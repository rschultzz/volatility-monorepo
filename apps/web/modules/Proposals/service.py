"""Proposals service — pure computation helpers for POST /api/proposals/pl-data.

No DB I/O in this module. All DB access is in routes.py.

Functions:
    build_evaluation_time   — expiration date → 16:00 ET → UTC datetime
    build_bsm_chain         — BSM call prices across spot ± range_pts
    build_grid_bounds       — regime-aware (lo, hi) for the price grid
    compute_initial_cost    — net premium at entry from BSM pricing
    compute_pl_curve        — P/L array across asymmetric regime-aware price grid
    compute_key_levels      — max profit, max loss, breakeven crossings
"""
from __future__ import annotations

import datetime as dt
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np

from packages.shared.pricing.engine import (
    black_scholes_price,
    compute_leg_value,
    compute_position_pl,
)

_ET  = ZoneInfo("America/New_York")
_UTC = ZoneInfo("UTC")

# Regimes that get a symmetric spot-centred grid (no magnet/pin extension).
_SYMMETRIC_REGIMES = {"amplification", "untethered", "broken-magnet"}


def build_evaluation_time(expir_date: dt.date) -> dt.datetime:
    """Return 16:00 ET on expir_date, converted to UTC.

    This is the standard options-market evaluation time for a position
    held through close on the expiration date.
    """
    naive = dt.datetime(expir_date.year, expir_date.month, expir_date.day, 16, 0)
    return naive.replace(tzinfo=_ET).astimezone(_UTC)


def build_bsm_chain(
    spot: float,
    atmiv: float,
    risk_free_rate: float,
    tte: float,
    *,
    range_pts: float = 300.0,
    spacing: float = 5.0,
) -> list[dict]:
    """BSM flat-vol call prices at strikes spot ± range_pts in `spacing` increments.

    Returns list of {'strike': float, 'call_price': float} dicts.
    Uses a flat volatility surface (atmiv for all strikes — MVP; no skew).
    """
    result: list[dict] = []
    lo = spot - range_pts
    hi = spot + range_pts
    k = lo
    while k <= hi + 1e-9:
        price = black_scholes_price(spot, k, tte, risk_free_rate, atmiv, "c")
        result.append({"strike": round(k, 4), "call_price": price})
        k += spacing
    return result


def compute_initial_cost(
    legs: list[dict],
    spot: float,
    evaluation_time: dt.datetime,
    market_state: dict,
) -> float:
    """Net premium paid at position entry using BSM pricing at spot.

    Positive = net debit; negative = net credit.
    Legs must have: strike, expiration (tz-aware datetime), flag, side, qty, iv.
    """
    return sum(
        compute_leg_value(leg, spot, evaluation_time, market_state)
        for leg in legs
    )


def build_grid_bounds(
    spot: float,
    implied_move: float,
    regime_block: Optional[dict],
    *,
    half_sigma: float = 1.5,
    max_sigma: float = 3.0,
) -> tuple[float, float]:
    """Compute asymmetric price-grid bounds based on regime.

    Returns (lo, hi) — the price range to cover in the P/L curve and edge-zone
    price axis.  Each side is capped at max_sigma × implied_move from spot so
    that an unusually distant magnet never produces an absurdly wide grid.

    Regime mapping:
        magnet-above   → [spot − half×IM,  max(spot + half×IM, drift_target + half×IM)]
        magnet-below   → [min(spot − half×IM, drift_target − half×IM), spot + half×IM]
        magnetic-pin   → [drift_target − half×IM, drift_target + half×IM]
        bounded        → [cz.lower_price − 0.5×IM, cz.upper_price + 0.5×IM]
        everything else → spot ± half×IM  (symmetric default)

    Cap: each bound is clamped to [spot − max_sigma×IM, spot + max_sigma×IM].

    Falls back to the symmetric default on any missing key or type error.
    """
    half = half_sigma * implied_move if implied_move > 0 else 50.0
    fallback_lo = spot - half
    fallback_hi = spot + half
    cap_lo = spot - (max_sigma * implied_move if implied_move > 0 else 150.0)
    cap_hi = spot + (max_sigma * implied_move if implied_move > 0 else 150.0)

    if not regime_block or not isinstance(regime_block, dict):
        return fallback_lo, fallback_hi

    regime = regime_block.get("regime", "")
    if not regime or regime in _SYMMETRIC_REGIMES:
        return fallback_lo, fallback_hi

    try:
        if regime == "magnet-above":
            drift = float(regime_block["drift_target"])
            hi = min(max(fallback_hi, drift + half), cap_hi)
            return fallback_lo, hi

        if regime == "magnet-below":
            drift = float(regime_block["drift_target"])
            lo = max(min(fallback_lo, drift - half), cap_lo)
            return lo, fallback_hi

        if regime == "magnetic-pin":
            drift = float(regime_block["drift_target"])
            lo = max(drift - half, cap_lo)
            hi = min(drift + half, cap_hi)
            return lo, hi

        if regime == "bounded":
            cz = regime_block["containment_zone"]
            lower_price = float(cz["lower_price"])
            upper_price = float(cz["upper_price"])
            buffer = 0.5 * implied_move if implied_move > 0 else 25.0
            lo = max(lower_price - buffer, cap_lo)
            hi = min(upper_price + buffer, cap_hi)
            return lo, hi

    except (KeyError, TypeError, ValueError):
        pass

    # Unrecognised regime or missing keys — symmetric fallback
    return fallback_lo, fallback_hi


def compute_pl_curve(
    legs: list[dict],
    spot: float,
    implied_move: float,
    evaluation_time: dt.datetime,
    market_state: dict,
    initial_cost: float,
    *,
    regime_block: Optional[dict] = None,
) -> dict:
    """P/L across a regime-aware price grid at evaluation_time.

    When regime_block is provided, the grid is extended asymmetrically toward
    the regime's directional side via build_grid_bounds (see that function for
    the per-regime mapping).  Falls back to a symmetric spot ± 1.5×IM grid when
    regime_block is None or unrecognised.

    Step size is 1 pt (n_points = round(hi − lo) + 1).
    Falls back to spot ± 50 if implied_move == 0.

    Returns {"prices": [float], "pnl": [float]}.
    """
    lo, hi = build_grid_bounds(spot, implied_move, regime_block)
    n_points = max(50, round(hi - lo) + 1)
    price_grid = np.linspace(lo, hi, n_points)
    pnl = compute_position_pl(legs, price_grid, evaluation_time, market_state, initial_cost)
    return {
        "prices": price_grid.tolist(),
        "pnl":    pnl.tolist(),
    }


def compute_key_levels(pnl: list[float], prices: list[float]) -> dict:
    """Max profit, max loss, and breakeven crossings from the P/L curve.

    A breakeven is a price level where the P/L crosses zero in either direction
    (loss → profit or profit → loss).  Three cases are handled:

    1. Sign change (a < 0 and b > 0 or vice versa): linear interpolation.
    2. Transition from exactly 0 to non-zero sign (e.g. OTM region to ITM region
       in a flat-cost spread): use prices[i+1] as the breakeven.

    To avoid reporting dozens of "breakevens" inside a flat-zero region (common
    at expiration when both legs are OTM and initial_cost=0), a transition only
    fires at the *first* index where pnl goes from non-positive to positive (or
    non-negative to negative).  Consecutive duplicates are deduplicated.

    Returns {"max_profit": float, "max_loss": float, "breakevens": [float]}.
    """
    if not pnl:
        return {"max_profit": None, "max_loss": None, "breakevens": []}

    max_profit = max(pnl)
    max_loss   = min(pnl)

    breakevens: list[float] = []
    for i in range(len(pnl) - 1):
        a, b = pnl[i], pnl[i + 1]
        if a * b < 0:
            # Strict sign change: interpolate the crossing
            t  = -a / (b - a)
            be = prices[i] + t * (prices[i + 1] - prices[i])
            breakevens.append(round(be, 2))
        elif a <= 0.0 < b:
            # Transition from non-profit (including flat-0) to profit
            breakevens.append(round(prices[i + 1], 2))
        elif b < 0.0 <= a:
            # Transition from non-loss to loss
            breakevens.append(round(prices[i + 1], 2))

    # Deduplicate very-close breakevens (can arise at adjacent points in a steep move)
    deduped: list[float] = []
    for be in breakevens:
        if not deduped or abs(be - deduped[-1]) > 1.0:
            deduped.append(be)

    return {
        "max_profit": round(max_profit, 4),
        "max_loss":   round(max_loss,   4),
        "breakevens": deduped,
    }
