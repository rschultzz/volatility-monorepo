"""Proposals service — pure computation helpers for POST /api/proposals/pl-data.

No DB I/O in this module. All DB access is in routes.py.

Functions:
    build_evaluation_time   — expiration date → 16:00 ET → UTC datetime
    build_bsm_chain         — BSM call prices across spot ± range_pts
    compute_initial_cost    — net premium at entry from BSM pricing
    compute_pl_curve        — P/L array across ±2σ price grid
    compute_key_levels      — max profit, max loss, breakeven crossings
"""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np

from packages.shared.pricing.engine import (
    black_scholes_price,
    compute_leg_value,
    compute_position_pl,
)

_ET  = ZoneInfo("America/New_York")
_UTC = ZoneInfo("UTC")


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


def compute_pl_curve(
    legs: list[dict],
    spot: float,
    implied_move: float,
    evaluation_time: dt.datetime,
    market_state: dict,
    initial_cost: float,
    *,
    range_sigma: float = 2.0,
    n_points: int = 200,
) -> dict:
    """P/L across spot ± range_sigma * implied_move at evaluation_time.

    Returns {"prices": [float], "pnl": [float]}.
    Falls back to spot ± 50 if implied_move == 0.
    """
    half = range_sigma * implied_move if implied_move > 0 else 50.0
    lo = spot - half
    hi = spot + half
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
