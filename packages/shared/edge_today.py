"""Today's edge — touch & close vs market, per horizon (CR-028 / CR-V).

For a given setup (today's magnet/GEX read), computes structural minus
market-implied edge for two events at each of t1 / t5 / t15 horizons:

  Touch edge  = P_struct(touch within N sessions) − min(1, 2 × |delta_tN|)
  Close edge  = P_struct(close past strike at session tN) − |delta_tN|

where:
  Structural touch (horizon-specific): fraction of K analogues that touched
    the magnet within HORIZON_SESSION_DAYS[tN] sessions (path-based, from
    reached_touch + days_to_reach in bt_daily_outcomes_active).
  Structural close (unconditional, fresh from session_close_tN): fraction of
    K analogues whose PROJECTED session close at tN is past the placed short
    strike — over ALL computed analogues, NOT touch-conditional.
  Market close (|delta_tN|): stored ORATS delta at the magnet-strike option
    at the expiration closest to tN trading days from trade_date.
  Market touch (2×|delta_tN|): reflection-principle approximation, capped at
    1.0 (flagged as approximation in the UI).

Coordinate space (inherited from CR-T):
  - Classify analogue closes in ES-forward space against `strike_es`.
  - SPX-cash strike is used only to fetch the option / read delta.
  - Projection reuses _project_close from structural_distribution.

Critical constraint: structural CLOSE must be unconditional.
  Do NOT use reached_close (proximity test, not per-{t1/t5/t15} past-strike).
  Do NOT use aggregate_post_touch_distribution (touch-conditional + band-relative).

Public entry points:
    compute_todays_edge(...)  → dict (per-horizon edge table + warnings)

Module constants:
    HORIZONS              — v1 horizon set; all consumers read from here.
    HORIZON_SESSION_DAYS  — maps horizon label → session count for filtering.

Supported regimes (v1):
    "magnet-above", "magnet-below"
    All others → returns None for all horizons (no past-strike notion).
"""
from __future__ import annotations

from typing import Optional

from packages.shared.stats import wilson_ci
from packages.shared.structural_distribution import _project_close

# ── Horizon configuration ────────────────────────────────────────────────────

# v1 horizon set. All callers, schema, and UI read from this constant.
# Adding a new horizon (e.g. "t10") requires:
#   (a) session_close_t10 column in bt_daily_outcomes_active + backfill,
#   (b) add "t10": 10 to HORIZON_SESSION_DAYS here,
#   (c) find the ~10-DTE SPX expiration for the market-side delta.
# No edge logic, response shape, or UI structure changes.
HORIZONS: list[str] = ["t1", "t5", "t15"]
HORIZON_SESSION_DAYS: dict[str, int] = {"t1": 1, "t5": 5, "t15": 15}

# Minimum non-null closes (or computed analogues) per horizon/side before the
# cell is flagged low_confidence. Per-cell, not global.
DEFAULT_MIN_ANALOGUES: int = 8


# ── Supported regime helpers ─────────────────────────────────────────────────

_SUPPORTED_REGIMES = frozenset({"magnet-above", "magnet-below"})


def _past_strike(projected_close: float, strike_es: float, regime: str) -> bool:
    """True if projected_close is past the short strike in the regime direction."""
    if regime == "magnet-above":
        return projected_close >= strike_es
    return projected_close <= strike_es   # magnet-below


# ── Pure computation helpers ─────────────────────────────────────────────────


def _touch_rate_within_sessions(
    analogues: list[dict],
    max_sessions: int,
) -> tuple[Optional[float], Optional[tuple[float, float]], int]:
    """Fraction of computed analogues that touched within max_sessions sessions.

    Returns (rate, (ci_lo, ci_hi), k_computed).
    Rate is None when k_computed == 0.

    Filtering:
      denominator — analogues with outcome_status == 'computed'
      numerator   — those where reached_touch AND days_to_reach <= max_sessions
    """
    computed = [a for a in analogues if a.get("outcome_status") == "computed"]
    k = len(computed)
    if k == 0:
        return None, None, 0

    n_touch = sum(
        1
        for a in computed
        if a.get("reached_touch")
        and a.get("days_to_reach") is not None
        and int(a["days_to_reach"]) <= max_sessions
    )
    rate = n_touch / k
    ci = wilson_ci(n_touch, k)
    return round(rate, 4), ci, k


def _unconditional_close_past_strike(
    analogues: list[dict],
    horizon_key: str,
    strike_es: float,
    regime: str,
    today_spot: float,
    today_implied_move: float,
) -> tuple[Optional[float], Optional[tuple[float, float]], int]:
    """Fraction of ALL computed analogues whose projected close at horizon_key
    is past the placed short strike in ES-forward space.

    This is the one new pure function for CR-V. It is UNCONDITIONAL — all
    computed analogues, not just touchers. Do NOT feed it the post-touch bars.

    Args:
        analogues:          List of analogue dicts from _rank_analogues_with_outcomes.
                            Each must have: outcome_status, session_close_tN,
                            session_open_t0, implied_move_1d.
        horizon_key:        e.g. "t1", "t5", "t15" — determines which
                            session_close_* column to read.
        strike_es:          Placed short strike in ES discounted-forward space.
        regime:             "magnet-above" or "magnet-below".
        today_spot:         Today's session open (ES / SPX as consistent with
                            the analogue data source).
        today_implied_move: Today's implied move 1d (same units as today_spot).

    Returns (rate, (ci_lo, ci_hi), n_close) where n_close is the denominator
    (analogues with a non-null session_close at this horizon).
    Rate is None when n_close == 0.
    """
    close_col = f"session_close_{horizon_key}"

    valid: list[bool] = []
    for a in analogues:
        if a.get("outcome_status") != "computed":
            continue
        close = a.get(close_col)
        anchor_spot = a.get("session_open_t0")
        anchor_im = a.get("implied_move_1d")
        if close is None or anchor_spot is None or anchor_im is None:
            continue
        try:
            c_f = float(close)
            asp_f = float(anchor_spot)
            aim_f = float(anchor_im)
        except (TypeError, ValueError):
            continue
        if aim_f <= 0:
            continue
        projected = _project_close(c_f, asp_f, aim_f, today_spot, today_implied_move)
        valid.append(_past_strike(projected, strike_es, regime))

    n = len(valid)
    if n == 0:
        return None, None, 0

    n_past = sum(valid)
    rate = n_past / n
    ci = wilson_ci(n_past, n)
    return round(rate, 4), ci, n


def _market_side(
    delta: Optional[float],
    spread_cost: Optional[float],
    spread_width: float,
) -> tuple[Optional[float], Optional[float]]:
    """Return (mkt_close, mkt_touch) for one horizon.

    Primary source: stored |delta|.
    Fallback: |spread_cost| / spread_width when delta is None.
    Both None: return (None, None).

    mkt_touch = min(1.0, 2 × mkt_close) — reflection-principle approximation.
    """
    mkt_close: Optional[float] = None

    if delta is not None:
        mkt_close = round(abs(float(delta)), 4)
    elif spread_cost is not None and spread_width > 0:
        mkt_close = round(abs(float(spread_cost)) / float(spread_width), 4)

    if mkt_close is None:
        return None, None

    mkt_touch = round(min(1.0, 2.0 * mkt_close), 4)
    return mkt_close, mkt_touch


# ── Public entry point ───────────────────────────────────────────────────────


def compute_todays_edge(
    *,
    regime: str,
    strike_es: float,
    today_spot: float,
    today_implied_move: float,
    analogues: list[dict],
    horizons: list[str],
    magnet_delta_by_horizon: dict[str, Optional[float]],
    spread_cost_by_horizon: dict[str, Optional[float]],
    spread_width: float,
    min_analogues: int = DEFAULT_MIN_ANALOGUES,
) -> dict:
    """Compute per-horizon touch & close edge for today's setup.

    Args:
        regime:                    "magnet-above" | "magnet-below". Other
                                   values return per_horizon=None.
        strike_es:                 Placed short strike in ES discounted-forward
                                   space (= magnet strike, rounded to listed).
        today_spot:                Today's session open (ES-forward units).
        today_implied_move:        Today's implied move 1d (same units).
        analogues:                 K ranked analogue dicts from
                                   _rank_analogues_with_outcomes. Each must
                                   carry outcome_status, reached_touch,
                                   days_to_reach, session_open_t0,
                                   implied_move_1d, and session_close_tN.
        horizons:                  Ordered list of horizon labels to compute.
                                   v1 = HORIZONS = ["t1","t5","t15"].
                                   Never hard-coded inside this function.
        magnet_delta_by_horizon:   {tN: abs(delta)} at the magnet-strike option
                                   for tN's expiration. None if cache miss.
        spread_cost_by_horizon:    {tN: net_debit} cross-check / fallback for
                                   the close-market prob. None if unavailable.
        spread_width:              Width of the spread in ES points (v1 = 10).
        min_analogues:             Per-cell low-confidence threshold.

    Returns:
        {
            "regime":      str,
            "per_horizon": [
                {
                    "horizon":         str,
                    "struct_touch":    float | None,
                    "struct_touch_ci": [float, float] | None,
                    "n_touch":         int,
                    "mkt_touch":       float | None,
                    "touch_edge":      float | None,
                    "struct_close":    float | None,
                    "struct_close_ci": [float, float] | None,
                    "n_close":         int,
                    "mkt_close":       float | None,
                    "close_edge":      float | None,
                    "low_confidence":  bool,
                },
                ...
            ] | None,       # None for unsupported regimes
            "warnings":    [str],
        }
    """
    warnings_out: list[str] = []

    if regime not in _SUPPORTED_REGIMES:
        return {
            "regime":      regime,
            "per_horizon": None,
            "warnings":    [f"edge not supported for regime {regime!r} (v1: magnet-above/below only)"],
        }

    per_horizon: list[dict] = []

    for tN in horizons:
        max_sess = HORIZON_SESSION_DAYS.get(tN)
        if max_sess is None:
            warnings_out.append(f"unknown horizon {tN!r}; skipped")
            continue

        # ── Structural touch (within max_sess sessions) ────────────────────
        struct_touch, touch_ci, n_touch = _touch_rate_within_sessions(
            analogues, max_sess
        )

        # ── Structural close (unconditional, fresh from session_close_tN) ──
        struct_close, close_ci, n_close = _unconditional_close_past_strike(
            analogues, tN, strike_es, regime, today_spot, today_implied_move
        )

        # ── Market side (stored delta primary; cost÷width fallback) ────────
        delta = magnet_delta_by_horizon.get(tN)
        spread_cost = spread_cost_by_horizon.get(tN)
        mkt_close, mkt_touch = _market_side(delta, spread_cost, spread_width)

        if mkt_close is None:
            warnings_out.append(
                f"no market-side data for horizon {tN!r} "
                "(delta and spread cost both unavailable)"
            )

        # ── Edge ───────────────────────────────────────────────────────────
        touch_edge = (
            round(struct_touch - mkt_touch, 4)
            if struct_touch is not None and mkt_touch is not None
            else None
        )
        close_edge = (
            round(struct_close - mkt_close, 4)
            if struct_close is not None and mkt_close is not None
            else None
        )

        # ── Low-confidence flag ────────────────────────────────────────────
        low_confidence = (n_touch < min_analogues) or (n_close < min_analogues)

        per_horizon.append({
            "horizon":         tN,
            "struct_touch":    struct_touch,
            "struct_touch_ci": list(touch_ci) if touch_ci is not None else None,
            "n_touch":         n_touch,
            "mkt_touch":       mkt_touch,
            "touch_edge":      touch_edge,
            "struct_close":    struct_close,
            "struct_close_ci": list(close_ci) if close_ci is not None else None,
            "n_close":         n_close,
            "mkt_close":       mkt_close,
            "close_edge":      close_edge,
            "low_confidence":  low_confidence,
        })

    return {
        "regime":      regime,
        "per_horizon": per_horizon,
        "warnings":    warnings_out,
    }
