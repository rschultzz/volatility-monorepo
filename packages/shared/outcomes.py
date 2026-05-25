"""Outcome computation for structural daily reads (CR-B).

Public API:
    compute_outcome(...) -> dict   — pure computation, no DB I/O.

The runner (scripts/cr_b_backfill_outcomes.py) is responsible for:
  - joining drift_target from orats_gex_landscape.walls
  - deriving dominant_bucket via argmax of dominance_* feature fields
  - fetching and aggregating RTH bars to daily OHLC
  - persisting the returned dict to bt_daily_outcomes
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import pandas as pd

from packages.shared.buckets import bucket_sessions

log = logging.getLogger(__name__)

_DIRECTIONAL_REGIMES = frozenset({"magnet-above", "magnet-below", "magnetic-pin"})


def compute_outcome(
    trade_date: date,
    regime: str,
    drift_target: Optional[float],
    dominant_bucket: str,
    expected_move: Optional[float],
    bars: pd.DataFrame,
) -> dict:
    """Compute outcome metrics for one structural read.

    Parameters
    ----------
    trade_date      : The trade date the structural read applies to.
    regime          : bt_daily_features.regime_at_classification value.
    drift_target    : Dominant GEX wall price from orats_gex_landscape.walls;
                      None if no landscape row exists for this date.
    dominant_bucket : Argmax of dominance_* fields in feature_vector — one of
                      '0DTE', '1-7 DTE', '8-30 DTE', '30+ DTE'.
    expected_move   : feature_vector['implied_move_1d'] — 1-day expected move
                      in points. Used as tolerance base (0.25×) and the EM
                      ratio denominator for actual_realized_em_pct.
    bars            : Daily RTH OHLC DataFrame. Index: date objects, one row
                      per session. Columns: open, high, low, close (float).
                      Must span trade_date through at least
                      trade_date + horizon_sessions to produce 'computed' rows.

    Returns
    -------
    dict with all bt_daily_outcomes fields except ticker, trade_date,
    feature_version, backfill_run_id, and computed_at — the runner fills those.

    outcome_status values
    ---------------------
    'computed'        — all metrics populated.
    'pending_history' — fewer RTH bars available than n_sessions requires;
                        metrics NULL.
    'na_regime'       — regime is non-directional (bounded, amplification,
                        untethered) or unknown; metrics NULL. Expected to be
                        the majority of rows (173+77+19 = 269 of 735 in
                        v0.5.0-rebuilt).
    'na_data'         — a required input was missing or unusable (no landscape
                        row, expected_move ≤ 0, empty/all-NaN bars, unknown
                        bucket label). Expected count for v0.5.0-rebuilt: 0.

    Session basis (E7)
    ------------------
    bars must contain one row per RTH session (06:30–13:00 PT = 13:30–20:00
    UTC). Each row's open is the 06:30 PT open; close is the 13:00 PT close.
    This matches the session definition in apps/web/modules/Bars/service.py
    and the Analogues module's _fetch_session_outcomes.

    Why ES bars work for SPX-anchored outcomes (E6)
    -----------------------------------------------
    drift_target is the price of the dominant GEX wall from
    orats_gex_landscape.walls. The landscape accumulates dealer GEX at each
    option's discounted_level:

        discounted_level = strike × exp((r − q) × T)

    where r is the short rate, q is the dividend yield, and T is time to
    expiry in years. This formula projects each SPX cash strike into forward
    price space — the same space ES futures trade in. A wall at price P in the
    landscape means there is a concentration of options whose ES-equivalent
    forward price ≈ P. ES bar prices are therefore directly comparable to
    drift_target without any SPX-to-ES conversion.

    The residual difference between discounted_level and the actual ES futures
    price arises from rate/dividend convention rounding and roll effects;
    empirically this is < 5 pts. This is negligible relative to the
    0.25 × expected_move tolerance used for reached_close (typically 10–14 pts
    on a 40–55 pt expected move). The Analogues module's _fetch_session_outcomes
    establishes this as the canonical codebase pattern: ES bar prices compared
    directly to SPX-derived landscape levels, no conversion.

    days_to_reach convention
    ------------------------
    0-indexed: 0 = target touched on trade_date itself (same session as the
    structural read), 1 = one session later, etc. NULL if not reached.
    """
    base: dict = {
        "regime_kind_at_classification":    regime,
        "dominant_bucket_at_classification": dominant_bucket,
        "horizon_sessions":                 None,
        "horizon_end_date":                 None,
        "outcome_status":                   None,
        "reached_touch":                    None,
        "reached_close":                    None,
        "days_to_reach":                    None,
        "max_excursion_in_direction":       None,
        "final_close_distance_from_target": None,
        "actual_realized_em_pct":           None,
    }

    # ── 1. na_regime: non-directional regimes have no v1 outcome ─────────────
    if regime not in _DIRECTIONAL_REGIMES:
        return {**base, "outcome_status": "na_regime"}

    # ── 2. Horizon session count (needed before na_data returns) ──────────────
    try:
        n_sessions = bucket_sessions(dominant_bucket)
    except KeyError:
        log.warning(
            "compute_outcome: unknown dominant_bucket %r for %s %s",
            dominant_bucket, trade_date, regime,
        )
        return {**base, "outcome_status": "na_data"}

    base["horizon_sessions"] = n_sessions

    # ── 3. na_data: skip-and-log on missing or invalid required inputs ────────
    # Lesson 2 from CR-021: missing inputs must skip-and-log, not zero-fallback.
    if drift_target is None:
        log.warning(
            "compute_outcome: drift_target is None for %s %s — missing landscape row?",
            trade_date, regime,
        )
        return {**base, "outcome_status": "na_data"}

    if expected_move is None or expected_move <= 0:
        log.warning(
            "compute_outcome: expected_move=%r for %s %s — skipping to avoid "
            "zero-division and sentinel-value corruption",
            expected_move, trade_date, regime,
        )
        return {**base, "outcome_status": "na_data"}

    if bars is None or bars.empty:
        log.warning(
            "compute_outcome: empty bars for %s %s", trade_date, regime,
        )
        return {**base, "outcome_status": "na_data"}

    required_cols = {"open", "high", "low", "close"}
    if not required_cols.issubset(bars.columns):
        log.warning(
            "compute_outcome: bars missing columns %s for %s %s",
            required_cols - set(bars.columns), trade_date, regime,
        )
        return {**base, "outcome_status": "na_data"}

    # ── 4. Filter bars to trade_date onward; drop all-NaN rows ───────────────
    bars_sorted = bars.sort_index()
    forward = bars_sorted[bars_sorted.index >= trade_date].dropna(
        subset=["high", "low", "close"]
    )

    if forward.empty:
        log.warning(
            "compute_outcome: no usable bars on/after trade_date %s for %s %s",
            trade_date, regime, drift_target,
        )
        return {**base, "outcome_status": "na_data"}

    # ── 5. pending_history: not enough sessions elapsed ───────────────────────
    if len(forward) < n_sessions:
        return {**base, "outcome_status": "pending_history"}

    # ── 6. Slice to horizon ───────────────────────────────────────────────────
    horizon = forward.iloc[:n_sessions]
    base["horizon_end_date"] = horizon.index[-1]

    drift_target_f  = float(drift_target)
    expected_move_f = float(expected_move)
    tolerance       = 0.25 * expected_move_f

    # ── 7. Direction-specific touch and excursion ─────────────────────────────
    if regime == "magnet-above":
        touch_series = horizon["high"] >= drift_target_f
        first_open   = float(horizon.iloc[0]["open"])
        # C5: clamp to 0.0 — max favorable upward move from session open
        max_excursion = max(0.0, float(horizon["high"].max()) - first_open)

    elif regime == "magnet-below":
        touch_series = horizon["low"] <= drift_target_f
        first_open   = float(horizon.iloc[0]["open"])
        # C5: clamp to 0.0 — max favorable downward move from session open
        max_excursion = max(0.0, first_open - float(horizon["low"].min()))

    else:  # magnetic-pin
        # reached_touch = any bar whose range overlaps the tolerance band
        # [drift_target − tol, drift_target + tol]
        touch_series = (
            (horizon["high"] >= (drift_target_f - tolerance))
            & (horizon["low"]  <= (drift_target_f + tolerance))
        )
        first_open    = float(horizon.iloc[0]["open"])
        up_excursion   = max(0.0, float(horizon["high"].max()) - first_open)
        down_excursion = max(0.0, first_open - float(horizon["low"].min()))
        max_excursion  = max(up_excursion, down_excursion)

    # ── 8. Compute metrics ────────────────────────────────────────────────────
    reached_touch = bool(touch_series.any())
    days_to_reach = int(touch_series.values.argmax()) if reached_touch else None

    final_close  = float(horizon.iloc[-1]["close"])
    reached_close = bool(abs(final_close - drift_target_f) <= tolerance)

    final_close_distance = final_close - drift_target_f

    horizon_high           = float(horizon["high"].max())
    horizon_low            = float(horizon["low"].min())
    actual_realized_em_pct = (horizon_high - horizon_low) / expected_move_f

    return {
        **base,
        "outcome_status":                   "computed",
        "reached_touch":                    reached_touch,
        "reached_close":                    reached_close,
        "days_to_reach":                    days_to_reach,
        "max_excursion_in_direction":       max_excursion,
        "final_close_distance_from_target": final_close_distance,
        "actual_realized_em_pct":           actual_realized_em_pct,
    }
