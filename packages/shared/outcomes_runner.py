"""Shared per-date outcome computation runner (CR-AA).

Extracted from scripts/cr_b_backfill_outcomes.py so both the CR-022 backfill
and the CR-AA sweep call a single implementation.  Do not re-implement this
logic inline — two divergent copies silently corrupt outcome output.

Public API
----------
compute_outcome_for_date(trade_date, regime, feature_vector, landscape, daily_bars)
    -> (outcome_dict, session_open_t0)

derive_dominant_bucket(feature_vector) -> str | None
direction_sanity(regime, drift_target, table_spot, trade_date) -> bool
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Optional

import pandas as pd

from packages.shared.outcomes import compute_outcome, pick_drift_target

log = logging.getLogger(__name__)

# Distance threshold for magnetic-pin sanity check.
# Matches near_dist_pts default in classify_regime — a dominant wall further
# than this from spot is not acting as a pin, so the outcome would be garbage.
_PIN_MAX_DISTANCE_PTS = 30.0


def derive_dominant_bucket(feature_vector: dict) -> Optional[str]:
    """Argmax of dominance_* fields. None if all are missing/zero."""
    candidates = {
        "0DTE":     feature_vector.get("dominance_0DTE",   0.0) or 0.0,
        "1-7 DTE":  feature_vector.get("dominance_1_7",    0.0) or 0.0,
        "8-30 DTE": feature_vector.get("dominance_8_30",   0.0) or 0.0,
        "30+ DTE":  feature_vector.get("dominance_30plus", 0.0) or 0.0,
    }
    if not any(candidates.values()):
        return None
    return max(candidates, key=candidates.__getitem__)


def direction_sanity(
    regime: str,
    drift_target: float,
    table_spot: float,
    trade_date: dt.date,
) -> bool:
    """Return False (and log) if drift_target contradicts regime direction or is implausibly far."""
    if regime == "magnet-above" and drift_target < table_spot:
        log.warning(
            "direction_sanity FAIL %s %s: magnet-above but drift_target=%.2f < spot=%.2f",
            trade_date, regime, drift_target, table_spot,
        )
        return False
    if regime == "magnet-below" and drift_target > table_spot:
        log.warning(
            "direction_sanity FAIL %s %s: magnet-below but drift_target=%.2f > spot=%.2f",
            trade_date, regime, drift_target, table_spot,
        )
        return False
    if regime == "magnetic-pin" and abs(drift_target - table_spot) > _PIN_MAX_DISTANCE_PTS:
        log.warning(
            "direction_sanity FAIL %s magnetic-pin: drift_target=%.2f too far from spot=%.2f (>%.0fpt)",
            trade_date, drift_target, table_spot, _PIN_MAX_DISTANCE_PTS,
        )
        return False
    return True


def compute_outcome_for_date(
    trade_date: dt.date,
    regime: str,
    feature_vector: dict,
    landscape: dict,
    daily_bars: pd.DataFrame,
) -> tuple[dict, Optional[float]]:
    """Derive bucket, drift_target, expected_move; run direction sanity; call compute_outcome.

    Parameters
    ----------
    trade_date     : Date of the structural read.
    regime         : bt_daily_features.regime_at_classification value.
    feature_vector : bt_daily_features.feature_vector JSONB dict.
    landscape      : {"walls": [...], "table_spot": float | None} from orats_gex_landscape.
    daily_bars     : RTH daily OHLC DataFrame (index: date objects; columns: open, high, low, close).

    Returns
    -------
    (outcome_dict, session_open_t0)
    outcome_dict    : All bt_daily_outcomes fields except ticker, trade_date,
                      feature_version, backfill_run_id, computed_at.
    session_open_t0 : First RTH bar open on trade_date; None if not in bars.
    """
    walls      = landscape.get("walls") or []
    table_spot = landscape.get("table_spot")

    dominant_bucket = derive_dominant_bucket(feature_vector)
    drift_target    = pick_drift_target(walls)
    expected_move   = feature_vector.get("implied_move_1d")
    if expected_move is not None:
        try:
            expected_move = float(expected_move)
        except (TypeError, ValueError):
            expected_move = None

    sanity_ok = True
    if drift_target is not None and table_spot is not None:
        sanity_ok = direction_sanity(regime, drift_target, table_spot, trade_date)

    session_open_t0 = (
        float(daily_bars.loc[trade_date, "open"])
        if trade_date in daily_bars.index
        else None
    )

    if not sanity_ok:
        outcome = {
            "regime_kind_at_classification":     regime,
            "dominant_bucket_at_classification": dominant_bucket,
            "horizon_sessions":                  None,
            "horizon_end_date":                  None,
            "outcome_status":                    "na_data",
            "reached_touch":                     None,
            "reached_close":                     None,
            "days_to_reach":                     None,
            "max_excursion_in_direction":        None,
            "final_close_distance_from_target":  None,
            "actual_realized_em_pct":            None,
        }
    else:
        outcome = compute_outcome(
            trade_date      = trade_date,
            regime          = regime or "",
            drift_target    = drift_target,
            dominant_bucket = dominant_bucket or "",
            expected_move   = expected_move,
            bars            = daily_bars,
        )

    return outcome, session_open_t0
