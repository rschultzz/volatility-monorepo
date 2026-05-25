"""Structural probability computation (CR-C).

Three layers per CR-021 Lesson 3 (compute / persist separation):
  stats.wilson_ci                   → pure math, in stats.py
  _aggregate_outcomes               → pure math, synthetic-testable
  _rank_analogues_with_outcomes     → DB I/O shell
  compute_structural_probability    → orchestrator (thin)

Public entry point:
    compute_structural_probability(today_features, conn, ...) → dict
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Optional

from packages.shared.canonical_version import CANONICAL_FEATURE_VERSION
from packages.shared.knn import feature_stats, rank_analogues
from packages.shared.stats import wilson_ci

log = logging.getLogger(__name__)


def _aggregate_outcomes(rows: list[dict]) -> dict:
    """Aggregate ranked-analogue rows into a probability summary.

    Pure math — no DB I/O. Testable with synthetic data.

    Args:
        rows: List of dicts with keys:
            outcome_status: str | None  ('computed', 'na_regime', etc.)
            reached_touch:  bool | None
            reached_close:  bool | None
            days_to_reach:  int | None
            max_excursion_in_direction: float | None
            implied_move_1d: float | None
            distance:        float

    Returns:
        Dict with outcome_status='ok' or 'no_data', plus aggregated fields.
    """
    n_na_regime  = sum(1 for r in rows if r.get("outcome_status") == "na_regime")
    n_pending    = sum(1 for r in rows if r.get("outcome_status") == "pending_history")
    n_na_data    = sum(1 for r in rows if r.get("outcome_status") == "na_data")
    n_no_outcome = sum(1 for r in rows if r.get("outcome_status") is None)
    computed     = [r for r in rows if r.get("outcome_status") == "computed"]
    k_with_outcomes = len(computed)

    if k_with_outcomes == 0:
        detail = []
        if n_na_regime:  detail.append(f"{n_na_regime} had na_regime")
        if n_pending:    detail.append(f"{n_pending} had pending_history")
        if n_na_data:    detail.append(f"{n_na_data} had na_data")
        if n_no_outcome: detail.append(f"{n_no_outcome} had no outcome row")
        note = f"No computed analogues found among {len(rows)} nearest neighbors."
        if detail:
            note += " " + ", ".join(detail) + "."
        return {
            "outcome_status":     "no_data",
            "k_with_outcomes":    0,
            "touch_rate":         None,
            "close_rate":         None,
            "touch_ci_lower":     None,
            "touch_ci_upper":     None,
            "mean_days_to_reach": None,
            "mean_excursion_pct": None,
            "note":               note,
        }

    # ── Rates ─────────────────────────────────────────────────────────────────
    n_touched = sum(1 for r in computed if r.get("reached_touch"))
    n_closed  = sum(1 for r in computed if r.get("reached_close"))
    touch_rate = n_touched / k_with_outcomes
    close_rate = n_closed  / k_with_outcomes
    ci_lower, ci_upper = wilson_ci(n_touched, k_with_outcomes)

    # ── Mean days to reach (only rows that actually touched) ──────────────────
    days_list = [
        r["days_to_reach"]
        for r in computed
        if r.get("reached_touch") and r.get("days_to_reach") is not None
    ]
    mean_days_to_reach = sum(days_list) / len(days_list) if days_list else None

    # ── Mean excursion pct (skip analogues with invalid implied_move_1d) ──────
    excursion_pcts = []
    n_excursion_skipped = 0
    for r in computed:
        im  = r.get("implied_move_1d")
        exc = r.get("max_excursion_in_direction")
        if im is None or im <= 0.0:
            if im is not None:
                log.warning(
                    "_aggregate_outcomes: skipping excursion ratio — "
                    "implied_move_1d=%r for %s",
                    im, r.get("trade_date", "?"),
                )
            n_excursion_skipped += 1
            continue
        if exc is None:
            continue
        excursion_pcts.append(exc / im)
    mean_excursion_pct = (
        sum(excursion_pcts) / len(excursion_pcts) if excursion_pcts else None
    )

    # ── Distance range ────────────────────────────────────────────────────────
    finite_dists = [
        r["distance"] for r in rows
        if isinstance(r.get("distance"), (int, float))
        and r["distance"] != float("inf")
    ]
    min_dist = min(finite_dists) if finite_dists else None
    max_dist = max(finite_dists) if finite_dists else None

    # ── Provenance note ───────────────────────────────────────────────────────
    status_detail = []
    if n_na_regime:  status_detail.append(f"{n_na_regime} had na_regime")
    if n_pending:    status_detail.append(f"{n_pending} had pending_history")
    if n_na_data:    status_detail.append(f"{n_na_data} had na_data")
    if n_no_outcome: status_detail.append(f"{n_no_outcome} had no outcome row")

    base = f"Based on {k_with_outcomes} analogues with computed outcomes"
    if status_detail:
        base += f" ({', '.join(status_detail)})"
    note_parts = [base + "."]

    if min_dist is not None and max_dist is not None and len(rows) > 1:
        note_parts.append(f"Distance range: {min_dist:.2f}σ – {max_dist:.2f}σ.")
    if n_excursion_skipped > 0:
        note_parts.append(
            f"{n_excursion_skipped} analogue(s) excluded from excursion mean "
            "(implied_move_1d ≤ 0)."
        )

    return {
        "outcome_status":     "ok",
        "k_with_outcomes":    k_with_outcomes,
        "touch_rate":         round(touch_rate, 4),
        "close_rate":         round(close_rate, 4),
        "touch_ci_lower":     round(ci_lower, 4) if ci_lower is not None else None,
        "touch_ci_upper":     round(ci_upper, 4) if ci_upper is not None else None,
        "mean_days_to_reach": round(mean_days_to_reach, 2) if mean_days_to_reach is not None else None,
        "mean_excursion_pct": round(mean_excursion_pct, 4) if mean_excursion_pct is not None else None,
        "note":               " ".join(note_parts),
    }


def _rank_analogues_with_outcomes(
    today_features: dict,
    conn,
    k: int,
    feature_version: str,
    *,
    ticker: str = "SPX",
    exclude_date: Optional[str] = None,
) -> list[dict]:
    """Load corpus, rank by KNN distance, join outcomes.

    DB I/O layer — pure aggregation math is in _aggregate_outcomes.

    Returns list of dicts: trade_date, distance, outcome_status,
    reached_touch, reached_close, days_to_reach,
    max_excursion_in_direction, implied_move_1d.
    """
    # 1. Load corpus feature vectors
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT trade_date, feature_vector
            FROM bt_daily_features_active
            WHERE ticker = %s AND feature_version = %s
            """,
            (ticker, feature_version),
        )
        corpus_rows = cur.fetchall()

    if not corpus_rows:
        log.warning(
            "_rank_analogues_with_outcomes: empty corpus ticker=%r fv=%r",
            ticker, feature_version,
        )
        return []

    candidates = [(d.isoformat(), v) for (d, v) in corpus_rows]
    fv_by_date  = {d: v for (d, v) in candidates}

    # 2. Rank
    stats  = feature_stats(v for (_, v) in candidates)
    ranked = rank_analogues(
        today_features, candidates, k,
        exclude_date=exclude_date, stats=stats,
    )
    if not ranked:
        return []

    # 3. Fetch outcomes for the top-K dates
    top_dates = [dt.date.fromisoformat(d) for (d, _) in ranked]
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT trade_date, outcome_status,
                   reached_touch, reached_close, days_to_reach,
                   max_excursion_in_direction
            FROM bt_daily_outcomes_active
            WHERE ticker = %s AND feature_version = %s
              AND trade_date = ANY(%s)
            """,
            (ticker, feature_version, top_dates),
        )
        outcome_rows = cur.fetchall()

    outcomes_by_date = {
        row[0].isoformat(): {
            "outcome_status":             row[1],
            "reached_touch":              row[2],
            "reached_close":              row[3],
            "days_to_reach":              row[4],
            "max_excursion_in_direction": float(row[5]) if row[5] is not None else None,
        }
        for row in outcome_rows
    }

    # 4. Assemble; implied_move_1d comes from the feature vector already in memory
    result = []
    for trade_date_iso, distance in ranked:
        outcome = outcomes_by_date.get(trade_date_iso, {})
        fv      = fv_by_date.get(trade_date_iso, {})
        raw_im  = fv.get("implied_move_1d")
        try:
            implied_move_1d = float(raw_im) if raw_im is not None else None
        except (TypeError, ValueError):
            implied_move_1d = None

        result.append({
            "trade_date":                 trade_date_iso,
            "distance":                   float(distance),
            "outcome_status":             outcome.get("outcome_status"),
            "reached_touch":              outcome.get("reached_touch"),
            "reached_close":              outcome.get("reached_close"),
            "days_to_reach":              outcome.get("days_to_reach"),
            "max_excursion_in_direction": outcome.get("max_excursion_in_direction"),
            "implied_move_1d":            implied_move_1d,
        })
    return result


def compute_structural_probability(
    today_features: dict,
    conn,
    k: int = 20,
    feature_version: str = CANONICAL_FEATURE_VERSION,
    *,
    ticker: str = "SPX",
    exclude_date: Optional[str] = None,
    regime_kind: Optional[str] = None,
) -> dict:
    """Given today's feature vector, compute structural probability.

    Calls _rank_analogues_with_outcomes (DB) → _aggregate_outcomes (math)
    and adds request-level context fields (regime_kind, k).

    Args:
        today_features:  Feature dict matching bt_daily_features_active.feature_vector.
        conn:            psycopg connection.
        k:               Number of nearest neighbours. Default 20.
        feature_version: Corpus to rank against. Defaults to CANONICAL_FEATURE_VERSION
                         (currently 'v0.5.0-rebuilt', 735 rows).
        ticker:          Instrument. Default 'SPX'.
        exclude_date:    ISO date string to exclude from ranking (for historical
                         lookups where today IS in the corpus).
        regime_kind:     Regime label for the response ('magnet-above', etc.).
                         If None, derived from the boolean flags in today_features.

    Returns:
        Dict with outcome_status, k, k_with_outcomes, touch_rate,
        touch_ci_lower/upper, close_rate, mean_days_to_reach,
        mean_excursion_pct, regime_kind, note.
    """
    rows   = _rank_analogues_with_outcomes(
        today_features, conn, k, feature_version,
        ticker=ticker, exclude_date=exclude_date,
    )
    result = _aggregate_outcomes(rows)
    result["k"]           = k
    result["regime_kind"] = regime_kind or _infer_regime_kind(today_features)
    return result


def _infer_regime_kind(features: dict) -> Optional[str]:
    """Derive regime label from the boolean flags in the feature vector."""
    if features.get("is_magnet_day"):
        direction = features.get("magnet_direction_signed")
        try:
            return "magnet-above" if float(direction) >= 0 else "magnet-below"
        except (TypeError, ValueError):
            return "magnet"
    if features.get("is_pin_day"):
        return "magnetic-pin"
    if features.get("is_bounded_day"):
        return "bounded"
    if features.get("is_amplification_day"):
        return "amplification"
    if features.get("is_untethered_day"):
        return "untethered"
    return None
