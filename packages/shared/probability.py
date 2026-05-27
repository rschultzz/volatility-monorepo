"""Structural probability computation (CR-C) + post-touch distribution (CR-I).

Three layers per CR-021 Lesson 3 (compute / persist separation):
  stats.wilson_ci                       → pure math, in stats.py
  _aggregate_outcomes                   → pure math, synthetic-testable
  classify_post_touch_positions         → pure math, synthetic-testable
  _pattern_label_from_fractions         → pure math, synthetic-testable
  aggregate_post_touch_distribution     → pure math, synthetic-testable
  _rank_analogues_with_outcomes         → DB I/O shell
  compute_structural_probability        → orchestrator (thin)

Public entry points:
    compute_structural_probability(today_features, conn, ...) → dict
    classify_post_touch_positions(days_to_reach, horizon_bars, ...) → dict
    aggregate_post_touch_distribution(analogues_with_outcomes, anchor_bucket, ...) → dict
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Optional

from packages.shared.canonical_version import CANONICAL_FEATURE_VERSION
from packages.shared.knn import feature_stats, rank_analogues
from packages.shared.stats import wilson_ci

log = logging.getLogger(__name__)


# ── Bucket label helpers ─────────────────────────────────────────────────────


def _derive_anchor_bucket(features: dict) -> Optional[str]:
    """Derive dominant bucket label from feature vector dominance fields.

    Replicates the CR-B runner's _derive_dominant_bucket logic.
    Returns None if no dominance values are present.
    """
    candidates = {
        "0DTE":    features.get("dominance_0DTE",   0.0) or 0.0,
        "1-7 DTE": features.get("dominance_1_7",    0.0) or 0.0,
        "8-30 DTE": features.get("dominance_8_30",  0.0) or 0.0,
        "30+ DTE":  features.get("dominance_30plus", 0.0) or 0.0,
    }
    if not any(candidates.values()):
        return None
    return max(candidates, key=candidates.__getitem__)


# ── Post-touch position classifier (CR-I Step 1a) ───────────────────────────


def classify_post_touch_positions(
    days_to_reach: int,
    horizon_bars,          # pandas DataFrame or any iloc-supporting sequence
    drift_target: float,
    tolerance: float,
    timeframes_sessions: tuple[int, ...] = (1, 5, 15),
) -> dict[int, Optional[int]]:
    """Classify the session-close position at each post-touch timeframe.

    For an analogue that touched the magnet, returns where price closed at
    T+1, T+5, and T+15 sessions after touch, relative to the tolerance band.

    Args:
        days_to_reach:       0-indexed per outcomes.py convention; 0 = magnet
                             touched on trade_date itself (same session as the
                             structural read). The touch session is at
                             horizon_bars.iloc[days_to_reach].
        horizon_bars:        Ordered sequence of session bars starting from
                             trade_date (index 0 = trade_date). Supports
                             pandas DataFrame with 'close' column or any
                             object where horizon_bars.iloc[i]['close'] works.
                             T+N close is at iloc[days_to_reach + N].
        drift_target:        Magnet price level (the GEX wall target).
        tolerance:           Half-width of the "at" band: 0.25 × implied_move_1d.
        timeframes_sessions: Timeframes to classify. Default (1, 5, 15).

    Returns:
        Dict mapping timeframe int → position int:
          -1  close < drift_target - tolerance   (below tolerance)
           0  |close - drift_target| <= tolerance (within tolerance, "at")
          +1  close > drift_target + tolerance   (above tolerance)
          None  bar not available (days_to_reach + N is out of bounds)

    Boundary note: the "at" band uses `abs(close - drift_target) <= tolerance`
    (<=, inclusive). With standard float arithmetic, exact equality is rare;
    the inclusive boundary means ties on the exact threshold count as "at".
    """
    n_bars = len(horizon_bars)
    result: dict[int, Optional[int]] = {}

    lower = drift_target - tolerance
    upper = drift_target + tolerance

    for tf in timeframes_sessions:
        idx = days_to_reach + tf
        if idx >= n_bars:
            result[tf] = None
            continue
        try:
            close = float(horizon_bars.iloc[idx]["close"])
        except (IndexError, KeyError, TypeError, ValueError):
            result[tf] = None
            continue
        # NaN guard (float('nan') != float('nan'))
        if close != close:
            result[tf] = None
            continue

        if close < lower:
            result[tf] = -1
        elif close > upper:
            result[tf] = 1
        else:
            result[tf] = 0

    return result


# ── Pattern label classifier (CR-I Step 1b helper) ───────────────────────────


def _pattern_label_from_fractions(fractions: dict) -> str:
    """Deterministic decision tree: 9-cell fraction dict → pattern label.

    Args:
        fractions: {
            "t1":  {"below": float, "at": float, "above": float},
            "t5":  {...},
            "t15": {...},
        }
        None values are treated as 0.0 (pre-backfill / missing timeframe data).

    Returns one of: "stepping-stone", "touch-and-reject", "touch-and-pin",
    "overshoot-then-revert", "slow-revert", "mixed".

    Decision order (earlier rules take priority):
      1. stepping-stone:        above > 0.50 at ALL THREE timeframes
      2. touch-and-reject:      below > 0.50 at ALL THREE timeframes
      3. touch-and-pin:         at-fraction STRICTLY largest in >= 2 of 3 tf
      4. overshoot-then-revert: above > 0.50 at T+1 AND T+5, below > 0.50 at T+15
      5. slow-revert:           above > 0.50 at T+1, monotonically decreasing
      6. mixed:                 everything else

    Boundary note on step 1/2: uses strict ">" (> 0.50), not ">=". A 50/50
    split (above = 0.50) does not trigger stepping-stone; it falls through.
    This matches the spec's intent — a true majority is required.
    """
    THRESH = 0.50

    def _get(tf_label: str, side: str) -> float:
        return fractions.get(tf_label, {}).get(side) or 0.0

    above_t1  = _get("t1",  "above");  above_t5  = _get("t5",  "above");  above_t15 = _get("t15", "above")
    below_t1  = _get("t1",  "below");  below_t5  = _get("t5",  "below");  below_t15 = _get("t15", "below")
    at_t1     = _get("t1",  "at");     at_t5     = _get("t5",  "at");     at_t15    = _get("t15", "at")

    # 1. stepping-stone
    if above_t1 > THRESH and above_t5 > THRESH and above_t15 > THRESH:
        return "stepping-stone"

    # 2. touch-and-reject
    if below_t1 > THRESH and below_t5 > THRESH and below_t15 > THRESH:
        return "touch-and-reject"

    # 3. touch-and-pin: at-fraction strictly largest in >= 2 timeframes
    at_dominant = (
        (at_t1  > above_t1  and at_t1  > below_t1)
        + (at_t5  > above_t5  and at_t5  > below_t5)
        + (at_t15 > above_t15 and at_t15 > below_t15)
    )
    if at_dominant >= 2:
        return "touch-and-pin"

    # 4. overshoot-then-revert: above dominant at T+1 and T+5, below dominant at T+15
    if above_t1 > THRESH and above_t5 > THRESH and below_t15 > THRESH:
        return "overshoot-then-revert"

    # 5. slow-revert: above > 0.50 at T+1, monotonically decreasing across timeframes
    if above_t1 > THRESH and above_t5 < above_t1 and above_t15 < above_t5:
        return "slow-revert"

    # 6. mixed
    return "mixed"


# ── Bucket-filtered post-touch aggregation (CR-I Step 1b) ───────────────────


def aggregate_post_touch_distribution(
    analogues_with_outcomes: list[dict],
    anchor_bucket: str,
    fallback_threshold: int = 7,
    pooled_minimum: int = 4,
) -> dict:
    """Aggregate post-touch positions across K=20 analogues.

    Produces the 9-cell below/at/above fraction matrix with Wilson CIs across
    T+1/T+5/T+15 timeframes, bucket-filtered for physical-regime consistency.

    Args:
        analogues_with_outcomes: List of outcome dicts (same format as
            _rank_analogues_with_outcomes output). Must include:
              - outcome_status: str ('computed', 'na_regime', etc.)
              - reached_touch: bool | None
              - dominant_bucket_at_classification: str | None
              - position_t1_post_touch: int | None  (-1/0/+1 or None)
              - position_t5_post_touch: int | None
              - position_t15_post_touch: int | None
            Missing keys are treated as None (pre-backfill safe).

        anchor_bucket: The anchor day's dominant bucket label (e.g. '8-30 DTE').
            Used for both the 0DTE pre-check and the same-bucket filter.

        fallback_threshold: Minimum same-bucket touchers to use strict mode.
            Default 7 (revised from spec's 10 based on Step 0 empirical check;
            mean ~9.7 same-bucket analogues × 82% touch rate ≈ 8 expected
            touchers — threshold 7 keeps ~70% of anchors in strict mode).

        pooled_minimum: Minimum total touchers for pooled-fallback mode.
            Default 4 (revised from 5 based on Step 0 empirical check).

    Returns dict with:
        filter_mode:       "strict" | "pooled-fallback" | "insufficient" |
                           "zero_dte_corpus_insufficient"
        denominator_t1/t5/t15: int (may differ due to NULL handling)
        same_bucket_n:     int
        total_touchers:    int
        fractions:         {"t1": {"below": f, "at": f, "above": f}, ...}
                           or None if insufficient/0DTE/pre-backfill
        wilson_cis:        {"t1": {"below": (lo,hi), ...}, ...} or None
        pattern_label:     str (one of six labels) or None

    Pre-backfill behaviour: if position columns are absent from the dicts
    (key missing or None), denominators are 0 and pattern_label is None.
    The filter_mode still reflects the bucket-filter decision so the UI
    can display "post-touch data not yet available" without crashing.

    0DTE pre-check: if anchor_bucket == "0DTE", returns immediately with
    filter_mode "zero_dte_corpus_insufficient". Only 3 0DTE days exist in
    the current corpus; no 0DTE anchor will find same-bucket analogues in
    K=20. This is a structural corpus-coverage fact, not a per-anchor issue.
    """
    # ── 0DTE corpus pre-check ────────────────────────────────────────────────
    if anchor_bucket == "0DTE":
        total_touchers = sum(
            1 for r in analogues_with_outcomes
            if r.get("outcome_status") == "computed" and r.get("reached_touch")
        )
        return {
            "filter_mode":     "zero_dte_corpus_insufficient",
            "denominator_t1":  0,
            "denominator_t5":  0,
            "denominator_t15": 0,
            "same_bucket_n":   0,
            "total_touchers":  total_touchers,
            "fractions":       None,
            "wilson_cis":      None,
            "pattern_label":   None,
        }

    # ── Filter to computed touchers ──────────────────────────────────────────
    touchers = [
        r for r in analogues_with_outcomes
        if r.get("outcome_status") == "computed" and r.get("reached_touch")
    ]
    total_touchers = len(touchers)

    # ── Same-bucket count ────────────────────────────────────────────────────
    same_bucket_touchers = [
        r for r in touchers
        if r.get("dominant_bucket_at_classification") == anchor_bucket
    ]
    same_bucket_n = len(same_bucket_touchers)

    # ── Determine filter mode and working pool ───────────────────────────────
    if same_bucket_n >= fallback_threshold:
        pool        = same_bucket_touchers
        filter_mode = "strict"
    elif total_touchers >= pooled_minimum:
        pool        = touchers
        filter_mode = "pooled-fallback"
    else:
        return {
            "filter_mode":     "insufficient",
            "denominator_t1":  0,
            "denominator_t5":  0,
            "denominator_t15": 0,
            "same_bucket_n":   same_bucket_n,
            "total_touchers":  total_touchers,
            "fractions":       None,
            "wilson_cis":      None,
            "pattern_label":   None,
        }

    # ── Per-timeframe aggregation ────────────────────────────────────────────
    fractions:   dict[str, dict] = {}
    wilson_cis_: dict[str, dict] = {}
    denominators: dict[str, int] = {}

    for tf in (1, 5, 15):
        col_key  = f"position_t{tf}_post_touch"
        tf_label = f"t{tf}"

        # Filter to rows where this timeframe's position is not None
        with_pos = [r for r in pool if r.get(col_key) is not None]
        n = len(with_pos)
        denominators[f"denominator_t{tf}"] = n

        if n == 0:
            # Pre-backfill: position columns not yet populated
            fractions[tf_label]    = {"below": None, "at": None, "above": None}
            wilson_cis_[tf_label]  = {
                "below": (None, None),
                "at":    (None, None),
                "above": (None, None),
            }
            continue

        n_below = sum(1 for r in with_pos if r.get(col_key) == -1)
        n_at    = sum(1 for r in with_pos if r.get(col_key) ==  0)
        n_above = sum(1 for r in with_pos if r.get(col_key) ==  1)

        fractions[tf_label] = {
            "below": round(n_below / n, 4),
            "at":    round(n_at    / n, 4),
            "above": round(n_above / n, 4),
        }
        wilson_cis_[tf_label] = {
            "below": wilson_ci(n_below, n),
            "at":    wilson_ci(n_at,    n),
            "above": wilson_ci(n_above, n),
        }

    # ── Pattern label (only when all timeframes have data) ───────────────────
    all_have_data = all(denominators[f"denominator_t{tf}"] > 0 for tf in (1, 5, 15))
    pattern_label = _pattern_label_from_fractions(fractions) if all_have_data else None

    return {
        "filter_mode":     filter_mode,
        "denominator_t1":  denominators["denominator_t1"],
        "denominator_t5":  denominators["denominator_t5"],
        "denominator_t15": denominators["denominator_t15"],
        "same_bucket_n":   same_bucket_n,
        "total_touchers":  total_touchers,
        "fractions":       fractions,
        "wilson_cis":      wilson_cis_,
        "pattern_label":   pattern_label,
    }


# ── Existing outcome aggregation (CR-C) ──────────────────────────────────────


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

    DB I/O layer — pure aggregation math is in _aggregate_outcomes and
    aggregate_post_touch_distribution.

    Returns list of dicts: trade_date, distance, outcome_status,
    reached_touch, reached_close, days_to_reach,
    max_excursion_in_direction, implied_move_1d,
    dominant_bucket_at_classification,
    position_t1_post_touch, position_t5_post_touch, position_t15_post_touch,
    session_open_t1,  session_high_t1,  session_low_t1,  session_close_t1,
    session_open_t5,  session_high_t5,  session_low_t5,  session_close_t5,
    session_open_t15, session_high_t15, session_low_t15, session_close_t15.

    position_tN_post_touch: SMALLINT -1/0/+1 or NULL. NULL before CR-I Step 2b
    backfill runs (aggregate_post_touch_distribution treats None as missing via
    .get(), producing fractions=None / pattern_label=None gracefully).
    After Step 2b backfill: carries the classified post-touch close position.

    session_*_tN: REAL or NULL. Populated by CR-G Step 0-A backfill for
    outcome_status IN ('computed', 'na_regime'). NULL for corpus-end rows
    (T+15 near end of bars) and quarterly-expiry Fridays (no RTH bars).
    Existing callers continue to work — new keys are accessed via .get().
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

    # 3. Fetch outcomes for the top-K dates (includes CR-G OHLC columns)
    top_dates = [dt.date.fromisoformat(d) for (d, _) in ranked]
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT trade_date, outcome_status,
                   reached_touch, reached_close, days_to_reach,
                   max_excursion_in_direction,
                   dominant_bucket_at_classification,
                   position_t1_post_touch,
                   position_t5_post_touch,
                   position_t15_post_touch,
                   session_open_t1,  session_high_t1,  session_low_t1,  session_close_t1,
                   session_open_t5,  session_high_t5,  session_low_t5,  session_close_t5,
                   session_open_t15, session_high_t15, session_low_t15, session_close_t15
            FROM bt_daily_outcomes_active
            WHERE ticker = %s AND feature_version = %s
              AND trade_date = ANY(%s)
            """,
            (ticker, feature_version, top_dates),
        )
        outcome_rows = cur.fetchall()

    def _f(v) -> Optional[float]:
        return float(v) if v is not None else None

    outcomes_by_date = {
        row[0].isoformat(): {
            "outcome_status":                    row[1],
            "reached_touch":                     row[2],
            "reached_close":                     row[3],
            "days_to_reach":                     row[4],
            "max_excursion_in_direction":        _f(row[5]),
            "dominant_bucket_at_classification": row[6],
            "position_t1_post_touch":            row[7],   # SMALLINT -1/0/+1 or NULL
            "position_t5_post_touch":            row[8],   # populated by CR-I Step 2b backfill
            "position_t15_post_touch":           row[9],
            # CR-G Step 0-A: session OHLC at T+1, T+5, T+15 sessions after trade_date
            "session_open_t1":                   _f(row[10]),
            "session_high_t1":                   _f(row[11]),
            "session_low_t1":                    _f(row[12]),
            "session_close_t1":                  _f(row[13]),
            "session_open_t5":                   _f(row[14]),
            "session_high_t5":                   _f(row[15]),
            "session_low_t5":                    _f(row[16]),
            "session_close_t5":                  _f(row[17]),
            "session_open_t15":                  _f(row[18]),
            "session_high_t15":                  _f(row[19]),
            "session_low_t15":                   _f(row[20]),
            "session_close_t15":                 _f(row[21]),
        }
        for row in outcome_rows
    }

    # 4. Assemble; implied_move_1d comes from the feature vector already in memory.
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
            "trade_date":                        trade_date_iso,
            "distance":                          float(distance),
            "outcome_status":                    outcome.get("outcome_status"),
            "reached_touch":                     outcome.get("reached_touch"),
            "reached_close":                     outcome.get("reached_close"),
            "days_to_reach":                     outcome.get("days_to_reach"),
            "max_excursion_in_direction":        outcome.get("max_excursion_in_direction"),
            "implied_move_1d":                   implied_move_1d,
            "dominant_bucket_at_classification": outcome.get("dominant_bucket_at_classification"),
            "position_t1_post_touch":            outcome.get("position_t1_post_touch"),
            "position_t5_post_touch":            outcome.get("position_t5_post_touch"),
            "position_t15_post_touch":           outcome.get("position_t15_post_touch"),
            # CR-G Step 0-A OHLC (None when bars unavailable)
            "session_open_t1":                   outcome.get("session_open_t1"),
            "session_high_t1":                   outcome.get("session_high_t1"),
            "session_low_t1":                    outcome.get("session_low_t1"),
            "session_close_t1":                  outcome.get("session_close_t1"),
            "session_open_t5":                   outcome.get("session_open_t5"),
            "session_high_t5":                   outcome.get("session_high_t5"),
            "session_low_t5":                    outcome.get("session_low_t5"),
            "session_close_t5":                  outcome.get("session_close_t5"),
            "session_open_t15":                  outcome.get("session_open_t15"),
            "session_high_t15":                  outcome.get("session_high_t15"),
            "session_low_t15":                   outcome.get("session_low_t15"),
            "session_close_t15":                 outcome.get("session_close_t15"),
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

    Also computes the post-touch close distribution via
    aggregate_post_touch_distribution and includes it as 'post_touch' in
    the response. Pre-Step-2 (before schema migration + backfill), the
    post_touch block will have None fractions/pattern_label — the filter_mode
    and denominator fields still reflect the bucket-filter decision so the
    frontend can display "data not yet available" without crashing.

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
        mean_excursion_pct, regime_kind, note, post_touch.
    """
    rows   = _rank_analogues_with_outcomes(
        today_features, conn, k, feature_version,
        ticker=ticker, exclude_date=exclude_date,
    )
    result = _aggregate_outcomes(rows)
    result["k"]           = k
    result["regime_kind"] = regime_kind or _infer_regime_kind(today_features)

    # ── Post-touch distribution (CR-I) ────────────────────────────────────────
    anchor_bucket = _derive_anchor_bucket(today_features)
    if anchor_bucket is not None:
        result["post_touch"] = aggregate_post_touch_distribution(
            rows, anchor_bucket,
        )
    else:
        result["post_touch"] = None

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
