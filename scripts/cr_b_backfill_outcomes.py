#!/usr/bin/env python3
"""CR-022 (CR-B) — Outcome computation backfill for bt_daily_outcomes.

For each target date (bt_daily_features row at v0.5.0-rebuilt not yet in
bt_daily_outcomes), derives dominant_bucket and drift_target from DB sources,
fetches pre-aggregated RTH daily OHLC bars, and calls compute_outcome() to
produce the outcome metrics.

All writes use ON CONFLICT DO NOTHING — safe on re-run.

Note on actual_realized_em_pct: the denominator is always implied_move_1d
(1-day expected move), regardless of the horizon length for the row's bucket.
This keeps values comparable across analogues with similar horizons but means
the metric is NOT interpretable as "did the structural prediction play out"
for >1-day buckets. CR-C consumers should weight accordingly.

Usage:
    python scripts/cr_b_backfill_outcomes.py
    python scripts/cr_b_backfill_outcomes.py --limit 3
    python scripts/cr_b_backfill_outcomes.py --from-date 2023-06-01 --to-date 2023-06-30
    python scripts/cr_b_backfill_outcomes.py --dry-run

Exit: 0 on success; 1 if any date failed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from packages.shared.backfill_safety import (
    assert_role_or_die,
    backfill_run,
    get_backfill_db_conn,
    update_run_progress,
    update_run_smoke,
)
from packages.shared.outcomes import compute_outcome, pick_drift_target

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

FEATURE_VERSION = "v0.5.0-rebuilt"

# Rows in bt_daily_features not yet in bt_daily_outcomes.
_TARGET_DATES_SQL = """
    SELECT f.trade_date, f.regime_at_classification, f.feature_vector
    FROM bt_daily_features f
    WHERE f.ticker = %s
      AND f.feature_version = %s
      AND f.active = TRUE
      AND NOT EXISTS (
          SELECT 1 FROM bt_daily_outcomes o
          WHERE o.ticker = f.ticker
            AND o.trade_date = f.trade_date
            AND o.feature_version = f.feature_version
      )
    ORDER BY f.trade_date
"""

_LANDSCAPE_SQL = """
    SELECT trade_date, walls, table_spot
    FROM orats_gex_landscape
    WHERE ticker = %s
      AND trade_date = ANY(%s)
"""

# Aggregate 1-minute RTH bars to daily OHLC.
# open  = first bar's open, high = max(high), low = min(low), close = last bar's close.
# RTH window: 06:30–13:00 PT = 13:30–20:00 UTC (matches Bars/service.py convention).
_RTH_BARS_SQL = """
    WITH rth AS (
        SELECT
            (datetime AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')::date AS session_date,
            open, high, low, close,
            ROW_NUMBER() OVER (
                PARTITION BY
                    (datetime AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')::date
                ORDER BY datetime ASC
            ) AS rn_asc,
            ROW_NUMBER() OVER (
                PARTITION BY
                    (datetime AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')::date
                ORDER BY datetime DESC
            ) AS rn_desc
        FROM ironbeam_es_1m_bars
        WHERE
            (datetime AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')::time
                BETWEEN '06:30:00' AND '13:00:00'
          AND (datetime AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')::date
                BETWEEN %s AND %s
    )
    SELECT
        session_date,
        MAX(CASE WHEN rn_asc  = 1 THEN open  END) AS open,
        MAX(high)                                   AS high,
        MIN(low)                                    AS low,
        MAX(CASE WHEN rn_desc = 1 THEN close END)  AS close
    FROM rth
    GROUP BY session_date
    ORDER BY session_date
"""

_INSERT_OUTCOME_SQL = """
    INSERT INTO bt_daily_outcomes (
        ticker, trade_date, feature_version,
        regime_kind_at_classification,
        dominant_bucket_at_classification,
        horizon_sessions,
        horizon_end_date,
        outcome_status,
        reached_touch,
        reached_close,
        days_to_reach,
        max_excursion_in_direction,
        final_close_distance_from_target,
        actual_realized_em_pct,
        session_open_t0,
        backfill_run_id
    ) VALUES (
        %s, %s, %s,
        %s, %s, %s, %s, %s,
        %s, %s, %s, %s, %s, %s,
        %s, %s
    )
    ON CONFLICT (ticker, trade_date, feature_version) DO NOTHING
"""


def _load_env() -> None:
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return
    with open(env_path) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())


def _derive_dominant_bucket(feature_vector: dict) -> Optional[str]:
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


# _pick_drift_target moved to packages/shared/outcomes.pick_drift_target (CR-I Step 2b).
# Imported above; this alias keeps existing call sites in this file unchanged.
_pick_drift_target = pick_drift_target


# Distance threshold for magnetic-pin sanity check.
# Matches near_dist_pts default in classify_regime — a dominant wall further
# than this from spot is not acting as a pin, so the outcome would be garbage.
_PIN_MAX_DISTANCE_PTS = 30.0


def _direction_sanity(
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


def _get_target_rows(
    conn,
    ticker: str,
    from_date: Optional[dt.date],
    to_date: Optional[dt.date],
    limit: Optional[int],
) -> list[tuple]:
    with conn.cursor() as cur:
        cur.execute(_TARGET_DATES_SQL, (ticker, FEATURE_VERSION))
        rows = cur.fetchall()
    if from_date:
        rows = [r for r in rows if r[0] >= from_date]
    if to_date:
        rows = [r for r in rows if r[0] <= to_date]
    if limit:
        rows = rows[:limit]
    return rows


def _fetch_landscape(conn, ticker: str, dates: list[dt.date]) -> dict[dt.date, dict]:
    """Return {trade_date: {"walls": [...], "table_spot": float}} for all dates."""
    with conn.cursor() as cur:
        cur.execute(_LANDSCAPE_SQL, (ticker, dates))
        rows = cur.fetchall()
    result = {}
    for (d, walls, table_spot) in rows:
        result[d] = {
            "walls":      walls if isinstance(walls, list) else [],
            "table_spot": float(table_spot) if table_spot is not None else None,
        }
    return result


def _fetch_daily_bars(conn, bar_from: dt.date, bar_to: dt.date) -> pd.DataFrame:
    """Aggregate RTH 1m bars to daily OHLC over [bar_from, bar_to]."""
    with conn.cursor() as cur:
        cur.execute(_RTH_BARS_SQL, (bar_from, bar_to))
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close"])
    df = pd.DataFrame(rows, columns=["session_date", "open", "high", "low", "close"])
    df = df.set_index("session_date")
    df.index = [d for d in df.index]        # keep as date objects
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _run_smoke(conn, ticker: str) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*),
                   COUNT(*) FILTER (WHERE outcome_status = 'computed'),
                   COUNT(*) FILTER (WHERE outcome_status = 'pending_history'),
                   COUNT(*) FILTER (WHERE outcome_status = 'na_regime'),
                   COUNT(*) FILTER (WHERE outcome_status = 'na_data'),
                   MIN(trade_date), MAX(trade_date)
            FROM bt_daily_outcomes
            WHERE ticker = %s AND feature_version = %s
            """,
            (ticker, FEATURE_VERSION),
        )
        total, n_computed, n_pending, n_na_regime, n_na_data, min_d, max_d = cur.fetchone()

    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM bt_daily_features "
            "WHERE ticker = %s AND feature_version = %s AND active = TRUE",
            (ticker, FEATURE_VERSION),
        )
        features_count = int(cur.fetchone()[0])

    # Every row written by this backfill carries backfill_run_id — no status filter.
    # A NULL here means a row was inserted outside this script, which is a real bug.
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*) FROM bt_daily_outcomes
            WHERE ticker = %s AND feature_version = %s
              AND backfill_run_id IS NULL
            """,
            (ticker, FEATURE_VERSION),
        )
        null_run_id = int(cur.fetchone()[0])

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*) FROM bt_daily_outcomes o
            WHERE o.ticker = %s AND o.feature_version = %s
              AND o.outcome_status NOT IN ('pending_history', 'na_regime', 'na_data')
              AND o.horizon_end_date > (
                  SELECT MAX(
                      (datetime AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')::date
                  ) FROM ironbeam_es_1m_bars
              )
            """,
            (ticker, FEATURE_VERSION),
        )
        future_horizon = int(cur.fetchone()[0])

    total_int = int(total) if total else 0
    # Expected ~0% divergence: one INSERT per feature row; ≤1% is defensive.
    pct_diff  = abs(total_int - features_count) / max(features_count, 1) * 100

    return {
        "outcomes_total":       total_int,
        "outcomes_computed":    int(n_computed or 0),
        "outcomes_pending":     int(n_pending or 0),
        "outcomes_na_regime":   int(n_na_regime or 0),
        "outcomes_na_data":     int(n_na_data or 0),
        "features_total":       features_count,
        "row_count_pct_diff":   round(pct_diff, 2),
        "null_run_id_count":    null_run_id,
        "future_horizon_count": future_horizon,
        "date_min":             str(min_d) if min_d else None,
        "date_max":             str(max_d) if max_d else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ticker",    default="SPX")
    ap.add_argument("--from-date", default=None, metavar="YYYY-MM-DD")
    ap.add_argument("--to-date",   default=None, metavar="YYYY-MM-DD")
    ap.add_argument("--limit",     type=int, default=None, metavar="N")
    ap.add_argument("--dry-run",   action="store_true")
    ap.add_argument("--batch",     type=int, default=50, metavar="N",
                    help="Update rows_inserted every N dates (default: 50)")
    args = ap.parse_args()

    ticker    = args.ticker
    from_date = dt.date.fromisoformat(args.from_date) if args.from_date else None
    to_date   = dt.date.fromisoformat(args.to_date)   if args.to_date   else None

    _load_env()
    conn = get_backfill_db_conn()
    assert_role_or_die(conn)

    target_rows = _get_target_rows(conn, ticker, from_date, to_date, args.limit)

    print(f"=== CR-022 Outcome Backfill ===")
    print(f"Ticker: {ticker}   Feature version: {FEATURE_VERSION}")
    print(f"Target dates: {len(target_rows)}")
    if target_rows:
        print(f"Range: {target_rows[0][0]} → {target_rows[-1][0]}")
    if args.dry_run:
        print("[DRY RUN — no writes]")
    print()

    if not target_rows:
        print("Nothing to do.")
        conn.close()
        sys.exit(0)

    dates = [r[0] for r in target_rows]

    # Load landscape and bars once across the full range.
    landscape_by_date = _fetch_landscape(conn, ticker, dates)

    # bar_to extends 90 calendar days past the last trade_date to cover 60-session horizons.
    bar_from = min(dates)
    bar_to   = max(dates) + dt.timedelta(days=90)
    daily_bars = _fetch_daily_bars(conn, bar_from, bar_to)
    log.info("Loaded %d RTH daily sessions (%s → %s)", len(daily_bars), bar_from, bar_to)

    if args.dry_run:
        for trade_date, regime, fv in target_rows:
            landscape    = landscape_by_date.get(trade_date, {})
            bucket       = _derive_dominant_bucket(fv or {})
            drift_target = _pick_drift_target(landscape.get("walls") or [])
            spot         = landscape.get("table_spot")
            print(f"  [dry-run] {trade_date} regime={regime!r} "
                  f"bucket={bucket!r} target={drift_target} spot={spot}")
        conn.close()
        sys.exit(0)

    n_inserted = 0
    n_skipped  = 0
    n_failed   = 0
    failed_dates: list[str] = []

    with backfill_run(conn, "CR-022") as run_id:
        print(f"Run ID: {run_id}")

        for i, (trade_date, regime, fv) in enumerate(target_rows, 1):
            try:
                feature_vector  = fv or {}
                landscape       = landscape_by_date.get(trade_date, {})
                walls           = landscape.get("walls") or []
                table_spot      = landscape.get("table_spot")
                dominant_bucket = _derive_dominant_bucket(feature_vector)
                drift_target    = _pick_drift_target(walls)
                expected_move   = feature_vector.get("implied_move_1d")
                if expected_move is not None:
                    try:
                        expected_move = float(expected_move)
                    except (TypeError, ValueError):
                        expected_move = None

                # Runner-side direction sanity: skip to na_data if the dominant wall
                # is on the wrong side of spot (directional) or too far from spot (pin).
                sanity_ok = True
                if drift_target is not None and table_spot is not None:
                    sanity_ok = _direction_sanity(regime, drift_target, table_spot, trade_date)

                # session_open_t0: FIRST RTH bar open on trade_date.
                # Computable from daily_bars regardless of outcome_status —
                # must be populated for all new rows so the pl-data endpoint
                # can normalise analogue returns without a separate backfill step.
                session_open_t0 = (
                    float(daily_bars.loc[trade_date, "open"])
                    if trade_date in daily_bars.index
                    else None
                )

                if not sanity_ok:
                    outcome = {
                        "regime_kind_at_classification":    regime,
                        "dominant_bucket_at_classification": dominant_bucket,
                        "horizon_sessions":                 None,
                        "horizon_end_date":                 None,
                        "outcome_status":                   "na_data",
                        "reached_touch":                    None,
                        "reached_close":                    None,
                        "days_to_reach":                    None,
                        "max_excursion_in_direction":       None,
                        "final_close_distance_from_target": None,
                        "actual_realized_em_pct":           None,
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

                with conn.cursor() as cur:
                    cur.execute(_INSERT_OUTCOME_SQL, (
                        ticker,
                        trade_date,
                        FEATURE_VERSION,
                        outcome["regime_kind_at_classification"],
                        outcome["dominant_bucket_at_classification"],
                        outcome["horizon_sessions"],
                        outcome["horizon_end_date"],
                        outcome["outcome_status"],
                        outcome["reached_touch"],
                        outcome["reached_close"],
                        outcome["days_to_reach"],
                        outcome["max_excursion_in_direction"],
                        outcome["final_close_distance_from_target"],
                        outcome["actual_realized_em_pct"],
                        session_open_t0,
                        run_id,
                    ))
                    if cur.rowcount == 1:
                        n_inserted += 1
                    else:
                        n_skipped += 1

            except Exception as exc:
                n_failed += 1
                failed_dates.append(str(trade_date))
                log.error("ERROR %s: %s", trade_date, exc, exc_info=True)

            if i % args.batch == 0 or i == len(target_rows):
                update_run_progress(conn, run_id, n_inserted)
                print(f"  [{i}/{len(target_rows)}] "
                      f"inserted={n_inserted} skipped={n_skipped} failed={n_failed}")

        smoke = _run_smoke(conn, ticker)
        smoke.update({
            "n_target":     len(target_rows),
            "n_inserted":   n_inserted,
            "n_skipped":    n_skipped,
            "n_failed":     n_failed,
            "failed_dates": failed_dates,
        })
        assessment = (
            f"{n_inserted} rows inserted, {n_skipped} skipped, {n_failed} failed"
            + (f" — FAILURES: {', '.join(failed_dates)}" if failed_dates else " — clean run")
        )
        update_run_smoke(conn, run_id, smoke, assessment)

        print(f"\n=== DONE ===")
        print(f"inserted={n_inserted}  skipped={n_skipped}  failed={n_failed}")
        print(f"Total outcome rows:    {smoke['outcomes_total']}")
        print(f"  computed={smoke['outcomes_computed']}  pending={smoke['outcomes_pending']}")
        print(f"  na_regime={smoke['outcomes_na_regime']}  na_data={smoke['outcomes_na_data']}")
        print(f"Features total:        {smoke['features_total']}  "
              f"row_count_pct_diff={smoke['row_count_pct_diff']}%")
        print(f"Null run_id count:     {smoke['null_run_id_count']}")
        print(f"Future horizon count:  {smoke['future_horizon_count']}")

    conn.close()
    sys.exit(1 if n_failed > 0 else 0)


if __name__ == "__main__":
    main()
