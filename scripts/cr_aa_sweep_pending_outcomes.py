#!/usr/bin/env python3
"""CR-AA — Pending outcome sweep: promote matured pending_history rows to computed/na_data.

Finds pending_history rows in bt_daily_outcomes whose horizon window has closed
(enough RTH sessions now exist in ironbeam_es_1m_bars), re-runs compute_outcome
via the shared helper, and UPDATEs each row to its terminal status.

Design constraints:
  - Promotes pending_history ONLY (WHERE outcome_status = 'pending_history').
    Never touches computed, na_regime, or na_data.
  - Horizon-gated: only rows where the Nth session on/after trade_date exists
    and is <= the latest fully-closed RTH session in ironbeam_es_1m_bars.
    (horizon_end_date is NULL for pending rows; maturity is computed Python-side
    from the actual RTH session list — see Step-0 finding in CR-036 spec.)
  - Uses get_backfill_db_conn() + assert_role_or_die + backfill_run scaffolding.
  - UPDATE guard: WHERE outcome_status = 'pending_history' prevents clobbering
    a row that changed between SELECT and UPDATE.
  - Idempotent: a second immediate run promotes 0 rows.

Usage:
    python scripts/cr_aa_sweep_pending_outcomes.py
    python scripts/cr_aa_sweep_pending_outcomes.py --dry-run
    python scripts/cr_aa_sweep_pending_outcomes.py --limit 5
    python scripts/cr_aa_sweep_pending_outcomes.py --from-date 2026-03-01

Exit: 0 on success; 1 if any row failed.
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
from packages.shared.buckets import bucket_sessions
from packages.shared.outcomes_runner import compute_outcome_for_date

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

FEATURE_VERSION = "v0.5.0-rebuilt"

_PENDING_ROWS_SQL = """
    SELECT o.trade_date,
           o.regime_kind_at_classification,
           o.dominant_bucket_at_classification,
           f.feature_vector
    FROM bt_daily_outcomes o
    JOIN bt_daily_features f
        ON f.ticker          = o.ticker
       AND f.trade_date      = o.trade_date
       AND f.feature_version = o.feature_version
    WHERE o.ticker          = %s
      AND o.feature_version = %s
      AND o.outcome_status  = 'pending_history'
    ORDER BY o.trade_date
"""

_LANDSCAPE_SQL = """
    SELECT trade_date, walls, table_spot
    FROM orats_gex_landscape
    WHERE ticker = %s
      AND trade_date = ANY(%s)
"""

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

_RTH_SESSION_DATES_SQL = """
    SELECT DISTINCT (datetime AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')::date
    FROM ironbeam_es_1m_bars
    WHERE (datetime AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')::time
              BETWEEN '06:30:00' AND '13:00:00'
    ORDER BY 1
"""

# UPDATE guard: WHERE outcome_status = 'pending_history' prevents clobbering
# a row that changed between our SELECT and this UPDATE.
_UPDATE_OUTCOME_SQL = """
    UPDATE bt_daily_outcomes
    SET outcome_status                    = %s,
        regime_kind_at_classification     = %s,
        dominant_bucket_at_classification = %s,
        horizon_sessions                  = %s,
        horizon_end_date                  = %s,
        reached_touch                     = %s,
        reached_close                     = %s,
        days_to_reach                     = %s,
        max_excursion_in_direction        = %s,
        final_close_distance_from_target  = %s,
        actual_realized_em_pct            = %s,
        session_open_t0                   = %s,
        backfill_run_id                   = %s,
        computed_at                       = NOW()
    WHERE ticker          = %s
      AND trade_date      = %s
      AND feature_version = %s
      AND outcome_status  = 'pending_history'
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


def _fetch_session_dates(conn) -> list[dt.date]:
    """Return sorted list of all distinct RTH session dates in ironbeam_es_1m_bars."""
    rows = conn.execute(_RTH_SESSION_DATES_SQL).fetchall()
    return sorted(r[0] for r in rows)


def _expected_horizon_end(
    trade_date: dt.date,
    bucket: str,
    session_dates: list[dt.date],
) -> Optional[dt.date]:
    """Return the Nth session on/after trade_date; None if not enough sessions exist."""
    try:
        n = bucket_sessions(bucket)
    except KeyError:
        return None
    forward = [d for d in session_dates if d >= trade_date]
    if len(forward) >= n:
        return forward[n - 1]
    return None


def _fetch_landscape(conn, ticker: str, dates: list[dt.date]) -> dict[dt.date, dict]:
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
    with conn.cursor() as cur:
        cur.execute(_RTH_BARS_SQL, (bar_from, bar_to))
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close"])
    df = pd.DataFrame(rows, columns=["session_date", "open", "high", "low", "close"])
    df = df.set_index("session_date")
    df.index = [d for d in df.index]
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _run_smoke(conn, ticker: str, n_promoted: int) -> dict:
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

    # Rows where horizon window is closed but outcome is still pending_history
    # (should be 0 after a clean sweep run).
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT MAX((datetime AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')::date)
            FROM ironbeam_es_1m_bars
            """
        )
        latest_session = cur.fetchone()[0]

    # Count remaining pending rows whose horizon is still open (not yet matured).
    # These are expected — they just haven't had enough sessions yet.
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*) FROM bt_daily_outcomes
            WHERE ticker = %s AND feature_version = %s
              AND outcome_status = 'pending_history'
            """,
            (ticker, FEATURE_VERSION),
        )
        remaining_pending = int(cur.fetchone()[0])

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

    return {
        "latest_session":       str(latest_session) if latest_session else None,
        "outcomes_total":       int(total) if total else 0,
        "outcomes_computed":    int(n_computed or 0),
        "outcomes_pending":     int(n_pending or 0),
        "outcomes_na_regime":   int(n_na_regime or 0),
        "outcomes_na_data":     int(n_na_data or 0),
        "n_promoted":           n_promoted,
        "remaining_pending":    remaining_pending,
        "null_run_id_count":    null_run_id,
        "date_min":             str(min_d) if min_d else None,
        "date_max":             str(max_d) if max_d else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ticker",    default="SPX")
    ap.add_argument("--from-date", default=None, metavar="YYYY-MM-DD",
                    help="Only sweep pending rows on/after this trade_date.")
    ap.add_argument("--limit",     type=int, default=None, metavar="N",
                    help="Cap number of matured rows to promote (for testing).")
    ap.add_argument("--dry-run",   action="store_true",
                    help="Print what would be promoted without writing.")
    args = ap.parse_args()

    ticker    = args.ticker
    from_date = dt.date.fromisoformat(args.from_date) if args.from_date else None

    _load_env()
    conn = get_backfill_db_conn()
    assert_role_or_die(conn)

    # Load all RTH session dates once (used for horizon-maturity check).
    session_dates = _fetch_session_dates(conn)
    if not session_dates:
        print("No RTH session dates found in ironbeam_es_1m_bars. Nothing to do.")
        conn.close()
        sys.exit(0)
    latest_session = session_dates[-1]

    # Fetch all pending_history rows (join with bt_daily_features for feature_vector).
    pending_rows = conn.execute(
        _PENDING_ROWS_SQL, (ticker, FEATURE_VERSION)
    ).fetchall()

    if from_date:
        pending_rows = [r for r in pending_rows if r[0] >= from_date]

    # Filter to matured rows: Nth session on/after trade_date exists and is <= latest_session.
    matured = []
    for trade_date, regime, bucket, fv in pending_rows:
        horizon_end = _expected_horizon_end(trade_date, bucket, session_dates)
        if horizon_end is not None and horizon_end <= latest_session:
            matured.append((trade_date, regime, bucket, fv))

    if args.limit:
        matured = matured[:args.limit]

    print("=== CR-AA Pending Outcome Sweep ===")
    print(f"Ticker: {ticker}   Feature version: {FEATURE_VERSION}")
    print(f"Latest RTH session: {latest_session}")
    print(f"Pending rows total: {len(pending_rows)}   Matured (window closed): {len(matured)}")
    if args.dry_run:
        print("[DRY RUN — no writes]")
    print()

    if not matured:
        print("No matured pending rows. Nothing to promote.")
        conn.close()
        sys.exit(0)

    dates = [r[0] for r in matured]

    if args.dry_run:
        for trade_date, regime, bucket, fv in matured:
            horizon_end = _expected_horizon_end(trade_date, bucket, session_dates)
            print(f"  [dry-run] {trade_date} regime={regime!r} bucket={bucket!r} "
                  f"horizon_end={horizon_end}")
        conn.close()
        sys.exit(0)

    landscape_by_date = _fetch_landscape(conn, ticker, dates)

    # Load bars from min(trade_date) through today to cover all horizon windows.
    bar_from   = min(dates)
    bar_to     = latest_session
    daily_bars = _fetch_daily_bars(conn, bar_from, bar_to)
    log.info("Loaded %d RTH daily sessions (%s → %s)", len(daily_bars), bar_from, bar_to)

    n_promoted = 0
    n_skipped  = 0   # rows where compute_outcome still returned pending_history
    n_failed   = 0
    failed_dates: list[str] = []

    with backfill_run(conn, "CR-AA") as run_id:
        print(f"Run ID: {run_id}")

        for i, (trade_date, regime, bucket, fv) in enumerate(matured, 1):
            try:
                landscape = landscape_by_date.get(trade_date, {})
                outcome, session_open_t0 = compute_outcome_for_date(
                    trade_date     = trade_date,
                    regime         = regime,
                    feature_vector = fv or {},
                    landscape      = landscape,
                    daily_bars     = daily_bars,
                )

                new_status = outcome["outcome_status"]

                if new_status == "pending_history":
                    # horizon-gated filter said it was matured but compute_outcome
                    # disagrees — log and skip rather than leaving unchanged.
                    log.warning(
                        "SKIP %s: horizon-gated filter predicted matured but "
                        "compute_outcome returned pending_history (bars gap?)",
                        trade_date,
                    )
                    n_skipped += 1
                    continue

                with conn.cursor() as cur:
                    cur.execute(_UPDATE_OUTCOME_SQL, (
                        new_status,
                        outcome["regime_kind_at_classification"],
                        outcome["dominant_bucket_at_classification"],
                        outcome["horizon_sessions"],
                        outcome["horizon_end_date"],
                        outcome["reached_touch"],
                        outcome["reached_close"],
                        outcome["days_to_reach"],
                        outcome["max_excursion_in_direction"],
                        outcome["final_close_distance_from_target"],
                        outcome["actual_realized_em_pct"],
                        session_open_t0,
                        run_id,
                        ticker,
                        trade_date,
                        FEATURE_VERSION,
                    ))
                    if cur.rowcount == 1:
                        n_promoted += 1
                        log.info("PROMOTED %s: pending_history → %s", trade_date, new_status)
                    else:
                        # rowcount == 0: row was already promoted by a concurrent run
                        n_skipped += 1
                        log.info("SKIPPED %s: no longer pending_history (concurrent update?)", trade_date)

            except Exception as exc:
                n_failed += 1
                failed_dates.append(str(trade_date))
                log.error("ERROR %s: %s", trade_date, exc, exc_info=True)

            if i % 20 == 0 or i == len(matured):
                update_run_progress(conn, run_id, n_promoted)
                print(f"  [{i}/{len(matured)}] "
                      f"promoted={n_promoted} skipped={n_skipped} failed={n_failed}")

        smoke = _run_smoke(conn, ticker, n_promoted)
        smoke.update({
            "n_matured":     len(matured),
            "n_skipped":     n_skipped,
            "n_failed":      n_failed,
            "failed_dates":  failed_dates,
        })
        assessment = (
            f"{n_promoted} rows promoted, {n_skipped} skipped, {n_failed} failed"
            + (f" — FAILURES: {', '.join(failed_dates)}" if failed_dates else " — clean run")
        )
        update_run_smoke(conn, run_id, smoke, assessment)

        print(f"\n=== DONE ===")
        print(f"promoted={n_promoted}  skipped={n_skipped}  failed={n_failed}")
        print(f"Latest RTH session:    {smoke['latest_session']}")
        print(f"Total outcome rows:    {smoke['outcomes_total']}")
        print(f"  computed={smoke['outcomes_computed']}  pending={smoke['outcomes_pending']}")
        print(f"  na_regime={smoke['outcomes_na_regime']}  na_data={smoke['outcomes_na_data']}")
        print(f"Remaining pending:     {smoke['remaining_pending']}")
        print(f"Null run_id count:     {smoke['null_run_id_count']}")

    conn.close()
    sys.exit(1 if n_failed > 0 else 0)


if __name__ == "__main__":
    main()
