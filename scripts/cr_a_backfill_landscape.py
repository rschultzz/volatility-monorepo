#!/usr/bin/env python3
"""CR-021 Step 3 — two-step landscape + feature backfill.

For each target date:
  Step A: ensure orats_gex_landscape row exists.
          If already present → skip (no UPDATE permission on this role).
          If absent → compute from orats_oi_gamma via compute_and_insert_landscape.
  Step B: compute features from the stored landscape and INSERT into
          bt_daily_features at feature_version='v0.5.0-rebuilt' with
          ON CONFLICT DO NOTHING (no UPDATE permission; safe on re-run).

The dash_backfill_writer role has INSERT+SELECT but NOT UPDATE on both
orats_gex_landscape and bt_daily_features, so this script never calls
the library ON CONFLICT DO UPDATE upserts directly — it either skips
existing rows or inserts fresh.

Note on orats_gex_landscape tracking: landscape rows written by this
backfill (for historical dates not already present) use version tag
'cr-021-backfill' but carry no per-row backfill_run_id because that
column does not exist on orats_gex_landscape. These rows are tracked
only via bt_backfill_runs.smoke_test_results (date range). A future
small CR should add safety columns to orats_gex_landscape for
architectural symmetry.

Usage:
    python scripts/cr_a_backfill_landscape.py
    python scripts/cr_a_backfill_landscape.py --from-date 2026-04-15 --to-date 2026-05-15 --limit 3
    python scripts/cr_a_backfill_landscape.py --dry-run

Exit: 0 on success or clean --limit run; 1 if any date failed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from psycopg.types.json import Jsonb

from packages.shared.backfill_safety import (
    assert_role_or_die,
    backfill_run,
    get_backfill_db_conn,
    update_run_progress,
    update_run_smoke,
)
from packages.shared.day_features import (
    _IMPLIED_MOVE_SQL,
    _LANDSCAPE_ROW_SQL,
    _materialize_payload,
    compute_feature_config_hash,
    extract_features,
)
from packages.shared.gex_landscape import compute_and_insert_landscape, compute_implied_move

FEATURE_VERSION = "v0.5.0-rebuilt"
LANDSCAPE_BACKFILL_VERSION = "cr-021-backfill"

# Match the EOD cron parameters exactly (apps/cron/job_orats_eod.py lines 251-257).
LANDSCAPE_SPREAD_COEF = 8.0
LANDSCAPE_RANGE_PTS   = 300.0
LANDSCAPE_STEP_PTS    = 1.0

# Custom INSERT — DO NOTHING because dash_backfill_writer has no UPDATE permission.
# Includes backfill_run_id so runs can be deactivated with a single query by run_id.
_INSERT_FEATURES_SQL = """
    INSERT INTO bt_daily_features
        (ticker, trade_date, feature_vector, feature_version,
         feature_config_hash, regime_at_classification, backfill_run_id)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (ticker, trade_date, feature_version) DO NOTHING
"""

# Target dates = dates with oi_gamma data AND orats_monies_minute data
# (both sources required; orats_monies_minute is needed for implied_move),
# minus dates already written at v0.5.0-rebuilt.
_TARGET_DATES_SQL = """
    SELECT DISTINCT d
    FROM (
        SELECT trade_date::date AS d FROM orats_oi_gamma WHERE ticker = %s
    ) g
    WHERE EXISTS (
        SELECT 1 FROM orats_monies_minute m
        WHERE m.ticker = %s
          AND m.trade_date::date = g.d
    )
    AND d NOT IN (
        SELECT trade_date FROM bt_daily_features
        WHERE ticker = %s AND feature_version = %s
    )
    ORDER BY d
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


def _get_target_dates(
    conn,
    ticker: str,
    from_date: dt.date | None,
    to_date: dt.date | None,
    limit: int | None,
) -> list[dt.date]:
    with conn.cursor() as cur:
        cur.execute(_TARGET_DATES_SQL, (ticker, ticker, ticker, FEATURE_VERSION))
        rows = cur.fetchall()
    dates = [r[0] for r in rows]
    if from_date:
        dates = [d for d in dates if d >= from_date]
    if to_date:
        dates = [d for d in dates if d <= to_date]
    if limit:
        dates = dates[:limit]
    return dates


def _ensure_landscape(conn, ticker: str, trade_date: dt.date, dry_run: bool) -> bool:
    """Return True if a landscape row exists or was just computed. False = no data."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM orats_gex_landscape WHERE ticker = %s AND trade_date = %s",
            (ticker, trade_date),
        )
        if cur.fetchone():
            return True

    if dry_run:
        print(f"  [dry-run] {trade_date}: would compute landscape from orats_oi_gamma")
        return True

    try:
        compute_and_insert_landscape(
            conn, ticker, trade_date,
            spread_coef=LANDSCAPE_SPREAD_COEF,
            range_pts=LANDSCAPE_RANGE_PTS,
            step_pts=LANDSCAPE_STEP_PTS,
            version=LANDSCAPE_BACKFILL_VERSION,
        )
        return True
    except (ValueError, Exception) as exc:
        print(f"  [landscape SKIP] {trade_date}: {exc}")
        return False


def _compute_and_insert_features(
    conn,
    ticker: str,
    trade_date: dt.date,
    run_id: str,
    dry_run: bool,
) -> bool:
    """Return True if a features row was inserted (or would be in dry-run)."""
    with conn.cursor() as cur:
        cur.execute(_LANDSCAPE_ROW_SQL, (ticker, trade_date))
        row = cur.fetchone()
    if not row or row[1] is None:
        print(f"  [features SKIP] {trade_date}: no landscape row after ensure step")
        return False
    landscape_rows, table_spot = row
    spot = float(table_spot)

    with conn.cursor() as cur:
        cur.execute(_IMPLIED_MOVE_SQL, (trade_date.isoformat(), ticker))
        iv_row = cur.fetchone()
    if iv_row and iv_row[0] is not None:
        try:
            implied_move = compute_implied_move(spot, float(iv_row[0]), dte=1.0)
        except (TypeError, ValueError) as exc:
            print(f"  [features SKIP] {trade_date}: implied_move computation failed: {exc}")
            return False
    else:
        print(f"  [features SKIP] {trade_date}: no implied_move data in orats_monies_minute")
        return False

    payload = _materialize_payload(landscape_rows, spot, implied_move)
    features = extract_features(payload, spot, implied_move)
    config_hash = compute_feature_config_hash(FEATURE_VERSION)
    regime = (payload.get("regime") or {}).get("regime") or None

    if dry_run:
        print(f"  [dry-run] {trade_date}: would insert features "
              f"spot={spot:.2f} im={implied_move:.4f} regime={regime!r}")
        return True

    with conn.cursor() as cur:
        cur.execute(_INSERT_FEATURES_SQL, (
            ticker, trade_date, Jsonb(features),
            FEATURE_VERSION, config_hash, regime, run_id,
        ))
        return cur.rowcount == 1


def _run_smoke(conn, ticker: str) -> dict:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*), MIN(trade_date), MAX(trade_date) "
            "FROM bt_daily_features "
            "WHERE ticker = %s AND feature_version = %s",
            (ticker, FEATURE_VERSION),
        )
        n_rows, min_dt, max_dt = cur.fetchone()

    with conn.cursor() as cur:
        cur.execute(
            "SELECT regime_at_classification, COUNT(*) "
            "FROM bt_daily_features "
            "WHERE ticker = %s AND feature_version = %s "
            "GROUP BY 1 ORDER BY 2 DESC",
            (ticker, FEATURE_VERSION),
        )
        regime_rows = cur.fetchall()

    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM bt_daily_features "
            "WHERE ticker = %s AND feature_version = %s "
            "  AND feature_vector IS NULL",
            (ticker, FEATURE_VERSION),
        )
        null_fv = int(cur.fetchone()[0])

    with conn.cursor() as cur:
        cur.execute(
            "SELECT COUNT(*) FROM bt_daily_features WHERE feature_version = 'v0.5.0'",
        )
        v050_count = int(cur.fetchone()[0])

    return {
        "n_rows": int(n_rows) if n_rows else 0,
        "date_min": str(min_dt) if min_dt else None,
        "date_max": str(max_dt) if max_dt else None,
        "regime_distribution": {(r[0] or "None"): int(r[1]) for r in regime_rows},
        "null_feature_vectors": null_fv,
        "v050_row_count_unchanged": v050_count,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ticker",    default="SPX")
    ap.add_argument("--from-date", default=None, metavar="YYYY-MM-DD")
    ap.add_argument("--to-date",   default=None, metavar="YYYY-MM-DD")
    ap.add_argument("--limit",     type=int, default=None, metavar="N")
    ap.add_argument("--dry-run",   action="store_true")
    ap.add_argument("--batch",     type=int, default=10, metavar="N",
                    help="Update rows_inserted every N dates (default: 10)")
    args = ap.parse_args()

    ticker    = args.ticker
    from_date = dt.date.fromisoformat(args.from_date) if args.from_date else None
    to_date   = dt.date.fromisoformat(args.to_date)   if args.to_date   else None

    _load_env()
    conn = get_backfill_db_conn()
    assert_role_or_die(conn)

    target_dates = _get_target_dates(conn, ticker, from_date, to_date, args.limit)

    print(f"=== CR-021 Landscape Backfill ===")
    print(f"Ticker: {ticker}   Feature version: {FEATURE_VERSION}")
    print(f"Target dates: {len(target_dates)}")
    if target_dates:
        print(f"Range: {target_dates[0]} → {target_dates[-1]}")
    if args.dry_run:
        print("[DRY RUN — no writes]")
    print()

    if not target_dates:
        print("Nothing to do.")
        conn.close()
        sys.exit(0)

    if args.dry_run:
        for d in target_dates:
            _ensure_landscape(conn, ticker, d, dry_run=True)
            _compute_and_insert_features(conn, ticker, d, run_id="dry-run", dry_run=True)
        conn.close()
        sys.exit(0)

    n_inserted = 0
    n_skipped  = 0
    n_failed   = 0
    failed_dates: list[str] = []

    with backfill_run(conn, "CR-021") as run_id:
        print(f"Run ID: {run_id}")
        for i, trade_date in enumerate(target_dates, 1):
            try:
                ok = _ensure_landscape(conn, ticker, trade_date, dry_run=False)
                if not ok:
                    n_failed += 1
                    failed_dates.append(str(trade_date))
                    continue

                inserted = _compute_and_insert_features(
                    conn, ticker, trade_date, run_id=run_id, dry_run=False
                )
                if inserted:
                    n_inserted += 1
                else:
                    n_skipped += 1

            except Exception as exc:
                n_failed += 1
                failed_dates.append(str(trade_date))
                print(f"  [ERROR] {trade_date}: {exc}")

            if i % args.batch == 0 or i == len(target_dates):
                update_run_progress(conn, run_id, n_inserted)
                print(f"  [{i}/{len(target_dates)}] "
                      f"inserted={n_inserted} skipped={n_skipped} failed={n_failed}")

        smoke = _run_smoke(conn, ticker)
        smoke.update({
            "n_target":     len(target_dates),
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
        print(f"Total v0.5.0-rebuilt rows: {smoke['n_rows']}")
        print(f"Date range: {smoke['date_min']} → {smoke['date_max']}")
        print(f"Regime distribution: {smoke['regime_distribution']}")
        print(f"Null feature_vectors: {smoke['null_feature_vectors']}")
        print(f"v0.5.0 row count (unchanged check): {smoke['v050_row_count_unchanged']}")

    conn.close()
    sys.exit(1 if n_failed > 0 else 0)


if __name__ == "__main__":
    main()
