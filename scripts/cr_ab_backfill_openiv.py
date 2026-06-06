#!/usr/bin/env python3
"""CR-AB Step 4 — re-backfill corpus onto open-straddle basis (v0.6.0-openiv).

Writes bt_daily_features rows at feature_version='v0.6.0-openiv' for all dates
in the v0.5.0-rebuilt corpus, sourcing implied_move_1d from the 06:33 PT
open-straddle ATM IV for each date. v0.5.0-rebuilt rows are left intact.

Hard gate (gate 0.6): before the main loop, counts corpus dates with no 06:33+
orats_monies_minute snapshot. If any are found, the script prints the affected
dates and exits 1 — surface these for a fallback decision before flipping the
canonical in Step 5.

Uses BACKFILL_DATABASE_URL + assert_role_or_die + backfill_run scaffolding.
Does INSERT ON CONFLICT DO NOTHING so re-running is safe.

Usage:
    python scripts/cr_ab_backfill_openiv.py
    python scripts/cr_ab_backfill_openiv.py --dry-run
    python scripts/cr_ab_backfill_openiv.py --from-date 2025-01-01
    python scripts/cr_ab_backfill_openiv.py --to-date 2025-06-30
    python scripts/cr_ab_backfill_openiv.py --limit 10

Exit: 0 on success (or clean --limit run); 1 on gate failure or any date error.
"""
from __future__ import annotations

import argparse
import datetime as dt
import logging
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
    _LANDSCAPE_ROW_SQL,
    _OPEN_STRADDLE_SQL,
    _materialize_payload,
    compute_feature_config_hash,
    extract_features,
)
from packages.shared.gex_landscape import compute_implied_move

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

TICKER = os.environ.get("TICKER", "SPX").strip()
FEATURE_VERSION = "v0.6.0-openiv"
SOURCE_VERSION  = "v0.5.0-rebuilt"

# INSERT with ON CONFLICT DO NOTHING — safe to re-run; includes backfill_run_id
# for row-level tracking (can deactivate a bad run via UPDATE WHERE backfill_run_id).
_INSERT_SQL = """
    INSERT INTO bt_daily_features
        (ticker, trade_date, feature_vector, feature_version,
         feature_config_hash, regime_at_classification, backfill_run_id)
    VALUES (%s, %s, %s, %s, %s, %s, %s)
    ON CONFLICT (ticker, trade_date, feature_version) DO NOTHING
"""

# Corpus dates at SOURCE_VERSION not yet written at FEATURE_VERSION.
_TARGET_DATES_SQL = """
    SELECT f.trade_date
    FROM bt_daily_features f
    WHERE f.ticker          = %s
      AND f.feature_version = %s
      AND f.active          = TRUE
      AND f.trade_date NOT IN (
          SELECT trade_date FROM bt_daily_features
          WHERE ticker = %s AND feature_version = %s
      )
    ORDER BY f.trade_date
"""

# Hard-gate query: corpus dates with no 06:33+ DTE>0 open straddle snapshot.
_MISSING_SNAPSHOTS_SQL = """
    SELECT f.trade_date
    FROM bt_daily_features f
    WHERE f.ticker          = %s
      AND f.feature_version = %s
      AND f.active          = TRUE
      AND NOT EXISTS (
          SELECT 1
          FROM orats_monies_minute m
          WHERE m.trade_date      = f.trade_date::text
            AND m.ticker          = %s
            AND m.atmiv           IS NOT NULL
            AND m.dte             > 0
            AND m.snapshot_pt::time >= '06:33'
      )
    ORDER BY f.trade_date
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


def _check_hard_gate(conn, ticker: str) -> list[dt.date]:
    """Return corpus dates missing a 06:33+ open straddle snapshot."""
    rows = conn.execute(
        _MISSING_SNAPSHOTS_SQL,
        (ticker, SOURCE_VERSION, ticker),
    ).fetchall()
    return [r[0] for r in rows]


def _get_target_dates(
    conn,
    ticker: str,
    from_date: dt.date | None,
    to_date: dt.date | None,
    limit: int | None,
) -> list[dt.date]:
    rows = conn.execute(
        _TARGET_DATES_SQL,
        (ticker, SOURCE_VERSION, ticker, FEATURE_VERSION),
    ).fetchall()
    dates = [r[0] for r in rows]
    if from_date:
        dates = [d for d in dates if d >= from_date]
    if to_date:
        dates = [d for d in dates if d <= to_date]
    if limit:
        dates = dates[:limit]
    return dates


def _compute_and_insert_one(
    conn,
    ticker: str,
    trade_date: dt.date,
    run_id: str,
    dry_run: bool,
) -> bool:
    """Return True if a row was inserted (or would be in dry-run)."""
    with conn.cursor() as cur:
        cur.execute(_LANDSCAPE_ROW_SQL, (ticker, trade_date))
        row = cur.fetchone()
    if not row or row[1] is None:
        log.warning("  [SKIP] %s: no landscape row", trade_date)
        return False
    landscape_rows, table_spot = row
    spot = float(table_spot)

    with conn.cursor() as cur:
        cur.execute(_OPEN_STRADDLE_SQL, (trade_date.isoformat(), ticker))
        iv_row = cur.fetchone()
    if not iv_row or iv_row[0] is None:
        log.warning("  [SKIP] %s: no 06:33+ open straddle snapshot", trade_date)
        return False

    try:
        iv_f = float(iv_row[0])
        implied_move = compute_implied_move(spot, iv_f, dte=1.0)
    except (TypeError, ValueError) as exc:
        log.warning("  [SKIP] %s: implied_move computation failed: %s", trade_date, exc)
        return False

    payload = _materialize_payload(landscape_rows, spot, implied_move)
    features = extract_features(payload, spot, implied_move)
    config_hash = compute_feature_config_hash(FEATURE_VERSION)
    regime = (payload.get("regime") or {}).get("regime") or None

    if dry_run:
        log.info("  [dry-run] %s: would insert spot=%.2f implied_move=%.4f regime=%r",
                 trade_date, spot, implied_move, regime)
        return True

    with conn.cursor() as cur:
        cur.execute(_INSERT_SQL, (
            ticker, trade_date,
            Jsonb(features), FEATURE_VERSION, config_hash, regime, run_id,
        ))
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Corpus re-backfill onto open-straddle basis.")
    ap.add_argument("--from-date", help="First trade_date to backfill (YYYY-MM-DD).")
    ap.add_argument("--to-date",   help="Last trade_date to backfill (YYYY-MM-DD).")
    ap.add_argument("--limit",     type=int, help="Cap the number of dates to process.")
    ap.add_argument("--dry-run",   action="store_true", help="Show plan without writing.")
    args = ap.parse_args()

    from_date = dt.date.fromisoformat(args.from_date) if args.from_date else None
    to_date   = dt.date.fromisoformat(args.to_date)   if args.to_date   else None

    _load_env()
    conn = get_backfill_db_conn()
    assert_role_or_die(conn)

    log.info("Corpus re-backfill: %s → %s  ticker=%s", SOURCE_VERSION, FEATURE_VERSION, TICKER)

    # ── Hard gate (0.6): confirm no corpus dates are missing the open straddle ──
    missing = _check_hard_gate(conn, TICKER)
    if missing:
        log.error(
            "GATE 0.6 FAIL: %d corpus date(s) have no 06:33+ open straddle snapshot "
            "— these would become NULL implied_move rows. Review before canonical flip.",
            len(missing),
        )
        for d in missing[:20]:
            log.error("  missing: %s", d)
        if len(missing) > 20:
            log.error("  ... and %d more", len(missing) - 20)
        conn.close()
        sys.exit(1)
    log.info("Gate 0.6 PASSED: all corpus dates have a 06:33+ open straddle snapshot.")

    dates = _get_target_dates(conn, TICKER, from_date, to_date, args.limit)
    if not dates:
        log.info("No dates to backfill — %s already complete for all corpus dates.", FEATURE_VERSION)
        conn.close()
        return

    log.info("Backfilling %d date(s): %s … %s", len(dates), dates[0], dates[-1])
    if args.dry_run:
        log.info("[dry-run] no data will be written")

    with backfill_run(conn, "CR-AB") as run_id:
        ok = 0
        skipped = 0
        failures: list[tuple[dt.date, str]] = []

        for i, trade_date in enumerate(dates, start=1):
            try:
                inserted = _compute_and_insert_one(conn, TICKER, trade_date, run_id, args.dry_run)
                if inserted:
                    ok += 1
                    log.info("  [%d/%d] %s  OK", i, len(dates), trade_date)
                else:
                    skipped += 1
            except Exception as exc:
                failures.append((trade_date, str(exc)))
                log.error("  [%d/%d] %s  FAIL: %s", i, len(dates), trade_date, exc)

            if i % 50 == 0:
                update_run_progress(conn, run_id, ok)

        smoke = {
            "n_dates_attempted": len(dates),
            "n_inserted":        ok,
            "n_skipped":         skipped,
            "n_failed":          len(failures),
            "first_date":        dates[0].isoformat() if dates else None,
            "last_date":         dates[-1].isoformat() if dates else None,
            "failures":          [(d.isoformat(), e) for d, e in failures[:10]],
        }
        update_run_smoke(
            conn, run_id, smoke,
            f"inserted {ok}/{len(dates)} {FEATURE_VERSION} rows; {skipped} skipped; {len(failures)} failed",
        )

    conn.close()

    log.info(
        "Done. inserted=%d skipped=%d failed=%d out of %d",
        ok, skipped, len(failures), len(dates),
    )
    if failures:
        log.error("Failures (first 10):")
        for d, e in failures[:10]:
            log.error("  %s: %s", d, e[:160])
        sys.exit(1)


if __name__ == "__main__":
    main()
