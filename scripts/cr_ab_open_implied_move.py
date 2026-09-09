#!/usr/bin/env python3
"""CR-AB — Post-open implied-move fill.

Runs at 13:35 UTC (06:35 PDT) after the 06:33 PT open-straddle snapshot is
available in orats_monies_minute. Finds the most recent bt_daily_features row
at the current canonical feature_version where implied_move_1d IS NULL (written
by the 03:01 PDT EOD cron), pins the 06:33 open straddle, recomputes the
sigma-normalized features, and UPDATEs the row.

Requires the DDL prerequisite from CR-037:
  infra/sql/bt_daily_features_backfill_writer_feature_update.sql
to be applied before the first run.

CR-BD decision 2 (2026-09-08): the job no longer assumes the clock. Before
filling it polls orats_monies_minute for the target date's 06:33 PT SPX
snapshot — up to --max-wait-min (120) minutes when the target date is today or
later in America/Los_Angeles, one probe for a past date — and exits 0 with a
logged "no open snapshot" if none appears (holiday, DST drift, ingest outage).
A target date that is not an NYSE trading day (a mis-stamped row) is skipped
immediately, exit 0, instead of stalling the poll.

Usage:
    python scripts/cr_ab_open_implied_move.py
    python scripts/cr_ab_open_implied_move.py --date 2026-06-06  # re-run specific date
    python scripts/cr_ab_open_implied_move.py --dry-run
    python scripts/cr_ab_open_implied_move.py --max-wait-min 0   # never wait

Exit: 0 on success or nothing to do (incl. no snapshot within the budget); 1 on failure.
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

from packages.shared.backfill_safety import (
    assert_role_or_die,
    backfill_run,
    get_backfill_db_conn,
    update_run_smoke,
)
from packages.shared.canonical_version import CANONICAL_FEATURE_VERSION
from packages.shared.day_features import compute_and_upsert_open_implied_move
from packages.shared.snapshot_poll import no_snapshot_line, now_pt, wait_for_open_snapshot
from packages.shared.trading_calendar import is_trading_day

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

TICKER = os.environ.get("TICKER", "SPX").strip()

# Find the most recent bt_daily_features row at the canonical version that
# still has a NULL implied_move_1d (EOD cron wrote non-IV features; post-open
# fill not yet applied).
_PENDING_DATE_SQL = """
    SELECT trade_date
    FROM bt_daily_features
    WHERE ticker          = %s
      AND feature_version = %s
      AND active          = TRUE
      AND (feature_vector->>'implied_move_1d') IS NULL
    ORDER BY trade_date DESC
    LIMIT 1
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


def skip_reason_for(trade_date: dt.date) -> Optional[str]:
    """CR-BD: a target date that is not an NYSE session can never have a 06:33 snapshot."""
    if not is_trading_day(trade_date):
        return (f"{trade_date} is not an NYSE trading day (holiday-mis-stamped row?) — "
                f"no 06:33 PT snapshot can exist; nothing to fill")
    return None


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Post-open implied-move fill (CR-AB).")
    ap.add_argument("--date", help="Target trade_date YYYY-MM-DD (default: auto-detect).")
    ap.add_argument("--dry-run", action="store_true", help="Show target date but don't update.")
    ap.add_argument("--max-wait-min", type=float, default=None,
                    help="CR-BD: poll budget for the target date's 06:33 PT snapshot (default: 120 if the date is "
                         "today or later in America/Los_Angeles, else one probe)")
    ap.add_argument("--poll-seconds", type=float, default=60.0)
    args = ap.parse_args(argv)

    _load_env()
    conn = get_backfill_db_conn()
    assert_role_or_die(conn)

    log.info("canonical feature_version=%s ticker=%s", CANONICAL_FEATURE_VERSION, TICKER)

    # Resolve target date.
    if args.date:
        trade_date = dt.date.fromisoformat(args.date)
        log.info("target trade_date=%s (explicit --date)", trade_date)
    else:
        row = conn.execute(
            _PENDING_DATE_SQL,
            (TICKER, CANONICAL_FEATURE_VERSION),
        ).fetchone()
        if not row:
            log.info(
                "No bt_daily_features row with NULL implied_move_1d at %s — nothing to do.",
                CANONICAL_FEATURE_VERSION,
            )
            conn.close()
            return
        trade_date = row[0]
        log.info("target trade_date=%s (auto-detected: most recent NULL implied_move_1d)", trade_date)

    reason = skip_reason_for(trade_date)
    if reason:
        log.warning(reason)
        conn.close()
        return

    if args.dry_run:
        log.info("[dry-run] would poll for the %s 06:33 PT snapshot, then call compute_and_upsert_open_implied_move "
                 "for (%s, %s, version=%s)", trade_date, TICKER, trade_date, CANONICAL_FEATURE_VERSION)
        conn.close()
        return

    # CR-BD decision 2: wait for the pin instead of assuming the clock
    snap, waited, attempts, budget = wait_for_open_snapshot(
        conn, TICKER, trade_date, max_wait_min=args.max_wait_min, poll_s=args.poll_seconds, log=log.info)
    if snap is None:
        log.warning(no_snapshot_line(TICKER, trade_date, waited, attempts, budget))
        conn.close()
        return
    if attempts > 1:
        log.info("06:33 PT snapshot appeared after %.1f min (%d probes) — now %s PT", waited, attempts, now_pt().strftime("%H:%M"))

    with backfill_run(conn, "CR-AB") as run_id:
        summary = compute_and_upsert_open_implied_move(
            conn,
            ticker=TICKER,
            trade_date=trade_date,
            version=CANONICAL_FEATURE_VERSION,
        )
        if summary["updated"]:
            log.info(
                "implied_move filled: (%s, %s) version=%s implied_move=%.4f n_features=%s",
                TICKER, trade_date, CANONICAL_FEATURE_VERSION,
                summary["implied_move"], summary["n_features"],
            )
        else:
            log.warning(
                "implied_move NOT filled for (%s, %s): no 06:33+ snapshot available",
                TICKER, trade_date,
            )

        update_run_smoke(
            conn,
            run_id,
            smoke_results={
                "ticker":          TICKER,
                "trade_date":      trade_date.isoformat(),
                "feature_version": CANONICAL_FEATURE_VERSION,
                "implied_move":    summary.get("implied_move"),
                "updated":         summary["updated"],
            },
            self_assessment=(
                f"filled {TICKER} {trade_date} implied_move={summary.get('implied_move')}"
                if summary["updated"]
                else f"no 06:33+ snapshot for {TICKER} {trade_date}"
            ),
        )

    conn.close()
    if not summary["updated"]:
        log.error("Post-open fill failed — implied_move_1d still NULL. Check orats_monies_minute.")
        sys.exit(1)


if __name__ == "__main__":
    main()
