#!/usr/bin/env python3
"""CR-026 (CR-G) Step 2.5a-bis — Expand session_open_t0 backfill to all outcome_status values.

Original Step 2.5a (2026-05-27) scoped to outcome_status IN ('computed', 'na_regime').
This excluded pending_history rows, which represent recent anchor dates whose T+15
horizon hasn't elapsed yet.

Surfaced 2026-05-28 during live-app verification: the POST /api/proposals/pl-data
endpoint returned "no session_open_t0 for (SPX, 2026-05-21)" because 2026-05-21 is
recent enough to still be pending_history.

Root cause of the original scope error: the outcome_status filter was inherited from
CR-I's "needs horizon data" pattern, which is wrong for session_open_t0. session_open_t0
= FIRST RTH bar open on trade_date — depends only on the trade_date's own bars, not on
horizon completion or regime classification. Any row with valid bars for its trade_date
should have session_open_t0 populated.

Fix: drop the outcome_status filter entirely. New scope:
    WHERE ticker = %s
      AND feature_version = %s
      AND session_open_t0 IS NULL

The _fetch_trade_date_open helper returns None when no bars exist (row logged + skipped).

Usage:
    python -u scripts/cr_g_step_2_5a_bis_backfill_session_open_t0.py \\
        2>&1 | tee scripts/logs/cr_g_step_2_5a_bis_$(date +%Y%m%d_%H%M%S).log
"""
from __future__ import annotations

import datetime as dt
import logging
import os
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env")

from packages.shared.backfill_safety import (
    assert_role_or_die,
    backfill_run,
    get_backfill_db_conn,
    update_run_progress,
    update_run_smoke,
)

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)

FEATURE_VERSION = "v0.5.0-rebuilt"
TICKER          = "SPX"
BATCH_SIZE      = 50
RTH_START_UTC   = dt.time(13, 30)   # 06:30 PT = 13:30 UTC
RTH_END_UTC     = dt.time(20,  0)   # 13:00 PT = 20:00 UTC


def _fetch_trade_date_open(conn, trade_date: dt.date) -> Optional[float]:
    """Return the FIRST RTH bar open on trade_date from ironbeam_es_1m_bars.

    Matches compute_outcome's horizon.iloc[0]["open"]:
      forward = bars_sorted[bars_sorted.index >= trade_date]
      first_open = float(forward.iloc[0]["open"])

    RTH session: 13:30–20:00 UTC (06:30–13:00 PT).
    Returns None if no bars found for trade_date.
    """
    start_ts = dt.datetime.combine(trade_date, RTH_START_UTC, tzinfo=dt.timezone.utc)
    end_ts   = dt.datetime.combine(trade_date, RTH_END_UTC,   tzinfo=dt.timezone.utc)

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT open
            FROM ironbeam_es_1m_bars
            WHERE datetime >= %s AND datetime < %s
            ORDER BY datetime ASC
            LIMIT 1
            """,
            (start_ts, end_ts),
        )
        row = cur.fetchone()

    return float(row[0]) if row else None


def main() -> None:
    conn = get_backfill_db_conn()
    assert_role_or_die(conn)
    log.info("Role verified: dash_backfill_writer ✓")

    with backfill_run(conn, "CR-G") as run_id:
        log.info("Backfill run registered (Step 2.5a-bis): run_id=%s", run_id)

        # ── Target rows: ALL outcome_status values where session_open_t0 IS NULL ──
        # No outcome_status filter — session_open_t0 depends only on trade_date bars,
        # not on horizon completion. Covers pending_history rows excluded by Step 2.5a.
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT trade_date, ticker, feature_version, outcome_status
                FROM bt_daily_outcomes
                WHERE ticker = %s
                  AND feature_version = %s
                  AND session_open_t0 IS NULL
                ORDER BY trade_date
                """,
                (TICKER, FEATURE_VERSION),
            )
            targets = cur.fetchall()

        total = len(targets)
        log.info(
            "Step 2.5a-bis target rows (session_open_t0 IS NULL, all statuses): %d",
            total,
        )

        if total == 0:
            log.info("Nothing to do — all rows already have session_open_t0.")
        else:
            # Log breakdown by outcome_status so we know what we're touching
            from collections import Counter
            status_counts = Counter(r[3] for r in targets)
            log.info("By outcome_status: %s", dict(status_counts))

        n_updated  = 0
        n_null     = 0   # trade_date has no RTH bars
        n_failed   = 0

        for i, (trade_date, ticker_col, fv_col, status) in enumerate(targets):
            try:
                open_price = _fetch_trade_date_open(conn, trade_date)
            except Exception as e:
                log.error("FAILED bar fetch for %s (status=%s): %s", trade_date, status, e)
                n_failed += 1
                continue

            if open_price is None:
                log.warning(
                    "No RTH bars for %s (status=%s) — session_open_t0 will remain NULL",
                    trade_date, status,
                )
                n_null += 1
                continue

            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE bt_daily_outcomes
                        SET session_open_t0 = %s,
                            backfill_run_id = %s
                        WHERE ticker = %s
                          AND trade_date = %s
                          AND feature_version = %s
                          AND session_open_t0 IS NULL
                        """,
                        (open_price, run_id, ticker_col, trade_date, fv_col),
                    )
            except Exception as e:
                log.error("UPDATE FAILED for %s (status=%s): %s", trade_date, status, e)
                n_failed += 1
                continue

            n_updated += 1

            if n_updated % BATCH_SIZE == 0:
                conn.commit()
                update_run_progress(conn, run_id, n_updated)
                log.info(
                    "Progress: %d/%d updated, %d no-bars, %d failed",
                    n_updated, total, n_null, n_failed,
                )

        conn.commit()
        log.info(
            "Step 2.5a-bis complete: %d updated, %d no-bars, %d failed (of %d total)",
            n_updated, n_null, n_failed, total,
        )

        # ── Smoke tests — no outcome_status filter (all statuses) ────────────────
        smoke: dict = {}

        with conn.cursor() as cur:
            # Smoke 1: rows with session_open_t0 populated (all statuses)
            cur.execute(
                "SELECT COUNT(*) FROM bt_daily_outcomes "
                "WHERE ticker = %s AND feature_version = %s "
                "  AND session_open_t0 IS NOT NULL",
                (TICKER, FEATURE_VERSION),
            )
            smoke["rows_with_t0"] = cur.fetchone()[0]

            # Smoke 2: rows still NULL (expected: 0 or only rows with no bars)
            cur.execute(
                "SELECT COUNT(*) FROM bt_daily_outcomes "
                "WHERE ticker = %s AND feature_version = %s "
                "  AND session_open_t0 IS NULL",
                (TICKER, FEATURE_VERSION),
            )
            smoke["rows_still_null"] = cur.fetchone()[0]

            # Smoke 3: range sanity — session_open_t0 should be in plausible ES range
            cur.execute(
                "SELECT MIN(session_open_t0), MAX(session_open_t0), AVG(session_open_t0) "
                "FROM bt_daily_outcomes "
                "WHERE ticker = %s AND feature_version = %s "
                "  AND session_open_t0 IS NOT NULL",
                (TICKER, FEATURE_VERSION),
            )
            row = cur.fetchone()
            smoke["t0_min"], smoke["t0_max"], smoke["t0_avg"] = (
                float(row[0]) if row[0] else None,
                float(row[1]) if row[1] else None,
                float(row[2]) if row[2] else None,
            )

            # Smoke 4: breakdown by outcome_status (all statuses)
            cur.execute(
                "SELECT outcome_status, "
                "       COUNT(*) AS total, "
                "       COUNT(session_open_t0) AS with_t0, "
                "       COUNT(*) - COUNT(session_open_t0) AS still_null "
                "FROM bt_daily_outcomes "
                "WHERE ticker = %s AND feature_version = %s "
                "GROUP BY outcome_status ORDER BY outcome_status",
                (TICKER, FEATURE_VERSION),
            )
            smoke["by_outcome_status"] = {
                r[0]: {"total": r[1], "with_t0": r[2], "still_null": r[3]}
                for r in cur.fetchall()
            }

        log.info("Smoke results: %s", smoke)

        assessment = (
            f"Step 2.5a-bis: {n_updated}/{total} rows updated; "
            f"{n_null} no-bars; {n_failed} failed; "
            f"remaining_null={smoke['rows_still_null']}; "
            f"t0_range=[{smoke.get('t0_min') or '?':.0f}, {smoke.get('t0_max') or '?':.0f}]; "
            f"by_status={smoke['by_outcome_status']}"
        )
        update_run_smoke(conn, run_id, smoke, assessment)
        log.info("Self-assessment: %s", assessment)

    conn.close()


if __name__ == "__main__":
    main()
