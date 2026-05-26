#!/usr/bin/env python3
"""CR-025 (CR-I) Step 2b — Backfill bt_daily_outcomes.position_tN_post_touch.

For each row where reached_touch=TRUE and position_t1_post_touch IS NULL:
  1. Fetch drift_target from orats_gex_landscape.walls (pick_drift_target)
  2. Fetch implied_move_1d from bt_daily_features_active.feature_vector
  3. Re-fetch RTH daily bars from ironbeam_es_1m_bars (trade_date + generous window)
  4. Call classify_post_touch_positions() to get -1/0/+1 per timeframe
  5. UPDATE bt_daily_outcomes with the three position columns + backfill_run_id

Idempotent: WHERE position_t1_post_touch IS NULL skips already-populated rows.

Data safety class: null_fill_update — eligible for unattended execution.
Pre-flight: SELECT current_user must return 'dash_backfill_writer'.

Usage:
    python -u scripts/cr_i_backfill_post_touch_positions.py 2>&1 | tee scripts/logs/cr_i_pass1_$(date +%Y%m%d_%H%M%S).log
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

import pandas as pd

from packages.shared.backfill_safety import (
    assert_role_or_die,
    backfill_run,
    get_backfill_db_conn,
    update_run_progress,
    update_run_smoke,
)
from packages.shared.outcomes import pick_drift_target
from packages.shared.probability import classify_post_touch_positions

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)

FEATURE_VERSION = "v0.5.0-rebuilt"
TICKER          = "SPX"
BATCH_SIZE      = 50           # rows per explicit conn.commit() heartbeat
RTH_START_UTC   = dt.time(13, 30)   # 06:30 PT = 13:30 UTC
RTH_END_UTC     = dt.time(20,  0)   # 13:00 PT = 20:00 UTC
# Calendar-day buffer past trade_date to cover days_to_reach + T+15:
# worst case: days_to_reach ≈ 20 sessions ≈ 30 calendar days; T+15 = 15 more sessions ≈ 22 cd
# 90-day buffer is generous but cheap (bars are per-minute, fetch is date-ranged).
BAR_WINDOW_CALENDAR_DAYS = 90


# ── Data loading helpers ─────────────────────────────────────────────────────


def _load_landscape(conn) -> dict[str, Optional[float]]:
    """Return {trade_date_iso: drift_target} for all SPX landscape rows.

    Uses pick_drift_target from packages/shared/outcomes.py — the canonical
    wall-selection logic.  Single bulk load avoids N per-row queries.
    """
    rows = conn.execute(
        "SELECT trade_date, walls FROM orats_gex_landscape WHERE ticker = %s",
        (TICKER,),
    ).fetchall()
    result = {}
    for trade_date, walls in rows:
        wall_list = walls if isinstance(walls, list) else []
        result[trade_date.isoformat()] = pick_drift_target(wall_list)
    log.info("Landscape loaded: %d rows", len(result))
    return result


def _load_implied_moves(conn) -> dict[str, Optional[float]]:
    """Return {trade_date_iso: implied_move_1d} from bt_daily_features_active."""
    rows = conn.execute(
        """
        SELECT trade_date,
               (feature_vector->>'implied_move_1d')::float AS implied_move_1d
        FROM bt_daily_features_active
        WHERE ticker = %s AND feature_version = %s
        """,
        (TICKER, FEATURE_VERSION),
    ).fetchall()
    result = {r[0].isoformat(): r[1] for r in rows}
    log.info("Implied moves loaded: %d rows", len(result))
    return result


def _fetch_rth_daily_bars(conn, start_date: dt.date, end_date: dt.date) -> pd.DataFrame:
    """Fetch RTH daily OHLC (close only) from ironbeam_es_1m_bars.

    Returns a DataFrame[trade_date (str), close (float)], one row per trading
    session between start_date and end_date.

    RTH: 13:30–20:00 UTC (06:30–13:00 Pacific).  'datetime' is the column name.
    """
    start_ts = dt.datetime.combine(start_date, RTH_START_UTC, tzinfo=dt.timezone.utc)
    end_ts   = dt.datetime.combine(end_date,   RTH_END_UTC,   tzinfo=dt.timezone.utc) + dt.timedelta(minutes=1)

    rows = conn.execute(
        """
        SELECT datetime, close
        FROM ironbeam_es_1m_bars
        WHERE datetime >= %s AND datetime < %s
        ORDER BY datetime
        """,
        (start_ts, end_ts),
    ).fetchall()

    if not rows:
        return pd.DataFrame(columns=["trade_date", "close"])

    df = pd.DataFrame(rows, columns=["datetime", "close"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    # Keep RTH minutes only (13:30 ≤ t < 20:00 UTC)
    hm = df["datetime"].dt.hour * 60 + df["datetime"].dt.minute
    df = df[(hm >= 13 * 60 + 30) & (hm < 20 * 60)].copy()
    df["trade_date"] = df["datetime"].dt.date.astype(str)

    # Last minute per session = daily RTH close
    return df.groupby("trade_date")["close"].last().reset_index()


# ── Main backfill ─────────────────────────────────────────────────────────────


def main() -> None:
    conn = get_backfill_db_conn()
    assert_role_or_die(conn)
    log.info("Role verified: dash_backfill_writer ✓")

    with backfill_run(conn, "CR-I") as run_id:
        log.info("Run registered: run_id=%s", run_id)

        # ── Pre-load lookup tables (avoid per-row queries) ────────────────────
        landscape    = _load_landscape(conn)
        implied_move = _load_implied_moves(conn)

        # ── Target rows: touched analogues with NULL position columns ─────────
        targets = conn.execute(
            """
            SELECT trade_date, days_to_reach, ticker, feature_version
            FROM bt_daily_outcomes
            WHERE ticker = %s
              AND feature_version = %s
              AND reached_touch = TRUE
              AND position_t1_post_touch IS NULL
            ORDER BY trade_date
            """,
            (TICKER, FEATURE_VERSION),
        ).fetchall()
        total = len(targets)
        log.info("Target rows (NULL position_t1): %d", total)

        n_updated = 0
        n_skipped = 0
        n_failed  = 0
        skip_reasons: dict[str, int] = {}

        for i, (trade_date, days_to_reach, ticker_col, fv_col) in enumerate(targets):
            trade_date_iso = trade_date.isoformat()

            # ── Resolve drift_target and tolerance ───────────────────────────
            drift_target = landscape.get(trade_date_iso)
            if drift_target is None:
                skip_reasons["no_drift_target"] = skip_reasons.get("no_drift_target", 0) + 1
                n_skipped += 1
                log.warning("Skipping %s: no drift_target in orats_gex_landscape", trade_date_iso)
                continue

            im_1d = implied_move.get(trade_date_iso)
            if not im_1d or im_1d <= 0:
                skip_reasons["no_implied_move"] = skip_reasons.get("no_implied_move", 0) + 1
                n_skipped += 1
                log.warning("Skipping %s: missing/invalid implied_move_1d=%r", trade_date_iso, im_1d)
                continue

            if days_to_reach is None:
                skip_reasons["null_days_to_reach"] = skip_reasons.get("null_days_to_reach", 0) + 1
                n_skipped += 1
                log.warning("Skipping %s: days_to_reach is NULL", trade_date_iso)
                continue

            tolerance = 0.25 * im_1d

            # ── Re-fetch bars: trade_date → trade_date + BAR_WINDOW_CALENDAR_DAYS ──
            # We need bars from trade_date through trade_date + days_to_reach + 15
            # sessions (T+15).  90-calendar-day window is generous (covers ≈60 sessions).
            bar_end = trade_date + dt.timedelta(days=BAR_WINDOW_CALENDAR_DAYS)
            try:
                bars_df = _fetch_rth_daily_bars(conn, trade_date, bar_end)
            except Exception as e:
                n_failed += 1
                log.error("Bar fetch failed for %s: %s", trade_date_iso, e)
                continue

            if bars_df.empty:
                skip_reasons["no_bars"] = skip_reasons.get("no_bars", 0) + 1
                n_skipped += 1
                log.warning("Skipping %s: no RTH bars found in window", trade_date_iso)
                continue

            # ── Classify post-touch positions ─────────────────────────────────
            try:
                positions = classify_post_touch_positions(
                    days_to_reach=days_to_reach,
                    horizon_bars=bars_df,
                    drift_target=drift_target,
                    tolerance=tolerance,
                    timeframes_sessions=(1, 5, 15),
                )
            except Exception as e:
                n_failed += 1
                log.error("classify_post_touch_positions failed for %s: %s", trade_date_iso, e)
                continue

            pos_t1  = positions.get(1)
            pos_t5  = positions.get(5)
            pos_t15 = positions.get(15)

            # ── UPDATE the row ────────────────────────────────────────────────
            try:
                conn.execute(
                    """
                    UPDATE bt_daily_outcomes
                    SET position_t1_post_touch  = %s,
                        position_t5_post_touch  = %s,
                        position_t15_post_touch = %s,
                        backfill_run_id         = %s
                    WHERE ticker = %s
                      AND feature_version = %s
                      AND trade_date = %s
                      AND position_t1_post_touch IS NULL
                    """,
                    (pos_t1, pos_t5, pos_t15, run_id, TICKER, FEATURE_VERSION, trade_date),
                )
            except Exception as e:
                n_failed += 1
                log.error("UPDATE failed for %s: %s", trade_date_iso, e)
                continue

            n_updated += 1

            if (i + 1) % BATCH_SIZE == 0:
                update_run_progress(conn, run_id, n_updated)
                log.info(
                    "Progress: %d/%d processed, %d updated, %d skipped, %d failed",
                    i + 1, total, n_updated, n_skipped, n_failed,
                )

        # ── Smoke checks ──────────────────────────────────────────────────────
        remaining_null = conn.execute(
            """
            SELECT COUNT(*) FROM bt_daily_outcomes
            WHERE ticker = %s AND feature_version = %s
              AND reached_touch = TRUE
              AND position_t1_post_touch IS NULL
            """,
            (TICKER, FEATURE_VERSION),
        ).fetchone()[0]

        out_of_range = conn.execute(
            """
            SELECT COUNT(*) FROM bt_daily_outcomes
            WHERE ticker = %s AND feature_version = %s
              AND position_t1_post_touch IS NOT NULL
              AND position_t1_post_touch NOT IN (-1, 0, 1)
            """,
            (TICKER, FEATURE_VERSION),
        ).fetchone()[0]

        value_dist_t1 = conn.execute(
            """
            SELECT position_t1_post_touch, COUNT(*)
            FROM bt_daily_outcomes
            WHERE ticker = %s AND feature_version = %s
              AND reached_touch = TRUE
            GROUP BY 1
            ORDER BY 1
            """,
            (TICKER, FEATURE_VERSION),
        ).fetchall()

        value_dist_t15 = conn.execute(
            """
            SELECT position_t15_post_touch, COUNT(*)
            FROM bt_daily_outcomes
            WHERE ticker = %s AND feature_version = %s
              AND reached_touch = TRUE
            GROUP BY 1
            ORDER BY 1
            """,
            (TICKER, FEATURE_VERSION),
        ).fetchall()

        smoke = {
            "n_updated":       n_updated,
            "n_skipped":       n_skipped,
            "n_failed":        n_failed,
            "skip_reasons":    skip_reasons,
            "remaining_null_t1": remaining_null,
            "out_of_range":    out_of_range,
            "value_dist_t1":   {str(k): v for k, v in value_dist_t1},
            "value_dist_t15":  {str(k): v for k, v in value_dist_t15},
        }

        status_ok = (n_failed == 0 and out_of_range == 0)
        assessment = (
            f"Pass 1 complete. updated={n_updated}, skipped={n_skipped}, failed={n_failed}. "
            f"remaining_null_t1={remaining_null} (expected: rows where T+1 bar unavailable). "
            f"out_of_range={out_of_range} (expect 0). "
            f"Status: {'clean' if status_ok else 'ISSUES — review failed/out_of_range counts'}."
        )

        if not status_ok:
            log.warning("SMOKE CHECK ISSUES: %s", assessment)
        else:
            log.info("Smoke checks clean: %s", assessment)

        update_run_smoke(conn, run_id, smoke, assessment)
        log.info("Run %s finalising with status='%s'", run_id, "completed" if status_ok else "suspect")

        if not status_ok:
            # Force the context manager to record 'suspect' instead of 'completed'
            conn.execute(
                "UPDATE bt_backfill_runs SET status = 'suspect' WHERE run_id = %s",
                (run_id,),
            )

    log.info("=== Pass 1 DONE ===  updated=%d  skipped=%d  failed=%d  remaining_null=%d",
             n_updated, n_skipped, n_failed, remaining_null)


if __name__ == "__main__":
    main()
