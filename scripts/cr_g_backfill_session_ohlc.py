#!/usr/bin/env python3
"""CR-026 (CR-G) Step 0-A — Backfill bt_daily_outcomes session OHLC columns.

For each row where outcome_status IN ('computed', 'na_regime') and
session_open_t1 IS NULL:
  1. Fetch RTH 1-minute bars from ironbeam_es_1m_bars for a 90-day window
     starting at trade_date
  2. Build a list of RTH session dates (one row = one trading session)
  3. For N in (1, 5, 15): find the Nth session after trade_date
       - session_open_tN   = FIRST bar open for that session
       - session_high_tN   = MAX bar high for that session
       - session_low_tN    = MIN bar low for that session
       - session_close_tN  = LAST bar close for that session
  4. NULL when that session falls beyond available bars (end-of-corpus)
  5. UPDATE bt_daily_outcomes with the 12 OHLC values + backfill_run_id

Scope: ~735 rows (computed + na_regime). OHLC computation is regime-agnostic —
no drift_target or tolerance needed. Includes bounded/amplification/untethered rows
so structural_distribution.py has OHLC for all K=20 analogue types.

NULL behaviour (T+15): rows near the corpus end where the 15th session is beyond
the ingested bars produce NULL in all four T+15 columns. Same pattern as
position_t15_post_touch. Expect a small number (~few) of T+15 NULLs.

Data safety class: null_fill_update — eligible for unattended execution.
Pre-flight: SELECT current_user must return 'dash_backfill_writer'.

Usage:
    python -u scripts/cr_g_backfill_session_ohlc.py 2>&1 | tee scripts/logs/cr_g_ohlc_$(date +%Y%m%d_%H%M%S).log
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

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)

FEATURE_VERSION         = "v0.5.0-rebuilt"
TICKER                  = "SPX"
BATCH_SIZE              = 50
RTH_START_UTC           = dt.time(13, 30)   # 06:30 PT = 13:30 UTC
RTH_END_UTC             = dt.time(20,  0)   # 13:00 PT = 20:00 UTC
# 90 calendar days covers trade_date + 15 sessions comfortably (~60 calendar days worst case)
BAR_WINDOW_CALENDAR_DAYS = 90
TARGET_TIMEFRAMES        = (1, 5, 15)


def _fetch_rth_minute_bars(
    conn, start_date: dt.date, end_date: dt.date
) -> pd.DataFrame:
    """Fetch RTH 1-minute bars from ironbeam_es_1m_bars.

    Returns DataFrame with columns: [session_date (str), datetime (UTC),
    open, high, low, close] filtered to RTH minutes (13:30–20:00 UTC).
    One row per 1-minute bar.
    """
    start_ts = dt.datetime.combine(start_date, RTH_START_UTC, tzinfo=dt.timezone.utc)
    end_ts   = dt.datetime.combine(end_date,   RTH_END_UTC,   tzinfo=dt.timezone.utc) + dt.timedelta(minutes=1)

    rows = conn.execute(
        """
        SELECT datetime, open, high, low, close
        FROM ironbeam_es_1m_bars
        WHERE datetime >= %s AND datetime < %s
        ORDER BY datetime
        """,
        (start_ts, end_ts),
    ).fetchall()

    if not rows:
        return pd.DataFrame(columns=["session_date", "datetime", "open", "high", "low", "close"])

    df = pd.DataFrame(rows, columns=["datetime", "open", "high", "low", "close"])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    # Filter to RTH minutes only (13:30 ≤ t < 20:00 UTC)
    hm = df["datetime"].dt.hour * 60 + df["datetime"].dt.minute
    df = df[(hm >= 13 * 60 + 30) & (hm < 20 * 60)].copy()
    df["session_date"] = df["datetime"].dt.date.astype(str)
    return df


def _compute_session_ohlc(
    bars: pd.DataFrame,
    trade_date: dt.date,
    timeframes: tuple[int, ...] = TARGET_TIMEFRAMES,
) -> dict[str, Optional[float]]:
    """Compute session OHLC at each timeframe after trade_date.

    trade_date is session 0. Session T+N is the Nth subsequent RTH session
    (T+1 = next session, T+5 = 5 sessions later, T+15 = 15 sessions later).

    Returns a flat dict: session_open_tN, session_high_tN, session_low_tN,
    session_close_tN for each N in timeframes. NULL (None) when the session
    is beyond available bars.
    """
    if bars.empty:
        result: dict[str, Optional[float]] = {}
        for tf in timeframes:
            for col in ("open", "high", "low", "close"):
                result[f"session_{col}_t{tf}"] = None
        return result

    # All distinct session dates in the bars, sorted ascending
    session_dates = sorted(bars["session_date"].unique())
    # trade_date itself is session 0; T+N is the Nth session after it
    trade_date_iso = trade_date.isoformat()

    try:
        origin_idx = session_dates.index(trade_date_iso)
    except ValueError:
        # trade_date not in bars — all NULL
        result = {}
        for tf in timeframes:
            for col in ("open", "high", "low", "close"):
                result[f"session_{col}_t{tf}"] = None
        return result

    result = {}
    for tf in timeframes:
        target_idx = origin_idx + tf
        if target_idx >= len(session_dates):
            # Session beyond available bars → NULL
            for col in ("open", "high", "low", "close"):
                result[f"session_{col}_t{tf}"] = None
            continue

        target_date = session_dates[target_idx]
        session_bars = bars[bars["session_date"] == target_date]
        if session_bars.empty:
            for col in ("open", "high", "low", "close"):
                result[f"session_{col}_t{tf}"] = None
            continue

        result[f"session_open_t{tf}"]  = float(session_bars["open"].iloc[0])
        result[f"session_high_t{tf}"]  = float(session_bars["high"].max())
        result[f"session_low_t{tf}"]   = float(session_bars["low"].min())
        result[f"session_close_t{tf}"] = float(session_bars["close"].iloc[-1])

    return result


def main() -> None:
    conn = get_backfill_db_conn()
    assert_role_or_die(conn)
    log.info("Role verified: dash_backfill_writer ✓")

    with backfill_run(conn, "CR-G") as run_id:
        log.info("Run registered: run_id=%s", run_id)

        # ── Target rows: outcome_status in ('computed', 'na_regime'), OHLC NULL ──
        targets = conn.execute(
            """
            SELECT trade_date, ticker, feature_version, outcome_status
            FROM bt_daily_outcomes
            WHERE ticker = %s
              AND feature_version = %s
              AND outcome_status IN ('computed', 'na_regime')
              AND session_open_t1 IS NULL
            ORDER BY trade_date
            """,
            (TICKER, FEATURE_VERSION),
        ).fetchall()
        total = len(targets)
        log.info("Target rows (NULL session_open_t1, computed+na_regime): %d", total)

        n_updated  = 0
        n_skipped  = 0
        n_failed   = 0
        n_t15_null = 0   # rows where T+15 session is beyond bars (expected small count)

        for i, (trade_date, ticker_col, fv_col, outcome_status) in enumerate(targets):
            trade_date_iso = trade_date.isoformat()

            # ── Fetch RTH bars: trade_date → trade_date + BAR_WINDOW_CALENDAR_DAYS ──
            end_date = trade_date + dt.timedelta(days=BAR_WINDOW_CALENDAR_DAYS)
            try:
                bars = _fetch_rth_minute_bars(conn, trade_date, end_date)
            except Exception as e:
                log.error("FAILED bars fetch for %s: %s", trade_date_iso, e)
                n_failed += 1
                continue

            if bars.empty:
                log.warning("No bars for %s (source=ironbeam_es_1m_bars)", trade_date_iso)
                n_skipped += 1
                continue

            # ── Compute session OHLC ──────────────────────────────────────────
            ohlc = _compute_session_ohlc(bars, trade_date)

            # Track T+15 NULLs (expected near corpus end)
            if ohlc.get("session_open_t15") is None:
                n_t15_null += 1

            # ── UPDATE bt_daily_outcomes ──────────────────────────────────────
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE bt_daily_outcomes
                        SET session_open_t1   = %(session_open_t1)s,
                            session_high_t1   = %(session_high_t1)s,
                            session_low_t1    = %(session_low_t1)s,
                            session_close_t1  = %(session_close_t1)s,
                            session_open_t5   = %(session_open_t5)s,
                            session_high_t5   = %(session_high_t5)s,
                            session_low_t5    = %(session_low_t5)s,
                            session_close_t5  = %(session_close_t5)s,
                            session_open_t15  = %(session_open_t15)s,
                            session_high_t15  = %(session_high_t15)s,
                            session_low_t15   = %(session_low_t15)s,
                            session_close_t15 = %(session_close_t15)s,
                            backfill_run_id   = %(run_id)s
                        WHERE ticker = %(ticker)s
                          AND trade_date = %(trade_date)s
                          AND feature_version = %(feature_version)s
                          AND session_open_t1 IS NULL
                        """,
                        {**ohlc, "run_id": run_id, "ticker": ticker_col,
                         "trade_date": trade_date, "feature_version": fv_col},
                    )
            except Exception as e:
                log.error("UPDATE FAILED for %s: %s", trade_date_iso, e)
                n_failed += 1
                continue

            n_updated += 1

            # ── Progress heartbeat ────────────────────────────────────────────
            if n_updated % BATCH_SIZE == 0:
                conn.commit()
                update_run_progress(conn, run_id, n_updated)
                log.info(
                    "Progress: %d/%d updated, %d skipped, %d failed, %d T+15 NULL",
                    n_updated, total, n_skipped, n_failed, n_t15_null,
                )

        # Final commit
        conn.commit()
        log.info(
            "Backfill complete: %d updated, %d skipped, %d failed, %d T+15 NULL (of %d total)",
            n_updated, n_skipped, n_failed, n_t15_null, total,
        )

        # ── Smoke tests ───────────────────────────────────────────────────────
        smoke: dict = {}

        # Smoke 1: rows updated count
        row = conn.execute(
            "SELECT COUNT(*) FROM bt_daily_outcomes "
            "WHERE ticker = %s AND feature_version = %s "
            "  AND outcome_status IN ('computed', 'na_regime') "
            "  AND session_open_t1 IS NOT NULL",
            (TICKER, FEATURE_VERSION),
        ).fetchone()
        smoke["rows_with_t1_ohlc"] = row[0]

        # Smoke 2: NULL T+15 count (expected small; corpus-end rows)
        row = conn.execute(
            "SELECT COUNT(*) FROM bt_daily_outcomes "
            "WHERE ticker = %s AND feature_version = %s "
            "  AND outcome_status IN ('computed', 'na_regime') "
            "  AND session_open_t1 IS NOT NULL "
            "  AND session_open_t15 IS NULL",
            (TICKER, FEATURE_VERSION),
        ).fetchone()
        smoke["rows_t1_populated_t15_null"] = row[0]

        # Smoke 3: by outcome_status
        rows = conn.execute(
            "SELECT outcome_status, COUNT(*) FROM bt_daily_outcomes "
            "WHERE ticker = %s AND feature_version = %s "
            "  AND session_open_t1 IS NOT NULL "
            "GROUP BY outcome_status ORDER BY outcome_status",
            (TICKER, FEATURE_VERSION),
        ).fetchall()
        smoke["by_outcome_status"] = {r[0]: r[1] for r in rows}

        # Smoke 4: sum invariant (high >= low for all non-null rows)
        row = conn.execute(
            "SELECT COUNT(*) FROM bt_daily_outcomes "
            "WHERE session_high_t1 IS NOT NULL AND session_high_t1 < session_low_t1",
        ).fetchone()
        smoke["high_lt_low_violations_t1"] = row[0]

        log.info("Smoke results: %s", smoke)

        assessment = (
            f"{n_updated}/{total} rows updated; "
            f"{smoke['rows_t1_populated_t15_null']} T+15 NULLs (corpus-end); "
            f"by_status={smoke['by_outcome_status']}; "
            f"high<low violations={smoke['high_lt_low_violations_t1']}"
        )
        update_run_smoke(conn, run_id, smoke, assessment)
        log.info("Self-assessment: %s", assessment)


if __name__ == "__main__":
    main()
