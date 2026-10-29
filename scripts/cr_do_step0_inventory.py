#!/usr/bin/env python3
"""CR-DO Step 0a (read-only, counts only) — derived tables other than bt_daily_outcomes /
vol_risk_premium that could hold values from CR-BZ-repaired bars.

Per B4 regime (train ≤ 2026-06-05, holdout 06-06 → 10-26, fwd ≥ 10-27, rb 2025) it counts:
  - setup_candidates events on a session with a CR-BZ repair, and events whose test / confirm /
    entry / retest minute is a repaired minute (no meta, outcome or price is read);
  - bt_signals rows on a session with a CR-BZ repair (live-panel snapshots; ts columns are text);
  - es_minute_features trade_date range (does it reach 2026?);
  - es_minutes vs ironbeam_es_1m_bars on the 783 repaired minutes (OHLCV equal?).
DB errors print the error type only.

Usage:
    apps/web/.venv/bin/python scripts/cr_do_step0_inventory.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(REPO_ROOT / ".env")

from packages.shared.backfill_safety import assert_role_or_die, get_backfill_db_conn  # noqa: E402

CRBZ_RUN = "88139854-3605-4d94-ab03-7bc34af5da5c"
REGIME = """CASE WHEN {d} < '2025-01-01' THEN 'pre2025' WHEN {d} <= '2025-12-31' THEN 'rb'
                 WHEN {d} <= '2026-06-05' THEN 'train' WHEN {d} <= '2026-10-26' THEN 'holdout' ELSE 'fwd' END"""
REPAIRED = f"""SELECT bar_ts FROM ironbeam_es_1m_bars_repair_log
               WHERE run_id = '{CRBZ_RUN}' AND target_table = 'ironbeam_es_1m_bars'"""
# a repaired minute belongs to the PT session date of its bar
REPAIRED_SESSIONS = f"SELECT DISTINCT (bar_ts AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')::date FROM ({REPAIRED}) r"


def _utc(col: str) -> str:   # naive PT wall clock → naive UTC (repair-log bar_ts convention)
    return f"date_trunc('minute', ({col} AT TIME ZONE 'America/Los_Angeles') AT TIME ZONE 'UTC')"


def _any_minute(cols) -> str:
    return " OR ".join(f"{_utc(c)} IN ({REPAIRED})" for c in cols)


QUERIES = {
    "CR-BZ repaired minutes by regime (PT session date)":
        f"SELECT {REGIME.format(d='s')}, count(*) FROM (SELECT (bar_ts AT TIME ZONE 'UTC' AT TIME ZONE "
        f"'America/Los_Angeles')::date AS s FROM ({REPAIRED}) r) x GROUP BY 1 ORDER BY 1",
    "setup_candidates: events / on a repaired session / a stamped minute is repaired, by regime":
        f"""SELECT {REGIME.format(d='session_date')}, count(*),
                   count(*) FILTER (WHERE session_date IN ({REPAIRED_SESSIONS})),
                   count(*) FILTER (WHERE {_any_minute(('test_ts_pt', 'confirm_ts_pt', 'entry_ts_pt', 'retest_ts_pt'))})
            FROM setup_candidates GROUP BY 1 ORDER BY 1""",
    "setup_candidates sets (cr_id, set_name, events)":
        "SELECT cr_id, set_name, count(*) FROM setup_candidates GROUP BY 1, 2 ORDER BY 1, 2",
    # signal_ts_pt etc. are free-form text (not all parse as timestamps): session-level count only
    "bt_signals: rows / on a repaired session, by regime":
        f"""SELECT {REGIME.format(d='trade_date')}, count(*),
                   count(*) FILTER (WHERE trade_date IN ({REPAIRED_SESSIONS}))
            FROM bt_signals GROUP BY 1 ORDER BY 1""",
    "es_minute_features trade_date range, rows":
        "SELECT min(trade_date), max(trade_date), count(*) FROM es_minute_features",
    "es_minutes vs ironbeam_es_1m_bars on repaired minutes (rows, OHLCV equal)":
        f"""SELECT count(*), count(*) FILTER (WHERE (e.open, e.high, e.low, e.close, e.volume)
                                                 = (b.open, b.high, b.low, b.close, b.volume))
            FROM ({REPAIRED}) r
            JOIN ironbeam_es_1m_bars b ON b.datetime AT TIME ZONE 'UTC' = r.bar_ts
            JOIN es_minutes e ON e.ts_utc = b.datetime""",
}


def main() -> int:
    try:
        conn = get_backfill_db_conn()
        assert_role_or_die(conn)
    except Exception as e:
        print(f"DB connect failed: {type(e).__name__}")
        return 1
    conn.read_only = True
    conn.autocommit = True
    for title, sql in QUERIES.items():
        print(f"== {title}")
        try:
            for row in conn.execute(sql).fetchall():
                print("   ", *row)
        except Exception as e:
            print(f"    DB error: {type(e).__name__}")
    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
