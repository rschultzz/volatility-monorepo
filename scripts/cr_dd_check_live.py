#!/usr/bin/env python3
"""CR-DD check 4b (DD8) — es_flow_price_1m (source 'ironbeam') vs ironbeam_es_trades, cell by cell, for one
Globex day [D-1 18:00 ET, D 17:00 ET). Mechanical (B4: the forward log is not read for direction or outcome):
every (minute, symbol, 0.25-pt price) cell's buy_vol, sell_vol and trade_count must equal the trades' own sums,
and no cell may exist on one side only.

Usage: python scripts/cr_dd_check_live.py --day 2026-10-27
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / ".env")

from packages.shared import flow_price_1m as F  # noqa: E402
from packages.shared.backfill_safety import get_backfill_db_conn  # noqa: E402

SQL = f"""
  WITH t AS (SELECT date_trunc('minute', ts_utc) minute_utc, symbol, floor(price * 4) / 4 price,
                    coalesce(sum(size) FILTER (WHERE aggressor_side = 'BUY'), 0) b,
                    coalesce(sum(size) FILTER (WHERE aggressor_side = 'SELL'), 0) s, count(*) n
             FROM {F.TRADES} WHERE ts_utc >= %(a)s AND ts_utc < %(b)s GROUP BY 1, 2, 3),
       c AS (SELECT minute_utc, symbol, price, buy_vol b, sell_vol s, trade_count n FROM {F.TABLE}
             WHERE source = 'ironbeam' AND minute_utc >= %(a)s AND minute_utc < %(b)s)
  SELECT count(*), count(*) FILTER (WHERE c.n IS NULL), count(*) FILTER (WHERE t.n IS NULL),
         count(*) FILTER (WHERE c.b = t.b AND c.s = t.s AND c.n = t.n),
         count(DISTINCT minute_utc), sum(t.b), sum(c.b), sum(t.s), sum(c.s), sum(t.n), sum(c.n)
  FROM t FULL JOIN c USING (minute_utc, symbol, price)
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", required=True)
    d = date.fromisoformat(ap.parse_args().day)
    a = datetime.combine(d - timedelta(days=1), time(18), F.ET)
    b = datetime.combine(d, time(17), F.ET)
    conn = get_backfill_db_conn()
    oldest = conn.execute(f"SELECT min(ts_utc) FROM {F.TRADES}").fetchone()[0]
    n, only_t, only_c, eq, mins, tb, cb, ts, cs, tn, cn = conn.execute(SQL, dict(a=a, b=b)).fetchone()
    print(f"Globex day {d}: [{a:%Y-%m-%d %H:%M %Z}, {b:%Y-%m-%d %H:%M %Z}); trades retained from {oldest:%Y-%m-%d %H:%M:%S}Z")
    print(f"cells {n:,} over {mins:,} minutes; equal {eq:,}; only in trades {only_t}; only in table {only_c}")
    print(f"buy  trades {tb:,} table {cb:,}\nsell trades {ts:,} table {cs:,}\ncount trades {tn:,} table {cn:,}")
    ok = n == eq and not only_t and not only_c and oldest <= a
    print("EXACT MATCH" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
