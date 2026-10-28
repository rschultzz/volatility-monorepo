#!/usr/bin/env python3
"""CR-DD — live es_flow_price_1m from ironbeam_es_trades (Render cron, */5 * * * 0-5 UTC).

Each run: every completed minute (packages/shared/flow_price_1m.is_held, DD4) in the last 47 h of retained
trades that is not yet stored for source 'ironbeam' is aggregated in one INSERT ... SELECT. Idempotent (stored
minutes are skipped; ON CONFLICT DO NOTHING) and safe to overlap. Runs every 5 min, so a minute is written
within ~10 min of completing (closing minutes after 18:00 ET), far ahead of the 01:00 UTC prune (rows live ~48 h).

Protocol (DD6): BACKFILL_DATABASE_URL + assert_role_or_die on every run; backfill_run(conn, "CR-DD-live") and
update_run_smoke only on runs that have minutes to write. The smoke re-sums the written minutes against
ironbeam_es_trades (buy / sell / count per minute and symbol must be equal) and fails the run otherwise.
A trade that arrives after its minute is stored is not added (insert-only; Step 0 S3).

Usage: python scripts/cron_flow_price_1m.py [--now 2026-10-28T03:30:00Z] [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

from packages.shared import flow_price_1m as F  # noqa: E402
from packages.shared.backfill_safety import (  # noqa: E402
    assert_role_or_die, backfill_run, get_backfill_db_conn, update_run_progress, update_run_smoke,
)

CR_ID = "CR-DD-live"

CHECK_SQL = f"""
  WITH w AS (SELECT DISTINCT minute_utc, symbol FROM unnest(%(m)s::timestamptz[], %(s)s::text[]) AS u(minute_utc, symbol)),
  a AS (SELECT f.minute_utc, f.symbol, sum(buy_vol) b, sum(sell_vol) s, sum(trade_count) n
        FROM {F.TABLE} f JOIN w USING (minute_utc, symbol) WHERE source = 'ironbeam' GROUP BY 1, 2),
  t AS (SELECT date_trunc('minute', ts_utc) minute_utc, symbol,
               coalesce(sum(size) FILTER (WHERE aggressor_side = 'BUY'), 0) b,
               coalesce(sum(size) FILTER (WHERE aggressor_side = 'SELL'), 0) s, count(*) n
        FROM {F.TRADES} WHERE ts_utc >= %(lo)s AND ts_utc < %(hi)s GROUP BY 1, 2)
  SELECT count(*), count(*) FILTER (WHERE t.b IS DISTINCT FROM a.b OR t.s IS DISTINCT FROM a.s
                                           OR t.n IS DISTINCT FROM a.n),
         coalesce(sum(a.b), 0), coalesce(sum(a.s), 0), coalesce(sum(a.n), 0)
  FROM a LEFT JOIN t USING (minute_utc, symbol)
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--now", help="ISO UTC time to act as now (testing)")
    ap.add_argument("--dry-run", action="store_true", help="count pending minutes only")
    a = ap.parse_args()
    now = datetime.fromisoformat(a.now.replace("Z", "+00:00")) if a.now else datetime.now(timezone.utc)
    since, end = F.window(now)
    p = dict(since=since, end=end, now=now)

    conn = get_backfill_db_conn()
    assert_role_or_die(conn)
    minutes, trades = conn.execute(F.LIVE_COUNT_SQL, p).fetchone()
    print(f"[flow_price_1m] now {now:%Y-%m-%d %H:%M:%S}Z window [{since:%m-%d %H:%M}, {end:%m-%d %H:%M}) UTC: "
          f"{minutes} pending minute-symbols, {trades} trades")
    if not minutes or a.dry_run:
        return

    with backfill_run(conn, CR_ID) as run_id:
        cur = conn.execute(F.LIVE_INSERT_SQL, p)
        written = cur.fetchall()
        update_run_progress(conn, run_id, len(written))
        keys = sorted({(m, s) for m, s in written})
        lo, hi = keys[0][0], max(k[0] for k in keys) + timedelta(minutes=1)
        n_min, n_bad, b, s, n = conn.execute(CHECK_SQL, dict(
            m=[k[0] for k in keys], s=[k[1] for k in keys], lo=lo, hi=hi)).fetchone()
        res = dict(now=now.isoformat(), rows=len(written), minutes=len(keys), minutes_checked=n_min,
                   minutes_mismatch=n_bad, buy_vol=int(b), sell_vol=int(s), trade_count=int(n),
                   first_minute=lo.isoformat(), last_minute=keys[-1][0].isoformat(),
                   symbols=sorted({k[1] for k in keys}))
        ok = n_bad == 0 and n_min == len(keys)
        update_run_smoke(conn, run_id, res, f"CR-DD live: {len(written)} rows, {len(keys)} minutes "
                         f"{lo:%m-%d %H:%M}..{keys[-1][0]:%m-%d %H:%M} UTC; trades re-sum "
                         + ("equal" if ok else f"MISMATCH on {n_bad}"))
        print(f"[flow_price_1m] wrote {len(written)} rows over {len(keys)} minutes "
              f"({lo:%m-%d %H:%M} .. {keys[-1][0]:%m-%d %H:%M} UTC); re-sum vs trades: "
              f"{'equal' if ok else f'MISMATCH {n_bad}'} (run {run_id})")
        if not ok:
            raise RuntimeError(f"re-sum mismatch on {n_bad} of {n_min} minutes")


if __name__ == "__main__":
    main()
