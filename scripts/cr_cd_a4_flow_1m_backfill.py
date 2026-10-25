"""CR-CD A4.2 — backfill es_flow_1m_databento / ironbeam_es_flow_1m from the 1-second tables.

    apps/web/.venv/bin/python scripts/cr_cd_a4_flow_1m_backfill.py --source databento|ironbeam [--month 2026-05 | --from-month 2025-12] [--dry-run]

Backfill protocol (CLAUDE.md): get_backfill_db_conn, assert_role_or_die, one backfill_run(conn,
"CR-CD-A4") per (source, calendar month UTC), update_run_smoke. INSERT ... SELECT <rollup> ...
ON CONFLICT DO NOTHING (packages/shared/flow_1m.py), so a re-run inserts 0. The Ironbeam end is the
run's start floored to the minute minus 10 min (the worker covers from there, A4.3).

Per-month smoke (all must hold, else the month's run aborts):
  - 1-minute rows in the month == distinct (symbol, minute) in the 1-second table;
  - sum(volume), sum(buy_vol), sum(sell_vol), sum(unknown_vol), sum(trade_count) equal;
  - sum(n_seconds) == 1-second row count.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
for line in (REPO / ".env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        import os
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

from packages.shared import flow_1m  # noqa: E402
from packages.shared.backfill_safety import (  # noqa: E402
    assert_role_or_die, backfill_run, get_backfill_db_conn, update_run_smoke,
)

UTC = dt.timezone.utc
TABLE_1S = {"databento": "es_flow_1s_databento", "ironbeam": "ironbeam_es_flow_1s"}
WHERE = "ts_utc >= %(a)s AND ts_utc < %(b)s"


def months(a: dt.datetime, b: dt.datetime):
    cur = dt.datetime(a.year, a.month, 1, tzinfo=UTC)
    while cur < b:
        nxt = dt.datetime(cur.year + (cur.month == 12), cur.month % 12 + 1, 1, tzinfo=UTC)
        yield max(cur, a), min(nxt, b)
        cur = nxt


def stats_1s(conn, t1s, a, b):
    return conn.execute(f"""
        SELECT count(DISTINCT (symbol, floor(extract(epoch FROM ts_utc) / 60))), sum(volume), sum(buy_vol),
               sum(sell_vol), sum(unknown_vol), sum(trade_count), count(*)
        FROM {t1s} WHERE {WHERE}""", {"a": a, "b": b}).fetchone()


def stats_1m(conn, t1m, a, b):
    return conn.execute(f"""
        SELECT count(*), sum(volume), sum(buy_vol), sum(sell_vol), sum(unknown_vol), sum(trade_count),
               sum(n_seconds)
        FROM {t1m} WHERE minute_utc >= %(a)s AND minute_utc < %(b)s""", {"a": a, "b": b}).fetchone()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=sorted(TABLE_1S), required=True)
    ap.add_argument("--month", help="YYYY-MM (UTC) only")
    ap.add_argument("--from-month", help="YYYY-MM (UTC): resume from this month")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    conn = get_backfill_db_conn()
    assert_role_or_die(conn)
    t1s = TABLE_1S[args.source]
    t1m = flow_1m.TABLE_1M[t1s]
    started = dt.datetime.now(UTC)
    lo, hi = conn.execute(f"SELECT min(ts_utc), max(ts_utc) FROM {t1s}").fetchone()
    a0 = lo.astimezone(UTC).replace(second=0, microsecond=0)
    b0 = hi.astimezone(UTC) + dt.timedelta(seconds=1)
    if args.source == "ironbeam":
        b0 = min(b0, started.replace(second=0, microsecond=0) - dt.timedelta(minutes=10))
    print(f"{t1s} -> {t1m}: [{a0.isoformat()}, {b0.isoformat()})")
    ins_sql = flow_1m.insert_sql(t1s, WHERE, on_conflict="nothing")
    summary = []
    for a, b in months(a0, b0):
        if args.month and a.strftime("%Y-%m") != args.month:
            continue
        if args.from_month and a.strftime("%Y-%m") < args.from_month:
            continue
        conn = get_backfill_db_conn()   # fresh connection per month: an SSL drop costs one month, not the run
        assert_role_or_die(conn)
        exp = stats_1s(conn, t1s, a, b)
        if args.dry_run:
            print(a.strftime("%Y-%m"), "1s distinct minutes", exp[0], "1s rows", exp[6])
            continue
        with backfill_run(conn, "CR-CD-A4") as run_id:
            t = time.time()
            cur = conn.execute(ins_sql, {"a": a, "b": b})
            inserted = cur.rowcount
            got = stats_1m(conn, t1m, a, b)
            smoke = {
                "source": args.source, "table": t1m, "from": a.isoformat(), "to": b.isoformat(),
                "inserted": inserted, "rows_1m": got[0], "distinct_minutes_1s": exp[0],
                "volume": [int(got[1] or 0), int(exp[1] or 0)], "buy_vol": [int(got[2] or 0), int(exp[2] or 0)],
                "sell_vol": [int(got[3] or 0), int(exp[3] or 0)], "unknown_vol": [int(got[4] or 0), int(exp[4] or 0)],
                "trade_count": [int(got[5] or 0), int(exp[5] or 0)], "n_seconds_vs_rows_1s": [int(got[6] or 0), int(exp[6])],
                "seconds": round(time.time() - t, 1),
            }
            ok = (got[0] == exp[0] and all(smoke[k][0] == smoke[k][1] for k in
                  ("volume", "buy_vol", "sell_vol", "unknown_vol", "trade_count", "n_seconds_vs_rows_1s")))
            smoke["pass"] = ok
            conn.execute("UPDATE bt_backfill_runs SET rows_inserted = %s WHERE run_id = %s", (inserted, run_id))
            update_run_smoke(conn, run_id, smoke, f"CR-CD A4 {t1m} {a:%Y-%m}: {'PASS' if ok else 'FAIL'}")
            print(json.dumps(smoke))
            if not ok:
                raise RuntimeError(f"smoke failed for {a:%Y-%m}")
            summary.append((a.strftime("%Y-%m"), inserted, got[0], ok, run_id))
    for row in summary:
        print(*row)


if __name__ == "__main__":
    main()
