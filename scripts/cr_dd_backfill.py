#!/usr/bin/env python3
"""CR-DD — Databento train backfill of es_flow_price_1m (source 'databento') from the CR-CA raw files.

Sessions 2026-01-06 .. 2026-06-05 (train; the holdout is never opened, B4). Contracts per session = CR-CA
contracts_for (front, plus the dying contract in roll weeks; DD1). Ticks come only from the CR-CA L7 DBN files
already on disk (DD2): a missing file is an error, nothing is downloaded. Each file is read, filtered to the
CR-CA window [06:15, 13:00) PT on ts_event (DD7), aggregated in memory with flow_price_1m.aggregate_trades
(side B = buy, A = sell, N counted in trade_count only; DD3) and inserted ON CONFLICT DO NOTHING. No ticks are
stored. One backfill_run(conn, "CR-DD") for the whole load.

Smoke (raises -> run aborted):
  * DB rows for the window = built rows; every session has exactly its contracts_for set; prices on the grid;
  * check 4a: on 3 sessions, per minute and contract, Σ(buy + sell) vs es_flow_1m_databento.volume and
    Σ(buy − sell) vs its buy_vol − sell_vol, each within 1 % of the minute's volume; same minute set.

Usage (apps/web/.venv/bin/python):
  CR_CA_RAW_DIR=/Users/ryan/code/volatility-monorepo/data/databento/raw python scripts/cr_dd_backfill.py [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cr_ca_step0 import (  # noqa: E402  (also installs output redaction)
    DATASET, RAW_DIR, contracts_for, in_window, ironbeam_symbol, raw_symbol, sessions, window_utc,
)
from packages.shared import flow_price_1m as F  # noqa: E402
from packages.shared.backfill_safety import (  # noqa: E402
    assert_role_or_die, backfill_run, get_backfill_db_conn, update_run_progress, update_run_smoke,
)

START, END = date(2026, 1, 6), date(2026, 6, 5)
HOLDOUT_START = date(2026, 6, 6)
CHECK_SESSIONS = [date(2026, 1, 6), date(2026, 3, 16), date(2026, 5, 15)]  # plain, roll week (2 contracts), late train
TOL = 0.01
CHUNK = 50_000
PG_TYPES = ["timestamptz", "text", "float8", "int", "int", "int", "text"]


def raw_path(sym: str, d: date) -> Path:
    return RAW_DIR / f"{DATASET}_trades_{sym}_{d:%Y%m%d}_0615-1300PT.dbn.zst"


def build() -> pd.DataFrame:
    import databento as db
    frames = []
    for d in sessions(START, END):
        assert d < HOLDOUT_START
        for y, m in contracts_for(d):
            sym = raw_symbol(y, m)
            p = raw_path(sym, d)
            if not p.exists():
                raise FileNotFoundError(f"{p.name} not on disk; CR-DD downloads nothing (DD2)")
            tr = db.DBNStore.from_file(str(p)).to_df(pretty_ts=True, map_symbols=True).reset_index()
            if set(tr["symbol"].unique()) - {sym} or not (tr["action"] == "T").all():
                raise RuntimeError(f"{sym} {d}: unexpected symbols or non-trade actions")
            tr["ts"] = pd.to_datetime(tr["ts_event"], utc=True)
            tr = tr[in_window(tr["ts"])]
            f = F.aggregate_trades(tr, ironbeam_symbol(y, m), "ts", "side", "B", "A", "databento")
            f["session"], f["raw_symbol"] = d, sym
            frames.append(f)
    return pd.concat(frames, ignore_index=True)


def insert(conn, df: pd.DataFrame) -> int:
    sql = (f"INSERT INTO {F.TABLE} ({', '.join(F.COLS)}) SELECT * FROM unnest("
           + ", ".join(f"%s::{t}[]" for t in PG_TYPES) + ") ON CONFLICT DO NOTHING")
    n = 0
    for i in range(0, len(df), CHUNK):
        part = df.iloc[i:i + CHUNK]
        arrays = [[t.to_pydatetime() for t in part["minute_utc"]], part["symbol"].tolist(),
                  part["price"].astype(float).tolist()]
        arrays += [part[c].astype("int64").tolist() for c in ("buy_vol", "sell_vol", "trade_count")]
        arrays.append(part["source"].tolist())
        n += conn.execute(sql, arrays).rowcount
    return n


def check_4a(conn, d: date) -> dict:
    s, e = window_utc(d)
    q = f"""
      WITH c AS (SELECT minute_utc, symbol, sum(buy_vol + sell_vol) v, sum(buy_vol - sell_vol) dl
                 FROM {F.TABLE} WHERE source = 'databento' AND minute_utc >= %(s)s AND minute_utc < %(e)s GROUP BY 1, 2),
           r AS (SELECT minute_utc, symbol, volume v, buy_vol - sell_vol dl
                 FROM es_flow_1m_databento WHERE minute_utc >= %(s)s AND minute_utc < %(e)s)
      SELECT count(*), count(*) FILTER (WHERE c.v IS NULL), count(*) FILTER (WHERE r.v IS NULL),
             count(*) FILTER (WHERE abs(c.v - r.v) > %(tol)s * r.v),
             count(*) FILTER (WHERE abs(c.dl - r.dl) > %(tol)s * r.v),
             count(*) FILTER (WHERE c.v = r.v AND c.dl = r.dl),
             sum(c.v), sum(r.v), sum(c.dl), sum(r.dl), max(abs(c.v - r.v)), max(abs(c.dl - r.dl))
      FROM c FULL JOIN r USING (minute_utc, symbol)"""
    n, only_ref, only_cells, v_bad, dl_bad, exact, cv, rv, cdl, rdl, mv, mdl = conn.execute(
        q, dict(s=s.to_pydatetime(), e=e.to_pydatetime(), tol=TOL)).fetchone()
    res = dict(session=str(d), minutes=n, missing_in_cells=only_ref, missing_in_flow_1m=only_cells,
               vol_outside_1pct=v_bad, delta_outside_1pct=dl_bad, minutes_exact=exact,
               sum_vol_cells=int(cv or 0), sum_vol_flow_1m=int(rv or 0), sum_delta_cells=int(cdl or 0),
               sum_delta_flow_1m=int(rdl or 0), max_abs_vol_diff=int(mv or 0), max_abs_delta_diff=int(mdl or 0))
    res["pass"] = bool(n and not (only_ref or only_cells or v_bad or dl_bad)
                       and abs(res["sum_vol_cells"] - res["sum_vol_flow_1m"]) <= TOL * res["sum_vol_flow_1m"]
                       and abs(res["sum_delta_cells"] - res["sum_delta_flow_1m"]) <= TOL * res["sum_vol_flow_1m"])
    return res


def smoke(conn, built: pd.DataFrame, inserted: int) -> dict:
    s, _ = window_utc(START)
    _, e = window_utc(END)
    db = pd.read_sql(f"SELECT minute_utc, symbol, price FROM {F.TABLE} WHERE source = 'databento' "
                     f"AND minute_utc >= %s AND minute_utc < %s", conn, params=(s.to_pydatetime(), e.to_pydatetime()))
    fails = []
    if len(db) != len(built):
        fails.append(f"DB rows {len(db)} != built {len(built)}")
    off = int(((db["price"] * 4) % 1 != 0).sum())
    if off:
        fails.append(f"{off} prices off the 0.25 grid")
    got = built.groupby("session")["raw_symbol"].agg(lambda x: sorted(set(x)))
    bad = [str(d) for d in sessions(START, END) if got.get(d) != sorted(raw_symbol(*c) for c in contracts_for(d))]
    if bad:
        fails.append(f"contract set wrong on {bad[:5]}")
    checks = [check_4a(conn, d) for d in CHECK_SESSIONS]
    fails += [f"4a {c['session']} fails" for c in checks if not c["pass"]]
    return dict(rows_built=int(len(built)), rows_inserted=int(inserted), rows_db=int(len(db)),
                sessions=int(built["session"].nunique()), session_contracts=int(built.groupby(["session", "raw_symbol"]).ngroups),
                symbols=sorted(built["symbol"].unique().tolist()), off_grid=off,
                buy_vol=int(built["buy_vol"].sum()), sell_vol=int(built["sell_vol"].sum()),
                trade_count=int(built["trade_count"].sum()), check_4a=checks, fails=fails)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    print(f"raw dir {RAW_DIR}")
    built = build()
    print(f"built {len(built):,} rows; {built['session'].nunique()} sessions; "
          f"{built.groupby(['session', 'raw_symbol']).ngroups} session-contracts; symbols {sorted(built['symbol'].unique())}")
    if a.dry_run:
        return
    conn = get_backfill_db_conn()
    assert_role_or_die(conn)
    with backfill_run(conn, "CR-DD") as run_id:
        n = insert(conn, built)
        update_run_progress(conn, run_id, n)
        res = smoke(conn, built, n)
        ok = not res["fails"]
        update_run_smoke(conn, run_id, res, f"CR-DD Databento backfill {START}..{END}: {res['rows_db']:,} rows, "
                         f"{res['sessions']} sessions; 4a " + ("PASS" if ok else "FAIL: " + "; ".join(res["fails"])))
        print(f"inserted {n:,}; DB {res['rows_db']:,}; off-grid {res['off_grid']}; run {run_id}")
        for c in res["check_4a"]:
            print("  4a", c)
        if not ok:
            raise RuntimeError(f"smoke failed: {res['fails']}")
        print("PASS")


if __name__ == "__main__":
    main()
