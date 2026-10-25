#!/usr/bin/env python3
"""CR-CA — Databento ES trades -> es_flow_1s_databento, month by month (Implementation step 1/4).

For each month of 2024-04-03 .. 2026-10-23 (A3.3):
  1. download one DBN file per session-contract window [06:15, 13:00) PT (L3 contracts; roll weeks store
     both) to the L7 raw path, reusing files already there (Q2 / Q4 / earlier runs);
  2. aggregate to 1 s with cr_ca_step0.aggregate_1s (ironbeam_flow_1s_aggregate rules; A1 symbol,
     raw_symbol, trade_date = UTC date, bucket on ts_event, end-exclusive window);
  3. INSERT ... ON CONFLICT DO NOTHING as dash_backfill_writer inside backfill_run(conn, "CR-CA");
  4. smoke from the DB: rows, symbols per session vs the L3 calendar, window bounds, trade_date convention,
     sign test corr(minute delta, minute close - open) > 0 on the front contract.  Any smoke failure raises
     (run marked aborted) and stops the remaining months.

Cost guard: the priced total in scripts/logs/cr_ca/cost_trades.csv (cr_ca_step0.py cost) must be <= the A3.4
cap before anything is downloaded.  Smokes on holdout months (> 2026-06-05) are mechanical feed checks only.

Usage (apps/web/.venv/bin/python):
  CR_CA_RAW_DIR=<path> python scripts/cr_ca_databento_flow.py --month 2024-04 [--month 2024-05 ...]
  CR_CA_RAW_DIR=<path> python scripts/cr_ca_databento_flow.py --all
  ... --dry-run   download + aggregate + in-memory checks, no DB writes
"""
from __future__ import annotations

import argparse
import sys
import time as _time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cr_ca_step0 import (  # noqa: E402  (also installs output redaction, A2.4)
    COST_CAP, DATASET, LOG_DIR, PT, R_END, R_START, RAW_DIR, WIN_END, WIN_START,
    aggregate_1s, client, contracts_for, front_for, in_window, ironbeam_symbol, raw_symbol, sessions,
    window_utc,
)
from packages.shared.backfill_safety import (  # noqa: E402
    assert_role_or_die, backfill_run, get_backfill_db_conn, update_run_progress, update_run_smoke,
)

TABLE = "es_flow_1s_databento"
COLS = ["ts_utc", "symbol", "raw_symbol", "trade_date", "volume", "notional", "trade_count",
        "buy_vol", "sell_vol", "unknown_vol", "buy_count", "sell_count", "unknown_count",
        "open", "high", "low", "close", "first_trade_ms", "last_trade_ms", "source"]
PG_TYPES = ["timestamptz", "text", "text", "date", "int", "float8", "int",
            "int", "int", "int", "int", "int", "int",
            "float8", "float8", "float8", "float8", "int8", "int8", "text"]
CHUNK = 50_000


def raw_path(sym: str, d: date) -> Path:
    return RAW_DIR / f"{DATASET}_trades_{sym}_{d:%Y%m%d}_0615-1300PT.dbn.zst"


def download(job, cl):
    sym, d = job
    p = raw_path(sym, d)
    if p.exists():
        return p, False
    s, e = window_utc(d)
    tmp = p.with_suffix(".part")
    for attempt in range(4):
        try:
            cl.timeseries.get_range(dataset=DATASET, schema="trades", symbols=[sym], stype_in="raw_symbol",
                                    start=s.isoformat(), end=e.isoformat(), path=str(tmp))
            tmp.rename(p)
            return p, True
        except Exception:
            if attempt == 3:
                raise
            _time.sleep(5 * (attempt + 1))


def month_jobs(month: str) -> list[tuple[str, date, tuple[int, int]]]:
    out = []
    for d in sessions(R_START, R_END):
        if f"{d:%Y-%m}" == month:
            for c in contracts_for(d):
                out.append((raw_symbol(*c), d, c))
    return out


def build_month(month: str) -> pd.DataFrame:
    import databento as db
    jobs = month_jobs(month)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cl = client()
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(lambda j: download((j[0], j[1]), cl), jobs))
    n_new = sum(new for _, new in res)
    mb = sum(p.stat().st_size for p, _ in res) / 1e6
    print(f"  {len(jobs)} session-contract files ({n_new} downloaded now, {len(jobs) - n_new} reused; {mb:,.0f} MB)")
    frames = []
    for (sym, d, (y, m)), (p, _) in zip(jobs, res):
        tr = db.DBNStore.from_file(str(p)).to_df(pretty_ts=True, map_symbols=True)
        if tr.empty:
            print(f"  WARNING {sym} {d}: no trades in the window")
            continue
        tr = tr.reset_index()
        bad = set(tr["symbol"].unique()) - {sym}
        if bad or not (tr["action"] == "T").all():
            raise RuntimeError(f"{sym} {d}: unexpected symbols {bad} or non-trade actions")
        f = aggregate_1s(tr, y, m)
        f = f[in_window(f["ts_utc"])]
        f["session"] = d
        frames.append(f)
    return pd.concat(frames, ignore_index=True)


def insert(conn, df: pd.DataFrame) -> int:
    cols = ", ".join(COLS)
    unnest = ", ".join(f"%s::{t}[]" for t in PG_TYPES)
    sql = (f"INSERT INTO {TABLE} ({cols}) SELECT * FROM unnest({unnest}) "
           f"ON CONFLICT (ts_utc, symbol) DO NOTHING")
    n = 0
    for i in range(0, len(df), CHUNK):
        part = df.iloc[i:i + CHUNK]
        arrays = []
        for c in COLS:
            v = part[c]
            if c == "ts_utc":
                arrays.append([t.to_pydatetime() for t in v])
            elif c in ("open", "high", "low", "close", "notional"):
                arrays.append(v.astype(float).tolist())
            elif c in ("symbol", "raw_symbol", "source", "trade_date"):
                arrays.append(v.tolist())
            else:
                arrays.append(v.astype("int64").tolist())
        cur = conn.execute(sql, arrays)
        n += cur.rowcount
    return n


def smoke(conn, month: str, built: pd.DataFrame, inserted: int) -> dict:
    ss = sorted(built["session"].unique()) if not built.empty else []
    exp_sessions = [d for d in sessions(R_START, R_END) if f"{d:%Y-%m}" == month]
    s0, _ = window_utc(exp_sessions[0])
    _, e1 = window_utc(exp_sessions[-1])
    db = pd.read_sql(
        f"SELECT ts_utc, symbol, raw_symbol, trade_date, volume, buy_vol, sell_vol, unknown_vol, open, close "
        f"FROM {TABLE} WHERE ts_utc >= %s AND ts_utc < %s", conn,
        params=(s0.to_pydatetime(), e1.to_pydatetime()))
    db["ts_utc"] = pd.to_datetime(db["ts_utc"], utc=True)
    db["session"] = db["ts_utc"].dt.tz_convert(PT).dt.date
    fails = []

    rows_db = len(db)
    if rows_db != len(built):
        fails.append(f"DB rows {rows_db} != built {len(built)}")
    outside = int((~in_window(db["ts_utc"])).sum())
    if outside:
        fails.append(f"{outside} rows outside [06:15, 13:00) PT")
    td_bad = int((pd.to_datetime(db["trade_date"]).dt.date != db["ts_utc"].dt.date).sum())
    if td_bad:
        fails.append(f"{td_bad} rows with trade_date != UTC date")

    sym_bad = []
    for d in exp_sessions:
        got = set(db.loc[db.session == d, "raw_symbol"].unique())
        want = {raw_symbol(*c) for c in contracts_for(d)}
        if got != want:
            sym_bad.append(f"{d}: got {sorted(got)} want {sorted(want)}")
    map_bad = db.groupby("raw_symbol")["symbol"].nunique().gt(1).sum()
    if sym_bad:
        fails.append("symbols: " + "; ".join(sym_bad[:5]))
    if map_bad:
        fails.append("raw_symbol maps to > 1 symbol")

    fr = db[[raw_symbol(*front_for(d)) == r for d, r in zip(db.session, db.raw_symbol)]].copy()
    fr["minute"] = fr["ts_utc"].dt.floor("min")
    fr = fr.sort_values("ts_utc")
    g = fr.groupby(["raw_symbol", "minute"])
    mins = pd.DataFrame({"delta": g["buy_vol"].sum() - g["sell_vol"].sum(),
                         "dp": g["close"].last() - g["open"].first()})
    sign = float(mins["delta"].corr(mins["dp"]))
    if not sign > 0:
        fails.append(f"sign test {sign:+.4f} <= 0")

    rows_per = db.groupby(["session", "raw_symbol"]).size()
    front_rows = [rows_per.get((d, raw_symbol(*front_for(d))), 0) for d in exp_sessions]
    res = dict(
        month=month, sessions=len(exp_sessions), sessions_with_rows=len(ss), rows_built=int(len(built)),
        rows_inserted=int(inserted), rows_db=int(rows_db),
        symbols=sorted(db["symbol"].unique().tolist()), raw_symbols=sorted(db["raw_symbol"].unique().tolist()),
        two_contract_sessions=int(sum(len(contracts_for(d)) == 2 for d in exp_sessions)),
        front_rows_per_session_min=int(min(front_rows)), front_rows_per_session_median=float(np.median(front_rows)),
        volume=int(db["volume"].sum()), unknown_vol_share=float(db["unknown_vol"].sum() / max(db["volume"].sum(), 1)),
        sign_test=round(sign, 4), front_minutes=int(len(mins)), outside_window=outside, trade_date_bad=td_bad,
        fails=fails,
    )
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", action="append", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    months = sorted({f"{d:%Y-%m}" for d in sessions(R_START, R_END)}) if a.all else a.month
    if not months:
        sys.exit("pass --month YYYY-MM or --all")

    cost = pd.read_csv(LOG_DIR / "cost_trades.csv")
    priced = cost["usd"].sum()
    if priced > COST_CAP:
        sys.exit(f"STOP: priced trades total ${priced:,.2f} > cap ${COST_CAP:.0f}")
    cost["month"] = pd.to_datetime(cost["session"]).dt.strftime("%Y-%m")
    print(f"cost guard: priced total ${priced:,.2f} <= ${COST_CAP:.0f}; raw dir {RAW_DIR}")

    conn = None
    if not a.dry_run:
        conn = get_backfill_db_conn()
        assert_role_or_die(conn)

    summary = []
    for month in months:
        t0 = _time.time()
        print(f"\n=== {month}  (priced ${cost.loc[cost.month == month, 'usd'].sum():.2f}) ===")
        built = build_month(month)
        print(f"  built {len(built):,} rows; symbols {sorted(built['raw_symbol'].unique())}")
        if a.dry_run:
            continue
        with backfill_run(conn, "CR-CA") as run_id:
            n = insert(conn, built)
            update_run_progress(conn, run_id, n)
            res = smoke(conn, month, built, n)
            res["run_id"] = run_id
            ok = not res["fails"]
            update_run_smoke(conn, run_id, res,
                             f"CR-CA {month}: {res['rows_db']:,} rows, {res['raw_symbols']}, sign {res['sign_test']:+.3f} "
                             + ("PASS" if ok else "FAIL: " + "; ".join(res["fails"])))
            print(f"  inserted {n:,}; DB {res['rows_db']:,}; sessions {res['sessions_with_rows']}/{res['sessions']}; "
                  f"front rows/session min {res['front_rows_per_session_min']:,} median "
                  f"{res['front_rows_per_session_median']:,.0f}; unknown share {res['unknown_vol_share']:.4f}; "
                  f"sign {res['sign_test']:+.4f}; {'PASS' if ok else 'FAIL'}  ({_time.time() - t0:.0f} s, run {run_id})")
            summary.append(res)
            if not ok:
                raise RuntimeError(f"smoke failed for {month}: {res['fails']}")
    if summary:
        out = pd.DataFrame(summary)
        p = LOG_DIR / "backfill_smoke.csv"
        out.to_csv(p, mode="a", header=not p.exists(), index=False)


if __name__ == "__main__":
    main()
