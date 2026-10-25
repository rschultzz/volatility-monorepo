#!/usr/bin/env python3
"""CR-CA Step 0 — Databento ES trades backfill diagnosis (no DB writes).

Phases (run in order; each prints its section of the Step 0 findings):
  cost     Q1a/b  metadata.get_cost per session window, trades + MBP-1, per contract per month.
  pull     Q2     download the overlap window (2026-01-12 .. 2026-01-23, ESH6), only if the
                  full-window trades total from `cost` is <= $150 (L8 / A1.6).
  overlap  Q2     aggregate to 1 s (A1 rules) and compare with ironbeam_es_flow_1s per minute.
  size     Q1c/d  project the RTH 1 s table size from the pulled window; DB free space.
  roll     Q3     roll calendar under L3.
  bars     Q4     Databento-built 1-minute OHLC vs ironbeam_es_1m_bars on 10 random 2025 sessions
                  (seed 20261025, the cloud run's sample).  Downloads trades for those sessions
                  only if the user passes --allow-q4-pull (it is not part of the A1.6 budget).
  ddl      Q5     proposed DDL and whether dash_backfill_writer can CREATE.

Window (A1.4): [06:15:00, 13:00:00) America/Los_Angeles on NYSE sessions, end-exclusive.
Symbols (L2/L3): outright quarterlies, stype_in=raw_symbol; front rolls on the Monday of the week
before expiry week; roll weeks (that Monday .. expiry Friday) store both contracts.

Run with apps/web/.venv/bin/python (arm64 psycopg).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from packages.shared.trading_calendar import is_trading_day  # noqa: E402


# ---------------------------------------------------------------- redaction (A2.4)

_URL_RE = re.compile(r"postgres(?:ql)?(?:\+\w+)?://[^\s'\"]+")


def redact(text: str) -> str:
    """Mask DATABASE_URL / BACKFILL_DATABASE_URL (and any postgres URL) in output."""
    for k in ("DATABASE_URL", "BACKFILL_DATABASE_URL"):
        v = os.environ.get(k)
        if v:
            for form in {v, v.replace("+psycopg", "")}:
                text = text.replace(form, f"<{k}>")
    return _URL_RE.sub("<postgres-url>", text)


class _RedactingStream:
    def __init__(self, stream):
        self._s = stream

    def write(self, text):
        return self._s.write(redact(text))

    def __getattr__(self, name):
        return getattr(self._s, name)


def install_redaction():
    """Route stdout/stderr (including tracebacks) through redact()."""
    if not isinstance(sys.stdout, _RedactingStream):
        sys.stdout = _RedactingStream(sys.stdout)
        sys.stderr = _RedactingStream(sys.stderr)


install_redaction()

DATASET = "GLBX.MDP3"
PT = ZoneInfo("America/Los_Angeles")
WIN_START, WIN_END = time(6, 15), time(13, 0)
R_START, R_END = date(2024, 4, 3), date(2025, 12, 31)
OVL_START, OVL_END, OVL_SYM = date(2026, 1, 12), date(2026, 1, 23), "ESH6"
COST_CAP = 215.0  # A2.1 (was 150, L8)
Q4_SEED = 20261025

RAW_DIR = ROOT / "data" / "databento" / "raw"
LOG_DIR = ROOT / "scripts" / "logs" / "cr_ca"
MONTH_CODES = {3: "H", 6: "M", 9: "U", 12: "Z"}


def hdr(s):
    print("\n" + "=" * 100 + f"\n{s}\n" + "=" * 100)


# ---------------------------------------------------------------- calendar (L3)

def third_friday(y: int, m: int) -> date:
    d = date(y, m, 15)
    return d + timedelta(days=(4 - d.weekday()) % 7)


def roll_monday(y: int, m: int) -> date:
    return third_friday(y, m) - timedelta(days=11)


def raw_symbol(y: int, m: int) -> str:
    return f"ES{MONTH_CODES[m]}{y % 10}"


def ironbeam_symbol(y: int, m: int) -> str:
    return f"XCME:ES.{MONTH_CODES[m]}{y % 100:02d}"


def quarterlies(a: date, b: date) -> list[tuple[int, int]]:
    out = []
    for y in range(a.year, b.year + 2):
        for m in (3, 6, 9, 12):
            out.append((y, m))
    return out


def front_for(d: date) -> tuple[int, int]:
    """Front of (y, m) = [roll Monday of the previous quarterly, own roll Monday)."""
    for y, m in quarterlies(d - timedelta(days=120), d):
        py, pm = (y, m - 3) if m > 3 else (y - 1, 12)
        if roll_monday(py, pm) <= d < roll_monday(y, m):
            return y, m
    raise ValueError(d)


def contracts_for(d: date) -> list[tuple[int, int]]:
    """Contracts stored on session d: the front, plus the dying contract during its roll weeks
    (its roll Monday .. its expiry Friday, the CR-BM/BO roll-week span)."""
    stored = {front_for(d)}
    for y, m in quarterlies(d - timedelta(days=120), d):
        if roll_monday(y, m) <= d <= third_friday(y, m):
            stored.add((y, m))
    return sorted(stored)


def sessions(a: date, b: date) -> list[date]:
    out, d = [], a
    while d <= b:
        if is_trading_day(d):
            out.append(d)
        d += timedelta(days=1)
    return out


def window_utc(d: date) -> tuple[pd.Timestamp, pd.Timestamp]:
    s = datetime.combine(d, WIN_START, PT)
    e = datetime.combine(d, WIN_END, PT)
    return pd.Timestamp(s).tz_convert("UTC"), pd.Timestamp(e).tz_convert("UTC")


# ---------------------------------------------------------------- Databento

def client():
    import databento as db
    return db.Historical(os.environ["DATABENTO_API_KEY"])


def cost_rows(schema: str) -> pd.DataFrame:
    jobs = []
    for d in sessions(R_START, R_END):
        for c in contracts_for(d):
            jobs.append((d, c, "R"))
    for d in sessions(OVL_START, OVL_END):
        jobs.append((d, (2026, 3), "Q2"))
    cl = client()

    def one(j):
        d, (y, m), part = j
        s, e = window_utc(d)
        usd = cl.metadata.get_cost(dataset=DATASET, schema=schema, symbols=[raw_symbol(y, m)],
                                   stype_in="raw_symbol", start=s.isoformat(), end=e.isoformat())
        size = cl.metadata.get_billable_size(dataset=DATASET, schema=schema, symbols=[raw_symbol(y, m)],
                                             stype_in="raw_symbol", start=s.isoformat(), end=e.isoformat())
        return dict(session=d, raw_symbol=raw_symbol(y, m), part=part, usd=float(usd), bytes=int(size))

    with ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(one, jobs))
    return pd.DataFrame(rows)


def phase_cost():
    hdr("Q1 — cost (metadata.get_cost per session window [06:15, 13:00) PT, stype_in=raw_symbol)")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for schema in ("trades", "mbp-1"):
        df = cost_rows(schema)
        df.to_csv(LOG_DIR / f"cost_{schema}.csv", index=False)
        df["month"] = pd.to_datetime(df["session"]).dt.strftime("%Y-%m")
        r = df[df.part == "R"]
        piv = r.pivot_table(index="month", columns="raw_symbol", values="usd", aggfunc="sum").fillna(0)
        print(f"\n--- {schema}: USD per month per contract (R window) ---")
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(piv.round(2).to_string())
        tot_r, tot_q2 = r.usd.sum(), df[df.part == "Q2"].usd.sum()
        gb_r = r.bytes.sum() / 1e9
        print(f"{schema}: R total ${tot_r:,.2f} ({gb_r:,.2f} GB billable, {len(r)} session-contracts); "
              f"Q2 overlap ${tot_q2:,.2f}; total ${tot_r + tot_q2:,.2f}")
        out[schema] = dict(r_usd=tot_r, q2_usd=tot_q2, total_usd=tot_r + tot_q2, r_gb=gb_r)
    (LOG_DIR / "cost_summary.json").write_text(json.dumps(out, indent=2))
    ok = out["trades"]["total_usd"] <= COST_CAP
    print(f"\nL8 gate: trades total ${out['trades']['total_usd']:,.2f} vs cap ${COST_CAP:.0f} -> "
          f"{'PASS (Q2 pull allowed)' if ok else 'STOP'}")


def fetch_trades(sym: str, days: list[date], tag: str) -> pd.DataFrame:
    """Download (or reuse) one DBN file per session window under the L7 path; return trades."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    import databento as db
    cl = None
    frames = []
    for d in days:
        p = RAW_DIR / f"{DATASET}_trades_{sym}_{d:%Y%m%d}_0615-1300PT.dbn.zst"
        if not p.exists():
            cl = cl or client()
            s, e = window_utc(d)
            cl.timeseries.get_range(dataset=DATASET, schema="trades", symbols=[sym], stype_in="raw_symbol",
                                    start=s.isoformat(), end=e.isoformat(), path=str(p))
            print(f"  [{tag}] downloaded {p.name} ({p.stat().st_size / 1e6:.1f} MB)")
        store = db.DBNStore.from_file(str(p))
        df = store.to_df(pretty_ts=True, map_symbols=True)
        if df.empty:
            continue
        df = df.reset_index()
        df["session"] = d
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def phase_pull():
    summ = json.loads((LOG_DIR / "cost_summary.json").read_text())
    tot = summ["trades"]["total_usd"]
    if tot > COST_CAP:
        sys.exit(f"STOP: full-window trades total ${tot:,.2f} > ${COST_CAP:.0f}; no Q2 pull (L8).")
    hdr(f"Q2 pull — {OVL_SYM} trades {OVL_START} .. {OVL_END}, [06:15, 13:00) PT (gate ${tot:,.2f} <= ${COST_CAP:.0f})")
    df = fetch_trades(OVL_SYM, sessions(OVL_START, OVL_END), "Q2")
    print(f"  {len(df):,} trades; columns {list(df.columns)}")
    print(f"  side counts: {df['side'].value_counts().to_dict()}")
    print(f"  action counts: {df['action'].value_counts().to_dict()}")
    print(f"  symbols: {df['symbol'].value_counts().to_dict()}")


# ---------------------------------------------------------------- aggregation (A1 + ironbeam rules)

def aggregate_1s(tr: pd.DataFrame, y: int, m: int) -> pd.DataFrame:
    """Same rules as apps/cron/ironbeam_flow_1s_aggregate.aggregate_1s, on Databento columns:
    bucket = ts_event floored to the second (A1.3); side B = buy aggressor, A = sell, N = unknown (L6);
    first/last_trade_ms from ts_event; trade_date = UTC date of ts_utc (A1.2)."""
    df = tr.copy()
    df["ts_ev"] = pd.to_datetime(df["ts_event"], utc=True)
    df["ts_sec"] = df["ts_ev"].dt.floor("s")
    is_buy, is_sell = df["side"] == "B", df["side"] == "A"
    df["buy_vol"] = df["size"].where(is_buy, 0)
    df["sell_vol"] = df["size"].where(is_sell, 0)
    df["unknown_vol"] = df["size"].where(~(is_buy | is_sell), 0)
    df["buy_count"] = is_buy.astype("int64")
    df["sell_count"] = is_sell.astype("int64")
    df["unknown_count"] = (~(is_buy | is_sell)).astype("int64")
    df["notional"] = df["price"] * df["size"]
    df["ts_ms"] = (df["ts_ev"].astype("int64") // 1_000_000).astype("int64")
    df = df.sort_values(["ts_ev", "sequence"], kind="stable")
    gb = df.groupby("ts_sec", sort=True)
    out = pd.DataFrame({
        "volume": gb["size"].sum().astype("int64"),
        "notional": gb["notional"].sum(),
        "trade_count": gb["size"].count().astype("int64"),
        "buy_vol": gb["buy_vol"].sum().astype("int64"),
        "sell_vol": gb["sell_vol"].sum().astype("int64"),
        "unknown_vol": gb["unknown_vol"].sum().astype("int64"),
        "buy_count": gb["buy_count"].sum().astype("int64"),
        "sell_count": gb["sell_count"].sum().astype("int64"),
        "unknown_count": gb["unknown_count"].sum().astype("int64"),
        "open": gb["price"].first(), "high": gb["price"].max(),
        "low": gb["price"].min(), "close": gb["price"].last(),
        "first_trade_ms": gb["ts_ms"].min(), "last_trade_ms": gb["ts_ms"].max(),
    }).reset_index().rename(columns={"ts_sec": "ts_utc"})
    out.insert(1, "symbol", ironbeam_symbol(y, m))
    out.insert(2, "raw_symbol", raw_symbol(y, m))
    out.insert(3, "trade_date", out["ts_utc"].dt.date)
    out["source"] = "databento"
    return out


def in_window(ts: pd.Series) -> pd.Series:
    t = ts.dt.tz_convert(PT).dt.time
    return (t >= WIN_START) & (t < WIN_END)


def to_minutes(f: pd.DataFrame) -> pd.DataFrame:
    f = f.sort_values("ts_utc").copy()
    f["minute"] = f["ts_utc"].dt.floor("min")
    f["delta"] = f["buy_vol"] - f["sell_vol"]
    g = f.groupby("minute")
    return pd.DataFrame({"volume": g["volume"].sum(), "delta": g["delta"].sum(),
                         "buy": g["buy_vol"].sum(), "sell": g["sell_vol"].sum(), "unk": g["unknown_vol"].sum(),
                         "open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                         "close": g["close"].last(), "n_sec": g.size()})


def db_conn():
    import psycopg
    return psycopg.connect(os.environ["DATABASE_URL"].replace("+psycopg", ""))


def overlap_flow():
    tr = fetch_trades(OVL_SYM, sessions(OVL_START, OVL_END), "Q2")
    return tr, aggregate_1s(tr, 2026, 3)


def phase_overlap():
    import numpy as np
    hdr("Q2 — overlap validation vs ironbeam_es_flow_1s (ESH6, 2026-01-12 .. 2026-01-23, [06:15, 13:00) PT)")
    tr, dbf = overlap_flow()
    dbf = dbf[in_window(dbf["ts_utc"])]
    s0, _ = window_utc(OVL_START)
    _, e1 = window_utc(OVL_END)
    with db_conn() as c:
        ib = pd.read_sql(
            "SELECT * FROM ironbeam_es_flow_1s WHERE symbol = %s AND ts_utc >= %s AND ts_utc < %s",
            c, params=("XCME:ES.H26", s0.to_pydatetime(), e1.to_pydatetime()))
    ib["ts_utc"] = pd.to_datetime(ib["ts_utc"], utc=True)
    ib = ib[in_window(ib["ts_utc"])]
    print(f"Databento 1 s rows {len(dbf):,} (volume {dbf.volume.sum():,}); Ironbeam 1 s rows {len(ib):,} "
          f"(volume {ib.volume.sum():,}); Ironbeam symbols {ib.symbol.unique().tolist()}")
    print(f"Ironbeam trade_date == UTC date of ts_utc: "
          f"{(pd.to_datetime(ib.trade_date).dt.date == ib.ts_utc.dt.date).mean():.4f}")
    print(f"Databento side shares (volume): B {dbf.buy_vol.sum() / dbf.volume.sum():.4f}  "
          f"A {dbf.sell_vol.sum() / dbf.volume.sum():.4f}  N {dbf.unknown_vol.sum() / dbf.volume.sum():.4f}; "
          f"Ironbeam unknown share {ib.unknown_vol.sum() / max(ib.volume.sum(), 1):.4f}")

    md, mi = to_minutes(dbf), to_minutes(ib)
    j = md.join(mi, how="inner", lsuffix="_db", rsuffix="_ib")
    print(f"minutes: Databento {len(md):,}, Ironbeam {len(mi):,}, both {len(j):,}; "
          f"Databento-only {len(md.index.difference(mi.index))}, Ironbeam-only {len(mi.index.difference(md.index))}")
    ratio = j.volume_ib / j.volume_db
    q = ratio.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
    print(f"\nvolume ratio Ironbeam / Databento per minute: median {q[0.5]:.4f}  p5 {q[0.05]:.4f}  "
          f"p25 {q[0.25]:.4f}  p75 {q[0.75]:.4f}  p95 {q[0.95]:.4f}; pooled {j.volume_ib.sum() / j.volume_db.sum():.4f}")
    per_day = j.groupby(j.index.tz_convert(PT).date).apply(lambda x: x.volume_ib.sum() / x.volume_db.sum())
    print("pooled ratio per session: " + ", ".join(f"{d:%m-%d} {v:.3f}" for d, v in per_day.items()))
    cd = j.delta_db.corr(j.delta_ib)
    print(f"\nper-minute delta (buy − sell) correlation Databento vs Ironbeam: {cd:.4f}")
    print(f"  buy corr {j.buy_db.corr(j.buy_ib):.4f}; sell corr {j.sell_db.corr(j.sell_ib):.4f}; "
          f"delta sign agreement {(np.sign(j.delta_db) == np.sign(j.delta_ib)).mean():.4f}")
    print(f"  delta slope (Ironbeam on Databento) {np.polyfit(j.delta_db, j.delta_ib, 1)[0]:.4f}")
    print("\nper-minute OHLC differences (Databento − Ironbeam, pts):")
    for k in ("open", "high", "low", "close"):
        d = (j[f"{k}_db"] - j[f"{k}_ib"])
        a = d.abs()
        print(f"  {k:5s}: median {d.median():+.3f}  p99 |d| {a.quantile(0.99):.2f}  max |d| {a.max():.2f}  "
              f"equal {(a < 1e-9).mean():.4f}  > 1 pt {(a > 1).sum()}")

    for name, f, mins in (("Databento", dbf, md), ("Ironbeam", ib, mi)):
        dp = mins["close"] - mins["open"]
        print(f"sign test {name}: corr(minute delta, minute close − open) = {mins['delta'].corr(dp):+.4f}")
    dbm = md.copy()
    dbm["dp"] = dbm["close"] - dbm["open"]
    sign = dbm["delta"].corr(dbm["dp"])
    stop = []
    if not 0.95 <= q[0.5] <= 1.05:
        stop.append("volume ratio median outside 0.95–1.05")
    if cd < 0.9:
        stop.append("delta corr < 0.9")
    if not sign > 0:
        stop.append("sign test failed")
    print(f"\nQ2 gate: {'PASS' if not stop else 'STOP — ' + '; '.join(stop)}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    j.to_parquet(LOG_DIR / "q2_minutes.parquet")
    dbf.to_parquet(LOG_DIR / "q2_databento_flow_1s.parquet")


def phase_size():
    hdr("Q1c/d — projected table size and DB free space")
    _, dbf = overlap_flow()
    dbf = dbf[in_window(dbf["ts_utc"])]
    n_sess = len(sessions(OVL_START, OVL_END))
    rows_per_sess = len(dbf) / n_sess
    with db_conn() as c, c.cursor() as cur:
        cur.execute("SELECT pg_database_size(current_database())")
        db_bytes = cur.fetchone()[0]
        cur.execute("""SELECT pg_total_relation_size('ironbeam_es_flow_1s'), pg_relation_size('ironbeam_es_flow_1s'),
                              (SELECT reltuples FROM pg_class WHERE relname = 'ironbeam_es_flow_1s')""")
        ib_total, ib_heap, ib_rows = cur.fetchone()
        cur.execute("SELECT count(*) FROM ironbeam_es_flow_1s")
        ib_exact = cur.fetchone()[0]
    bpr_total = ib_total / ib_exact
    n_r = sum(len(contracts_for(d)) for d in sessions(R_START, R_END))
    n_r_front = len(sessions(R_START, R_END))
    # dying-contract rows in roll weeks are sparser than front rows; measured below from cost bytes ratio
    cost = pd.read_csv(LOG_DIR / "cost_trades.csv")
    cost = cost[cost.part == "R"]
    cost["session"] = pd.to_datetime(cost["session"]).dt.date
    fr = {d: raw_symbol(*front_for(d)) for d in sessions(R_START, R_END)}
    cost["is_front"] = [fr[d] == s for d, s in zip(cost.session, cost.raw_symbol)]
    q2 = pd.read_csv(LOG_DIR / "cost_trades.csv")
    q2 = q2[q2.part == "Q2"]
    q2_bytes_per_sess = q2.bytes.sum() / len(q2)
    print(f"Q2 window measured: {len(dbf):,} 1 s rows over {n_sess} sessions = {rows_per_sess:,.0f} rows/session "
          f"(max possible {6.75 * 3600:,.0f})")
    print(f"ironbeam_es_flow_1s: {ib_exact:,} rows, {ib_total / 1e9:.3f} GB total ({ib_heap / 1e9:.3f} GB heap) "
          f"= {bpr_total:.1f} bytes/row incl. indexes (PK + 3 secondary indexes)")
    # rows ~ scale with seconds that have >= 1 trade; activity proxy = billable trade bytes per session
    front = cost[cost.is_front]
    dying = cost[~cost.is_front]
    est_front = (front.bytes / q2_bytes_per_sess).clip(upper=1.0 / (rows_per_sess / (6.75 * 3600)))
    rows_front = (rows_per_sess * est_front).sum()
    rows_front_flat = rows_per_sess * len(front)
    rows_dying_upper = rows_per_sess * len(dying)
    # the new table has two extra text columns (raw_symbol, source) ~ +14 bytes/row
    bpr_new = bpr_total + 14
    lo = rows_front * bpr_new / 1e9
    hi = (rows_front_flat + rows_dying_upper) * bpr_new / 1e9
    print(f"R: {n_r_front} sessions, {n_r} session-contracts ({len(dying)} dying-contract roll-week rows)")
    print(f"projected rows: front scaled by trade bytes {rows_front:,.0f}; flat upper bound "
          f"(every session-contract at the Q2 rate) {rows_front_flat + rows_dying_upper:,.0f}")
    print(f"projected size at {bpr_new:.0f} bytes/row: {lo:.2f} GB (scaled, front only) .. {hi:.2f} GB (flat upper bound)")
    print(f"current DB size {db_bytes / 1e9:.2f} GB on a 30 GB disk (spec; autoscaling off) -> "
          f"free ≈ {30 - db_bytes / 1e9:.2f} GB; after load {db_bytes / 1e9 + lo:.2f} .. {db_bytes / 1e9 + hi:.2f} GB")
    print(f"Q1 gate (projected <= 27 GB): {'PASS' if db_bytes / 1e9 + hi <= 27 else 'STOP'}")


def phase_roll():
    hdr("Q3 — roll calendar (L3): front until the Monday of the week before expiry week; roll weeks store both")
    ss = sessions(R_START, R_END)
    rows = []
    for d in ss:
        cs = contracts_for(d)
        rows.append(dict(session=d, contracts=",".join(raw_symbol(*c) for c in cs),
                         front=raw_symbol(*front_for(d)), n=len(cs)))
    df = pd.DataFrame(rows)
    print(f"{'contract':9s} {'ironbeam':12s} {'front first':12s} {'front last':12s} {'front n':>7s}  "
          f"{'stored first':12s} {'stored last':12s} {'stored n':>8s}")
    for y, m in quarterlies(R_START, R_END):
        sym = raw_symbol(y, m)
        f = df[df.front == sym]
        st = df[df.contracts.str.contains(sym)]
        if st.empty:
            continue
        print(f"{sym:9s} {ironbeam_symbol(y, m):12s} {str(f.session.min()):12s} {str(f.session.max()):12s} "
              f"{len(f):7d}  {str(st.session.min()):12s} {str(st.session.max()):12s} {len(st):8d}")
    two = df[df.n == 2]
    print(f"\nsessions with two contracts stored: {len(two)} of {len(df)}")
    for (y, m) in quarterlies(R_START, R_END):
        w = two[[c.split(",")[0] == raw_symbol(y, m) for c in two.contracts]]
        if not w.empty:
            print(f"  roll {raw_symbol(y, m)} -> next: {w.session.min()} .. {w.session.max()} "
                  f"({len(w)} sessions; expiry {third_friday(y, m)})")
    print(f"sessions with zero contracts: {(df.n == 0).sum()}")
    df.to_csv(LOG_DIR / "q3_roll_calendar.csv", index=False)


def q4_sample() -> list[date]:
    import random
    ss = sessions(date(2025, 1, 1), date(2025, 12, 31))
    return sorted(random.Random(Q4_SEED).sample(ss, 10))


def phase_bars(allow_pull: bool):
    hdr(f"Q4 — Databento-built 1-minute OHLC (front) vs ironbeam_es_1m_bars, 10 random 2025 sessions (seed {Q4_SEED})")
    samp = q4_sample()
    print("sample: " + ", ".join(f"{d}{' (roll wk)' if len(contracts_for(d)) == 2 else ''}" for d in samp))
    if not allow_pull:
        cl = client()
        usd = 0.0
        for d in samp:
            s, e = window_utc(d)
            usd += float(cl.metadata.get_cost(dataset=DATASET, schema="trades", symbols=[raw_symbol(*front_for(d))],
                                              stype_in="raw_symbol", start=s.isoformat(), end=e.isoformat()))
        print(f"Q4 pull not run (needs --allow-q4-pull); cost for the 10 sessions ${usd:.2f}")
        return
    rows = []
    with db_conn() as c:
        for d in samp:
            y, m = front_for(d)
            tr = fetch_trades(raw_symbol(y, m), [d], "Q4")
            f = aggregate_1s(tr, y, m)
            f = f[in_window(f["ts_utc"])]
            md = to_minutes(f)
            s, e = window_utc(d)
            b = pd.read_sql("SELECT datetime, open, high, low, close, volume, source FROM ironbeam_es_1m_bars "
                            "WHERE datetime >= %s AND datetime < %s", c,
                            params=(s.to_pydatetime(), e.to_pydatetime()))
            b["datetime"] = pd.to_datetime(b["datetime"], utc=True)
            b = b.set_index("datetime")
            j = md.join(b, how="inner", rsuffix="_bar")
            diffs = {k: (j[k] - j[f"{k}_bar"]) for k in ("open", "high", "low", "close")}
            dc = diffs["close"]
            rows.append(dict(session=d, front=raw_symbol(y, m), roll_week=len(contracts_for(d)) == 2,
                             bar_sources=",".join(sorted(b.source.dropna().unique())), minutes=len(j),
                             db_minutes=len(md), bar_minutes=len(b),
                             close_median=dc.median(), close_abs_max=dc.abs().max(),
                             any_gt1=int(sum((diffs[k].abs() > 1).sum() for k in diffs)),
                             close_equal=(dc.abs() < 1e-9).mean(),
                             vol_ratio_bar_over_db=b.volume.sum() / md.volume.sum()))
    out = pd.DataFrame(rows)
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(out.round(3).to_string(index=False))
    out.to_csv(LOG_DIR / "q4_bars.csv", index=False)


DDL = """CREATE TABLE es_flow_1s_databento (
  ts_utc         timestamptz NOT NULL,
  symbol         text        NOT NULL,        -- Ironbeam format, e.g. XCME:ES.H26 (A1.1)
  raw_symbol     text        NOT NULL,        -- CME/Databento, e.g. ESH6 (A1.1)
  trade_date     date        NULL,            -- UTC date of ts_utc (A1.2)
  volume         integer     NOT NULL DEFAULT 0,
  notional       double precision NOT NULL DEFAULT 0,
  trade_count    integer     NOT NULL DEFAULT 0,
  buy_vol        integer     NOT NULL DEFAULT 0,
  sell_vol       integer     NOT NULL DEFAULT 0,
  unknown_vol    integer     NOT NULL DEFAULT 0,
  buy_count      integer     NOT NULL DEFAULT 0,
  sell_count     integer     NOT NULL DEFAULT 0,
  unknown_count  integer     NOT NULL DEFAULT 0,
  open           double precision NULL,
  high           double precision NULL,
  low            double precision NULL,
  close          double precision NULL,
  first_trade_ms bigint NULL,
  last_trade_ms  bigint NULL,
  source         text        NOT NULL DEFAULT 'databento',
  PRIMARY KEY (ts_utc, symbol)
);
CREATE INDEX idx_es_flow_1s_databento_trade_date ON es_flow_1s_databento (trade_date);
GRANT SELECT, INSERT ON es_flow_1s_databento TO dash_backfill_writer;"""


def phase_ddl():
    hdr("Q5 — DDL and backfill-role privileges (read-only checks)")
    print(DDL)
    import psycopg
    with psycopg.connect(os.environ["BACKFILL_DATABASE_URL"].replace("+psycopg", "")) as c, c.cursor() as cur:
        cur.execute("SELECT current_user, has_schema_privilege(current_user, 'public', 'CREATE'), "
                    "has_database_privilege(current_user, current_database(), 'CREATE'), "
                    "(SELECT rolcreatedb FROM pg_roles WHERE rolname = current_user), "
                    "to_regclass('public.es_flow_1s_databento')")
        u, sch, dbc, cdb, exists = cur.fetchone()
    print(f"\nbackfill role: {u}; CREATE on schema public: {sch}; CREATE on database: {dbc}; "
          f"rolcreatedb: {cdb}; table exists already: {exists is not None}")
    with db_conn() as c, c.cursor() as cur:
        cur.execute("SELECT column_name, data_type FROM information_schema.columns "
                    "WHERE table_name = 'ironbeam_es_flow_1s' ORDER BY ordinal_position")
        print("ironbeam_es_flow_1s columns: " + ", ".join(f"{a} {b}" for a, b in cur.fetchall()))
        cur.execute("SELECT indexdef FROM pg_indexes WHERE tablename = 'ironbeam_es_flow_1s'")
        for (i,) in cur.fetchall():
            print("  " + i)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["cost", "pull", "overlap", "size", "roll", "bars", "ddl"])
    ap.add_argument("--allow-q4-pull", action="store_true")
    a = ap.parse_args()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    {"cost": phase_cost, "pull": phase_pull, "overlap": phase_overlap, "size": phase_size, "roll": phase_roll,
     "bars": lambda: phase_bars(a.allow_q4_pull), "ddl": phase_ddl}[a.phase]()
