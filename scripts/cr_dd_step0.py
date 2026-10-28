#!/usr/bin/env python3
"""CR-DD Step 0 — es_flow_price_1m diagnosis (read-only; no DB writes, no Databento download).

  cost   metadata.get_cost per session-contract window (CR-CA contracts_for, [06:15, 13:00) PT) for
         the train sessions 2026-01-06 .. 2026-06-05; the $75 gate; which windows already have a CR-CA
         raw DBN file on disk (reused, no new spend).
  size   rows per session from one train DBN file aggregated to (minute, price) in memory; rows per day
         from ironbeam_es_trades (2026-10-27 Globex day); projected table size.
  live   ironbeam_es_trades: aggressor_side values, off-grid prices, ingest lateness (received_at vs the
         trade's minute), retention window vs the 01:00 UTC prune; owner / role privileges.

Run with apps/web/.venv/bin/python.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cr_ca_step0 import (  # noqa: E402  (also installs output redaction)
    DATASET, ROOT, RAW_DIR, client, contracts_for, db_conn, in_window, raw_symbol, sessions, window_utc,
)

START, END = date(2026, 1, 6), date(2026, 6, 5)
COST_CAP = 75.0
LOG_DIR = ROOT / "scripts" / "logs" / "cr_dd"


def raw_path(sym: str, d: date) -> Path:
    return RAW_DIR / f"{DATASET}_trades_{sym}_{d:%Y%m%d}_0615-1300PT.dbn.zst"


def jobs():
    return [(d, raw_symbol(*c)) for d in sessions(START, END) for c in contracts_for(d)]


def phase_cost():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    cl = client()

    def one(j):
        d, sym = j
        s, e = window_utc(d)
        kw = dict(dataset=DATASET, schema="trades", symbols=[sym], stype_in="raw_symbol",
                  start=s.isoformat(), end=e.isoformat())
        return dict(session=d, raw_symbol=sym, usd=float(cl.metadata.get_cost(**kw)),
                    bytes=int(cl.metadata.get_billable_size(**kw)), on_disk=raw_path(sym, d).exists())

    with ThreadPoolExecutor(8) as ex:
        df = pd.DataFrame(list(ex.map(one, jobs())))
    df.to_csv(LOG_DIR / "cost_trades.csv", index=False)
    tot = df.usd.sum()
    print(f"sessions {df.session.nunique()}, session-contracts {len(df)}, "
          f"two-contract sessions {int((df.groupby('session').size() == 2).sum())}")
    print(df.groupby("raw_symbol").agg(n=("usd", "size"), usd=("usd", "sum"), gb=("bytes", lambda b: b.sum() / 1e9))
          .round(3).to_string())
    print(f"priced total ${tot:,.2f} ({df.bytes.sum() / 1e9:.2f} GB billable); per session "
          f"max ${df.groupby('session').usd.sum().max():.3f}")
    print(f"raw DBN on disk (CR-CA L7): {int(df.on_disk.sum())} / {len(df)}; "
          f"to download: ${df.loc[~df.on_disk, 'usd'].sum():,.2f}")
    ok = tot <= COST_CAP
    print(f"gate: ${tot:,.2f} vs ${COST_CAP:.0f} -> {'PASS' if ok else 'STOP'}")
    (LOG_DIR / "cost_summary.json").write_text(json.dumps(
        dict(priced_usd=float(tot), cap=COST_CAP, on_disk=int(df.on_disk.sum()), n=len(df),
             to_download_usd=float(df.loc[~df.on_disk, "usd"].sum()), passed=bool(ok)), indent=2))


def phase_size():
    import databento as db
    d, sym = date(2026, 1, 6), "ESH6"
    tr = db.DBNStore.from_file(str(raw_path(sym, d))).to_df(pretty_ts=True, map_symbols=True).reset_index()
    tr["ts"] = pd.to_datetime(tr["ts_event"], utc=True)
    tr = tr[in_window(tr["ts"])]
    off = int(((tr["price"] * 4).round() != tr["price"] * 4).sum())
    cells = tr.groupby([tr["ts"].dt.floor("min"), (tr["price"] * 4 // 1) / 4]).size()
    print(f"Databento {sym} {d}: {len(tr):,} trades, side {tr['side'].value_counts().to_dict()}, off-grid {off}, "
          f"(minute, price) rows {len(cells):,} over {cells.index.get_level_values(0).nunique()} minutes")
    with db_conn() as c:
        r = c.execute("""SELECT count(*) FROM (SELECT DISTINCT date_trunc('minute', ts_utc), symbol, price
                         FROM ironbeam_es_trades WHERE ts_utc >= '2026-10-26 22:00Z' AND ts_utc < '2026-10-27 21:00Z') x""").fetchone()
        db_gb = c.execute("SELECT pg_database_size(current_database()) / 1e9").fetchone()[0]
    print(f"Ironbeam 2026-10-27 Globex day: (minute, symbol, price) rows {r[0]:,}")
    per_row = 100  # bytes incl. PK index, estimated for 7 narrow columns
    bf = len(cells) * len(jobs())
    print(f"projection: backfill ≈ {bf:,} rows (≈ {bf * per_row / 1e9:.2f} GB); live ≈ {r[0] * 260:,} rows/yr "
          f"(≈ {r[0] * 260 * per_row / 1e9:.2f} GB/yr); DB now {float(db_gb):.2f} GB")


def phase_live():
    qs = {
        "aggressor_side": "SELECT symbol, aggressor_side, count(*), sum(size) FROM ironbeam_es_trades GROUP BY 1, 2 ORDER BY 1, 2",
        "off_grid": "SELECT count(*) FILTER (WHERE price * 4 <> round(price * 4)), count(*) FROM ironbeam_es_trades",
        "span": "SELECT min(ts_utc), max(ts_utc) FROM ironbeam_es_trades",
        "lateness_s": """SELECT percentile_cont(ARRAY[0.5, 0.99, 0.999]) WITHIN GROUP
                           (ORDER BY extract(epoch FROM received_at - ts_utc)),
                         max(extract(epoch FROM received_at - ts_utc)),
                         count(*) FILTER (WHERE received_at >= date_trunc('minute', ts_utc) + interval '3 min'),
                         count(*) FILTER (WHERE received_at IS NULL), count(*)
                         FROM ironbeam_es_trades""",
        "late_rows": """SELECT ts_utc, received_at, size, aggressor_side FROM ironbeam_es_trades
                        WHERE received_at >= date_trunc('minute', ts_utc) + interval '3 min' ORDER BY ts_utc""",
        "trade_id_null": "SELECT count(*) FILTER (WHERE trade_id IS NULL), count(*) - count(DISTINCT trade_key) FROM ironbeam_es_trades",
        "backfill_create": "SELECT has_schema_privilege('dash_backfill_writer', 'public', 'CREATE')",
        "target_exists": "SELECT to_regclass('es_flow_price_1m')",
    }
    with db_conn() as c:
        for k, q in qs.items():
            print(f"{k}: {c.execute(q).fetchall()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["cost", "size", "live"])
    {"cost": phase_cost, "size": phase_size, "live": phase_live}[ap.parse_args().phase]()
