"""CR-DV Step 0 — replay /api/ironbeam/flow/cells on 2026-10-30 RTH with "now" simulated (read-only, no server).

Each poll at T calls the real flow_v3._build(now_utc=T) with every DB read cut to what existed at T:
  trades     : ironbeam_es_trades with received_at <= T (in memory, same sums as fetch_trade_cells)
  price_1m   : es_flow_price_1m minutes readable by T (CR-DD-live run completion, q2_lag.py)
  flow 1-s   : ironbeam_es_flow_1s rows with ts_utc < T - 2 s (the worker's max lateness)
  bar closes : ironbeam_es_1m_bars minutes closed by T
Reports each final mark's first appearance vs its confirm time, and marks shown that are not final.
Smoke (after the CR-DV change): also minute close -> first provisional / first confirmed, per final mark, and the
polls that picked each tier. Usage: replay_live.py [step_s] [label]
"""
import json, sys, time, datetime as dt
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "apps/web"), str(Path(__file__).parent)]
import numpy as np, pandas as pd
from sqlalchemy import create_engine
import os
from db import conn
from modules.Ironbeam import flow_v2, flow_v3
from packages.shared import flow_heat as fh

STEP = int(sys.argv[1]) if len(sys.argv) > 1 else 15
LABEL = sys.argv[2] if len(sys.argv) > 2 else "before"
U = dt.timezone.utc; SYM = "XCME:ES.Z26"
A, B = dt.datetime(2026,10,30,13,30,tzinfo=U), dt.datetime(2026,10,30,20,0,tzinfo=U)
engine = create_engine(os.environ["DATABASE_URL"], pool_size=2, max_overflow=0,
                       connect_args={"options": "-c default_transaction_read_only=on"})
c = conn()
tr = pd.DataFrame(c.execute("select ts_utc, received_at, price, size, aggressor_side from ironbeam_es_trades "
                            "where symbol=%s and ts_utc>=%s and ts_utc<%s", (SYM, A, B)).fetchall(),
                  columns=["ts", "rcv", "price", "size", "side"])
tr["t"] = (tr["ts"].map(lambda x: x.timestamp()) // 60 * 60).astype(np.int64)
tr["rcv_e"] = tr["rcv"].map(lambda x: x.timestamp()); tr["ts_e"] = tr["ts"].map(lambda x: x.timestamp())
p1m = pd.DataFrame(c.execute("select extract(epoch from minute_utc)::bigint, price, buy_vol::float8, sell_vol::float8 "
                             "from es_flow_price_1m where symbol=%s and source='ironbeam' and minute_utc>=%s and minute_utc<%s",
                             (SYM, A, B)).fetchall(), columns=["t", "price", "buy", "sell"])
stored = sorted(p1m["t"].unique()); lags = json.load(open(ROOT / "scripts/logs/cr-dv/q2_lags.json"))["lags"]
readable = dict(zip(stored, [t + 60 + l for t, l in zip(stored, lags)]))
print(f"trades {len(tr)}, price_1m rows {len(p1m)}", flush=True)
NOW = {"T": None}
orig_flow, orig_bars = flow_v2.fetch_flow_minutes, flow_v2.fetch_bar_closes

def f_trade_cells(engine, symbol, a, b):
    T = NOW["T"].timestamp(); k = (tr["ts_e"] >= a.timestamp()) & (tr["ts_e"] < b.timestamp()) & (tr["rcv_e"] <= T)
    g = tr[k].groupby(["t", "price", "side"], as_index=False)["size"].sum().astype({"size": float})
    return g.rename(columns={"side": "side"})[["t", "price", "side", "size"]]
def f_price_1m(engine, symbol, source, a, b):
    T = NOW["T"].timestamp(); ok = p1m["t"].map(lambda t: readable[t] <= T)
    return p1m[ok & (p1m["t"] >= a.timestamp()) & (p1m["t"] < b.timestamp())].reset_index(drop=True)
def f_flow(engine, table, ranges, now_utc=None):
    cut = NOW["T"] - dt.timedelta(seconds=2)
    return orig_flow(engine, table, [(a, min(b, cut)) for a, b in ranges if a < cut], now_utc=NOW["T"])
def f_bars(engine, a, b):
    return orig_bars(engine, a, min(b, flow_v2._floor_min(NOW["T"])))
flow_v3.fetch_trade_cells, flow_v3.fetch_price_1m = f_trade_cells, f_price_1m
flow_v3._trades_oldest = lambda e, s: dt.datetime(2026,10,29,1,1,tzinfo=U)
flow_v3._exists_price_1m = lambda e: True
flow_v2.fetch_flow_minutes, flow_v2.fetch_bar_closes = f_flow, f_bars

def poll(T):
    NOW["T"] = T
    pl, _ = flow_v3._build(engine, "2026-10-30", session="RTH", until_pt=None, params=fh.AbsorbParams(),
                           session_date=dt.date(2026,10,30), session_note=None, now_utc=T)
    seg = next(s for s in pl["segments"] if s["name"] == "rth")
    return seg["tier"], {(m["kind"], m["time"], m["price"]): m for m in (seg["marks"] or [])}

def pct(x):
    return "n/a" if not x else (f"median {np.median(x):.0f}, p95 {np.percentile(x,95):.0f}, min {min(x):.0f}, max {max(x):.0f} (n {len(x)})")

T_final = dt.datetime(2026,10,31,0,0,tzinfo=U)
_, final = poll(T_final)
print("final marks:", len(final), flush=True)
first, shown_not_final, tiers, t0 = {}, {}, {}, time.time()
first_prov, first_conf, tier_polls = {}, {}, {}
T = A + dt.timedelta(seconds=STEP)
while T <= B + dt.timedelta(minutes=5):
    tier, mk = poll(T)
    tiers[tier] = tiers.get(tier, 0) + 1
    tier_polls.setdefault(tier, []).append(T.strftime("%H:%M:%S"))
    for k, m in mk.items():
        (first_prov if m.get("provisional") else first_conf).setdefault(k, T.timestamp())
    for k in mk:
        first.setdefault(k, T.timestamp())
        if k not in final: shown_not_final.setdefault(k, []).append(T.timestamp())
    T += dt.timedelta(seconds=STEP)
print(f"polls {sum(tiers.values())} tiers {tiers} in {time.time()-t0:.0f}s")
lag = [first[k] - final[k]["confirm_time"] for k in final if k in first]
print(f"final marks seen live {len(lag)}/{len(final)}; first-seen - confirm_time s: median {np.median(lag):.0f}, "
      f"p5 {np.percentile(lag,5):.0f}, p95 {np.percentile(lag,95):.0f}, min {min(lag):.0f}, max {max(lag):.0f}")
early = [k for k in final if k in first and first[k] < final[k]["confirm_time"]]
print(f"final marks first shown BEFORE their confirm time: {len(early)}")
print(f"marks shown live that are not in the final set: {len(shown_not_final)}")
for k, ts in list(shown_not_final.items())[:10]:
    print("   ", k[0], dt.datetime.fromtimestamp(k[1], U).strftime("%H:%M"), k[2], "shown", len(ts), "polls from",
          dt.datetime.fromtimestamp(ts[0], U).strftime("%H:%M:%S"))
for tier, ts in tier_polls.items():
    print(f"tier {tier}: {len(ts)} polls, first {ts[0]}, last {ts[-1]}")
conf_shown_not_final = [k for k in first_conf if k not in final]
print(f"confirmed marks shown that are not final marks: {len(conf_shown_not_final)}; provisional-only candidates "
      f"never confirmed: {len([k for k in first_prov if k not in final])}")
close_to_prov = [first_prov[k] - (final[k]["time"] + 60) for k in final if k in first_prov]
close_to_conf = [first_conf[k] - (final[k]["time"] + 60) for k in final if k in first_conf]
hold_to_conf = [first_conf[k] - final[k]["confirm_time"] for k in final if k in first_conf]
print("minute close -> first provisional s:", pct(close_to_prov))
print("minute close -> first confirmed s:  ", pct(close_to_conf))
print("hold end (confirm_time) -> first confirmed s:", pct(hold_to_conf))
early_conf = [k for k in final if k in first_conf and first_conf[k] < final[k]["confirm_time"]]
print(f"final marks confirmed BEFORE their hold ended: {len(early_conf)}")
json.dump({"close_to_prov": close_to_prov, "close_to_conf": close_to_conf, "hold_to_conf": hold_to_conf,
           "early_conf": len(early_conf), "conf_not_final": len(conf_shown_not_final), "tier_polls": tier_polls},
          open(ROOT / f"scripts/logs/cr-dv/replay_{LABEL}_detail.json", "w"))
json.dump({"lag": lag, "final": len(final), "early": len(early), "phantom": len(shown_not_final), "tiers": tiers},
          open(ROOT / f"scripts/logs/cr-dv/replay_{LABEL}.json", "w"))
