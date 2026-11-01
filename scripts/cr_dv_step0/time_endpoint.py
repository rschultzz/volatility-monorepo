"""CR-DV pre-merge — /api/ironbeam/flow/cells response time on 2026-10-30 after 12:00 PT, real DB reads (read-only).

The real flow_v3.build_cells_payload(now_utc=T) + json.dumps, every 15 s from 19:00:15 to 20:00:00 UTC, with every
read's range end cut at T (what the live DB held then); the in-process caches persist across polls as on the server.
One cold poll at 19:00:00 (empty caches) is reported separately. Timed from the laptop.
Usage: time_endpoint.py [--root TREE] [--lag] [--label L]
  --root  the checkout whose code is timed (default: this one), e.g. a detached origin/main worktree
  --lag   also cut es_flow_price_1m to the minutes readable by then (CR-DD-live run completion, q2_lags.json), as live
"""
import argparse, json, os, sys, time, datetime as dt
from pathlib import Path
HERE = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--root", default=str(HERE)); ap.add_argument("--lag", action="store_true"); ap.add_argument("--label", default="branch")
ARGS = ap.parse_args()
ROOT = Path(ARGS.root)
sys.path[:0] = [str(ROOT), str(ROOT / "apps/web")]
from dotenv import load_dotenv
load_dotenv(HERE / ".env")
import numpy as np
from sqlalchemy import create_engine
from modules.Ironbeam import flow_v2, flow_v3
from packages.shared import flow_heat as fh

U = dt.timezone.utc
engine = create_engine(os.environ["DATABASE_URL"], pool_size=3, max_overflow=2,
                       connect_args={"options": "-c default_transaction_read_only=on"})
NOW = {"T": None}
o_tc, o_p1, o_fl, o_bc = flow_v3.fetch_trade_cells, flow_v3.fetch_price_1m, flow_v2.fetch_flow_minutes, flow_v2.fetch_bar_closes
flow_v3.fetch_trade_cells = lambda e, s, a, b: o_tc(e, s, a, min(b, NOW["T"]))
LAGS = json.load(open(HERE / "scripts/logs/cr-dv/q2_lags.json"))["lags"]
RTH_A = dt.datetime(2026, 10, 30, 13, 30, tzinfo=U)
READABLE = [RTH_A + dt.timedelta(minutes=i + 1, seconds=l) for i, l in enumerate(LAGS)]   # minute i's readable time


def p1m_end(T):
    """Exclusive end of the stored minutes readable at T (the cron writes in order; RTH minute i = 13:30 + i)."""
    if not ARGS.lag:
        return T
    n = sum(1 for r in READABLE if r <= T)
    return RTH_A + dt.timedelta(minutes=n)


flow_v3.fetch_price_1m = lambda e, s, src, a, b: o_p1(e, s, src, a, max(a, min(b, p1m_end(NOW["T"]))))
flow_v2.fetch_flow_minutes = lambda e, t, r, now_utc=None: o_fl(e, t, [(a, min(b, NOW["T"] - dt.timedelta(seconds=2))) for a, b in r], now_utc=NOW["T"])
flow_v2.fetch_bar_closes = lambda e, a, b: o_bc(e, a, min(b, flow_v2._floor_min(NOW["T"])))
o_last = getattr(flow_v3, "_trades_last", None)
if o_last is not None:
    flow_v3._trades_last = lambda e, s: min(o_last(e, s), NOW["T"])   # same ordered read; value capped at T


def poll(T):
    NOW["T"] = T
    t0 = time.perf_counter()
    p, st = flow_v3.build_cells_payload(engine, "2026-10-30", session="FULL", until_pt=None, params=fh.AbsorbParams(),
                                        session_date=dt.date(2026, 10, 30), session_note=None, now_utc=T)
    body = json.dumps(p)
    ms = (time.perf_counter() - t0) * 1000
    rth = next(s for s in p["segments"] if s["name"] == "rth")
    assert ROOT.as_posix() in flow_v3.__file__, flow_v3.__file__
    return ms, st, rth["tier"], len(body), len(rth["cells"]["t"])


for c in (flow_v3._CACHE, flow_v3._STATE, flow_v2._CACHE, flow_v2._WATERMARK, flow_v2._TABLE_STATE):
    c.clear()
cold = poll(dt.datetime(2026, 10, 30, 19, 0, 0, tzinfo=U))
print(f"cold poll 12:00:00 PT (empty caches): {cold[0]:.0f} ms, tier {cold[2]}, body {cold[3]/1e6:.2f} MB, rth cells {cold[4]}")
rows = []
T = dt.datetime(2026, 10, 30, 19, 0, 15, tzinfo=U)
while T <= dt.datetime(2026, 10, 30, 20, 0, 0, tzinfo=U):
    rows.append((T.strftime("%H:%M:%S"),) + poll(T))
    T += dt.timedelta(seconds=15)
ms = np.array([r[1] for r in rows])
print(f"warm polls 12:00:15 -> 13:00:00 PT: n {len(ms)}, p50 {np.percentile(ms,50):.0f} ms, p95 {np.percentile(ms,95):.0f} ms, "
      f"max {ms.max():.0f} ms (at {rows[int(ms.argmax())][0]} UTC); tiers {sorted({r[3] for r in rows})}; "
      f"status {sorted({r[2] for r in rows})}; body MB p50 {np.median([r[4] for r in rows])/1e6:.2f}")
slow = sorted(rows, key=lambda r: -r[1])[:5]
print("slowest:", [(r[0], round(r[1])) for r in slow])
print(f"code: {flow_v3.__file__}  lag: {ARGS.lag}")
json.dump({"cold_ms": cold[0], "warm": [(r[0], r[1]) for r in rows]},
          open(HERE / f"scripts/logs/cr-dv/time_endpoint_{ARGS.label}{'_lag' if ARGS.lag else ''}.json", "w"))
