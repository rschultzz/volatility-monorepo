"""Q2 — when each 10-30 RTH minute became readable in es_flow_price_1m (no insert ts: use CR-DD-live runs)."""
import json, datetime as dt
import numpy as np
from db import conn
c = conn()
print("server tz:", c.execute("show timezone").fetchone()[0])
runs = c.execute("""select started_at, completed_at, status, rows_inserted, smoke_test_results->>'first_minute',
                    smoke_test_results->>'last_minute', smoke_test_results->>'minutes'
                    from bt_backfill_runs where cr_id='CR-DD-live' and started_at >= '2026-10-30 13:00' and started_at < '2026-10-30 21:00'
                    order by started_at""").fetchall()
print("runs on 10-30 13:00-21:00:", len(runs))
for r in runs[:4]: print("  ", r)
U = dt.timezone.utc
A, B = dt.datetime(2026,10,30,13,30,tzinfo=U), dt.datetime(2026,10,30,20,0,tzinfo=U)
stored = [r[0] for r in c.execute("select distinct minute_utc from es_flow_price_1m where source='ironbeam' and symbol='XCME:ES.Z26' and minute_utc>=%s and minute_utc<%s order by 1",(A,B))]
print("stored RTH minutes:", len(stored))
spans = [(dt.datetime.fromisoformat(f), dt.datetime.fromisoformat(l), comp.replace(tzinfo=U)) for _, comp, st, _, f, l, _ in runs if st == 'completed' and f]
lags, unassigned, gaps = [], 0, []
for m in stored:
    cover = [comp for f, l, comp in spans if f <= m <= l]
    if not cover: unassigned += 1; continue
    lags.append((min(cover) - (m + dt.timedelta(minutes=1))).total_seconds())
L = np.array(lags)
print(f"assigned {len(L)}, unassigned {unassigned}")
print(f"lag s (readable - minute end): median {np.median(L):.0f}, p95 {np.percentile(L,95):.0f}, max {L.max():.0f}, min {L.min():.0f}")
st = sorted(comp for *_, comp in spans)
gap = np.diff([s.timestamp() for s in st]) if len(st) > 1 else []
print("run completion spacing s: median", np.median(gap), "max", max(gap) if len(gap) else None)
print("run durations s:", np.percentile([(r[1]-r[0]).total_seconds() for r in runs if r[1]], [50,95,100]))
json.dump(dict(lags=lags), open("../logs/cr-dv/q2_lags.json","w"))
