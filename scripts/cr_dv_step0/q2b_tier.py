"""Q2b — which tier flow_v3.choose_tier would pick at each poll on 10-30 RTH (stored-by-T vs minutes with flow by T)."""
import json, datetime as dt
import numpy as np
from db import conn
c = conn(); U = dt.timezone.utc
A, B = dt.datetime(2026,10,30,13,30,tzinfo=U), dt.datetime(2026,10,30,20,0,tzinfo=U)
lags = json.load(open("../logs/cr-dv/q2_lags.json"))["lags"]
stored = [r[0] for r in c.execute("select distinct minute_utc from es_flow_price_1m where source='ironbeam' and symbol='XCME:ES.Z26' and minute_utc>=%s and minute_utc<%s order by 1",(A,B))]
readable = {m: m + dt.timedelta(minutes=1, seconds=l) for m, l in zip(stored, lags)}
flow_min = sorted(r[0] for r in c.execute("select distinct date_trunc('minute', ts_utc) from ironbeam_es_flow_1s where symbol='XCME:ES.Z26' and ts_utc>=%s and ts_utc<%s and volume>0",(A,B)))
print("flow_1s RTH minutes with volume:", len(flow_min))
res = []
for i in range(390):
    T = A + dt.timedelta(minutes=i+1, seconds=15)           # a poll 15 s after minute i closes
    have = sum(1 for m in flow_min if m < T)                  # incl. the forming minute (worker is ~live)
    st = sum(1 for m, r in readable.items() if r <= T)
    res.append((T, "price_1m" if st >= 0.99 * have else "trades", have, st))
first_p = next((r for r in res if r[1] == "price_1m"), None)
n_p = sum(r[1] == "price_1m" for r in res)
print(f"polls served from price_1m: {n_p}/390; first at {first_p[0]:%H:%M:%S} UTC" if first_p else "never price_1m")
flips = sum(1 for a, b in zip(res, res[1:]) if a[1] != b[1])
print("tier flips:", flips)
for r in res[::30]: print(f"  {r[0]:%H:%M:%S} {r[1]:8s} have {r[2]} stored {r[3]}")
