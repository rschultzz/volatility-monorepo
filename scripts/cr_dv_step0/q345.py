"""Q3 cells from trades vs es_flow_price_1m; Q4 tail query cost; Q5 RTH trade-gap distribution. Read-only."""
import time, datetime as dt
import numpy as np
from db import conn
c = conn(); U = dt.timezone.utc; SYM = "XCME:ES.Z26"
GA, GB = dt.datetime(2026,10,29,22,0,tzinfo=U), dt.datetime(2026,10,30,21,0,tzinfo=U)   # Globex day 10-30
print("== Q3 Globex day 2026-10-30 [10-29 22:00, 10-30 21:00) UTC, symbol", SYM)
print("off-grid trade prices:", c.execute("select count(*) from ironbeam_es_trades where symbol=%s and ts_utc>=%s and ts_utc<%s and price*4 <> floor(price*4)", (SYM,GA,GB)).fetchone()[0])
print("sides:", c.execute("select aggressor_side, count(*), sum(size) from ironbeam_es_trades where symbol=%s and ts_utc>=%s and ts_utc<%s group by 1", (SYM,GA,GB)).fetchall())
CMP = """
 with t as (select date_trunc('minute', ts_utc) m, floor(price*4)/4 p,
                   coalesce(sum(size) filter (where aggressor_side='BUY'),0) b,
                   coalesce(sum(size) filter (where aggressor_side='SELL'),0) s, count(*) n
            from ironbeam_es_trades where symbol=%(s)s and ts_utc>=%(a)s and ts_utc<%(b)s {extra} group by 1,2),
      f as (select minute_utc m, price p, buy_vol b, sell_vol s, trade_count n from es_flow_price_1m
            where source='ironbeam' and symbol=%(s)s and minute_utc>=%(a)s and minute_utc<%(b)s)
 select count(*) filter (where t.m is not null and f.m is not null) both_,
        count(*) filter (where f.m is null) only_trades, count(*) filter (where t.m is null) only_stored,
        count(*) filter (where t.m is not null and f.m is not null and (t.b<>f.b or t.s<>f.s or t.n<>f.n)) diff,
        count(distinct coalesce(t.m,f.m)) filter (where f.m is null or t.m is null or t.b<>f.b or t.s<>f.s or t.n<>f.n) minutes_off,
        count(distinct coalesce(t.m,f.m)) minutes
 from t full join f on t.m=f.m and t.p=f.p"""
for label, extra in (("all trades now", ""), ("trades received by minute end + 15 s", "and received_at <= date_trunc('minute', ts_utc) + interval '75 seconds'"),
                     ("received by minute end + 5 s", "and received_at <= date_trunc('minute', ts_utc) + interval '65 seconds'")):
    r = c.execute(CMP.format(extra=extra), dict(s=SYM, a=GA, b=GB)).fetchone()
    print(f"  {label}: cells both {r[0]}, only trades {r[1]}, only stored {r[2]}, differ {r[3]}; minutes off {r[4]}/{r[5]}")
RA, RB = dt.datetime(2026,10,30,13,30,tzinfo=U), dt.datetime(2026,10,30,20,0,tzinfo=U)
r = c.execute(CMP.format(extra=""), dict(s=SYM, a=RA, b=RB)).fetchone(); print(f"  RTH only, all trades: both {r[0]} only_t {r[1]} only_s {r[2]} differ {r[3]} minutes {r[5]}")
r = c.execute(CMP.format(extra="and received_at <= date_trunc('minute', ts_utc) + interval '75 seconds'"), dict(s=SYM, a=RA, b=RB)).fetchone()
print(f"  RTH only, received by end+15s: both {r[0]} only_t {r[1]} only_s {r[2]} differ {r[3]} minutes off {r[4]}/{r[5]}")
late = c.execute("""select count(*), coalesce(sum(size),0) from ironbeam_es_trades where symbol=%s and ts_utc>=%s and ts_utc<%s
                    and received_at > date_trunc('minute', ts_utc) + interval '75 seconds'""", (SYM,RA,RB)).fetchone()
print("  RTH prints received > 15 s after their minute closed:", late)
lat = c.execute("""select percentile_cont(array[0.5,0.99,0.999]) within group (order by extract(epoch from received_at-ts_utc)), max(extract(epoch from received_at-ts_utc))
                   from ironbeam_es_trades where symbol=%s and ts_utc>=%s and ts_utc<%s""", (SYM,RA,RB)).fetchone()
print("  RTH lateness s p50/p99/p99.9, max:", lat)

print("== Q4 tail query cost (fetch_trade_cells SQL, index (symbol, ts_utc DESC))")
Q = """select (floor(extract(epoch from ts_utc)/60)*60)::bigint t, price, aggressor_side, sum(size)::float8
       from ironbeam_es_trades where symbol=%s and ts_utc>=%s and ts_utc<%s group by 1,2,3"""
for label, a, b in (("8-min tail 10:00-10:08 PT", dt.datetime(2026,10,30,17,0,tzinfo=U), dt.datetime(2026,10,30,17,8,tzinfo=U)),
                    ("8-min tail at open 06:30-06:38 PT", RA, RA+dt.timedelta(minutes=8)),
                    ("15-min tail 06:30-06:45", RA, RA+dt.timedelta(minutes=15)),
                    ("full RTH (first live poll today)", RA, RB)):
    ts = []
    for _ in range(5):
        t0 = time.perf_counter(); rows = c.execute(Q, (SYM, a, b)).fetchall(); ts.append(time.perf_counter()-t0)
    print(f"  {label}: {len(rows)} rows, wall ms median {np.median(ts)*1000:.0f} max {max(ts)*1000:.0f}")
plan = c.execute("explain (analyze, buffers) " + Q, (SYM, dt.datetime(2026,10,30,17,0,tzinfo=U), dt.datetime(2026,10,30,17,8,tzinfo=U))).fetchall()
print("  plan (8-min tail):"); [print("    ", p[0]) for p in plan]

print("== Q5 gaps between trades during RTH (ironbeam_es_flow_1s, seconds with volume), symbol = day's max volume")
G = """with d as (select trade_date, symbol, sum(volume) v from ironbeam_es_flow_1s
                 where ts_utc >= now() - interval '120 days' group by 1,2),
            f as (select distinct on (trade_date) trade_date, symbol from d order by trade_date, v desc),
            s as (select x.ts_utc, x.symbol from ironbeam_es_flow_1s x join f using (trade_date, symbol)
                  where x.volume > 0 and (x.ts_utc at time zone 'America/Los_Angeles')::time >= '06:30'
                    and (x.ts_utc at time zone 'America/Los_Angeles')::time < '13:00'),
            g as (select ts_utc, (ts_utc at time zone 'America/Los_Angeles')::date d,
                         extract(epoch from ts_utc - lag(ts_utc) over (partition by symbol, (ts_utc at time zone 'America/Los_Angeles')::date order by ts_utc)) gap
                  from s)
       select d, ts_utc, gap from g where gap is not null"""
rows = c.execute(G).fetchall()
gaps = np.array([float(r[2]) for r in rows]); days = sorted({r[0] for r in rows})
print(f"  sessions {len(days)} ({days[0]} .. {days[-1]}), gaps {len(gaps)}")
print("  gap s percentiles 50/99/99.9/99.99/99.999:", np.percentile(gaps, [50, 99, 99.9, 99.99, 99.999]))
for thr in (5, 10, 15, 20, 30, 45, 60, 120, 300):
    print(f"    gaps > {thr:>3}s: {int((gaps > thr).sum())}")
big = sorted([r for r in rows if float(r[2]) > 20], key=lambda r: -float(r[2]))[:25]
for d, ts, g in big:
    print(f"    {d} gap {float(g):7.0f}s ending {ts.astimezone(dt.timezone(dt.timedelta(hours=-7))):%H:%M:%S} PDT-ish")
import collections
mx = collections.defaultdict(float)
for d, ts, g in rows: mx[d] = max(mx[d], float(g))
print("  per-session max gap s (sorted):", sorted(round(v) for v in mx.values()))
missing = [d for d in days]
