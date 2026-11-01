"""Q5b — RTH gaps > 20 s: were they feed stalls (bars show volume the flow lacks) or quiet tape? Read-only."""
import datetime as dt
from db import conn
c = conn()
rows = c.execute("""
with d as (select trade_date, symbol, sum(volume) v from ironbeam_es_flow_1s where ts_utc >= now() - interval '120 days' group by 1,2),
f as (select distinct on (trade_date) trade_date, symbol from d order by trade_date, v desc),
s as (select x.ts_utc, x.symbol from ironbeam_es_flow_1s x join f using (trade_date, symbol)
      where x.volume > 0 and (x.ts_utc at time zone 'America/Los_Angeles')::time between '06:30' and '13:00'),
g as (select symbol, lag(ts_utc) over w a, ts_utc b from s
      window w as (partition by symbol, (ts_utc at time zone 'America/Los_Angeles')::date order by ts_utc))
select g.a, g.b, extract(epoch from g.b-g.a) gap,
  (select coalesce(sum(volume),0) from ironbeam_es_1m_bars m where m.datetime >= (date_trunc('minute', g.a) + interval '1 minute') at time zone 'UTC'
       and m.datetime < date_trunc('minute', g.b) at time zone 'UTC') bar_vol_inside,
  (select count(*) from ironbeam_es_1m_bars m where m.datetime >= (date_trunc('minute', g.a) + interval '1 minute') at time zone 'UTC'
       and m.datetime < date_trunc('minute', g.b) at time zone 'UTC') whole_minutes_inside
from g where extract(epoch from g.b-g.a) > 20 order by gap desc""").fetchall()
stall = quiet = 0
for a, b, gap, bv, nm in rows:
    kind = "n/a (no whole minute)" if nm == 0 else ("STALL (bars traded)" if bv > 0 else "quiet (bars 0 / missing)")
    print(f"{a.astimezone(dt.timezone(dt.timedelta(hours=-7))):%Y-%m-%d %H:%M:%S} gap {float(gap):6.0f}s  whole min {nm}  bar vol {bv}  {kind}")
