"""CR-BZ Q3/Q4: for 2026 minutes where bar is > 1 tick off the flow-built minute, classify the bar
as HEAD (matches flow seconds [0, s)), TAIL (matches flow seconds [s, 60)), FULL-other, or NONE.
Read-only. Excludes the CR-BF patch window."""
import pandas as pd
from db import q
SQL = """
WITH f AS (
  SELECT date_trunc('minute', ts_utc) AS m, count(DISTINCT symbol) nsym,
         (array_agg(open ORDER BY ts_utc))[1] o, max(high) h, min(low) l,
         (array_agg(close ORDER BY ts_utc DESC))[1] c, sum(volume) v, count(*) n
  FROM ironbeam_es_flow_1s WHERE ts_utc >= '2026-01-01' GROUP BY 1)
SELECT b.datetime, b.open bo, b.high bh, b.low bl, b.close bc, b.volume bv, f.o, f.h, f.l, f.c, f.v, f.n, f.nsym
FROM ironbeam_es_1m_bars b JOIN f ON f.m = b.datetime AT TIME ZONE 'UTC'
WHERE b.datetime >= '2026-01-01' AND NOT (b.datetime >= '2026-09-13 22:00' AND b.datetime < '2026-09-18 21:00')
  AND greatest(abs(b.high-f.h),abs(b.low-f.l),abs(b.close-f.c)) > 0.25
"""
cols, rows = q(SQL)
d = pd.DataFrame(rows, columns=cols)
d["datetime"] = pd.to_datetime(d["datetime"]).dt.tz_localize("UTC")
d["pt"] = d["datetime"].dt.tz_convert("America/Los_Angeles")
d["rth"] = (d.pt.dt.hour*60 + d.pt.dt.minute).between(390, 779)
print("defect minutes", len(d), "rth", int(d.rth.sum()))
d["bar_vol_lt"] = d.bv < d.v
print(d.groupby(["rth", "bar_vol_lt"]).size())
# per-second for bar_vol_lt minutes
sub = d[d.bar_vol_lt]
mins = [x.to_pydatetime() for x in sub.datetime]
cols, srows = q("""SELECT date_trunc('minute', ts_utc) m, extract(second from ts_utc)::int s, open, high, low, close, volume
                   FROM ironbeam_es_flow_1s WHERE date_trunc('minute', ts_utc) = ANY(%s) ORDER BY ts_utc""", (mins,))
sec = pd.DataFrame(srows, columns=cols)
sec["m"] = pd.to_datetime(sec["m"], utc=True)
out = []
for _, r in sub.iterrows():
    g = sec[sec.m == r.datetime]
    if g.empty: continue
    tag, cut = "NONE", None
    def ohlc(x): return (x.open.iloc[0], x.high.max(), x.low.min(), x.close.iloc[-1])
    tgt = (r.bo, r.bh, r.bl, r.bc)
    for s in sorted(g.s.unique()):
        t = g[g.s >= s]
        if all(abs(a-b) <= 0.25 for a, b in zip(ohlc(t), tgt)): tag, cut = "TAIL", s; break
    if tag == "NONE":
        for s in sorted(g.s.unique(), reverse=True):
            h = g[g.s <= s]
            if all(abs(a-b) <= 0.25 for a, b in zip(ohlc(h), tgt)): tag, cut = "HEAD", s; break
    out.append(dict(datetime=r.datetime, pt=r.pt, rth=r.rth, tag=tag, cut=cut, bv=r.bv, fv=r.v,
                    vfrac=r.bv / r.v if r.v else None, err=max(abs(r.bh-r.h), abs(r.bl-r.l), abs(r.bc-r.c))))
o = pd.DataFrame(out)
print(o.groupby(["rth", "tag"]).agg(n=("tag", "size"), med_cut=("cut", "median"), med_vfrac=("vfrac", "median"), med_err=("err", "median"), max_err=("err", "max")))
o["hhmm"] = o.pt.dt.strftime("%H:%M")
print("top PT minutes:\n", o.hhmm.value_counts().head(8))
print("by month x tag:\n", o.assign(mon=o.pt.dt.strftime("%Y-%m")).pivot_table(index="mon", columns="tag", values="err", aggfunc="size", fill_value=0))
o.to_csv("q3_shape.csv", index=False)
