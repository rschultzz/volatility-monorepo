"""CR-CD Step 0 Q4 — 2026-05-01 worked day (train). READ-ONLY. Prototype of the L4/L5 series and markers.

Prototype only (scratch): the operational readings of L4/L5 used here are listed in the spec's Step 0
findings and are proposals for flow_features.py, not frozen definitions. Every value at bar t uses
bars <= t only (bar t = the minute starting at t, evaluated at its close).
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, sys.argv[1])
from packages.shared.backfill_safety import get_backfill_db_conn, assert_role_or_die  # noqa: E402

DAY = "2026-05-01"
PTZ = "America/Los_Angeles"
conn = get_backfill_db_conn()
assert_role_or_die(conn)
conn.execute("SET default_transaction_read_only = on")
assert conn.execute("SHOW default_transaction_read_only").fetchone()[0] == "on"


def q(sql, params=None):
    cur = conn.execute(sql, params or ())
    return pd.DataFrame(cur.fetchall(), columns=[c.name for c in cur.description])


def sessions_before(table, n):
    s = q(f"""SELECT DISTINCT (ts_utc AT TIME ZONE '{PTZ}')::date AS d FROM {table}
              WHERE ts_utc >= '2026-03-15' AND ts_utc < '{DAY} 07:00+00'
                AND extract(isodow FROM (ts_utc AT TIME ZONE '{PTZ}')) < 6
                AND (ts_utc AT TIME ZONE '{PTZ}')::time BETWEEN '06:30' AND '12:59'
              ORDER BY 1""")
    return list(s["d"])[-n:]


def minutes(table, days, t0="06:30", t1="13:00"):
    """Flow-built minutes, dominant (RTH-volume) symbol per day."""
    return q(f"""
        WITH s AS (
          SELECT d, symbol FROM (
            SELECT (ts_utc AT TIME ZONE '{PTZ}')::date AS d, symbol, sum(volume) v,
                   row_number() OVER (PARTITION BY (ts_utc AT TIME ZONE '{PTZ}')::date ORDER BY sum(volume) DESC) r
            FROM {table} WHERE (ts_utc AT TIME ZONE '{PTZ}')::date = ANY(%s)
              AND (ts_utc AT TIME ZONE '{PTZ}')::time >= '06:30' AND (ts_utc AT TIME ZONE '{PTZ}')::time < '13:00'
            GROUP BY 1, 2) x WHERE r = 1)
        SELECT (f.ts_utc AT TIME ZONE '{PTZ}')::date AS d, s.symbol,
               date_trunc('minute', f.ts_utc AT TIME ZONE '{PTZ}') AS m,
               (array_agg(open ORDER BY ts_utc))[1] AS o, max(high) AS h, min(low) AS l,
               (array_agg(close ORDER BY ts_utc DESC))[1] AS c,
               sum(buy_vol)::float AS buy, sum(sell_vol)::float AS sell, sum(volume)::float AS vol
        FROM {table} f JOIN s ON s.symbol = f.symbol AND s.d = (f.ts_utc AT TIME ZONE '{PTZ}')::date
        WHERE (f.ts_utc AT TIME ZONE '{PTZ}')::time >= %s AND (f.ts_utc AT TIME ZONE '{PTZ}')::time < %s
        GROUP BY 1, 2, 3 ORDER BY 1, 3""", (days, t0, t1))


def full_index(g, d):
    idx = pd.date_range(f"{d} 06:30", f"{d} 12:59", freq="1min")
    g = g.set_index("m").reindex(idx)
    g[["buy", "sell", "vol"]] = g[["buy", "sell", "vol"]].fillna(0.0)
    g["c"] = g["c"].ffill()
    for k in ("o", "h", "l"):
        g[k] = g[k].fillna(g["c"])
    return g


def abs_impact(g):
    """Rolling 5-min |price move| / (|net| / 1000); NaN where net == 0 or not 5 bars yet."""
    net5 = (g["buy"] - g["sell"]).rolling(5).sum()
    mv5 = (g["c"] - g["c"].shift(5)).abs()
    return (mv5 / (net5.abs() / 1000.0)).where(net5 != 0)


def features(g, base_med, base_p20):
    g = g.copy()
    g["delta"] = g["buy"] - g["sell"]
    g["cvd"] = g["delta"].cumsum()
    g["share_raw"] = (g["delta"] / g["vol"]).where(g["vol"] > 0)
    g["share"] = g["share_raw"].ewm(span=3, adjust=False, ignore_na=True).mean()
    g["imp"] = abs_impact(g)
    g["imp_rel"] = g["imp"] / base_med
    g["absorb"] = (g["imp_rel"] <= base_p20 / base_med) & (g["share"].abs() >= 0.15)
    n = len(g)
    t = g.index
    L, H, C, cvd, dlt = g["l"].values, g["h"].values, g["c"].values, g["cvd"].values, g["delta"].values
    vol = g["vol"].values
    marks = []

    def add(i, kind, side, text):
        marks.append(dict(t=str(t[i])[11:16], i=i, kind=kind, side=side, text=text))

    # Divergence: new 15-min low (high) at bar i, CVD at i above (below) CVD at the prior low's bar by
    # >= 10 % of the CVD range over [i-15, i]. One marker per side per 5 bars.
    last = {"bull": -99, "bear": -99}
    for i in range(15, n):
        w = slice(i - 15, i)
        rng = cvd[i - 15:i + 1].max() - cvd[i - 15:i + 1].min()
        if rng <= 0:
            continue
        if L[i] < L[w].min():
            k = i - 15 + int(np.argmin(L[w]))
            if cvd[i] >= cvd[k] + 0.10 * rng and i - last["bull"] >= 5:
                add(i, "divergence", "bull", "Sellers fading: price made a new 15-min low, but cumulative "
                    "delta is higher than at the last low")
                last["bull"] = i
        if H[i] > H[w].max():
            k = i - 15 + int(np.argmax(H[w]))
            if cvd[i] <= cvd[k] - 0.10 * rng and i - last["bear"] >= 5:
                add(i, "divergence", "bear", "Buyers fading: price made a new 15-min high, but cumulative "
                    "delta is lower than at the last high")
                last["bear"] = i

    # Push decay: a push = first bar within tol of the trailing 30-min high (low) after a retreat of more
    # than tol + 2 pts; push delta = net of the 2 bars before it, signed toward the edge. Marker when a
    # push's delta <= 0.6 x the previous push's (previous > 0, within 30 min).
    tol, retreat = 1.0, 2.0
    for side, sg in (("high", 1), ("low", -1)):
        inpush, prev = False, None
        for i in range(30, n):
            ref = H[i - 30:i].max() if sg == 1 else L[i - 30:i].min()
            px = H[i] if sg == 1 else L[i]
            near = sg * (px - ref) >= -tol
            far = sg * (px - ref) < -(tol + retreat)
            if near and not inpush:
                p = sg * dlt[i - 2:i].sum()
                if prev is not None and i - prev[0] <= 30 and prev[1] > 0 and p <= 0.6 * prev[1]:
                    who = "Buyers" if sg == 1 else "Sellers"
                    add(i, "push_decay", "bear" if sg == 1 else "bull",
                        f"{who} weaker on the retest: push into the 30-min {side} had {p:+.0f} net vs "
                        f"{prev[1]:+.0f} on the previous push")
                prev, inpush = (i, p), True
            elif far:
                inpush = False

    # Fading: over the trailing 30 bars, the dominant side's delta share in the last 10 bars minus the
    # first 10 bars, crossing below 0. One marker per 10 bars.
    lastf, prevle = -99, None
    for i in range(29, n):
        a, b = slice(i - 29, i - 19), slice(i - 9, i + 1)
        if vol[a].sum() <= 0 or vol[b].sum() <= 0:
            prevle = None
            continue
        s = np.sign(dlt[i - 29:i + 1].sum()) or 1.0
        le = s * (dlt[b].sum() / vol[b].sum() - dlt[a].sum() / vol[a].sum())
        if prevle is not None and prevle >= 0 > le and i - lastf >= 10:
            who = "Buyers" if s > 0 else "Sellers"
            add(i, "fading", "bear" if s > 0 else "bull",
                f"{who} fading: their share of volume in the last 10 min is below the first 10 of the last 30")
            lastf = i
        prevle = le
    return g, sorted(marks, key=lambda m: m["i"])


out = {}
for table, tag in (("es_flow_1s_databento", "databento"), ("ironbeam_es_flow_1s", "ironbeam")):
    prior = sessions_before(table, 20)
    df = minutes(table, prior + [pd.Timestamp(DAY).date()])
    imps = []
    for d in prior:
        g = full_index(df[df["d"] == d].drop(columns=["d", "symbol"]), d)
        imps.append(abs_impact(g).dropna())
    pool = pd.concat(imps)
    med, p20 = float(pool.median()), float(pool.quantile(0.2))
    today = df[df["d"] == pd.Timestamp(DAY).date()]
    sym = today["symbol"].iloc[0]
    g, marks = features(full_index(today.drop(columns=["d", "symbol"]), DAY), med, p20)
    out[tag] = dict(symbol=sym, prior=[str(prior[0]), str(prior[-1]), len(prior)], base_med=med, base_p20=p20,
                    absorb_bars=int(g["absorb"].sum()), marks=marks,
                    series=[dict(t=str(ts)[11:16], o=r.o, h=r.h, l=r.l, c=r.c, delta=r.delta, cvd=r.cvd,
                                 share=None if pd.isna(r.share) else r.share,
                                 imp_rel=None if pd.isna(r.imp_rel) else r.imp_rel, absorb=bool(r.absorb),
                                 vol=r.vol) for ts, r in g.iterrows()])

# Current indicator, exactly as the route computes it in its FULL / 1m / ema_len 14 default
# (Ironbeam 1 s, prior 15:00 PT -> 13:00 PT, reindexed per second, 1 min sums, EMA(buy) - EMA(sell)).
sym_ib = out["ironbeam"]["symbol"]
raw = q("""SELECT ts_utc, buy_vol::float AS buy, sell_vol::float AS sell FROM ironbeam_es_flow_1s
           WHERE symbol = %s AND ts_utc >= '2026-04-30 22:00+00' AND ts_utc < '2026-05-01 20:00+00'
           ORDER BY ts_utc""", (sym_ib,))
raw["ts_utc"] = pd.to_datetime(raw["ts_utc"], utc=True)
idx = pd.date_range("2026-04-30 22:00", "2026-05-01 20:00", freq="1s", tz="UTC", inclusive="left")
r1 = raw.set_index("ts_utc").reindex(idx).fillna(0.0).resample("1min").sum()
cur = (r1["buy"].ewm(span=14, adjust=False).mean() - r1["sell"].ewm(span=14, adjust=False).mean())
cur.index = cur.index.tz_convert(PTZ)
cur = cur[(cur.index.strftime("%H:%M") >= "06:30") & (cur.index.strftime("%H:%M") < "13:00")]
out["current"] = [dict(t=ts.strftime("%H:%M"), v=float(v)) for ts, v in cur.items()]

# Chart candles as served (ironbeam_es_1m_bars), RTH
bars = q(f"""SELECT (datetime AT TIME ZONE 'UTC') AT TIME ZONE '{PTZ}' AS m, open o, high h, low l, close c FROM ironbeam_es_1m_bars
             WHERE datetime >= '2026-05-01 13:30+00' AND datetime < '2026-05-01 20:00+00' ORDER BY 1""")
out["bars"] = [dict(t=str(r.m)[11:16], o=r.o, h=r.h, l=r.l, c=r.c) for r in bars.itertuples()]
print(json.dumps(out, default=float))
