"""CR-CD Step 0 Q3 — source switching. READ-ONLY (default_transaction_read_only asserted).

Mechanical coverage/symbol counts over all Databento dates (CR-CA A3.3 allows mechanical checks
on holdout rows); anything value-based (impact scale) is train-only (< 2026-06-05 21:00 UTC).
"""
import json
import sys
import time
from decimal import Decimal

import numpy as np
import pandas as pd

sys.path.insert(0, sys.argv[1])
from packages.shared.backfill_safety import get_backfill_db_conn, assert_role_or_die  # noqa: E402

conn = get_backfill_db_conn()
assert_role_or_die(conn)
conn.execute("SET default_transaction_read_only = on")
conn.execute("SET statement_timeout = '900s'")
assert conn.execute("SHOW default_transaction_read_only").fetchone()[0] == "on"


def q(sql, params=None):
    cur = conn.execute(sql, params or ())
    cols = [c.name for c in cur.description]
    df = pd.DataFrame(cur.fetchall(), columns=cols)
    for c in df.columns:
        if len(df) and any(isinstance(v, Decimal) for v in df[c].head(50)):
            df[c] = df[c].astype(float)
    return df


out = {}
PTZ = "America/Los_Angeles"

# 1. Databento coverage per PT date and symbol (RTH 06:30-13:00 volume picks the dominant symbol)
t = time.time()
cov = q(f"""
    SELECT (ts_utc AT TIME ZONE '{PTZ}')::date AS d, symbol, count(*) AS n,
           sum(volume) FILTER (WHERE (ts_utc AT TIME ZONE '{PTZ}')::time >= '06:30') AS rth_vol,
           min(ts_utc AT TIME ZONE '{PTZ}') AS first_pt, max(ts_utc AT TIME ZONE '{PTZ}') AS last_pt
    FROM es_flow_1s_databento GROUP BY 1, 2 ORDER BY 1, 2""")
out["db_cov_s"] = round(time.time() - t, 1)
dom = cov.sort_values("rth_vol").groupby("d").tail(1).set_index("d")
out["db_sessions"] = int(cov["d"].nunique())
out["db_two_symbol_sessions"] = int((cov.groupby("d")["symbol"].nunique() > 1).sum())
out["db_first_last_pt"] = [str(cov["first_pt"].min()), str(cov["last_pt"].max())]
ft = dom["first_pt"].dt.time.astype(str)
lt = dom["last_pt"].dt.time.astype(str)
out["db_dom_first_pt_counts"] = ft.str[:5].value_counts().head(5).to_dict()
out["db_dom_last_pt_counts"] = lt.str[:5].value_counts().head(8).to_dict()
out["db_early_close_sessions"] = [str(d) for d in dom.index[dom["last_pt"].dt.hour < 12]]

# 2. RTH max gap of the dominant symbol per session (Databento and Ironbeam)
def gaps(table, d0, d1):
    return q(f"""
        WITH r AS (
          SELECT (ts_utc AT TIME ZONE '{PTZ}')::date AS d, symbol, ts_utc,
                 extract(epoch FROM ts_utc - lag(ts_utc) OVER (PARTITION BY symbol,
                     (ts_utc AT TIME ZONE '{PTZ}')::date ORDER BY ts_utc)) AS g
          FROM {table}
          WHERE ts_utc >= %s AND ts_utc < %s
            AND (ts_utc AT TIME ZONE '{PTZ}')::time >= '06:30'
            AND (ts_utc AT TIME ZONE '{PTZ}')::time < '13:00')
        SELECT d, symbol, count(*) AS n, max(g) AS max_gap,
               count(*) FILTER (WHERE g > 15) AS n_gt15, count(*) FILTER (WHERE g > 60) AS n_gt60
        FROM r GROUP BY 1, 2""", (d0, d1))


t = time.time()
gdb = gaps("es_flow_1s_databento", "2024-04-01", "2026-10-25")
out["db_gap_s"] = round(time.time() - t, 1)
gdb = gdb.merge(dom[["symbol"]].reset_index(), on=["d", "symbol"])
out["db_rth_maxgap"] = {"p50": float(gdb["max_gap"].median()), "p95": float(gdb["max_gap"].quantile(.95)),
                        "max": float(gdb["max_gap"].max()),
                        "sessions_gt15": int((gdb["max_gap"] > 15).sum()),
                        "sessions_gt60": int((gdb["max_gap"] > 60).sum()),
                        "worst": gdb.nlargest(8, "max_gap")[["d", "max_gap"]].astype(str).values.tolist()}

ib = q(f"""
    SELECT (ts_utc AT TIME ZONE '{PTZ}')::date AS d, symbol, count(*) AS n,
           count(*) FILTER (WHERE (ts_utc AT TIME ZONE '{PTZ}')::time >= '06:30'
                              AND (ts_utc AT TIME ZONE '{PTZ}')::time < '13:00') AS n_rth,
           sum(volume) FILTER (WHERE (ts_utc AT TIME ZONE '{PTZ}')::time >= '06:30'
                              AND (ts_utc AT TIME ZONE '{PTZ}')::time < '13:00') AS rth_vol,
           min(ts_utc) AS first_utc
    FROM ironbeam_es_flow_1s GROUP BY 1, 2 ORDER BY 1, 2""")
out["ib_first_utc"] = str(ib["first_utc"].min())
out["ib_last_date"] = str(ib["d"].max())
ibdom = ib[ib["rth_vol"].notna()].sort_values("rth_vol").groupby("d").tail(1).set_index("d")
gib = gaps("ironbeam_es_flow_1s", "2025-12-20", "2026-10-26").merge(ibdom[["symbol"]].reset_index(),
                                                                   on=["d", "symbol"])
out["ib_rth_maxgap"] = {"sessions": int(len(gib)), "sessions_gt15": int((gib["max_gap"] > 15).sum()),
                        "sessions_gt60": int((gib["max_gap"] > 60).sum())}

# 3. Dominant-symbol agreement on sessions both tables cover
j = dom[["symbol"]].join(ibdom[["symbol"]], how="inner", lsuffix="_db", rsuffix="_ib")
dis = j[j["symbol_db"] != j["symbol_ib"]]
out["sym_overlap_sessions"] = int(len(j))
out["sym_disagree"] = [[str(d), r.symbol_db, r.symbol_ib] for d, r in dis.iterrows()]
out["db_only_after_ib_start"] = [str(d) for d in dom.index if d >= pd.Timestamp("2025-12-29").date()
                                 and d not in ibdom.index]
out["ib_after_db_end"] = [str(d) for d in ibdom.index if d > dom.index.max()]

# 4. Which Databento symbol matches the chart's bars (ironbeam_es_1m_bars) on two-symbol sessions
two = cov.groupby("d")["symbol"].nunique()
two_days = [d for d in two.index[two > 1]]
t = time.time()
m = q(f"""
    WITH f AS (
      SELECT date_trunc('minute', ts_utc) AS m, symbol,
             (array_agg(close ORDER BY ts_utc DESC))[1] AS c, sum(volume) AS v
      FROM es_flow_1s_databento
      WHERE (ts_utc AT TIME ZONE '{PTZ}')::date = ANY(%s)
        AND (ts_utc AT TIME ZONE '{PTZ}')::time >= '06:30'
      GROUP BY 1, 2)
    SELECT (f.m AT TIME ZONE '{PTZ}')::date AS d, f.symbol, count(*) AS n,
           avg((abs(f.c - b.close) < 0.01)::int)::float8 AS eqs, percentile_cont(.5) WITHIN GROUP
             (ORDER BY f.c - b.close) AS med_diff, sum(f.v)::float8 AS vol
    FROM f JOIN ironbeam_es_1m_bars b ON b.datetime = f.m
    GROUP BY 1, 2 ORDER BY 1, 2""", (two_days,))
out["bars_match_s"] = round(time.time() - t, 1)
best = m.sort_values("eqs").groupby("d").tail(1).set_index("d")
volw = m.sort_values("vol").groupby("d").tail(1).set_index("d")
cmp_ = best[["symbol", "eqs"]].join(volw[["symbol"]], rsuffix="_vol").join(dom[["symbol"]], rsuffix="_dom")
out["two_symbol_days"] = len(two_days)
out["bars_match_best_eq_dist"] = cmp_["eqs"].describe().round(3).to_dict()
out["bars_symbol_not_rth_dominant"] = [[str(d), r.symbol, round(r.eqs, 3), r.symbol_dom]
                                       for d, r in cmp_.iterrows() if r.symbol != r.symbol_dom]
out["bars_no_symbol_matches"] = [[str(d), r.symbol, round(r.eqs, 3)] for d, r in cmp_.iterrows() if r.eqs < 0.9]

# 5. Impact scale Databento vs Ironbeam, train only, RTH 5-min buckets, dominant symbol per table
def five(table, symmap):
    rows = []
    for d, sym in symmap.items():
        rows.append((d, sym))
    df = q(f"""
        WITH s AS (SELECT * FROM unnest(%s::date[], %s::text[]) AS s(d, sym))
        SELECT (f.ts_utc AT TIME ZONE '{PTZ}')::date AS d,
               to_timestamp(floor(extract(epoch FROM f.ts_utc) / 300) * 300) AS b,
               sum(buy_vol - sell_vol) AS net, sum(volume) AS vol,
               (array_agg(close ORDER BY ts_utc DESC))[1] AS c
        FROM {table} f JOIN s ON s.d = (f.ts_utc AT TIME ZONE '{PTZ}')::date AND s.sym = f.symbol
        WHERE f.ts_utc < '2026-06-05 21:00+00'
          AND (f.ts_utc AT TIME ZONE '{PTZ}')::time >= '06:25'
          AND (f.ts_utc AT TIME ZONE '{PTZ}')::time < '13:00'
        GROUP BY 1, 2 ORDER BY 1, 2""", ([r[0] for r in rows], [r[1] for r in rows]))
    df["dc"] = df.groupby("d")["c"].diff()
    df = df[df["b"].dt.tz_convert(PTZ).dt.time >= pd.Timestamp("06:30").time()]
    df = df[(df["net"] != 0) & df["dc"].notna()]
    df["abs_imp"] = df["dc"].abs() / (df["net"].abs() / 1000.0)
    df["sgn_imp"] = np.sign(df["net"]) * df["dc"] / (df["net"].abs() / 1000.0)
    return df


tr = [d for d in j.index if d < pd.Timestamp("2026-06-05").date() and d >= pd.Timestamp("2026-01-06").date()]
fdb = five("es_flow_1s_databento", {d: j.loc[d, "symbol_db"] for d in tr})
fib = five("ironbeam_es_flow_1s", {d: j.loc[d, "symbol_ib"] for d in tr})
sd = fdb.groupby("d")[["abs_imp", "sgn_imp"]].median()
si = fib.groupby("d")[["abs_imp", "sgn_imp"]].median()
r = (si / sd).dropna()
out["scale_sessions"] = int(len(r))
out["scale_ib_over_db"] = {k: {"median": round(float(r[k].median()), 3),
                               "p10": round(float(r[k].quantile(.1)), 3),
                               "p90": round(float(r[k].quantile(.9)), 3)} for k in r.columns}
mm = fdb.merge(fib, on=["d", "b"], suffixes=("_db", "_ib"))
out["share_ratio_minutes"] = int(len(mm))
ds_db, ds_ib = mm["net_db"] / mm["vol_db"], mm["net_ib"] / mm["vol_ib"]
out["delta_share_5m"] = {"slope_ib_on_db": round(float(np.polyfit(ds_db, ds_ib, 1)[0]), 3),
                         "corr": round(float(np.corrcoef(ds_db, ds_ib)[0, 1]), 3)}
out["net_5m_slope_ib_on_db"] = round(float(np.polyfit(mm["net_db"], mm["net_ib"], 1)[0]), 3)

# 6. L9: one-session fetch time, Databento and Ironbeam (2026-05-01 RTH window, dominant symbol)
for table, sym in (("es_flow_1s_databento", dom.loc[pd.Timestamp("2026-05-01").date(), "symbol"]),
                   ("ironbeam_es_flow_1s", ibdom.loc[pd.Timestamp("2026-05-01").date(), "symbol"])):
    ts = []
    for _ in range(3):
        t = time.time()
        n = len(q(f"""SELECT ts_utc, buy_vol, sell_vol, volume, open, high, low, close FROM {table}
                      WHERE symbol = %s AND ts_utc >= '2026-05-01 13:15+00' AND ts_utc < '2026-05-01 20:00+00'
                      ORDER BY ts_utc""", (sym,)))
        ts.append(round(time.time() - t, 3))
    out[f"fetch_{table}"] = {"rows": n, "seconds": ts}
    t = time.time()
    n = len(q(f"""SELECT ts_utc, buy_vol, sell_vol, volume, open, high, low, close FROM {table}
                  WHERE symbol = %s AND ts_utc >= '2026-04-01 13:15+00' AND ts_utc < '2026-05-01 20:00+00'
                    AND (ts_utc AT TIME ZONE '{PTZ}')::time >= '06:30'
                    AND (ts_utc AT TIME ZONE '{PTZ}')::time < '13:00'""", (sym,)))
    out[f"fetch_20sess_{table}"] = {"rows": n, "seconds": round(time.time() - t, 2)}

print(json.dumps(out, indent=1, default=str))
