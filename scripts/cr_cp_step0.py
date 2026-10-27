#!/usr/bin/env python3
"""CR-CP Step 0 (re-run after A1: primary = close box + close > 2-pt restart) — counts, distributions, A/A. NO range outcome is computed. READ-ONLY, train only.

Spec: specs/CR-CP-range-edge-flow.md, "Step 0". Writes scripts/logs/cr_cp/step0.json.
  1. es_flow_1m_databento vs the es_flow_1s_databento minute aggregate (C3).
  2. funnel, median IM, swing_d (C2, C4, C18).
  3-5. ranges, decisions, fills per definition (primary + each C12 readout).
  6. splits on the primary (levels C14, time C16, skew C15).
  7. A/A on the random baseline (C13): only random same-minute entries are raced.
"""
from __future__ import annotations

import json
import time as _time
from collections import Counter

import numpy as np
import pandas as pd

import cr_cp_lib as C

T0 = _time.time()
R: dict = {}


def log(*a):
    print(f"[{_time.time() - T0:7.1f}s]", *a, flush=True)


def qtiles(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    if not x.size:
        return None
    return dict(n=int(x.size), p5=float(np.percentile(x, 5)), p25=float(np.percentile(x, 25)),
                p50=float(np.median(x)), p75=float(np.percentile(x, 75)), p95=float(np.percentile(x, 95)))


# ------------------------------------------------------------------------------------------------ 1. m1 vs 1s
log("1. es_flow_1m_databento vs es_flow_1s_databento (train RTH)")
a, b = C.pt(C.D_FROM, 6, 30), C.pt(C.D_TO, 13, 0)
C.guard_ts(a, b)
cmp_sql = """
WITH s AS (
    SELECT date_trunc('minute', ts_utc) AS m, symbol,
           (array_agg(open ORDER BY ts_utc))[1] AS o, max(high) AS h, min(low) AS l,
           (array_agg(close ORDER BY ts_utc DESC))[1] AS c, sum(volume) AS v, sum(buy_vol) AS bv,
           sum(sell_vol) AS sv, count(*) AS ns
    FROM es_flow_1s_databento
    WHERE ts_utc >= %(a)s AND ts_utc < %(b)s
      AND (ts_utc AT TIME ZONE 'America/Los_Angeles')::time >= '06:30'
      AND (ts_utc AT TIME ZONE 'America/Los_Angeles')::time < '13:00'
    GROUP BY 1, 2
), m AS (
    SELECT minute_utc + make_interval(mins => %(shift)s) AS m, symbol, open AS o, high AS h, low AS l, close AS c,
           volume AS v, buy_vol AS bv, sell_vol AS sv, n_seconds AS ns
    FROM es_flow_1m_databento
    WHERE minute_utc >= %(a)s - interval '1 minute' AND minute_utc < %(b)s + interval '1 minute'
)
SELECT count(*) FILTER (WHERE s.m IS NOT NULL AND m.m IS NOT NULL) AS both_,
       count(*) FILTER (WHERE m.m IS NULL) AS only_1s,
       count(*) FILTER (WHERE s.m IS NULL) AS only_1m,
       count(*) FILTER (WHERE s.o = m.o) AS eq_o, count(*) FILTER (WHERE s.h = m.h) AS eq_h,
       count(*) FILTER (WHERE s.l = m.l) AS eq_l, count(*) FILTER (WHERE s.c = m.c) AS eq_c,
       count(*) FILTER (WHERE s.v = m.v) AS eq_v, count(*) FILTER (WHERE s.bv = m.bv) AS eq_buy,
       count(*) FILTER (WHERE s.sv = m.sv) AS eq_sell, count(*) FILTER (WHERE s.ns = m.ns) AS eq_nsec,
       max(abs(s.c - m.c)) AS maxdiff_c, max(abs(s.v - m.v)) AS maxdiff_v
FROM s FULL OUTER JOIN (SELECT * FROM m WHERE m >= %(a)s AND m < %(b)s
                          AND (m AT TIME ZONE 'America/Los_Angeles')::time >= '06:30'
                          AND (m AT TIME ZONE 'America/Los_Angeles')::time < '13:00') m
  ON s.m = m.m AND s.symbol = m.symbol
"""
R["m1_vs_1s"] = {}
for shift in (0, 1):
    r = C.q(cmp_sql, dict(a=a, b=b, shift=shift)).iloc[0].to_dict()
    R["m1_vs_1s"][f"shift_{shift}"] = {k: (float(v) if v is not None else None) for k, v in r.items()}
    log("  shift", shift, R["m1_vs_1s"][f"shift_{shift}"])
tod = C.q("""SELECT min((minute_utc AT TIME ZONE 'America/Los_Angeles')::time)::text AS lo,
                    max((minute_utc AT TIME ZONE 'America/Los_Angeles')::time)::text AS hi
             FROM es_flow_1m_databento WHERE minute_utc >= %s AND minute_utc < %s""", (a, b)).iloc[0].to_dict()
R["m1_window_pt"] = tod
log("  1m PT window", tod)

# ----------------------------------------------------------------------------------------------- 2. universe
log("2. load minutes, IM, build")
m1 = C.load_minutes()
im = C.load_im()
D = C.build(m1, im)
R["funnel"] = D["funnel"]
R["median_im"] = D["MED_IM"]
assert abs(D["MED_IM"] - 65.71) < 0.01, D["MED_IM"]   # A0: CR-CO A1.1 IM_train
R["swing_d"] = qtiles(list(D["SW"].values()))
R["n_sessions"] = len(D["SESS"])
R["missing_minutes"] = dict(sessions_with_any=int(sum(D["MISS"][d]["n_missing"] > 0 for d in D["SESS"])),
                            total=int(sum(D["MISS"][d]["n_missing"] for d in D["SESS"])),
                            max_run=qtiles([D["MISS"][d]["max_run"] for d in D["SESS"]]))
import cr_cm_lib as CM  # noqa: E402  (roll_week only here; CM's own load runs in step 6)
R["roll_week_sessions_kept"] = [str(d) for d in D["SESS"] if CM.roll_week(d)]
syms = Counter(D["SYM"][d] for d in D["SESS"])
R["front_symbols"] = dict(syms)
for f in D["funnel"]:
    log("  funnel", f[0], f[1], "->", f[2])
log("  usable", len(D["SESS"]), "median IM", round(D["MED_IM"], 2), "swing_d", R["swing_d"])


# ------------------------------------------------------------------------------------------- 3-5. schedules
def summarise(S: pd.DataFrame, legs: dict) -> dict:
    out = dict(n_ranges=int(len(S)), sessions_with_range=int(S["d"].nunique()) if len(S) else 0,
               ranges_per_session=qtiles(S.groupby("d").size().reindex(D["SESS"], fill_value=0).to_numpy())
               if len(S) else None,
               legs_per_session=qtiles(list(legs.values())),
               width=qtiles(S["width"]) if len(S) else None,
               leg_pts=qtiles(S["leg_pts"]) if len(S) else None)
    if not len(S):
        return out
    out["dstatus"] = dict(Counter(S["dstatus"]))
    call = S[S["dstatus"] == "call"]
    out["calls"] = dict(short=int((call["side"] == -1).sum()), long=int((call["side"] == 1).sum()))
    out["call_by_approach"] = {f"approach {'up' if k[0] == 1 else 'down'} -> {'long' if k[1] == 1 else 'short'}": int(v)
                               for k, v in call.groupby(["dir", "side"]).size().items()}
    out["bars_q_to_decision"] = qtiles(call["t"] - call["q"])
    out["share_at_decision"] = qtiles(call["share"])
    out["prog_at_decision"] = qtiles(call["prog"])
    nc = S[S["dstatus"] == "no call"]
    out["share_no_call_at_end"] = qtiles(nc["share"])
    out["prog_no_call_at_end"] = qtiles(nc["prog"])
    for nm in ("ch", "op"):
        out[f"{nm}_fill"] = dict(Counter(call[f"{nm}_status"]))
        out[f"{nm}_fill_share"] = float((call[f"{nm}_status"] == "fill").mean()) if len(call) else None
        out[f"{nm}_edge"] = dict(Counter(call.loc[call[f"{nm}_status"] == "fill", f"{nm}_edge"]))
    both = call[(call["ch_status"] == "fill") & (call["op_status"] == "fill")]
    out["both_filled"] = int(len(both))
    out["chosen_fills_sessions"] = int(call.loc[call["ch_status"] == "fill", "d"].nunique())
    if len(nc) and "nl_status" in nc:
        out["no_call_edge_fills"] = dict(long=dict(Counter(nc["nl_status"])), short=dict(Counter(nc["ns_status"])))
    return out


log("3-5. schedules")
SCH = {}
R["schedules"] = {}
for nm, p in C.VARIANTS.items():
    S = C.schedule(D, p)
    SCH[nm] = S
    R["schedules"][nm] = summarise(S, C.leg_counts(D, p))
    s = R["schedules"][nm]
    log(f"  {nm:16s} ranges {s['n_ranges']:4d}  calls {s.get('calls')}  chosen fills "
        f"{(s.get('ch_fill') or {}).get('fill', 0)}  status {s.get('dstatus')}")
SCH["primary"].to_parquet(C.OUT / "step0_primary_schedule.parquet")
hb = R["schedules"]["hl_box"]                          # A1: hl_box = Step 0 v1's primary (parity with edd21bb)
assert hb["n_ranges"] == 322 and hb["ch_fill"].get("fill") == 115, (hb["n_ranges"], hb["ch_fill"])

# per-session range counts (primary)
P0 = SCH["primary"]
R["primary_ranges_by_session"] = {str(d): int(n) for d, n in P0.groupby("d").size().items()}

# ---------------------------------------------------------------------------------------------- 6. splits
log("6. splits (levels, time, skew)")
LV = C.load_levels()
R["levels_universe"] = dict(n=len(LV["USABLE"]), funnel=[f[:3] for f in LV["funnel"]])
P0 = C.tag_levels(P0, LV)
P0 = C.tag_skew(P0)
P0 = C.tag_a9(P0, LV)
P0["hour"] = (390 + P0["t"]) // 60
P0["first"] = np.where(P0["n_in_session"] == 1, "first range", "later")
call = P0[P0["dstatus"] == "call"]
fills = call[call["ch_status"] == "fill"]
R["splits"] = {}
for col in ("lv_pos", "lv_strength", "lv_kind", "hour", "first", "skew", "g1down_like", "a9"):
    R["splits"][col] = dict(
        ranges={str(k): int(v) for k, v in P0[col].value_counts().items()},
        calls={str(k): int(v) for k, v in call[col].value_counts().items()},
        chosen_fills={str(k): int(v) for k, v in fills[col].value_counts().items()})
R["splits"]["lv_pos_by_approach"] = {f"{'up' if k[0] == 1 else 'down'} / {k[1]}": int(v)
                                    for k, v in fills.groupby(["dir", "lv_pos"]).size().items()}
for k, v in R["splits"].items():
    log("  ", k, v.get("chosen_fills", v))

# ------------------------------------------------------------------------------------------------- 7. A/A
log("7. A/A on the random baseline (random same-minute entries only)")
BT = C.baseline_table(D)
BT = BT[BT["res"].isin(["win", "loss"])].copy()
BT["w"] = (BT["res"] == "win").astype(float)
keys = list(zip(fills["ch_j"].astype(int), fills["side"].astype(int), fills["ch_edge"]))
R["aa_keys"] = len(keys)
rng = np.random.default_rng(C.SEED_AA)
sess = np.array(D["SESS"], dtype=object)
grp = {k: g for k, g in BT.groupby(["m", "h", "edge"])}
diffs, excl = [], []
for _ in range(C.N_AA):
    perm = rng.permutation(len(sess))
    A = set(sess[perm[: len(sess) // 2]])
    vals, nx = [], 0
    for k in keys:
        g = grp.get(k)
        if g is None:
            nx += 1
            continue
        inA = g["d"].isin(A).to_numpy()
        ga, gb = g[inA], g[~inA]
        if not len(ga) or not len(gb):
            nx += 1
            continue
        vals.append(float(ga["w"].iloc[rng.integers(len(ga))]) - float(gb["w"].mean()))
    diffs.append(float(np.mean(vals)) if vals else np.nan)
    excl.append(nx / max(len(keys), 1))
diffs = np.array(diffs)
R["aa"] = dict(n_splits=C.N_AA, mean_diff_pp=100 * float(np.nanmean(diffs)), median_diff_pp=100 * float(np.nanmedian(diffs)),
               sd_diff_pp=100 * float(np.nanstd(diffs)), p5_pp=100 * float(np.nanpercentile(diffs, 5)),
               p95_pp=100 * float(np.nanpercentile(diffs, 95)), mean_excluded_share=float(np.mean(excl)),
               baseline_win_rate=float(BT["w"].mean()), baseline_n=int(len(BT)),
               baseline_edge_share=float((BT["edge"] == "EDGE").mean()))
log("  A/A", R["aa"])

# A1.5 gate
n_fill, n_sess = int(len(fills)), int(fills["d"].nunique())
R["a1_gate"] = dict(chosen_fills=n_fill, sessions=n_sess, aa_mean_pp=R["aa"]["mean_diff_pp"],
                    passed=bool(n_fill >= 40 and n_sess >= 30 and abs(R["aa"]["mean_diff_pp"]) <= 1.0))
log("  A1.5 gate", R["a1_gate"])

R["elapsed_s"] = _time.time() - T0
(C.OUT / "step0.json").write_text(json.dumps(R, indent=1, default=str))
log("wrote", C.OUT / "step0.json")
