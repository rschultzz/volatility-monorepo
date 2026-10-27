#!/usr/bin/env python3
"""CR-CR Step 0 — counts, distributions, A/A on R-a. NO range outcome is computed. READ-ONLY, R-a only.

Spec: specs/CR-CR-effort-range-ra.md (E1-E9). Writes scripts/logs/cr_cr/step0.json.
  0. parity (E1): under the default, cr_cp_lib resolves train dates / guards / logs/cr_cp (subprocess, no data).
  1. es_flow_1m_databento vs the es_flow_1s_databento minute aggregate on R-a RTH (E4); symbols present (E3).
  2. funnel, IM, swing_d (C2 with E2's constant), roll-window sessions kept with their symbol and volume share (E3).
  3-5. ranges, calls, fills: primary (effort rule) + the E6 readout definitions.
  6. split counts on the primary (levels C14, call side, hour, G1-down flag).
  7. A/A on the random baseline (C13); E9 gate.
Summaries copied unchanged from scripts/cr_cp_step0.py.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time as _time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
chk = subprocess.run([sys.executable, "-c", "import cr_cp_lib as C; print(C.PERIOD, C.OUT.name, C.D_FROM, C.D_TO, "
                      "C.FLOOR.isoformat(), C.CAP.isoformat())"], cwd=HERE,
                     env={k: v for k, v in os.environ.items() if k not in ("CR_CP_PERIOD", "CR_CL_PERIOD")},
                     capture_output=True, text=True)
PARITY = chk.stdout.split()[-6:]
assert chk.returncode == 0 and PARITY == ["train", "cr_cp", "2026-01-06", "2026-06-05", "2026-01-01T00:00:00+00:00",
                                          "2026-06-05T21:00:00+00:00"], (chk.stdout, chk.stderr)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import cr_cr_lib as K  # noqa: E402

C = K.C
T0 = _time.time()
R: dict = dict(parity_default=PARITY)


def log(*a):
    print(f"[{_time.time() - T0:7.1f}s]", *a, flush=True)


def qtiles(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    if not x.size:
        return None
    return dict(n=int(x.size), p5=float(np.percentile(x, 5)), p25=float(np.percentile(x, 25)),
                p50=float(np.median(x)), p75=float(np.percentile(x, 75)), p95=float(np.percentile(x, 95)))


# ------------------------------------------------------------------------------------------------ 1. m1 vs 1s
log("1. es_flow_1m_databento vs es_flow_1s_databento (R-a RTH)")
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
syms = C.q("""SELECT symbol, raw_symbol, count(*) AS n, min(minute_utc) AS lo, max(minute_utc) AS hi
              FROM es_flow_1m_databento WHERE minute_utc >= %s AND minute_utc < %s GROUP BY 1, 2 ORDER BY 4""", (a, b))
R["symbols"] = [dict(symbol=r.symbol, raw_symbol=r.raw_symbol, n=int(r.n), lo=str(r.lo), hi=str(r.hi))
                for r in syms.itertuples()]
assert not any("c.0" in str(s) for s in syms["symbol"]) and not any("c.0" in str(s) for s in syms["raw_symbol"])
log("  symbols", R["symbols"])

# ----------------------------------------------------------------------------------------------- 2. universe
log("2. load minutes, IM, build")
m1 = C.load_minutes()
im = C.load_im()
D = C.build(m1, im)
R["funnel"] = D["funnel"]
R["median_im_used"] = D["MED_IM"]
assert D["MED_IM"] == C.IM_TRAIN_CR_CO
R["im_ra"] = qtiles(list(D["IM"].values()))
R["swing_d"] = qtiles(list(D["SW"].values()))
R["swing_d_lt6"] = sorted(str(d) for d, v in D["SW"].items() if v < 6.0)
R["n_sessions"] = len(D["SESS"])
R["missing_minutes"] = dict(sessions_with_any=int(sum(D["MISS"][d]["n_missing"] > 0 for d in D["SESS"])),
                            total=int(sum(D["MISS"][d]["n_missing"] for d in D["SESS"])),
                            max_run=qtiles([D["MISS"][d]["max_run"] for d in D["SESS"]]))
import cr_cm_lib as CM  # noqa: E402

vol = m1.copy()
vol["d"] = pd.to_datetime(vol["minute_utc"], utc=True).dt.tz_convert(C.PT).dt.date
vs = vol.groupby(["d", "symbol"])["vol"].sum()
R["roll_week_sessions_kept"] = {
    str(d): dict(symbol=D["SYM"][d], vol_share=float(vs[(d, D["SYM"][d])] / vs[d].sum()), n_symbols=int(vs[d].size))
    for d in D["SESS"] if CM.roll_week(d)}
R["sessions_two_symbols"] = int(sum(vs[d].size > 1 for d in D["SESS"]))
R["front_symbols"] = dict(Counter(D["SYM"][d] for d in D["SESS"]))
for f in D["funnel"]:
    log("  funnel", f[0], f[1], "->", f[2])
log("  usable", len(D["SESS"]), "IM_train", D["MED_IM"], "IM R-a", R["im_ra"], "swing_d", R["swing_d"])
log("  roll-window sessions kept", len(R["roll_week_sessions_kept"]), "front", R["front_symbols"])


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


def cpp_ratio_at(B, r0, t):
    """Diagnostic: cpp_up / cpp_down at the decision bar (the C12 quantity side_call thresholds)."""
    sl = slice(r0, t + 1)
    o, c, v = B["o"][sl], B["c"][sl], B["vol"][sl]
    up, dn = c > o, c < o
    if not (up.any() and dn.any()):
        return np.nan
    cu, cd = np.sum(v[up]) / np.sum(np.abs(c - o)[up]), np.sum(v[dn]) / np.sum(np.abs(c - o)[dn])
    return float(cu / cd) if cd > 0 else np.nan


log("3-5. schedules")
SCH = {}
R["schedules"] = {}
for nm, p in K.VARIANTS.items():
    S = C.schedule(D, p)
    SCH[nm] = S
    R["schedules"][nm] = summarise(S, C.leg_counts(D, p))
    s = R["schedules"][nm]
    log(f"  {nm:16s} ranges {s['n_ranges']:4d}  calls {s.get('calls')}  chosen fills "
        f"{(s.get('ch_fill') or {}).get('fill', 0)} on {s.get('chosen_fills_sessions')} sessions  status {s.get('dstatus')}")
P0 = SCH["primary"]
P0["cpp_ratio"] = [cpp_ratio_at(D["BARS"][r.d], r.r0, r.t) for r in P0.itertuples()]
R["cpp_ratio_at_decision"] = qtiles(P0.loc[P0["dstatus"] == "call", "cpp_ratio"])
R["cpp_ratio_no_call_at_end"] = qtiles(P0.loc[P0["dstatus"] == "no call", "cpp_ratio"])
P0.to_parquet(C.OUT / "step0_primary_schedule.parquet")
R["primary_ranges_by_session"] = {str(d): int(n) for d, n in P0.groupby("d").size().items()}

# ---------------------------------------------------------------------------------------------- 6. splits
log("6. splits (levels, call side, hour, G1-down flag)")
LV = C.load_levels()
R["levels_universe"] = dict(n=len(LV["USABLE"]), funnel=[f[:3] for f in LV["funnel"]])
P0 = C.tag_levels(P0, LV)
P0 = C.tag_skew(P0)
P0["hour"] = (390 + P0["t"]) // 60
P0["call_side"] = np.where(P0["side"] == 1, "long", np.where(P0["side"] == -1, "short", "none"))
call = P0[P0["dstatus"] == "call"]
fills = call[call["ch_status"] == "fill"]
R["splits"] = {}
for col in ("lv_strength", "lv_pos", "lv_kind", "call_side", "hour", "g1down_like", "skew"):
    R["splits"][col] = dict(
        ranges={str(k): int(v) for k, v in P0[col].value_counts().items()},
        calls={str(k): int(v) for k, v in call[col].value_counts().items()},
        chosen_fills={str(k): int(v) for k, v in fills[col].value_counts().items()})
for k, v in R["splits"].items():
    log("  ", k, v["chosen_fills"])

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

n_fill, n_sess = int(len(fills)), int(fills["d"].nunique())
R["gate"] = dict(chosen_fills=n_fill, sessions=n_sess, aa_mean_pp=R["aa"]["mean_diff_pp"],
                 passed=bool(n_fill >= 40 and n_sess >= 30 and abs(R["aa"]["mean_diff_pp"]) <= 1.0))
log("  E9 gate", R["gate"])

R["elapsed_s"] = _time.time() - T0
(C.OUT / "step0.json").write_text(json.dumps(R, indent=1, default=str))
log("wrote", C.OUT / "step0.json")
