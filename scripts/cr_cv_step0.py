#!/usr/bin/env python3
"""CR-CV Step 0 (C9) — counts, 1-s data checks, fills, ±10 reproduction. READ-ONLY, train only.

No ±2 / ±4 outcome and no primary-fill outcome of any size is computed here. The only races are ±10: (a) each CR's own
race function on its own bars against the stored outcomes of the original run, and (b) the 1-s touch-fill model (C5)
against each CR's reported win rates (C8: within 1 pp, else STOP). Writes scripts/logs/cr_cv/step0.json.
"""
from __future__ import annotations

import json
import sys
import time as _time
from collections import Counter

import numpy as np
import pandas as pd

import cr_cv_lib as V
import cr_cl_lib as CL
import cr_ct_lib as CT

T0 = _time.time()
C = V.C
R: dict = {}
FAIL: list[str] = []


def log(*a):
    print(f"[{_time.time() - T0:7.1f}s]", *a, flush=True)


def check(name, ok, detail=None):
    R.setdefault("checks", {})[name] = dict(ok=bool(ok), detail=detail)
    log("CHECK", "ok  " if ok else "FAIL", name, "" if detail is None else detail)
    if not ok:
        FAIL.append(name)


ro = C.conn().execute("SELECT current_setting('default_transaction_read_only'), current_user").fetchone()
check("read-only connection", ro[0] == "on", list(ro))

A = V.build_all()
TR = A["TR"]
log("schedule rows", len(TR))

# ------------------------------------------------------------------------------------------------ event sets vs ref
ref = V.REF_DIR
G1, cov1 = A["G1"], A["cov1"]
rcl = pd.read_parquet(ref / "cl_legs.parquet")
rcl = rcl[rcl["cell"] == "G1-down"].reset_index(drop=True)
keys = ["d", "ci", "p0", "dir", "edge", "tercile", "e", "s"]
check("S1 event set = CR-CL stored G1-down legs", len(G1) == len(rcl) == 49 and
      G1[keys].reset_index(drop=True).equals(rcl[keys]), dict(n=len(G1), ref=len(rcl)))
D1 = A["D1"]
o_cl = [CL.race(D1["BARS"][r.d], r.ci, r.p0, 1, 10.0) for r in G1.itertuples()]
check("S1 CR-CL race reproduces stored r10", o_cl == list(rcl["r10"]), dict(Counter(o_cl)))
o_cl_opp = [CL.race(D1["BARS"][r.d], r.ci, r.p0, -1, 10.0) for r in G1.itertuples()]
check("S1 covered legs", int(cov1.sum()) == 47, int(cov1.sum()))

S2 = A["S2"]
rcp = pd.read_parquet(ref / "cp_trades.parquet").reset_index(drop=True)
k2 = ["d", "t", "side", "hi", "lo", "ch_status", "op_status"]
same2 = len(S2) == len(rcp) == 235 and S2[k2].reset_index(drop=True).equals(rcp[k2])
for nm in ("ch", "op"):
    for col in (f"{nm}_j", f"{nm}_px"):
        a, b = S2[col].astype(float).to_numpy(), rcp[col].astype(float).to_numpy()
        same2 &= bool(np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(a[~np.isnan(a)], b[~np.isnan(b)]))
check("S2 event set = CR-CP stored schedule (calls, fills)", same2, dict(n=len(S2), ref=len(rcp)))
Dcp = A["Dcp"]
o_cp = {}
for nm, sg in (("ch", 1), ("op", -1)):
    out = []
    for r in S2.itertuples():
        st, j, px = getattr(r, f"{nm}_status"), getattr(r, f"{nm}_j"), getattr(r, f"{nm}_px")
        out.append(C.race_from(Dcp["BARS"][r.d], int(j), float(px), sg * int(r.side), 10.0) if st == "fill" else None)
    o_cp[nm] = out
    stored = [x if isinstance(x, str) else None for x in rcp[f"{nm}_res"]]
    check(f"S2 CR-CP race reproduces stored {nm}_res", out == stored, dict(Counter(out)))

S3 = A["S3"]
rcs = pd.read_parquet(ref / "cs_trades.parquet").reset_index(drop=True)
k3 = ["d", "r", "h", "px", "b", "dir"]
check("S3 event set = CR-CS stored fades", len(S3) == len(rcs) == 58 and S3[k3].reset_index(drop=True).equals(rcs[k3]),
      dict(n=len(S3), ref=len(rcs)))
o_cs = {nm: [CL.race(Dcp["BARS"][r.d], int(r.r), float(r.px), sg * int(r.h), 10.0) for r in S3.itertuples()]
        for nm, sg in (("ch", 1), ("op", -1))}
for nm in ("ch", "op"):
    check(f"S3 CR-CL race reproduces stored {nm}_res", o_cs[nm] == list(rcs[f"{nm}_res"]), dict(Counter(o_cs[nm])))

P4 = A["P4"]
rct = pd.read_parquet(ref / "ct_events.parquet")
rct = rct[rct["cls10"].isin(["FAST", "SLOW"])].reset_index(drop=True)
k4 = ["d", "L", "j", "b", "k1s", "cls10", "fam"]
check("S4 event set = CR-CT stored FAST / SLOW first touches", len(P4) == len(rct) == 160 and
      P4[k4].reset_index(drop=True).equals(rct[k4].astype(P4[k4].dtypes.to_dict())), dict(n=len(P4), ref=len(rct)))
o_ct = [CT.race1s(A["S1ct"][r.d], int(r.k1s), float(r.L), int(r.b), 10.0) for r in P4.itertuples()]
check("S4 CR-CT race reproduces stored res10 (break)", o_ct == list(rct["res10"]), dict(Counter(o_ct)))

R["events"] = {
    "S1": dict(n=len(G1), sessions=int(G1["d"].nunique()), covered=int(cov1.sum()),
               on_jan02_jan05=int(G1["d"].astype(str).isin(["2026-01-02", "2026-01-05"]).sum()),
               universe=len(A["UNIV"]["S1"])),
    "S2": dict(n=len(S2), sessions=int(S2["d"].nunique()), chosen_short=int((S2["side"] == -1).sum()),
               chosen_long=int((S2["side"] == 1).sum()), universe=len(A["UNIV"]["S2"])),
    "S3": dict(n=len(S3), sessions=int(S3["d"].nunique()), universe=len(A["UNIV"]["S3"])),
    "S4": dict(n=len(P4), sessions=int(P4["d"].nunique()), FAST=int((P4["cls10"] == "FAST").sum()),
               SLOW=int((P4["cls10"] == "SLOW").sum()), universe=len(A["UNIV"]["S4"])),
}
log("events", json.dumps(R["events"]))

# ------------------------------------------------------------------------------------------------ 1-s data (C2)
S1S = A["S1S"]
n_rows = sum(len(s["sec"]) for s in S1S.values())
bad_oc = sum(int(((s["o"] < s["l"]) | (s["o"] > s["h"]) | (s["c"] < s["l"]) | (s["c"] > s["h"])).sum())
             for s in S1S.values())
check("1-s o, c within [l, h]", bad_oc == 0, dict(rows=n_rows, pairs=len(S1S), bad=bad_oc))


def agg_equal(bars, S):
    m = V.minute_agg(S)
    have = np.isfinite(m["o"])
    ok = np.array_equal(have, np.isfinite(bars["c"]))
    diff = 0
    for k in "ohlc":
        diff += int((np.abs(m[k][have] - bars[k][have]) > 1e-9).sum())
    return ok, diff, int(have.sum())


mis = []
n_min = 0
for d in sorted(set(G1["d"])):
    ok, diff, n = agg_equal(D1["BARS"][d], S1S[(d, D1["SYM"][d])])
    n_min += n
    if not ok or diff:
        mis.append(("S1", str(d), ok, diff))
for d in Dcp["SESS"]:
    ok, diff, n = agg_equal(Dcp["BARS"][d], S1S[(d, Dcp["SYM"][d])])
    n_min += n
    if not ok or diff or n != V.NB:
        mis.append(("S2-S4", str(d), ok, diff, n))
check("1-s minute aggregate = each setup's minute bars", not mis, dict(minutes=n_min, mismatches=mis[:10]))
ctd = 0
for d in Dcp["SESS"]:
    a, b = A["S1ct"][d], S1S[(d, Dcp["SYM"][d])]
    if not (np.array_equal(a["sec"], b["sec"]) and all(np.array_equal(a[k], b[k]) for k in "hlc")):
        ctd += 1
check("1-s h / l / c = CR-CT's 1-s cache", ctd == 0, dict(sessions_differing=ctd))

# ------------------------------------------------------------------------------------------------ fills (C9)
fills = {}
for (st, sd, md), g in TR.groupby(["setup", "side", "model"]):
    fills[f"{st}/{sd}/{md}"] = dict(n=len(g), fills=int((g["status"] == "fill").sum()),
                                    sessions=int(g.loc[g["status"] == "fill", "d"].nunique()),
                                    status=dict(Counter(g["status"])))
R["fills"] = fills
for k, v in fills.items():
    log("fills", k, v)
tt = {}
for st in ("S2", "S4"):
    for sd in ("hyp", "opp"):
        p, t = fills[f"{st}/{sd}/primary"]["fills"], fills[f"{st}/{sd}/touch"]["fills"]
        tt[f"{st}/{sd}"] = dict(touch_fills=t, trade_through_fills=p, share=round(p / t, 4) if t else None)
R["trade_through_share"] = tt
log("trade-through share", json.dumps(tt))

# S2 touch model must find CR-CP's own fill minute; S4 touch model = CR-CT's touch second
s2t = TR[(TR["setup"] == "S2") & (TR["model"] == "touch")]
mm = 0
for r in s2t.itertuples():
    row = S2.iloc[r.ev]
    nm = "ch" if r.side == "hyp" else "op"
    oj = getattr(row, f"{nm}_j")
    if (r.status == "fill") != (getattr(row, f"{nm}_status") == "fill") or (
            r.status == "fill" and int(r.sub.split("=")[1]) != int(oj)):
        mm += 1
check("S2 touch-fill minute = CR-CP fill minute", mm == 0, dict(mismatches=mm))
k4m = 0
for r in P4.itertuples():
    lf = V.level_fill(S1S[(r.d, Dcp["SYM"][r.d])], int(r.j), float(r.L), int(r.b), -int(r.b), 0.0)
    k4m += lf is None or lf[0] != int(r.k1s)
check("S4 touch second = CR-CT k1s", k4m == 0, dict(mismatches=int(k4m)))

# ------------------------------------------------------------------------------------------------ ±10 reproduction (C8b)
def rate(outs):
    w, l_ = sum(o == "win" for o in outs), sum(o == "loss" for o in outs)
    return (w / (w + l_) if w + l_ else None), w, l_


touch10 = {}
for (st, sd), g in TR[TR["model"] == "touch"].groupby(["setup", "side"]):
    outs = [V.outcome(A, r, V.REF) for r in g.itertuples()]
    touch10[(st, sd)] = (g, outs)

targets = {   # reported win rates of the original runs (specs), side -> (pp, source)
    ("S1", "hyp"): (66.0, "CR-CL G1-down covered 31 / 47"),
    ("S2", "hyp"): (46.4, "CR-CP chosen"), ("S2", "opp"): (55.2, "CR-CP opposite"),
    ("S3", "hyp"): (39.2, "CR-CS fade"), ("S3", "opp"): (60.8, "CR-CS continuation"),
}
repro = {}


def rep(name, outs_new, target_pp, src):
    p, w, l_ = rate(outs_new)
    ok = p is not None and abs(100 * p - target_pp) <= 1.0
    repro[name] = dict(touch_fill_1s_pp=None if p is None else round(100 * p, 2), wins=w, losses=l_,
                       original_pp=target_pp, source=src, within_1pp=ok)
    check(f"±10 reproduction {name}", ok, repro[name])


g, outs = touch10[("S1", "hyp")]
cov = [cov1[e] for e in g["ev"]]
rep("S1 hyp (covered)", [o for o, c in zip(outs, cov) if c], 66.0, "CR-CL G1-down covered 31 / 47")
p0, w0, l0 = rate(o_cl)
rep("S1 hyp (all 49)", outs, round(100 * p0, 2), f"CR-CL race, all 49 legs {w0} / {w0 + l0}")
po, wo, lo_ = rate(o_cl_opp)
rep("S1 opp (all 49)", touch10[("S1", "opp")][1], round(100 * po, 2), f"CR-CL race h = -1, {wo} / {wo + lo_}")
for k in (("S2", "hyp"), ("S2", "opp"), ("S3", "hyp"), ("S3", "opp")):
    rep(f"{k[0]} {k[1]}", touch10[k][1], *targets[k])
# S4: break rates by class, and the pooled hypothesis / opposite derived from the stored outcomes
for sd in ("hyp", "opp"):
    g, outs = touch10[("S4", sd)]
    for cls in ("FAST", "SLOW"):
        sub = [o for o, s in zip(outs, g["sub"]) if s == cls]
        is_break = (sd == "hyp") == (cls == "FAST")
        brk = sub if is_break else [{"win": "loss", "loss": "win"}.get(o, o) for o in sub]
        tgt = 56.5 if cls == "FAST" else 54.3
        if sd == "hyp":
            rep(f"S4 {cls} break (from {sd} side)", brk, tgt, f"CR-CT {cls} break rate")
st_o = dict(zip(range(len(P4)), o_ct))
hyp_orig = [st_o[i] if c == "FAST" else {"win": "loss", "loss": "win"}.get(st_o[i], st_o[i])
            for i, c in enumerate(P4["cls10"])]
ph, wh, lh = rate(hyp_orig)
rep("S4 hyp pooled", touch10[("S4", "hyp")][1], round(100 * ph, 2), f"CR-CT stored, FAST break + SLOW fade {wh} / {wh + lh}")
opp_orig = [{"win": "loss", "loss": "win"}.get(o, o) for o in hyp_orig]
pp_, wp, lp = rate(opp_orig)
rep("S4 opp pooled", touch10[("S4", "opp")][1], round(100 * pp_, 2), f"CR-CT stored, FAST fade + SLOW break {wp} / {wp + lp}")
R["repro10"] = repro

# ------------------------------------------------------------------------------------------------ gate
gate = {}
for st in ("S1", "S2", "S3", "S4"):
    for sd in ("hyp", "opp"):
        f = fills[f"{st}/{sd}/primary"]["fills"]
        gate[f"{st}/{sd}"] = dict(primary_fills=f, status="ok" if f >= V.MIN_N else "UNDERPOWERED by fills")
R["gate"] = dict(cells=gate, data_and_repro_checks_failed=FAIL, passed=not FAIL)
R["n_schedule_rows"] = len(TR)
(V.OUT / "step0.json").write_text(json.dumps(R, indent=2, default=str))
TR.to_parquet(V.OUT / "step0_schedule.parquet")
log("GATE", json.dumps(R["gate"]))
sys.exit(1 if FAIL else 0)
