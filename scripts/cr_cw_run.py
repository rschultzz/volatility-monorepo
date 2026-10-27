#!/usr/bin/env python3
"""CR-CW run — the single outcome run on train (spec C12-C16). READ-ONLY, train only. Run once.

H1 primary: break win rate CLEAN − CHOPPY over CR-CT's 422 first touches (10-min path efficiency terciles from
step0.json), entry at the level at the touch second (R2 level-entry variant, labelled), ±10 1-s race to 13:00 PT
(CR-CT race1s on CR-CT's 1-s arrays — identical to CR-CT's outcome).
H2 primary: continuation win rate HOLD − FILL over the deduped H2 trades (break 6 / pullback 3), entry at the
turn-bar close, ±10 1-s race from the next bar to 13:00 PT.
Session-cluster bootstrap 10,000 draws, seed 20260920; two-sided bootstrap p; Holm m = 2; B1 verdicts (C14).
Readouts unadjusted, no verdicts. Asserts the Step 0 gate, cuts and event sets. Writes logs/cr_cw/results.json.
"""
from __future__ import annotations

import json
import time as _time
from collections import Counter

import numpy as np
import pandas as pd

import cr_cw_lib as W
import cr_ct_lib as T
import cr_cl_lib as CL

T0 = _time.time()
OUT = W.OUT / "results.json"
assert not OUT.exists(), "CR-CW is run once; results.json already exists"
S0J = json.loads((W.OUT / "step0.json").read_text())
assert S0J["gate"]["H1"]["passed"] and S0J["gate"]["H2"]["passed"], S0J["gate"]


def log(*a):
    print(f"[{_time.time() - T0:7.1f}s]", *a, flush=True)


D, S_ct, S1, P = W.build_all()
SESS = D["SESS"]
assert len(SESS) == S0J["n_sessions"] and len(P) == S0J["n_touches"]
SI = {d: i for i, d in enumerate(SESS)}
nS = len(SESS)
for w, v in S0J["h1_cuts"].items():
    c, n = W.tercile_cuts(P[f"eff{w}"])
    assert np.allclose(c, v["cuts"]) and n == v["n"], (w, c, v)
    P[f"h1{w}"] = W.h1_class(P[f"eff{w}"], v["cuts"])
ct = json.loads(W.CT_STEP0.read_text())
c10 = ct["cuts"]["10"]["cuts"]
P["cls10"] = T.classify(P, "speed10", "relvol10", c10)
P["spd_terc"] = np.where(P["speed10"] >= c10[1], "top", np.where(P["speed10"] <= c10[0], "bottom", "middle"))
retr_med = S0J["retrace"]["median"]
P["retr_hl"] = np.where(P["retr10"] <= retr_med, "low", "high")
P["retr_band"] = [W.retr_band(x) for x in P["retr10"]]

VARIANTS = {"primary": (W.BRK, W.PB)}
VARIANTS.update({f"brk{int(x)}": (x, W.PB) for x in W.BRK_READ})
VARIANTS.update({f"pb{int(x)}": (W.BRK, x) for x in W.PB_READ})
H2 = {}
for name, (brk, pb) in VARIANTS.items():
    H = W.h2_events(D, S1, P, brk, pb)
    tr = H[(H["stage"] == "trade") & ~H["dup"]].copy()
    want = S0J["h2"][name]["classes"]
    got = {k: dict(n=int(len(g)), sessions=int(g["d"].nunique())) for k, g in tr.groupby("cls")}
    assert got == want, (name, got, want)
    H2[name] = tr

WT = CL.boot_weights(nS, W.N_BOOT, np.random.default_rng(W.SEED))
ONE = np.ones((1, nS))
R: dict = dict(n_sessions=nS, n_boot=W.N_BOOT, seed=W.SEED,
               entry_h1="R2 level price at the touch second (labelled)", entry_h2="turn-bar close (C11)")

log("outcomes")
for col, X in (("res10", W.X1), ("res20", W.X2)):
    P[col] = [T.race1s(S_ct[r.d], int(r.k1s), r.L, int(r.b), X) for r in P.itertuples()]
P["mfe"] = [T.mfe1s(S_ct[r.d], int(r.k1s), r.L, int(r.b)) for r in P.itertuples()]
for tr in H2.values():
    for col, X in (("res10", W.X1), ("res20", W.X2)):
        tr[col] = [W.h2_race(S1[e["d"]], e, X) for e in tr.to_dict("records")]
    tr["mfe"] = [W.h2_mfe(S1[e["d"]], e) for e in tr.to_dict("records")]
Q = H2["primary"]


# ------------------------------------------------------------------------------------------------ helpers
def sc(rows):
    n, w = np.zeros(nS), np.zeros(nS)
    for d, x in rows:
        n[SI[d]] += 1
        w[SI[d]] += x
    return n, w


def rate(Wt, n, w):
    with np.errstate(invalid="ignore", divide="ignore"):
        return (Wt @ w) / (Wt @ n)


def ci(arr):
    a = arr[np.isfinite(arr)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if a.size else [None, None]


def win_rows(E, col="res10"):
    m = E[col].isin(["win", "loss"])
    return [(d, 1.0 if x == "win" else 0.0) for d, x in zip(E.loc[m, "d"], E.loc[m, col])]


def one(E, col="res10"):
    n, w = sc(win_rows(E, col))
    out = dict(n=int(len(E)), resolved=int(n.sum()), sessions=int((n > 0).sum()), res=dict(Counter(E[col])))
    if n.sum():
        p, b = rate(ONE, n, w)[0], rate(WT, n, w)
        lo, hi = ci(b - 0.5)
        out.update(win_rate=float(p), vs50_pp=dict(est=100 * (p - .5), ci=[100 * lo, 100 * hi]))
    return out


def diff(A, B, col="res10", draws=False):
    na, wa = sc(win_rows(A, col))
    nb, wb = sc(win_rows(B, col))
    out = dict(a=one(A, col), b=one(B, col))
    if na.sum() and nb.sum():
        pa, pb = rate(ONE, na, wa)[0], rate(ONE, nb, wb)[0]
        dr = rate(WT, na, wa) - rate(WT, nb, wb)
        lo, hi = ci(dr)
        out["diff_pp"] = dict(est=100 * (pa - pb), ci=[100 * lo, 100 * hi])
        if draws:
            out["_draws"], out["_na"], out["_nb"] = dr, int(na.sum()), int(nb.sum())
    return out


def mean_diff(A, B, col):
    def rows(E):
        x = E[col].astype(float)
        m = x.notna()
        return sc(zip(E.loc[m, "d"], x[m]))
    na, sa = rows(A)
    nb, sb = rows(B)
    out = dict(a_mean=float(rate(ONE, na, sa)[0]) if na.sum() else None, a_n=int(na.sum()),
               b_mean=float(rate(ONE, nb, sb)[0]) if nb.sum() else None, b_n=int(nb.sum()))
    if na.sum() and nb.sum():
        lo, hi = ci(rate(WT, na, sa) - rate(WT, nb, sb))
        out["diff"] = dict(est=out["a_mean"] - out["b_mean"], ci=[lo, hi])
    return out


def mfe_read(A, B):
    return dict(mean=mean_diff(A, B, "mfe"), a_q=[float(x) for x in A["mfe"].quantile([.25, .5, .75])],
                b_q=[float(x) for x in B["mfe"].quantile([.25, .5, .75])],
                a_reach10=float((A["mfe"] >= 10).mean()), b_reach10=float((B["mfe"] >= 10).mean()))


BT = T.baseline_table(D)
BT = BT[BT["res"].isin(["win", "loss"])]
base_rows = [(d, (m, h, e), 1.0 if r == "win" else 0.0)
             for d, m, h, e, r in BT[["d", "m", "h", "edge", "res"]].itertuples(index=False)]


def vs_base(E, jcol):
    m = E["res10"].isin(["win", "loss"])
    g = [(d, (int(j), int(b), e), 1.0 if r == "win" else 0.0)
         for d, j, b, e, r in E.loc[m, ["d", jcol, "b", "edge", "res10"]].itertuples(index=False)]
    if not g:
        return None
    M, _ = CL.strata_matrices(SESS, g, base_rows)
    pe, n_cov, cov = CL.matched_diff(ONE, M)
    be, _, _ = CL.matched_diff(WT, M)
    lo, hi = ci(be)
    return dict(est_pp=100 * float(pe[0]), ci=[100 * lo, 100 * hi], n_covered=float(n_cov[0]), coverage=float(cov[0]))


def grp(E, col, a, b):
    return E[E[col] == a], E[E[col] == b]


# ------------------------------------------------------------------------------------------------ primaries + Holm
p1 = diff(*grp(P, "h110", "CLEAN", "CHOPPY"), draws=True)
p2 = diff(*grp(Q, "cls", "HOLD", "FILL"), draws=True)
cells = {}
for name, p in (("H1", p1), ("H2", p2)):
    cells[name] = dict(p=W.two_sided_p(p["_draws"]), na=p["_na"], nb=p["_nb"], draws=p["_draws"],
                       ci95=p["diff_pp"]["ci"], ran=True)
W.holm(cells)
R["primary"] = {}
for name, p in (("H1", p1), ("H2", p2)):
    c = cells[name]
    p["holm"] = dict(p=c["p"], p_holm=c["p_holm"], rank=c["rank"], level=c["holm_level"],
                     ci_holm_pp=[100 * x for x in c["ci_holm"]], verdict=c["verdict"])
    for k in ("_draws", "_na", "_nb"):
        p.pop(k)
    R["primary"][name] = p
    log("PRIMARY", name, json.dumps(p, default=str))

# ------------------------------------------------------------------------------------------------ H1 readouts
CLEAN, CHOP = grp(P, "h110", "CLEAN", "CHOPPY")
R1: dict = {}
R1["all_first_touches"] = one(P)
R1["middle"] = one(P[P["h110"] == "MIDDLE"])
R1["pm20"] = diff(CLEAN, CHOP, "res20")
R1["mfe"] = mfe_read(CLEAN, CHOP)
R1["vs_random"] = dict(clean=vs_base(CLEAN, "j"), choppy=vs_base(CHOP, "j"), all=vs_base(P, "j"))
R1["within_ct_class"] = {k: diff(*grp(g, "h110", "CLEAN", "CHOPPY")) for k, g in P.groupby("cls10")}
R1["within_speed_tercile"] = {k: diff(*grp(g, "h110", "CLEAN", "CHOPPY")) for k, g in P.groupby("spd_terc")}
R1["retrace_low_vs_high"] = diff(*grp(P, "retr_hl", "low", "high"))
R1["retrace_bands"] = {k: one(g) for k, g in P.groupby("retr_band")}
R1["window5"] = diff(*grp(P, "h15", "CLEAN", "CHOPPY"))
R1["window20"] = diff(*grp(P, "h120", "CLEAN", "CHOPPY"))
R1["a7_leg"] = diff(*grp(P, "h1A7", "CLEAN", "CHOPPY"))
R1["by_family"] = {f: diff(*grp(g, "h110", "CLEAN", "CHOPPY")) for f, g in P.groupby("fam")}
R["readouts_h1"] = R1

# ------------------------------------------------------------------------------------------------ H2 readouts
HOLD, FILL = grp(Q, "cls", "HOLD", "FILL")
R2: dict = {}
R2["all_trades"] = one(Q)
R2["pm20"] = diff(HOLD, FILL, "res20")
R2["mfe"] = mfe_read(HOLD, FILL)
R2["vs_random"] = dict(hold=vs_base(HOLD, "t"), fill=vs_base(FILL, "t"), all=vs_base(Q, "t"))
R2["bottom_quartile"] = diff(*grp(Q, "cls_q", "HOLD", "FILL"))
R2["final_band_excluded"] = diff(*grp(Q, "cls_x", "HOLD", "FILL"))
R2["variants"] = {k: diff(*grp(H2[k], "cls", "HOLD", "FILL")) for k in VARIANTS if k != "primary"}
lt = Q.assign(lt=Q["light"].map({True: "light", False: "heavy"}))
R2["light_heavy"] = {k: diff(*grp(g, "cls", "HOLD", "FILL")) for k, g in lt.groupby("lt")}
R2["by_family"] = {f: diff(*grp(g, "cls", "HOLD", "FILL")) for f, g in Q.groupby("fam")}
R2["by_snap_final_band"] = {("snap = final band" if k else "snap earlier"): diff(*grp(g, "cls", "HOLD", "FILL"))
                            for k, g in Q.groupby(Q["snap"] == Q["nb"] - 1)}
R["readouts_h2"] = R2
for k, v in {**R1, **R2}.items():
    log(k, json.dumps(v, default=str)[:500])

P.drop(columns=[c for c in P.columns if c.startswith("why")]).assign(d=P["d"].astype(str)).to_parquet(
    W.OUT / "run_h1_events.parquet")
Q.drop(columns=["band_vol"]).assign(d=Q["d"].astype(str)).to_parquet(W.OUT / "run_h2_trades.parquet")
OUT.write_text(json.dumps(json.loads(json.dumps(R, default=str)), indent=1))
log("done")
