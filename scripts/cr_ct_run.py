#!/usr/bin/env python3
"""CR-CT run — the single outcome run on train (spec C10-C15). READ-ONLY, train only. Run once.

Primary (C11): on pooled first touches (F1, F2, RUN; C5), break win rate of FAST minus SLOW, each over its own
resolved ±10 races (C10, entry at the level at the touch second = the R2 level-entry variant); session-cluster
bootstrap 10,000 draws, seed 20260920, percentile 95 % CI; B1 verdict; >= 40 resolved in each cell else
UNDERPOWERED. One hypothesis, no Holm. Readouts (no verdict, unadjusted): FAST vs 50 % (break), SLOW vs 50 % (fade);
±20; move at +30 / +60; MFE before -10; matched random baseline (C12); speed alone; volume alone; windows 5 / 20;
before she goes (C14); family; time of day; second touch; strong line within 3 pts and A8 put / call; A7 leg.
Reads the tercile cuts from step0.json (C9) and asserts the event set is unchanged. Writes logs/cr_ct/results.json.
"""
from __future__ import annotations

import json
import time as _time
from collections import Counter

import numpy as np
import pandas as pd

import cr_ct_lib as T
import cr_cp_lib as C
import cr_cl_lib as CL

T0 = _time.time()
OUT = T.OUT / "results.json"
assert not OUT.exists(), "CR-CT is run once; results.json already exists"
S0J = json.loads((T.OUT / "step0.json").read_text())
assert S0J["gate"]["passed"], S0J["gate"]


def log(*a):
    print(f"[{_time.time() - T0:7.1f}s]", *a, flush=True)


D = T.build()
SESS = D["SESS"]
assert len(SESS) == S0J["n_sessions"]
SI = {d: i for i, d in enumerate(SESS)}
nS = len(SESS)
S1 = T.load_1s(D)
ev, _ = T.events(D, S1)
ev["no1s"] = ev["k1s"].isna()
P = ev[(ev["touch"] == 1) & ~ev["dup"] & ~ev["no1s"]].copy()
S2 = ev[(ev["touch"] == 2) & ~ev["dup"] & ~ev["no1s"]].copy()
assert len(P) == S0J["primary_events"]["n"]
CUTS = {w: v["cuts"] for w, v in S0J["cuts"].items()}
for w in CUTS:
    c, n = T.cuts(P, f"speed{w}", f"relvol{w}")
    assert np.allclose(c, CUTS[w]) and n == S0J["cuts"][w]["n"], (w, c, CUTS[w])
    P[f"cls{w}"] = T.classify(P, f"speed{w}", f"relvol{w}", CUTS[w])
    S2[f"cls{w}"] = T.classify(S2, f"speed{w}", f"relvol{w}", CUTS[w])
P = T.tag_strong(P, C.load_levels())

RNG = np.random.default_rng(T.SEED)
WT = CL.boot_weights(nS, T.N_BOOT, RNG)
ONE = np.ones((1, nS))
R: dict = dict(n_sessions=nS, n_boot=T.N_BOOT, seed=T.SEED, entry="R2 level price at the touch second (labelled)")


def outcomes(E):
    E = E.copy()
    r10, r20, mfe, m30, m60 = [], [], [], [], []
    for r in E.itertuples():
        s, k = S1[r.d], int(r.k1s)
        r10.append(T.race1s(s, k, r.L, r.b, T.X1))
        r20.append(T.race1s(s, k, r.L, r.b, T.X2))
        mfe.append(T.mfe1s(s, k, r.L, r.b))
        m30.append(T.move_at(s, k, r.L, r.b, 30))
        m60.append(T.move_at(s, k, r.L, r.b, 60))
    E["res10"], E["res20"], E["mfe"], E["mv30"], E["mv60"] = r10, r20, mfe, m30, m60
    return E


log("outcomes")
P = outcomes(P)
S2 = outcomes(S2)


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


def verdict(lo, hi, na, nb):
    if min(na, nb) < T.MIN_CELL:
        return "UNDERPOWERED"
    return "PASS" if lo > 0 else ("REFUTED" if hi < 0 else "INCONCLUSIVE")


def win_rows(E, col="res10"):
    m = E[col].isin(["win", "loss"])
    return [(d, 1.0 if x == "win" else 0.0) for d, x in zip(E.loc[m, "d"], E.loc[m, col])]


def one(E, col="res10"):
    n, w = sc(win_rows(E, col))
    out = dict(n=int(len(E)), resolved=int(n.sum()), sessions=int((n > 0).sum()), res=dict(Counter(E[col])))
    if n.sum():
        p, b = rate(ONE, n, w)[0], rate(WT, n, w)
        lo, hi = ci(b - 0.5)
        out.update(break_wr=float(p), vs50_pp=dict(est=100 * (p - .5), ci=[100 * lo, 100 * hi]))
    return out


def diff(A, B, col="res10", with_verdict=False):
    na, wa = sc(win_rows(A, col))
    nb, wb = sc(win_rows(B, col))
    out = dict(a=one(A, col), b=one(B, col))
    if na.sum() and nb.sum():
        pa, pb = rate(ONE, na, wa)[0], rate(ONE, nb, wb)[0]
        lo, hi = ci(rate(WT, na, wa) - rate(WT, nb, wb))
        out["diff_pp"] = dict(est=100 * (pa - pb), ci=[100 * lo, 100 * hi])
        if with_verdict:
            out["verdict"] = verdict(lo, hi, int(na.sum()), int(nb.sum()))
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


def fs(E, col="cls10"):
    return E[E[col] == "FAST"], E[E[col] == "SLOW"]


# ------------------------------------------------------------------------------------------------ primary
F, S = fs(P)
R["primary"] = diff(F, S, with_verdict=True)
R["primary"]["slow_fade_wr"] = 1 - R["primary"]["b"].get("break_wr", np.nan)
v = R["primary"]["b"].get("vs50_pp")
R["primary"]["slow_fade_vs50_pp"] = dict(est=-v["est"], ci=[-v["ci"][1], -v["ci"][0]]) if v else None
log("PRIMARY", json.dumps(R["primary"], default=str))

# ------------------------------------------------------------------------------------------------ readouts
RD: dict = {}
RD["all_first_touches"] = one(P)
RD["neither"] = one(P[P["cls10"] == "NEITHER"])
RD["pm20"] = diff(F, S, col="res20")
RD["move30"] = mean_diff(F, S, "mv30")
RD["move60"] = mean_diff(F, S, "mv60")
RD["mfe"] = dict(mean=mean_diff(F, S, "mfe"), fast_median=float(F["mfe"].median()), slow_median=float(S["mfe"].median()),
                 fast_q=[float(x) for x in F["mfe"].quantile([.25, .5, .75])],
                 slow_q=[float(x) for x in S["mfe"].quantile([.25, .5, .75])],
                 fast_reach10=float((F["mfe"] >= 10).mean()), slow_reach10=float((S["mfe"] >= 10).mean()))

BT = T.baseline_table(D)
BT = BT[BT["res"].isin(["win", "loss"])]
base_rows = [(d, (m, h, e), 1.0 if r == "win" else 0.0) for d, m, h, e, r in BT[["d", "m", "h", "edge", "res"]].itertuples(index=False)]


def vs_base(E):
    m = E["res10"].isin(["win", "loss"])
    g = [(d, (int(j), int(b), e), 1.0 if r == "win" else 0.0)
         for d, j, b, e, r in E.loc[m, ["d", "j", "b", "edge", "res10"]].itertuples(index=False)]
    if not g:
        return None
    M, _ = CL.strata_matrices(SESS, g, base_rows)
    pe, n_cov, cov = CL.matched_diff(ONE, M)
    be, _, _ = CL.matched_diff(WT, M)
    lo, hi = ci(be)
    return dict(est_pp=100 * float(pe[0]), ci=[100 * lo, 100 * hi], n_covered=float(n_cov[0]), coverage=float(cov[0]))


RD["vs_random"] = dict(fast=vs_base(F), slow=vs_base(S), all=vs_base(P))
c10 = CUTS["10"]
RD["speed_alone"] = diff(P[P["speed10"] >= c10[1]], P[P["speed10"] <= c10[0]])
RD["volume_alone"] = diff(P[P["relvol10"] >= 1], P[P["relvol10"] < 1])
RD["window5"] = diff(*fs(P, "cls5"))
RD["window20"] = diff(*fs(P, "cls20"))
RD["a7_leg"] = diff(*fs(P, "clsA7"))
RD["before_she_goes"] = diff(P[P["bsg"]], P[~P["bsg"]])
RD["by_family"] = {f: diff(*fs(g)) for f, g in P.groupby("fam")}
RD["by_kind"] = {k: one(g) for k, g in P.groupby("kind")}
RD["by_b"] = {str(b): diff(*fs(g)) for b, g in P.groupby("b")}
RD["first_hour"] = {("first hour" if h else "later"): diff(*fs(g)) for h, g in P.groupby("first_hour")}
RD["second_touch"] = dict(fast_slow=diff(*fs(S2)), all=one(S2))
RD["strong"] = dict(strong_vs_none=diff(P[P["strong"] == "strong"], P[P["strong"] == "none"]),
                    within_strong=diff(*fs(P[P["strong"] == "strong"])))
RD["a8"] = dict(put_vs_call=diff(P[P["a8"] == "put"], P[P["a8"] == "call"]),
                put=diff(*fs(P[P["a8"] == "put"])), call=diff(*fs(P[P["a8"] == "call"])))
RD["edge"] = {e: diff(*fs(g)) for e, g in P.groupby("edge")}
R["readouts"] = RD
for k, v in RD.items():
    log(k, json.dumps(v, default=str)[:600])

keep = [c for c in P.columns if c not in ("why5", "why20", "whyA7")]
P[keep].assign(d=P["d"].astype(str)).to_parquet(T.OUT / "run_primary_events.parquet")
S2.assign(d=S2["d"].astype(str)).to_parquet(T.OUT / "run_second_touch_events.parquet")
OUT.write_text(json.dumps(json.loads(json.dumps(R, default=str)), indent=1))
log("done")
