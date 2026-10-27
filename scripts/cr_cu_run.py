#!/usr/bin/env python3
"""CR-CU run — the single outcome run (spec U7, U8, U10). READ-ONLY, 2023-05-01 .. 2024-04-02 only. Run once.

Primary (U7): on deduped second touches (F1, F2, RUN), break win rate of SLOW minus FAST on CR-CT's frozen 10-min cuts,
each over its own resolved ±10 races (C1: entry at the level at the touch bar = the R2 level-entry variant, raced on
ES.c.0 1-min bars from the next bar); session-cluster bootstrap 10,000 draws, seed 20260920, percentile 95 % CI; B1
verdict; >= 40 resolved in each cell else UNDERPOWERED. One hypothesis, no Holm. Readouts (no verdict, unadjusted, U8):
first touches FAST - SLOW; each second-touch cell vs 50 %; all second touches; ±20; family; windows 5 / 20; matched
random baseline; MFE before -10; bar-j excursions; without expiry-week Mon / Tue.
Asserts the Step 0 gate, session count, event set and cuts. Writes logs/cr_cu/results.json.
"""
from __future__ import annotations

import json
import time as _time
from collections import Counter

import numpy as np

import cr_cu_lib as U
import cr_ct_lib as T
import cr_cl_lib as CL

T0 = _time.time()
OUT = U.OUT / "results.json"
assert not OUT.exists(), "CR-CU is run once; results.json already exists"
S0J = json.loads((U.OUT / "step0.json").read_text())
assert S0J["gate"]["passed"], S0J["gate"]
assert S0J["cuts"] == U.CUTS
U.check_frozen()


def log(*a):
    print(f"[{_time.time() - T0:7.1f}s]", *a, flush=True)


D = U.build()
SESS = D["SESS"]
assert len(SESS) == S0J["n_sessions"]
SI = {d: i for i, d in enumerate(SESS)}
nS = len(SESS)
ev, _ = U.events(D)
P = ev[(ev["touch"] == 1) & ~ev["dup"]].copy()
S2 = ev[(ev["touch"] == 2) & ~ev["dup"]].copy()
assert len(S2) == S0J["second_events"]["n"] and len(P) == S0J["first_events"]["n"]
for w in U.CUTS:
    got = {k: v["n"] for k, v in S2.groupby(f"cls{w}").size().rename("n").to_frame().to_dict("index").items()}
    assert got == {k: v["n"] for k, v in S0J["second_classes"][w].items()}, (w, got)

RNG = np.random.default_rng(U.SEED)
WT = CL.boot_weights(nS, U.N_BOOT, RNG)
ONE = np.ones((1, nS))
R: dict = dict(n_sessions=nS, n_boot=U.N_BOOT, seed=U.SEED,
               entry="R2 level price at the touch bar (labelled); race on ES.c.0 1-min bars from the next bar (C1)")


def outcomes(E):
    E = E.copy()
    r10, r20, mfe, ib = [], [], [], []
    for r in E.itertuples():
        B = D["BARS"][r.d]
        r10.append(U.race1m(B, r.j, r.L, r.b, U.X1))
        r20.append(U.race1m(B, r.j, r.L, r.b, U.X2))
        mfe.append(U.mfe1m(B, r.j, r.L, r.b))
        ib.append(U.intrabar(B, r.j, r.L, r.b))
    E["res10"], E["res20"], E["mfe"], E["bar_j"] = r10, r20, mfe, ib
    return E


log("outcomes")
S2 = outcomes(S2)
P = outcomes(P)


# ------------------------------------------------------------------------------------------------ helpers (CR-CT run)
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
    if min(na, nb) < U.MIN_CELL:
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


def sf(E, col="cls10"):
    """(SLOW, FAST): the CR-CU primary sign."""
    return E[E[col] == "SLOW"], E[E[col] == "FAST"]


# ------------------------------------------------------------------------------------------------ primary
S, F = sf(S2)
R["primary"] = diff(S, F, with_verdict=True)
log("PRIMARY (SLOW - FAST, second touches)", json.dumps(R["primary"], default=str))

# ------------------------------------------------------------------------------------------------ readouts
RD: dict = {}
RD["second_fast"] = one(F)
RD["second_slow"] = one(S)
RD["second_all"] = one(S2)
RD["second_neither"] = one(S2[S2["cls10"] == "NEITHER"])
RD["first_touch_fast_minus_slow"] = diff(P[P["cls10"] == "FAST"], P[P["cls10"] == "SLOW"])
RD["first_all"] = one(P)
RD["pm20"] = diff(S, F, col="res20")
RD["mfe"] = dict(mean=mean_diff(S, F, "mfe"), slow_q=[float(x) for x in S["mfe"].quantile([.25, .5, .75])],
                 fast_q=[float(x) for x in F["mfe"].quantile([.25, .5, .75])],
                 slow_reach10=float((S["mfe"] >= 10).mean()), fast_reach10=float((F["mfe"] >= 10).mean()))
RD["by_family"] = {f: diff(*sf(g)) for f, g in S2.groupby("fam")}
RD["window5"] = diff(*sf(S2, "cls5"))
RD["window20"] = diff(*sf(S2, "cls20"))
RD["no_expiry_mon_tue"] = diff(*sf(S2[~S2["mon_tue_expiry"]]))
RD["by_b"] = {str(b): diff(*sf(g)) for b, g in S2.groupby("b")}
RD["edge"] = {e: diff(*sf(g)) for e, g in S2.groupby("edge")}
RD["bar_j"] = {c: S2.loc[S2["cls10"] == c, "bar_j"].value_counts().to_dict() for c in ("FAST", "SLOW", "NEITHER")}

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


RD["vs_random"] = dict(second_fast=vs_base(F), second_slow=vs_base(S), second_all=vs_base(S2))
R["readouts"] = RD
for k, v in RD.items():
    log(k, json.dumps(v, default=str)[:700])

S2.assign(d=S2["d"].astype(str)).to_parquet(U.OUT / "run_second_touch_events.parquet")
P.assign(d=P["d"].astype(str)).to_parquet(U.OUT / "run_first_touch_events.parquet")
OUT.write_text(json.dumps(json.loads(json.dumps(R, default=str)), indent=1))
log("done")
