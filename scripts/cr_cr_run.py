#!/usr/bin/env python3
"""CR-CR run — the single outcome run on R-a (spec E1-E10). READ-ONLY, R-a only. Run once.

Primary (CR-CP C11 with the effort rule, E5): on ranges where the contracts/pt rule calls a side, race win rate (±10,
C10) of the chosen side minus the opposite side, each over its own resolved fills; session-cluster bootstrap 10,000
draws, seed 20260920, percentile 95 % CI; B1 verdict; >= 40 resolved chosen-side trades else underpowered. One
hypothesis, no Holm. Readouts (E6, no verdict): each side vs 50 %, true pairs, C13 matched random baseline, ratio
1.3 / 2.0, swing fixed 12, C14 level split, call side, decision hour, G1-down flag; ±20 race and MFE (A11).
Helpers (`outcomes` .. `mean_ci`) copied unchanged from scripts/cr_cp_run.py. Writes scripts/logs/cr_cr/results.json.
"""
from __future__ import annotations

import json
import time as _time
from collections import Counter

import numpy as np
import pandas as pd

import cr_cr_lib as K
import cr_cp_lib as C
import cr_cl_lib as CL

assert C.PERIOD == "ra" and CL.PERIOD == "ra"
T0 = _time.time()
OUT = C.OUT / "results.json"
assert not OUT.exists(), "CR-CR is run once; results.json already exists"
S0J = json.loads((C.OUT / "step0.json").read_text())
assert S0J["gate"]["passed"], S0J["gate"]


def log(*a):
    print(f"[{_time.time() - T0:7.1f}s]", *a, flush=True)


m1, im = C.load_minutes(), C.load_im()
D = C.build(m1, im)
SESS = D["SESS"]
SI = {d: i for i, d in enumerate(SESS)}
nS = len(SESS)
assert nS == S0J["n_sessions"] and D["MED_IM"] == C.IM_TRAIN_CR_CO
RNG = np.random.default_rng(C.SEED)
WT = CL.boot_weights(nS, C.N_BOOT, RNG)
ONE = np.ones((1, nS))
R: dict = dict(n_sessions=nS, n_boot=C.N_BOOT, seed=C.SEED, period="ra", primary=None)


# ------------------------------------------------------------------------------------------------ helpers
def outcomes(S: pd.DataFrame, p: C.P, X: float):
    """Adds {ch,op}_res (win / loss / tie / unresolved / gap / None) for each filled side."""
    S = S.copy()
    for nm, sg in (("ch", 1), ("op", -1)):
        res = []
        for r in S.itertuples():
            st = getattr(r, f"{nm}_status")
            if st != "fill":
                res.append(None)
                continue
            j, px, h = int(getattr(r, f"{nm}_j")), float(getattr(r, f"{nm}_px")), sg * int(r.side)
            res.append(C.race_from(D["BARS"][r.d], j, px, h, X, fill_bar=(p.entry == "edge")))
        S[f"{nm}_res"] = res
    return S


def sess_counts(rows):
    """rows: iterable of (d, w). Returns (n, w) per session arrays."""
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
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if a.size else [None, None], int(a.size)


def verdict(lo, hi, n):
    if n < C.MIN_TRADES:
        return "UNDERPOWERED"
    return "PASS" if lo > 0 else ("REFUTED" if hi < 0 else "INCONCLUSIVE")


def resolved(S, nm):
    m = S[f"{nm}_res"].isin(["win", "loss"])
    return S[m], [(d, 1.0 if x == "win" else 0.0) for d, x in zip(S.loc[m, "d"], S.loc[m, f"{nm}_res"])]


def compare(S, with_verdict=False, single=False):
    """Chosen − opposite (each over its own resolved fills) + each vs 50 % + true pairs."""
    _, rc = resolved(S, "ch")
    _, ro = resolved(S, "op")
    nc, wc = sess_counts(rc)
    no, wo = sess_counts(ro)
    out = dict(chosen_resolved=int(nc.sum()), opposite_resolved=int(no.sum()),
               chosen_sessions=int((nc > 0).sum()),
               chosen_res=dict(Counter(S["ch_res"].dropna())), opposite_res=dict(Counter(S["op_res"].dropna())))
    if nc.sum() == 0:
        return out
    pc, bc = rate(ONE, nc, wc)[0], rate(WT, nc, wc)
    out["chosen_wr"] = float(pc)
    (lo, hi), nb = ci(bc - 0.5)
    out["chosen_minus_50_pp"] = dict(est=100 * (pc - 0.5), ci=[100 * lo, 100 * hi], draws=nb)
    if single or no.sum() == 0:
        return out
    po, bo = rate(ONE, no, wo)[0], rate(WT, no, wo)
    out["opposite_wr"] = float(po)
    (lo2, hi2), _ = ci(bo - 0.5)
    out["opposite_minus_50_pp"] = dict(est=100 * (po - 0.5), ci=[100 * lo2, 100 * hi2])
    (dlo, dhi), nb = ci(bc - bo)
    out["diff_pp"] = dict(est=100 * (pc - po), ci=[100 * dlo, 100 * dhi], draws=nb)
    if with_verdict:
        out["verdict"] = verdict(dlo, dhi, int(nc.sum()))
    pr = S[S["ch_res"].isin(["win", "loss"]) & S["op_res"].isin(["win", "loss"])]
    if len(pr):
        n_, d_ = np.zeros(nS), np.zeros(nS)
        for r in pr.itertuples():
            n_[SI[r.d]] += 1
            d_[SI[r.d]] += (r.ch_res == "win") - (r.op_res == "win")
        pe, be = rate(ONE, n_, d_)[0], rate(WT, n_, d_)
        (plo, phi), _ = ci(be)
        out["pairs_diff_pp"] = dict(n=int(len(pr)), est=100 * pe, ci=[100 * plo, 100 * phi])
    return out


def side_rows(S, nm, fn):
    """Per filled side: (d, value) of fn(B, j, px, h)."""
    sg = 1 if nm == "ch" else -1
    out = []
    for r in S[S[f"{nm}_status"] == "fill"].itertuples():
        out.append((r.d, fn(D["BARS"][r.d], int(getattr(r, f"{nm}_j")), float(getattr(r, f"{nm}_px")),
                         sg * int(r.side))))
    return out


def mean_ci(rows):
    rows = [(d, v) for d, v in rows if v is not None and np.isfinite(v)]
    n, s = sess_counts(rows)
    if n.sum() == 0:
        return None
    (lo, hi), _ = ci(rate(WT, n, s))
    return dict(n=int(n.sum()), mean=float(rate(ONE, n, s)[0]), ci=[lo, hi])


# ------------------------------------------------------------------------------------------------ primary
P = K.PRIMARY
log("primary schedule + outcomes")
S0 = C.schedule(D, P)
st = S0J["schedules"]["primary"]
assert len(S0) == st["n_ranges"] and int((S0["ch_status"] == "fill").sum()) == st["ch_fill"]["fill"], \
    "schedule differs from Step 0"
S = outcomes(S0[S0["dstatus"] == "call"], P, 10.0)
R["primary"] = compare(S, with_verdict=True)
log("PRIMARY", json.dumps(R["primary"]))

# C13 matched random baseline
log("random baseline (C13)")
BT = C.baseline_table(D)
BT = BT[BT["res"].isin(["win", "loss"])]
base_rows = [(d, (int(m), int(h), e), 1.0 if x == "win" else 0.0)
             for d, m, h, e, x in zip(BT["d"], BT["m"], BT["h"], BT["edge"], BT["res"])]
R["vs_random"] = {}
for nm, sg in (("ch", 1), ("op", -1)):
    rr = S[S[f"{nm}_res"].isin(["win", "loss"])]
    grp_rows = [(r.d, (int(getattr(r, f"{nm}_j")), sg * int(r.side), getattr(r, f"{nm}_edge")),
                 1.0 if getattr(r, f"{nm}_res") == "win" else 0.0) for r in rr.itertuples()]
    M, _ = CL.strata_matrices(SESS, grp_rows, base_rows)
    pe, ncov, cov = CL.matched_diff(ONE, M)
    be, _, _ = CL.matched_diff(WT, M)
    (lo, hi), nb = ci(be)
    R["vs_random"][nm] = dict(est_pp=100 * float(pe[0]), ci_pp=[100 * lo, 100 * hi], covered=int(ncov[0]),
                              coverage=float(cov[0]), draws=nb)
log("vs random", R["vs_random"])

# A11 readouts: ±20 race, max favourable before −10
log("readouts: ±20, MFE")
S20 = outcomes(S0[S0["dstatus"] == "call"], P, 20.0)
R["race20"] = compare(S20)
R["mfe_before_minus10"] = {}
for nm in ("ch", "op"):
    v = [x for _, x in side_rows(S, nm, lambda B, j, px, h: CL.mfe_before(B, j, px, h, 10.0))]
    R["mfe_before_minus10"][nm] = dict(n=len(v), p25=float(np.percentile(v, 25)), median=float(np.median(v)),
                                       p75=float(np.percentile(v, 75)), share_ge10=float(np.mean(np.array(v) >= 10)),
                                       share_ge20=float(np.mean(np.array(v) >= 20))) if v else None
log("race20", R["race20"].get("diff_pp"), "mfe", R["mfe_before_minus10"])

# ------------------------------------------------------------------------------------------------ splits
log("splits")
LV = C.load_levels()
T = C.tag_skew(C.tag_levels(S, LV))
T["hour"] = (390 + T["t"]) // 60
T["call_side"] = np.where(T["side"] == 1, "long", "short")
R["splits"] = {}
for col in ("lv_strength", "lv_pos", "lv_kind", "call_side", "hour", "g1down_like"):
    R["splits"][col] = {}
    for k, g in T.groupby(col):
        c = compare(g)
        n_ch = c.get("chosen_resolved", 0)
        tag = "descriptive (< 40)" if n_ch < C.MIN_TRADES else "readout"
        if col == "lv_kind" and k in ("put", "call") and n_ch < C.MIN_TRADES:
            tag = "not readable on R-a (< 40)"
        R["splits"][col][str(k)] = dict(status=tag, **c)
T.to_parquet(C.OUT / "run_primary_trades.parquet")

# ------------------------------------------------------------------------------------------------ grid
log("readout grid")
R["grid"] = {}
for nm, p in K.VARIANTS.items():
    if nm == "primary":
        continue
    Sv = C.schedule(D, p)
    Sv = outcomes(Sv[Sv["dstatus"] == "call"], p, p.X)
    R["grid"][nm] = compare(Sv)
    g = R["grid"][nm]
    log(f"  {nm:16s}", g.get("chosen_resolved"), g.get("diff_pp"))

R["elapsed_s"] = _time.time() - T0
OUT.write_text(json.dumps(R, indent=1, default=str))
log("wrote", OUT)
