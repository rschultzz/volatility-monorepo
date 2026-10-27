#!/usr/bin/env python3
"""CR-CM T0 run — strong GEX levels, real vs placebo. READ-ONLY. Single run.

Spec: specs/CR-CM-strong-level-t0.md (M1–M5) + Amendment A1. Harness: scripts/cr_cm_lib.py (A17 code + chart rule).

  - Placebo (A1.2): A16.4 bins also >= 10 pts from every CR-CB-set level and every in-window strong level, for S,
    X100 and W. Readout: the A17 literal placebo (`placebo` without `avoid`).
  - Hypotheses (A1.3): H_call = strong calls on train; H_put = strong puts on train + R-a pooled (EXPLORATORY).
    Statistic = M2 matched (Σ_s w_s (R_s − P_s), s ∈ {EDGE, INTERIOR}), restricted-mean time to escape 0.25·IM.
    Bootstrap per scope: 10,000 session-multinomial resamples, default_rng(20260920), over that scope's sessions
    (train 89, R-a 147, pooled 236 = R-a then train). One-sided p = share of defined resamples <= 0 (floor 1e-4);
    undefined resamples (a stratum with real tests and no placebo session) are dropped and counted.
    Holm over {H_call, H_put}, one-sided α 0.025. Call confirmation: calls on R-a, p <= 0.025 and lo > 0.
  - Verdicts (B1): pass = Holm-rejected and lo > 0; refuted = hi < 0; else inconclusive (also unmatchable).
  - Readouts (no gate), every scope: unmatched, EDGE-only, INTERIOR-only, approach cells, all-type pooled, dwell,
    excl. IM < 25, literal placebo, X100, W, S-first, and S − W on real tests (session-weighted, unmatched).

Usage: /Users/ryan/code/volatility-monorepo/apps/web/.venv/bin/python -u scripts/cr_cm_t0.py
"""
from __future__ import annotations

import json
import signal

import numpy as np
import pandas as pd

import cr_cm_lib as C

signal.alarm(4 * 3600)

DS = {n: C.load(n) for n in ("ra", "train")}


def strict_plv(D, k):
    out = {}
    for d_ in D.USABLE:
        avoid = [x["L"] for x in D.SETS["CB"][d_]] + [x["L"] for x in D.SETS["S"][d_]]
        out[d_] = C.placebo(D.SETS[k][d_], D.OPEN[d_], D.IM[d_], avoid=avoid)
    return out


# tests + measures per dataset, set and placebo rule
T = {}
for n, D in DS.items():
    for k in ("S", "X100", "W"):
        R = C.add_measures(D, C.tests(D, D.SETS[k]))
        Pst = C.add_measures(D, C.tests(D, strict_plv(D, k)))
        Plit = C.add_measures(D, C.tests(D, D.PLV[k]))
        for E in (R, Pst, Plit):
            E["ds"] = n
        T[(n, k)] = dict(real=R, strict=Pst, literal=Plit)


class Scope:
    pass


def scope(names):
    s = Scope()
    s.USABLE = [d_ for n in names for d_ in DS[n].USABLE]
    s.SIDX = {d_: i for i, d_ in enumerate(s.USABLE)}
    s.nS = len(s.USABLE)
    s.IM = {d_: DS[n].IM[d_] for n in names for d_ in DS[n].USABLE}
    s.names = names
    s.W = C.boot_weights(s.nS)
    return s


SC = {"train": scope(["train"]), "ra": scope(["ra"]), "pooled": scope(["ra", "train"])}


def frames(sc, k, arm):
    return pd.concat([T[(n, k)][arm] for n in sc.names], ignore_index=True)


def compare(sc, Er, Ep, m="esc", matched=True):
    one = np.ones((1, sc.nS))
    fn = C.matched_diff if matched else C.unmatched_diff
    pt_ = float(fn(sc, one, Er, Ep, m)[0])
    bd = fn(sc, sc.W, Er, Ep, m)
    ok = np.isfinite(bd)
    out = dict(n_real_tests=int(len(Er)), n_placebo_tests=int(len(Ep)),
               n_real_sessions=int(Er["d"].nunique()) if len(Er) else 0,
               n_placebo_sessions=int(Ep["d"].nunique()) if len(Ep) else 0,
               edge_share_real=round(float(Er["edge"].mean()), 3) if len(Er) else None,
               edge_share_placebo=round(float(Ep["edge"].mean()), 3) if len(Ep) else None,
               resamples_undefined=int((~ok).sum()))
    mr, pr, _ = C.sess_arrays(sc, Er, m)
    mp, pp, _ = C.sess_arrays(sc, Ep, m)
    out.update(real=round(float(C.wmean(one, mr, pr)[0]), 3), placebo=round(float(C.wmean(one, mp, pp)[0]), 3))
    if not len(Er) or not len(Ep) or not np.isfinite(pt_) or ok.sum() < 100:
        out.update(diff=None, lo=None, hi=None, p_one_sided=None, unmatchable=True)
        return out
    lo, hi = np.percentile(bd[ok], [2.5, 97.5])
    out.update(diff=round(pt_, 3), lo=round(float(lo), 3), hi=round(float(hi), 3),
               p_one_sided=max(float(np.mean(bd[ok] <= 0)), 1.0 / C.N_BOOT), unmatchable=False)
    return out


TYPES = {"put": lambda E: E[E["type"] == "put"], "call": lambda E: E[E["type"] == "call"], "all": lambda E: E}
CELLS = {f"{ty}/{'below' if s == 1 else 'above'}": (lambda E, ty=ty, s=s: E[(E["type"] == ty) & (E["side"] == s)])
         for ty, s in C.CELLS}


def block(sc, k, arm="strict", filt=lambda E: E):
    R, P = filt(frames(sc, k, "real")), filt(frames(sc, k, arm))
    b = {}
    for h, f in {**TYPES, **CELLS}.items():
        r, p = f(R), f(P)
        b[h] = dict(primary=compare(sc, r, p), unmatched=compare(sc, r, p, matched=False),
                    edge_only=compare(sc, r[r["edge"]], p[p["edge"]], matched=False),
                    interior_only=compare(sc, r[~r["edge"]], p[~p["edge"]], matched=False),
                    dwell=compare(sc, r, p, m="dwell"))
    return b


def verdict(r, rejected):
    if r.get("unmatchable"):
        return "inconclusive (unmatchable)"
    if rejected and r["lo"] > 0:
        return "pass"
    return "refuted" if r["hi"] < 0 else "inconclusive"


RES = dict(sessions={k: dict(n=s.nS, first=str(s.USABLE[0]), last=str(s.USABLE[-1])) for k, s in SC.items()})

# placebo counts per cell (A1.2), both rules
RES["placebo_counts"] = {}
for n in ("train", "ra"):
    for k in ("S", "X100", "W"):
        RES["placebo_counts"][f"{n}/{k}"] = {
            arm: {h: dict(tested_levels=int(f(T[(n, k)][arm]).groupby(["d", "L"]).ngroups) if len(f(T[(n, k)][arm])) else 0,
                          tests=int(len(f(T[(n, k)][arm]))),
                          sessions=int(f(T[(n, k)][arm])["d"].nunique()) if len(f(T[(n, k)][arm])) else 0)
                  for h, f in {**TYPES, **CELLS}.items()}
            for arm in ("real", "strict", "literal")}
        RES["placebo_counts"][f"{n}/{k}"]["placebo_levels_strict"] = int(sum(len(v) for v in strict_plv(DS[n], k).values()))
        RES["placebo_counts"][f"{n}/{k}"]["placebo_levels_literal"] = int(sum(len(v) for v in DS[n].PLV[k].values()))

# ------------------------------------------------------------------------------------------------ primary
S_train, S_pool, S_ra = block(SC["train"], "S"), block(SC["pooled"], "S"), block(SC["ra"], "S")
H = {"H_call (train)": S_train["call"]["primary"], "H_put (train + R-a, EXPLORATORY)": S_pool["put"]["primary"]}
order = sorted(H, key=lambda k: (H[k]["p_one_sided"] is None, H[k]["p_one_sided"] or 1.0))
stop = False
for i, k in enumerate(order):
    thr = C.ALPHA / (len(order) - i)
    p = H[k]["p_one_sided"]
    rej = (not stop) and p is not None and p <= thr
    stop = stop or not rej
    H[k].update(holm_threshold=thr, holm_rejected=bool(rej), verdict=verdict(H[k], rej))
conf = dict(S_ra["call"]["primary"])
conf_rej = conf.get("p_one_sided") is not None and conf["p_one_sided"] <= C.ALPHA
conf.update(rejected=bool(conf_rej), verdict=verdict(conf, conf_rej))
call_v = H["H_call (train)"]["verdict"]
RES["primary"] = dict(holm=H, call_confirmation_ra=conf,
                      call_overall=("confirmed" if call_v == "pass" and conf["verdict"] == "pass"
                                    else "train-only" if call_v == "pass" else call_v),
                      put_overall=H["H_put (train + R-a, EXPLORATORY)"]["verdict"] + " (EXPLORATORY)")

# ---------------------------------------------------------------------------------------------- readouts
RES["S"] = {"train": S_train, "ra": S_ra, "pooled": S_pool}
RES["S_literal_placebo"] = {k: block(s, "S", arm="literal") for k, s in SC.items()}
RES["S_first_touch"] = {k: block(s, "S", filt=lambda E: E[E["first_touch"]]) for k, s in SC.items()}


def no_low_im(sc):
    return lambda E: E[[sc.IM[d_] >= C.IM_FLAG for d_ in E["d"]]] if len(E) else E


RES["S_excl_low_im"] = {k: block(s, "S", filt=no_low_im(s)) for k, s in SC.items()}
RES["X100"] = {k: block(s, "X100") for k, s in SC.items()}
RES["W"] = {k: block(s, "W") for k, s in SC.items()}
RES["S_minus_W_real"] = {k: {h: compare(s, f(frames(s, "S", "real")), f(frames(s, "W", "real")), matched=False)
                             for h, f in TYPES.items()} for k, s in SC.items()}

C.OUT_DIR.mkdir(parents=True, exist_ok=True)
(C.OUT_DIR / "t0.json").write_text(json.dumps(RES, indent=1, default=str))


def line(lbl, v):
    if v.get("diff") is None:
        return f"  {lbl:34s} n {v['n_real_tests']}/{v['n_placebo_tests']}  unmatchable"
    return (f"  {lbl:34s} n {v['n_real_tests']}/{v['n_placebo_tests']} sess {v['n_real_sessions']}/"
            f"{v['n_placebo_sessions']}  {v['real']:.2f} vs {v['placebo']:.2f}  diff {v['diff']:+.2f} "
            f"[{v['lo']:+.2f}, {v['hi']:+.2f}]  p {v['p_one_sided']:.4f}  edge {v['edge_share_real']}/"
            f"{v['edge_share_placebo']}")


print("PRIMARY (matched, time to escape, min; real − placebo)")
for k, v in RES["primary"]["holm"].items():
    print(line(k, v), f" Holm thr {v['holm_threshold']:.4f} -> {v['verdict']}")
print(line("call confirmation R-a", conf), f" -> {conf['verdict']}")
print("call overall:", RES["primary"]["call_overall"], "| put overall:", RES["primary"]["put_overall"])
for title, blk in (("S", RES["S"]), ("S literal placebo", RES["S_literal_placebo"]), ("S first touch", RES["S_first_touch"]),
                   ("S excl IM<25", RES["S_excl_low_im"]), ("X100", RES["X100"]), ("W", RES["W"])):
    print(f"\n{title}")
    for sc_, b in blk.items():
        for h in ("put", "call", "all"):
            print(line(f"{sc_} {h}", b[h]["primary"]))
print("\nS − W on real tests (unmatched)")
for sc_, b in RES["S_minus_W_real"].items():
    for h, v in b.items():
        print(line(f"{sc_} {h}", v))
