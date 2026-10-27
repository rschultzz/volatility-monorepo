#!/usr/bin/env python3
"""CR-CM Step 0 — strong levels, touches per cell, placebo coverage, A/A. READ-ONLY; no real-level outcome.

Spec: specs/CR-CM-strong-level-t0.md, "Step 0". For train and R-a: the universe funnel; chart lines and colour span
per session; strong levels per session (calls / puts; before and after the ±1·IM window); the strong set under the
full-session band; X100 / W / CR-CB-set counts; S ∩ CR-CB overlap; S levels with a neighbour within 10 pts; tests per
cell and per EDGE / INTERIOR stratum (real and placebo; underpowered flag < 30 tests or < 15 sessions); S-first
counts; placebo coverage; the A/A (spec M5) on S placebo tests (unmatched and matched). Escape is computed on
placebo tests only (A/A). A listing of S levels on three train sessions for Ryan to check against the chart.

Usage: /Users/ryan/code/volatility-monorepo/apps/web/.venv/bin/python -u scripts/cr_cm_step0.py
"""
from __future__ import annotations

import json
import signal

import numpy as np
import pandas as pd

import cr_cm_lib as C

signal.alarm(4 * 3600)
HYP = {"put": lambda E: E[E["type"] == "put"], "call": lambda E: E[E["type"] == "call"], "pooled": lambda E: E}
CELLS = {f"{ty}/{'below' if s == 1 else 'above'}": (lambda E, ty=ty, s=s: E[(E["type"] == ty) & (E["side"] == s)])
         for ty, s in C.CELLS}


def dist(v):
    v = np.asarray(list(v), dtype=float)
    if not v.size:
        return dict(n=0)
    return dict(n=int(v.size), mean=round(float(v.mean()), 3), p25=float(np.percentile(v, 25)),
                p50=float(np.median(v)), p75=float(np.percentile(v, 75)), min=float(v.min()), max=float(v.max()))


def cnt(E):
    return dict(tests=int(len(E)), sessions=int(E["d"].nunique()) if len(E) else 0)


def strata(R, P):
    out = {}
    for s, nm in ((True, "EDGE"), (False, "INTERIOR")):
        r, p = R[R["edge"] == s], P[P["edge"] == s]
        c = dict(real=cnt(r), placebo=cnt(p))
        c["underpowered"] = bool(min(c["real"]["tests"], c["placebo"]["tests"]) < 30
                                 or min(c["real"]["sessions"], c["placebo"]["sessions"]) < 15)
        out[nm] = c
    out["edge_share"] = dict(real=round(float(R["edge"].mean()), 3) if len(R) else None,
                             placebo=round(float(P["edge"].mean()), 3) if len(P) else None)
    return out


def step0(name):
    D = C.load(name)
    U = D.USABLE
    res = dict(funnel=D.funnel, sessions=dict(n=D.nS, first=str(U[0]), last=str(U[-1]),
                                              symbols=sorted({D.SYM[x] for x in U})))
    res["im"] = dist(D.IM.values())
    res["low_im_sessions"] = {str(x): round(D.IM[x], 2) for x in U if D.IM[x] < C.IM_FLAG}

    # chart lines and strong levels
    res["chart"] = dict(
        displayed_per_session=dist(len(D.CHART[x]) for x in U),
        span_bn=dist(D.SPAN[x] / 1e9 for x in U),
        strong_all_per_session=dist(len(D.STRONG_ALL[x]) for x in U),
        strong_all_calls=dist(sum(y["net"] > 0 for y in D.STRONG_ALL[x]) for x in U),
        strong_all_puts=dist(sum(y["net"] < 0 for y in D.STRONG_ALL[x]) for x in U),
        strong_all_net_bn=dict(calls=dist(y["net"] / 1e9 for x in U for y in D.STRONG_ALL[x] if y["net"] > 0),
                               puts=dist(y["net"] / 1e9 for x in U for y in D.STRONG_ALL[x] if y["net"] < 0)),
        strong_outside_window=int(sum(len(D.STRONG_ALL[x]) - len(D.SETS["S"][x]) for x in U)),
    )
    S = D.SETS["S"]
    res["S"] = dict(
        levels=int(sum(len(S[x]) for x in U)),
        calls=int(sum(y["net"] > 0 for x in U for y in S[x])),
        puts=int(sum(y["net"] < 0 for x in U for y in S[x])),
        per_session=dist(len(S[x]) for x in U),
        calls_per_session=dist(sum(y["net"] > 0 for y in S[x]) for x in U),
        puts_per_session=dist(sum(y["net"] < 0 for y in S[x]) for x in U),
        sessions_with_none=int(sum(len(S[x]) == 0 for x in U)),
        sessions_with_call=int(sum(any(y["net"] > 0 for y in S[x]) for x in U)),
        sessions_with_put=int(sum(any(y["net"] < 0 for y in S[x]) for x in U)),
        dist_from_open_over_im=dist(abs(y["L"] - D.OPEN[x]) / D.IM[x] for x in U for y in S[x]),
    )
    # full-session band (chart viewed after the close) vs the band at the open
    full = {x: {round(y["L"], 2) for y in D.CHART_FULL[x] if y["strong"]} for x in U}
    opn = {x: {round(y["L"], 2) for y in D.STRONG_ALL[x]} for x in U}
    res["full_band_check"] = dict(sessions_differing=int(sum(full[x] != opn[x] for x in U)),
                                  only_full=int(sum(len(full[x] - opn[x]) for x in U)),
                                  only_open=int(sum(len(opn[x] - full[x]) for x in U)),
                                  span_ratio_full_over_open=dist(D.SPAN_FULL[x] / D.SPAN[x] for x in U))
    CB = D.SETS["CB"]
    res["overlap"] = dict(
        S_within_2_of_CB=int(sum(any(abs(y["L"] - z["L"]) <= C.SAME_LVL for z in CB[x]) for x in U for y in S[x])),
        S_with_neighbour_within_10=int(sum(
            any(0 < abs(y["L"] - z["L"]) <= C.NEAR_LVL for z in S[x]) or
            any(C.SAME_LVL < abs(y["L"] - z["L"]) <= C.NEAR_LVL for z in CB[x]) for x in U for y in S[x])),
        CB_levels=int(sum(len(CB[x]) for x in U)),
        CB_minus_strong_W=int(sum(len(D.SETS["W"][x]) for x in U)),
        X100_levels=int(sum(len(D.SETS["X100"][x]) for x in U)),
        X100_calls=int(sum(y["net"] > 0 for x in U for y in D.SETS["X100"][x])),
        X100_puts=int(sum(y["net"] < 0 for x in U for y in D.SETS["X100"][x])),
        X100_per_session=dist(len(D.SETS["X100"][x]) for x in U),
        S_subset_of_X100=int(sum(any(abs(y["L"] - z["L"]) < 0.5 for z in D.SETS["X100"][x]) for x in U for y in S[x])),
    )

    # tests, strata, placebo coverage
    res["sets"] = {}
    T = {}
    for k in ("S", "X100", "W", "CB"):
        R, P = C.tests(D, D.SETS[k]), C.tests(D, D.PLV[k])
        T[k] = (R, P)
        blk = dict(levels_real=int(sum(len(D.SETS[k][x]) for x in U)),
                   levels_placebo=int(sum(len(D.PLV[k][x]) for x in U)))
        par = {(str(x), y["parent"]) for x in U for y in D.PLV[k][x]}
        blk["real_levels_with_placebo_share"] = round(
            float(np.mean([(str(x), y["L"]) in par for x in U for y in D.SETS[k][x]])) if blk["levels_real"] else 0, 3)
        blk["hyp"] = {h: dict(real=cnt(f(R)), placebo=cnt(f(P)), strata=strata(f(R), f(P))) for h, f in HYP.items()}
        blk["cells"] = {c: dict(real=cnt(f(R)), placebo=cnt(f(P)), strata=strata(f(R), f(P))) for c, f in CELLS.items()}
        if k == "S":
            blk["first_touch"] = {h: dict(real=cnt(f(R[R["first_touch"]])), placebo=cnt(f(P[P["first_touch"]])),
                                          strata=strata(f(R[R["first_touch"]]), f(P[P["first_touch"]])))
                                  for h, f in HYP.items()}
            blk["placebo_bins_within_10_of_CB"] = round(float(np.mean(
                [any(abs(y["L"] - z["L"]) < C.PLC_SEP for z in CB[x]) for x in U for y in D.PLV["S"][x]])), 3)
        res["sets"][k] = blk

    # A/A on S placebo tests (escape on placebo only)
    _, PS = T["S"]
    PSm = C.add_measures(D, PS)
    res["AA"] = {h: dict(unmatched=C.aa(D, f(PSm), matched=False), matched=C.aa(D, f(PSm), matched=True))
                 for h, f in HYP.items()}
    res["AA_gate"] = dict(train_rule="both Holm hypotheses (put, call) pass, unmatched and matched",
                          passed=bool(all(res["AA"][h][v].get("PASS", False) for h in ("put", "call")
                                          for v in ("unmatched", "matched"))))
    return D, res


OUT = {}
for name in ("train", "ra"):
    D, r = step0(name)
    OUT[name] = r
    if name == "train":
        show = [x for x in D.USABLE if str(x) in ("2026-05-01",)] + D.USABLE[-2:]
        OUT["train"]["listing"] = {
            str(x): dict(open=D.OPEN[x], im=round(D.IM[x], 2), span_bn=round(D.SPAN[x] / 1e9, 1),
                         displayed=len(D.CHART[x]),
                         strong=[dict(L=y["L"], net_bn=round(y["net"] / 1e9, 1), gross_bn=round(y["mag"] / 1e9, 1),
                                      in_window=any(abs(y["L"] - z["L"]) < 0.5 for z in D.SETS["S"][x]))
                                 for y in sorted(D.STRONG_ALL[x], key=lambda z: z["L"])])
            for x in show}

C.OUT_DIR.mkdir(parents=True, exist_ok=True)
(C.OUT_DIR / "step0.json").write_text(json.dumps(OUT, indent=1, default=str))
print(json.dumps(OUT, indent=1, default=str))
