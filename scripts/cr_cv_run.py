#!/usr/bin/env python3
"""CR-CV run — the single outcome run (C6, C7, C10, C11). READ-ONLY, train only. Run once.

Primary: 16 cells = setup (S1-S4) x side (hypothesis / opposite) x race T (2, 4 pts), primary fill model (C3), 1-s race
(C4). Per cell E = (2p - 1)T - 0.50; session-cluster bootstrap 10,000 draws, seed 20260920 (one matrix per setup
universe); Holm over the 16 cells; PASS / REFUTED / INCONCLUSIVE / UNDERPOWERED (C7). Readouts (no verdicts): costs 0.25 /
0.75; win rate vs break-even; touch-fill model; ±10 under both models; S4 FAST-break / SLOW-fade split; fill shares.
Writes scripts/logs/cr_cv/results.json and trades.parquet.
"""
from __future__ import annotations

import json
import time as _time
from collections import Counter

import numpy as np
import pandas as pd

import cr_cv_lib as V

T0 = _time.time()
OUT = V.OUT / "results.json"
assert not OUT.exists(), "CR-CV is run once; results.json already exists"
S0 = json.loads((V.OUT / "step0.json").read_text())
assert S0["gate"]["passed"], S0["gate"]


def log(*a):
    print(f"[{_time.time() - T0:7.1f}s]", *a, flush=True)


A = V.build_all()
TR = A["TR"]
assert len(TR) == S0["n_schedule_rows"]
for k, v in S0["fills"].items():                     # schedule unchanged since Step 0
    st, sd, md = k.split("/")
    g = TR[(TR["setup"] == st) & (TR["side"] == sd) & (TR["model"] == md)]
    assert len(g) == v["n"] and int((g["status"] == "fill").sum()) == v["fills"], (k, v)
log("schedule asserted = Step 0", len(TR))

for T in V.SIZES + (V.REF,):
    TR[f"r{int(T)}"] = [V.outcome(A, r, T) for r in TR.itertuples()]
log("outcomes done")

SETUPS = ("S1", "S2", "S3", "S4")
SI = {s: {d: i for i, d in enumerate(A["UNIV"][s])} for s in SETUPS}
WT = {s: V.boot_weights(len(SI[s])) for s in SETUPS}          # C6: one matrix per setup universe, seed 20260920
ONE = {s: np.ones((1, len(SI[s]))) for s in SETUPS}
NAMES = {"S1": ("long after G1-down", "short"), "S2": ("chosen edge", "opposite edge"),
         "S3": ("fade", "continuation"), "S4": ("FAST break + SLOW fade", "FAST fade + SLOW break")}


def cell(st, sd, model, T, cost, sub=None, keep_draws=False):
    g = TR[(TR["setup"] == st) & (TR["side"] == sd) & (TR["model"] == model)]
    if sub is not None:
        g = g[g["sub"] == sub]
    col = f"r{int(T)}"
    cnt = Counter(g[col])
    n, w = V.sess_counts(zip(g["d"], g[col]), SI[st])
    out = dict(setup=st, side=sd, side_name=NAMES[st][0 if sd == "hyp" else 1], model=model, T=T, cost=cost,
               events=len(g), fills=int((g["status"] == "fill").sum()), n=int(n.sum()), wins=int(w.sum()),
               losses=int(n.sum() - w.sum()), ties=cnt.get("tie", 0), unresolved=cnt.get("unresolved", 0),
               no_fill=cnt.get("no fill", 0), sessions=int((n > 0).sum()))
    if n.sum() == 0:
        return out
    p = float(V.boot_p(ONE[st], n, w)[0])
    pb = V.boot_p(WT[st], n, w)
    Eb = V.expectancy(pb, T, cost)
    be = V.break_even(T, cost)
    out.update(p=p, E=float(V.expectancy(p, T, cost)), ci95=V.pct_ci(Eb, 0.95), p_ci95=V.pct_ci(pb, 0.95),
               break_even=be, p_minus_be_pp=100 * (p - be), p_value=V.two_sided_p(Eb),
               draws=int(np.isfinite(Eb).sum()))
    if keep_draws:
        out["E_draws"] = Eb
    return out


# ------------------------------------------------------------------------------------------------ primary (16 cells)
prim = []
for st in SETUPS:
    for sd in ("hyp", "opp"):
        for T in V.SIZES:
            c = cell(st, sd, "primary", T, V.COST, keep_draws=True)
            prim.append(c)
hc = [dict(p=c["p_value"], n=c["n"], E_draws=c.pop("E_draws"), ci95=c["ci95"]) for c in prim]   # V.holm's 'p' = test p
V.holm(hc)
for c, h in zip(prim, hc):
    c.update({k: h[k] for k in ("rank", "p_holm", "holm_level", "ci_holm", "verdict")})
R: dict = dict(primary=prim, n_boot=V.N_BOOT, seed=V.SEED, cost=V.COST, holm_m=V.M_HOLM,
               any_pass=any(c["verdict"] == "PASS" for c in prim),
               label="mechanical brackets (B6), not Ryan's trade; S1 is a measurement only (C12)")
for c in prim:
    log(f"PRIMARY {c['setup']} {c['side']:3s} ±{int(c['T'])}: n {c['n']:3d} p {100 * c['p']:.1f} % "
        f"E {c['E']:+.2f} [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}] p {c['p_value']:.4f} holm {c['p_holm']:.4f} "
        f"-> {c['verdict']}")

# ------------------------------------------------------------------------------------------------ readouts
rd: dict = {}
rd["costs"] = [cell(st, sd, "primary", T, cst) for st in SETUPS for sd in ("hyp", "opp") for T in V.SIZES
               for cst in V.COSTS_READ]
rd["touch_fill"] = [cell(st, sd, "touch", T, V.COST) for st in SETUPS for sd in ("hyp", "opp") for T in V.SIZES]
rd["ref10"] = [cell(st, sd, md, V.REF, V.COST) for st in SETUPS for sd in ("hyp", "opp") for md in ("primary", "touch")]
rd["s4_split"] = [cell("S4", sd, md, T, V.COST, sub=sb) for sd in ("hyp", "opp") for sb in ("FAST", "SLOW")
                  for md in ("primary",) for T in V.SIZES + (V.REF,)]
rd["trade_through_share"] = S0["trade_through_share"]
R["readouts"] = rd
for c in rd["ref10"]:
    log(f"REF10 {c['setup']} {c['side']} {c['model']}: n {c['n']} p {100 * c.get('p', np.nan):.1f} %")

TR.to_parquet(V.OUT / "trades.parquet")
OUT.write_text(json.dumps(R, indent=2, default=float))
log("ANY PASS:", R["any_pass"])
