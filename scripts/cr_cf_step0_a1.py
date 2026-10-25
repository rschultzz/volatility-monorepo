#!/usr/bin/env python3
"""CR-CF Step 0 re-run under Amendment A1 (train) — counts and the A/A gate. READ-ONLY; no real-level measure.

Spec: specs/CR-CF-which-levels-price-respects.md, Amendment A1 (A1.1–A1.5). Construction from scripts/cr_cf_lib.py
(shared with the Stage 1 script). Counts as Step 0 Q1, per family and side, plus first-side-only tests and the
A1.1 / A1.2 dropped placebo tests. A/A as Step 0 Q2 (200 pseudo-level splits, default_rng(20261025); 2,000 session
resamples per split, default_rng(20260920); session-weighted half-1 − half-2 of the primary; PASS if the CI excludes
0 in <= 10 % of splits and |median| <= 3 min), per family × {pooled, below, above}, all families pooled, and the
first-side-only placebo tests. Gate (A1.5): every one of the 24 Holm cells (F1–F5, F7) passes.

Usage: apps/web/.venv/bin/python -u scripts/cr_cf_step0_a1.py | tee scripts/logs/cr_cf/step0_a1.log
"""
from __future__ import annotations

import json
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
signal.alarm(3 * 3600)
import numpy as np  # noqa: E402

import cr_cf_lib as C  # noqa: E402

HOLM_FAMS = ["F1", "F2", "F3", "F4", "F5_100", "F5_50", "F5_25", "F7"]
R, P = C.build(measure_real=False)
assert "esc" not in R.columns                       # no real-level measure in Step 0
nS, USABLE, SIDX = C.nS, C.USABLE, C.SIDX


def dist(v):
    v = np.asarray(v, float)
    return dict(median=float(np.median(v)), p25=float(np.percentile(v, 25)), p75=float(np.percentile(v, 75)),
                zero_share=round(float(np.mean(v == 0)), 3))


Q1 = {}
for f in C.FAMILIES:
    Rf, Pf = R[R["fam"] == f], P[P["fam"] == f]
    side = lambda E, s: dict(tests=int((E["side"] == s).sum()), sessions=int(E[E["side"] == s]["d"].nunique()))
    Q1[f] = dict(
        levels_per_session=dist([1] * nS if f == "F4" else [len(C.LEV[d][f]) for d in USABLE]),
        placebo_levels_per_session=dist([C.PLC_K] * nS if f == "F4" else [len(C.PLV[d][f]) for d in USABLE]),
        real=dict(tests=len(Rf), sessions=int(Rf["d"].nunique()), below=side(Rf, 1), above=side(Rf, -1),
                  first_side_only=int(Rf["first"].sum()), pre10=int(Rf["pre10"].sum()),
                  post10=int((~Rf["pre10"]).sum()), tests_per_session=dist([(Rf["d"] == d).sum() for d in USABLE])),
        placebo=dict(tests=len(Pf), sessions=int(Pf["d"].nunique()), below=side(Pf, 1), above=side(Pf, -1),
                     first_side_only=int(Pf["first"].sum()), pre10=int(Pf["pre10"].sum()),
                     post10=int((~Pf["pre10"]).sum()), censored=int(Pf["censored"].sum()),
                     tests_per_session=dist([(Pf["d"] == d).sum() for d in USABLE])),
        gex_overlap_tests=dict(real=round(float(Rf["ovl"].mean()), 3), placebo=round(float(Pf["ovl"].mean()), 3)),
    )
    if f != "F4":
        pl = [x["L"] for d in USABLE for x in C.PLV[d][f]]
        Q1[f]["placebo_on_25_multiple_share"] = round(float(np.mean([x % 25 == 0 for x in pl])), 3)
Q1["_dropped_placebo_tests"] = C.DROPPED
Q1["_missing"] = {f: len(v) for f, v in C.MISSING.items()}


def aa(E):
    if E.empty:
        return dict(n_tests=0)
    keys = sorted(set(E["key"]))
    if len(keys) < 4:
        return dict(n_tests=int(len(E)), pseudo_levels=len(keys), note="too few pseudo-levels")
    rs_, rb_ = np.random.default_rng(C.SEED_DRAW), np.random.default_rng(C.SEED_BOOT)
    sx = np.array([SIDX[d] for d in E["d"]])
    val = E["esc"].to_numpy(float)
    stats, excl = [], 0
    for _ in range(C.N_SPLIT):
        perm = rs_.permutation(len(keys))
        h1 = {keys[k] for k in perm[: len(keys) // 2]}
        g = np.array([k not in h1 for k in E["key"]]).astype(int)
        c = np.zeros((nS, 2, 2))
        np.add.at(c, (sx, g, 0), val)
        np.add.at(c, (sx, g, 1), 1)
        m = np.where(c[:, :, 1] > 0, c[:, :, 0] / np.maximum(c[:, :, 1], 1), 0.0)
        pres = (c[:, :, 1] > 0).astype(float)
        stats.append((m[:, 0] * pres[:, 0]).sum() / pres[:, 0].sum() - (m[:, 1] * pres[:, 1]).sum() / pres[:, 1].sum())
        Wb = rb_.multinomial(nS, np.full(nS, 1 / nS), size=C.N_BOOT_SPLIT).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            bst = (Wb @ (m[:, 0] * pres[:, 0])) / (Wb @ pres[:, 0]) - (Wb @ (m[:, 1] * pres[:, 1])) / (Wb @ pres[:, 1])
        lo, hi = np.nanpercentile(bst, [2.5, 97.5])
        excl += int(lo > 0 or hi < 0)
    stats = np.array(stats)
    share, med = excl / C.N_SPLIT, float(np.median(stats))
    return dict(n_tests=int(len(E)), pseudo_levels=len(keys), sessions=int(E["d"].nunique()),
                excl_share=round(share, 3), median=round(med, 3),
                PASS=bool(share <= C.AA_EXCL_MAX and abs(med) <= C.AA_MED_TOL))


Q2 = {}
for f in C.FAMILIES + ["all"]:
    E = P if f == "all" else P[P["fam"] == f]
    Q2[f] = {"pooled": aa(E), "below": aa(E[E["side"] == 1]), "above": aa(E[E["side"] == -1]),
             "first_side_only": aa(E[E["first"]])}
GATE = {f"{f}/{c}": Q2[f][c].get("PASS", False) for f in HOLM_FAMS for c in ("pooled", "below", "above")}
RES = dict(sessions=dict(n=nS, first=str(USABLE[0]), last=str(USABLE[-1])), Q1=Q1, Q2=Q2, gate=GATE,
           gate_pass=all(GATE.values()))
C.OUT_DIR.mkdir(parents=True, exist_ok=True)
(C.OUT_DIR / "step0_a1.json").write_text(json.dumps(RES, indent=1, default=str))

print("sessions", RES["sessions"], "dropped", C.DROPPED, "missing", Q1["_missing"])
print("\nQ1 counts under A1")
for f in C.FAMILIES:
    v = Q1[f]
    print(f"  {f:7s} lv {v['levels_per_session']['median']:.0f} plc lv {v['placebo_levels_per_session']['median']:.0f} | "
          f"real {v['real']['tests']} ({v['real']['sessions']} s) below {v['real']['below']} above {v['real']['above']} "
          f"first {v['real']['first_side_only']} pre/post10 {v['real']['pre10']}/{v['real']['post10']} | "
          f"plc {v['placebo']['tests']} ({v['placebo']['sessions']} s) below {v['placebo']['below']} above "
          f"{v['placebo']['above']} first {v['placebo']['first_side_only']} | ovl {v['gex_overlap_tests']}"
          + (f" | plc on 25s {v['placebo_on_25_multiple_share']}" if "placebo_on_25_multiple_share" in v else ""))
print("\nQ2 A/A under A1")
for f in C.FAMILIES + ["all"]:
    print(f"  {f:7s} " + "  ".join(f"{c}: n {v.get('n_tests')} s {v.get('sessions')} excl {v.get('excl_share')} med "
                                    f"{v.get('median')} {'PASS' if v.get('PASS') else 'FAIL' if 'PASS' in v else v.get('note', '')}"
                                    for c, v in Q2[f].items()))
print("\nA1.5 gate (24 Holm cells):", "PASS" if RES["gate_pass"] else "FAIL", [k for k, v in GATE.items() if not v])
