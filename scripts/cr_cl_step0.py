"""CR-CL Step 0 — counts, beyond-noise share, match coverage, A/A. READ-ONLY. No group outcome is computed.

Spec: specs/CR-CL-skew-beyond-noise.md (K8, C13). 2026 train `y26` only (R-a, 2025, holdout never loaded).
Race outcomes are computed ONLY on baseline (in-noise) legs and ONLY inside the A/A splits; only the split
statistics are reported. Writes scripts/logs/cr_cl/step0.json.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import cr_cj_lib as L
import cr_cl_lib as C

D = C.load()
R = {"sessions": len(D["SESS"]), "first": str(D["SESS"][0]), "last": str(D["SESS"][-1])}
assert max(D["SESS"]) <= L.PERIODS[C.PERIOD][1]


def bl(b):
    return L.bucket_label(int(b))


def cell_counts(W, ratio_lo=0.0, ratio_hi=1.0):
    """Per cell: legs, covered (stratum has >= 1 baseline leg, pre-outcome), coverage, EDGE share."""
    P = W[~W["late"]]
    base = P[P["grp"] == "base"]
    bkeys = {C.stratum(r) for _, r in base.iterrows()}
    out = {}
    for g, dr in C.CELLS:
        m = C.group_mask(P, g, dr, ratio_lo, ratio_hi)
        G = P[m]
        cov = [C.stratum(r) in bkeys for _, r in G.iterrows()]
        out[f"{g}-{'up' if dr == 1 else 'down'}"] = dict(
            n=len(G), sessions=int(G["d"].nunique()), covered=int(sum(cov)),
            coverage=round(float(np.mean(cov)), 3) if len(G) else None,
            edge_share=round(float((G["edge"] == "EDGE").mean()), 3) if len(G) else None,
            gate_upper_bound_ok=bool(sum(cov) >= C.MIN_CELL))
    out["baseline"] = {("up" if dr == 1 else "down"): dict(
        n=int((base["dir"] == dr).sum()), edge_share=round(float((base[base["dir"] == dr]["edge"] == "EDGE").mean()), 3))
        for dr in (1, -1)}
    return out


def funnel(W):
    P = W[~W["late"]]
    f = {"legs": len(W), "primary (e <= 11:00)": len(P), "late (11:00 < e <= 12:45)": int(W["late"].sum())}
    f.update({f"primary|{k}": int(v) for k, v in P["reason"].value_counts().items()})
    f.update({f"primary|grp={k}": int(v) for k, v in P["grp"].value_counts().items()})
    return f


# --------------------------------------------------------------------------------------------- per swing
LEGS = {}
for sw in (12.0, 8.0, 16.0):
    W, cuts = C.add_terciles(C.build_legs(D, sw))
    LEGS[sw] = W
    key = f"swing{int(sw)}"
    R[key] = {"funnel": funnel(W), "size_tercile_cuts_pts": [round(c, 2) for c in cuts],
              "cells": cell_counts(W)}
    if sw == 12.0:
        R[key]["cells_ratio_cut_-0.5_1.5"] = cell_counts(W, -0.5, 1.5)
        Lt = W[W["late"]].copy()
        Lt["late"] = False                                     # count late legs with the same machinery
        R[key]["late_cells"] = cell_counts(Lt)

W = LEGS[12.0]
P = W[~W["late"]]

# beyond-noise share by 30-min start bucket (ratio-defined legs, primary + late)
ok = W[W["reason"] == "ok"]
R["beyond_share_by_start_bucket"] = {
    bl(b): dict(n=len(g), beyond=round(float(g["beyond"].astype(bool).mean()), 3),
                g1=int(((g["beyond"] == True) & (g["ratio"] < 0)).sum()),  # noqa: E712
                g2=int(((g["beyond"] == True) & (g["ratio"] > 1)).sum()))  # noqa: E712
    for b, g in ok.groupby("sbucket")}

# stratum occupancy (baseline, primary)
base = P[P["grp"] == "base"]
occ = base.groupby(["dir", "tercile", "cbucket", "edge"]).size()
R["baseline_strata"] = {"n_strata": int(len(occ)), "median_legs_per_stratum": float(occ.median()),
                        "strata_with_1_leg": int((occ == 1).sum())}

# coverage under the size x time match only (no EDGE key) — to show what the B7 key costs
bkeys2 = {(int(r["dir"]), int(r["tercile"]), int(r["cbucket"])) for _, r in base.iterrows()}
R["coverage_without_edge_key"] = {}
for g, dr in C.CELLS:
    G = P[C.group_mask(P, g, dr)]
    cov = [(int(r["dir"]), int(r["tercile"]), int(r["cbucket"])) in bkeys2 for _, r in G.iterrows()]
    R["coverage_without_edge_key"][f"{g}-{'up' if dr == 1 else 'down'}"] = dict(n=len(G), covered=int(sum(cov)))

# GEX readout counts (no outcomes)
lev, why = C.gex_levels(D)
R["gex_sessions"] = {"with_levels": len(lev), "without": len(why),
                     "why": pd.Series(list(why.values())).value_counts().to_dict() if why else {},
                     "levels_per_session_median": float(np.median([len(v) for v in lev.values()])) if lev else None}
Wg = C.tag_gex(P, lev)
R["gex_tag_by_cell"] = {}
for g, dr in C.CELLS:
    G = Wg[C.group_mask(Wg, g, dr)]
    R["gex_tag_by_cell"][f"{g}-{'up' if dr == 1 else 'down'}"] = G["gex"].value_counts().to_dict()
R["gex_tag_baseline"] = Wg[Wg["grp"] == "base"]["gex"].value_counts().to_dict()

# ------------------------------------------------------------------------------- A/A (C13): baseline only
sess = sorted(base["d"].unique())
si = {d: i for i, d in enumerate(sess)}
res = {}
for hs in (1, -1):                                       # h = hs * dir: +1 continuation, -1 against
    res[hs] = [C.race(D["BARS"][r.d], r.ci, r.p0, hs * r.dir, 10.0) for r in base.itertuples()]
rng = np.random.default_rng(C.SEED)
Wb = C.boot_weights(len(sess), 500, rng)
AA = {}
for hs, hname in ((1, "continue"), (-1, "against")):
    outc = np.array(res[hs])
    keep = np.isin(outc, ["win", "loss"])
    for dr in (1, -1):
        diffs, excl = [], []
        for _ in range(200):
            A = set(rng.permutation(sess)[: len(sess) // 2])
            grp, bas = [], []
            for (r, o, k) in zip(base.itertuples(), outc, keep):
                if not k or r.dir != dr:
                    continue
                row = (r.d, (r.dir, r.tercile, r.cbucket, r.edge), 1.0 if o == "win" else 0.0)
                (grp if r.d in A else bas).append(row)
            M, _ = C.strata_matrices(sess, grp, bas)
            dpt, _, _ = C.matched_diff(np.ones((1, len(sess))), M)
            db, _, _ = C.matched_diff(Wb, M)
            lo, hi = np.nanpercentile(db, [2.5, 97.5])
            diffs.append(float(dpt[0]))
            excl.append(bool(lo > 0 or hi < 0))
        dd = np.array(diffs)
        AA[f"{hname}|{'up' if dr == 1 else 'down'}"] = dict(
            median=round(float(np.nanmedian(dd)), 4), p2_5=round(float(np.nanpercentile(dd, 2.5)), 4),
            p97_5=round(float(np.nanpercentile(dd, 97.5)), 4), share_ci_excludes_0=round(float(np.mean(excl)), 3))
R["aa_pm10"] = AA

(C.OUT / "step0.json").write_text(json.dumps(R, indent=2, default=str))
print(json.dumps(R, indent=1, default=str))
