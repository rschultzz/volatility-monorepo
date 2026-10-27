"""CR-CL run — does a beyond-noise skew move predict the next move? READ-ONLY. Run ONCE (C15).

Spec: specs/CR-CL-skew-beyond-noise.md (K1-K10, C1-C16, A1, Approvals, A2). 2026 train `y26` only (R-a, 2025 and the
holdout are never loaded). Definitions live in scripts/cr_cl_lib.py (shared with Step 0).

  primary (A2.1):  G1-down (decline leg, put skew fell beyond noise, ratio < 0) -> up. +/-10 race from the
                   confirmation bar close; win rate minus the 3-key matched in-noise baseline; 95 % session-cluster
                   bootstrap CI (10,000 draws, seed 20260920); PASS / REFUTED / INCONCLUSIVE; >= 40 resolved + covered.
  readouts (A2.3): G1-up, G2-up, G2-down; swing 8 / 16; ratio cut -0.5 / 1.5; late legs; G1-down by GEX tag.
                   Unadjusted 95 % CIs, no verdicts, never pooled.
Writes scripts/logs/cr_cl/results.json and legs.parquet.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import cr_cl_lib as C

N_BOOT = 10_000
D = C.load()
SESS = D["SESS"]
SI = {d: i for i, d in enumerate(SESS)}
WT = C.boot_weights(len(SESS), N_BOOT, np.random.default_rng(C.SEED))        # A2.5: one matrix for every cell
ONES = np.ones((1, len(SESS)))


def outcomes(r, h):
    B = D["BARS"][r.d]
    return dict(r10=C.race(B, r.ci, r.p0, h, 10.0), r20=C.race(B, r.ci, r.p0, h, 20.0),
                m30=C.move_at(B, r.ci, r.p0, h, 30), m60=C.move_at(B, r.ci, r.p0, h, 60),
                mfe=C.mfe_before(B, r.ci, r.p0, h, 10.0))


def key(r):
    return (int(r.dir), int(r.tercile), int(r.cbucket), r.edge)


def matched(grows, brows):
    M, _ = C.strata_matrices(SESS, grows, brows)
    dpt, ncov, cov = C.matched_diff(ONES, M)
    db, _, _ = C.matched_diff(WT, M)
    db = db[np.isfinite(db)]
    # baseline rate as the group-weighted stratum mean (point estimate)
    Gn, Bn, Bw = M["Gn"].sum(0), M["Bn"].sum(0), M["Bw"].sum(0)
    c = Bn > 0
    brate = float((Gn[c] * Bw[c] / Bn[c]).sum() / Gn[c].sum()) if Gn[c].sum() else None
    grate = float((M["Gw"].sum(0)[c]).sum() / Gn[c].sum()) if Gn[c].sum() else None
    out = dict(n_covered=int(ncov[0]), coverage=round(float(cov[0]), 3) if np.isfinite(cov[0]) else None,
               group=None if grate is None else round(grate, 4), baseline=None if brate is None else round(brate, 4),
               diff=round(float(dpt[0]), 4) if np.isfinite(dpt[0]) else None, n_draws_defined=int(len(db)))
    if len(db):
        lo, hi = np.percentile(db, [2.5, 97.5])
        out.update(ci95=[round(float(lo), 4), round(float(hi), 4)],
                   p_two_sided=round(float(min(1.0, 2 * min((db <= 0).mean(), (db >= 0).mean()))), 4))
    return out


def race_rows(df, col):
    rows = []
    for r in df.itertuples():
        o = getattr(r, col)
        if o in ("win", "loss"):
            rows.append((r.d, key(r), 1.0 if o == "win" else 0.0))
    return rows


def value_rows(df, col):
    return [(r.d, key(r), float(getattr(r, col))) for r in df.itertuples() if np.isfinite(getattr(r, col))]


def mfe_summary(df):
    v = df["mfe"].to_numpy(float)
    if not len(v):
        return None
    r10 = v >= 10
    return dict(n=len(v), q=dict(zip(("p25", "p50", "p75", "p90"), np.round(np.percentile(v, [25, 50, 75, 90]), 2).tolist())),
                share_ge10=round(float(r10.mean()), 3),
                share_ge15_given_10=round(float((v[r10] >= 15).mean()), 3) if r10.any() else None,
                share_ge20_given_10=round(float((v[r10] >= 20).mean()), 3) if r10.any() else None)


def evaluate(G, base, h):
    G = G.copy()
    Bs = base.copy()
    for df in (G, Bs):
        o = [outcomes(r, h) for r in df.itertuples()]
        for k in ("r10", "r20", "m30", "m60", "mfe"):
            df[k] = [x[k] for x in o]
    res = dict(n_legs=len(G), n_sessions=int(G["d"].nunique()), n_baseline=len(Bs),
               counts_r10=G["r10"].value_counts().to_dict(), counts_r10_baseline=Bs["r10"].value_counts().to_dict(),
               counts_r20=G["r20"].value_counts().to_dict(), counts_r20_baseline=Bs["r20"].value_counts().to_dict(),
               edge_share=round(float((G["edge"] == "EDGE").mean()), 3) if len(G) else None)
    res["race10"] = matched(race_rows(G, "r10"), race_rows(Bs, "r10"))
    res["race20"] = matched(race_rows(G, "r20"), race_rows(Bs, "r20"))
    res["move30"] = matched(value_rows(G, "m30"), value_rows(Bs, "m30"))
    res["move60"] = matched(value_rows(G, "m60"), value_rows(Bs, "m60"))
    res["mfe_before_-10"] = dict(cell=mfe_summary(G), baseline=mfe_summary(Bs))
    return res, G


def verdict(r):
    n = r["race10"]["n_covered"]
    if n < C.MIN_CELL:
        return "INCONCLUSIVE (underpowered)"
    lo, hi = r["race10"]["ci95"]
    return "PASS" if lo > 0 else ("REFUTED" if hi < 0 else "INCONCLUSIVE")


def cells_for(W, ratio_lo=0.0, ratio_hi=1.0, only=None):
    out, legs = {}, []
    base = W[W["grp"] == "base"]
    for g, dr in C.CELLS:
        name = f"{g}-{'up' if dr == 1 else 'down'}"
        if only and name not in only:
            continue
        G = W[C.group_mask(W, g, dr, ratio_lo, ratio_hi)]
        r, Gx = evaluate(G, base[base["dir"] == dr], C.h_of(g, dr))
        r["h"] = "up" if C.h_of(g, dr) == 1 else "down"
        out[name] = r
        legs.append(Gx.assign(cell=name))
    return out, legs


R = {"sessions": len(SESS), "n_boot": N_BOOT, "seed": C.SEED}
ALL_LEGS = []

# ------------------------------------------------------------------------------------------ swing 12
W12, cuts12 = C.add_terciles(C.build_legs(D, 12.0))
P12 = W12[~W12["late"]]
assert len(W12) == 1233 and int((W12["reason"] == "ok").sum()) == 764      # Step 0 parity (CR-CK)
main, lg = cells_for(P12)
ALL_LEGS += lg
prim = main["G1-down"]
prim["verdict"] = verdict(prim)
R["primary_G1_down"] = prim
R["readout_swing12"] = {k: v for k, v in main.items() if k != "G1-down"}

# ratio cut -0.5 / 1.5
R["readout_ratio_cut"], _ = cells_for(P12, -0.5, 1.5)

# late legs (own baseline, same strata)
Lt = W12[W12["late"]]
R["readout_late"], _ = cells_for(Lt)

# GEX tags on G1-down (A8): each tag's legs vs the full matched baseline
lev, _why = C.gex_levels(D)
Pg = C.tag_gex(P12, lev)
G1d = Pg[C.group_mask(Pg, "G1", -1)]
baseg = Pg[(Pg["grp"] == "base") & (Pg["dir"] == -1)]
R["readout_gex_G1_down"] = {}
for tag in ("put", "call", "none", "no levels"):
    sub = G1d[G1d["gex"] == tag]
    if len(sub):
        R["readout_gex_G1_down"][tag], _ = evaluate(sub, baseg, 1)

# swing 8 / 16
for sw in (8.0, 16.0):
    W, _ = C.add_terciles(C.build_legs(D, sw))
    R[f"readout_swing{int(sw)}"], _ = cells_for(W[~W["late"]])

(C.OUT / "results.json").write_text(json.dumps(R, indent=2, default=str))
try:
    pd.concat(ALL_LEGS).to_parquet(C.OUT / "legs.parquet")
except Exception as exc:                                                     # results.json is already saved
    print("legs.parquet not written:", exc)
print(json.dumps(R, indent=1, default=str))
