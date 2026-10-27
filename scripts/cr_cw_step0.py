#!/usr/bin/env python3
"""CR-CW Step 0 — data checks, H1 tercile cuts and CLEAN / CHOPPY counts, H2 break / pullback / turn funnel and
HOLD / FILL counts, snap-band check, A/A per hypothesis. NO touch or trade outcome is computed (the only races run
are the random-baseline entries for the A/A, CR-CT C12). READ-ONLY, train only.

Spec: specs/CR-CW-momentum-path-snap.md. Writes scripts/logs/cr_cw/step0.json and step0_*.parquet (features only).
Gate (C15), per hypothesis: each group >= 40 events on >= 30 sessions and |A/A mean| <= 1 pp.
"""
from __future__ import annotations

import json
import time as _time

import numpy as np
import pandas as pd

import cr_cw_lib as W
import cr_ct_lib as T

T0 = _time.time()
R: dict = {}


def log(*a):
    print(f"[{_time.time() - T0:7.1f}s]", *a, flush=True)


def jd(x):
    return json.loads(json.dumps(x, default=str))


def counts(df, col):
    return {k: dict(n=int(len(g)), sessions=int(g["d"].nunique())) for k, g in df.groupby(col)}


log("1. universe, CR-CT first touches, 1-s rows with volume")
D, S_ct, S1, P = W.build_all()
SESS = D["SESS"]
R["n_sessions"], R["n_touches"] = len(SESS), len(P)
log("  sessions", len(SESS), "first touches", len(P))

log("2. data checks")
mism, n_all, hl_n, hl_v, v_all, gaps = 0, 0, 0, 0.0, 0.0, []
for d in SESS:
    s, (rv, i30) = S1[d], W.rth_view(S1[d])
    c = S_ct[d]
    assert np.array_equal(rv["sec"], c["sec"]), d
    for k in ("h", "l", "c"):
        mism += int((rv[k] != c[k]).sum())
    n_all += len(rv["sec"])
    wide = (rv["h"] - rv["l"]) > W.TICK + 1e-9
    hl_n += int(wide.sum())
    hl_v += float(rv["v"][wide].sum())
    v_all += float(rv["v"].sum())
    gaps.append(int(np.diff(rv["sec"]).max()))
    off = np.abs(rv["c"] / W.TICK - np.rint(rv["c"] / W.TICK)).max()
    assert off < 1e-6, (d, off)
R["check_1s_equal_ct"] = dict(rows=n_all, hlc_mismatches=mism)
R["check_hl_gt_tick"] = dict(share_seconds=hl_n / n_all, share_volume=hl_v / v_all, n=hl_n)
R["check_max_gap_s"] = dict(median=float(np.median(gaps)), max=int(max(gaps)))
log("  1-s vs CR-CT", R["check_1s_equal_ct"], "h-l > tick", R["check_hl_gt_tick"], "gaps", R["check_max_gap_s"])
# H1 net = CR-CT speed numerator (1-min closes = last 1-s close)
dn = []
for r in P.itertuples():
    if r.net10 is not None and np.isfinite(r.net10) and r.speed10 is not None and np.isfinite(r.speed10):
        dn.append(abs(r.net10 - r.speed10 * 10 * D["IM"][r.d]))
R["check_net_vs_ct_speed"] = dict(n=len(dn), max_abs=float(max(dn)), n_gt_0_01=int(np.sum(np.array(dn) > 0.01)))
log("  net10 vs CR-CT speed10", R["check_net_vs_ct_speed"])

log("3. H1 cuts and classes (no outcome)")
CUTS = {}
for w in [str(W.W_PRIMARY)] + [str(x) for x in W.W_READ] + ["A7"]:
    c, n = W.tercile_cuts(P[f"eff{w}"])
    CUTS[w] = dict(cuts=c, n=n)
    P[f"h1{w}"] = W.h1_class(P[f"eff{w}"], c)
R["h1_cuts"] = CUTS
R["h1_classes"] = {w: counts(P, f"h1{w}") for w in CUTS}
ct = json.loads(W.CT_STEP0.read_text())
c10 = ct["cuts"]["10"]["cuts"]
cc, nn = T.cuts(P, "speed10", "relvol10")
assert np.allclose(cc, c10) and nn == ct["cuts"]["10"]["n"], (cc, c10)
P["cls10"] = T.classify(P, "speed10", "relvol10", c10)
R["h1_by_ct_class"] = {k: counts(g, "h110") for k, g in P.groupby("cls10")}
R["h1_by_family"] = {k: counts(g, "h110") for k, g in P.groupby("fam")}
P["retr_band"] = [W.retr_band(x) for x in P["retr10"]]
R["retrace"] = dict(quantiles=[float(x) for x in P["retr10"].dropna().quantile([.1, .25, .5, .75, .9])],
                    bands=counts(P, "retr_band"), median=float(P["retr10"].dropna().median()))
R["h1_eff_quantiles"] = [float(x) for x in P["eff10"].dropna().quantile([.05, .25, .5, .75, .95])]
R["h1_eff_speed_corr"] = float(P[["eff10", "speed10"]].astype(float).corr().iloc[0, 1])
R["h1_eff_retr_corr"] = float(P[["eff10", "retr10"]].astype(float).corr(method="spearman").iloc[0, 1])
log("  cuts", CUTS)
log("  classes 10", R["h1_classes"]["10"])
log("  x CR-CT class", R["h1_by_ct_class"])
log("  retrace", R["retrace"], "corr eff~speed", R["h1_eff_speed_corr"], "eff~retr", R["h1_eff_retr_corr"])

log("4. H2 funnel, snap band, HOLD / FILL (no outcome)")
VARIANTS = {"primary": (W.BRK, W.PB)}
VARIANTS.update({f"brk{int(x)}": (x, W.PB) for x in W.BRK_READ})
VARIANTS.update({f"pb{int(x)}": (W.BRK, x) for x in W.PB_READ})
H2 = {}
R["h2"] = {}
for name, (brk, pb) in VARIANTS.items():
    H = W.h2_events(D, S1, P, brk, pb)
    H2[name] = H
    tr = H[(H["stage"] == "trade") & ~H["dup"]]
    R["h2"][name] = dict(brk=brk, pb=pb, stages=H["stage"].value_counts().to_dict(), dup=int(H["dup"].sum()),
                         classes=counts(tr, "cls"), classes_q=counts(tr, "cls_q"),
                         light=counts(tr.assign(lt=tr["light"].map({True: "light", False: "heavy"})), "lt"))
    log(" ", name, R["h2"][name])
H = H2["primary"]
br = H[H["stage"] != "no_break"]
tr = H[(H["stage"] == "trade") & ~H["dup"]].copy()
R["snap"] = dict(
    n_breaks=int(len(br)), bands_median=float(br["nb"].median()),
    snap_index=br["snap"].value_counts().sort_index().to_dict(),
    zero_volume_snap=int(sum(min(b) == 0 for b in br["band_vol"])),
    snap_over_median_band=[float(x) for x in pd.Series([min(b) / np.median(b) if np.median(b) > 0 else np.nan
                                                        for b in br["band_vol"]]).quantile([.1, .5, .9])],
    trade_snap_index=tr["snap"].value_counts().sort_index().to_dict(),
    trade_by_cls_snap=tr.groupby(["cls", "snap"]).size().rename("n").reset_index().to_dict("records"),
    pb_ext_quantiles=[float(x) for x in tr["pb_ext"].quantile([.1, .5, .9])],
    P_quantiles=[float(x) for x in tr["P"].quantile([.1, .5, .9])],
    turn_minus_pullback_bar=[float(x) for x in pd.Series(
        [int(e.t) - (int(S1[e.d]["sec"][int(e.r)] // 60) - W.PRE) for e in tr.itertuples()]).quantile([.1, .5, .9])],
    turn_bar_quantiles=[float(x) for x in tr["t"].quantile([.1, .5, .9])],
    by_family=counts(tr, "fam"), by_b=counts(tr.assign(bb=tr["b"].astype(str)), "bb"),
    edge=tr["edge"].value_counts().to_dict())
log("  snap", R["snap"])

log("5. A/A on the random baseline (random same-minute entries only)")
BT = T.baseline_table(D)
R["baseline_res"] = BT["res"].value_counts().to_dict()
BT = BT[BT["res"].isin(["win", "loss"])].copy()
BT["w"] = (BT["res"] == "win").astype(float)
cs = P[P["h110"].isin(["CLEAN", "CHOPPY"])]
R["aa_h1"] = W.aa(list(zip(cs["j"].astype(int), cs["b"].astype(int), cs["edge"])), BT, SESS)
R["aa_h2"] = W.aa(list(zip(tr["t"].astype(int), tr["b"].astype(int), tr["edge"])), BT, SESS)
log("  A/A H1", R["aa_h1"])
log("  A/A H2", R["aa_h2"])

log("6. gate")


def gate(cl, a, b, aa_):
    x, y = cl.get(a, dict(n=0, sessions=0)), cl.get(b, dict(n=0, sessions=0))
    ok = (min(x["n"], y["n"]) >= W.MIN_CELL and min(x["sessions"], y["sessions"]) >= W.MIN_SESS
          and abs(aa_["mean_diff_pp"]) <= 1.0)
    return dict(a=x, b=y, aa_mean_pp=aa_["mean_diff_pp"], passed=bool(ok))


R["gate"] = dict(H1=gate(R["h1_classes"]["10"], "CLEAN", "CHOPPY", R["aa_h1"]),
                 H2=gate(R["h2"]["primary"]["classes"], "HOLD", "FILL", R["aa_h2"]))
log("GATE", R["gate"])

keep = [c for c in P.columns if not c.startswith("why")]
P[keep].assign(d=P["d"].astype(str)).to_parquet(W.OUT / "step0_h1_events.parquet")
for name, Hx in H2.items():
    Hx.drop(columns=["band_vol"]).assign(d=Hx["d"].astype(str)).to_parquet(W.OUT / f"step0_h2_{name}.parquet")
(W.OUT / "step0.json").write_text(json.dumps(jd(R), indent=1))
log("done")
