#!/usr/bin/env python3
"""CR-CU Step 0 — universe after exclusions, levels, touches, FAST / SLOW / NEITHER counts on CR-CT's frozen cuts, A/A.
NO touch outcome is computed (the only races run are the random-baseline entries for the A/A, U9). READ-ONLY,
2023-05-01 .. 2024-04-02 only.

Spec: specs/CR-CU-second-touch-2023.md. Writes scripts/logs/cr_cu/step0.json and step0_events.parquet (features only).
Gate (kickoff): second-touch FAST and SLOW each >= 40 on >= 30 sessions and |A/A mean| <= 1 pp.
"""
from __future__ import annotations

import json
import time as _time
from collections import Counter

import numpy as np

import cr_cu_lib as U
import cr_ct_lib as T

T0 = _time.time()
R: dict = {}


def log(*a):
    print(f"[{_time.time() - T0:7.1f}s]", *a, flush=True)


def jd(x):
    return json.loads(json.dumps(x, default=str))


R["frozen_checked_against"] = U.check_frozen()
R["cuts"] = U.CUTS
log("frozen constants", R["frozen_checked_against"])

log("1. universe")
D = U.build()
SESS = D["SESS"]
R["sources"] = D["sources"]
assert set(D["sources"]) == {"databento"}, D["sources"]
R["funnel"] = D["funnel"]
R["n_sessions"] = len(SESS)
R["im_median"] = float(np.median(list(D["IM"].values())))
R["im_q"] = [float(x) for x in np.percentile(list(D["IM"].values()), [10, 50, 90])]
R["n_priors"] = len(D["PRI"])
for row in D["funnel"]:
    log("  ", row[:3])
log("  sessions", len(SESS), "IM p10/50/90", R["im_q"], "priors", len(D["PRI"]))

log("2. levels and touches (features only)")
ev, why = U.events(D)
R["level_missing"] = {k: dict(n=len(v), reasons=dict(Counter(x[1] for x in v))) for k, v in why.items()}
lev_all = {f: [] for f in T.FAM_PRIORITY}
for d in SESS:
    c = Counter(x["fam"] for x in U.levels(D, d, {}))
    for f in lev_all:
        lev_all[f].append(c.get(f, 0))
R["levels_per_session"] = {f: dict(median=float(np.median(v)), q25=float(np.percentile(v, 25)),
                                   q75=float(np.percentile(v, 75)), zero_share=float(np.mean(np.array(v) == 0)))
                           for f, v in lev_all.items()}
log("  missing", R["level_missing"])
log("  levels / session", R["levels_per_session"])
tab = ev.groupby(["touch", "fam"]).agg(n=("d", "size"), sessions=("d", "nunique"), dup=("dup", "sum")).reset_index()
R["touches"] = tab.to_dict("records")
log("  touches\n", tab.to_string(index=False))

P = ev[(ev["touch"] == 1) & ~ev["dup"]].copy()
S2 = ev[(ev["touch"] == 2) & ~ev["dup"]].copy()


def cell_counts(df, col):
    return {k: dict(n=int(len(x)), sessions=int(x["d"].nunique())) for k, x in df.groupby(col)}


for nm, E in (("second", S2), ("first", P)):
    R[f"{nm}_events"] = dict(n=len(E), sessions=int(E["d"].nunique()), by_family=E["fam"].value_counts().to_dict(),
                             by_b=E["b"].value_counts().to_dict(), edge=E["edge"].value_counts().to_dict(),
                             first_hour=int(E["first_hour"].sum()), mon_tue_expiry=int(E["mon_tue_expiry"].sum()),
                             undefined={w: E[f"why{w}"].fillna("ok").value_counts().to_dict() for w in U.CUTS})
    R[f"{nm}_classes"] = {w: cell_counts(E, f"cls{w}") for w in U.CUTS}
    R[f"{nm}_relvol10_q"] = [float(x) for x in E["relvol10"].dropna().quantile([.1, .25, .5, .75, .9])]
    R[f"{nm}_speed10_share_vs_cuts"] = dict(
        below_lo=float((E["speed10"] <= U.CUTS["10"][0]).mean()), above_hi=float((E["speed10"] >= U.CUTS["10"][1]).mean()))
    log(f"  {nm} touches", R[f"{nm}_events"])
    log(f"  {nm} classes", R[f"{nm}_classes"])
    log(f"  {nm} relvol10 q", R[f"{nm}_relvol10_q"], "speed tails", R[f"{nm}_speed10_share_vs_cuts"])
R["second_classes_by_family"] = {f: cell_counts(g, "cls10") for f, g in S2.groupby("fam")}
R["second_classes_no_mon_tue"] = cell_counts(S2[~S2["mon_tue_expiry"]], "cls10")
log("  second by family", R["second_classes_by_family"])
log("  second without expiry-week Mon/Tue", R["second_classes_no_mon_tue"])

log("3. A/A on the random baseline (random same-minute entries only; keys = second-touch FAST u SLOW)")
BT = T.baseline_table(D)
R["baseline_res"] = BT["res"].value_counts().to_dict()
BT = BT[BT["res"].isin(["win", "loss"])].copy()
BT["w"] = (BT["res"] == "win").astype(float)
FS = S2[S2["cls10"].isin(["FAST", "SLOW"])]
keys = list(zip(FS["j"].astype(int), FS["b"].astype(int), FS["edge"]))
rng = np.random.default_rng(U.SEED_AA)
sess = np.array(SESS, dtype=object)
grp = {k: g for k, g in BT.groupby(["m", "h", "edge"])}
diffs, excl = [], []
for _ in range(U.N_AA):
    perm = rng.permutation(len(sess))
    A = set(sess[perm[: len(sess) // 2]])
    vals, nx = [], 0
    for k in keys:
        g = grp.get(k)
        if g is None:
            nx += 1
            continue
        inA = g["d"].isin(A).to_numpy()
        ga, gb = g[inA], g[~inA]
        if not len(ga) or not len(gb):
            nx += 1
            continue
        vals.append(float(ga["w"].iloc[rng.integers(len(ga))]) - float(gb["w"].mean()))
    diffs.append(float(np.mean(vals)) if vals else np.nan)
    excl.append(nx / max(len(keys), 1))
diffs = np.array(diffs)
R["aa"] = dict(n_splits=U.N_AA, keys=len(keys), mean_diff_pp=100 * float(np.nanmean(diffs)),
               median_diff_pp=100 * float(np.nanmedian(diffs)), sd_diff_pp=100 * float(np.nanstd(diffs)),
               p5_pp=100 * float(np.nanpercentile(diffs, 5)), p95_pp=100 * float(np.nanpercentile(diffs, 95)),
               mean_excluded_share=float(np.mean(excl)), baseline_win_rate=float(BT["w"].mean()),
               baseline_n=int(len(BT)), baseline_edge_share=float((BT["edge"] == "EDGE").mean()))
log("  A/A", R["aa"])

c = R["second_classes"]["10"]
f, s = c.get("FAST", dict(n=0, sessions=0)), c.get("SLOW", dict(n=0, sessions=0))
passed = (f["n"] >= U.MIN_CELL and s["n"] >= U.MIN_CELL and f["sessions"] >= U.MIN_SESS
          and s["sessions"] >= U.MIN_SESS and abs(R["aa"]["mean_diff_pp"]) <= 1.0)
R["gate"] = dict(fast=f, slow=s, aa_mean_pp=R["aa"]["mean_diff_pp"], passed=bool(passed))
log("GATE", R["gate"])

ev.assign(d=ev["d"].astype(str)).to_parquet(U.OUT / "step0_events.parquet")
(U.OUT / "step0.json").write_text(json.dumps(jd(R), indent=1))
log("done")
