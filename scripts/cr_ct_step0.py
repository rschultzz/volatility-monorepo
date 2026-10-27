#!/usr/bin/env python3
"""CR-CT Step 0 — universe, levels, touches, tercile cuts, FAST / SLOW / NEITHER counts, A/A. NO touch outcome is
computed (the only races run are the random-baseline entries for the A/A, C12). READ-ONLY, train only.

Spec: specs/CR-CT-approach-speed.md. Writes scripts/logs/cr_ct/step0.json and step0_events.parquet (features only).
Gate (kickoff): FAST and SLOW each >= 40 primary touches on >= 30 sessions and |A/A mean| <= 1 pp.
"""
from __future__ import annotations

import json
import time as _time
from collections import Counter

import numpy as np
import pandas as pd

import cr_ct_lib as T
import cr_cp_lib as C

T0 = _time.time()
R: dict = {}


def log(*a):
    print(f"[{_time.time() - T0:7.1f}s]", *a, flush=True)


def jd(x):
    return json.loads(json.dumps(x, default=str))


log("1. universe")
D = T.build()
SESS = D["SESS"]
R["funnel"] = D["funnel"]
R["n_sessions"] = len(SESS)
R["im_median"] = float(np.median(list(D["IM"].values())))
R["im_lt25"] = [str(d) for d in SESS if D["IM"][d] < 25]
R["priors_available"] = [str(d) for d in D["PRI"] if d < T.D_FROM]
for row in D["funnel"]:
    log("  ", row[:3])
log("  sessions", len(SESS), "IM median", R["im_median"])

log("2. 1-s bars")
S1 = T.load_1s(D)
R["sessions_without_1s"] = [str(d) for d in SESS if d not in S1]

log("3. levels and touches (features only)")
ev, why = T.events(D, S1)
R["level_missing"] = {k: dict(n=len(v), detail=v) for k, v in why.items()}
lvc = ev.drop_duplicates(["d", "fam", "L"]).groupby(["fam", "d"]).size()
R["levels_per_session_touched"] = {f: dict(median=float(g.median()), q25=float(g.quantile(.25)),
                                           q75=float(g.quantile(.75))) for f, g in lvc.groupby(level=0)}
lev_all = {f: [] for f in T.FAM_PRIORITY}
for d in SESS:
    lv = T.levels(D, d, {})
    c = Counter(x["fam"] for x in lv)
    for f in lev_all:
        lev_all[f].append(c.get(f, 0))
R["levels_per_session"] = {f: dict(median=float(np.median(v)), q25=float(np.percentile(v, 25)),
                                   q75=float(np.percentile(v, 75)), zero_share=float(np.mean(np.array(v) == 0)))
                           for f, v in lev_all.items()}
log("  levels / session", R["levels_per_session"])

ev["no1s"] = ev["k1s"].isna()
tab = ev.groupby(["touch", "fam"]).agg(n=("d", "size"), sessions=("d", "nunique"), dup=("dup", "sum"),
                                         no1s=("no1s", "sum")).reset_index()
R["touches"] = tab.to_dict("records")
log("  touches\n", tab.to_string(index=False))

P = ev[(ev["touch"] == 1) & ~ev["dup"] & ~ev["no1s"]].copy()      # primary event set (C5, C6)
S2 = ev[(ev["touch"] == 2) & ~ev["dup"] & ~ev["no1s"]].copy()
R["primary_events"] = dict(n=len(P), sessions=int(P["d"].nunique()),
                           by_family=P["fam"].value_counts().to_dict(), by_kind=P["kind"].value_counts().to_dict(),
                           by_b=P["b"].value_counts().to_dict(), first_hour=int(P["first_hour"].sum()),
                           edge=P["edge"].value_counts().to_dict())
R["undefined_reasons"] = {w: P[f"why{w}"].fillna("ok").value_counts().to_dict()
                          for w in [str(T.W_PRIMARY)] + [str(x) for x in T.W_READ] + ["A7"]}
log("  primary events", R["primary_events"])
log("  undefined", R["undefined_reasons"])

log("4. tercile cuts (train first touches, no outcome)")
CUTS = {}
for w in [str(T.W_PRIMARY)] + [str(x) for x in T.W_READ] + ["A7"]:
    c, n = T.cuts(P, f"speed{w}", f"relvol{w}")
    CUTS[w] = dict(cuts=c, n=n)
R["cuts"] = CUTS
log("  cuts", CUTS)

for w in CUTS:
    P[f"cls{w}"] = T.classify(P, f"speed{w}", f"relvol{w}", CUTS[w]["cuts"])
    S2[f"cls{w}"] = T.classify(S2, f"speed{w}", f"relvol{w}", CUTS[w]["cuts"])


def cell_counts(df, col):
    g = df.groupby(col)
    return {k: dict(n=int(len(x)), sessions=int(x["d"].nunique())) for k, x in g}


R["classes"] = {w: cell_counts(P, f"cls{w}") for w in CUTS}
R["classes_by_family"] = {f: cell_counts(g, "cls10") for f, g in P.groupby("fam")}
R["classes_by_b"] = {str(b): cell_counts(g, "cls10") for b, g in P.groupby("b")}
R["classes_by_first_hour"] = {str(h): cell_counts(g, "cls10") for h, g in P.groupby("first_hour")}
R["classes_second_touch"] = cell_counts(S2, "cls10")
R["speed_alone"] = dict(top=int((P["speed10"] >= CUTS["10"]["cuts"][1]).sum()),
                        bottom=int((P["speed10"] <= CUTS["10"]["cuts"][0]).sum()))
R["volume_alone"] = dict(heavy=int((P["relvol10"] >= 1).sum()), light=int((P["relvol10"] < 1).sum()))
R["relvol10_quantiles"] = [float(x) for x in P["relvol10"].dropna().quantile([.1, .25, .5, .75, .9])]
R["bsg"] = dict(flagged=int(P["bsg"].sum()), n=len(P), pullbacks=P["n_pullbacks"].value_counts().sort_index().to_dict())
log("  classes (10 min)", R["classes"]["10"])
log("  by family", R["classes_by_family"])
log("  second touch", R["classes_second_touch"])
log("  BSG", R["bsg"])

log("5. strong lines (readout tags, no outcome)")
LV = C.load_levels()
P = T.tag_strong(P, LV)
R["strong"] = P.groupby(["strong", "a8"]).size().rename("n").reset_index().to_dict("records")
log("  strong", R["strong"])

log("6. A/A on the random baseline (random same-minute entries only)")
BT = T.baseline_table(D)
R["baseline_res"] = BT["res"].value_counts().to_dict()
BT = BT[BT["res"].isin(["win", "loss"])].copy()
BT["w"] = (BT["res"] == "win").astype(float)
FS = P[P["cls10"].isin(["FAST", "SLOW"])]
keys = list(zip(FS["j"].astype(int), FS["b"].astype(int), FS["edge"]))
rng = np.random.default_rng(T.SEED_AA)
sess = np.array(SESS, dtype=object)
grp = {k: g for k, g in BT.groupby(["m", "h", "edge"])}
diffs, excl = [], []
for _ in range(T.N_AA):
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
R["aa"] = dict(n_splits=T.N_AA, keys=len(keys), mean_diff_pp=100 * float(np.nanmean(diffs)),
               median_diff_pp=100 * float(np.nanmedian(diffs)), sd_diff_pp=100 * float(np.nanstd(diffs)),
               p5_pp=100 * float(np.nanpercentile(diffs, 5)), p95_pp=100 * float(np.nanpercentile(diffs, 95)),
               mean_excluded_share=float(np.mean(excl)), baseline_win_rate=float(BT["w"].mean()),
               baseline_n=int(len(BT)), baseline_edge_share=float((BT["edge"] == "EDGE").mean()))
log("  A/A", R["aa"])

f, s = R["classes"]["10"].get("FAST", dict(n=0, sessions=0)), R["classes"]["10"].get("SLOW", dict(n=0, sessions=0))
passed = (f["n"] >= T.MIN_CELL and s["n"] >= T.MIN_CELL and f["sessions"] >= T.MIN_SESS
          and s["sessions"] >= T.MIN_SESS and abs(R["aa"]["mean_diff_pp"]) <= 1.0)
R["gate"] = dict(fast=f, slow=s, aa_mean_pp=R["aa"]["mean_diff_pp"], passed=bool(passed))
log("GATE", R["gate"])

keep = ["d", "fam", "kind", "L", "known", "touch", "j", "b", "arm_bar", "k1s", "edge", "first_hour", "dup", "no1s",
        "bsg", "n_pullbacks"] + [c for c in ev.columns if c.startswith(("speed", "relvol", "why"))]
ev[keep].assign(d=ev["d"].astype(str)).to_parquet(T.OUT / "step0_events.parquet")
(T.OUT / "step0.json").write_text(json.dumps(jd(R), indent=1))
log("done")
