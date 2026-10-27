#!/usr/bin/env python3
"""CR-CS Step 0 — counts, readout cells, A/A. NO fade / continuation outcome is computed. READ-ONLY, train only.

Spec: specs/CR-CS-range-stop-run.md C14. Writes scripts/logs/cr_cs/step0.json and step0_trades.parquet.
Only the A/A reads outcomes, and only of random same-minute entries (CR-CP C13).
"""
from __future__ import annotations

import json
import time as _time
from collections import Counter

import numpy as np
import pandas as pd

import cr_cs_lib as S
from cr_cs_lib import C

T0 = _time.time()
R: dict = {}


def log(*a):
    print(f"[{_time.time() - T0:7.1f}s]", *a, flush=True)


def qtiles(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    if not x.size:
        return None
    return dict(n=int(x.size), p5=float(np.percentile(x, 5)), p25=float(np.percentile(x, 25)),
                p50=float(np.median(x)), p75=float(np.percentile(x, 75)), p95=float(np.percentile(x, 95)))


# ---------------------------------------------------------------------------------------------- caches, universe
log("caches + universe (C1)")
m1, D = S.load()
ts = pd.to_datetime(m1["minute_utc"], utc=True)
R["m1_cache"] = dict(rows=int(len(m1)), first=str(ts.min()), last=str(ts.max()))
assert ts.min() >= C.FLOOR and ts.max() < C.CAP
R["universe"] = dict(sessions=len(D["SESS"]), med_im=D["MED_IM"], funnel=D["funnel"],
                     swing=qtiles(list(D["SW"].values())))
log("  ", R["m1_cache"], len(D["SESS"]), D["MED_IM"])

RG = S.ranges(D)
R["ranges"] = dict(n=len(RG), sessions=len({d for d, _ in RG}), width=qtiles([rg["width"] for _, rg in RG]),
                   q_hour_pt=dict(Counter(int((390 + rg["q"]) // 60) for _, rg in RG)))
assert len(RG) == 470, len(RG)                       # CR-CP Step 0 v2 primary
log("  ranges", R["ranges"]["n"], "sessions", R["ranges"]["sessions"])


def summary(EV, T, n_drop):
    br = EV[EV["dir"] != 0] if len(EV) else EV
    out = dict(
        breaks=int(len(br)), breaks_up=int((br["dir"] == 1).sum()), breaks_down=int((br["dir"] == -1).sum()),
        both_edges_bars=int((EV["kind"] == "both edges").sum()) if len(EV) else 0,
        ranges_with_a_break=int(br["ri"].nunique()) if len(br) else 0,
        burst=int(br["burst"].sum()), no_burst=int((~br["burst"]).sum()),
        burst_failure=int((br["burst"] & (br["kind"] == "failure")).sum()),
        burst_breakout=int((br["burst"] & (br["kind"] == "breakout")).sum()),
        no_burst_failure=int((~br["burst"] & (br["kind"] == "failure")).sum()),
        no_burst_breakout=int((~br["burst"] & (br["kind"] == "breakout")).sum()),
        overlap_dropped=int(n_drop), fades=int(len(T)), fade_sessions=int(T["d"].nunique()) if len(T) else 0,
        fades_up_sweep=int((T["dir"] == 1).sum()) if len(T) else 0,
        fades_down_sweep=int((T["dir"] == -1).sum()) if len(T) else 0,
    )
    return out


# ------------------------------------------------------------------------------------------------ primary + grid
R["grid"] = {}
for nm, sp in S.VARIANTS.items():
    EV = S.events(D, RG, sp)
    T, nd = S.trades(D, EV)
    R["grid"][nm] = summary(EV, T, nd)
    log(f"  {nm:10s}", R["grid"][nm])
    if nm == "primary":
        EVp, Tp = EV, T

br = EVp[EVp["dir"] != 0]
R["primary_distributions"] = dict(
    break_size=qtiles(br["size"]), vol_ratio_all_breaks=qtiles(br["vratio"]),
    vol_ratio_fades=qtiles(Tp["vratio"]), sweep_bars_fades=dict(Counter((Tp["x"] - Tp["b"] + 1).astype(int))),
    reentry_lag_fades=dict(Counter((Tp["r"] - Tp["b"]).astype(int))),
    sweep_extreme_beyond_edge_fades=qtiles(np.where(Tp["dir"] == 1, Tp["ext"] - Tp["hi"], Tp["lo"] - Tp["ext"])),
    stop_distance_fades=qtiles(np.where(Tp["dir"] == 1, Tp["ext"] + 1 - Tp["px"], Tp["px"] - (Tp["ext"] - 1))),
    dshare_fades=qtiles(Tp["dshare"]), break_minus_q=qtiles(br["b"] - br["q"]),
    ranges_per_session_with_fade=dict(Counter(Counter(Tp["d"]).values())),
)
R["breakout_readout_trades"] = int(((br["kind"] == "breakout") & br["burst"] & br["bo_j"].notna()).sum())
log("  distributions", R["primary_distributions"])

# ------------------------------------------------------------------------------------------------ split counts
LV = C.load_levels()
Tp = S.tag_levels(Tp, LV)
med_ds = float(Tp["dshare"].median())
Tp["dshare_half"] = np.where(Tp["dshare"] >= med_ds, "top half", "bottom half")
Tp["sweep_dir"] = np.where(Tp["dir"] == 1, "up sweep (short)", "down sweep (long)")
R["dshare_median"] = med_ds
R["splits"] = {col: {str(k): int(v) for k, v in Tp[col].value_counts().items()}
               for col in ("sweep_dir", "dshare_half", "lv_pos", "lv_kind", "hour", "edge")}
log("  splits", R["splits"])
Tp.to_parquet(S.OUT / "step0_trades.parquet")
EVp.to_parquet(S.OUT / "step0_events.parquet")

# ------------------------------------------------------------------------------------------------ A/A (C10)
log("A/A on the random baseline (random same-minute entries only)")
BT = C.baseline_table(D)
BT = BT[BT["res"].isin(["win", "loss"])].copy()
BT["w"] = (BT["res"] == "win").astype(float)
keys = list(zip(Tp["r"].astype(int), Tp["h"].astype(int), Tp["edge"]))
rng = np.random.default_rng(C.SEED_AA)
sess = np.array(D["SESS"], dtype=object)
grp = {k: g for k, g in BT.groupby(["m", "h", "edge"])}
diffs, excl = [], []
for _ in range(C.N_AA):
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
R["aa"] = dict(n_keys=len(keys), n_splits=C.N_AA, mean_diff_pp=100 * float(np.nanmean(diffs)),
               median_diff_pp=100 * float(np.nanmedian(diffs)), sd_diff_pp=100 * float(np.nanstd(diffs)),
               p5_pp=100 * float(np.nanpercentile(diffs, 5)), p95_pp=100 * float(np.nanpercentile(diffs, 95)),
               mean_excluded_share=float(np.mean(excl)), baseline_n=int(len(BT)),
               baseline_edge_share=float((BT["edge"] == "EDGE").mean()))
log("  A/A", R["aa"])

g = R["grid"]["primary"]
R["gate"] = dict(fades=g["fades"], sessions=g["fade_sessions"], aa_mean_pp=R["aa"]["mean_diff_pp"],
                 passed=bool(g["fades"] >= 40 and g["fade_sessions"] >= 30 and abs(R["aa"]["mean_diff_pp"]) <= 1.0))
log("  gate", R["gate"])
R["elapsed_s"] = _time.time() - T0
(S.OUT / "step0.json").write_text(json.dumps(R, indent=1, default=str))
log("wrote", S.OUT / "step0.json")
