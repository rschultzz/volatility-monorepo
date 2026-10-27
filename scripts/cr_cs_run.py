#!/usr/bin/env python3
"""CR-CS run — the single outcome run. READ-ONLY, train only. Run once.

Primary (kickoff, C6, C9): on the 58 Step 0 fades, ±10 stop-free race win rate of the fade minus the continuation side
(same entry price and bar, other direction); session-cluster bootstrap 10,000 draws, seed 20260920, percentile 95 % CI;
B1 verdict; >= 40 resolved fades else UNDERPOWERED. Everything else is a readout (no verdict).
Writes scripts/logs/cr_cs/results.json and run_trades.parquet.
"""
from __future__ import annotations

import json
import time as _time
from collections import Counter

import numpy as np
import pandas as pd

import cr_cl_lib as CL
import cr_cs_lib as S
from cr_cs_lib import C

T0 = _time.time()
OUT = S.OUT / "results.json"
assert not OUT.exists(), "CR-CS is run once; results.json already exists"


def log(*a):
    print(f"[{_time.time() - T0:7.1f}s]", *a, flush=True)


m1, D = S.load()
SESS = D["SESS"]
SI = {d: i for i, d in enumerate(SESS)}
nS = len(SESS)
RNG = np.random.default_rng(C.SEED)
WT = CL.boot_weights(nS, C.N_BOOT, RNG)
ONE = np.ones((1, nS))
R: dict = dict(n_sessions=nS, n_boot=C.N_BOOT, seed=C.SEED)
RG = S.ranges(D)
assert len(RG) == 470


# ------------------------------------------------------------------------------------------------ helpers
def with_outcomes(T: pd.DataFrame, X: float) -> pd.DataFrame:
    """Fade = 'ch' (side h), continuation = 'op' (side -h), both from c[r] (C6 / C7)."""
    T = T.copy()
    T["side"] = T["h"]
    for nm, sg in (("ch", 1), ("op", -1)):
        T[f"{nm}_status"] = "fill"
        T[f"{nm}_j"], T[f"{nm}_px"] = T["r"], T["px"]
        T[f"{nm}_res"] = [CL.race(D["BARS"][d], int(r), float(px), sg * int(h), X)
                          for d, r, px, h in zip(T["d"], T["r"], T["px"], T["h"])]
    return T


def sess_counts(rows):
    n, w = np.zeros(nS), np.zeros(nS)
    for d, x in rows:
        n[SI[d]] += 1
        w[SI[d]] += x
    return n, w


def rate(Wt, n, w):
    with np.errstate(invalid="ignore", divide="ignore"):
        return (Wt @ w) / (Wt @ n)


def ci(arr):
    a = arr[np.isfinite(arr)]
    return [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))] if a.size else [None, None], int(a.size)


def verdict(lo, hi, n):
    if n < C.MIN_TRADES:
        return "UNDERPOWERED"
    return "PASS" if lo > 0 else ("REFUTED" if hi < 0 else "INCONCLUSIVE")


def resolved(T, nm):
    m = T[f"{nm}_res"].isin(["win", "loss"])
    return [(d, 1.0 if x == "win" else 0.0) for d, x in zip(T.loc[m, "d"], T.loc[m, f"{nm}_res"])]


def compare(T, with_verdict=False, single=False):
    """CR-CP run `compare`: chosen (fade) − opposite (continuation), each vs 50 %, true pairs."""
    nc, wc = sess_counts(resolved(T, "ch"))
    no, wo = sess_counts(resolved(T, "op"))
    out = dict(n=int(len(T)), chosen_resolved=int(nc.sum()), opposite_resolved=int(no.sum()),
               chosen_sessions=int((nc > 0).sum()),
               chosen_res=dict(Counter(T["ch_res"].dropna())), opposite_res=dict(Counter(T["op_res"].dropna())))
    if nc.sum() == 0:
        return out
    pc, bc = rate(ONE, nc, wc)[0], rate(WT, nc, wc)
    out["chosen_wr"] = float(pc)
    (lo, hi), nb = ci(bc - 0.5)
    out["chosen_minus_50_pp"] = dict(est=100 * (pc - 0.5), ci=[100 * lo, 100 * hi], draws=nb)
    if single or no.sum() == 0:
        return out
    po, bo = rate(ONE, no, wo)[0], rate(WT, no, wo)
    out["opposite_wr"] = float(po)
    (dlo, dhi), nb = ci(bc - bo)
    out["diff_pp"] = dict(est=100 * (pc - po), ci=[100 * dlo, 100 * dhi], draws=nb)
    if with_verdict:
        out["verdict"] = verdict(dlo, dhi, int(nc.sum()))
    return out


def mean_ci(rows):
    rows = [(d, v) for d, v in rows if v is not None and np.isfinite(v)]
    n, s = sess_counts(rows)
    if n.sum() == 0:
        return None
    (lo, hi), _ = ci(rate(WT, n, s))
    return dict(n=int(n.sum()), mean=float(rate(ONE, n, s)[0]), ci=[lo, hi])


def stop_race(B, r, px, h, stop_px, X=10.0):
    """C12 (mechanical, B6): stop at the sweep extreme + 1 vs +X, first touched from bar r + 1."""
    for j in range(r + 1, C.NB):
        fav = (B["h"][j] - px >= X) if h == 1 else (px - B["l"][j] >= X)
        adv = (B["l"][j] <= stop_px) if h == 1 else (B["h"][j] >= stop_px)
        if fav and adv:
            return "tie"
        if adv:
            return "stopped"
        if fav:
            return "target"
    return "neither"


# ------------------------------------------------------------------------------------------------ primary
log("primary schedule + outcomes")
EV = S.events(D, RG, S.PRIMARY)
T0_, nd = S.trades(D, EV)
assert len(T0_) == 58 and T0_["d"].nunique() == 41 and nd == 1, "schedule differs from Step 0"
T = with_outcomes(T0_, 10.0)
R["primary"] = compare(T, with_verdict=True)
log("PRIMARY", json.dumps(R["primary"]))

# B7 matched random baseline (C10)
log("random baseline (C10)")
BT = C.baseline_table(D)
BT = BT[BT["res"].isin(["win", "loss"])]
base_rows = [(d, (int(m), int(h), e), 1.0 if x == "win" else 0.0)
             for d, m, h, e, x in zip(BT["d"], BT["m"], BT["h"], BT["edge"], BT["res"])]
R["vs_random"] = {}
for nm, sg in (("ch", 1), ("op", -1)):
    rr = T[T[f"{nm}_res"].isin(["win", "loss"])]
    grp_rows = [(r.d, (int(r.r), sg * int(r.h), C.edge_tag(D["BARS"][r.d], int(r.r), float(r.px), sg * int(r.h),
                                                            include_j=True)),
                 1.0 if getattr(r, f"{nm}_res") == "win" else 0.0) for r in rr.itertuples()]
    M, _ = CL.strata_matrices(SESS, grp_rows, base_rows)
    pe, ncov, cov = CL.matched_diff(ONE, M)
    be, _, _ = CL.matched_diff(WT, M)
    (lo, hi), nb = ci(be)
    R["vs_random"][nm] = dict(est_pp=100 * float(pe[0]), ci_pp=[100 * lo, 100 * hi], covered=int(ncov[0]),
                              coverage=float(cov[0]), draws=nb)
log("vs random", R["vs_random"])

# ------------------------------------------------------------------------------------------------ readouts
log("readouts: ±20, +30/+60, MFE, stop")
R["race20"] = compare(with_outcomes(T0_, 20.0))
R["move"] = {}
for k in (30, 60):
    R["move"][f"+{k}"] = {nm: mean_ci([(r.d, CL.move_at(D["BARS"][r.d], int(r.r), float(r.px), sg * int(r.h), k))
                                       for r in T.itertuples()]) for nm, sg in (("fade", 1), ("continuation", -1))}
R["mfe_before_minus10"] = {}
for nm, sg in (("fade", 1), ("continuation", -1)):
    v = np.array([CL.mfe_before(D["BARS"][r.d], int(r.r), float(r.px), sg * int(r.h), 10.0) for r in T.itertuples()])
    R["mfe_before_minus10"][nm] = dict(n=int(v.size), p25=float(np.percentile(v, 25)), median=float(np.median(v)),
                                       p75=float(np.percentile(v, 75)), share_ge10=float(np.mean(v >= 10)),
                                       share_ge20=float(np.mean(v >= 20)))
st = []
for r in T.itertuples():
    stop_px = r.ext + 1.0 if r.h == -1 else r.ext - 1.0
    st.append(stop_race(D["BARS"][r.d], int(r.r), float(r.px), int(r.h), stop_px))
T["stop_res"] = st
R["stop_at_sweep_extreme_plus1"] = dict(
    label="mechanical readout (B6): stop at the sweep extreme + 1 pt vs +10",
    counts=dict(Counter(st)), share_stopped_first=float(np.mean(np.array(st) == "stopped")),
    target_share_of_resolved=mean_ci([(d, 1.0 if x == "target" else 0.0) for d, x in zip(T["d"], st)
                                      if x in ("target", "stopped")]))
log("race20", R["race20"].get("diff_pp"), "move", R["move"], "mfe", R["mfe_before_minus10"],
    "stop", R["stop_at_sweep_extreme_plus1"])

# breakout-direction trades (burst breakouts, entry c[b + 4], sweep direction)
bo = EV[(EV["dir"] != 0) & (EV["kind"] == "breakout") & EV["burst"].astype(bool) & EV["bo_j"].notna()].copy()
bo["r"], bo["h"] = bo["bo_j"].astype(int), bo["dir"].astype(int)
bo["px"] = [float(D["BARS"][d]["c"][j]) for d, j in zip(bo["d"], bo["r"])]
R["breakout_direction"] = compare(with_outcomes(bo, 10.0), single=True)
log("breakout direction", R["breakout_direction"])

# ------------------------------------------------------------------------------------------------ splits
log("splits")
LV = C.load_levels()
T = S.tag_levels(T, LV)
T["dshare_half"] = np.where(T["dshare"] >= T["dshare"].median(), "top half", "bottom half")
T["sweep_dir"] = np.where(T["dir"] == 1, "up sweep (short)", "down sweep (long)")
R["splits"] = {}
for col in ("sweep_dir", "dshare_half", "lv_pos", "lv_kind", "hour", "edge"):
    R["splits"][col] = {}
    for k, g in T.groupby(col):
        c = compare(g)
        n_ch = c.get("chosen_resolved", 0)
        tag = "descriptive (< 40)" if n_ch < C.MIN_TRADES else "readout"
        if col == "lv_kind" and k in ("put", "call") and n_ch < C.MIN_TRADES:
            tag = "not readable on train (< 40)"
        R["splits"][col][str(k)] = dict(status=tag, **c)
T.to_parquet(S.OUT / "run_trades.parquet")

# ------------------------------------------------------------------------------------------------ grid
log("readout grid")
R["grid"] = {}
for nm, sp in S.VARIANTS.items():
    if nm == "primary":
        continue
    Tv, _ = S.trades(D, S.events(D, RG, sp))
    R["grid"][nm] = compare(with_outcomes(Tv, 10.0))
    g = R["grid"][nm]
    log(f"  {nm:10s}", g.get("chosen_resolved"), g.get("diff_pp"), g.get("chosen_minus_50_pp"))

R["elapsed_s"] = _time.time() - T0
OUT.write_text(json.dumps(R, indent=1, default=str))
log("wrote", OUT)
