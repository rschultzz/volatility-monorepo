"""CR-CK Amendment A1 — flat-price time drift of 0DTE put skew. READ-ONLY, no outcomes, 2026 train only.

Spec: specs/CR-CK-skew-stickiness.md (A1.1-A1.7). Windows of 15/30/60 min on a 5-min start grid from 06:30, ending
<= 12:59 PT, kept when |ES move| < 0.05 %. Drift = trend change (C3). Compared with the C6 noise band; the 06:30
bucket's band is measured here (C6 method) for the comparison only. Writes scripts/logs/cr_ck/a1_drift.json.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import cr_cj_lib as L
from packages.shared import skew_expected as se

SEED = 20261026
RTH0 = 6 * 60 + 30
LAST = L.NB - 1                          # 12:59 PT, the last snapshot index
LENGTHS = (15, 30, 60)
FLAT_PCT = 0.05
MIN_N = 30
OUT = L.REPO_ROOT / "scripts" / "logs" / "cr_ck"

MIN = L.skew_minutes(L.load_skew("y26"))
_, BARS = L.front_bars(L.load_es("y26"))
SESS = [d for d in L.sessions("y26") if d in MIN and d in BARS]
PX = {d: L.es_close_at_snapshot(BARS, d, MIN) for d in SESS}

rows, band0 = [], []
for d in SESS:
    put, px = MIN[d]["put"], PX[d]
    for Lm in LENGTHS:
        for s in range(0, LAST - Lm + 1, 5):
            e = s + Lm
            if s < 30 and np.isfinite(put[s]) and np.isfinite(put[e]):
                band0.append((Lm, put[e] - put[s]))                 # A1.4: 06:30 band, C6 method
            if not (np.isfinite(px[s]) and np.isfinite(px[e])):
                continue
            r = 100.0 * (px[e] / px[s] - 1.0)
            if abs(r) >= FLAT_PCT:
                continue
            idx = np.arange(s, e + 1)
            ok = np.isfinite(put[idx])
            tr = se.trend_change_pp((idx[ok] + RTH0).tolist(), put[idx][ok].tolist(), s + RTH0, e + RTH0)
            if tr is None:
                continue
            dend = put[e] - put[s] if np.isfinite(put[s]) and np.isfinite(put[e]) else np.nan
            rows.append((d, s // 30, Lm, tr, dend, r))
W = pd.DataFrame(rows, columns=["d", "b", "L", "trend", "dend", "r"])
B0 = pd.DataFrame(band0, columns=["L", "dend"]).groupby("L")["dend"].std()


def band(b, Lm):
    if b == 0:
        return float(B0[Lm])
    return se.noise_band_pp(Lm, RTH0 + 30 * b)


def boot(g, n=2000):
    groups = [x["trend"].to_numpy() for _, x in g.groupby("d")]
    rng = np.random.default_rng(SEED)
    k = len(groups)
    mean, med = [], []
    for _ in range(n):
        v = np.concatenate([groups[i] for i in rng.integers(0, k, k)])
        mean.append(v.mean())
        med.append(np.median(v))
    return [round(float(np.percentile(mean, 2.5)), 3), round(float(np.percentile(mean, 97.5)), 3)], \
           [round(float(np.percentile(med, 2.5)), 3), round(float(np.percentile(med, 97.5)), 3)]


cells = []
for (b, Lm), g in W.groupby(["b", "L"]):
    bd = band(int(b), int(Lm))
    mean, med = float(g["trend"].mean()), float(g["trend"].median())
    ci_mean, ci_med = boot(g)
    cells.append(dict(bucket=L.bucket_label(int(b)), b=int(b), L=int(Lm), n=len(g), sessions=int(g["d"].nunique()),
                      mean=round(mean, 3), mean_ci=ci_mean, median=round(med, 3), median_ci=ci_med,
                      median_endpoint=round(float(g["dend"].median()), 3), band=round(bd, 3),
                      median_exceeds=bool(len(g) >= MIN_N and abs(med) > bd),
                      mean_exceeds=bool(len(g) >= MIN_N and abs(mean) > bd)))
C = pd.DataFrame(cells)
R = {"windows": len(W), "sessions": len(SESS), "band_0630": {int(k): round(float(v), 3) for k, v in B0.items()},
     "any_median_exceeds": bool(C["median_exceeds"].any()), "any_mean_exceeds": bool(C["mean_exceeds"].any()),
     "cells": cells}
(OUT / "a1_drift.json").write_text(json.dumps(R, indent=2))
print({k: v for k, v in R.items() if k != "cells"})
pd.set_option("display.width", 250)
print(C.drop(columns=["b"]).to_string(index=False))
