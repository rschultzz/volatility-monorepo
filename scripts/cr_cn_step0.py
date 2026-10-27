"""CR-CN Step 0 — CR-CL Step 0 on R-a, plus the quote lag and the 2024 noise band. READ-ONLY. No group outcome.

Spec: specs/CR-CN-skew-g1down-ra.md (K9, N1-N5, N9, A1). Period `ra` only (2024-04-03 .. 2024-12-31); 2026 train, 2025
and the holdout are never loaded.

  1. parity (A1):  under the default the CR-CL module resolves PERIOD 'y26' and OUT logs/cr_cl (checked in a
                   subprocess, nothing loaded); no "y26" literal remains outside the default.
  2. CR-CL Step 0: scripts/cr_cl_step0.py unchanged, run under CR_CL_PERIOD=ra -> scripts/logs/cr_cn/step0.json
                   (funnel, cells, coverage, EDGE, terciles, GEX counts, A/A on baseline legs only).
  3. quote lag (N3): label - quote_date on the loaded 0DTE PM RTH rows; share dropped (> 5 min); lag at e on legs.
  4. noise (N4):   the CR-CK C6 band measured on R-a (code copied from scripts/cr_ck_stickiness.py), vs the shipped
                   NOISE_BAND_PP. Report only; the band is not changed and no count under it is computed.
Writes scripts/logs/cr_cn/step0_extra.json.
"""
from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ------------------------------------------------------------------------------------------------ 1. parity (A1)
chk = subprocess.run([sys.executable, "-c", "import cr_cl_lib as C; print(C.PERIOD, C.OUT.name)"], cwd=HERE,
                     env={k: v for k, v in os.environ.items() if k != "CR_CL_PERIOD"}, capture_output=True, text=True)
assert chk.returncode == 0 and chk.stdout.split()[-2:] == ["y26", "cr_cl"], (chk.stdout, chk.stderr)
for f in ("cr_cl_lib.py", "cr_cl_step0.py"):
    lines = [ln for ln in (HERE / f).read_text().splitlines() if '"y26"' in ln]
    assert all("CR_CL_PERIOD" in ln or "PERIOD ==" in ln for ln in lines), (f, lines)

# ------------------------------------------------------------------------------------------- 2. CR-CL Step 0 on ra
os.environ["CR_CL_PERIOD"] = "ra"
sys.path.insert(0, str(HERE))
G = runpy.run_path(str(HERE / "cr_cl_step0.py"), run_name="__main__")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import cr_cj_lib as L  # noqa: E402
import cr_cl_lib as C  # noqa: E402
from packages.shared import skew_expected as se  # noqa: E402

assert C.PERIOD == "ra" and C.OUT.name == "cr_cn"
D, LEGS = G["D"], G["LEGS"]
assert min(D["SESS"]) >= L.PERIODS["ra"][0] and max(D["SESS"]) <= L.PERIODS["ra"][1]
X = {"period": "ra", "sessions": len(D["SESS"]), "first": str(D["SESS"][0]), "last": str(D["SESS"][-1]),
     "sessions_missing": [str(d) for d in L.sessions("ra") if d not in D["SESS"]]}

# --------------------------------------------------------------------------------------------- 3. quote lag (N3)
sk = L.load_skew("ra")
lab = pd.to_datetime(sk["snap_shot_date"], utc=True)
qts = pd.to_datetime(sk["quote_date"], utc=True, errors="coerce")
lag = (lab - qts).dt.total_seconds()
X["quote_lag_all_rows_s"] = dict(n=int(len(lag)), p5=float(np.nanpercentile(lag, 5)), p50=float(np.nanmedian(lag)),
                                 p95=float(np.nanpercentile(lag, 95)), share_gt_300=round(float((lag.abs() > 300).mean()), 4),
                                 share_unparsed=round(float(qts.isna().mean()), 4))


def lag_at(d, i):
    m = D["MIN"][d]
    j = m["row"][i]
    if j < 0:
        return np.nan
    r = m["rows"].iloc[j]
    return (pd.Timestamp(r["snap_shot_date"]) - pd.Timestamp(r["quote_date"])).total_seconds() if pd.notna(r["quote_date"]) else np.nan


W = LEGS[12.0]
P = W[~W["late"]]
X["quote_lag_at_e_s"] = {}
for name, sub in (("primary_ratio_defined", P[P["reason"] == "ok"]),
                  ("G1-down", P[C.group_mask(P, "G1", -1)]), ("baseline_down", P[(P["grp"] == "base") & (P["dir"] == -1)])):
    v = np.array([lag_at(r.d, r.e) for r in sub.itertuples()], float)
    X["quote_lag_at_e_s"][name] = dict(n=int(np.isfinite(v).sum()), p50=float(np.nanmedian(v)) if len(v) else None,
                                       p95=float(np.nanpercentile(v, 95)) if len(v) else None)
# quote minute of the snapshot at e vs the leg's extreme bar (end_i = e - 1): bars behind
qbehind = np.array([(r.e - 1) - (D["MIN"][r.d]["qidx"][r.e] - 1) for r in P[P["reason"] == "ok"].itertuples()], float)
X["quote_minute_behind_extreme_bar"] = {str(int(k)): int(v) for k, v in pd.Series(qbehind).value_counts().sort_index().items()}

# ---------------------------------------------------------------------------- 4. 2024 noise band (N4, CR-CK C6)
DOM_HI = L.NB - 15
band_rows = []
for d in D["SESS"]:
    put = D["MIN"][d]["put"]
    for Lm in se.NOISE_BAND_LENGTHS:
        for s in range(C.DOM_LO, DOM_HI - Lm + 1, 5):
            e = s + Lm
            if np.isfinite(put[s]) and np.isfinite(put[e]):
                band_rows.append((d, Lm, s // 30, put[e] - put[s]))
B = pd.DataFrame(band_rows, columns=["d", "L", "b", "dend"])
sd = B.groupby(["b", "L"]).agg(n=("dend", "size"), sd_end=("dend", "std")).reset_index()
band24, ratio = {}, {}
for b in range(1, 13):
    vals, last = [], None
    for Lm in se.NOISE_BAND_LENGTHS:
        cell = sd[(sd["b"] == b) & (sd["L"] == Lm)]
        v = float(cell["sd_end"].iloc[0]) if len(cell) and int(cell["n"].iloc[0]) >= 30 else None
        last = v if v is not None else last
        vals.append(None if last is None else round(last, 3))
    band24[L.bucket_label(b)] = vals
    ratio[L.bucket_label(b)] = [None if v is None else round(v / s_, 2) for v, s_ in zip(vals, se.NOISE_BAND_PP[b])]
X["noise_band_lengths_min"] = list(se.NOISE_BAND_LENGTHS)
X["noise_band_2024_pp"] = band24
X["noise_band_shipped_pp"] = {L.bucket_label(b): list(v) for b, v in se.NOISE_BAND_PP.items()}
X["noise_band_ratio_2024_over_shipped"] = ratio
pr = np.array([r for b in range(1, 9) for r in ratio[L.bucket_label(b)] if r is not None])
X["noise_ratio_summary_07_00_to_10_30"] = dict(p25=float(np.percentile(pr, 25)), p50=float(np.median(pr)),
                                               p75=float(np.percentile(pr, 75)))
# put-skew level on R-a (context for the band; vol pts)
X["put_skew_level_pp"] = dict(p50=float(np.nanmedian(np.concatenate([D["MIN"][d]["put"] for d in D["SESS"]]))))

(C.OUT / "step0_extra.json").write_text(json.dumps(X, indent=2, default=str))
print(json.dumps(X, indent=1, default=str))
