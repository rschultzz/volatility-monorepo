"""CR-CK Part 1 — the stickiness ratio on 2026 train. READ-ONLY, no outcomes.

Spec: specs/CR-CK-skew-stickiness.md (C1-C9). 2026 train only (CR-CJ loaders, period `y26`; R-a and 2025 are never
loaded). Uses packages/shared/skew_expected.py for every read. Writes scripts/logs/cr_ck/results.json.

  noise band (C6): SD of the endpoint change in put skew, by window length and 30-min start bucket
  samples (C7):    swing-12 legs; 20 random 15-min and 20 random 30-min windows per session
  reads (C3-C5):   trend change, sticky-strike expected, ratio when |ES move| >= 0.15 %, noise flag
"""
from __future__ import annotations

import importlib.util
import json
from datetime import date

import numpy as np
import pandas as pd

import cr_cj_lib as L
from packages.shared import skew_expected as se
from packages.shared.flow_features import swing_legs
from packages.shared.surface_compare import k_for_abs_delta

SEED = 20261026
DOM_LO, DOM_HI = 30, L.NB - 15            # snapshot minute index: start >= 07:00, end <= 12:45 PT
RTH0 = 6 * 60 + 30                        # minute of day of index 0
LEGSW = 12.0
BAND_MIN_N = 30                           # a band cell needs >= 30 windows
OUT = L.REPO_ROOT / "scripts" / "logs" / "cr_ck"
OUT.mkdir(parents=True, exist_ok=True)

_spec = importlib.util.spec_from_file_location("skew_callbacks", L.REPO_ROOT / "apps/web/modules/Skew/callbacks.py")
SK = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SK)

# ------------------------------------------------------------------------------------------------ load (y26 only)
MIN = L.skew_minutes(L.load_skew("y26"))
_, BARS = L.front_bars(L.load_es("y26"))
SESS = [d for d in L.sessions("y26") if d in MIN and d in BARS]
PX = {d: L.es_close_at_snapshot(BARS, d, MIN) for d in SESS}
R = {"sessions": len(SESS)}


def vols_at(d: date, i: int) -> dict | None:
    j = MIN[d]["row"][i]
    if j < 0:
        return None
    row = MIN[d]["rows"].iloc[j]
    return {n: float(row[f"vol{n}"]) for n in range(5, 100, 5)}


# ------------------------------------------------------------------------------------------- C6: noise band
band_rows = []
for d in SESS:
    put = MIN[d]["put"]
    for Lm in se.NOISE_BAND_LENGTHS:
        for s in range(DOM_LO, DOM_HI - Lm + 1, 5):
            e = s + Lm
            if np.isfinite(put[s]) and np.isfinite(put[e]):
                idx = np.arange(s, e + 1)
                ok = np.isfinite(put[idx])
                tr = np.nan
                if ok.sum() >= max(se.TREND_MIN_POINTS, se.TREND_MIN_COVERAGE * (Lm + 1)):
                    x, y = idx[ok].astype(float), put[idx][ok]
                    tr = np.polyfit(x, y, 1)[0] * Lm
                band_rows.append((d, Lm, s // 30, put[e] - put[s], tr))
B = pd.DataFrame(band_rows, columns=["d", "L", "b", "dend", "dtrend"])
sd = B.groupby(["b", "L"]).agg(n=("dend", "size"), sd_end=("dend", "std"), sd_trend=("dtrend", "std")).reset_index()
band = {}
for b in range(1, 13):
    vals, last = [], None
    for Lm in se.NOISE_BAND_LENGTHS:
        cell = sd[(sd["b"] == b) & (sd["L"] == Lm)]
        v = float(cell["sd_end"].iloc[0]) if len(cell) and int(cell["n"].iloc[0]) >= BAND_MIN_N else None
        last = v if v is not None else last              # unmeasured longer lengths carry the longest measured value
        vals.append(round(last, 3))
    band[b] = tuple(vals)
R["noise_band_pp"] = {L.bucket_label(b): v for b, v in band.items()}
R["noise_band_cells"] = [dict(bucket=L.bucket_label(int(r.b)), L=int(r.L), n=int(r.n), sd_end=round(float(r.sd_end), 3),
                              sd_trend=round(float(r.sd_trend), 3)) for r in sd.itertuples()]


# ----------------------------------------------------------------------------------------------- C7: samples
def read(d, s, e, kind, legdir=0):
    put, px = MIN[d]["put"], PX[d]
    v = vols_at(d, s)
    if v is None or not (np.isfinite(px[s]) and np.isfinite(px[e])) or not np.isfinite(put[s]):
        return None
    idx = np.arange(s, e + 1)
    ok = np.isfinite(put[idx])
    rd = se.stickiness_read((idx[ok] + RTH0).tolist(), put[idx][ok].tolist(), int(s + RTH0), int(e + RTH0),
                            v, float(px[s]), float(px[e]), band_table=band)
    dend = put[e] - put[s] if np.isfinite(put[e]) else np.nan
    return dict(d=d, kind=kind, s=s, e=e, hour=(s + RTH0) // 60, b=s // 30, legdir=legdir, r=rd.es_move_pct,
                actual=rd.actual_pp, expected=rd.expected_pp, ratio=rd.ratio, reason=rd.reason, band=rd.band_pp,
                beyond=rd.beyond_noise, dend=dend)


rows = []
rng = np.random.default_rng(SEED)
for d in SESS:
    for h in (15, 30):
        for _ in range(20):
            s = int(rng.integers(DOM_LO, DOM_HI - h + 1))
            rows.append(read(d, s, s + h, f"rand{h}"))
n_legs_all = 0
for d in SESS:
    H, Lo = BARS[d]["h"], BARS[d]["l"]
    if not (np.isfinite(H).all() and np.isfinite(Lo).all()):
        continue
    for lg in swing_legs(H, Lo, LEGSW):
        s, e = lg["start_i"] + 1, lg["end_i"] + 1
        if s >= DOM_LO and e <= DOM_HI:
            n_legs_all += 1
            rows.append(read(d, s, e, "leg", lg["dir"]))
W = pd.DataFrame([r for r in rows if r is not None])
W["dir"] = np.where(W["r"] > 0, "up", np.where(W["r"] < 0, "down", "flat"))
R["counts"] = {"legs_in_domain": n_legs_all,
               **{f"{k}|{reason}": int(n) for (k, reason), n in W.groupby(["kind", "reason"]).size().items()}}


# ------------------------------------------------------------------------------------- parity readout (C4)
def callbacks_ss(d, s, e):
    """The Skew module's sticky-strike put-skew change (no add-ons; they cancel, CR-CJ Step 3), ES move for spot."""
    m = MIN[d]
    a, b = m["rows"].iloc[m["row"][s]], m["rows"].iloc[m["row"][e]]
    exp_iso = d.isoformat()
    T_now, T_prev = SK._T_from_row_snapshot(b, exp_iso), SK._T_from_row_snapshot(a, exp_iso)
    k_prev, s_prev = SK._prev_smile_interp(a, T_prev)
    k_shift = np.log(PX[d][e] / PX[d][s])
    atm_now, _, _ = SK._skews_from_row(b)
    _, _, put_prev = SK._skews_from_row(a)
    k_p25_now = k_for_abs_delta(0.25, is_put=True, sigma=atm_now, T=T_now)
    exp_put = (SK._interp_linear_extrap(k_p25_now + k_shift, k_prev, s_prev)
               - SK._interp_linear_extrap(k_shift, k_prev, s_prev)) * 100.0
    return exp_put - put_prev


par = W[(W["kind"] == "rand15") & W["expected"].notna()].copy()
cb = []
for row in par.itertuples():
    try:
        cb.append(callbacks_ss(row.d, row.s, row.e) if MIN[row.d]["row"][row.e] >= 0 else np.nan)
    except Exception:
        cb.append(np.nan)
par["cb"] = cb
par = par[par["cb"].notna()]
X = np.c_[np.ones(len(par)), par["r"]]
beta_mine = float(np.linalg.lstsq(X, par["expected"], rcond=None)[0][1])
beta_cb = float(np.linalg.lstsq(X, par["cb"], rcond=None)[0][1])
R["parity_rand15"] = {"n": len(par), "beta_ck_vol_pts_per_1pct": round(beta_mine, 2),
                      "beta_callbacks_vol_pts_per_1pct": round(beta_cb, 2),
                      "cb_intercept": round(float(np.linalg.lstsq(X, par["cb"], rcond=None)[0][0]), 3),
                      "corr": round(float(np.corrcoef(par["expected"], par["cb"])[0, 1]), 3)}


# ------------------------------------------------------------------------------------------- C8: tables
QS = (10, 25, 50, 75, 90)


def summarize(df):
    ok = df[df["reason"] == "ok"]
    out = {"n": len(df), "n_ratio": len(ok),
           "share_move_too_small": round(float((df["reason"] == "move too small").mean()), 3) if len(df) else None,
           "share_insufficient": round(float((df["reason"] == "insufficient data").mean()), 3) if len(df) else None}
    bn = df["beyond"].dropna().astype(bool)
    out["share_beyond_noise_all"] = round(float(bn.mean()), 3) if len(bn) else None
    if len(ok):
        q = np.percentile(ok["ratio"], QS)
        out.update({f"p{k}": round(float(v), 2) for k, v in zip(QS, q)})
        out["share_lt0"] = round(float((ok["ratio"] < 0).mean()), 3)
        out["share_0_1"] = round(float(((ok["ratio"] >= 0) & (ok["ratio"] <= 1)).mean()), 3)
        out["share_gt1"] = round(float((ok["ratio"] > 1).mean()), 3)
        bo = ok["beyond"].dropna().astype(bool)
        out["share_beyond_noise_ratio_defined"] = round(float(bo.mean()), 3) if len(bo) else None
        out["median_abs_actual"] = round(float(ok["actual"].abs().median()), 3)
        out["median_abs_expected"] = round(float(ok["expected"].abs().median()), 3)
        out["median_endpoint_ratio"] = round(float((ok["dend"] / ok["expected"]).median()), 2)
    return out


def boot(df, n=2000):
    ok = df[df["reason"] == "ok"]
    groups = {d: g for d, g in ok.groupby("d")}
    ds = list(groups)
    rng_ = np.random.default_rng(SEED)
    stats = []
    for _ in range(n):
        g = pd.concat([groups[d] for d in rng_.choice(ds, len(ds))])
        rr = g["ratio"].to_numpy()
        stats.append((np.median(rr), (rr < 0).mean(), (rr > 1).mean(), g["beyond"].astype(float).mean()))
    a = np.array(stats)
    lo, hi = np.percentile(a, 2.5, axis=0), np.percentile(a, 97.5, axis=0)
    return {k: [round(float(lo[i]), 3), round(float(hi[i]), 3)]
            for i, k in enumerate(("median", "share_lt0", "share_gt1", "share_beyond_noise_ratio_defined"))}


T = {}
for k in ("leg", "rand15", "rand30"):
    df = W[W["kind"] == k]
    T[k] = {"all": summarize(df), "ci95": boot(df)}
    T[k]["by_dir"] = {dr: summarize(df[df["dir"] == dr]) for dr in ("up", "down")}
    T[k]["by_hour"] = {f"{h:02d}:00": summarize(df[df["hour"] == h]) for h in range(7, 13)}
R["tables"] = T
(OUT / "results.json").write_text(json.dumps(R, indent=2, default=str))
W.to_parquet(OUT / "windows.parquet")
print(json.dumps({k: R[k] for k in ("sessions", "counts", "parity_rand15", "noise_band_pp")}, indent=1))
for k in T:
    print(k, json.dumps(T[k]["all"]), json.dumps(T[k]["ci95"]))
    for dr in ("up", "down"):
        print("  ", dr, json.dumps(T[k]["by_dir"][dr]))
    for h, v in T[k]["by_hour"].items():
        print("  ", h, {x: v.get(x) for x in ("n", "n_ratio", "p25", "p50", "p75", "share_lt0", "share_gt1",
                                              "share_beyond_noise_ratio_defined")})
