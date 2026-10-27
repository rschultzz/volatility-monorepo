"""CR-CO Step 0 — CR-CL Step 0 on 2023-01-03 .. 2024-04-02, plus coverage, quote timing, roll weeks, era noise
band and the vol-scaled swing counts. READ-ONLY. No group outcome.

Spec: specs/CR-CO-skew-g1down-2023.md (K7, O1-O11). Period `y23` only; R-a, 2025 and the holdout are never loaded.
The one 2026 train read is O8's median `implied_move_1d` (a pre-open feature, no price / skew / outcome).

  1. parity (O2):     under the default the CR-CL module resolves PERIOD 'y26' and OUT logs/cr_cl (subprocess,
                      nothing loaded); no "y26" literal remains outside the default / the OUT map.
  2. coverage (O3/O4): per session 0DTE PM skew minutes, Databento ES RTH minutes, non-Databento rows; third Fridays.
  3. quote timing (O6): label - quote_date by month; 1-min dSPX / dvol50 vs dES correlation at lags -20..+5 on the
                      label and quote-time axes, by month; month established / excluded by the O6 rule.
  4. roll weeks (O5): window sessions' RTH volume ratio and missing minutes; flags; rule -> exclusions.
  5. exclusions:      written to scripts/logs/cr_co/exclusions.json (read by cr_cl_lib.load under y23).
  6. CR-CL Step 0:    scripts/cr_cl_step0.py unchanged under CR_CL_PERIOD=y23 -> scripts/logs/cr_co/step0.json.
  7. extras:          G1-down count under no / rule / all-window exclusion; quote lag at e; 2023 noise band vs
                      shipped (report only); vol-scaled swing counts (O8).
Writes scripts/logs/cr_co/step0_extra.json.
"""
from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ------------------------------------------------------------------------------------------------ 1. parity (O2)
chk = subprocess.run([sys.executable, "-c", "import cr_cl_lib as C; print(C.PERIOD, C.OUT.name)"], cwd=HERE,
                     env={k: v for k, v in os.environ.items() if k != "CR_CL_PERIOD"}, capture_output=True, text=True)
assert chk.returncode == 0 and chk.stdout.split()[-2:] == ["y26", "cr_cl"], (chk.stdout, chk.stderr)
for f in ("cr_cl_lib.py", "cr_cl_step0.py"):
    lines = [ln for ln in (HERE / f).read_text().splitlines() if '"y26"' in ln]
    assert all("CR_CL_PERIOD" in ln or "PERIOD ==" in ln or "}[PERIOD]" in ln for ln in lines), (f, lines)

os.environ["CR_CL_PERIOD"] = "y23"
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import cr_cj_lib as L  # noqa: E402
import cr_cl_lib as C  # noqa: E402
from packages.shared import skew_expected as se  # noqa: E402

assert C.PERIOD == "y23" and C.OUT.name == "cr_co"
EXF = C.OUT / "exclusions.json"
if EXF.exists():
    EXF.unlink()                                            # recomputed below from bars / quotes only
LO, HI = L.PERIODS["y23"]
SESS = L.sessions("y23")
X = {"period": "y23", "calendar_sessions": len(SESS), "first": str(SESS[0]), "last": str(SESS[-1])}


def third_friday(y, m):
    d = date(y, m, 15)
    return d + timedelta(days=(4 - d.weekday()) % 7)


# ------------------------------------------------------------------------------------------- 2. coverage (O3/O4)
SK = L.load_skew("y23")
L.load_es("y23")
ESALL = pd.read_parquet(L.OUT_DIR / "es_y23.parquet")
MIN = L.skew_minutes(SK)
_, BARS = L.front_bars(ESALL[ESALL["symbol"] == "ES.c.0"])

skmin = SK.groupby("d")["snapshot_pt"].nunique()
esdb = ESALL[ESALL["symbol"] == "ES.c.0"].groupby("d").size()
esother = ESALL[ESALL["symbol"] != "ES.c.0"].groupby(["d", "symbol"]).size()
cov = pd.DataFrame({"skew_minutes": skmin, "es_databento_minutes": esdb}).reindex(SESS).fillna(0).astype(int)
X["coverage"] = {
    "skew_first_session": str(skmin.index.min()) if len(skmin) else None,
    "skew_last_session": str(skmin.index.max()) if len(skmin) else None,
    "sessions_with_skew": int((cov["skew_minutes"] > 0).sum()),
    "sessions_with_es_databento": int((cov["es_databento_minutes"] > 0).sum()),
    "sessions_with_both": int(((cov["skew_minutes"] > 0) & (cov["es_databento_minutes"] > 0)).sum()),
    "no_skew": [str(d) for d in cov.index[cov["skew_minutes"] == 0]],
    "skew_lt_80pct_minutes": {str(d): int(v) for d, v in cov["skew_minutes"].items() if 0 < v < 0.8 * L.NB},
    "no_es_databento": [str(d) for d in cov.index[cov["es_databento_minutes"] == 0]],
    "es_databento_lt_390": {str(d): int(v) for d, v in cov["es_databento_minutes"].items() if 0 < v < L.NB},
    "non_databento_rows": {f"{d}|{s}": int(n) for (d, s), n in esother.items()},
    "skew_minutes_by_month_median": {str(k): float(v) for k, v in
                                     cov.groupby(pd.to_datetime(cov.index).to_period("M"))["skew_minutes"].median().items()},
}
# first SPX monies row of any expiry in the period (is the pre-coverage gap a table gap or a 0DTE-only gap?)
X["coverage"]["monies_any_expiry_first_session_in_period"] = str(L.q(
    "SELECT min(trade_date) AS t FROM orats_monies_minute WHERE ticker = 'SPX' AND trade_date >= %s AND trade_date <= %s",
    (LO.isoformat(), HI.isoformat()))["t"].iloc[0])
# third Fridays: which expiry_tod rows exist for the same-day expiry
tf = [third_friday(y, m) for y in (2023, 2024) for m in range(1, 13) if LO <= third_friday(y, m) <= HI]
tfr = L.q("""
    SELECT trade_date, expiry_tod, count(*) AS n FROM orats_monies_minute
    WHERE ticker = 'SPX' AND trade_date = ANY(%s) AND expir_date = trade_date
      AND snapshot_pt::time >= '06:30' AND snapshot_pt::time < '13:00'
    GROUP BY 1, 2 ORDER BY 1, 2
""", ([d.isoformat() for d in tf],))
X["third_fridays"] = {str(d): {r.expiry_tod: int(r.n) for r in tfr[tfr["trade_date"] == d.isoformat()].itertuples()}
                      for d in tf}

# --------------------------------------------------------------------------------------- 3. quote timing (O6)
lab = pd.to_datetime(SK["snap_shot_date"], utc=True)
qts = pd.to_datetime(SK["quote_date"], utc=True, errors="coerce")
SK["lag_s"] = (lab - qts).dt.total_seconds()
SK["month"] = pd.to_datetime(SK["d"]).dt.to_period("M").astype(str)
QT = {}
for mo, g in SK.groupby("month"):
    v = g["lag_s"]
    QT[mo] = dict(n=int(len(v)), p5=float(np.nanpercentile(v, 5)) if v.notna().any() else None,
                  p50=float(np.nanmedian(v)) if v.notna().any() else None,
                  p95=float(np.nanpercentile(v, 95)) if v.notna().any() else None,
                  share_gt_300=round(float((v.abs() > 300).mean()), 4), share_unparsed=round(float(v.isna().mean()), 4))
X["quote_lag_by_month_s"] = QT

LAGS = list(range(-20, 6))


def lag_corr(days, series, mode):
    out = {}
    for lag in LAGS:
        a, b = [], []
        for d in days:
            if d not in MIN or d not in BARS:
                continue
            y = MIN[d][series]
            px = L.es_close_at_snapshot(BARS, d, MIN if mode == "quote" else None)
            pxl = np.full(L.NB, np.nan)
            if lag >= 0:
                pxl[:L.NB - lag] = px[lag:]
            else:
                pxl[-lag:] = px[:L.NB + lag]
            u, w = np.diff(y), np.diff(pxl)
            k = np.isfinite(u) & np.isfinite(w)
            k[:30] = False                                     # skip 06:30-07:00 (CR-CJ A1)
            a.append(u[k]); b.append(w[k])
        a = np.concatenate(a) if a else np.array([])
        b = np.concatenate(b) if b else np.array([])
        out[lag] = round(float(np.corrcoef(a, b)[0, 1]), 3) if len(a) > 30 else None
    return out


months = sorted(SK["month"].unique())
ALN, established, excluded_months = {}, {}, {}
for mo in months:
    days = [d for d in SESS if str(d)[:7] == mo]
    r = {f"{s}|{m}": lag_corr(days, s, m) for s in ("spx", "atm") for m in ("label", "quote")}
    ALN[mo] = r
    sq, aq = r["spx|quote"], r["atm|quote"]
    sq_ok = sq[0] is not None and sq[0] >= 0.3 and sq[0] == max(v for v in sq.values() if v is not None)
    aq_ok = aq[0] is not None and aq[0] <= -0.2 and aq[0] == min(v for v in aq.values() if v is not None)
    rule = "stock_price" if sq_ok else ("vol50" if (sq[0] is None or sq[0] < 0.3) and aq_ok else None)
    q = QT[mo]
    if q["share_unparsed"] > 0.2 or q["share_gt_300"] > 0.2:
        rule = None
    established[mo] = rule
    if rule is None:
        excluded_months[mo] = dict(spx_quote_lag0=sq[0], spx_quote_peak_lag=max(sq, key=lambda k: -9 if sq[k] is None else sq[k]),
                                   atm_quote_lag0=aq[0], share_unparsed=q["share_unparsed"], share_gt_300=q["share_gt_300"])
X["alignment_lag_corr_by_month"] = ALN
X["quote_time_established_by_month"] = established
X["quote_time_excluded_months"] = excluded_months

# --------------------------------------------------------------------------------------------- 4. roll weeks (O5)
expiries = [third_friday(y, m) for y in (2023, 2024) for m in (3, 6, 9, 12) if LO <= third_friday(y, m) <= HI]
vol = ESALL[ESALL["symbol"] == "ES.c.0"].groupby("d")["vol"].sum()
win = {}
for ex in expiries:
    mon = ex - timedelta(days=11)
    win[ex] = [d for d in SESS if mon <= d <= ex]
in_win = {d for v in win.values() for d in v}
ref_vol = float(vol[[d for d in vol.index if d not in in_win]].median())


def missing(d):
    if d not in BARS:
        return L.NB, L.NB
    ok = np.isfinite(BARS[d]["h"]) & np.isfinite(BARS[d]["l"])
    run, mx = 0, 0
    for k in ok:
        run = 0 if k else run + 1
        mx = max(mx, run)
    return int((~ok).sum()), mx


ROLL, roll_excl = {}, set()
for ex, days in win.items():
    rows = {}
    for d in days:
        n_miss, mx = missing(d)
        vr = float(vol.get(d, 0.0)) / ref_vol
        flag = bool(vr < 0.25 or n_miss > 5 or mx > C.GAP_MAX)
        rows[str(d)] = dict(vol_ratio=round(vr, 3), missing_min=n_miss, max_missing_run=mx, flagged=flag)
    nflag = sum(r["flagged"] for r in rows.values())
    whole = nflag >= len(rows) / 2
    for d, r in rows.items():
        if r["flagged"] or whole:
            roll_excl.add(d)
    ROLL[str(ex)] = dict(window=f"{days[0]}..{days[-1]}" if days else None, sessions=rows, flagged=nflag,
                         whole_window_excluded=bool(whole))
X["roll_ref_median_rth_volume"] = ref_vol
X["roll_windows"] = ROLL
# the same metrics on non-window sessions (how unusual are the flags?)
nw = [d for d in SESS if d not in in_win and d in BARS]
nwm = np.array([missing(d)[0] for d in nw])
X["non_window_sessions"] = dict(n=len(nw), missing_min_p50=float(np.median(nwm)), missing_gt5=int((nwm > 5).sum()),
                                vol_ratio_p5=round(float(np.percentile([vol[d] / ref_vol for d in nw], 5)), 3))

# ---------------------------------------------------------------------------------------------- 5. exclusions
qt_excl = {str(d) for d in SESS if str(d)[:7] in excluded_months}
EX = sorted(roll_excl | qt_excl)
EXF.write_text(json.dumps({"sessions": EX, "roll": sorted(roll_excl), "quote_time": sorted(qt_excl)}, indent=1))
X["exclusions"] = dict(n=len(EX), roll=sorted(roll_excl), quote_time_months=sorted(excluded_months))

# ------------------------------------------------------------------------------------------- 6. CR-CL Step 0 on y23
G = runpy.run_path(str(HERE / "cr_cl_step0.py"), run_name="__main__")
D, LEGS = G["D"], G["LEGS"]
assert min(D["SESS"]) >= LO and max(D["SESS"]) <= HI
assert not (set(map(str, D["SESS"])) & set(EX))
X["sessions_in_test"] = len(D["SESS"])
X["sessions_missing_from_test"] = [str(d) for d in SESS if d not in D["SESS"]]

# ------------------------------------------------------------------------------------------------------ 7. extras
W = LEGS[12.0]
P = W[~W["late"]]
g1d = P[C.group_mask(P, "G1", -1)]
X["G1_down_count"] = {"rule_exclusions": int(len(g1d)),
                      "all_window_sessions_excluded": int((~g1d["d"].isin(in_win)).sum())}
EXF.rename(EXF.with_suffix(".tmp"))                     # count with no exclusions (counts only), then restore
try:
    D0 = C.load()
    W0, _ = C.add_terciles(C.build_legs(D0, 12.0))
    P0 = W0[~W0["late"]]
    X["G1_down_count"]["no_exclusions"] = int(C.group_mask(P0, "G1", -1).sum())
    X["G1_down_count"]["no_exclusions_sessions"] = len(D0["SESS"])
finally:
    EXF.with_suffix(".tmp").rename(EXF)


def lag_at(d, i):
    m = D["MIN"][d]
    j = m["row"][i]
    if j < 0:
        return np.nan
    r = m["rows"].iloc[j]
    return (pd.Timestamp(r["snap_shot_date"]) - pd.Timestamp(r["quote_date"])).total_seconds() if pd.notna(r["quote_date"]) else np.nan


X["quote_lag_at_e_s"] = {}
for name, sub in (("primary_ratio_defined", P[P["reason"] == "ok"]), ("G1-down", g1d),
                  ("baseline_down", P[(P["grp"] == "base") & (P["dir"] == -1)])):
    v = np.array([lag_at(r.d, r.e) for r in sub.itertuples()], float)
    X["quote_lag_at_e_s"][name] = dict(n=int(np.isfinite(v).sum()), p50=float(np.nanmedian(v)) if len(v) else None,
                                       p95=float(np.nanpercentile(v, 95)) if len(v) else None)
qbehind = np.array([(r.e - 1) - (D["MIN"][r.d]["qidx"][r.e] - 1) for r in P[P["reason"] == "ok"].itertuples()], float)
X["quote_minute_behind_extreme_bar"] = {str(int(k)): int(v) for k, v in pd.Series(qbehind).value_counts().sort_index().items()}

# 2023 noise band (O7, CR-CK C6 method as CR-CN N4) — report only
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
band, ratio = {}, {}
for b in range(1, 13):
    vals, last = [], None
    for Lm in se.NOISE_BAND_LENGTHS:
        cell = sd[(sd["b"] == b) & (sd["L"] == Lm)]
        v = float(cell["sd_end"].iloc[0]) if len(cell) and int(cell["n"].iloc[0]) >= 30 else None
        last = v if v is not None else last
        vals.append(None if last is None else round(last, 3))
    band[L.bucket_label(b)] = vals
    ratio[L.bucket_label(b)] = [None if v is None else round(v / s_, 2) for v, s_ in zip(vals, se.NOISE_BAND_PP[b])]
X["noise_band_lengths_min"] = list(se.NOISE_BAND_LENGTHS)
X["noise_band_y23_pp"] = band
X["noise_band_ratio_y23_over_shipped"] = ratio
pr = np.array([r for b in range(1, 9) for r in ratio[L.bucket_label(b)] if r is not None])
X["noise_ratio_summary_07_00_to_10_30"] = dict(p25=float(np.percentile(pr, 25)), p50=float(np.median(pr)),
                                               p75=float(np.percentile(pr, 75)))
X["put_skew_level_pp"] = dict(p50=float(np.nanmedian(np.concatenate([D["MIN"][d]["put"] for d in D["SESS"]]))))

# vol-scaled swing (O8) — counts only
from packages.shared.canonical_version import CANONICAL_FEATURE_VERSION  # noqa: E402

lo26, hi26 = L.PERIODS["y26"]
L.guard(lo26, hi26)
IMQ = """SELECT trade_date, (feature_vector->>'implied_move_1d')::float AS im FROM bt_daily_features
         WHERE feature_version = %s AND ticker = 'SPX' AND active AND trade_date BETWEEN %s AND %s"""
im26 = L.q(IMQ, (CANONICAL_FEATURE_VERSION, lo26, hi26)).drop_duplicates("trade_date")
im26 = im26[im26["trade_date"].isin(L.sessions("y26"))]["im"].dropna()
IM_TRAIN = float(im26.median())
im23 = L.q(IMQ, (CANONICAL_FEATURE_VERSION, LO, HI)).drop_duplicates("trade_date").set_index("trade_date")["im"]
parts, no_im = [], []
for d in D["SESS"]:
    v = im23.get(d)
    if v is None or not np.isfinite(v) or v <= 0:
        no_im.append(str(d))
        continue
    Dd = dict(D, SESS=[d])
    parts.append(C.build_legs(Dd, 12.0 * float(v) / IM_TRAIN))
WV, cutsv = C.add_terciles(pd.concat(parts, ignore_index=True))
swd = np.array([12.0 * float(im23[d]) / IM_TRAIN for d in D["SESS"] if str(d) not in no_im])
X["vol_scaled_swing"] = dict(im_train_median_pts=round(IM_TRAIN, 2), im_train_n=int(len(im26)),
                             im_y23_median_pts=round(float(np.median([im23[d] for d in D["SESS"] if str(d) not in no_im])), 2),
                             swing_pts=dict(p5=round(float(np.percentile(swd, 5)), 2), p50=round(float(np.median(swd)), 2),
                                            p95=round(float(np.percentile(swd, 95)), 2)),
                             sessions_no_im=no_im, funnel=G["funnel"](WV),
                             size_tercile_cuts_pts=[round(c, 2) for c in cutsv], cells=G["cell_counts"](WV))

(C.OUT / "step0_extra.json").write_text(json.dumps(X, indent=2, default=str))
print(json.dumps({k: v for k, v in X.items() if k not in ("alignment_lag_corr_by_month",)}, indent=1, default=str))
