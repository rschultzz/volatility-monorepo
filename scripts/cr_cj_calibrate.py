"""CR-CJ Steps 1-5 — noise, fit, current-curve comparison, balance, out-of-sample. READ-ONLY, no outcomes.

Spec: specs/CR-CJ-skew-expected-calibration.md (C1-C14). Uses only 0DTE PM skew snapshots and contemporaneous
Databento ES closes (scripts/cr_cj_lib.py). Writes scripts/logs/cr_cj/results.json and results.md.

Models (vol pts; r = ES % move over the window):
  alpha model (C7): dY = alpha_b(start bucket) + beta * r                       fixed windows only
  minutes model:    dY = sum_b theta_b * minutes_in_b + beta * r                 any window or leg
  curve (C9):       expected dY = beta * r + sum_b theta_b * minutes_in_b, theta_b = mean 1-min (dY - beta * r1)
"""
from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date

import numpy as np
import pandas as pd
from scipy import stats

import cr_cj_lib as L
from packages.shared.flow_features import swing_legs
from packages.shared.surface_compare import k_for_abs_delta

SEED = 20261026
DOM_LO, DOM_HI = 30, L.NB - 15            # snapshot minute index: start >= 07:00, end <= 12:45 PT
NBK = 13                                  # 30-min buckets 06:30 .. 13:00
DOM_B = list(range(1, 13))                # buckets inside the fit domain (07:00 .. 12:45)
LEGSW = 12.0

_spec = importlib.util.spec_from_file_location("skew_callbacks", L.REPO_ROOT / "apps/web/modules/Skew/callbacks.py")
SK = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(SK)
assert SK.BETA_VOLPTS_PER_1PCT == 4.5 and SK.THETA_ATM_PP_PER_SQRT_YEAR == -638 and SK.BETA_MAX_SHIFT_PP == 6.0

# ------------------------------------------------------------------------------------------------ load
MIN, BARS, PER = {}, {}, {}
for p in L.PERIODS:
    MIN.update(L.skew_minutes(L.load_skew(p)))
    BARS.update(L.front_bars(L.load_es(p))[1])
    for d in L.sessions(p):
        PER[d] = p
SESS = [d for d in sorted(PER) if d in MIN and d in BARS]
PX = {d: L.es_close_at_snapshot(BARS, d, MIN) for d in SESS}
OK = {d: np.isfinite(MIN[d]["put"]) & np.isfinite(MIN[d]["atm"]) & np.isfinite(PX[d]) for d in SESS}
R = {"sessions": {p: sum(PER[d] == p for d in SESS) for p in L.PERIODS}}


def minutes_in_buckets(s: int, e: int) -> np.ndarray:
    """Minutes of (s, e] spent in each 30-min bucket (minute j counts in bucket of j - 1 -> j, i.e. (j-1)//30)."""
    v = np.zeros(NBK)
    j = np.arange(s, e)                     # 1-min steps j -> j + 1
    np.add.at(v, j // 30, 1.0)
    return v


def years_to_close(i):
    """T in years to 16:00 ET (= 13:00 PT) on a 365-day clock, as the Skew module (snapshot minute index i)."""
    return np.maximum((L.NB - np.asarray(i, float)) / (365.0 * 24 * 60), SK.EPS_T)


def window_row(d, s, e, kind, legdir=0):
    m = MIN[d]
    r = 100.0 * (PX[d][e] / PX[d][s] - 1.0)
    return dict(d=d, period=PER[d], kind=kind, s=s, e=e, b=s // 30, r=r, dput=m["put"][e] - m["put"][s],
                datm=m["atm"][e] - m["atm"][s], dsqrtT=float(np.sqrt(years_to_close(e)) - np.sqrt(years_to_close(s))),
                dpx=PX[d][e] - PX[d][s], legdir=legdir, mins=minutes_in_buckets(s, e))


# ------------------------------------------------------------------------------------------- samples
rows = []
for d in SESS:
    ok = OK[d]
    for h in (5, 15, 30):
        for s in range(DOM_LO, DOM_HI - h + 1, h):
            if ok[s] and ok[s + h]:
                rows.append(window_row(d, s, s + h, f"w{h}"))
    for s in range(1, DOM_LO - 5 + 1, 5):                       # readout: 06:31 .. 07:00
        if ok[s] and ok[s + 5]:
            rows.append(window_row(d, s, s + 5, "open5"))
    for s in range(DOM_HI, L.NB - 5, 5):                         # readout: 12:45 .. 12:59
        if ok[s] and ok[s + 5]:
            rows.append(window_row(d, s, s + 5, "late5"))
rng = np.random.default_rng(SEED)
for d in SESS:
    ok = OK[d]
    for _ in range(20):
        h = int(rng.integers(15, 31))
        s = int(rng.integers(DOM_LO, DOM_HI - h + 1))
        if ok[s] and ok[s + h]:
            rows.append(window_row(d, s, s + h, "rand"))
for d in SESS:
    H, Lo = BARS[d]["h"], BARS[d]["l"]
    if not (np.isfinite(H).all() and np.isfinite(Lo).all()):
        continue
    for lg in swing_legs(H, Lo, LEGSW):
        s, e = lg["start_i"] + 1, lg["end_i"] + 1               # snapshot aligned to the extreme bar's close (C4)
        if s >= DOM_LO and e <= DOM_HI and OK[d][s] and OK[d][e]:
            rows.append(window_row(d, s, e, "leg", lg["dir"]))
W = pd.DataFrame(rows)
W = W[W["r"] != 0].reset_index(drop=True)
R["counts"] = {k: int(n) for k, n in W.groupby("kind").size().items()}
R["counts_by_period"] = {f"{k}|{p}": int(n) for (k, p), n in W.groupby(["kind", "period"]).size().items()}


# ------------------------------------------------------------------------------------------- OLS tools
def ols_cl(X, y, g):
    X, y = np.asarray(X, float), np.asarray(y, float)
    codes = pd.factorize(pd.Series(g))[0]
    G, (N, K) = codes.max() + 1, X.shape
    XtXi = np.linalg.pinv(X.T @ X)
    b = XtXi @ X.T @ y
    u = y - X @ b
    S = np.zeros((G, K))
    np.add.at(S, codes, X * u[:, None])
    V = (G / (G - 1)) * ((N - 1) / (N - K)) * XtXi @ (S.T @ S) @ XtXi
    return b, V, dict(n=int(N), sessions=int(G), r2=float(1 - (u @ u) / ((y - y.mean()) @ (y - y.mean()))))


def wald_equal(b, V, idx):
    Rm = np.zeros((len(idx) - 1, len(b)))
    for i in range(len(idx) - 1):
        Rm[i, idx[i]], Rm[i, idx[i + 1]] = 1.0, -1.0
    Rb = Rm @ b
    w = float(Rb @ np.linalg.pinv(Rm @ V @ Rm.T) @ Rb)
    return dict(chi2=round(w, 2), df=len(idx) - 1, p=float(stats.chi2.sf(w, len(idx) - 1)))


def ci(b, V, i):
    se = float(np.sqrt(V[i, i]))
    return dict(est=round(float(b[i]), 3), se=round(se, 3), lo=round(float(b[i] - 1.96 * se), 3),
                hi=round(float(b[i] + 1.96 * se), 3))


def dummies(bk, levels):
    return np.column_stack([(bk == lv).astype(float) for lv in levels])


def fit_alpha(df, y, variant="single"):
    """C7 alpha model on fixed windows: alpha_b + beta r (single / per bucket / up-down)."""
    lv = sorted(df["b"].unique())
    D = dummies(df["b"].to_numpy(), lv)
    r = df["r"].to_numpy()
    if variant == "single":
        X = np.column_stack([D, r]); names = [f"a{b}" for b in lv] + ["beta"]
    elif variant == "bucket":
        X = np.column_stack([D, D * r[:, None]]); names = [f"a{b}" for b in lv] + [f"beta{b}" for b in lv]
    elif variant == "updown":
        X = np.column_stack([D, np.maximum(r, 0), np.minimum(r, 0)]); names = [f"a{b}" for b in lv] + ["beta_up", "beta_dn"]
    elif variant == "sqrtT":
        X = np.column_stack([np.ones(len(df)), df["dsqrtT"].to_numpy(), r]); names = ["a", "gamma", "beta"]
    b, V, info = ols_cl(X, df[y].to_numpy(), df["d"].astype(str))
    out = {"info": info, "coef": {n: ci(b, V, i) for i, n in enumerate(names) if not n.startswith("a")}}
    if variant == "bucket":
        out["wald_equal_beta"] = wald_equal(b, V, [names.index(f"beta{b_}") for b_ in lv])
    if variant == "updown":
        out["wald_equal_beta"] = wald_equal(b, V, [names.index("beta_up"), names.index("beta_dn")])
    if variant == "sqrtT":
        out["coef"]["a"] = ci(b, V, 0)
    return out


def fit_minutes(df, y):
    """Minutes model: sum_b theta_b * minutes_b + beta r, any window or leg."""
    M = np.vstack(df["mins"].to_numpy())[:, DOM_B]
    X = np.column_stack([M, df["r"].to_numpy()])
    b, V, info = ols_cl(X, df[y].to_numpy(), df["d"].astype(str))
    return {"info": info, "beta": ci(b, V, len(DOM_B)),
            "theta": {L.bucket_label(bb): round(float(b[i]), 4) for i, bb in enumerate(DOM_B)}}


# ------------------------------------------------------------------------------- Step 1: noise
noise = {}
for y in ("put", "atm"):
    per_b = {b: [] for b in range(NBK)}
    for d in SESS:
        v = MIN[d][y]
        dv = np.diff(v)
        for j in range(L.NB - 1):                                # change j -> j + 1, bucket of j
            if np.isfinite(dv[j]):
                per_b[j // 30].append(dv[j])
    noise[y] = {"sd_1min": {L.bucket_label(b): round(float(np.std(per_b[b])), 3) for b in range(NBK)},
                "mean_abs_1min": {L.bucket_label(b): round(float(np.mean(np.abs(per_b[b]))), 3) for b in range(NBK)}}
    vr = {}
    for h in (5, 15, 30):
        sd_h, vr_h = {}, {}
        for b in range(NBK):
            ch = []
            for d in SESS:
                v = MIN[d][y]
                for s in range(30 * b, min(30 * b + 30, L.NB - h), h):
                    if np.isfinite(v[s]) and np.isfinite(v[s + h]):
                        ch.append(v[s + h] - v[s])
            if len(ch) > 50:
                sd_h[L.bucket_label(b)] = round(float(np.std(ch)), 3)
                vr_h[L.bucket_label(b)] = round(float(np.var(ch) / (h * np.var(per_b[b]))), 3)
        vr[h] = {"sd": sd_h, "variance_ratio": vr_h}
    noise[y]["by_horizon"] = vr
R["step1_noise"] = noise

# ------------------------------------------------------------------------------- Step 2: fits
fits = {}
for y in ("dput", "datm"):
    fy = {}
    for k in ("w5", "w15", "w30"):
        df = W[W["kind"] == k]
        fy[k] = {v: fit_alpha(df, y, v) for v in ("single", "bucket", "updown", "sqrtT")}
        fy[k]["by_period"] = {p: fit_alpha(df[df["period"] == p], y, "single")["coef"]["beta"] for p in L.PERIODS}
        fy[k]["minutes"] = fit_minutes(df, y)
    leg = W[W["kind"] == "leg"]
    fy["leg"] = {"minutes": fit_minutes(leg, y),
                 "up": fit_minutes(leg[leg["legdir"] == 1], y)["beta"],
                 "down": fit_minutes(leg[leg["legdir"] == -1], y)["beta"],
                 "by_period": {p: fit_minutes(leg[leg["period"] == p], y)["beta"] for p in L.PERIODS}}
    for k in ("open5", "late5"):
        df = W[W["kind"] == k]
        b, V, info = ols_cl(np.column_stack([np.ones(len(df)), df["r"]]), df[y], df["d"].astype(str))
        fy[k] = {"beta": ci(b, V, 1), "info": info}
    fits[y] = fy
R["step2_fits"] = fits

# C10: recommended beta horizon
bp = {k: fits["dput"][k]["single"]["coef"]["beta"]["est"] for k in ("w5", "w15", "w30")}
plateau = all(abs(bp[k] - bp["w15"]) <= 0.25 * abs(bp["w15"]) for k in bp)
R["c10"] = {"beta_put_by_window": bp, "plateau": plateau, "source": "windows" if plateau else "legs"}


# ------------------------------------------------------------------------------- the curve (C9)
def theta_from_1min(sessions, y, beta_of_bucket):
    """theta_b = mean over 1-min changes in bucket b of (dY - beta_b * r1)."""
    acc = {b: [] for b in range(NBK)}
    for d in sessions:
        v, px = MIN[d][{"dput": "put", "datm": "atm"}[y]], PX[d]
        dv, r1 = np.diff(v), 100.0 * (px[1:] / px[:-1] - 1.0)
        for j in range(L.NB - 1):
            if np.isfinite(dv[j]) and np.isfinite(r1[j]):
                acc[j // 30].append(dv[j] - beta_of_bucket(j // 30, r1[j]) * r1[j])
    return np.array([float(np.mean(acc[b])) if acc[b] else 0.0 for b in range(NBK)])


def make_curve(sessions, y, variant):
    """Fit beta(s) on the C10 source sample (15-min windows if plateau, else legs) restricted to `sessions`,
    then theta from 1-min changes. variant: single | bucket | updown | bucket_updown."""
    ss = set(sessions)
    src = W[(W["kind"] == ("w15" if plateau else "leg")) & W["d"].isin(ss)]
    M = np.vstack(src["mins"].to_numpy())
    r = src["r"].to_numpy()
    if variant == "single":
        X = np.column_stack([M[:, DOM_B], r])
    elif variant == "updown":
        X = np.column_stack([M[:, DOM_B], np.maximum(r, 0), np.minimum(r, 0)])
    elif variant in ("bucket", "bucket_updown"):
        Wt = M[:, DOM_B] / M[:, DOM_B].sum(1, keepdims=True)    # minute-weighted bucket shares
        if variant == "bucket":
            X = np.column_stack([M[:, DOM_B], Wt * r[:, None]])
        else:
            X = np.column_stack([M[:, DOM_B], Wt * np.maximum(r, 0)[:, None], Wt * np.minimum(r, 0)[:, None]])
    b, V, _ = ols_cl(X, src[y].to_numpy(), src["d"].astype(str))
    nb = len(DOM_B)
    beta = np.zeros((NBK, 2))                                   # [bucket, up/down]
    if variant == "single":
        beta[:, :] = b[nb]
    elif variant == "updown":
        beta[:, 0], beta[:, 1] = b[nb], b[nb + 1]
    elif variant == "bucket":
        beta[DOM_B, 0] = beta[DOM_B, 1] = b[nb:nb + nb]
    else:
        beta[DOM_B, 0], beta[DOM_B, 1] = b[nb:2 * nb], b[2 * nb:3 * nb]
    if variant in ("bucket", "bucket_updown"):                  # edge buckets outside the domain: nearest
        beta[0], beta[12] = beta[1], beta[12] if M[:, 12].sum() else beta[11]
    theta = theta_from_1min(sessions, y, lambda bk, r1: beta[bk, 0 if r1 >= 0 else 1])
    return dict(beta=beta, theta=theta, variant=variant, y=y)


def curve_expected(c, row):
    w = row["mins"]
    sh = w / w.sum()
    bb = float(sh @ c["beta"][:, 0 if row["r"] >= 0 else 1])
    return bb * row["r"] + float(w @ c["theta"])


# ------------------------------------------------------------------------------- Step 3: current curve
def current_expected(d, s, e):
    """(expected d put skew, expected d ATM) in vol pts from the callbacks.py Expected-SS block (CR-CI C18)."""
    m = MIN[d]
    a, b = m["rows"].iloc[m["row"][s]], m["rows"].iloc[m["row"][e]]
    exp_iso = d.isoformat()
    stock_now, prev_stock = float(b["stock_price"]), float(a["stock_price"])
    atm_now, _, put_now = SK._skews_from_row(b)
    _, _, put_prev = SK._skews_from_row(a)
    T_now, prev_T = SK._T_from_row_snapshot(b, exp_iso), SK._T_from_row_snapshot(a, exp_iso)
    k_prev, s_prev = SK._prev_smile_interp(a, prev_T)
    k_shift = np.log(stock_now / prev_stock)
    exp_atm_shape = SK._interp_linear_extrap(k_shift, k_prev, s_prev)
    ret_frac = (stock_now - prev_stock) / prev_stock
    level_shift_pp = max(-SK.BETA_MAX_SHIFT_PP, min(SK.BETA_MAX_SHIFT_PP, (-ret_frac) * 100.0 * SK.BETA_VOLPTS_PER_1PCT))
    droot = max(0.0, np.sqrt(max(prev_T, SK.EPS_T)) - np.sqrt(max(T_now, SK.EPS_T)))
    atm_exp = exp_atm_shape + level_shift_pp / 100.0 + SK.THETA_ATM_PP_PER_SQRT_YEAR * droot / 100.0
    k_p25_now = k_for_abs_delta(0.25, is_put=True, sigma=atm_now, T=T_now)
    exp_p25 = SK._interp_linear_extrap(k_p25_now + k_shift, k_prev, s_prev) + (atm_exp - exp_atm_shape)
    exp_put = (exp_p25 - atm_exp) * 100.0
    # the same model without the beta/theta add-ons, to show they cancel out of skew
    exp_put_ss = (SK._interp_linear_extrap(k_p25_now + k_shift, k_prev, s_prev) - exp_atm_shape) * 100.0
    return exp_put - put_prev, atm_exp * 100.0 - float(a["vol50"]) * 100.0, exp_put_ss - put_prev


_cache = L.OUT_DIR / "current_model.pkl"
if _cache.exists() and pd.read_pickle(_cache)[["d", "kind", "s", "e"]].equals(W[["d", "kind", "s", "e"]]):
    cur = pd.read_pickle(_cache)["cur"].tolist()
else:
    cur = []
    for row in W.itertuples():
        if row.kind in ("w15", "rand", "leg", "w5", "w30"):
            try:
                cur.append(current_expected(row.d, row.s, row.e))
            except Exception:
                cur.append((np.nan, np.nan, np.nan))
        else:
            cur.append((np.nan, np.nan, np.nan))
    W[["d", "kind", "s", "e"]].assign(cur=cur).to_pickle(_cache)
W["cur_dput"], W["cur_datm"], W["cur_dput_ss"] = zip(*cur)
R["current_model_failures"] = {k: int(W[(W["kind"] == k)]["cur_dput"].isna().sum()) for k in ("w5", "w15", "w30", "rand", "leg")}

cmp_ = {}
for k in ("w5", "w15", "w30", "leg"):
    df = W[(W["kind"] == k) & W["cur_dput"].notna()]
    out = {}
    for y, cy in (("dput", "cur_dput"), ("datm", "cur_datm")):
        if k == "leg":
            act = fit_minutes(df, y)["beta"]
            tmp = df.assign(_m=df[cy])
            imp = fit_minutes(tmp, "_m")["beta"]
        else:
            act = fit_alpha(df, y, "single")["coef"]["beta"]
            imp = fit_alpha(df.assign(_m=df[cy]), "_m", "single")["coef"]["beta"]
        out[y] = {"fitted_actual_beta": act, "current_model_implied_beta": imp}
    out["skew_addons_cancel_max_abs_pp"] = float(np.nanmax(np.abs(df["cur_dput"] - df["cur_dput_ss"])))
    out["mean_current_expected_dput"] = round(float(df["cur_dput"].mean()), 3)
    out["mean_actual_dput"] = round(float(df["dput"].mean()), 3)
    cmp_[k] = out
R["step3_compare"] = cmp_


# ------------------------------------------------------------------------------- Step 4: balance
def reads(res, r):
    s = np.sign(res) * np.sign(r)
    return np.where(~np.isfinite(res) | (res == 0), 0, s)      # +1 divergent, -1 confirming, 0 none


def share(df, col, boot=True):
    x = df[["d", col]].copy()
    x = x[x[col] != 0]
    if not len(x):
        return None
    g = x.groupby("d")[col].agg(div=lambda v: int((v > 0).sum()), n="size")
    est = g["div"].sum() / g["n"].sum()
    out = dict(share=round(float(est), 3), n=int(g["n"].sum()), sessions=int(len(g)))
    if boot:
        rng_ = np.random.default_rng(SEED)
        idx = rng_.integers(0, len(g), size=(2000, len(g)))
        dv, nn = g["div"].to_numpy()[idx].sum(1), g["n"].to_numpy()[idx].sum(1)
        bs = dv / nn
        out.update(lo=round(float(np.percentile(bs, 2.5)), 3), hi=round(float(np.percentile(bs, 97.5)), 3))
    return out


def balance_table(df, cols):
    out = {}
    terc = pd.qcut(df["r"].abs(), 3, labels=["small", "mid", "large"])
    for name, col in cols.items():
        t = {"pooled": share(df, col), "up": share(df[df["r"] > 0], col), "down": share(df[df["r"] < 0], col)}
        t["by_size"] = {str(lv): share(df[terc == lv], col, boot=False) for lv in ("small", "mid", "large")}
        t["by_bucket"] = {L.bucket_label(b): {"up": share(df[(df["b"] == b) & (df["r"] > 0)], col, boot=False),
                                              "down": share(df[(df["b"] == b) & (df["r"] < 0)], col, boot=False)}
                          for b in sorted(df["b"].unique())}
        out[name] = t
    return out


def balanced_everywhere(tab, lo=0.45, hi=0.55):
    ok = lo <= tab["pooled"]["share"] <= hi and lo <= tab["up"]["share"] <= hi and lo <= tab["down"]["share"] <= hi
    bad = []
    for bl, ud in tab["by_bucket"].items():
        for k, v in ud.items():
            if v and not (lo <= v["share"] <= hi):
                bad.append((bl, k, v["share"], v["n"]))
    return ok and not bad, bad


SAMPLES = {"rand": W[(W["kind"] == "rand") & W["cur_dput"].notna()].copy(),
           "leg": W[(W["kind"] == "leg") & W["cur_dput"].notna()].copy()}
CURVES = {v: make_curve(SESS, "dput", v) for v in ("single", "updown", "bucket", "bucket_updown")}
for nm, df in SAMPLES.items():
    df["read_raw"] = reads(df["dput"].to_numpy(), df["r"].to_numpy())
    df["read_cur"] = reads((df["dput"] - df["cur_dput"]).to_numpy(), df["r"].to_numpy())
    for v, c in CURVES.items():
        e = np.array([curve_expected(c, rw) for _, rw in df.iterrows()])
        df[f"read_fit_{v}"] = reads((df["dput"] - e).to_numpy(), df["r"].to_numpy())

bal = {}
for nm, df in SAMPLES.items():
    cols = {"raw": "read_raw", "current": "read_cur", **{f"fit_{v}": f"read_fit_{v}" for v in CURVES}}
    bal[nm] = balance_table(df, cols)
R["step4_balance"] = bal

# C8: single beta suffices?
c8 = {}
for v in ("single", "updown", "bucket", "bucket_updown"):
    okv, bad = balanced_everywhere(bal["rand"][f"fit_{v}"])
    c8[v] = {"balanced_everywhere": okv, "cells_outside": bad}
chosen = next((v for v in ("single", "updown", "bucket", "bucket_updown") if c8[v]["balanced_everywhere"]), None)
R["c8"] = {"by_variant": c8, "chosen": chosen}
REC = chosen or "single"

# ------------------------------------------------------------------------------- Step 5: OOS
y26 = [d for d in SESS if PER[d] == "y26"]
ra = [d for d in SESS if PER[d] == "ra"]
half = len(y26) // 2
splits = {"y26_first_half->second_half (K8 primary)": (y26[:half], y26[half:]),
          "ra->y26 (readout)": (ra, y26), "y26->ra (readout)": (y26, ra)}
oos = {}
for nm, (tr, te) in splits.items():
    c = make_curve(tr, "dput", REC)
    te_s = set(te)
    out = {"fit_sessions": [str(tr[0]), str(tr[-1]), len(tr)], "test_sessions": [str(te[0]), str(te[-1]), len(te)],
           "beta": c["beta"][DOM_B].round(3).tolist(), "theta": c["theta"].round(4).tolist()}
    for sn, df in SAMPLES.items():
        x = df[df["d"].isin(te_s)].copy()
        e = np.array([curve_expected(c, rw) for _, rw in x.iterrows()])
        x["read_oos"] = reads((x["dput"] - e).to_numpy(), x["r"].to_numpy())
        t = balance_table(x, {"fit_oos": "read_oos", "current": "read_cur", "raw": "read_raw"})
        out[sn] = {k: {kk: t[k][kk] for kk in ("pooled", "up", "down", "by_size")} for k in t}
    oos[nm] = out
R["step5_oos"] = oos

# ------------------------------------------------------------------------------- recommended + ATM curve
rec = CURVES[REC]
rec_atm = make_curve(SESS, "datm", "single")
c26 = make_curve(y26, "dput", REC)
R["recommended"] = {
    "variant": REC, "source_sample": "w15" if plateau else "leg",
    "put_beta_by_bucket_up_down": {L.bucket_label(b): rec["beta"][b].round(3).tolist() for b in DOM_B},
    "put_theta_per_min_by_bucket": {L.bucket_label(b): round(float(rec["theta"][b]), 4) for b in range(NBK)},
    "atm_beta": round(float(rec_atm["beta"][1, 0]), 3),
    "atm_theta_per_min_by_bucket": {L.bucket_label(b): round(float(rec_atm["theta"][b]), 4) for b in range(NBK)},
    "y26_only_put_beta_by_bucket_up_down": {L.bucket_label(b): c26["beta"][b].round(3).tolist() for b in DOM_B},
    "y26_only_put_theta_per_min_by_bucket": {L.bucket_label(b): round(float(c26["theta"][b]), 4) for b in range(NBK)},
}

(L.OUT_DIR / "results.json").write_text(json.dumps(R, indent=1, default=str))
print(json.dumps({k: R[k] for k in ("sessions", "counts", "c10", "c8", "recommended")}, indent=1, default=str))
