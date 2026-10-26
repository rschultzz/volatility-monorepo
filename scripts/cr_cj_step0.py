"""CR-CJ Step 0 — coverage, symbols, ES cross-check, alignment, sample counts. READ-ONLY, no fits.

Writes scripts/logs/cr_cj/step0.json and prints a summary.
"""
from __future__ import annotations

import json
from datetime import date, datetime, time

import numpy as np
import pandas as pd

import cr_cj_lib as L
from packages.shared.flow_features import swing_legs

OUT = {}
SK, ES, SYM, BARS, MIN = {}, {}, {}, {}, {}
for p in L.PERIODS:
    SK[p] = L.load_skew(p)
    ES[p] = L.load_es(p)
    s, b = L.front_bars(ES[p])
    SYM.update(s)
    BARS.update(b)
    MIN.update(L.skew_minutes(SK[p]))

# ------------------------------------------------------------------ coverage per period
cov = {}
for p in L.PERIODS:
    ss = L.sessions(p)
    sk_days = set(SK[p]["d"])
    es_days = set(BARS) & set(ss)
    per = []
    for d in ss:
        m = MIN.get(d)
        n_sk = int(np.isfinite(m["put"]).sum()) if m else 0
        n_es = int(np.isfinite(BARS[d]["c"]).sum()) if d in BARS else 0
        both = 0
        if m and d in BARS:
            both = int((np.isfinite(m["put"]) & np.isfinite(L.es_close_at_snapshot(BARS, d, MIN))).sum())
        per.append(dict(d=d, sk=n_sk, es=n_es, both=both))
    per = pd.DataFrame(per)
    usable = per[per["both"] >= 0.9 * (L.NB - 1)]
    cov[p] = dict(sessions=len(ss), with_skew=int(len(sk_days & set(ss))), with_es=len(es_days),
                  usable_ge90pct_minutes=int(len(usable)), usable_share=round(len(usable) / len(ss), 3),
                  skew_minutes_median=float(per["sk"].median()), es_minutes_median=float(per["es"].median()),
                  short_sessions=[(str(r.d), int(r.sk), int(r.es)) for r in per.itertuples() if r.both < 0.9 * (L.NB - 1)])
OUT["coverage"] = cov

# third Fridays: is there a pm 0DTE row?
tf = {}
for p in L.PERIODS:
    for d in L.sessions(p):
        if d.weekday() == 4 and 15 <= d.day <= 21:
            tf[str(d)] = int(np.isfinite(MIN[d]["put"]).sum()) if d in MIN else 0
OUT["third_friday_pm_minutes"] = tf

# ------------------------------------------------------------------ symbols
symv = pd.concat([ES[p] for p in L.PERIODS]).groupby(["d", "symbol"])["vol"].sum().reset_index()
two = symv.groupby("d").filter(lambda g: len(g) > 1)
share = {str(d): round(float(g["vol"].max() / g["vol"].sum()), 3) for d, g in two.groupby("d")}
OUT["symbols"] = dict(by_symbol={s: [str(min(ds)), str(max(ds)), len(ds)]
                                 for s, ds in pd.Series(SYM).groupby(pd.Series(SYM)).groups.items()},
                      two_symbol_sessions=len(share), front_share_min=min(share.values()) if share else None)

# ------------------------------------------------------------------ Databento vs Ironbeam close, 2026 sample
rng = np.random.default_rng(20261026)
smp = sorted(rng.choice([d for d in L.sessions("y26") if d in BARS], 10, replace=False))
diffs = []
for d in smp:
    L.guard(d, d)
    ib = L.q("""SELECT datetime, close::float AS c FROM ironbeam_es_1m_bars
                WHERE datetime >= %s AND datetime < %s ORDER BY datetime""", (L.pt(d, 6, 30), L.pt(d, 13, 0)))
    ib["i"] = ((pd.to_datetime(ib["datetime"], utc=True) - pd.Timestamp(L.pt(d, 6, 30))).dt.total_seconds() // 60).astype(int)
    db = BARS[d]["c"]
    x = np.array([c - db[i] for i, c in zip(ib["i"], ib["c"]) if 0 <= i < L.NB and np.isfinite(db[i])])
    diffs.append(x)
x = np.concatenate(diffs)
OUT["es_xcheck_2026"] = dict(sessions=[str(d) for d in smp], minutes=int(len(x)), equal_share=round(float((x == 0).mean()), 4),
                             abs_le_1tick=round(float((np.abs(x) <= 0.25).mean()), 4),
                             p99_abs=float(np.percentile(np.abs(x), 99)), max_abs=float(np.abs(x).max()))

# ------------------------------------------------------------------ alignment: 1-min dSPX(stock_price) vs dES, lags
lag_corr = {}
for mode in ("label_C4", "quote_A1"):
  for p in L.PERIODS:
    for lag in range(-3, 4):
        a, b = [], []
        for d in L.sessions(p):
            if d not in MIN or d not in BARS:
                continue
            spx, px = MIN[d]["spx"], L.es_close_at_snapshot(BARS, d, MIN if mode == "quote_A1" else None)
            pxl = np.full(L.NB, np.nan)                        # ES price at T + lag, on snapshot T's index
            if lag >= 0:
                pxl[:L.NB - lag] = px[lag:]
            else:
                pxl[-lag:] = px[:L.NB + lag]
            u, v = np.diff(spx), np.diff(pxl)
            k = np.isfinite(u) & np.isfinite(v)
            k[:30] = False                                     # skip 06:30-07:00
            a.append(u[k]); b.append(v[k])
        a, b = np.concatenate(a), np.concatenate(b)
        lag_corr.setdefault(mode, {}).setdefault(p, {})[lag] = round(float(np.corrcoef(a, b)[0, 1]), 3)
OUT["alignment_lag_corr"] = lag_corr      # lag 0 = the mode's pairing

# quote lag vs label by period (A1)
OUT["quote_lag_s"] = {}
for p in L.PERIODS:
    lag = (pd.to_datetime(SK[p]["snap_shot_date"], utc=True) - pd.to_datetime(SK[p]["quote_date"], utc=True)).dt.total_seconds()
    OUT["quote_lag_s"][p] = dict(p5=float(lag.quantile(.05)), p50=float(lag.median()), p95=float(lag.quantile(.95)),
                                 abs_gt_300=int((lag.abs() > 300).sum()), n=int(len(lag)))

# ------------------------------------------------------------------ sample counts
cnt = {}
for p in L.PERIODS:
    c = dict(w5=0, w15=0, w30=0, legs=0, legs_domain=0, sessions=0)
    for d in L.sessions(p):
        if d not in MIN or d not in BARS:
            continue
        c["sessions"] += 1
        put, px = MIN[d]["put"], L.es_close_at_snapshot(BARS, d, MIN)
        ok = np.isfinite(put) & np.isfinite(px)
        for h in (5, 15, 30):
            for s in range(30, L.NB - 15 - h + 1, h):          # start >= 07:00, end <= 12:45
                c[f"w{h}"] += int(ok[s] and ok[s + h])
        H, Lo = BARS[d]["h"], BARS[d]["l"]
        if np.isfinite(H).all() and np.isfinite(Lo).all():
            lg = swing_legs(H, Lo, 12.0)
            c["legs"] += len(lg)
            c["legs_domain"] += sum(1 for x in lg if x["start_i"] + 1 >= 30 and x["end_i"] + 1 <= L.NB - 15
                                    and ok[x["start_i"] + 1] and ok[x["end_i"] + 1])
    cnt[p] = c
OUT["counts"] = cnt

(L.OUT_DIR / "step0.json").write_text(json.dumps(OUT, indent=1, default=str))
print(json.dumps(OUT, indent=1, default=str))
