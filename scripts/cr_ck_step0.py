"""CR-CK Step 0 — coverage and counts. READ-ONLY, no outcomes, no fits.

Spec: specs/CR-CK-skew-stickiness.md. Part 1 data is 2026 train only (CR-CJ loaders, period `y26`); R-a and 2025 are
never loaded. Part 2's live session (2026-10-26) is checked for coverage only. Writes scripts/logs/cr_ck/step0.json.
"""
from __future__ import annotations

import json
from datetime import date

import numpy as np

import cr_cj_lib as L
from packages.shared.flow_features import swing_legs

OUT = L.REPO_ROOT / "scripts" / "logs" / "cr_ck"
OUT.mkdir(parents=True, exist_ok=True)
DOM_LO, DOM_HI = 30, L.NB - 15
LIVE = date(2026, 10, 26)
R = {}

# ---------------------------------------------------------------- 2026 train coverage (CR-CJ loaders, y26 only)
sk = L.load_skew("y26")
MIN = L.skew_minutes(sk)
_, BARS = L.front_bars(L.load_es("y26"))
sess = L.sessions("y26")
usable = []
for d in sess:
    if d not in MIN or d not in BARS:
        continue
    px = L.es_close_at_snapshot(BARS, d, MIN)
    ok = np.isfinite(MIN[d]["put"]) & np.isfinite(px)
    if ok[DOM_LO:DOM_HI + 1].mean() >= 0.9:
        usable.append(d)
R["train_2026"] = {"sessions": len(sess), "usable_ge_90pct_domain_minutes": len(usable),
                   "first": str(sess[0]), "last": str(sess[-1])}

# swing-12 legs inside the domain (C7 alignment)
n_leg = 0
for d in usable:
    H, Lo = BARS[d]["h"], BARS[d]["l"]
    if not (np.isfinite(H).all() and np.isfinite(Lo).all()):
        continue
    for lg in swing_legs(H, Lo, 12.0):
        s, e = lg["start_i"] + 1, lg["end_i"] + 1
        if s >= DOM_LO and e <= DOM_HI:
            n_leg += 1
R["train_2026"]["swing12_legs_in_domain"] = n_leg
lens = []
for d in usable:
    H, Lo = BARS[d]["h"], BARS[d]["l"]
    if np.isfinite(H).all() and np.isfinite(Lo).all():
        lens += [lg["end_i"] - lg["start_i"] for lg in swing_legs(H, Lo, 12.0)
                 if lg["start_i"] + 1 >= DOM_LO and lg["end_i"] + 1 <= DOM_HI]
R["train_2026"]["leg_minutes_p10_p50_p90_max"] = [int(x) for x in np.percentile(lens, [10, 50, 90, 100])]

# ---------------------------------------------------------------- live session coverage (Part 2), counts only
lv = L.q("""
    SELECT count(*) AS n, min(snapshot_pt) AS a, max(snapshot_pt) AS b,
           count(*) FILTER (WHERE snapshot_pt::time >= '07:20' AND snapshot_pt::time <= '08:50') AS n_window,
           percentile_cont(0.5) WITHIN GROUP (ORDER BY extract(epoch FROM (snap_shot_date::timestamptz - quote_date::timestamptz))) AS lag_med_s
    FROM orats_monies_minute
    WHERE ticker = 'SPX' AND trade_date = %s AND expir_date = trade_date AND expiry_tod = 'pm'
      AND snapshot_pt::time >= '06:30' AND snapshot_pt::time < '13:00'
""", (LIVE.isoformat(),)).iloc[0]
bars = L.q("""
    SELECT count(*) AS n, array_agg(DISTINCT source) AS src FROM ironbeam_es_1m_bars
    WHERE datetime >= %s AND datetime < %s
""", (L.pt(LIVE, 6, 30), L.pt(LIVE, 13, 0))).iloc[0]
flow = L.q("""
    SELECT count(DISTINCT date_trunc('minute', ts_utc)) AS n_min FROM ironbeam_es_flow_1s
    WHERE ts_utc >= %s AND ts_utc < %s
""", (L.pt(LIVE, 6, 30), L.pt(LIVE, 13, 0))).iloc[0]
R["live_2026_10_26"] = {"skew_rows_rth": int(lv["n"]), "first": str(lv["a"]), "last": str(lv["b"]),
                        "skew_rows_07_20_08_50": int(lv["n_window"]), "quote_lag_median_s": float(lv["lag_med_s"]),
                        "ironbeam_bars_rth": int(bars["n"]), "bar_source": list(bars["src"]),
                        "flow_minutes_rth": int(flow["n_min"])}

(OUT / "step0.json").write_text(json.dumps(R, indent=2, default=str))
print(json.dumps(R, indent=2, default=str))
