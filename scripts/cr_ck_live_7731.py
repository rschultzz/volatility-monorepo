"""CR-CK Part 2 — stickiness on 2026-10-26 around ES 7731. LIVE READOUT, NOT A TEST. READ-ONLY.

Spec: specs/CR-CK-skew-stickiness.md (C10, C11). One session after the holdout cut, read only for this readout and
never pooled with Part 1. ES = `ironbeam_es_1m_bars` (no Databento after 2026-10-23), cross-checked against the
`ironbeam_es_flow_1s` minute close; where an endpoint differs by > 1 tick the flow close is used (field-conventions).
Writes scripts/logs/cr_ck/live_2026_10_26.json.
"""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd

import cr_cj_lib as L
from packages.shared import skew_expected as se
from packages.shared.flow_features import swing_legs

D = date(2026, 10, 26)
RTH0 = 6 * 60 + 30
LEVEL = 7731.0
OUT = L.REPO_ROOT / "scripts" / "logs" / "cr_ck"


def hm(i: int) -> str:
    m = RTH0 + int(i)
    return f"{m // 60:02d}:{m % 60:02d}"


def idx(h: int, m: int) -> int:
    return h * 60 + m - RTH0


sk = L.q(f"""
    SELECT trade_date, snapshot_pt, snap_shot_date, quote_date, stock_price, {", ".join(L.VOLS)}
    FROM orats_monies_minute
    WHERE ticker = 'SPX' AND trade_date = %s AND expir_date = trade_date AND expiry_tod = 'pm'
      AND snapshot_pt::time >= '06:30' AND snapshot_pt::time < '13:00'
    ORDER BY snapshot_pt
""", (D.isoformat(),))
for c in ["stock_price"] + L.VOLS:
    sk[c] = sk[c].astype(float)
sk["d"] = D
M = L.skew_minutes(sk)[D]                    # A1 quote-time index, same code as Part 1

bars = L.q("""
    SELECT datetime, open::float o, high::float h, low::float l, close::float c FROM ironbeam_es_1m_bars
    WHERE datetime >= %s AND datetime < %s ORDER BY datetime
""", (L.pt(D, 6, 30), L.pt(D, 13, 0)))
flow = L.q("""
    SELECT date_trunc('minute', ts_utc) AS m, (array_agg(close ORDER BY ts_utc DESC))[1]::float AS c
    FROM ironbeam_es_flow_1s WHERE ts_utc >= %s AND ts_utc < %s GROUP BY 1 ORDER BY 1
""", (L.pt(D, 6, 30), L.pt(D, 13, 0)))
start = pd.Timestamp(L.pt(D, 6, 30))


def arr(df, tcol, col):
    a = np.full(L.NB, np.nan)
    i = ((pd.to_datetime(df[tcol], utc=True) - start).dt.total_seconds() // 60).astype(int).to_numpy()
    ok = (i >= 0) & (i < L.NB)
    a[i[ok]] = df[col].to_numpy(float)[ok]
    return a


H, Lo, C = arr(bars, "datetime", "h"), arr(bars, "datetime", "l"), arr(bars, "datetime", "c")
CF = arr(flow, "m", "c")
R = {"note": "live readout, not a test", "bars_vs_flow_close": {
    "minutes": int(np.isfinite(C).sum()), "equal": int((np.abs(C - CF) < 1e-9).sum()),
    "gt_1_tick": int((np.abs(C - CF) > 0.25 + 1e-9).sum())}}


def es_at_quote(i: int) -> tuple[float, str]:
    """ES close at snapshot i's quote minute Q (close of the bar starting Q - 1); flow if the bar is > 1 tick off."""
    q = M["qidx"][i]
    if q < 1:
        return np.nan, "none"
    b, f = C[q - 1], CF[q - 1]
    if np.isfinite(f) and (not np.isfinite(b) or abs(b - f) > 0.25 + 1e-9):
        return float(f), "flow"
    return float(b), "bar"


def window(name: str, s: int, e: int) -> dict:
    put = M["put"]
    j = M["row"][s]
    row = M["rows"].iloc[j]
    vols = {n: float(row[f"vol{n}"]) for n in range(5, 100, 5)}
    (es0, src0), (es1, src1) = es_at_quote(s), es_at_quote(e)
    ii = np.arange(s, e + 1)
    ok = np.isfinite(put[ii])
    rd = se.stickiness_read((ii[ok] + RTH0).tolist(), put[ii][ok].tolist(), s + RTH0, e + RTH0, vols, es0, es1)
    return dict(window=name, start=hm(s), end=hm(e), minutes=int(e - s), snapshots=int(ok.sum()),
                es_start=es0, es_end=es1, es_src=f"{src0}/{src1}", es_move_pct=round(rd.es_move_pct, 3),
                es_move_pts=round(es1 - es0, 2), put_skew_start=round(float(put[s]), 2),
                put_skew_end=round(float(put[e]), 2), endpoint_change=round(float(put[e] - put[s]), 2),
                actual_trend=None if rd.actual_pp is None else round(rd.actual_pp, 2),
                expected_ss=None if rd.expected_pp is None else round(rd.expected_pp, 2),
                ratio=None if rd.ratio is None else round(rd.ratio, 2), reason=rd.reason,
                band=None if rd.band_pp is None else round(rd.band_pp, 2), beyond_noise=rd.beyond_noise)


# C11b: the swing-12 legs on the Ironbeam bars; the one containing 07:30-07:50, and that leg cut at the first 7731 trade
legs = swing_legs(H, Lo, 12.0)
R["swing12_legs_0700_0900"] = [dict(dir=lg["dir"], start=hm(lg["start_i"]), end=hm(lg["end_i"]),
                                    confirm=hm(lg["confirm_i"]), start_px=lg["start_px"], end_px=lg["end_px"])
                               for lg in legs if idx(7, 0) <= lg["end_i"] and lg["start_i"] <= idx(9, 0)]
leg = next(lg for lg in legs if lg["start_i"] <= idx(7, 40) <= lg["end_i"])
s_leg, e_leg = leg["start_i"] + 1, leg["end_i"] + 1          # snapshot at the extreme bar's close (Part 1 alignment)
touch = next(i for i in range(leg["start_i"], leg["end_i"] + 1) if H[i] >= LEVEL)
R["first_7731_bar"] = hm(touch)

rows = [
    window("a. approach as named", idx(7, 30), idx(7, 50)),
    window("b1. swing-12 leg -> first 7731 trade", s_leg, touch + 1),
    window("b2. swing-12 leg, full", s_leg, e_leg),
    window("c. range as named", idx(7, 50), idx(8, 48)),
]
R["windows"] = rows
(OUT / "live_2026_10_26.json").write_text(json.dumps(R, indent=2, default=str))
print(json.dumps({k: v for k, v in R.items() if k != "windows"}, indent=1, default=str))
print(pd.DataFrame(rows).to_string(index=False))
