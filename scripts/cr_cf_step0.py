#!/usr/bin/env python3
"""CR-CF Step 0 (Stage 1, train 2026-01-06 .. 2026-06-05) — which levels does price respect. READ-ONLY.

Spec: specs/CR-CF-which-levels-price-respects.md (L1–L12, Step 0 Q1–Q3). Price paths only; no trade outcome.
Step 0 computes NO real-level T0 measure: Q1 counts levels and tests (real and placebo), Q2 is the A/A on placebo
tests only (primary measure), Q3 lists levels on 2026-05-01. The real − placebo screen is Stage 1, not run here.

Copied unchanged from scripts/cr_cb_t0_ra_a17.py (CR-CB A17; copy, not import): connection / read-only guard,
third_friday / roll_week / pt / ep, front symbol = largest RTH volume per PT date, Databento 1-s loader, rth_slice,
universe rules (roll week; no ES 15:45 ET bar on A; quarterly-expiry Friday; no Databento RTH flow; early close =
last RTH print < 12:55 PT; no canonical IM; no orats_oi_gamma rows; contract check), A1.1 select_levels (the
Corrected GEX levels, used only for the L11 overlap count), the A17 test (first armed test, arming
max(0.10·IM, 4), 2-pt band), restricted-mean time to escape, session-weighted means, seeds. From
scripts/cr_cb_step0_v3.py: the 2026 IM / train date guard, overnight() and swings15() (A15.1), the A/A split design
(200 splits by pseudo-level, 2,000 session resamples per split).

Choices fixed here before the run (the spec leaves them open):
  - Guard: every read is in [2026-01-01 00:00 UTC, 2026-06-05 21:00 UTC). 2025 is not read: F6 is missing for a
    session whose prior week has a day before 2026-01-01 (week of 01-05..01-09); F1 needs the prior session in 2026.
  - Contract consistency (L2): F1 / F6 use today's front symbol's Databento RTH prints, and only prior sessions whose
    own front (largest RTH volume) is that symbol; otherwise the family is missing that session (counted). F2 uses
    ironbeam_es_1m_bars (no symbol column) 15:00 PT prior session → 06:30 PT, and is kept only if the close of the
    bar starting 06:29 PT is within 2 pts of today's-symbol Databento last close in [06:29, 06:30) PT.
  - F1 prior RTH high / low / close; F2 overnight high / low; F3 opening range high / low from [06:30, 07:00) PT,
    armed from 07:00:00 PT; F6 prior calendar week (Mon–Fri) RTH high / low; all armed from 06:30 PT.
  - F4 VWAP from 06:30: per-second typical price (h + l + c) / 3 × volume, cumulative; the level at second k is the
    VWAP through second k − 1 (causal); moving. Arming, band, escape and dwell are measured against VWAP at that
    second. Tests from the second RTH row.
  - F5 round numbers, sub-families R100 (multiples of 100), R50 (of 50, not 100), R25 (of 25, not 50); armed 06:30.
  - F7 confirmed swings per A15.1 (swings15 copied from v3: zigzag at max(8, 0.15·IM), unviolated 600 s); a swing
    is armed from its known time max(confirmation, swing + 600 s).
  - L4 window: static levels within ±1·IM_d of the RTH open (first Databento RTH open); F4 always. Within a family
    and session, levels closer than 0.5 pt are merged (first kind kept).
  - L7 placebo, static families: per real (in-window) level, the 4 nearest 5-pt bins on the same side of the open,
    within ±1·IM_d, ≥ 10 pts from every level of that family in the session (all levels of the family, in or out of
    window; for F7 every kept swing of the day, known or not); dedup within family; each placebo inherits its
    parent's armed-from time (F3 07:00, F7 the parent swing's known time; a bin claimed by several parents takes the
    earliest). F5: bins must also be > 3 pts from every multiple of 25 (so bins ≡ 5, 10, 15, 20 mod 25).
    F4: 4 offsets per session, sign ±1 with p ½ and magnitude U[10, 30] pts, from one default_rng(20261025) stream
    drawn in session order; placebo level = VWAP(t) + offset.
  - Cells (L8): approach from below (side +1) / above (−1), plus pooled, per family. No put / call split.
  - Primary measure (A/A only in Step 0): min(minutes to a trade ≥ 0.25·IM_d from the level, 120, minutes to
    13:00 PT).
  - Q2 A/A per family × {pooled, below, above} and over all families pooled: placebo tests only. Pseudo-level keys
    (session, family, placebo price or VWAP offset) shuffled with default_rng(20261025) and cut into halves 200 times.
    Statistic = session-weighted mean of the half-1 tests − session-weighted mean of the half-2 tests (A17.3
    weighting: per-session mean, then mean over sessions with ≥ 1 test in that half). CI: 2,000 session-multinomial
    resamples per split, one default_rng(20260920) stream across splits; 95 % percentile. PASS if the CI excludes 0 in
    ≤ 10 % of splits and the median statistic is within ±3 min (the CR-BW ±3 tolerance in this measure's units).
  - Q1 also reports (counts, no outcome): sessions with ≥ 1 test, tests before / after 10:00 PT (L11), the L11
    overlap share (static family levels within 5 pts of a Corrected GEX level of that session; F4 at its test
    second), the share of non-F5 placebo levels that sit on a 25-multiple, and IM < 25 sessions.
  - Q3 2026-05-01: every family level in [7266, 7282] known by 08:10 PT with its known time and its first A17 test
    (time, side; the Corrected GEX levels in the band too); VWAP over 07:30–08:10;
    the price path 07:30–08:10 (low and its time, minute closes); every F7 swing (kept or not) and every earlier
    session minute low in [7266, 7282]; Corrected GEX levels in that band. No outcome.

Usage: apps/web/.venv/bin/python -u scripts/cr_cf_step0.py | tee scripts/logs/cr_cf/step0.log
"""
from __future__ import annotations

import json
import os
import signal
import sys
import warnings
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parent.parent
for line in (REPO_ROOT / ".env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())
sys.path.insert(0, str(REPO_ROOT))

warnings.filterwarnings("ignore")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from packages.shared.backfill_safety import assert_role_or_die, get_backfill_db_conn  # noqa: E402
from packages.shared.canonical_version import CANONICAL_FEATURE_VERSION  # noqa: E402
from packages.shared.level_placement import (  # noqa: E402
    bucket_levels, es_snapshot_bar_start_utc, place_levels)
from packages.shared.trading_calendar import is_trading_day, next_trading_day, prev_trading_day  # noqa: E402

signal.alarm(3 * 3600)

PT, UTC = ZoneInfo("America/Los_Angeles"), timezone.utc
D_FROM, D_TO = date(2026, 1, 6), date(2026, 6, 5)
FLOOR_TS, HOLDOUT_TS = datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 6, 5, 21, 0, tzinfo=UTC)
WORKED = date(2026, 5, 1)
WB_LO, WB_HI = 7266.0, 7282.0
DTE_MAX, MAG_MIN, SPACING, WIN_IM, TOP_N = 30, 40e9, 5.0, 1.0, 6
ARM_IM, ARM_MIN, BAND, IM_FLAG = 0.10, 4.0, 2.0, 25.0
PLC_BIN, PLC_SEP, PLC_K, R25_EXCL = 5.0, 10.0, 4, 3.0
VW_OFF_LO, VW_OFF_HI = 10.0, 30.0
SW_PTS, SW_IM, SW_HOLD = 8.0, 0.15, 600
ESC_IM, ESC_CAP = 0.25, 120
OVL_PTS, MERGE_PTS = 5.0, 0.5
SEED_DRAW, SEED_BOOT, N_SPLIT, N_BOOT_SPLIT = 20261025, 20260920, 200, 2_000
AA_EXCL_MAX, AA_MED_TOL = 0.10, 3.0
FAMILIES = ["F1", "F2", "F3", "F4", "F5_100", "F5_50", "F5_25", "F6", "F7"]
OUT_DIR = REPO_ROOT / "scripts" / "logs" / "cr_cf"

CONN = get_backfill_db_conn()
assert_role_or_die(CONN)
CONN.execute("SET default_transaction_read_only = on")
CONN.execute("SET statement_timeout = '600s'")
assert CONN.execute("SELECT current_setting('default_transaction_read_only')").fetchone()[0] == "on"


def q(sql, params=None) -> pd.DataFrame:
    cur = CONN.execute(sql, params)
    return pd.DataFrame(cur.fetchall(), columns=[c.name for c in cur.description])


def third_friday(y, m):
    d = date(y, m, 15)
    return d + timedelta(days=(4 - d.weekday()) % 7)


def roll_week(d):
    if d.month not in (3, 6, 9, 12):
        return False
    tf = third_friday(d.year, d.month)
    return tf - timedelta(days=11) <= d <= tf


def pt(d, hh, mm, ss=0):
    return datetime.combine(d, time(hh, mm, ss), PT).astimezone(UTC)


def ep(x):
    return int(x.timestamp())


def hms(t):
    return datetime.fromtimestamp(int(t), UTC).astimezone(PT).strftime("%H:%M:%S")


def guard(t0, t1):
    assert FLOOR_TS <= t0 and t1 <= HOLDOUT_TS, (t0, t1)
    return t0, t1


def trading_days(a, b):
    out, d = [], a
    while d <= b:
        if is_trading_day(d):
            out.append(d)
        d += timedelta(days=1)
    return out


SESSIONS = trading_days(D_FROM, D_TO)
QEXP = {third_friday(2026, m) for m in (3, 6, 9, 12)}
ALLDAYS = trading_days(date(2026, 1, 2), D_TO)                 # 2026 sessions only (prior session / week)

# ----------------------------------------------------------------------------------------------- loads
sym = q("""
    SELECT (ts_utc AT TIME ZONE 'America/Los_Angeles')::date AS d, symbol, sum(volume) AS vol
    FROM es_flow_1s_databento
    WHERE ts_utc >= %s AND ts_utc < %s
      AND (ts_utc AT TIME ZONE 'America/Los_Angeles')::time >= '06:30'
      AND (ts_utc AT TIME ZONE 'America/Los_Angeles')::time < '13:00'
    GROUP BY 1, 2
""", guard(FLOOR_TS, HOLDOUT_TS)).sort_values("vol")
SYM = dict(zip(sym["d"], sym["symbol"]))


def load_flow(d_, symbol):
    f = q("""SELECT ts_utc, open, high, low, close, volume FROM es_flow_1s_databento
             WHERE symbol = %s AND ts_utc >= %s AND ts_utc < %s ORDER BY ts_utc""",
          (symbol, *guard(pt(d_, 6, 15), pt(d_, 13, 0))))
    return dict(t=np.array([ep(x) for x in f["ts_utc"]], dtype=np.int64), o=f["open"].to_numpy(float),
                h=f["high"].to_numpy(float), l=f["low"].to_numpy(float), c=f["close"].to_numpy(float),
                v=f["volume"].to_numpy(float))


FLOW = {d_: load_flow(d_, SYM[d_]) for d_ in SESSIONS if d_ in SYM}

oig = q("""
    SELECT trade_date, expir_date, strike::float AS strike, stock_price::float AS stock_price, gex_call, gex_put
    FROM orats_oi_gamma
    WHERE ticker = 'SPX' AND trade_date BETWEEN %s AND %s
      AND expir_date >= trade_date AND expir_date <= trade_date + %s
""", (D_FROM - timedelta(days=7), D_TO, DTE_MAX + 7))
oig["d"] = [t if is_trading_day(t) else next_trading_day(t) for t in oig["trade_date"]]
oig = oig[(oig["d"] >= D_FROM) & (oig["d"] <= D_TO)]
oig = oig[[d_ <= e <= d_ + timedelta(days=DTE_MAX) for e, d_ in zip(oig["expir_date"], oig["d"])]]
oig["call"] = oig["gex_call"].astype(float)
oig["put"] = -oig["gex_put"].abs().astype(float)

feat = q("""
    SELECT trade_date, (feature_vector->>'implied_move_1d')::float AS im FROM bt_daily_features
    WHERE feature_version = %s AND ticker = 'SPX' AND active AND trade_date BETWEEN %s AND %s
""", (CANONICAL_FEATURE_VERSION, D_FROM, D_TO)).set_index("trade_date")["im"]
feat = feat[~feat.index.duplicated()]

starts = {es_snapshot_bar_start_utc(prev_trading_day(d_)): d_ for d_ in SESSIONS}
assert min(starts) >= FLOOR_TS.replace(tzinfo=None) and max(starts) < HOLDOUT_TS.replace(tzinfo=None)
es = q("SELECT datetime, close::float AS close FROM ironbeam_es_1m_bars WHERE datetime = ANY(%s)", (list(starts),))
PTGT = {starts[pd.Timestamp(t).to_pydatetime()]: c for t, c in zip(es["datetime"], es["close"])
        if c is not None and np.isfinite(c) and c > 0}

ACHK = {}
for st, d_ in starts.items():
    a_ = prev_trading_day(d_)
    if a_ not in SYM:
        continue
    r = q("""SELECT close FROM es_flow_1s_databento WHERE symbol = %s AND ts_utc >= %s AND ts_utc < %s
             ORDER BY ts_utc DESC LIMIT 1""",
          (SYM[a_], *guard(st.replace(tzinfo=UTC), st.replace(tzinfo=UTC) + timedelta(minutes=1))))
    if len(r):
        ACHK[d_] = (SYM[a_], float(r["close"][0]))


def rth_slice(F, d_):
    return np.searchsorted(F["t"], ep(pt(d_, 6, 30))), np.searchsorted(F["t"], ep(pt(d_, 13, 0)))


OPEN, LAST = {}, {}
for d_, F in FLOW.items():
    a, b = rth_slice(F, d_)
    if b > a:
        OPEN[d_], LAST[d_] = float(F["o"][a]), int(F["t"][b - 1])

U, funnel = list(SESSIONS), []
for name, rule in [
    ("roll week", lambda d_: roll_week(d_)),
    ("no ES 15:45 ET bar on A", lambda d_: d_ not in PTGT),
    ("quarterly-expiry Friday", lambda d_: d_ in QEXP),
    ("no Databento RTH flow", lambda d_: d_ not in OPEN),
    ("early close (last RTH print < 12:55 PT)", lambda d_: LAST[d_] < ep(pt(d_, 12, 55))),
    ("no canonical IM / IM <= 0", lambda d_: not (d_ in feat.index and pd.notna(feat[d_]) and feat[d_] > 0)),
    ("no orats_oi_gamma rows", lambda d_: d_ not in set(oig["d"])),
    ("contract check (A symbol != D symbol, or |P_target − A flow 15:45 close| > 2)",
     lambda d_: d_ not in ACHK or ACHK[d_][0] != SYM[d_] or abs(ACHK[d_][1] - PTGT[d_]) > 2.0),
]:
    rm = [d_ for d_ in U if rule(d_)]
    U = [d_ for d_ in U if not rule(d_)]
    funnel.append([name, len(rm), len(U), [str(x) for x in rm]])
USABLE = U
IM = {d_: float(feat[d_]) for d_ in USABLE}
SIDX = {d_: k for k, d_ in enumerate(USABLE)}
nS = len(USABLE)
assert WORKED in USABLE, "2026-05-01 not in the universe"


# --------------------------------------------------------------------------------------- GEX (L11 only)
def select_levels(g, ptgt, o, im):
    g = place_levels(g, ptgt)
    g = g[g["level_corrected"].notna()].copy()
    g["level"] = bucket_levels(g["level_corrected"].to_numpy(), 1.0)
    bk = g.groupby("level").agg(call=("call", "sum"), put=("put", "sum")).reset_index()
    bk["mag"] = bk["call"].abs() + bk["put"].abs()
    bk["net"] = bk["call"] + bk["put"]
    bk = bk[bk["mag"] >= MAG_MIN].sort_values("mag", ascending=False)
    kept = []
    for r in bk.itertuples():
        if all(abs(r.level - k.level) >= SPACING for k in kept):
            kept.append(r)
    win = [k for k in kept if abs(k.level - o) <= WIN_IM * im]
    return [dict(L=float(k.level), mag=float(k.mag), net=float(k.net)) for k in win[:TOP_N]]


GEX = {d_: [x["L"] for x in select_levels(g, PTGT[d_], OPEN[d_], IM[d_])]
       for d_, g in oig[oig["d"].isin(USABLE)].groupby("d")}


# ------------------------------------------------------------------------------------ family levels
def rth_hlc(d_, symbol):
    r = q("""SELECT max(high)::float h, min(low)::float l, (array_agg(close ORDER BY ts_utc DESC))[1]::float c,
                    count(*) n
             FROM es_flow_1s_databento WHERE symbol = %s AND ts_utc >= %s AND ts_utc < %s""",
          (symbol, *guard(pt(d_, 6, 30), pt(d_, 13, 0))))
    return None if not r["n"][0] else dict(h=r["h"][0], l=r["l"][0], c=r["c"][0])


def overnight(d_):
    p_ = prev_trading_day(d_)
    t0, t1 = guard(pt(p_, 15, 0), pt(d_, 6, 30))
    r = q("""SELECT max(high)::float h, min(low)::float l, count(*) n FROM ironbeam_es_1m_bars
             WHERE datetime >= %s AND datetime < %s""", (t0.replace(tzinfo=None), t1.replace(tzinfo=None)))
    if not r["n"][0]:
        return None, "no bars"
    ib = q("SELECT close::float c FROM ironbeam_es_1m_bars WHERE datetime = %s",
           (pt(d_, 6, 29).replace(tzinfo=None),))
    F = FLOW[d_]
    k0, k1 = np.searchsorted(F["t"], ep(pt(d_, 6, 29))), np.searchsorted(F["t"], ep(pt(d_, 6, 30)))
    if not len(ib) or k1 <= k0:
        return None, "no 06:29 bar or flow"
    if abs(ib["c"][0] - F["c"][k1 - 1]) > 2.0:
        return None, f"06:29 bar close {ib['c'][0]} vs flow {F['c'][k1 - 1]}"
    return dict(h=r["h"][0], l=r["l"][0]), None


def swings15(F, d_, im):
    """Zigzag at max(8, 0.15·IM); a swing is kept only if not violated for 10 min after it forms (A15.1)."""
    a, b = rth_slice(F, d_)
    thr = max(SW_PTS, SW_IM * im)
    t, h, l = F["t"], F["h"], F["l"]
    out, hi, lo, hi_i, lo_i, mode = [], h[a], l[a], a, a, 0
    for k in range(a, b):
        if h[k] > hi:
            hi, hi_i = h[k], k
        if l[k] < lo:
            lo, lo_i = l[k], k
        if mode >= 0 and l[k] <= hi - thr and hi_i < k:
            e_ = np.searchsorted(t, t[hi_i] + SW_HOLD, side="right")
            ok = not np.any(h[hi_i + 1:e_] > hi) and t[hi_i] + SW_HOLD <= ep(pt(d_, 13, 0))
            out.append(dict(px=float(hi), kind="swing_high", at=int(t[hi_i]), known=max(int(t[k]), int(t[hi_i]) + SW_HOLD),
                            kept=ok))
            mode, lo, lo_i = -1, l[k], k
        elif mode <= 0 and h[k] >= lo + thr and lo_i < k:
            e_ = np.searchsorted(t, t[lo_i] + SW_HOLD, side="right")
            ok = not np.any(l[lo_i + 1:e_] < lo) and t[lo_i] + SW_HOLD <= ep(pt(d_, 13, 0))
            out.append(dict(px=float(lo), kind="swing_low", at=int(t[lo_i]), known=max(int(t[k]), int(t[lo_i]) + SW_HOLD),
                            kept=ok))
            mode, hi, hi_i = 1, h[k], k
    return out


def merge(lv):
    out = []
    for x in sorted(lv, key=lambda z: (z["known"], z["L"])):
        if all(abs(x["L"] - y["L"]) >= MERGE_PTS for y in out):
            out.append(x)
    return out


MISSING = {f: [] for f in FAMILIES}
SWALL = {}


def family_levels(d_):
    """All levels of each static family for the session (in and out of the L4 window); F4 handled separately."""
    F, s_ = FLOW[d_], SYM[d_]
    t630, t700 = ep(pt(d_, 6, 30)), ep(pt(d_, 7, 0))
    fam = {f: [] for f in FAMILIES if f != "F4"}
    p_ = prev_trading_day(d_)
    if p_ in ALLDAYS and SYM.get(p_) == s_ and (r := rth_hlc(p_, s_)) is not None:
        fam["F1"] = [dict(L=r["h"], kind="prior_high", known=t630), dict(L=r["l"], kind="prior_low", known=t630),
                     dict(L=r["c"], kind="prior_close", known=t630)]
    else:
        MISSING["F1"].append(str(d_))
    on, why = overnight(d_)
    if on is not None:
        fam["F2"] = [dict(L=on["h"], kind="overnight_high", known=t630),
                     dict(L=on["l"], kind="overnight_low", known=t630)]
    else:
        MISSING["F2"].append([str(d_), why])
    a, _ = rth_slice(F, d_)
    k7 = np.searchsorted(F["t"], t700)
    if k7 > a:
        fam["F3"] = [dict(L=float(F["h"][a:k7].max()), kind="or_high", known=t700),
                     dict(L=float(F["l"][a:k7].min()), kind="or_low", known=t700)]
    else:
        MISSING["F3"].append(str(d_))
    o, im = OPEN[d_], IM[d_]
    for k_ in np.arange(np.floor((o - 2 * im) / 25) * 25, o + 2 * im + 25, 25):
        f = "F5_100" if k_ % 100 == 0 else ("F5_50" if k_ % 50 == 0 else "F5_25")
        fam[f].append(dict(L=float(k_), kind=f, known=t630))
    mon = d_ - timedelta(days=d_.weekday())
    pw = [x for x in trading_days(mon - timedelta(days=7), mon - timedelta(days=3))]
    if pw and min(pw) >= date(2026, 1, 1) and (pw_ok := [x for x in pw if SYM.get(x) == s_]):
        hl = [rth_hlc(x, s_) for x in pw_ok]
        hl = [x for x in hl if x is not None]
        if hl:
            fam["F6"] = [dict(L=max(x["h"] for x in hl), kind="prior_week_high", known=t630),
                         dict(L=min(x["l"] for x in hl), kind="prior_week_low", known=t630)]
    if not fam["F6"]:
        MISSING["F6"].append(str(d_))
    sw = swings15(F, d_, im)
    SWALL[d_] = sw
    fam["F7"] = [dict(L=x["px"], kind=x["kind"], known=x["known"]) for x in sw if x["kept"]]
    return {f: merge(v) for f, v in fam.items()}


FAMALL = {d_: family_levels(d_) for d_ in USABLE}
LEV = {d_: {f: [x for x in v if abs(x["L"] - OPEN[d_]) <= WIN_IM * IM[d_]] for f, v in FAMALL[d_].items()}
       for d_ in USABLE}


def placebo(d_, f):
    o, im = OPEN[d_], IM[d_]
    allf = np.array([x["L"] for x in FAMALL[d_][f]]) if FAMALL[d_][f] else np.array([np.inf])
    lo, hi = np.ceil((o - im) / PLC_BIN) * PLC_BIN, np.floor((o + im) / PLC_BIN) * PLC_BIN
    grid = np.arange(lo, hi + 0.5, PLC_BIN)
    grid = grid[np.min(np.abs(grid[:, None] - allf[None, :]), axis=1) >= PLC_SEP]
    if f.startswith("F5"):
        grid = grid[np.abs(grid - np.round(grid / 25) * 25) > R25_EXCL]
    pool = {}
    for x in LEV[d_][f]:
        s = np.sign(x["L"] - o)
        cand = grid[np.sign(grid - o) == s]
        cand = cand[np.lexsort((cand, np.abs(cand - x["L"])))][:PLC_K]
        for b_ in cand:
            k_ = float(b_)
            if k_ not in pool or x["known"] < pool[k_]["known"]:
                pool[k_] = dict(L=k_, kind="placebo", known=x["known"])
    return list(pool.values())


PLV = {d_: {f: placebo(d_, f) for f in FAMILIES if f != "F4"} for d_ in USABLE}
rng_vw = np.random.default_rng(SEED_DRAW)
VWOFF = {}
for d_ in USABLE:
    sg = rng_vw.choice([-1.0, 1.0], size=PLC_K)
    mg = rng_vw.uniform(VW_OFF_LO, VW_OFF_HI, size=PLC_K)
    VWOFF[d_] = [float(x) for x in sg * mg]


def vwap(F, d_):
    a, b = rth_slice(F, d_)
    tp = (F["h"][a:b] + F["l"][a:b] + F["c"][a:b]) / 3
    cv, cq = np.cumsum(tp * F["v"][a:b]), np.cumsum(F["v"][a:b])
    out = np.full(len(F["t"]), np.nan)
    out[a + 1:b] = (cv / cq)[:-1]                        # VWAP through the previous second (causal)
    return out


VW = {d_: vwap(FLOW[d_], d_) for d_ in USABLE}


# ------------------------------------------------------------------------------------------------ tests
def tests_for(F, d_, Lv, im, k_from):
    """A17 test with a per-row level array Lv (constant for static levels) and an armed-from row."""
    a, b = rth_slice(F, d_)
    a = max(a, k_from)
    h, l = F["h"], F["l"]
    arm = max(ARM_IM * im, ARM_MIN)
    assert arm > BAND
    with np.errstate(invalid="ignore"):
        below = np.nonzero(h[a:b] <= Lv[a:b] - arm)[0] + a
        above = np.nonzero(l[a:b] >= Lv[a:b] + arm)[0] + a
        hit_b = np.nonzero(h[a:b] >= Lv[a:b] - BAND)[0] + a
        hit_a = np.nonzero(l[a:b] <= Lv[a:b] + BAND)[0] + a
    out, pos = [], a

    def nxt(arr, k):
        i = np.searchsorted(arr, k)
        return int(arr[i]) if i < len(arr) else None

    while True:
        ib, ia = nxt(below, pos), nxt(above, pos)
        if ib is None and ia is None:
            break
        if ia is None or (ib is not None and ib < ia):
            jt, side = nxt(hit_b, ib), 1
        else:
            jt, side = nxt(hit_a, ia), -1
        if jt is None:
            break
        out.append((jt, side))
        pos = jt + 1
    return out[:1]                                  # A17.1: first test only


def run(d_, f, lv, Lv, placebo_key=None, measure=False):
    F = FLOW[d_]
    _, b = rth_slice(F, d_)
    t, h, l = F["t"], F["h"], F["l"]
    k_from = int(np.searchsorted(t, lv["known"]))
    rows = []
    for j, side in tests_for(F, d_, Lv, IM[d_], k_from):
        t0 = int(t[j])
        r = dict(d=d_, fam=f, L=lv["L"], kind=lv["kind"], side=side, t=t0, key=placebo_key,
                 pre10=t0 < ep(pt(d_, 10, 0)),
                 ovl=bool(GEX.get(d_)) and min(abs(Lv[j] - g) for g in GEX[d_]) <= OVL_PTS)
        if measure:
            close13 = ep(pt(d_, 13, 0))
            e_ = ESC_IM * IM[d_]
            esc = np.nonzero((h[j:b] >= Lv[j:b] + e_) | (l[j:b] <= Lv[j:b] - e_))[0]
            esc_min = (t[j + int(esc[0])] - t0) / 60 if len(esc) else np.inf
            r.update(esc=min(esc_min, ESC_CAP, (close13 - t0) / 60), censored=not np.isfinite(esc_min))
        rows.append(r)
    return rows


rr, pp = [], []
for d_ in USABLE:
    F = FLOW[d_]
    n_ = len(F["t"])
    for f in FAMILIES:
        if f == "F4":
            known = int(F["t"][rth_slice(F, d_)[0] + 1])
            rr += run(d_, f, dict(L=np.nan, kind="vwap", known=known), VW[d_])
            for off in VWOFF[d_]:
                pp += run(d_, f, dict(L=off, kind="placebo", known=known), VW[d_] + off, (str(d_), f, off), True)
            continue
        for lv in LEV[d_][f]:
            rr += run(d_, f, lv, np.full(n_, lv["L"]))
        for lv in PLV[d_][f]:
            pp += run(d_, f, lv, np.full(n_, lv["L"]), (str(d_), f, lv["L"]), True)
R = pd.DataFrame(rr)
P = pd.DataFrame(pp)
assert "esc" not in R.columns                       # Step 0: no real-level T0 measure


# --------------------------------------------------------------------------------------- Q1 counts
def dist(v):
    v = np.asarray(v, float)
    return dict(n=int(len(v)), median=float(np.median(v)), p25=float(np.percentile(v, 25)),
                p75=float(np.percentile(v, 75)), mean=round(float(v.mean()), 2), zero_share=round(float(np.mean(v == 0)), 3))


Q1 = {}
for f in FAMILIES:
    if f == "F4":
        nlev = [1] * nS
        nplv = [PLC_K] * nS
        ovl_l = None
    else:
        nlev = [len(LEV[d_][f]) for d_ in USABLE]
        nplv = [len(PLV[d_][f]) for d_ in USABLE]
        L_ = [(d_, x["L"]) for d_ in USABLE for x in LEV[d_][f]]
        ovl_l = round(float(np.mean([bool(GEX.get(d_)) and min(abs(x - g) for g in GEX[d_]) <= OVL_PTS
                                     for d_, x in L_])), 3) if L_ else None
    Rf, Pf = R[R["fam"] == f], P[P["fam"] == f]
    tr = [int((Rf["d"] == d_).sum()) for d_ in USABLE]
    tp = [int((Pf["d"] == d_).sum()) for d_ in USABLE]
    Q1[f] = dict(
        levels_per_session=dist(nlev), placebo_levels_per_session=dist(nplv),
        tests_real=int(len(Rf)), tests_placebo=int(len(Pf)),
        tests_real_per_session=dist(tr), tests_placebo_per_session=dist(tp),
        sessions_with_test=dict(real=int(Rf["d"].nunique()), placebo=int(Pf["d"].nunique())),
        by_side=dict(real={"below": int((Rf["side"] == 1).sum()), "above": int((Rf["side"] == -1).sum())},
                     placebo={"below": int((Pf["side"] == 1).sum()), "above": int((Pf["side"] == -1).sum())}),
        sessions_by_side=dict(real={"below": int(Rf[Rf["side"] == 1]["d"].nunique()),
                                    "above": int(Rf[Rf["side"] == -1]["d"].nunique())},
                              placebo={"below": int(Pf[Pf["side"] == 1]["d"].nunique()),
                                       "above": int(Pf[Pf["side"] == -1]["d"].nunique())}),
        tests_pre_post_10pt=dict(real=[int(Rf["pre10"].sum()), int((~Rf["pre10"]).sum())] if len(Rf) else [0, 0],
                                 placebo=[int(Pf["pre10"].sum()), int((~Pf["pre10"]).sum())] if len(Pf) else [0, 0]),
        gex_overlap_share_levels=ovl_l,
        gex_overlap_share_tests=dict(real=round(float(Rf["ovl"].mean()), 3) if len(Rf) else None,
                                     placebo=round(float(Pf["ovl"].mean()), 3) if len(Pf) else None),
        placebo_censored=int(Pf["censored"].sum()) if len(Pf) else 0,
        missing_sessions=len(MISSING.get(f, [])),
    )
    if f not in ("F4",) and not f.startswith("F5"):
        pl = [x["L"] for d_ in USABLE for x in PLV[d_][f]]
        Q1[f]["placebo_on_25_multiple_share"] = round(float(np.mean([x % 25 == 0 for x in pl])), 3) if pl else None
Q1["_kinds_real_tests"] = R.groupby(["fam", "kind"]).size().reset_index().values.tolist()
Q1["_swings"] = dict(before_hold=dist([len(SWALL[d_]) for d_ in USABLE]),
                     kept=dist([sum(s_["kept"] for s_ in SWALL[d_]) for d_ in USABLE]))
Q1["_low_im"] = {str(d_): IM[d_] for d_ in USABLE if IM[d_] < IM_FLAG}
Q1["_im_dist"] = dist(list(IM.values()))
Q1["_missing"] = MISSING
Q1["_gex_levels_per_session"] = dist([len(GEX.get(d_, [])) for d_ in USABLE])


# ----------------------------------------------------------------------------------------- Q2 A/A


def aa(E):
    if E.empty:
        return dict(n_tests=0)
    keys = sorted(set(E["key"]))
    if len(keys) < 4:
        return dict(n_tests=int(len(E)), pseudo_levels=len(keys), note="too few pseudo-levels")
    rs_, rb_ = np.random.default_rng(SEED_DRAW), np.random.default_rng(SEED_BOOT)
    sx = np.array([SIDX[d_] for d_ in E["d"]])
    val = E["esc"].to_numpy(float)
    stats, excl = [], 0
    for _ in range(N_SPLIT):
        perm = rs_.permutation(len(keys))
        h1 = {keys[k] for k in perm[: len(keys) // 2]}
        g = np.array([k not in h1 for k in E["key"]]).astype(int)
        c = np.zeros((nS, 2, 2))
        np.add.at(c, (sx, g, 0), val)
        np.add.at(c, (sx, g, 1), 1)
        m = np.where(c[:, :, 1] > 0, c[:, :, 0] / np.maximum(c[:, :, 1], 1), 0.0)     # session means
        pres = (c[:, :, 1] > 0).astype(float)
        stats.append((m[:, 0] * pres[:, 0]).sum() / pres[:, 0].sum() - (m[:, 1] * pres[:, 1]).sum() / pres[:, 1].sum())
        Wb = rb_.multinomial(nS, np.full(nS, 1 / nS), size=N_BOOT_SPLIT).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            bst = (Wb @ (m[:, 0] * pres[:, 0])) / (Wb @ pres[:, 0]) - (Wb @ (m[:, 1] * pres[:, 1])) / (Wb @ pres[:, 1])
        lo, hi = np.nanpercentile(bst, [2.5, 97.5])
        excl += int(lo > 0 or hi < 0)
    stats = np.array(stats)
    share = excl / N_SPLIT
    med = float(np.median(stats))
    return dict(n_tests=int(len(E)), pseudo_levels=len(keys), sessions=int(E["d"].nunique()),
                excl_share=round(share, 3), median=round(med, 3),
                p10=round(float(np.percentile(stats, 10)), 3), p90=round(float(np.percentile(stats, 90)), 3),
                PASS=bool(share <= AA_EXCL_MAX and abs(med) <= AA_MED_TOL))


Q2 = {}
for f in FAMILIES + ["all"]:
    E = P if f == "all" else P[P["fam"] == f]
    Q2[f] = {"pooled": aa(E), "below": aa(E[E["side"] == 1]), "above": aa(E[E["side"] == -1])}
Q2["_PASS_all_cells"] = all(v["PASS"] for f in FAMILIES + ["all"] for v in Q2[f].values() if "PASS" in v)


# ---------------------------------------------------------------------------------- Q3 worked day
d_ = WORKED
F = FLOW[d_]
t810, t730 = ep(pt(d_, 8, 10)), ep(pt(d_, 7, 30))
near = []
for f in FAMILIES:
    if f == "F4":
        continue
    for x in FAMALL[d_][f]:
        if WB_LO <= x["L"] <= WB_HI and x["known"] <= t810:
            ft = run(d_, f, x, np.full(len(F["t"]), x["L"]))
            near.append(dict(family=f, kind=x["kind"], level=x["L"], known_pt=hms(x["known"]),
                             in_L4_window=bool(abs(x["L"] - OPEN[d_]) <= IM[d_]),
                             first_test=[hms(ft[0]["t"]), "below" if ft[0]["side"] == 1 else "above"] if ft else None))
k0, k1 = np.searchsorted(F["t"], t730), np.searchsorted(F["t"], t810)
vw = VW[d_][k0:k1]
low_k = k0 + int(np.argmin(F["l"][k0:k1]))
mins = []
for m_ in range(40):
    ta = t730 + 60 * m_
    a_, b_ = np.searchsorted(F["t"], ta), np.searchsorted(F["t"], ta + 60)
    if b_ > a_:
        mins.append([hms(ta)[:5], float(F["l"][a_:b_].min()), float(F["h"][a_:b_].max()), float(F["c"][b_ - 1]),
                     round(float(VW[d_][b_ - 1]), 2)])
a_, _ = rth_slice(F, d_)
earlier = []
for m_ in range(int((t730 - ep(pt(d_, 6, 30))) / 60)):
    ta = ep(pt(d_, 6, 30)) + 60 * m_
    x0, x1 = np.searchsorted(F["t"], ta), np.searchsorted(F["t"], ta + 60)
    if x1 > x0 and WB_LO <= F["l"][x0:x1].min() <= WB_HI:
        earlier.append([hms(ta)[:5], float(F["l"][x0:x1].min())])
Q3 = dict(
    session=str(d_), symbol=SYM[d_], open=OPEN[d_], im=IM[d_], band=[WB_LO, WB_HI],
    family_levels_in_band=near,
    all_family_levels={f: [[x["kind"], x["L"], hms(x["known"])] for x in FAMALL[d_][f]] for f in FAMALL[d_]},
    vwap_0730_0810=dict(min=round(float(np.nanmin(vw)), 2), max=round(float(np.nanmax(vw)), 2),
                        at_low=round(float(VW[d_][low_k]), 2)),
    low_0730_0810=dict(px=float(F["l"][low_k]), t=hms(F["t"][low_k])),
    minutes_0730_0810=mins,
    swings_all=[[x["kind"], x["px"], hms(x["at"]), hms(x["known"]), x["kept"]] for x in SWALL[d_]],
    earlier_minute_lows_in_band_before_0730=earlier,
    gex_levels=GEX.get(d_, []),
    gex_in_band=[g for g in GEX.get(d_, []) if WB_LO - 5 <= g <= WB_HI + 5],
    gex_in_band_first_test={str(g): [[hms(z["t"]), "below" if z["side"] == 1 else "above"]
                                     for z in run(d_, "GEX", dict(L=g, kind="gex", known=int(F["t"][a_])),
                                                  np.full(len(F["t"]), g))]
                            for g in GEX.get(d_, []) if WB_LO - 5 <= g <= WB_HI + 5},
)

RES = dict(funnel=funnel, sessions=dict(n=nS, first=str(USABLE[0]), last=str(USABLE[-1]),
                                        symbols=sorted({SYM[x] for x in USABLE})),
           Q1=Q1, Q2=Q2, Q3=Q3)
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "step0.json").write_text(json.dumps(RES, indent=1, default=str))

for f_ in funnel:
    print(f_)
print("sessions", RES["sessions"])
print("\nQ1 counts (levels/session median [IQR]; tests real / placebo; sessions with a test real / placebo)")
for f in FAMILIES:
    v = Q1[f]
    lp = v["levels_per_session"]
    print(f"  {f:7s} lv {lp['median']:.0f} [{lp['p25']:.0f}, {lp['p75']:.0f}] zero {lp['zero_share']:.2f} | plc lv "
          f"{v['placebo_levels_per_session']['median']:.0f} | tests {v['tests_real']}/{v['tests_placebo']} "
          f"(per sess {v['tests_real_per_session']['median']:.0f}/{v['tests_placebo_per_session']['median']:.0f}) "
          f"| sess {v['sessions_with_test']['real']}/{v['sessions_with_test']['placebo']} | side real "
          f"{v['by_side']['real']} plc {v['by_side']['placebo']} | sess/side real {v['sessions_by_side']['real']} plc "
          f"{v['sessions_by_side']['placebo']} | pre/post10 {v['tests_pre_post_10pt']} | GEX ovl lv "
          f"{v['gex_overlap_share_levels']} tests {v['gex_overlap_share_tests']} | missing {v['missing_sessions']}"
          + (f" | plc on 25s {v['placebo_on_25_multiple_share']}" if "placebo_on_25_multiple_share" in v else ""))
print("  swings", Q1["_swings"], "low IM", Q1["_low_im"], "IM", Q1["_im_dist"])
print("\nQ2 A/A (placebo only; excl share, median min; PASS)")
for f in FAMILIES + ["all"]:
    print(f"  {f:7s} " + "  ".join(f"{c}: n {v.get('n_tests')} pl {v.get('pseudo_levels')} excl {v.get('excl_share')} "
                                    f"med {v.get('median')} {'PASS' if v.get('PASS') else 'FAIL' if 'PASS' in v else v.get('note', '')}"
                                    for c, v in Q2[f].items()))
print("  all cells pass:", Q2["_PASS_all_cells"])
print("\nQ3 2026-05-01")
print(json.dumps({k: v for k, v in Q3.items() if k != "minutes_0730_0810"}, indent=1, default=str))
for m_ in Q3["minutes_0730_0810"]:
    print("  ", m_)
