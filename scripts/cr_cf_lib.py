"""CR-CF shared construction under Amendment A1 — universe, level families, placebo, tests, measures. READ-ONLY.

Imported by scripts/cr_cf_step0_a1.py (counts + A/A, placebo measures only) and scripts/cr_cf_stage1.py (the
Stage 1 screen). Importing it loads train data (2026-01-06 .. 2026-06-05; every flow / bar read asserted inside
[2026-01-01 00:00 UTC, 2026-06-05 21:00 UTC)) and builds levels and placebo; it computes no measure by itself.

Copied unchanged from scripts/cr_cf_step0.py (Step 0, commit c4421d4): connection / guard, loaders, universe,
GEX A1.1 levels (L11 only), F1–F7 family levels, VWAP, the F4 offsets, merge, L4 window. Changed under A1:
  - A1.2: F1, F2, F3, F6 placebo bins > 3 pts from every 25-multiple (as F5); F4 placebo tests dropped when
    VWAP + offset at the test second is <= 3 pts from a 25-multiple.
  - A1.1: F7 placebo per parent swing: 5-pt bins on the parent's side of the open within ±1·IM, nearest to the
    parent first; excluded if < 10 pts from any kept swing known at or before the parent's known time, or <= 3 pts
    from a 25-multiple; 4 per parent; armed from the parent's known time (earliest parent wins). A placebo test is
    dropped if, at the test second, the bin is < 10 pts from any kept swing known at or before that second.
  - A1.4: tests = first test from below and first from above, independently (each after its own first arming,
    max(0.10·IM, 4), 2-pt band, from the armed-from row); flag `first` = the earlier of the two (L5 readout).
  - Measures (only when asked): esc = min(minutes to a trade >= 0.25·IM from the level, 120, minutes to 13:00);
    dwell = seconds with forward-filled 1-s close within level ± max(3, 0.05·IM) over [t, min(t + 3600, 13:00));
    lvr = log(path after / path before), path = Σ|Δ close| on 10-s samples over [t − 300, t] (clipped at 06:30) and
    [t, t + 300] (clipped at 13:00), as CR-CB v3. For F4 the level is VWAP (or VWAP + offset) at each second.
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
DWELL_S, ZONE_PTS, ZONE_IM = 3600, 3.0, 0.05                   # L6 secondary
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


def off25(x):
    return np.abs(x - np.round(x / 25) * 25) > R25_EXCL


def placebo(d_, f):
    o, im = OPEN[d_], IM[d_]
    lo, hi = np.ceil((o - im) / PLC_BIN) * PLC_BIN, np.floor((o + im) / PLC_BIN) * PLC_BIN
    grid = np.arange(lo, hi + 0.5, PLC_BIN)
    grid = grid[off25(grid)]                                       # A1.2 (all static families)
    pool = {}
    if f == "F7":                                                  # A1.1
        sw = [x for x in SWALL[d_] if x["kept"]]
        for x in LEV[d_][f]:
            s = np.sign(x["L"] - o)
            known = np.array([z["px"] for z in sw if z["known"] <= x["known"]] or [np.inf])
            cand = grid[(np.sign(grid - o) == s) & (np.min(np.abs(grid[:, None] - known[None, :]), axis=1) >= PLC_SEP)]
            cand = cand[np.lexsort((cand, np.abs(cand - x["L"])))][:PLC_K]
            for b_ in cand:
                k_ = float(b_)
                if k_ not in pool or x["known"] < pool[k_]["known"]:
                    pool[k_] = dict(L=k_, kind="placebo", known=x["known"])
        return list(pool.values())
    allf = np.array([x["L"] for x in FAMALL[d_][f]]) if FAMALL[d_][f] else np.array([np.inf])
    grid = grid[np.min(np.abs(grid[:, None] - allf[None, :]), axis=1) >= PLC_SEP]
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
def tests_sides(F, d_, Lv, im, k_from):
    """A1.4: first test from below and first from above, each after its own first arming (A17 rule per side)."""
    a, b = rth_slice(F, d_)
    a = max(a, k_from)
    h, l = F["h"], F["l"]
    arm = max(ARM_IM * im, ARM_MIN)
    assert arm > BAND
    out = []
    with np.errstate(invalid="ignore"):
        for side, armed, hit in ((1, h[a:b] <= Lv[a:b] - arm, h[a:b] >= Lv[a:b] - BAND),
                                 (-1, l[a:b] >= Lv[a:b] + arm, l[a:b] <= Lv[a:b] + BAND)):
            ka = np.nonzero(armed)[0]
            if not len(ka):
                continue
            kh = np.nonzero(hit[ka[0]:])[0]
            if len(kh):
                out.append((a + int(ka[0]) + int(kh[0]), side))
    out.sort()
    return [(j, s, i == 0) for i, (j, s) in enumerate(out)]


DROPPED = {"F4_round25": 0, "F7_swing_near": 0}


def run(d_, f, lv, Lv, placebo_key=None, measure=False):
    F = FLOW[d_]
    _, b = rth_slice(F, d_)
    t, h, l, c = F["t"], F["h"], F["l"], F["c"]
    k_from = int(np.searchsorted(t, lv["known"]))
    close13, t630 = ep(pt(d_, 13, 0)), ep(pt(d_, 6, 30))
    rows = []
    for j, side, first in tests_sides(F, d_, Lv, IM[d_], k_from):
        t0 = int(t[j])
        if placebo_key is not None and f == "F4" and not off25(Lv[j]):
            DROPPED["F4_round25"] += 1
            continue
        if placebo_key is not None and f == "F7":
            near = [z["px"] for z in SWALL[d_] if z["kept"] and z["known"] <= t0]
            if near and min(abs(lv["L"] - z) for z in near) < PLC_SEP:
                DROPPED["F7_swing_near"] += 1
                continue
        r = dict(d=d_, fam=f, L=lv["L"], kind=lv["kind"], side=side, first=first, t=t0, key=placebo_key,
                 pre10=t0 < ep(pt(d_, 10, 0)),
                 ovl=bool(GEX.get(d_)) and min(abs(Lv[j] - g) for g in GEX[d_]) <= OVL_PTS)
        if measure:
            im = IM[d_]
            esc = np.nonzero((h[j:b] >= Lv[j:b] + ESC_IM * im) | (l[j:b] <= Lv[j:b] - ESC_IM * im))[0]
            esc_min = (t[j + int(esc[0])] - t0) / 60 if len(esc) else np.inf
            g = np.arange(t0, min(t0 + DWELL_S, close13))
            kk = np.clip(np.searchsorted(t, g, side="right") - 1, 0, None)
            dwell = int(np.sum(np.abs(c[kk] - Lv[kk]) <= max(ZONE_PTS, ZONE_IM * im)))

            def path(ta, tb):
                gg = np.arange(ta, tb + 1, 10)
                k2 = np.searchsorted(t, gg, side="right") - 1
                k2 = k2[k2 >= 0]
                return float(np.sum(np.abs(np.diff(c[k2])))) if len(k2) > 1 else np.nan
            before, after = path(max(t0 - 300, t630), t0), path(t0, min(t0 + 300, close13))
            lvr = np.log(after / before) if before and after and np.isfinite(before) and np.isfinite(after) else np.nan
            r.update(esc=min(esc_min, ESC_CAP, (close13 - t0) / 60), censored=not np.isfinite(esc_min),
                     dwell=dwell, lvr=lvr)
        rows.append(r)
    return rows


def build(measure_real: bool):
    rr, pp = [], []
    for d_ in USABLE:
        F = FLOW[d_]
        n_ = len(F["t"])
        for f in FAMILIES:
            if f == "F4":
                known = int(F["t"][rth_slice(F, d_)[0] + 1])
                rr += run(d_, f, dict(L=np.nan, kind="vwap", known=known), VW[d_], None, measure_real)
                for off in VWOFF[d_]:
                    pp += run(d_, f, dict(L=off, kind="placebo", known=known), VW[d_] + off, (str(d_), f, off), True)
                continue
            for lv in LEV[d_][f]:
                rr += run(d_, f, lv, np.full(n_, lv["L"]), None, measure_real)
            for lv in PLV[d_][f]:
                pp += run(d_, f, lv, np.full(n_, lv["L"]), (str(d_), f, lv["L"]), True)
    return pd.DataFrame(rr), pd.DataFrame(pp)
