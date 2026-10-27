#!/usr/bin/env python3
"""CR-CM shared harness — T0 at the chart's STRONG GEX levels. READ-ONLY.

Spec: specs/CR-CM-strong-level-t0.md. Imported by scripts/cr_cm_step0.py (counts + A/A, placebo measures only) and
the later outcome script. `load(dataset)` builds one dataset ("train" 2026-01-06 → 2026-06-05, or "ra" 2024-04-03 →
2024-12-31); every DB read is asserted inside that dataset's guard (train: >= 2026-01-01 and < 2026-06-05 21:00
UTC; R-a: < 2025-01-01), so R-b (2025) and the holdout are never read.

Copied unchanged from scripts/cr_cb_t0_ra_a17.py (CR-CB A16 + A17): the connection guard, universe funnel (roll
week, 15:45 ET bar on A, quarterly-expiry Friday, Databento RTH flow, early close, canonical IM, orats_oi_gamma rows,
contract check), the session's largest-RTH-volume Databento symbol, the A1.1 level selection (`select_levels`, used
for the W readout), the A16.4 placebo, `tests_for` (first armed test, arming max(0.10·IM, 4), 2-pt band), the escape
and dwell measures, session means and the bootstrap weights (seed 20260920).

New here (spec M1, M2):
  - `chart_levels`: the Ironbeam Corrected chart's displayed lines (orats_oi_gamma trade_date = D, expir_date >= D,
    discounted_level IS NOT NULL, ticker SPX; place_levels with P_target; 1-pt bucket; call = Σ gex_call, put =
    −|Σ gex_put|; band = ironbeam_es_1m_bars low/high of prior 15:00 PT → 06:30 PT ± 150; gross >= 40 $bn; 5-pt
    spacing by gross desc; <= 80), span = nanpercentile(|net| of displayed, 95) (GEX_COLOR_ABS_MAX unset), strong =
    |net| >= span (t clipped to 0 / 1: indigo / yellow).
  - sets S (strong, ±1·IM of the open), X100 (displayed |net| >= 100 $bn, ±1·IM), W (A1.1 set minus levels within
    2 pts of any pre-window strong level); S-first is a flag on S tests.
  - per test: `edge` (L >= running RTH high − 2 or L <= running RTH low + 2, extremes from 06:30 PT through the test
    second inclusive; CR-CF A2 D3) and `first_touch` (no RTH second before the test whose range meets [L − 2, L + 2]).
  - `matched_diff`: Σ_s w_s (R_s − P_s) over s in {EDGE, INTERIOR}, w_s = real test share in s (per resample).
"""
from __future__ import annotations

import os
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
DATASETS = {
    "train": dict(d_from=date(2026, 1, 6), d_to=date(2026, 6, 5),
                  floor=datetime(2026, 1, 1, tzinfo=UTC), cap=datetime(2026, 6, 5, 21, 0, tzinfo=UTC)),
    "ra": dict(d_from=date(2024, 4, 3), d_to=date(2024, 12, 31),
               floor=datetime(2024, 3, 1, tzinfo=UTC), cap=datetime(2025, 1, 1, tzinfo=UTC)),
}
TICK = 0.25
DTE_MAX, MAG_MIN, SPACING, WIN_IM, TOP_N = 30, 40e9, 5.0, 1.0, 6
ARM_IM, ARM_MIN, BAND, IM_FLAG = 0.10, 4.0, 2.0, 25.0
PLC_BIN, PLC_SEP, PLC_K = 5.0, 10.0, 4
O_C = {("put", 1): 9.75, ("put", -1): 8.25, ("call", 1): 6.25, ("call", -1): 5.25}
ESC_IM, ESC_CAP, DWELL_S = 0.25, 120, 3600
SEED_BOOT, N_BOOT, ALPHA = 20260920, 10_000, 0.025
SEED_DRAW, N_SPLIT, N_BOOT_SPLIT, AA_EXCL_MAX, AA_MED_TOL = 20261025, 200, 2_000, 0.10, 3.0
CELLS = [("put", 1), ("put", -1), ("call", 1), ("call", -1)]
# chart (apps/web/modules/Ironbeam/callbacks.py, code defaults; GEX_COLOR_ABS_MAX unset)
CH_PAD, CH_MAX, CH_PCT, CH_BUCKET, X100 = 150.0, 80, 95.0, 1.0, 100e9
SAME_LVL, NEAR_LVL = 2.0, 10.0
OUT_DIR = REPO_ROOT / "scripts" / "logs" / "cr_cm"

CONN = get_backfill_db_conn()
assert_role_or_die(CONN)
CONN.execute("SET default_transaction_read_only = on")
CONN.execute("SET statement_timeout = '900s'")
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


def rth_slice(F, d_):
    return np.searchsorted(F["t"], ep(pt(d_, 6, 30))), np.searchsorted(F["t"], ep(pt(d_, 13, 0)))


# ---------------------------------------------------------------------------------------------- levels
def select_levels(g, ptgt, o, im):
    """CR-CB A1.1 (A17 code, unchanged)."""
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


def chart_levels(g, ptgt, band_lo, band_hi):
    """Ironbeam Corrected chart: displayed lines and the colour span (spec 'Strong levels')."""
    g = place_levels(g, ptgt)
    g = g[g["level_corrected"].notna()].copy()
    g["level"] = bucket_levels(g["level_corrected"].to_numpy(), CH_BUCKET)
    bk = g.groupby("level").agg(call=("gc", "sum"), praw=("gp", "sum")).reset_index()
    bk["put"] = -bk["praw"].abs()
    bk["net"] = bk["call"] + bk["put"]
    bk = bk[(bk["level"] >= band_lo) & (bk["level"] <= band_hi)].copy()
    bk["mag"] = bk["call"].abs() + bk["put"].abs()
    bk = bk[bk["mag"] >= MAG_MIN].sort_values("mag", ascending=False)
    kept = []
    for r in bk.itertuples():
        if all(abs(r.level - k.level) >= SPACING for k in kept):
            kept.append(r)
        if len(kept) >= CH_MAX:
            break
    disp = [dict(L=float(k.level), mag=float(k.mag), net=float(k.net)) for k in kept]
    a = np.abs(np.array([x["net"] for x in disp], dtype=float))
    a = a[np.isfinite(a)]
    if a.size:
        p = float(np.nanpercentile(a, CH_PCT))
        span = p if p > 0 else (float(a.max()) if a.max() > 0 else 1.0)
    else:
        span = 1.0
    for x in disp:
        x["strong"] = bool(abs(x["net"]) >= span)
    return disp, span


def placebo(levels, o, im, avoid=None):
    """A16.4 / A15.6 (A17 code): around the set's own real levels. `avoid` (CR-CM A1.2): extra level prices the bins
    must also sit >= 10 pts from; None = A17 unchanged."""
    real = np.array([x["L"] for x in levels] + [float(a) for a in (avoid or [])])
    pool = {}
    for x in levels:
        s = np.sign(x["L"] - o)
        lo, hi = np.ceil((o - im) / PLC_BIN) * PLC_BIN, np.floor((o + im) / PLC_BIN) * PLC_BIN
        cand = np.arange(lo, hi + 0.5, PLC_BIN)
        cand = cand[(np.sign(cand - o) == s) & (np.min(np.abs(cand[:, None] - real[None, :]), axis=1) >= PLC_SEP)]
        cand = cand[np.lexsort((cand, np.abs(cand - x["L"])))][:PLC_K]
        for b in cand:
            pool.setdefault(float(b), dict(L=float(b), mag=x["mag"], net=x["net"], parent=x["L"]))
    return list(pool.values())


# ------------------------------------------------------------------------------------------------ tests
def tests_for(F, d_, L, im):
    """A17 code, unchanged: the first armed test (either side)."""
    a, b = rth_slice(F, d_)
    h, l = F["h"], F["l"]
    arm = max(ARM_IM * im, ARM_MIN)
    assert arm > BAND
    below = np.nonzero(h[a:b] <= L - arm)[0] + a
    above = np.nonzero(l[a:b] >= L + arm)[0] + a
    hit_b = np.nonzero(h[a:b] >= L - BAND)[0] + a
    hit_a = np.nonzero(l[a:b] <= L + BAND)[0] + a
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
    return out[:1]


class Data:
    pass


def load(name: str) -> Data:
    cfg = DATASETS[name]
    D_FROM, D_TO, FLOOR, CAP = cfg["d_from"], cfg["d_to"], cfg["floor"], cfg["cap"]

    def guard_ts(lo, hi):
        assert lo >= FLOOR and hi <= CAP, (lo, hi)

    def guard_d(lo, hi):
        assert lo >= FLOOR.date() and hi < CAP.date() + timedelta(days=1) and hi <= D_TO, (lo, hi)

    SESSIONS, d = [], D_FROM
    while d <= D_TO:
        if is_trading_day(d):
            SESSIONS.append(d)
        d += timedelta(days=1)
    QEXP = {third_friday(y, m) for y in {D_FROM.year, D_TO.year} for m in (3, 6, 9, 12)}
    CAP_TS = min(pt(D_TO, 13, 0), CAP)

    lo_ts = pt(prev_trading_day(D_FROM), 0, 0)
    guard_ts(lo_ts, CAP_TS)
    sym = q("""
        SELECT (ts_utc AT TIME ZONE 'America/Los_Angeles')::date AS d, symbol, sum(volume) AS vol
        FROM es_flow_1s_databento
        WHERE ts_utc >= %s AND ts_utc < %s
          AND (ts_utc AT TIME ZONE 'America/Los_Angeles')::time >= '06:30'
          AND (ts_utc AT TIME ZONE 'America/Los_Angeles')::time < '13:00'
        GROUP BY 1, 2
    """, (lo_ts, CAP_TS)).sort_values("vol")
    SYM = dict(zip(sym["d"], sym["symbol"]))

    def load_flow(d_, symbol):
        guard_ts(pt(d_, 6, 15), min(pt(d_, 13, 0), CAP_TS))
        f = q("""SELECT ts_utc, open, high, low, close FROM es_flow_1s_databento
                 WHERE symbol = %s AND ts_utc >= %s AND ts_utc < %s AND ts_utc < %s ORDER BY ts_utc""",
              (symbol, pt(d_, 6, 15), pt(d_, 13, 0), CAP_TS))
        return dict(t=np.array([ep(x) for x in f["ts_utc"]], dtype=np.int64), o=f["open"].to_numpy(float),
                    h=f["high"].to_numpy(float), l=f["low"].to_numpy(float), c=f["close"].to_numpy(float))

    FLOW = {d_: load_flow(d_, SYM[d_]) for d_ in SESSIONS if d_ in SYM}

    oig_lo = max(D_FROM - timedelta(days=7), FLOOR.date())
    guard_d(oig_lo, D_TO)
    oig = q("""
        SELECT trade_date, expir_date, strike::float AS strike, stock_price::float AS stock_price, gex_call, gex_put
        FROM orats_oi_gamma
        WHERE ticker = 'SPX' AND trade_date BETWEEN %s AND %s
          AND expir_date >= trade_date AND expir_date <= trade_date + %s
    """, (oig_lo, D_TO, DTE_MAX + 7))
    oig["d"] = [t if is_trading_day(t) else next_trading_day(t) for t in oig["trade_date"]]
    oig = oig[(oig["d"] >= D_FROM) & (oig["d"] <= D_TO)]
    oig = oig[[d_ <= e <= d_ + timedelta(days=DTE_MAX) for e, d_ in zip(oig["expir_date"], oig["d"])]]
    oig["call"] = oig["gex_call"].astype(float)
    oig["put"] = -oig["gex_put"].abs().astype(float)

    guard_d(D_FROM, D_TO)
    feat = q("""
        SELECT trade_date, (feature_vector->>'implied_move_1d')::float AS im FROM bt_daily_features
        WHERE feature_version = %s AND ticker = 'SPX' AND active AND trade_date BETWEEN %s AND %s
    """, (CANONICAL_FEATURE_VERSION, D_FROM, D_TO)).set_index("trade_date")["im"]
    feat = feat[~feat.index.duplicated()]

    starts = {es_snapshot_bar_start_utc(prev_trading_day(d_)): d_ for d_ in SESSIONS}
    guard_ts(min(starts).replace(tzinfo=UTC), max(starts).replace(tzinfo=UTC))
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
              (SYM[a_], st.replace(tzinfo=UTC), st.replace(tzinfo=UTC) + timedelta(minutes=1)))
        if len(r):
            ACHK[d_] = (SYM[a_], float(r["close"][0]))

    OPEN, LAST = {}, {}
    for d_, F in FLOW.items():
        a, b = rth_slice(F, d_)
        if b > a:
            OPEN[d_], LAST[d_] = float(F["o"][a]), int(F["t"][b - 1])

    U, funnel = list(SESSIONS), []
    for nm, rule in [
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
        funnel.append([nm, len(rm), len(U), [str(x) for x in rm]])
    USABLE = U
    IM = {d_: float(feat[d_]) for d_ in USABLE}

    # chart rows: every live expiry, aggregated per (D, strike, forward) — exact for the per-level sums
    guard_d(D_FROM, D_TO)
    ch = q("""
        SELECT trade_date AS d, strike::float AS strike, stock_price::float AS stock_price,
               sum(gex_call)::float AS gc, sum(gex_put)::float AS gp
        FROM orats_oi_gamma
        WHERE ticker = 'SPX' AND trade_date = ANY(%s) AND discounted_level IS NOT NULL AND expir_date >= trade_date
        GROUP BY 1, 2, 3
    """, (USABLE,))
    ch["gc"] = ch["gc"].fillna(0.0)
    ch["gp"] = ch["gp"].fillna(0.0)

    def band(d_, end_hh, end_mm):
        lo_, hi_ = pt(d_ - timedelta(days=1), 15, 0), pt(d_, end_hh, end_mm)
        guard_ts(lo_, hi_)
        r = q("""SELECT min(low)::float lo, max(high)::float hi FROM ironbeam_es_1m_bars
                 WHERE datetime >= %s AND datetime < %s""",
              (lo_.replace(tzinfo=None), hi_.replace(tzinfo=None)))
        return float(r["lo"][0]) - CH_PAD, float(r["hi"][0]) + CH_PAD

    CHART, CHART_FULL, SPAN, SPAN_FULL = {}, {}, {}, {}
    chg = dict(tuple(ch.groupby("d")))
    for d_ in USABLE:
        g = chg.get(d_, ch.iloc[:0])
        CHART[d_], SPAN[d_] = chart_levels(g, PTGT[d_], *band(d_, 6, 30))
        CHART_FULL[d_], SPAN_FULL[d_] = chart_levels(g, PTGT[d_], *band(d_, 13, 0))

    oigg = dict(tuple(oig[oig["d"].isin(USABLE)].groupby("d")))
    CB = {d_: select_levels(oigg[d_], PTGT[d_], OPEN[d_], IM[d_]) for d_ in USABLE}

    def win(lv, d_):
        return [x for x in lv if abs(x["L"] - OPEN[d_]) <= WIN_IM * IM[d_]]

    STRONG_ALL = {d_: [x for x in CHART[d_] if x["strong"]] for d_ in USABLE}
    SETS = {
        "S": {d_: win(STRONG_ALL[d_], d_) for d_ in USABLE},
        "X100": {d_: win([x for x in CHART[d_] if abs(x["net"]) >= X100], d_) for d_ in USABLE},
        "W": {d_: [x for x in CB[d_] if all(abs(x["L"] - s["L"]) > SAME_LVL for s in STRONG_ALL[d_])]
              for d_ in USABLE},
        "CB": CB,
    }
    PLV = {k: {d_: placebo(v[d_], OPEN[d_], IM[d_]) for d_ in USABLE} for k, v in SETS.items()}

    D = Data()
    D.__dict__.update(name=name, cfg=cfg, USABLE=USABLE, IM=IM, FLOW=FLOW, OPEN=OPEN, SYM=SYM, PTGT=PTGT,
                      funnel=funnel, CHART=CHART, CHART_FULL=CHART_FULL, SPAN=SPAN, SPAN_FULL=SPAN_FULL,
                      STRONG_ALL=STRONG_ALL, SETS=SETS, PLV=PLV, SIDX={d_: k for k, d_ in enumerate(USABLE)},
                      nS=len(USABLE))
    return D


def tests(D: Data, levels: dict) -> pd.DataFrame:
    """One row per first armed test, with type, side, EDGE and first-touch tags. No outcome."""
    rows = []
    for d_ in D.USABLE:
        F = D.FLOW[d_]
        a, _ = rth_slice(F, d_)
        h, l = F["h"], F["l"]
        for lv in levels[d_]:
            ty = "put" if lv["net"] < 0 else "call"
            for j, side in tests_for(F, d_, lv["L"], D.IM[d_]):
                rh, rl = float(h[a:j + 1].max()), float(l[a:j + 1].min())
                prior = (h[a:j] >= lv["L"] - BAND) & (l[a:j] <= lv["L"] + BAND)
                rows.append(dict(d=d_, L=lv["L"], type=ty, side=side, j=int(j), net=lv["net"],
                                 edge=bool(lv["L"] >= rh - BAND or lv["L"] <= rl + BAND),
                                 first_touch=not bool(prior.any())))
    return pd.DataFrame(rows, columns=["d", "L", "type", "side", "j", "net", "edge", "first_touch"])


def add_measures(D: Data, E: pd.DataFrame) -> pd.DataFrame:
    """A17 measures on given tests: restricted-mean escape (min) and zone dwell (s)."""
    esc_v, dw_v = [], []
    for r in E.itertuples():
        F = D.FLOW[r.d]
        _, b = rth_slice(F, r.d)
        t, h, l, c = F["t"], F["h"], F["l"], F["c"]
        close13 = ep(pt(r.d, 13, 0))
        j, im = r.j, D.IM[r.d]
        t0 = int(t[j])
        esc = np.nonzero((h[j:b] >= r.L + ESC_IM * im) | (l[j:b] <= r.L - ESC_IM * im))[0]
        esc_min = (t[j + int(esc[0])] - t0) / 60 if len(esc) else np.inf
        cap = min(ESC_CAP, (close13 - t0) / 60)
        O = O_C[(r.type, r.side)]
        g = np.arange(t0, min(t0 + DWELL_S, close13))
        kk = np.searchsorted(t, g, side="right") - 1
        px = c[np.clip(kk, 0, None)]
        esc_v.append(min(esc_min, cap))
        dw_v.append(int(np.sum(np.abs(px - r.L) <= O)))
    E = E.copy()
    E["esc"], E["dwell"] = esc_v, dw_v
    return E


# ------------------------------------------------------------------------------------------- statistics
def boot_weights(nS, n=N_BOOT, seed=SEED_BOOT):
    return np.random.default_rng(seed).multinomial(nS, np.full(nS, 1 / nS), size=n).astype(float)


def sess_arrays(D: Data, E: pd.DataFrame, m: str):
    """Per-session sum, count and mean of measure m (A17.3)."""
    s_ = np.zeros((D.nS, 2))
    if len(E):
        ix = [D.SIDX[x] for x in E["d"]]
        np.add.at(s_, (ix, 0), E[m].to_numpy(float))
        np.add.at(s_, (ix, 1), 1)
    pres = (s_[:, 1] > 0).astype(float)
    mean = np.where(pres > 0, s_[:, 0] / np.maximum(s_[:, 1], 1), 0.0)
    return mean, pres, s_[:, 1]


def wmean(Wm, mean, pres):
    with np.errstate(invalid="ignore", divide="ignore"):
        return (Wm @ (mean * pres)) / (Wm @ pres)


def unmatched_diff(D, Wm, Er, Ep, m):
    mr, pr, _ = sess_arrays(D, Er, m)
    mp, pp, _ = sess_arrays(D, Ep, m)
    return wmean(Wm, mr, pr) - wmean(Wm, mp, pp)


def matched_diff(D, Wm, Er, Ep, m):
    """Spec M2: Σ_s w_s (R_s − P_s), s ∈ {EDGE, INTERIOR}; w_s from the real arm's test counts under Wm."""
    _, _, nr_all = sess_arrays(D, Er, m)
    den = Wm @ nr_all
    out = 0.0
    for s in (True, False):
        Rs, Ps = Er[Er["edge"] == s], Ep[Ep["edge"] == s]
        mr, pr, nr = sess_arrays(D, Rs, m)
        mp, pp, _ = sess_arrays(D, Ps, m)
        with np.errstate(invalid="ignore", divide="ignore"):
            w = (Wm @ nr) / den
        term = w * (wmean(Wm, mr, pr) - wmean(Wm, mp, pp))
        out = out + np.where(w > 0, term, 0.0)
    return out


def aa(D: Data, E: pd.DataFrame, matched: bool):
    """CR-CF Step 0 A/A on placebo tests: 200 pseudo-level splits, 2,000 session resamples each."""
    if E.empty:
        return dict(n_tests=0)
    E = E.assign(key=[f"{d_}|{L}" for d_, L in zip(E["d"], E["L"])])
    keys = sorted(set(E["key"]))
    if len(keys) < 4:
        return dict(n_tests=int(len(E)), pseudo_levels=len(keys), note="too few pseudo-levels")
    rs_, rb_ = np.random.default_rng(SEED_DRAW), np.random.default_rng(SEED_BOOT)
    one = np.ones((1, D.nS))
    stats, excl, undefined = [], 0, 0
    fn = matched_diff if matched else unmatched_diff
    for _ in range(N_SPLIT):
        perm = rs_.permutation(len(keys))
        h1 = {keys[k] for k in perm[: len(keys) // 2]}
        in1 = E["key"].isin(h1)
        A_, B_ = E[in1], E[~in1]
        st = float(fn(D, one, A_, B_, "esc")[0])
        Wb = rb_.multinomial(D.nS, np.full(D.nS, 1 / D.nS), size=N_BOOT_SPLIT).astype(float)
        bst = fn(D, Wb, A_, B_, "esc")
        if not np.isfinite(st) or np.isfinite(bst).sum() < N_BOOT_SPLIT // 2:
            undefined += 1
            continue
        stats.append(st)
        lo, hi = np.nanpercentile(bst, [2.5, 97.5])
        excl += int(lo > 0 or hi < 0)
    n_def = len(stats)
    if not n_def:
        return dict(n_tests=int(len(E)), pseudo_levels=len(keys), note="no defined split")
    share, med = excl / n_def, float(np.median(stats))
    return dict(n_tests=int(len(E)), pseudo_levels=len(keys), sessions=int(E["d"].nunique()),
                splits_defined=n_def, splits_undefined=undefined, excl_share=round(share, 3), median=round(med, 3),
                PASS=bool(share <= AA_EXCL_MAX and abs(med) <= AA_MED_TOL))
