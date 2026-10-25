#!/usr/bin/env python3
"""CR-CE Stage 1 run — first-touch fade, stop-free direction (screen). READ-ONLY: SELECTs only, no API calls.

Spec: specs/CR-CE-first-touch-fade-direction.md — L1–L11, Step 0, Amendment A1. One run. Every statistic is computed
and written to scripts/logs/cr_ce/stage1.json before anything is printed.

Event construction, sessions, levels, placebo, baseline, data guards: identical to scripts/cr_ce_step0.py (code copied
unchanged; its docstring fixes those choices). Train 2026-01-09 .. 2026-06-05 (86 sessions, A1.1); Databento flow
only, no 2025 row and no holdout row read.

Fixed before the run (not stated in the CR / A1):
  - primary set (A1.2): real armed touches with D > 0 (803); placebo the same rule at A1.2 pseudo-levels.
  - fade sign s = +1 approached from above (long), −1 from below (short). Touch price = L.
  - price P(x) = close of the last Databento row with ts <= x; missing if that row is > 15 s old (counted).
  - stop-free move at H: m = s · (P(t + H) − L). Horizons: +30 min, +60 min (primary), 13:00 PT (x = 12:59:59).
    A1.3: at +30 / +60 an event is used only if t + H <= 13:00 PT. Truncated readout: x = min(t + H, 12:59:59).
  - random entries (L6 ii): per real event, 5 draws from default_rng(20261025) (u ~ U[0,1), drawn once per event in
    event order); entry second r = lo + floor(u · (hi − lo + 1)), lo = max(06:30 PT, t − 3600),
    hi = min(t + 3600, 13:00 PT − H) for +30/+60 (so the draw reaches its horizon), min(t + 3600, 12:59:59) for 13:00
    and the truncated horizons (exit x_r = min(r + H, 12:59:59)). Random move = s · (P(x_r) − P(r)), same s as the
    event. Event's random value = mean of its valid draws; an event with none is dropped from that comparison.
  - comparisons, per cell (L7) and pooled: vs placebo = mean(real) − mean(placebo), event-weighted, unpaired;
    vs random = mean over real events of (m − random value), paired; vs zero = mean(real).
  - L10 bootstrap: session-cluster, 10 000 resamples, seed 20260920, one multinomial session-weight matrix reused
    for every statistic; real, placebo and random resampled jointly by session; 95 % percentile CI.
    One-sided bootstrap p = share of resamples <= 0.
  - Stage 1 screen verdict per cell (L9 form, CR-CB A10.3 screen, three-verdict rule):
      SHORTLISTED  if vs placebo > 0 with CI lower > 0 AND vs random > 0 with CI lower > 0 (+60 min, unadjusted);
      REFUTED      if the CI upper bound vs placebo is < 0;
      INCONCLUSIVE otherwise.
    Holm across the four cells (L9, a Stage 2 rule) is reported as a readout: per comparison family, Holm-adjusted
    one-sided p, "passes after Holm" if both adjusted p < 0.025.
  - L8 (a) A_rel terciles (Step 0 cut points 1.521 / 2.929): +60 min mean per tercile, real and placebo; low − high
    (real) and (low − high real) − (low − high placebo), with CI. Pooled and per cell.
  - L8 (b) MAE before +10 pts: from the touch row, per 1-s row, favourable excursion fav = high − L (long) /
    L − low (short), adverse adv = L − low (long) / high − L (short). k = first row with
    fav >= 10 (up to 13:00 PT). MAE = max(0, adv over rows [touch, k)); the k row's own adverse print is not
    counted (order inside a second is unknown); k = touch row -> MAE 0. Reported: share reaching +10, and among
    those MAE p25/50/75/90, share MAE <= 2 / 4 / 6 / 8 / 10 pts, minutes to +10 (median); real and placebo.
    Not-reached events: median MAE to 13:00.
  - L8 (c) share of moves > 0 at +30, +60, 13:00: real, placebo, random (mean over events of the share of its valid
    draws > 0); real − placebo share with CI.
  - L8 (d) levels with mag >= 100 $bn (placebo by parent mag): the primary block.
  - L8 (e) A9.5′ re-arm: for each primary real (placebo) event, the first row after the touch with adv >= 2.0 pts
    (8 ticks, the CR-BW L6 stop distance); then the first later row whose range contains L = attempt 2, same s,
    entry L. Deduplicated on (session, level, attempt-2 second). No D condition at attempt 2. The primary block
    (+60 min, random entries drawn around the attempt-2 second) on these events.
  - readouts: first touch of the level in the session only (A1.2); truncated-at-13:00 horizons (A1.3).

Usage: /Users/ryan/code/volatility-monorepo/apps/web/.venv/bin/python -u scripts/cr_ce_stage1.py | tee scripts/logs/cr_ce/stage1.log
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

signal.alarm(3600)

PT, UTC = ZoneInfo("America/Los_Angeles"), timezone.utc
D_FROM, D_TO, TRAIN_FROM = date(2026, 1, 2), date(2026, 6, 5), date(2026, 1, 6)
HOLDOUT_TS = datetime(2026, 6, 5, 21, 0, tzinfo=UTC)
TICK = 0.25
DTE_MAX, MAG_MIN, SPACING, WIN_IM, TOP_N, STRONG = 30, 40e9, 5.0, 1.0, 6, 100e9
ARM_IM, WIN_S, GAP_MAX = 0.10, 300, 15
BASE_N, BASE_MIN, BASE_W = 20, 5, 300
PLC_BIN, PLC_SEP, PLC_K = 5.0, 10.0, 2
HORIZONS = {"30m": 1800, "60m": 3600}
SEED_SPLIT, SEED_BOOT, N_SPLIT, N_BOOT_SPLIT = 20261025, 20260920, 200, 2000
ROLL_MISMATCH_EXTRA = {date(2026, 6, 15)}
MAIN_CHECKOUT = Path("/Users/ryan/code/volatility-monorepo")
CRBW_EVENTS = MAIN_CHECKOUT / "scripts" / "logs" / "cr_bw" / "events_real.pkl"
FIXTURE = REPO_ROOT / "research" / "gallery" / "cc-fixture.json"
OUT_DIR = REPO_ROOT / "scripts" / "logs" / "cr_ce"

CONN = get_backfill_db_conn()
assert_role_or_die(CONN)
CONN.execute("SET default_transaction_read_only = on")
CONN.execute("SET statement_timeout = '600s'")
assert CONN.execute("SELECT current_setting('default_transaction_read_only')").fetchone()[0] == "on"


def q(sql, params=None) -> pd.DataFrame:
    cur = CONN.execute(sql, params)
    return pd.DataFrame(cur.fetchall(), columns=[c.name for c in cur.description])


def third_friday(y: int, m: int) -> date:
    d = date(y, m, 15)
    return d + timedelta(days=(4 - d.weekday()) % 7)


def roll_week(d: date) -> bool:
    if d.month not in (3, 6, 9, 12):
        return False
    tf = third_friday(d.year, d.month)
    return tf - timedelta(days=11) <= d <= tf


def trading_days(a: date, b: date) -> list[date]:
    out, d = [], a
    while d <= b:
        if is_trading_day(d):
            out.append(d)
        d += timedelta(days=1)
    return out


def pt(d: date, hh: int, mm: int, ss: int = 0) -> datetime:
    return datetime.combine(d, time(hh, mm, ss), PT).astimezone(UTC)


def ep(x: datetime) -> int:
    return int(x.timestamp())


def pts(x) -> list:
    return [float(v) for v in np.percentile(x, [5, 25, 50, 75, 95])] if len(x) else []


SESSIONS = trading_days(D_FROM, D_TO)
QEXP = {third_friday(y, m) for y in (2025, 2026) for m in (3, 6, 9, 12)}
INFO: dict = {}
LO_TS = pt(D_FROM, 5, 30)
assert LO_TS >= datetime(2026, 1, 1, tzinfo=UTC)

# ---------------------------------------------------------------------------------------------- loads
sym = q("""
    SELECT trade_date, symbol, raw_symbol, count(*) AS n, sum(volume) AS vol
    FROM es_flow_1s_databento
    WHERE ts_utc >= %s AND ts_utc < %s
      AND (ts_utc AT TIME ZONE 'America/Los_Angeles')::time >= '06:30'
      AND (ts_utc AT TIME ZONE 'America/Los_Angeles')::time < '13:00'
    GROUP BY 1, 2, 3 ORDER BY 1, 5 DESC
""", (LO_TS, HOLDOUT_TS))
multi = sorted({d for d, g in sym.groupby("trade_date") if len(g) > 1})
INFO["symbols"] = dict(sessions=int(sym["trade_date"].nunique()),
                       two_symbol_sessions=[str(d) for d in multi],
                       by_symbol={s: [str(g["trade_date"].min()), str(g["trade_date"].max()), int(len(g))]
                                  for s, g in sym.groupby("symbol")})
SYM = {d: g.iloc[0]["symbol"] for d, g in sym.groupby("trade_date")}  # most-volume symbol; asserted unique below

FLOW: dict[date, dict] = {}
for d in SESSIONS:
    if d not in SYM:
        continue
    f = q("""
        SELECT ts_utc, buy_vol, sell_vol, open, high, low, close
        FROM es_flow_1s_databento
        WHERE symbol = %s AND ts_utc >= %s AND ts_utc < %s AND ts_utc >= %s AND ts_utc < %s
        ORDER BY ts_utc
    """, (SYM[d], pt(d, 5, 30), pt(d, 13, 0), LO_TS, HOLDOUT_TS))
    FLOW[d] = dict(t=np.array([ep(x) for x in f["ts_utc"]], dtype=np.int64),
                   buy=f["buy_vol"].to_numpy(np.int64), sell=f["sell_vol"].to_numpy(np.int64),
                   o=f["open"].to_numpy(float), h=f["high"].to_numpy(float), l=f["low"].to_numpy(float),
                   c=f["close"].to_numpy(float))

oig = q("""
    SELECT trade_date, expir_date, strike::float AS strike, stock_price::float AS stock_price, gex_call, gex_put
    FROM orats_oi_gamma
    WHERE ticker = 'SPX' AND trade_date BETWEEN %s AND %s
      AND expir_date >= trade_date AND expir_date <= trade_date + %s
""", (D_FROM - timedelta(days=7), D_TO, DTE_MAX + 7))
oig["d"] = [t if is_trading_day(t) else next_trading_day(t) for t in oig["trade_date"]]
oig = oig[(oig["d"] >= D_FROM) & (oig["d"] <= D_TO)]
oig = oig[[d <= e <= d + timedelta(days=DTE_MAX) for e, d in zip(oig["expir_date"], oig["d"])]]
oig["call"] = oig["gex_call"].astype(float)
oig["put"] = -oig["gex_put"].abs().astype(float)

feat = q("""
    SELECT trade_date, (feature_vector->>'implied_move_1d')::float AS im
    FROM bt_daily_features
    WHERE feature_version = %s AND ticker = 'SPX' AND active AND trade_date BETWEEN %s AND %s
""", (CANONICAL_FEATURE_VERSION, D_FROM, D_TO)).set_index("trade_date")["im"]
feat = feat[~feat.index.duplicated()]

es_starts = {es_snapshot_bar_start_utc(prev_trading_day(d)): d for d in SESSIONS if prev_trading_day(d) >= D_FROM}
es = q("SELECT datetime, close::float AS close FROM ironbeam_es_1m_bars WHERE datetime = ANY(%s)",
       (list(es_starts.keys()),))
PTGT = {es_starts[pd.Timestamp(t).to_pydatetime()]: c for t, c in zip(es["datetime"], es["close"])
        if c is not None and np.isfinite(c) and c > 0}

# Ironbeam RTH open per session, for the open comparison only (CR-BW A2.1 definition)
ib_open = q("""
    SELECT DISTINCT ON (trade_date) trade_date, open::float AS o
    FROM ironbeam_es_flow_1s
    WHERE ts_utc >= %s AND ts_utc < %s
      AND (ts_utc AT TIME ZONE 'America/Los_Angeles')::time >= '06:30'
      AND (ts_utc AT TIME ZONE 'America/Los_Angeles')::time < '13:00'
    ORDER BY trade_date, ts_utc
""", (LO_TS, HOLDOUT_TS)).set_index("trade_date")["o"]


def rth_slice(d):
    f = FLOW[d]
    return np.searchsorted(f["t"], ep(pt(d, 6, 30))), np.searchsorted(f["t"], ep(pt(d, 13, 0)))


def max_gap(d, t0, t1):
    """Largest RTH flow gap (s) within [t0, t1], clipped to [06:30, 13:00] PT."""
    f = FLOW[d]["t"]
    t0 = max(t0, ep(pt(d, 6, 30)))
    t1 = min(t1, ep(pt(d, 13, 0)))
    if t1 <= t0:
        return 0
    a, b = np.searchsorted(f, t0), np.searchsorted(f, t1, side="right")
    return int(np.diff(np.concatenate([[t0], f[a:b], [t1]])).max())


OPEN = {}
for d in FLOW:
    a, b = rth_slice(d)
    if b > a:
        OPEN[d] = float(FLOW[d]["o"][a])

ex_a = {d for d in SESSIONS if roll_week(d)}
ex_b = {d for d in SESSIONS if d not in PTGT and d != D_FROM}
ex_c = {d for d in SESSIONS if d in QEXP or d in ROLL_MISMATCH_EXTRA}
L14 = [d for d in SESSIONS if d not in ex_a | ex_b | ex_c]
assert not (set(multi) & set(L14)), f"two symbols on a non-roll-week session: {set(multi) & set(L14)}"


def base_windows(d):
    f = FLOW[d]
    t0 = ep(pt(d, 6, 30))
    out = []
    for k in range(78):
        ws, we = t0 + BASE_W * k, t0 + BASE_W * (k + 1)
        a, b = np.searchsorted(f["t"], ws), np.searchsorted(f["t"], we)
        if a == 0 or b <= a or max_gap(d, ws, we) > GAP_MAX:
            continue
        net = int(f["buy"][a:b].sum() - f["sell"][a:b].sum())
        if net:
            out.append(np.sign(net) * (f["c"][b - 1] - f["c"][a - 1]) / (abs(net) / 1000.0))
    return np.array(out)


BASE_ELIG = [d for d in L14 if d in FLOW]
AW = {d: base_windows(d) for d in BASE_ELIG}
BASE_ELIG = [d for d in BASE_ELIG if len(AW[d])]
B, BN = {}, {}
for d in SESSIONS:
    prior = [x for x in BASE_ELIG if x < d][-BASE_N:]
    if len(prior) >= BASE_MIN:
        B[d] = float(np.median(np.concatenate([AW[x] for x in prior])))
        BN[d] = len(prior)

U = [d for d in SESSIONS if d >= TRAIN_FROM]
funnel = [["NYSE sessions 2026-01-06 .. 2026-06-05", 0, len(U)]]
for name, rule in [
    ("L14a roll week", lambda d: d in ex_a),
    ("L14b no ES 15:45 ET bar on A", lambda d: d in ex_b),
    ("L14c roll mismatch", lambda d: d in ex_c),
    ("no Databento flow / no RTH open", lambda d: d not in OPEN),
    ("no canonical bt_daily_features row", lambda d: d not in feat.index),
    ("IM null or <= 0", lambda d: d in feat.index and not (pd.notna(feat[d]) and feat[d] > 0)),
    ("no orats_oi_gamma rows", lambda d: d not in set(oig["d"])),
    ("< 5 prior 2026 Databento baseline sessions", lambda d: d not in B),
]:
    rm = [d for d in U if rule(d)]
    U = [d for d in U if not rule(d)]
    funnel.append([name, len(rm), len(U), [str(x) for x in rm]])
USABLE = U
IM = {d: float(feat[d]) for d in USABLE}
gaps = {d: max_gap(d, ep(pt(d, 6, 30)), ep(pt(d, 13, 0))) for d in USABLE}
open_diff = [OPEN[d] - float(ib_open[d]) for d in USABLE if d in ib_open.index]
INFO["sessions"] = dict(
    funnel=funnel, n=len(USABLE), first=str(USABLE[0]), last=str(USABLE[-1]),
    rth_max_gap_s=dict(pcts=pts(list(gaps.values())), max=max(gaps.values()),
                       over_15s=sorted([[str(d), g] for d, g in gaps.items() if g > GAP_MAX])),
    open_minus_ironbeam=dict(n=len(open_diff), equal=int(sum(abs(x) < 1e-9 for x in open_diff)),
                             abs_pcts=pts(np.abs(open_diff)), max_abs=float(np.max(np.abs(open_diff)))),
    baseline=dict(pcts=pts(list(B[d] for d in USABLE)), min=min(B[d] for d in USABLE), max=max(B[d] for d in USABLE),
                  sessions_lt_20_prior=sum(BN[d] < BASE_N for d in USABLE),
                  min_prior=min(BN[d] for d in USABLE)))

# --------------------------------------------------------------------------------------------- levels
LEVELS = {}
for d, g in oig[oig["d"].isin(USABLE)].groupby("d"):
    g = place_levels(g, PTGT[d])
    g = g[g["level_corrected"].notna()]
    g["level"] = bucket_levels(g["level_corrected"].to_numpy(), 1.0)
    bk = g.groupby("level").agg(call=("call", "sum"), put=("put", "sum")).reset_index()
    bk["mag"] = bk["call"].abs() + bk["put"].abs()
    bk["net"] = bk["call"] + bk["put"]
    bk = bk[bk["mag"] >= MAG_MIN].sort_values("mag", ascending=False)
    kept = []
    for r in bk.itertuples():
        if all(abs(r.level - k.level) >= SPACING for k in kept):
            kept.append(r)
    win = [k for k in kept if abs(k.level - OPEN[d]) <= WIN_IM * IM[d]]
    LEVELS[d] = [dict(L=float(k.level), mag=float(k.mag), net=float(k.net)) for k in win[:TOP_N]]


def placebo(d):
    o, im = OPEN[d], IM[d]
    real = np.array([x["L"] for x in LEVELS[d]])
    pool = {}
    for x in LEVELS[d]:
        s = np.sign(x["L"] - o)
        lo, hi = np.ceil((o - im) / PLC_BIN) * PLC_BIN, np.floor((o + im) / PLC_BIN) * PLC_BIN
        cand = np.arange(lo, hi + 0.5, PLC_BIN)
        cand = cand[(np.sign(cand - o) == s) & (np.min(np.abs(cand[:, None] - real[None, :]), axis=1) >= PLC_SEP)]
        cand = cand[np.lexsort((cand, np.abs(cand - x["L"])))][:PLC_K]
        for b in cand:
            pool.setdefault(float(b), dict(L=float(b), mag=x["mag"], net=x["net"]))
    return list(pool.values())


PLV = {d: placebo(d) for d in USABLE}
INFO["levels"] = dict(real=sum(len(v) for v in LEVELS.values()), placebo=sum(len(v) for v in PLV.values()),
                      real_ge100=sum(x["mag"] >= STRONG for v in LEVELS.values() for x in v),
                      real_neg_pos=[sum(x["net"] < 0 for v in LEVELS.values() for x in v),
                                    sum(x["net"] > 0 for v in LEVELS.values() for x in v)],
                      per_session=pts([len(v) for v in LEVELS.values()]))


# --------------------------------------------------------------------------------------------- events
def events_for(d, levels, tag):
    f = FLOW[d]
    a, b = rth_slice(d)
    t, h, l, c = f["t"], f["h"], f["l"], f["c"]
    arm = ARM_IM * IM[d]
    out = []
    for lv in levels:
        L = lv["L"]
        hh, ll = h[a:b], l[a:b]
        far = np.nonzero((ll >= L + arm) | (hh <= L - arm))[0]
        cont = np.nonzero((ll <= L) & (L <= hh))[0]
        start, n_touch = 0, 0
        while True:
            fi = np.searchsorted(far, start)
            if fi >= len(far):
                break
            ci = np.searchsorted(cont, far[fi] + 1)
            if ci >= len(cont):
                break
            i = int(cont[ci]) + a
            side = 1 if l[i - 1] > L else (-1 if h[i - 1] < L else 0)
            ti = int(t[i])
            ev = dict(d=d, tag=tag, L=L, mag=lv["mag"], net=lv["net"], i=i, t=ti, side=side, k=n_touch,
                      pre_gap=max_gap(d, ti - WIN_S, ti))
            w0 = np.searchsorted(t, ti - WIN_S)
            buy, sell = int(f["buy"][w0:i].sum()), int(f["sell"][w0:i].sum())
            D = (sell - buy) if side == 1 else (buy - sell)
            p_start = c[w0 - 1] if w0 > 0 else np.nan
            dP = (p_start - c[i - 1]) if side == 1 else (c[i - 1] - p_start)
            ev["D"] = D
            ev["A_rel"] = dP / (D / 1000.0) / B[d] if D > 0 else np.nan
            out.append(ev)
            n_touch += 1
            start = cont[ci] + 1
    return out


def build(levels, tag):
    E = pd.DataFrame(sum((events_for(d, levels[d], tag) for d in USABLE), []))
    n0 = len(E)
    n_side0 = int((E["side"] == 0).sum())
    n_gap = int(((E["pre_gap"] > GAP_MAX) & (E["side"] != 0)).sum())
    E = E[(E["pre_gap"] <= GAP_MAX) & (E["side"] != 0)].copy()
    E["first"] = E["k"] == 0
    E["cell"] = np.where(E["net"] < 0, "put", "call") + np.where(E["side"] == 1, "_above", "_below")
    E["sidx"] = [USABLE.index(d) for d in E["d"]]
    return E, dict(touches=n0, side0=n_side0, pre_gap_drop=n_gap, kept=len(E),
                   eligible=int((E["D"] > 0).sum()), d_le0=int((E["D"] <= 0).sum()))


R, INFO["events_real"] = build(LEVELS, "real")
P, INFO["events_placebo"] = build(PLV, "placebo")
CELLS = ["put_below", "put_above", "call_below", "call_above"]
ER, EP = R[R["D"] > 0], P[P["D"] > 0]
CUT = np.percentile(ER["A_rel"], [100 / 3, 200 / 3])
for E in (R, P):
    x = E["A_rel"].to_numpy()
    E["terc"] = np.where(np.isnan(x), -1, np.where(x <= CUT[0], 0, np.where(x <= CUT[1], 1, 2)))
ER, EP = R[R["D"] > 0], P[P["D"] > 0]

# ============================================================================================ Stage 1
assert len(USABLE) == 86 and int((R["D"] > 0).sum()) == 803, "universe differs from Step 0 / A1"
assert np.allclose(CUT, [1.5210904958523719, 2.9293496108232437]), "tercile cut points differ from Step 0"
SEED_RAND, N_RAND, N_BOOT = 20261025, 5, 10_000
STOP_PTS, MAE_TARGET = 2.0, 10.0
HKEYS = {"30m": (1800, False), "60m": (3600, False), "1300": (None, False),
         "30m_trunc": (1800, True), "60m_trunc": (3600, True)}
ER, EP = R[R["D"] > 0].copy(), P[P["D"] > 0].copy()
INFO["missing_price"] = {}


def end_of(d):
    return ep(pt(d, 13, 0))


def price_at(d, x):
    f = FLOW[d]["t"]
    j = np.searchsorted(f, x, side="right") - 1
    if j < 0 or x - f[j] > GAP_MAX:
        return np.nan
    return float(FLOW[d]["c"][j])


def exit_ts(d, t, hk):
    H, trunc = HKEYS[hk]
    end = end_of(d)
    if H is None:
        return end - 1
    if trunc:
        return min(t + H, end - 1)
    return t + H if t + H <= end else None


def add_moves(E, name):
    for hk in HKEYS:
        mv, miss = [], 0
        for x in E.itertuples():
            tx = exit_ts(x.d, x.t, hk)
            if tx is None:
                mv.append(np.nan)
                continue
            p = price_at(x.d, tx)
            miss += int(not np.isfinite(p))
            mv.append(x.side * (p - x.L) if np.isfinite(p) else np.nan)
        E[f"m_{hk}"] = mv
        INFO["missing_price"][f"{name}_{hk}"] = miss


def add_random(E):
    U = np.random.default_rng(SEED_RAND).random((len(E), N_RAND))
    for hk in HKEYS:
        H, trunc = HKEYS[hk]
        rv, rp = [], []
        for k, x in enumerate(E.itertuples()):
            end = end_of(x.d)
            lo = max(ep(pt(x.d, 6, 30)), x.t - 3600)
            hi = min(x.t + 3600, end - 1) if (H is None or trunc) else min(x.t + 3600, end - H)
            vals = []
            if hi >= lo:
                for u in U[k]:
                    r = lo + int(np.floor(u * (hi - lo + 1)))
                    xr = end - 1 if H is None else min(r + H, end - 1)
                    p0, p1 = price_at(x.d, r), price_at(x.d, xr)
                    if np.isfinite(p0) and np.isfinite(p1):
                        vals.append(x.side * (p1 - p0))
            rv.append(np.mean(vals) if vals else np.nan)
            rp.append(np.mean(np.array(vals) > 0) if vals else np.nan)
        E[f"r_{hk}"] = rv
        E[f"rpos_{hk}"] = rp


def add_mae(E):
    reach, mae, tmin = [], [], []
    for x in E.itertuples():
        f = FLOW[x.d]
        _, b = rth_slice(x.d)
        h, l = f["h"][x.i:b], f["l"][x.i:b]
        fav = (h - x.L) if x.side == 1 else (x.L - l)
        adv = (x.L - l) if x.side == 1 else (h - x.L)
        k = np.nonzero(fav >= MAE_TARGET)[0]
        if len(k):
            k = int(k[0])
            reach.append(True)
            mae.append(float(max(0.0, adv[:k].max())) if k > 0 else 0.0)
            tmin.append((int(f["t"][x.i + k]) - x.t) / 60.0)
        else:
            reach.append(False)
            mae.append(float(max(0.0, adv.max())) if len(adv) else np.nan)
            tmin.append(np.nan)
    E["reach10"], E["mae"], E["t10_min"] = reach, mae, tmin


def attempt2(E, tag):
    rows, seen = [], set()
    for x in E.itertuples():
        f = FLOW[x.d]
        _, b = rth_slice(x.d)
        h, l = f["h"][x.i + 1:b], f["l"][x.i + 1:b]
        adv = (x.L - l) if x.side == 1 else (h - x.L)
        j = np.nonzero(adv >= STOP_PTS)[0]
        if not len(j):
            continue
        j0 = int(j[0])
        c = np.nonzero((l[j0 + 1:] <= x.L) & (x.L <= h[j0 + 1:]))[0]
        if not len(c):
            continue
        i2 = x.i + 1 + j0 + 1 + int(c[0])
        key = (x.d, x.L, int(f["t"][i2]))
        if key in seen:
            continue
        seen.add(key)
        rows.append(dict(d=x.d, tag=tag, L=x.L, mag=x.mag, net=x.net, i=i2, t=int(f["t"][i2]), side=x.side,
                         cell=x.cell, sidx=x.sidx, first=x.first, terc=x.terc))
    return pd.DataFrame(rows)


add_moves(ER, "real")
add_moves(EP, "placebo")
add_random(ER)
add_mae(ER)
add_mae(EP)
A2R, A2P = attempt2(ER, "real_a2"), attempt2(EP, "placebo_a2")
for E, nm in ((A2R, "real_a2"), (A2P, "placebo_a2")):
    add_moves(E, nm)
add_random(A2R)

nS = len(USABLE)
W = np.random.default_rng(SEED_BOOT).multinomial(nS, np.full(nS, 1 / nS), size=N_BOOT).astype(float)


def sess(E, col):
    """per-session (sum, count) of a column over rows where it is finite."""
    e = E[np.isfinite(E[col].to_numpy(float))]
    s, c = np.zeros(nS), np.zeros(nS)
    np.add.at(s, e["sidx"].to_numpy(), e[col].to_numpy(float))
    np.add.at(c, e["sidx"].to_numpy(), 1)
    return s, c


def ci(obs, b):
    lo, hi = np.nanpercentile(b, [2.5, 97.5])
    return dict(est=float(obs), lo=float(lo), hi=float(hi), p=float(np.nanmean(b <= 0)))


def mean_ci(E, col):
    s, c = sess(E, col)
    if c.sum() == 0:
        return dict(n=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        b = (W @ s) / (W @ c)
    return dict(n=int(c.sum()), **ci(s.sum() / c.sum(), b))


def diff_ci(E1, c1, E2, c2):
    s1, n1 = sess(E1, c1)
    s2, n2 = sess(E2, c2)
    if n1.sum() == 0 or n2.sum() == 0:
        return dict(n1=int(n1.sum()), n2=int(n2.sum()))
    with np.errstate(invalid="ignore", divide="ignore"):
        b = (W @ s1) / (W @ n1) - (W @ s2) / (W @ n2)
    return dict(n1=int(n1.sum()), n2=int(n2.sum()), **ci(s1.sum() / n1.sum() - s2.sum() / n2.sum(), b))


def block(Er, Ep, hk="60m"):
    Er = Er.copy()
    Ep = Ep.copy()
    Er["pr"] = Er[f"m_{hk}"] - Er[f"r_{hk}"]
    Er["pos"] = np.where(np.isfinite(Er[f"m_{hk}"]), (Er[f"m_{hk}"] > 0).astype(float), np.nan)
    Ep["pos"] = np.where(np.isfinite(Ep[f"m_{hk}"]), (Ep[f"m_{hk}"] > 0).astype(float), np.nan)
    rp = Er[f"rpos_{hk}"].to_numpy(float)
    return dict(real=mean_ci(Er, f"m_{hk}"), placebo=mean_ci(Ep, f"m_{hk}"), random=mean_ci(Er, f"r_{hk}"),
                vs_placebo=diff_ci(Er, f"m_{hk}", Ep, f"m_{hk}"), vs_random=mean_ci(Er, "pr"),
                share_pos=dict(real=mean_ci(Er, "pos"), placebo=mean_ci(Ep, "pos"),
                               random=float(np.nanmean(rp)) if np.isfinite(rp).any() else None,
                               real_minus_placebo=diff_ci(Er, "pos", Ep, "pos")))


def by_cell(Er, Ep, hk="60m"):
    out = {c: block(Er[Er["cell"] == c], Ep[Ep["cell"] == c], hk) for c in CELLS}
    out["pooled"] = block(Er, Ep, hk)
    return out


def verdict(b):
    vp, vr = b["vs_placebo"], b["vs_random"]
    if "lo" not in vp or "lo" not in vr:
        return "n/a"
    if vp["est"] > 0 and vp["lo"] > 0 and vr["est"] > 0 and vr["lo"] > 0:
        return "SHORTLISTED"
    if vp["hi"] < 0:
        return "REFUTED"
    return "INCONCLUSIVE"


def holm(ps):
    order = np.argsort(ps)
    m, adj, run = len(ps), np.zeros(len(ps)), 0.0
    for r, k in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[k]))
        adj[k] = run
    return adj


RES: dict = dict(info=INFO)
PRIM = by_cell(ER, EP, "60m")
RES["primary_60m"] = PRIM
RES["verdict"] = {c: verdict(PRIM[c]) for c in CELLS + ["pooled"]}
hp = holm(np.array([PRIM[c]["vs_placebo"]["p"] for c in CELLS]))
hr = holm(np.array([PRIM[c]["vs_random"]["p"] for c in CELLS]))
RES["holm"] = {c: dict(vs_placebo_adj_p=float(hp[k]), vs_random_adj_p=float(hr[k]),
                       pass_after_holm=bool(hp[k] < 0.025 and hr[k] < 0.025 and PRIM[c]["vs_placebo"]["est"] > 0
                                            and PRIM[c]["vs_random"]["est"] > 0))
               for k, c in enumerate(CELLS)}
RES["horizons"] = {hk: by_cell(ER, EP, hk) for hk in ("30m", "1300")}
RES["truncated"] = {hk: by_cell(ER, EP, hk) for hk in ("30m_trunc", "60m_trunc")}
RES["first_touch_only"] = by_cell(ER[ER["first"]], EP[EP["first"]], "60m")
RES["first_touch_only_verdict"] = {c: verdict(RES["first_touch_only"][c]) for c in CELLS + ["pooled"]}

# L8 (a) A_rel terciles


def terc_block(Er, Ep):
    out = {}
    for k, nm in enumerate(("low", "mid", "high")):
        out[nm] = dict(real=mean_ci(Er[Er["terc"] == k], "m_60m"), placebo=mean_ci(Ep[Ep["terc"] == k], "m_60m"))
    lo_r, hi_r = Er[Er["terc"] == 0], Er[Er["terc"] == 2]
    lo_p, hi_p = Ep[Ep["terc"] == 0], Ep[Ep["terc"] == 2]
    s = [sess(x, "m_60m") for x in (lo_r, hi_r, lo_p, hi_p)]
    with np.errstate(invalid="ignore", divide="ignore"):
        br = (W @ s[0][0]) / (W @ s[0][1]) - (W @ s[1][0]) / (W @ s[1][1])
        bp = (W @ s[2][0]) / (W @ s[2][1]) - (W @ s[3][0]) / (W @ s[3][1])
    obs_r = s[0][0].sum() / s[0][1].sum() - s[1][0].sum() / s[1][1].sum()
    obs_p = s[2][0].sum() / s[2][1].sum() - s[3][0].sum() / s[3][1].sum()
    out["low_minus_high_real"] = ci(obs_r, br)
    out["low_minus_high_placebo"] = ci(obs_p, bp)
    out["diff_in_diff"] = ci(obs_r - obs_p, br - bp)
    return out


RES["L8a_terciles"] = {c: terc_block(ER[ER["cell"] == c], EP[EP["cell"] == c]) for c in CELLS}
RES["L8a_terciles"]["pooled"] = terc_block(ER, EP)

# L8 (b) MAE before +10


def mae_block(E):
    r = E[E["reach10"]]
    nr = E[~E["reach10"]]
    m = r["mae"].to_numpy(float)
    return dict(n=int(len(E)), reach_share=float(E["reach10"].mean()) if len(E) else None, n_reach=int(len(r)),
                mae_p25_50_75_90=[float(v) for v in np.percentile(m, [25, 50, 75, 90])] if len(m) else [],
                share_mae_le={str(k): float(np.mean(m <= k)) for k in (2, 4, 6, 8, 10)} if len(m) else {},
                t10_median_min=float(np.median(r["t10_min"])) if len(r) else None,
                not_reached_mae_median=float(np.nanmedian(nr["mae"])) if len(nr) else None)


RES["L8b_mae"] = {c: dict(real=mae_block(ER[ER["cell"] == c]), placebo=mae_block(EP[EP["cell"] == c]))
                  for c in CELLS}
RES["L8b_mae"]["pooled"] = dict(real=mae_block(ER), placebo=mae_block(EP))
RES["L8b_mae_first_touch"] = dict(real=mae_block(ER[ER["first"]]), placebo=mae_block(EP[EP["first"]]))

# L8 (c) share positive at each horizon (in each block's share_pos); collected for the table
RES["L8c_share_pos"] = {hk: {c: (RES["primary_60m"] if hk == "60m" else RES["horizons"][hk])[c]["share_pos"]
                             for c in CELLS + ["pooled"]} for hk in ("30m", "60m", "1300")}

# L8 (d) strong levels
RES["L8d_ge100"] = by_cell(ER[ER["mag"] >= STRONG], EP[EP["mag"] >= STRONG], "60m")
RES["L8d_ge100_verdict"] = {c: verdict(RES["L8d_ge100"][c]) for c in CELLS + ["pooled"]}

# L8 (e) A9.5′ re-arm
RES["L8e_attempt2"] = dict(n_real=int(len(A2R)), n_placebo=int(len(A2P)), block=by_cell(A2R, A2P, "60m"))

OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "stage1.json").write_text(json.dumps(RES, indent=1, default=str))
try:
    ER.to_pickle(OUT_DIR / "stage1_events_real.pkl")
    EP.to_pickle(OUT_DIR / "stage1_events_placebo.pkl")
except Exception as exc:  # event dumps are a convenience; stage1.json is the record
    print("event dump failed:", exc)

# ------------------------------------------------------------------------------------------- printing
print(json.dumps(RES, indent=1, default=str))
print()
for c in CELLS + ["pooled"]:
    b = PRIM[c]
    vp, vr = b["vs_placebo"], b["vs_random"]
    print(f"{c:11s} n {b['real']['n']:4d}  real {b['real']['est']:+.2f}  "
          f"vs placebo {vp['est']:+.2f} [{vp['lo']:+.2f}, {vp['hi']:+.2f}]  "
          f"vs random {vr['est']:+.2f} [{vr['lo']:+.2f}, {vr['hi']:+.2f}]  -> {RES['verdict'][c]}")
