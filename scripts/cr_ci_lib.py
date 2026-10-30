#!/usr/bin/env python3
"""CR-CI shared machinery — wide-stop touch trade at CR-BW level touches. READ-ONLY: SELECTs only, no API calls.

Spec: specs/CR-CI-wide-stop-touch-trade.md — L0–L6 (Ryan's kickoff), C1–C10 (pre-run choices). Imported by
scripts/cr_ci_step0.py and the CR-CI run script; executes its loads on import (as the CR-CE scripts do at module level).

Part 1 (universe, levels, placebo levels, touch events): scripts/cr_ce_stage1.py lines 52–400, copied unchanged — the
CR-CE 86-session universe and its 803 re-armed touches with D > 0. Train 2026-01-09 .. 2026-06-05, Databento flow only;
no 2025 row, no R-a row and no holdout row is read (every flow query is bounded by HOLDOUT_TS).

Part 2 (CR-CI): 1-minute bars from the Databento 1-s flow (C3), the setup_candidates confirm-bar entry (C1), CR-BW
direction (C2), EDGE / INTERIOR (C5), the exit simulator (C4), random entries (C7), the session-cluster bootstrap and the
EDGE-post-stratified placebo comparison (C6, C8).
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

# ================================================================================================ CR-CI
assert len(USABLE) == 86 and int((R["D"] > 0).sum()) == 803, "universe differs from CR-CE Step 0 / A1"
COST = 0.5
GRID = [(S, T) for S in (10, 15, 20) for T in (20, 30)]
CENTRE = (15, 20)
IM_GRID = [(k, T) for k in (0.25, 0.4) for T in (20, 30)]
SEED_RAND, N_RAND, SEED_BOOT, N_BOOT, SEED_AA = 20261025, 5, 20260920, 10_000, 20261026
MIN_STRATUM, CONFIRM_BARS, NB = 5, 10, 390  # 390 one-minute bars 06:30 .. 12:59 PT; bar 389 = 12:59
RAND_WIN = 60
STRATA = [f"{c}|{e}" for c in CELLS for e in ("EDGE", "INTERIOR")]


def cfg_name(c):
    return f"S{c[0]}_T{c[1]}" if isinstance(c[0], int) else f"IM{c[0]}_T{c[1]}"


CONFIGS = [cfg_name(c) for c in GRID + IM_GRID]
CFG = {cfg_name(c): c for c in GRID + IM_GRID}


# ------------------------------------------------------------------------------- C3 one-minute bars
def make_bars(d):
    f = FLOW[d]
    a, b = rth_slice(d)
    t0 = ep(pt(d, 6, 30))
    t = f["t"][a:b]
    o, h, l, c = (np.full(NB, np.nan) for _ in range(4))
    if len(t):
        idx = (t - t0) // 60
        u, first = np.unique(idx, return_index=True)
        last = np.r_[first[1:] - 1, len(idx) - 1]
        o[u] = f["o"][a:b][first]
        c[u] = f["c"][a:b][last]
        h[u] = np.maximum.reduceat(f["h"][a:b], first)
        l[u] = np.minimum.reduceat(f["l"][a:b], first)
    return dict(t0=t0, o=o, h=h, l=l, c=c)


def ironbeam_bars(d):
    """C3 alternative (setup_candidates series): ironbeam_es_1m_bars, bar start naive UTC, RTH 06:30 .. 12:59 PT."""
    t0 = ep(pt(d, 6, 30))
    b = q("SELECT datetime, open::float AS o, high::float AS h, low::float AS l, close::float AS c "
          "FROM ironbeam_es_1m_bars WHERE datetime >= %s AND datetime < %s AND datetime < %s ORDER BY datetime",
          (pt(d, 6, 30).replace(tzinfo=None), pt(d, 13, 0).replace(tzinfo=None), HOLDOUT_TS.replace(tzinfo=None)))
    o, h, l, c = (np.full(NB, np.nan) for _ in range(4))
    for x in b.itertuples():
        k = (ep(pd.Timestamp(x.datetime).tz_localize(UTC).to_pydatetime()) - t0) // 60
        if 0 <= k < NB:
            o[k], h[k], l[k], c[k] = x.o, x.h, x.l, x.c
    return dict(t0=t0, o=o, h=h, l=l, c=c)


BAR_SOURCE = os.environ.get("CR_CI_BARS", "databento")
assert BAR_SOURCE in ("databento", "ironbeam")
BARS = {d: (make_bars(d) if BAR_SOURCE == "databento" else ironbeam_bars(d)) for d in FLOW}
INFO["bar_source"] = BAR_SOURCE
INFO["bars"] = dict(missing_rth_minutes={str(d): int(np.isnan(BARS[d]["c"]).sum()) for d in USABLE
                                         if np.isnan(BARS[d]["c"]).any()})


# ------------------------------------------------------------------------------- C1 entry, C5 EDGE
def confirm(d, t, L, s):
    """(confirm bar index j, entry price) or a reason string. setup_candidates rule (scripts/cr_cc_build_fixture.py)."""
    B = BARS[d]
    k = (t - B["t0"]) // 60
    if k < 0 or k >= NB:
        return "touch outside RTH"
    for j in range(k + 1, min(k + 1 + CONFIRM_BARS, NB)):
        c = B["c"][j]
        if np.isfinite(c) and ((s == 1 and c > L) or (s == -1 and c < L)):
            if j >= NB - 1:
                return "confirm at 12:59 (no exit bar)"
            return j, float(c)
    return "no confirm within 10 bars"


def is_edge(d, i, L):
    f = FLOW[d]
    a, _ = rth_slice(d)
    hi, lo = float(f["h"][a:i + 1].max()), float(f["l"][a:i + 1].min())
    return bool(L >= hi - 2.0 or L <= lo + 2.0)


def cc_direction(d, t, L):
    """CR-CC builder's direction (close 5 one-minute bars before the touch bar vs L); C2 disagreement count only."""
    B = BARS[d]
    k = (t - B["t0"]) // 60
    if k < 5 or not np.isfinite(B["c"][k - 5]):
        return 0
    return 1 if B["c"][k - 5] > L else -1


def trades_from(E):
    """Touch events -> trade rows (confirm-bar entry). Returns (trades, funnel dict)."""
    rows, why = [], {}
    for x in E.itertuples():
        r = confirm(x.d, x.t, x.L, x.side)
        if isinstance(r, str):
            why[r] = why.get(r, 0) + 1
            continue
        j, entry = r
        rows.append(dict(d=x.d, sidx=x.sidx, cell=x.cell, first=bool(x.first), L=x.L, mag=x.mag, net=x.net,
                         side=int(x.side), t=int(x.t), i=int(x.i), j=int(j), entry=entry,
                         edge=is_edge(x.d, int(x.i), x.L), cc_dir=cc_direction(x.d, int(x.t), x.L)))
    T = pd.DataFrame(rows)
    T["stratum"] = T["cell"] + "|" + np.where(T["edge"], "EDGE", "INTERIOR")
    return T, dict(events=int(len(E)), trades=int(len(T)), dropped=why)


# ------------------------------------------------------------------------------- C4 exit simulator
def simulate(d, j, entry, s, stop_px, T):
    """Exits on bars j+1 .. 389. Returns (pnl after cost, kind, exit bar, MAE before exit bar, missing bars on path)."""
    B = BARS[d]
    h, l, o = B["h"][j + 1:], B["l"][j + 1:], B["o"][j + 1:]
    tp = entry + s * T
    with np.errstate(invalid="ignore"):
        hs = (l <= stop_px) if s == 1 else (h >= stop_px)
        ht = (h >= tp) if s == 1 else (l <= tp)
    hit = np.nonzero(hs | ht)[0]
    adv = (entry - l) if s == 1 else (h - entry)
    if len(hit):
        m = int(hit[0])
        if hs[m]:
            fill = min(stop_px, o[m]) if s == 1 else max(stop_px, o[m])
            fill = stop_px if not np.isfinite(fill) else fill
            kind = "stop"
        else:
            fill, kind = tp, "target"
    else:
        fin = np.nonzero(np.isfinite(B["c"][j + 1:]))[0]
        if not len(fin):
            return np.nan, "none", -1, np.nan, 0
        m = int(fin[-1])
        fill, kind = float(B["c"][j + 1 + m]), "time"
    mae = float(max(0.0, np.nanmax(adv[:m]))) if m > 0 and np.isfinite(adv[:m]).any() else 0.0
    miss = int(np.isnan(B["c"][j + 1:j + 2 + m]).sum())
    return float(s * (fill - entry) - COST), kind, j + 1 + m, mae, miss


def stop_dist(name, d):
    a, _ = CFG[name]
    return float(a) if isinstance(a, int) else float(a) * IM[d]


def add_trades(T, names=CONFIGS):
    """Per config: pnl, kind, R, pnl_R, mae (before exit bar), min_to_exit (exit bar start - entry time)."""
    for nm in names:
        Tg = CFG[nm][1]
        out = []
        for x in T.itertuples():
            S = stop_dist(nm, x.d)
            sp = x.L - x.side * S
            p, kind, m, mae, miss = simulate(x.d, x.j, x.entry, x.side, sp, Tg)
            R_ = S + abs(x.entry - x.L)
            out.append((p, kind, R_, p / R_, mae, float(m - (x.j + 1)) if m >= 0 else np.nan, miss))
        o = list(zip(*out)) if out else [[]] * 7
        for k, col in enumerate(("pnl", "kind", "R", "pnlR", "mae", "mins", "miss")):
            T[f"{col}_{nm}"] = list(o[k])
    return T


# ------------------------------------------------------------------------------- C7 random entries
def random_draws(T):
    """Per trade, 5 random entry bars (CR-CE generator and seed). Returns list of lists of bar indices (None invalid)."""
    U = np.random.default_rng(SEED_RAND).random((len(T), N_RAND))
    out = []
    for k, x in enumerate(T.itertuples()):
        lo, hi = max(0, x.j + 1 - RAND_WIN), min(x.j + 1 + RAND_WIN, NB - 2)
        rs = []
        for u in U[k]:
            r = lo + int(np.floor(u * (hi - lo + 1)))
            rs.append(r if np.isfinite(BARS[x.d]["c"][r]) else None)
        out.append(rs)
    return out


def add_random(T, draws, names=CONFIGS):
    for nm in names:
        Tg = CFG[nm][1]
        vals, valsR, pos = [], [], []
        for x, rs in zip(T.itertuples(), draws):
            S = stop_dist(nm, x.d)
            R_ = S + abs(x.entry - x.L)
            v = []
            for r in rs:
                if r is None:
                    continue
                e = float(BARS[x.d]["c"][r])
                p = simulate(x.d, r, e, x.side, e - x.side * R_, Tg)[0]
                if np.isfinite(p):
                    v.append(p)
            vals.append(np.mean(v) if v else np.nan)
            valsR.append(np.mean(v) / R_ if v else np.nan)
            pos.append(np.mean(np.array(v) > 0) if v else np.nan)
        T[f"rnd_{nm}"], T[f"rndR_{nm}"], T[f"rndwin_{nm}"] = vals, valsR, pos
    return T


# ------------------------------------------------------------------------------- A1 opposite direction, race
def add_opposite(T, names=CONFIGS):
    """C11: break trade, side −s, same entry, stop = entry + s·R (same risk as the fade), target entry − s·T."""
    for nm in names:
        Tg = CFG[nm][1]
        v = []
        for x in T.itertuples():
            R_ = stop_dist(nm, x.d) + abs(x.entry - x.L)
            v.append(simulate(x.d, x.j, x.entry, -x.side, x.entry + x.side * R_, Tg)[0])
        T[f"opp_{nm}"] = v
        T[f"oppR_{nm}"] = np.array(v) / T[f"R_{nm}"].to_numpy(float) if f"R_{nm}" in T else np.nan
    return T


def add_race(T, targets=(20, 30)):
    """C12: +1 fade target first, −1 break target first, 0 same bar, NaN neither by 12:59 PT (no stops)."""
    for Tg in targets:
        v = []
        for x in T.itertuples():
            B = BARS[x.d]
            h, l = B["h"][x.j + 1:], B["l"][x.j + 1:]
            with np.errstate(invalid="ignore"):
                up, dn = h >= x.entry + Tg, l <= x.entry - Tg
            f, b = (up, dn) if x.side == 1 else (dn, up)
            i = np.nonzero(f | b)[0]
            v.append(np.nan if not len(i) else (0.0 if f[i[0]] and b[i[0]] else (1.0 if f[i[0]] else -1.0)))
        T[f"race_T{Tg}"] = v
    return T


# ------------------------------------------------------------------------------- C6, C8 statistics
nS = len(USABLE)


def weights(n_boot, seed):
    return np.random.default_rng(seed).multinomial(nS, np.full(nS, 1 / nS), size=n_boot).astype(float)


def sess(E, col):
    """per-session (sum, count) of a column over rows where it is finite."""
    v = E[col].to_numpy(float) if len(E) else np.array([])
    e = E[np.isfinite(v)] if len(E) else E
    s, c = np.zeros(nS), np.zeros(nS)
    if len(e):
        np.add.at(s, e["sidx"].to_numpy(), e[col].to_numpy(float))
        np.add.at(c, e["sidx"].to_numpy(), 1)
    return s, c


def ci(obs, b):
    lo, hi = np.nanpercentile(b, [2.5, 97.5])
    return dict(est=float(obs), lo=float(lo), hi=float(hi), p=float(np.nanmean(b <= 0)))


def mean_ci(E, col, W):
    s, c = sess(E, col)
    if c.sum() == 0:
        return dict(n=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        b = (W @ s) / (W @ c)
    return dict(n=int(c.sum()), **ci(s.sum() / c.sum(), b))


def covered_strata(Ep):
    n = Ep.groupby("stratum").size() if len(Ep) else pd.Series(dtype=int)
    return {k for k in STRATA if int(n.get(k, 0)) >= MIN_STRATUM}


def matched_diff(Er, Ep, col, W, cov=None, colp=None):
    """real mean (covered strata) − placebo mean post-stratified to the real stratum mix (C6)."""
    colp = colp or col
    cov = covered_strata(Ep) if cov is None else cov
    Er = Er[Er["stratum"].isin(cov)]
    ks = [k for k in STRATA if k in set(Er["stratum"])]
    if not ks:
        return dict(n_real=0)
    sr, cr = sess(Er, col)
    one = np.ones(nS)
    with np.errstate(invalid="ignore", divide="ignore"):
        real_b, real_o = (W @ sr) / (W @ cr), sr.sum() / cr.sum()
        tot_b, tot_o = W @ cr, cr.sum()
        plc_b, plc_o = np.zeros(len(W)), 0.0
        for k in ks:
            _, ck = sess(Er[Er["stratum"] == k], col)
            spk, cpk = sess(Ep[Ep["stratum"] == k], colp)
            plc_b = plc_b + (W @ ck) / tot_b * ((W @ spk) / (W @ cpk))
            plc_o += ck.sum() / tot_o * (spk.sum() / cpk.sum())
    return dict(n_real=int(cr.sum()), n_placebo=int(len(Ep[Ep["stratum"].isin(ks)])), placebo_matched=float(plc_o),
                **ci(real_o - plc_o, real_b - plc_b))


def edge_shares(Er, Ep, cov=None):
    cov = covered_strata(Ep) if cov is None else cov
    Erc = Er[Er["stratum"].isin(cov)]
    out = dict(real=float(Er["edge"].mean()) if len(Er) else None,
               placebo_raw=float(Ep["edge"].mean()) if len(Ep) else None)
    # matched placebo EDGE share = real stratum mix applied to stratum EDGE indicators
    out["placebo_matched"] = float(Erc["edge"].mean()) if len(Erc) else None
    return out


def coverage(Er, Ep, rnd_col=None):
    cov = covered_strata(Ep)
    out = dict(n_real=int(len(Er)), placebo_covered=float(Er["stratum"].isin(cov).mean()) if len(Er) else None)
    if rnd_col is not None:
        out["random_covered"] = float(np.isfinite(Er[rnd_col].to_numpy(float)).mean()) if len(Er) else None
    return out


def holm(ps):
    order = np.argsort(ps)
    m, adj, run = len(ps), np.zeros(len(ps)), 0.0
    for r, k in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[k]))
        adj[k] = run
    return adj


# ------------------------------------------------------------------------------- trade sets
TR, FUN_R = trades_from(R[R["D"] > 0].copy())
TP, FUN_P = trades_from(P[P["D"] > 0].copy())
INFO["trade_funnel"] = dict(real=FUN_R, placebo=FUN_P)


# ================================================================================================ A2
from packages.shared import flow_features as ff  # noqa: E402
from packages.shared.surface_compare import k_for_abs_delta  # noqa: E402

SWINGS, SWING_PRIMARY = (8.0, 12.0, 16.0), 12.0
ABSORB_MIN, FREE_MAX, SNAP_MAX_AGE = 1.5, 1.0, 300
RACE_PTS = 20.0


def rth_minutes(d):
    """C14: completed Databento minutes 06:30 .. 12:59 PT (arrays of length NB): h, l, c, vol, delta."""
    f = FLOW[d]
    a, b = rth_slice(d)
    F = {k: f[k][a:b] for k in ("t", "o", "h", "l", "c", "buy", "sell")}
    F["vol"] = F["buy"] + F["sell"]
    t0 = ep(pt(d, 6, 30))
    m = ff.complete_minutes(ff.minutes_from_seconds(F)[list(ff.COLUMNS)], t0, t0 + 60 * NB)
    return dict(h=m["h"].to_numpy(float), l=m["l"].to_numpy(float), c=m["c"].to_numpy(float),
                vol=m["vol"].to_numpy(float), delta=(m["buy"] - m["sell"]).to_numpy(float), t0=t0)


MIN = {d: rth_minutes(d) for d in USABLE}


def approach_leg(d, k, side, swing):
    """C15/C16. Returns dict (dir, start_i, start_px, ext, pts, push, cpp, prev_cpp, ratio) or a reason string."""
    m = MIN[d]
    if k < 1:
        return "touch in first bar"
    legs = ff.leg_effort(ff.swing_legs(m["h"][:k + 1], m["l"][:k + 1], swing), m["vol"][:k + 1], m["delta"][:k + 1])
    if not legs:
        return "no confirmed leg"
    last = legs[-1]
    dirn, si, spx = -last["dir"], last["end_i"], float(last["end_px"])
    if dirn != -side:
        return "in-progress leg against the approach"
    seg = slice(si + 1, k + 1)
    if si >= k:
        return "leg has no bars"
    ext = float(m["h"][seg].max()) if dirn == 1 else float(m["l"][seg].min())
    pts_ = abs(ext - spx)
    if pts_ <= 0:
        return "zero-point leg"
    vol, dl = float(m["vol"][seg].sum()), float(m["delta"][seg].sum())
    cpp = vol / pts_
    prev = [lg for lg in legs if lg["dir"] == dirn]
    pc = prev[-1]["cpp"] if prev else None
    ratio = cpp / pc if (pc is not None and pc > 0) else np.nan
    return dict(dir=dirn, start_i=int(si), start_px=spx, ext=ext, pts=pts_, push=dl * dirn, cpp=cpp,
                prev_cpp=pc if pc is not None else np.nan, ratio=ratio, mins=int(k - si))


def effort_class(ratio):
    if not np.isfinite(ratio):
        return "neutral"
    return "ABSORBED" if ratio >= ABSORB_MIN else ("FREE" if ratio <= FREE_MAX else "neutral")


# ------------------------------------------------------------------------------- C17–C20 skew
import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location("skew_callbacks", REPO_ROOT / "apps" / "web" / "modules" / "Skew" / "callbacks.py")
SK = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(SK)
assert SK.BETA_VOLPTS_PER_1PCT == 4.5 and SK.THETA_ATM_PP_PER_SQRT_YEAR == -638 and SK.BETA_MAX_SHIFT_PP == 6.0


def skew_expiry(d):
    r = q("""SELECT DISTINCT expir_date, expiry_tod FROM orats_monies_minute
             WHERE ticker = 'SPX' AND trade_date = %s AND expir_date >= %s ORDER BY expir_date LIMIT 6""",
          (d.isoformat(), d.isoformat()))
    pm = r[r["expiry_tod"].astype(str).str.lower() == "pm"]["expir_date"].tolist()
    if not pm:
        return None, None
    return pm[0], bool(pm[0] != d.isoformat())


def skew_rows(d, exp_iso):
    r = q("""SELECT * FROM orats_monies_minute WHERE ticker = 'SPX' AND trade_date = %s AND expir_date = %s
             AND snapshot_pt >= %s AND snapshot_pt < %s ORDER BY snapshot_pt""",
          (d.isoformat(), exp_iso, f"{d.isoformat()} 06:30:00", f"{d.isoformat()} 13:00:00"))
    r["snap_pt"] = pd.to_datetime(r["snapshot_pt"])
    r["snap_ep"] = [ep(x.to_pydatetime().replace(tzinfo=PT)) for x in r["snap_pt"]]
    return r.reset_index(drop=True)


SKEXP, SKROWS = {}, {}
for _d in USABLE:
    SKEXP[_d] = skew_expiry(_d)
    SKROWS[_d] = skew_rows(_d, SKEXP[_d][0]) if SKEXP[_d][0] else pd.DataFrame()


def snap_at(d, bar_i):
    """C17: latest snapshot <= the bar's close; None if missing or > 5 min older."""
    r = SKROWS[d]
    if r.empty:
        return None
    x = MIN[d]["t0"] + 60 * (bar_i + 1)
    j = np.searchsorted(r["snap_ep"].to_numpy(), x, side="right") - 1
    if j < 0 or x - int(r["snap_ep"].iloc[j]) > SNAP_MAX_AGE:
        return None
    return r.iloc[j]


def residual_put_skew(prev_row, now_row, exp_iso):
    """C18: the callbacks.py Expected-SS block, prev = leg start, now = touch. Returns put_skew_now − exp_put_skew (pp)."""
    stock_now, prev_stock = float(now_row["stock_price"]), float(prev_row["stock_price"])
    atm_now, _, put_now = SK._skews_from_row(now_row)
    T_now, prev_T = SK._T_from_row_snapshot(now_row, exp_iso), SK._T_from_row_snapshot(prev_row, exp_iso)
    k_prev, s_prev = SK._prev_smile_interp(prev_row, prev_T)
    k_shift = np.log(stock_now / prev_stock) if (prev_stock and stock_now) else 0.0
    exp_atm_shape = SK._interp_linear_extrap(k_shift, k_prev, s_prev)
    ret_frac = (stock_now - prev_stock) / prev_stock
    level_shift_pp = max(-SK.BETA_MAX_SHIFT_PP, min(SK.BETA_MAX_SHIFT_PP, (-ret_frac) * 100.0 * SK.BETA_VOLPTS_PER_1PCT))
    droot = max(0.0, np.sqrt(max(prev_T, SK.EPS_T)) - np.sqrt(max(T_now, SK.EPS_T)))
    atm_exp = exp_atm_shape + level_shift_pp / 100.0 + SK.THETA_ATM_PP_PER_SQRT_YEAR * droot / 100.0
    k_p25_now = k_for_abs_delta(0.25, is_put=True, sigma=atm_now, T=T_now)
    exp_p25 = SK._interp_linear_extrap(k_p25_now + k_shift, k_prev, s_prev) + (atm_exp - exp_atm_shape)
    return float(put_now - (exp_p25 - atm_exp) * 100.0)


def skew_read(d, start_i, k, dirn):
    """C19: (residual change, raw change, read) with read in {divergent, confirming, none}."""
    a, b = snap_at(d, start_i), snap_at(d, k)
    if a is None or b is None:
        return np.nan, np.nan, "none", "missing snapshot"
    raw = float((b["vol75"] - b["vol50"]) * 100.0 - (a["vol75"] - a["vol50"]) * 100.0)
    try:
        res = residual_put_skew(a, b, SKEXP[d][0])
    except Exception as exc:  # module block failure -> missing (C18)
        return np.nan, raw, "none", f"module: {type(exc).__name__}"
    if not np.isfinite(res) or res == 0:
        return res, raw, "none", "zero or non-finite"
    return res, raw, ("divergent" if np.sign(res) == dirn else "confirming"), "ok"


def raw_read(raw, dirn):
    if not np.isfinite(raw) or raw == 0:
        return "none"
    return "divergent" if np.sign(raw) == dirn else "confirming"


def call(push, eff, sk):
    if push > 0 and eff == "ABSORBED" and sk == "divergent":
        return "FADE"
    if push > 0 and eff == "FREE" and sk == "confirming":
        return "BREAK"
    return "NO TRADE"


def call_delta(push, eff):
    return "FADE" if (push > 0 and eff == "ABSORBED") else ("BREAK" if (push > 0 and eff == "FREE") else "NO TRADE")


def call_skew(sk):
    return {"divergent": "FADE", "confirming": "BREAK"}.get(sk, "NO TRADE")


def a2_reads(E):
    """Touch events -> one row per event with entry (C14) and A2 reads at every swing size."""
    rows = []
    for x in E.itertuples():
        m = MIN[x.d]
        k = int((x.t - m["t0"]) // 60)
        row = dict(d=x.d, sidx=x.sidx, cell=x.cell, first=bool(x.first), L=x.L, side=int(x.side), t=int(x.t), k=k,
                   entry=float(m["c"][k]))
        for sw in SWINGS:
            tag = f"s{int(sw)}"
            lg = approach_leg(x.d, k, x.side, sw)
            if isinstance(lg, str):
                row.update({f"{tag}_why": lg, f"{tag}_call": "NO TRADE", f"{tag}_call_delta": "NO TRADE",
                            f"{tag}_call_skew": "NO TRADE", f"{tag}_call_raw": "NO TRADE"})
                continue
            res, raw, sk, why = skew_read(x.d, lg["start_i"], k, lg["dir"])
            eff = effort_class(lg["ratio"])
            row.update({f"{tag}_why": "ok", f"{tag}_push": lg["push"], f"{tag}_ratio": lg["ratio"],
                        f"{tag}_pts": lg["pts"], f"{tag}_mins": lg["mins"], f"{tag}_eff": eff,
                        f"{tag}_res": res, f"{tag}_raw": raw, f"{tag}_skew": sk, f"{tag}_skew_why": why,
                        f"{tag}_call": call(lg["push"], eff, sk), f"{tag}_call_delta": call_delta(lg["push"], eff),
                        f"{tag}_call_skew": call_skew(sk),
                        f"{tag}_call_raw": call(lg["push"], eff, raw_read(raw, lg["dir"]))})
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------- C24 any-price population
def swing_levels(d, swing=SWING_PRIMARY):
    """Confirmed swing extremes in RTH with the epoch second they become known (confirm bar close)."""
    m = MIN[d]
    legs = ff.swing_legs(m["h"], m["l"], swing)
    out = {}
    for lg in legs:
        px, known = float(lg["end_px"]), m["t0"] + 60 * (lg["confirm_i"] + 1)
        if px not in out or known < out[px]:
            out[px] = known
    return [dict(L=px, mag=0.0, net=0.0, t_known=tk) for px, tk in out.items()]


def events_from(d, levels, tag):
    """events_for (CR-CE, unchanged rule) with each level armed only from its t_known."""
    out = []
    for lv in levels:
        for ev in events_for(d, [lv], tag):
            if ev["t"] > lv["t_known"]:
                out.append(ev)
    return out


def build_anyprice():
    rows = []
    for d in USABLE:
        rows += events_from(d, swing_levels(d), "anyprice")
    E = pd.DataFrame(rows)
    n0 = len(E)
    E = E[(E["pre_gap"] <= GAP_MAX) & (E["side"] != 0)].copy()
    E["first"] = E.groupby(["d", "L"])["t"].transform("min") == E["t"]   # first touch after the level is known
    E["cell"] = np.where(E["side"] == 1, "above", "below")
    E["sidx"] = [USABLE.index(d) for d in E["d"]]
    return E[E["D"] > 0].copy(), dict(touches=n0, kept=int(len(E)), d_gt0=int((E["D"] > 0).sum()))
