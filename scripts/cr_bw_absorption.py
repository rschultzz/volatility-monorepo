#!/usr/bin/env python3
"""CR-BW run — aggressor absorption at levels (T1, T2, S1–S9). READ-ONLY: SELECTs only, no API calls.

Spec: specs/CR-BW-aggressor-absorption-at-levels.md — L1–L11, Amendment A1 (A1.1–A1.5, A1.7; A1.6's drop revoked),
Amendment A2 (A2.1–A2.3). One run. Every statistic is computed and written to
scripts/logs/cr_bw/results.json before anything is printed.

  --counts-only   stop after event construction and print tercile counts (no race is run; used for the A2
                  restatement before this script's commit).

Definitions (from scripts/cr_bw_step0_a1.py unless marked A2):
  - connection: get_backfill_db_conn() + default_transaction_read_only = on (asserted); statement_timeout 600 s;
    Python alarm 3 h. Holdout guard: sessions <= 2026-06-05, flow ts_utc < 2026-06-05 21:00 UTC.
  - sessions: NYSE trading days 2025-12-26 .. 2026-06-05; L14 (a) roll week, (b) no ES 15:45 ET bar on A,
    (c) roll mismatch; flow rows present; canonical bt_daily_features IM > 0; orats_oi_gamma rows;
    >= 5 prior baseline sessions (A1.5).
  - A2.1 RTH open O_d: open of the first flow row at or after 06:30:00 PT (no 1m-bar fallback).
  - A1.1 levels: [D, D+30] rows, place_levels at the ES 15:45 ET close on A, 1-pt buckets, mag = |call| + |put|
    >= 40e9, 5-pt spacing by descending mag over the whole universe, |L - O_d| <= IM_d, top 6 by mag.
  - A1.2 placebo: per real level the 2 nearest 5-pt bins on the same side of the open, not real, >= 10 pts from
    every real level, within ±IM_d; ties to the lower bin; dedup per session; parent mag/net kept.
  - touch (L5), approach side, fill (L6: 60 s, 1 tick through), predictor (L7: 300 s, D, dP, A, A_rel = A / B_d).
  - A1.4 pre-window: drop an event with an RTH flow gap > 15 s in [t - 300, t].
  - A2.1: no outlier-minute drop on the primary path.
  - L8 terciles: cut points over all real events with D > 0 after the pre-window rule (before fills and race);
    placebo uses the same cut points.
  - race (L6): fade long from above (short from below). Entry at L, filled at the first bar within 60 s
    trading 1 tick through L. Target L ± R, hit on a trade 1 tick through it; stop L ∓ R. From the bar after the
    fill bar; the fill bar counts a stop, not a target; same bar both -> loss; neither by 13:00 PT -> unresolved
    (excluded from rates). Race end = resolving bar, else 13:00 PT.
  - A2.3: drop a filled event whose race window [fill bar time, race end] has an RTH flow gap > 15 s.
    Applied per R (the window depends on R). Drops counted per tercile.
  - T1 (real): win rate low tercile − high tercile, R = 8 ticks. Pass: >= +8 pts, CI lower > 0, >= 150 resolved in
    each tercile. T2: T1(real) − T1(placebo). Pass: >= +5 pts, CI lower > 0.
  - L10 bootstrap: session-cluster, 10 000 resamples, seed 20260920, 95 % percentile; one weight matrix
    (multinomial session counts) reused for every statistic; real and placebo resampled jointly by session.
  - S1: R = 4, 6, 8 ticks: win rate and net expectancy per resolved trade in ticks: win +R, loss −(R + 1)
    (stop + 1 tick slippage), minus 0.36 tick ($4.50 round trip, $12.50 per tick); by tercile, real and placebo;
    break-even win rate (R + 1.36) / (2R + 1).
  - S2: T1 and T2 at R = 4, 6 ticks and R = 0.20·IM_d (rounded to the nearest tick per session).
  - S3: T1 and T2 within negative-net and positive-net levels (placebo by parent sign).
  - S4: T1 and T2 within touches before / at-or-after 10:00 PT.
  - S5: no-fill share by tercile, real and placebo (eligible events).
  - S6: D <= 0 events: win rate real − placebo (R = 8), with CI.
  - S7: A over 60 s and 900 s windows (same events; D_w > 0 required; own cut points over real); T1 and T2.
  - S8: levels with mag >= 100e9 (placebo by parent mag): n per tercile, T1, T2, S1 table. No conclusion below
    50 per tercile.
  - S9 (A2.2): T1 and T2 with the A1.6 drop applied: an event is dropped if a flow-minute vs ironbeam_es_1m_bars
    difference > 1 pt (any of O/H/L/C, RTH) overlaps its pre-window [t − 300, t] or race window.

Usage: apps/web/.venv/bin/python -u scripts/cr_bw_absorption.py | tee scripts/logs/cr_bw/run.log
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

COUNTS_ONLY = "--counts-only" in sys.argv
signal.alarm(3 * 3600)

PT, UTC = ZoneInfo("America/Los_Angeles"), timezone.utc
D_FROM, D_TO = date(2025, 12, 26), date(2026, 6, 5)
HOLDOUT_TS = datetime(2026, 6, 5, 21, 0, tzinfo=UTC)
TICK, COST_TICKS = 0.25, 4.50 / 12.50
DTE_MAX, MAG_MIN, SPACING, WIN_IM, TOP_N, STRONG = 30, 40e9, 5.0, 1.0, 6, 100e9
ARM_IM, WIN_S, FILL_S, GAP_MAX, OUTLIER_PTS = 0.10, 300, 60, 15, 1.0
BASE_N, BASE_MIN, BASE_W = 20, 5, 300
PLC_BIN, PLC_SEP, PLC_K = 5.0, 10.0, 2
R_PRIMARY, R_LADDER = 8, (4, 6, 8)
SEED_BOOT, N_BOOT = 20260920, 10_000
ROLL_MISMATCH_EXTRA = {date(2026, 6, 15)}
OUT_DIR = REPO_ROOT / "scripts" / "logs" / "cr_bw"

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


SESSIONS = trading_days(D_FROM, D_TO)
QEXP = {third_friday(y, m) for y in (2025, 2026) for m in (3, 6, 9, 12)}
INFO: dict = {}

# ---------------------------------------------------------------------------------------------- loads
sym = q("""
    SELECT trade_date, symbol, sum(volume) AS vol
    FROM ironbeam_es_flow_1s
    WHERE ts_utc >= %s AND ts_utc < %s
      AND (ts_utc AT TIME ZONE 'America/Los_Angeles')::time >= '06:30'
      AND (ts_utc AT TIME ZONE 'America/Los_Angeles')::time < '13:00'
    GROUP BY 1, 2
""", (pt(D_FROM, 0, 0), HOLDOUT_TS))
assert not sym.duplicated("trade_date").any(), "more than one RTH symbol on a session"
SYM = dict(zip(sym["trade_date"], sym["symbol"]))

FLOW: dict[date, dict] = {}
for d in SESSIONS:
    if d not in SYM:
        continue
    f = q("""
        SELECT ts_utc, buy_vol, sell_vol, open, high, low, close
        FROM ironbeam_es_flow_1s
        WHERE symbol = %s AND ts_utc >= %s AND ts_utc < %s AND ts_utc < %s
        ORDER BY ts_utc
    """, (SYM[d], pt(d, 5, 30), pt(d, 13, 0), HOLDOUT_TS))
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

es_starts = {es_snapshot_bar_start_utc(prev_trading_day(d)): d for d in SESSIONS}
es = q("SELECT datetime, close::float AS close FROM ironbeam_es_1m_bars WHERE datetime = ANY(%s)",
       (list(es_starts.keys()),))
PTGT = {es_starts[pd.Timestamp(t).to_pydatetime()]: c for t, c in zip(es["datetime"], es["close"])
        if c is not None and np.isfinite(c) and c > 0}


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
ex_b = {d for d in SESSIONS if d not in PTGT}
ex_c = {d for d in SESSIONS if d in QEXP or d in ROLL_MISMATCH_EXTRA}
L14 = [d for d in SESSIONS if d not in ex_a | ex_b | ex_c]


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
B = {}
for d in SESSIONS:
    prior = [x for x in BASE_ELIG if x < d][-BASE_N:]
    if len(prior) >= BASE_MIN:
        B[d] = float(np.median(np.concatenate([AW[x] for x in prior])))

U = list(SESSIONS)
funnel = []
for name, rule in [
    ("L14a roll week", lambda d: d in ex_a),
    ("L14b no ES 15:45 ET bar on A", lambda d: d in ex_b),
    ("L14c roll mismatch", lambda d: d in ex_c),
    ("no flow rows / no RTH open", lambda d: d not in OPEN),
    ("no canonical bt_daily_features row", lambda d: d not in feat.index),
    ("IM null or <= 0", lambda d: d in feat.index and not (pd.notna(feat[d]) and feat[d] > 0)),
    ("no orats_oi_gamma rows", lambda d: d not in set(oig["d"])),
    ("< 5 prior baseline sessions", lambda d: d not in B),
]:
    rm = [d for d in U if rule(d)]
    U = [d for d in U if not rule(d)]
    funnel.append([name, len(rm), len(U)])
USABLE = U
IM = {d: float(feat[d]) for d in USABLE}
INFO["sessions"] = dict(funnel=funnel, n=len(USABLE), first=str(USABLE[0]), last=str(USABLE[-1]),
                        open_2026_03_04=OPEN.get(date(2026, 3, 4)))

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
                      real_ge100=sum(x["mag"] >= STRONG for v in LEVELS.values() for x in v))


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
        start = 0
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
            j1 = min(np.searchsorted(t, ti + FILL_S, side="right"), b)
            fj = np.nonzero(l[i:j1] <= L - TICK)[0] if side == 1 else np.nonzero(h[i:j1] >= L + TICK)[0]
            ev = dict(d=d, tag=tag, L=L, mag=lv["mag"], net=lv["net"], i=i, t=ti, side=side,
                      fill_i=i + int(fj[0]) if len(fj) else -1, pre_gap=max_gap(d, ti - WIN_S, ti))
            for w in (WIN_S, 60, 900):
                w0 = np.searchsorted(t, ti - w)
                buy, sell = int(f["buy"][w0:i].sum()), int(f["sell"][w0:i].sum())
                D = (sell - buy) if side == 1 else (buy - sell)
                p_start = c[w0 - 1] if w0 > 0 else np.nan
                dP = (p_start - c[i - 1]) if side == 1 else (c[i - 1] - p_start)
                sfx = "" if w == WIN_S else f"_{w}"
                ev[f"D{sfx}"] = D
                ev[f"A_rel{sfx}"] = dP / (D / 1000.0) / B[d] if D > 0 else np.nan
            out.append(ev)
            start = cont[ci] + 1
    return out


def build(levels, tag):
    E = pd.DataFrame(sum((events_for(d, levels[d], tag) for d in USABLE), []))
    n0 = len(E)
    E = E[(E["pre_gap"] <= GAP_MAX) & (E["side"] != 0)].copy()
    return E, n0


R, nR0 = build(LEVELS, "real")
P, nP0 = build(PLV, "placebo")
ER = R[R["D"] > 0]
CUT = np.percentile(ER["A_rel"], [100 / 3, 200 / 3])
CUTS = {"": CUT}
for w in (60, 900):
    e = R[R[f"D_{w}"] > 0]
    CUTS[f"_{w}"] = np.percentile(e[f"A_rel_{w}"], [100 / 3, 200 / 3])
for E in (R, P):
    for sfx, ct in CUTS.items():
        x = E[f"A_rel{sfx}"].to_numpy()
        E[f"terc{sfx}"] = np.where(np.isnan(x), -1, np.where(x <= ct[0], 0, np.where(x <= ct[1], 1, 2)))
    E["filled"] = E["fill_i"] >= 0
    E["sidx"] = [USABLE.index(d) for d in E["d"]]


def tcounts(E):
    e = E[E["D"] > 0]
    return [[int((e["terc"] == k).sum()), int(((e["terc"] == k) & e["filled"]).sum())] for k in range(3)]


INFO["events"] = dict(real_touch=nR0, real_kept=len(R), real_eligible=int((R["D"] > 0).sum()),
                      real_d_le0=int((R["D"] <= 0).sum()), placebo_touch=nP0, placebo_kept=len(P),
                      placebo_eligible=int((P["D"] > 0).sum()), cut=CUT.tolist(),
                      real_terc=tcounts(R), placebo_terc=tcounts(P))

if COUNTS_ONLY:
    print(json.dumps(INFO, indent=1, default=str))
    print("real terciles [eligible, filled]:", INFO["events"]["real_terc"],
          "GATE >= 150 filled:", "PASS" if min(x[1] for x in INFO["events"]["real_terc"]) >= 150 else "FAIL")
    sys.exit(0)


# ----------------------------------------------------------------------------------------------- race
def race(d, fi, L, s, Rt):
    """(1 win / 0 loss / -1 unresolved, end epoch)."""
    f = FLOW[d]
    _, b = rth_slice(d)
    Rp = Rt * TICK
    if (s == 1 and f["l"][fi] <= L - Rp) or (s == -1 and f["h"][fi] >= L + Rp):
        return 0, int(f["t"][fi])
    if s == 1:
        tgt, stp = f["h"][fi + 1:b] >= L + Rp + TICK, f["l"][fi + 1:b] <= L - Rp
    else:
        tgt, stp = f["l"][fi + 1:b] <= L - Rp - TICK, f["h"][fi + 1:b] >= L + Rp
    ti, si = np.nonzero(tgt)[0], np.nonzero(stp)[0]
    if not len(ti) and not len(si):
        return -1, ep(pt(d, 13, 0))
    k = min(ti[0] if len(ti) else 10**9, si[0] if len(si) else 10**9)
    return (1 if len(ti) and (not len(si) or ti[0] < si[0]) else 0), int(f["t"][fi + 1 + k])


def r_ticks(d, spec):
    return spec if isinstance(spec, int) else max(1, int(round(0.20 * IM[d] / TICK)))


R_SPECS = {4: 4, 6: 6, 8: 8, "0.20IM": "im"}
for E in (R, P):
    fl = E["filled"].to_numpy()
    for key, spec in R_SPECS.items():
        res, gap = np.full(len(E), -2), np.zeros(len(E), int)
        ends = np.zeros(len(E), np.int64)
        for k, x in enumerate(E.itertuples()):
            if not fl[k]:
                continue
            rt = r_ticks(x.d, spec)
            r_, end = race(x.d, int(x.fill_i), x.L, x.side, rt)
            res[k], ends[k] = r_, end
            gap[k] = max_gap(x.d, int(FLOW[x.d]["t"][int(x.fill_i)]), end)
        E[f"res_{key}"] = res
        E[f"gapdrop_{key}"] = fl & (gap > GAP_MAX)
        E[f"end_{key}"] = ends

# -------------------------------------------------------------------------- S9 outlier minutes (A1.6)
BARS = q("""SELECT datetime, open::float o, high::float h, low::float l, close::float c FROM ironbeam_es_1m_bars
            WHERE datetime >= %s AND datetime < %s""",
         (pt(D_FROM, 6, 30).replace(tzinfo=None), HOLDOUT_TS.replace(tzinfo=None)))
BARS["m"] = [int(pd.Timestamp(x).tz_localize("UTC").timestamp()) for x in BARS["datetime"]]
BARS = BARS.set_index("m")
OUT_MIN = {}
for d in USABLE:
    f = FLOW[d]
    a, b = rth_slice(d)
    df = pd.DataFrame(dict(t=f["t"][a:b], o=f["o"][a:b], h=f["h"][a:b], l=f["l"][a:b], c=f["c"][a:b]))
    df["m"] = df["t"] // 60 * 60
    m = df.groupby("m").agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"))
    j = m.join(BARS[["o", "h", "l", "c"]], rsuffix="_b", how="inner")
    bad = np.zeros(len(j), bool)
    for k in "ohlc":
        bad |= (j[k] - j[f"{k}_b"]).abs().to_numpy() > OUTLIER_PTS
    OUT_MIN[d] = np.array(sorted(j.index[bad]), dtype=np.int64)


def outlier_in(d, t0, t1):
    m = OUT_MIN.get(d, np.array([], np.int64))
    return bool(np.any((m + 60 > t0) & (m <= t1)))


for E in (R, P):
    E["s9_drop"] = [outlier_in(x.d, x.t - WIN_S, max(x.t, x.end_8)) for x in E.itertuples()]

# ------------------------------------------------------------------------------------------ bootstrap
nS = len(USABLE)
W = np.random.default_rng(SEED_BOOT).multinomial(nS, np.full(nS, 1 / nS), size=N_BOOT).astype(float)


def counts(E, key="8", terc="terc", extra=None, dcol="D"):
    """session × 3 terciles × (wins, resolved) for eligible, filled, kept, resolved events."""
    m = (E[dcol] > 0) & E["filled"] & ~E[f"gapdrop_{key}"] & (E[f"res_{key}"] >= 0) & (E[terc] >= 0)
    if extra is not None:
        m &= extra
    e = E[m]
    c = np.zeros((nS, 3, 2))
    np.add.at(c, (e["sidx"].to_numpy(), e[terc].to_numpy(), 0), e[f"res_{key}"].to_numpy())
    np.add.at(c, (e["sidx"].to_numpy(), e[terc].to_numpy(), 1), 1)
    return c


def rates(c):
    """tercile win rates for c of shape (..., 3, 2)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return c[..., 0] / c[..., 1]


def t1(c):
    r = rates(c)
    return r[..., 0] - r[..., 2]


def boot(cr, cp=None):
    """observed and bootstrap draws of T1 (cp None) or T2 (real − placebo)."""
    obs = t1(cr.sum(0)) - (t1(cp.sum(0)) if cp is not None else 0)
    br = t1(np.einsum("bs,sij->bij", W, cr))
    bp = t1(np.einsum("bs,sij->bij", W, cp)) if cp is not None else 0
    b = br - bp
    lo, hi = np.nanpercentile(b, [2.5, 97.5])
    return dict(est=float(obs) * 100, lo=float(lo) * 100, hi=float(hi) * 100)


def block(key="8", terc="terc", er=None, ep_=None, dcol="D"):
    cr, cp = counts(R, key, terc, er, dcol), counts(P, key, terc, ep_, dcol)
    return dict(real_n=cr.sum(0)[:, 1].astype(int).tolist(), real_wr=(rates(cr.sum(0)) * 100).round(2).tolist(),
                placebo_n=cp.sum(0)[:, 1].astype(int).tolist(),
                placebo_wr=(rates(cp.sum(0)) * 100).round(2).tolist(), T1=boot(cr), T1_placebo=boot(cp),
                T2=boot(cr, cp))


RES: dict = dict(info=INFO)

# drops and unresolved (R = 8)
for name, E in (("real", R), ("placebo", P)):
    e = E[(E["D"] > 0) & E["filled"]]
    RES[f"drops_{name}"] = dict(
        eligible_filled=[int((e["terc"] == k).sum()) for k in range(3)],
        race_gap_drop=[int(((e["terc"] == k) & e["gapdrop_8"]).sum()) for k in range(3)],
        unresolved=[int(((e["terc"] == k) & ~e["gapdrop_8"] & (e["res_8"] == -1)).sum()) for k in range(3)])

PRIM = block("8")
RES["primary"] = PRIM
n_ok = min(PRIM["real_n"]) >= 150
RES["T1_pass"] = bool(PRIM["T1"]["est"] >= 8 and PRIM["T1"]["lo"] > 0 and n_ok)
RES["T2_pass"] = bool(PRIM["T2"]["est"] >= 5 and PRIM["T2"]["lo"] > 0)
RES["T1_parts"] = dict(est_ge_8=PRIM["T1"]["est"] >= 8, ci_lo_gt_0=PRIM["T1"]["lo"] > 0, n_ge_150=n_ok)
RES["T2_parts"] = dict(est_ge_5=PRIM["T2"]["est"] >= 5, ci_lo_gt_0=PRIM["T2"]["lo"] > 0)
if RES["T1_pass"] and RES["T2_pass"]:
    RES["verdict"] = "T1 and T2 pass: absorption at OI levels carries information"
elif RES["T1_pass"]:
    RES["verdict"] = "T1 passes, T2 fails: absorption works but the GEX levels add nothing"
else:
    RES["verdict"] = "T1 fails: this absorption measure does not predict reversals at this scale"


# S1 cost reality
def s1_table(extra_r=None, extra_p=None):
    out = {}
    for Rt in R_LADDER:
        key = str(Rt)
        row = dict(breakeven=round((Rt + 1 + COST_TICKS) / (2 * Rt + 1) * 100, 2))
        for name, E, ex in (("real", R, extra_r), ("placebo", P, extra_p)):
            c = counts(E, key, "terc", ex).sum(0)
            wr = rates(c)
            exp_ = wr * Rt - (1 - wr) * (Rt + 1) - COST_TICKS
            allc = c.sum(0)
            wa = allc[0] / allc[1] if allc[1] else np.nan
            row[name] = dict(n=c[:, 1].astype(int).tolist(), wr=(wr * 100).round(2).tolist(),
                             exp_ticks=np.round(exp_, 3).tolist(), all_n=int(allc[1]),
                             all_wr=round(float(wa) * 100, 2),
                             all_exp=round(float(wa * Rt - (1 - wa) * (Rt + 1) - COST_TICKS), 3))
        out[key] = row
    return out


RES["S1"] = s1_table()
RES["S2"] = {k: block(k) for k in ("4", "6", "0.20IM")}
RES["S3"] = dict(negative=block(er=R["net"] < 0, ep_=P["net"] < 0), positive=block(er=R["net"] > 0, ep_=P["net"] > 0))
t10r = R["t"].to_numpy() < np.array([ep(pt(d, 10, 0)) for d in R["d"]])
t10p = P["t"].to_numpy() < np.array([ep(pt(d, 10, 0)) for d in P["d"]])
RES["S4"] = dict(before_10=block(er=pd.Series(t10r, R.index), ep_=pd.Series(t10p, P.index)),
                 after_10=block(er=pd.Series(~t10r, R.index), ep_=pd.Series(~t10p, P.index)))
RES["S5"] = {name: [round(float(1 - E[(E["D"] > 0) & (E["terc"] == k)]["filled"].mean()) * 100, 2) for k in range(3)]
             for name, E in (("real", R), ("placebo", P))}


def s6():
    def c(E):
        m = (E["D"] <= 0) & E["filled"] & ~E["gapdrop_8"] & (E["res_8"] >= 0)
        e = E[m]
        out = np.zeros((nS, 2))
        np.add.at(out, (e["sidx"].to_numpy(), 0), e["res_8"].to_numpy())
        np.add.at(out, (e["sidx"].to_numpy(), 1), 1)
        return out
    cr, cp = c(R), c(P)
    est = cr[:, 0].sum() / cr[:, 1].sum() - cp[:, 0].sum() / cp[:, 1].sum()
    br, bp = W @ cr, W @ cp
    b = br[:, 0] / br[:, 1] - bp[:, 0] / bp[:, 1]
    lo, hi = np.nanpercentile(b, [2.5, 97.5])
    return dict(real_n=int(cr[:, 1].sum()), real_wr=round(cr[:, 0].sum() / cr[:, 1].sum() * 100, 2),
                placebo_n=int(cp[:, 1].sum()), placebo_wr=round(cp[:, 0].sum() / cp[:, 1].sum() * 100, 2),
                diff=dict(est=est * 100, lo=lo * 100, hi=hi * 100))


RES["S6"] = s6()
RES["S7"] = {f"{w}s": block(terc=f"terc_{w}", dcol=f"D_{w}") for w in (60, 900)}
RES["S8"] = dict(block=block(er=R["mag"] >= STRONG, ep_=P["mag"] >= STRONG),
                 S1=s1_table(R["mag"] >= STRONG, P["mag"] >= STRONG))
RES["S9"] = dict(dropped_real=int(((R["D"] > 0) & R["s9_drop"]).sum()),
                 dropped_placebo=int(((P["D"] > 0) & P["s9_drop"]).sum()),
                 block=block(er=~R["s9_drop"], ep_=~P["s9_drop"]))

OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "results.json").write_text(json.dumps(RES, indent=1, default=str))
try:
    R.to_pickle(OUT_DIR / "events_real.pkl")
    P.to_pickle(OUT_DIR / "events_placebo.pkl")
except Exception as exc:  # event dumps are a convenience; results.json is the record
    print("event dump failed:", exc)

# ------------------------------------------------------------------------------------------- printing
print(json.dumps(RES, indent=1, default=str))
p = RES["primary"]
print(f"\nT1 (real, R = 8): low − high = {p['T1']['est']:+.2f} pts [{p['T1']['lo']:+.2f}, {p['T1']['hi']:+.2f}]; "
      f"resolved n {p['real_n']}  -> {'PASS' if RES['T1_pass'] else 'FAIL'} {RES['T1_parts']}")
print(f"T2 (real − placebo): {p['T2']['est']:+.2f} pts [{p['T2']['lo']:+.2f}, {p['T2']['hi']:+.2f}]  -> "
      f"{'PASS' if RES['T2_pass'] else 'FAIL'} {RES['T2_parts']}")
print("VERDICT:", RES["verdict"])
