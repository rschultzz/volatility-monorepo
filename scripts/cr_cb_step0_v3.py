#!/usr/bin/env python3
"""CR-CB Step 0 v3 (Stage 1) under A1–A15 — Databento flow. READ-ONLY: SELECTs only, no API calls.

Spec: specs/CR-CB-failed-retest-at-level.md (Framework, L1–L11, A1–A15). A15.8 order: T0 (A15.9) first; then
range-duration distribution, reference-level and zone counts, fail/break shares, counts per cell at each R_min,
placebo counts, the A/A; then the round-2 gallery (research/gallery/cr-cb-stage1-r2.json, outcome null).

NO trade outcome is reported. Internal stop uses (as v2): the A9.5′ re-arm (train counts are attempt 1 only;
attempts 2–3 only in the gallery and on 2026-10-23) and the Q5 A/A hold-to-close inside the splits.
The A15.2 fail/break share and the A15.9 T0 measures are price-path readouts the spec places in Step 0.

Unchanged from scripts/cr_cb_step0_v2.py (copied): connection/guard, Databento loader (A9.1), universe, σ5_d / B_d
/ skew β_d from prior train sessions, A1.1 levels, A13 clusters, L5 tests, O_c per (cell, d) from real GEX tests
(A9.3), the edge-feature code (A4.3, A5.1, A8) and rules FR1–FR12, stall side, H-PNB / H-PSD / H-CNB.

Choices fixed here before the run (A15 leaves them open):
  - A15.6 placebo: the 4 nearest 5-pt bins per real level, same side of the open, within ±IM, >= 10 pts from every
    real level; dedup; parent sign. Clustered like real levels; zones use the session's refs.
  - A15.1 refs: prior RTH H/L/C (Databento), overnight H/L (ironbeam_es_1m_bars, prior 15:00 → 06:30 PT); swings
    by zigzag at max(8, 0.15·IM), kept only if price does not trade beyond the swing for 600 s after the swing bar;
    a kept swing is known from max(confirmation, swing + 600 s). Zone at test time t: the d-cluster plus the
    nearest ref known before t strictly beyond the cluster's far member in the approach direction, within e, with
    zone span <= 10 pts; else the cluster alone. The zone's far edge (arming price) is the ref if one is added.
    Tests run on each candidate arming price and are kept only when that price is the zone's far edge at the test.
  - O_c per (cell, d) is the A9.3 measurement at GEX arming levels (v2 values), also used for ref-armed zones.
  - Two-sided zone (A15.9/A15.10), real prices: [zone core low − O_c, zone core high + O_c].
  - Detector (short space; longs negated): from the test, running max M; broken if M > arm + O_c before
    confirmation (low <= M − P, high < M). H at confirmation may sit on either side of the level (A15.10).
    PL = the running low after H; the range forms (PL fixed) at the first bar after confirmation with high >= running
    low + max(4, 0.10·IM) (A15.3). If a new high beyond H occurs before PL is fixed, H takes it (and must still be
    <= arm + O_c). PL must be inside the zone, else "outzone". Invalidation: first bar trading >= H + thr or
    <= PL − thr, thr = max(3, 0.5·(H − PL)) (A15.4). Developed = the range still valid R_min after it formed, and
    before the window end (confirmation + W). Edge events: first touch within tol of H / PL after development, while
    valid and before the window end. Orders (A15.2), each placed only after development: fade limits H − tol /
    PL + tol (1 tick through), break stops H + 1 / PL − 1 (+1 tick); counted separately (exclusive per rule).
  - A15.2 fail / break per touched edge, from the touch: top fail = a trade <= H − P before one >= H + 1 (break);
    bottom mirrored; neither by 13:00 = none. Reported per edge cell, real and placebo.
  - A9.5′: a stopped top fade (S-x 2) inside the zone re-arms (new attempt from the stop bar). Episodes per
    (session, cluster, side) are sequential; a cluster side stops after its first episode with a developed range.
  - Gap rule [test, edge touch] > 15 s drops an edge event; A9.6 de-dup on (session, fade direction, fade fill
    second), credited to the higher-magnitude arming member.
  - Centre: O_c 80th, P 6, W 90, tol 1, d 5, e 4, R_min 20, attempts 2, stop S-x 2. Q3: (d, e, R_min) at the A1
    centre (27), and the A1 grid (81) x R_min at d 5, e 4. Cells for counts = type × approach × edge.
  - Range-wide features (A5.1) from range formation (PL fixed) to the touch (A15.3). Big move A15.5.
  - T0 (A15.9), every armed test at its zone arming price (d 0 / e 0 and the centre d 5 / e 4), real vs placebo:
    (d) time to escape = minutes until a trade >= 0.25·IM from the arming price, either side; summarised as the
    restricted mean of min(escape, 120 min, time to 13:00) (censoring-safe) and the share escaping within 60 min;
    (b′) seconds inside the two-sided zone over [t, t + 60 min) on forward-filled 1-s closes (capped at 13:00);
    (a′) range formation = a developed range (R_min 10 / 20 / 30, W 180 so the hold fits) whose PL fixes within
    60 min of the test, both boundaries in the zone; (c) log speed ratio as v2. Session-cluster bootstrap.
  - Range-duration distribution: every PL-fixed range (attempt 1, centre params): minutes from formation to
    invalidation, censored at 13:00 (descriptive price path).
  - Q5 A/A: placebo fade fills at both edges (attempts <= 2), kept, S-x 2 stop beyond the edge, hold to 13:00,
    net 0.09 pts; 200 splits by pseudo-level cluster; as v2.
  - Gallery round 2 (A15.8): composition as round 1 (40 random edge events stratified over the 8 edge cells, seed
    20261024; all H-PNB / H-PSD; all 2026-10-23 events at 7681/7687 at d 5 and 10, e 4, all attempts). Meta adds
    zone bounds (core and two-sided), the arming kind, range bounds / height / formation time / minutes held before
    the touch, and SS side = above / below / straddle of the zone core. set_name "cr-cb:round-2".
  - 2026-10-23: refs = overnight + swings (10-22 not read).

Usage: apps/web/.venv/bin/python -u scripts/cr_cb_step0_v3.py | tee scripts/logs/cr_cb/step0_v3.log
"""
from __future__ import annotations

import itertools
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
HOLDOUT_TS = datetime(2026, 6, 5, 21, 0, tzinfo=UTC)
ILLUS = date(2026, 10, 23)
TICK, COST_PTS = 0.25, 4.50 / 50.0
DTE_MAX, MAG_MIN, SPACING, WIN_IM, TOP_N = 30, 40e9, 5.0, 1.0, 6
ARM_IM, BAND, GAP_MAX, RETREAT = 0.10, 2.0, 15, 2.0
PLC_BIN, PLC_SEP, PLC_K = 5.0, 10.0, 4                     # A15.6
O_PCTS, PULLS, WINS, TOLS, DS = (60, 80, 95), (4.0, 6.0, 10.0), (30, 90, 180), (0.5, 1.0, 2.0), (0, 5, 10)
CENTRE = (80, 6.0, 90, 1.0)
D_C, ATT_C, STOP_C = 5, 2, 2.0
BASE_N, BASE_MIN, BASE_W, PUSH_W = 20, 5, 300, 120
SEED_DRAW, SEED_BOOT, SEED_GAL, N_BOOT, N_SPLIT, N_BOOT_SPLIT = 20261025, 20260920, 20261024, 10_000, 200, 2_000
ROLL_MISMATCH_EXTRA = {date(2026, 6, 15)}
TYPES, SIDES = ("put", "call"), (1, -1)
CELLS = [(ty, s) for ty in TYPES for s in SIDES]
OUT_DIR = REPO_ROOT / "scripts" / "logs" / "cr_cb"
GALLERY = REPO_ROOT / "research" / "gallery" / "cr-cb-stage1-r2.json"
R_MINS, R_C, E_GRID, E_C = (10, 20, 30), 20, (0, 4, 8), 4
REB_PTS, REB_IM, INV_PTS, INV_FRAC = 4.0, 0.10, 3.0, 0.5   # A15.3, A15.4
SW_PTS, SW_IM, SW_HOLD, ZONE_MAX = 8.0, 0.15, 600, 10.0     # A15.1
ESC_IM, ESC_CAP, T0_WIN = 0.25, 120, 3600                   # A15.9

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


def hms(t) -> str:
    return datetime.fromtimestamp(int(t), UTC).astimezone(PT).strftime("%H:%M:%S")


def iso_pt(t) -> str:
    return datetime.fromtimestamp(int(t), UTC).astimezone(PT).isoformat()


SESSIONS = trading_days(D_FROM, D_TO)
QEXP = {third_friday(y, m) for y in (2025, 2026) for m in (3, 6, 9, 12)}


# ---------------------------------------------------------------------------------------------- loads
def load_flow(table: str, d: date, symbol: str, cap: datetime) -> dict:
    f = q(f"""
        SELECT ts_utc, buy_vol, sell_vol, volume, open, high, low, close
        FROM {table}
        WHERE symbol = %s AND ts_utc >= %s AND ts_utc < %s AND ts_utc < %s
        ORDER BY ts_utc
    """, (symbol, pt(d, 6, 15), pt(d, 13, 0), cap))
    return dict(t=np.array([ep(x) for x in f["ts_utc"]], dtype=np.int64),
                buy=f["buy_vol"].to_numpy(np.int64), sell=f["sell_vol"].to_numpy(np.int64),
                vol=f["volume"].to_numpy(np.int64), o=f["open"].to_numpy(float), h=f["high"].to_numpy(float),
                l=f["low"].to_numpy(float), c=f["close"].to_numpy(float))


def front_symbols(table: str, t0: datetime, t1: datetime) -> dict:
    s = q(f"""
        SELECT (ts_utc AT TIME ZONE 'America/Los_Angeles')::date AS d, symbol, sum(volume) AS vol
        FROM {table}
        WHERE ts_utc >= %s AND ts_utc < %s
          AND (ts_utc AT TIME ZONE 'America/Los_Angeles')::time >= '06:30'
          AND (ts_utc AT TIME ZONE 'America/Los_Angeles')::time < '13:00'
        GROUP BY 1, 2
    """, (t0, t1)).sort_values("vol")
    return dict(zip(s["d"], s["symbol"]))   # last (largest volume) wins


def load_oig(d0: date, d1: date) -> pd.DataFrame:
    g = q("""
        SELECT trade_date, expir_date, strike::float AS strike, stock_price::float AS stock_price, gex_call, gex_put
        FROM orats_oi_gamma
        WHERE ticker = 'SPX' AND trade_date BETWEEN %s AND %s
          AND expir_date >= trade_date AND expir_date <= trade_date + %s
    """, (d0 - timedelta(days=7), d1, DTE_MAX + 7))
    g["d"] = [t if is_trading_day(t) else next_trading_day(t) for t in g["trade_date"]]
    g = g[(g["d"] >= d0) & (g["d"] <= d1)]
    g = g[[d <= e <= d + timedelta(days=DTE_MAX) for e, d in zip(g["expir_date"], g["d"])]]
    g["call"] = g["gex_call"].astype(float)
    g["put"] = -g["gex_put"].abs().astype(float)
    return g


def load_im(d0: date, d1: date) -> pd.Series:
    f = q("""
        SELECT trade_date, (feature_vector->>'implied_move_1d')::float AS im
        FROM bt_daily_features
        WHERE feature_version = %s AND ticker = 'SPX' AND active AND trade_date BETWEEN %s AND %s
    """, (CANONICAL_FEATURE_VERSION, d0, d1)).set_index("trade_date")["im"]
    return f[~f.index.duplicated()]


def load_ptgt(days: list[date]) -> dict:
    starts = {es_snapshot_bar_start_utc(prev_trading_day(d)): d for d in days}
    es = q("SELECT datetime, close::float AS close FROM ironbeam_es_1m_bars WHERE datetime = ANY(%s)",
           (list(starts.keys()),))
    return {starts[pd.Timestamp(t).to_pydatetime()]: c for t, c in zip(es["datetime"], es["close"])
            if c is not None and np.isfinite(c) and c > 0}


def load_skew(days: list[date]) -> dict:
    s = q("""
        SELECT trade_date, snapshot_pt, vol25, vol50, vol75, stock_price
        FROM orats_monies_minute
        WHERE ticker IN ('SPX', 'SPXW') AND trade_date = ANY(%s) AND expir_date = trade_date AND expiry_tod = 'pm'
        ORDER BY trade_date, snapshot_pt
    """, ([str(d) for d in days],))
    out = {}
    for td, g in s.groupby("trade_date"):
        g = g.drop_duplicates("snapshot_pt", keep="last")
        d = date.fromisoformat(td)
        tt = np.array([ep(datetime.combine(d, x.time(), PT)) for x in g["snapshot_pt"]], np.int64)
        out[d] = dict(t=tt, ps=(g["vol75"] - g["vol50"]).to_numpy(float), cs=(g["vol25"] - g["vol50"]).to_numpy(float),
                      spot=g["stock_price"].to_numpy(float))
    return out


SYM_DB = front_symbols("es_flow_1s_databento", pt(D_FROM, 0, 0), HOLDOUT_TS)
SYM_IB = front_symbols("ironbeam_es_flow_1s", pt(D_FROM, 0, 0), HOLDOUT_TS)
FLOW, SRC = {}, {}
for d in SESSIONS:
    if d in SYM_DB:
        FLOW[d], SRC[d] = load_flow("es_flow_1s_databento", d, SYM_DB[d], HOLDOUT_TS), "databento"
    elif d in SYM_IB:
        FLOW[d], SRC[d] = load_flow("ironbeam_es_flow_1s", d, SYM_IB[d], HOLDOUT_TS), "ironbeam"
oig = load_oig(D_FROM, D_TO)
feat = load_im(D_FROM, D_TO)
PTGT = load_ptgt(SESSIONS)


def rth_slice(F, d):
    return np.searchsorted(F["t"], ep(pt(d, 6, 30))), np.searchsorted(F["t"], ep(pt(d, 13, 0)))


def max_gap(F, d, t0, t1):
    """Largest RTH flow gap (s) within [t0, t1], clipped to [06:30, 13:00] PT."""
    f = F["t"]
    t0 = max(t0, ep(pt(d, 6, 30)))
    t1 = min(t1, ep(pt(d, 13, 0)))
    if t1 <= t0:
        return 0
    a, b = np.searchsorted(f, t0), np.searchsorted(f, t1, side="right")
    return int(np.diff(np.concatenate([[t0], f[a:b], [t1]])).max())


def rth_open(F, d):
    a, b = rth_slice(F, d)
    return float(F["o"][a]) if b > a else None


OPEN = {d: o for d in FLOW if (o := rth_open(FLOW[d], d)) is not None}
ex_a = {d for d in SESSIONS if roll_week(d)}
ex_b = {d for d in SESSIONS if d not in PTGT}
ex_c = {d for d in SESSIONS if d in QEXP or d in ROLL_MISMATCH_EXTRA}
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
]:
    rm = [d for d in U if rule(d)]
    U = [d for d in U if not rule(d)]
    funnel.append([name, len(rm), len(U), [str(x) for x in rm]])
USABLE = U
IM = {d: float(feat[d]) for d in USABLE}
SIDX = {d: k for k, d in enumerate(USABLE)}
SKEW = load_skew(USABLE)


# ------------------------------------------------------------------------------------- per-session scales
def windows(F, d):
    t0 = ep(pt(d, 6, 30))
    for k in range(78):
        ws, we = t0 + BASE_W * k, t0 + BASE_W * (k + 1)
        a, b = np.searchsorted(F["t"], ws), np.searchsorted(F["t"], we)
        if a == 0 or b <= a or max_gap(F, d, ws, we) > GAP_MAX:
            continue
        yield a, b


def sess_stats(d):
    F = FLOW[d]
    dc, imp = [], []
    for a, b in windows(F, d):
        dc.append(abs(F["c"][b - 1] - F["c"][a - 1]))
        net = int(F["buy"][a:b].sum() - F["sell"][a:b].sum())
        if net:
            imp.append(np.sign(net) * (F["c"][b - 1] - F["c"][a - 1]) / (abs(net) / 1000.0))
    sk = SKEW.get(d)
    xs, ys_p, ys_c = [], [], []
    if sk is not None and len(sk["t"]) > 10:
        t0 = ep(pt(d, 7, 0))
        for k in range(72):
            ta, tb = t0 + 300 * k, t0 + 300 * (k + 1)
            ia, ib = np.searchsorted(sk["t"], ta, side="right") - 1, np.searchsorted(sk["t"], tb, side="right") - 1
            if ia < 0 or ib <= ia or sk["t"][ib] > ep(pt(d, 12, 45)):
                continue
            xs.append(np.log(sk["spot"][ib] / sk["spot"][ia]))
            ys_p.append(sk["ps"][ib] - sk["ps"][ia])
            ys_c.append(sk["cs"][ib] - sk["cs"][ia])
    return np.array(dc), np.array(imp), np.array(xs), np.array(ys_p), np.array(ys_c)


STATS = {d: sess_stats(d) for d in USABLE}
SIG5, BD, BETA = {}, {}, {}
for d in USABLE:
    prior = [x for x in USABLE if x < d][-BASE_N:]
    if len(prior) >= BASE_MIN:
        SIG5[d] = float(np.median(np.concatenate([STATS[x][0] for x in prior])))
        BD[d] = float(np.median(np.concatenate([STATS[x][1] for x in prior])))
        x = np.concatenate([STATS[p][2] for p in prior])
        BETA[d] = {k: float(np.polyfit(x, np.concatenate([STATS[p][j] for p in prior]), 1)[0]) if len(x) > 20
                   else np.nan for k, j in (("ps", 3), ("cs", 4))}


# --------------------------------------------------------------------------------------------- levels
def select_levels(g: pd.DataFrame, ptgt: float, o: float, im: float, top=TOP_N):
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
    return [dict(L=float(k.level), mag=float(k.mag), net=float(k.net)) for k in win[:top]], win


LEVELS = {}
for d, g in oig[oig["d"].isin(USABLE)].groupby("d"):
    LEVELS[d], _ = select_levels(g, PTGT[d], OPEN[d], IM[d])


def placebo(d, levels):
    o, im = OPEN[d], IM[d]
    real = np.array([x["L"] for x in levels])
    pool = {}
    for x in levels:
        s = np.sign(x["L"] - o)
        lo, hi = np.ceil((o - im) / PLC_BIN) * PLC_BIN, np.floor((o + im) / PLC_BIN) * PLC_BIN
        cand = np.arange(lo, hi + 0.5, PLC_BIN)
        cand = cand[(np.sign(cand - o) == s) & (np.min(np.abs(cand[:, None] - real[None, :]), axis=1) >= PLC_SEP)]
        cand = cand[np.lexsort((cand, np.abs(cand - x["L"])))][:PLC_K]
        for b in cand:
            pool.setdefault(float(b), dict(L=float(b), mag=x["mag"], net=x["net"]))
    return list(pool.values())


PLV = {d: placebo(d, LEVELS[d]) for d in USABLE}


def clusters(levels, dd):
    """Chain levels with consecutive gap <= dd. Returns cluster dicts."""
    xs = sorted(levels, key=lambda x: x["L"])
    groups, cur = [], []
    for x in xs:
        if cur and x["L"] - cur[-1]["L"] > dd:
            groups.append(cur)
            cur = []
        cur.append(x)
    if cur:
        groups.append(cur)
    out = []
    for g in groups:
        net = sum(x["net"] for x in g)
        out.append(dict(members=[x["L"] for x in g], top=g[-1]["L"], bot=g[0]["L"], net=net,
                        mag=sum(x["mag"] for x in g), type="put" if net < 0 else "call",
                        mixed=len({np.sign(x["net"]) for x in g}) > 1, n=len(g), span=g[-1]["L"] - g[0]["L"],
                        top_mag=g[-1]["mag"], bot_mag=g[0]["mag"]))
    return out


CL = {dd: {d: clusters(LEVELS[d], dd) for d in USABLE} for dd in DS}
CLP = {dd: {d: clusters(PLV[d], dd) for d in USABLE} for dd in DS}


# ---------------------------------------------------------------------------------------------- tests
def tests_for(F, d, L, im):
    """Armed tests on level L (CR-BW re-arm, 2-pt band): list of (bar index, side)."""
    a, b = rth_slice(F, d)
    h, l = F["h"], F["l"]
    arm = ARM_IM * im
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
    return out


def cluster_tests(F, d, cls, im):
    rows = []
    for k, c in enumerate(cls):
        for side, L, mag in ((1, c["top"], c["top_mag"]), (-1, c["bot"], c["bot_mag"])):
            for j, s in tests_for(F, d, L, im):
                if s == side:
                    rows.append(dict(d=d, ci=k, L=L, side=side, type=c["type"], mixed=c["mixed"], n=c["n"],
                                     members=c["members"], net=c["net"], mag=c["mag"], arm_mag=mag, j=j,
                                     t=int(F["t"][j])))
    return rows


def space(F, side):
    return (F["h"], F["l"], F["c"], 1.0) if side == 1 else (-F["l"], -F["h"], -F["c"], -1.0)


def overshoot(F, d, j, side, L, im):
    _, b = rth_slice(F, d)
    H, Lo, _, sg = space(F, side)
    Ls = sg * L
    M = np.maximum.accumulate(H[j:b])
    ok = (Lo[j:b] <= M - ARM_IM * im) & (H[j:b] < M)
    k = np.nonzero(ok)[0]
    e = (b - 1) if not len(k) else j + int(k[0])
    return max(0.0, float(M[e - j]) - Ls), not len(k), e


# ------------------------------------------------------------------------------- tests and O_c (Q2)
TESTS = {dd: sum((cluster_tests(FLOW[d], d, CL[dd][d], IM[d]) for d in USABLE), []) for dd in DS}
TESTS_P = {dd: sum((cluster_tests(FLOW[d], d, CLP[dd][d], IM[d]) for d in USABLE), []) for dd in DS}
OC, Q2 = {}, {}
for dd in DS:
    for r in TESTS[dd]:
        F = FLOW[r["d"]]
        ov, cens, e = overshoot(F, r["d"], r["j"], r["side"], r["L"], IM[r["d"]])
        r.update(ov=ov, cens=cens, ogap=max_gap(F, r["d"], r["t"], int(F["t"][e])))
    T = pd.DataFrame(TESTS[dd])
    T = T[T["ogap"] <= GAP_MAX]
    for cell in CELLS:
        v = T[(T["type"] == cell[0]) & (T["side"] == cell[1])]["ov"].to_numpy()
        OC[(dd, cell)] = {p: float(np.round(np.percentile(v, p) / TICK) * TICK) for p in O_PCTS}
        Q2[f"d{dd} {cell[0]}/{'below' if cell[1] == 1 else 'above'}"] = dict(
            n=len(v), zero_share=round(float((v == 0).mean()), 3),
            pct={p: round(float(np.percentile(v, p)), 2) for p in (25, 50, 60, 75, 80, 90, 95, 99)},
            O_c=OC[(dd, cell)])


def O_of(dd, op):
    return {c: OC[(dd, c)][op] for c in CELLS}


# put vs call overshoot difference (d = 0), session-cluster bootstrap
rngq = np.random.default_rng(SEED_BOOT)
BI = rngq.integers(0, len(USABLE), size=(2000, len(USABLE)))
T0d = pd.DataFrame(TESTS[0])
T0d = T0d[T0d["ogap"] <= GAP_MAX]
Q2_DIFF = {}
for side in SIDES:
    bs = {(ty, d): T0d[(T0d["type"] == ty) & (T0d["side"] == side) & (T0d["d"] == d)]["ov"].to_numpy()
          for ty in TYPES for d in USABLE}
    for nm, fn in (("p80", lambda v: np.percentile(v, 80)), ("median", np.median)):
        obs = fn(np.concatenate([bs[("put", d)] for d in USABLE])) - fn(np.concatenate([bs[("call", d)] for d in USABLE]))
        bb = [fn(np.concatenate([bs[("put", USABLE[k])] for k in row])) -
              fn(np.concatenate([bs[("call", USABLE[k])] for k in row])) for row in BI]
        lo, hi = np.percentile(bb, [2.5, 97.5])
        Q2_DIFF[f"{'below' if side == 1 else 'above'} {nm} put−call"] = [round(float(obs), 2), round(float(lo), 2),
                                                                          round(float(hi), 2)]



# ------------------------------------------------------------------------------- A15.1 refs and zones
def prior_rth(d):
    p_ = prev_trading_day(d)
    r = q("""SELECT symbol, sum(volume) v, max(high) h, min(low) l, (array_agg(close ORDER BY ts_utc DESC))[1] c
             FROM es_flow_1s_databento WHERE ts_utc >= %s AND ts_utc < %s AND ts_utc < %s
             GROUP BY 1 ORDER BY v DESC LIMIT 1""", (pt(p_, 6, 30), pt(p_, 13, 0), HOLDOUT_TS))
    return None if not len(r) else dict(h=float(r["h"][0]), l=float(r["l"][0]), c=float(r["c"][0]))


def overnight(d):
    p_ = prev_trading_day(d)
    r = q("""SELECT max(high)::float h, min(low)::float l, count(*) n FROM ironbeam_es_1m_bars
             WHERE datetime >= %s AND datetime < %s""",
          (pt(p_, 15, 0).replace(tzinfo=None), pt(d, 6, 30).replace(tzinfo=None)))
    return None if not r["n"][0] else dict(h=float(r["h"][0]), l=float(r["l"][0]))


def swings15(F, d, im):
    """Zigzag at max(8, 0.15·IM); a swing is kept only if not violated for 10 min after it forms (A15.1)."""
    a, b = rth_slice(F, d)
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
            ok = not np.any(h[hi_i + 1:e_] > hi) and t[hi_i] + SW_HOLD <= ep(pt(d, 13, 0))
            out.append(dict(px=float(hi), kind="swing_high", at=int(t[hi_i]), known=max(int(t[k]), int(t[hi_i]) + SW_HOLD),
                            kept=ok))
            mode, lo, lo_i = -1, l[k], k
        elif mode <= 0 and h[k] >= lo + thr and lo_i < k:
            e_ = np.searchsorted(t, t[lo_i] + SW_HOLD, side="right")
            ok = not np.any(l[lo_i + 1:e_] < lo) and t[lo_i] + SW_HOLD <= ep(pt(d, 13, 0))
            out.append(dict(px=float(lo), kind="swing_low", at=int(t[lo_i]), known=max(int(t[k]), int(t[lo_i]) + SW_HOLD),
                            kept=ok))
            mode, hi, hi_i = 1, h[k], k
    return out


def refs_for(d, F, im, with_prior=True):
    t0 = ep(pt(d, 6, 30))
    out = []
    if with_prior and (pr := prior_rth(d)) is not None:
        out += [dict(px=pr["h"], kind="prior_high", known=t0), dict(px=pr["l"], kind="prior_low", known=t0),
                dict(px=pr["c"], kind="prior_close", known=t0)]
    if (on := overnight(d)) is not None:
        out += [dict(px=on["h"], kind="overnight_high", known=t0), dict(px=on["l"], kind="overnight_low", known=t0)]
    sw = swings15(F, d, im)
    return out + [s_ for s_ in sw if s_["kept"]], sw


def zone_for(c, side, refs, tt, e):
    """A15.1: the cluster + at most one ref, the nearest beyond its far member within e, span <= 10."""
    far = c["top"] if side == 1 else c["bot"]
    best = None
    if e > 0:
        for r in refs:
            if r["known"] >= tt:
                continue
            dist_ = (r["px"] - far) * side
            span = (r["px"] - c["bot"]) if side == 1 else (c["top"] - r["px"])
            if 0 < dist_ <= e and span <= ZONE_MAX and (best is None or dist_ < best[0]):
                best = (dist_, r)
    if best is None:
        return dict(arm=far, kind="gex", lo=c["bot"], hi=c["top"])
    r = best[1]
    return dict(arm=r["px"], kind=r["kind"], lo=min(c["bot"], r["px"]), hi=max(c["top"], r["px"]))


def zone_tests(F, d, cls, refs, im, e):
    rows = []
    for ci, c in enumerate(cls):
        for side in SIDES:
            far = c["top"] if side == 1 else c["bot"]
            cands = {far} | {r["px"] for r in refs if 0 < (r["px"] - far) * side <= e and
                             ((r["px"] - c["bot"]) if side == 1 else (c["top"] - r["px"])) <= ZONE_MAX}
            for X in cands:
                for j, s_ in tests_for(F, d, X, im):
                    if s_ != side:
                        continue
                    tt = int(F["t"][j])
                    z = zone_for(c, side, refs, tt, e)
                    if z["arm"] != X:
                        continue
                    rows.append(dict(d=d, ci=ci, L=X, side=side, type=c["type"], mixed=c["mixed"], n=c["n"],
                                     members=c["members"], net=c["net"], mag=c["mag"],
                                     arm_mag=c["top_mag"] if side == 1 else c["bot_mag"], arm_kind=z["kind"],
                                     zlo=z["lo"], zhi=z["hi"], span=z["hi"] - z["lo"], j=j, t=tt))
    return rows


REFS, SWALL = {}, {}
for d in USABLE:
    REFS[d], SWALL[d] = refs_for(d, FLOW[d], IM[d])


# ----------------------------------------------------------------------------------- detector (A15)
def attempt3(F, d, start, r, O, P, W, tol, rmin, m0=None):
    """One attempt (short space). Range per A15.3/A15.4; edges only from a developed range (A15.10)."""
    _, b = rth_slice(F, d)
    t = F["t"]
    side = r["side"]
    H, Lo, _, sg = space(F, side)
    Ls = sg * r["L"]
    zlo_s = Ls - r["span"] - O                  # far side of the two-sided zone in short space
    seg = H[start:b].copy()
    if m0 is not None:
        seg[0] = max(seg[0], m0)
    M = np.maximum.accumulate(seg)
    brk = np.nonzero(M > Ls + O)[0]
    cf = np.nonzero((Lo[start:b] <= M - P) & (H[start:b] < M))[0]
    kb = int(brk[0]) if len(brk) else 10**9
    kc = int(cf[0]) if len(cf) else 10**9
    if kb == 10**9 and kc == 10**9:
        return "none", b - 1, {}
    if kb <= kc:
        return "broken", start + kb, {}
    c = start + kc
    hi_i = start + int(np.argmax(H[start:c + 1]))
    wend = min(int(np.searchsorted(t, t[c] + W * 60, side="right")), b)
    reb = max(REB_PTS, REB_IM * IM[d])
    rl = np.minimum.accumulate(Lo[hi_i:b])
    rb_ = np.nonzero(H[c + 1:wend] >= rl[c + 1 - hi_i:wend - hi_i] + reb)[0]
    det = dict(c=c, hi_i=hi_i, H=float(M[kc]), wend=wend)
    if not len(rb_):
        return "norange", wend - 1, det
    fix = c + 1 + int(rb_[0])
    h0 = hi_i                                   # PL = the pullback low after the confirmed extreme
    PL = float(Lo[h0:fix + 1].min())
    pl_i = h0 + int(np.argmin(Lo[h0:fix + 1]))
    Hx = float(H[h0:fix + 1].max())             # a new high before PL is fixed extends H
    hi_i = h0 + int(np.argmax(H[h0:fix + 1]))
    det.update(H=Hx, hi_i=hi_i, PL=PL, pl_i=pl_i, pl_fix=fix)
    if Hx > Ls + O:
        return "broken", fix, det
    if PL < zlo_s:
        return "outzone", fix, det
    thr = max(INV_PTS, INV_FRAC * (Hx - PL))
    iv = np.nonzero((H[fix + 1:b] >= Hx + thr) | (Lo[fix + 1:b] <= PL - thr))[0]
    inv = fix + 1 + int(iv[0]) if len(iv) else b
    det.update(thr=thr, inv=inv, life_s=int((t[inv] if inv < b else ep(pt(d, 13, 0))) - t[fix]), life_cens=inv >= b)
    dev_i = int(np.searchsorted(t, t[fix] + rmin * 60))
    det["dev_i"] = dev_i
    if inv <= dev_i or dev_i >= wend:
        return "shortrange", min(inv, wend) - 1, det
    lim = min(inv, wend)

    def first(mask, a0):
        k = np.nonzero(mask)[0]
        return a0 + int(k[0]) if len(k) else None
    det["top_i"] = first(H[dev_i:lim] >= Hx - tol, dev_i)
    det["bot_i"] = first(Lo[dev_i:lim] <= PL + tol, dev_i)
    det["fade_i"] = first(H[dev_i:lim] >= Hx - tol + TICK, dev_i)
    det["bfade_i"] = first(Lo[dev_i:lim] <= PL + tol - TICK, dev_i)
    det["brk_i"] = first(H[dev_i:lim] >= Hx + 1.0, dev_i)
    det["bbrk_i"] = first(Lo[dev_i:lim] <= PL - 1.0, dev_i)
    # A15.2 fail / break share from price paths, per touched edge
    for edge, ti in (("top", det["top_i"]), ("bot", det["bot_i"])):
        if ti is None:
            det[f"fb_{edge}"] = None
            continue
        if edge == "top":
            fl, bk = first(Lo[ti:b] <= Hx - P, ti), first(H[ti:b] >= Hx + 1.0, ti)
        else:
            fl, bk = first(H[ti:b] >= PL + P, ti), first(Lo[ti:b] <= PL - 1.0, ti)
        det[f"fb_{edge}"] = ("none" if fl is None and bk is None else "break" if fl is None or
                             (bk is not None and bk <= fl) else "fail")
    end = det["fade_i"] if det["fade_i"] is not None else lim - 1
    return "developed", end, det


def episode3(F, d, r, O, P, W, tol, rmin, max_att):
    _, b = rth_slice(F, d)
    H, _, _, sg = space(F, r["side"])
    Ls = sg * r["L"]
    out, start, m0 = [], r["j"], None
    for k in range(max_att):
        st, end, det = attempt3(F, d, start, r, O, P, W, tol, rmin, m0)
        out.append(dict(att=k + 1, status=st, start=start, **det))
        if st != "developed" or det.get("fade_i") is None or k + 1 == max_att:
            return out, end
        fi = det["fade_i"]
        hit = np.nonzero(H[fi:b] >= det["H"] + STOP_C)[0]        # A9.5′: top fade stopped (S-x 2)?
        if not len(hit):
            return out, b - 1
        si = fi + int(hit[0])
        out[-1]["stop_i"] = si
        if H[si] > Ls + O:
            return out, si
        start, m0 = si, float(H[si])
    return out, start


def run3(tests, cfg, O_of_, rmin, max_att, F_by_day=None):
    """Sequential episodes per (session, cluster, side); stop after the first episode with a developed range."""
    op, P, W, tol = cfg
    F_by_day = F_by_day or FLOW
    eps, by = [], {}
    for r in tests:
        by.setdefault((r["d"], r["ci"], r["side"]), []).append(r)
    for (d, ci, side), rs in by.items():
        F, end = F_by_day[d], -1
        for r in sorted(rs, key=lambda x: x["j"]):
            if r["j"] <= end:
                continue
            atts, end = episode3(F, d, r, O_of_[(r["type"], side)], P, W, tol, rmin, max_att)
            eps.append(dict(r, O=O_of_[(r["type"], side)], atts=atts))
            if any(a["status"] == "developed" for a in atts):
                break
    return eps


def edge_rows(eps, F_by_day=None):
    """One row per touched edge of a developed range (no features); gap rule + A9.6 de-dup on the fade fill."""
    F_by_day = F_by_day or FLOW
    rows = []
    for e in eps:
        F = F_by_day[e["d"]]
        for a in e["atts"]:
            if a["status"] != "developed":
                continue
            for edge in ("top", "bot"):
                ti = a.get(f"{edge}_i")
                if ti is None:
                    continue
                fi = a["fade_i"] if edge == "top" else a["bfade_i"]
                bi = a["brk_i"] if edge == "top" else a["bbrk_i"]
                fade_real = -e["side"] if edge == "top" else e["side"]
                rows.append(dict(e=e, a=a, d=e["d"], type=e["type"], side=e["side"], edge=edge, att=a["att"],
                                 arm_mag=e["arm_mag"], fade_i=fi, brk_i=bi, fb=a[f"fb_{edge}"],
                                 gap=max_gap(F, e["d"], e["t"], int(F["t"][ti])),
                                 key=(e["d"], fade_real, int(F["t"][fi])) if fi is not None else None))
    best = {}
    for k, x in enumerate(rows):
        if x["key"] is not None and (x["key"] not in best or x["arm_mag"] > rows[best[x["key"]]]["arm_mag"]):
            best[x["key"]] = k
    for k, x in enumerate(rows):
        x["dup"] = x["key"] is not None and best[x["key"]] != k
        x["keep"] = x["gap"] <= GAP_MAX and not x["dup"]
        x["cell"] = f"{x['type']}/{'below' if x['side'] == 1 else 'above'}/{x['edge']}"
    return rows


ECELLS = [f"{ty}/{ap}/{ed}" for ty in TYPES for ap in ("below", "above") for ed in ("top", "bot")]


def tally3(eps):
    rows = [x for x in edge_rows(eps) if x["att"] == 1]
    a1 = [e["atts"][0] for e in eps]
    st = pd.Series([a["status"] for a in a1], dtype=object)
    out = dict(episodes=len(eps), **{k: int((st == k).sum()) for k in
                                     ("none", "broken", "norange", "outzone", "shortrange", "developed")})
    out["cells"] = {c: dict(edge_events=sum(x["keep"] and x["cell"] == c for x in rows),
                            fade_fills=sum(x["keep"] and x["cell"] == c and x["fade_i"] is not None for x in rows),
                            break_fills=sum(x["gap"] <= GAP_MAX and x["cell"] == c and x["brk_i"] is not None
                                            for x in rows),
                            dup=sum(x["dup"] and x["cell"] == c for x in rows)) for c in ECELLS}
    return out


# --------------------------------------------------------------------------------------- tests per e
ZT = {(dd, e): sum((zone_tests(FLOW[d], d, CL[dd][d], REFS[d], IM[d], e) for d in USABLE), [])
      for dd in DS for e in E_GRID}
ZTP = {(dd, e): sum((zone_tests(FLOW[d], d, CLP[dd][d], REFS[d], IM[d], e) for d in USABLE), [])
       for dd in DS for e in E_GRID}

# Q3 counts: (d, e, R_min) at the A1 centre; A1 grid x R_min at the centre d, e
GRES3 = {}
for dd in DS:
    for e in E_GRID:
        for rm in R_MINS:
            GRES3[(dd, e, rm, CENTRE)] = dict(real=tally3(run3(ZT[(dd, e)], CENTRE, O_of(dd, 80), rm, 1)),
                                              placebo=tally3(run3(ZTP[(dd, e)], CENTRE, O_of(dd, 80), rm, 1)))
GRID = list(itertools.product(O_PCTS, PULLS, WINS, TOLS))
for rm in R_MINS:
    for cfg in GRID:
        if (D_C, E_C, rm, cfg) not in GRES3:
            GRES3[(D_C, E_C, rm, cfg)] = dict(real=tally3(run3(ZT[(D_C, E_C)], cfg, O_of(D_C, cfg[0]), rm, 1)),
                                              placebo=tally3(run3(ZTP[(D_C, E_C)], cfg, O_of(D_C, cfg[0]), rm, 1)))


def kept_cell(k, c, arm="real"):
    return GRES3[k][arm]["cells"][c]["fade_fills"]


Q3 = {}
Q3["by_d_e_rmin"] = {f"d{dd} e{e} R{rm}": {arm: {c: v["fade_fills"] for c, v in GRES3[(dd, e, rm, CENTRE)][arm]["cells"].items()}
                                           for arm in ("real", "placebo")}
                     for dd in DS for e in E_GRID for rm in R_MINS}
Q3["centre_detail"] = {f"R{rm}": GRES3[(D_C, E_C, rm, CENTRE)] for rm in R_MINS}
Q3["ge40"] = {f"R{rm}": {c: sum(kept_cell((D_C, E_C, rm, g), c) >= 40 for g in GRID) for c in ECELLS} for rm in R_MINS}
Q3["max"] = {f"R{rm}": {c: max(kept_cell((D_C, E_C, rm, g), c) for g in GRID) for c in ECELLS} for rm in R_MINS}
Q3["pooled_type_ge40"] = {f"R{rm}": {f"{ty}/{ed}": sum(kept_cell((D_C, E_C, rm, g), f"{ty}/below/{ed}") +
                                                       kept_cell((D_C, E_C, rm, g), f"{ty}/above/{ed}") >= 40
                                                       for g in GRID)
                                     for ty in TYPES for ed in ("top", "bot")} for rm in R_MINS}
tots = {k: sum(v["real"]["cells"][c]["fade_fills"] for c in ECELLS) for k, v in GRES3.items() if k[0] == D_C and k[1] == E_C}
kmin, kmax = min(tots, key=tots.get), max(tots, key=tots.get)
Q3["fewest"] = dict(cfg=str(kmin), real={c: kept_cell(kmin, c) for c in ECELLS})
Q3["most"] = dict(cfg=str(kmax), real={c: kept_cell(kmax, c) for c in ECELLS})
Q3["tests"] = {f"d{dd} e{e}": dict(real=len(ZT[(dd, e)]), placebo=len(ZTP[(dd, e)]),
                                   real_ref_armed=sum(r["arm_kind"] != "gex" for r in ZT[(dd, e)]))
               for dd in DS for e in E_GRID}

# ------------------------------------------------------------------------- centre run (internal attempts)
EC = run3(ZT[(D_C, E_C)], CENTRE, O_of(D_C, 80), R_C, 3)
ECP = run3(ZTP[(D_C, E_C)], CENTRE, O_of(D_C, 80), R_C, 3)


# ------------------------------------------------------------------------------- features at edge events
def flow_sum(F, t0, t1):
    a, b = np.searchsorted(F["t"], t0), np.searchsorted(F["t"], t1)
    return int(F["buy"][a:b].sum() - F["sell"][a:b].sum()), int(F["vol"][a:b].sum()), a, b


def close_at(F, tt):
    k = np.searchsorted(F["t"], tt, side="right") - 1
    return float(F["c"][max(k, 0)])


def skew_at(d, tt, key):
    sk = SKEW.get(d)
    if sk is None:
        return np.nan, np.nan
    tt = max(tt, ep(pt(d, 7, 0)))
    k = np.searchsorted(sk["t"], tt, side="right") - 1
    if k < 0 or tt - sk["t"][k] > 300:
        if tt < sk["t"][0] and sk["t"][0] - tt <= 300:
            k = 0
        else:
            return np.nan, np.nan
    return float(sk[key][k]), float(sk["spot"][k])


def skew_slope(d, t0, t1, key, resid):
    sk = SKEW.get(d)
    if sk is None:
        return np.nan
    t0 = max(t0, ep(pt(d, 7, 0)))
    m = (sk["t"] >= t0) & (sk["t"] <= t1)
    if m.sum() < 3:
        return np.nan
    y = sk[key][m].copy()
    if resid:
        b = BETA.get(d, {}).get(key, np.nan)
        if not np.isfinite(b):
            return np.nan
        y = y - b * np.log(sk["spot"][m])
    return float(np.polyfit((sk["t"][m] - t0) / 600.0, y, 1)[0])


def edge_features(F, d, e, a, edge):
    """Features of one edge event (edge 'top' / 'bot', short space). Positive = toward the touched edge."""
    side, L = e["side"], e["L"]
    _, _, C, sg = space(F, side)
    t = F["t"]
    ti = a["top_i"] if edge == "top" else a["bot_i"]
    te = int(t[ti])
    u = 1 if edge == "top" else -1             # short-space direction toward the edge
    real_dir = sg * u                          # real direction toward the edge
    x = dict(edge=edge, real_edge="top" if real_dir == 1 else "bottom", t=te)
    net, vol, i0, _ = flow_sum(F, te - PUSH_W, te)
    x["P"] = real_dir * net
    mv = u * (C[ti - 1] - C[max(i0 - 1, 0)]) if ti > 0 else np.nan
    x["A_rel"] = (mv / (x["P"] / 1000.0) / BD[d]) if (x["P"] > 0 and d in BD) else np.nan
    set_i = a["hi_i"] if edge == "top" else a["pl_i"]
    ns, _, _, _ = flow_sum(F, int(t[set_i]) - PUSH_W, int(t[set_i]))
    pset = real_dir * ns
    x["Q"] = x["P"] / pset if pset > 0 else np.nan
    # pushes
    Hs, Los, _, _ = space(F, side)
    edge_px = a["H"] if edge == "top" else a["PL"]
    pushes, inpush = [], False
    for k in range(set_i, ti + 1):
        near = (Hs[k] >= edge_px - a["tol"]) if edge == "top" else (Los[k] <= edge_px + a["tol"])
        far = (Hs[k] < edge_px - a["tol"] - RETREAT) if edge == "top" else (Los[k] > edge_px + a["tol"] + RETREAT)
        if near and not inpush:
            n_, _, _, _ = flow_sum(F, int(t[k]) - PUSH_W, int(t[k]))
            pushes.append(real_dir * n_)
            inpush = True
        elif far:
            inpush = False
    x["n_push"] = len(pushes)
    x["PD"] = (float(np.polyfit(np.arange(len(pushes)), pushes, 1)[0]) / np.mean(pushes)
               if len(pushes) >= 2 and np.mean(pushes) > 0 else np.nan)
    tc = int(t[a["pl_fix"]])                 # A15.3: range-wide features from range formation
    x["range_s"] = te - tc
    th = (te - tc) / 3.0
    n1, v1, _, _ = flow_sum(F, tc, tc + th)
    n3, v3, _, _ = flow_sum(F, te - th, te)
    x["LE"] = (real_dir * n3 / v3 - real_dir * n1 / v1) if v1 > 0 and v3 > 0 else np.nan
    nr, vr, ra, rb = flow_sum(F, tc, te)
    netr = real_dir * nr
    mvr = u * (C[ti - 1] - C[a["pl_fix"]]) if ti > a["pl_fix"] else 0.0
    x["RA"] = (mvr / (netr / 1000.0) / BD[d]) if (netr > 0 and d in BD) else np.nan
    x["DV"] = netr / vr if vr > 0 else np.nan
    seg = sg * F["c"][ra:rb]
    x["SS_share"] = float(np.mean(seg > sg * L)) if rb > ra else np.nan
    w = F["vol"][ra:rb]
    x["SS_vwap"] = (float(np.sum(seg * w) / w.sum()) - sg * L) if rb > ra and w.sum() else np.nan
    # skew
    pre_t = te < ep(pt(d, 7, 0))
    jt = e["j"]
    tt = int(t[jt])
    w0 = np.searchsorted(t, tt - 3600)
    st_i = w0 + int(np.argmin(Los[w0:jt + 1]))     # opposite extreme of the move in (short-space low)
    ts_ = int(t[st_i])
    dP = sg * (Hs[jt] - Los[st_i])                 # real price change over the move in
    x["move_in_pts"] = dP
    for key in ("ps", "cs"):
        s0, sp0 = skew_at(d, ts_, key)
        s1, sp1 = skew_at(d, tt, key)
        ds = s1 - s0 if not pre_t else np.nan
        x[f"d{key}"] = ds
        b = BETA.get(d, {}).get(key, np.nan)
        x[f"d{key}_res"] = ds - b * np.log(sp1 / sp0) if np.isfinite(ds) and np.isfinite(b) else np.nan
        for sfx, val in (("", ds), ("_res", x[f"d{key}_res"])):
            x[f"div_{key}{sfx}"] = ("na" if not np.isfinite(val) else
                                    "bull" if (dP < 0 and val < 0) else "bear" if (dP > 0 and val > 0) else "none")
        x[f"d{key}_per10"] = ds / abs(dP) * 10 if np.isfinite(ds) and dP != 0 else np.nan
        sgn = 1 if x["real_edge"] == "top" else -1
        for rs in (False, True):
            sl = skew_slope(d, tc, te, key, rs) if not pre_t else np.nan
            x[f"trend_{key}{'_res' if rs else ''}"] = sgn * sl if np.isfinite(sl) else np.nan
        mv_slope = ds / ((tt - ts_) / 600.0) if np.isfinite(ds) and tt > ts_ else np.nan
        x[f"trend_chg_{key}"] = x[f"trend_{key}"] - sgn * mv_slope if np.isfinite(mv_slope) else np.nan
    # big move toward the level (short-space up move into the test)
    w15 = np.searchsorted(t, tt - 900)
    big15 = Hs[jt] - Los[w15:jt + 1].min()
    big60 = Hs[jt] - Los[w0:jt + 1].min()
    x["big_move"] = bool((d in SIG5 and big15 >= 5 * SIG5[d]) or big60 >= 0.5 * IM.get(d, np.inf))  # A15.5
    x["big15_sig"] = big15 / SIG5[d] if d in SIG5 else np.nan
    return x




def edge_events3(eps, F_by_day=None):
    F_by_day = F_by_day or FLOW
    out = []
    for x in edge_rows(eps, F_by_day):
        e, a = x["e"], x["a"]
        F = F_by_day[e["d"]]
        a["tol"] = CENTRE[3]
        f = edge_features(F, e["d"], e, a, x["edge"])
        ti = a[f"{x['edge']}_i"]
        sg = e["side"]
        rng_lo, rng_hi = sorted([sg * a["H"], sg * a["PL"]])
        side_of = "above" if rng_lo > e["zhi"] else ("below" if rng_hi < e["zlo"] else "straddle")
        out.append(dict(d=e["d"], ci=e["ci"], L=e["L"], arm_kind=e["arm_kind"], zlo=e["zlo"], zhi=e["zhi"],
                        zone_lo=e["zlo"] - e["O"], zone_hi=e["zhi"] + e["O"],
                        side=sg, type=e["type"], mixed=e["mixed"], n_members=e["n"], members=e["members"],
                        net=e["net"], mag=e["mag"], O=e["O"], att=a["att"], gap=x["gap"], dup=x["dup"],
                        keep=x["keep"], cell=x["cell"], H=a["H"], PL=a["PL"], test_t=e["t"],
                        conf_t=int(F["t"][a["c"]]), form_t=int(F["t"][a["pl_fix"]]),
                        held_s=int(F["t"][ti] - F["t"][a["pl_fix"]]), range_h=a["H"] - a["PL"],
                        rng_lo=rng_lo, rng_hi=rng_hi, ss_side=side_of, fb=x["fb"],
                        fade_i=x["fade_i"], brk_i=x["brk_i"], **f))
    return pd.DataFrame(out)


EV = edge_events3(EC)
EVP = edge_events3(ECP)

TERC_VARS = ["A_rel", "P", "LE", "RA", "DV", "trend_ps", "dcs_res"]
CUTS = {}
for cell, g in EV[EV["keep"] & (EV["att"] == 1)].groupby("cell"):
    for v in TERC_VARS:
        xx = g[v].dropna()
        CUTS[(cell, v)] = np.percentile(xx, [100 / 3, 200 / 3]) if len(xx) >= 6 else None


def terc(E, v):
    out = np.full(len(E), -1)
    for k, (cell, xx) in enumerate(zip(E["cell"], E[v])):
        ct = CUTS.get((cell, v))
        if ct is not None and np.isfinite(xx):
            out[k] = 0 if xx <= ct[0] else (1 if xx <= ct[1] else 2)
    return out


def rules(E):
    if not len(E):
        return E
    for v in TERC_VARS:
        E[f"t_{v}"] = terc(E, v)
    fr = {}
    fr["FR1"] = np.where(E["t_A_rel"] == 0, "fade", np.where(E["t_A_rel"] == 2, "break", "none"))
    fr["FR2"] = np.where(E["Q"] < 0.75, "fade", np.where(E["Q"] > 1.25, "break", "none"))
    fr["FR3"] = np.where(E["P"] <= 0, "fade", np.where(E["t_P"] == 2, "break", "none"))
    fr["FR4"] = np.where((fr["FR1"] == fr["FR2"]) & (fr["FR1"] != "none"), fr["FR1"], "none")
    fr["FR5"] = np.where((E["PD"] < 0) & (E["t_LE"] == 0), "fade", "none")
    fr["FR6"] = np.where((E["P"] > 0) & (E["t_RA"] == 0), "fade",
                         np.where((E["t_RA"] == 2) & (E["LE"] >= 0), "break", "none"))
    fr["FR7"] = np.where((E["SS_share"] >= 0.6) & (E["DV"] > 0), "break",
                         np.where((E["SS_share"] <= 0.4) & (E["DV"] <= 0), "fade", "none"))
    fr["FR8"] = np.where(E["DV"] <= 0, "fade", np.where((E["t_DV"] == 2) & (E["PD"] >= 0), "break", "none"))
    fr["FR10"] = np.where(((E["real_edge"] == "bottom") & (E["div_ps"] == "bull")) |
                          ((E["real_edge"] == "top") & (E["div_ps"] == "bear")), "fade", "none")
    fr["FR11"] = np.where(((fr["FR5"] == "fade") | (fr["FR6"] == "fade")) & (fr["FR10"] == "fade"), "fade", "none")
    fr["FR12"] = np.where(E["t_trend_ps"] == 2, "fade", np.where(E["t_trend_ps"] == 0, "break", "none"))
    fr["STALL"] = np.where(E["SS_share"] >= 0.6, "break", np.where(E["SS_share"] <= 0.4, "fade", "none"))
    for k, v in fr.items():
        E[k] = v
    b = (E["FR5"] == "fade") | (E["FR6"] == "fade")
    pnb = (E["type"] == "put") & (E["side"] == -1) & (E["edge"] == "top")
    psd = (E["type"] == "call") & (E["side"] == 1) & (E["edge"] == "top")
    E["c_a"], E["c_b"] = E["big_move"], b
    E["c_c_pnb"], E["c_c_psd"] = E["div_ps"] == "bull", E["div_ps"] == "bear"
    E["H_PNB"] = pnb & E["c_a"] & b & E["c_c_pnb"]
    E["H_PSD"] = psd & E["c_a"] & b & E["c_c_psd"]
    E["H_CNB"] = psd & E["c_a"] & b & (E["t_dcs_res"] == 0)
    E["is_pnb_cell"], E["is_psd_cell"] = pnb, psd
    return E


EV, EVP = rules(EV), rules(EVP)
FRS = ["FR1", "FR2", "FR3", "FR4", "FR5", "FR6", "FR7", "FR8", "FR10", "FR11", "FR12", "STALL"]
FEATS = ["P", "A_rel", "Q", "PD", "LE", "RA", "DV", "SS_share", "SS_vwap", "n_push", "range_s", "held_s",
         "range_h", "dps", "dps_res", "dcs", "dcs_res", "trend_ps", "trend_ps_res", "trend_cs", "trend_chg_ps",
         "big15_sig"]


def dist(x):
    x = pd.Series(x, dtype=float).dropna()
    if not len(x):
        return dict(n=0)
    return dict(n=int(len(x)), p10=round(float(x.quantile(.1)), 3), p25=round(float(x.quantile(.25)), 3),
                p50=round(float(x.median()), 3), p75=round(float(x.quantile(.75)), 3),
                p90=round(float(x.quantile(.9)), 3))


A1E = EV[EV["keep"] & (EV["att"] == 1)]
A1P = EVP[EVP["keep"] & (EVP["att"] == 1)] if len(EVP) else EVP
ADD = {}
ADD["edge_counts"] = {c: dict(real=int((A1E["cell"] == c).sum()), placebo=int((A1P["cell"] == c).sum()))
                      for c in ECELLS}
ADD["ss_side"] = {c: g["ss_side"].value_counts().to_dict() for c, g in A1E.groupby("cell")}
ADD["ss_side_all"] = A1E["ss_side"].value_counts().to_dict()
ADD["feature_dist_all"] = {f: dist(A1E[f]) for f in FEATS}
ADD["feature_dist"] = {c: {f: dist(g[f]) for f in FEATS} for c, g in A1E.groupby("cell")}
ADD["rule_decisions_all"] = {r: A1E[r].value_counts().to_dict() for r in FRS}
ADD["rule_decisions"] = {c: {r: g[r].value_counts().to_dict() for r in FRS} for c, g in A1E.groupby("cell")}
ADD["push_counts"] = A1E["n_push"].value_counts().sort_index().to_dict()
ADD["na_counts"] = {f: int(A1E[f].isna().sum()) for f in FEATS}
ADD["skew_coverage"] = dict(events=len(A1E), ps=int(A1E["dps"].notna().sum()), trend_ps=int(A1E["trend_ps"].notna().sum()))
ADD["divergence_by_level_type"] = {f"{ty}/{re}": A1E[(A1E["type"] == ty) & (A1E["real_edge"] == re)]["div_ps"]
                                   .value_counts().to_dict() for ty in TYPES for re in ("top", "bottom")}


def named(E):
    out = {}
    if not len(E):
        return out
    for nm, cell, cc in (("H_PNB", "is_pnb_cell", "c_c_pnb"), ("H_PSD", "is_psd_cell", "c_c_psd")):
        g = E[E[cell]]
        out[nm] = dict(cell_events=len(g), full=int(g[nm].sum()), a=int(g["c_a"].sum()), b=int(g["c_b"].sum()),
                       c=int(g[cc].sum()), ab=int((g["c_a"] & g["c_b"]).sum()), ac=int((g["c_a"] & g[cc]).sum()),
                       bc=int((g["c_b"] & g[cc]).sum()), skew_na=int(g["dps"].isna().sum()))
    out["H_CNB_readout"] = int(E["H_CNB"].sum())
    return out


ADD["named_real"] = named(A1E)
ADD["named_placebo"] = named(A1P)

# A15.2 fail / break shares (price paths; developed edges; attempt 1; centre)
ADD["fail_break"] = {c: dict(real=A1E[A1E["cell"] == c]["fb"].value_counts().to_dict(),
                             placebo=A1P[A1P["cell"] == c]["fb"].value_counts().to_dict() if len(A1P) else {})
                     for c in ECELLS}
for nm, E in (("real", A1E), ("placebo", A1P)):
    v = E["fb"].value_counts()
    ADD[f"fail_break_all_{nm}"] = dict(fail=int(v.get("fail", 0)), brk=int(v.get("break", 0)),
                                       none=int(v.get("none", 0)),
                                       fail_share=round(float(v.get("fail", 0) / max(v.get("fail", 0) + v.get("break", 0), 1)), 3))

# range-duration distribution (every PL-fixed range, attempt 1, centre params, real and placebo)
for nm, eps in (("real", EC), ("placebo", ECP)):
    life = [a["life_s"] / 60 for e in eps for a in e["atts"][:1] if a.get("life_s") is not None]
    cens = [a["life_cens"] for e in eps for a in e["atts"][:1] if a.get("life_s") is not None]
    life = np.array(life)
    ADD[f"range_life_min_{nm}"] = dict(**dist(life), censored=int(np.sum(cens)),
                                       ge10=round(float(np.mean(life >= 10)), 3), ge20=round(float(np.mean(life >= 20)), 3),
                                       ge30=round(float(np.mean(life >= 30)), 3))
    st = pd.Series([e["atts"][0]["status"] for e in eps]).value_counts().to_dict()
    ADD[f"episode_status_{nm}"] = st

# A15.1 / A14.4 reference-level and zone counts
rc = pd.DataFrame([dict(d=d, kind=r["kind"]) for d in USABLE for r in REFS[d]])
ADD["refs_per_session"] = {k: dist(rc[rc["kind"] == k].groupby("d").size().reindex(USABLE, fill_value=0))
                           for k in sorted(rc["kind"].unique())}
ADD["swings_before_hold_filter"] = dist([len(SWALL[d]) for d in USABLE])
ADD["swings_kept"] = dist([sum(s_["kept"] for s_ in SWALL[d]) for d in USABLE])
ADD["zones"] = {}
for e in E_GRID:
    zt = ZT[(D_C, e)]
    kinds = pd.Series([r["arm_kind"] for r in zt]).value_counts().to_dict()
    ADD["zones"][f"e{e}"] = dict(tests=len(zt), armed_by=kinds,
                                 ref_armed_share=round(sum(r["arm_kind"] != "gex" for r in zt) / max(len(zt), 1), 3),
                                 span=dist([r["span"] for r in zt]))


# ----------------------------------------------------------------------------------------------- T0
def t0_rows(tests, dd):
    Oc = O_of(dd, 80)
    rows = []
    for r in tests:
        d, F = r["d"], FLOW[r["d"]]
        _, b = rth_slice(F, d)
        t, c, h, l = F["t"], F["c"], F["h"], F["l"]
        O = Oc[(r["type"], r["side"])]
        zlo, zhi = r["zlo"] - O, r["zhi"] + O
        t0 = r["t"]
        close13 = ep(pt(d, 13, 0))
        t1 = min(t0 + T0_WIN, close13)
        g = np.arange(t0, t1)
        kk = np.searchsorted(t, g, side="right") - 1
        px = c[np.clip(kk, 0, None)]
        dwell = int(np.sum((px >= zlo) & (px <= zhi)))
        esc = np.nonzero((h[r["j"]:b] >= r["L"] + ESC_IM * IM[d]) | (l[r["j"]:b] <= r["L"] - ESC_IM * IM[d]))[0]
        cap = min(ESC_CAP, (close13 - t0) / 60)
        esc_min = (t[r["j"] + int(esc[0])] - t0) / 60 if len(esc) else np.inf
        rm = min(esc_min, cap)

        def path(ta, tb):
            gg = np.arange(ta, tb + 1, 10)
            k2 = np.searchsorted(t, gg, side="right") - 1
            k2 = k2[k2 >= 0]
            return float(np.sum(np.abs(np.diff(c[k2])))) if len(k2) > 1 else np.nan
        before = path(max(t0 - 300, ep(pt(d, 6, 30))), t0)
        after = path(t0, min(t0 + 300, close13))
        lvr = np.log(after / before) if before and after and np.isfinite(before) and np.isfinite(after) else np.nan
        row = dict(d=d, type=r["type"], side=r["side"], dwell=dwell, esc_rm=rm, esc60=float(esc_min <= 60),
                   esc_cens=float(not np.isfinite(esc_min)), lvr=lvr)
        for rmin in R_MINS:
            st, _, det = attempt3(F, d, r["j"], r, O, CENTRE[1], 180, CENTRE[3], rmin)
            ok = st == "developed" and t[det["pl_fix"]] - t0 <= T0_WIN
            row[f"formed_R{rmin}"] = float(bool(ok))
        rows.append(row)
    return pd.DataFrame(rows)


W_T0 = np.random.default_rng(SEED_BOOT).multinomial(len(USABLE), np.full(len(USABLE), 1 / len(USABLE)),
                                                     size=N_BOOT).astype(float)
T0_METRICS = ["esc_rm", "esc60", "dwell", "formed_R10", "formed_R20", "formed_R30", "lvr"]


def t0_compare(R, P_):
    out = {}
    for cell in CELLS + [("all", 0)]:
        def sel(E):
            return E if cell[0] == "all" else E[(E["type"] == cell[0]) & (E["side"] == cell[1])]
        r, p = sel(R), sel(P_)
        blk = dict(n_real=len(r), n_placebo=len(p))
        for m in T0_METRICS:
            def sums(E):
                s_ = np.zeros((len(USABLE), 2))
                x = E[m].astype(float)
                ok = x.notna()
                np.add.at(s_, ([SIDX[d] for d in E["d"][ok]], 0), x[ok].to_numpy())
                np.add.at(s_, ([SIDX[d] for d in E["d"][ok]], 1), 1)
                return s_
            sr, sp = sums(r), sums(p)
            br, bp = W_T0 @ sr, W_T0 @ sp
            with np.errstate(invalid="ignore", divide="ignore"):
                bd = br[:, 0] / br[:, 1] - bp[:, 0] / bp[:, 1]
            rv, pv = sr[:, 0].sum() / max(sr[:, 1].sum(), 1), sp[:, 0].sum() / max(sp[:, 1].sum(), 1)
            lo, hi = np.nanpercentile(bd, [2.5, 97.5])
            blk[m] = dict(real=round(float(rv), 4), placebo=round(float(pv), 4),
                          diff=[round(float(rv - pv), 4), round(float(lo), 4), round(float(hi), 4)])
        out[f"{cell[0]}/{'below' if cell[1] == 1 else ('above' if cell[1] == -1 else '')}"] = blk
    return out


T0 = {}
for dd, e in ((0, 0), (D_C, E_C)):
    R0, P0 = t0_rows(ZT[(dd, e)], dd), t0_rows(ZTP[(dd, e)], dd)
    T0[f"d{dd} e{e}"] = t0_compare(R0, P0)
    T0[f"d{dd} e{e}"]["esc_rm_dist_real"] = dist(R0["esc_rm"])
    T0[f"d{dd} e{e}"]["esc_rm_dist_placebo"] = dist(P0["esc_rm"])
    T0[f"d{dd} e{e}"]["censored"] = dict(real=int(R0["esc_cens"].sum()), placebo=int(P0["esc_cens"].sum()))


# ----------------------------------------------------------------------------------- 2026-10-23 (A9.7)
sym_i = q("""SELECT symbol, sum(volume) v FROM es_flow_1s_databento WHERE ts_utc >= %s AND ts_utc < %s
             GROUP BY 1 ORDER BY v DESC""", (pt(ILLUS, 6, 30), pt(ILLUS, 13, 0)))
F_I = load_flow("es_flow_1s_databento", ILLUS, sym_i["symbol"].iloc[0], pt(ILLUS, 13, 0))
FLOW[ILLUS] = F_I
IM_I = float(load_im(ILLUS, ILLUS)[ILLUS])
IM[ILLUS] = IM_I
PT_I = load_ptgt([ILLUS])[ILLUS]
O_I = rth_open(F_I, ILLUS)
OPEN[ILLUS] = O_I
SKEW.update(load_skew([ILLUS]))
LV_I, _ = select_levels(load_oig(ILLUS, ILLUS), PT_I, O_I, IM_I)
REFS_I, SW_I = refs_for(ILLUS, F_I, IM_I, with_prior=False)
ILL = dict(symbol=sym_i["symbol"].iloc[0], im=IM_I, p_target=PT_I, open=O_I, top6=[x["L"] for x in LV_I],
           refs=[dict(px=r["px"], kind=r["kind"], known=hms(r["known"])) for r in REFS_I])
ILL_EV = {}


def ill_table(eps, F):
    rows = []
    for e in sorted(eps, key=lambda x: x["t"]):
        sg = e["side"]
        for a in e["atts"]:
            row = dict(arm=e["L"], arm_kind=e["arm_kind"], members=e["members"], approach="below" if sg == 1 else "above",
                       O_c=e["O"], att=a["att"], test_t=hms(e["t"]), status=a["status"])
            if "pl_fix" in a:
                row.update(extreme=sg * a["H"], extreme_t=hms(F["t"][a["hi_i"]]), PL=sg * a["PL"],
                           formed_t=hms(F["t"][a["pl_fix"]]),
                           life_min=round(a["life_s"] / 60, 1) if "life_s" in a else None)
            for nm in ("top_i", "bot_i", "fade_i", "bfade_i", "brk_i", "bbrk_i"):
                if a.get(nm) is not None:
                    row[nm.replace("_i", "_t")] = hms(F["t"][a[nm]])
            if a.get("fade_i") is not None:
                row["top_entry"] = sg * (a["H"] - CENTRE[3])
            rows.append(row)
    return rows


for dd in (D_C, 10):
    cls = clusters(LV_I, dd)
    tests = [r for r in zone_tests(F_I, ILLUS, cls, REFS_I, IM_I, E_C) if any(abs(m - 7687) <= 6 for m in r["members"])]
    eps = run3(tests, CENTRE, O_of(dd, 80), R_C, 3)
    ev = edge_events3(eps)
    ev = rules(ev) if len(ev) else ev
    ILL_EV[dd] = ev
    ILL[f"d{dd}_table"] = ill_table(eps, F_I)


# ---------------------------------------------------------------------------------------------- Q5 A/A
def hold_to_close(F, d, side, fi, entry_s, stop_s, edge):
    """Short space: top fade = short at entry_s, stop above; bottom fade = long, stop below."""
    _, b = rth_slice(F, d)
    Hs, Los, Cs, _ = space(F, side)
    if edge == "top":
        hit = np.nonzero(Hs[fi:b] >= stop_s)[0]
        return (entry_s - (stop_s + TICK) if len(hit) else entry_s - Cs[b - 1]) - COST_PTS, \
            (fi + int(hit[0]) if len(hit) else b - 1)
    hit = np.nonzero(Los[fi:b] <= stop_s)[0]
    return ((stop_s - TICK) - entry_s if len(hit) else Cs[b - 1] - entry_s) - COST_PTS, \
        (fi + int(hit[0]) if len(hit) else b - 1)


pl = []
for _, x in EVP[EVP["keep"] & (EVP["att"] <= ATT_C)].iterrows():
    if pd.isna(x["fade_i"]):
        continue
    F = FLOW[x["d"]]
    fi = int(x["fade_i"])
    if x["edge"] == "top":
        pnl, k = hold_to_close(F, x["d"], x["side"], fi, x["H"] - CENTRE[3], x["H"] + STOP_C, "top")
    else:
        pnl, k = hold_to_close(F, x["d"], x["side"], fi, x["PL"] + CENTRE[3], x["PL"] - STOP_C, "bot")
    if max_gap(F, x["d"], int(F["t"][fi]), int(F["t"][k])) > GAP_MAX:
        continue
    pl.append(dict(d=x["d"], key=(x["d"], x["ci"]), pnl=pnl))
PLd = pd.DataFrame(pl)
keys = sorted(set(PLd["key"]))
rs_, rb_ = np.random.default_rng(SEED_DRAW), np.random.default_rng(SEED_BOOT)
nS = len(USABLE)
sx = np.array([SIDX[d] for d in PLd["d"]])
stats, hws, excl = [], [], 0
for _ in range(N_SPLIT):
    perm = rs_.permutation(len(keys))
    h1 = {keys[k] for k in perm[: len(keys) // 2]}
    g = np.array([k not in h1 for k in PLd["key"]]).astype(int)
    c = np.zeros((nS, 2, 2))
    np.add.at(c, (sx, g, 0), PLd["pnl"].to_numpy())
    np.add.at(c, (sx, g, 1), 1)
    tt = c.sum(0)
    stats.append(tt[0, 0] / tt[0, 1] - tt[1, 0] / tt[1, 1])
    Wb = rb_.multinomial(nS, np.full(nS, 1 / nS), size=N_BOOT_SPLIT).astype(float)
    bc = np.einsum("bs,sij->bij", Wb, c)
    with np.errstate(invalid="ignore", divide="ignore"):
        bst = bc[:, 0, 0] / bc[:, 0, 1] - bc[:, 1, 0] / bc[:, 1, 1]
    lo, hi = np.nanpercentile(bst, [2.5, 97.5])
    hws.append((hi - lo) / 2)
    excl += int(lo > 0 or hi < 0)
stats = np.array(stats)
Q5 = dict(placebo_trades=len(PLd), pseudo_clusters=len(keys), median=round(float(np.median(stats)), 3),
          mean=round(float(stats.mean()), 3), sd=round(float(stats.std(ddof=1)), 3),
          pcts={p: round(float(np.percentile(stats, p)), 2) for p in (5, 25, 75, 95)},
          ci_halfwidth_median=round(float(np.median(hws)), 3),
          sd_over_hw=round(float(stats.std(ddof=1) / (np.median(hws) / 1.96)), 3), excl_share=excl / N_SPLIT)
Q5["PASS"] = bool(Q5["excl_share"] <= 0.10 and abs(Q5["median"]) <= 3)
del PLd, pl


# ------------------------------------------------------------------------------------- gallery round 2
def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def fnum(v, r=4):
    return None if v is None or not np.isfinite(v) else round(float(v), r)


def gal_event(x, k, F, tag):
    sg = x["side"]
    real_top = x["real_edge"] == "top"
    edge_px = sg * (x["H"] if x["edge"] == "top" else x["PL"])
    entry = edge_px - CENTRE[3] if real_top else edge_px + CENTRE[3]
    stop = edge_px + STOP_C if real_top else edge_px - STOP_C
    fi = int(x["fade_i"]) if pd.notna(x["fade_i"]) else None
    meta = dict(tag=tag, cell=x["cell"], attempt=int(x["att"]), O_c=x["O"], arming_level=x["L"],
                arming_kind=x["arm_kind"], cluster_members=x["members"], cluster_mixed=bool(x["mixed"]),
                zone=dict(core_lo=x["zlo"], core_hi=x["zhi"], lo=x["zlo"] - x["O"], hi=x["zhi"] + x["O"]),
                range=dict(top=x["rng_hi"], bottom=x["rng_lo"], height=fnum(x["range_h"], 2),
                           formed_ts_pt=iso_pt(x["form_t"]), held_before_edge_min=round(x["held_s"] / 60, 1)),
                ss_side=x["ss_side"], edge=x["real_edge"], edge_touch_ts_pt=iso_pt(x["t"]),
                flow={f: fnum(x[f]) for f in ("P", "A_rel", "Q", "PD", "LE", "RA", "DV", "SS_share", "SS_vwap")},
                skew=dict(div_ps=x["div_ps"], div_ps_res=x["div_ps_res"], div_cs=x["div_cs"], dps=fnum(x["dps"], 5),
                          trend_ps=fnum(x["trend_ps"], 5), trend_ps_res=fnum(x["trend_ps_res"], 5),
                          trend_chg_ps=fnum(x["trend_chg_ps"], 5)),
                big_move=bool(x["big_move"]), rules={r: x[r] for r in FRS},
                named=dict(H_PNB=bool(x["H_PNB"]), H_PSD=bool(x["H_PSD"]), H_CNB=bool(x["H_CNB"])))
    return dict(id=f"CR-CB:stage1-r2:{k}", session_date=str(x["d"]), cell=x["cell"],
                direction="short" if real_top else "long", level_price=x["L"],
                level_net_bn=round(x["net"] / 1e9, 2), level_gross_bn=round(x["mag"] / 1e9, 2),
                entry_ts_pt=iso_pt(F["t"][fi]) if fi is not None else None, entry_price=entry, stop_price=stop,
                extreme_price=edge_px, test_ts_pt=iso_pt(x["test_t"]), confirm_ts_pt=iso_pt(x["conf_t"]),
                retest_ts_pt=iso_pt(x["t"]), meta=meta, outcome=None)


pool = EV[EV["keep"]]
rng_g = np.random.default_rng(SEED_GAL)
cells_g = sorted(pool["cell"].unique())
avail = {c: int((pool["cell"] == c).sum()) for c in cells_g}
alloc, left = {c: 0 for c in cells_g}, 40
while left > 0 and any(alloc[c] < avail[c] for c in cells_g):
    for c in sorted(cells_g, key=lambda c: -avail[c]):
        if left and alloc[c] < avail[c]:
            alloc[c] += 1
            left -= 1
picked = []
for c in cells_g:
    idx = pool.index[pool["cell"] == c].to_numpy()
    picked += list(rng_g.choice(idx, alloc[c], replace=False))
gal, k = [], 0
for i in picked:
    k += 1
    gal.append(gal_event(EV.loc[i], k, FLOW[EV.loc[i, "d"]], "random"))
for i in EV.index[(EV["H_PNB"] | EV["H_PSD"]) & EV["keep"]]:
    if i in picked:
        continue
    k += 1
    gal.append(gal_event(EV.loc[i], k, FLOW[EV.loc[i, "d"]], "H-PNB" if EV.loc[i, "H_PNB"] else "H-PSD"))
for dd, ev in ILL_EV.items():
    for _, x in (ev.iterrows() if len(ev) else []):
        k += 1
        gal.append(gal_event(x, k, F_I, f"2026-10-23 d{dd}"))
GALLERY.parent.mkdir(parents=True, exist_ok=True)
GALLERY.write_text(json.dumps(dict(cr_id="CR-CB", set_name="cr-cb:round-2",
                                   generated_at=datetime.now(UTC).isoformat(), events=clean(gal)), indent=1))

# --------------------------------------------------------------------------------------------- output
lv_n = pd.Series([len(LEVELS[d]) for d in USABLE])
Q1 = dict(funnel=funnel, n=len(USABLE), first=str(USABLE[0]), last=str(USABLE[-1]),
          source=pd.Series(SRC).reindex(USABLE).value_counts().to_dict(),
          levels=dict(total=int(lv_n.sum()), put=sum(x["net"] < 0 for v in LEVELS.values() for x in v),
                      call=sum(x["net"] > 0 for v in LEVELS.values() for x in v)),
          placebo_pseudo_levels=sum(len(v) for v in PLV.values()),
          placebo_per_session=dist([len(PLV[d]) for d in USABLE]))
RES = dict(T0=T0, Q1=Q1, Q2=dict(cells=Q2, put_vs_call_d0=Q2_DIFF), Q3=Q3, ADD=ADD, ILLUS=ILL, Q5=Q5,
           gallery=dict(path=str(GALLERY.relative_to(REPO_ROOT)), events=len(gal), alloc=alloc,
                        random=sum(g_["meta"]["tag"] == "random" for g_ in gal),
                        named=sum(g_["meta"]["tag"] in ("H-PNB", "H-PSD") for g_ in gal),
                        illus=sum(g_["meta"]["tag"].startswith("2026-10-23") for g_ in gal)))
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "step0_v3.json").write_text(json.dumps(clean(RES), indent=1, default=str))
print("=== T0 ===")
print(json.dumps(clean(T0), indent=1, default=str))
print("=== rest ===")
print(json.dumps(clean({k: v for k, v in RES.items() if k not in ("T0", "ILLUS")}), indent=1, default=str))
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
for dd in (D_C, 10):
    print(f"\n=== 2026-10-23 (Databento, d {dd}, e {E_C}, R_min {R_C}, all attempts, no outcomes) ===")
    print(pd.DataFrame(ILL[f"d{dd}_table"]).to_string(index=False))
print("\nQ5 A/A:", "PASS" if Q5["PASS"] else "FAIL", Q5)
