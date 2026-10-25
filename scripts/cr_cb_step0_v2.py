#!/usr/bin/env python3
"""CR-CB Step 0 (Stage 1) under A1–A13 — Databento flow. READ-ONLY: SELECTs only, no API calls.

Spec: specs/CR-CB-failed-retest-at-level.md (Framework, L1–L11, A1–A13, A9.5′, A9.8, A9.9). Re-runs Step 0 on the
full design. Order of output: T0 (A7.1/A7.5) first, then Q1–Q4 and every Step 0 addition (A3.3, A4.8, A5.5, A6.6,
A8.8, A13.5), then the A/A (Q5), then the gallery export (A9.9).

NO trade outcome is reported at real levels or for placebo setups as a set. Two internal uses of a stop, both
required by the spec and neither reported as a statistic:
  - A9.5′ re-arm needs to know whether a fade was stopped. Train counts are reported for attempt 1 only; attempts
    2–3 exist only in the gallery (attempt number in meta, as A9.9 asks) and the 2026-10-23 illustration.
  - Q5 A/A computes hold-to-close P&L on placebo setups inside the splits; only the split statistic is reported.

Reused from scripts/cr_cb_step0.py (CR-BW A2 loaders): connection/guard, sessions and L14 exclusions, A1.1 levels,
placebo (L10, 15-pt separation), max_gap, the L5 test, the O_c measurement (A9.3 reading), the 1-tick-through fill.

Operational choices fixed here before the run (not stated in the spec):
  Data
  - Flow: es_flow_1s_databento (A9.1), per session the symbol with the largest RTH [06:30, 13:00) volume. Ironbeam
    fallback only for a session with no Databento RTH rows (counted; expected 0). Window loaded [06:15, 13:00) PT.
    Train guard ts_utc < 2026-06-05 21:00 UTC; 2026-10-23 read in one separate block (A9.7, Ryan's pre-registered
    illustration; field-conventions marks rows after 2026-06-05 as holdout).
  - Universe: CR-BW A2 exclusions (L14 a–c, canonical IM, orats_oi_gamma rows) applied to Databento. 2026-03-03,
    removed in CR-BW for missing Ironbeam flow, has Databento flow and is kept if it passes the rest.
  - Scales with no lookahead and no pre-train reads (A10: 2025 is R-b, not read): σ5_d (CR-BX L4: median |5-min
    change| over non-overlapping RTH 300 s windows) and B_d (CR-BW L7 impact baseline, median of
    sign(net)·Δclose/(|net|/1000) over 300 s windows with net ≠ 0 and gap ≤ 15 s) from the prior >= 5 (up to 20)
    usable train sessions, Databento scale. The first 5 sessions have no σ5_d / B_d: their A_rel, RA and the
    3·σ5_d leg are NA (counted). Skew regression β_d (A5.4) the same way.
  Clusters (A13)
  - Real levels sorted by price, chained when consecutive gap <= d; d ∈ {0, 5, 10}. Arming level for approach from
    below = the top member, from above = the bottom member; tests run on the arming level and keep that side only.
    Type = sign of summed net; mixed = members disagree. Placebo pseudo-levels are clustered with the same rule;
    their type is the sign of the summed parent nets.
  - "Reached far level" (A13.5): per cluster with >= 2 members and per approach side, each armed test of the near
    member (the first member met: bottom from below, top from above); reached = a trade within the 2-pt band of the
    far member before price is again >= 0.10·IM back from the near member, or 13:00.
  Cells
  - cell = (type, side); edge cells add the edge (short-space top = the extreme on the level side, the L7 retest
    edge; bottom = PL). Longs run as shorts on the negated path.
  Detector (A1, A9.3–A9.6, A9.5′, A4.1–A4.2)
  - O_c per (cell, d) from all real armed tests at arming levels, A9.3 reading (as the first Step 0).
  - Confirmation: low <= M − P with the high below M (no 0.10·IM floor, A9.4). Broken: M > L + O_c first.
    H < L at confirmation -> "short of level" (no setup).
  - Range: top = H. Bottom PL = running low after H; PL is fixed at the first bar after confirmation whose high is
    >= running low + 2 pts (A5.1's 2-pt retreat). Range invalid at the first bar beyond H + stop or PL − stop
    (stop = S-x 2, the centre), or at the window end (confirmation + W). Top event = first bar with high >= H − tol;
    bottom event = first bar after PL is fixed with low <= PL + tol; at most one per boundary per range, each only
    while the range is valid.
  - Fade fills: top limit at H − tol, bottom limit at PL + tol, 1 tick through. Break fills: stop order at H + 1 /
    PL − 1 (A3.2), filled at +1 tick (fill price only; no outcome).
  - A9.5′: if the top fade is stopped at H + 2 (S-x 2) with price inside L + O_c, a new attempt starts at the stop
    bar: running max from there, new confirmation, new range. Max attempts per level per session = grid {1, 2, 3}.
    Only the top-fade stop re-arms (a stopped bottom fade means price left the level side). A level's later tests
    are used only while no top fade on it has been filled (one filled episode per level; re-arms inside it).
  - One trade per fill (A9.6/A9.8): top-fade fills with the same session, real direction and fill second are one
    trade, credited to the higher-magnitude arming level (that level's cell). Applied to counts and the gallery.
  - Centre configuration: O_c 80th, P 6, W 90, tol 1, attempts 2, d 5, stop S-x 2. Counts depend on (O, P, W, tol,
    d) for attempt 1; 243 configurations are counted.
  - Gap rule: an edge event / setup is dropped if [test, event] has an RTH flow gap > 15 s (A1.4); the A/A also
    drops on [fill, exit].
  Flow features at each edge event (centre only; positive = toward the touched edge)
  - P: net aggressor volume over [t − 120, t). A_rel = (close move toward the edge over the same window / (P/1000))
    / B_d, P > 0. Q = P / P_set, P_set = the same push measure in the 120 s before the boundary's extreme bar;
    defined when P_set > 0.
  - pushes: from the boundary's extreme bar to the touch; a push starts when price is within tol of the edge and
    ends when price retreats > tol + 2 pts from it; P per push over the 120 s before its first bar. PD = OLS slope of
    P on push index ÷ mean P, >= 2 pushes and mean P > 0, else NA.
  - Range time [confirmation, touch]. LE = delta share (net / volume) of the last third − the first third.
    RA = (move toward the edge / (net/1000)) / B_d, net > 0. DV = net / volume over the range.
    SS share = share of rows beyond the arming level in the approach direction; SS_vwap = (VWAP − level), signed.
  Skew (A5.4, A7.3, A8)
  - orats_monies_minute, ticker SPX/SPXW, expir_date = trade_date, expiry_tod = 'pm'; value at t = latest snapshot
    at or before t within 5 min. PS = vol75 − vol50, CS = vol25 − vol50. Edge touches before 07:00 PT get NA skew;
    a skew read before 07:00 uses the first snapshot >= 07:00.
  - Move in (A8.2): from the extreme opposite the approach within the 60 min before the test to the test bar.
    ΔP (real pts), ΔPS, ΔCS. Bullish = ΔP < 0 and ΔPS < 0; bearish = ΔP > 0 and ΔPS > 0. Residual = Δskew −
    β_d·Δlog(SPX `stock_price`), β_d from OLS of 5-min Δskew on Δlog spot over the prior 5–20 usable sessions.
  - Range trend (A8.3): OLS slope of PS (and residual PS) over the range minutes per 10 min, >= 3 snapshots; signed
    favourable for the fade at that edge: + at the real top edge, − at the real bottom edge. Slope change = range
    slope − move-in slope (ΔPS per 10 min), same sign.
  Big move (A6.1a): toward the level, (test price − the 15-min opposite extreme) >= 3·σ5_d, or vs the 60-min
    session extreme >= 0.5·IM_d.
  Rules: tercile cuts per edge cell on real centre events (non-NA), applied to placebo. FR1–FR8, FR10–FR12 and stall
    side (A7.2) as written; FR9 inputs reported, not fitted (A9.8). H-PNB: put cell, from above, real bottom edge
    (= the level-side top in short space), (a) big move, (b) FR5 or FR6 fade, (c) bullish raw divergence. H-PSD:
    call, from below, real top edge, (a), (b), (c) bearish raw divergence. H-CNB readout: call, from below, top,
    (a), (b), residual CS in its lowest tercile.
  T0 (A7.1): every armed test at the arming level (d = 0 and centre d), real vs placebo, per cell: (a) range formed
    (centre params, attempt 1), (b) dwell = seconds within ±2 pts of the level over [t, t + 1800) on forward-filled
    1-s closes (capped at 13:00), (c) velocity ratio = path length of 10-s sampled closes over 5 min after ÷ 5 min
    before, log ratio reported. real − placebo with session-cluster bootstrap (10 000, seed 20260920).
  A3.3 resolution: per confirmed range (attempt 1, centre): F = top fade filled first, K = break fill first,
    neither within the window.
  Q5 A/A: placebo top-fade trades at the centre (attempts <= 2), pooled, stop S-x 2, hold to the last RTH print,
    net of 0.09 pts. 200 splits by pseudo-level cluster (seed 20261025), 2,000 session-cluster resamples each
    (seed 20260920). Pass: <= 10 % of splits exclude 0 and median within ±3 pts.
  Gallery (A9.9): CR-CC L10. 40 edge events at the centre drawn with default_rng(20261024), stratified across the
    edge cells (equal allocation, remainder to the largest cells; all of a cell if it has fewer), from all attempts;
    all H-PNB and H-PSD events; all 2026-10-23 events at the 7681/7687 levels (all attempts, d 5 and 10).
    direction = the fade direction at that edge; entry/stop = the top-fade (or bottom-fade) limit and S-x 2 stop.

  A14 (added after the first v2 run, which had already computed T0; T0 and everything before A14 are unchanged):
  - Reference levels: prior session RTH [06:30, 13:00) high / low / last close from es_flow_1s_databento (front
    symbol of that session; all prior sessions are 2026-01-05 or later); overnight high / low from
    ironbeam_es_1m_bars over [prior session 15:00 PT, 06:30 PT) (Databento flow holds RTH only); swings from a
    zigzag on today's 1-s flow from 06:30 with threshold max(4, 0.05·IM_d): a swing high is confirmed (known) at the
    first bar trading thr below the running high since the last swing low, and vice versa. Each ref carries its
    type and known-from time (06:30 for prior / overnight, confirmation time for swings).
  - Zone at time t (A14.2): the A13 cluster at the centre d (5) plus every ref known before t within e of a zone
    member, chained; GEX clusters linked through refs merge. e ∈ {0, 4, 8}; e = 0 means refs at the same price only.
  - A14.4 counts: refs per session by type; zones at 13:00 (all refs known) per e, their spans and type mix;
    and, for every real armed test at a centre-d arming level, whether the zone known at the test time has a far
    edge (approach direction) that is a ref. The detector itself is not re-armed at ref edges in Step 0 (counts
    only); the gallery meta carries the e = 4 zone members known at each event's test time.
  - 2026-10-23: overnight and swing refs only. The prior session (10-22) is not read (holdout; A9.7 authorises
    10-23 only).

Usage: apps/web/.venv/bin/python -u scripts/cr_cb_step0_v2.py | tee scripts/logs/cr_cb/step0_v2.log
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
PLC_BIN, PLC_SEP, PLC_K = 5.0, 15.0, 2
O_PCTS, PULLS, WINS, TOLS, DS = (60, 80, 95), (4.0, 6.0, 10.0), (30, 90, 180), (0.5, 1.0, 2.0), (0, 5, 10)
CENTRE = (80, 6.0, 90, 1.0)
D_C, ATT_C, STOP_C = 5, 2, 2.0
BASE_N, BASE_MIN, BASE_W, PUSH_W = 20, 5, 300, 120
SEED_DRAW, SEED_BOOT, SEED_GAL, N_BOOT, N_SPLIT, N_BOOT_SPLIT = 20261025, 20260920, 20261024, 10_000, 200, 2_000
ROLL_MISMATCH_EXTRA = {date(2026, 6, 15)}
TYPES, SIDES = ("put", "call"), (1, -1)
CELLS = [(ty, s) for ty in TYPES for s in SIDES]
OUT_DIR = REPO_ROOT / "scripts" / "logs" / "cr_cb"
GALLERY = REPO_ROOT / "research" / "gallery" / "cr-cb-stage1.json"

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


# ------------------------------------------------------------------------------------ detector (A9.5′)
def attempt(F, d, start, side, L, O, P, W, tol, m0=None):
    """One attempt from bar `start` (short space). Returns (status, end idx, detail)."""
    _, b = rth_slice(F, d)
    t = F["t"]
    H, Lo, _, sg = space(F, side)
    Ls = sg * L
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
    Hx = float(M[kc])
    hi_i = start + int(np.argmax(H[start:c + 1]))
    det = dict(H=Hx, hi_i=hi_i, c=c)
    if Hx < Ls:
        return "short", c, det
    wend = min(int(np.searchsorted(t, t[c] + W * 60, side="right")), b)
    # bottom PL: running low after H, fixed at the first bar with high >= running low + RETREAT
    rl = np.minimum.accumulate(Lo[hi_i:wend])
    reb = np.nonzero(H[c + 1:wend] >= rl[c + 1 - hi_i:] + RETREAT)[0]
    pl_fix = c + 1 + int(reb[0]) if len(reb) else None
    PL = float(rl[pl_fix - hi_i]) if pl_fix is not None else None
    pl_i = (hi_i + int(np.argmin(Lo[hi_i:pl_fix + 1]))) if pl_fix is not None else None
    # range invalidation: beyond H + stop or PL − stop, or window end
    inv_top = np.nonzero(H[c + 1:wend] > Hx + STOP_C)[0]
    inv = c + 1 + int(inv_top[0]) if len(inv_top) else wend
    if pl_fix is not None:
        inv_b = np.nonzero(Lo[pl_fix + 1:wend] < PL - STOP_C)[0]
        if len(inv_b):
            inv = min(inv, pl_fix + 1 + int(inv_b[0]))
    det.update(PL=PL, pl_i=pl_i, pl_fix=pl_fix, wend=wend, inv=inv)
    # top edge (retest), fade and break fills
    top = np.nonzero(H[c + 1:wend] >= Hx - tol)[0]
    det["top_i"] = c + 1 + int(top[0]) if len(top) and c + 1 + int(top[0]) <= inv else None
    ff = np.nonzero(H[c + 1:wend] >= Hx - tol + TICK)[0]
    det["fade_i"] = c + 1 + int(ff[0]) if len(ff) else None
    kf = np.nonzero(H[c + 1:wend] >= Hx + 1.0)[0]
    det["brk_i"] = c + 1 + int(kf[0]) if len(kf) else None
    # bottom edge
    det["bot_i"] = det["bfade_i"] = det["bbrk_i"] = None
    if pl_fix is not None:
        lim = inv if det["top_i"] is None else inv
        bb = np.nonzero(Lo[pl_fix + 1:lim + 1] <= PL + tol)[0]
        if len(bb):
            det["bot_i"] = pl_fix + 1 + int(bb[0])
        bf = np.nonzero(Lo[pl_fix + 1:wend] <= PL + tol - TICK)[0]
        det["bfade_i"] = pl_fix + 1 + int(bf[0]) if len(bf) else None
        bk = np.nonzero(Lo[pl_fix + 1:wend] <= PL - 1.0)[0]
        det["bbrk_i"] = pl_fix + 1 + int(bk[0]) if len(bk) else None
    if det["fade_i"] is None:
        return "nofill", wend - 1, det
    return "filled", det["fade_i"], det


def episode(F, d, r, O, P, W, tol, max_att):
    """All attempts from a test row r. Returns (list of attempt dicts, end idx, filled flag)."""
    _, b = rth_slice(F, d)
    H, _, _, sg = space(F, r["side"])
    Ls = sg * r["L"]
    out, start, m0 = [], r["j"], None
    for k in range(max_att):
        st, end, det = attempt(F, d, start, r["side"], r["L"], O, P, W, tol, m0)
        out.append(dict(att=k + 1, status=st, start=start, **det))
        if st != "filled":
            return out, end, any(a["status"] == "filled" for a in out)
        if k + 1 == max_att:
            return out, end, True
        fi = det["fade_i"]
        stp = det["H"] + STOP_C
        hit = np.nonzero(H[fi:b] >= stp)[0]
        if not len(hit):
            return out, b - 1, True
        si = fi + int(hit[0])
        out[-1]["stop_i"] = si
        if H[si] > Ls + O:            # beyond the zone: the level failed, no re-arm
            return out, si, True
        start, m0 = si, float(H[si])
    return out, start, True


def run_detector(tests, cfg, O_of, max_att, F_by_day=None):
    """Sequential episodes per (session, cluster, side); one filled episode per arming level."""
    op, P, W, tol = cfg
    F_by_day = F_by_day or FLOW
    eps = []
    by = {}
    for r in tests:
        by.setdefault((r["d"], r["ci"], r["side"]), []).append(r)
    for (d, ci, side), rs in by.items():
        F = F_by_day[d]
        end = -1
        for r in sorted(rs, key=lambda x: x["j"]):
            if r["j"] <= end:
                continue
            O = O_of[(r["type"], side)]
            atts, end, filled = episode(F, d, r, O, P, W, tol, max_att)
            pre = max_gap(F, d, r["t"], int(F["t"][min(end, len(F["t"]) - 1)]))
            eps.append(dict(r, O=O, atts=atts, gap=pre))
            if filled:
                break
    return eps


def dedupe_fills(eps, F_by_day=None):
    """A9.6: top-fade fills sharing (session, real direction, fill second) -> credited to the higher arming mag."""
    F_by_day = F_by_day or FLOW
    best = {}
    for e in eps:
        for a in e["atts"]:
            if a["status"] == "filled":
                key = (e["d"], e["side"], int(F_by_day[e["d"]]["t"][a["fade_i"]]))
                if key not in best or e["arm_mag"] > best[key][0]["arm_mag"]:
                    best[key] = (e, a["att"])
    keep = {(id(e), att) for e, att in best.values()}
    for e in eps:
        for a in e["atts"]:
            a["dup"] = a["status"] == "filled" and (id(e), a["att"]) not in keep
    return eps


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


# ------------------------------------------------------------------------------------ Q3 grid counts
def tally(eps):
    res = {}
    for cell in CELLS:
        e = [x for x in eps if x["type"] == cell[0] and x["side"] == cell[1]]
        a1 = [x["atts"][0] for x in e]
        st = pd.Series([a["status"] for a in a1], dtype=object)
        kept = [a for x, a in zip(e, a1) if a["status"] == "filled" and x["gap"] <= GAP_MAX and not a.get("dup")]
        res[f"{cell[0]}/{'short' if cell[1] == 1 else 'long'}"] = dict(
            episodes=len(e), broken=int((st == "broken").sum()), short_of_level=int((st == "short").sum()),
            none=int((st == "none").sum()), confirmed=int(st.isin(["nofill", "filled"]).sum()),
            top_events=sum(a.get("top_i") is not None for a in a1),
            bottom_events=sum(a.get("bot_i") is not None for a in a1),
            filled=int((st == "filled").sum()), dup=sum(bool(a.get("dup")) for a in a1), kept=len(kept))
    return res


GRID = list(itertools.product(O_PCTS, PULLS, WINS, TOLS))
GRES = {}
for dd in DS:
    for cfg in GRID:
        er = dedupe_fills(run_detector(TESTS[dd], cfg, O_of(dd, cfg[0]), 1))
        epl = dedupe_fills(run_detector(TESTS_P[dd], cfg, O_of(dd, cfg[0]), 1))
        GRES[(dd, cfg)] = dict(real=tally(er), placebo=tally(epl))


def tot(key, arm="real"):
    return sum(v["kept"] for v in GRES[key][arm].values())


ALLK = sorted(GRES, key=tot)
Q3 = dict(centre={str(k): GRES[k] for k in GRES if k[1] == CENTRE},
          fewest=dict(cfg=str(ALLK[0]), **GRES[ALLK[0]]), most=dict(cfg=str(ALLK[-1]), **GRES[ALLK[-1]]))
cellnames = list(GRES[ALLK[0]]["real"].keys())
Q3["ge40"] = {f"d{dd}": {c: sum(GRES[(dd, g)]["real"][c]["kept"] >= 40 for g in GRID) for c in cellnames} for dd in DS}
Q3["max"] = {f"d{dd}": {c: max(GRES[(dd, g)]["real"][c]["kept"] for g in GRID) for c in cellnames} for dd in DS}
Q3["pooled_type_ge40"] = {f"d{dd}": {ty: sum(GRES[(dd, g)]["real"][f"{ty}/short"]["kept"] +
                                             GRES[(dd, g)]["real"][f"{ty}/long"]["kept"] >= 40 for g in GRID)
                                     for ty in TYPES} for dd in DS}
Q3["test_counts"] = {f"d{dd}": dict(real=len(TESTS[dd]), placebo=len(TESTS_P[dd])) for dd in DS}

# ------------------------------------------------------------------------ centre run with attempts (internal)
EC = dedupe_fills(run_detector(TESTS[D_C], CENTRE, O_of(D_C, 80), 3))
ECP = dedupe_fills(run_detector(TESTS_P[D_C], CENTRE, O_of(D_C, 80), 3))


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
    tc = int(t[a["c"]])
    x["range_s"] = te - tc
    th = (te - tc) / 3.0
    n1, v1, _, _ = flow_sum(F, tc, tc + th)
    n3, v3, _, _ = flow_sum(F, te - th, te)
    x["LE"] = (real_dir * n3 / v3 - real_dir * n1 / v1) if v1 > 0 and v3 > 0 else np.nan
    nr, vr, ra, rb = flow_sum(F, tc, te)
    netr = real_dir * nr
    mvr = u * (C[ti - 1] - C[a["c"]]) if ti > a["c"] else 0.0
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
    x["big_move"] = bool((d in SIG5 and big15 >= 3 * SIG5[d]) or big60 >= 0.5 * IM.get(d, np.inf))
    x["big15_sig"] = big15 / SIG5[d] if d in SIG5 else np.nan
    return x


def edge_events(eps, F_by_day=None, im=None):
    F_by_day = F_by_day or FLOW
    rows = []
    for e in eps:
        F = F_by_day[e["d"]]
        for a in e["atts"]:
            a["tol"] = CENTRE[3]
            for edge in ("top", "bot"):
                if a.get(f"{edge}_i") is None:
                    continue
                gap = max_gap(F, e["d"], e["t"], int(F["t"][a[f"{edge}_i"]]))
                x = edge_features(F, e["d"], e, a, edge)
                rows.append(dict(d=e["d"], ci=e["ci"], L=e["L"], side=e["side"], type=e["type"], mixed=e["mixed"],
                                 n_members=e["n"], members=e["members"], net=e["net"], mag=e["mag"], O=e["O"],
                                 att=a["att"], gap=gap, H=a["H"], PL=a["PL"], test_t=e["t"],
                                 conf_t=int(F["t"][a["c"]]), dup=bool(a.get("dup")) and edge == "top",
                                 fade_i=a["fade_i"] if edge == "top" else a["bfade_i"],
                                 brk_i=a["brk_i"] if edge == "top" else a["bbrk_i"], **x))
    return pd.DataFrame(rows)


EV = edge_events(EC)
EVP = edge_events(ECP)
for E in (EV, EVP):
    E["cell"] = E["type"] + "/" + np.where(E["side"] == 1, "below", "above") + "/" + E["edge"]
    E["keep"] = (E["gap"] <= GAP_MAX) & ~E["dup"]

# terciles per edge cell on real attempt-1 kept events
TERC_VARS = ["A_rel", "P", "LE", "RA", "DV", "trend_ps", "dcs_res"]
CUTS = {}
base = EV[EV["keep"] & (EV["att"] == 1)]
for cell, g in base.groupby("cell"):
    for v in TERC_VARS:
        x = g[v].dropna()
        CUTS[(cell, v)] = np.percentile(x, [100 / 3, 200 / 3]) if len(x) >= 6 else None


def terc(E, v):
    out = np.full(len(E), -1)
    for k, (cell, x) in enumerate(zip(E["cell"], E[v])):
        ct = CUTS.get((cell, v))
        if ct is not None and np.isfinite(x):
            out[k] = 0 if x <= ct[0] else (1 if x <= ct[1] else 2)
    return out


def rules(E):
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
FEATS = ["P", "A_rel", "Q", "PD", "LE", "RA", "DV", "SS_share", "SS_vwap", "n_push", "range_s", "dps", "dps_res",
         "dcs", "dcs_res", "trend_ps", "trend_ps_res", "trend_cs", "trend_chg_ps", "big15_sig"]


def dist(x):
    x = pd.Series(x).dropna()
    if not len(x):
        return dict(n=0)
    return dict(n=int(len(x)), p10=round(float(x.quantile(.1)), 3), p25=round(float(x.quantile(.25)), 3),
                p50=round(float(x.median()), 3), p75=round(float(x.quantile(.75)), 3),
                p90=round(float(x.quantile(.9)), 3))


A1E = EV[EV["keep"] & (EV["att"] == 1)]
A1P = EVP[EVP["keep"] & (EVP["att"] == 1)]
ADD = {}
ADD["edge_counts"] = {c: dict(real=int((A1E["cell"] == c).sum()), placebo=int((A1P["cell"] == c).sum()))
                      for c in sorted(set(A1E["cell"]) | set(A1P["cell"]))}
ADD["feature_dist"] = {c: {f: dist(g[f]) for f in FEATS} for c, g in A1E.groupby("cell")}
ADD["feature_dist_all"] = {f: dist(A1E[f]) for f in FEATS}
ADD["rule_decisions"] = {c: {r: g[r].value_counts().to_dict() for r in FRS} for c, g in A1E.groupby("cell")}
ADD["rule_decisions_placebo"] = {c: {r: g[r].value_counts().to_dict() for r in FRS} for c, g in A1P.groupby("cell")}
ADD["push_counts"] = A1E["n_push"].value_counts().sort_index().to_dict()
ADD["range_duration_s"] = dict(**dist(A1E["range_s"]), share_ge_30min=round(float((A1E["range_s"] >= 1800).mean()), 3))
ADD["na_counts"] = {f: int(A1E[f].isna().sum()) for f in FEATS}
ADD["skew_coverage"] = dict(events=len(A1E), ps=int(A1E["dps"].notna().sum()), ps_res=int(A1E["dps_res"].notna().sum()),
                            cs=int(A1E["dcs"].notna().sum()), trend_ps=int(A1E["trend_ps"].notna().sum()),
                            before_0700=int((A1E["t"] < np.array([ep(pt(d, 7, 0)) for d in A1E["d"]])).sum()))
ADD["divergence"] = {c: {k: g[k].value_counts().to_dict() for k in ("div_ps", "div_ps_res", "div_cs")}
                     for c, g in A1E.groupby("cell")}
ADD["divergence_by_level_type"] = {f"{ty}/{re}": A1E[(A1E["type"] == ty) & (A1E["real_edge"] == re)]["div_ps"]
                                   .value_counts().to_dict() for ty in TYPES for re in ("top", "bottom")}
ADD["ps_trend_dist"] = {c: dist(g["trend_ps"]) for c, g in A1E.groupby("cell")}


def named(E):
    out = {}
    for nm, cell, cc in (("H_PNB", "is_pnb_cell", "c_c_pnb"), ("H_PSD", "is_psd_cell", "c_c_psd")):
        g = E[E[cell]]
        out[nm] = dict(cell_events=len(g), full=int(g[nm].sum()), a=int(g["c_a"].sum()), b=int(g["c_b"].sum()),
                       c=int(g[cc].sum()), ab=int((g["c_a"] & g["c_b"]).sum()), ac=int((g["c_a"] & g[cc]).sum()),
                       bc=int((g["c_b"] & g[cc]).sum()), skew_na=int(g["dps"].isna().sum()),
                       sig5_na=int(g["big15_sig"].isna().sum()))
    out["H_CNB_readout"] = int(E["H_CNB"].sum())
    return out


ADD["named_real"] = named(A1E)
ADD["named_placebo"] = named(A1P)
ADD["fr9_inputs"] = "PD, LE, RA, DV, SS_share, SS_vwap, P, A_rel, Q, dps_res, dcs_res: reported above; not fitted (A9.8)"

# A3.3 resolution share (attempt 1, centre, confirmed ranges)
res = {}
for cell in CELLS:
    e = [x for x in EC if x["type"] == cell[0] and x["side"] == cell[1] and x["atts"][0]["status"] in ("nofill", "filled")]
    f_ = k_ = n_ = 0
    for x in e:
        a = x["atts"][0]
        fi, ki = a.get("fade_i"), a.get("brk_i")
        if fi is None and ki is None:
            n_ += 1
        elif ki is None or (fi is not None and fi <= ki):
            f_ += 1
        else:
            k_ += 1
    res[f"{cell[0]}/{'below' if cell[1] == 1 else 'above'}"] = dict(confirmed=len(e), F=f_, K=k_, neither=n_)
ADD["A3_resolution"] = res


# ------------------------------------------------------------------------------------------- A13.5
def reached_far(dd):
    out = dict(clusters=0, multi=0, spans=[], members=[], tests=0, reached=0, by_side={1: [0, 0], -1: [0, 0]})
    for d in USABLE:
        F = FLOW[d]
        _, b = rth_slice(F, d)
        for c in CL[dd][d]:
            out["clusters"] += 1
            out["members"].append(c["n"])
            if c["n"] < 2:
                continue
            out["multi"] += 1
            out["spans"].append(c["span"])
            for side, near, far in ((1, c["bot"], c["top"]), (-1, c["top"], c["bot"])):
                H, Lo, _, sg = space(F, side)
                for j, s in tests_for(F, d, near, IM[d]):
                    if s != side:
                        continue
                    back = np.nonzero(H[j:b] <= sg * near - ARM_IM * IM[d])[0]
                    e = j + int(back[0]) if len(back) else b
                    hit = bool(np.any(H[j:e] >= sg * far - BAND))
                    out["tests"] += 1
                    out["reached"] += hit
                    out["by_side"][side][0] += 1
                    out["by_side"][side][1] += hit
    sp = np.array(out.pop("spans") or [0])
    mem = pd.Series(out.pop("members")).value_counts().sort_index().to_dict()
    out.update(member_counts=mem, span=dict(median=float(np.median(sp)), p90=float(np.percentile(sp, 90)),
                                            max=float(sp.max())),
               reached_share=round(out["reached"] / out["tests"], 3) if out["tests"] else None,
               by_side={("below" if k == 1 else "above"): v for k, v in out["by_side"].items()})
    return out


ADD["clusters"] = {f"d{dd}": reached_far(dd) for dd in DS}
ADD["clusters"]["types"] = {f"d{dd}": dict(put=sum(c["type"] == "put" for d in USABLE for c in CL[dd][d]),
                                           call=sum(c["type"] == "call" for d in USABLE for c in CL[dd][d]),
                                           mixed=sum(c["mixed"] for d in USABLE for c in CL[dd][d])) for dd in DS}


# ----------------------------------------------------------------------------------------------- T0
def t0_metrics(tests, eps_lookup, F_by_day=None):
    F_by_day = F_by_day or FLOW
    rows = []
    for r in tests:
        F, d = F_by_day[r["d"]], r["d"]
        a, b = rth_slice(F, d)
        t, c = F["t"], F["c"]
        t0 = r["t"]
        t1 = min(t0 + 1800, ep(pt(d, 13, 0)))
        grid = np.arange(t0, t1)
        k = np.searchsorted(t, grid, side="right") - 1
        dwell = int(np.sum(np.abs(c[np.clip(k, 0, None)] - r["L"]) <= BAND))

        def path(ta, tb):
            g = np.arange(ta, tb + 1, 10)
            kk = np.searchsorted(t, g, side="right") - 1
            kk = kk[kk >= 0]
            return float(np.sum(np.abs(np.diff(c[kk])))) if len(kk) > 1 else np.nan
        before = path(max(t0 - 300, ep(pt(d, 6, 30))), t0)
        after = path(t0, min(t0 + 300, ep(pt(d, 13, 0))))
        vr = np.log(after / before) if before and after and np.isfinite(before) and np.isfinite(after) else np.nan
        rows.append(dict(d=d, type=r["type"], side=r["side"], dwell=dwell, lvr=vr,
                         formed=eps_lookup.get((d, r["ci"], r["side"], r["j"]), False)))
    return pd.DataFrame(rows)


def formed_lookup(dd, tests):
    """T0 (a): each test on its own (not sequential): confirmed in-zone extreme with a fixed PL, centre params."""
    look, Oc = {}, O_of(dd, 80)
    for r in tests:
        st, _, det = attempt(FLOW[r["d"]], r["d"], r["j"], r["side"], r["L"], Oc[(r["type"], r["side"])],
                             CENTRE[1], CENTRE[2], CENTRE[3])
        look[(r["d"], r["ci"], r["side"], r["j"])] = st in ("nofill", "filled") and det.get("pl_fix") is not None
    return look


W_T0 = np.random.default_rng(SEED_BOOT).multinomial(len(USABLE), np.full(len(USABLE), 1 / len(USABLE)),
                                                     size=N_BOOT).astype(float)


def t0_compare(R, P_):
    out = {}
    for cell in CELLS + [("all", 0)]:
        def sel(E):
            return E if cell[0] == "all" else E[(E["type"] == cell[0]) & (E["side"] == cell[1])]
        r, p = sel(R), sel(P_)
        blk = dict(n_real=len(r), n_placebo=len(p))
        for m in ("formed", "dwell", "lvr"):
            def sums(E):
                s = np.zeros((len(USABLE), 2))
                x = E[m].astype(float)
                ok = x.notna()
                np.add.at(s, ([SIDX[d] for d in E["d"][ok]], 0), x[ok].to_numpy())
                np.add.at(s, ([SIDX[d] for d in E["d"][ok]], 1), 1)
                return s
            sr, sp = sums(r), sums(p)
            br, bp = W_T0 @ sr, W_T0 @ sp
            with np.errstate(invalid="ignore", divide="ignore"):
                bd = br[:, 0] / br[:, 1] - bp[:, 0] / bp[:, 1]
            est = sr[:, 0].sum() / max(sr[:, 1].sum(), 1) - sp[:, 0].sum() / max(sp[:, 1].sum(), 1)
            lo, hi = np.nanpercentile(bd, [2.5, 97.5])
            blk[m] = dict(real=round(sr[:, 0].sum() / max(sr[:, 1].sum(), 1), 4),
                          placebo=round(sp[:, 0].sum() / max(sp[:, 1].sum(), 1), 4),
                          diff=[round(float(est), 4), round(float(lo), 4), round(float(hi), 4)])
        out[f"{cell[0]}/{'below' if cell[1] == 1 else ('above' if cell[1] == -1 else '')}"] = blk
    return out


T0 = {}
for dd in (0, D_C):
    R0 = t0_metrics(TESTS[dd], formed_lookup(dd, TESTS[dd]))
    P0 = t0_metrics(TESTS_P[dd], formed_lookup(dd, TESTS_P[dd]))
    T0[f"d{dd}"] = t0_compare(R0, P0)
    T0[f"d{dd}"]["dwell_dist_real"] = dist(R0["dwell"])
    T0[f"d{dd}"]["dwell_dist_placebo"] = dist(P0["dwell"])


# ------------------------------------------------------------------------------------- Q4 detector table
def q4_rows(E, eps, d, F):
    rows = []
    for e in sorted([x for x in eps if x["d"] == d], key=lambda x: x["t"]):
        sg = e["side"]
        for a in e["atts"]:
            if a["status"] in ("none", "broken"):
                rows.append(dict(session=str(d), arm_level=e["L"], members=e["members"], type=e["type"],
                                 approach="below" if sg == 1 else "above", O_c=e["O"], att=a["att"],
                                 test_t=hms(e["t"]), status=a["status"]))
                continue
            row = dict(session=str(d), arm_level=e["L"], members=e["members"], type=e["type"],
                       approach="below" if sg == 1 else "above", O_c=e["O"], att=a["att"], test_t=hms(e["t"]),
                       status=a["status"], extreme=sg * a["H"], extreme_t=hms(F["t"][a["hi_i"]]),
                       confirm_t=hms(F["t"][a["c"]]),
                       PL=(sg * a["PL"]) if a.get("PL") is not None else "",
                       PL_t=hms(F["t"][a["pl_i"]]) if a.get("pl_i") is not None else "")
            for edge in ("top", "bot"):
                if a.get(f"{edge}_i") is None:
                    continue
                ev = E[(E["d"] == d) & (E["L"] == e["L"]) & (E["side"] == sg) & (E["att"] == a["att"]) &
                       (E["edge"] == edge) & (E["test_t"] == e["t"])]
                if len(ev):
                    x = ev.iloc[0]
                    row[f"{edge}_t"] = hms(x["t"])
                    row[f"{edge}_rules"] = " ".join(f"{r}:{x[r][0].upper()}" for r in FRS if x[r] != "none")
                    row[f"{edge}_flow"] = (f"P {x['P']:.0f} Q {x['Q']:.2f} DV {x['DV']:.2f} SS {x['SS_share']:.2f}"
                                           if np.isfinite(x["Q"]) else f"P {x['P']:.0f} DV {x['DV']:.2f}")
            if a["status"] == "filled":
                row["shared_fill_dup"] = bool(a.get("dup"))
                row["fade_fill_t"] = hms(F["t"][a["fade_i"]])
                row["entry"] = sg * (a["H"] - CENTRE[3])
                row["stop_Sx2"] = sg * (a["H"] + STOP_C)
            rows.append(row)
    return rows


Q4_DAYS = sorted(np.random.default_rng(SEED_DRAW).choice(np.array(USABLE, dtype=object), 5, replace=False))
Q4 = sum((q4_rows(EV, EC, d, FLOW[d]) for d in Q4_DAYS), [])
# train table: attempt 1 only (attempts >= 2 imply a stopped fade)
Q4 = [r for r in Q4 if r["att"] == 1]

# --------------------------------------------------------------------------- 2026-10-23 illustration (A9.7)
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
gaps_i = [(hms(F_I["t"][k]), int(F_I["t"][k + 1] - F_I["t"][k])) for k in np.nonzero(np.diff(F_I["t"]) > 15)[0]
          if ep(pt(ILLUS, 6, 30)) <= F_I["t"][k] < ep(pt(ILLUS, 13, 0))]
ILL = dict(symbol=sym_i["symbol"].iloc[0], im=IM_I, p_target=PT_I, open=O_I, top6=[x["L"] for x in LV_I],
           rth_rows=int(np.diff(rth_slice(F_I, ILLUS))[0]), gaps_gt15s=gaps_i, sig5_note="σ5/B_d of 10-23: NA (no "
           "prior-session scale read outside train); big-move 3σ leg and A_rel/RA are NA there")
ILL_EV = {}  # filled below
for dd in (D_C, 10):
    cls = clusters(LV_I, dd)
    tests = [r for r in cluster_tests(F_I, ILLUS, cls, IM_I) if any(abs(m - 7687) <= 6 for m in r["members"])]
    eps = dedupe_fills(run_detector(tests, CENTRE, O_of(dd, 80), 3))
    ev = edge_events(eps)
    if len(ev):
        ev["cell"] = ev["type"] + "/" + np.where(ev["side"] == 1, "below", "above") + "/" + ev["edge"]
        ev["keep"] = (ev["gap"] <= GAP_MAX) & ~ev["dup"]
        ev = rules(ev)
    ILL_EV[dd] = (eps, ev)
    ILL[f"d{dd}_table"] = q4_rows(ev, eps, ILLUS, F_I) if len(ev) else []


# ---------------------------------------------------------------------------------------------- A14
E_GRID, E_C = (0, 4, 8), 4


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
    return None if not r["n"][0] else dict(h=float(r["h"][0]), l=float(r["l"][0]), n=int(r["n"][0]))


def swings(F, d, im):
    a, b = rth_slice(F, d)
    thr = max(4.0, 0.05 * im)
    t, h, l = F["t"], F["h"], F["l"]
    out, hi, lo, hi_i, lo_i, mode = [], h[a], l[a], a, a, 0
    for k in range(a, b):
        if h[k] > hi:
            hi, hi_i = h[k], k
        if l[k] < lo:
            lo, lo_i = l[k], k
        if mode >= 0 and l[k] <= hi - thr and hi_i < k:
            out.append(dict(px=float(hi), kind="swing_high", known=int(t[k]), at=int(t[hi_i])))
            mode, lo, lo_i = -1, l[k], k
        elif mode <= 0 and h[k] >= lo + thr and lo_i < k:
            out.append(dict(px=float(lo), kind="swing_low", known=int(t[k]), at=int(t[lo_i])))
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
    return out + swings(F, d, im)


def zone_at(cls, refs, tt, e, ci):
    """Zone containing cluster ci at time tt: members (price, kind) chained within e."""
    known = [r for r in refs if r["known"] < tt]
    mem = [(m, "gex") for m in cls[ci]["members"]]
    other = [(m, "gex", k) for k, c in enumerate(cls) if k != ci for m in c["members"]]
    changed = True
    while changed:
        changed = False
        for r in known:
            if (r["px"], r["kind"]) not in mem and any(abs(r["px"] - m) <= e for m, _ in mem):
                mem.append((r["px"], r["kind"]))
                changed = True
        for m, kind, k in other:
            if (m, kind) not in mem and any(abs(m - x) <= e for x, kk in mem if kk != "gex"):
                mem.append((m, kind))
                changed = True
    return sorted(mem)


def zone_type(mem):
    kinds = {k for _, k in mem}
    out = ["gex"]
    for tag, ks in (("swing", ("swing_high", "swing_low")), ("prior", ("prior_high", "prior_low", "prior_close")),
                    ("overnight", ("overnight_high", "overnight_low"))):
        if kinds & set(ks):
            out.append(tag)
    return "+".join(out)


REFS = {d: refs_for(d, FLOW[d], IM[d]) for d in USABLE}
A14 = {}
rc = pd.DataFrame([dict(d=d, kind=r["kind"]) for d in USABLE for r in REFS[d]])
A14["refs_per_session"] = {k: dist(rc[rc["kind"] == k].groupby("d").size().reindex(USABLE, fill_value=0))
                           for k in sorted(rc["kind"].unique())}
A14["refs_missing"] = dict(prior=[str(d) for d in USABLE if not any(r["kind"] == "prior_high" for r in REFS[d])],
                           overnight=[str(d) for d in USABLE if not any(r["kind"] == "overnight_high" for r in REFS[d])])
for e in E_GRID:
    zones, spans, types, merged = 0, [], {}, 0
    for d in USABLE:
        cls, seen = CL[D_C][d], set()
        for ci in range(len(cls)):
            z = tuple(zone_at(cls, REFS[d], ep(pt(d, 13, 0)), e, ci))
            if z in seen:
                merged += 1
                continue
            seen.add(z)
            zones += 1
            px = [m for m, _ in z]
            spans.append(max(px) - min(px))
            zt = zone_type(z)
            types[zt] = types.get(zt, 0) + 1
    far_ref = any_ref = n = 0
    by_kind = {}
    for r in TESTS[D_C]:
        cls = CL[D_C][r["d"]]
        z = zone_at(cls, REFS[r["d"]], r["t"], e, r["ci"])
        far = max(z) if r["side"] == 1 else min(z)
        n += 1
        any_ref += any(k != "gex" for _, k in z)
        if far[1] != "gex":
            far_ref += 1
            by_kind[far[1]] = by_kind.get(far[1], 0) + 1
    A14[f"e{e}"] = dict(zones_at_close=zones, clusters_merged_into_other_zones=merged, span=dist(spans),
                        zone_types=types, tests=n, tests_zone_has_ref=any_ref, tests_far_edge_is_ref=far_ref,
                        far_edge_ref_share=round(far_ref / n, 3) if n else None, far_edge_ref_kind=by_kind)


def gal_zone(x, refs, cls_by_day):
    cls = cls_by_day[x["d"]]
    ci = next((k for k, c in enumerate(cls) if c["members"] == list(x["members"])), None)
    if ci is None:
        return None
    return [dict(price=m, kind=k) for m, k in zone_at(cls, refs, int(x["test_t"]), E_C, ci)]

# ---------------------------------------------------------------------------------------------- Q5 A/A
def hold_to_close(F, d, side, fi, entry_s, H):
    _, b = rth_slice(F, d)
    Hs, _, Cs, _ = space(F, side)
    stp = H + STOP_C
    hit = np.nonzero(Hs[fi:b] >= stp)[0]
    if len(hit):
        return entry_s - (stp + TICK) - COST_PTS, fi + int(hit[0])
    return entry_s - Cs[b - 1] - COST_PTS, b - 1


pl = []
for e in ECP:
    F = FLOW[e["d"]]
    for a in e["atts"][:ATT_C]:
        if a["status"] != "filled" or a.get("dup") or e["gap"] > GAP_MAX:
            continue
        pnl, k = hold_to_close(F, e["d"], e["side"], a["fade_i"], a["H"] - CENTRE[3], a["H"])
        if max_gap(F, e["d"], int(F["t"][a["fade_i"]]), int(F["t"][k])) > GAP_MAX:
            continue
        pl.append(dict(d=e["d"], key=(e["d"], e["ci"]), pnl=pnl))
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


# ------------------------------------------------------------------------------------- gallery (A9.9)
def nz(v):
    return None if v is None or (isinstance(v, float) and not np.isfinite(v)) else v


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def gal_event(x, k, F, tag, zone=None):
    sg = x["side"]
    real_top = x["real_edge"] == "top"
    direction = "short" if real_top else "long"
    edge_px = sg * (x["H"] if x["edge"] == "top" else x["PL"])
    entry = edge_px - CENTRE[3] if real_top else edge_px + CENTRE[3]
    stop = edge_px + STOP_C if real_top else edge_px - STOP_C
    meta = dict(tag=tag, cell=x["cell"], attempt=int(x["att"]), O_c=x["O"], arming_level=x["L"],
                cluster_members=x["members"], cluster_mixed=bool(x["mixed"]),
                range=dict(top=sg * x["H"] if sg == 1 else nz(sg * x["PL"]),
                           bottom=nz(sg * x["PL"]) if sg == 1 else sg * x["H"]),
                edge=x["real_edge"], edge_touch_ts_pt=iso_pt(x["t"]),
                flow={f: (None if not np.isfinite(x[f]) else round(float(x[f]), 4)) for f in
                      ("P", "A_rel", "Q", "PD", "LE", "RA", "DV", "SS_share", "SS_vwap")},
                skew=dict(div_ps=x["div_ps"], div_ps_res=x["div_ps_res"], div_cs=x["div_cs"],
                          dps=None if not np.isfinite(x["dps"]) else round(float(x["dps"]), 5),
                          trend_ps=None if not np.isfinite(x["trend_ps"]) else round(float(x["trend_ps"]), 5),
                          trend_ps_res=None if not np.isfinite(x["trend_ps_res"]) else round(float(x["trend_ps_res"]), 5),
                          trend_chg_ps=None if not np.isfinite(x["trend_chg_ps"]) else round(float(x["trend_chg_ps"]), 5)),
                big_move=bool(x["big_move"]), rules={r: x[r] for r in FRS}, zone_e4=zone,
                named=dict(H_PNB=bool(x["H_PNB"]), H_PSD=bool(x["H_PSD"]), H_CNB=bool(x["H_CNB"])))
    fi = int(x["fade_i"]) if pd.notna(x["fade_i"]) else None
    return dict(id=f"CR-CB:stage1:{k}", session_date=str(x["d"]), cell=x["cell"], direction=direction,
                level_price=x["L"], level_net_bn=round(x["net"] / 1e9, 2), level_gross_bn=round(x["mag"] / 1e9, 2),
                entry_ts_pt=iso_pt(F["t"][fi]) if fi is not None else None, entry_price=entry, stop_price=stop,
                extreme_price=edge_px, test_ts_pt=iso_pt(x["test_t"]), confirm_ts_pt=iso_pt(x["conf_t"]),
                retest_ts_pt=iso_pt(x["t"]), meta=meta, outcome=None)


pool = EV[EV["keep"]].copy()
rng_g = np.random.default_rng(SEED_GAL)
cells_g = sorted(pool["cell"].unique())
alloc = {c: 0 for c in cells_g}
left, avail = 40, {c: int((pool["cell"] == c).sum()) for c in cells_g}
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
    x = EV.loc[i]
    gal.append(gal_event(x, k, FLOW[x["d"]], "random", gal_zone(x, REFS[x["d"]], CL[D_C])))
for i in EV.index[(EV["H_PNB"] | EV["H_PSD"]) & EV["keep"]]:
    if i in picked:
        continue
    k += 1
    x = EV.loc[i]
    gal.append(gal_event(x, k, FLOW[x["d"]], "H-PNB" if x["H_PNB"] else "H-PSD", gal_zone(x, REFS[x["d"]], CL[D_C])))
REFS_I = refs_for(ILLUS, F_I, IM_I, with_prior=False)
CL_I = {dd: {ILLUS: clusters(LV_I, dd)} for dd in (D_C, 10)}
ILL["refs"] = [dict(px=r["px"], kind=r["kind"], known=hms(r["known"])) for r in REFS_I]
for dd, (_, ev) in ILL_EV.items():
    for _, x in ev.iterrows():
        k += 1
        gal.append(gal_event(x, k, F_I, f"2026-10-23 d{dd}", gal_zone(x, REFS_I, CL_I[dd])))
GALLERY.parent.mkdir(parents=True, exist_ok=True)
GALLERY.write_text(json.dumps(dict(cr_id="CR-CB", set_name="stage1-step0", generated_at=datetime.now(UTC).isoformat(),
                                   events=clean(gal)), indent=1, default=str))

# --------------------------------------------------------------------------------------------- output
lv_n = pd.Series([len(LEVELS[d]) for d in USABLE])
Q1 = dict(funnel=funnel, n=len(USABLE), first=str(USABLE[0]), last=str(USABLE[-1]),
          source=pd.Series(SRC).reindex(USABLE).value_counts().to_dict(),
          symbols=sorted({SYM_DB.get(d, "") for d in USABLE}),
          levels=dict(total=int(lv_n.sum()), median=float(lv_n.median()), min=int(lv_n.min()),
                      put=sum(x["net"] < 0 for v in LEVELS.values() for x in v),
                      call=sum(x["net"] > 0 for v in LEVELS.values() for x in v)),
          placebo=sum(len(v) for v in PLV.values()), sig5=dist(list(SIG5.values())), B_d=dist(list(BD.values())),
          no_scale_sessions=[str(d) for d in USABLE if d not in SIG5],
          skew_sessions=len([d for d in USABLE if d in SKEW]),
          beta_ps=dist([BETA[d]["ps"] for d in BETA]))
RES = dict(T0=T0, A14=A14, Q1=Q1, Q2=dict(cells=Q2, put_vs_call_d0=Q2_DIFF), Q3=Q3, ADD=ADD,
           Q4=dict(days=[str(d) for d in Q4_DAYS], table=Q4), ILLUS=ILL, Q5=Q5,
           gallery=dict(path=str(GALLERY.relative_to(REPO_ROOT)), events=len(gal),
                        random=sum(g["meta"]["tag"] == "random" for g in gal),
                        alloc=alloc, named=sum(g["meta"]["tag"] in ("H-PNB", "H-PSD") for g in gal),
                        illus=sum(g["meta"]["tag"].startswith("2026-10-23") for g in gal)))
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "step0_v2.json").write_text(json.dumps(RES, indent=1, default=str))
print("=== T0 ===")
print(json.dumps(T0, indent=1, default=str))
print("=== A14 ===")
print(json.dumps(A14, indent=1, default=str))
print("=== Q1 / Q2 / Q3 / additions / Q5 ===")
print(json.dumps({k: v for k, v in RES.items() if k not in ("T0", "A14", "Q4", "ILLUS")}, indent=1, default=str))
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
print("\n=== Q4 table (centre, attempt 1, no outcomes) ===")
print(pd.DataFrame(Q4).to_string(index=False))
for dd in (D_C, 10):
    print(f"\n=== 2026-10-23 (Databento, d {dd}, all attempts, no outcomes) ===")
    print(pd.DataFrame(ILL[f"d{dd}_table"]).to_string(index=False))
print("\nQ5 A/A:", "PASS" if Q5["PASS"] else "FAIL", Q5)
