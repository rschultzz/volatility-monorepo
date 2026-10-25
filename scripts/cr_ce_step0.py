#!/usr/bin/env python3
"""CR-CE Step 0 (Stage 1) — event counts on Databento, A/A on placebo stop-free moves, CR-CC fixture match.

READ-ONLY: SELECTs only, no API calls. No outcome at real levels is computed. Stop-free moves are computed only
for placebo events inside the A/A splits, and only the split statistic's distribution is reported.

Spec: specs/CR-CE-first-touch-fade-direction.md (L1–L11, Step 0). Event construction is CR-BW's
(specs/CR-BW-aggressor-absorption-at-levels.md: L5, L7, A1.1, A1.2, A1.4, A1.5, A2.1; code in
scripts/cr_bw_absorption.py) with the flow source swapped to `es_flow_1s_databento` (L2).

Operational choices fixed before the run (not stated in the CR):
  - connection: get_backfill_db_conn() + default_transaction_read_only = on (asserted); statement_timeout 600 s;
    Python alarm 1 h.
  - data guards: Databento flow reads are bounded to ts_utc in [2026-01-02 05:30 PT, 2026-06-05 21:00 UTC).
    No 2025 Databento row is read (R-b, CR-CB A10.2 / A10.4) and nothing after 2026-06-05 (holdout).
  - sessions: NYSE days 2026-01-02 .. 2026-06-05. The train universe is 2026-01-06 .. 2026-06-05 (L3) under the
    CR-BW exclusions: L14 (a) roll week, (b) no ES 15:45 ET bar on A, (c) roll mismatch; Databento flow present;
    RTH open; canonical bt_daily_features IM > 0; orats_oi_gamma rows; >= 5 prior baseline sessions (A1.5).
  - front symbol per session: the one symbol with RTH rows in es_flow_1s_databento (asserted unique per kept
    session; roll weeks are excluded by L14a and are the only sessions holding two contracts, CR-CA L3).
  - RTH open (A2.1): open of the first Databento row at or after 06:30:00 PT.
  - Databento-scale baseline (L2, L7, A1.5): B_d = median A_w over non-overlapping 300 s RTH windows (78 per
    session; dropped if an RTH gap > 15 s or net = 0) in the prior up-to-20 baseline sessions, >= 5 required.
    Baseline sessions = 2026 NYSE days passing L14 (a), (c) and, where A >= 2026-01-02, (b), with Databento flow.
    2026-01-02 is admitted without the L14b check (its A is 2025-12-31, not read). Since no 2025 flow may be
    read, the first sessions lack 5 prior baseline sessions; they are excluded and counted.
  - levels (A1.1) and placebo (A1.2): exactly as cr_bw_absorption.py, with P_target the ironbeam_es_1m_bars
    15:45 ET close on A (CR-BT, unchanged) and the Databento RTH open.
  - touch / arming / approach side (L5), pre-window gap rule (A1.4: drop if RTH gap > 15 s in [t − 300, t]),
    D, ΔP, A_rel (L7): exactly as cr_bw_absorption.py on Databento rows.
  - event (L2) = an armed touch with D > 0 (flow into the level), after the A1.4 drop. Every armed touch counts
    (CR-BW L5 re-arm). The first armed touch of a level in a session is flagged (`first`) and counted separately.
  - cells (L7): put-heavy = level net < 0, call-heavy = net > 0; approached from below = side −1 (fade short),
    from above = side +1 (fade long). Placebo events take the parent level's net sign.
  - A_rel terciles: cut points over the real Databento events (D > 0); counts only.
  - CR-BW Ironbeam comparison: scripts/logs/cr_bw/events_real.pkl from the main checkout (CR-BW run output).
    Only identity / predictor columns are read (d, L, net, mag, side, t, D); its outcome columns are not.
  - horizon availability (path time only): an event has horizon H if t + H <= 13:00 PT; 13:00 PT always.
  - A/A (CR-BW A1.3 design, applied to the stop-free move): placebo events (D > 0, kept) with a +60 min price.
    Stop-free move at H = s · (P(t + H) − L), s = +1 from above (fade long), −1 from below; P(x) = close of the
    last Databento row with ts <= x; missing if that row is older than 15 s (counted). Split unit = pseudo-level
    (d, L) carrying >= 1 measured event; 200 splits, default_rng(20261025) permutations, halves = first
    floor(n/2) vs the rest. Statistic = mean move (half 1) − mean move (half 2), event-weighted, in points.
    CI: session-cluster bootstrap, 2,000 resamples per split (multinomial session weights, one default_rng(20260920)
    stream across splits), 95 % percentile; both halves resampled jointly by session.
    Gate (A1.3 thresholds as written): CI excludes 0 in <= 10 % of splits AND median statistic within ±3 (points
    here). Pooled at +60 min is the gate; +30 min, 13:00 PT and the four cells at +60 min are reported.
  - fixture match: research/gallery/cc-fixture.json (session_date, level_price, meta.touch_utc only; its outcome
    fields are not read). Match = a Databento event (any D) at the same session and level; reported: same second
    yes/no, Δt to the nearest Databento touch at that level, side, D on both feeds, first-touch flag, usable
    session.

Usage: /Users/ryan/code/volatility-monorepo/apps/web/.venv/bin/python -u scripts/cr_ce_step0.py | tee scripts/logs/cr_ce/step0.log
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


def cell_table(E):
    rows = {}
    for c in CELLS + ["pooled"]:
        e = E if c == "pooled" else E[E["cell"] == c]
        end = np.array([ep(pt(d, 13, 0)) for d in e["d"]]) if len(e) else np.array([])
        rows[c] = dict(n=int(len(e)), first=int(e["first"].sum()), sessions=int(e["d"].nunique()),
                       ge100=int((e["mag"] >= STRONG).sum()),
                       terc=[int((e["terc"] == k).sum()) for k in range(3)],
                       h30=int((e["t"].to_numpy() + 1800 <= end).sum()),
                       h60=int((e["t"].to_numpy() + 3600 <= end).sum()))
    return rows


INFO["cut"] = CUT.tolist()
INFO["cells_real"] = cell_table(ER)
INFO["cells_placebo"] = cell_table(EP)
INFO["cells_real_all_touches_any_D"] = {c: int((R["cell"] == c).sum()) for c in CELLS}

# ---------------------------------------------------------------------------- CR-BW Ironbeam comparison
IB = pd.read_pickle(CRBW_EVENTS)[["d", "L", "net", "mag", "side", "t", "D"]].copy()
IB["d"] = pd.to_datetime(IB["d"]).dt.date
IB["cell"] = np.where(IB["net"] < 0, "put", "call") + np.where(IB["side"] == 1, "_above", "_below")
IBE = IB[IB["D"] > 0]
INFO["cells_ironbeam_crbw"] = {c: int((IBE["cell"] == c).sum()) for c in CELLS} | {"pooled": int(len(IBE))}
INFO["ironbeam_crbw_sessions"] = int(IB["d"].nunique())


def match(A, Bdf, tol):
    """for each row of A, the Bdf row at the same (d, L) with |Δt| <= tol (nearest), else None."""
    idx = {}
    for j, r in enumerate(Bdf.itertuples()):
        idx.setdefault((r.d, r.L), []).append((r.t, j))
    out = []
    for r in A.itertuples():
        best = None
        for tb, j in idx.get((r.d, r.L), []):
            if abs(tb - r.t) <= tol and (best is None or abs(tb - r.t) < abs(best[0] - r.t)):
                best = (tb, j)
        out.append(best[1] if best else -1)
    return np.array(out)


common = sorted(set(IB["d"]) & set(USABLE))
IBc, Rc = IB[IB["d"].isin(common)], R[R["d"].isin(common)]
ov = {}
for tol in (0, 2, 60):
    m = match(IBc, Rc, tol)
    hit = m >= 0
    both = IBc[hit].assign(D_db=Rc["D"].to_numpy()[m[hit]])
    ov[f"tol_{tol}s"] = dict(ironbeam_kept_on_common=int(len(IBc)), found_in_databento=int(hit.sum()),
                             D_pos_agree=int(((both["D"] > 0) == (both["D_db"] > 0)).sum()),
                             ib_pos_db_le0=int(((both["D"] > 0) & (both["D_db"] <= 0)).sum()),
                             ib_le0_db_pos=int(((both["D"] <= 0) & (both["D_db"] > 0)).sum()))
mr = match(Rc, IBc, 2)
ov["databento_kept_on_common"] = int(len(Rc))
ov["databento_found_in_ironbeam_2s"] = int((mr >= 0).sum())
ib_lv = {(r.d, r.L) for r in IBc.itertuples()}
db_lv = {(d, x["L"]) for d in common for x in LEVELS[d]}
ov["levels_with_touch_ironbeam_not_in_databento_levels"] = len(ib_lv - db_lv)
ov["common_sessions"] = len(common)
m0 = match(IBc, Rc, 0)
both, dd = IBc[m0 >= 0], Rc.iloc[m0[m0 >= 0]]
ov["D_ratio_databento_over_ironbeam_median_where_both_pos"] = float(np.median(
    (dd["D"].to_numpy() / both["D"].to_numpy())[(both["D"].to_numpy() > 0) & (dd["D"].to_numpy() > 0)]))
INFO["overlap"] = ov

# -------------------------------------------------------------------------------------- fixture match
fx = json.loads(FIXTURE.read_text())
FX = []
for e in fx["events"]:
    d = date.fromisoformat(e["session_date"])
    L = float(e["level_price"])
    tt = int(pd.Timestamp(e["meta"]["touch_utc"]).tz_localize("UTC").timestamp()) \
        if pd.Timestamp(e["meta"]["touch_utc"]).tzinfo is None else int(pd.Timestamp(e["meta"]["touch_utc"]).timestamp())
    row = dict(id=e["id"], d=str(d), L=L, touch_utc=e["meta"]["touch_utc"], usable=d in USABLE,
               level_in_databento_set=d in LEVELS and any(abs(x["L"] - L) < 1e-9 for x in LEVELS[d]))
    ib = IB[(IB["d"] == d) & (IB["L"] == L)]
    ibm = ib.iloc[(ib["t"] - tt).abs().argsort()[:1]] if len(ib) else ib
    row["ironbeam"] = (dict(dt_s=int(ibm["t"].iloc[0] - tt), side=int(ibm["side"].iloc[0]), D=int(ibm["D"].iloc[0]),
                            first=bool(ib["t"].min() == ibm["t"].iloc[0])) if len(ibm) else None)
    if d in FLOW and d in feat.index and pd.notna(feat[d]) and feat[d] > 0:
        # touch, side and D do not depend on B_d; outside the universe IM comes from feat and B_d may be undefined
        had_im, had_b = d in IM, d in B
        IM.setdefault(d, float(feat[d]))
        B.setdefault(d, np.nan)
        lv = [x for x in LEVELS.get(d, []) if abs(x["L"] - L) < 1e-9] or [dict(L=L, mag=np.nan, net=np.nan)]
        evs = pd.DataFrame(events_for(d, lv, "fx"))
        if not had_im:
            del IM[d]
        if not had_b:
            del B[d]
        if len(evs):
            near = evs.iloc[(evs["t"] - tt).abs().argsort()[:1]].iloc[0]
            row["databento"] = dict(same_second=bool(near["t"] == tt), dt_s=int(near["t"] - tt),
                                    side=int(near["side"]), D=int(near["D"]), first=bool(near["k"] == 0),
                                    pre_gap=int(near["pre_gap"]), n_touches_at_level=int(len(evs)),
                                    in_event_set=bool(row["usable"] and row["level_in_databento_set"]
                                                      and near["pre_gap"] <= GAP_MAX and near["side"] != 0
                                                      and near["D"] > 0))
        else:
            row["databento"] = None
    else:
        row["databento"] = None
    FX.append(row)
INFO["fixture"] = FX

# ------------------------------------------------------------------------- A/A on placebo stop-free moves


def price_at(d, x):
    f = FLOW[d]["t"]
    j = np.searchsorted(f, x, side="right") - 1
    if j < 0 or x - f[j] > GAP_MAX:
        return np.nan
    return float(FLOW[d]["c"][j])


AA_H = {"30m": 1800, "60m": 3600, "1300": None}
for key, H in AA_H.items():
    mv = []
    for x in EP.itertuples():
        end = ep(pt(x.d, 13, 0))
        tx = end - 1 if H is None else x.t + H
        if tx > end:
            mv.append(np.nan)
            continue
        p = price_at(x.d, tx)
        mv.append(x.side * (p - x.L) if np.isfinite(p) else np.nan)
    EP[f"mv_{key}"] = mv
INFO["aa_measured"] = {k: dict(n=int(EP[f"mv_{k}"].notna().sum()),
                               no_horizon=int(sum(1 for x in EP.itertuples() if AA_H[k] is not None
                                                  and x.t + AA_H[k] > ep(pt(x.d, 13, 0)))))
                       for k in AA_H}

nS = len(USABLE)
RNG_B = np.random.default_rng(SEED_BOOT)


def aa(E, col):
    e = E[E[col].notna()]
    units = sorted({(d, L) for d, L in zip(e["d"], e["L"])})
    if len(units) < 4:
        return dict(n_events=int(len(e)), n_units=len(units), pass_=None)
    uidx = {u: k for k, u in enumerate(units)}
    eu = np.array([uidx[(d, L)] for d, L in zip(e["d"], e["L"])])
    es, ev = e["sidx"].to_numpy(), e[col].to_numpy()
    rng = np.random.default_rng(SEED_SPLIT)
    stats, excl, hw = [], 0, []
    for _ in range(N_SPLIT):
        perm = rng.permutation(len(units))
        half = np.zeros(len(units), int)
        half[perm[len(units) // 2:]] = 1
        h = half[eu]
        sums, cnts = np.zeros((nS, 2)), np.zeros((nS, 2))
        np.add.at(sums, (es, h), ev)
        np.add.at(cnts, (es, h), 1)
        obs = sums[:, 0].sum() / cnts[:, 0].sum() - sums[:, 1].sum() / cnts[:, 1].sum()
        W = RNG_B.multinomial(nS, np.full(nS, 1 / nS), size=N_BOOT_SPLIT).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            bs = (W @ sums[:, 0]) / (W @ cnts[:, 0]) - (W @ sums[:, 1]) / (W @ cnts[:, 1])
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        stats.append(obs)
        hw.append((hi - lo) / 2)
        excl += int(lo > 0 or hi < 0)
    s = np.array(stats)
    med_hw = float(np.median(hw))
    return dict(n_events=int(len(e)), n_units=len(units), n_sessions=int(len(set(e["d"]))),
                median=float(np.median(s)), mean=float(s.mean()), sd=float(s.std(ddof=1)),
                p5_25_75_95=[float(v) for v in np.percentile(s, [5, 25, 75, 95])],
                min=float(s.min()), max=float(s.max()), median_halfwidth=med_hw,
                calib_ratio=float(s.std(ddof=1) / (med_hw / 1.96)),
                excl_share=excl / N_SPLIT, excl_n=excl,
                pass_=bool(excl / N_SPLIT <= 0.10 and abs(float(np.median(s))) <= 3.0))


AA = {"pooled_60m": aa(EP, "mv_60m"), "pooled_30m": aa(EP, "mv_30m"), "pooled_1300": aa(EP, "mv_1300")}
for c in CELLS:
    AA[f"{c}_60m"] = aa(EP[EP["cell"] == c], "mv_60m")
INFO["aa"] = AA
INFO["gate_aa_pass"] = AA["pooled_60m"]["pass_"]

OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "step0.json").write_text(json.dumps(INFO, indent=1, default=str))
print(json.dumps(INFO, indent=1, default=str))
print("\nA/A gate (pooled, +60 min):", "PASS" if INFO["gate_aa_pass"] else "FAIL")
