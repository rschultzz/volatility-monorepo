#!/usr/bin/env python3
"""CR-BW Step 0 — diagnosis for the aggressor-absorption-at-levels test. READ-ONLY: SELECTs only, no API calls.

Spec: specs/CR-BW-aggressor-absorption-at-levels.md (Step 0 gate questions 1–6).

  1  flow OHLC: ironbeam_es_flow_1s 1-s OHLC aggregated to minutes vs ironbeam_es_1m_bars, 10 kept sessions
  2  symbol(s) per session, contract used, and P_target (1m bar 15:45 ET on A) vs the same contract's flow print
  3  universe and counts: sessions after each exclusion; levels; armed touch events; T1-eligible (D > 0) and
     per-tercile counts; entry-fill rate (price path only); placebo counts. Gate: >= 150 filled events per tercile
  4  baseline B_d distribution and stability; sessions without 20 prior baseline sessions
  5  worked event (random eligible event, seed 20261025): level, rows, 300 s path, D, dP, A, B_d, A_rel
  6  A/A control: two independent placebo draws through the full harness; T2-form difference and CI only

NO race is run at any real level or at the L9 placebo levels. The race harness runs only on the two A/A
placebo draws, and only their difference and CI are printed (no win rates).

Definitions fixed here before the run:
  - connection: get_backfill_db_conn() + SET default_transaction_read_only = on (role asserted);
    statement_timeout 600 s; Python wall-time alarm 3 h.
  - holdout guard: every read is bounded to session dates <= 2026-06-05 (flow ts < 2026-06-05 21:00 UTC).
  - session D: NYSE trading day in [2025-12-26, 2026-06-05]. A = prev_trading_day(D). Times America/Los_Angeles
    via zoneinfo; RTH = [06:30:00, 13:00:00) PT.
  - CR-BT / CR-BU L14 exclusions: (a) roll week = Monday of the week before the quarterly third Friday through
    that Friday; (b) no positive ES 15:45 ET bar close on A (ironbeam_es_1m_bars, es_snapshot_bar_start_utc);
    (c) roll mismatch = D a quarterly expiry Friday or D = 2026-06-15 (CR-BT A1.4).
  - flow series: ironbeam_es_flow_1s rows of the session's symbol (Q2: one symbol per session). One row per
    second with at least one trade; a "1-second bar" is such a row. Flow-complete session (data rule, Step 0):
    first RTH row <= 06:30:59, last RTH row >= 12:59:00, no gap between consecutive RTH rows > 60 s.
  - IM_d = canonical bt_daily_features implied_move_1d (CANONICAL_FEATURE_VERSION, SPX, active); null/<= 0 out.
  - RTH open O_d = open of the first flow row at or after 06:30:00 PT.
  - levels (L4): orats_oi_gamma ticker SPX, session D (holiday-stamped store dates -> next_trading_day),
    expir_date in [D, D+7]; L = place_levels(rows, P_target) unchanged, P_target = ES 15:45 ET bar close on A;
    bin = bucket_levels(L, 5); net = sum(gex_call - |gex_put|) per bin; level = bin with |net| >= 25e9 and
    |bin - O_d| <= 1.0 * IM_d. Both signs.
  - touch (L5), per level, from the first RTH bar: "far" bar = low >= L + 0.10*IM (above) or high <= L - 0.10*IM
    (below). Event = first bar with low <= L <= high after a far bar; then disarm and repeat from the next bar.
    Approach side = the side of the bar immediately before the touch (low > L above, high < L below).
    Diagnostic: crossings between consecutive bars with no print at L (not events, counted).
  - fill (L6, price path only): fade = long when approached from above, short from below. Filled if within
    [t, t + 60 s] a bar trades 1 tick through L (low <= L - 0.25 long; high >= L + 0.25 short).
  - predictor (L7): window [t - 300 s, t). D = (sell - buy) if approached from above, (buy - sell) from below.
    dP = in the approach direction, close of the last bar before t minus close of the last bar before t - 300 s
    (above: P_start - P_end). A = dP / (D / 1000) pts per 1,000 contracts, defined only when D > 0.
    Touches in the first 5 RTH minutes use pre-06:30 flow for the window (counted).
  - baseline B_d: windows [O_ts + 300k, O_ts + 300(k+1)), k = 0..77, O_ts = 06:30:00 PT. Per window
    net = buy - sell; windows with net = 0 dropped; A_w = sign(net) * dP_w / (|net| / 1000), dP_w = close of the
    last bar in the window minus close of the last bar before it. B_d = median of A_w pooled over the 20 prior
    baseline sessions (flow-complete, not roll week; sessions before D only). Fewer than 20 -> session excluded.
    A_rel = A / B_d.
  - terciles (L8): A_rel cut points (33.3 / 66.7 percentiles) over all T1-eligible real events; placebo events
    are assigned with the same cut points.
  - placebo (L9): for each usable session s a donor s' != s drawn uniformly from usable sessions; pseudo-levels
    O_s + (L_i(s') - O_s') for each real level of s', dropping any within 5 pts of a real level of s; same
    touch / fill / predictor rules (IM_s, B_s of the host session). Donor sign kept for S3.
    L9 draw: numpy default_rng(20261025). A/A draws: default_rng(SeedSequence(20261025).spawn(2)).
  - race (A/A only, L6, R = 8 ticks): long: target hit when high >= L + R + 0.25, stop when low <= L - R; short
    mirrored. From the bar after the fill bar; the fill bar itself counts a stop (not a target). Same bar both
    -> loss. Neither by 13:00 PT -> unresolved (excluded).
  - A/A statistic (T2 form): [low - high tercile win rate](draw 1) - [same](draw 2); session-cluster bootstrap
    10 000, seed 20260920, 95 % percentile. (T2 form because T1 at placebo levels is not expected to be 0:
    absorption without levels is a legitimate T1-pass / T2-fail outcome.)
  - Q1 sample: 10 sessions from the usable set, default_rng(20261025). Q5 worked event: default_rng(20261025)
    fresh instance, uniform over eligible real events.

Usage: apps/web/.venv/bin/python -u scripts/cr_bw_step0.py | tee scripts/logs/cr_bw/step0.log
"""
from __future__ import annotations

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

pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 400)
pd.set_option("display.max_columns", 40)

signal.alarm(3 * 3600)

PT, ET, UTC = ZoneInfo("America/Los_Angeles"), ZoneInfo("America/New_York"), timezone.utc
D_FROM, D_TO = date(2025, 12, 26), date(2026, 6, 5)
HOLDOUT_TS = datetime(2026, 6, 5, 21, 0, tzinfo=UTC)
TICK = 0.25
BIN, NET_MIN, WIN_IM, ARM_IM = 5.0, 25e9, 1.0, 0.10
WIN_S, FILL_S, BASE_N, BASE_W = 300, 60, 20, 300
R_TICKS = 8
PLC_MIN_SEP = 5.0
SEED, SEED_BOOT, N_BOOT = 20261025, 20260920, 10_000
ROLL_MISMATCH_EXTRA = {date(2026, 6, 15)}

CONN = get_backfill_db_conn()
assert_role_or_die(CONN)
CONN.execute("SET default_transaction_read_only = on")
CONN.execute("SET statement_timeout = '600s'")
assert CONN.execute("SELECT current_setting('default_transaction_read_only')").fetchone()[0] == "on"


def q(sql, params=None) -> pd.DataFrame:
    cur = CONN.execute(sql, params)
    return pd.DataFrame(cur.fetchall(), columns=[c.name for c in cur.description])


def hdr(s):
    print("\n" + "=" * 100 + f"\n{s}\n" + "=" * 100, flush=True)


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

# ---------------------------------------------------------------------------------------------- loads
hdr("LOADS")
sym = q("""
    SELECT trade_date, symbol, count(*) AS n, sum(volume) AS vol
    FROM ironbeam_es_flow_1s
    WHERE ts_utc >= %s AND ts_utc < %s
      AND (ts_utc AT TIME ZONE 'America/Los_Angeles')::time >= '06:30'
      AND (ts_utc AT TIME ZONE 'America/Los_Angeles')::time < '13:00'
    GROUP BY 1, 2 ORDER BY 1, 2
""", (pt(D_FROM, 0, 0), HOLDOUT_TS))
sym_by_d = {d: g.sort_values("vol", ascending=False) for d, g in sym.groupby("trade_date")}
SYM = {d: g["symbol"].iloc[0] for d, g in sym_by_d.items()}
multi = {d: list(g["symbol"]) for d, g in sym_by_d.items() if len(g) > 1}
print(f"sessions with RTH flow rows: {len(SYM)}; sessions with > 1 RTH symbol: {len(multi)} {multi}")

FLOW: dict[date, dict] = {}
for d in SESSIONS:
    if d not in SYM:
        continue
    f = q("""
        SELECT ts_utc, buy_vol, sell_vol, unknown_vol, open, high, low, close
        FROM ironbeam_es_flow_1s
        WHERE symbol = %s AND ts_utc >= %s AND ts_utc < %s AND ts_utc < %s
        ORDER BY ts_utc
    """, (SYM[d], pt(d, 6, 0), pt(d, 13, 0), HOLDOUT_TS))
    t = np.array([ep(x) for x in f["ts_utc"]], dtype=np.int64)
    FLOW[d] = dict(t=t, buy=f["buy_vol"].to_numpy(np.int64), sell=f["sell_vol"].to_numpy(np.int64),
                   unk=f["unknown_vol"].to_numpy(np.int64), o=f["open"].to_numpy(float),
                   h=f["high"].to_numpy(float), l=f["low"].to_numpy(float), c=f["close"].to_numpy(float))
print(f"flow loaded for {len(FLOW)} sessions; rows {sum(len(v['t']) for v in FLOW.values()):,}")

oig = q("""
    SELECT trade_date, expir_date, strike::float AS strike, stock_price::float AS stock_price, gex_call, gex_put
    FROM orats_oi_gamma
    WHERE ticker = 'SPX' AND trade_date BETWEEN %s AND %s
      AND expir_date >= trade_date AND expir_date <= trade_date + 10
""", (D_FROM - timedelta(days=7), D_TO))
oig["store_date"] = oig["trade_date"]
oig["d"] = [t if is_trading_day(t) else next_trading_day(t) for t in oig["trade_date"]]
remapped = sorted(set(oig.loc[oig["d"] != oig["store_date"], "store_date"]))
print(f"orats_oi_gamma holiday-stamped store dates remapped: {[str(x) for x in remapped]}")
oig = oig[(oig["d"] >= D_FROM) & (oig["d"] <= D_TO)]
oig = oig[[d <= e <= d + timedelta(days=7) for e, d in zip(oig["expir_date"], oig["d"])]]
oig["net"] = oig["gex_call"] - oig["gex_put"].abs()

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
    a = np.searchsorted(f["t"], ep(pt(d, 6, 30)))
    b = np.searchsorted(f["t"], ep(pt(d, 13, 0)))
    return a, b


def flow_complete(d):
    if d not in FLOW:
        return False, "no flow"
    f = FLOW[d]
    a, b = rth_slice(d)
    if b - a < 2:
        return False, "no RTH rows"
    t = f["t"][a:b]
    first_ok = t[0] <= ep(pt(d, 6, 30, 59))
    last_ok = t[-1] >= ep(pt(d, 12, 59))
    gap = int(np.diff(t).max())
    ok = first_ok and last_ok and gap <= 60
    return ok, f"first {datetime.fromtimestamp(t[0], PT):%H:%M:%S} last {datetime.fromtimestamp(t[-1], PT):%H:%M:%S} maxgap {gap}s"


FC = {d: flow_complete(d) for d in SESSIONS}

# ------------------------------------------------------------------------------------- baseline windows


def base_windows(d):
    """A_w over the 78 non-overlapping 300 s RTH windows (net != 0)."""
    f = FLOW[d]
    t0 = ep(pt(d, 6, 30))
    out = []
    for k in range(78):
        a = np.searchsorted(f["t"], t0 + BASE_W * k)
        b = np.searchsorted(f["t"], t0 + BASE_W * (k + 1))
        if a == 0 or b <= a:
            continue
        net = int(f["buy"][a:b].sum() - f["sell"][a:b].sum())
        if net == 0:
            continue
        dp = f["c"][b - 1] - f["c"][a - 1]
        out.append(np.sign(net) * dp / (abs(net) / 1000.0))
    return np.array(out)


BASE_ELIG = [d for d in SESSIONS if FC[d][0] and not roll_week(d)]
AW = {d: base_windows(d) for d in BASE_ELIG}
B = {}
for d in SESSIONS:
    prior = [x for x in BASE_ELIG if x < d][-BASE_N:]
    if len(prior) == BASE_N:
        B[d] = float(np.median(np.concatenate([AW[x] for x in prior])))

# ---------------------------------------------------------------------------------------- Q3 exclusions
hdr("Q3a — SESSION COUNTS (no outcomes)")
print(f"Trading sessions in [{D_FROM}, {D_TO}]: {len(SESSIONS)}")
ex_a = {d for d in SESSIONS if roll_week(d)}
ex_b = {d for d in SESSIONS if d not in PTGT}
ex_c = {d for d in SESSIONS if d in QEXP or d in ROLL_MISMATCH_EXTRA}
U = list(SESSIONS)
steps = []
for name, rule in [
    ("L3/L14a roll week", lambda d: d in ex_a),
    ("L3/L14b no ES 15:45 ET bar on A", lambda d: d in ex_b),
    ("L3/L14c roll mismatch", lambda d: d in ex_c),
    ("flow incomplete (RTH)", lambda d: not FC[d][0]),
    ("no canonical bt_daily_features row", lambda d: d not in feat.index),
    ("IM null or <= 0", lambda d: d in feat.index and not (pd.notna(feat[d]) and feat[d] > 0)),
    ("no orats_oi_gamma rows", lambda d: d not in set(oig["d"])),
    ("< 20 prior baseline sessions (B_d)", lambda d: d not in B),
]:
    rm = [d for d in U if rule(d)]
    U = [d for d in U if not rule(d)]
    steps.append((name, len(rm), len(U), rm))
print(f"\n{'sequential exclusion':40s} {'removed':>8s} {'remaining':>10s}")
print(f"{'start':40s} {'':>8s} {len(SESSIONS):>10d}")
for name, n, rem, lst in steps:
    print(f"{name:40s} {n:>8d} {rem:>10d}   {[str(x) for x in lst] if n <= 30 else ''}")
print("flow-incomplete detail:", {str(d): FC[d][1] for d in SESSIONS if not FC[d][0]})
USABLE = U
print(f"\nUSABLE SESSIONS: {len(USABLE)} ({USABLE[0]} .. {USABLE[-1]})")

IM = {d: float(feat[d]) for d in USABLE}
OPEN, OPEN_I = {}, {}
for d in USABLE:
    a, _ = rth_slice(d)
    OPEN[d], OPEN_I[d] = float(FLOW[d]["o"][a]), a

# ------------------------------------------------------------------------------------------ Q2 symbols
hdr("Q2 — SYMBOL PER SESSION")
print("RTH symbol by month (sessions):")
sm = pd.DataFrame([(d.strftime("%Y-%m"), SYM.get(d, "—")) for d in SESSIONS], columns=["m", "sym"])
print(sm.groupby(["m", "sym"]).size().unstack(fill_value=0).to_string())
print(f"usable sessions with > 1 RTH symbol: {[str(d) for d in USABLE if d in multi]}")
print("symbol switches across all sessions:",
      [(str(a), SYM[a], str(b), SYM[b]) for a, b in zip(SESSIONS, SESSIONS[1:]) if a in SYM and b in SYM and SYM[a] != SYM[b]])
# P_target (1m bar 15:45 ET on A) vs the D-symbol flow print in that minute on A
rows = []
for d in USABLE:
    a_day = prev_trading_day(d)
    s = datetime.combine(a_day, time(15, 45), ET).astimezone(UTC)
    f = q("""SELECT close FROM ironbeam_es_flow_1s WHERE symbol = %s AND ts_utc >= %s AND ts_utc < %s
             ORDER BY ts_utc DESC LIMIT 1""", (SYM[d], s, s + timedelta(minutes=1)))
    rows.append(dict(d=d, ptgt=PTGT[d], flow=float(f["close"].iloc[0]) if len(f) else np.nan))
PX = pd.DataFrame(rows)
PX["diff"] = (PX["ptgt"] - PX["flow"]).abs()
print(f"P_target vs same-symbol flow close in the 15:45 ET minute on A: n {PX['diff'].notna().sum()}, "
      f"no flow print {PX['diff'].isna().sum()}, median |diff| {PX['diff'].median():.2f}, p90 "
      f"{PX['diff'].quantile(.9):.2f}, max {PX['diff'].max():.2f}")
print(PX.sort_values("diff", ascending=False).head(5).to_string(index=False))

# ------------------------------------------------------------------------------------------ Q1 OHLC
hdr("Q1 — FLOW 1-s OHLC vs ironbeam_es_1m_bars (10 usable sessions, seed 20261025)")
q1 = sorted(np.random.default_rng(SEED).choice(np.array(USABLE, dtype=object), size=10, replace=False))
res = []
for d in q1:
    f = FLOW[d]
    a, b = rth_slice(d)
    df = pd.DataFrame(dict(t=f["t"][a:b], o=f["o"][a:b], h=f["h"][a:b], l=f["l"][a:b], c=f["c"][a:b]))
    df["m"] = df["t"] // 60 * 60
    m = df.groupby("m").agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"))
    bars = q("""SELECT datetime, open::float o, high::float h, low::float l, close::float c FROM ironbeam_es_1m_bars
                WHERE datetime >= %s AND datetime < %s""",
             (pt(d, 6, 30).replace(tzinfo=None), pt(d, 13, 0).replace(tzinfo=None)))
    bars["m"] = [int(pd.Timestamp(x).tz_localize("UTC").timestamp()) for x in bars["datetime"]]
    j = m.join(bars.set_index("m")[["o", "h", "l", "c"]], rsuffix="_b", how="inner")
    for k in "ohlc":
        j[f"d{k}"] = (j[k] - j[f"{k}_b"]).abs()
    j["d"] = d
    res.append(j)
    print(f"{d}: minutes flow {len(m)}, bars {len(bars)}, joined {len(j)}; median |dH| {j['dh'].median():.2f} "
          f"|dL| {j['dl'].median():.2f} |dC| {j['dc'].median():.2f}; max {j[['dh', 'dl', 'dc']].max().max():.2f}")
J = pd.concat(res)
for k in "ohlc":
    print(f"  {k.upper()}: median |diff| {J[f'd{k}'].median():.3f}  p90 {J[f'd{k}'].quantile(.9):.3f}  p99 "
          f"{J[f'd{k}'].quantile(.99):.3f}  max {J[f'd{k}'].max():.2f}  exact {(J[f'd{k}'] == 0).mean():.1%}")
q1_med = max(J["dh"].median(), J["dl"].median(), J["dc"].median())
print(f"GATE Q1 (median of H/L/C |diff| <= 0.25): {'PASS' if q1_med <= 0.25 else 'FAIL — STOP'}  (max of the three "
      f"medians {q1_med:.3f})")

# --------------------------------------------------------------------------------------------- levels
LEVELS, LVROWS = {}, {}
for d, g in oig[oig["d"].isin(USABLE)].groupby("d"):
    g = place_levels(g, PTGT[d])
    g["bin"] = bucket_levels(g["level_corrected"].to_numpy(), BIN)
    bins = g.groupby("bin")["net"].sum()
    o, im = OPEN[d], IM[d]
    sel = bins[(bins.abs() >= NET_MIN) & ((bins.index - o).__abs__() <= WIN_IM * im)]
    LEVELS[d] = [(float(b), float(n)) for b, n in sel.items()]
    LVROWS[d] = g

# --------------------------------------------------------------------------------------------- events


def events_for(d, levels, tag):
    """levels: list of (L, net). Returns event dicts (touch, approach, fill, predictor). No outcomes."""
    f = FLOW[d]
    a, b = rth_slice(d)
    t, h, l, c = f["t"], f["h"], f["l"], f["c"]
    im = IM[d]
    arm = ARM_IM * im
    out, ncross = [], 0
    for L, net in levels:
        hh, ll = h[a:b], l[a:b]
        far = np.nonzero((ll >= L + arm) | (hh <= L - arm))[0]
        cont = np.nonzero((ll <= L) & (L <= hh))[0]
        # crossings with no print at L (diagnostic)
        above = ll > L
        below = hh < L
        ncross += int(np.sum((above[:-1] & below[1:]) | (below[:-1] & above[1:])))
        start = 0
        while True:
            fi = np.searchsorted(far, start)
            if fi >= len(far):
                break
            ci = np.searchsorted(cont, far[fi] + 1)
            if ci >= len(cont):
                break
            i = int(cont[ci]) + a                       # absolute index of the touch bar
            side = 1 if l[i - 1] > L else (-1 if h[i - 1] < L else 0)   # +1 from above, -1 from below
            ti = int(t[i])
            # fill (price path only)
            j1 = min(np.searchsorted(t, ti + FILL_S, side="right"), b)
            if side == 1:
                fj = np.nonzero(l[i:j1] <= L - TICK)[0]
            else:
                fj = np.nonzero(h[i:j1] >= L + TICK)[0]
            fill_i = i + int(fj[0]) if len(fj) else -1
            ev = dict(d=d, tag=tag, L=L, net=net, i=i, t=ti, side=side, fill_i=fill_i,
                      early=ti < ep(pt(d, 6, 35)))
            for w in (WIN_S, 60, 900):
                w0 = np.searchsorted(t, ti - w)
                buy, sell = int(f["buy"][w0:i].sum()), int(f["sell"][w0:i].sum())
                D = (sell - buy) if side == 1 else (buy - sell)
                p_end = c[i - 1]
                p_start = c[w0 - 1] if w0 > 0 else np.nan
                dP = (p_start - p_end) if side == 1 else (p_end - p_start)
                A = dP / (D / 1000.0) if D > 0 else np.nan
                sfx = "" if w == WIN_S else f"_{w}"
                ev.update({f"D{sfx}": D, f"dP{sfx}": dP, f"A{sfx}": A, f"A_rel{sfx}": A / B[d] if D > 0 else np.nan})
            out.append(ev)
            start = cont[ci] + 1
    return out, ncross


hdr("Q3b — LEVELS, EVENTS, ELIGIBILITY, FILLS (no outcomes)")
nlev = pd.Series({d: len(LEVELS.get(d, [])) for d in USABLE})
print(f"levels per session: median {nlev.median():.0f}, IQR {nlev.quantile(.25):.0f}–{nlev.quantile(.75):.0f}, "
      f"min {nlev.min()}, max {nlev.max()}; sessions with 0 levels {int((nlev == 0).sum())}; total {nlev.sum()}")
allL = pd.DataFrame([dict(d=d, L=L, net=n, dist=L - OPEN[d], z=(L - OPEN[d]) / IM[d]) for d in USABLE
                     for L, n in LEVELS.get(d, [])])
print(f"level sign: negative-net {int((allL['net'] < 0).sum())}, positive-net {int((allL['net'] > 0).sum())}; "
      f"|net| median {allL['net'].abs().median() / 1e9:.1f} $bn; |dist| median {allL['dist'].abs().median():.1f} pts; "
      f"IM median {np.median(list(IM.values())):.1f}")
EVR, ncr = [], 0
for d in USABLE:
    e, n = events_for(d, LEVELS.get(d, []), "real")
    EVR += e
    ncr += n
R = pd.DataFrame(EVR)
per = R.groupby("d").size().reindex(USABLE, fill_value=0)
print(f"armed touch events: {len(R)}; per session median {per.median():.0f}, IQR {per.quantile(.25):.0f}–"
      f"{per.quantile(.75):.0f}, max {per.max()}; sessions with >= 1 event {int((per > 0).sum())}")
print(f"approach side: from above {int((R['side'] == 1).sum())}, from below {int((R['side'] == -1).sum())}, "
      f"undetermined {int((R['side'] == 0).sum())}; crossings with no print at L (not events): {ncr}")
print(f"events whose 300 s window starts before 06:30 PT (touch < 06:35): {int(R['early'].sum())}")
elig = R[(R["D"] > 0) & (R["side"] != 0)].copy()
print(f"T1-eligible (D > 0): {len(elig)}; D <= 0 (S6): {int(((R['D'] <= 0) & (R['side'] != 0)).sum())}")
cut = np.percentile(elig["A_rel"], [100 / 3, 200 / 3])
print(f"A_rel tercile cut points: {cut.round(4)}; A_rel quantiles "
      f"{np.percentile(elig['A_rel'], [1, 5, 25, 50, 75, 95, 99]).round(3)}")


def tercile(x):
    return np.where(x <= cut[0], 0, np.where(x <= cut[1], 1, 2))


elig["terc"] = tercile(elig["A_rel"].to_numpy())
elig["filled"] = elig["fill_i"] >= 0
tt = elig.groupby("terc").agg(n=("filled", "size"), filled=("filled", "sum"))
tt["fill_rate"] = (tt["filled"] / tt["n"]).round(3)
tt.index = ["low (absorption)", "mid", "high (continuation)"]
print(tt.to_string())
print(f"entry-fill rate, all events {(R['fill_i'] >= 0).mean():.3f}; eligible {elig['filled'].mean():.3f}")
print(f"S3 split (eligible, filled): negative-net {int((elig['filled'] & (elig['net'] < 0)).sum())}, positive-net "
      f"{int((elig['filled'] & (elig['net'] > 0)).sum())}; S4 before/after 10:00 PT "
      f"{int((elig['filled'] & (elig['t'] < [ep(pt(x, 10, 0)) for x in elig['d']])).sum())} / "
      f"{int((elig['filled'] & (elig['t'] >= [ep(pt(x, 10, 0)) for x in elig['d']])).sum())}")
gate3 = int(tt["filled"].min())
print(f"GATE Q3 (>= 150 filled — could-resolve — eligible events in every tercile): "
      f"{'PASS' if gate3 >= 150 else 'FAIL — STOP'} (min {gate3})")
if gate3 < 150:
    sys.exit("STOP: gate Q3 failed")


def placebo_levels(rng):
    donors = {}
    lv = {}
    for s in USABLE:
        others = [x for x in USABLE if x != s]
        dn = others[int(rng.integers(len(others)))]
        donors[s] = dn
        real = np.array([x for x, _ in LEVELS.get(s, [])])
        pl = []
        for L, n in LEVELS.get(dn, []):
            P = OPEN[s] + (L - OPEN[dn])
            if len(real) and np.min(np.abs(real - P)) < PLC_MIN_SEP:
                continue
            pl.append((P, n))
        lv[s] = pl
    return lv, donors


PLV0, _ = placebo_levels(np.random.default_rng(SEED))
EVP = []
for d in USABLE:
    EVP += events_for(d, PLV0[d], "placebo")[0]
P0 = pd.DataFrame(EVP)
pe = P0[(P0["D"] > 0) & (P0["side"] != 0)].copy()
pe["terc"] = tercile(pe["A_rel"].to_numpy())
pe["filled"] = pe["fill_i"] >= 0
print(f"\nPLACEBO (L9 draw): pseudo-levels {sum(len(v) for v in PLV0.values())}; events {len(P0)}; eligible "
      f"{len(pe)}; filled eligible {int(pe['filled'].sum())}; per tercile (n / filled): "
      f"{pe.groupby('terc')['filled'].agg(['size', 'sum']).values.tolist()}")

# ------------------------------------------------------------------------------------------ Q4 baseline
hdr("Q4 — BASELINE B_d")
Bs = pd.Series({d: B[d] for d in USABLE})
print(f"B_d over usable sessions (pts per 1,000 contracts): median {Bs.median():.4f}, IQR {Bs.quantile(.25):.4f}–"
      f"{Bs.quantile(.75):.4f}, min {Bs.min():.4f}, max {Bs.max():.4f}; B_d <= 0: {int((Bs <= 0).sum())}")
r = (Bs / Bs.shift(1)).dropna()
print(f"day-to-day ratio B_d / B_(d-1): median {r.median():.3f}, p5 {r.quantile(.05):.3f}, p95 {r.quantile(.95):.3f}")
own = pd.Series({d: float(np.median(AW[d])) for d in BASE_ELIG})
print(f"single-session median A_w (baseline-eligible sessions, n {len(own)}): median {own.median():.4f}, IQR "
      f"{own.quantile(.25):.4f}–{own.quantile(.75):.4f}; windows per session median "
      f"{np.median([len(AW[d]) for d in BASE_ELIG]):.0f}")
own_u = own.reindex(USABLE)
print(f"corr(B_d, same-session median A_w) over usable sessions: {np.corrcoef(Bs, own_u)[0, 1]:.3f}")
print("B_d by month:", Bs.groupby([d.strftime('%Y-%m') for d in Bs.index]).median().round(4).to_dict())
first_b = min(B)
print(f"first session with 20 prior baseline sessions: {first_b}; sessions in window before it "
      f"(excluded for B_d, all causes counted in Q3a): {len([d for d in SESSIONS if d < first_b])}")

# ------------------------------------------------------------------------------------------ Q5 worked
hdr("Q5 — WORKED EVENT (random eligible real event, seed 20261025); outcome hidden")
ev = elig.iloc[int(np.random.default_rng(SEED).integers(len(elig)))]
d = ev["d"]
f = FLOW[d]
print(f"session {d}; A {prev_trading_day(d)}; symbol {SYM[d]}; P_target {PTGT[d]:.2f}; RTH open {OPEN[d]:.2f}; "
      f"IM {IM[d]:.2f}")
print(f"level L {ev['L']:.2f}; bin net {ev['net'] / 1e9:+.2f} $bn; touch {datetime.fromtimestamp(ev['t'], PT):%H:%M:%S} PT; "
      f"approach {'from above (fade long)' if ev['side'] == 1 else 'from below (fade short)'}; entry filled "
      f"{'yes' if ev['fill_i'] >= 0 else 'no'}")
g = LVROWS[d]
g = g[g["bin"] == ev["L"]].copy()
g["dte"] = [(e - d).days for e in g["expir_date"]]
g["net_bn"] = g["net"] / 1e9
print(g[["strike", "expir_date", "dte", "stock_price", "level_corrected", "net_bn"]].sort_values("net_bn")
      .round(2).to_string(index=False))
i = int(ev["i"])
w0 = np.searchsorted(f["t"], ev["t"] - WIN_S)
path = pd.DataFrame(dict(t=f["t"][w0:i], buy=f["buy"][w0:i], sell=f["sell"][w0:i], c=f["c"][w0:i]))
path["slot"] = (path["t"] - (ev["t"] - WIN_S)) // 30
agg = path.groupby("slot").agg(buy=("buy", "sum"), sell=("sell", "sum"), close=("c", "last"))
agg.index = [f"-{WIN_S - 30 * k}s..-{WIN_S - 30 * (k + 1)}s" for k in agg.index]
print(f"300 s path (30 s slots; window start reference close {f['c'][w0 - 1]:.2f}):")
print(agg.to_string())
print(f"D = {ev['D']:,} contracts (signed into the level); dP = {ev['dP']:+.2f} pts (approach direction); "
      f"A = {ev['A']:.4f} pts/1k; B_d = {B[d]:.4f}; A_rel = {ev['A_rel']:.3f}; tercile "
      f"{['low', 'mid', 'high'][int(ev['terc'])]}")
print(f"S7 windows: A_60 {ev['A_60']:.4f} (D {ev['D_60']}), A_900 {ev['A_900']:.4f} (D {ev['D_900']})")

# ------------------------------------------------------------------------------------------- Q6 A/A
hdr("Q6 — A/A CONTROL (two independent placebo draws, full harness, R = 8 ticks)")


def race(d, e):
    """1 win, 0 loss, -1 unresolved, from the fill. A/A only."""
    f = FLOW[d]
    _, b = rth_slice(d)
    L, s, fi = e.L, e.side, int(e.fill_i)
    Rp = R_TICKS * TICK
    if s == 1:
        tgt = f["h"][fi + 1:b] >= L + Rp + TICK
        stp = f["l"][fi + 1:b] <= L - Rp
        if f["l"][fi] <= L - Rp:
            return 0
    else:
        tgt = f["l"][fi + 1:b] <= L - Rp - TICK
        stp = f["h"][fi + 1:b] >= L + Rp
        if f["h"][fi] >= L + Rp:
            return 0
    ti, si = np.nonzero(tgt)[0], np.nonzero(stp)[0]
    if not len(ti) and not len(si):
        return -1
    if not len(si):
        return 1
    if not len(ti):
        return 0
    return 1 if ti[0] < si[0] else 0


SIDX = {d: k for k, d in enumerate(USABLE)}


def harness(lv):
    ev = []
    for d in USABLE:
        ev += events_for(d, lv[d], "aa")[0]
    E = pd.DataFrame(ev)
    E = E[(E["D"] > 0) & (E["side"] != 0) & (E["fill_i"] >= 0)].copy()
    E["terc"] = tercile(E["A_rel"].to_numpy())
    E["res"] = [race(r.d, r) for r in E.itertuples()]
    E = E[E["res"] >= 0]
    cnt = np.zeros((len(USABLE), 3, 2))           # session × tercile × (wins, resolved)
    for r in E.itertuples():
        cnt[SIDX[r.d], r.terc, 0] += r.res
        cnt[SIDX[r.d], r.terc, 1] += 1
    return cnt, E


ss = np.random.SeedSequence(SEED).spawn(2)
C1, E1 = harness(placebo_levels(np.random.default_rng(ss[0]))[0])
C2, E2 = harness(placebo_levels(np.random.default_rng(ss[1]))[0])


def stat(w):
    s1 = np.tensordot(w, C1, axes=1)
    s2 = np.tensordot(w, C2, axes=1)
    d1 = s1[0, 0] / s1[0, 1] - s1[2, 0] / s1[2, 1]
    d2 = s2[0, 0] / s2[0, 1] - s2[2, 0] / s2[2, 1]
    return d1 - d2


obs = stat(np.ones(len(USABLE)))
brng = np.random.default_rng(SEED_BOOT)
boots = np.empty(N_BOOT)
for k in range(N_BOOT):
    boots[k] = stat(np.bincount(brng.integers(len(USABLE), size=len(USABLE)), minlength=len(USABLE)))
lo, hi = np.nanpercentile(boots, [2.5, 97.5])
print(f"resolved events: draw 1 {len(E1)} (per tercile {np.bincount(E1['terc'], minlength=3).tolist()}), draw 2 "
      f"{len(E2)} (per tercile {np.bincount(E2['terc'], minlength=3).tolist()}); sessions {len(USABLE)}")
print(f"A/A [low − high](draw 1) − [low − high](draw 2) = {obs * 100:+.2f} pts; 95 % session-cluster CI "
      f"[{lo * 100:+.2f}, {hi * 100:+.2f}]   GATE CI covers 0: {'PASS' if lo <= 0 <= hi else 'FAIL — STOP'}")
print("\nDONE")
