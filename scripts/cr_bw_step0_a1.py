#!/usr/bin/env python3
"""CR-BW Step 0 under Amendment A1. READ-ONLY: SELECTs only, no API calls.

Spec: specs/CR-BW-aggressor-absorption-at-levels.md (Step 0, Amendment A1.1–A1.7).

  A1.6  Q1 outlier trace: every RTH minute where the flow-aggregated O/H/L/C differs from ironbeam_es_1m_bars by
        > 1 pt, over all kept sessions (count, sessions, times, symbol, field, the 1-s rows, raw trades if retained)
  Q3    sessions after each rule; levels per session (A1.1) and how many reach >= 100 $bn; armed touch events;
        event drops (A1.6, A1.4); T1-eligible and per-tercile counts, real and placebo (A1.2); entry-fill rate.
        Gate: every real tercile >= 150 eligible filled events.
  Q4    B_d under A1.5
  A1.3  A/A calibration: 200 splits of the placebo pseudo-levels; per split the T2-form statistic and its
        2,000-resample session-cluster CI. Pass: CI excludes 0 in <= 10 % of splits and median within ±3 pts.

NO race is run at real levels. The race runs only on placebo events, and only inside the A/A splits; only the
split statistics' distribution is printed (no win rates).

Definitions fixed here before the run (unchanged from scripts/cr_bw_step0.py unless marked A1):
  - connection: get_backfill_db_conn() + default_transaction_read_only = on (asserted); statement_timeout 600 s;
    Python alarm 3 h. Holdout guard: sessions <= 2026-06-05, flow ts_utc < 2026-06-05 21:00 UTC.
  - sessions: NYSE trading days 2025-12-26 .. 2026-06-05; L14 (a) roll week, (b) no ES 15:45 ET bar on A,
    (c) roll mismatch; canonical bt_daily_features IM > 0; orats_oi_gamma rows present.
    A1.4: no session-level flow rule. A session needs flow rows (one symbol, Q2) and an RTH open.
  - RTH open O_d: open of the first flow row in [06:30:00, 06:31:00) PT; if none, the open of the
    ironbeam_es_1m_bars bar starting 06:30 PT (Q1: same series); none -> session out. Fallbacks counted.
  - A1.1 levels: orats_oi_gamma SPX, session D (holiday store dates -> next_trading_day), expir_date in
    [D, D+30]; L = place_levels(rows, P_target) unchanged (P_target = ES 15:45 ET bar close on A);
    1-pt buckets (bucket_levels); per bucket call = sum gex_call, put = -sum |gex_put| (chart convention);
    mag = |call| + |put|; net = call + put. Chart selection re-implemented from callbacks._select_levels:
    mag >= 40e9, descending mag, keep if >= 5 pts from every kept level (GEX_MIN_LEVEL_SPACING 5). The chart
    applies its price band (the session's own bars) before spacing; that band is lookahead, so spacing runs over
    all buckets of the [D, D+30] universe. Then |L - O_d| <= 1.0 * IM_d, then the 6 largest by mag.
  - A1.2 placebo: for each real level L (side s = sign(L - O_d)), candidate 5-pt bins b (multiples of 5) with
    sign(b - O_d) = s, |b - O_d| <= IM_d, b not equal to a real level, min |b - real| >= 10; the 2 nearest to L
    (ties -> lower b). Pseudo-levels pooled per session (deduplicated); each keeps the parent's mag/net.
  - touch (L5), approach side, fill (L6, 60 s, 1 tick through), predictor (L7, 300 s; D, dP, A) as before.
  - A1.5 baseline: baseline sessions = sessions passing L14 (a)-(c) with flow; windows of 300 s from 06:30 PT,
    a window dropped if it contains an RTH flow gap > 15 s or has net = 0. B_d = median A_w pooled over the
    trailing up to 20 prior baseline sessions; >= 5 required.
  - A1.4 gap: max RTH gap in [t0, t1] = the largest of (first row - t0), consecutive row diffs, (t1 - last row),
    with t0 clipped to 06:30:00. Event dropped if > 15 s on the pre-touch window [t - 300, t]. Race window
    [t, race end]: applied in the A/A (race known); for real events Step 0 reports only an upper bound
    (gap > 15 s anywhere in [t, 13:00)), because the race end is an outcome.
  - A1.6: an event is dropped if a > 1 pt outlier minute overlaps its pre-touch window [t - 300, t]; race-window
    overlap handled as A1.4 (A/A applied; real upper bound reported).
  - terciles (L8): A_rel cut points over all kept T1-eligible real events; placebo uses the same cut points.
  - race (A/A only, R = 8 ticks): from the bar after the fill; the fill bar counts a stop, not a target; same bar
    both -> loss; neither by 13:00 PT -> unresolved (excluded). Race end = resolving bar (or 13:00).
  - A1.3 split: the placebo pseudo-levels (session, bin) are shuffled with default_rng(20261025) and cut into two
    halves of equal count (odd -> half 1 one larger), 200 times from the same generator; events follow their
    pseudo-level. Statistic: [low - high tercile win rate](half 1) - [same](half 2), placebo events kept after
    every drop rule including the race-window rule. CI: session-cluster bootstrap, 2,000 resamples per split,
    one generator default_rng(20260920) across splits, 95 % percentile.

Usage: apps/web/.venv/bin/python -u scripts/cr_bw_step0_a1.py | tee scripts/logs/cr_bw/step0_a1.log
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
DTE_MAX, MAG_MIN, SPACING, WIN_IM, TOP_N, STRONG = 30, 40e9, 5.0, 1.0, 6, 100e9
ARM_IM, WIN_S, FILL_S, GAP_MAX, OUTLIER_PTS = 0.10, 300, 60, 15, 1.0
BASE_N, BASE_MIN, BASE_W = 20, 5, 300
PLC_BIN, PLC_SEP, PLC_K = 5.0, 10.0, 2
R_TICKS = 8
SEED, SEED_BOOT, N_SPLIT, N_BOOT = 20261025, 20260920, 200, 2_000
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


def hms(t: int) -> str:
    return datetime.fromtimestamp(t, PT).strftime("%H:%M:%S")


SESSIONS = trading_days(D_FROM, D_TO)
QEXP = {third_friday(y, m) for y in (2025, 2026) for m in (3, 6, 9, 12)}

# ---------------------------------------------------------------------------------------------- loads
hdr("LOADS")
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
    """, (SYM[d], pt(d, 6, 0), pt(d, 13, 0), HOLDOUT_TS))
    FLOW[d] = dict(t=np.array([ep(x) for x in f["ts_utc"]], dtype=np.int64),
                   buy=f["buy_vol"].to_numpy(np.int64), sell=f["sell_vol"].to_numpy(np.int64),
                   o=f["open"].to_numpy(float), h=f["high"].to_numpy(float), l=f["low"].to_numpy(float),
                   c=f["close"].to_numpy(float))
print(f"flow loaded for {len(FLOW)} sessions; rows {sum(len(v['t']) for v in FLOW.values()):,}")

BARS = q("""SELECT datetime, open::float o, high::float h, low::float l, close::float c FROM ironbeam_es_1m_bars
            WHERE datetime >= %s AND datetime < %s""",
         (pt(D_FROM, 6, 30).replace(tzinfo=None), HOLDOUT_TS.replace(tzinfo=None)))
BARS["m"] = [int(pd.Timestamp(x).tz_localize("UTC").timestamp()) for x in BARS["datetime"]]
BARS = BARS.set_index("m")

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
    """Largest RTH flow gap (s) within [t0, t1], t0 clipped to 06:30 PT, t1 to 13:00 PT."""
    f = FLOW[d]["t"]
    t0 = max(t0, ep(pt(d, 6, 30)))
    t1 = min(t1, ep(pt(d, 13, 0)))
    if t1 <= t0:
        return 0
    a, b = np.searchsorted(f, t0), np.searchsorted(f, t1, side="right")
    pts = np.concatenate([[t0], f[a:b], [t1]])
    return int(np.diff(pts).max())


def rth_open(d):
    if d in FLOW:
        f = FLOW[d]
        a, _ = rth_slice(d)
        if a < len(f["t"]) and f["t"][a] < ep(pt(d, 6, 31)):
            return float(f["o"][a]), "flow"
    m = ep(pt(d, 6, 30))
    if m in BARS.index:
        return float(BARS.loc[m, "o"]), "1m bar"
    return None, None


# ------------------------------------------------------------------------------------------ sessions
hdr("Q3a — SESSIONS AFTER EACH RULE")
OPEN, OPEN_SRC = {}, {}
for d in SESSIONS:
    o, s = rth_open(d)
    if o is not None:
        OPEN[d], OPEN_SRC[d] = o, s
ex_a = {d for d in SESSIONS if roll_week(d)}
ex_b = {d for d in SESSIONS if d not in PTGT}
ex_c = {d for d in SESSIONS if d in QEXP or d in ROLL_MISMATCH_EXTRA}
L14 = [d for d in SESSIONS if d not in ex_a | ex_b | ex_c]


# A1.5 baseline
def base_windows(d):
    f = FLOW[d]
    t0 = ep(pt(d, 6, 30))
    out, dropped = [], 0
    for k in range(78):
        ws, we = t0 + BASE_W * k, t0 + BASE_W * (k + 1)
        a, b = np.searchsorted(f["t"], ws), np.searchsorted(f["t"], we)
        if a == 0 or b <= a or max_gap(d, ws, we) > GAP_MAX:
            dropped += 1
            continue
        net = int(f["buy"][a:b].sum() - f["sell"][a:b].sum())
        if net == 0:
            dropped += 1
            continue
        out.append(np.sign(net) * (f["c"][b - 1] - f["c"][a - 1]) / (abs(net) / 1000.0))
    return np.array(out), dropped


BASE_ELIG = [d for d in L14 if d in FLOW]
AW, AWD = {}, {}
for d in BASE_ELIG:
    AW[d], AWD[d] = base_windows(d)
BASE_ELIG = [d for d in BASE_ELIG if len(AW[d])]
B, BN = {}, {}
for d in SESSIONS:
    prior = [x for x in BASE_ELIG if x < d][-BASE_N:]
    if len(prior) >= BASE_MIN:
        B[d] = float(np.median(np.concatenate([AW[x] for x in prior])))
        BN[d] = len(prior)

U = list(SESSIONS)
print(f"{'rule':44s} {'removed':>8s} {'remaining':>10s}")
print(f"{'trading sessions':44s} {'':>8s} {len(U):>10d}")
for name, rule in [
    ("L14a roll week", lambda d: d in ex_a),
    ("L14b no ES 15:45 ET bar on A", lambda d: d in ex_b),
    ("L14c roll mismatch", lambda d: d in ex_c),
    ("no flow rows (session)", lambda d: d not in FLOW),
    ("no RTH open (flow or 1m bar)", lambda d: d not in OPEN),
    ("no canonical bt_daily_features row", lambda d: d not in feat.index),
    ("IM null or <= 0", lambda d: d in feat.index and not (pd.notna(feat[d]) and feat[d] > 0)),
    ("no orats_oi_gamma rows", lambda d: d not in set(oig["d"])),
    ("< 5 prior baseline sessions (A1.5)", lambda d: d not in B),
]:
    rm = [d for d in U if rule(d)]
    U = [d for d in U if not rule(d)]
    print(f"{name:44s} {len(rm):>8d} {len(U):>10d}   {[str(x) for x in rm] if len(rm) <= 12 else ''}")
USABLE = U
IM = {d: float(feat[d]) for d in USABLE}
print(f"USABLE SESSIONS: {len(USABLE)} ({USABLE[0]} .. {USABLE[-1]}); RTH open from the 1m-bar fallback: "
      f"{[str(d) for d in USABLE if OPEN_SRC[d] != 'flow']}")
print(f"baseline windows dropped (gap > 15 s or net 0), baseline sessions {len(BASE_ELIG)}: median "
      f"{np.median([AWD[d] for d in BASE_ELIG]):.0f} of 78 per session, max {max(AWD[d] for d in BASE_ELIG)}")

# ------------------------------------------------------------------------------------ A1.6 outliers
hdr("A1.6 — Q1 OUTLIER TRACE (flow-minute vs 1m bar, > 1 pt, RTH, all usable sessions)")
OUT = []
for d in USABLE:
    f = FLOW[d]
    a, b = rth_slice(d)
    df = pd.DataFrame(dict(t=f["t"][a:b], o=f["o"][a:b], h=f["h"][a:b], l=f["l"][a:b], c=f["c"][a:b]))
    df["m"] = df["t"] // 60 * 60
    m = df.groupby("m").agg(o=("o", "first"), h=("h", "max"), l=("l", "min"), c=("c", "last"))
    j = m.join(BARS[["o", "h", "l", "c"]], rsuffix="_b", how="inner")
    for k in "ohlc":
        bad = j[(j[k] - j[f"{k}_b"]).abs() > OUTLIER_PTS]
        for mm, r in bad.iterrows():
            OUT.append(dict(d=d, m=int(mm), field=k.upper(), flow=r[k], bar=r[f"{k}_b"], diff=r[k] - r[f"{k}_b"],
                            sym=SYM[d]))
O = pd.DataFrame(OUT)
compared = sum(len(FLOW[d]["t"]) > 0 for d in USABLE)
print(f"sessions compared {compared}; outlier (minute, field) pairs {len(O)}; distinct minutes "
      f"{O[['d', 'm']].drop_duplicates().shape[0] if len(O) else 0}; sessions with any "
      f"{O['d'].nunique() if len(O) else 0}")
if len(O):
    O["time_pt"] = [hms(x) for x in O["m"]]
    print(O.groupby("field").size().to_dict(), "| by symbol", O.groupby("sym").size().to_dict())
    print(O[["d", "time_pt", "sym", "field", "flow", "bar", "diff"]].to_string(index=False))
    # 1-s rows behind the worst 5 outlier minutes, and raw trades if retained
    for r in O.reindex(O["diff"].abs().sort_values(ascending=False).index).drop_duplicates(["d", "m"]).head(5).itertuples():
        f = FLOW[r.d]
        a, b = np.searchsorted(f["t"], r.m), np.searchsorted(f["t"], r.m + 60)
        s = pd.DataFrame(dict(t=[hms(x) for x in f["t"][a:b]], o=f["o"][a:b], h=f["h"][a:b], l=f["l"][a:b],
                              c=f["c"][a:b], vol=f["buy"][a:b] + f["sell"][a:b]))
        ext = s[(s["h"] == s["h"].max()) | (s["l"] == s["l"].min())]
        bar = BARS.loc[r.m]
        print(f"\n{r.d} {r.time_pt} PT {r.sym}: bar O/H/L/C {bar['o']}/{bar['h']}/{bar['l']}/{bar['c']}; flow minute "
              f"O/H/L/C {s['o'].iloc[0]}/{s['h'].max()}/{s['l'].min()}/{s['c'].iloc[-1]}; flow seconds {len(s)}")
        print("  1-s rows at the flow extremes:\n" + ext.to_string(index=False))
        tr = q("""SELECT ts_utc, price, size, aggressor_side FROM ironbeam_es_trades
                  WHERE symbol = %s AND ts_utc >= %s AND ts_utc < %s
                  ORDER BY abs(price - %s) DESC LIMIT 5""",
               (r.sym, datetime.fromtimestamp(r.m, UTC), datetime.fromtimestamp(r.m + 60, UTC), float(bar["c"])))
        print("  raw trades furthest from the bar close:", "none retained" if tr.empty else "\n" + tr.to_string(index=False))
OUT_MIN = {}
for r in (O.itertuples() if len(O) else []):
    OUT_MIN.setdefault(r.d, set()).add(r.m)


def outlier_in(d, t0, t1):
    return any(t0 < m + 60 and m <= t1 for m in OUT_MIN.get(d, ()))


# --------------------------------------------------------------------------------------- A1.1 levels
LEVELS, LVALL = {}, {}
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
    o, im = OPEN[d], IM[d]
    win = [k for k in kept if abs(k.level - o) <= WIN_IM * im]
    LVALL[d] = len(win)
    LEVELS[d] = [dict(L=float(k.level), mag=float(k.mag), net=float(k.net)) for k in win[:TOP_N]]


# -------------------------------------------------------------------------------------- A1.2 placebo
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
            w0 = np.searchsorted(t, ti - WIN_S)
            buy, sell = int(f["buy"][w0:i].sum()), int(f["sell"][w0:i].sum())
            D = (sell - buy) if side == 1 else (buy - sell)
            p_start = c[w0 - 1] if w0 > 0 else np.nan
            dP = (p_start - c[i - 1]) if side == 1 else (c[i - 1] - p_start)
            A = dP / (D / 1000.0) if D > 0 else np.nan
            out.append(dict(d=d, tag=tag, L=L, mag=lv["mag"], net=lv["net"], i=i, t=ti, side=side,
                            fill_i=i + int(fj[0]) if len(fj) else -1, D=D, dP=dP, A=A,
                            A_rel=A / B[d] if D > 0 else np.nan,
                            pre_gap=max_gap(d, ti - WIN_S, ti), pre_out=outlier_in(d, ti - WIN_S, ti),
                            post_gap_ub=max_gap(d, ti, ep(pt(d, 13, 0))),
                            post_out_ub=outlier_in(d, ti, ep(pt(d, 13, 0)))))
            start = cont[ci] + 1
    return out


def funnel(E, name):
    n0 = len(E)
    E = E[~E["pre_out"]]
    n1 = len(E)
    E = E[E["pre_gap"] <= GAP_MAX]
    n2 = len(E)
    print(f"{name}: armed touch events {n0}; A1.6 outlier in pre-window −{n0 - n1}; A1.4 gap > 15 s in pre-window "
          f"−{n1 - n2}; kept {n2}; side undetermined {int((E['side'] == 0).sum())}")
    return E


hdr("Q3b — LEVELS, EVENTS, ELIGIBILITY, FILLS (no outcomes)")
nl = pd.Series({d: len(LEVELS.get(d, [])) for d in USABLE})
nall = pd.Series({d: LVALL.get(d, 0) for d in USABLE})
AL = pd.DataFrame([dict(d=d, **x) for d in USABLE for x in LEVELS.get(d, [])])
print(f"chart lines (mag >= 40 $bn, spaced) within ±1 IM before the top-6 cut: median {nall.median():.0f}, IQR "
      f"{nall.quantile(.25):.0f}–{nall.quantile(.75):.0f}, max {nall.max()}")
print(f"levels per session (A1.1, top 6): median {nl.median():.0f}, IQR {nl.quantile(.25):.0f}–{nl.quantile(.75):.0f}, "
      f"min {nl.min()}; sessions with 0 {int((nl == 0).sum())}; with 6 {int((nl == 6).sum())}; total {nl.sum()}")
print(f"level mag $bn: median {AL['mag'].median() / 1e9:.1f}, IQR {AL['mag'].quantile(.25) / 1e9:.1f}–"
      f"{AL['mag'].quantile(.75) / 1e9:.1f}; >= 100 $bn: {int((AL['mag'] >= STRONG).sum())} levels on "
      f"{AL.loc[AL['mag'] >= STRONG, 'd'].nunique()} sessions; net sign −/+ {int((AL['net'] < 0).sum())}/"
      f"{int((AL['net'] > 0).sum())}; |L − open| median {(AL['L'] - AL['d'].map(OPEN)).abs().median():.1f} pts")
npl = pd.Series({d: len(PLV[d]) for d in USABLE})
print(f"placebo pseudo-levels (A1.2): total {npl.sum()} (target {PLC_K} × {nl.sum()} = {PLC_K * nl.sum()} before dedup); "
      f"per session median {npl.median():.0f}, IQR {npl.quantile(.25):.0f}–{npl.quantile(.75):.0f}")

R = funnel(pd.DataFrame(sum((events_for(d, LEVELS.get(d, []), "real") for d in USABLE), [])), "REAL")
P = funnel(pd.DataFrame(sum((events_for(d, PLV[d], "placebo") for d in USABLE), [])), "PLACEBO")
R, P = R[R["side"] != 0].copy(), P[P["side"] != 0].copy()
per = R.groupby("d").size().reindex(USABLE, fill_value=0)
print(f"real kept events per session: median {per.median():.0f}, IQR {per.quantile(.25):.0f}–{per.quantile(.75):.0f}, "
      f"max {per.max()}")
ER, EP = R[R["D"] > 0].copy(), P[P["D"] > 0].copy()
print(f"T1-eligible (D > 0): real {len(ER)}, placebo {len(EP)}; D <= 0 (S6): real {int((R['D'] <= 0).sum())}, "
      f"placebo {int((P['D'] <= 0).sum())}")
cut = np.percentile(ER["A_rel"], [100 / 3, 200 / 3])
print(f"A_rel tercile cut points (real eligible): {cut.round(4)}")


def terc(x):
    return np.where(x <= cut[0], 0, np.where(x <= cut[1], 1, 2))


for E in (ER, EP):
    E["terc"] = terc(E["A_rel"].to_numpy())
    E["filled"] = E["fill_i"] >= 0


def tab(E):
    t = E.groupby("terc").agg(eligible=("filled", "size"), filled=("filled", "sum"))
    t["fill_rate"] = (t["filled"] / t["eligible"]).round(3)
    t["filled_ub_race_drop"] = E[E["filled"] & ((E["post_gap_ub"] > GAP_MAX) | E["post_out_ub"])].groupby("terc").size()
    t["filled_ub_race_drop"] = t["filled_ub_race_drop"].fillna(0).astype(int)
    t["filled_min_after_ub"] = t["filled"] - t["filled_ub_race_drop"]
    t["filled_ge100bn"] = E[E["filled"] & (E["mag"] >= STRONG)].groupby("terc").size()
    t["filled_ge100bn"] = t["filled_ge100bn"].fillna(0).astype(int)
    t.index = ["low (absorption)", "mid", "high (continuation)"]
    return t


TR, TP = tab(ER), tab(EP)
print("\nREAL (eligible, D > 0):\n" + TR.to_string())
print("\nPLACEBO (eligible, D > 0; real cut points):\n" + TP.to_string())
print(f"\nentry-fill rate: real all kept events {(R['fill_i'] >= 0).mean():.3f}, eligible {ER['filled'].mean():.3f}; "
      f"placebo all {(P['fill_i'] >= 0).mean():.3f}, eligible {EP['filled'].mean():.3f}")
print(f"real eligible filled, S3 neg/pos net {int((ER['filled'] & (ER['net'] < 0)).sum())}/"
      f"{int((ER['filled'] & (ER['net'] > 0)).sum())}; S4 before/after 10:00 PT "
      f"{int((ER['filled'] & (ER['t'] < [ep(pt(x, 10, 0)) for x in ER['d']])).sum())}/"
      f"{int((ER['filled'] & (ER['t'] >= [ep(pt(x, 10, 0)) for x in ER['d']])).sum())}")
gate = int(TR["filled"].min())
print(f"GATE (every real tercile >= 150 eligible filled events): {'PASS' if gate >= 150 else 'FAIL — STOP'} "
      f"(min {gate}; min after the race-window upper-bound drop {int(TR['filled_min_after_ub'].min())})")

hdr("Q4 — BASELINE B_d (A1.5)")
Bs = pd.Series({d: B[d] for d in USABLE})
r = (Bs / Bs.shift(1)).dropna()
print(f"B_d median {Bs.median():.4f}, IQR {Bs.quantile(.25):.4f}–{Bs.quantile(.75):.4f}, range {Bs.min():.4f}–"
      f"{Bs.max():.4f}; day ratio p5/median/p95 {r.quantile(.05):.3f}/{r.median():.3f}/{r.quantile(.95):.3f}; "
      f"sessions on < 20 prior: {int((pd.Series(BN).reindex(USABLE) < 20).sum())} (min {min(BN[d] for d in USABLE)})")

if gate < 150:
    sys.exit("STOP: tercile gate failed")

# ------------------------------------------------------------------------------------------ A1.3 A/A
hdr("A1.3 — A/A CALIBRATION (200 splits of placebo pseudo-levels, R = 8 ticks)")


def race(d, fi, L, s):
    """(result, end_t): 1 win, 0 loss, -1 unresolved. Placebo only."""
    f = FLOW[d]
    _, b = rth_slice(d)
    Rp = R_TICKS * TICK
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
    res = 1 if len(ti) and (not len(si) or ti[0] < si[0]) else 0
    return res, int(f["t"][fi + 1 + k])


A = EP[EP["filled"]].copy()
rr = [race(x.d, int(x.fill_i), x.L, x.side) for x in A.itertuples()]
A["res"], A["end"] = [x[0] for x in rr], [x[1] for x in rr]
A["race_gap"] = [max_gap(x.d, x.t, x.end) for x in A.itertuples()]
A["race_out"] = [outlier_in(x.d, x.t, x.end) for x in A.itertuples()]
n0 = len(A)
A = A[(A["race_gap"] <= GAP_MAX) & ~A["race_out"]]
n1 = len(A)
A = A[A["res"] >= 0]
print(f"placebo eligible filled {n0}; race-window drops (gap > 15 s or outlier) −{n0 - n1}; unresolved −{n1 - len(A)}; "
      f"resolved {len(A)}")

S = {d: k for k, d in enumerate(USABLE)}
PL = sorted({(x.d, x.L) for x in EP.itertuples()})
PLI = {p: k for k, p in enumerate(PL)}
A["sidx"] = [S[d] for d in A["d"]]
A["pidx"] = [PLI[(d, L)] for d, L in zip(A["d"], A["L"])]
nS = len(USABLE)
rng = np.random.default_rng(SEED)
brng = np.random.default_rng(SEED_BOOT)
W = None


def counts(mask):
    """session × 6 array: (wins, resolved) for low and high terciles of the events in mask."""
    out = np.zeros((nS, 4))
    sub = A[mask]
    for col, (t, what) in enumerate([(0, "w"), (0, "n"), (2, "w"), (2, "n")]):
        s = sub[sub["terc"] == t]
        out[:, col] = np.bincount(s["sidx"], weights=s["res"] if what == "w" else None, minlength=nS)
    return out


def tdiff(c):
    return c[..., 0] / c[..., 1] - c[..., 2] / c[..., 3]


stats, excl, los, his = [], 0, [], []
for k in range(N_SPLIT):
    perm = rng.permutation(len(PL))
    h1 = np.zeros(len(PL), bool)
    h1[perm[: (len(PL) + 1) // 2]] = True
    m1 = h1[A["pidx"].to_numpy()]
    c1, c2 = counts(m1), counts(~m1)
    obs = tdiff(c1.sum(0)) - tdiff(c2.sum(0))
    w = np.stack([np.bincount(brng.integers(nS, size=nS), minlength=nS) for _ in range(N_BOOT)]).astype(float)
    bs = tdiff(w @ c1) - tdiff(w @ c2)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    stats.append(obs)
    los.append(lo)
    his.append(hi)
    excl += not (lo <= 0 <= hi)
st = np.array(stats) * 100
share = excl / N_SPLIT
print(f"pseudo-levels split {len(PL)}; splits {N_SPLIT}; bootstrap {N_BOOT} per split")
print(f"statistic (pts): median {np.median(st):+.2f}, mean {st.mean():+.2f}, sd {st.std(ddof=1):.2f}, p5 "
      f"{np.percentile(st, 5):+.2f}, p25 {np.percentile(st, 25):+.2f}, p75 {np.percentile(st, 75):+.2f}, p95 "
      f"{np.percentile(st, 95):+.2f}, min {st.min():+.2f}, max {st.max():+.2f}")
hw = (np.array(his) - np.array(los)) * 100 / 2
print(f"bootstrap CI half-width (pts): median {np.median(hw):.2f}; split-to-split sd of the statistic / median "
      f"half-width × 1.96 = {st.std(ddof=1) / (np.median(hw) / 1.96):.2f} (≈ 1 if the CI is calibrated)")
print("histogram (pts):", pd.cut(st, np.arange(-40, 41, 5)).value_counts().sort_index().to_dict())
print(f"share of splits whose CI excludes 0: {share:.1%} ({excl}/{N_SPLIT})")
ok = share <= 0.10 and abs(np.median(st)) <= 3
print(f"GATE A1.3 (excludes 0 in <= 10 % AND |median| <= 3 pts): {'PASS' if ok else 'FAIL — STOP'}")
print("\nDONE")
