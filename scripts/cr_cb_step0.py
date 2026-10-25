#!/usr/bin/env python3
"""CR-CB Step 0 (Stage 1) — failed retest at a level. READ-ONLY: SELECTs only, no API calls.

Spec: specs/CR-CB-failed-retest-at-level.md (L1–L11, Amendment A1). Answers Step 0 Q1–Q5. Computes NO exit
outcome at real levels or for placebo setups as a set; the only outcome computed is the hold-to-close (L8 stop)
P&L of placebo setups inside the A/A splits (Q5), reported only as the split statistic's distribution.

Reused unchanged from scripts/cr_bw_absorption.py (CR-BW A2): connection and read-only guard, flow loader
(ironbeam_es_flow_1s, one symbol per RTH session), sessions and L14 exclusions, flow RTH open (A2.1), A1.1
levels (place_levels + bucket_levels, mag >= 40 $bn, 5-pt spacing, ±1·IM, top 6), max_gap, 1-tick-through fill,
stop + 1 tick slippage, $4.50 round trip, session-cluster bootstrap, A1.3 split design.

Operational choices fixed here before the run (not stated in the spec):
  - Universe: CR-BW A2's 88 sessions. CR-BW's ">= 5 prior baseline sessions" rule only removed sessions before
    2026-01-06, so the window starts at 2026-01-06 and the B_d baseline is not computed (CR-CB has no A_rel).
    Asserted: 88 sessions.
  - Holdout guard: flow ts_utc < 2026-06-05 21:00 UTC for train. 2026-10-23 is read in one separate block
    (flow, orats_oi_gamma, IM, P_target for that date only) for the Q4 illustration; no outcome is computed there.
  - Cells: sign = put-heavy (net < 0) / call-heavy (net > 0); approach from below -> short setup, from above ->
    long setup. Placebo pseudo-levels take their parent's sign.
  - Longs are run as shorts on the negated price path (h' = -l, l' = -h, L' = -L).
  - Test (L5): CR-BW arming with a 2-pt band. Armed below when a 1-s bar has high <= L - 0.10·IM (above: low >=
    L + 0.10·IM); test = first later bar with high >= L - 2 (from below) / low <= L + 2 (from above). After a test
    the level disarms until price is again >= 0.10·IM away (CR-BW re-arm). Every test is listed; the detector
    skips tests that start before the previous detector episode on that level ended.
  - O_c measurement (L5, pre-entry paths only): for every real test, M = running max (short space) from the
    test bar. Episode ends at the first bar with low <= M - 0.10·IM whose high is below M (the max was set
    earlier). Overshoot = max(0, M_end - L). No pullback by 13:00 -> overshoot = M at the last RTH bar
    (censored; counted). "Moves 0.10·IM back" is read as a 0.10·IM pullback from the running extreme, the same
    construction L6 uses for H. Reported for comparison: the alternative reading (max excursion until price is
    back 0.10·IM on the approach side of L). Tests with an RTH flow gap > 15 s in [test, episode end] are not
    used. O_c per cell = the 60th / 80th / 95th percentile (A1.1); the 80th is the grid centre. Rounded to tick.
  - Local extreme (L6), short space: from the test bar, M = running max. Broken = M > L + O_c before
    confirmation. Confirmation = first bar with low <= M - max(P, 0.10·IM) and high < M. H = M at confirmation.
    H must lie in the zone [L, L + O_c]; H < L ("extreme short of the level") is no setup. No confirmation and
    no break by 13:00 -> none.
  - Failed retest (L7): window (confirmation bar, confirmation time + W min], capped at 13:00 PT. Retest = first
    bar with high >= H - tol. Fill = first bar with high >= H - tol + 1 tick (entry at H - tol). The "without
    trading above H + 1 tick" condition is checked on bars before the fill bar (it can only fail on a jump
    between seconds); a fill bar that also trades through the stop is a filled entry (its loss is an outcome).
  - One setup per level per session (L7) = the first filled entry. Detector episodes on a level are
    sequential: a new test is considered only after the previous episode ends (break, no setup at
    confirmation, retest window expiry, or fill).
  - Gap rule (CR-BW A1.4 / A2.3): a setup is dropped if [test bar, fill bar] contains an RTH flow gap > 15 s.
    The A/A hold-to-close additionally drops a placebo setup whose [fill, exit] window has a gap > 15 s.
  - Placebo (L10): for each real level, the 2 nearest 5-pt bins on the same side of the open, within ±1·IM,
    not within 15 pts of any real level; ties to the lower bin; dedup per session; parent mag/net kept. The
    cell's O_c is the real one.
  - Counts depend on (O pct, P, W, tol) only; the stop buffer does not change counts. Q3 reports the grid centre
    (80th, 6, 90, tol 1) and the configurations with the fewest and most filled entries (A1.5), with all 81.
  - Q4 illustration: 2026-10-23's flow has an RTH hole 07:52:46 -> ~09:56 PT (found on the first run), which
    covers the move Ryan describes. The detector is therefore also run on that day's ironbeam_es_1m_bars
    (each minute treated as one bar, t = minute start), labelled as a 1-minute proxy. Still no outcome.
  - Q3 also reports filled entries per cell for all 81 configurations, and setups that share one fill
    (same session, side and fill second) on adjacent levels.
  - Q4: 5 sessions drawn from the usable set with default_rng(20261025).choice(..., 5, replace=False); every
    setup (filled entry, centre config) plus every test that reached confirmation. No outcome column.
  - Q5 A/A (CR-BW A1.3 adapted): placebo filled setups, centre config, stop buffer 2 (L8), hold to the last RTH
    print before 13:00 PT, stop filled at stop + 1 tick, same bar fill + stop -> stop. Net P&L in points minus
    0.09 pts ($4.50 / $50). 200 splits of the pseudo-levels carrying >= 1 setup (default_rng(20261025)); per
    split the statistic = mean net (half 1) - mean net (half 2), session-cluster bootstrap 2,000 resamples
    (one default_rng(20260920) stream across splits). Pass: CI excludes 0 in <= 10 % of splits and median
    statistic within ±3 pts (CR-BW's ±3 is percentage points of win rate; here points of P&L).

Usage: apps/web/.venv/bin/python -u scripts/cr_cb_step0.py | tee scripts/logs/cr_cb/step0.log
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
ARM_IM, BAND, GAP_MAX = 0.10, 2.0, 15
PLC_BIN, PLC_SEP, PLC_K = 5.0, 15.0, 2
O_PCTS, PULLS, WINS, TOLS, STOPS = (60, 80, 95), (4.0, 6.0, 10.0), (30, 90, 180), (0.5, 1.0, 2.0), (1.0, 2.0, 4.0)
CENTRE = (80, 6.0, 90, 1.0)
STOP_CENTRE = 2.0
SEED_DRAW, SEED_BOOT, N_BOOT, N_SPLIT, N_BOOT_SPLIT = 20261025, 20260920, 10_000, 200, 2_000
ROLL_MISMATCH_EXTRA = {date(2026, 6, 15)}
CELLS = [("put", 1), ("put", -1), ("call", 1), ("call", -1)]  # side 1 = from below (short), -1 = from above (long)
OUT_DIR = REPO_ROOT / "scripts" / "logs" / "cr_cb"

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


def hms(d: date, t: int) -> str:
    return datetime.fromtimestamp(int(t), UTC).astimezone(PT).strftime("%H:%M:%S")


SESSIONS = trading_days(D_FROM, D_TO)
QEXP = {third_friday(y, m) for y in (2025, 2026) for m in (3, 6, 9, 12)}
INFO: dict = {}


# ---------------------------------------------------------------------------------------------- loads
def load_flow(d: date, symbol: str, cap: datetime) -> dict:
    f = q("""
        SELECT ts_utc, buy_vol, sell_vol, open, high, low, close
        FROM ironbeam_es_flow_1s
        WHERE symbol = %s AND ts_utc >= %s AND ts_utc < %s AND ts_utc < %s
        ORDER BY ts_utc
    """, (symbol, pt(d, 5, 30), pt(d, 13, 0), cap))
    return dict(t=np.array([ep(x) for x in f["ts_utc"]], dtype=np.int64),
                buy=f["buy_vol"].to_numpy(np.int64), sell=f["sell_vol"].to_numpy(np.int64),
                o=f["open"].to_numpy(float), h=f["high"].to_numpy(float), l=f["low"].to_numpy(float),
                c=f["close"].to_numpy(float))


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
FLOW: dict[date, dict] = {d: load_flow(d, SYM[d], HOLDOUT_TS) for d in SESSIONS if d in SYM}
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
assert len(USABLE) == 88, f"expected CR-BW A2's 88 sessions, got {len(USABLE)}"
IM = {d: float(feat[d]) for d in USABLE}


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


LEVELS, WINLINES = {}, {}
for d, g in oig[oig["d"].isin(USABLE)].groupby("d"):
    LEVELS[d], w = select_levels(g, PTGT[d], OPEN[d], IM[d])
    WINLINES[d] = len(w)


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
for lv in itertools.chain(*LEVELS.values(), *PLV.values()):
    lv["sign"] = "put" if lv["net"] < 0 else "call"


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
        # the first arming after pos decides the side; price cannot arm on the other side without first
        # entering the 2-pt band, so the test is the first band touch after that arming
        if ia is None or (ib is not None and ib < ia):
            jt, side = nxt(hit_b, ib), 1
        else:
            jt, side = nxt(hit_a, ia), -1
        if jt is None:
            break
        out.append((jt, side))
        pos = jt + 1
    return out


def space(F, side):
    """Short-space arrays: (high, low, sign) with longs negated."""
    return (F["h"], F["l"], 1.0) if side == 1 else (-F["l"], -F["h"], -1.0)


def overshoot(F, d, j, side, L, im):
    """O_c measurement from test bar j: (overshoot, censored, end idx, alt overshoot)."""
    _, b = rth_slice(F, d)
    H, Lo, sg = space(F, side)
    Ls = sg * L
    M = np.maximum.accumulate(H[j:b])
    pull = ARM_IM * im
    ok = (Lo[j:b] <= M - pull) & (H[j:b] < M)
    k = np.nonzero(ok)[0]
    cens = not len(k)
    e = (b - 1) if cens else j + int(k[0])
    ov = max(0.0, float(M[e - j]) - Ls)
    back = np.nonzero(H[j:b] <= Ls - pull)[0]          # alternative reading: back 0.10·IM below L
    e2 = (b - 1) if not len(back) else j + int(back[0])
    alt = max(0.0, float(M[e2 - j]) - Ls)
    return ov, cens, e, alt


def detect(F, d, j, side, L, im, O, P, W, tol, floor_im=ARM_IM):
    """Detector episode from test bar j (short space). Returns (status, end idx, detail dict)."""
    _, b = rth_slice(F, d)
    t = F["t"]
    H, Lo, sg = space(F, side)
    Ls = sg * L
    M = np.maximum.accumulate(H[j:b])
    brk = np.nonzero(M > Ls + O)[0]
    peff = max(P, floor_im * im)
    cf = np.nonzero((Lo[j:b] <= M - peff) & (H[j:b] < M))[0]
    kb = int(brk[0]) if len(brk) else 10**9
    kc = int(cf[0]) if len(cf) else 10**9
    if kb == 10**9 and kc == 10**9:
        return "none", b - 1, {}
    if kb <= kc:
        return "broken", j + kb, {}
    c = j + kc
    Hx = float(M[kc])
    hi_i = j + int(np.argmax(H[j:c + 1]))
    det = dict(H=sg * Hx, H_t=int(t[hi_i]), conf_t=int(t[c]), conf_px=sg * float(Lo[c]))
    if Hx < Ls:
        return "short", c, det
    wend = min(int(np.searchsorted(t, t[c] + W * 60, side="right")), b)
    seg = H[c + 1:wend]
    rt = np.nonzero(seg >= Hx - tol)[0]
    if not len(rt):
        return "noretest", wend - 1, det
    det["retest_t"] = int(t[c + 1 + int(rt[0])])
    fl = np.nonzero(seg >= Hx - tol + TICK)[0]
    if not len(fl):
        return "unfilled", wend - 1, det
    fi = c + 1 + int(fl[0])
    if np.any(H[c + 1:fi] > Hx + TICK):
        return "overrun", fi, det
    det.update(fill_i=fi, fill_t=int(t[fi]), entry=sg * (Hx - tol))
    return "filled", fi, det


# ---------------------------------------------------------------------------------- Q2: O_c per cell
def all_tests(levels_by_day, F_by_day, im_by_day, days):
    rows = []
    for d in days:
        F = F_by_day[d]
        for lv in levels_by_day[d]:
            for j, side in tests_for(F, d, lv["L"], im_by_day[d]):
                rows.append(dict(d=d, L=lv["L"], sign=lv["sign"], mag=lv["mag"], net=lv["net"], j=j, side=side,
                                 t=int(F["t"][j])))
    return rows


TR = all_tests(LEVELS, FLOW, IM, USABLE)
TP = all_tests(PLV, FLOW, IM, USABLE)
for r in TR:
    F = FLOW[r["d"]]
    ov, cens, e, alt = overshoot(F, r["d"], r["j"], r["side"], r["L"], IM[r["d"]])
    r.update(ov=ov, cens=cens, alt=alt, ogap=max_gap(F, r["d"], r["t"], int(F["t"][e])))
TRd = pd.DataFrame(TR)
OV = TRd[TRd["ogap"] <= GAP_MAX]
OC = {}
q2 = {}
for cell in CELLS:
    x = OV[(OV["sign"] == cell[0]) & (OV["side"] == cell[1])]
    v = x["ov"].to_numpy()
    OC[cell] = {p: float(np.round(np.percentile(v, p) / TICK) * TICK) for p in O_PCTS}
    q2[f"{cell[0]}/{'below-short' if cell[1] == 1 else 'above-long'}"] = dict(
        n=len(v), censored=int(x["cens"].sum()), zero_share=round(float((v == 0).mean()), 3),
        pct={p: round(float(np.percentile(v, p)), 2) for p in (10, 25, 50, 60, 75, 80, 90, 95, 99)},
        O_c=OC[cell], ov_over_im_median=round(float(np.median(v / np.array([IM[d] for d in x["d"]]))), 3),
        alt_pct={p: round(float(np.percentile(x["alt"], p)), 2) for p in (50, 60, 80, 95)})

# put vs call, same approach side: session-cluster bootstrap of the p80 and median difference
sidx = {d: k for k, d in enumerate(USABLE)}
rng_q2 = np.random.default_rng(SEED_BOOT)
BOOT_IDX = rng_q2.integers(0, len(USABLE), size=(N_BOOT, len(USABLE)))
q2_diff = {}
by_sess = {(c, d): OV[(OV["sign"] == c[0]) & (OV["side"] == c[1]) & (OV["d"] == d)]["ov"].to_numpy()
           for c in CELLS for d in USABLE}
for side in (1, -1):
    for stat, fn in (("p80", lambda v: np.percentile(v, 80)), ("median", np.median)):
        obs = fn(np.concatenate([by_sess[(("put", side), d)] for d in USABLE])) - \
            fn(np.concatenate([by_sess[(("call", side), d)] for d in USABLE]))
        bs = []
        for row in BOOT_IDX[:2000]:
            pv = np.concatenate([by_sess[(("put", side), USABLE[k])] for k in row])
            cv = np.concatenate([by_sess[(("call", side), USABLE[k])] for k in row])
            if len(pv) and len(cv):
                bs.append(fn(pv) - fn(cv))
        lo, hi = np.percentile(bs, [2.5, 97.5])
        q2_diff[f"{'below-short' if side == 1 else 'above-long'} {stat} put−call"] = \
            dict(est=round(float(obs), 2), lo=round(float(lo), 2), hi=round(float(hi), 2), n_boot=len(bs))


# ----------------------------------------------------------------------------- detector over the grid
def run_detector(tests, F_by_day, im_by_day, O_of, P, W, tol, floor_im=ARM_IM):
    """One setup per level per session; sequential episodes. Returns list of episode dicts."""
    out = []
    by_level: dict = {}
    for r in tests:
        by_level.setdefault((r["d"], r["L"]), []).append(r)
    for (d, L), rs in by_level.items():
        F = F_by_day[d]
        end = -1
        for r in sorted(rs, key=lambda x: x["j"]):
            if r["j"] <= end:
                continue
            O = O_of[(r["sign"], r["side"])]
            st, end, det = detect(F, d, r["j"], r["side"], L, im_by_day[d], O, P, W, tol, floor_im)
            ep_ = dict(r, status=st, O=O, **det)
            if st == "filled":
                ep_["pre_gap"] = max_gap(F, d, r["t"], det["fill_t"])
            out.append(ep_)
            if st == "filled":
                break
    return out


def tally(eps):
    res = {}
    for cell in CELLS:
        e = [x for x in eps if x["sign"] == cell[0] and x["side"] == cell[1]]
        st = pd.Series([x["status"] for x in e], dtype=object)
        filled = [x for x in e if x["status"] == "filled"]
        kept = [x for x in filled if x["pre_gap"] <= GAP_MAX]
        res[f"{cell[0]}/{'short' if cell[1] == 1 else 'long'}"] = dict(
            episodes=len(e), broken=int((st == "broken").sum()), none=int((st == "none").sum()),
            short_of_level=int((st == "short").sum()),
            confirmed=int(st.isin(["noretest", "unfilled", "overrun", "filled"]).sum()),
            retest=int(st.isin(["unfilled", "overrun", "filled"]).sum()), overrun=int((st == "overrun").sum()),
            filled=len(filled), gap_drop=len(filled) - len(kept), kept=len(kept))
    return res


GRID = list(itertools.product(O_PCTS, PULLS, WINS, TOLS))
GRES = {}
CENTRE_EPS = {}
for cfg in GRID:
    op, P, W, tol = cfg
    O_of = {c: OC[c][op] for c in CELLS}
    er = run_detector(TR, FLOW, IM, O_of, P, W, tol)
    epl = run_detector(TP, FLOW, IM, O_of, P, W, tol)
    GRES[cfg] = dict(real=tally(er), placebo=tally(epl))
    if cfg == CENTRE:
        CENTRE_EPS = dict(real=er, placebo=epl)


def tot(cfg, arm):
    return sum(v["kept"] for v in GRES[cfg][arm].values())


ranked = sorted(GRID, key=lambda c: tot(c, "real"))
q3 = dict(centre=dict(cfg=CENTRE, **GRES[CENTRE]), fewest=dict(cfg=ranked[0], **GRES[ranked[0]]),
          most=dict(cfg=ranked[-1], **GRES[ranked[-1]]),
          all=[dict(cfg=c, real=tot(c, "real"), placebo=tot(c, "placebo"),
                    min_cell=min(v["kept"] for v in GRES[c]["real"].values())) for c in GRID])
q3["per_cell_all"] = [dict(cfg=c, **{k: v["kept"] for k, v in GRES[c]["real"].items()},
                           **{"plc " + k: v["kept"] for k, v in GRES[c]["placebo"].items()}) for c in GRID]
q3["per_cell_max"] = {k: max(GRES[c]["real"][k]["kept"] for c in GRID) for k in GRES[CENTRE]["real"]}
for arm in ("real", "placebo"):
    f_ = [x for x in CENTRE_EPS[arm] if x["status"] == "filled" and x["pre_gap"] <= GAP_MAX]
    keys_ = pd.Series([(x["d"], x["side"], x["fill_t"]) for x in f_])
    q3[f"shared_fill_{arm}"] = dict(setups=len(f_), distinct_fills=int(keys_.nunique()))
q3["test_counts"] = dict(real=len(TR), placebo=len(TP),
                         real_by_cell={f"{c[0]}/{c[1]}": int(((TRd["sign"] == c[0]) & (TRd["side"] == c[1])).sum())
                                       for c in CELLS})

# ------------------------------------------------------------------------------------- Q4 detector table
rng4 = np.random.default_rng(SEED_DRAW)
Q4_DAYS = sorted(rng4.choice(np.array(USABLE, dtype=object), 5, replace=False))


def table_rows(eps, d, im, F):
    rows = []
    for x in sorted([e for e in eps if e["d"] == d and "H" in e], key=lambda e: e["t"]):
        rows.append(dict(
            session=str(d), level=x["L"], sign=x["sign"], mag_bn=round(x["mag"] / 1e9, 1),
            net_bn=round(x["net"] / 1e9, 1), setup="short" if x["side"] == 1 else "long", O_c=x["O"],
            zone=f"{x['L']:.2f}–{x['L'] + x['side'] * x['O']:.2f}",
            test_t=hms(d, x["t"]), test_px=(F["h"] if x["side"] == 1 else F["l"])[x["j"]],
            extreme=x["H"], extreme_t=hms(d, x["H_t"]), confirm_t=hms(d, x["conf_t"]), confirm_px=x["conf_px"],
            status=x["status"], retest_t=hms(d, x["retest_t"]) if "retest_t" in x else "",
            fill_t=hms(d, x["fill_t"]) if "fill_t" in x else "",
            entry=x.get("entry", ""), stop=(x["H"] + x["side"] * STOP_CENTRE) if x["status"] == "filled" else "",
            pre_gap_s=x.get("pre_gap", "")))
    return rows


Q4 = []
for d in Q4_DAYS:
    Q4 += table_rows(CENTRE_EPS["real"], d, IM[d], FLOW[d])

# 2026-10-23 illustration (holdout day, detector only, no outcome)
sym_i = q("""SELECT symbol, count(*) n FROM ironbeam_es_flow_1s WHERE ts_utc >= %s AND ts_utc < %s
             GROUP BY 1 ORDER BY n DESC""", (pt(ILLUS, 6, 30), pt(ILLUS, 13, 0)))
F_I = load_flow(ILLUS, sym_i["symbol"].iloc[0], pt(ILLUS, 13, 0))
IM_I = float(load_im(ILLUS, ILLUS)[ILLUS])
PT_I = load_ptgt([ILLUS])[ILLUS]
O_I = rth_open(F_I, ILLUS)
LV_I, WIN_I = select_levels(load_oig(ILLUS, ILLUS), PT_I, O_I, IM_I)
lines_i = [dict(L=float(k.level), mag=float(k.mag), net=float(k.net)) for k in WIN_I]
for lv in lines_i:
    lv["sign"] = "put" if lv["net"] < 0 else "call"
    lv["top6"] = any(abs(lv["L"] - x["L"]) < 1e-9 for x in LV_I)
near = sorted(lines_i, key=lambda x: abs(x["L"] - 7686))[:3]
T_I = all_tests({ILLUS: near}, {ILLUS: F_I}, {ILLUS: IM_I}, [ILLUS])
O_of_c = {c: OC[c][CENTRE[0]] for c in CELLS}
E_I = run_detector(T_I, {ILLUS: F_I}, {ILLUS: IM_I}, O_of_c, CENTRE[1], CENTRE[2], CENTRE[3])
Q4_ILLUS = dict(symbol=sym_i["symbol"].iloc[0], rth_rows=int(rth_slice(F_I, ILLUS)[1] - rth_slice(F_I, ILLUS)[0]),
                im=IM_I, p_target=PT_I, open=O_I, lines_near_7686=near,
                top6=[x["L"] for x in LV_I],
                tests=[dict(L=r["L"], t=hms(ILLUS, r["t"]), side=r["side"]) for r in T_I],
                episodes=[dict(L=e["L"], test_t=hms(ILLUS, e["t"]), side=e["side"], status=e["status"],
                               O=e["O"]) for e in E_I],
                table=table_rows(E_I, ILLUS, IM_I, F_I))
Q4_ILLUS["flow_gaps_gt60s"] = [(hms(ILLUS, F_I["t"][k]), int(F_I["t"][k + 1] - F_I["t"][k]))
                               for k in np.nonzero(np.diff(F_I["t"]) > 60)[0]
                               if pt(ILLUS, 6, 30).timestamp() <= F_I["t"][k] < pt(ILLUS, 13, 0).timestamp()]
del F_I
# 1-minute proxy for the illustration (flow hole); minute bars as bars, t = minute start
bm = q("""SELECT datetime, open::float o, high::float h, low::float l, close::float c FROM ironbeam_es_1m_bars
          WHERE datetime >= %s AND datetime < %s ORDER BY datetime""",
       (pt(ILLUS, 6, 30).replace(tzinfo=None), pt(ILLUS, 13, 0).replace(tzinfo=None)))
F_M = dict(t=np.array([int(pd.Timestamp(x).tz_localize("UTC").timestamp()) for x in bm["datetime"]], np.int64),
           o=bm["o"].to_numpy(), h=bm["h"].to_numpy(), l=bm["l"].to_numpy(), c=bm["c"].to_numpy())
O_M = rth_open(F_M, ILLUS)
T_M = all_tests({ILLUS: near}, {ILLUS: F_M}, {ILLUS: IM_I}, [ILLUS])
E_M = run_detector(T_M, {ILLUS: F_M}, {ILLUS: IM_I}, O_of_c, CENTRE[1], CENTRE[2], CENTRE[3])
Q4_ILLUS["minute_proxy"] = dict(bars=len(F_M["t"]), open=O_M, table=table_rows(E_M, ILLUS, IM_I, F_M))
# readout only: the same proxy with the confirmation pullback = P alone (no 0.10·IM floor)
E_M0 = run_detector(T_M, {ILLUS: F_M}, {ILLUS: IM_I}, O_of_c, CENTRE[1], CENTRE[2], CENTRE[3], floor_im=0.0)
Q4_ILLUS["minute_proxy_no_im_floor"] = table_rows(E_M0, ILLUS, IM_I, F_M)


# --------------------------------------------------------------------------------------------- Q5 A/A
def hold_to_close(F, d, fi, entry, Hx, side, stop_buf):
    """Net P&L (pts) of a filled setup held to the last RTH print, L8 stop. Returns (pnl, exit idx)."""
    _, b = rth_slice(F, d)
    H, Lo, sg = space(F, side)
    stp = sg * Hx + stop_buf
    hit = np.nonzero(H[fi:b] >= stp)[0]
    e_s = sg * entry
    if len(hit):
        k = fi + int(hit[0])
        return e_s - (stp + TICK) - COST_PTS, k
    return e_s - sg * F["c"][b - 1] - COST_PTS, b - 1


PS = [x for x in CENTRE_EPS["placebo"] if x["status"] == "filled" and x["pre_gap"] <= GAP_MAX]
pl = []
for x in PS:
    F = FLOW[x["d"]]
    pnl, k = hold_to_close(F, x["d"], x["fill_i"], x["entry"], x["H"], x["side"], STOP_CENTRE)
    if max_gap(F, x["d"], x["fill_t"], int(F["t"][k])) > GAP_MAX:
        continue
    pl.append(dict(d=x["d"], L=x["L"], pnl=pnl))
PL = pd.DataFrame(pl)
keys = sorted({(r.d, r.L) for r in PL.itertuples()})
rs, rb = np.random.default_rng(SEED_DRAW), np.random.default_rng(SEED_BOOT)
nS = len(USABLE)
PL["s"] = [sidx[d] for d in PL["d"]]
stats, his, excl = [], [], 0
for _ in range(N_SPLIT):
    perm = rs.permutation(len(keys))
    half1 = {keys[k] for k in perm[: len(keys) // 2]}
    g = np.array([(r.d, r.L) in half1 for r in PL.itertuples()])
    c = np.zeros((nS, 2, 2))
    np.add.at(c, (PL["s"].to_numpy(), (~g).astype(int), 0), PL["pnl"].to_numpy())
    np.add.at(c, (PL["s"].to_numpy(), (~g).astype(int), 1), 1)
    tot_ = c.sum(0)
    st = tot_[0, 0] / tot_[0, 1] - tot_[1, 0] / tot_[1, 1]
    Wb = rb.multinomial(nS, np.full(nS, 1 / nS), size=N_BOOT_SPLIT).astype(float)
    bc = np.einsum("bs,sij->bij", Wb, c)
    with np.errstate(invalid="ignore", divide="ignore"):
        bst = bc[:, 0, 0] / bc[:, 0, 1] - bc[:, 1, 0] / bc[:, 1, 1]
    lo, hi = np.nanpercentile(bst, [2.5, 97.5])
    stats.append(st)
    his.append((hi - lo) / 2)
    excl += int(lo > 0 or hi < 0)
stats = np.array(stats)
q5 = dict(placebo_filled_centre=len(PS), after_exit_gap_drop=len(PL), pseudo_levels=len(keys),
          median=round(float(np.median(stats)), 3), mean=round(float(stats.mean()), 3),
          sd=round(float(stats.std(ddof=1)), 3),
          pcts={p: round(float(np.percentile(stats, p)), 2) for p in (5, 25, 75, 95)},
          ci_halfwidth_median=round(float(np.median(his)), 3),
          sd_over_hw=round(float(stats.std(ddof=1) / (np.median(his) / 1.96)), 3),
          excl_share=excl / N_SPLIT)
q5["PASS"] = bool(q5["excl_share"] <= 0.10 and abs(q5["median"]) <= 3)

# ------------------------------------------------------------------------------------------------ Q1
lv_n = pd.Series([len(LEVELS[d]) for d in USABLE])
q1 = dict(funnel=funnel, n=len(USABLE), first=str(USABLE[0]), last=str(USABLE[-1]),
          lines_in_window=dict(median=float(np.median(list(WINLINES.values())))),
          levels=dict(total=int(lv_n.sum()), median=float(lv_n.median()), q25=float(lv_n.quantile(.25)),
                      q75=float(lv_n.quantile(.75)), min=int(lv_n.min()),
                      put=sum(x["sign"] == "put" for v in LEVELS.values() for x in v),
                      call=sum(x["sign"] == "call" for v in LEVELS.values() for x in v)),
          placebo=dict(total=sum(len(v) for v in PLV.values()),
                       put=sum(x["sign"] == "put" for v in PLV.values() for x in v),
                       call=sum(x["sign"] == "call" for v in PLV.values() for x in v),
                       median=float(np.median([len(PLV[d]) for d in USABLE]))),
          symbols=sorted(set(SYM[d] for d in USABLE)))

RES = dict(q1=q1, q2=dict(cells=q2, put_vs_call=q2_diff, O_c={f"{k[0]}/{k[1]}": v for k, v in OC.items()},
                          ogap_dropped=int((TRd["ogap"] > GAP_MAX).sum())),
           q3={k: v for k, v in q3.items()}, q4=dict(days=[str(d) for d in Q4_DAYS], table=Q4),
           q4_illus=Q4_ILLUS, q5=q5)
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "step0.json").write_text(json.dumps(RES, indent=1, default=str))
print(json.dumps({k: v for k, v in RES.items() if k not in ("q4",)}, indent=1, default=str))
print("\nQ4 table (centre config; no outcomes)")
print(pd.DataFrame(Q4).to_string(index=False))
print("\nQ4 2026-10-23 illustration")
print(pd.DataFrame(Q4_ILLUS["table"]).to_string(index=False))
print("\nQ4 2026-10-23 1-minute proxy (flow hole)")
print(pd.DataFrame(Q4_ILLUS["minute_proxy"]["table"]).to_string(index=False))
print("\nQ4 2026-10-23 1-minute proxy, no 0.10·IM pullback floor (readout)")
print(pd.DataFrame(Q4_ILLUS["minute_proxy_no_im_floor"]).to_string(index=False))
print("\nQ5 A/A:", "PASS" if q5["PASS"] else "FAIL", q5)
