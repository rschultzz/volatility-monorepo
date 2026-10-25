#!/usr/bin/env python3
"""CR-BX Step 0 — capitulation (family A) and move → box → break (family B). READ-ONLY: SELECTs only.

Spec: specs/CR-BX-capitulation-seller-exhaustion.md (Step 0 gates 1–7; kickoff: k from family A counts only,
drop — do not loosen — a family under 150 per tercile, S11(b) box counts).

NO race is run at any real event. Races run only on placebo entries / placebo boxes inside the A/A splits, and
only the split-statistic distribution is printed.

Session universe, flow loader, gap rule and fill model are copied from the CR-BW run script
(scripts/cr_bw_absorption.py on the unmerged branch feat/CR-BW-aggressor-absorption-at-levels, commit 51f3e3a).

Definitions fixed here before the run:
  - connection: get_backfill_db_conn() + default_transaction_read_only = on (asserted); statement_timeout 600 s;
    Python alarm 3 h. Holdout guard: sessions <= 2026-06-05; flow ts_utc < 2026-06-05 21:00 UTC.
  - sessions (L3, CR-BW A2 universe): NYSE trading days 2025-12-26 .. 2026-06-05 minus L14 (a) roll week,
    (b) no ES 15:45 ET bar on A, (c) roll mismatch; flow rows with an RTH open; canonical bt_daily_features
    IM > 0 and orats_oi_gamma rows (kept so the universe equals CR-BW's and S4/S9 levels exist);
    >= 5 prior scale sessions (L4). Scale sessions = L14-passing sessions with flow (CR-BW A1.5 set).
  - flow: ironbeam_es_flow_1s rows of the session's single RTH symbol; a row = a second with >= 1 trade.
    Volume = the `volume` column (buy + sell + unknown).
  - L4 σw_d (w = 5, 10, 30 min): median |close(last row of window) − close(last row before window)| over
    non-overlapping RTH windows of w minutes from 06:30 PT in the prior >= 5 (up to 20) scale sessions; a window
    is dropped if it holds an RTH flow gap > 15 s.
  - gap (CR-BW A1.4): largest of (first row − t0), consecutive row diffs, (t1 − last row) in [t0, t1], t0
    clipped to 06:30 PT, t1 to 13:00 PT.
  - L5 condition at row t (06:35:00 <= t <= 12:45:00 PT): lo300/hi300 = min low / max high over rows in
    [t − 300, t]. Down: low(t) = lo300 and low(t) <= hi300 − k·σ5_d. Up: high(t) = hi300 and
    high(t) >= lo300 + k·σ5_d. Events: condition rows in time order, one per direction per 15 min (900 s
    after the previous accepted event of that direction).
  - family A eligibility: pre-window [t − 300, t] gap <= 15 s and D(300, 0) > 0. The race-window part of the gap
    rule needs the race end (an outcome), so Step 0 applies it only inside the A/A.
  - L7 D(a, b) = sum over rows with t − a <= time <= t − b of (sell − buy) for a down move, (buy − sell) for an
    up move. X = D(60, 0) / (D(300, 0) / 5). S7: X30 = D(30, 0) / (D(300, 0) / 10).
  - L5 k: the largest of 3.0, 2.5, 2.0 with >= 150 eligible family A events in every X tercile (terciles over
    that k's eligible events). Family B uses the same k; dropped if any P tercile < 150 eligible boxes.
  - L6 trade (A/A only): fade. Entry at the open of the next row after t, + 1 tick against (long: open + 0.25;
    short: open − 0.25). Long target E + R, hit on high >= E + R + 0.25; stop E − R, hit on low <= E − R; short
    mirrored. From the row after the entry row; the entry row counts a stop, not a target; same row both -> loss;
    neither by 13:00 PT -> unresolved. R = 8 ticks.
  - L9 placebo entries: rows s with |s − t| <= 1800, 06:35 <= s <= 12:45, no L5 condition row (either direction,
    same k) in [s − 300, s], pre-window gap <= 15 s; 5 drawn without replacement, default_rng(20261025),
    events in time order. Trade direction = the event's.
  - L12 box: impulses = family A events (all, before eligibility). Box windows are rows e with the window
    (e − 180, e]; range = max high − min low of its rows. First e with e − 180 >= t, e <= t + 900, e <= 12:45 PT
    and range <= 0.5·σ5_d. t_B = e. Box = [min low, max high].
  - L14 P = D_box / V_box over the window rows, D signed in the impulse direction. Box eligibility: gap <= 15 s
    over [min(box start, t_B − 300), t_B] and V_box > 0. Terciles of P over eligible boxes.
  - L15 placebo boxes: rows e' with |e' − t_B| <= 1800, 06:35 <= e' <= 12:45, range rule, no L5 condition row
    (either direction) in [e' − 900, e'], window (e' − 180, e'] not overlapping the real box window, the same
    gap rule; up to 5 drawn without replacement with pairwise |e'1 − e'2| >= 180, default_rng(20261025) fresh.
    Placebo impulse direction = sign(close at box start − close 300 s earlier); 0 -> dropped.
  - L13 race (A/A only): from the row after t_B; reversal = a trade 8 ticks beyond the box boundary against the
    impulse (down impulse: high >= box_high + 2.0), continuation = 8 ticks beyond with it (low <= box_low − 2.0);
    same row both -> neither (excluded); unresolved by 13:00 excluded. Race-window gap rule applied.
  - S11(b) counts: σ10_d / σ30_d; windows of 600 s / 1800 s, range <= 0.5·σ10_d / 0.5·σ30_d, e − w >= t,
    e <= t + 3600, e <= 12:45; same eligibility; own P terciles.
  - Gate 6 A/A (CR-BW A1.3): 200 splits of the placebo entries (A) / placebo boxes (B) into two halves by
    parent event (shuffle events, default_rng(20261025) fresh per family, first half = ceil(n/2)); statistic
    A: [low − high X-tercile win rate](half 1) − (half 2); B: [high − low P-tercile reversal rate](half 1) −
    (half 2), each using the parent's tercile. Session-cluster bootstrap, 2 000 per split, one
    default_rng(20260920) stream per family. Pass: CI excludes 0 in <= 10 % of splits and |median| <= 3 pts.
  - Gate 5 worked event: default_rng(20261025) fresh, uniform over eligible family A events at the chosen k.

Usage: apps/web/.venv/bin/python -u scripts/cr_bx_step0.py | tee scripts/logs/cr_bx/step0.log
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
from packages.shared.level_placement import es_snapshot_bar_start_utc  # noqa: E402
from packages.shared.trading_calendar import is_trading_day, next_trading_day, prev_trading_day  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 200)
signal.alarm(3 * 3600)

PT, UTC = ZoneInfo("America/Los_Angeles"), timezone.utc
D_FROM, D_TO = date(2025, 12, 26), date(2026, 6, 5)
HOLDOUT_TS = datetime(2026, 6, 5, 21, 0, tzinfo=UTC)
TICK, GAP_MAX = 0.25, 15
K_LIST = (3.0, 2.5, 2.0)
COOLDOWN, PLC_SPAN, PLC_N = 900, 1800, 5
BOX_W, BOX_SEARCH, BOX_FRAC = 180, 900, 0.5
R_TICKS, BREAK_TICKS = 8, 8
SCALE_N, SCALE_MIN = 20, 5
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
    return datetime.fromtimestamp(int(t), PT).strftime("%H:%M:%S")


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
        SELECT ts_utc, volume, buy_vol, sell_vol, open, high, low, close
        FROM ironbeam_es_flow_1s
        WHERE symbol = %s AND ts_utc >= %s AND ts_utc < %s AND ts_utc < %s
        ORDER BY ts_utc
    """, (SYM[d], pt(d, 6, 30), pt(d, 13, 0), HOLDOUT_TS))
    if f.empty:
        continue
    FLOW[d] = dict(t=np.array([ep(x) for x in f["ts_utc"]], dtype=np.int64),
                   vol=f["volume"].to_numpy(np.int64), buy=f["buy_vol"].to_numpy(np.int64),
                   sell=f["sell_vol"].to_numpy(np.int64), o=f["open"].to_numpy(float),
                   h=f["high"].to_numpy(float), l=f["low"].to_numpy(float), c=f["close"].to_numpy(float))
print(f"flow loaded (RTH) for {len(FLOW)} sessions; rows {sum(len(v['t']) for v in FLOW.values()):,}")

oig_days = set(q("""SELECT DISTINCT trade_date FROM orats_oi_gamma
                    WHERE ticker = 'SPX' AND trade_date BETWEEN %s AND %s""",
                 (D_FROM - timedelta(days=7), D_TO))["trade_date"])
oig_days = {t if is_trading_day(t) else next_trading_day(t) for t in oig_days}
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


def max_gap(d, t0, t1):
    f = FLOW[d]["t"]
    t0 = max(int(t0), ep(pt(d, 6, 30)))
    t1 = min(int(t1), ep(pt(d, 13, 0)))
    if t1 <= t0:
        return 0
    a, b = np.searchsorted(f, t0), np.searchsorted(f, t1, side="right")
    return int(np.diff(np.concatenate([[t0], f[a:b], [t1]])).max())


ex_a = {d for d in SESSIONS if roll_week(d)}
ex_b = {d for d in SESSIONS if d not in PTGT}
ex_c = {d for d in SESSIONS if d in QEXP or d in ROLL_MISMATCH_EXTRA}
SCALE_ELIG = [d for d in SESSIONS if d not in ex_a | ex_b | ex_c and d in FLOW]


def abs_changes(d, w):
    f = FLOW[d]
    t0 = ep(pt(d, 6, 30))
    out = []
    for k in range(23400 // w):
        ws, we = t0 + w * k, t0 + w * (k + 1)
        a, b = np.searchsorted(f["t"], ws), np.searchsorted(f["t"], we)
        if a == 0 or b <= a or max_gap(d, ws, we) > GAP_MAX:
            continue
        out.append(abs(f["c"][b - 1] - f["c"][a - 1]))
    return np.array(out)


CH = {w: {d: abs_changes(d, w) for d in SCALE_ELIG} for w in (300, 600, 1800)}
SIG = {w: {} for w in CH}
for d in SESSIONS:
    prior = [x for x in SCALE_ELIG if x < d][-SCALE_N:]
    if len(prior) >= SCALE_MIN:
        for w in CH:
            SIG[w][d] = float(np.median(np.concatenate([CH[w][x] for x in prior])))

# ------------------------------------------------------------------------------------------ gate 1
hdr("GATE 1 — UNIVERSE")
U = list(SESSIONS)
print(f"{'rule':44s} {'removed':>8s} {'remaining':>10s}\n{'trading sessions':44s} {'':>8s} {len(U):>10d}")
for name, rule in [
    ("L14a roll week", lambda d: d in ex_a),
    ("L14b no ES 15:45 ET bar on A", lambda d: d in ex_b),
    ("L14c roll mismatch", lambda d: d in ex_c),
    ("no flow rows / no RTH open", lambda d: d not in FLOW),
    ("no canonical bt_daily_features row", lambda d: d not in feat.index),
    ("IM null or <= 0", lambda d: d in feat.index and not (pd.notna(feat[d]) and feat[d] > 0)),
    ("no orats_oi_gamma rows", lambda d: d not in oig_days),
    ("< 5 prior scale sessions (L4)", lambda d: d not in SIG[300]),
]:
    rm = [d for d in U if rule(d)]
    U = [d for d in U if not rule(d)]
    print(f"{name:44s} {len(rm):>8d} {len(U):>10d}   {[str(x) for x in rm] if len(rm) <= 12 else ''}")
USABLE = U
print(f"USABLE SESSIONS: {len(USABLE)} ({USABLE[0]} .. {USABLE[-1]})")
for w in (300, 600, 1800):
    s = pd.Series({d: SIG[w][d] for d in USABLE})
    r = (s / s.shift(1)).dropna()
    print(f"σ{w // 60}_d (pts): median {s.median():.3f}, IQR {s.quantile(.25):.3f}–{s.quantile(.75):.3f}, range "
          f"{s.min():.3f}–{s.max():.3f}; day ratio p5/p95 {r.quantile(.05):.3f}/{r.quantile(.95):.3f}")
S5 = pd.Series({d: SIG[300][d] for d in USABLE})
print("σ5_d by month:", S5.groupby([d.strftime("%Y-%m") for d in S5.index]).median().round(3).to_dict())

# ------------------------------------------------------------------------------------- per-session prep
PREP = {}
for d in USABLE:
    f = FLOW[d]
    idx = pd.to_datetime(f["t"], unit="s")
    lo300 = pd.Series(f["l"], idx).rolling("300s", closed="both").min().to_numpy()
    hi300 = pd.Series(f["h"], idx).rolling("300s", closed="both").max().to_numpy()
    rng_w = {}
    for w in (180, 600, 1800):
        hi = pd.Series(f["h"], idx).rolling(f"{w}s").max().to_numpy()
        lo = pd.Series(f["l"], idx).rolling(f"{w}s").min().to_numpy()
        rng_w[w] = (hi - lo, hi, lo)
    inwin = (f["t"] >= ep(pt(d, 6, 35))) & (f["t"] <= ep(pt(d, 12, 45)))
    cum = dict(net=np.concatenate([[0], np.cumsum(f["buy"] - f["sell"])]),
               vol=np.concatenate([[0], np.cumsum(f["vol"])]))
    PREP[d] = dict(lo300=lo300, hi300=hi300, rng=rng_w, inwin=inwin, cum=cum)


def net_between(d, t0, t1):
    """buy − sell over rows with t0 <= time <= t1."""
    f, c = FLOW[d]["t"], PREP[d]["cum"]["net"]
    a, b = np.searchsorted(f, t0), np.searchsorted(f, t1, side="right")
    return int(c[b] - c[a])


def vol_between(d, t0, t1):
    f, c = FLOW[d]["t"], PREP[d]["cum"]["vol"]
    a, b = np.searchsorted(f, t0), np.searchsorted(f, t1, side="right")
    return int(c[b] - c[a])


def conditions(d, k):
    f, p = FLOW[d], PREP[d]
    s = SIG[300][d]
    down = p["inwin"] & (f["l"] == p["lo300"]) & (f["l"] <= p["hi300"] - k * s)
    up = p["inwin"] & (f["h"] == p["hi300"]) & (f["h"] >= p["lo300"] + k * s)
    return down, up


def family_a(k):
    ev, cond_any = [], {}
    for d in USABLE:
        f = FLOW[d]
        down, up = conditions(d, k)
        cond_any[d] = f["t"][down | up]
        for dirn, mask in ((-1, down), (1, up)):
            last = -10**12
            for i in np.nonzero(mask)[0]:
                t = int(f["t"][i])
                if t - last < COOLDOWN:
                    continue
                last = t

                D300 = dirn * net_between(d, t - 300, t)
                D60 = dirn * net_between(d, t - 60, t)
                D30 = dirn * net_between(d, t - 30, t)
                ev.append(dict(d=d, i=int(i), t=t, dirn=dirn, D300=D300, D60=D60, D30=D30,
                               X=D60 / (D300 / 5) if D300 > 0 else np.nan,
                               X30=D30 / (D300 / 10) if D300 > 0 else np.nan,
                               pre_gap=max_gap(d, t - 300, t),
                               mag=(PREP[d]["hi300"][i] - f["l"][i]) / SIG[300][d] if dirn == -1
                               else (f["h"][i] - PREP[d]["lo300"][i]) / SIG[300][d]))
    E = pd.DataFrame(ev)
    E["elig"] = (E["pre_gap"] <= GAP_MAX) & (E["D300"] > 0)
    return E, cond_any


def terc_of(x, cut):
    return np.where(x <= cut[0], 0, np.where(x <= cut[1], 1, 2))


# ------------------------------------------------------------------------------------------ gate 2 A
hdr("GATE 2 — COUNTS (no outcomes): FAMILY A")
FA, COND = {}, {}
for k in K_LIST:
    E, cnd = family_a(k)
    FA[k], COND[k] = E, cnd
    el = E[E["elig"]]
    cut = np.percentile(el["X"], [100 / 3, 200 / 3]) if len(el) else [np.nan, np.nan]
    tc = np.bincount(terc_of(el["X"].to_numpy(), cut), minlength=3) if len(el) else [0, 0, 0]
    per = E.groupby("d").size().reindex(USABLE, fill_value=0)
    print(f"k = {k}: events {len(E)} (down {int((E['dirn'] == -1).sum())}, up {int((E['dirn'] == 1).sum())}); "
          f"per session median {per.median():.0f}, IQR {per.quantile(.25):.0f}–{per.quantile(.75):.0f}; "
          f"gap drops (pre-window) {int((E['pre_gap'] > GAP_MAX).sum())}; D(300,0) <= 0 "
          f"{int(((E['pre_gap'] <= GAP_MAX) & (E['D300'] <= 0)).sum())}; eligible {len(el)}; per tercile "
          f"{list(map(int, tc))}")
K = next((k for k in K_LIST if len(FA[k][FA[k]["elig"]]) and
          min(np.bincount(terc_of(FA[k][FA[k]["elig"]]["X"].to_numpy(),
                                  np.percentile(FA[k][FA[k]["elig"]]["X"], [100 / 3, 200 / 3])), minlength=3)) >= 150),
         None)
print(f"\nk CHOSEN (L5, family A counts only): {K}")
if K is None:
    sys.exit("STOP: family A below 150 per tercile at every k — family A dropped; family B has no k")
A = FA[K].copy()
AE = A[A["elig"]].copy()
CUT_A = np.percentile(AE["X"], [100 / 3, 200 / 3])
AE["terc"] = terc_of(AE["X"].to_numpy(), CUT_A)
print(f"X tercile cut points: {CUT_A.round(4)}; S2 move-size split of eligible (|move| / σ5): 2–3σ "
      f"{int(((AE['mag'] >= 2) & (AE['mag'] < 3)).sum())}, 3–4σ {int(((AE['mag'] >= 3) & (AE['mag'] < 4)).sum())}, "
      f">= 4σ {int((AE['mag'] >= 4).sum())}; S3 buy-after-down / sell-after-up "
      f"{int((AE['dirn'] == -1).sum())}/{int((AE['dirn'] == 1).sum())}; S5 before/after 10:00 PT "
      f"{int((AE['t'] < [ep(pt(x, 10, 0)) for x in AE['d']]).sum())}/"
      f"{int((AE['t'] >= [ep(pt(x, 10, 0)) for x in AE['d']]).sum())}")


# ------------------------------------------------------------------------------------------ boxes
def find_boxes(E, w, search, sig_w):
    rows = []
    for x in E.itertuples():
        d, t = x.d, x.t
        f, (rg, hi, lo) = FLOW[d], PREP[d]["rng"][w]
        s = SIG[sig_w][d]
        a = np.searchsorted(f["t"], t + w)
        b = np.searchsorted(f["t"], min(t + search, ep(pt(d, 12, 45))), side="right")
        ok = np.nonzero(rg[a:b] <= BOX_FRAC * s)[0]
        if not len(ok):
            rows.append(dict(d=d, t=t, dirn=x.dirn, found=False))
            continue
        j = a + int(ok[0])
        e = int(f["t"][j])
        dn = x.dirn * net_between(d, e - w + 1, e)
        v = vol_between(d, e - w + 1, e)
        rows.append(dict(d=d, t=t, dirn=x.dirn, found=True, j=j, tB=e, box_hi=hi[j], box_lo=lo[j],
                         range=hi[j] - lo[j], wait=e - t, V=v, P=dn / v if v > 0 else np.nan,
                         gap=max_gap(d, min(e - w, e - 300), e)))
    Bx = pd.DataFrame(rows)
    for col in ("gap", "V", "P", "range", "wait"):
        if col not in Bx:
            Bx[col] = np.nan
    Bx["elig"] = Bx["found"] & (Bx["gap"] <= GAP_MAX) & (Bx["V"] > 0)
    return Bx


hdr(f"GATE 2 — COUNTS (no outcomes): FAMILY B at k = {K}")
BX = find_boxes(A, BOX_W, BOX_SEARCH, 300)
BE = BX[BX["elig"]].copy()
CUT_B = np.percentile(BE["P"], [100 / 3, 200 / 3])
BE["terc"] = terc_of(BE["P"].to_numpy(), CUT_B)
tcB = np.bincount(BE["terc"], minlength=3)
perB = BE.groupby("d").size().reindex(USABLE, fill_value=0)
print(f"impulses {len(BX)}; with a qualifying box {int(BX['found'].sum())}; gap drops "
      f"{int((BX['found'] & (BX['gap'] > GAP_MAX)).sum())}; V = 0 {int((BX['found'] & (BX['V'] <= 0)).sum())}; "
      f"eligible boxes {len(BE)}; per session median {perB.median():.0f}; P cut points {CUT_B.round(4)}; per tercile "
      f"{tcB.tolist()}")
FAM_B = bool(tcB.min() >= 150)
print(f"family B: {'KEPT' if FAM_B else 'DROPPED (< 150 per tercile at k; not loosened)'}")

hdr("GATE 3 — X DISTRIBUTION (family A eligible)")
print(AE["X"].describe(percentiles=[.05, .1, .25, .5, .75, .9, .95]).round(3).to_string())
print(f"X <= 0 (flow flipped): {int((AE['X'] <= 0).sum())} ({(AE['X'] <= 0).mean():.1%}); X < 1 (fading): "
      f"{(AE['X'] < 1).mean():.1%}; S7 X30 median {AE['X30'].median():.3f}")


# ----------------------------------------------------------------------------------- gate 4 placebo A
def placebo_a(E):
    rng = np.random.default_rng(SEED)
    rows, short = [], 0
    for x in E.sort_values(["d", "t"]).itertuples():
        d, t = x.d, x.t
        f = FLOW[d]
        lo_t, hi_t = max(t - PLC_SPAN, ep(pt(d, 6, 35))), min(t + PLC_SPAN, ep(pt(d, 12, 45)))
        a, b = np.searchsorted(f["t"], lo_t), np.searchsorted(f["t"], hi_t, side="right")
        cand = f["t"][a:b]
        ca = COND[K][d]
        # no L5 condition row in [s − 300, s]
        k1 = np.searchsorted(ca, cand - 300)
        k2 = np.searchsorted(ca, cand, side="right")
        cand_i = np.arange(a, b)[(k2 - k1) == 0]
        cand_i = np.array([i for i in cand_i if max_gap(d, f["t"][i] - 300, f["t"][i]) <= GAP_MAX], dtype=int)
        if len(cand_i) < PLC_N:
            short += 1
        pick = rng.choice(cand_i, size=min(PLC_N, len(cand_i)), replace=False) if len(cand_i) else []
        for i in pick:
            rows.append(dict(d=d, parent=(d, t), i=int(i), t=int(f["t"][i]), dirn=x.dirn, terc=x.terc))
    return pd.DataFrame(rows), short


hdr("GATE 4 — PLACEBO FEASIBILITY")
PA, shortA = placebo_a(AE)
nA = PA.groupby("parent").size()
print(f"family A: placebo entries {len(PA)} for {len(AE)} eligible events; per event median {nA.median():.0f}; "
      f"events with < 5 eligible placebo seconds {shortA}")


def placebo_b(Bx, w=BOX_W, sig_w=300):
    rng = np.random.default_rng(SEED)
    rows, short = [], 0
    for x in Bx.sort_values(["d", "tB"]).itertuples():
        d = x.d
        f, (rg, hi, lo) = FLOW[d], PREP[d]["rng"][w]
        s = SIG[sig_w][d]
        lo_t, hi_t = max(x.tB - PLC_SPAN, ep(pt(d, 6, 35))), min(x.tB + PLC_SPAN, ep(pt(d, 12, 45)))
        a, b = np.searchsorted(f["t"], lo_t), np.searchsorted(f["t"], hi_t, side="right")
        idx = np.arange(a, b)
        idx = idx[rg[a:b] <= BOX_FRAC * s]
        te = f["t"][idx]
        idx, te = idx[np.abs(te - x.tB) >= w], te[np.abs(te - x.tB) >= w]       # no overlap with the real box
        ca = COND[K][d]
        k1, k2 = np.searchsorted(ca, te - 900), np.searchsorted(ca, te, side="right")
        idx, te = idx[(k2 - k1) == 0], te[(k2 - k1) == 0]
        ok = []
        for i, e in zip(idx, te):
            if max_gap(d, min(e - w, e - 300), e) > GAP_MAX or vol_between(d, e - w + 1, e) <= 0:
                continue
            a0 = np.searchsorted(f["t"], e - w, side="right") - 1          # last row at or before box start
            a1 = np.searchsorted(f["t"], e - w - 300, side="right") - 1
            if a0 < 0 or a1 < 0 or f["c"][a0] == f["c"][a1]:
                continue
            ok.append((int(i), int(e), int(np.sign(f["c"][a0] - f["c"][a1]))))
        order = rng.permutation(len(ok))
        chosen = []
        for k_ in order:
            if all(abs(ok[k_][1] - c[1]) >= w for c in chosen):
                chosen.append(ok[k_])
            if len(chosen) == PLC_N:
                break
        if len(chosen) < PLC_N:
            short += 1
        for i, e, dirn in chosen:
            dn = dirn * net_between(d, e - w + 1, e)
            rows.append(dict(d=d, parent=(d, x.tB), j=i, tB=e, dirn=dirn, box_hi=hi[i], box_lo=lo[i],
                             P=dn / vol_between(d, e - w + 1, e), terc=x.terc))
    return pd.DataFrame(rows), short


PB, shortB = placebo_b(BE)
nB = PB.groupby("parent").size().reindex(list(zip(BE["d"], BE["tB"])), fill_value=0)
print(f"family B: placebo boxes {len(PB)} for {len(BE)} eligible boxes; per box median {nB.median():.0f}, "
      f"mean {nB.mean():.2f}; boxes with 0 placebo {int((nB == 0).sum())}; with < 5 {shortB}")

# ------------------------------------------------------------------------------------------ gate 7
hdr("GATE 7 — BOX GEOMETRY (family B eligible)")
print(f"box range: median {BE['range'].median():.2f} pts ({BE['range'].median() / TICK:.0f} ticks), IQR "
      f"{BE['range'].quantile(.25):.2f}–{BE['range'].quantile(.75):.2f}; 0.5·σ5 median "
      f"{0.5 * np.median([SIG[300][d] for d in BE['d']]):.2f}")
print(f"time from impulse to box end: median {BE['wait'].median():.0f} s, IQR {BE['wait'].quantile(.25):.0f}–"
      f"{BE['wait'].quantile(.75):.0f} s (box window fixed at 180 s)")
print(f"boxes narrower than 4 ticks (< 1.00 pt): {int((BE['range'] < 1.0).sum())} ({(BE['range'] < 1.0).mean():.1%})")

hdr("S11(b) — LONGER BOX COUNTS (no outcomes)")
for w, sw in ((600, 600), (1800, 1800)):
    bx = find_boxes(A, w, 3600, sw)
    be = bx[bx["elig"]]
    cut = np.percentile(be["P"], [100 / 3, 200 / 3]) if len(be) >= 3 else [np.nan, np.nan]
    tc = np.bincount(terc_of(be["P"].to_numpy(), cut), minlength=3).tolist() if len(be) >= 3 else [len(be), 0, 0]
    print(f"{w // 60}-min window (range <= 0.5·σ{sw // 60}): impulses {len(bx)}; found {int(bx['found'].sum())}; "
          f"gap drops {int((bx['found'] & (bx['gap'] > GAP_MAX)).sum())}; eligible {len(be)}; per tercile {tc}; "
          f"median range {be['range'].median() if len(be) else float('nan'):.2f} pts; median wait "
          f"{be['wait'].median() if len(be) else float('nan'):.0f} s")

# ------------------------------------------------------------------------------------------ gate 5
hdr("GATE 5 — WORKED EVENT (family A, random eligible, seed 20261025); outcome hidden")
ev = AE.iloc[int(np.random.default_rng(SEED).integers(len(AE)))]
d, t, i = ev["d"], int(ev["t"]), int(ev["i"])
f = FLOW[d]
print(f"session {d}; symbol {SYM[d]}; σ5_d {SIG[300][d]:.3f}; k {K}; event {hms(t)} PT, "
      f"{'DOWN move → fade long' if ev['dirn'] == -1 else 'UP move → fade short'}")
print(f"300 s window: low {PREP[d]['lo300'][i]:.2f}, high {PREP[d]['hi300'][i]:.2f}; move "
      f"{ev['mag']:.2f} σ5 ({ev['mag'] * SIG[300][d]:.2f} pts)")
w0 = np.searchsorted(f["t"], t - 300)
path = pd.DataFrame(dict(t=f["t"][w0:i + 1], buy=f["buy"][w0:i + 1], sell=f["sell"][w0:i + 1],
                         lo=f["l"][w0:i + 1], hi=f["h"][w0:i + 1], c=f["c"][w0:i + 1]))
path["slot"] = (path["t"] - (t - 300)) // 30
agg = path.groupby("slot").agg(buy=("buy", "sum"), sell=("sell", "sum"), low=("lo", "min"), high=("hi", "max"),
                               close=("c", "last"))
agg.index = [f"-{300 - 30 * k}s..-{max(300 - 30 * (k + 1), 0)}s" for k in agg.index]
print(agg.to_string())
print(f"D(300,0) {ev['D300']:,}; D(60,0) {ev['D60']:,}; D(30,0) {ev['D30']:,}; X {ev['X']:.3f} (tercile "
      f"{['low/exhausted', 'mid', 'high'][int(ev['terc'])]}); X30 {ev['X30']:.3f}")
nx = i + 1
print(f"entry second {hms(f['t'][nx])} PT, open {f['o'][nx]:.2f} → entry {f['o'][nx] + (TICK if ev['dirn'] == -1 else -TICK):.2f} "
      f"({'buy' if ev['dirn'] == -1 else 'sell'}, +1 tick slippage); target/stop ±{R_TICKS} ticks")


# ------------------------------------------------------------------------------------------ gate 6
def race_a(d, i_ev, dirn):
    """L6 fade from the row after i_ev. (1 win / 0 loss / -1 unresolved, entry t, end t). A/A only."""
    f = FLOW[d]
    n = len(f["t"])
    e_i = i_ev + 1
    if e_i >= n:
        return -1, None, None
    long = dirn == -1
    E = f["o"][e_i] + (TICK if long else -TICK)
    Rp = R_TICKS * TICK
    if (long and f["l"][e_i] <= E - Rp) or (not long and f["h"][e_i] >= E + Rp):
        return 0, int(f["t"][e_i]), int(f["t"][e_i])
    if long:
        tgt, stp = f["h"][e_i + 1:] >= E + Rp + TICK, f["l"][e_i + 1:] <= E - Rp
    else:
        tgt, stp = f["l"][e_i + 1:] <= E - Rp - TICK, f["h"][e_i + 1:] >= E + Rp
    ti, si = np.nonzero(tgt)[0], np.nonzero(stp)[0]
    if not len(ti) and not len(si):
        return -1, int(f["t"][e_i]), ep(pt(d, 13, 0))
    k = min(ti[0] if len(ti) else 10**9, si[0] if len(si) else 10**9)
    return (1 if len(ti) and (not len(si) or ti[0] < si[0]) else 0), int(f["t"][e_i]), int(f["t"][e_i + 1 + k])


def race_b(d, j, dirn, bhi, blo):
    """L13 from the row after j: 1 reversal / 0 continuation / -1 excluded (tie or unresolved). A/A only."""
    f = FLOW[d]
    Bp = BREAK_TICKS * TICK
    up_hit = f["h"][j + 1:] >= bhi + Bp
    dn_hit = f["l"][j + 1:] <= blo - Bp
    ui, di = np.nonzero(up_hit)[0], np.nonzero(dn_hit)[0]
    if not len(ui) and not len(di):
        return -1, ep(pt(d, 13, 0))
    fu, fd = (ui[0] if len(ui) else 10**9), (di[0] if len(di) else 10**9)
    if fu == fd:
        return -1, int(f["t"][j + 1 + fu])
    rev_up = dirn == -1                     # down impulse: reversal = upside break
    first_up = fu < fd
    return (1 if first_up == rev_up else 0), int(f["t"][j + 1 + min(fu, fd)])


def aa(df, res_col, hi_terc, lo_terc, label):
    S = {d: k for k, d in enumerate(USABLE)}
    df = df[df[res_col] >= 0].copy()
    df["sidx"] = [S[d] for d in df["d"]]
    parents = sorted(set(df["parent"]))
    pidx = {p: k for k, p in enumerate(parents)}
    df["pidx"] = [pidx[p] for p in df["parent"]]
    nS = len(USABLE)
    rng, brng = np.random.default_rng(SEED), np.random.default_rng(SEED_BOOT)

    def counts(m):
        c = np.zeros((nS, 4))
        sub = df[m]
        for col, (tt, what) in enumerate([(hi_terc, "w"), (hi_terc, "n"), (lo_terc, "w"), (lo_terc, "n")]):
            s = sub[sub["terc"] == tt]
            c[:, col] = np.bincount(s["sidx"], weights=s[res_col] if what == "w" else None, minlength=nS)
        return c

    def diff(c):
        with np.errstate(invalid="ignore", divide="ignore"):
            return c[..., 0] / c[..., 1] - c[..., 2] / c[..., 3]

    st, excl = [], 0
    for _ in range(N_SPLIT):
        perm = rng.permutation(len(parents))
        h1 = np.zeros(len(parents), bool)
        h1[perm[: (len(parents) + 1) // 2]] = True
        m1 = h1[df["pidx"].to_numpy()]
        c1, c2 = counts(m1), counts(~m1)
        obs = diff(c1.sum(0)) - diff(c2.sum(0))
        w = np.stack([np.bincount(brng.integers(nS, size=nS), minlength=nS) for _ in range(N_BOOT)]).astype(float)
        lo, hi = np.nanpercentile(diff(w @ c1) - diff(w @ c2), [2.5, 97.5])
        st.append(obs)
        excl += not (lo <= 0 <= hi)
    st = np.array(st) * 100
    share = excl / N_SPLIT
    ok = share <= 0.10 and abs(np.median(st)) <= 3
    print(f"{label}: resolved placebo {len(df)} in {len(parents)} parent events; statistic median {np.median(st):+.2f}, "
          f"mean {st.mean():+.2f}, sd {st.std(ddof=1):.2f}, p5 {np.percentile(st, 5):+.2f}, p95 "
          f"{np.percentile(st, 95):+.2f}; CI excludes 0 in {share:.1%} ({excl}/{N_SPLIT})  -> "
          f"{'PASS' if ok else 'FAIL — STOP'}")
    return ok


hdr("GATE 6 — A/A CALIBRATION (placebo only, R = 8 ticks)")
rr = [race_a(x.d, x.i, x.dirn) for x in PA.itertuples()]
PA["res"] = [r[0] for r in rr]
PA["race_gap"] = [max_gap(x.d, r[1], r[2]) if r[1] is not None else 0 for x, r in zip(PA.itertuples(), rr)]
n0 = len(PA)
PA = PA[PA["race_gap"] <= GAP_MAX]
print(f"family A placebo: race-window gap drops −{n0 - len(PA)}; unresolved −{int((PA['res'] == -1).sum())}")
okA = aa(PA, "res", 0, 2, "FAMILY A (low − high X tercile)")
okB = True
if FAM_B:
    rb = [race_b(x.d, x.j, x.dirn, x.box_hi, x.box_lo) for x in PB.itertuples()]
    PB["res"] = [r[0] for r in rb]
    PB["race_gap"] = [max_gap(x.d, x.tB, r[1]) for x, r in zip(PB.itertuples(), rb)]
    n0 = len(PB)
    PB = PB[PB["race_gap"] <= GAP_MAX]
    print(f"family B placebo: race-window gap drops −{n0 - len(PB)}; excluded (tie/unresolved) "
          f"−{int((PB['res'] == -1).sum())}")
    okB = aa(PB, "res", 2, 0, "FAMILY B (high − low P tercile)")
print(f"\nGATE 6: family A {'PASS' if okA else 'FAIL'}; family B "
      f"{('PASS' if okB else 'FAIL') if FAM_B else 'n/a (dropped)'}")
print("\nDONE")
