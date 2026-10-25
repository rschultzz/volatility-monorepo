#!/usr/bin/env python3
"""CR-BU run — put-node support test under Amendment A1: T1 and S1–S8. READ-ONLY: SELECTs only, no API calls.

Spec: specs/CR-BU-put-node-support-test.md (L1–L14, A1, pass criterion, S1–S7, S8 addendum). ONE RUN.

Run discipline: every statistic is computed and written to scripts/logs/cr_bu/results.json BEFORE anything
is printed, so a crash prints no outcome. Once any outcome has printed there is no re-run (L12).

Universe and node rules (unchanged from scripts/cr_bu_step0.py / cr_bu_step0_a1.py; asserted before outcomes):
  - 433 usable train sessions 2024-04-03 .. 2026-06-05 (L3, L14 a/b/c, canonical IM > 0, SPX-cash open, rows).
  - primary candidates (A1.1): 0–7 DTE `orats_oi_gamma` rows, net = gex_call − |gex_put|, corrected L =
    place_levels(rows, S0) (S0 = MIN stock_price of the nearest expiry >= D), bucket_levels(L, 5); window
    open − 1.5·IM <= bin <= open − 0.25·IM, net < 0; greedy by net ascending (tie: higher bin), >= 10 pts
    apart, at most 3. Asserted: 365 candidate days, 151 touched (Step 0 under A1).
  - event (A1.2): first clean print strictly after the open print, through 13:00 PT, with px <= E_i =
    N_i + 0.05·IM; earliest touch wins; same-minute tie -> highest E.
  - race (L9): from the event print inclusive at the event E; win = first print >= E + R before first print
    <= E − R; neither by 13:00 PT -> unresolved (excluded from rates). One print per minute -> no tie.
  - placebo (A1.3): event day d with z-set {z_i = (open_d − N_i)/IM_d}; every other usable day e gets
    P_i = open_e − z_i·IM_e, E_i = P_i + 0.05·IM_e, the same first-touch rule and race (R scaled by IM_e,
    or the fixed 15 pts). Placebo rate for d = wins / touched resolved e.
  - statistic: mean over event days of (actual − placebo), actual ∈ {0, 1}; placebo = that day's rate.
  - bootstrap (L11): calendar months of the event date, 10 000 resamples, numpy default_rng(20260920),
    95 % percentile; a fresh rng with that seed for every statistic.

T1 (gated): R = 0.20·IM. Pass = diff >= +8 pts AND CI lower bound > 0 AND resolved n >= 150. Nothing else
can change the verdict.

Secondary readouts (not gated), definitions fixed here before the run:
  S1  T1 split by first touch < 10:00 PT vs >= 10:00 PT; each event's placebo counts only e whose
      first touch falls in the same bucket (time-matched).
  S2  approach speed = |px(touch) − px(first print >= touch − 15 min, floored at the open print)| / IM,
      over touched events; placebo = mean over touched e of the same quantity at the pseudo-touch.
  S3  T1 with candidates from (a) 0DTE rows only (expir_date = D), (b) all live expiries (expir_date >= D;
      rows loaded with 0.9·stock_price <= strike <= 1.05·stock_price, which contains every bin the window
      can reach). Each definition carries its own events and z-sets.
  S4  T1 with candidates from old placement: L_old = discounted_level × S0 ÷ ES, ES = close of the
      ironbeam_es_1m_bars bar starting 15:45 America/New_York on A (CR-BT L4); 0–7 DTE rows, same rules.
  S5  monetization: primary events with next_trading_day(D) <= 2026-06-05 (no holdout read). Node-bin rows
      with expir_date > D; ΔputOI = Σ put_oi on session D+1's row (same strike, expiry) − Σ put_oi on D's row,
      rows missing on D+1 skipped (counted); rel = Δ / Σ put_oi(D) (rows with Σ put_oi(D) > 0).
      Bounce (win) vs non-bounce (loss) among T1-resolved events: mean / median rel, mean diff + CI.
      Matched-distance control: every usable non-event day e (no primary event, D+1 <= train end); for each
      event's touched z, bin = bucket_levels(open_e − z·IM_e, 5) over e's corrected 0–7 DTE rows with
      expir_date > e; control_d = mean rel over e with a measurable bin; event − control mean diff + CI.
  S6  close above the node: over touched events (resolved or not), actual = SPX-cash close (last clean
      print) > N; placebo = share of touched e with close_e > the touched P_e.
  S7  ladder: R = 0.10, 0.30, 0.40·IM and fixed 15 SPX pts; win rate vs placebo (diff + CI) and mean points
      per resolved event: actual (2w − 1)·R_d, placebo (2p − 1)·R_d (placebo rate applied at d's own R),
      R_d = mult·IM_d or 15.
  S8  T1 restricted to events whose touched node net <= −50 $bn, and <= −100 $bn (stored gex ÷ 1e9); same
      per-event placebo as T1; n, actual, placebo, diff, CI, plus the same split by year (each year
      bootstrapped over its own months).

Usage: apps/web/.venv/bin/python -u scripts/cr_bu_support.py | tee scripts/logs/cr_bu/run.log
"""
from __future__ import annotations

import json
import os
import signal
import sys
import warnings
from datetime import date, time, timedelta
from pathlib import Path

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
from packages.shared import spx_cash  # noqa: E402
from packages.shared.trading_calendar import is_trading_day, next_trading_day, prev_trading_day  # noqa: E402

D_FROM, D_TO = date(2024, 4, 3), date(2026, 6, 5)
BIN, MIN_SEP, MAX_CAND = 5.0, 10.0, 3
WIN_FAR, WIN_NEAR, ENTRY_OFF = 1.5, 0.25, 0.05
SEED_BOOT, N_BOOT = 20260920, 10_000
ROLL_MISMATCH_EXTRA = {date(2026, 6, 15)}
T10 = time(10, 0)
RACES = {"r10": ("im", 0.10), "r20": ("im", 0.20), "r30": ("im", 0.30), "r40": ("im", 0.40), "f15": ("pts", 15.0)}
PASS_DIFF, PASS_N = 0.08, 150
OUT_DIR = REPO_ROOT / "scripts" / "logs" / "cr_bu"
LOAD_WALL_SECONDS = 1800
STATEMENT_TIMEOUT = "600s"


class LoadTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise LoadTimeout(f"DB loads exceeded {LOAD_WALL_SECONDS} s wall time")


# ------------------------------------------------------------------------------------------- DB loads
signal.signal(signal.SIGALRM, _alarm)
signal.alarm(LOAD_WALL_SECONDS)                      # Python-side wall guard (a dropped socket never returns)
CONN = get_backfill_db_conn()
assert_role_or_die(CONN)
CONN.execute("SET default_transaction_read_only = on")
CONN.execute(f"SET statement_timeout = '{STATEMENT_TIMEOUT}'")
assert CONN.execute("SELECT current_setting('default_transaction_read_only')").fetchone()[0] == "on"


def q(sql, params=None) -> pd.DataFrame:
    cur = CONN.execute(sql, params)
    return pd.DataFrame(cur.fetchall(), columns=[c.name for c in cur.description])


def third_friday(y: int, m: int) -> date:
    d = date(y, m, 15)
    return d + timedelta(days=(4 - d.weekday()) % 7)


def roll_week(d: date) -> bool:  # CR-BM/CR-BO definition
    if d.month not in (3, 6, 9, 12):
        return False
    tf = third_friday(d.year, d.month)
    return tf - timedelta(days=11) <= d <= tf


def remap(df: pd.DataFrame) -> pd.DataFrame:
    df["d"] = [t if is_trading_day(t) else next_trading_day(t) for t in df["trade_date"]]
    return df[(df["d"] >= D_FROM) & (df["d"] <= D_TO)]


SESSIONS = [D_FROM + timedelta(days=i) for i in range((D_TO - D_FROM).days + 1)]
SESSIONS = [d for d in SESSIONS if is_trading_day(d)]
QEXP = {third_friday(y, m) for y in (2024, 2025, 2026) for m in (3, 6, 9, 12)}

print("loading …", flush=True)
near = remap(q("""
    SELECT trade_date, expir_date, strike::float AS strike, stock_price::float AS stock_price,
           gex_call, gex_put, put_oi, discounted_level
    FROM orats_oi_gamma
    WHERE ticker = 'SPX' AND trade_date BETWEEN %s AND %s
      AND expir_date >= trade_date AND expir_date <= trade_date + 10
""", (D_FROM - timedelta(days=7), D_TO)))
near = near[[d <= e <= d + timedelta(days=7) for e, d in zip(near["expir_date"], near["d"])]].copy()
near["net"] = near["gex_call"] - near["gex_put"].abs()

allx = remap(q("""
    SELECT trade_date, expir_date, strike::float AS strike, stock_price::float AS stock_price,
           gex_call - abs(gex_put) AS net
    FROM orats_oi_gamma
    WHERE ticker = 'SPX' AND trade_date BETWEEN %s AND %s AND expir_date >= trade_date
      AND strike BETWEEN 0.9 * stock_price AND 1.05 * stock_price
""", (D_FROM - timedelta(days=7), D_TO)))
allx = allx[[e >= d for e, d in zip(allx["expir_date"], allx["d"])]].copy()

feat = q("""
    SELECT trade_date, (feature_vector->>'implied_move_1d')::float AS im
    FROM bt_daily_features
    WHERE feature_version = %s AND ticker = 'SPX' AND active AND trade_date BETWEEN %s AND %s
""", (CANONICAL_FEATURE_VERSION, D_FROM, D_TO)).set_index("trade_date")["im"]
feat = feat[~feat.index.duplicated()]

es_starts = {es_snapshot_bar_start_utc(prev_trading_day(d)): d for d in SESSIONS}
es = q("SELECT datetime, close::float AS close FROM ironbeam_es_1m_bars WHERE datetime = ANY(%s)",
       (list(es_starts.keys()),))
ES = {es_starts[pd.Timestamp(t).to_pydatetime()]: float(c) for t, c in zip(es["datetime"], es["close"])
      if c is not None and np.isfinite(c) and c > 0}

daily, _, clean = spx_cash.load_spx_daily_bars(CONN, D_FROM, D_TO)
signal.alarm(0)
CONN.close()
print("loads done", flush=True)

# ------------------------------------------------------------------------------------------- universe
PRINTS = {d: g.sort_values("minute") for d, g in clean.groupby("session_date")}
have_rows = set(near["d"])
USABLE = [d for d in SESSIONS
          if not roll_week(d) and d in ES and d not in QEXP and d not in ROLL_MISMATCH_EXTRA
          and d in feat.index and pd.notna(feat[d]) and feat[d] > 0 and d in daily.index and d in have_rows]
assert len(USABLE) == 433, f"universe {len(USABLE)} != 433 — stop"
US = set(USABLE)
OPEN = {d: float(daily.loc[d, "open"]) for d in USABLE}
CLOSE = {d: float(daily.loc[d, "close"]) for d in USABLE}
IM = {d: float(feat[d]) for d in USABLE}
PX = {d: PRINTS[d]["px"].to_numpy() for d in USABLE}
MINS = {d: PRINTS[d]["minute"].to_numpy() for d in USABLE}

S0 = {}
for d, g in near[near["d"].isin(US)].groupby("d"):
    S0[d] = float(g.loc[g["expir_date"] == g["expir_date"].min(), "stock_price"].min())


def window(bins: pd.Series, d: date) -> pd.Series:
    o, im = OPEN[d], IM[d]
    return bins[(bins.index >= o - WIN_FAR * im) & (bins.index <= o - WIN_NEAR * im)]


def corrected(g: pd.DataFrame, d: date) -> pd.DataFrame:
    g = place_levels(g, S0[d])
    g["bin"] = bucket_levels(g["level_corrected"].to_numpy(), BIN)
    return g


NEAR_BY_D = {d: corrected(g.copy(), d) for d, g in near[near["d"].isin(US)].groupby("d")}
for d, g in NEAR_BY_D.items():
    g["bin_old"] = bucket_levels((g["discounted_level"] * S0[d] / ES[d]).to_numpy(), BIN)

DEFS = {
    "primary": {d: window(g.groupby("bin")["net"].sum(), d) for d, g in NEAR_BY_D.items()},
    "s3_0dte": {d: window(g[g["expir_date"] == d].groupby("bin")["net"].sum(), d) for d, g in NEAR_BY_D.items()},
    "s4_old": {d: window(g.groupby("bin_old")["net"].sum(), d) for d, g in NEAR_BY_D.items()},
}
DEFS["s3_all"] = {}
for d, g in allx[allx["d"].isin(US)].groupby("d"):
    g = corrected(g.copy(), d)
    DEFS["s3_all"][d] = window(g.groupby("bin")["net"].sum(), d)


def candidates(bins: pd.Series) -> list[tuple[float, float]]:
    order = sorted(bins[bins < 0].items(), key=lambda kv: (kv[1], -kv[0]))
    out: list[tuple[float, float]] = []
    for b, n in order:
        if all(abs(b - c) >= MIN_SEP for c, _ in out):
            out.append((float(b), float(n)))
            if len(out) == MAX_CAND:
                break
    return out


def touch_index(px: np.ndarray, e: float) -> int:
    hit = np.nonzero(px[1:] <= e)[0]
    return int(hit[0]) + 1 if len(hit) else -1


def first_touch(px: np.ndarray, es_: list[float]) -> tuple[int, int]:
    ti = [touch_index(px, e) for e in es_]
    hits = [(i, k) for k, i in enumerate(ti) if i >= 0]
    if not hits:
        return -1, -1
    i0 = min(i for i, _ in hits)
    return max((k for i, k in hits if i == i0), key=lambda kk: es_[kk]), i0


def race(px: np.ndarray, i: int, e: float, r: float) -> int:
    seg = px[i:]
    up = np.nonzero(seg >= e + r)[0]
    dn = np.nonzero(seg <= e - r)[0]
    if not len(up) and not len(dn):
        return -1
    if not len(dn):
        return 1
    if not len(up):
        return 0
    return 1 if up[0] < dn[0] else 0


def r_of(key: str, d: date) -> float:
    kind, v = RACES[key]
    return v * IM[d] if kind == "im" else v


def evaluate(d: date, zs: tuple[float, ...]) -> dict | None:
    """First touch of the levels open − z·IM on day d; None if untouched."""
    im, o = IM[d], OPEN[d]
    levels = [o - z * im for z in zs]
    es_ = [lv + ENTRY_OFF * im for lv in levels]
    px = PX[d]
    k, i = first_touch(px, es_)
    if k < 0:
        return None
    t = pd.Timestamp(MINS[d][i])
    j = int(np.searchsorted(MINS[d], np.datetime64(t - pd.Timedelta(minutes=15)), side="left"))
    return dict(k=k, i=i, N=levels[k], E=es_[k], early=t.time() < T10, minute=str(t),
                approach=abs(px[i] - px[j]) / im, close_above=CLOSE[d] > levels[k],
                race={key: race(px, i, es_[k], r_of(key, d)) for key in RACES})


def build(defn: str) -> tuple[dict, dict]:
    """(events, placebo results) for one candidate definition."""
    cands = {d: candidates(DEFS[defn].get(d, pd.Series(dtype=float))) for d in USABLE}
    events = {}
    for d in USABLE:
        if not cands[d]:
            continue
        zs = tuple((OPEN[d] - b) / IM[d] for b, _ in cands[d])
        ev = evaluate(d, zs)
        if ev is not None:
            ev.update(zs=zs, rank=ev["k"] + 1, net_bn=cands[d][ev["k"]][1] / 1e9, n_cand=len(cands[d]))
            events[d] = ev
    plc = {d: {e: evaluate(e, ev["zs"]) for e in USABLE if e != d} for d, ev in events.items()}
    return events, plc


def boot_ci(dates: list[date], vals: np.ndarray) -> tuple[float, float]:
    months = np.array([f"{d.year}-{d.month:02d}" for d in dates])
    um = np.unique(months)
    idx = {m: np.nonzero(months == m)[0] for m in um}
    rng = np.random.default_rng(SEED_BOOT)
    b = np.empty(N_BOOT)
    for i in range(N_BOOT):
        b[i] = vals[np.concatenate([idx[m] for m in rng.choice(um, size=len(um), replace=True)])].mean()
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(lo), float(hi)


def rate_stat(events: dict, plc: dict, key: str = "r20", keep=None, match_early: bool = False) -> dict:
    rows = []
    for d, ev in events.items():
        if keep is not None and not keep(d, ev):
            continue
        a = ev["race"][key]
        if a < 0:
            continue
        res = [p["race"][key] for p in plc[d].values()
               if p is not None and p["race"][key] >= 0 and (not match_early or p["early"] == ev["early"])]
        if not res:
            continue
        rows.append((d, a, float(np.mean(res)), len(res)))
    if not rows:
        return dict(n=0)
    ds = [r[0] for r in rows]
    a = np.array([r[1] for r in rows], float)
    p = np.array([r[2] for r in rows], float)
    lo, hi = boot_ci(ds, a - p)
    out = dict(n=len(rows), actual=a.mean(), placebo=p.mean(), diff=(a - p).mean(), ci=[lo, hi],
               median_placebo_n=float(np.median([r[3] for r in rows])))
    if key in RACES:
        rd = np.array([r_of(key, d) for d in ds])
        out.update(pts_actual=float(((2 * a - 1) * rd).mean()), pts_placebo=float(((2 * p - 1) * rd).mean()))
        out["pts_diff"] = out["pts_actual"] - out["pts_placebo"]
    return out


def value_stat(events: dict, plc: dict, field: str) -> dict:
    rows = []
    for d, ev in events.items():
        res = [float(p[field]) for p in plc[d].values() if p is not None]
        if res:
            rows.append((d, float(ev[field]), float(np.mean(res))))
    ds = [r[0] for r in rows]
    a = np.array([r[1] for r in rows])
    p = np.array([r[2] for r in rows])
    lo, hi = boot_ci(ds, a - p)
    return dict(n=len(rows), actual=a.mean(), placebo=p.mean(), diff=(a - p).mean(), ci=[lo, hi])


# --------------------------------------------------------------------------------------------- compute
print("computing …", flush=True)
R: dict = {}
EVS, PLC = {}, {}
for defn in ("primary", "s3_0dte", "s3_all", "s4_old"):
    EVS[defn], PLC[defn] = build(defn)
ev0, pl0 = EVS["primary"], PLC["primary"]
n_cand_days = sum(1 for d in USABLE if candidates(DEFS["primary"][d]))
assert n_cand_days == 365 and len(ev0) == 151, f"primary {n_cand_days}/{len(ev0)} != 365/151 — stop"

t1 = rate_stat(ev0, pl0, "r20")
unresolved = sum(1 for ev in ev0.values() if ev["race"]["r20"] < 0)
t1["touched"], t1["unresolved"] = len(ev0), unresolved
t1["verdict"] = ("PASS" if t1["n"] >= PASS_N and t1["diff"] >= PASS_DIFF and t1["ci"][0] > 0 else "FAIL")
R["T1"] = t1

R["S1"] = {"before_10": rate_stat(ev0, pl0, keep=lambda d, ev: ev["early"], match_early=True),
           "after_10": rate_stat(ev0, pl0, keep=lambda d, ev: not ev["early"], match_early=True)}
R["S2"] = value_stat(ev0, pl0, "approach")
R["S3"] = {"0dte_only": dict(rate_stat(EVS["s3_0dte"], PLC["s3_0dte"]), touched=len(EVS["s3_0dte"])),
           "all_expiries": dict(rate_stat(EVS["s3_all"], PLC["s3_all"]), touched=len(EVS["s3_all"]))}
R["S4"] = dict(rate_stat(EVS["s4_old"], PLC["s4_old"]), touched=len(EVS["s4_old"]))

# S5 monetization
put_oi = {(d, e, k): v for d, e, k, v in zip(near["d"], near["expir_date"], near["strike"], near["put_oi"])}


def rel_doi(d: date, rows: pd.DataFrame) -> tuple[float, int] | None:
    d1 = next_trading_day(d)
    if d1 > D_TO:
        return None
    rows = rows[rows["expir_date"] > d]
    base = nxt = 0.0
    missing = 0
    for e, k, v in zip(rows["expir_date"], rows["strike"], rows["put_oi"]):
        v1 = put_oi.get((d1, e, k))
        if v1 is None:
            missing += 1
            continue
        base += float(v)
        nxt += float(v1)
    return ((nxt - base) / base, missing) if base > 0 else None


s5_rows, s5_missing = [], 0
for d, ev in ev0.items():
    g = NEAR_BY_D[d]
    r = rel_doi(d, g[g["bin"] == ev["N"]])
    if r is None:
        continue
    s5_missing += r[1]
    s5_rows.append(dict(d=d, rel=r[0], win=ev["race"]["r20"], z=(OPEN[d] - ev["N"]) / IM[d]))
S5 = pd.DataFrame(s5_rows)
non_event = [e for e in USABLE if e not in ev0 and next_trading_day(e) <= D_TO]
ctrl = []
for _, row in S5.iterrows():
    vals = []
    for e in non_event:
        g = NEAR_BY_D[e]
        b = float(bucket_levels(np.array([OPEN[e] - row["z"] * IM[e]]), BIN)[0])
        r = rel_doi(e, g[g["bin"] == b])
        if r is not None:
            vals.append(r[0])
    ctrl.append(float(np.mean(vals)) if vals else np.nan)
S5["ctrl"] = ctrl
bn = S5[S5["win"] == 1]
nb = S5[S5["win"] == 0]
s5 = dict(n_events=len(S5), rows_missing_next_day=s5_missing,
          bounce=dict(n=len(bn), mean=bn["rel"].mean(), median=bn["rel"].median()),
          non_bounce=dict(n=len(nb), mean=nb["rel"].mean(), median=nb["rel"].median()),
          unresolved_n=int((S5["win"] < 0).sum()))
if len(bn) and len(nb):
    # difference of means, month-cluster bootstrap over the pooled resolved events
    res = S5[S5["win"] >= 0].reset_index(drop=True)
    months = np.array([f"{d.year}-{d.month:02d}" for d in res["d"]])
    um = np.unique(months)
    idx = {m: np.nonzero(months == m)[0] for m in um}
    rng = np.random.default_rng(SEED_BOOT)
    w, rel = res["win"].to_numpy(), res["rel"].to_numpy()
    bs = []
    for _ in range(N_BOOT):
        sel = np.concatenate([idx[m] for m in rng.choice(um, size=len(um), replace=True)])
        ww, rr = w[sel], rel[sel]
        if (ww == 1).any() and (ww == 0).any():
            bs.append(rr[ww == 1].mean() - rr[ww == 0].mean())
    s5["bounce_minus_non"] = dict(diff=bn["rel"].mean() - nb["rel"].mean(),
                                  ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))])
cm = S5.dropna(subset=["ctrl"])
s5["event_vs_control"] = dict(n=len(cm), event_mean=cm["rel"].mean(), control_mean=cm["ctrl"].mean(),
                              diff=(cm["rel"] - cm["ctrl"]).mean(),
                              ci=list(boot_ci(list(cm["d"]), (cm["rel"] - cm["ctrl"]).to_numpy())))
R["S5"] = s5

R["S6"] = value_stat(ev0, pl0, "close_above")
R["S7"] = {key: rate_stat(ev0, pl0, key) for key in ("r10", "r30", "r40", "f15")}


def s8(floor_bn: float) -> dict:
    keep = lambda d, ev: ev["net_bn"] <= floor_bn  # noqa: E731
    out = {"all": rate_stat(ev0, pl0, keep=keep)}
    for y in (2024, 2025, 2026):
        out[str(y)] = rate_stat(ev0, pl0, keep=lambda d, ev, y=y: keep(d, ev) and d.year == y)
    return out


R["S8"] = {"le_-50bn": s8(-50.0), "le_-100bn": s8(-100.0)}

OUT_DIR.mkdir(parents=True, exist_ok=True)
pd.DataFrame([{**{k: v for k, v in ev.items() if k not in ("race", "zs")}, **ev["race"], "d": d,
               "zs": json.dumps(ev["zs"])} for d, ev in ev0.items()]).to_csv(OUT_DIR / "events_primary.csv", index=False)
S5.to_csv(OUT_DIR / "s5_rows.csv", index=False)
(OUT_DIR / "results.json").write_text(json.dumps(R, indent=2, default=float))

# ----------------------------------------------------------------------------------------------- print


def pct(x):
    return f"{x * 100:6.2f}"


def line(name, s):
    if not s.get("n"):
        return f"{name:28s} n 0"
    out = (f"{name:28s} n {s['n']:4d}  actual {pct(s['actual'])}  placebo {pct(s['placebo'])}  "
           f"diff {s['diff'] * 100:+6.2f}  CI [{s['ci'][0] * 100:+6.2f}, {s['ci'][1] * 100:+6.2f}]")
    if "pts_actual" in s:
        out += f"  pts {s['pts_actual']:+6.2f} vs {s['pts_placebo']:+6.2f} ({s['pts_diff']:+.2f})"
    return out


print("\n" + "=" * 110)
print(f"T1 — bounce at the put node, R = 0.20·IM (A1)   touched {t1['touched']}, unresolved {t1['unresolved']}")
print(line("T1", t1))
print(f"pass rule: diff >= +8.00 AND CI lower > 0 AND resolved n >= 150   ->   VERDICT: {t1['verdict']}")
print("=" * 110)
print("\nS1 time of day (time-matched placebo)")
print(line("  first touch < 10:00 PT", R["S1"]["before_10"]))
print(line("  first touch >= 10:00 PT", R["S1"]["after_10"]))
s2 = R["S2"]
print(f"\nS2 approach |Δ15m| / IM: n {s2['n']}  actual {s2['actual']:.4f}  placebo {s2['placebo']:.4f}  "
      f"diff {s2['diff']:+.4f}  CI [{s2['ci'][0]:+.4f}, {s2['ci'][1]:+.4f}]")
print("\nS3 node definitions")
print(line(f"  0DTE only (touched {R['S3']['0dte_only']['touched']})", R["S3"]["0dte_only"]))
print(line(f"  all expiries (touched {R['S3']['all_expiries']['touched']})", R["S3"]["all_expiries"]))
print(f"\nS4 old placement (discounted_level × S0 ÷ ES), touched {R['S4']['touched']}")
print(line("  old placement", R["S4"]))
print(f"\nS5 monetization (rel ΔputOI D→D+1, expiries > D): events {s5['n_events']}, rows missing on D+1 "
      f"{s5['rows_missing_next_day']}, unresolved {s5['unresolved_n']}")
print(f"  bounce     n {s5['bounce']['n']:3d}  mean {s5['bounce']['mean']:+.4f}  median {s5['bounce']['median']:+.4f}")
print(f"  non-bounce n {s5['non_bounce']['n']:3d}  mean {s5['non_bounce']['mean']:+.4f}  median "
      f"{s5['non_bounce']['median']:+.4f}")
if "bounce_minus_non" in s5:
    b = s5["bounce_minus_non"]
    print(f"  bounce − non-bounce {b['diff']:+.4f}  CI [{b['ci'][0]:+.4f}, {b['ci'][1]:+.4f}]")
ec = s5["event_vs_control"]
print(f"  event vs matched-distance control: n {ec['n']}  event {ec['event_mean']:+.4f}  control "
      f"{ec['control_mean']:+.4f}  diff {ec['diff']:+.4f}  CI [{ec['ci'][0]:+.4f}, {ec['ci'][1]:+.4f}]")
s6 = R["S6"]
print(f"\nS6 close above N: n {s6['n']}  actual {pct(s6['actual'])}  placebo {pct(s6['placebo'])}  "
      f"diff {s6['diff'] * 100:+6.2f}  CI [{s6['ci'][0] * 100:+6.2f}, {s6['ci'][1] * 100:+6.2f}]")
print("\nS7 target ladder (pts = mean points per resolved event, (2p − 1)·R_d)")
for key, lab in (("r10", "R = 0.10·IM"), ("r30", "R = 0.30·IM"), ("r40", "R = 0.40·IM"), ("f15", "R = 15 pts")):
    print(line(f"  {lab}", R["S7"][key]))
print(line("  (T1) R = 0.20·IM", rate_stat(ev0, pl0, "r20")))
print("\nS8 node size (touched node net)")
for fk, lab in (("le_-50bn", "net <= −50 $bn"), ("le_-100bn", "net <= −100 $bn")):
    for part in ("all", "2024", "2025", "2026"):
        print(line(f"  {lab} {part}", R["S8"][fk][part]))
print(f"\nresults: {OUT_DIR / 'results.json'}\nDONE")
