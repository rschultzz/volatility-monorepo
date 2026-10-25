#!/usr/bin/env python3
"""CR-CB Amendment A16 — pre-registered T0 on R-a (2024-04-03 .. 2024-12-31). READ-ONLY: SELECTs only.

Spec: specs/CR-CB-failed-retest-at-level.md, Amendment A16 (A16.1–A16.9). Price paths only: no trade outcome, no
placebo P&L, no A/A. One run. Every statistic is written to scripts/logs/cr_cb/t0_ra.json before anything prints.

Fixed here (A16 and the train T0 of scripts/cr_cb_step0_v3.py, unchanged where not stated):
  - Guard: every read is dated < 2025-01-01 (flow ts_utc < 2024-12-31 13:00 PT; orats_oi_gamma trade_date <=
    2024-12-31; 1m bars only on A days <= 2024-12-30). Asserted on the loaded frames.
  - Sessions: NYSE trading days 2024-04-03 .. 2024-12-31; exclusions in order, each counted: roll week (Monday of
    the week before expiry week → expiry Friday); no ES 15:45 ET bar on A; quarterly-expiry Friday; no Databento
    RTH flow; early close (last RTH print before 12:55 PT); no canonical IM / IM <= 0; no orats_oi_gamma rows;
    contract check: A's and D's largest-volume Databento symbols must match, and P_target must equal the close of
    A's flow in the 15:45 ET minute within 2 pts (2024 bars are Databento ES.c.0; this catches a frame mismatch).
  - RTH open = first flow print >= 06:30 PT. Levels: A1.1 (place_levels at P_target; 1-pt buckets; mag >= 40 $bn;
    5-pt spacing; within ±IM of the open; top 6). Single levels (d 0, e 0). Sign: net < 0 put, > 0 call.
  - Placebo A15.6: 4 nearest 5-pt bins per real level (same side of the open, within ±IM, >= 10 pts from every
    real level; ties to the lower bin); dedup; parent sign.
  - Tests: armed L5 tests (0.10·IM arming, 2-pt band, CR-BW re-arm), every one, both sides; side = approach.
  - O_c frozen (A16.3): put/below 9.75, put/above 8.25, call/below 6.25, call/above 5.25.
  - Primary: esc = min(minutes to first trade >= 0.25·IM from the level, 120, minutes to 13:00 PT).
    Secondary: seconds of forward-filled 1-s closes inside [L − O_c, L + O_c] over [t, min(t + 3600, 13:00)).
  - Bootstrap: one 10 000 × n_sessions multinomial weight matrix (seed 20260920) applied to per-session sums of
    real and placebo jointly; diff = ratio of weighted sums (mean real − mean placebo); 95 % percentile CI.
  - Holm (A16.7): family = 4 cells + pooled, primary only; one-sided p = share of resamples with diff <= 0,
    floor 1e-4; step-down at alpha 0.025; pass = Holm-rejected and CI lower > 0.
  - Verdict (A16.8): any pass → "pass"; else per hypothesis "refuted" if CI upper < 0, else "inconclusive".

Usage: apps/web/.venv/bin/python -u scripts/cr_cb_t0_ra.py | tee scripts/logs/cr_cb/t0_ra.log
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

signal.alarm(3 * 3600)

PT, UTC = ZoneInfo("America/Los_Angeles"), timezone.utc
D_FROM, D_TO = date(2024, 4, 3), date(2024, 12, 31)
LIMIT_DATE = date(2025, 1, 1)
TICK = 0.25
DTE_MAX, MAG_MIN, SPACING, WIN_IM, TOP_N = 30, 40e9, 5.0, 1.0, 6
ARM_IM, BAND = 0.10, 2.0
PLC_BIN, PLC_SEP, PLC_K = 5.0, 10.0, 4
O_C = {("put", 1): 9.75, ("put", -1): 8.25, ("call", 1): 6.25, ("call", -1): 5.25}
ESC_IM, ESC_CAP, DWELL_S = 0.25, 120, 3600
SEED_BOOT, N_BOOT, ALPHA = 20260920, 10_000, 0.025
CELLS = [("put", 1), ("put", -1), ("call", 1), ("call", -1)]
OUT_DIR = REPO_ROOT / "scripts" / "logs" / "cr_cb"

CONN = get_backfill_db_conn()
assert_role_or_die(CONN)
CONN.execute("SET default_transaction_read_only = on")
CONN.execute("SET statement_timeout = '600s'")
assert CONN.execute("SELECT current_setting('default_transaction_read_only')").fetchone()[0] == "on"


def q(sql, params=None) -> pd.DataFrame:
    cur = CONN.execute(sql, params)
    return pd.DataFrame(cur.fetchall(), columns=[c.name for c in cur.description])


def third_friday(y, m):
    d = date(y, m, 15)
    return d + timedelta(days=(4 - d.weekday()) % 7)


def roll_week(d):
    if d.month not in (3, 6, 9, 12):
        return False
    tf = third_friday(d.year, d.month)
    return tf - timedelta(days=11) <= d <= tf


def pt(d, hh, mm, ss=0):
    return datetime.combine(d, time(hh, mm, ss), PT).astimezone(UTC)


def ep(x):
    return int(x.timestamp())


SESSIONS, d = [], D_FROM
while d <= D_TO:
    if is_trading_day(d):
        SESSIONS.append(d)
    d += timedelta(days=1)
QEXP = {third_friday(2024, m) for m in (3, 6, 9, 12)}
CAP_TS = pt(D_TO, 13, 0)
assert CAP_TS < datetime(2025, 1, 1, tzinfo=UTC)

# ----------------------------------------------------------------------------------------------- loads
sym = q("""
    SELECT (ts_utc AT TIME ZONE 'America/Los_Angeles')::date AS d, symbol, sum(volume) AS vol
    FROM es_flow_1s_databento
    WHERE ts_utc >= %s AND ts_utc < %s
      AND (ts_utc AT TIME ZONE 'America/Los_Angeles')::time >= '06:30'
      AND (ts_utc AT TIME ZONE 'America/Los_Angeles')::time < '13:00'
    GROUP BY 1, 2
""", (pt(prev_trading_day(D_FROM), 0, 0), CAP_TS)).sort_values("vol")
assert sym["d"].max() < LIMIT_DATE
SYM = dict(zip(sym["d"], sym["symbol"]))


def load_flow(d_, symbol):
    f = q("""SELECT ts_utc, open, high, low, close FROM es_flow_1s_databento
             WHERE symbol = %s AND ts_utc >= %s AND ts_utc < %s AND ts_utc < %s ORDER BY ts_utc""",
          (symbol, pt(d_, 6, 15), pt(d_, 13, 0), CAP_TS))
    return dict(t=np.array([ep(x) for x in f["ts_utc"]], dtype=np.int64), o=f["open"].to_numpy(float),
                h=f["high"].to_numpy(float), l=f["low"].to_numpy(float), c=f["close"].to_numpy(float))


FLOW = {d_: load_flow(d_, SYM[d_]) for d_ in SESSIONS if d_ in SYM}

oig = q("""
    SELECT trade_date, expir_date, strike::float AS strike, stock_price::float AS stock_price, gex_call, gex_put
    FROM orats_oi_gamma
    WHERE ticker = 'SPX' AND trade_date BETWEEN %s AND %s
      AND expir_date >= trade_date AND expir_date <= trade_date + %s
""", (D_FROM - timedelta(days=7), D_TO, DTE_MAX + 7))
assert oig["trade_date"].max() < LIMIT_DATE
oig["d"] = [t if is_trading_day(t) else next_trading_day(t) for t in oig["trade_date"]]
oig = oig[(oig["d"] >= D_FROM) & (oig["d"] <= D_TO)]
oig = oig[[d_ <= e <= d_ + timedelta(days=DTE_MAX) for e, d_ in zip(oig["expir_date"], oig["d"])]]
oig["call"] = oig["gex_call"].astype(float)
oig["put"] = -oig["gex_put"].abs().astype(float)

feat = q("""
    SELECT trade_date, (feature_vector->>'implied_move_1d')::float AS im FROM bt_daily_features
    WHERE feature_version = %s AND ticker = 'SPX' AND active AND trade_date BETWEEN %s AND %s
""", (CANONICAL_FEATURE_VERSION, D_FROM, D_TO)).set_index("trade_date")["im"]
feat = feat[~feat.index.duplicated()]

starts = {es_snapshot_bar_start_utc(prev_trading_day(d_)): d_ for d_ in SESSIONS}
assert max(starts) < datetime(2025, 1, 1)                     # naive UTC bar starts
es = q("SELECT datetime, close::float AS close FROM ironbeam_es_1m_bars WHERE datetime = ANY(%s)", (list(starts),))
PTGT = {starts[pd.Timestamp(t).to_pydatetime()]: c for t, c in zip(es["datetime"], es["close"])
        if c is not None and np.isfinite(c) and c > 0}

# flow close of A's front symbol in the 15:45 ET minute
ACHK = {}
for st, d_ in starts.items():
    a_ = prev_trading_day(d_)
    if a_ not in SYM:
        continue
    r = q("""SELECT close FROM es_flow_1s_databento WHERE symbol = %s AND ts_utc >= %s AND ts_utc < %s
             ORDER BY ts_utc DESC LIMIT 1""", (SYM[a_], st.replace(tzinfo=UTC), st.replace(tzinfo=UTC) + timedelta(minutes=1)))
    if len(r):
        ACHK[d_] = (SYM[a_], float(r["close"][0]))


def rth_slice(F, d_):
    return np.searchsorted(F["t"], ep(pt(d_, 6, 30))), np.searchsorted(F["t"], ep(pt(d_, 13, 0)))


OPEN, LAST = {}, {}
for d_, F in FLOW.items():
    a, b = rth_slice(F, d_)
    if b > a:
        OPEN[d_], LAST[d_] = float(F["o"][a]), int(F["t"][b - 1])

U, funnel = list(SESSIONS), []
for name, rule in [
    ("roll week", lambda d_: roll_week(d_)),
    ("no ES 15:45 ET bar on A", lambda d_: d_ not in PTGT),
    ("quarterly-expiry Friday", lambda d_: d_ in QEXP),
    ("no Databento RTH flow", lambda d_: d_ not in OPEN),
    ("early close (last RTH print < 12:55 PT)", lambda d_: LAST[d_] < ep(pt(d_, 12, 55))),
    ("no canonical IM / IM <= 0", lambda d_: not (d_ in feat.index and pd.notna(feat[d_]) and feat[d_] > 0)),
    ("no orats_oi_gamma rows", lambda d_: d_ not in set(oig["d"])),
    ("contract check (A symbol != D symbol, or |P_target − A flow 15:45 close| > 2)",
     lambda d_: d_ not in ACHK or ACHK[d_][0] != SYM[d_] or abs(ACHK[d_][1] - PTGT[d_]) > 2.0),
]:
    rm = [d_ for d_ in U if rule(d_)]
    U = [d_ for d_ in U if not rule(d_)]
    funnel.append([name, len(rm), len(U), [str(x) for x in rm]])
USABLE = U
IM = {d_: float(feat[d_]) for d_ in USABLE}
SIDX = {d_: k for k, d_ in enumerate(USABLE)}


# ---------------------------------------------------------------------------------------------- levels
def select_levels(g, ptgt, o, im):
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
    return [dict(L=float(k.level), mag=float(k.mag), net=float(k.net)) for k in win[:TOP_N]]


LEVELS = {d_: select_levels(g, PTGT[d_], OPEN[d_], IM[d_]) for d_, g in oig[oig["d"].isin(USABLE)].groupby("d")}


def placebo(d_):
    o, im = OPEN[d_], IM[d_]
    real = np.array([x["L"] for x in LEVELS[d_]])
    pool = {}
    for x in LEVELS[d_]:
        s = np.sign(x["L"] - o)
        lo, hi = np.ceil((o - im) / PLC_BIN) * PLC_BIN, np.floor((o + im) / PLC_BIN) * PLC_BIN
        cand = np.arange(lo, hi + 0.5, PLC_BIN)
        cand = cand[(np.sign(cand - o) == s) & (np.min(np.abs(cand[:, None] - real[None, :]), axis=1) >= PLC_SEP)]
        cand = cand[np.lexsort((cand, np.abs(cand - x["L"])))][:PLC_K]
        for b in cand:
            pool.setdefault(float(b), dict(L=float(b), mag=x["mag"], net=x["net"]))
    return list(pool.values())


PLV = {d_: placebo(d_) for d_ in USABLE}


# ------------------------------------------------------------------------------------------------ tests
def tests_for(F, d_, L, im):
    a, b = rth_slice(F, d_)
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


def measures(levels):
    rows = []
    for d_ in USABLE:
        F = FLOW[d_]
        _, b = rth_slice(F, d_)
        t, h, l, c = F["t"], F["h"], F["l"], F["c"]
        close13 = ep(pt(d_, 13, 0))
        for lv in levels[d_]:
            ty = "put" if lv["net"] < 0 else "call"
            for j, side in tests_for(F, d_, lv["L"], IM[d_]):
                t0 = int(t[j])
                esc = np.nonzero((h[j:b] >= lv["L"] + ESC_IM * IM[d_]) | (l[j:b] <= lv["L"] - ESC_IM * IM[d_]))[0]
                esc_min = (t[j + int(esc[0])] - t0) / 60 if len(esc) else np.inf
                cap = min(ESC_CAP, (close13 - t0) / 60)
                O = O_C[(ty, side)]
                g = np.arange(t0, min(t0 + DWELL_S, close13))
                kk = np.searchsorted(t, g, side="right") - 1
                px = c[np.clip(kk, 0, None)]
                rows.append(dict(d=d_, type=ty, side=side, esc=min(esc_min, cap), censored=not np.isfinite(esc_min),
                                 capped=bool(esc_min > cap), dwell=int(np.sum(np.abs(px - lv["L"]) <= O))))
    return pd.DataFrame(rows)


R = measures(LEVELS)
P = measures(PLV)

# ------------------------------------------------------------------------------------------- statistics
nS = len(USABLE)
W = np.random.default_rng(SEED_BOOT).multinomial(nS, np.full(nS, 1 / nS), size=N_BOOT).astype(float)


def sums(E, m):
    s = np.zeros((nS, 2))
    np.add.at(s, ([SIDX[x] for x in E["d"]], 0), E[m].to_numpy(float))
    np.add.at(s, ([SIDX[x] for x in E["d"]], 1), 1)
    return s


def compare(sel, m):
    r, p = sel(R), sel(P)
    sr, sp = sums(r, m), sums(p, m)
    br, bp = W @ sr, W @ sp
    with np.errstate(invalid="ignore", divide="ignore"):
        bd = br[:, 0] / br[:, 1] - bp[:, 0] / bp[:, 1]
    rv, pv = sr[:, 0].sum() / sr[:, 1].sum(), sp[:, 0].sum() / sp[:, 1].sum()
    lo, hi = np.nanpercentile(bd, [2.5, 97.5])
    return dict(n_real=len(r), n_placebo=len(p), real=round(float(rv), 4), placebo=round(float(pv), 4),
                diff=round(float(rv - pv), 4), lo=round(float(lo), 4), hi=round(float(hi), 4),
                p_one_sided=max(float(np.nanmean(bd <= 0)), 1.0 / N_BOOT))


HYP = {f"{ty}/{'below' if s == 1 else 'above'}": (lambda E, ty=ty, s=s: E[(E["type"] == ty) & (E["side"] == s)])
       for ty, s in CELLS}
HYP["pooled"] = lambda E: E
RES = dict(funnel=funnel, sessions=dict(n=nS, first=str(USABLE[0]), last=str(USABLE[-1]),
                                        symbols=sorted({SYM[x] for x in USABLE})),
           levels=dict(real=sum(len(v) for v in LEVELS.values()), placebo=sum(len(v) for v in PLV.values()),
                       real_put=sum(x["net"] < 0 for v in LEVELS.values() for x in v),
                       real_call=sum(x["net"] > 0 for v in LEVELS.values() for x in v)),
           tests=dict(real=len(R), placebo=len(P), real_censored=int(R["censored"].sum()),
                      placebo_censored=int(P["censored"].sum()), real_capped=int(R["capped"].sum()),
                      placebo_capped=int(P["capped"].sum())),
           primary={k: compare(f, "esc") for k, f in HYP.items()},
           secondary={k: compare(f, "dwell") for k, f in HYP.items()})

# Holm, one-sided, alpha 0.025, primary only
order = sorted(RES["primary"], key=lambda k: RES["primary"][k]["p_one_sided"])
m_, stop = len(order), False
for i, k in enumerate(order):
    thr = ALPHA / (m_ - i)
    rej = (not stop) and RES["primary"][k]["p_one_sided"] <= thr
    stop = stop or not rej
    blk = RES["primary"][k]
    blk.update(holm_threshold=thr, holm_rejected=bool(rej), passed=bool(rej and blk["lo"] > 0),
               verdict="pass" if (rej and blk["lo"] > 0) else ("refuted" if blk["hi"] < 0 else "inconclusive"))
RES["A16_pass"] = any(v["passed"] for v in RES["primary"].values())
RES["verdict"] = "pass" if RES["A16_pass"] else "fail"

OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "t0_ra.json").write_text(json.dumps(RES, indent=1, default=str))
print(json.dumps({k: v for k, v in RES.items() if k != "funnel"}, indent=1, default=str))
for f_ in funnel:
    print(f_[:3])
print("\nPRIMARY time to escape (min, real − placebo):")
for k, v in RES["primary"].items():
    print(f"  {k:12s} n {v['n_real']}/{v['n_placebo']}  {v['real']:.2f} vs {v['placebo']:.2f}  diff {v['diff']:+.2f} "
          f"[{v['lo']:+.2f}, {v['hi']:+.2f}]  p {v['p_one_sided']:.4f}  Holm thr {v['holm_threshold']:.4f}  "
          f"{v['verdict']}")
print("A16:", RES["verdict"].upper())
