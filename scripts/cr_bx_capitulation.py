#!/usr/bin/env python3
"""CR-BX run — family A only (capitulation fade): T1, T2, S1–S7, S14. READ-ONLY: SELECTs only.

Spec: specs/CR-BX-capitulation-seller-exhaustion.md (L1–L11, Step 0 findings: k = 3.0; family B dropped at Step 0,
not run). One run. Every statistic is computed and written to scripts/logs/cr_bx/results.json before anything is
printed.

Universe, σ5_d, L5 events, L7 predictor, eligibility, L9 placebo and the L6 race are those of
scripts/cr_bx_step0.py (commit 7b065fb), copied unchanged. Additional definitions fixed here before the run:
  - k = 3.0 (Step 0, from family A counts only).
  - eligibility: pre-window [t − 300, t] RTH gap <= 15 s and D(300, 0) > 0; then, after the race, the race
    window [entry second, race end] RTH gap <= 15 s (L3). Unresolved by 13:00 PT excluded and counted.
    The same rules for placebo entries.
  - terciles: X cut points over all eligible events (after the pre-window rule, before the race-window rule),
    exactly as Step 0; placebo entries carry the parent's tercile.
  - T1: win rate low − high X tercile at R = 8. Pass: >= +8 pts, CI lower > 0, >= 150 resolved per tercile.
  - T2: low-tercile win rate − win rate of the placebo entries whose parent is low-tercile, R = 8.
    Pass: >= +8 pts, CI lower > 0.
  - L10 bootstrap: session-cluster, 10 000 multinomial resamples, seed 20260920; one weight matrix reused for
    every statistic; real and placebo resampled jointly by session.
  - S1: R = 4, 6, 8: win rate and net expectancy per resolved trade in ticks (win +R, loss −(R + 1), minus
    0.36 tick) by tercile, real and placebo (parent tercile); break-even win rate (R + 1.36) / (2R + 1).
  - S2: move size |move| / σ5_d in [3, 4) and >= 4 (no 2–3σ events exist at k = 3): low-tercile win rate −
    matched placebo, and T1, per bucket.
  - S3: direction (buy after down / sell after up): T1 and T2 within each.
  - S4: event extreme (low for down, high for up) within 2 pts of a CR-BW A1.1 level vs not; levels rebuilt as
    in CR-BW (orats_oi_gamma [D, D+30], place_levels at the ES 15:45 ET close on A, 1-pt buckets,
    |call| + |put| >= 40e9, 5-pt spacing by magnitude over the universe, |L − open| <= IM, top 6; open = first
    RTH flow open). T1 and T2 within each.
  - S5: event before / at-or-after 10:00 PT: T1 and T2 within each.
  - S6: X <= 0 events: win rate − matched placebo (placebo entries of X <= 0 parents), with CI.
  - S7: X30 terciles (own cut points over eligible): T1 and T2-form (low X30 tercile − its matched placebo).
  - S14: signed move in the fade direction from the entry price to the last close at or before entry + 30 min,
    + 60 min (capped at 13:00 PT) and the 13:00 PT close, in σ5_d units; mean by X tercile, real and placebo;
    low − high and low − matched placebo with CI. Uses every eligible event / placebo entry passing the
    pre-window rule (no race needed, so no race-window rule).

Usage: apps/web/.venv/bin/python -u scripts/cr_bx_capitulation.py | tee scripts/logs/cr_bx/run.log
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
D_FROM, D_TO = date(2025, 12, 26), date(2026, 6, 5)
HOLDOUT_TS = datetime(2026, 6, 5, 21, 0, tzinfo=UTC)
TICK, GAP_MAX, COST_TICKS = 0.25, 15, 4.50 / 12.50
K = 3.0
COOLDOWN, PLC_SPAN, PLC_N = 900, 1800, 5
R_PRIMARY, R_LADDER = 8, (4, 6, 8)
SCALE_N, SCALE_MIN = 20, 5
DTE_MAX, MAG_MIN, SPACING, TOP_N, NEAR_PTS = 30, 40e9, 5.0, 6, 2.0
SEED, SEED_BOOT, N_BOOT = 20261025, 20260920, 10_000
ROLL_MISMATCH_EXTRA = {date(2026, 6, 15)}
OUT_DIR = REPO_ROOT / "scripts" / "logs" / "cr_bx"

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


CH = {d: abs_changes(d, 300) for d in SCALE_ELIG}
SIG = {}
for d in SESSIONS:
    prior = [x for x in SCALE_ELIG if x < d][-SCALE_N:]
    if len(prior) >= SCALE_MIN:
        SIG[d] = float(np.median(np.concatenate([CH[x] for x in prior])))
oig_days = set(oig["d"])
USABLE = [d for d in SESSIONS if d not in ex_a | ex_b | ex_c and d in FLOW and d in feat.index
          and pd.notna(feat[d]) and feat[d] > 0 and d in oig_days and d in SIG]
IM = {d: float(feat[d]) for d in USABLE}
OPEN = {d: float(FLOW[d]["o"][0]) for d in USABLE}
INFO = dict(sessions=len(USABLE), first=str(USABLE[0]), last=str(USABLE[-1]))

PREP = {}
for d in USABLE:
    f = FLOW[d]
    idx = pd.to_datetime(f["t"], unit="s")
    PREP[d] = dict(lo300=pd.Series(f["l"], idx).rolling("300s", closed="both").min().to_numpy(),
                   hi300=pd.Series(f["h"], idx).rolling("300s", closed="both").max().to_numpy(),
                   inwin=(f["t"] >= ep(pt(d, 6, 35))) & (f["t"] <= ep(pt(d, 12, 45))),
                   cum=np.concatenate([[0], np.cumsum(f["buy"] - f["sell"])]))


def net_between(d, t0, t1):
    f, c = FLOW[d]["t"], PREP[d]["cum"]
    a, b = np.searchsorted(f, t0), np.searchsorted(f, t1, side="right")
    return int(c[b] - c[a])


# ------------------------------------------------------------------------------------- family A events
EV, COND = [], {}
for d in USABLE:
    f, p = FLOW[d], PREP[d]
    s = SIG[d]
    down = p["inwin"] & (f["l"] == p["lo300"]) & (f["l"] <= p["hi300"] - K * s)
    up = p["inwin"] & (f["h"] == p["hi300"]) & (f["h"] >= p["lo300"] + K * s)
    COND[d] = f["t"][down | up]
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
            EV.append(dict(d=d, i=int(i), t=t, dirn=dirn, D300=D300,
                           X=D60 / (D300 / 5) if D300 > 0 else np.nan,
                           X30=D30 / (D300 / 10) if D300 > 0 else np.nan,
                           pre_gap=max_gap(d, t - 300, t),
                           ext=f["l"][i] if dirn == -1 else f["h"][i],
                           mag=(p["hi300"][i] - f["l"][i]) / s if dirn == -1 else (f["h"][i] - p["lo300"][i]) / s))
A = pd.DataFrame(EV)
INFO["events"] = len(A)
A = A[(A["pre_gap"] <= GAP_MAX) & (A["D300"] > 0)].reset_index(drop=True)
CUT = np.percentile(A["X"], [100 / 3, 200 / 3])
CUT30 = np.percentile(A["X30"], [100 / 3, 200 / 3])


def terc_of(x, cut):
    return np.where(x <= cut[0], 0, np.where(x <= cut[1], 1, 2))


A["terc"] = terc_of(A["X"].to_numpy(), CUT)
A["terc30"] = terc_of(A["X30"].to_numpy(), CUT30)
A["pid"] = np.arange(len(A))
INFO.update(eligible=len(A), cut=CUT.tolist(), cut30=CUT30.tolist(),
            per_tercile=np.bincount(A["terc"], minlength=3).tolist())
# Step 0 consistency (commit 7b065fb): fail before any outcome if the universe or events differ
assert len(USABLE) == 88 and INFO["events"] == 1831 and len(A) == 1291, INFO
assert INFO["per_tercile"] == [431, 430, 430], INFO

# ----------------------------------------------------------------------------------------- placebo
rng = np.random.default_rng(SEED)
prow = []
for x in A.sort_values(["d", "t"]).itertuples():
    d, t = x.d, x.t
    f = FLOW[d]
    lo_t, hi_t = max(t - PLC_SPAN, ep(pt(d, 6, 35))), min(t + PLC_SPAN, ep(pt(d, 12, 45)))
    a, b = np.searchsorted(f["t"], lo_t), np.searchsorted(f["t"], hi_t, side="right")
    cand = f["t"][a:b]
    ca = COND[d]
    k1, k2 = np.searchsorted(ca, cand - 300), np.searchsorted(ca, cand, side="right")
    ci = np.arange(a, b)[(k2 - k1) == 0]
    ci = np.array([i for i in ci if max_gap(d, f["t"][i] - 300, f["t"][i]) <= GAP_MAX], dtype=int)
    pick = rng.choice(ci, size=min(PLC_N, len(ci)), replace=False) if len(ci) else []
    for i in pick:
        prow.append(dict(d=d, i=int(i), t=int(f["t"][i]), dirn=x.dirn, terc=x.terc, terc30=x.terc30,
                         pid=x.pid, X=x.X, mag=x.mag, ext=x.ext, t_parent=t))
P = pd.DataFrame(prow)
INFO["placebo_entries"] = len(P)


# -------------------------------------------------------------------------------------------- race
def race(d, i_ev, dirn, Rt):
    """L6 fade from the row after i_ev: (1 / 0 / -1 unresolved, entry price, entry t, end t)."""
    f = FLOW[d]
    e_i = i_ev + 1
    if e_i >= len(f["t"]):
        return -1, np.nan, None, None
    long = dirn == -1
    E = f["o"][e_i] + (TICK if long else -TICK)
    Rp = Rt * TICK
    if (long and f["l"][e_i] <= E - Rp) or (not long and f["h"][e_i] >= E + Rp):
        return 0, E, int(f["t"][e_i]), int(f["t"][e_i])
    if long:
        tgt, stp = f["h"][e_i + 1:] >= E + Rp + TICK, f["l"][e_i + 1:] <= E - Rp
    else:
        tgt, stp = f["l"][e_i + 1:] <= E - Rp - TICK, f["h"][e_i + 1:] >= E + Rp
    ti, si = np.nonzero(tgt)[0], np.nonzero(stp)[0]
    if not len(ti) and not len(si):
        return -1, E, int(f["t"][e_i]), ep(pt(d, 13, 0))
    k = min(ti[0] if len(ti) else 10**9, si[0] if len(si) else 10**9)
    return (1 if len(ti) and (not len(si) or ti[0] < si[0]) else 0), E, int(f["t"][e_i]), int(f["t"][e_i + 1 + k])


def close_at(d, t1):
    f = FLOW[d]
    j = np.searchsorted(f["t"], min(t1, ep(pt(d, 13, 0)) - 1), side="right") - 1
    return f["c"][j]


for E in (A, P):
    for Rt in R_LADDER:
        rr = [race(x.d, x.i, x.dirn, Rt) for x in E.itertuples()]
        E[f"res_{Rt}"] = [r[0] for r in rr]
        E[f"keep_{Rt}"] = [r[2] is not None and max_gap(x.d, r[2], r[3]) <= GAP_MAX
                           for x, r in zip(E.itertuples(), rr)]
        if Rt == R_PRIMARY:
            E["entry"] = [r[1] for r in rr]
            E["t_entry"] = [r[2] for r in rr]
    sgn = -E["dirn"].to_numpy()                 # fade direction: long (+1) after a down move
    for lab, dt in (("30m", 1800), ("60m", 3600), ("close", None)):
        px = [close_at(x.d, int(x.t_entry + dt) if dt else ep(pt(x.d, 13, 0))) if pd.notna(x.t_entry) else np.nan
              for x in E.itertuples()]
        E[f"mv_{lab}"] = sgn * (np.array(px) - E["entry"].to_numpy()) / np.array([SIG[x] for x in E["d"]])

# ------------------------------------------------------------------------------------------ S4 levels
LEVELS = {}
for d, g in oig[oig["d"].isin(USABLE)].groupby("d"):
    g = place_levels(g, PTGT[d])
    g = g[g["level_corrected"].notna()]
    g["level"] = bucket_levels(g["level_corrected"].to_numpy(), 1.0)
    bk = g.groupby("level").agg(call=("call", "sum"), put=("put", "sum")).reset_index()
    bk["mag"] = bk["call"].abs() + bk["put"].abs()
    bk = bk[bk["mag"] >= MAG_MIN].sort_values("mag", ascending=False)
    kept = []
    for r in bk.itertuples():
        if all(abs(r.level - k.level) >= SPACING for k in kept):
            kept.append(r)
    LEVELS[d] = np.array([k.level for k in kept if abs(k.level - OPEN[d]) <= IM[d]][:TOP_N])
A["near"] = [bool(len(LEVELS.get(x.d, [])) and np.min(np.abs(LEVELS[x.d] - x.ext)) <= NEAR_PTS) for x in A.itertuples()]
P["near"] = P["pid"].map(A.set_index("pid")["near"])

# ------------------------------------------------------------------------------------------ bootstrap
S_IDX = {d: k for k, d in enumerate(USABLE)}
nS = len(USABLE)
W = np.random.default_rng(SEED_BOOT).multinomial(nS, np.full(nS, 1 / nS), size=N_BOOT).astype(float)
for E in (A, P):
    E["sidx"] = [S_IDX[d] for d in E["d"]]


def cnt(E, Rt=R_PRIMARY, mask=None, terc="terc"):
    """session × 3 terciles × (wins, resolved)."""
    m = E[f"keep_{Rt}"] & (E[f"res_{Rt}"] >= 0)
    if mask is not None:
        m &= mask
    e = E[m]
    c = np.zeros((nS, 3, 2))
    np.add.at(c, (e["sidx"].to_numpy(), e[terc].to_numpy(), 0), e[f"res_{Rt}"].to_numpy())
    np.add.at(c, (e["sidx"].to_numpy(), e[terc].to_numpy(), 1), 1)
    return c


def wr(c):
    with np.errstate(invalid="ignore", divide="ignore"):
        return c[..., 0] / c[..., 1]


def ci(fn, *cs):
    obs = fn(*[c.sum(0) for c in cs])
    b = fn(*[np.einsum("bs,sij->bij", W, c) for c in cs])
    lo, hi = np.nanpercentile(b, [2.5, 97.5])
    return dict(est=round(float(obs) * 100, 2), lo=round(float(lo) * 100, 2), hi=round(float(hi) * 100, 2))


def t1(c):
    r = wr(c)
    return r[..., 0] - r[..., 2]


def t2(cr, cp):
    return wr(cr)[..., 0] - wr(cp)[..., 0]


def block(Rt=R_PRIMARY, mr=None, mp=None, terc="terc"):
    cr, cp = cnt(A, Rt, mr, terc), cnt(P, Rt, mp, terc)
    return dict(real_n=cr.sum(0)[:, 1].astype(int).tolist(), real_wr=(wr(cr.sum(0)) * 100).round(2).tolist(),
                placebo_n=cp.sum(0)[:, 1].astype(int).tolist(),
                placebo_wr=(wr(cp.sum(0)) * 100).round(2).tolist(), T1=ci(t1, cr), T2=ci(t2, cr, cp))


RES: dict = dict(info=INFO)
RES["drops"] = {name: dict(race_gap=[int(((E["terc"] == k) & ~E["keep_8"]).sum()) for k in range(3)],
                           unresolved=[int(((E["terc"] == k) & E["keep_8"] & (E["res_8"] == -1)).sum())
                                       for k in range(3)])
                for name, E in (("real", A), ("placebo", P))}
PRIM = block()
RES["primary"] = PRIM
n_ok = min(PRIM["real_n"]) >= 150
RES["T1_parts"] = dict(est_ge_8=PRIM["T1"]["est"] >= 8, ci_lo_gt_0=PRIM["T1"]["lo"] > 0, n_ge_150=n_ok)
RES["T2_parts"] = dict(est_ge_8=PRIM["T2"]["est"] >= 8, ci_lo_gt_0=PRIM["T2"]["lo"] > 0)
RES["T1_pass"] = all(RES["T1_parts"].values())
RES["T2_pass"] = all(RES["T2_parts"].values())
if RES["T1_pass"] and RES["T2_pass"]:
    RES["verdict"] = "Family A: T1 and T2 pass — candidate edge (earns the single pre-registered holdout read)"
elif RES["T1_pass"]:
    RES["verdict"] = "Family A: T1 only — exhaustion sorts events but does not beat random entries"
elif RES["T2_pass"]:
    RES["verdict"] = "Family A: T2 only (not a verdict-table case) — recorded; no candidate edge"
else:
    RES["verdict"] = ("Family A: neither passes — recorded. Family B dropped at Step 0. Neither family passes: "
                      "the order-flow scalping line stops here on train")

S1 = {}
for Rt in R_LADDER:
    row = dict(breakeven=round((Rt + 1 + COST_TICKS) / (2 * Rt + 1) * 100, 2))
    for name, E in (("real", A), ("placebo", P)):
        c = cnt(E, Rt).sum(0)
        w = wr(c)
        tot = c.sum(0)
        wa = tot[0] / tot[1]
        row[name] = dict(n=c[:, 1].astype(int).tolist(), wr=(w * 100).round(2).tolist(),
                         exp=np.round(w * Rt - (1 - w) * (Rt + 1) - COST_TICKS, 3).tolist(), all_n=int(tot[1]),
                         all_wr=round(float(wa) * 100, 2), all_exp=round(float(wa * Rt - (1 - wa) * (Rt + 1) - COST_TICKS), 3))
    S1[str(Rt)] = row
RES["S1"] = S1
RES["S2"] = {lab: block(mr=m(A["mag"]), mp=m(P["mag"])) for lab, m in
             (("3-4sigma", lambda x: (x >= 3) & (x < 4)), (">=4sigma", lambda x: x >= 4))}
RES["S3"] = {lab: block(mr=A["dirn"] == v, mp=P["dirn"] == v) for lab, v in (("buy_after_down", -1), ("sell_after_up", 1))}
RES["S4"] = {lab: block(mr=A["near"] == v, mp=P["near"] == v) for lab, v in (("near_level", True), ("not_near", False))}
RES["S4"]["near_count"] = int(A["near"].sum())
b10a = A["t"].to_numpy() < np.array([ep(pt(d, 10, 0)) for d in A["d"]])
b10p = P["t_parent"].to_numpy() < np.array([ep(pt(d, 10, 0)) for d in P["d"]])
RES["S5"] = dict(before_10=block(mr=pd.Series(b10a, A.index), mp=pd.Series(b10p, P.index)),
                 after_10=block(mr=pd.Series(~b10a, A.index), mp=pd.Series(~b10p, P.index)))


def s6():
    mr, mp = A["X"] <= 0, P["X"] <= 0
    cr, cp = cnt(A, mask=mr).sum(1, keepdims=True), cnt(P, mask=mp).sum(1, keepdims=True)
    return dict(real_n=int(cr.sum(0)[0, 1]), real_wr=round(float(wr(cr.sum(0))[0]) * 100, 2),
                placebo_n=int(cp.sum(0)[0, 1]), placebo_wr=round(float(wr(cp.sum(0))[0]) * 100, 2),
                diff=ci(lambda a, b: wr(a)[..., 0] - wr(b)[..., 0], cr, cp))


RES["S6"] = s6()
RES["S7"] = block(terc="terc30")


def s14():
    out = {}
    for lab in ("30m", "60m", "close"):
        def agg(E):
            e = E[E[f"mv_{lab}"].notna()]
            s = np.zeros((nS, 3, 2))
            np.add.at(s, (e["sidx"].to_numpy(), e["terc"].to_numpy(), 0), e[f"mv_{lab}"].to_numpy())
            np.add.at(s, (e["sidx"].to_numpy(), e["terc"].to_numpy(), 1), 1)
            return s
        sr, sp = agg(A), agg(P)
        mean = lambda c: c[..., 0] / c[..., 1]  # noqa: E731
        def cim(fn, *cs):
            obs = fn(*[c.sum(0) for c in cs])
            b = fn(*[np.einsum("bs,sij->bij", W, c) for c in cs])
            lo, hi = np.nanpercentile(b, [2.5, 97.5])
            return dict(est=round(float(obs), 4), lo=round(float(lo), 4), hi=round(float(hi), 4))
        out[lab] = dict(real_mean_sigma=mean(sr.sum(0)).round(4).tolist(), real_n=sr.sum(0)[:, 1].astype(int).tolist(),
                        placebo_mean_sigma=mean(sp.sum(0)).round(4).tolist(),
                        low_minus_high=cim(lambda c: mean(c)[..., 0] - mean(c)[..., 2], sr),
                        low_minus_placebo=cim(lambda a, b: mean(a)[..., 0] - mean(b)[..., 0], sr, sp),
                        share_fade_side_real=[round(float((A.loc[A["terc"] == k, f"mv_{lab}"] > 0).mean()) * 100, 2)
                                              for k in range(3)],
                        share_fade_side_placebo=[round(float((P.loc[P["terc"] == k, f"mv_{lab}"] > 0).mean()) * 100, 2)
                                                 for k in range(3)])
    return out


RES["S14"] = s14()

OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "results.json").write_text(json.dumps(RES, indent=1, default=str))
try:
    A.to_pickle(OUT_DIR / "events_real.pkl")
    P.to_pickle(OUT_DIR / "events_placebo.pkl")
except Exception as exc:  # convenience dumps; results.json is the record
    print("event dump failed:", exc)

# ------------------------------------------------------------------------------------------- printing
print(json.dumps(RES, indent=1, default=str))
p = RES["primary"]
print(f"\nT1 (low − high X tercile, R = 8): {p['T1']['est']:+.2f} pts [{p['T1']['lo']:+.2f}, {p['T1']['hi']:+.2f}]; "
      f"resolved {p['real_n']} -> {'PASS' if RES['T1_pass'] else 'FAIL'} {RES['T1_parts']}")
print(f"T2 (low tercile − matched placebo): {p['T2']['est']:+.2f} pts [{p['T2']['lo']:+.2f}, {p['T2']['hi']:+.2f}] -> "
      f"{'PASS' if RES['T2_pass'] else 'FAIL'} {RES['T2_parts']}")
print("VERDICT:", RES["verdict"])
