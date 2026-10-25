#!/usr/bin/env python3
"""CR-BU Step 0 re-count under Amendment A1. READ-ONLY: SELECTs only, no API calls.

Spec: specs/CR-BU-put-node-support-test.md, Amendment A1 (multi-candidate node rule replaces L7/L8/L10).
Universe, loads and L1–L6 / L14 definitions are those of scripts/cr_bu_step0.py (unchanged; 433 usable).

  counts: days with >= 1 candidate, candidates per day, touched days, z of the touched candidate,
          rank of the touched candidate (1/2/3), first-touch time split (before / after 10:00 PT),
          touched node |net| ($bn = stored gex ÷ 1e9) median / IQR / by year
  counts: touched days if candidates were restricted to bins with net <= −100 $bn (by year) and <= −50 $bn
          (counts only — no floor adopted)
  A/A:    pseudo z-sets on random usable days through the full A1 harness; actual − placebo and its CI only

NO race is run at any actual node. The race harness runs only at A/A pseudo-levels and their placebo levels.

Definitions fixed here before the run:
  - candidates (A1.1): window bins (5-pt bucket_levels of corrected L, 0–7 DTE rows, P_target = S0) with
    open − 1.5·IM <= bin <= open − 0.25·IM and net < 0; sorted by net ascending (ties: higher bin first);
    greedy: take a bin if it is >= 10 pts from every bin already taken; at most 3. Rank = selection order.
  - net-floor variants: the same greedy run on window bins with net <= floor only.
  - event (A1.2): each candidate's E_i = N_i + 0.05·IM and its first touch index (first clean print strictly
    after the open print, through 13:00 PT, px <= E_i). Event = the candidate with the earliest touch index.
    Same-minute tie (one print per minute, so a gap through several E): the highest E — the first level
    the falling path crosses. Tie count is reported.
  - race (L9, A/A only): from the event print inclusive, at the event E, R = 0.20·IM; win = first print
    >= E + R before first print <= E − R; neither -> unresolved.
  - placebo (A1.3): for event day d with z-set {z_i}, every other usable day e gets P_i = open_e − z_i·IM_e,
    E_i = P_i + 0.05·IM_e, the same first-touch rule and race (R = 0.20·IM_e); rate = wins / resolved touched e.
  - bootstrap (L11): calendar months of the event date, 10 000 resamples, seed 20260920, 95 % percentile.
  - A/A: rng = default_rng(20261024); n_pseudo = days with >= 1 candidate; pseudo days drawn without
    replacement from usable days; z-sets drawn with replacement from the actual candidate z-sets.

Usage: apps/web/.venv/bin/python -u scripts/cr_bu_step0_a1.py | tee scripts/logs/cr_bu/step0_a1.log
"""
from __future__ import annotations

import os
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

pd.set_option("display.width", 220)

D_FROM, D_TO = date(2024, 4, 3), date(2026, 6, 5)
BIN, MIN_SEP, MAX_CAND = 5.0, 10.0, 3
WIN_FAR, WIN_NEAR, ENTRY_OFF, R_MULT = 1.5, 0.25, 0.05, 0.20
SEED_RNG, SEED_BOOT, N_BOOT = 20261024, 20260920, 10_000
ROLL_MISMATCH_EXTRA = {date(2026, 6, 15)}
T10 = time(10, 0)

CONN = get_backfill_db_conn()
assert_role_or_die(CONN)
CONN.execute("SET default_transaction_read_only = on")
assert CONN.execute("SELECT current_setting('default_transaction_read_only')").fetchone()[0] == "on"


def q(sql, params=None) -> pd.DataFrame:
    cur = CONN.execute(sql, params)
    return pd.DataFrame(cur.fetchall(), columns=[c.name for c in cur.description])


def hdr(s):
    print("\n" + "=" * 100 + f"\n{s}\n" + "=" * 100)


def third_friday(y: int, m: int) -> date:
    d = date(y, m, 15)
    return d + timedelta(days=(4 - d.weekday()) % 7)


def roll_week(d: date) -> bool:  # CR-BM/CR-BO definition
    if d.month not in (3, 6, 9, 12):
        return False
    tf = third_friday(d.year, d.month)
    return tf - timedelta(days=11) <= d <= tf


SESSIONS = [D_FROM + timedelta(days=i) for i in range((D_TO - D_FROM).days + 1)]
SESSIONS = [d for d in SESSIONS if is_trading_day(d)]
QEXP = {third_friday(y, m) for y in (2024, 2025, 2026) for m in (3, 6, 9, 12)}

# ---------------------------------------------------------------- loads + universe (as cr_bu_step0.py)
oig = q("""
    SELECT trade_date, expir_date, strike::float AS strike, stock_price::float AS stock_price,
           gex_call, gex_put
    FROM orats_oi_gamma
    WHERE ticker = 'SPX' AND trade_date BETWEEN %s AND %s
      AND expir_date >= trade_date AND expir_date <= trade_date + 10
""", (D_FROM - timedelta(days=7), D_TO))
oig["d"] = [t if is_trading_day(t) else next_trading_day(t) for t in oig["trade_date"]]
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
es_ok = {es_starts[pd.Timestamp(t).to_pydatetime()] for t, c in zip(es["datetime"], es["close"])
         if c is not None and np.isfinite(c) and c > 0}

daily, _, clean = spx_cash.load_spx_daily_bars(CONN, D_FROM, D_TO)
PRINTS = {d: g.sort_values("minute") for d, g in clean.groupby("session_date")}
have_rows = set(oig["d"])
USABLE = [d for d in SESSIONS
          if not roll_week(d) and d in es_ok and d not in QEXP and d not in ROLL_MISMATCH_EXTRA
          and d in feat.index and pd.notna(feat[d]) and feat[d] > 0 and d in daily.index and d in have_rows]
print(f"usable sessions: {len(USABLE)} (Step 0: 433)")
assert len(USABLE) == 433, "universe differs from Step 0 — stop"

OPEN = {d: float(daily.loc[d, "open"]) for d in USABLE}
IM = {d: float(feat[d]) for d in USABLE}
PX = {d: PRINTS[d]["px"].to_numpy() for d in USABLE}
MIN = {d: PRINTS[d]["minute"].to_numpy() for d in USABLE}

# window bins per day (net in stored gex units)
WBINS = {}
for d, g in oig[oig["d"].isin(USABLE)].groupby("d"):
    s0 = float(g.loc[g["expir_date"] == g["expir_date"].min(), "stock_price"].min())
    g = place_levels(g, s0)
    b = g.groupby(bucket_levels(g["level_corrected"].to_numpy(), BIN))["net"].sum()
    o, im = OPEN[d], IM[d]
    WBINS[d] = b[(b.index >= o - WIN_FAR * im) & (b.index <= o - WIN_NEAR * im)]


def candidates(bins: pd.Series, floor: float = 0.0) -> list[tuple[float, float]]:
    """A1.1 greedy: [(bin, net)] in rank order; net < 0 (or <= floor when floor < 0)."""
    neg = bins[bins < 0] if floor >= 0 else bins[bins <= floor]
    order = sorted(neg.items(), key=lambda kv: (kv[1], -kv[0]))
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


def first_touch(px: np.ndarray, es_: list[float]) -> tuple[int, int, bool]:
    """(candidate position, print index, tie) for the A1.2 event; (-1, -1, False) if none touched."""
    ti = [touch_index(px, e) for e in es_]
    hits = [(i, k) for k, i in enumerate(ti) if i >= 0]
    if not hits:
        return -1, -1, False
    i0 = min(i for i, _ in hits)
    same = [k for i, k in hits if i == i0]
    k = max(same, key=lambda kk: es_[kk])
    return k, i0, len(same) > 1


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


# -------------------------------------------------------------------------------- counts under A1
hdr("A1 RE-COUNT (low path only; no race at any actual node)")
CAND = {d: candidates(WBINS[d]) for d in USABLE}
CD = [d for d in USABLE if CAND[d]]
ncand = pd.Series({d: len(CAND[d]) for d in CD})
print(f"days with >= 1 candidate: {len(CD)} of {len(USABLE)}")
print(f"candidates per day: {ncand.value_counts().sort_index().to_dict()}  (mean {ncand.mean():.2f})")

EVT = {}
ties = 0
for d in CD:
    es_ = [b + ENTRY_OFF * IM[d] for b, _ in CAND[d]]
    k, i, tie = first_touch(PX[d], es_)
    ties += tie
    if k >= 0:
        b, n = CAND[d][k]
        EVT[d] = dict(rank=k + 1, N=b, net_bn=n / 1e9, z=(OPEN[d] - b) / IM[d], E=es_[k],
                      minute=pd.Timestamp(MIN[d][i]), n_cand=len(CAND[d]))
E = pd.DataFrame(EVT).T
print(f"\nTOUCHED DAYS: {len(E)} of {len(CD)} candidate days ({len(E) / len(CD):.1%})   "
      f"gate n >= 150: {'PASS' if len(E) >= 150 else 'FAIL — STOP (CR-BU closes as underpowered on train)'}")
print(f"same-minute multi-candidate ties: {ties}")
zt = E["z"].astype(float)
print(f"z of touched candidate: median {zt.median():.3f}, IQR {zt.quantile(.25):.3f}–{zt.quantile(.75):.3f}, "
      f"min {zt.min():.3f}, max {zt.max():.3f}")
print(f"touched rank: {E['rank'].value_counts().sort_index().to_dict()}")
early = sum(m.time() < T10 for m in E["minute"])
print(f"first touch before 10:00 PT / after: {early} / {len(E) - early}")
an = E["net_bn"].astype(float).abs()
print(f"touched node |net| $bn: median {an.median():.2f}, IQR {an.quantile(.25):.2f}–{an.quantile(.75):.2f}")
yr = np.array([d.year for d in E.index])
print("by year:")
print(pd.DataFrame({"n": an.groupby(yr).size(), "median": an.groupby(yr).median().round(2),
                    "p25": an.groupby(yr).quantile(.25).round(2),
                    "p75": an.groupby(yr).quantile(.75).round(2)}).to_string())
print(f"candidate days by year: {pd.Series([d.year for d in CD]).value_counts().sort_index().to_dict()}")

hdr("NET-FLOOR COUNTS (counts only; no floor adopted)")
for fl_bn in (-100.0, -50.0):
    tdays = []
    for d in USABLE:
        c = candidates(WBINS[d], fl_bn * 1e9)
        if c and first_touch(PX[d], [b + ENTRY_OFF * IM[d] for b, _ in c])[0] >= 0:
            tdays.append(d)
    by = pd.Series([d.year for d in tdays]).value_counts().sort_index().to_dict()
    cdays = sum(1 for d in USABLE if candidates(WBINS[d], fl_bn * 1e9))
    print(f"net <= {fl_bn:.0f} $bn: candidate days {cdays}, touched {len(tdays)}; by year {by}")

# --------------------------------------------------------------------------------------- A/A under A1
hdr("A/A CONTROL UNDER A1 (pseudo z-sets; full harness)")
ZSETS = [[(OPEN[d] - b) / IM[d] for b, _ in CAND[d]] for d in CD]


def outcome_zset(d: date, zs: list[float]) -> int:
    """-2 untouched, else race (1 / 0 / -1) at the first-touched pseudo-level."""
    im = IM[d]
    es_ = [OPEN[d] - zz * im + ENTRY_OFF * im for zz in zs]
    k, i, _ = first_touch(PX[d], es_)
    return -2 if k < 0 else race(PX[d], i, es_[k], R_MULT * im)


rng = np.random.default_rng(SEED_RNG)
n_pseudo = len(CD)
pdays = list(rng.choice(np.array(USABLE, dtype=object), size=n_pseudo, replace=False))
pzi = rng.integers(len(ZSETS), size=n_pseudo)
ev = []
n_touch_pseudo = n_unres_pseudo = 0
for d, j in zip(pdays, pzi):
    zs = ZSETS[j]
    a = outcome_zset(d, zs)
    n_touch_pseudo += a != -2
    n_unres_pseudo += a == -1
    if a < 0:
        continue
    res = np.array([outcome_zset(e, zs) for e in USABLE if e != d])
    ok = res >= 0
    if ok.sum():
        ev.append(dict(d=d, actual=a, placebo=res[ok].mean(), n_plc=int(ok.sum())))
EV = pd.DataFrame(ev)
dv = (EV["actual"] - EV["placebo"]).to_numpy()
months = np.array([f"{d.year}-{d.month:02d}" for d in EV["d"]])
um = np.unique(months)
idx = {m: np.nonzero(months == m)[0] for m in um}
brng = np.random.default_rng(SEED_BOOT)
boots = np.empty(N_BOOT)
for b in range(N_BOOT):
    boots[b] = dv[np.concatenate([idx[m] for m in brng.choice(um, size=len(um), replace=True)])].mean()
lo, hi = np.percentile(boots, [2.5, 97.5])
print(f"pseudo z-sets drawn {n_pseudo}; touched {n_touch_pseudo}, unresolved {n_unres_pseudo}; "
      f"touched+resolved pseudo events {len(EV)}; months {len(um)}; "
      f"median placebo n per event {EV['n_plc'].median():.0f}")
print(f"A/A actual − placebo = {dv.mean() * 100:+.2f} pts; 95 % month-cluster CI [{lo * 100:+.2f}, {hi * 100:+.2f}]"
      f"   CI covers 0: {'PASS' if lo <= 0 <= hi else 'FAIL — STOP'}")
print("\nDONE")
