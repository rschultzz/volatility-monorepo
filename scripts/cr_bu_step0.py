#!/usr/bin/env python3
"""CR-BU Step 0 — diagnosis for the put-node support test. READ-ONLY: SELECTs only, no API calls.

Spec: specs/CR-BU-put-node-support-test.md (Step 0 gate questions 1–6; Q1 is a git/spec read, reported in the spec).

  2  counts: usable train sessions after L3 / each L14 clause / data availability; days with a node (L7);
     z_d distribution; days whose node is touched (L8, low path only)
  3  node anatomy: 0DTE vs 1–7 DTE share of node-bin net, median |net| ($bn), median distance (pts)
  4  worked day (random node day, seed 20261024): node rows, bin, corrected L vs discounted_level, touch minute
  5  A/A control: pseudo-nodes on random usable days at z drawn from the actual z_d distribution, through
     the full L8–L11 harness; actual − placebo with the month-cluster bootstrap CI (must cover 0)
  6  IM source: canonical bt_daily_features version, IM null/0 counts

NO race is run at any actual node. The race harness runs only at A/A pseudo-node levels and their
leave-one-out placebo levels; only actual − placebo and its CI are printed (no win rates).

Definitions fixed here before the run:
  - connection: get_backfill_db_conn() + SET default_transaction_read_only = on (role asserted).
  - session D: NYSE trading day in [2024-04-03, 2026-06-05] (L3). orats_oi_gamma.trade_date not a trading
    day -> remapped to next_trading_day (CR-BK A1.6); collisions with a properly stamped row are reported.
  - L3 floor count: trading days 2023-05-01 .. 2024-04-02 that carry orats_oi_gamma rows (train start per
    field-conventions; the sessions L3 removes).
  - L14 (a) roll week: Monday of the week before the quarterly third Friday through that Friday (D tested).
    (b) no ES bar starting 15:45 America/New_York on A = prev_trading_day(D), or its close not positive.
    (c) roll mismatch per CR-BT A1.4: D = a quarterly expiry Friday (2024-06-21 .. 2026-03-20) or D = 2026-06-15.
  - data: canonical bt_daily_features row (CANONICAL_FEATURE_VERSION, ticker SPX, active); IM =
    feature_vector->>'implied_move_1d', null or <= 0 excluded (L5); SPX-cash open (L4, spx_cash hybrid
    series, open = first clean print >= 06:33 PT); node rows present.
  - node rows (L6): ticker SPX, expir_date in [D, D+7]; net = gex_call − |gex_put|. S0 = MIN(stock_price)
    of the nearest expiry >= D. L = place_levels(rows, S0) unchanged (F_e = row stock_price).
  - node (L7): bins = bucket_levels(L, 5) (PG-compatible); bin net = sum; window bins with
    open − 1.5·IM <= bin <= open − 0.25·IM; N = the most negative bin net (< 0); none -> no node.
  - z_d = (open − N)/IM. E = N + 0.05·IM. Touch (L8) = first clean print strictly after the open print,
    through 13:00 PT, with px <= E. The series is one print per minute, so low = high = the print.
  - race (L9, A/A only): from the entry print inclusive, win = first print >= E + R before first print
    <= E − R, R = 0.20·IM; neither -> unresolved (excluded). One print per minute, so no same-minute tie.
  - placebo (L10): for an event on day d at z, every other usable day e at P_e = open_e − z·IM_e,
    E_e = P_e + 0.05·IM_e, R_e = 0.20·IM_e; rate = wins / resolved touched e.
  - bootstrap (L11): calendar months of the event date, 10 000 resamples, seed 20260920, 95 % percentile.
  - rng = numpy default_rng(20261024): first the worked day (uniform over node days), then A/A:
    n_pseudo = number of node days, days drawn without replacement from usable days, z drawn with
    replacement from the actual z_d.

Usage: apps/web/.venv/bin/python -u scripts/cr_bu_step0.py | tee scripts/logs/cr_bu/step0.log
"""
from __future__ import annotations

import os
import sys
import warnings
from datetime import date, datetime, timedelta
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
pd.set_option("display.max_rows", 400)
pd.set_option("display.max_columns", 40)

TRAIN_START_FIELD = date(2023, 5, 1)          # field-conventions train start (L3 floor count)
D_FROM, D_TO = date(2024, 4, 3), date(2026, 6, 5)
BIN = 5.0
WIN_FAR, WIN_NEAR, ENTRY_OFF, R_MULT = 1.5, 0.25, 0.05, 0.20
SEED_RNG, SEED_BOOT, N_BOOT = 20261024, 20260920, 10_000
ROLL_MISMATCH_EXTRA = {date(2026, 6, 15)}     # CR-BT A1.4 (2026-03-20 is a quarterly expiry Friday)

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


def trading_days(a: date, b: date) -> list[date]:
    out, d = [], a
    while d <= b:
        if is_trading_day(d):
            out.append(d)
        d += timedelta(days=1)
    return out


SESSIONS = trading_days(D_FROM, D_TO)
QEXP = {third_friday(y, m) for y in (2024, 2025, 2026) for m in (3, 6, 9, 12)}

# ---------------------------------------------------------------------------------------------- loads
hdr("LOADS")
oig = q("""
    SELECT trade_date, expir_date, strike::float AS strike, stock_price::float AS stock_price,
           gex_call, gex_put, discounted_level
    FROM orats_oi_gamma
    WHERE ticker = 'SPX' AND trade_date BETWEEN %s AND %s
      AND expir_date >= trade_date AND expir_date <= trade_date + 10
""", (D_FROM - timedelta(days=7), D_TO))
oig["store_date"] = oig["trade_date"]
oig["d"] = [t if is_trading_day(t) else next_trading_day(t) for t in oig["trade_date"]]
remapped = sorted(set(oig.loc[oig["d"] != oig["store_date"], "store_date"]))
collide = sorted({next_trading_day(t) for t in remapped} & set(oig.loc[oig["d"] == oig["store_date"], "d"]))
print(f"orats_oi_gamma rows (expir within 10 cd of store date): {len(oig):,}; holiday-stamped store dates "
      f"remapped: {[str(x) for x in remapped]}; collisions with a properly stamped row: {[str(x) for x in collide]}")
oig = oig[(oig["d"] >= D_FROM) & (oig["d"] <= D_TO)]
oig = oig[[d <= e <= d + timedelta(days=7) for e, d in zip(oig["expir_date"], oig["d"])]]
oig["net"] = oig["gex_call"] - oig["gex_put"].abs()
print(f"node-universe rows (expir in [D, D+7]): {len(oig):,}; sessions with rows: {oig['d'].nunique()}; "
      f"gex_put > 0 rows: {(oig['gex_put'] > 0).sum()}; null stock_price/strike: "
      f"{oig['stock_price'].isna().sum()}/{oig['strike'].isna().sum()}; null discounted_level: "
      f"{oig['discounted_level'].isna().sum()}")

feat = q("""
    SELECT trade_date, (feature_vector->>'implied_move_1d')::float AS im
    FROM bt_daily_features
    WHERE feature_version = %s AND ticker = 'SPX' AND active AND trade_date BETWEEN %s AND %s
""", (CANONICAL_FEATURE_VERSION, D_FROM, D_TO)).set_index("trade_date")["im"]
dup_feat = feat.index.duplicated().sum()
feat = feat[~feat.index.duplicated()]

es_starts = {es_snapshot_bar_start_utc(prev_trading_day(d)): d for d in SESSIONS}
es = q("SELECT datetime, close::float AS close FROM ironbeam_es_1m_bars WHERE datetime = ANY(%s)",
       (list(es_starts.keys()),))
es_ok = {es_starts[pd.Timestamp(t).to_pydatetime()] for t, c in zip(es["datetime"], es["close"])
         if c is not None and np.isfinite(c) and c > 0}

daily, dropped, clean = spx_cash.load_spx_daily_bars(CONN, D_FROM, D_TO)
print(f"SPX cash: {len(daily)} sessions with prints, {len(clean):,} clean prints, {len(dropped):,} dropped")
PRINTS = {d: g.sort_values("minute") for d, g in clean.groupby("session_date")}

# ---------------------------------------------------------------------------------------- Q2 exclusions
hdr("Q2 — SESSION COUNTS (no outcomes)")
floor_n = q("""SELECT count(DISTINCT trade_date) AS n FROM orats_oi_gamma
               WHERE ticker = 'SPX' AND trade_date >= %s AND trade_date < %s""",
            (TRAIN_START_FIELD, D_FROM))["n"].iloc[0]
floor_td = len(trading_days(TRAIN_START_FIELD, D_FROM - timedelta(days=1)))
print(f"L3 floor: trading days {TRAIN_START_FIELD} .. {D_FROM - timedelta(days=1)} = {floor_td}; "
      f"with orats_oi_gamma rows = {floor_n} (removed)")
print(f"Trading sessions in [{D_FROM}, {D_TO}]: {len(SESSIONS)}")

ex_a = {d for d in SESSIONS if roll_week(d)}
ex_b = {d for d in SESSIONS if d not in es_ok}
ex_c = {d for d in SESSIONS if d in QEXP or d in ROLL_MISMATCH_EXTRA}
print(f"L14 gross (each clause alone):  (a) roll week {len(ex_a)}  (b) no ES 15:45 ET bar on A {len(ex_b)}  "
      f"(c) roll mismatch {len(ex_c)}")
print(f"  (b) sessions: {[str(x) for x in sorted(ex_b)]}")
print(f"  (c) sessions: {[str(x) for x in sorted(ex_c)]}")
print(f"  (b) outside (a): {[str(x) for x in sorted(ex_b - ex_a)]}")
print(f"  (c) outside (a)∪(b): {[str(x) for x in sorted(ex_c - ex_a - ex_b)]}")

U = list(SESSIONS)
steps = []
for name, rule in [
    ("L14a roll week", lambda d: d in ex_a),
    ("L14b no ES 15:45 ET bar on A", lambda d: d in ex_b),
    ("L14c roll mismatch", lambda d: d in ex_c),
    ("no canonical bt_daily_features row", lambda d: d not in feat.index),
    ("IM null or <= 0", lambda d: d in feat.index and not (pd.notna(feat[d]) and feat[d] > 0)),
    ("no SPX-cash open (L4)", lambda d: d not in daily.index),
    ("no node rows (orats_oi_gamma)", lambda d: d not in set(oig["d"])),
]:
    rm = [d for d in U if rule(d)]
    U = [d for d in U if not rule(d)]
    steps.append((name, len(rm), len(U), [str(x) for x in rm]))
print(f"\n{'sequential exclusion':40s} {'removed':>8s} {'remaining':>10s}")
print(f"{'start':40s} {'':>8s} {len(SESSIONS):>10d}")
for name, n, rem, lst in steps:
    print(f"{name:40s} {n:>8d} {rem:>10d}   {lst if n <= 40 else ''}")
print(f"canonical feature rows in range: {len(feat)} (duplicate trade_date rows: {dup_feat}); sessions without one: "
      f"{[str(d) for d in SESSIONS if d not in feat.index]}")
USABLE = U
print(f"\nUSABLE SESSIONS: {len(USABLE)}")

# ------------------------------------------------------------------------------------------------ nodes
OPEN = {d: float(daily.loc[d, "open"]) for d in USABLE}
IM = {d: float(feat[d]) for d in USABLE}
nodes, node_rows = {}, {}
for d, g in oig[oig["d"].isin(USABLE)].groupby("d"):
    s0 = float(g.loc[g["expir_date"] == g["expir_date"].min(), "stock_price"].min())
    g = place_levels(g, s0)
    g["bin"] = bucket_levels(g["level_corrected"].to_numpy(), BIN)
    g["s0"] = s0
    o, im = OPEN[d], IM[d]
    bins = g.groupby("bin")["net"].sum()
    w = bins[(bins.index >= o - WIN_FAR * im) & (bins.index <= o - WIN_NEAR * im) & (bins < 0)]
    if len(w):
        b = float(w.idxmin())
        nodes[d] = dict(N=b, net=float(w.min()), z=(o - b) / im, dist=o - b, E=b + ENTRY_OFF * im, s0=s0)
        node_rows[d] = g[g["bin"] == b]
ND = sorted(nodes)
print(f"days with a node: {len(ND)} of {len(USABLE)} usable ({len(USABLE) - len(ND)} with no negative bin in window)")


def touch_index(px: np.ndarray, e: float) -> int:
    """First index >= 1 (strictly after the open print) with px <= e; -1 if none."""
    hit = np.nonzero(px[1:] <= e)[0]
    return int(hit[0]) + 1 if len(hit) else -1


for d in ND:
    p = PRINTS[d]
    i = touch_index(p["px"].to_numpy(), nodes[d]["E"])
    nodes[d]["touch_i"] = i
    nodes[d]["touch_min"] = p["minute"].iloc[i] if i >= 0 else None
T = [d for d in ND if nodes[d]["touch_i"] >= 0]
z = pd.Series({d: nodes[d]["z"] for d in ND})
print("\nz_d distribution (days with a node):")
print(z.describe(percentiles=[.05, .1, .25, .5, .75, .9, .95]).round(3).to_string())
hist = pd.cut(z, np.arange(0.25, 1.5001, 0.125), include_lowest=True).value_counts().sort_index()
print(hist.to_string())
print(f"\nTOUCHED (low path only): {len(T)} of {len(ND)} node days ({len(T) / max(len(ND), 1):.1%})"
      f"   gate n >= 150: {'PASS' if len(T) >= 150 else 'FAIL — STOP'}")
tz = pd.Series({d: nodes[d]["z"] for d in T})
print(f"z_d touched: median {tz.median():.3f}; untouched: median {z.drop(T).median():.3f}")
early = sum(1 for d in T if nodes[d]["touch_min"].time() < datetime.strptime("10:00", "%H:%M").time())
print(f"S1 split counts (touch before 10:00 PT / after): {early} / {len(T) - early}")
by_year = pd.Series([d.year for d in T]).value_counts().sort_index()
print(f"touched by year: {by_year.to_dict()}; node days by year: "
      f"{pd.Series([d.year for d in ND]).value_counts().sort_index().to_dict()}")

# ---------------------------------------------------------------------------------------- Q3 anatomy
hdr("Q3 — NODE ANATOMY (node bins, all node days)")
rows = []
for d in ND:
    g = node_rows[d]
    zero = float(g.loc[g["expir_date"] == d, "net"].sum())
    tot = float(g["net"].sum())
    rows.append(dict(d=d, net_bn=tot / 1e9, net0_bn=zero / 1e9, net17_bn=(tot - zero) / 1e9,
                     share0=zero / tot, dist=nodes[d]["dist"], n_rows=len(g), n_exp=g["expir_date"].nunique(),
                     n_strikes=g["strike"].nunique()))
A = pd.DataFrame(rows).set_index("d")
print(f"aggregate share of node net: 0DTE {A['net0_bn'].sum() / A['net_bn'].sum():.1%}, "
      f"1–7 DTE {A['net17_bn'].sum() / A['net_bn'].sum():.1%}  (signed sums; a share can exceed 100 % "
      f"when the other leg is net positive)")
print(f"per-day share0: median {A['share0'].median():.1%}, IQR {A['share0'].quantile(.25):.1%}–"
      f"{A['share0'].quantile(.75):.1%}; days 0DTE-majority (share0 > 50 %): {(A['share0'] > .5).sum()} / {len(A)}")
print(f"median node |net| {A['net_bn'].abs().median():.2f} $bn (IQR {A['net_bn'].abs().quantile(.25):.2f}–"
      f"{A['net_bn'].abs().quantile(.75):.2f}; units: stored gex ÷ 1e9, the chart's 'B')")
print(f"median node distance open − N {A['dist'].median():.1f} pts (IQR {A['dist'].quantile(.25):.1f}–"
      f"{A['dist'].quantile(.75):.1f}); median IM {np.median([IM[d] for d in ND]):.1f} pts")
print(f"rows per node bin: median {A['n_rows'].median():.0f}; expiries {A['n_exp'].median():.0f}; strikes "
      f"{A['n_strikes'].median():.0f}")
At = A.loc[T]
print(f"touched subset: 0DTE aggregate share {At['net0_bn'].sum() / At['net_bn'].sum():.1%}, median |net| "
      f"{At['net_bn'].abs().median():.2f} $bn, median distance {At['dist'].median():.1f} pts")
print("by year (all node days): median |net| $bn / median dist / median share0")
print(A.groupby([d.year for d in A.index]).agg(net=("net_bn", lambda x: x.abs().median()),
                                               dist=("dist", "median"), share0=("share0", "median")).round(3))

# ------------------------------------------------------------------------------------- Q4 worked day
rng = np.random.default_rng(SEED_RNG)
WD = ND[int(rng.integers(len(ND)))]
hdr(f"Q4 — WORKED DAY {WD} (random node day, seed {SEED_RNG}); race hidden")
n = nodes[WD]
print(f"A = {prev_trading_day(WD)}; S0 {n['s0']:.2f}; open {OPEN[WD]:.2f}; IM {IM[WD]:.2f}; window "
      f"[{OPEN[WD] - WIN_FAR * IM[WD]:.2f}, {OPEN[WD] - WIN_NEAR * IM[WD]:.2f}]")
print(f"node bin N = {n['N']:.0f}; net {n['net'] / 1e9:.2f} $bn; z {n['z']:.3f}; E = {n['E']:.2f}; touch minute "
      f"{n['touch_min']}")
g = node_rows[WD].copy()
g["dte"] = [(e - WD).days for e in g["expir_date"]]
g["net_bn"] = g["net"] / 1e9
g["L_minus_dl"] = g["level_corrected"] - g["discounted_level"]
print(g[["strike", "expir_date", "dte", "stock_price", "discounted_level", "level_corrected", "L_minus_dl",
         "net_bn"]].sort_values("net_bn").round(2).to_string(index=False))
# the same day's neighbouring window bins, for context
gg = place_levels(oig[oig["d"] == WD], n["s0"])
gg["bin"] = bucket_levels(gg["level_corrected"].to_numpy(), BIN)
bb = (gg.groupby("bin")["net"].sum() / 1e9)
bb = bb[(bb.index >= OPEN[WD] - WIN_FAR * IM[WD]) & (bb.index <= OPEN[WD] - WIN_NEAR * IM[WD])]
print("window bins, five most negative ($bn):", bb.nsmallest(5).round(2).to_dict())
# where the same OI sits under discounted_level binning
gd = oig[oig["d"] == WD].copy()
gd["bin_dl"] = bucket_levels(gd["discounted_level"].to_numpy(), BIN)
print("same window under discounted_level bins, five most negative ($bn):",
      (gd.groupby("bin_dl")["net"].sum() / 1e9).loc[lambda s: (s.index >= OPEN[WD] - WIN_FAR * IM[WD])
                                                    & (s.index <= OPEN[WD] - WIN_NEAR * IM[WD])]
      .nsmallest(5).round(2).to_dict())
p = PRINTS[WD]
i = n["touch_i"]
if i >= 0:
    print("prints around the touch (path to touch only):")
    print(p.iloc[max(0, i - 3): i + 1][["minute", "px"]].to_string(index=False))

# ----------------------------------------------------------------------------------------- Q5 A/A
hdr("Q5 — A/A CONTROL (pseudo-nodes; full L8–L11 harness)")
PX = {d: PRINTS[d]["px"].to_numpy() for d in USABLE}


def race(px: np.ndarray, i: int, e: float, r: float) -> int:
    """1 win, 0 loss, -1 unresolved; from print i inclusive."""
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


def outcome_at_z(d: date, zz: float) -> int:
    """-2 untouched, else race result (1 / 0 / -1) at P = open − z·IM."""
    im = IM[d]
    e = OPEN[d] - zz * im + ENTRY_OFF * im
    i = touch_index(PX[d], e)
    return -2 if i < 0 else race(PX[d], i, e, R_MULT * im)


n_pseudo = len(ND)
pdays = list(rng.choice(np.array(USABLE, dtype=object), size=n_pseudo, replace=False))
pz = rng.choice(z.to_numpy(), size=n_pseudo, replace=True)
ev = []
for d, zz in zip(pdays, pz):
    a = outcome_at_z(d, zz)
    if a < 0:
        continue                          # untouched or unresolved -> not a resolved event
    res = np.array([outcome_at_z(e, zz) for e in USABLE if e != d])
    ok = res >= 0
    if ok.sum() == 0:
        continue
    ev.append(dict(d=d, z=zz, actual=a, placebo=res[ok].mean(), n_plc=int(ok.sum())))
EV = pd.DataFrame(ev)
diff = EV["actual"] - EV["placebo"]
months = np.array([f"{d.year}-{d.month:02d}" for d in EV["d"]])
um = np.unique(months)
idx = {m: np.nonzero(months == m)[0] for m in um}
brng = np.random.default_rng(SEED_BOOT)
dv = diff.to_numpy()
boots = np.empty(N_BOOT)
for b in range(N_BOOT):
    pick = brng.choice(um, size=len(um), replace=True)
    sel = np.concatenate([idx[m] for m in pick])
    boots[b] = dv[sel].mean()
lo, hi = np.percentile(boots, [2.5, 97.5])
print(f"pseudo-nodes drawn {n_pseudo}; touched+resolved pseudo events {len(EV)}; months {len(um)}; "
      f"median placebo n per event {EV['n_plc'].median():.0f}")
print(f"A/A actual − placebo = {dv.mean() * 100:+.2f} pts; 95 % month-cluster CI [{lo * 100:+.2f}, {hi * 100:+.2f}]"
      f"   gate CI covers 0: {'PASS' if lo <= 0 <= hi else 'FAIL — STOP'}")

# ------------------------------------------------------------------------------------------- Q6 IM
hdr("Q6 — IM SOURCE")
print(f"canonical version: {CANONICAL_FEATURE_VERSION} (packages/shared/canonical_version.py), ticker SPX, active")
print(f"rows in [{D_FROM}, {D_TO}]: {len(feat)}; IM null: {feat.isna().sum()}; IM <= 0: {(feat <= 0).sum()} "
      f"{[str(d) for d in feat.index[(feat <= 0) | feat.isna()]]}")
print(f"IM over usable sessions: median {np.median(list(IM.values())):.2f} pts, IQR "
      f"{np.percentile(list(IM.values()), 25):.2f}–{np.percentile(list(IM.values()), 75):.2f}")
print("\nDONE")
