#!/usr/bin/env python3
"""CR-BT Step 0 — diagnosis for corrected level placement. READ-ONLY: SELECTs only (read-only session), no API calls.

Spec: specs/CR-BT-corrected-level-placement.md (Step 0 gate questions 1–6; 5 and 6 are code reads, not here).

  1  stock_price is a per-expiry forward: F_e/S0 − 1 vs calendar days to expiry, 20 random train dates + 2026-10-23
  2  |F_matched − ES(15:45 ET bar close)| on the expiry matched to the front ES contract (gate: median ≤ 5 pts)
     + L4 alignment: the same residual at neighbouring bar starts 15:40–15:50 ET
  3  ES contract identity at the snapshot (2026 front month incl. the 2026-09 patch window; 2023–25 roll-week counts)
  4  worked day 2026-10-23: top-10 |GEX| rows, then the selected lines under Current vs Corrected

Definitions fixed here before the run:
  - session date D = orats_oi_gamma.trade_date; API date A = prev_trading_day(D) (job_orats_eod: D = next_trading_day(A)).
  - live rows = expir_date >= D (the chart's filter). S0 = stock_price of the nearest live expiry (CR-BK A1.1).
  - snapshot ES = close of the ironbeam_es_1m_bars bar whose UTC start is 15:45 America/New_York on A (zoneinfo).
  - front ES contract per the ingest rule (apps/cron/ironbeam_ingest.py, and CR-BO's "dying contract through expiry"):
    the first quarterly (Mar/Jun/Sep/Dec) third Friday >= A. Matched SPX expiry = the live expir_date nearest it.
  - train = sessions 2023-05-01 … 2026-06-05; sample = 20 train dates drawn with numpy seed 17, plus 2026-10-23.

Usage: apps/web/.venv/bin/python -u scripts/cr_bt_step0.py
"""
from __future__ import annotations

import os
import sys
import warnings
from datetime import date, datetime, time, timedelta
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
import psycopg2  # noqa: E402

from packages.shared.trading_calendar import prev_trading_day  # noqa: E402

pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 400)
pd.set_option("display.max_columns", 40)

ET = ZoneInfo("America/New_York")
PT = ZoneInfo("America/Los_Angeles")
UTC = ZoneInfo("UTC")
TRAIN_START, TRAIN_END = date(2023, 5, 1), date(2026, 6, 5)
WORKED_DAY = date(2026, 10, 23)
PATCH_FROM, PATCH_TO = datetime(2026, 9, 13, 22, 0), datetime(2026, 9, 18, 20, 59)

# Chart constants (apps/web/modules/Ironbeam/callbacks.py defaults; env overrides not set locally)
GEX_LEVEL_BUCKET, GEX_LEVEL_PADDING, GEX_ABS_THRESHOLD = 1.0, 150.0, 1e10
GEX_MIN_LEVEL_SPACING, GEX_MAX_LEVELS_PER_DAY = 5.0, 80


def conn():
    url = os.environ["DATABASE_URL"].replace("postgresql+psycopg2://", "postgresql://").replace(
        "postgresql+psycopg://", "postgresql://")
    c = psycopg2.connect(url)
    c.set_session(readonly=True, autocommit=True)
    return c


def q(sql, params=None):
    with conn() as c:
        return pd.read_sql(sql, c, params=params)


def third_friday(y: int, m: int) -> date:
    d = date(y, m, 15)
    return d + timedelta(days=(4 - d.weekday()) % 7)


def quarterlies_from(a: date, n: int = 2) -> list[date]:
    out, y, m = [], a.year, a.month
    while len(out) < n:
        if m in (3, 6, 9, 12) and third_friday(y, m) >= a:
            out.append(third_friday(y, m))
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def roll_week(d: date) -> bool:  # CR-BM/CR-BO definition
    if d.month not in (3, 6, 9, 12):
        return False
    tf = third_friday(d.year, d.month)
    return tf - timedelta(days=11) <= d <= tf


def et_bar_utc(a: date, hh: int, mm: int) -> datetime:
    return datetime.combine(a, time(hh, mm), tzinfo=ET).astimezone(UTC).replace(tzinfo=None)


def hdr(s):
    print("\n" + "=" * 100 + f"\n{s}\n" + "=" * 100)


# ------------------------------------------------------------------ data
hdr("LOAD")
exp = q("""
    SELECT trade_date, expir_date,
           COUNT(*)                                                   AS n_rows,
           MIN(stock_price)::float8                                   AS f_min,
           MAX(stock_price)::float8                                   AS f_max,
           SUM(CASE WHEN stock_price IS NULL OR stock_price = 0 THEN 1 ELSE 0 END) AS n_bad_f,
           SUM(CASE WHEN discounted_level IS NULL THEN 1 ELSE 0 END)  AS n_null_dl,
           SUM(ABS(COALESCE(gex_call,0)) + ABS(COALESCE(gex_put,0)))  AS gex_abs,
           SUM(CASE WHEN stock_price IS NULL OR stock_price = 0
                    THEN ABS(COALESCE(gex_call,0)) + ABS(COALESCE(gex_put,0)) ELSE 0 END) AS gex_abs_bad_f,
           SUM(CASE WHEN discounted_level IS NULL
                    THEN ABS(COALESCE(gex_call,0)) + ABS(COALESCE(gex_put,0)) ELSE 0 END) AS gex_abs_null_dl
    FROM orats_oi_gamma
    WHERE ticker = 'SPX' AND expir_date >= trade_date
    GROUP BY trade_date, expir_date
    ORDER BY trade_date, expir_date
""")
exp["trade_date"] = pd.to_datetime(exp["trade_date"]).dt.date
exp["expir_date"] = pd.to_datetime(exp["expir_date"]).dt.date
dates = sorted(exp["trade_date"].unique())
print(f"live (trade_date, expir_date) groups: {len(exp):,}; sessions {len(dates)} ({dates[0]} → {dates[-1]})")
print(f"groups whose stock_price varies within the expiry (f_max − f_min > 0.005): "
      f"{int(((exp.f_max - exp.f_min) > 0.005).sum())}")

api = {d: prev_trading_day(d) for d in dates}

# ES bars 15:40–15:50 ET on every API date
want = {}
for d in dates:
    a = api[d]
    for mm in range(40, 51):
        want[et_bar_utc(a, 15, mm)] = (d, mm)
es = q("SELECT datetime, close, source FROM ironbeam_es_1m_bars WHERE datetime = ANY(%s)",
       (list(want.keys()),))
es["datetime"] = pd.to_datetime(es["datetime"])
es["d"] = es["datetime"].map(lambda t: want[t.to_pydatetime()][0])
es["mm"] = es["datetime"].map(lambda t: want[t.to_pydatetime()][1])
es_wide = es.pivot_table(index="d", columns="mm", values="close")
es_src = es[es.mm == 45].set_index("d")["source"]
print(f"ES 15:45 ET bar present for {es_src.size}/{len(dates)} sessions")

# ------------------------------------------------------------------ per-session frame
rows = []
for d, g in exp.groupby("trade_date"):
    a = api[d]
    g = g.sort_values("expir_date")
    good = g[(g.f_min > 0)]
    if good.empty:
        continue
    s0 = float(good.iloc[0].f_min)
    es_expiry = quarterlies_from(a, 2)
    es_close = es_wide.loc[d, 45] if d in es_wide.index and 45 in es_wide.columns else np.nan
    rec = dict(d=d, a=a, s0=s0, es45=es_close, es_src=es_src.get(d, None),
               es_exp_front=es_expiry[0], es_exp_next=es_expiry[1])
    for tag, tf in (("front", es_expiry[0]), ("next", es_expiry[1])):
        idx = (good["expir_date"].map(lambda e: abs((e - tf).days))).idxmin()
        rec[f"m_{tag}_exp"] = good.loc[idx, "expir_date"]
        rec[f"m_{tag}_f"] = float(good.loc[idx, "f_min"])
        rec[f"m_{tag}_gap_days"] = abs((good.loc[idx, "expir_date"] - tf).days)
    rows.append(rec)
S = pd.DataFrame(rows).set_index("d")
S["diff_front"] = S["m_front_f"] - S["es45"]
S["diff_next"] = S["m_next_f"] - S["es45"]
S["train"] = [(TRAIN_START <= d <= TRAIN_END) for d in S.index]
S["roll_A"] = [roll_week(a) for a in S["a"]]
S["roll_D"] = [roll_week(d) for d in S.index]
S["a_is_es_expiry"] = S["a"] == S["es_exp_front"]

train_dates = [d for d in S.index if S.loc[d, "train"]]
rng = np.random.default_rng(17)
sample = sorted(rng.choice(np.array(train_dates, dtype=object), size=20, replace=False).tolist()) + [WORKED_DAY]
print("sample:", ", ".join(str(d) for d in sample))

# ------------------------------------------------------------------ Q1
hdr("Q1 — stock_price is a per-expiry forward: F_e/S0 − 1 vs calendar days to expiry (from the API date)")
q1 = []
for d in sample:
    g = exp[exp.trade_date == d].copy()
    a = api[d]
    s0 = S.loc[d, "s0"]
    g["cd"] = g["expir_date"].map(lambda e: (e - a).days)
    g["r_bp"] = (g["f_min"] / s0 - 1) * 1e4
    gg = g[g.f_min > 0].sort_values("cd")
    dr = np.diff(gg["r_bp"].to_numpy())
    pts = np.diff(gg["f_min"].to_numpy())
    viol0 = int((dr < 0).sum())
    viol1 = int((pts < -1.0).sum())  # forward falls by more than 1 pt between consecutive expiries
    rho = pd.Series(gg["cd"].to_numpy()).corr(pd.Series(gg["r_bp"].to_numpy()), method="spearman")
    # carry per calendar day implied by the farthest expiry ≤ 120 cd
    far = gg[gg.cd <= 120].iloc[-1]
    q1.append(dict(
        d=d, n_exp=len(g), n_bad_f_exp=int((g.n_bad_f > 0).sum()), bad_rows=int(g.n_bad_f.sum()),
        bad_gex_share=float(g.gex_abs_bad_f.sum() / max(g.gex_abs.sum(), 1)),
        null_dl_rows=int(g.n_null_dl.sum()), null_dl_gex_share=float(g.gex_abs_null_dl.sum() / max(g.gex_abs.sum(), 1)),
        s0=round(s0, 2), spearman=round(rho, 4), n_down_any=viol0, n_down_gt1pt=viol1,
        r_7cd_bp=round(float(np.interp(7, gg.cd, gg.r_bp)), 1),
        r_30cd_bp=round(float(np.interp(30, gg.cd, gg.r_bp)), 1),
        r_90cd_bp=round(float(np.interp(90, gg.cd, gg.r_bp)), 1),
        pts_per_cd_to_120=round((far.f_min - s0) / max(far.cd, 1), 3),
    ))
q1 = pd.DataFrame(q1)
print(q1.to_string(index=False))

print("\nnon-monotone detail (consecutive expiries where F falls by > 1 pt):")
for d in sample:
    g = exp[(exp.trade_date == d) & (exp.f_min > 0)].sort_values("expir_date")
    f = g["f_min"].to_numpy()
    for i in np.where(np.diff(f) < -1.0)[0]:
        print(f"  {d}: {g.iloc[i].expir_date} F={f[i]:.2f} → {g.iloc[i+1].expir_date} F={f[i+1]:.2f} "
              f"(Δ {f[i+1]-f[i]:+.2f}); |GEX| share of the later expiry "
              f"{g.iloc[i+1].gex_abs / g.gex_abs.sum():.4f}")

allg = exp
print(f"\nall sessions, live rows: stock_price null/0 on {int(allg.n_bad_f.sum()):,} of {int(allg.n_rows.sum()):,} rows "
      f"({allg.gex_abs_bad_f.sum() / allg.gex_abs.sum():.2e} of |GEX|); "
      f"discounted_level NULL on {int(allg.n_null_dl.sum()):,} rows ({allg.gex_abs_null_dl.sum() / allg.gex_abs.sum():.2e} of |GEX|)")
bad_by_date = allg[allg.n_bad_f > 0].groupby("trade_date").n_bad_f.sum()
if len(bad_by_date):
    print("sessions with any null/0 stock_price (live):", bad_by_date.to_dict())
nd = allg[allg.n_null_dl > 0].groupby("trade_date").agg(rows=("n_null_dl", "sum"), gex=("gex_abs_null_dl", "sum"))
print(f"sessions with any NULL discounted_level (live): {len(nd)}")
if len(nd):
    print(nd.tail(15).to_string())

# ------------------------------------------------------------------ Q1b
hdr("Q1b — regime: per session, Spearman(F_e, cd) over live expiries ≤ 120 cd, and the ≤120-cd carry in pts/cd")
reg = []
for d, g in exp[exp.f_min > 0].groupby("trade_date"):
    a = api[d]
    g = g.assign(cd=g["expir_date"].map(lambda e: (e - a).days)).sort_values("cd")
    g = g[g.cd <= 120]
    if len(g) < 5:
        continue
    rho = pd.Series(g["cd"].to_numpy()).corr(pd.Series(g["f_min"].to_numpy()), method="spearman")
    slope = np.polyfit(g["cd"].to_numpy(float), g["f_min"].to_numpy(float), 1)[0]
    reg.append(dict(d=d, rho=rho, slope=slope))
reg = pd.DataFrame(reg).set_index("d")
reg["forward_like"] = (reg.rho > 0.9) & (reg.slope > 0)
reg["ym"] = [f"{d.year}-{d.month:02d}" for d in reg.index]
print(reg.groupby("ym").agg(n=("rho", "size"), forward_like=("forward_like", "sum"),
                            rho_med=("rho", "median"), slope_med=("slope", "median")).round(3).to_string())
fl = reg["forward_like"]
last_bad = fl[~fl].index.max()
first_run = fl[fl.index > last_bad].index.min() if pd.notna(last_bad) else fl.index.min()
print(f"last NOT forward-like session: {last_bad}; forward-like on every session from {first_run} on")
switch = [d for d, prev in zip(fl.index[1:], fl.to_numpy()[:-1]) if fl[d] != prev]
print("state changes (session where forward_like flips):", [(str(d), bool(fl[d])) for d in switch][:40])
bad_train = int((~fl[[TRAIN_START <= d <= TRAIN_END for d in fl.index]]).sum())
print(f"train sessions NOT forward-like: {bad_train} of {sum(TRAIN_START <= d <= TRAIN_END for d in fl.index)}")

# ------------------------------------------------------------------ Q2
hdr("Q2 — |F_matched − ES(15:45 ET bar close)|, matched = live SPX expiry nearest the front ES expiry (ingest rule)")


def stats(x: pd.Series) -> str:
    x = x.dropna().abs()
    if x.empty:
        return "n 0"
    return f"n {x.size:4d}  median {x.median():6.2f}  p90 {x.quantile(.9):6.2f}  max {x.max():7.2f}"


samp = S.loc[sample]
print(samp[["a", "es_src", "es_exp_front", "m_front_exp", "m_front_gap_days", "m_front_f", "es45", "diff_front",
            "m_next_f", "diff_next", "roll_A", "a_is_es_expiry"]].round(2).to_string())
print("\nGATE (sample, n 21):  ", stats(samp["diff_front"]))
print("  sample excl. API-date roll weeks:", stats(samp.loc[~samp.roll_A, "diff_front"]))
print("\nsupplementary, every session with an ES 15:45 bar:")
print("  all                    ", stats(S["diff_front"]))
print("  excl. roll weeks (A)   ", stats(S.loc[~S.roll_A, "diff_front"]))
print("  roll weeks (A) only    ", stats(S.loc[S.roll_A, "diff_front"]))
for y in sorted({d.year for d in S.index}):
    m = np.array([d.year == y for d in S.index])
    print(f"  {y} excl. roll weeks     ", stats(S.loc[m & ~S.roll_A.to_numpy(), "diff_front"]),
          f" | signed median {S.loc[m & ~S.roll_A.to_numpy(), 'diff_front'].median():+.2f}")
pre = np.array([d < date(2024, 4, 3) for d in S.index])
print("  era < 2024-04-03, excl. roll weeks ", stats(S.loc[pre & ~S.roll_A.to_numpy(), "diff_front"]),
      f" | signed median {S.loc[pre & ~S.roll_A.to_numpy(), 'diff_front'].median():+.2f}")
print("  era ≥ 2024-04-03, excl. roll weeks ", stats(S.loc[~pre & ~S.roll_A.to_numpy(), "diff_front"]),
      f" | signed median {S.loc[~pre & ~S.roll_A.to_numpy(), 'diff_front'].median():+.2f}")
print("  era ≥ 2024-04-03, all              ", stats(S.loc[~pre, "diff_front"]))
print("  sample, era ≥ 2024-04-03           ", stats(samp.loc[[d >= date(2024, 4, 3) for d in samp.index], "diff_front"]))
print("  sample, era < 2024-04-03           ", stats(samp.loc[[d < date(2024, 4, 3) for d in samp.index], "diff_front"]))
print("  matched-expiry gap to the ES third Friday (days):",
      S["m_front_gap_days"].value_counts().sort_index().to_dict())
print("  NOTE matched expiry is always a third Friday (when gap 0) → the AM/PM overwrite under the PK applies to it "
      "on every date; F_e is one row's stockPrice, AM and PM same-date forwards differ by hours of carry only.")

big = S[(S.diff_front.abs() > 5) & ~S.roll_A].sort_values("d") if False else S[(S.diff_front.abs() > 5) & ~S.roll_A]
print(f"\nnon-roll-week sessions with |diff| > 5: {len(big)}")
print(big[["a", "es_src", "m_front_exp", "m_front_f", "es45", "diff_front", "diff_next"]].round(2).head(40).to_string())

hdr("Q2b — L4 alignment: median |F_matched − ES close| by bar start (ET), non-roll-week sessions")
for mm in range(40, 51):
    if mm not in es_wide.columns:
        continue
    dd = (S["m_front_f"] - es_wide[mm].reindex(S.index))[~S.roll_A]
    print(f"  15:{mm:02d} ET start (close ≈ 15:{mm+1:02d}): median |d| {dd.abs().median():5.2f}   sd {dd.std():5.2f}   n {dd.notna().sum()}")

missing = S[S.es45.isna()]
print(f"\nsessions with no ES 15:45 ET bar on the API date: {len(missing)}")
print("  ", ", ".join(f"{d}(A {a})" for d, a in zip(missing.index, missing.a)))

# ------------------------------------------------------------------ Q3
hdr("Q3 — ES contract identity at the snapshot")
s26 = S[[d.year == 2026 for d in S.index]].copy()
s26["closer"] = np.where(s26.diff_front.abs() <= s26.diff_next.abs(), "front", "next")
print("2026: which quarterly forward is the ES 15:45 bar closer to (front = ingest rule):",
      s26["closer"].value_counts().to_dict())
win = s26[[any(abs((d - third_friday(2026, m)).days) <= 16 for m in (3, 6, 9)) for d in s26.index]]
print(win[["a", "es_src", "es_exp_front", "m_front_f", "m_next_f", "es45", "diff_front", "diff_next", "closer"]]
      .round(2).to_string())
nxt = s26[s26.closer == "next"]
print("\n2026 sessions where the bar matches the NEXT contract:", ", ".join(str(d) for d in nxt.index) or "none")

print("\n2026 contract switches inside the bar series (|1-min close-to-open jump| > 30 pts, 2026-03-01 → 2026-10-01):")
jumps = q("""
    SELECT datetime, open, prev_close, source FROM (
        SELECT datetime, open, source, LAG(close) OVER (ORDER BY datetime) AS prev_close
        FROM ironbeam_es_1m_bars WHERE datetime >= '2026-03-01' AND datetime < '2026-10-01') x
    WHERE ABS(open - prev_close) > 30 ORDER BY datetime
""")
for _, j in jumps.iterrows():
    t_et = pd.Timestamp(j.datetime).tz_localize(UTC).tz_convert(ET)
    print(f"  {j.datetime} UTC ({t_et:%Y-%m-%d %H:%M %a} ET)  {j.prev_close:.2f} → {j.open:.2f} "
          f"({j.open - j.prev_close:+.2f}) source {j.source}")
print("  → a session D whose bars span a switch, or whose API-date snapshot is on the other side of one, has levels "
      "and bars in different contracts")

tr = S[S.train]
print(f"\ntrain sessions: {len(tr)}")
for y in (2023, 2024, 2025, 2026):
    m = np.array([d.year == y for d in tr.index])
    print(f"  {y}: session in roll week {int(tr.roll_D[m].sum()):3d} | API (snapshot) date in roll week "
          f"{int(tr.roll_A[m].sum()):3d} | API date = ES expiry Friday {int(tr.a_is_es_expiry[m].sum())}")
print(f"  2023–2025 total: session-in-roll-week "
      f"{int(tr.roll_D[np.array([d.year <= 2025 for d in tr.index])].sum())}, "
      f"snapshot-in-roll-week {int(tr.roll_A[np.array([d.year <= 2025 for d in tr.index])].sum())}")

# ------------------------------------------------------------------ Q4
hdr("Q4 — worked day 2026-10-23")
D = WORKED_DAY
A = api[D]
p_es = float(S.loc[D, "es45"])
print(f"session {D}, API date {A}, ES 15:45 ET bar close {p_es:.2f} (source {S.loc[D, 'es_src']}), "
      f"S0 {S.loc[D, 's0']:.2f}, ES−S0 {p_es - S.loc[D, 's0']:+.2f}")
r = q("""
    SELECT expir_date, dte, strike::float8 AS strike, stock_price::float8 AS f, discounted_level,
           short_rate, div_yield, gex_call, gex_put, call_oi, put_oi
    FROM orats_oi_gamma
    WHERE ticker = 'SPX' AND trade_date = %s AND expir_date >= trade_date
""", (D,))
r["expir_date"] = pd.to_datetime(r["expir_date"]).dt.date
r["cd"] = r["expir_date"].map(lambda e: (e - A).days)
r["L"] = r["strike"] * p_es / r["f"]
r["abs_gex"] = r["gex_call"].abs() + r["gex_put"].abs()
r["cur_shift"] = r["discounted_level"] - r["strike"]
r["cor_shift"] = r["L"] - r["strike"]
r["L_minus_dl"] = r["L"] - r["discounted_level"]
top = r.sort_values("abs_gex", ascending=False).head(10)
print(top[["strike", "expir_date", "dte", "cd", "f", "discounted_level", "L", "cur_shift", "cor_shift",
           "L_minus_dl", "abs_gex"]].round(2).to_string(index=False))

print("\nchat-estimate table vs measured (shift = level − strike, median over all live rows of the expiry):")
per_exp = r.groupby("expir_date").agg(cd=("cd", "first"), dte=("dte", "first"), f=("f", "first"),
                                      cur=("cur_shift", "median"), cor=("cor_shift", "median"),
                                      gex=("abs_gex", "sum"))
for target, est_cur, est_cor in ((0, 1, 25), (7, 10, 22), (30, 38, 11), (55, 68, 0), (90, 112, -16)):
    i = (per_exp["dte"] - target).abs().idxmin()
    row = per_exp.loc[i]
    print(f"  ~{target:>2} DTE → {i} (dte {int(row.dte)}, cd {int(row.cd)}): current {row.cur:+7.2f} (chat {est_cur:+}), "
          f"corrected {row.cor:+7.2f} (chat {est_cor:+}), current error {row.cur - row.cor:+7.2f} "
          f"(chat {est_cur - est_cor:+})")
print(per_exp.round(2).to_string())

bars = q("SELECT high, low FROM ironbeam_es_1m_bars WHERE datetime >= %s AND datetime < %s",
         (datetime.combine(D - timedelta(days=1), time(15, 0), tzinfo=PT).astimezone(UTC).replace(tzinfo=None),
          datetime.combine(D, time(13, 0), tzinfo=PT).astimezone(UTC).replace(tzinfo=None)))
band_min, band_max = float(bars.low.min()) - GEX_LEVEL_PADDING, float(bars.high.max()) + GEX_LEVEL_PADDING
print(f"\nband [{band_min:.2f}, {band_max:.2f}] from {len(bars)} bars")


def grouped(key: str) -> pd.DataFrame:
    rr = r[r[key].notna() & r["discounted_level"].notna()].copy()
    rr["level"] = (np.round(rr[key] / GEX_LEVEL_BUCKET) * GEX_LEVEL_BUCKET).astype(int).astype(float)
    g = rr.groupby("level").agg(call_gamma=("gex_call", "sum"), put_raw=("gex_put", "sum")).reset_index()
    g["put_gamma"] = -g["put_raw"].abs()
    g["net_gamma"] = g["call_gamma"] + g["put_gamma"]
    return g


def select_levels(df: pd.DataFrame) -> pd.DataFrame:  # replica of callbacks._select_levels
    df = df[(df.level >= band_min) & (df.level <= band_max)].copy()
    df["mag"] = df.call_gamma.abs() + df.put_gamma.abs()
    df = df[df.mag >= GEX_ABS_THRESHOLD].sort_values("mag", ascending=False)
    keep, kept = [], []
    for _, x in df.iterrows():
        if all(abs(x.level - k) >= GEX_MIN_LEVEL_SPACING for k in kept):
            keep.append(x)
            kept.append(x.level)
        if len(keep) >= GEX_MAX_LEVELS_PER_DAY:
            break
    return pd.DataFrame(keep)


cur = select_levels(grouped("discounted_level"))
cor = select_levels(grouped("L"))
print(f"selected lines: Current {len(cur)}, Corrected {len(cor)}")
show = lambda x: x.head(12)[["level", "net_gamma", "mag"]].assign(
    net_bn=lambda z: z.net_gamma / 1e9, mag_bn=lambda z: z.mag / 1e9)[["level", "net_bn", "mag_bn"]].round(1)
print("\nCurrent (top 12 by magnitude):\n" + show(cur).to_string(index=False))
print("\nCorrected (top 12 by magnitude):\n" + show(cor).to_string(index=False))
n_dl_buckets = r.groupby("strike")["discounted_level"].apply(lambda s: s.round().nunique())
n_L_buckets = r.groupby("strike")["L"].apply(lambda s: s.round().nunique())
print(f"\nstrikes whose rows land in >1 bucket: Current {int((n_dl_buckets > 1).sum())}/{len(n_dl_buckets)}, "
      f"Corrected {int((n_L_buckets > 1).sum())}/{len(n_L_buckets)}")
print("\nDONE")
