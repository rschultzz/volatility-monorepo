#!/usr/bin/env python3
"""CR-BI Step 0 Q4 — quantify the cr_g / cr_i fixed-UTC session-window error. Read-only.

cr_g / cr_i keep ES bars with 13:30 <= t < 20:00 UTC, grouped by UTC date. In PST that is
05:30–11:59 PT. For every canonical row this script
  1. replicates cr_g's T+N session pick and close (must equal the stored value — proves the mechanism),
  2. takes the true close on that same date with a PT-aware window (12:59 PT bar close = cr_g's own
     PDT convention; 13:00 PT bar close = the canonical t0 convention),
  3. reports the distribution of (true − stored) for rows whose T+N session falls in PST,
  4. counts position_tN_post_touch labels (cr_i) that change with the true close.

Usage: apps/web/.venv/bin/python scripts/cr_bi_step0_window_error.py
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from dotenv import load_dotenv  # noqa: E402
load_dotenv(REPO_ROOT / ".env")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from packages.shared.backfill_safety import get_backfill_db_conn  # noqa: E402
from packages.shared.canonical_version import CANONICAL_FEATURE_VERSION  # noqa: E402
from packages.shared.outcomes import pick_drift_target  # noqa: E402
from packages.shared.trading_calendar import is_trading_day  # noqa: E402

PT = ZoneInfo("America/Los_Angeles")
TICKER = "SPX"


def is_pst(d: dt.date) -> bool:
    return dt.datetime(d.year, d.month, d.day, 12, tzinfo=PT).utcoffset() == dt.timedelta(hours=-8)


def dist(s: pd.Series, label: str) -> None:
    s = s.dropna()
    if s.empty:
        print(f"  {label}: n=0"); return
    a = s.abs()
    print(f"  {label}: n={len(s)} mean {s.mean():+.2f} median {s.median():+.2f} | |x| median {a.median():.2f} "
          f"p75 {a.quantile(.75):.2f} p90 {a.quantile(.9):.2f} p99 {a.quantile(.99):.2f} max {a.max():.2f} "
          f"| >5: {(a > 5).sum()} >10: {(a > 10).sum()} >25: {(a > 25).sum()}")


def main() -> None:
    conn = get_backfill_db_conn()
    conn.execute("SET default_transaction_read_only = on")
    bars = pd.DataFrame(conn.execute(
        "SELECT datetime, open, high, low, close FROM ironbeam_es_1m_bars WHERE datetime >= '2023-04-25' ORDER BY 1"
    ).fetchall(), columns=["datetime", "open", "high", "low", "close"])
    bars["datetime"] = pd.to_datetime(bars["datetime"], utc=True)
    hm = bars["datetime"].dt.hour * 60 + bars["datetime"].dt.minute
    g = bars[(hm >= 810) & (hm < 1200)].copy()                     # cr_g / cr_i window
    g["sd"] = g["datetime"].dt.date
    g_close = g.groupby("sd")["close"].last()
    g_sessions = sorted(g_close.index)
    g_idx = {d: i for i, d in enumerate(g_sessions)}

    pt = bars["datetime"].dt.tz_convert(PT)
    bars["pd"], bars["pt"] = pt.dt.date, pt.dt.time
    c1259 = bars[bars["pt"] == dt.time(12, 59)].set_index("pd")["close"]
    c1300 = bars[bars["pt"] == dt.time(13, 0)].set_index("pd")["close"]

    cols = ("trade_date, outcome_status, reached_touch, days_to_reach, session_close_t1, session_close_t5, "
            "session_close_t15, position_t1_post_touch, position_t5_post_touch, position_t15_post_touch")
    o = pd.DataFrame(conn.execute(
        f"SELECT {cols} FROM bt_daily_outcomes WHERE ticker=%s AND feature_version=%s AND active ORDER BY 1",
        (TICKER, CANONICAL_FEATURE_VERSION)).fetchall(), columns=[c.strip() for c in cols.split(",")])

    print(f"canonical rows {len(o)}; cr_g-style sessions {len(g_sessions)} "
          f"(non-NYSE-trading-day sessions among them: {sum(1 for d in g_sessions if not is_trading_day(d))})")
    for n in (1, 5, 15):
        col = f"session_close_t{n}"
        r = o[o[col].notna()].copy()
        r["tdate"] = [g_sessions[g_idx[d] + n] if d in g_idx and g_idx[d] + n < len(g_sessions) else None
                      for d in r["trade_date"]]
        r = r[r["tdate"].notna()]
        r["replica"] = [g_close[d] for d in r["tdate"]]
        r["pst"] = [is_pst(d) for d in r["tdate"]]
        r["true_1259"] = [c1259.get(d, np.nan) for d in r["tdate"]]
        r["true_1300"] = [c1300.get(d, np.nan) for d in r["tdate"]]
        match = (r[col].astype(float) - r["replica"]).abs() < 0.01
        print(f"\n== T+{n}: stored non-NULL {len(r)}; replica == stored on {int(match.sum())} "
              f"({len(r) - int(match.sum())} differ); PST-session rows {int(r.pst.sum())}, PDT {int((~r.pst).sum())}")
        if (~match).any():
            print(r.loc[~match, ["trade_date", "tdate", col, "replica"]].head(8).to_string(index=False))
        for lab, sub in (("PST", r[r.pst]), ("PDT", r[~r.pst])):
            dist(sub["true_1259"] - sub[col].astype(float), f"{lab} true 12:59-bar close − stored")
            dist(sub["true_1300"] - sub[col].astype(float), f"{lab} true 13:00-bar close − stored")
        hol = r[[not is_trading_day(d) for d in r["tdate"]]]
        print(f"  rows whose T+{n} 'session' is not an NYSE trading day: {len(hol)}")

    # ── cr_i: post-touch position labels with the true close ────────────────
    ls = {d: pick_drift_target(w if isinstance(w, list) else []) for d, w in conn.execute(
        "SELECT trade_date, walls FROM orats_gex_landscape WHERE ticker=%s", (TICKER,)).fetchall()}
    im = {d: v for d, v in conn.execute(
        "SELECT trade_date, (feature_vector->>'implied_move_1d')::float FROM bt_daily_features "
        "WHERE ticker=%s AND feature_version=%s AND active", (TICKER, CANONICAL_FEATURE_VERSION)).fetchall()}
    print("\n== cr_i position_tN_post_touch (stored vs true 13:00-bar close on the same session)")
    for n in (1, 5, 15):
        col = f"position_t{n}_post_touch"
        r = o[o[col].notna() & o["days_to_reach"].notna()]
        tot = chg = pst_tot = pst_chg = rep_bad = 0
        for t in r.itertuples():
            d = t.trade_date
            if d not in g_idx or ls.get(d) is None or not im.get(d):
                continue
            i = g_idx[d] + int(t.days_to_reach) + n
            if i >= len(g_sessions):
                continue
            sd, tgt, tol = g_sessions[i], float(ls[d]), 0.25 * im[d]
            lab = lambda c: -1 if c < tgt - tol else (1 if c > tgt + tol else 0)  # noqa: E731
            if lab(g_close[sd]) != getattr(t, col):
                rep_bad += 1
            tc = c1300.get(sd, np.nan)
            if np.isnan(tc):
                continue
            tot += 1; pst = is_pst(sd); pst_tot += pst
            if lab(tc) != getattr(t, col):
                chg += 1; pst_chg += pst
        print(f"  T+{n}: rows {tot}; label changes {chg} ({pst_chg} of {pst_tot} PST-session rows, "
              f"{chg - pst_chg} of {tot - pst_tot} PDT); replica ≠ stored on {rep_bad}")


if __name__ == "__main__":
    main()
