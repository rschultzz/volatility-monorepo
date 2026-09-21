#!/usr/bin/env python3
"""CR-BH Step 0 — read-only diagnostics for gate questions 3, 5 and 6.

  Q3  ES session window vs SPX-cash window (open/close print comparison)
  Q5  minute-sampling understatement of high/low, ES as proxy, 50 random outcome dates
  Q6  bad-print filter: every dropped minute; partial days classified

Read-only (default_transaction_read_only = on). Usage:
    apps/web/.venv/bin/python scripts/cr_bh_step0_diagnosis.py
"""
from __future__ import annotations

import datetime as dt
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(REPO_ROOT / ".env")

import pandas as pd  # noqa: E402

from packages.shared.backfill_safety import get_backfill_db_conn  # noqa: E402
from packages.shared.canonical_version import CANONICAL_FEATURE_VERSION  # noqa: E402
from scripts.cr_bh_spx_cash import daily_ohlc, fetch_spx_minutes, filter_bad_prints  # noqa: E402

TICKER = "SPX"
SEED = 20260920
PARTIAL_MIN_MINUTES = 380
CACHE = REPO_ROOT / "scripts" / ".cache" / "cr_bh_spx_minutes.pkl"

# NYSE 13:00 ET (10:00 PT) early closes inside the corpus
EARLY_CLOSES = {dt.date(*d) for d in [
    (2023, 7, 3), (2023, 11, 24), (2024, 7, 3), (2024, 11, 29), (2024, 12, 24),
    (2025, 7, 3), (2025, 11, 28), (2025, 12, 24),
]}

_ES_MINUTES_SQL = """
    SELECT (datetime AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles') AS pt, open, high, low, close
    FROM ironbeam_es_1m_bars
    WHERE (datetime AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')::date = ANY(%s)
      AND (datetime AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')::time BETWEEN '06:30:00' AND '13:00:00'
    ORDER BY 1
"""


def main() -> None:
    conn = get_backfill_db_conn()
    conn.execute("SET default_transaction_read_only = on")
    conn.commit()

    rows = conn.execute(
        "SELECT trade_date, active FROM bt_daily_outcomes WHERE ticker=%s AND feature_version=%s ORDER BY 1",
        (TICKER, CANONICAL_FEATURE_VERSION),
    ).fetchall()
    outcome_dates = [r[0] for r in rows]
    print(f"outcome rows {len(rows)} ({sum(1 for r in rows if r[1])} active) {outcome_dates[0]} → {outcome_dates[-1]}")

    if CACHE.exists():
        minutes = pd.read_pickle(CACHE)
    else:
        minutes = fetch_spx_minutes(conn, outcome_dates[0], dt.date.today())
        CACHE.parent.mkdir(exist_ok=True)
        minutes.to_pickle(CACHE)
    print(f"SPX minutes: {len(minutes)} rows, {minutes['session_date'].nunique()} days; "
          f"rows/minute max {minutes['n_rows'].max()}")

    # ── Q6 bad prints ────────────────────────────────────────────────────────
    clean, dropped = filter_bad_prints(minutes)
    print(f"\n== Q6 bad prints: {len(dropped)} dropped minutes on {dropped['session_date'].nunique()} days")
    for _, r in dropped.iterrows():
        print(f"  {r['minute']}  spx={r['spx']}  ref={r['ref']:.2f}  {r['reason']}")
    # sensitivity
    import scripts.cr_bh_spx_cash as m
    for thr in (0.0015, 0.003, 0.005, 0.01):
        m.NEIGHBOR_MAX_DEV = thr
        _, d = filter_bad_prints(minutes)
        print(f"  sensitivity NEIGHBOR_MAX_DEV={thr}: {len(d)} dropped on {d['session_date'].nunique()} days")
    m.NEIGHBOR_MAX_DEV = 0.005

    # ── Q6 partial days ──────────────────────────────────────────────────────
    daily = daily_ohlc(clean)
    g = clean.groupby("session_date")["minute"]
    cov = pd.DataFrame({"n": g.size(), "first": g.min().dt.time, "last": g.max().dt.time})
    cov["max_gap_min"] = clean.groupby("session_date")["minute"].apply(
        lambda s: int(s.diff().dt.total_seconds().max() / 60) if len(s) > 1 else 0)
    missing = sorted(set(outcome_dates) - set(cov.index))
    print(f"\n== Q6 coverage: outcome dates with no SPX minutes: {missing}")
    part = cov[cov["n"] < PARTIAL_MIN_MINUTES]
    print(f"partial days (< {PARTIAL_MIN_MINUTES} minutes): {len(part)} "
          f"({sum(1 for d in part.index if d in set(outcome_dates))} are outcome dates)")
    for d, r in part.iterrows():
        if d in EARLY_CLOSES:
            kind = "early close"
        elif r["first"] > dt.time(6, 35):
            kind = "outage (late start)"
        elif r["last"] < dt.time(12, 55):
            kind = "outage (early end)"
        else:
            kind = "outage (intraday gap)"
        print(f"  {d}  n={r['n']:3d}  {r['first']}–{r['last']}  max_gap={r['max_gap_min']:3d}m  "
              f"{'outcome-date' if d in set(outcome_dates) else 'not-outcome'}  {kind}")

    # ── Q3 open / close print comparison ─────────────────────────────────────
    cm = clean.set_index("minute")["spx"]
    o30, o33, c1259, c1300 = [], [], [], []
    for d in daily.index:
        def at(h, mi):
            return cm.get(pd.Timestamp(dt.datetime.combine(d, dt.time(h, mi))))
        a, b = at(6, 30), at(6, 33)
        if a is not None and b is not None:
            o33.append(abs(b - a))
        a, b = at(12, 59), at(13, 0)
        if a is not None and b is not None:
            c1300.append(abs(b - a))
    print(f"\n== Q3 SPX |06:33 − 06:30| median {pd.Series(o33).median():.2f} p90 {pd.Series(o33).quantile(.9):.2f} "
          f"max {max(o33):.2f} (n={len(o33)})")
    print(f"   SPX |13:00 − 12:59| median {pd.Series(c1300).median():.2f} p90 {pd.Series(c1300).quantile(.9):.2f} "
          f"max {max(c1300):.2f} (n={len(c1300)})")
    first_is_0630 = (cov["first"] == dt.time(6, 30)).mean()
    last_is_1300 = (cov["last"] == dt.time(13, 0)).mean()
    print(f"   days whose first print is 06:30: {first_is_0630:.1%}; last print 13:00: {last_is_1300:.1%}")

    # ── Q5 minute-sampling understatement, ES proxy ──────────────────────────
    rng = random.Random(SEED)
    sample = sorted(rng.sample([d for d in outcome_dates], 50))
    es = pd.DataFrame(conn.execute(_ES_MINUTES_SQL, (sample,)).fetchall(),
                      columns=["pt", "open", "high", "low", "close"])
    es["d"] = pd.to_datetime(es["pt"]).dt.date
    q = es.groupby("d").agg(true_hi=("high", "max"), true_lo=("low", "min"),
                            samp_hi=("close", "max"), samp_lo=("close", "min"), n=("close", "size"))
    q["hi_under"] = q["true_hi"] - q["samp_hi"]
    q["lo_under"] = q["samp_lo"] - q["true_lo"]
    q["range_under"] = q["hi_under"] + q["lo_under"]
    print(f"\n== Q5 ES proxy, {len(q)} days (seed {SEED}); 1m closes only vs true 1m high/low, pts")
    for c in ("hi_under", "lo_under", "range_under"):
        s = q[c]
        print(f"   {c:12s} median {s.median():.2f}  mean {s.mean():.2f}  p90 {s.quantile(.9):.2f}  max {s.max():.2f}")
    print("   sample dates:", ", ".join(str(d) for d in sample))


if __name__ == "__main__":
    main()
