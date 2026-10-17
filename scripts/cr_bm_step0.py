#!/usr/bin/env python3
"""CR-BM Step 0 — diagnosis for T-S3c (condor legs by wing). READ-ONLY: SELECTs only, no run row.

Answers (specs/CR-BM-condor-legs-by-wing.md, Step 0):
  Q1 coverage: of the 732 CR-AS universe dates, how many have both wings on both structures
  Q2 CR-AS fill rule vs the harness settlement: CR-AS close (spot_price, last snapshot ≤ 13:00 PT)
     vs the harness's SPX-cash settlement (spx_cash.settlement_prints, CR-BI); AM-settled legs
  Q3 roll-week, holiday and early-close dates in the universe
  Q4 CR-BK parquet join count; bt_vol_state_daily join count
  Q5 DAILY-CAPTURE (CR-AU) runs: any train-date condor captures outside the CR-AS set

The universe is the CR-AS full-sample read as run (scripts/logs/cr_as_analysis_full_fee0.65.json,
run 85b690ec): its rows carry the strikes, IM, regime and CR-AS settlement used.

Usage: apps/web/.venv/bin/python -u scripts/cr_bm_step0.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for line in (REPO_ROOT / ".env").read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())
os.environ["DATABASE_URL"] = os.environ["BACKFILL_DATABASE_URL"]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

from packages.shared.backfill_safety import assert_role_or_die, get_backfill_db_conn
from packages.shared.backtest.condor_0dte import CondorBox
from packages.shared.options_cache.opra import format_opra
from packages.shared.spx_cash import build_series, fetch_minutes, settlement_prints
from packages.shared.trading_calendar import NYSE_HOLIDAYS, is_trading_day
from scripts.cr_as_capture_0dte_condor_legs import OPRA_ROOT, UNIVERSE_END, WINDOW_END, WINDOW_START

CR_AS_JSON = REPO_ROOT / "scripts" / "logs" / "cr_as_analysis_full_fee0.65.json"
BK_PROFILES = REPO_ROOT / "scripts" / "logs" / "cr_bk" / "profiles.parquet"
OUT = REPO_ROOT / "scripts" / "output" / "cr_bm" / "step0.md"
EARLY_CLOSES = {date(2023, 7, 3), date(2023, 11, 24), date(2024, 7, 3), date(2024, 11, 29), date(2024, 12, 24),
                date(2025, 7, 3), date(2025, 11, 28), date(2025, 12, 24)}


def third_friday(y: int, m: int) -> date:
    d = date(y, m, 15)
    return d + timedelta(days=(4 - d.weekday()) % 7)


def roll_week(d: date) -> bool:
    """ES quarterly roll week: Mon–Fri of the week before the third Friday (roll Monday ≈ 4 sessions prior)
    through expiry Friday, Mar/Jun/Sep/Dec."""
    if d.month not in (3, 6, 9, 12):
        return False
    tf = third_friday(d.year, d.month)
    return tf - timedelta(days=11) <= d <= tf


def main() -> None:
    L = []
    say = lambda s="": (L.append(s), print(s))
    cr = json.load(open(CR_AS_JSON))
    rows = cr["results"]
    say(f"# CR-BM Step 0 — {datetime.now():%Y-%m-%d %H:%M} PT\n")
    say(f"Source: `{CR_AS_JSON.relative_to(REPO_ROOT)}` (CR-AS run 85b690ec, $0.65). Rows {len(rows)}; date status {cr['status']}; box status {cr['box_status']}.\n")

    # Q1 coverage
    by_date = defaultdict(dict)
    for r in rows:
        by_date[r["date"]][r["box"]] = r
    dates = sorted(by_date)
    both = [d for d in dates if {"pm0.5", "pm1"} <= set(by_date[d])]
    say("## Q1 coverage\n")
    say(f"- CR-AS universe 732 dates; priced dates {cr['status'].get('priced_date')}; dates with ≥ 1 priced box {len(dates)}.")
    say(f"- ±0.5 boxes {sum('pm0.5' in v for v in by_date.values())}; ±1.0 boxes {sum('pm1' in v for v in by_date.values())}; **both structures on the same date: {len(both)}**.")
    say(f"- Dates with only one structure: {[d for d in dates if d not in set(both)]}")
    assert max(dates) <= UNIVERSE_END.isoformat()

    conn = get_backfill_db_conn()
    assert_role_or_die(conn)

    # Q1b: every priced box has all four legs with ≥ 1 bar in the window (both wings present)
    say("\n### Q1b both wings captured per priced box\n")
    missing_leg = Counter()
    am_dates = set()
    am_detail = {}
    for d in dates:
        td = date.fromisoformat(d)
        boxes = [CondorBox(td, 0.5 if b == "pm0.5" else 1.0, *_strikes(r), 0.0, r["im"]) for b, r in by_date[d].items()]
        opras = sorted({format_opra(OPRA_ROOT, td, t, k) for b in boxes for k, t in b.contracts()})
        got = conn.execute(
            "SELECT strike, option_type, bool_or(expiry_tod = 'am'), count(*) FILTER (WHERE expiry_tod = 'am'), count(*) FILTER (WHERE expiry_tod IS DISTINCT FROM 'am') FROM orats_options_minute "
            "WHERE opra_symbol = ANY(%s) AND snapshot_pt BETWEEN %s AND %s GROUP BY 1, 2",
            (opras, datetime.combine(td, WINDOW_START), datetime.combine(td, WINDOW_END))).fetchall()
        have = {(float(s), t) for s, t, *_ in got}
        if any(r[2] for r in got):
            am_dates.add(d)
            am_detail[d] = (sum(r[3] for r in got), sum(r[4] for r in got))
        for b in boxes:
            for c in b.contracts():
                if c not in have:
                    missing_leg[f"{b.label}:{c[1]}"] += 1
    say(f"- Priced boxes missing a leg in the 06:30–06:45 window: {dict(missing_leg) or 'none'} (CR-AS priced a box only if all four legs had bars).")
    say(f"- Dates whose captured legs carry `expiry_tod = 'am'` (AM-settled SPX monthly): **{len(am_dates)}** {sorted(am_dates)}")
    say(f"- Bars on those dates, (am rows, non-am rows) in the window: {am_detail}")

    # Q2 settlement
    say("\n## Q2 CR-AS close vs harness settlement (spx_cash.settlement_prints)\n")
    clean, _ = build_series(fetch_minutes(conn, date.fromisoformat(dates[0]), date.fromisoformat(dates[-1])))
    settle = {d.isoformat() if hasattr(d, "isoformat") else str(d): float(v) for d, v in settlement_prints(clean).items()}
    diffs, absent = [], []
    for d in dates:
        c_as = next(iter(by_date[d].values()))["spx_close"]
        h = settle.get(d)
        if h is None:
            absent.append(d)
        else:
            diffs.append((d, h - c_as))
    dv = np.array([x for _, x in diffs])
    say(f"- Dates with a harness settlement: {len(diffs)} / {len(dates)}; absent: {len(absent)} {absent}")
    say(f"- harness − CR-AS close (pts): exact (|Δ| < 0.005) {int((np.abs(dv) < 0.005).sum())}; |Δ| ≥ 0.01 {int((np.abs(dv) >= 0.01).sum())}; "
        f"≥ 1 pt {int((np.abs(dv) >= 1).sum())}; ≥ 5 pts {int((np.abs(dv) >= 5).sum())}; median |Δ| {np.median(np.abs(dv)):.2f}; "
        f"p95 |Δ| {np.percentile(np.abs(dv), 95):.2f}; max |Δ| {np.abs(dv).max():.2f}")
    seg = Counter()
    for d, x in diffs:
        if abs(x) >= 0.01:
            td = date.fromisoformat(d)
            seg["2023 lagged" if (date(2023, 5, 1) <= td <= date(2023, 9, 7) or date(2023, 10, 24) <= td <= date(2023, 11, 8))
                else ("pre-cutover other" if td < date(2023, 11, 9) else str(td.year))] += 1
    say(f"- Disagreeing dates (|Δ| ≥ 0.01) by segment: {dict(seg)}")
    # does the disagreement change any wing's intrinsic payoff?
    flips = Counter()
    for d, x in diffs:
        h = settle[d]
        for b, r in by_date[d].items():
            lp, sp, sc, lc = r["strikes"]
            pl_as = max(0, sp - r["spx_close"]) - max(0, lp - r["spx_close"])
            cl_as = max(0, r["spx_close"] - sc) - max(0, r["spx_close"] - lc)
            pl_h = max(0, sp - h) - max(0, lp - h)
            cl_h = max(0, h - sc) - max(0, h - lc)
            if abs(pl_h - pl_as) >= 0.005:
                flips[f"{b}:put"] += 1
            if abs(cl_h - cl_as) >= 0.005:
                flips[f"{b}:call"] += 1
    say(f"- Boxes whose wing intrinsic changes under the harness settlement: {dict(flips)}")
    worst = sorted(diffs, key=lambda t: -abs(t[1]))[:10]
    say(f"- Ten largest |Δ|: {[(d, round(x, 2)) for d, x in worst]}")

    # Q3 roll weeks / holidays / early closes
    say("\n## Q3 roll weeks, holidays, early closes\n")
    tds = [date.fromisoformat(d) for d in dates]
    say(f"- Non-trading days (NYSE calendar) in the universe: {[d.isoformat() for d in tds if not is_trading_day(d)]}")
    say(f"- NYSE holidays in the universe: {[d.isoformat() for d in tds if d in NYSE_HOLIDAYS]}")
    say(f"- Early-close days in the universe: {[d.isoformat() for d in tds if d in EARLY_CLOSES]}")
    rw = [d for d in tds if roll_week(d)]
    say(f"- ES roll-week dates (week before third Friday through expiry, quarterly): {len(rw)}")
    tf = [d for d in tds if d.weekday() == 4 and 15 <= d.day <= 21]
    say(f"- Third Fridays in the universe: {len(tf)}")

    # Q4 joins
    say("\n## Q4 joins\n")
    prof = pd.read_parquet(BK_PROFILES)
    prof = prof[prof.wing == "linear"]
    bk_dates = {str(d)[:10] for d in prof.d.unique()}
    say(f"- CR-BK `profiles.parquet` (wing = linear) dates: {len(bk_dates)}; universe dates joined: **{sum(d in bk_dates for d in dates)} / {len(dates)}**; "
        f"missing: {[d for d in dates if d not in bk_dates][:40]}{' …' if sum(d not in bk_dates for d in dates) > 40 else ''}")
    vs = {r[0].isoformat(): r[1] for r in conn.execute(
        "SELECT trade_date, iv_7d_pct FROM bt_vol_state_daily WHERE ticker = 'SPX' AND version = 'cr-bl-v1' AND trade_date <= %s",
        (UNIVERSE_END,)).fetchall()}
    say(f"- `bt_vol_state_daily` cr-bl-v1 rows ≤ split: {len(vs)}; universe dates with a row {sum(d in vs for d in dates)}; "
        f"with non-NULL `iv_7d_pct` **{sum(vs.get(d) is not None for d in dates)} / {len(dates)}**; first non-NULL {min((d for d in dates if vs.get(d) is not None), default=None)}")
    reg = Counter(next(iter(v.values()))["regime"] for v in by_date.values())
    say(f"- CR-AS regime labels (from the CR-AS rows): {dict(reg)}")

    # Q5 DAILY-CAPTURE
    say("\n## Q5 DAILY-CAPTURE (CR-AU) runs\n")
    runs = conn.execute("SELECT cr_id, count(*), min(started_at), max(started_at) FROM bt_backfill_runs "
                        "WHERE cr_id LIKE 'DAILY-CAPTURE%%' OR cr_id LIKE 'CR-AS%%' GROUP BY 1 ORDER BY 1").fetchall()
    for r in runs:
        say(f"- `{r[0]}`: {r[1]} runs, {r[2]:%Y-%m-%d} → {r[3]:%Y-%m-%d}")
    extra = conn.execute(
        "SELECT count(DISTINCT snapshot_pt::date) FROM orats_options_minute "
        "WHERE snapshot_pt::date <= %s AND expir_date::date = snapshot_pt::date AND snapshot_pt::time BETWEEN '06:30' AND '06:45' "
        "AND NOT (snapshot_pt::date = ANY(%s))", (UNIVERSE_END, tds)).fetchone()
    say(f"- Train dates with any 0DTE bar in 06:30–06:45 PT outside the CR-AS priced set: {extra[0]}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n")
    print(f"\nwritten {OUT}")


def _strikes(r):
    lp, sp, sc, lc = r["strikes"]
    return sp, lp, sc, lc


if __name__ == "__main__":
    main()
