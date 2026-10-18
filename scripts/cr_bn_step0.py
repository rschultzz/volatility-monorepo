#!/usr/bin/env python3
"""CR-BN Step 0 — AM/PM mixing in 0DTE SPX bars of orats_options_minute. READ-ONLY.

Counts only. No quote is priced and no P&L is read (holdout dates included in
the counts are not analysed).

  Q1  third Fridays 2023-05-01 → today: 0DTE rows by expiry_tod, all-day and in
      the 06:30–06:45 PT capture window; mixed = both 'am' and non-'am' rows.
  Q2  expiry_tod inventory over all 0DTE SPX rows since 2023-05-01.
  Q3  non-third-Friday 0DTE dates carrying 'am' rows.
  Q4  per-minute coverage on mixed dates: (opra, minute) slots held by an 'am'
      row — the PK is (opra_symbol, snapshot_pt), so the PM quote for that
      minute was not stored.

Usage:
    apps/web/.venv/bin/python scripts/cr_bn_step0.py [--out scripts/output/cr_bn/step0.md]
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from datetime import date, datetime, time, timedelta
from pathlib import Path


def _find_dotenv() -> Path | None:
    current = Path(__file__).resolve().parent
    for _ in range(8):
        c = current / ".env"
        if c.exists():
            return c
        current = current.parent
    return None


_env_path = _find_dotenv()
if _env_path:
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

repo_root = str(Path(__file__).parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from packages.shared.backfill_safety import get_backfill_db_conn
from packages.shared.backtest.condor_0dte import is_third_friday

TICKER = "SPX"
START = date(2023, 5, 1)
SPLIT = date(2026, 6, 5)          # train ≤ SPLIT; holdout after
WINDOW_START, WINDOW_END = time(6, 30), time(6, 45)

BY_DATE_TOD_SQL = """
SELECT trade_date_d, coalesce(expiry_tod, '<null>') AS tod, count(*) AS n,
       count(*) FILTER (WHERE snapshot_pt::time BETWEEN %s AND %s) AS n_win,
       count(DISTINCT opra_symbol) AS n_opra
FROM orats_options_minute
WHERE ticker = %s AND trade_date_d BETWEEN %s AND %s AND expir_date_d = trade_date_d
GROUP BY 1, 2 ORDER BY 1, 2
"""
SLOTS_SQL = """
SELECT opra_symbol, count(*) FILTER (WHERE expiry_tod = 'am') AS n_am,
       count(*) FILTER (WHERE expiry_tod IS DISTINCT FROM 'am') AS n_pm
FROM orats_options_minute
WHERE ticker = %s AND trade_date_d = %s AND expir_date_d = trade_date_d
  AND snapshot_pt::time BETWEEN %s AND %s
GROUP BY 1 ORDER BY 1
"""
DAILY_RUNS_SQL = """
SELECT cr_id, count(*), min(started_at), max(started_at) FROM bt_backfill_runs
WHERE cr_id IN ('DAILY-CAPTURE', 'DAILY-CAPTURE-MATURE') GROUP BY 1 ORDER BY 1
"""


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="scripts/output/cr_bn/step0.md")
    ap.add_argument("--end", default=date.today().isoformat())
    args = ap.parse_args(argv)
    end = date.fromisoformat(args.end)

    conn = get_backfill_db_conn()
    conn.execute("SET TRANSACTION READ ONLY")
    out: list[str] = []
    say = out.append
    say(f"# CR-BN Step 0 — {datetime.now():%Y-%m-%d %H:%M} PT\n")
    say(f"Read-only. `orats_options_minute`, ticker {TICKER}, 0DTE = `expir_date_d = trade_date_d`, "
        f"{START} → {end}. Train ≤ {SPLIT}, holdout after. Counts only.\n")

    rows = conn.execute(BY_DATE_TOD_SQL, (WINDOW_START, WINDOW_END, TICKER, START, end)).fetchall()
    by_date: dict[date, dict[str, tuple[int, int, int]]] = defaultdict(dict)
    for td, tod, n, n_win, n_opra in rows:
        by_date[td][tod] = (int(n), int(n_win), int(n_opra))

    # Q2
    inv = defaultdict(lambda: [0, 0])
    for td, d in by_date.items():
        for tod, (n, _, _) in d.items():
            inv[tod][0] += n
            inv[tod][1] += 1
    say("## Q2 expiry_tod inventory (0DTE SPX rows)\n")
    say("| expiry_tod | rows | dates |\n|---|---:|---:|")
    for tod in sorted(inv):
        say(f"| {tod} | {inv[tod][0]:,} | {inv[tod][1]} |")
    say(f"\nDates with any 0DTE row: {len(by_date)} ({min(by_date)} → {max(by_date)}).\n" if by_date else "\nNo 0DTE rows.\n")

    # Q1
    tfs = [START + timedelta(days=i) for i in range((end - START).days + 1)]
    tfs = [d for d in tfs if is_third_friday(d)]
    say("## Q1 third Fridays\n")
    say(f"Third Fridays in range: {len(tfs)} (train {sum(d <= SPLIT for d in tfs)}, holdout {sum(d > SPLIT for d in tfs)}).\n")
    say("| date | split | 0DTE rows am / non-am | in 06:30–06:45 am / non-am | opras am / non-am | mixed |")
    say("|---|---|---|---|---|---|")
    mixed: list[date] = []
    with_rows = 0
    for d in tfs:
        rec = by_date.get(d)
        if not rec:
            continue
        with_rows += 1
        am = rec.get("am", (0, 0, 0))
        non = [v for k, v in rec.items() if k != "am"]
        non_n = sum(v[0] for v in non); non_w = sum(v[1] for v in non); non_o = sum(v[2] for v in non)
        is_mixed = am[0] > 0 and non_n > 0
        if is_mixed:
            mixed.append(d)
        flag = "**mixed**" if is_mixed else ("am only" if am[0] else "")
        say(f"| {d} | {'train' if d <= SPLIT else 'holdout'} | {am[0]} / {non_n} | {am[1]} / {non_w} | {am[2]} / {non_o} | {flag} |")
    say("")
    am_any = [d for d in tfs if by_date.get(d, {}).get("am")]
    say(f"- Third Fridays with any 0DTE row: {with_rows}; with any `am` row: {len(am_any)}; "
        f"**mixed (am + non-am): {len(mixed)}** — train {sum(d <= SPLIT for d in mixed)}, holdout {sum(d > SPLIT for d in mixed)}.")
    say(f"- Mixed dates: {[d.isoformat() for d in mixed]}")
    say(f"- `am`-only third Fridays: {[d.isoformat() for d in am_any if d not in mixed]}")
    hold_tf = [d for d in tfs if d > SPLIT]
    say(f"- Holdout third Fridays and whether 0DTE rows exist: "
        f"{[(d.isoformat(), bool(by_date.get(d))) for d in hold_tf]}\n")
    say("DAILY-CAPTURE runs (`bt_backfill_runs`):\n")
    for cr_id, n, lo, hi in conn.execute(DAILY_RUNS_SQL).fetchall():
        say(f"- `{cr_id}`: {n} runs, {lo:%Y-%m-%d} → {hi:%Y-%m-%d}")
    say("")

    # Q3
    say("## Q3 non-third-Friday 0DTE dates with `am` rows\n")
    other = sorted(d for d, rec in by_date.items() if "am" in rec and not is_third_friday(d))
    say(f"{len(other)}: {[d.isoformat() for d in other]}\n")

    # Q4
    say("## Q4 minute slots taken by `am` rows on mixed dates (06:30–06:45 PT)\n")
    say("PK is `(opra_symbol, snapshot_pt)` and `insert_bars` is `ON CONFLICT DO NOTHING`, so each "
        "(contract, minute) holds one row — AM or PM, not both.\n")
    say("| date | opras | slots am | slots non-am | opras with no non-am bar |")
    say("|---|---:|---:|---:|---:|")
    for d in mixed:
        sl = conn.execute(SLOTS_SQL, (TICKER, d, WINDOW_START, WINDOW_END)).fetchall()
        n_am = sum(r[1] for r in sl); n_pm = sum(r[2] for r in sl)
        zero_pm = sum(1 for r in sl if r[2] == 0)
        say(f"| {d} | {len(sl)} | {n_am} | {n_pm} | {zero_pm} |")
    say("")

    conn.rollback()
    text = "\n".join(out) + "\n"
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
