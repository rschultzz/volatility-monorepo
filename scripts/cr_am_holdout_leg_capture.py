#!/usr/bin/env python3
"""CR-AM Step 2 — capture holdout leg quotes into orats_options_minute. NO READ.

For every canonical magnet-above date after --split-date (default 2026-06-05)
whose outcome_status is computed or pending_history:
  1. target  = pick_drift_target(orats_gex_landscape.walls)  (canonical wall pick)
     also logs the CR-AH-style target (_materialize_payload(...)['regime']['drift_target'])
     and fetches the union of strikes when the two round to different 5-pt strikes.
  2. expiry  = nth_business_day(trade_date, 15)  (holiday-aware, from cr_ah_step4_analysis)
  3. legs    = round5(target) - 10 / + 0 / + 10  SPX calls  (debit: target-10/target;
               credit: target/target+10)
  4. fetch_option_bars for the entry-day RTH window (06:30-13:00 PT) and the
     expiry settlement window (12:50-13:00 PT). A settlement window that has not
     happened yet is SKIPPED (not fetched, not recorded as empty) so it can be
     fetched later.

Output is dates / targets / legs / bars_written / cache_hits / 404s ONLY.
This script never computes net_price, payoff, or P&L (ADR 2026-09-05
"Holdout Split Moves to 2026-06-05": capture is a data step, not a read).

Usage:
    PYTHONUNBUFFERED=1 python -u scripts/cr_am_holdout_leg_capture.py [--dry-run] [--split-date 2026-06-05] [--cr-id CR-AM-capture]
"""
from __future__ import annotations

import argparse
import os
import sys
import time as _time
from datetime import date, datetime, time
from pathlib import Path

# ── ENV (before any import that reads DATABASE_URL) ──────────────────────────
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

_bak_url = os.environ.get("BACKFILL_DATABASE_URL", "").strip()
if not _bak_url:
    sys.exit("ERROR: BACKFILL_DATABASE_URL not set.")
os.environ["DATABASE_URL"] = _bak_url   # options_cache repo reads DATABASE_URL

repo_root = str(Path(__file__).parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# ── IMPORTS ───────────────────────────────────────────────────────────────────
from packages.shared.backfill_safety import (
    assert_role_or_die, backfill_run, get_backfill_db_conn, update_run_smoke,
)
from packages.shared.canonical_version import CANONICAL_FEATURE_VERSION
from packages.shared.day_features import _LANDSCAPE_ROW_SQL, _OPEN_STRADDLE_SQL, _materialize_payload
from packages.shared.gex_landscape import compute_implied_move
from packages.shared.options_cache.fetcher import fetch_option_bars
from packages.shared.options_cache.http_client import OratsPermanentError
from packages.shared.options_cache.opra import format_opra
from packages.shared.outcomes import pick_drift_target
from scripts.cr_ah_step4_analysis import DTE_TARGET, nth_business_day, round5

TICKER = "SPX"
OPRA_ROOT = "SPX"


def _load_dates(conn, split_date: date) -> list[tuple[date, str]]:
    return conn.execute(
        """
        SELECT o.trade_date, o.outcome_status
        FROM bt_daily_outcomes o
        JOIN bt_daily_features f
          ON f.ticker = o.ticker AND f.trade_date = o.trade_date
         AND f.feature_version = o.feature_version AND f.active
        WHERE o.ticker = %s AND o.feature_version = %s AND o.active
          AND o.trade_date > %s
          AND f.regime_at_classification = 'magnet-above'
          AND o.outcome_status IN ('computed', 'pending_history')
        ORDER BY o.trade_date
        """,
        (TICKER, CANONICAL_FEATURE_VERSION, split_date),
    ).fetchall()


def _wall_target(conn, trade_date: date):
    row = conn.execute(
        "SELECT walls FROM orats_gex_landscape WHERE ticker = %s AND trade_date = %s",
        (TICKER, trade_date),
    ).fetchone()
    if not row:
        return None
    walls = row[0] if isinstance(row[0], list) else []
    return pick_drift_target(walls)


def _payload_target(conn, trade_date: date):
    """CR-AH Step 4's target for the same date (regime drift_target from the materialized payload)."""
    row = conn.execute(_LANDSCAPE_ROW_SQL, (TICKER, trade_date)).fetchone()
    if not row or row[0] is None or row[1] is None:
        return None
    landscape_rows, spot = row
    iv_row = conn.execute(
        _OPEN_STRADDLE_SQL,
        (trade_date.isoformat(), TICKER, datetime.combine(trade_date, time(6, 33, 0))),
    ).fetchone()
    if not iv_row or iv_row[0] is None:
        return None
    try:
        im = compute_implied_move(float(spot), float(iv_row[0]), dte=1.0)
    except Exception:
        return None
    if not im:
        return None
    dt_ = (_materialize_payload(landscape_rows, float(spot), im).get("regime") or {}).get("drift_target")
    return float(dt_) if dt_ is not None else None


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="CR-AM holdout leg capture (no read)")
    ap.add_argument("--split-date", type=date.fromisoformat, default=date(2026, 6, 5))
    ap.add_argument("--cr-id", default="CR-AM-capture")
    ap.add_argument("--dry-run", action="store_true", help="plan only: list dates/legs/windows, no fetch, no run row")
    args = ap.parse_args(argv)

    today = date.today()
    conn = get_backfill_db_conn()
    assert_role_or_die(conn)
    print(f"CR-AM holdout leg capture  split_date={args.split_date}  cr_id={args.cr_id}  dry_run={args.dry_run}  today={today}")

    dates = _load_dates(conn, args.split_date)
    print(f"Holdout stream: {len(dates)} magnet-above dates > {args.split_date} "
          f"({sum(1 for _, s in dates if s == 'computed')} computed, "
          f"{sum(1 for _, s in dates if s == 'pending_history')} pending)")

    plans = []
    for trade_date, status in dates:
        t_wall = _wall_target(conn, trade_date)
        t_payload = _payload_target(conn, trade_date)
        strikes: set[float] = set()
        for t in (t_wall, t_payload):
            if t is not None:
                base = float(round5(t))
                strikes.update({base - 10.0, base, base + 10.0})
        expiry = nth_business_day(trade_date, DTE_TARGET)
        plans.append({
            "trade_date": trade_date, "status": status, "target_wall": t_wall,
            "target_payload": t_payload, "strikes": sorted(strikes), "expiry": expiry,
            "settlement_in_future": expiry > today,
        })
        note = ""
        if t_wall is not None and t_payload is not None and round5(t_wall) != round5(t_payload):
            note = "  (targets round to different strikes → union fetched)"
        elif t_wall is None:
            note = "  (no positive-GEX wall → wall target None)"
        print(f"  {trade_date} {status:<15} wall={t_wall} payload={t_payload} "
              f"strikes={[int(s) for s in sorted(strikes)]} expiry={expiry}"
              f"{' settlement-in-future' if expiry > today else ''}{note}")

    if args.dry_run:
        print("dry-run: no fetches, no run row.")
        return

    counters = {"dates": len(plans), "dates_no_target": 0, "legs_planned": 0,
                "entry_windows_fetched": 0, "settle_windows_fetched": 0, "settle_windows_deferred": 0,
                "bars_written": 0, "cache_hits": 0, "gaps_filled": 0,
                "orats_404": 0, "fetch_exceptions": 0, "orats_404_detail": [], "exception_detail": []}

    with backfill_run(conn, args.cr_id) as run_id:
        print(f"\nRun ID: {run_id}\n")
        t0 = _time.perf_counter()
        for p in plans:
            td = p["trade_date"]
            if not p["strikes"]:
                counters["dates_no_target"] += 1
                print(f"  {td}: no target — skipped")
                continue
            opras = [format_opra(OPRA_ROOT, p["expiry"], "C", s) for s in p["strikes"]]
            counters["legs_planned"] += len(opras)
            entry_start = datetime(td.year, td.month, td.day, 6, 30)
            entry_end   = datetime(td.year, td.month, td.day, 13, 0)
            ex = p["expiry"]
            settle_start = datetime(ex.year, ex.month, ex.day, 12, 50)
            settle_end   = datetime(ex.year, ex.month, ex.day, 13, 0)
            print(f"  {td}  legs={[o for o in opras]}")
            for label, start, end in (("entry-day", entry_start, entry_end), ("settlement", settle_start, settle_end)):
                if label == "settlement" and p["settlement_in_future"]:
                    counters["settle_windows_deferred"] += 1
                    print(f"      {label}: expiry {ex} is in the future — deferred (not fetched, not recorded)")
                    continue
                try:
                    r = fetch_option_bars(opras, start, end, source="historical_backfill", record_empty_windows=True)
                except OratsPermanentError as exc:
                    counters["orats_404"] += 1
                    counters["orats_404_detail"].append(f"{td} {label}: {exc}")
                    print(f"      {label}: ORATS 4xx — {exc}")
                    continue
                except Exception as exc:  # G5: anything other than a 404 is reported, not fatal
                    counters["fetch_exceptions"] += 1
                    counters["exception_detail"].append(f"{td} {label}: {type(exc).__name__}: {exc}")
                    print(f"      {label}: EXCEPTION {type(exc).__name__}: {exc}")
                    continue
                counters["entry_windows_fetched" if label == "entry-day" else "settle_windows_fetched"] += 1
                counters["bars_written"] += r.bars_written
                counters["cache_hits"] += r.cache_hits
                counters["gaps_filled"] += r.gaps_filled
                print(f"      {label}: bars_written={r.bars_written} cache_hits={r.cache_hits} gaps_filled={r.gaps_filled}")

        elapsed = round(_time.perf_counter() - t0, 1)
        counters["elapsed_s"] = elapsed
        summary = (f"captured {counters['entry_windows_fetched']} entry windows + "
                   f"{counters['settle_windows_fetched']} settlement windows "
                   f"({counters['settle_windows_deferred']} deferred, future expiry) over {counters['dates']} dates; "
                   f"bars_written={counters['bars_written']} 404s={counters['orats_404']} "
                   f"exceptions={counters['fetch_exceptions']}; no P&L computed")
        update_run_smoke(conn, run_id, counters, summary)
        print(f"\nSUMMARY: {summary}")
        if counters["orats_404_detail"]:
            print("404 detail:"); [print("  " + d) for d in counters["orats_404_detail"]]
        if counters["exception_detail"]:
            print("exception detail:"); [print("  " + d) for d in counters["exception_detail"]]


if __name__ == "__main__":
    main()
