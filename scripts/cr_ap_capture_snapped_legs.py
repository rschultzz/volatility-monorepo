#!/usr/bin/env python3
"""CR-AP Step 1a — capture the SNAPPED legs for an explicit date list. NO READ.

Sibling of scripts/cr_am_holdout_leg_capture.py (same fetcher, same role,
same run record), for signal dates whose honest legs were never fetched:
CR-AO's snapping moved 21 CR-AN trades onto listed strikes that June's
round5 backfill did not capture, so the harness dropped them from the clean
sample (CR-AP halt at G1).

For each date:
  1. target  = CR-AH's drift_target for the date (payload path), spot = table_spot
  2. expiry  = nth_business_day(date, 15)
  3. legs    = snap_vertical_legs(target, -10) for a debit date, (+10) for a
               credit date — exactly what the harness's leg builder does
  4. windows = entry-day RTH 06:30-13:00 PT for every snapped leg;
               settlement 12:50-13:00 PT at expiry;
               for debit dates, the harness's exit-on-touch window
               [touch_pt, touch_pt + 90 min] from detect_touch (rth_touch:
               the touch minute; gap_touch: the next 06:30 PT open), for the
               debit pair only. No window when there is no actionable touch.

Logs dates / legs / bars / 404s only. Never computes net_price, payoff or P&L.

Usage:
    PYTHONUNBUFFERED=1 python -u scripts/cr_ap_capture_snapped_legs.py \
        --dates 2023-11-10,... --debit-dates 2023-11-10,... --credit-dates 2023-12-22,... \
        [--cr-id CR-AP-capture] [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import sys
import time as _time
from datetime import date, datetime, timedelta
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
from packages.shared.options_cache.fetcher import fetch_option_bars
from packages.shared.options_cache.http_client import OratsPermanentError
from packages.shared.options_cache.opra import format_opra
from packages.shared.options_cache.strikes import StrikeNotListed, snap_vertical_legs
from scripts.cr_ah_step4_analysis import DTE_TARGET, detect_touch, nth_business_day
from scripts.cr_am_holdout_leg_capture import _payload_target

TICKER = "SPX"
OPRA_ROOT = "SPX"
TOUCH_WINDOW_MIN = 90     # get_touch_pos_val: touch_datetime_pt → + 90 minutes


def _parse_dates(s: str | None) -> set[date]:
    return {date.fromisoformat(x.strip()) for x in s.split(",")} if s else set()


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="CR-AP snapped-leg capture for an explicit date list (no read)")
    ap.add_argument("--dates", required=True, help="comma-separated ISO signal dates")
    ap.add_argument("--debit-dates", default=None, help="subset of --dates traded as debit (touch window fetched)")
    ap.add_argument("--credit-dates", default=None, help="subset of --dates traded as credit (default: all --dates)")
    ap.add_argument("--cr-id", default="CR-AP-capture")
    ap.add_argument("--dry-run", action="store_true", help="plan only: legs and windows, no fetch, no run row")
    args = ap.parse_args(argv)

    dates = sorted(_parse_dates(args.dates))
    debit_dates = _parse_dates(args.debit_dates)
    credit_dates = _parse_dates(args.credit_dates) if args.credit_dates else set(dates)
    today = date.today()

    conn = get_backfill_db_conn()
    assert_role_or_die(conn)
    print(f"CR-AP snapped-leg capture  cr_id={args.cr_id}  dry_run={args.dry_run}  dates={len(dates)}  "
          f"debit={len(debit_dates & set(dates))}  credit={len(credit_dates & set(dates))}  today={today}")

    plans = []
    for td in dates:
        target = _payload_target(conn, td)
        row = conn.execute("SELECT table_spot FROM orats_gex_landscape WHERE ticker=%s AND trade_date=%s", (TICKER, td)).fetchone()
        spot = float(row[0]) if row and row[0] is not None else None
        expiry = nth_business_day(td, DTE_TARGET)
        plan = {"trade_date": td, "target": target, "spot": spot, "expiry": expiry, "strikes": set(),
                "debit_pair": None, "credit_pair": None, "touch": None, "unlistable": None}
        if target is None:
            plan["unlistable"] = "no drift_target for the date"
        else:
            try:
                if td in debit_dates:
                    d_ = snap_vertical_legs(target, -10, expiry, td, conn, toward=spot)
                    plan["debit_pair"] = (d_.other, d_.anchor, d_.width_actual, d_.prior_close)
                    plan["strikes"].update({d_.other, d_.anchor})
                if td in credit_dates:
                    c_ = snap_vertical_legs(target, +10, expiry, td, conn, toward=spot)
                    plan["credit_pair"] = (c_.anchor, c_.other, c_.width_actual, c_.prior_close)
                    plan["strikes"].update({c_.anchor, c_.other})
            except StrikeNotListed as exc:
                plan["unlistable"] = str(exc)
        if td in debit_dates and target is not None:
            res, touch_pt = detect_touch(conn, td, expiry, target)
            plan["touch"] = (res, touch_pt)
        plans.append(plan)
        dp, cp, t = plan["debit_pair"], plan["credit_pair"], plan["touch"]
        print(f"  {td}  target={target if target is None else f'{target:.2f}'} spot={spot} expiry={expiry}"
              f"{'  debit ' + f'{dp[0]:g}/{dp[1]:g} (w {dp[2]:g}, chain {dp[3]})' if dp else ''}"
              f"{'  credit ' + f'{cp[0]:g}/{cp[1]:g} (w {cp[2]:g})' if cp else ''}"
              f"{'  touch=' + t[0] + (f' @ {t[1]:%Y-%m-%d %H:%M} PT' if t[1] else '') if t else ''}"
              f"{'  UNLISTABLE: ' + plan['unlistable'] if plan['unlistable'] else ''}")

    if args.dry_run:
        print("dry-run: no fetches, no run row.")
        return

    counters = {"dates": len(plans), "dates_unlistable": 0, "legs_planned": 0,
                "entry_windows": 0, "settle_windows": 0, "touch_windows": 0, "touch_windows_none": 0,
                "bars_written": 0, "cache_hits": 0, "gaps_filled": 0,
                "orats_404": 0, "fetch_exceptions": 0, "orats_404_detail": [], "exception_detail": []}

    with backfill_run(conn, args.cr_id) as run_id:
        print(f"\nRun ID: {run_id}\n")
        t0 = _time.perf_counter()

        def fetch(label, td, opras, start, end):
            try:
                r = fetch_option_bars(opras, start, end, source="historical_backfill", record_empty_windows=True)
            except OratsPermanentError as exc:
                counters["orats_404"] += 1
                counters["orats_404_detail"].append(f"{td} {label} {opras}: {exc}")
                print(f"      {label}: ORATS 4xx — {exc}")
                return False
            except Exception as exc:
                counters["fetch_exceptions"] += 1
                counters["exception_detail"].append(f"{td} {label}: {type(exc).__name__}: {exc}")
                print(f"      {label}: EXCEPTION {type(exc).__name__}: {exc}")
                return False
            counters["bars_written"] += r.bars_written
            counters["cache_hits"] += r.cache_hits
            counters["gaps_filled"] += r.gaps_filled
            print(f"      {label}: bars_written={r.bars_written} cache_hits={r.cache_hits} gaps_filled={r.gaps_filled}")
            return True

        for p in plans:
            td, ex = p["trade_date"], p["expiry"]
            if p["unlistable"] or not p["strikes"]:
                counters["dates_unlistable"] += 1
                print(f"  {td}: skipped — {p['unlistable']}")
                continue
            opras = [format_opra(OPRA_ROOT, ex, "C", s) for s in sorted(p["strikes"])]
            counters["legs_planned"] += len(opras)
            print(f"  {td}  legs={opras}")
            if fetch("entry-day", td, opras, datetime(td.year, td.month, td.day, 6, 30), datetime(td.year, td.month, td.day, 13, 0)):
                counters["entry_windows"] += 1
            if ex > today:
                print(f"      settlement: expiry {ex} is in the future — deferred")
            elif fetch("settlement", td, opras, datetime(ex.year, ex.month, ex.day, 12, 50), datetime(ex.year, ex.month, ex.day, 13, 0)):
                counters["settle_windows"] += 1
            if p["touch"]:
                res, touch_pt = p["touch"]
                if touch_pt is not None and res in ("rth_touch", "gap_touch") and p["debit_pair"]:
                    d_opras = [format_opra(OPRA_ROOT, ex, "C", s) for s in (p["debit_pair"][0], p["debit_pair"][1])]
                    if fetch(f"touch-window ({res} {touch_pt:%m-%d %H:%M}→+{TOUCH_WINDOW_MIN}m)", td, d_opras,
                             touch_pt, touch_pt + timedelta(minutes=TOUCH_WINDOW_MIN)):
                        counters["touch_windows"] += 1
                else:
                    counters["touch_windows_none"] += 1
                    print(f"      touch-window: none ({res})")

        counters["elapsed_s"] = round(_time.perf_counter() - t0, 1)
        summary = (f"captured {counters['entry_windows']} entry + {counters['settle_windows']} settlement + "
                   f"{counters['touch_windows']} touch windows over {counters['dates']} dates "
                   f"({counters['dates_unlistable']} unlistable, {counters['touch_windows_none']} debit dates without actionable touch); "
                   f"bars_written={counters['bars_written']} 404s={counters['orats_404']} exceptions={counters['fetch_exceptions']}; no P&L computed")
        update_run_smoke(conn, run_id, counters, summary)
        print(f"\nSUMMARY: {summary}")
        for d in counters["orats_404_detail"]:
            print("  404:", d)
        for d in counters["exception_detail"]:
            print("  exception:", d)


if __name__ == "__main__":
    main()
