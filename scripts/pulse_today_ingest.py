# scripts/pulse_today_ingest.py
from __future__ import annotations

import argparse
import os
import time
import datetime as dt
import pandas as pd  # still needed indirectly by shared packages

# repo import path
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[0].parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from packages.shared.options_orats import ET_TZ, pt_minute_to_et, fetch_one_minute_monies
from packages.shared.ingest.monies_ingest import upsert_from_dashboard_minute
from packages.shared.cache.day_cache import refresh_today_if_needed


def is_rth(now_et: dt.datetime) -> bool:
    if now_et.weekday() >= 5:  # Sat/Sun
        return False
    t = now_et.time()
    return dt.time(9, 30) <= t <= dt.time(16, 0)


def minute_floor_et(ts_et: dt.datetime) -> dt.datetime:
    return ts_et.replace(second=0, microsecond=0)


def default_monthly_expiry_for(trade_date_iso: str) -> str:
    """
    Given a trade_date 'YYYY-MM-DD', return the 3rd Friday of the *following* month
    as 'YYYY-MM-DD'. This matches the dash's default expiry behavior.
    """
    trade_date = dt.date.fromisoformat(trade_date_iso)

    # Move to following month
    if trade_date.month == 12:
        year = trade_date.year + 1
        month = 1
    else:
        year = trade_date.year
        month = trade_date.month + 1

    first_of_month = dt.date(year, month, 1)
    # weekday: Mon=0 ... Fri=4
    days_to_friday = (4 - first_of_month.weekday()) % 7
    first_friday = first_of_month + dt.timedelta(days=days_to_friday)
    third_friday = first_friday + dt.timedelta(days=14)

    return third_friday.isoformat()


def ingest_minute(ticker: str, expiry_iso: str, ts_et: dt.datetime) -> int:
    """Fetch one minute from ORATS, upsert to DB, return #expiries upserted."""
    df = fetch_one_minute_monies(ts_et, ticker, expiry_iso)
    if df is None or df.empty:
        return 0
    return upsert_from_dashboard_minute(df, ticker=ticker)


def main():
    ap = argparse.ArgumentParser(
        description="Pulse ingest + refresh cache for *today* during RTH."
    )
    ap.add_argument("--ticker", default="SPX")
    ap.add_argument(
        "--expiry",
        required=False,
        help="Expiry as YYYY-MM-DD. If omitted or 'auto', use 3rd Friday of following month.",
    )
    ap.add_argument(
        "--backfill",
        type=int,
        default=0,
        help="also ingest the last N minutes (0 = just current)",
    )
    ap.add_argument(
        "--follow",
        action="store_true",
        help="loop every 30s until RTH ends",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="run even outside RTH (useful after-hours testing)",
    )
    args = ap.parse_args()

    if not os.getenv("ORATS_API_KEY"):
        raise SystemExit("ORATS_API_KEY is not set")
    if not os.getenv("DATABASE_URL"):
        raise SystemExit("DATABASE_URL is not set")

    while True:
        now_et = dt.datetime.now(ET_TZ)
        trade_date_iso = now_et.date().isoformat()

        # Decide expiry: explicit wins, otherwise default to next-month 3rd Friday
        expiry_arg = (args.expiry or "").strip()
        if not expiry_arg or expiry_arg.lower() == "auto":
            expiry_iso = default_monthly_expiry_for(trade_date_iso)
            print(
                f"[pulse] no explicit expiry; using default {expiry_iso} "
                f"for trade_date {trade_date_iso}"
            )
        else:
            expiry_iso = expiry_arg

        if not (args.force or is_rth(now_et)):
            print(
                f"[pulse] {now_et:%Y-%m-%d %H:%M:%S %Z} outside RTH — nothing to do."
            )
            if not args.follow:
                return
            time.sleep(30)
            continue

        # build list of minutes to ingest: backfill N, then current
        minutes: list[dt.datetime] = []
        base = minute_floor_et(now_et)
        for i in range(args.backfill, -1, -1):
            minutes.append(base - dt.timedelta(minutes=i))

        total = 0
        for ts in minutes:
            n = ingest_minute(args.ticker, expiry_iso, ts)
            total += n
            # small pacing for API courtesy
            time.sleep(0.1)

        # refresh the in-process day cache so the Smile callback gets instant “DB”
        df_or_tuple = refresh_today_if_needed(args.ticker, trade_date_iso, expiry_iso)
        day_df = df_or_tuple[0] if isinstance(df_or_tuple, tuple) else df_or_tuple
        rows = 0 if day_df is None or day_df.empty else len(day_df)

        label = f"{args.ticker} {trade_date_iso} {expiry_iso}"
        print(f"[pulse] upserted {total} expiry-rows; day-cache now has {rows} rows for {label}")

        if not args.follow:
            return
        # loop ~every 30s while market open
        time.sleep(30)


if __name__ == "__main__":
    main()
