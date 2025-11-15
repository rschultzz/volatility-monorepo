# scripts/pulse_today_ingest.py
from __future__ import annotations
import argparse, os, time
import datetime as dt
import pandas as pd

# repo import path
import sys, pathlib
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
    return dt.time(9,30) <= t <= dt.time(16,0)

def minute_floor_et(ts_et: dt.datetime) -> dt.datetime:
    return ts_et.replace(second=0, microsecond=0)

def ingest_minute(ticker: str, expiry_iso: str, ts_et: dt.datetime) -> int:
    """Fetch one minute from ORATS, upsert to DB, return #expiries upserted."""
    df = fetch_one_minute_monies(ts_et, ticker, expiry_iso)
    if df is None or df.empty:
        return 0
    return upsert_from_dashboard_minute(df, ticker=ticker)

def main():
    ap = argparse.ArgumentParser(description="Pulse ingest + refresh cache for *today* during RTH.")
    ap.add_argument("--ticker", default="SPX")
    ap.add_argument("--expiry", required=True, help="YYYY-MM-DD")
    ap.add_argument("--backfill", type=int, default=0, help="also ingest the last N minutes (0 = just current)")
    ap.add_argument("--follow", action="store_true", help="loop every 30s until RTH ends")
    ap.add_argument("--force", action="store_true", help="run even outside RTH (useful after-hours testing)")
    args = ap.parse_args()

    if not os.getenv("ORATS_API_KEY"):
        raise SystemExit("ORATS_API_KEY is not set")
    if not os.getenv("DATABASE_URL"):
        raise SystemExit("DATABASE_URL is not set")

    while True:
        now_et = dt.datetime.now(ET_TZ)
        trade_date_iso = now_et.date().isoformat()

        if not (args.force or is_rth(now_et)):
            print(f"[pulse] {now_et:%Y-%m-%d %H:%M:%S %Z} outside RTH — nothing to do.")
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
            n = ingest_minute(args.ticker, args.expiry, ts)
            total += n
            # small pacing for API courtesy
            time.sleep(0.1)

        # refresh the in-process day cache so the Smile callback gets instant “DB”
        df_or_tuple = refresh_today_if_needed(args.ticker, trade_date_iso, args.expiry)
        day_df = df_or_tuple[0] if isinstance(df_or_tuple, tuple) else df_or_tuple
        rows = 0 if day_df is None or day_df.empty else len(day_df)

        label = f"{args.ticker} {trade_date_iso} {args.expiry}"
        print(f"[pulse] upserted {total} expiry-rows; day-cache now has {rows} rows for {label}")

        if not args.follow:
            return
        # loop ~every 30s while market open
        time.sleep(30)

if __name__ == "__main__":
    main()
