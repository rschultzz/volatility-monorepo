#!/usr/bin/env python3
from __future__ import annotations
import os
import argparse
import datetime as dt
import sys
from pathlib import Path

# --- Make repo root importable (so `packages.shared...` works) ---
HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]  # repo root (since this file is in repo_root/scripts/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from packages.shared.cache.day_cache import get_day_df, refresh_today_if_needed, cache_info

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default=os.getenv("TICKER", "SPX"))
    ap.add_argument("--date",   required=True, help="YYYY-MM-DD (trade_date)")
    ap.add_argument("--expiry", required=True, help="YYYY-MM-DD (expiry_date)")
    args = ap.parse_args()

    dburl = os.environ.get("DATABASE_URL")
    if not dburl:
        raise SystemExit("DATABASE_URL is not set. Export it, then re-run.\n"
                         "Example:\n  export DATABASE_URL='postgresql+psycopg://USER:PASS@HOST:PORT/curve_trading?sslmode=require'")

    df = get_day_df(args.ticker, args.date, args.expiry)
    print(f"[OK] Loaded {len(df)} rows from DB for {args.ticker} {args.date} {args.expiry}")
    print(df.head(5).to_string(index=False))

    if args.date == dt.date.today().isoformat():
        new_df, added = refresh_today_if_needed(args.ticker, args.date, args.expiry)
        print(f"[OK] Today refresh appended {added} rows; total now {len(new_df)}")

    print("Cache info:", cache_info())

if __name__ == "__main__":
    main()
