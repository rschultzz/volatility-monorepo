#!/usr/bin/env python3
from __future__ import annotations
import os
import sys
import argparse
from pathlib import Path

# --- Make repo root importable (so `packages.shared...` works) ---
HERE = Path(__file__).resolve()
ROOT = HERE.parents[1]  # repo root (since this file is in repo_root/scripts/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import text
from packages.shared.cache.day_cache import get_day_df, cache_info
from packages.shared.utils.data_io import tx


def list_expiries_in_db(ticker: str, trade_date: str) -> list[str]:
    sql = text("""
        SELECT expiry_date
        FROM orats_monies_minute
        WHERE ticker = :t AND trade_date = :d
        GROUP BY 1
        ORDER BY 1
    """)
    with tx() as conn:
        rows = conn.execute(sql, {"t": ticker, "d": trade_date}).mappings().all()
    return [r["expiry_date"] for r in rows]


def main():
    ap = argparse.ArgumentParser(description="Warm in-memory day cache from DB.")
    ap.add_argument("--ticker", default=os.getenv("TICKER", "SPX"))
    ap.add_argument("--date", required=True, help="Trade date YYYY-MM-DD")
    ap.add_argument(
        "--expiry",
        action="append",
        help="Expiry YYYY-MM-DD (repeatable). If omitted, use --all-expiries."
    )
    ap.add_argument("--all-expiries", action="store_true",
                    help="Warm all expiries present in DB for this trade date.")
    args = ap.parse_args()

    if "DATABASE_URL" not in os.environ:
        raise SystemExit("DATABASE_URL is not set. Export it, then re-run.")

    expiries: list[str]
    if args.expiry:
        expiries = args.expiry
    elif args.all_expiries:
        expiries = list_expiries_in_db(args.ticker, args.date)
        if not expiries:
            print(f"[WARN] No expiries found in DB for {args.ticker} {args.date}")
            return
    else:
        raise SystemExit("Provide at least one --expiry or pass --all-expiries.")

    total_rows = 0
    for e in expiries:
        df = get_day_df(args.ticker, args.date, e)
        print(f"[OK] Warmed {args.ticker} {args.date} {e}: {len(df)} rows")
        total_rows += len(df)

    print("Cache info:", cache_info())
    print(f"[DONE] Warmed {len(expiries)} expiry(ies), total {total_rows} rows.")


if __name__ == "__main__":
    main()
