#!/usr/bin/env python
"""
Rebuild 1-minute ES bars from Ironbeam trades and write them to CSV.

Usage:
  python scripts/rebuild_es_bars_from_trades.py 2025-12-01

- If no date is provided, defaults to today's UTC date.
- Uses:
    DATABASE_URL (required)
    IRONBEAM_TRADES_TABLE (optional, default "ironbeam_es_trades")
"""

from __future__ import annotations

import os
import sys
import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import create_engine, text

# --- Load .env if available (for DATABASE_URL, etc.) ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

DB_TRADES_TABLE = os.getenv("IRONBEAM_TRADES_TABLE", "ironbeam_es_trades")


def _normalize_db_url(url: str) -> str:
    """
    Render often gives postgres://; SQLAlchemy prefers postgresql+psycopg://
    """
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://") and "+psycopg" not in url:
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def _get_db_url() -> str:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set in the environment")
    return _normalize_db_url(db_url)


def parse_trade_date_from_argv() -> dt.date:
    if len(sys.argv) > 1:
        try:
            return dt.datetime.strptime(sys.argv[1], "%Y-%m-%d").date()
        except ValueError:
            print(f"Invalid date '{sys.argv[1]}', expected YYYY-MM-DD. Falling back to today (UTC).")
    # default: today (UTC)
    return dt.datetime.utcnow().date()


def main():
    trade_date = parse_trade_date_from_argv()
    print(f"[INFO] Building 1m bars from trades for date {trade_date.isoformat()} (UTC).")

    # UTC window: [00:00, 24:00) for that date
    start_utc = dt.datetime.combine(trade_date, dt.time(0, 0), tzinfo=dt.timezone.utc)
    end_utc = start_utc + dt.timedelta(days=1)

    db_url = _get_db_url()
    engine = create_engine(db_url, pool_pre_ping=True)
    print(f"[INFO] Using DB: {db_url}")
    print(f"[INFO] Trades table: {DB_TRADES_TABLE}")

    # ---- Fetch trades ----
    with engine.connect() as con:
        df_trades = pd.read_sql(
            text(
                f"""
                SELECT ts_utc, price, size
                FROM {DB_TRADES_TABLE}
                WHERE ts_utc >= :start AND ts_utc < :end
                ORDER BY ts_utc ASC
                """
            ),
            con,
            params={"start": start_utc, "end": end_utc},
            parse_dates=["ts_utc"],
        )

    if df_trades.empty:
        print(f"[WARN] No trades found for {trade_date.isoformat()} in {DB_TRADES_TABLE}.")
        return

    print(f"[INFO] Loaded {len(df_trades)} trades.")

    # ---- Normalize timestamps to UTC tz-aware ----
    if df_trades["ts_utc"].dt.tz is None:
        # If the column is naive, treat it as UTC
        df_trades["ts_utc"] = df_trades["ts_utc"].dt.tz_localize(dt.timezone.utc)
    else:
        # If it already has a tz, convert to UTC
        df_trades["ts_utc"] = df_trades["ts_utc"].dt.tz_convert(dt.timezone.utc)

    # Floor to the minute in UTC
    df_trades["minute"] = df_trades["ts_utc"].dt.floor("min")

    # ---- Group into 1-minute OHLCV ----
    bars_from_trades = (
        df_trades
        .groupby("minute")
        .agg(
            open=("price", "first"),
            high=("price", "max"),
            low=("price", "min"),
            close=("price", "last"),
            volume=("size", "sum"),
            trade_count=("size", "count"),
        )
        .reset_index()
        .rename(columns={"minute": "datetime_utc"})
    )

    # Add PT version for visual checks
    bars_from_trades["datetime_pt"] = (
        bars_from_trades["datetime_utc"]
        .dt.tz_convert(ZoneInfo("America/Los_Angeles"))
    )

    # Make datetime_utc naive for easier joins with ironbeam_es_1m_bars (which is naive UTC)
    bars_from_trades["datetime_utc"] = (
        bars_from_trades["datetime_utc"]
        .dt.tz_convert(dt.timezone.utc)
        .dt.tz_localize(None)
    )

    print(f"[INFO] Built {len(bars_from_trades)} 1-minute bars from trades.")

    # ---- Save to CSV ----
    out_name = f"es_bars_from_trades_{trade_date.isoformat()}.csv"
    cols_order = [
        "datetime_utc",
        "datetime_pt",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trade_count",
    ]
    bars_from_trades[cols_order].to_csv(out_name, index=False)
    print(f"[DONE] Wrote {len(bars_from_trades)} bars to {out_name}")


if __name__ == "__main__":
    main()

