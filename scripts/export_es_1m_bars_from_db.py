#!/usr/bin/env python
"""
Export 1-minute ES bars from ironbeam_es_1m_bars for a given date,
and (optionally) join with bars rebuilt from trades.

Usage:
  python scripts/export_es_1m_bars_from_db.py 2025-12-01

- If no date is provided, defaults to today's UTC date.
- Uses:
    DATABASE_URL (required)
    IRONBEAM_BARS_TABLE (optional, default "ironbeam_es_1m_bars")

Outputs:
  es_bars_from_db_YYYY-MM-DD.csv
  es_bars_compare_YYYY-MM-DD.csv   (only if es_bars_from_trades_YYYY-MM-DD.csv exists)
"""

from __future__ import annotations

import os
import sys
import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd
from sqlalchemy import create_engine, text

# --- Load .env if available ---
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

DB_BARS_TABLE = os.getenv("IRONBEAM_BARS_TABLE", "ironbeam_es_1m_bars")


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


def parse_date_from_argv() -> dt.date:
    if len(sys.argv) > 1:
        try:
            return dt.datetime.strptime(sys.argv[1], "%Y-%m-%d").date()
        except ValueError:
            print(f"Invalid date '{sys.argv[1]}', expected YYYY-MM-DD. Falling back to today (UTC).")
    return dt.datetime.utcnow().date()


def main():
    trade_date = parse_date_from_argv()
    print(f"[INFO] Exporting ES 1m bars from DB for date {trade_date.isoformat()} (UTC).")

    # UTC window [00:00, 24:00) for that date
    start_utc = dt.datetime.combine(trade_date, dt.time(0, 0), tzinfo=dt.timezone.utc)
    end_utc = start_utc + dt.timedelta(days=1)

    # For a TIMESTAMP WITHOUT TIME ZONE column we should pass naive UTC
    start_naive = start_utc.replace(tzinfo=None)
    end_naive = end_utc.replace(tzinfo=None)

    db_url = _get_db_url()
    engine = create_engine(db_url, pool_pre_ping=True)
    print(f"[INFO] Using DB: {db_url}")
    print(f"[INFO] Bars table: {DB_BARS_TABLE}")

    # ---- Fetch bars ----
    with engine.connect() as con:
        df_db = pd.read_sql(
            text(
                f"""
                SELECT datetime, open, high, low, close, volume
                FROM {DB_BARS_TABLE}
                WHERE datetime >= :start AND datetime < :end
                ORDER BY datetime ASC
                """
            ),
            con,
            params={"start": start_naive, "end": end_naive},
            parse_dates=["datetime"],
        )

    if df_db.empty:
        print(f"[WARN] No bars found for {trade_date.isoformat()} in {DB_BARS_TABLE}.")
        return

    print(f"[INFO] Loaded {len(df_db)} bars from DB.")

    # Treat datetime column as UTC-naive
    df_db = df_db.copy()
    df_db["datetime_utc"] = df_db["datetime"]

    # Add PT timestamp for visual checking
    df_db["datetime_pt"] = (
        df_db["datetime_utc"]
        .dt.tz_localize(ZoneInfo("UTC"))
        .dt.tz_convert(ZoneInfo("America/Los_Angeles"))
    )

    # Rename OHLCV columns to make compare-join clearer
    df_db = df_db.rename(
        columns={
            "open": "open_db",
            "high": "high_db",
            "low": "low_db",
            "close": "close_db",
            "volume": "volume_db",
        }
    )

    # Save DB-only bars
    out_db_name = f"es_bars_from_db_{trade_date.isoformat()}.csv"
    cols_order = [
        "datetime_utc",
        "datetime_pt",
        "open_db",
        "high_db",
        "low_db",
        "close_db",
        "volume_db",
    ]
    df_db[cols_order].to_csv(out_db_name, index=False)
    print(f"[DONE] Wrote {len(df_db)} bars to {out_db_name}")

    # ---- Optional: compare vs bars-from-trades CSV ----
    trades_csv = f"es_bars_from_trades_{trade_date.isoformat()}.csv"
    if os.path.exists(trades_csv):
        print(f"[INFO] Found {trades_csv}, building compare CSV...")

        df_trades_bars = pd.read_csv(trades_csv, parse_dates=["datetime_utc"])

        # Ensure datetime_utc is naive for join
        if df_trades_bars["datetime_utc"].dt.tz is not None:
            df_trades_bars["datetime_utc"] = (
                df_trades_bars["datetime_utc"]
                .dt.tz_convert(ZoneInfo("UTC"))
                .dt.tz_localize(None)
            )

        df_trades_bars = df_trades_bars.rename(
            columns={
                "open": "open_from_trades",
                "high": "high_from_trades",
                "low": "low_from_trades",
                "close": "close_from_trades",
                "volume": "volume_from_trades",
                "trade_count": "trade_count_from_trades",
            }
        )

        # Outer join so we see any missing minutes on either side
        df_compare = df_db.merge(
            df_trades_bars,
            on="datetime_utc",
            how="outer",
            suffixes=("", "_from_trades"),
        )

        df_compare = df_compare.sort_values("datetime_utc")

        # Add simple diffs (where both sides exist)
        for col in ["open", "high", "low", "close"]:
            db_col = f"{col}_db"
            tr_col = f"{col}_from_trades"
            diff_col = f"{col}_diff"
            if db_col in df_compare.columns and tr_col in df_compare.columns:
                df_compare[diff_col] = df_compare[db_col] - df_compare[tr_col]

        compare_name = f"es_bars_compare_{trade_date.isoformat()}.csv"
        df_compare.to_csv(compare_name, index=False)
        print(f"[DONE] Wrote joined comparison to {compare_name}")
    else:
        print(
            f"[INFO] {trades_csv} not found; skipping compare CSV. "
            # (If you want comparison, run the trades->bars script first.)
        )


if __name__ == "__main__":
    main()
