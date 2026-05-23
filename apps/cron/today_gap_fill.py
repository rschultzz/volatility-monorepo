#!/usr/bin/env python
"""
ORATS one-minute "gap fill" script.

- Intended to be run as a cron job (e.g. every hour)
- Looks at today's trading session (ET) from 09:30 to min(now, 16:00)
- Fetches ORATS one-minute monies snapshots for each minute
- Transforms them exactly like your existing backfill script
- Writes into Postgres with ON CONFLICT DO NOTHING, so:
    * minutes you already have are ignored
    * missing minutes get filled
"""

import os
import re
import time
import io
import datetime as dt

import requests
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.dialects.postgresql import insert as pg_insert
from zoneinfo import ZoneInfo

# --- Configuration (prefer env vars) ---
ORATS_API_KEY = os.getenv("ORATS_API_KEY") or "YOUR_ORATS_API_KEY"
DATABASE_URL = (
    os.getenv("DATABASE_URL")
    or "postgresql+psycopg://user:pass@host/dbname?sslmode=require"
)
DB_TABLE_NAME = "orats_monies_minute"

BASE_URL = "https://api.orats.io"
ENDPOINT = "/datav2/hist/live/one-minute/monies/implied.csv"
TICKER = "SPX"

ET = ZoneInfo("America/New_York")


def camel_to_snake(name: str) -> str:
    """Converts a camelCase string to snake_case."""
    name = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def get_today_trading_minutes_et() -> tuple[str, list[str]]:
    """
    Returns (trade_date_str, [HH:MM, ...]) for today's trading
    session in ET, from 09:30 to min(now, 16:00).

    If it's a weekend or before market open, returns an empty list.
    """
    now_et = dt.datetime.now(ET)
    trade_date = now_et.date()
    trade_date_str = trade_date.strftime("%Y-%m-%d")

    # Weekend: nothing to do
    if now_et.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        return trade_date_str, []

    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)

    # Before market open: nothing to do yet
    if now_et < market_open:
        return trade_date_str, []

    end_time = min(now_et, market_close)

    minutes: list[str] = []
    current = market_open
    while current <= end_time:
        minutes.append(current.strftime("%H:%M"))
        current += dt.timedelta(minutes=1)

    return trade_date_str, minutes


def transform_orats_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Add snapshot_pt (PT-local naive timestamp)
    - Convert all columns to snake_case to match DB schema
    """
    if df.empty:
        return df

    # ORATS snapShotDate is a UTC ISO string
    utc_ts = pd.to_datetime(df["snapShotDate"], errors="coerce", utc=True)
    df["snapshot_pt"] = (
        utc_ts.dt.tz_convert("America/Los_Angeles").dt.tz_localize(None)
    )

    # Rename columns to snake_case (snapShotDate -> snap_shot_date, etc.)
    df.columns = [camel_to_snake(col) for col in df.columns]
    return df


def upsert_dataframe(df: pd.DataFrame, engine, table: Table) -> int:
    """
    Insert dataframe rows into Postgres with ON CONFLICT DO NOTHING,
    so duplicates (by primary key) are ignored.

    Returns number of rows actually inserted.
    """
    if df.empty:
        return 0

    records = df.to_dict(orient="records")
    inserted = 0
    chunk_size = 500  # keep parameter count under Postgres limit

    with engine.begin() as conn:
        for i in range(0, len(records), chunk_size):
            chunk = records[i : i + chunk_size]
            stmt = pg_insert(table).values(chunk).on_conflict_do_nothing()
            result = conn.execute(stmt)
            # rowcount = number of rows actually inserted (non-duplicates)
            inserted += result.rowcount or 0

    return inserted


def run_today_gap_fill():
    # Basic sanity checks
    if not ORATS_API_KEY or "YOUR_ORATS_API_KEY" in ORATS_API_KEY:
        raise RuntimeError("ORATS_API_KEY is not set (env ORATS_API_KEY).")
    if not DATABASE_URL or "user:pass@host/dbname" in DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set (env DATABASE_URL).")

    trade_date_str, trading_minutes = get_today_trading_minutes_et()
    if not trading_minutes:
        print("Market not open yet or non-trading day – nothing to backfill.")
        return

    print(f"[INIT] Gap-fill backfill for {trade_date_str} (ET).")
    print(
        f"[INIT] Minutes from {trading_minutes[0]} to {trading_minutes[-1]} ET "
        f"({len(trading_minutes)} total)."
    )

    all_data: list[pd.DataFrame] = []
    total_requests = len(trading_minutes)

    for idx, trade_time in enumerate(trading_minutes, start=1):
        trade_datetime_str = (
            f"{trade_date_str.replace('-', '')}{trade_time.replace(':', '')}"
        )
        params = {"ticker": TICKER, "tradeDate": trade_datetime_str, "token": ORATS_API_KEY}
        url = f"{BASE_URL}{ENDPOINT}"

        print(
            f"[{idx}/{total_requests}] Fetching {TICKER} {trade_datetime_str} "
            f"(ET minute {trade_time})..."
        )

        try:
            resp = requests.get(url, params=params, timeout=45)
            resp.raise_for_status()
            csv_text = resp.text.strip()

            # ORATS may return empty or HTML if no data / error
            if not csv_text or csv_text.startswith("<"):
                print("  -> No CSV data returned, skipping.")
                continue

            df = pd.read_csv(io.StringIO(csv_text))
            if df.empty:
                print("  -> Empty CSV, skipping.")
                continue

            all_data.append(df)

        except requests.exceptions.RequestException as e:
            print(f"  -> Request failed for {trade_time}: {e}")
        except Exception as e:
            print(f"  -> Failed to parse CSV for {trade_time}: {e}")

        # Be gentle with ORATS rate limits
        time.sleep(0.6)

    if not all_data:
        print("No data fetched for today's session. Exiting.")
        return

    print("\n[TRANSFORM] Combining and transforming fetched data...")
    full_df = pd.concat(all_data, ignore_index=True)
    print(f"  -> Total rows fetched from ORATS: {len(full_df)}")

    full_df = transform_orats_df(full_df)

    print("\n[DB] Connecting to database and upserting rows...")
    engine = create_engine(DATABASE_URL)
    metadata = MetaData()
    table = Table(DB_TABLE_NAME, metadata, autoload_with=engine)

    inserted = upsert_dataframe(full_df, engine, table)
    print(f"  -> Inserted {inserted} new rows (duplicates by primary key were ignored).")
    print("[DONE] Today's gap-fill job complete.")


if __name__ == "__main__":
    run_today_gap_fill()
