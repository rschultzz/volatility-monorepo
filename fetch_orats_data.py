import requests
import pandas as pd
import io
import re
import time
import datetime as dt
from sqlalchemy import create_engine

# --- Prerequisites ---
# pip install requests pandas sqlalchemy psycopg2-binary

# --- Configuration ---
ORATS_API_KEY = "cd809e2a-287c-4af7-9b05-a344df894745"
DATABASE_URL = "postgresql+psycopg://rschultz:5hUHvSVPDyVXhz7acgJZvlvnj7nFMDap@dpg-d38sm515pdvs738rknj0-a.oregon-postgres.render.com/curve_trading?sslmode=require"
DB_TABLE_NAME = "orats_monies_minute"

BASE_URL = "https://api.orats.io"
ENDPOINT = "/datav2/hist/live/one-minute/monies/implied.csv"
TICKER = "SPX"

# --- Date Range for Backfill ---
START_DATE = "2025-11-03"
END_DATE = "2025-11-07"

def camel_to_snake(name):
    """Converts a camelCase string to snake_case."""
    name = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name).lower()

def get_trading_minutes(start_time="09:30", end_time="16:00"):
    """Generates HH:MM strings for each minute in a time range."""
    minutes = []
    current_time = dt.datetime.strptime(start_time, "%H:%M")
    end = dt.datetime.strptime(end_time, "%H:%M")
    while current_time <= end:
        minutes.append(current_time.strftime("%H:%M"))
        current_time += dt.timedelta(minutes=1)
    return minutes

def get_trading_days(start_date_str: str, end_date_str: str) -> list[str]:
    """Generates a list of trading days (Mon-Fri) within a date range."""
    days = []
    start_date = dt.datetime.strptime(start_date_str, "%Y-%m-%d").date()
    end_date = dt.datetime.strptime(end_date_str, "%Y-%m-%d").date()
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() < 5: # Monday is 0 and Sunday is 6
            days.append(current_date.strftime("%Y-%m-%d"))
        current_date += dt.timedelta(days=1)
    return days

def run_historical_backfill():
    """
    Fetches historical ORATS data for a specific date range and appends it
    to the database.
    """
    if not ORATS_API_KEY or not DATABASE_URL:
        print("Error: API Key and Database URL must be set.")
        return

    all_data = []
    trading_days = get_trading_days(START_DATE, END_DATE)
    trading_minutes = get_trading_minutes()
    total_minutes = len(trading_days) * len(trading_minutes)
    minute_count = 0

    print(f"Starting backfill for {len(trading_days)} trading days from {START_DATE} to {END_DATE}...")

    for trade_date in trading_days:
        for trade_time in trading_minutes:
            minute_count += 1
            trade_datetime_str = f"{trade_date.replace('-', '')}{trade_time.replace(':', '')}"
            params = {"ticker": TICKER, "tradeDate": trade_datetime_str, "token": ORATS_API_KEY}
            url = f"{BASE_URL}{ENDPOINT}"

            print(f"({minute_count}/{total_minutes}) Fetching data for {trade_date} {trade_time} ET...")

            try:
                response = requests.get(url, params=params, timeout=45)
                response.raise_for_status()
                csv_text = response.text.strip()
                if not csv_text or csv_text.startswith("<"):
                    continue
                df = pd.read_csv(io.StringIO(csv_text))
                if not df.empty:
                    all_data.append(df)
            except requests.exceptions.RequestException as e:
                print(f"  -> Request failed for {trade_time}: {e}")
            except Exception as e:
                print(f"  -> Failed to parse CSV for {trade_time}: {e}")
            
            time.sleep(0.6)

    if not all_data:
        print("No data was fetched for the specified date range. Exiting.")
        return

    print("\nCombining all fetched data...")
    full_history_df = pd.concat(all_data, ignore_index=True)
    print(f"Total rows fetched: {len(full_history_df)}")

    # --- Transform Data ---
    try:
        print("Transforming data...")
        utc_timestamps = pd.to_datetime(full_history_df['snapShotDate'], errors='coerce', utc=True)
        full_history_df['snapshot_pt'] = utc_timestamps.dt.tz_convert('America/Los_Angeles').dt.tz_localize(None)
        full_history_df.columns = [camel_to_snake(col) for col in full_history_df.columns]
        print("  -> Data transformed successfully.")
    except Exception as e:
        print(f"An error occurred during data transformation: {e}")
        return

    # --- Write to Database ---
    try:
        print("Connecting to the database...")
        engine = create_engine(DATABASE_URL)

        print(f"Appending {len(full_history_df)} rows to the '{DB_TABLE_NAME}' table...")
        # Use 'append'. The existing primary key will prevent duplicate rows.
        full_history_df.to_sql(DB_TABLE_NAME, engine, if_exists='append', index=False)
        print("  -> Data appended successfully.")
        print("\nHistorical backfill complete.")

    except Exception as e:
        if "violates unique constraint" in str(e) or "duplicate key value" in str(e):
            print("  -> Some rows were duplicates and were skipped by the database, as expected.")
        else:
            print(f"An error occurred while writing to the database: {e}")

if __name__ == "__main__":
    run_historical_backfill()
