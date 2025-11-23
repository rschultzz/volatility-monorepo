import requests
import pandas as pd
import io
import re
import datetime as dt
from sqlalchemy import create_engine

# --- Prerequisites ---
# This script requires the following packages to be installed:
# pip install requests pandas sqlalchemy psycopg2-binary

# --- Configuration ---
# For Render, set these as environment variables
ORATS_API_KEY = "**************4745"
DATABASE_URL = "postgresql+psycopg://rschultz:*************Dap@dpg-d38sm515pdvs738rknj0-a.oregon-postgres.render.com/curve_trading?sslmode=require"
DB_TABLE_NAME = "orats_monies_minute"

BASE_URL = "https://api.orats.io"
ENDPOINT = "/datav2/hist/live/one-minute/monies/implied.csv"
TICKER = "SPX"

def camel_to_snake(name):
    """Converts a camelCase string to snake_case."""
    name = re.sub(r'(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name).lower()

def run_ingest_for_previous_minute():
    """
    Fetches ORATS data for the previous minute, transforms it, and appends it
    to the database. Designed to be run as a cron job every minute.
    """
    if not ORATS_API_KEY or not DATABASE_URL:
        print("Error: ORATS_API_KEY and DATABASE_URL environment variables must be set.")
        return

    # --- Determine the target timestamp ---
    # Get the current time in ET and go back one minute
    et_tz = dt.timezone(dt.timedelta(hours=-4), name="ET") # EDT
    now_et = dt.datetime.now(et_tz)
    target_et = now_et - dt.timedelta(minutes=1)
    
    # Check if markets are open (e.g., 9:30 AM to 4:00 PM ET)
    market_open = dt.time(9, 30)
    market_close = dt.time(16, 0)
    if not (market_open <= target_et.time() <= market_close and target_et.weekday() < 5):
        print(f"Skipping ingest: {target_et.strftime('%Y-%m-%d %H:%M')} ET is outside market hours.")
        return

    trade_datetime_str = target_et.strftime("%Y%m%d%H%M")
    print(f"Fetching data for {TICKER} at {target_et.strftime('%Y-%m-%d %H:%M')} ET...")

    # --- Fetch Data ---
    params = {"ticker": TICKER, "tradeDate": trade_datetime_str, "token": ORATS_API_KEY}
    url = f"{BASE_URL}{ENDPOINT}"

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        csv_text = response.text.strip()
        if not csv_text or csv_text.startswith("<"):
            print("  -> No data returned from API.")
            return
        df = pd.read_csv(io.StringIO(csv_text))
        if df.empty:
            print("  -> No data returned for the specified parameters.")
            return
        print(f"  -> Successfully fetched {len(df)} rows.")
    except Exception as e:
        print(f"  -> An error occurred during data fetch: {e}")
        return

    # --- Transform Data ---
    try:
        utc_timestamps = pd.to_datetime(df['snapShotDate'], errors='coerce', utc=True)
        df['snapshot_pt'] = utc_timestamps.dt.tz_convert('America/Los_Angeles').dt.tz_localize(None)
        df.columns = [camel_to_snake(col) for col in df.columns]
    except Exception as e:
        print(f"  -> An error occurred during data transformation: {e}")
        return

    # --- Write to Database ---
    try:
        engine = create_engine(DATABASE_URL)
        # Use 'append'. The primary key on the table will prevent duplicate rows.
        df.to_sql(DB_TABLE_NAME, engine, if_exists='append', index=False)
        print(f"  -> Successfully wrote {len(df)} rows to the '{DB_TABLE_NAME}' table.")
    except Exception as e:
        # Catching integrity errors is normal if the job runs twice for the same minute
        if "violates unique constraint" in str(e) or "duplicate key value" in str(e):
            print("  -> Data for this timestamp already exists. No new rows added.")
        else:
            print(f"  -> An error occurred while writing to the database: {e}")

if __name__ == "__main__":
    run_ingest_for_previous_minute()
