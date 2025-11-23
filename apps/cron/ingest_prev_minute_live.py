import os
import io
import re
import datetime as dt

import requests
import pandas as pd
from sqlalchemy import create_engine
from zoneinfo import ZoneInfo

ORATS_API_KEY = os.environ.get("ORATS_API_KEY")
DATABASE_URL = os.environ.get("DATABASE_URL")
DB_TABLE_NAME = "orats_monies_minute"

BASE_URL = "https://api.orats.io"
ENDPOINT = "/datav2/hist/live/one-minute/monies/implied.csv"
TICKER = "SPX"

def camel_to_snake(name: str) -> str:
    """Converts a camelCase string to snake_case."""
    name = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def run_ingest_for_previous_minute() -> None:
    """
    Fetch ORATS one-minute monies for the *previous* minute, transform it
    exactly like your historical backfill script, and append to Postgres.

    Designed to be run once per minute by a Render cron job.
    """
    if not ORATS_API_KEY or not DATABASE_URL:
        print("Error: ORATS_API_KEY and DATABASE_URL must be set in env.")
        return

    # --- Determine target ET minute (DST-aware) ---
    et = ZoneInfo("America/New_York")
    now_et = dt.datetime.now(et)
    target_et = now_et - dt.timedelta(minutes=1)

    # RTH guard: Mon–Fri, 09:30–16:00 ET
    market_open = dt.time(9, 30)
    market_close = dt.time(16, 0)
    if not (target_et.weekday() < 5 and market_open <= target_et.time() <= market_close):
        print(f"Skipping ingest: {target_et.strftime('%Y-%m-%d %H:%M')} ET is outside RTH.")
        return

    trade_datetime_str = target_et.strftime("%Y%m%d%H%M")
    print(f"Fetching {TICKER} for {target_et.strftime('%Y-%m-%d %H:%M')} ET "
          f"(tradeDate={trade_datetime_str})")

    params = {
        "ticker": TICKER,
        "tradeDate": trade_datetime_str,
        "token": ORATS_API_KEY,
    }
    url = f"{BASE_URL}{ENDPOINT}"

    # --- Fetch CSV ---
    try:
        r = requests.get(url, params=params, timeout=30)
        print(f"  -> GET {r.url}")
        r.raise_for_status()
        csv_text = r.text.strip()

        if not csv_text or csv_text.startswith("<"):
            print("  -> No CSV data returned for this minute.")
            return

        df = pd.read_csv(io.StringIO(csv_text))
        if df.empty:
            print("  -> Empty DataFrame for this minute.")
            return

        print(f"  -> Fetched {len(df)} rows.")
    except Exception as e:
        print(f"  -> Error during fetch: {e}")
        return

    # --- Transform (MATCHING your historical script) ---
    try:
        # Same as in your backfill: use snapShotDate to create snapshot_pt in PT
        utc_ts = pd.to_datetime(df["snapShotDate"], errors="coerce", utc=True)
        df["snapshot_pt"] = (
            utc_ts.dt.tz_convert("America/Los_Angeles")
                  .dt.tz_localize(None)
        )

        # Same camelCase → snake_case mapping
        df.columns = [camel_to_snake(col) for col in df.columns]

        print("  -> Transform complete.")
    except Exception as e:
        print(f"  -> Error during transform: {e}")
        return

    # --- Write to DB (same table as your historical script) ---
    try:
        engine = create_engine(DATABASE_URL)
        df.to_sql(DB_TABLE_NAME, engine, if_exists="append", index=False)
        print(f"  -> Wrote {len(df)} rows to '{DB_TABLE_NAME}'.")
    except Exception as e:
        msg = str(e)
        if "violates unique constraint" in msg or "duplicate key value" in msg:
            print("  -> Duplicate data for this minute; nothing new inserted.")
        else:
            print(f"  -> DB write error: {e}")


if __name__ == "__main__":
    run_ingest_for_previous_minute()
