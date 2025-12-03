#!/usr/bin/env python
"""
Pulse intraday ingest for SPX (or any ticker) using ORATS snapshot/monies/implied.

- Runs once per minute (Render Cron).
- Computes the current ET minute and calls:
    https://api.orats.io/datav2/snapshot/monies/implied
  with tradeDate=YYYYMMDDHHMM in ET.

- Filters to the configured ticker (PULSE_TICKER, default: SPX).
- Renames ORATS columns to match existing DB schema:
    tradeDate        -> trade_date
    expirDate        -> expir_date
    snapShotEstTime  -> snap_shot_est_time
    snapShotDate     -> snap_shot_date
- Uses (ticker, snap_shot_date) as the per-minute key:
    DELETE existing rows for that ticker+minute,
    then INSERT the fresh rows.

Environment variables expected:

    ORATS_API_KEY          (required)  - ORATS Intraday API token
    CURVE_DB_URL           (preferred) - Postgres URL for curve_trading
    DATABASE_URL           (fallback)  - alternative env var name
    ORATS_MONIES_TABLE     (optional)  - default: "orats_monies_minute"
    PULSE_TICKER           (optional)  - default: "SPX"

"""

import io
import os
import sys
import datetime as dt
from zoneinfo import ZoneInfo

import requests
import pandas as pd
from sqlalchemy import create_engine, text


# ---------- Config ----------

ORATS_API_KEY = os.getenv("ORATS_API_KEY")
if not ORATS_API_KEY:
    print("[cron] ERROR: ORATS_API_KEY env var is not set", file=sys.stderr)
    sys.exit(1)

DB_URL = os.getenv("CURVE_DB_URL") or os.getenv("DATABASE_URL")
if not DB_URL:
    print("[cron] ERROR: CURVE_DB_URL or DATABASE_URL env var is not set", file=sys.stderr)
    sys.exit(1)

DB_TABLE_NAME = os.getenv("ORATS_MONIES_TABLE", "orats_monies_minute")
TICKER = os.getenv("PULSE_TICKER", "SPX")

ORATS_SNAPSHOT_URL = "https://api.orats.io/datav2/snapshot/monies/implied"
NY_TZ = ZoneInfo("America/New_York")


def _normalize_db_url(url: str) -> str:
    """
    Ensure SQLAlchemy-friendly driver.
    Examples:
      postgres://...        -> postgresql+psycopg://...
      postgresql://...      -> postgresql+psycopg://...
    """
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://") and "+psycopg" not in url:
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def get_engine() -> "Engine":
    url = _normalize_db_url(DB_URL)
    return create_engine(url, future=True)


def compute_trade_date_key(now_et: dt.datetime) -> str:
    """
    Return ORATS tradeDate string in ET in format YYYYMMDDHHMM.

    We floor to the current minute (no offset) – ORATS snapshot API
    will return the latest completed snapshot for that minute.
    """
    floored = now_et.replace(second=0, microsecond=0)
    return floored.strftime("%Y%m%d%H%M")


def fetch_orats_snapshot(trade_date_key: str) -> pd.DataFrame:
    """
    Call ORATS snapshot/monies/implied for the given tradeDate minute.

    Returns a pandas DataFrame (can be empty on errors or no data).
    """
    params = {
        "token": ORATS_API_KEY,
        "tradeDate": trade_date_key,
    }

    print(f"[cron] GET {ORATS_SNAPSHOT_URL} tradeDate={trade_date_key}")
    try:
        resp = requests.get(ORATS_SNAPSHOT_URL, params=params, timeout=30)
    except Exception as e:
        print(f"[cron] ERROR: request failed: {e}", file=sys.stderr)
        return pd.DataFrame()

    print(f"[cron] ORATS response status={resp.status_code}")
    if resp.status_code != 200:
        # Log a small slice of body for debugging but avoid flooding logs
        body_preview = resp.text[:300].replace("\n", "\\n")
        print(f"[cron] ERROR: non-200 response: {body_preview}", file=sys.stderr)
        return pd.DataFrame()

    text_data = resp.text.strip()
    if not text_data:
        print("[cron] WARNING: empty CSV body from ORATS")
        return pd.DataFrame()

    try:
        df = pd.read_csv(io.StringIO(text_data))
    except Exception as e:
        print(f"[cron] ERROR parsing CSV: {e}", file=sys.stderr)
        return pd.DataFrame()

    return df


def prepare_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to our ticker and rename/convert columns to match DB schema.
    """
    # Filter to our ticker (snapshot endpoint can contain many tickers)
    if "ticker" not in df_raw.columns:
        print("[cron] ERROR: 'ticker' column missing from ORATS CSV", file=sys.stderr)
        return pd.DataFrame()

    df = df_raw[df_raw["ticker"] == TICKER].copy()
    if df.empty:
        print(f"[cron] WARNING: no rows for ticker={TICKER} in snapshot")
        return df

    # Rename ORATS columns -> snake_case DB schema
    rename_map = {
        "tradeDate": "trade_date",
        "expirDate": "expir_date",
        "snapShotEstTime": "snap_shot_est_time",
        "snapShotDate": "snap_shot_date",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # Convert date/time columns
    if "snap_shot_date" in df.columns:
        df["snap_shot_date"] = pd.to_datetime(df["snap_shot_date"], utc=True)

    if "trade_date" in df.columns:
        # ORATS tradeDate from snapshot is just a date (YYYY-MM-DD)
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date

    # You already have ticker column; ensure it's correct type
    df["ticker"] = df["ticker"].astype(str)

    return df


def upsert_minute(engine, df: pd.DataFrame) -> None:
    """
    Use (ticker, snap_shot_date) as the per-minute key.

    - Determine the unique snap_shot_date for this batch.
    - DELETE any existing rows for (ticker, snap_shot_date).
    - INSERT the new rows via pandas.to_sql.
    """
    if "snap_shot_date" not in df.columns:
        print("[cron] ERROR: snap_shot_date column missing after prepare_dataframe", file=sys.stderr)
        return

    unique_snaps = df["snap_shot_date"].dropna().unique()
    if len(unique_snaps) == 0:
        print("[cron] ERROR: no non-null snap_shot_date values", file=sys.stderr)
        return
    if len(unique_snaps) > 1:
        print(f"[cron] WARNING: multiple snap_shot_date values ({len(unique_snaps)}); "
              f"using the first one as key", file=sys.stderr)

    snap_dt = pd.to_datetime(unique_snaps[0])
    print(f"[cron] minute key snap_shot_date={snap_dt.isoformat()}")

    with engine.begin() as conn:
        # 1) Delete any existing snapshot for this ticker + minute
        delete_sql = text(f"""
            DELETE FROM {DB_TABLE_NAME}
            WHERE ticker = :ticker
              AND snap_shot_date = :snap_shot_date
        """)
        deleted = conn.execute(delete_sql, {"ticker": TICKER, "snap_shot_date": snap_dt}).rowcount
        print(f"[cron] deleted {deleted} existing rows for ticker={TICKER} @ {snap_dt.isoformat()}")

        # 2) Insert the fresh batch
        df.to_sql(DB_TABLE_NAME, con=conn, if_exists="append", index=False, method="multi")
        print(f"[cron] inserted {len(df)} rows for ticker={TICKER} @ {snap_dt.isoformat()}")


def main() -> None:
    now_et = dt.datetime.now(NY_TZ)
    trade_date_key = compute_trade_date_key(now_et)
    pretty_time = now_et.strftime("%Y-%m-%d %H:%M ET")

    print(
        f"[cron] Fetching {TICKER} for {pretty_time} "
        f"(tradeDate={trade_date_key})"
    )

    df_raw = fetch_orats_snapshot(trade_date_key)
    if df_raw.empty:
        print(f"[cron] no data from ORATS for tradeDate={trade_date_key}")
        return

    print(f"[cron] fetched {len(df_raw)} raw rows from ORATS")
    df_prepped = prepare_dataframe(df_raw)
    if df_prepped.empty:
        print(f"[cron] no rows for ticker={TICKER} after filtering/prep")
        return

    print(f"[cron] {len(df_prepped)} rows for ticker={TICKER} after prep")

    engine = get_engine()
    upsert_minute(engine, df_prepped)


if __name__ == "__main__":
    main()
