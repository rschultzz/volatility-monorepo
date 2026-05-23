#!/usr/bin/env python
"""
Ironbeam ES 1m bars historical data populator (REST version via /market/trades).

- Uses Ironbeam REST "Get Trades" endpoint to pull raw trades for ES
  over a given UTC date range, then aggregates them into 1-minute
  OHLCV bars and upserts into Postgres.

- No WebSocket or TimeBars subscription here. This is meant as a
  one-off historical backfill tool.

Config:
  - Set START_DATE / END_DATE below (timezone-aware, UTC).
  - Control demo vs live with IRONBEAM_ENV in .env:
      IRONBEAM_ENV=demo  (default)
      IRONBEAM_ENV=live

Environment variables needed are listed at the bottom of this file.
"""

import datetime as dt
from typing import List, Dict, Any
import os
import time

import requests
import pandas as pd
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.dialects.postgresql import insert as pg_insert
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ---------------------------------------------------------------------------
# CONFIGURABLE SETTINGS
# ---------------------------------------------------------------------------

# Set the start and end dates for historical data population (UTC)
START_DATE = dt.datetime(2025, 11, 17, tzinfo=dt.timezone.utc)
END_DATE   = dt.datetime(2025, 11, 19, 23, 59, 59, tzinfo=dt.timezone.utc)

# demo | live (anything else defaults to demo)
IRONBEAM_ENV = os.environ.get("IRONBEAM_ENV", "demo").lower()

if IRONBEAM_ENV == "live":
    API_BASE = os.environ.get("IRONBEAM_API_BASE", "https://live.ironbeamapi.com/v2")
else:
    API_BASE = os.environ.get("IRONBEAM_API_BASE", "https://demo.ironbeamapi.com/v2")

USERNAME = os.environ.get("IRONBEAM_USERNAME")

if IRONBEAM_ENV == "live":
    PASSWORD_OR_APIKEY = os.environ.get("LIVE_API_IRONBEAM")
else:
    PASSWORD_OR_APIKEY = os.environ.get("DEMO_API_IRONBEAM")

TENANT_API_KEY = os.environ.get("IRONBEAM_TENANT_API_KEY", "")

DB_BARS_TABLE = os.environ.get("IRONBEAM_BARS_TABLE", "ironbeam_es_1m_bars")

# Debug toggles
DEBUG_HTTP = True

MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
    "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
    "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

# Max trades per REST call (Ironbeam allows 1..100)
MAX_TRADES_PER_CALL = 100

# Sleep between calls just to be polite / avoid 429s
REQUEST_PAUSE_SECONDS = 0.1

# ---------------------------------------------------------------------------
# DB HELPERS
# ---------------------------------------------------------------------------

def _normalize_db_url(url: str) -> str:
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

# ---------------------------------------------------------------------------
# AUTH / SYMBOL DISCOVERY
# ---------------------------------------------------------------------------

def authenticate() -> str:
    if not USERNAME or not PASSWORD_OR_APIKEY:
        raise RuntimeError(
            "IRONBEAM_USERNAME and DEMO_API_IRONBEAM / LIVE_API_IRONBEAM must be set "
            "depending on IRONBEAM_ENV (demo|live)."
        )

    url = f"{API_BASE}/auth"
    payload: Dict[str, Any] = {
        "username": USERNAME,
        "password": PASSWORD_OR_APIKEY,
    }
    if TENANT_API_KEY:
        payload["apikey"] = TENANT_API_KEY

    print(f"[AUTH] POST {url} (env={IRONBEAM_ENV})")
    resp = requests.post(url, json=payload)
    if DEBUG_HTTP:
        print("AUTH status:", resp.status_code)
    resp.raise_for_status()
    data = resp.json()
    token = data.get("token")
    if not token:
        raise RuntimeError(f"Auth failed, no token in response: {data}")
    print("[AUTH] Authenticated OK.")
    return token


def discover_es_front_month(token: str) -> str:
    """
    Use the 'Get symbol futures' search endpoint to find the front-month ES.
    """
    url = f"{API_BASE}/info/symbol/search/futures/XCME/ES"
    headers = {"Authorization": f"Bearer {token}"}
    print(f"[SYMBOL] GET {url}")
    resp = requests.get(url, headers=headers)
    if DEBUG_HTTP:
        print("SYMBOL FUTURES status:", resp.status_code)
    resp.raise_for_status()
    data = resp.json()

    if data.get("status") != "OK":
        raise RuntimeError(f"Get symbol futures failed: {data}")

    symbols = data.get("symbols", [])
    if not symbols:
        raise RuntimeError("No ES futures returned – check entitlements.")

    today = dt.date.today()

    def maturity(rec: Dict[str, Any]) -> dt.date:
        month_str = str(rec.get("maturityMonth", "")).upper()
        year = int(rec.get("maturityYear"))
        month = MONTH_MAP.get(month_str, 1)
        return dt.date(year, month, 1)

    symbols_sorted = sorted(symbols, key=maturity)

    for rec in symbols_sorted:
        if maturity(rec) >= today:
            print(f"[SYMBOL] Using ES symbol (front month): {rec['symbol']} (maturity {maturity(rec)})")
            return rec["symbol"]

    last = symbols_sorted[-1]
    print("[SYMBOL] All maturities < today, using last symbol:", last["symbol"])
    return last["symbol"]

# ---------------------------------------------------------------------------
# TRADES FETCHER (REST: /market/trades)
# ---------------------------------------------------------------------------

def fetch_trades_for_range(
    token: str,
    symbol: str,
    start_dt: dt.datetime,
    end_dt: dt.datetime,
    max_per_call: int = MAX_TRADES_PER_CALL,
) -> List[Dict[str, Any]]:
    """
    Pull raw trades from Ironbeam REST "Get Trades" over [start_dt, end_dt] in UTC.

    Endpoint:
      GET /market/trades/{symbol}/{from}/{to}/{max}/{earlier}

    We page BACKWARDS by moving the 'to' bound earlier and keeping 'from'
    fixed at start_dt, so we can collect more than the last 100 trades
    if the API supports it.
    """
    if ":" not in symbol:
        symbol = f"XCME:{symbol}"

    headers = {"Authorization": f"Bearer {token}"}

    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    all_trades: List[Dict[str, Any]] = []
    current_to = end_ms
    page_num = 0

    print(f"[TRADES] Fetching trades for {symbol}")
    print(f"[TRADES] Requested range (ms): {start_ms} -> {end_ms}")

    while True:
        page_num += 1
        url = f"{API_BASE}/market/trades/{symbol}/{start_ms}/{current_to}/{max_per_call}/false"
        if DEBUG_HTTP:
            print(f"[TRADES] Page {page_num} GET {url}")

        resp = requests.get(url, headers=headers)
        if DEBUG_HTTP:
            print(f"[TRADES] Page {page_num} status:", resp.status_code)

        resp.raise_for_status()
        data = resp.json()

        trades = data.get("traders") or data.get("trades") or []
        if not trades:
            print(f"[TRADES] Page {page_num}: no trades returned; stopping pagination.")
            break

        # Debug: show time range of this page
        try:
            df_page = pd.DataFrame(trades)
            if "sendTime" in df_page.columns:
                def _to_dt(st):
                    try:
                        v = float(st)
                    except Exception:
                        return None
                    # seconds vs ms
                    if v > 10_000_000_000:
                        return dt.datetime.fromtimestamp(v / 1000.0, tz=dt.timezone.utc)
                    else:
                        return dt.datetime.fromtimestamp(v, tz=dt.timezone.utc)

                dts = df_page["sendTime"].apply(_to_dt)
                print(
                    f"[TRADES] Page {page_num}: {len(df_page)} trades, "
                    f"time range {dts.min()} -> {dts.max()}"
                )
            else:
                print(f"[TRADES] Page {page_num}: {len(trades)} trades (no sendTime field?)")
        except Exception as e:
            print(f"[TRADES] Page {page_num}: debug DataFrame failed: {e}")

        all_trades.extend(trades)

        # Find earliest sendTime in this page (in ms) to move 'to' backwards
        earliest_ms = None
        for t in trades:
            st = t.get("sendTime")
            if st is None:
                continue
            try:
                v = float(st)
            except (TypeError, ValueError):
                continue
            # Normalize to ms
            if v > 10_000_000_000:
                ms = int(v)
            else:
                ms = int(v * 1000.0)
            if earliest_ms is None or ms < earliest_ms:
                earliest_ms = ms

        if earliest_ms is None:
            print("[TRADES] No valid sendTime in this page; stopping to avoid loop.")
            break

        # Stop if we've reached or gone past the requested start
        if earliest_ms <= start_ms:
            print("[TRADES] Reached start of requested window; stopping pagination.")
            break

        # If we got fewer than max_per_call trades, there's probably no more data
        if len(trades) < max_per_call:
            print("[TRADES] Page returned fewer than max_per_call trades; likely no more older trades.")
            break

        # Move 'to' back to just before earliest trade in this page
        current_to = earliest_ms - 1
        if current_to <= start_ms:
            print("[TRADES] Next 'to' would be <= start; stopping pagination.")
            break

        time.sleep(REQUEST_PAUSE_SECONDS)

    print(f"[TRADES] Total trades fetched: {len(all_trades)}")
    return all_trades


# ---------------------------------------------------------------------------
# TRADES -> 1m BARS
# ---------------------------------------------------------------------------

def trades_to_1m_bars(trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Aggregate raw trades into 1-minute OHLCV bars in UTC.
    """
    if not trades:
        print("[BARS] No trades to aggregate.")
        return []

    records: List[Dict[str, Any]] = []
    for t in trades:
        send_time = t.get("sendTime")
        price = t.get("price")
        size = t.get("size") or t.get("totalVolume")

        if send_time is None or price is None or size is None:
            continue

        try:
            st = float(send_time)
            # Handle seconds vs ms
            if st > 10_000_000_000:
                ts = st / 1000.0
            else:
                ts = st
            dt_utc = dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc)
        except Exception:
            continue

        try:
            price_f = float(price)
            size_f = float(size)
        except (TypeError, ValueError):
            continue

        records.append(
            {
                "datetime": dt_utc,
                "price": price_f,
                "size": size_f,
            }
        )

    if not records:
        print("[BARS] No valid trade records after parsing.")
        return []

    df = pd.DataFrame(records)

    # Use datetime as the index for resampling
    df = df.set_index("datetime").sort_index()

    # Resample to 1-minute bars
    ohlc = df["price"].resample("1min").ohlc()
    vol = df["size"].resample("1min").sum()

    bars_df = ohlc.join(vol.rename("volume")).reset_index()

    # Enforce the requested [START_DATE, END_DATE] window
    bars_df = bars_df[
        (bars_df["datetime"] >= START_DATE) & (bars_df["datetime"] <= END_DATE)
    ]

    if bars_df.empty:
        print("[BARS] No 1m bars in requested window after aggregation.")
        return []

    print(
        "[BARS] Built",
        len(bars_df),
        "1m bars. Range:",
        bars_df["datetime"].min(),
        "->",
        bars_df["datetime"].max(),
    )

    return bars_df.to_dict(orient="records")



# ---------------------------------------------------------------------------
# DB WRITER (UPSERT on datetime)
# ---------------------------------------------------------------------------

def write_bars_to_db(bars: List[Dict[str, Any]], engine):
    if not bars:
        print("[DB] No bars to write.")
        return

    df = (
        pd.DataFrame(bars)
        .drop_duplicates(subset=["datetime"])
        .sort_values("datetime")
    )

    if df.empty:
        print("[DB] No new bars to write after deduplication.")
        return

    print(f"[DB] Preparing to upsert {len(df)} rows into {DB_BARS_TABLE}.")
    print("[DB] Datetime range:", df["datetime"].min(), "->", df["datetime"].max())

    metadata = MetaData()
    bars_table = Table(DB_BARS_TABLE, metadata, autoload_with=engine)

    rows = df.to_dict(orient="records")

    with engine.begin() as conn:
        stmt = pg_insert(bars_table).values(rows)

        # PK / unique index is on "datetime"
        stmt = stmt.on_conflict_do_nothing(index_elements=["datetime"])

        conn.execute(stmt)

    print("[DB] Upsert completed (new rows inserted, existing datetimes ignored).")

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    db_url = _get_db_url()
    engine = create_engine(db_url, pool_pre_ping=True)

    print(f"[INIT] Populating historical data into table '{DB_BARS_TABLE}'")
    print(f"[INIT] Date range (UTC): {START_DATE.isoformat()} to {END_DATE.isoformat()}")
    print(f"[INIT] IRONBEAM_ENV={IRONBEAM_ENV}, API_BASE={API_BASE}")

    try:
        token = authenticate()
        es_symbol = discover_es_front_month(token)
        print(f"[INIT] ES symbol for backfill: {es_symbol}")

        trades = fetch_trades_for_range(token, es_symbol, START_DATE, END_DATE)
        bars = trades_to_1m_bars(trades)
        write_bars_to_db(bars, engine)

        print(f"[DONE] Successfully processed {len(bars)} 1m bars into {DB_BARS_TABLE}.")
    except Exception as e:
        print(f"[ERROR] A critical error occurred: {e}")


if __name__ == "__main__":
    main()
