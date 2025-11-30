#!/usr/bin/env python
"""
Ironbeam ES 1m bars historical data populator.

- Sets start and end dates for data population.
- Authenticates with Ironbeam API.
- Discovers the front-month ES symbol.
- Opens a WebSocket stream and subscribes to historical 1-minute bars for the specified date range.
- Writes the fetched data to the Postgres database.
- This is a one-time script that will exit after fetching the data.
"""

import json
import datetime as dt
from typing import List, Dict, Any
import os
import time
import threading

import requests
import websocket  # websocket-client
import pandas as pd
import ssl
import certifi
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ---------------------------------------------------------------------------
# CONFIGURABLE SETTINGS
# ---------------------------------------------------------------------------

# Set the start and end dates for historical data population
START_DATE = dt.datetime(2025, 11, 17, tzinfo=dt.timezone.utc)
END_DATE = dt.datetime(2025, 11, 19, 23, 59, 59, tzinfo=dt.timezone.utc)

API_BASE = os.environ.get("IRONBEAM_API_BASE", "https://demo.ironbeamapi.com/v2")
WS_BASE = os.environ.get("IRONBEAM_WS_BASE", "wss://demo.ironbeamapi.com/v2/stream")

USERNAME = os.environ.get("IRONBEAM_USERNAME")
PASSWORD_OR_APIKEY = os.environ.get("DEMO_API_IRONBEAM")
TENANT_API_KEY = os.environ.get("IRONBEAM_TENANT_API_KEY", "")

DB_BARS_TABLE = os.environ.get("IRONBEAM_BARS_TABLE", "ironbeam_es_1m_bars")

# How long to wait for the historical data dump before closing the connection
DATA_WAIT_TIMEOUT_SECONDS = 30

# Debug toggles
DEBUG_HTTP = True
DEBUG_WS_RAW = False

MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
    "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
    "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

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
        raise RuntimeError("IRONBEAM_USERNAME and DEMO_API_IRONBEAM must be set")
    url = f"{API_BASE}/auth"
    payload = {"username": USERNAME, "password": PASSWORD_OR_APIKEY}
    if TENANT_API_KEY:
        payload["apikey"] = TENANT_API_KEY
    print(f"[AUTH] POST {url}")
    resp = requests.post(url, json=payload)
    if DEBUG_HTTP:
        print("AUTH status:", resp.status_code)
    resp.raise_for_status()
    data = resp.json()
    token = data.get("token")
    if not token:
        raise RuntimeError(f"Auth failed, no token in response: {data}")
    print("Authenticated OK.")
    return token

def discover_es_front_month(token: str) -> str:
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
            print(f"Using ES symbol (front month): {rec['symbol']} (maturity {maturity(rec)})")
            return rec["symbol"]
    last = symbols_sorted[-1]
    print("All maturities < today, using last symbol:", last["symbol"])
    return last["symbol"]

def create_stream(token: str) -> str:
    url = f"{API_BASE}/stream/create"
    headers = {"Authorization": f"Bearer {token}"}
    print(f"[STREAM] GET {url}")
    resp = requests.get(url, headers=headers)
    if DEBUG_HTTP:
        print("STREAM CREATE status:", resp.status_code)
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "OK":
        raise RuntimeError(f"stream/create failed: {data}")
    stream_id = data["streamId"]
    print("Stream created:", stream_id)
    return stream_id

# ---------------------------------------------------------------------------
# HISTORICAL DATA HANDLING
# ---------------------------------------------------------------------------

def subscribe_historical_bars(token: str, stream_id: str, symbol: str, start_dt: dt.datetime, end_dt: dt.datetime) -> Dict[str, Any]:
    if ":" not in symbol:
        symbol = f"XCME:{symbol}"
    url = f"{API_BASE}/indicator/{stream_id}/timeBars/subscribe"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "symbol": symbol,
        "period": 1,
        "barType": "MINUTE",
        "startDate": int(start_dt.timestamp() * 1000),
        "endDate": int(end_dt.timestamp() * 1000),
    }
    print(f"[SUBSCRIBE] Historical TimeBars POST {url} {payload}")
    resp = requests.post(url, headers=headers, json=payload)
    print("HISTORICAL SUBSCRIBE status:", resp.status_code)
    if resp.status_code == 400 and "Can't subscribe to time bars" in resp.text:
        print("Got 'Can't subscribe to time bars' 400; continuing anyway (known Ironbeam quirk).")
        try:
            return resp.json()
        except Exception:
            return {}
    resp.raise_for_status()
    return resp.json()

def _to_utc_from_epoch(ts_raw: Any) -> dt.datetime | None:
    if ts_raw is None: return None
    try:
        ts = float(ts_raw)
        if ts > 10_000_000_000: ts /= 1000.0
        return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc)
    except (TypeError, ValueError, OSError):
        return None

def parse_time_bars_from_message(msg: Dict[str, Any]) -> List[Dict[str, Any]]:
    bars: List[Dict[str, Any]] = []
    if "ti" not in msg: return bars
    entries = msg["ti"]
    if isinstance(entries, dict): entries = [entries]
    for bar in entries:
        if not isinstance(bar, dict): continue
        t_utc = _to_utc_from_epoch(bar.get("t"))
        if t_utc is None: continue
        try:
            o, h, l, c = float(bar.get("o")), float(bar.get("h")), float(bar.get("l")), float(bar.get("c"))
        except (TypeError, ValueError):
            continue
        v = bar.get("v")
        v = float(v) if v is not None else None
        bars.append({"datetime": t_utc, "open": o, "high": h, "low": l, "close": c, "volume": v})
    return bars

# ---------------------------------------------------------------------------
# DB WRITER
# ---------------------------------------------------------------------------

import traceback
from sqlalchemy import create_engine

from sqlalchemy import MetaData, Table
from sqlalchemy.dialects.postgresql import insert as pg_insert

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

    # Log what we're about to insert
    print(f"[DB] Preparing to upsert {len(df)} rows into {DB_BARS_TABLE}.")
    print("[DB] Datetime range:", df["datetime"].min(), "->", df["datetime"].max())

    # Reflect the existing table structure
    metadata = MetaData()
    bars_table = Table(DB_BARS_TABLE, metadata, autoload_with=engine)

    rows = df.to_dict(orient="records")

    with engine.begin() as conn:
        stmt = pg_insert(bars_table).values(rows)

        # IMPORTANT: conflict target must match your PK / unique index
        # Here your PK is on (datetime), per the error message.
        stmt = stmt.on_conflict_do_nothing(
            index_elements=["datetime"]
        )

        conn.execute(stmt)

    print("[DB] Upsert completed (new rows inserted, existing datetimes ignored).")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    db_url = _get_db_url()
    engine = create_engine(db_url, pool_pre_ping=True)
    print(f"[INIT] Populating historical data into table '{DB_BARS_TABLE}'")
    print(f"[INIT] Date range: {START_DATE.isoformat()} to {END_DATE.isoformat()}")

    all_bars: List[Dict[str, Any]] = []

    try:
        token = authenticate()
        es_symbol = discover_es_front_month(token)
        stream_id = create_stream(token)
        ws_url = f"{WS_BASE}/{stream_id}?token={token}"
        print("Opening WebSocket:", ws_url)

        def on_open(ws):
            print("WebSocket opened, subscribing to historical time bars...")
            # Set a timer to close the websocket after the timeout
            def close_ws():
                print("Data collection timeout reached. Closing WebSocket.")
                ws.close()
            threading.Timer(DATA_WAIT_TIMEOUT_SECONDS, close_ws).start()
            print(f"Data collection will stop in {DATA_WAIT_TIMEOUT_SECONDS} seconds.")
            
            try:
                subscribe_historical_bars(token, stream_id, es_symbol, START_DATE, END_DATE)
            except Exception as e:
                print(f"Subscribe error in on_open: {e}")
                ws.close()

        def on_message(ws, message: str):
            nonlocal all_bars
            if DEBUG_WS_RAW: print("WS RAW:", message)
            try:
                msg = json.loads(message)
            except json.JSONDecodeError:
                return
            if "p" in msg: return # Ignore pings
            new_bars = parse_time_bars_from_message(msg)
            if new_bars:
                all_bars.extend(new_bars)
                print(f"Got {len(new_bars)} new bars (total {len(all_bars)})")

        def on_error(ws, error):
            print(f"WS error: {error}")

        def on_close(ws, code, reason):
            print(f"WebSocket closed: {code} {reason}")

        ws_app = websocket.WebSocketApp(ws_url, on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close)
        ws_app.run_forever(sslopt={"ca_certs": certifi.where(), "cert_reqs": ssl.CERT_REQUIRED})

        print("\n--- Data Collection Finished ---")
        write_bars_to_db(all_bars, engine)
        print(f"Successfully processed {len(all_bars)} bars from the stream.")

    except Exception as e:
        print(f"[ERROR] A critical error occurred: {e}")

if __name__ == "__main__":
    main()
