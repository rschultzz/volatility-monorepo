#!/usr/bin/env python
"""
Ironbeam ES 1m bars backfill from demo -> Postgres

- Auth with /auth
- Discover front-month ES via /info/symbol/search/futures/XCME/ES
- Create streamId
- Open WebSocket stream
- In on_open, subscribe to 1-minute Time Bars for ES
- Collect bars from 'ti' and write them into ironbeam_es_1m_bars
  using bar *start* time (Ironbeam's t minus 1 minute).
"""

import json
import datetime as dt
from typing import List, Dict, Any
import os
import time
import ssl

import requests
import websocket  # websocket-client
import pandas as pd
import certifi
from sqlalchemy import create_engine

# ---------------------------------------------------------------------------
# BASIC CONFIG
# ---------------------------------------------------------------------------

API_BASE = os.environ.get("IRONBEAM_API_BASE", "https://demo.ironbeamapi.com/v2")
WS_BASE = os.environ.get("IRONBEAM_WS_BASE", "wss://demo.ironbeamapi.com/v2/stream")

USERNAME = "51395669"
PASSWORD_OR_APIKEY = "b7e9f2cd9e9c46e5bdd4f32b090b6ca4"
TENANT_API_KEY = os.environ.get("IRONBEAM_TENANT_API_KEY", "")

DB_TABLE_NAME = os.environ.get("IRONBEAM_BARS_TABLE", "ironbeam_es_1m_bars")

# How many 1-minute bars to initially load
LOAD_SIZE = int(os.environ.get("IRONBEAM_LOAD_SIZE", "2000"))

# Debug toggles
DEBUG_HTTP = os.environ.get("IRONBEAM_DEBUG_HTTP", "true").lower() == "true"
DEBUG_WS_RAW = os.environ.get("IRONBEAM_DEBUG_WS_RAW", "false").lower() == "true"

MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
    "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
    "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


# ---------------------------------------------------------------------------
# DB HELPERS
# ---------------------------------------------------------------------------

def _normalize_db_url(url: str) -> str:
    """
    Render often gives postgres://; SQLAlchemy prefers postgresql+psycopg://
    """
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://") and "+psycopg" not in url:
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def _get_db_engine():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL environment variable is not set")
    db_url = _normalize_db_url(db_url)
    return create_engine(db_url, pool_pre_ping=True)


# ---------------------------------------------------------------------------
# AUTH / SYMBOL DISCOVERY
# ---------------------------------------------------------------------------

def authenticate() -> str:
    """
    POST /auth to obtain a bearer token.
    """
    if not USERNAME or not PASSWORD_OR_APIKEY:
        raise RuntimeError("IRONBEAM_USERNAME and IRONBEAM_PASSWORD must be set")

    url = f"{API_BASE}/auth"
    payload = {
        "username": USERNAME,
        "password": PASSWORD_OR_APIKEY,
    }
    if TENANT_API_KEY:
        payload["apikey"] = TENANT_API_KEY

    resp = requests.post(url, json=payload)
    if DEBUG_HTTP:
        print("AUTH status:", resp.status_code)
        print("AUTH body:", resp.text[:500])
    resp.raise_for_status()
    data = resp.json()

    token = data.get("token")
    if not token:
        raise RuntimeError(f"Auth failed, no token in response: {data}")
    print("Authenticated OK.")
    return token


def discover_es_front_month(token: str) -> str:
    """
    Use GET /info/symbol/search/futures/XCME/ES to get ES futures, then pick front month.
    """
    url = f"{API_BASE}/info/symbol/search/futures/XCME/ES"
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(url, headers=headers)
    if DEBUG_HTTP:
        print("SYMBOL FUTURES status:", resp.status_code)
        print("SYMBOL FUTURES body:", resp.text[:500])
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
    """
    Create a streamId for WebSocket streaming.
    """
    url = f"{API_BASE}/stream/create"
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(url, headers=headers)
    if DEBUG_HTTP:
        print("STREAM CREATE status:", resp.status_code)
        print("STREAM CREATE body:", resp.text[:500])
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "OK":
        raise RuntimeError(f"stream/create failed: {data}")
    stream_id = data["streamId"]
    print("Stream created:", stream_id)
    return stream_id


# ---------------------------------------------------------------------------
# TIME BARS SUBSCRIBE (HTTP) + PARSE (WS)
# ---------------------------------------------------------------------------

def subscribe_time_bars(token: str, stream_id: str, symbol: str) -> Dict[str, Any]:
    """
    POST /indicator/{streamId}/timeBars/subscribe for 1-minute bars.

    NOTE: symbol must be like "XCME:ES.Z25" (exchange-prefixed).
    """
    if ":" not in symbol:
        symbol = f"XCME:{symbol}"

    url = f"{API_BASE}/indicator/{stream_id}/timeBars/subscribe"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "symbol": symbol,
        "period": 1,
        "barType": "MINUTE",
        "loadSize": LOAD_SIZE,
    }
    resp = requests.post(url, headers=headers, json=payload)
    print("TIME BARS SUBSCRIBE status:", resp.status_code)
    if DEBUG_HTTP:
        print("TIME BARS SUBSCRIBE body:", resp.text[:500])

    # Known Ironbeam quirk: 400/"Can't subscribe to time bars" can still mean
    # the subscription is active and data will flow on the stream.
    if resp.status_code == 400 and "Can't subscribe to time bars" in resp.text:
        print("Got 'Can't subscribe to time bars' 400; continuing anyway (known Ironbeam quirk).")
        try:
            return resp.json()
        except Exception:
            return {}

    resp.raise_for_status()
    return resp.json()


def parse_time_bars_from_message(msg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse time bars from a WebSocket message.

    Ironbeam's 't' appears to be the bar *end* time; here we store the bar
    *start* time (t - 1 minute) so that 15:00–15:01 trades are labeled 15:00.
    """
    bars: List[Dict[str, Any]] = []

    if "ti" not in msg:
        return bars

    entries = msg["ti"]
    if isinstance(entries, dict):
        entries = [entries]

    for bar in entries:
        if not isinstance(bar, dict):
            continue

        if "t" in bar:
            t = bar.get("t")
            if t is None:
                continue
            try:
                ts = float(t)
                # Heuristic: ms vs s
                if ts > 10_000_000_000:
                    ts /= 1000.0
                # Ironbeam gives UTC; this is bar END
                dt_utc_end = dt.datetime.utcfromtimestamp(ts)
                # We want bar START (1 minute earlier)
                dt_utc = dt_utc_end - dt.timedelta(minutes=1)
            except Exception:
                continue

            try:
                o = float(bar.get("o"))
                h = float(bar.get("h"))
                l = float(bar.get("l"))
                c = float(bar.get("c"))
            except (TypeError, ValueError):
                continue

            v = bar.get("v")
            v = float(v) if v is not None else None

            bars.append(
                {
                    "datetime": dt_utc,
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "volume": v,
                }
            )

    return bars


# ---------------------------------------------------------------------------
# MAIN – OPEN WS, SUBSCRIBE, COLLECT, WRITE TO DB
# ---------------------------------------------------------------------------

def main():
    engine = _get_db_engine()
    print(f"[INIT] Writing to table '{DB_TABLE_NAME}'")

    # 1) Auth
    token = authenticate()

    # 2) Discover front-month ES symbol (e.g. "ES.Z25")
    es_symbol = discover_es_front_month(token)

    # 3) Create stream
    stream_id = create_stream(token)

    # 4) Open WebSocket, then subscribe in on_open
    ws_url = f"{WS_BASE}/{stream_id}?token={token}"
    print("Opening WebSocket:", ws_url)

    all_bars: List[Dict[str, Any]] = []
    stop_ts = time.time() + 10  # stop after ~10 seconds or when LOAD_SIZE reached

    def on_open(ws):
        print("WebSocket opened, subscribing to time bars...")
        try:
            info = subscribe_time_bars(token, stream_id, es_symbol)
            print("Subscribe response:", info)
        except Exception as e:
            print("Subscribe error in on_open:", e)

    def on_message(ws, message: str):
        nonlocal all_bars
        if DEBUG_WS_RAW:
            print("WS RAW:", message)

        try:
            msg = json.loads(message)
        except json.JSONDecodeError:
            return

        # Ignore pings, etc.
        if "p" in msg and len(msg.keys()) == 1:
            return

        new_bars = parse_time_bars_from_message(msg)
        if new_bars:
            all_bars.extend(new_bars)
            print(f"Got {len(new_bars)} new bars (total {len(all_bars)})")

        # Either we grabbed enough bars, or time window expired: close WS.
        if time.time() > stop_ts or len(all_bars) >= LOAD_SIZE:
            print("Closing WebSocket, enough data collected.")
            ws.close()

    def on_error(ws, error):
        print("WS error:", error)

    def on_close(ws, code, reason):
        print("WebSocket closed:", code, reason)

    ws_app = websocket.WebSocketApp(
        ws_url,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )

    # This blocks until ws.close() is called or an error occurs
    ws_app.run_forever(
        sslopt={
            "ca_certs": certifi.where(),
            "cert_reqs": ssl.CERT_REQUIRED,
        }
    )

    # -----------------------------------------------------------------------
    # After WS closes: build DataFrame and write to DB
    # -----------------------------------------------------------------------
    if not all_bars:
        print("No bars collected from stream.")
        return

    df = (
        pd.DataFrame(all_bars)
        .drop_duplicates(subset=["datetime"])
        .sort_values("datetime")
    )

    print(f"Total unique bars collected (after de-dup): {len(df)}")
    print("Sample head:")
    print(df.head())
    print("Sample tail:")
    print(df.tail())

    # Write into DB
    with engine.begin() as connection:
        df.to_sql(DB_TABLE_NAME, connection, if_exists="append", index=False)

    print(f"Wrote {len(df)} rows into {DB_TABLE_NAME}.")


if __name__ == "__main__":
    main()
