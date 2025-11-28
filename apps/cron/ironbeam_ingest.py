import json
import datetime as dt
from typing import List, Dict, Any
import os
import time

import requests
import websocket  # websocket-client
import pandas as pd
import ssl
import certifi
from sqlalchemy import create_engine

# ---------------------------------------------------------------------------
# BASIC CONFIG – EDIT THESE
# ---------------------------------------------------------------------------

# HTTP REST base – v2 per docs
API_BASE = os.environ.get("IRONBEAM_API_BASE", "https://demo.ironbeamapi.com/v2")

# WS base – IMPORTANT: default to v1/stream because docs are inconsistent and
# many users report v1 being the working WS path.
WS_BASE = os.environ.get("IRONBEAM_WS_BASE", "wss://demo.ironbeamapi.com/v1/stream")

USERNAME = os.environ.get("IRONBEAM_USERNAME", "51395669")
PASSWORD_OR_APIKEY = os.environ.get("IRONBEAM_PASSWORD", "b7e9f2cd9e9c46e5bdd4f32b090b6ca4")
TENANT_API_KEY = os.environ.get("IRONBEAM_TENANT_API_KEY", "")

DB_TABLE_NAME = os.environ.get("IRONBEAM_BARS_TABLE", "ironbeam_es_1m_bars")

LOAD_SIZE = 2000

DEBUG_HTTP = True
DEBUG_WS_RAW = False

MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4,
    "MAY": 5, "JUN": 6, "JUL": 7, "AUG": 8,
    "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

def _get_db_url() -> str:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set in the environment")
    return db_url

# ---------------------------------------------------------------------------
# AUTH / SYMBOL DISCOVERY
# ---------------------------------------------------------------------------

def authenticate() -> str:
    """
    POST /auth to obtain a bearer token.
    """
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
    # unchanged from your version
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

    Ironbeam docs show both GET and POST; use POST here to match their newer examples.
    """
    url = f"{API_BASE}/stream/create"
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.post(url, headers=headers, json={})
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

# ... keep your subscribe_time_bars, parse_time_bars_from_message, write_bars_to_db
# exactly as they are now ...

def main():
    db_url = _get_db_url()
    engine = create_engine(db_url, pool_pre_ping=True)

    token = authenticate()
    es_symbol = discover_es_front_month(token)
    stream_id = create_stream(token)

    ws_url = f"{WS_BASE}/{stream_id}?token={token}"
    print("Opening WebSocket:", ws_url)

    def on_open(ws):
        print("WebSocket opened, subscribing to time bars...")
        try:
            subscribe_time_bars(token, stream_id, es_symbol)
        except Exception as e:
            print("Subscribe error in on_open:", e)

    def on_message(ws, message: str):
        if DEBUG_WS_RAW:
            print("WS RAW:", message)
        try:
            msg = json.loads(message)
        except json.JSONDecodeError:
            return

        if "p" in msg:
            # ping
            return

        new_bars = parse_time_bars_from_message(msg)
        if new_bars:
            write_bars_to_db(new_bars, engine)

    def on_error(ws, error):
        # websocket-client will pass a WebSocketBadStatusException here on handshake issues
        print("WS error:", error)

    def on_close(ws, code, reason):
        print("WebSocket closed:", code, reason)

    while True:
        ws_app = websocket.WebSocketApp(
            ws_url,
            on_open=on_open,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
        )
        ws_app.run_forever(
            sslopt={
                "ca_certs": certifi.where(),
                "cert_reqs": ssl.CERT_REQUIRED,
            },
            ping_interval=60,
            ping_timeout=10,
        )
        print("WebSocket disconnected, reconnecting in 5 seconds...")
        time.sleep(5)

if __name__ == "__main__":
    main()
