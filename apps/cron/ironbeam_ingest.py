#!/usr/bin/env python
"""
Ironbeam ES 1m bars + tick trades ingest worker for Render.

- Auth with /auth
- Discover front-month ES via /info/symbol/search/futures/XCME/ES
- Create streamId (GET /stream/create)
- Open WebSocket stream (wss://.../stream/{streamId}?token=...)
- Subscribe to:
    * 1-minute Time Bars indicator  -> 'ti' messages  -> BAR table
    * Trades stream                 -> 'tr' messages  -> TICK table
- Parse and write both into Postgres.

Environment variables expected:

  IRONBEAM_USERNAME        (required)
  IRONBEAM_PASSWORD        (required – password or API key)
  IRONBEAM_TENANT_API_KEY  (optional)
  DATABASE_URL             (required – Postgres URL)
  IRONBEAM_BARS_TABLE      (optional – default "ironbeam_es_1m_bars")
  IRONBEAM_TRADES_TABLE    (optional – default "ironbeam_es_trades")
  IRONBEAM_API_BASE        (optional – default https://demo.ironbeamapi.com/v2)
  IRONBEAM_WS_BASE         (optional – default wss://demo.ironbeamapi.com/v2/stream)
  IRONBEAM_LOAD_SIZE       (optional – initial bar load, default 2000)
"""

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
# BASIC CONFIG
# ---------------------------------------------------------------------------

API_BASE = os.environ.get("IRONBEAM_API_BASE", "https://demo.ironbeamapi.com/v2")
WS_BASE = os.environ.get("IRONBEAM_WS_BASE", "wss://demo.ironbeamapi.com/v2/stream")

USERNAME = os.environ.get("IRONBEAM_USERNAME")
PASSWORD_OR_APIKEY = os.environ.get("IRONBEAM_PASSWORD")
TENANT_API_KEY = os.environ.get("IRONBEAM_TENANT_API_KEY", "")

DB_BARS_TABLE = os.environ.get("IRONBEAM_BARS_TABLE", "ironbeam_es_1m_bars")
DB_TRADES_TABLE = os.environ.get("IRONBEAM_TRADES_TABLE", "ironbeam_es_trades")

# How many 1-minute bars to initially load
LOAD_SIZE = int(os.environ.get("IRONBEAM_LOAD_SIZE", "2000"))

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


# ---------------------------------------------------------------------------
# AUTH / SYMBOL DISCOVERY
# ---------------------------------------------------------------------------

def authenticate() -> str:
    """
    POST /auth to obtain a bearer token.
    Matches your working demo script.
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

    print(f"[AUTH] POST {url}")
    resp = requests.post(url, json=payload)
    if DEBUG_HTTP:
        print("AUTH status:", resp.status_code)
        print("AUTH body:", resp.text[:500])
    resp.raise_for_status()
    data = resp.json()

    token = data.get("token")
    if not token:
        raise RuntimeError(f"Auth failed, no token in response: {data}")
    print("[AUTH] Authenticated OK.")
    return token


def discover_es_front_month(token: str) -> str:
    """
    Use GET /info/symbol/search/futures/XCME/ES to get ES futures, then pick front month.
    Copied from your working demo.
    """
    url = f"{API_BASE}/info/symbol/search/futures/XCME/ES"
    headers = {"Authorization": f"Bearer {token}"}
    print(f"[SYMBOL] GET {url}")
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
            print(f"[SYMBOL] Using ES symbol (front month): {rec['symbol']} (maturity {maturity(rec)})")
            return rec["symbol"]

    last = symbols_sorted[-1]
    print("[SYMBOL] All maturities < today, using last symbol:", last["symbol"])
    return last["symbol"]


def create_stream(token: str) -> str:
    """
    Create a streamId for WebSocket streaming.

    IMPORTANT: This uses GET, exactly like your working script.
    """
    url = f"{API_BASE}/stream/create"
    headers = {"Authorization": f"Bearer {token}"}
    print(f"[STREAM] GET {url}")
    resp = requests.get(url, headers=headers)
    if DEBUG_HTTP:
        print("STREAM CREATE status:", resp.status_code)
        print("STREAM CREATE body:", resp.text[:500])
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") != "OK":
        raise RuntimeError(f"stream/create failed: {data}")
    stream_id = data["streamId"]
    print("[STREAM] Stream created:", stream_id)
    return stream_id


# ---------------------------------------------------------------------------
# SUBSCRIPTIONS: TIME BARS + TRADES
# ---------------------------------------------------------------------------

def subscribe_time_bars(token: str, stream_id: str, symbol: str) -> Dict[str, Any]:
    """
    POST /indicator/{streamId}/timeBars/subscribe for 1-minute bars.

    NOTE: symbol must be like "XCME:ES.Z25" (exchange-prefixed).
    Copied from your working script.
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
    print(f"[SUBSCRIBE] TimeBars POST {url} {payload}")
    resp = requests.post(url, headers=headers, json=payload)
    print("[SUBSCRIBE] TIME BARS status:", resp.status_code)
    if DEBUG_HTTP:
        print("TIME BARS body:", resp.text[:500])

    # Known Ironbeam quirk: 400/"Can't subscribe to time bars" can still mean
    # the subscription is active and data will flow on the stream.
    if resp.status_code == 400 and "Can't subscribe to time bars" in resp.text:
        print("[SUBSCRIBE] 'Can't subscribe to time bars' 400; continuing anyway.")
        try:
            return resp.json()
        except Exception:
            return {}

    resp.raise_for_status()
    return resp.json()


def subscribe_trades(token: str, stream_id: str, symbol: str) -> Dict[str, Any]:
    """
    GET /market/trades/subscribe/{streamId}?symbols=XCME:ES.Z25

    Subscribes to live trades, which arrive on the WebSocket in the 'tr' field.
    """
    if ":" not in symbol:
        symbol = f"XCME:{symbol}"

    url = f"{API_BASE}/market/trades/subscribe/{stream_id}"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"symbols": symbol}

    print(f"[SUBSCRIBE] Trades GET {url} {params}")
    resp = requests.get(url, headers=headers, params=params)
    print("[SUBSCRIBE] TRADES status:", resp.status_code)
    if DEBUG_HTTP:
        print("TRADES body:", resp.text[:500])

    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# PARSERS
# ---------------------------------------------------------------------------

def parse_time_bars_from_message(msg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse time bars from a WebSocket message.
    (Same 'ti' structure you already had.)
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

        t = bar.get("t")
        if t is None:
            continue

        try:
            ts = float(t)
            # ms vs s heuristic
            if ts > 10_000_000_000:
                ts /= 1000.0
            dt_utc = dt.datetime.fromtimestamp(ts, dt.timezone.utc)
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
        try:
            v_f = float(v) if v is not None else None
        except (TypeError, ValueError):
            v_f = None

        bars.append(
            {
                "datetime": dt_utc,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": v_f,
            }
        )

    return bars


def parse_trades_from_message(msg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse live trades from WebSocket 'tr' messages.

    The Ironbeam docs say trades appear under 'tr' when subscribed.
    Schema can vary slightly, so this parser is defensive:
      - Timestamp: tries keys 't', 'sendTime', 'tt'
      - Price:     'p' or 'price' or 'la'
      - Size:      'sz' or 'size' or 'q'
      - Total vol: 'tv' or 'totalVolume'
    """
    trades: List[Dict[str, Any]] = []

    if "tr" not in msg:
        return trades

    entries = msg["tr"]
    if isinstance(entries, dict):
        entries = [entries]

    for tr in entries:
        if not isinstance(tr, dict):
            continue

        # Timestamp
        ts_val = None
        for key in ("t", "sendTime", "tt"):
            val = tr.get(key)
            if val is not None:
                try:
                    ts_val = float(val)
                    break
                except (TypeError, ValueError):
                    continue

        if ts_val is None:
            continue

        if ts_val > 10_000_000_000:
            ts_val /= 1000.0

        try:
            dt_utc = dt.datetime.fromtimestamp(ts_val, dt.timezone.utc)
        except Exception:
            continue

        # Price (required)
        price_val = tr.get("p") or tr.get("price") or tr.get("la")
        if price_val is None:
            continue

        try:
            price = float(price_val)
        except (TypeError, ValueError):
            continue

        # Size, total volume (optional)
        size_val = tr.get("sz") or tr.get("size") or tr.get("q")
        tv_val = tr.get("tv") or tr.get("totalVolume")

        try:
            size = float(size_val) if size_val is not None else None
        except (TypeError, ValueError):
            size = None

        try:
            tv = float(tv_val) if tv_val is not None else None
        except (TypeError, ValueError):
            tv = None

        symbol = tr.get("s") or tr.get("symbol")

        trades.append(
            {
                "datetime": dt_utc,
                "symbol": symbol,
                "price": price,
                "size": size,
                "total_volume": tv,
            }
        )

    return trades


# ---------------------------------------------------------------------------
# DB WRITERS
# ---------------------------------------------------------------------------

def write_bars_to_db(bars: List[Dict[str, Any]], engine):
    """
    Append 1m bars to Postgres using SQLAlchemy.
    Drops duplicate datetimes inside the batch.
    """
    if not bars:
        return

    df = pd.DataFrame(bars).drop_duplicates(subset=["datetime"])
    if df.empty:
        return

    try:
        with engine.begin() as connection:
            df.to_sql(DB_BARS_TABLE, connection, if_exists="append", index=False)
        print(f"[DB] Wrote {len(df)} rows to {DB_BARS_TABLE}.")
    except Exception as e:
        msg = str(e)
        if "violates unique constraint" in msg or "duplicate key value" in msg:
            print("[DB] Bars duplicates found, skipping.")
        else:
            print(f"[DB] Bars write error: {e}")


def write_trades_to_db(trades: List[Dict[str, Any]], engine):
    """
    Append ticks (trades) to Postgres.
    De-dupes by (datetime, price, size) within the batch.
    """
    if not trades:
        return

    df = pd.DataFrame(trades).drop_duplicates(subset=["datetime", "price", "size"])
    if df.empty:
        return

    try:
        with engine.begin() as connection:
            df.to_sql(DB_TRADES_TABLE, connection, if_exists="append", index=False)
        print(f"[DB] Wrote {len(df)} rows to {DB_TRADES_TABLE}.")
    except Exception as e:
        msg = str(e)
        if "violates unique constraint" in msg or "duplicate key value" in msg:
            print("[DB] Trades duplicates found, skipping.")
        else:
            print(f"[DB] Trades write error: {e}")


# ---------------------------------------------------------------------------
# MAIN LOOP – RUN FOREVER, RECONNECT ON DISCONNECT
# ---------------------------------------------------------------------------

def run_worker():
    db_url = _get_db_url()
    engine = create_engine(db_url, pool_pre_ping=True)
    print(f"[INIT] Using DB bars table '{DB_BARS_TABLE}', trades table '{DB_TRADES_TABLE}'")

    # Optional: full websocket trace logs
    # if DEBUG_WS_RAW:
    #     websocket.enableTrace(True)

    while True:
        try:
            print("=== New auth / stream cycle starting ===")
            token = authenticate()
            es_symbol = discover_es_front_month(token)
            stream_id = create_stream(token)

            ws_url = f"{WS_BASE}/{stream_id}?token={token}"
            print("[WS] Opening WebSocket:", ws_url)

            def on_open(ws):
                print("[WS] Opened, subscribing to time bars + trades...")
                # Time bars subscription (existing behavior)
                try:
                    info_tb = subscribe_time_bars(token, stream_id, es_symbol)
                    print("[WS] TimeBars subscribe:", info_tb)
                except Exception as e:
                    print("[WS] Subscribe TimeBars error:", e)

                # Trades subscription (new)
                try:
                    info_tr = subscribe_trades(token, stream_id, es_symbol)
                    print("[WS] Trades subscribe:", info_tr)
                except Exception as e:
                    print("[WS] Subscribe Trades error:", e)

            def on_message(ws, message: str):
                if DEBUG_WS_RAW:
                    print("WS RAW:", message)

                try:
                    msg = json.loads(message)
                except json.JSONDecodeError:
                    return

                # Ignore ping messages
                if "p" in msg:
                    return

                # 1m bars
                new_bars = parse_time_bars_from_message(msg)
                if new_bars:
                    print(f"[WS] Got {len(new_bars)} new time bars.")
                    write_bars_to_db(new_bars, engine)

                # tick trades
                new_trades = parse_trades_from_message(msg)
                if new_trades:
                    print(f"[WS] Got {len(new_trades)} new trades.")
                    write_trades_to_db(new_trades, engine)

            def on_error(ws, error):
                print("[WS] error:", error)

            def on_close(ws, code, reason):
                print("[WS] closed:", code, reason)

            ws_app = websocket.WebSocketApp(
                ws_url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
            )

            # Blocks until close/error
            ws_app.run_forever(
                sslopt={
                    "ca_certs": certifi.where(),
                    "cert_reqs": ssl.CERT_REQUIRED,
                },
                ping_interval=60,
                ping_timeout=10,
            )

        except Exception as e:
            print("[TOP-LEVEL] worker error:", e)

        print("[TOP-LEVEL] WebSocket disconnected or error; reconnecting in 5 seconds...")
        time.sleep(5)


if __name__ == "__main__":
    run_worker()
