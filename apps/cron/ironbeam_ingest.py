#!/usr/bin/env python
"""
Ironbeam ES 1m bars + trades ingest worker for Render.

- Auth with /auth
- Discover front-month ES via /info/symbol/search/futures/XCME/ES
- Create streamId (GET /stream/create)
- Open WebSocket stream (wss://demo.ironbeamapi.com/v2/stream/{streamId}?token=...)
- Subscribe to:
    * 1-minute Time Bars indicator for ES
    * Trades stream for ES
- Parse 'ti' (time bars) and 'tr' (trades) messages and write to Postgres tables.

Environment variables expected:

  IRONBEAM_USERNAME        (required)
  IRONBEAM_PASSWORD        (required – password or API key)
  IRONBEAM_TENANT_API_KEY  (optional)
  DATABASE_URL             (required – Postgres URL)
  IRONBEAM_BARS_TABLE      (optional – default "ironbeam_es_1m_bars")
  IRONBEAM_TRADES_TABLE    (optional – default "ironbeam_es_trades")
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
DEBUG_WS_RAW = False   # set True temporarily if you want to see raw WS messages

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
    print("Authenticated OK.")
    return token


def discover_es_front_month(token: str) -> str:
    """
    Use GET /info/symbol/search/futures/XCME/ES to get ES futures, then pick front month.
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
            print(f"Using ES symbol (front month): {rec['symbol']} (maturity {maturity(rec)})")
            return rec["symbol"]

    last = symbols_sorted[-1]
    print("All maturities < today, using last symbol:", last["symbol"])
    return last["symbol"]


def create_stream(token: str) -> str:
    """
    Create a streamId for WebSocket streaming.

    /stream/create is GET in v2.
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
    print("Stream created:", stream_id)
    return stream_id

# ---------------------------------------------------------------------------
# SUBSCRIPTIONS
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
    print(f"[SUBSCRIBE] TimeBars POST {url} {payload}")
    resp = requests.post(url, headers=headers, json=payload)
    print("TIME BARS SUBSCRIBE status:", resp.status_code)
    if DEBUG_HTTP:
        print("TIME BARS SUBSCRIBE body:", resp.text[:500])

    # Quirk: 400/"Can't subscribe to time bars" can still mean the stream is alive.
    if resp.status_code == 400 and "Can't subscribe to time bars" in resp.text:
        print("Got 'Can't subscribe to time bars' 400; continuing anyway (known Ironbeam quirk).")
        try:
            return resp.json()
        except Exception:
            return {}

    resp.raise_for_status()
    return resp.json()


def subscribe_trades(token: str, stream_id: str, symbol: str) -> Dict[str, Any]:
    """
    GET /market/trades/subscribe/{streamId}?symbols=XCME:ES.Z25

    This is the low-latency trades stream. Data arrives in the 'tr' field
    on the WebSocket payload.
    """
    if ":" not in symbol:
        symbol = f"XCME:{symbol}"

    url = f"{API_BASE}/market/trades/subscribe/{stream_id}"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"symbols": symbol}  # single symbol; API expects comma-separated list
    print(f"[SUBSCRIBE] Trades GET {url} params={params}")
    resp = requests.get(url, headers=headers, params=params)
    print("TRADES SUBSCRIBE status:", resp.status_code)
    if DEBUG_HTTP:
        print("TRADES SUBSCRIBE body:", resp.text[:500])

    resp.raise_for_status()
    try:
        data = resp.json()
    except Exception:
        data = {}
    return data

# ---------------------------------------------------------------------------
# PARSERS
# ---------------------------------------------------------------------------

def _to_utc_from_epoch(ts_raw: Any) -> dt.datetime | None:
    if ts_raw is None:
        return None
    try:
        ts = float(ts_raw)
    except (TypeError, ValueError):
        return None
    # Heuristic: milliseconds vs seconds
    if ts > 10_000_000_000:
        ts /= 1000.0
    try:
        return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc)
    except Exception:
        return None


def parse_time_bars_from_message(msg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse time bars from a WebSocket message. Looks for 'ti' entries.
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

        t_utc = _to_utc_from_epoch(bar.get("t"))
        if t_utc is None:
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
                "datetime": t_utc,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": v,
            }
        )

    return bars


def parse_trades_from_message(msg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse trades from a WebSocket message. Looks for 'tr' entries.

    The docs don't fully spell out the streaming trade object schema, but
    it should be consistent with /market/trades and quotes:

      symbol: "XCME:ES.U16"      -> s or symbol
      price:  1.13535            -> p or price or la (last)
      size:   1                  -> sz or size or q
      totalVolume: 1             -> tv or totalVolume
      sendTime: 1234567890       -> sendTime or t (epoch)
      tickDirection: "INVALID"   -> tickDirection
      aggressorSide: 0           -> aggressorSide
      tradeId: 2131220200101     -> tradeId
      sequenceNumber: 12132123   -> sequenceNumber
      tradeDate: "20200101"      -> tradeDate
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

        ts_utc = _to_utc_from_epoch(tr.get("sendTime") or tr.get("t"))
        if ts_utc is None:
            continue

        symbol = tr.get("s") or tr.get("symbol")
        price = tr.get("p") or tr.get("price") or tr.get("la")
        size = tr.get("sz") or tr.get("size") or tr.get("q")
        total_volume = tr.get("tv") or tr.get("totalVolume")

        try:
            price = float(price)
        except (TypeError, ValueError):
            continue

        try:
            size = int(size) if size is not None else None
        except (TypeError, ValueError):
            size = None

        try:
            total_volume = float(total_volume) if total_volume is not None else None
        except (TypeError, ValueError):
            total_volume = None

        trade_id = tr.get("tradeId")
        seq = tr.get("sequenceNumber")
        aggr = tr.get("aggressorSide")
        tick_dir = tr.get("tickDirection")
        trade_date = tr.get("tradeDate")

        trades.append(
            {
                "ts_utc": ts_utc,
                "symbol": symbol,
                "price": price,
                "size": size,
                "total_volume": total_volume,
                "trade_id": trade_id,
                "sequence_number": seq,
                "aggressor_side": aggr,
                "tick_direction": tick_dir,
                "trade_date": trade_date,
            }
        )

    return trades

# ---------------------------------------------------------------------------
# DB WRITERS
# ---------------------------------------------------------------------------

def write_bars_to_db(bars: List[Dict[str, Any]], engine):
    """
    Append time bars to Postgres using SQLAlchemy.
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
            print(f"[DB] Bars duplicates found, skipping batch of {len(df)}.")
        else:
            print(f"[DB] Bars write error: {e}")


def write_trades_to_db(trades: List[Dict[str, Any]], engine):
    """
    Append trades to Postgres.
    """
    if not trades:
        return

    df = pd.DataFrame(trades).drop_duplicates(subset=["ts_utc", "price", "size"])
    if df.empty:
        return

    try:
        with engine.begin() as connection:
            df.to_sql(DB_TRADES_TABLE, connection, if_exists="append", index=False)
        print(f"[DB] Wrote {len(df)} trades to {DB_TRADES_TABLE}.")
    except Exception as e:
        msg = str(e)
        if "violates unique constraint" in msg or "duplicate key value" in msg:
            print(f"[DB] Trades duplicates found, skipping batch of {len(df)}.")
        else:
            print(f"[DB] Trades write error: {e}")

# ---------------------------------------------------------------------------
# MAIN LOOP – RUN FOREVER, RECONNECT ON DISCONNECT
# ---------------------------------------------------------------------------

def run_worker():
    db_url = _get_db_url()
    engine = create_engine(db_url, pool_pre_ping=True)
    print(f"[INIT] Using DB tables bars='{DB_BARS_TABLE}', trades='{DB_TRADES_TABLE}'")

    # Optional: enable full websocket trace logs
    # if DEBUG_WS_RAW:
    #     websocket.enableTrace(True)

    while True:
        try:
            print("=== New auth / stream cycle starting ===")
            token = authenticate()
            es_symbol = discover_es_front_month(token)
            stream_id = create_stream(token)

            ws_url = f"{WS_BASE}/{stream_id}?token={token}"
            print("Opening WebSocket:", ws_url)

            def on_open(ws):
                print("WebSocket opened, subscribing to TimeBars and Trades...")
                try:
                    tb_info = subscribe_time_bars(token, stream_id, es_symbol)
                    print("[SUBSCRIBE] TimeBars response:", tb_info)
                except Exception as e:
                    print("[SUBSCRIBE] TimeBars error in on_open:", e)

                try:
                    tr_info = subscribe_trades(token, stream_id, es_symbol)
                    print("[SUBSCRIBE] Trades response:", tr_info)
                except Exception as e:
                    print("[SUBSCRIBE] Trades error in on_open:", e)

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

                # Time bars
                new_bars = parse_time_bars_from_message(msg)
                if new_bars:
                    print(f"[WS] Got {len(new_bars)} new time bars.")
                    write_bars_to_db(new_bars, engine)

                # Trades
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

        print("WebSocket disconnected or error; reconnecting in 5 seconds...")
        time.sleep(5)


if __name__ == "__main__":
    run_worker()
