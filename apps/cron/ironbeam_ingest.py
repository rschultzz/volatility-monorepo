#!/usr/bin/env python
"""
Ironbeam ES 1m bars + trades ingest worker for Render.

- Auth with /auth
- Discover front-month ES via /info/symbol/search/futures/XCME/ES
- Create streamId (GET /stream/create)
- Open WebSocket stream (wss://.../stream/{streamId}?token=...)
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
  IRONBEAM_API_BASE        (optional – defaults to demo: https://demo.ironbeamapi.com/v2)
  IRONBEAM_WS_BASE         (optional – defaults to demo: wss://demo.ironbeamapi.com/v2/stream)
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
from sqlalchemy import create_engine, text

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

# Debug toggles (can override via env if you want)
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
        print("AUTH body:", resp.text[:300])
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
        print("SYMBOL FUTURES body:", resp.text[:300])
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
        print("STREAM CREATE body:", resp.text[:300])
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
        print("TIME BARS SUBSCRIBE body:", resp.text[:300])

    # Known quirk: 400/"Can't subscribe to time bars" but data still flows.
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
    """
    if ":" not in symbol:
        symbol = f"XCME:{symbol}"

    url = f"{API_BASE}/market/trades/subscribe/{stream_id}"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"symbols": symbol}
    print(f"[SUBSCRIBE] Trades GET {url} params={params}")
    resp = requests.get(url, headers=headers, params=params)
    print("TRADES SUBSCRIBE status:", resp.status_code)
    if DEBUG_HTTP:
        print("TRADES SUBSCRIBE body:", resp.text[:300])

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

    NOTE: Ironbeam's 't' for timeBars appears to be the bar *end* timestamp
    (end of the 1-minute interval). For plotting and alignment with
    TradingView we want the bar *start* time, so we subtract 1 minute.
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

        t_utc_end = _to_utc_from_epoch(bar.get("t"))
        if t_utc_end is None:
            continue

        # Store bar START time (1 minute earlier) so that the first RTH bar
        # shows up at 15:00 PT instead of 15:01, etc.
        t_utc_start = t_utc_end - dt.timedelta(minutes=1)

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
                "datetime": t_utc_start,
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

        # Timestamp: Ironbeam uses 'st' (sendTime ms)
        ts_raw = tr.get("st") or tr.get("sendTime") or tr.get("t")
        ts_utc = _to_utc_from_epoch(ts_raw)
        if ts_utc is None:
            ts_utc = dt.datetime.now(dt.timezone.utc)

        symbol = tr.get("s") or tr.get("symbol")
        price = tr.get("p") or tr.get("price") or tr.get("la") or tr.get("l")
        size = tr.get("sz") or tr.get("size") or tr.get("q") or tr.get("qty")
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

        trade_id = tr.get("tradeId")  # not present in sample, so stays None
        seq = tr.get("sq") or tr.get("sequenceNumber")
        aggr = tr.get("as") or tr.get("aggressorSide")
        tick_dir = tr.get("td") or tr.get("tickDirection")
        trade_date = tr.get("tdt") or tr.get("tradeDate")

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
# DB WRITERS (with cross-check against existing rows)
# ---------------------------------------------------------------------------

def write_bars_to_db(bars: List[Dict[str, Any]], engine):
    """
    Append time bars to Postgres, skipping any datetimes already present.

    This is defensive against:
      - snapshot bars on initial subscribe
      - any replayed bars after reconnect
    """
    if not bars:
        return

    df = pd.DataFrame(bars)
    if df.empty or "datetime" not in df.columns:
        return

    # Normalize datetimes to timezone-aware UTC
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)

    # Drop dupes inside this batch
    df = df.drop_duplicates(subset=["datetime"])
    if df.empty:
        return

    min_dt = df["datetime"].min()
    max_dt = df["datetime"].max()

    try:
        with engine.begin() as connection:
            # Check what we already have for this time range
            existing = pd.read_sql(
                text(f"""
                    SELECT datetime
                    FROM {DB_BARS_TABLE}
                    WHERE datetime >= :min_dt AND datetime <= :max_dt
                """),
                connection,
                params={"min_dt": min_dt, "max_dt": max_dt},
                parse_dates=["datetime"],
            )

            if not existing.empty:
                existing["datetime"] = pd.to_datetime(existing["datetime"], utc=True)
                existing_set = set(existing["datetime"].tolist())
                before = len(df)
                df = df[~df["datetime"].isin(existing_set)]
                skipped = before - len(df)
                if skipped:
                    print(f"[DB] Skipping {skipped} existing bars out of {before} in {DB_BARS_TABLE}.")

            if df.empty:
                print("[DB] No new bars to insert (all already present in DB).")
                return

            df.to_sql(DB_BARS_TABLE, connection, if_exists="append", index=False)
            print(f"[DB] Wrote {len(df)} new bars to {DB_BARS_TABLE}.")

    except Exception as e:
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
    print(f"[INIT] API_BASE={API_BASE}, WS_BASE={WS_BASE}")

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
                    print("[WS] bad JSON message")
                    return

                # Ignore pure ping messages (Ironbeam sends "p" occasionally)
                if list(msg.keys()) == ["p"]:
                    return

                # Time bars
                new_bars = parse_time_bars_from_message(msg)
                if new_bars:
                    print(f"[WS] Got {len(new_bars)} time bars.")
                    write_bars_to_db(new_bars, engine)

                # Trades
                new_trades = parse_trades_from_message(msg)
                if new_trades:
                    # Log just the first one for sanity
                    print(f"[WS] Got {len(new_trades)} trades. First trade: {new_trades[0]}")
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
