#!/usr/bin/env python
"""
Ironbeam ES trades-only ingest worker for Render.

- Auth with /auth
- Discover front-month ES via /info/symbol/search/futures/XCME/ES
- Create streamId (/stream/create)
- Open WebSocket stream (wss://.../stream/{streamId}?token=...)
- Subscribe to ES trades
- Parse 'tr' messages and write to Postgres table ironbeam_es_trades

Table schema (existing):

  ts_utc          TIMESTAMPTZ
  symbol          TEXT
  price           DOUBLE PRECISION
  size            BIGINT
  total_volume    TEXT
  trade_id        TEXT
  sequence_number TEXT
  aggressor_side  TEXT
  tick_direction  TEXT
  trade_date      TEXT

Environment variables:

  IRONBEAM_ENV            ("live" or "demo"; default "demo")
  IRONBEAM_USERNAME       (required)
  IRONBEAM_PASSWORD       (required – password or API key)
  IRONBEAM_TENANT_API_KEY (optional – sent as 'apikey')
  DATABASE_URL            (required – Postgres URL)
  IRONBEAM_TRADES_TABLE   (optional – default "ironbeam_es_trades")
"""

import json
import datetime as dt
from typing import List, Dict, Any, Optional
import os
import time
import traceback

import requests
import websocket  # websocket-client
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# ---------------------------------------------------------------------------
# BASIC CONFIG – LIVE vs DEMO
# ---------------------------------------------------------------------------

IRONBEAM_ENV = os.environ.get("IRONBEAM_ENV", "demo").lower().strip()

if IRONBEAM_ENV == "live":
    API_BASE = "https://live.ironbeamapi.com/v2"
    WS_BASE = "wss://live.ironbeamapi.com/v2"
else:
    API_BASE = "https://demo.ironbeamapi.com/v2"
    WS_BASE = "wss://demo.ironbeamapi.com/v2"

IRONBEAM_USERNAME = os.environ.get("IRONBEAM_USERNAME")
IRONBEAM_PASSWORD = os.environ.get("IRONBEAM_PASSWORD")
IRONBEAM_TENANT_API_KEY = os.environ.get("IRONBEAM_TENANT_API_KEY", "")

DB_URL = os.environ.get("DATABASE_URL") or os.environ.get("CURVE_DB_URL")
DB_TRADES_TABLE = os.environ.get("IRONBEAM_TRADES_TABLE", "ironbeam_es_trades")

HTTP_TIMEOUT = 10
RECONNECT_DELAY_SECONDS = 5

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


def get_engine() -> Engine:
    if not DB_URL:
        raise RuntimeError("DATABASE_URL or CURVE_DB_URL is not set")
    url = _normalize_db_url(DB_URL)
    print(f"[DB] Connecting with URL={url!r}")
    engine = create_engine(url, pool_pre_ping=True, future=True)
    return engine


def ensure_trades_table(engine: Engine) -> None:
    """
    Ensure the ironbeam_es_trades table exists with the expected schema.

    NOTE: This uses CREATE TABLE IF NOT EXISTS (no DROP), so it won't
    destroy any data on restart. If you want to wipe old data once, run:
      TRUNCATE TABLE ironbeam_es_trades;
    manually in Postgres.
    """
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {DB_TRADES_TABLE} (
        ts_utc          TIMESTAMPTZ,
        symbol          TEXT,
        price           DOUBLE PRECISION,
        size            BIGINT,
        total_volume    TEXT,
        trade_id        TEXT,
        sequence_number TEXT,
        aggressor_side  TEXT,
        tick_direction  TEXT,
        trade_date      TEXT
    );
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))
    print(f"[DB] Ensured trades table exists: {DB_TRADES_TABLE}")


def insert_trades(engine: Engine, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    sql = text(
        f"""
        INSERT INTO {DB_TRADES_TABLE} (
            ts_utc,
            symbol,
            price,
            size,
            total_volume,
            trade_id,
            sequence_number,
            aggressor_side,
            tick_direction,
            trade_date
        ) VALUES (
            :ts_utc,
            :symbol,
            :price,
            :size,
            :total_volume,
            :trade_id,
            :sequence_number,
            :aggressor_side,
            :tick_direction,
            :trade_date
        )
        """
    )
    with engine.begin() as conn:
        conn.execute(sql, rows)
    print(f"[DB] Inserted {len(rows)} trades")


# ---------------------------------------------------------------------------
# AUTH / SYMBOL DISCOVERY
# ---------------------------------------------------------------------------


def auth() -> str:
    """
    POST /auth to obtain a bearer token.

    We mirror your original working ingest:

    - payload keys: username, password, apikey
    - for DEMO: send JSON
    - for LIVE: send form-encoded (data=...), which avoids the 415 on live.
    """
    if not IRONBEAM_USERNAME or not IRONBEAM_PASSWORD:
        raise RuntimeError("IRONBEAM_USERNAME and IRONBEAM_PASSWORD must be set")

    url = f"{API_BASE}/auth"
    payload: Dict[str, Any] = {
        "username": IRONBEAM_USERNAME,
        "password": IRONBEAM_PASSWORD,
    }
    if IRONBEAM_TENANT_API_KEY:
        payload["apikey"] = IRONBEAM_TENANT_API_KEY

    print(f"[AUTH] POST {url} (env={IRONBEAM_ENV}, has_apikey={bool(IRONBEAM_TENANT_API_KEY)})")

    if IRONBEAM_ENV == "live":
        # Live has been returning 415 when we send JSON; the earlier live
        # test script that hit 400 used form-encoded, so we do the same here.
        resp = requests.post(url, data=payload, timeout=HTTP_TIMEOUT)
    else:
        # Demo worker that you had working used JSON.
        resp = requests.post(url, json=payload, timeout=HTTP_TIMEOUT)

    print(f"[AUTH] Status: {resp.status_code}")
    try:
        body = resp.json()
    except Exception:
        body = resp.text
    print(f"[AUTH] Raw body: {body!r}")

    resp.raise_for_status()

    if isinstance(body, dict):
        token = body.get("token")
    else:
        token = None
    if not token:
        raise RuntimeError(f"No token in auth response: {body!r}")
    print("[AUTH] Authenticated OK.")
    return token


def discover_es_front_month(token: str) -> str:
    """
    Use GET /info/symbol/search/futures/XCME/ES to get ES futures.
    For now we just take the first symbol (Ironbeam usually returns
    front-month first).
    """
    url = f"{API_BASE}/info/symbol/search/futures/XCME/ES"
    headers = {"Authorization": f"Bearer {token}"}
    print(f"[SYMBOL] GET {url}")
    resp = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT)
    print(f"[SYMBOL] Status: {resp.status_code}")
    resp.raise_for_status()
    data = resp.json()

    symbols = data.get("symbols", [])
    if not symbols:
        raise RuntimeError(f"No ES futures returned – response: {data!r}")

    sym = symbols[0]["symbol"]
    month = symbols[0].get("maturityMonth")
    year = symbols[0].get("maturityYear")
    desc = symbols[0].get("description")
    print(f"[SYMBOL] Using ES symbol: {sym} ({month} {year}) - {desc}")
    return sym


def create_stream(token: str) -> str:
    """
    Create a streamId for WebSocket streaming.

    /stream/create is GET in v2.
    """
    url = f"{API_BASE}/stream/create"
    headers = {"Authorization": f"Bearer {token}"}
    print(f"[STREAM] GET {url}")
    resp = requests.get(url, headers=headers, timeout=HTTP_TIMEOUT)
    print(f"[STREAM] Status: {resp.status_code}")
    resp.raise_for_status()
    data = resp.json()
    stream_id = data.get("streamId")
    if not stream_id:
        raise RuntimeError(f"stream/create failed: {data!r}")
    print(f"[STREAM] Stream created: {stream_id}")
    return stream_id


def subscribe_trades(token: str, stream_id: str, symbol: str) -> None:
    """
    GET /market/trades/subscribe/{streamId}?symbols=XCME:ES...
    """
    if ":" not in symbol:
        symbol = f"XCME:{symbol}"

    url = f"{API_BASE}/market/trades/subscribe/{stream_id}"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"symbols": symbol}
    print(f"[SUBSCRIBE] Trades GET {url} params={params}")
    resp = requests.get(url, headers=headers, params=params, timeout=HTTP_TIMEOUT)
    print(f"[SUBSCRIBE] Trades status: {resp.status_code}")
    try:
        body = resp.json()
    except Exception:
        body = resp.text
    print(f"[SUBSCRIBE] Trades body: {str(body)[:300]!r}")
    resp.raise_for_status()
    print("[SUBSCRIBE] Trades subscribed.")


# ---------------------------------------------------------------------------
# PARSERS
# ---------------------------------------------------------------------------


def _to_utc_from_epoch(ts_raw: Any) -> Optional[dt.datetime]:
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


def parse_trades_from_message(msg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse trades from a WebSocket message. Looks for 'tr' entries.

    This is adapted from your original worker's parse_trades_from_message,
    with light normalization so it matches the ironbeam_es_trades schema.
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

        # Timestamp: Ironbeam uses 'st' (sendTime ms) or 'sendTime'
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

        # we'll store total_volume as text in the DB, so stringify it
        try:
            if total_volume is not None:
                total_volume = str(total_volume)
        except Exception:
            total_volume = None

        trade_id = tr.get("tradeId")
        seq = tr.get("sq") or tr.get("sequenceNumber")
        aggr = tr.get("as") or tr.get("aggressorSide")
        tick_dir = tr.get("td") or tr.get("tickDirection")
        trade_date = tr.get("tdt") or tr.get("tradeDate")

        trades.append(
            {
                "ts_utc": ts_utc,
                "symbol": str(symbol) if symbol is not None else None,
                "price": price,
                "size": size,
                "total_volume": total_volume,
                "trade_id": str(trade_id) if trade_id is not None else None,
                "sequence_number": str(seq) if seq is not None else None,
                "aggressor_side": str(aggr) if aggr is not None else None,
                "tick_direction": str(tick_dir) if tick_dir is not None else None,
                "trade_date": str(trade_date) if trade_date is not None else None,
            }
        )

    return trades


# ---------------------------------------------------------------------------
# WEBSOCKET LOOP
# ---------------------------------------------------------------------------


class TradeStream:
    def __init__(self, engine: Engine):
        self.engine = engine
        self.token: Optional[str] = None
        self.symbol: Optional[str] = None
        self.stream_id: Optional[str] = None

    def on_open(self, ws: websocket.WebSocketApp) -> None:
        print("[WS] Connection opened")
        try:
            if not self.token or not self.symbol or not self.stream_id:
                print("[WS] Missing token/symbol/stream_id on open; closing")
                ws.close()
                return
            subscribe_trades(self.token, self.stream_id, self.symbol)
        except Exception:
            print("[WS] Error during subscribe_trades:")
            traceback.print_exc()
            ws.close()

    def on_message(self, ws: websocket.WebSocketApp, message: str) -> None:
        try:
            data = json.loads(message)
        except Exception:
            print(f"[WS] Non-JSON message: {message!r}")
            return

        # Ignore ping
        if "p" in data and len(data) == 1:
            return

        trades = parse_trades_from_message(data)
        if trades:
            print(f"[WS] Got {len(trades)} trades. First: {trades[0]}")
            try:
                insert_trades(self.engine, trades)
            except Exception:
                print("[WS] DB insert failure:")
                traceback.print_exc()

    def on_error(self, ws: websocket.WebSocketApp, error: Exception) -> None:
        print(f"[WS] Error: {error}")
        traceback.print_exc()

    def on_close(self, ws: websocket.WebSocketApp, close_status_code, close_msg) -> None:
        print(f"[WS] Closed: code={close_status_code} msg={close_msg}")


def run_forever(engine: Engine) -> None:
    print(f"[INIT] IRONBEAM_ENV={IRONBEAM_ENV}, API_BASE={API_BASE}, WS_BASE={WS_BASE}")
    print(f"[INIT] Trades table={DB_TRADES_TABLE}")

    stream = TradeStream(engine)

    while True:
        try:
            print("=== New auth / stream cycle starting ===")
            stream.token = auth()
            stream.symbol = discover_es_front_month(stream.token)
            stream.stream_id = create_stream(stream.token)

            ws_url = f"{WS_BASE}/stream/{stream.stream_id}?token={stream.token}"
            print(f"[WS] Connecting to {ws_url}")

            ws_app = websocket.WebSocketApp(
                ws_url,
                on_open=stream.on_open,
                on_message=stream.on_message,
                on_error=stream.on_error,
                on_close=stream.on_close,
            )

            # Blocks until connection closes or error raises
            ws_app.run_forever(ping_interval=60, ping_timeout=10)

            print(f"[LOOP] WebSocket stopped; reconnecting in {RECONNECT_DELAY_SECONDS}s")
            time.sleep(RECONNECT_DELAY_SECONDS)

        except KeyboardInterrupt:
            print("[LOOP] KeyboardInterrupt - exiting")
            break
        except Exception:
            print("[LOOP] Unhandled exception in main loop:")
            traceback.print_exc()
            print(f"[LOOP] Sleeping {RECONNECT_DELAY_SECONDS}s before retry")
            time.sleep(RECONNECT_DELAY_SECONDS)


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------


def main() -> None:
    engine = get_engine()
    ensure_trades_table(engine)
    run_forever(engine)


if __name__ == "__main__":
    main()
