#!/usr/bin/env python
"""
Ironbeam ES trades ingest worker for Render.

- Auth with /auth
- Discover front-month ES via /info/symbol/search/futures/XCME/ES
- Create streamId (/stream/create)
- Open WebSocket stream (wss://{demo|live}.ironbeamapi.com/v2/stream/{streamId}?token=...)
- Subscribe to trades for ES via /market/trades/subscribe/{streamId}
- Listen for 'tr' messages and write trades into Postgres table ironbeam_es_trades

Table schema (must already exist or will be created IF NOT EXISTS as):

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

Environment variables expected:

  # Ironbeam auth / environment
  IRONBEAM_ENV               (optional: "demo" or "live"; default "demo")
  IRONBEAM_USERNAME          (required)
  IRONBEAM_PASSWORD          (required – password or API key)
  IRONBEAM_TENANT_API_KEY    (optional – only if your tenant requires it)
  IRONBEAM_SYMBOL_OVERRIDE   (optional – e.g. "XCME:ES.H25"; skips front-month auto-discovery)

  # Database
  DATABASE_URL               (preferred, e.g. Render Postgres URL)
  # or CURVE_DB_URL          (fallback if DATABASE_URL not set)

  # Table
  IRONBEAM_TRADES_TABLE      (optional; default "ironbeam_es_trades")

Render worker start command:

  python scripts/ironbeam_trades_ingest.py
"""

import json
import os
import time
import traceback
import datetime as dt
from typing import Any, Dict, List, Optional

import requests
import websocket  # websocket-client
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# ---------- Config ----------

IRONBEAM_ENV = os.environ.get("IRONBEAM_ENV", "demo").lower()
if IRONBEAM_ENV not in {"demo", "live"}:
    IRONBEAM_ENV = "demo"

API_BASE = (
    "https://live.ironbeamapi.com/v2"
    if IRONBEAM_ENV == "live"
    else "https://demo.ironbeamapi.com/v2"
)

WS_BASE = (
    "wss://live.ironbeamapi.com/v2"
    if IRONBEAM_ENV == "live"
    else "wss://demo.ironbeamapi.com/v2"
)

IRONBEAM_USERNAME = os.environ.get("IRONBEAM_USERNAME")
IRONBEAM_PASSWORD = os.environ.get("IRONBEAM_PASSWORD")
IRONBEAM_TENANT_API_KEY = os.environ.get("IRONBEAM_TENANT_API_KEY")
IRONBEAM_SYMBOL_OVERRIDE = os.environ.get("IRONBEAM_SYMBOL_OVERRIDE")

DB_URL = os.environ.get("DATABASE_URL") or os.environ.get("CURVE_DB_URL")
DB_TRADES_TABLE = os.environ.get("IRONBEAM_TRADES_TABLE", "ironbeam_es_trades")

RECONNECT_DELAY_SECONDS = 5
HTTP_TIMEOUT = 10


# ---------- Helpers ----------

def _normalize_db_url(url: str) -> str:
    """
    Ensure SQLAlchemy-friendly driver. Examples:
      postgres://user:pass@host/db  -> postgresql+psycopg://user:pass@host/db
      postgresql://...              -> postgresql+psycopg://...
      postgresql+psycopg://...      -> unchanged
    """
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql://") and "+psycopg" not in url:
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def get_engine() -> Engine:
    if not DB_URL:
        raise RuntimeError("DATABASE_URL or CURVE_DB_URL env var must be set")
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


def epoch_to_utc(ts: int) -> Optional[dt.datetime]:
    """
    Convert epoch (seconds or milliseconds) to aware UTC datetime.
    """
    try:
        # crude heuristic: ms vs sec
        if ts > 10_000_000_000:  # way in the future if seconds => assume ms
            seconds = ts / 1000.0
        else:
            seconds = ts
        return dt.datetime.fromtimestamp(seconds, tz=dt.timezone.utc)
    except Exception:
        return None


# ---------- Ironbeam API helpers ----------

def auth() -> str:
    """
    Call /auth and return bearer token.
    """
    if not IRONBEAM_USERNAME or not IRONBEAM_PASSWORD:
        raise RuntimeError("IRONBEAM_USERNAME and IRONBEAM_PASSWORD must be set")

    payload: Dict[str, Any] = {
        "username": IRONBEAM_USERNAME,
        "password": IRONBEAM_PASSWORD,
    }
    if IRONBEAM_TENANT_API_KEY:
        # Some tenants require apikey, some don't. If it's wrong you'll see 400.
        payload["apikey"] = IRONBEAM_TENANT_API_KEY

    url = f"{API_BASE}/auth"
    print(f"[AUTH] POST {url}")
    resp = requests.post(url, data=payload, timeout=HTTP_TIMEOUT)
    print(f"[AUTH] Status: {resp.status_code}")
    try:
        body = resp.json()
    except Exception:
        body = resp.text
    print(f"[AUTH] Raw body: {body!r}")

    resp.raise_for_status()
    token = body.get("token")
    if not token:
        raise RuntimeError(f"No token in auth response: {body!r}")
    print("[AUTH] Got token")
    return token


def resolve_front_month_es(token: str) -> str:
    """
    Use /info/symbol/search/futures/XCME/ES to grab the front ES symbol.
    If IRONBEAM_SYMBOL_OVERRIDE is set, just return that.
    """
    if IRONBEAM_SYMBOL_OVERRIDE:
        print(f"[INFO] Using override symbol: {IRONBEAM_SYMBOL_OVERRIDE}")
        return IRONBEAM_SYMBOL_OVERRIDE

    url = f"{API_BASE}/info/symbol/search/futures/XCME/ES"
    print(f"[INFO] GET {url}")
    resp = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}"},
        timeout=HTTP_TIMEOUT,
    )
    print(f"[INFO] Status: {resp.status_code}")
    resp.raise_for_status()
    data = resp.json()
    symbols = data.get("symbols") or []
    if not symbols:
        raise RuntimeError(f"No futures symbols returned for ES: {data!r}")

    # For now, just take the first symbol. Ironbeam usually returns front-month first.
    sym = symbols[0]["symbol"]
    month = symbols[0].get("maturityMonth")
    year = symbols[0].get("maturityYear")
    desc = symbols[0].get("description")
    print(f"[INFO] Using ES symbol: {sym} ({month} {year}) - {desc}")
    return sym


def create_stream(token: str) -> str:
    """
    Call /stream/create to get a streamId.

    (GET works in practice; docs examples sometimes use POST.)
    """
    url = f"{API_BASE}/stream/create"
    print(f"[STREAM] GET {url}")
    resp = requests.get(url, headers={"Authorization": f"Bearer {token}"}, timeout=HTTP_TIMEOUT)
    print(f"[STREAM] Status: {resp.status_code}")
    try:
        body = resp.json()
    except Exception:
        body = resp.text
    print(f"[STREAM] Body: {body!r}")
    resp.raise_for_status()
    stream_id = body.get("streamId")
    if not stream_id:
        raise RuntimeError(f"No streamId in response: {body!r}")
    print(f"[STREAM] Got streamId={stream_id}")
    return stream_id


def subscribe_trades(token: str, stream_id: str, symbol: str) -> None:
    """
    Call /market/trades/subscribe/{streamId}?symbols=XCME:ES....
    """
    url = f"{API_BASE}/market/trades/subscribe/{stream_id}"
    params = {"symbols": symbol}
    headers = {"Authorization": f"Bearer {token}"}
    print(f"[SUB] GET {url} symbols={symbol}")
    resp = requests.get(url, headers=headers, params=params, timeout=HTTP_TIMEOUT)
    print(f"[SUB] Status: {resp.status_code}")
    try:
        body = resp.json()
    except Exception:
        body = resp.text
    print(f"[SUB] Body: {body!r}")
    resp.raise_for_status()
    print("[SUB] Subscribed to trades")


# ---------- DB insert ----------

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


# ---------- WebSocket loop ----------

class TradeStream:
    def __init__(self, engine: Engine):
        self.engine = engine
        self.token: Optional[str] = None
        self.symbol: Optional[str] = None
        self.stream_id: Optional[str] = None

    # WebSocket callbacks use these signatures

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

        # Ping: { "p": {"ping": "ping"} }
        if "p" in data:
            return

        # Reset / info messages
        if "r" in data:
            print(f"[WS] Reset message: {data['r']}")
            return

        # Trades payload lives under "tr"
        trades = data.get("tr")
        if not trades:
            # Uncomment for debugging other payloads:
            # print(f"[WS] Non-trades message: {data.keys()}")
            return

        rows: List[Dict[str, Any]] = []
        for t in trades:
            # Streaming may use either long field names (REST style) or shorter ones;
            # we defensively check both.
            symbol = t.get("symbol") or t.get("s") or self.symbol
            price = t.get("price") or t.get("p")
            size = t.get("size") or t.get("sz")
            total_volume = t.get("totalVolume") or t.get("tv")
            sequence_number = t.get("sequenceNumber") or t.get("sn")
            aggressor_side = t.get("aggressorSide") or t.get("as") or t.get("aggressor")
            tick_direction = t.get("tickDirection") or t.get("td")
            trade_date = t.get("tradeDate")

            trade_id = t.get("tradeId") or t.get("id")

            # Time: most reliable is sendTime; fall back to generic trade time fields.
            ts_raw = (
                t.get("sendTime")
                or t.get("tt")
                or t.get("t")
            )

            dt_utc: Optional[dt.datetime] = None
            if ts_raw is not None:
                try:
                    dt_utc = epoch_to_utc(int(ts_raw))
                except Exception:
                    dt_utc = None

            # If tradeDate not provided, derive from ts_utc for consistency with REST
            if trade_date is None and dt_utc is not None:
                trade_date = dt_utc.strftime("%Y%m%d")

            # Basic sanity: must at least have symbol + price + size
            if symbol is None or price is None or size is None:
                print("[WS] Skipping trade with missing essentials:", t)
                continue

            try:
                row = {
                    "ts_utc": dt_utc,
                    "symbol": str(symbol),
                    "price": float(price),
                    "size": int(size),
                    "total_volume": str(total_volume) if total_volume is not None else None,
                    "trade_id": str(trade_id) if trade_id is not None else None,
                    "sequence_number": str(sequence_number) if sequence_number is not None else None,
                    "aggressor_side": str(aggressor_side) if aggressor_side is not None else None,
                    "tick_direction": str(tick_direction) if tick_direction is not None else None,
                    "trade_date": str(trade_date) if trade_date is not None else None,
                }
                rows.append(row)
            except Exception:
                print("[WS] Failed to normalize trade payload; skipping:")
                print(t)
                traceback.print_exc()

        if rows:
            try:
                insert_trades(self.engine, rows)
            except Exception:
                print("[WS] DB insert failure:")
                traceback.print_exc()

    def on_error(self, ws: websocket.WebSocketApp, error: Exception) -> None:
        print(f"[WS] Error: {error}")
        traceback.print_exc()

    def on_close(self, ws: websocket.WebSocketApp, close_status_code, close_msg) -> None:
        print(f"[WS] Closed: code={close_status_code} msg={close_msg}")


def run_forever(engine: Engine) -> None:
    """
    Infinite loop:
      - auth
      - resolve symbol
      - create stream
      - open websocket
      - on drop, sleep and repeat
    """

    print(f"[INIT] IRONBEAM_ENV={IRONBEAM_ENV}, API_BASE={API_BASE}")
    print(f"[INIT] WS_BASE={WS_BASE}")
    print(f"[INIT] Trades table={DB_TRADES_TABLE}")

    stream = TradeStream(engine)

    while True:
        try:
            print("[LOOP] Authenticating...")
            stream.token = auth()

            print("[LOOP] Resolving ES symbol...")
            stream.symbol = resolve_front_month_es(stream.token)

            print("[LOOP] Creating new streamId...")
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
            ws_app.run_forever(ping_interval=20, ping_timeout=10)

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


# ---------- Entry point ----------

def main() -> None:
    engine = get_engine()
    ensure_trades_table(engine)
    run_forever(engine)


if __name__ == "__main__":
    main()
