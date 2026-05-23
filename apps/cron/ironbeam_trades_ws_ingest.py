#!/usr/bin/env python3
"""
Ironbeam trades -> Postgres via WebSocket stream (live or demo).

Flow:
- POST /v2/auth -> token
- GET  /v2/stream/create -> streamId
- WS   wss://.../v2/stream/{streamId}?token={token}
- GET  /v2/market/trades/subscribe/{streamId}?symbols=...

Env:
  DATABASE_URL

  IRONBEAM_ENV=live|demo          (default live)
  IRONBEAM_SYMBOL=XCME:ES.H26     (default shown)

  # Preferred (your names):
  IRONBEAM_LIVE_USERNAME
  IRONBEAM_LIVE_PASSWORD
  IRONBEAM_LIVE_API_KEY

  # Fallbacks (if you prefer):
  IRONBEAM_USERNAME
  IRONBEAM_PASSWORD
  IRONBEAM_API_KEY  (or IRONBEAM_TENANT_API_KEY)

Tuning:
  IRONBEAM_TRADES_FLUSH_SEC=0.5
  IRONBEAM_TRADES_MAX_BUFFER=5000
"""

import json
import os
import time
import threading
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import websocket
from sqlalchemy import create_engine, text

import ssl
import certifi



# ----------------------- Load .env automatically -----------------------
try:
    from dotenv import load_dotenv

    def _find_dotenv(start: Path) -> Path | None:
        for p in [start] + list(start.parents):
            cand = p / ".env"
            if cand.exists():
                return cand
        return None

    here = Path(__file__).resolve()
    dotenv_path = _find_dotenv(here.parent)

    if dotenv_path:
        load_dotenv(dotenv_path, override=False)
    # also try cwd as a fallback
    load_dotenv(Path.cwd() / ".env", override=False)

except Exception:
    pass



# ----------------------- Helpers -----------------------
def _env_first(*names: str) -> Optional[str]:
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return None


def _sqlalchemy_url(raw: str) -> str:
    if raw.startswith("postgres://"):
        return raw.replace("postgres://", "postgresql+psycopg://", 1)
    if raw.startswith("postgresql://"):
        return raw.replace("postgresql://", "postgresql+psycopg://", 1)
    return raw



def to_ts_utc(send_time: Any) -> Optional[dt.datetime]:
    if send_time is None:
        return None
    try:
        x = int(send_time)
    except Exception:
        return None

    # Ironbeam sendTime in your REST sample is epoch milliseconds (≈1.7e12)
    if x > 10_000_000_000:  # treat as ms
        return dt.datetime.fromtimestamp(x / 1000.0, tz=dt.timezone.utc)
    return dt.datetime.fromtimestamp(x, tz=dt.timezone.utc)


# ----------------------- Config -----------------------
ENV = os.getenv("IRONBEAM_ENV", "live").lower().strip()
BASE_HTTP = "https://live.ironbeamapi.com/v2" if ENV == "live" else "https://demo.ironbeamapi.com/v2"
BASE_WSS = "wss://live.ironbeamapi.com/v2" if ENV == "live" else "wss://demo.ironbeamapi.com/v2"

AUTH_URL = f"{BASE_HTTP}/auth"
STREAM_CREATE_URL = f"{BASE_HTTP}/stream/create"
SUB_TRADES_URL_TMPL = f"{BASE_HTTP}/market/trades/subscribe" + "/{streamId}"

SYMBOL = os.getenv("IRONBEAM_SYMBOL", "XCME:ES.H26")

DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise SystemExit(
        "Missing DATABASE_URL. If it's in .env, install python-dotenv OR `source .env` before running."
    )
DB_URL = _sqlalchemy_url(DB_URL)

engine = create_engine(DB_URL, pool_pre_ping=True, future=True)

FLUSH_EVERY_SEC = float(os.getenv("IRONBEAM_TRADES_FLUSH_SEC", "0.5"))
MAX_BUFFER = int(os.getenv("IRONBEAM_TRADES_MAX_BUFFER", "5000"))

INSERT_SQL = text(
    """
    INSERT INTO ironbeam_es_trades
      (trade_key, ts_utc, symbol, price, size,
       total_volume, trade_id, sequence_number,
       aggressor_side, tick_direction, trade_date)
    VALUES
      (:trade_key, :ts_utc, :symbol, :price, :size,
       :total_volume, :trade_id, :sequence_number,
       :aggressor_side, :tick_direction, :trade_date)
    ON CONFLICT (trade_key) DO NOTHING
    """
)


def get_creds() -> Dict[str, str]:
    """
    Ironbeam auth wants username + password + apikey.
    Use your IRONBEAM_LIVE_* names first, then fall back.
    """
    username = _env_first("IRONBEAM_LIVE_USERNAME", "IRONBEAM_USERNAME")
    password = _env_first("IRONBEAM_LIVE_PASSWORD", "IRONBEAM_PASSWORD")
    api_key = _env_first("IRONBEAM_LIVE_API_KEY", "IRONBEAM_TENANT_API_KEY", "IRONBEAM_API_KEY")

    if not username or not password or not api_key:
        raise SystemExit(
            "Missing Ironbeam creds. Need:\n"
            "  IRONBEAM_LIVE_USERNAME\n"
            "  IRONBEAM_LIVE_PASSWORD\n"
            "  IRONBEAM_LIVE_API_KEY\n"
            "(or the IRONBEAM_USERNAME / IRONBEAM_PASSWORD / IRONBEAM_API_KEY fallbacks)"
        )

    return {"username": username, "password": password, "apikey": api_key}


def auth_token(session: requests.Session) -> str:
    r = session.post(AUTH_URL, json=get_creds(), timeout=10)
    r.raise_for_status()
    j = r.json()
    token = j.get("token")
    if not token:
        raise RuntimeError(f"Auth response missing token: {j}")
    return token


def create_stream(session: requests.Session, token: str) -> str:
    r = session.get(STREAM_CREATE_URL, headers={"Authorization": f"Bearer {token}"}, timeout=10)
    r.raise_for_status()
    j = r.json()
    stream_id = j.get("streamId") or j.get("streamid") or j.get("id")
    if not stream_id:
        raise RuntimeError(f"Stream create response missing streamId: {j}")
    return stream_id


def subscribe_trades(session: requests.Session, token: str, stream_id: str, symbols: List[str]) -> None:
    url = SUB_TRADES_URL_TMPL.format(streamId=stream_id)
    params = {"symbols": ",".join(symbols)}
    r = session.get(url, headers={"Authorization": f"Bearer {token}"}, params=params, timeout=30)

    if r.status_code == 401:
        raise PermissionError("subscribe_trades: 401 Unauthorized (token expired?)")

    r.raise_for_status()



def _to_int(v: Any) -> Optional[int]:
    try:
        return int(v)
    except Exception:
        return None


def normalize_trade(rec: Dict[str, Any], symbol_default: str) -> Optional[Dict[str, Any]]:
    # Compact WS fields:
    # s=symbol, p=price, sz=size, st=sendTime(ms), sq=sequence, td=tickDir(enum), as=aggr(enum), tdt=YYYYMMDD, ch=change
    symbol = rec.get("symbol") or rec.get("s") or symbol_default
    price = rec.get("price") if rec.get("price") is not None else rec.get("p")
    size  = rec.get("size")  if rec.get("size")  is not None else rec.get("sz")

    # timestamp (ms) - WS uses "st"
    send_time = (
        rec.get("sendTime")
        if rec.get("sendTime") is not None
        else rec.get("t")
        if rec.get("t") is not None
        else rec.get("st")
    )
    ts_utc = to_ts_utc(send_time)

    if ts_utc is None or price is None or size is None:
        return None

    seq = rec.get("sequenceNumber") if rec.get("sequenceNumber") is not None else rec.get("sq")
    trade_date = rec.get("tradeDate") if rec.get("tradeDate") is not None else rec.get("tdt")
    change = rec.get("change") if rec.get("change") is not None else rec.get("ch")

    # enums -> human strings
    td = rec.get("tickDirection") if rec.get("tickDirection") is not None else rec.get("td")
    tick_dir_map = {1: "PLUS", 2: "MINUS", 3: "SAME"}
    tick_direction = tick_dir_map.get(int(td), str(td)) if td is not None else None

    ag = rec.get("aggressorSide") if rec.get("aggressorSide") is not None else rec.get("as")
    aggr_map = {1: "BUY", 2: "SELL"}
    aggressor_side = aggr_map.get(int(ag), str(ag)) if ag is not None else None

    seq_i = _to_int(seq)

    # stable unique key for dedupe
    if trade_date is not None and seq_i is not None:
        trade_key = f"{symbol}|{trade_date}|{seq_i}"
    else:
        trade_key = f"{symbol}|{int(send_time)}|{float(price)}|{int(size)}"

    return {
        "trade_key": trade_key,
        "ts_utc": ts_utc,
        "symbol": str(symbol),
        "price": float(price),
        "size": int(size),

        # store sequence in BOTH fields if you want; sequence is the useful dedupe key here
        "total_volume": seq_i,
        "trade_id": None,
        "sequence_number": seq_i,

        "aggressor_side": aggressor_side,
        "tick_direction": tick_direction,
        "trade_date": None if trade_date is None else str(trade_date),
        "change": float(change) if change is not None else None,
    }



# ----------------------- Stream ingestor -----------------------
class TradeIngestor:
    def __init__(self, symbols: List[str]) -> None:
        self.symbols = symbols
        self.session = requests.Session()
        self.token = ""
        self.stream_id = ""

        self.buffer: List[Dict[str, Any]] = []
        self.lock = threading.Lock()
        self.last_flush = time.time()

        # --- debug counters ---
        self.msg_count = 0
        self.ping_count = 0
        self.trade_msg_count = 0
        self.last_dbg = 0.0


    def flush(self, force: bool = False) -> None:
        now = time.time()
        if not force and (now - self.last_flush) < FLUSH_EVERY_SEC and len(self.buffer) < 200:
            return

        with self.lock:
            if not self.buffer:
                return
            rows = self.buffer
            self.buffer = []
            self.last_flush = now

        try:
            with engine.begin() as conn:
                conn.execute(INSERT_SQL, rows)
        except Exception as e:
            print(f"[db] insert failed ({len(rows)} rows): {e}")
            # best-effort requeue
            with self.lock:
                self.buffer = rows + self.buffer

    def on_message(self, ws: websocket.WebSocketApp, message: str) -> None:
        self.msg_count += 1

        try:
            payload = json.loads(message)
        except Exception:
            # If you ever hit this, print the first bytes to see if it's not JSON
            if time.time() - self.last_dbg > 5:
                self.last_dbg = time.time()
                print("[ws] non-json frame sample:", message[:120])
            return

        # 1) Ping messages (JSON p) every ~5 seconds means stream is alive
        if "p" in payload:
            self.ping_count += 1
            if self.ping_count % 12 == 0:  # about once/minute
                print(f"[ws] ping ok (pings={self.ping_count}, msgs={self.msg_count})")
            return

        # 2) Reset/info messages
        if "r" in payload:
            print("[ws] reset/info:", payload["r"])

        # 3) Trades
        trades = payload.get("tr")
        if trades is None:
            # occasionally print keys so we can see what the stream is sending
            if time.time() - self.last_dbg > 10:
                self.last_dbg = time.time()
                print("[ws] msg keys:", list(payload.keys()))
            return

        self.trade_msg_count += 1

        if isinstance(trades, dict):
            trades = [trades]
        if not isinstance(trades, list):
            print("[ws] unexpected tr type:", type(trades))
            return

        new_rows = []
        for rec in trades:
            if isinstance(rec, dict):
                row = normalize_trade(rec, symbol_default=self.symbols[0])
                if row:
                    new_rows.append(row)

        if not new_rows:
            print("[ws] got tr but nothing normalized; sample:", trades[0] if trades else None)
            return

        # show a quick sample so you know you’re getting prints
        first = new_rows[0]
        print(f"[ws] trades n={len(new_rows)} first ts={first['ts_utc']} px={first['price']} sz={first['size']}")

        with self.lock:
            self.buffer.extend(new_rows)
            if len(self.buffer) > MAX_BUFFER:
                self.buffer = self.buffer[-MAX_BUFFER:]

        self.flush(force=False)

    def on_open(self, ws: websocket.WebSocketApp) -> None:
        print(f"[ws] opened stream_id={self.stream_id} subscribing trades for {self.symbols} ...")

        def _sub():
            try:
                subscribe_trades(self.session, self.token, self.stream_id, self.symbols)
                print("[ws] subscribed OK")
            except PermissionError as e:
                print(f"[ws] subscribe unauthorized: {e} -> closing socket to re-auth")
                try:
                    ws.close()
                except Exception:
                    pass
            except Exception as e:
                print(f"[ws] subscribe failed: {e}")

        threading.Thread(target=_sub, daemon=True).start()

    def on_close(self, ws: websocket.WebSocketApp, code: int, msg: str) -> None:
        print(f"[ws] closed code={code} msg={msg}")
        self.flush(force=True)

    def on_error(self, ws: websocket.WebSocketApp, error: Any) -> None:
        print(f"[ws] error: {error}")

    def run_forever(self) -> None:
        backoff = 2.0  # seconds
        while True:
            try:
                self.token = auth_token(self.session)
                self.stream_id = create_stream(self.session, self.token)
                wss_url = f"{BASE_WSS}/stream/{self.stream_id}?token={self.token}"

                print(f"[auth] OK env={ENV} symbols={self.symbols} stream_id={self.stream_id}")
                print(f"[ws] connecting {BASE_WSS}/stream/{self.stream_id}?token=<redacted>")

                ws = websocket.WebSocketApp(
                    wss_url,
                    on_open=self.on_open,
                    on_message=self.on_message,
                    on_error=self.on_error,
                    on_close=self.on_close,
                )

                # If we got here, things are healthy; reset backoff
                backoff = 2.0

                ws.run_forever(
                    ping_interval=20,
                    ping_timeout=10,
                    sslopt={
                        "cert_reqs": ssl.CERT_REQUIRED,
                        "ca_certs": certifi.where(),
                    },
                )

            except Exception as e:
                print(f"[loop] exception: {e}")

            # flush any buffered rows before retrying
            self.flush(force=True)

            # exponential backoff with cap
            print(f"[loop] reconnecting in {backoff:.1f}s ...")
            time.sleep(backoff)
            backoff = min(60.0, backoff * 2.0)


if __name__ == "__main__":
    ing = TradeIngestor(symbols=[SYMBOL])
    ing.run_forever()
