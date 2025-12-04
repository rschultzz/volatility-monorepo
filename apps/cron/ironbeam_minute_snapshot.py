#!/usr/bin/env python
"""
Ironbeam ES 1m bars snapshot -> Postgres

Intended usage:
- Run this script periodically (e.g. via Render Cron every minute or every few minutes).
- On each run it:
    * Authenticates to Ironbeam (demo or live, based on env vars)
    * Discovers the ES front-month futures symbol
    * Creates a streamId
    * Subscribes to 1-minute timeBars
    * Collects up to LOAD_SIZE bars as a snapshot
    * Normalizes timestamps to bar *start* times (Ironbeam's 't' appears to be bar end)
    * Writes any bars with datetime > max(datetime) in the DB into ironbeam_es_1m_bars

Environment variables expected:

  IRONBEAM_USERNAME         (required)
  IRONBEAM_PASSWORD         (required – password or API key)
  IRONBEAM_TENANT_API_KEY   (optional)
  IRONBEAM_API_BASE         (optional – default demo: https://demo.ironbeamapi.com/v2)
  IRONBEAM_WS_BASE          (optional – default demo: wss://demo.ironbeamapi.com/v2/stream)
  IRONBEAM_LOAD_SIZE        (optional – default "2000")
  IRONBEAM_DEBUG_HTTP       (optional – default "true")
  IRONBEAM_DEBUG_WS_RAW     (optional – default "false")

  DATABASE_URL              (required – Postgres URL)
  IRONBEAM_BARS_TABLE       (optional – default "ironbeam_es_1m_bars")
"""

import os
import json
import time
import ssl
import datetime as dt
from typing import List, Dict, Any

import requests
import websocket  # websocket-client
import pandas as pd
import certifi
from sqlalchemy import create_engine, text

# ---------------------------------------------------------------------------
# BASIC CONFIG FROM ENV
# ---------------------------------------------------------------------------

API_BASE = os.environ.get("IRONBEAM_API_BASE", "https://live.ironbeamapi.com/v2")
WS_BASE = os.environ.get("IRONBEAM_WS_BASE", "wss://live.ironbeamapi.com/v2/stream")

# API_BASE = os.environ.get("IRONBEAM_API_BASE", "https://demo.ironbeamapi.com/v2")
# WS_BASE = os.environ.get("IRONBEAM_WS_BASE", "wss://demo.ironbeamapi.com/v2/stream")

USERNAME = os.environ.get("IRONBEAM_USERNAME")
PASSWORD_OR_APIKEY = os.environ.get("IRONBEAM_PASSWORD")
TENANT_API_KEY = os.environ.get("IRONBEAM_TENANT_API_KEY", "")

DB_TABLE_NAME = os.environ.get("IRONBEAM_BARS_TABLE", "ironbeam_es_1m_bars")

LOAD_SIZE = int(os.environ.get("IRONBEAM_LOAD_SIZE", "2000"))

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
        raise RuntimeError("DATABASE_URL environment variable is not set")
    return _normalize_db_url(db_url)


def _get_engine():
    db_url = _get_db_url()
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
    print("[AUTH] Authenticated OK.")
    return token


def discover_es_front_month(token: str) -> str:
    """
    Use GET /info/symbol/search/futures/XCME/ES to get ES futures,
    then pick the nearest contract whose (year, month) >= today's (year, month).

    This keeps DEC as front month for the whole of December instead of
    rolling to MAR on Dec 1.
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
    curr_year, curr_month = today.year, today.month

    def ym(rec: Dict[str, Any]) -> tuple[int, int]:
        month_str = str(rec.get("maturityMonth", "")).upper()
        year = int(rec.get("maturityYear"))
        month = MONTH_MAP.get(month_str, 1)
        return year, month

    symbols_sorted = sorted(symbols, key=ym)

    for rec in symbols_sorted:
        y, m = ym(rec)
        if (y > curr_year) or (y == curr_year and m >= curr_month):
            print(
                f"[SYMBOL] Using ES symbol (front month): {rec['symbol']} "
                f"(maturityYear={y}, maturityMonth={rec.get('maturityMonth')})"
            )
            return rec["symbol"]

    # Fallback: last available symbol
    last = symbols_sorted[-1]
    y, m = ym(last)
    print(
        "[SYMBOL] All contract months < current month, using last symbol:",
        last["symbol"],
        f"(maturityYear={y}, maturityMonth={last.get('maturityMonth')})",
    )
    return last["symbol"]


def create_stream(token: str) -> str:
    """
    Create a streamId for WebSocket streaming.
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
    print("[STREAM] Stream created:", stream_id)
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
    print(f"[SUBSCRIBE] TimeBars POST {url} {payload}")
    resp = requests.post(url, headers=headers, json=payload)
    print("[SUBSCRIBE] TIME BARS status:", resp.status_code)
    if DEBUG_HTTP:
        print("[SUBSCRIBE] TIME BARS body:", resp.text[:300])

    # Known Ironbeam quirk: 400/"Can't subscribe to time bars" can still mean
    # the subscription is active and data will flow on the stream.
    if resp.status_code == 400 and "Can't subscribe to time bars" in resp.text:
        print("[SUBSCRIBE] Got 'Can't subscribe to time bars' 400; continuing anyway.")
        try:
            return resp.json()
        except Exception:
            return {}

    resp.raise_for_status()
    return resp.json()


def parse_time_bars_from_message(msg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Parse time bars from a WebSocket message.

    Ironbeam's 't' for timeBars appears to be the bar *end* timestamp
    (end of the 1-minute interval). For our DB we store the bar *start*
    time, so we subtract 1 minute.
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
            # Heuristic: ms vs s
            if ts > 10_000_000_000:
                ts /= 1000.0
            dt_utc_end = dt.datetime.utcfromtimestamp(ts)
        except Exception:
            continue

        # shift to bar start
        dt_utc_start = dt_utc_end - dt.timedelta(minutes=1)

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
                "datetime": dt_utc_start,
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": v,
            }
        )

    return bars


# ---------------------------------------------------------------------------
# DB WRITE
# ---------------------------------------------------------------------------

def write_snapshot_to_db(all_bars: List[Dict[str, Any]], engine):
    """
    Take the collected bars, de-duplicate, and insert any bars strictly newer
    than the latest datetime in DB_TABLE_NAME.
    """
    if not all_bars:
        print("[DB] No bars collected in this run, nothing to write.")
        return

    df = (
        pd.DataFrame(all_bars)
        .drop_duplicates(subset=["datetime"])
        .sort_values("datetime")
    )

    if df.empty:
        print("[DB] Snapshot dataframe is empty after dedup.")
        return

    print(
        f"[DB] Snapshot min datetime={df['datetime'].min()}, "
        f"max datetime={df['datetime'].max()}, rows={len(df)}"
    )

    with engine.begin() as conn:
        # Fetch the latest bar we have in the DB
        result = conn.execute(
            text(f"SELECT MAX(datetime) AS max_dt FROM {DB_TABLE_NAME}")
        )
        row = result.fetchone()
        max_dt = row.max_dt if row and row.max_dt is not None else None

        if max_dt is not None:
            print(f"[DB] Current DB max(datetime) = {max_dt}")
            df_new = df[df["datetime"] > max_dt]
        else:
            print("[DB] Table is empty; inserting all snapshot bars.")
            df_new = df

        if df_new.empty:
            print("[DB] No new bars to insert; DB is up to date for this snapshot.")
            return

        df_new.to_sql(DB_TABLE_NAME, conn, if_exists="append", index=False)
        print(f"[DB] Inserted {len(df_new)} new bars into {DB_TABLE_NAME}.")


# ---------------------------------------------------------------------------
# MAIN – SINGLE SNAPSHOT RUN
# ---------------------------------------------------------------------------

def main():
    engine = _get_engine()

    token = authenticate()
    es_symbol = discover_es_front_month(token)
    stream_id = create_stream(token)

    ws_url = f"{WS_BASE}/{stream_id}?token={token}"
    print("[WS] Opening WebSocket:", ws_url)

    all_bars: List[Dict[str, Any]] = []
    # Give the stream a little time to deliver the snapshot
    stop_ts = time.time() + 10  # ~10 seconds window

    def on_open(ws):
        print("[WS] WebSocket opened, subscribing to time bars...")
        try:
            info = subscribe_time_bars(token, stream_id, es_symbol)
            print("[WS] Subscribe response:", info)
        except Exception as e:
            print("[WS] Subscribe error in on_open:", e)

    def on_message(ws, message: str):
        nonlocal all_bars
        if DEBUG_WS_RAW:
            print("[WS RAW]", message)

        try:
            msg = json.loads(message)
        except json.JSONDecodeError:
            return

        # Ignore ping-only messages
        if list(msg.keys()) == ["p"]:
            return

        new_bars = parse_time_bars_from_message(msg)
        if new_bars:
            all_bars.extend(new_bars)
            print(f"[WS] Got {len(new_bars)} new bars (total {len(all_bars)})")

        # Either we grabbed enough bars, or time window expired: close WS.
        if time.time() > stop_ts or len(all_bars) >= LOAD_SIZE:
            print("[WS] Closing WebSocket, enough data collected for this snapshot.")
            ws.close()

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
        }
    )

    # After WebSocket closes, write snapshot to DB
    write_snapshot_to_db(all_bars, engine)


if __name__ == "__main__":
    main()
