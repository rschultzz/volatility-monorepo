#!/usr/bin/env python3
"""
Ironbeam LIVE trades POC (REST)

- Loads .env
- Authenticates (Bearer token) via POST /v2/auth
- Finds a front-month ES future via GET /v2/info/symbol/search/futures/XCME/ES
- Converts it to exchange-qualified symbol "XCME:ES.Z25"
- Fetches a few recent market trades via GET /v2/market/trades/{symbol}/{from}/{to}/{max}/{earlier}
- Writes trades to a CSV

Required .env:
  IRONBEAM_LIVE_USERNAME=...
  IRONBEAM_LIVE_API_KEY=...

Optional (some account types):
  IRONBEAM_LIVE_PASSWORD=...
"""

from __future__ import annotations

import csv
import os
import sys
import time
import json
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests
from dotenv import load_dotenv


LIVE_BASE = "https://live.ironbeamapi.com/v2"

# for this POC we hardcode ES on XCME; easy to parameterize later
FUT_EXCHANGE = "XCME"
FUT_ROOT = "ES"


@dataclass
class IronbeamCreds:
    username: str
    api_key: str
    password: Optional[str] = None


def _now_ms() -> int:
    return int(time.time() * 1000)


def _iso_utc(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat()


def load_creds() -> IronbeamCreds:
    load_dotenv()
    username = (os.getenv("IRONBEAM_LIVE_USERNAME") or "").strip()
    api_key = (os.getenv("IRONBEAM_LIVE_API_KEY") or "").strip()
    password = (os.getenv("IRONBEAM_LIVE_PASSWORD") or "").strip() or None

    if not username or not api_key:
        raise SystemExit(
            "Missing env vars. Need IRONBEAM_LIVE_USERNAME and IRONBEAM_LIVE_API_KEY in your .env"
        )
    return IronbeamCreds(username=username, api_key=api_key, password=password)


def auth_token(session: requests.Session, creds: IronbeamCreds) -> str:
    url = f"{LIVE_BASE}/auth"
    headers = {"Accept": "application/json", "Content-Type": "application/json"}

    payloads: List[Dict[str, Any]] = []
    if creds.password:
        payloads.append({"username": creds.username, "password": creds.password, "apikey": creds.api_key})
    payloads.append({"username": creds.username, "password": creds.api_key})
    payloads.append({"username": creds.username, "password": creds.api_key, "apikey": creds.api_key})

    last_err = None
    for i, payload in enumerate(payloads, start=1):
        resp = session.post(url, headers=headers, json=payload, timeout=20)
        if resp.ok:
            data = resp.json()
            token = data.get("token")
            if not token:
                raise RuntimeError(f"Auth succeeded but no 'token' in response: {data}")
            return token

        try:
            body = resp.json()
        except Exception:
            body = resp.text
        last_err = f"[auth try {i}] HTTP {resp.status_code}: {body}"

    raise RuntimeError(f"Unable to authenticate. Last error: {last_err}")


def get_front_month_es_exch_symbol(session: requests.Session, token: str) -> str:
    """
    Symbol search returns things like "ES.Z25".
    Market-data endpoints (including trades) expect "XCME:ES.Z25". :contentReference[oaicite:1]{index=1}
    """
    url = f"{LIVE_BASE}/info/symbol/search/futures/{FUT_EXCHANGE}/{FUT_ROOT}"
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}

    resp = session.get(url, headers=headers, timeout=20)
    resp.raise_for_status()
    data = resp.json()

    symbols = data.get("symbols") or []
    if not symbols:
        raise RuntimeError(f"No symbols returned from {url}. Response: {data}")

    raw = symbols[0].get("symbol")
    if not raw:
        raise RuntimeError(f"Unexpected symbol search response: {data}")

    # Ensure exchange-qualified "XCME:..."
    if ":" in raw:
        return raw
    return f"{FUT_EXCHANGE}:{raw}"


def get_recent_trades(
    session: requests.Session,
    token: str,
    exch_symbol: str,
    max_records: int = 25,
    lookback_hours: int = 72,
    earlier: bool = False,
) -> List[Dict[str, Any]]:
    to_ms = _now_ms()
    from_ms = int((datetime.now(timezone.utc) - timedelta(hours=lookback_hours)).timestamp() * 1000)

    # Keep ":" and "." unescaped; Ironbeam examples show ":" in symbol. :contentReference[oaicite:2]{index=2}
    sym_enc = quote(exch_symbol, safe=":.")  # encode anything weird, but preserve : and .
    url = f"{LIVE_BASE}/market/trades/{sym_enc}/{from_ms}/{to_ms}/{max_records}/{str(earlier).lower()}"

    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    resp = session.get(url, headers=headers, timeout=20)

    if not resp.ok:
        # Print body for fast debugging
        try:
            body = resp.json()
        except Exception:
            body = resp.text
        raise RuntimeError(f"Trades request failed HTTP {resp.status_code} for {url}\nBody: {body}")

    data = resp.json()

    # Docs sample uses "traders" key (likely a naming quirk). :contentReference[oaicite:3]{index=3}
    for key in ("trades", "traders", "Trades", "trade", "tr"):
        v = data.get(key)
        if isinstance(v, list):
            return v

    # Fallback: find any list-of-dicts field
    for _, v in data.items():
        if isinstance(v, list) and (not v or isinstance(v[0], dict)):
            return v

    return []


def write_trades_csv(trades: List[Dict[str, Any]], out_path: str) -> None:
    if not trades:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            f.write("no_trades_returned\n")
        return

    fieldnames = sorted({k for t in trades for k in t.keys()})
    if "fetchedAtUtc" not in fieldnames:
        fieldnames.append("fetchedAtUtc")
    if "sendTimeIsoUtc" not in fieldnames:
        fieldnames.append("sendTimeIsoUtc")

    fetched_at = datetime.now(timezone.utc).isoformat()

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for t in trades:
            row = dict(t)
            row["fetchedAtUtc"] = fetched_at
            st = row.get("sendTime")
            if isinstance(st, (int, float)):
                st_int = int(st)
                if st_int > 10_000_000_000:  # ms
                    row["sendTimeIsoUtc"] = _iso_utc(st_int)
                elif st_int > 1_000_000_000:  # seconds
                    row["sendTimeIsoUtc"] = datetime.fromtimestamp(st_int, tz=timezone.utc).isoformat()
            w.writerow(row)


def main() -> None:
    creds = load_creds()
    session = requests.Session()

    print("Authenticating to Ironbeam LIVE…")
    token = auth_token(session, creds)
    print("✅ Got Bearer token")

    print("Finding front-month ES symbol…")
    exch_symbol = get_front_month_es_exch_symbol(session, token)
    print(f"✅ Using exch symbol: {exch_symbol}")

    print("Fetching recent market trades (POC)…")
    trades = get_recent_trades(session, token, exch_symbol=exch_symbol, max_records=25, lookback_hours=72, earlier=False)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"ironbeam_trades_{exch_symbol.replace(':','-')}_{ts}.csv"

    write_trades_csv(trades, out_path)

    print(f"✅ Wrote {len(trades)} trades to: {out_path}")
    if trades:
        print("Sample trade (first row):")
        print(json.dumps(trades[0], indent=2)[:2000])


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
        sys.exit(0)
