#!/usr/bin/env python
"""
Ironbeam LIVE trades test (fixed sendTime units).

- Auths to /v2/auth
- Fetches recent trades via /market/trades/{symbol}/{from}/{to}/{max}/{earlier}
- Builds 1-minute OHLCV bars from those trades

Env:
  IRONBEAM_USERNAME
  IRONBEAM_PASSWORD
  IRONBEAM_TENANT_API_KEY   (or IRONBEAM_API_KEY)
  IRONBEAM_SYMBOL           e.g. "XCME:ES.H26"
  LOOKBACK_MINUTES          default 5
  MAX_RECORDS               default 500
"""

import os
import time
import requests
import pandas as pd

BASE_URL = "https://live.ironbeamapi.com/v2"
AUTH_URL = f"{BASE_URL}/auth"
TRADES_URL_TEMPLATE = BASE_URL + "/market/trades/{symbol}/{from_ms}/{to_ms}/{max_records}/{earlier}"


def _get_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise SystemExit(f"Missing env var: {name}")
    return v


def main() -> None:
    # IMPORTANT: don't hardcode creds in the file
    username = _get_env("IRONBEAM_USERNAME")
    password = _get_env("IRONBEAM_PASSWORD")
    api_key = os.getenv("IRONBEAM_TENANT_API_KEY") or os.getenv("IRONBEAM_API_KEY")
    if not api_key:
        raise SystemExit("Missing env var: IRONBEAM_TENANT_API_KEY (or IRONBEAM_API_KEY)")

    symbol = os.getenv("IRONBEAM_SYMBOL", "XCME:ES.H26")
    lookback_minutes = int(os.getenv("LOOKBACK_MINUTES", "5"))
    max_records = int(os.getenv("MAX_RECORDS", "500"))

    print(f"Using symbol: {symbol}")

    # ---- AUTH ----
    auth_payload = {"username": username, "password": password, "apikey": api_key}
    print(f"\nPOST {AUTH_URL}")
    r = requests.post(AUTH_URL, json=auth_payload, timeout=10)
    print("AUTH status:", r.status_code)
    print("AUTH raw body:", r.text)
    r.raise_for_status()
    token = r.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}

    # ---- TRADES ----
    now_ms = int(time.time() * 1000)
    from_ms = now_ms - lookback_minutes * 60 * 1000

    url = TRADES_URL_TEMPLATE.format(
        symbol=symbol,
        from_ms=from_ms,
        to_ms=now_ms,
        max_records=max_records,
        earlier="false",
    )
    print(f"\nGET {url}")
    t = requests.get(url, headers=headers, timeout=10)
    print("TRADES status:", t.status_code)
    print("TRADES raw body (first 500 chars):", t.text[:500])
    t.raise_for_status()

    data = t.json()
    rows = data.get("traders") or data.get("Trades") or data.get("trades") or []
    if not rows:
        print("\nNo trades returned.")
        return

    df = pd.DataFrame(rows)
    print("\nRaw trades (head):")
    print(df.head())

    # ---- FIX: sendTime is epoch MILLISECONDS ----
    st = pd.to_numeric(df.get("sendTime"), errors="coerce")
    if st.isna().all():
        raise SystemExit("No usable sendTime values in response")

    # Auto-detect (but for your sample, this will choose 'ms')
    unit = "ms" if st.median() > 1e11 else "s"
    df["ts_utc"] = pd.to_datetime(st, unit=unit, utc=True)
    df = df.sort_values(["ts_utc", "totalVolume"], na_position="last")

    # ---- 1-min bars from trades ----
    df = df.set_index("ts_utc")
    bars = (
        df.resample("1min")
        .agg(
            open=("price", "first"),
            high=("price", "max"),
            low=("price", "min"),
            close=("price", "last"),
            volume=("size", "sum"),
            trades=("price", "count"),
        )
        .dropna(subset=["open", "high", "low", "close"], how="any")
    )

    print("\n1-minute bars (tail):")
    print(bars.tail(10))


if __name__ == "__main__":
    main()
