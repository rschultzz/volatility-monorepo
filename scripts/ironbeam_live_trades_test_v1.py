#!/usr/bin/env python
"""
Minimal Ironbeam LIVE API test.

- Authenticates against https://live.ironbeamapi.com/v2/auth
- Fetches recent trades via /market/trades/{symbol}/{from}/{to}/{max}/{earlier}
- Builds simple 1-minute OHLCV bars from those trades.

Required env vars (same idea as your worker):

  IRONBEAM_USERNAME         # your Ironbeam username / account id
  IRONBEAM_PASSWORD         # password (or API password)
  IRONBEAM_TENANT_API_KEY   # tenant / API key (or use IRONBEAM_API_KEY)
  IRONBEAM_SYMBOL           # e.g. "XCME:ES.H25"  (set to your current ES front month)

Run with:  python ironbeam_live_trades_test.py
"""

import os
import time
import datetime as dt

import requests
import pandas as pd


BASE_URL = "https://live.ironbeamapi.com/v2"
AUTH_URL = f"{BASE_URL}/auth"
TRADES_URL_TEMPLATE = (
    BASE_URL + "/market/trades/{symbol}/{from_ms}/{to_ms}/{max_records}/{earlier}"
)


def main() -> None:
    username = "23233577"
    password = "f20ae06eb7184cd8999321af363024ab"
    api_key = "f20ae06eb7184cd8999321af363024ab"
    symbol = os.getenv("IRONBEAM_SYMBOL", "XCME:ES.H25")  # change to your live symbol

    if not (username and password and api_key):
        raise SystemExit(
            "Set IRONBEAM_USERNAME, IRONBEAM_PASSWORD and "
            "IRONBEAM_TENANT_API_KEY (or IRONBEAM_API_KEY) in your environment."
        )

    print(f"Using symbol: {symbol}")

    # ---- 1) AUTH ----
    auth_payload = {
        "username": username,
        "password": password,
        "apikey": api_key,
    }

    print(f"\nPOST {AUTH_URL}")
    auth_resp = requests.post(AUTH_URL, json=auth_payload, timeout=10)
    print("AUTH status:", auth_resp.status_code)
    print("AUTH raw body:", auth_resp.text)

    auth_resp.raise_for_status()
    token = auth_resp.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}

    # ---- 2) GET RECENT TRADES ----
    # last 5 minutes
    now_ms = int(time.time() * 1000)
    from_ms = now_ms - 5 * 60 * 1000
    to_ms = now_ms

    trades_url = TRADES_URL_TEMPLATE.format(
        symbol=symbol,
        from_ms=from_ms,
        to_ms=to_ms,
        max_records=100,   # up to 100 trades
        earlier="false",   # path param is literally 'true' or 'false'
    )

    print(f"\nGET {trades_url}")
    trades_resp = requests.get(trades_url, headers=headers, timeout=10)
    print("TRADES status:", trades_resp.status_code)
    print("TRADES raw body:", trades_resp.text)

    trades_resp.raise_for_status()
    data = trades_resp.json()

    # The docs use the key "traders" in the sample; be defensive:
    rows = data.get("traders") or data.get("Trades") or data.get("trades") or []
    if not rows:
        print("\nNo trades returned in the last 5 minutes.")
        return

    df_trades = pd.DataFrame(rows)
    print("\nRaw trades (head):")
    print(df_trades.head())

    # ---- 3) BUILD 1-MINUTE BARS ----
    if "sendTime" not in df_trades.columns:
        print("\nNo 'sendTime' column in response; cannot build bars.")
        return

    # sendTime in docs looks like epoch seconds; if timestamps look crazy,
    # change unit='ms' to unit='s' or vice versa.
    df_trades["ts"] = pd.to_datetime(df_trades["sendTime"], unit="s", utc=True)
    df_trades = df_trades.set_index("ts")

    bars = (
        df_trades.resample("1T")
        .agg(
            open=("price", "first"),
            high=("price", "max"),
            low=("price", "min"),
            close=("price", "last"),
            volume=("size", "sum"),
        )
        .dropna(how="all")
    )

    print("\n1-minute bars:")
    print(bars.tail())


if __name__ == "__main__":
    main()

