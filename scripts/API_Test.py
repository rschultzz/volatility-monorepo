# scripts/API_Test.py

import requests
import pandas as pd
from io import StringIO

# =========================
# CONFIG – EDIT THESE ONLY
# =========================
API_KEY = "cd809e2a-287c-4af7-9b05-a344df894745"  # <--- paste token

TICKER = "SPX"

# Minute you want to inspect (ET)
# Example: 2025-11-17 09:31 ET -> "202511170931"
LIVE_TRADE_TS = "202511170931"

# Expiration you care about
LIVE_EXPIRY = "2025-12-19"     # YYYY-MM-DD

# Daily / hist sample (a trade date you know works)
HIST_TRADE_DATE = "2025-11-17"  # YYYY-MM-DD

BASE = "https://api.orats.io/datav2"


def _print_header(title: str) -> None:
    print("\n" + "=" * 5, title, "=" * 5)


# -----------------------------
# 1) LIVE one-minute CSV
# -----------------------------
def live_one_minute_csv():
    url = f"{BASE}/hist/live/one-minute/monies/implied.csv"
    params = {
        "token": API_KEY,
        "ticker": TICKER,
        "tradeDate": LIVE_TRADE_TS,  # yyyymmddhhmm
        "expiry": LIVE_EXPIRY,
    }
    r = requests.get(url, params=params, timeout=10)
    _print_header("LIVE ONE-MINUTE CSV")
    print("URL:", r.url)
    print("Status:", r.status_code)
    if r.status_code != 200:
        print(r.text)
        return None

    df = pd.read_csv(StringIO(r.text))
    if "expirDate" in df.columns:
        df = df[df["expirDate"] == LIVE_EXPIRY]

    if df.empty:
        print("No rows for this expiry.")
        return None

    row = df.iloc[0]
    print("row:")
    print(row[["quoteDate", "expirDate", "stockPrice", "vol50", "atmiv"]])

    return {
        "quoteDate": row["quoteDate"],
        "expirDate": row["expirDate"],
        "stockPrice": float(row["stockPrice"]),
        "vol50": float(row["vol50"]),
        "atmiv": float(row["atmiv"]),
    }


# -----------------------------
# 2) LIVE one-minute JSON range
# -----------------------------
def live_one_minute_json():
    url = f"{BASE}/hist/live/one-minute/monies/implied"
    params = {
        "token": API_KEY,
        "ticker": TICKER,
        "tradeDate": f"{LIVE_TRADE_TS},{LIVE_TRADE_TS}",
        "expiry": LIVE_EXPIRY,
    }
    r = requests.get(url, params=params, timeout=10)
    _print_header("LIVE ONE-MINUTE RANGE (JSON)")
    print("URL:", r.url)
    print("Status:", r.status_code)
    if r.status_code != 200:
        print(r.text)
        return None

    # ORATS sometimes returns 200 with non-JSON; guard for that
    try:
        payload = r.json()
    except ValueError:
        print("Non-JSON body from live JSON endpoint:")
        print(r.text[:500])
        return None

    data = payload.get("data", [])
    if not data:
        print("No data[] in JSON response.")
        return None

    row = data[0]
    out = {
        "quoteDate": row.get("quoteDate"),
        "expirDate": row.get("expirDate"),
        "stockPrice": row.get("stockPrice"),
        "vol50": row.get("vol50"),
        "atmiv": row.get("atmiv"),
    }
    print("row:")
    print(out)
    return out


# -----------------------------
# 3) HIST monies JSON (EOD)
#    matches ORATS doc example:
#    .../hist/monies/implied?token=...&ticker=SPX&tradeDate=2025-11-17&fields=tradeDate,vol50
# -----------------------------
def hist_monies_json():
    url = f"{BASE}/hist/monies/implied"
    params = {
        "token": API_KEY,
        "ticker": TICKER,
        "tradeDate": HIST_TRADE_DATE,
        # Keep this aligned with the working example from ORATS:
        "fields": "tradeDate,vol50,vol5",
    }
    r = requests.get(url, params=params, timeout=10)
    _print_header("HIST MONIES JSON (EOD)")
    print("URL:", r.url)
    print("Status:", r.status_code)
    if r.status_code != 200:
        print(r.text)
        return None

    try:
        payload = r.json()
    except ValueError:
        print("Non-JSON body from hist JSON endpoint:")
        print(r.text[:500])
        return None

    data = payload.get("data", [])
    if not data:
        print("No data[] in JSON response.")
        return None

    df = pd.DataFrame(data)
    # This endpoint is EOD only; may have many expiries. Just show first N rows.
    print(df.head())
    # If you want a specific expiry later, we can add a filter here.

    return df


def main():
    live_csv = live_one_minute_csv()
    live_json = live_one_minute_json()
    hist_json_df = hist_monies_json()

    _print_header("SUMMARY COMPARISON")
    print("LIVE CSV:", live_csv)
    print("LIVE JSON:", live_json)
    if hist_json_df is not None:
        print("HIST JSON rows:", len(hist_json_df))
    else:
        print("HIST JSON: None")


if __name__ == "__main__":
    main()
