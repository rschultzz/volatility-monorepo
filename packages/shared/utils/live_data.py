from __future__ import annotations

import os
import io
from typing import Optional

import pandas as pd
import requests


def fetch_live_orats_data(ticker: str = "SPX") -> Optional[pd.DataFrame]:
    """
    Fetches live one-minute data from the ORATS API.
    """
    token = os.getenv("ORATS_API_TOKEN", "cd809e2a-287c-4af7-9b05-a344df894745")  # Fallback for development
    if not token:
        print("Warning: ORATS_API_TOKEN environment variable is not set.")
        return None

    url = f"https://api.orats.io/datav2/live/one-minute/monies/implied?token={token}&ticker={ticker}"

    try:
        response = requests.get(url)
        response.raise_for_status()

        if not response.text.strip():
            print("No data in live ORATS response (empty).")
            return None

        df = pd.read_csv(io.StringIO(response.text))

        # Rename columns to match the database schema (snake_case)
        rename_map = {
            "tradeDate": "trade_date",
            "expirDate": "expir_date",
            "stockPrice": "stock_price",
            "snapShotDate": "snap_shot_date",
        }
        df.rename(columns=rename_map, inplace=True)

        if "expir_date" in df.columns and "trade_date" in df.columns:
            return df
        else:
            print("Missing 'expir_date' or 'trade_date' column in live ORATS response.")
            return None

    except requests.exceptions.RequestException as e:
        print(f"Error fetching live ORATS data: {e}")
        return None
    except Exception as e:
        print(f"Error processing live ORATS data: {e}")
        return None
