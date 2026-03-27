#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any

import requests
import pandas as pd
from dotenv import load_dotenv


DEFAULT_URL = "https://api.orats.io/datav2/hist/live/one-minute/strikes/chain"


def load_env() -> None:
    """Load .env from common repo locations without exposing secrets."""
    candidates = [
        Path.cwd() / ".env",
        Path.cwd().parent / ".env",
        Path(__file__).resolve().parent / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]
    for path in candidates:
        if path.exists():
            load_dotenv(path, override=False)
            return
    load_dotenv(override=False)


def normalize_trade_date(raw: str) -> str:
    """
    Normalize a few friendly inputs into the ORATS history format.

    ORATS docs for this endpoint specify EST/ET timestamp format YYYYMMDDHHMM.
    Examples accepted here:
      - 202603261015
      - 2026-03-26 10:15
      - 2026-03-26T10:15
    """
    raw = raw.strip()
    fmts = [
        "%Y%m%d%H%M",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M",
        "%Y/%m/%d %H:%M",
    ]
    for fmt in fmts:
        try:
            dt = datetime.strptime(raw, fmt)
            return dt.strftime("%Y%m%d%H%M")
        except ValueError:
            pass
    raise ValueError(
        "trade-date must look like YYYYMMDDHHMM or 'YYYY-MM-DD HH:MM' in Eastern time"
    )



def flatten_json_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [pd.json_normalize(item, sep="_").to_dict(orient="records")[0] if isinstance(item, dict) else {"value": item} for item in payload]

    if isinstance(payload, dict):
        # Common wrappers first
        for key in ("data", "items", "results", "rows"):
            value = payload.get(key)
            if isinstance(value, list):
                return pd.json_normalize(value, sep="_").to_dict(orient="records")

        # If the dict itself looks like a single row, preserve it.
        return pd.json_normalize(payload, sep="_").to_dict(orient="records")

    return [{"value": payload}]



def response_to_dataframe(resp: requests.Response) -> pd.DataFrame:
    content_type = (resp.headers.get("content-type") or "").lower()
    text_body = resp.text.strip()

    # Try JSON first if content type suggests it, or body looks like JSON.
    if "json" in content_type or text_body[:1] in "[{":
        try:
            payload = resp.json()
            records = flatten_json_records(payload)
            return pd.DataFrame(records)
        except Exception:
            pass

    # Then try CSV/plain text.
    if text_body:
        try:
            return pd.read_csv(StringIO(text_body))
        except Exception:
            pass

    raise ValueError("Could not parse ORATS response as JSON or CSV")



def build_output_path(output: str | None, ticker: str, trade_date: str) -> Path:
    if output:
        return Path(output)
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"orats_{ticker.lower()}_strikes_chain_{trade_date}.csv"



def main() -> int:
    parser = argparse.ArgumentParser(
        description="Test ORATS live strikes chain history endpoint and write the response to CSV."
    )
    parser.add_argument("--ticker", required=True, help="Ticker, e.g. SPX")
    parser.add_argument(
        "--trade-date",
        required=True,
        help="Eastern time minute to request. Example: 202603261015 or '2026-03-26 10:15'",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional CSV path. Default: outputs/orats_<ticker>_strikes_chain_<tradeDate>.csv",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"ORATS endpoint URL. Default: {DEFAULT_URL}",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Request timeout in seconds. Default: 60",
    )
    args = parser.parse_args()

    load_env()
    api_key = os.getenv("ORATS_API_KEY")
    if not api_key:
        print("ERROR: ORATS_API_KEY was not found in your .env or environment.", file=sys.stderr)
        return 1

    try:
        trade_date = normalize_trade_date(args.trade_date)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    params = {
        "token": api_key,
        "ticker": args.ticker,
        "tradeDate": trade_date,
    }

    print("Requesting ORATS strikes chain history...")
    print(f"  URL: {args.url}")
    print(f"  ticker: {args.ticker}")
    print(f"  tradeDate: {trade_date} (Eastern)")

    try:
        resp = requests.get(args.url, params=params, timeout=args.timeout)
    except requests.RequestException as e:
        print(f"ERROR: request failed: {e}", file=sys.stderr)
        return 1

    if not resp.ok:
        print(f"ERROR: ORATS returned HTTP {resp.status_code}", file=sys.stderr)
        print(resp.text[:2000], file=sys.stderr)
        return 1

    try:
        df = response_to_dataframe(resp)
    except Exception as e:
        print(f"ERROR: could not parse response: {e}", file=sys.stderr)
        print(resp.text[:2000], file=sys.stderr)
        return 1

    output_path = build_output_path(args.output, args.ticker, trade_date)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, quoting=csv.QUOTE_MINIMAL)

    print(f"Success. Wrote {len(df):,} rows to {output_path}")
    print(f"Columns ({len(df.columns)}): {', '.join(map(str, df.columns))}")
    if {"callVolume", "putVolume"}.issubset(df.columns):
        print("Found callVolume and putVolume columns.")
    else:
        print("Did not find both callVolume and putVolume columns in this response.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
