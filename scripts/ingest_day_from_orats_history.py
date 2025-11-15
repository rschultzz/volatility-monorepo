# scripts/ingest_day_from_orats_history.py
from __future__ import annotations
import argparse, os, io, sys, pathlib, datetime as dt
import pandas as pd
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# repo import path
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from packages.shared.ingest.monies_ingest import upsert_from_dashboard_minute
from packages.shared.cache.day_cache import get_day_df

BASE = "https://api.orats.io"

def _session():
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=Retry(
        total=3, backoff_factor=0.25, status_forcelist=[429,500,502,503,504]
    )))
    return s

def fetch_hist_monies_day_df(ticker: str, trade_date: str, token: str) -> pd.DataFrame:
    """
    Pulls *minute-level* monies for a given trade_date with the buckets your smile needs.
    Returns a raw CSV-as-DataFrame (one row per minute x expiry).
    """
    url = f"{BASE}/datav2/hist/monies/implied.csv"
    # Request only the columns we need for the smile + a few helpful fields
    fields = ",".join([
        "vol90","vol85","vol80","vol75","vol70","vol65","vol50",
        "vol35","vol30","vol25","vol20","vol15","vol10",
        "stockPrice","expirDate","quoteDate"
    ])
    params = {"ticker": ticker, "tradeDate": trade_date, "fields": fields, "token": token}
    r = _session().get(url, params=params, headers={"User-Agent":"day-ingest/0.1"}, timeout=60)
    r.raise_for_status()
    txt = r.text.strip()
    if not txt or txt.startswith("<"):
        raise RuntimeError("ORATS returned empty/HTML for this date")
    df = pd.read_csv(io.StringIO(txt))
    if df.empty:
        raise RuntimeError("No rows from ORATS for this date")
    return df

def minute_iso_utc(series) -> pd.Series:
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    return ts.dt.strftime("%Y-%m-%dT%H:%M:%SZ")

def main():
    ap = argparse.ArgumentParser(description="Ingest a full trading day of ORATS monies into DB.")
    ap.add_argument("--ticker", default="SPX")
    ap.add_argument("--date", required=True, help="YYYY-MM-DD (trade date)")
    ap.add_argument("--warm-expiry", default=None, help="YYYY-MM-DD (option expiry) to warm cache for")
    args = ap.parse_args()

    token = os.getenv("ORATS_API_KEY")
    db = os.getenv("DATABASE_URL")
    if not token: raise SystemExit("ORATS_API_KEY not set")
    if not db:     raise SystemExit("DATABASE_URL not set")

    # 1) Fetch raw ORATS day
    raw = fetch_hist_monies_day_df(args.ticker, args.date, token)

    # 2) Normalize timestamp + expiry column names for our upsert helper
    #    upsert_from_dashboard_minute accepts any of:
    #      minute_col in ("ts","quoteDate","quotedate","timestamp","time","minute")
    #      expiry_col in ("expiry","expiration","exp","exp_date","expdate","expirationdate","expirDate")
    if "quoteDate" not in raw.columns:
        # Try a few alternates; if none, bail with a clear message
        for alt in ("ts","timestamp","time","minute"):
            if alt in raw.columns:
                raw["quoteDate"] = raw[alt]
                break
    if "quoteDate" not in raw.columns:
        raise SystemExit("No quoteDate/ts/timestamp column present in ORATS CSV")

    # Ensure UTC ISO for minute stamps
    raw["quoteDate"] = minute_iso_utc(raw["quoteDate"])

    # 3) Iterate each minute and upsert all expiries for that minute in one call
    #    (raw already has one row per expiry per minute with volXX columns)
    #    We keep ORATS 'expirDate' as-is; the upsert helper will recognize it.
    minutes = sorted(raw["quoteDate"].unique())
    total_rows = 0
    for i, m in enumerate(minutes, 1):
        df_min = raw.loc[raw["quoteDate"] == m].copy()
        # Upsert into DB (maps volXX -> canonical p/c/atm smile under the hood)
        total_rows += upsert_from_dashboard_minute(df_min, ticker=args.ticker, store_volxx=False)
        if i % 25 == 0:
            print(f"[ingest] {i}/{len(minutes)} minutes…")

    print(f"[OK] Upserted rows across expiries for {args.ticker} {args.date}: {total_rows}")

    # 4) Optionally warm the in-process day cache for a specific expiry
    if args.warm_expiry:
        df = get_day_df(args.ticker, args.date, args.warm_expiry)
        n = 0 if df is None or df.empty else len(df)
        print(f"[cache] day_cache warm: {n} rows for {args.ticker} {args.date} {args.warm_expiry}")

if __name__ == "__main__":
    main()
