# pragma: no cover
# Build today's minutes from the one-minute endpoint and upsert them.
import os, sys, datetime as dt
import pandas as pd

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from packages.shared.utils.data_io import init_orats_monies_schema
from packages.shared.ingest.monies_ingest import upsert_minute_payload
from packages.shared.options_orats import fetch_monies_minutes_for_day_via_one_minute

SMILE_COLS = ["P10","P15","P20","P25","P30","P35","ATM","C35","C30","C25","C20","C15","C10"]

def _build_payloads(df_day: pd.DataFrame, ticker: str) -> list[dict]:
    """Make one payload per unique minute (taking last row per expiry each minute)."""
    # confirm required columns
    if "ts" not in df_day.columns or "expiry" not in df_day.columns:
        return []

    # minute buckets
    df = df_day.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts","expiry"])
    payloads = []
    for ts_utc, block in df.groupby(df["ts"]):
        # last row per expiry at this minute
        per_exp = block.sort_values(["expiry","ts"]).groupby("expiry", as_index=False).tail(1)
        # map smile columns if present
        lower = {c.lower(): c for c in per_exp.columns}
        minute = pd.Timestamp(ts_utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        trade_date = pd.Timestamp(ts_utc).date()
        expirations = []
        for _, row in per_exp.iterrows():
            exp_date = row["expiry"]
            dte = int(max((exp_date - trade_date).days, 0)) if isinstance(exp_date, dt.date) else 0
            smile = {}
            for name in SMILE_COLS:
                col = lower.get(name.lower())
                if col is None: continue
                v = pd.to_numeric(row[col], errors="coerce")
                if pd.notna(v):
                    key = "atm" if name.upper() == "ATM" else name.lower()
                    smile[key] = float(v)
            expirations.append({
                "expiry_date": exp_date.isoformat(),
                "dte": dte,
                "forward": None,
                "smile": smile,
            })
        payloads.append({"ticker": ticker, "minute": minute, "underlying": None, "rf_rate": None, "div_yield": None, "expirations": expirations})
    return payloads

def main():
    ticker = os.getenv("TICKER","SPX")
    today = dt.date.today()
    init_orats_monies_schema()

    # pull the whole day via one-minute endpoint
    df_day = fetch_monies_minutes_for_day_via_one_minute(ticker=ticker, trade_date=today, expiry=None, max_minutes=None)
    if df_day is None or df_day.empty:
        print("No data from one-minute endpoint for today.")
        return

    total = 0
    for payload in _build_payloads(df_day, ticker):
        total += upsert_minute_payload(payload)
    print(f"[OK] Upserted {total} rows across expiries for {ticker} on {today}")

if __name__ == "__main__":
    main()
