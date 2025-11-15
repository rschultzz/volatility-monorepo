# pragma: no cover
import os, sys, datetime as dt
import pandas as pd

if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from packages.shared.utils.data_io import init_orats_monies_schema
from packages.shared.ingest.monies_ingest import upsert_minute_payload
from packages.shared.options_orats import ET_TZ, fetch_intraday_implied_monies_history

# Map ORATS volXX columns into your smile keys
VOL_MAP = {
    "P10": [90], "P15": [85], "P20": [80], "P25": [75], "P30": [70], "P35": [65],
    "C35": [35], "C30": [30], "C25": [25], "C20": [20], "C15": [15], "C10": [10],
}
SMILE_ORDER = ["P10","P15","P20","P25","P30","P35","ATM","C35","C30","C25","C20","C15","C10"]

def _find_col(df, names):
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None

def _vol_aliases(delta_int: int):
    d = str(delta_int)
    return [f"vol {d}", f"vol{d}", f"vol_{d}"]

def _build_payloads(df_day: pd.DataFrame, ticker: str) -> list[dict]:
    if df_day is None or df_day.empty:
        return []
    if "ts" not in df_day.columns or "expiry" not in df_day.columns:
        return []

    df = df_day.copy()
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts","expiry"])

    spot_col = _find_col(df, ["stockPrice","underlying","price","spot","stockprice","underlyingprice"])
    payloads = []

    for ts_utc, block in df.groupby(df["ts"]):
        per_exp = block.sort_values(["expiry","ts"]).groupby("expiry", as_index=False).tail(1)

        underlying = None
        if spot_col and spot_col in per_exp.columns:
            s = pd.to_numeric(per_exp[spot_col], errors="coerce")
            if s.notna().any():
                underlying = float(s.dropna().iloc[0])

        trade_date = pd.Timestamp(ts_utc).tz_convert("UTC").date()
        minute = pd.Timestamp(ts_utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        lower = {c.lower(): c for c in per_exp.columns}
        atm_col = _find_col(per_exp, ["atmiv","ATM","atm","iv_atm","atm_iv"])

        expirations = []
        for _, row in per_exp.iterrows():
            exp_date = row["expiry"]
            dte = int(max((exp_date - trade_date).days, 0))

            smile = {}
            if atm_col:
                v = pd.to_numeric(row[atm_col], errors="coerce")
                if pd.notna(v):
                    smile["atm"] = float(v)

            for name, deltas in VOL_MAP.items():
                for d in deltas:
                    col = None
                    for alias in _vol_aliases(d):
                        if alias.lower() in lower:
                            col = lower[alias.lower()]
                            break
                    if col:
                        v = pd.to_numeric(row[col], errors="coerce")
                        if pd.notna(v):
                            smile[name.lower()] = float(v)
                            break

            expirations.append({
                "expiry_date": exp_date.isoformat(),
                "dte": dte,
                "forward": None,
                "smile": smile,
            })

        payloads.append({
            "ticker": ticker,
            "minute": minute,
            "underlying": underlying,
            "rf_rate": None,
            "div_yield": None,
            "expirations": expirations,
        })
    return payloads

def main():
    ticker = os.getenv("TICKER","SPX")
    range_minutes = int(os.getenv("RANGE_MINUTES", "15"))
    dry_run = os.getenv("DRY_RUN","0") == "1"

    now_et = pd.Timestamp.now(tz=ET_TZ).to_pydatetime()
    open_et = dt.datetime.combine(now_et.date(), dt.time(9, 30), tzinfo=ET_TZ)
    close_et = dt.datetime.combine(now_et.date(), dt.time(16, 0), tzinfo=ET_TZ)

    # If before open, pull **yesterday's** close window. If after close, clamp to today's close.
    def _prev_bd(d: dt.date) -> dt.date:
        while d.weekday() >= 5:
            d -= dt.timedelta(days=1)
        return d

    if now_et < open_et:
        # pre-market: target yesterday’s last N minutes ending at 16:00
        y = _prev_bd(now_et.date() - dt.timedelta(days=1))
        open_et = dt.datetime.combine(y, dt.time(9, 30), tzinfo=ET_TZ)
        close_et = dt.datetime.combine(y, dt.time(16, 0), tzinfo=ET_TZ)
        end_et = close_et
    elif now_et > close_et:
        # after-hours: end at today’s close
        end_et = close_et
    else:
        # during RTH: end at 'now'
        end_et = now_et

    start_et = max(open_et, end_et - dt.timedelta(minutes=range_minutes))

    init_orats_monies_schema()

    print(f"[INFO] Fetching {ticker} range: {start_et} → {end_et} (ET), ~{range_minutes} minutes")
    df_day = fetch_intraday_implied_monies_history(ticker, start_et, end_et)
    if df_day is None or df_day.empty:
        print("[ERROR] No data returned from ORATS history endpoint for this window.")
        return

    # Quick stats
    n_rows = len(df_day)
    n_minutes = df_day["ts"].nunique() if "ts" in df_day.columns else "?"
    n_exps = df_day["expiry"].nunique() if "expiry" in df_day.columns else "?"
    print(f"[INFO] Got {n_rows} rows | minutes={n_minutes} | expiries={n_exps} | cols={list(df_day.columns)[:12]}")

    if dry_run:
        print("[DRY_RUN] Skipping DB writes.")
        return

    total = 0
    payloads = _build_payloads(df_day, ticker)
    for i, payload in enumerate(payloads, 1):
        total += upsert_minute_payload(payload)
        if i % 25 == 0:
            print(f"[INGEST] {i}/{len(payloads)} minutes processed…")

    print(f"[OK] Upserted {total} rows across expiries for {ticker} (window {start_et.date()})")

if __name__ == "__main__":
    main()
