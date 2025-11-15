# pragma: no cover
# Full-smile smoke test that reuses YOUR existing day/minute fetcher in packages.shared.options_orats.
# It auto-discovers the function name and calls it, then upserts one minute with P10..ATM..C10.

import os
import sys
import importlib
import datetime as dt
import pandas as pd

# Ensure repo root on sys.path
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from packages.shared.utils.data_io import init_orats_monies_schema
from packages.shared.ingest.monies_ingest import upsert_minute_payload, read_minutes_df

# --- which columns your dash expects (case-insensitive) ---
SMILE_COLS = ["P10","P15","P20","P25","P30","P35","ATM","C35","C30","C25","C20","C15","C10"]
TS_CANDS   = ["ts","quoteDate","quotedate","timestamp","time","minute"]
EXP_CANDS  = ["expiry","expiration","exp","exp_date","expdate","expirationdate"]
SPOT_CANDS = ["underlying","spot","price","stockprice","underlyingprice"]

def _find_col(df: pd.DataFrame, names: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None

def _prev_bd(d: dt.date) -> dt.date:
    while d.weekday() >= 5:
        d -= dt.timedelta(days=1)
    return d

def _discover_day_fetch():
    """
    Import packages.shared.options_orats and return your existing 'day minutes' function.
    Tries common names; uses whatever exists in your file.
    """
    m = importlib.import_module("packages.shared.options_orats")
    candidates = [
        "fetch_monies_minutes_for_day",
        "fetch_intraday_monies_minutes",
        "fetch_monies_intraday",
        "get_monies_minutes_for_day",
        "monies_minutes_for_day",
        "fetch_orats_monies_day",
    ]
    for name in candidates:
        fn = getattr(m, name, None)
        if callable(fn):
            return fn, name
    raise RuntimeError(
        "Couldn't find a day/minute fetcher in packages.shared.options_orats.\n"
        "Add one of these function names (or tell me the actual name):\n"
        f"{', '.join(candidates)}"
    )

def _call_day_fetch(fn, ticker: str, trade_date: dt.date):
    """
    Call your function with flexible args/kwargs (handles different signatures).
    Returns a DataFrame or None.
    """
    combos = [
        # common: (ticker, trade_date, expiry=None)
        ((ticker, trade_date), {"expiry": None}),
        # kwargs with trade_date
        ((), {"ticker": ticker, "trade_date": trade_date, "expiry": None}),
        # YYYYMMDD string
        ((ticker, trade_date.strftime("%Y%m%d")), {}),
        ((), {"ticker": ticker, "date": trade_date, "expiry": None}),
        ((), {"ticker": ticker, "date_yyyymmdd": trade_date.strftime("%Y%m%d")}),
        # minimal
        ((ticker,), {"trade_date": trade_date}),
        ((ticker,), {"trade_date": trade_date.strftime("%Y%m%d")}),
    ]
    for args, kwargs in combos:
        try:
            df = fn(*args, **kwargs)
            if df is not None and len(df):
                return df
        except TypeError:
            continue
        except Exception:
            # swallow and try next signature
            continue
    return None

def _pick_working_day(day_fetch, preferred: dt.date, ticker: str) -> tuple[dt.date, pd.DataFrame | None]:
    d = preferred
    for _ in range(6):
        df = _call_day_fetch(day_fetch, ticker, d)
        if df is not None and len(df):
            print(f"[INFO] Using day fetcher on {d} (rows: {len(df)})")
            return d, df
        d = _prev_bd(d - dt.timedelta(days=1))
    return preferred, None

def _build_payload_from_day(df_day: pd.DataFrame, ticker: str, when_utc: pd.Timestamp | None = None) -> dict:
    ts_col = _find_col(df_day, TS_CANDS)
    exp_col = _find_col(df_day, EXP_CANDS)
    if not ts_col or not exp_col:
        raise RuntimeError(f"Required columns not found. Have: {list(df_day.columns)}")

    df = df_day.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_col, exp_col])
    df[exp_col] = pd.to_datetime(df[exp_col], errors="coerce").dt.date

    when_utc = pd.Timestamp(df[ts_col].max(), tz="UTC") if when_utc is None else pd.Timestamp(when_utc, tz="UTC")
    at_or_before = df[df[ts_col] <= when_utc]
    if at_or_before.empty:
        raise RuntimeError("No rows at or before requested minute.")

    per_exp = at_or_before.sort_values([exp_col, ts_col]).groupby(exp_col, as_index=False).tail(1)

    # map smile columns present (case-insensitive)
    lower = {c.lower(): c for c in per_exp.columns}
    present = {name: lower.get(name.lower()) for name in SMILE_COLS if lower.get(name.lower())}
    if not present:
        print("[WARN] No full-smile columns found; this endpoint might be ATM-only for your plan.")
    else:
        missing = [c for c in SMILE_COLS if c not in present]
        if missing:
            print(f"[INFO] Available smile cols: {list(present.keys())}; missing (ok): {missing}")

    # optional spot
    spot_col = _find_col(per_exp, SPOT_CANDS)
    underlying = None
    if spot_col:
        s = pd.to_numeric(per_exp[spot_col], errors="coerce")
        if s.notna().any():
            underlying = float(s.dropna().iloc[0])

    minute_utc = when_utc.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    trade_date = when_utc.tz_convert("UTC").date()

    expirations = []
    for _, row in per_exp.iterrows():
        exp_date = row[exp_col]
        dte = int(max((exp_date - trade_date).days, 0))
        smile = {}
        for name, orig in present.items():
            v = pd.to_numeric(row[orig], errors="coerce")
            if pd.notna(v):
                key = "atm" if name.upper() == "ATM" else name.lower()
                smile[key] = float(v)
        expirations.append({
            "expiry_date": exp_date.isoformat(),
            "dte": dte,
            "forward": None,
            "smile": smile,
        })

    return {
        "ticker": ticker,
        "minute": minute_utc,
        "underlying": underlying,
        "rf_rate": None,
        "div_yield": None,
        "expirations": expirations,
    }

def main():
    ticker = os.getenv("TICKER", "SPX")
    td_env = os.getenv("TEST_TRADE_DATE")
    trade_date = dt.date.fromisoformat(td_env) if td_env else dt.date.today()

    init_orats_monies_schema()

    day_fetch, used_name = _discover_day_fetch()
    print(f"[INFO] Using your day fetcher: packages.shared.options_orats.{used_name}")

    used_date, df_day = _pick_working_day(day_fetch, trade_date, ticker)
    if df_day is None or df_day.empty:
        print("Day-level fetch returned no rows after trying recent business days. "
              "If your dash calls a *different* function for minutes, tell me its name and I'll point to it.")
        return
    if used_date != trade_date:
        print(f"[INFO] Using {used_date} instead of {trade_date} (preferred had no data).")

    payload = _build_payload_from_day(df_day, ticker=ticker, when_utc=None)

    n_rows = upsert_minute_payload(payload)
    out = read_minutes_df(used_date, ticker=ticker)
    print(f"[OK] Upserted {n_rows} rows for {ticker} at {payload['minute']}")
    print(f"[OK] DB now has {len(out)} rows for {ticker} on {used_date}")
    if not out.empty:
        cols = [c for c in ["minute_ts","expiry_date","underlying","atm_iv","smile"] if c in out.columns]
        print(out.tail(5)[cols].to_string(index=False))

if __name__ == "__main__":
    main()
