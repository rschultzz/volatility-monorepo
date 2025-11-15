# pragma: no cover
# Smoke test: use the SAME columns as the live dashboard.
# 1) Prefer your dashboard's "minutes for day" function (expected to return P10..C10, ATM)
# 2) Fallback to one-minute CSV (usually ATM-only)

import os
import sys
import datetime as dt
import pandas as pd

from packages.shared.utils.data_io import init_orats_monies_schema
from packages.shared.ingest.monies_ingest import upsert_minute_payload, read_minutes_df

# --- Try to import the SAME function your dashboard uses ---
# Adjust the import below if your project keeps it elsewhere.
_DASH_FETCH = None
try:
    # Preferred: day-level minutes with the dashboard's exact column layout
    # e.g., returns a DataFrame with columns like:
    # ['ts','expiry','underlying','P10','P15','P20','P25','P30','P35','ATM','C35','C30','C25','C20','C15','C10', ...]
    from packages.shared.options_orats import fetch_monies_minutes_for_day as _DASH_FETCH  # type: ignore
except Exception:
    _DASH_FETCH = None

# Fallback: your minimal utils client for one-minute CSV (may be ATM-only)
from packages.shared.options_orats import fetch_one_minute_monies as _FALLBACK_FETCH

# --- Column setup: EXACTLY what the live dashboard plots ---
# If your dashboard uses different labels (e.g., lowercase), edit THIS dict once.
SMILE_COLS = {
    "ATM": "atm",
    "P10": "p10",
    "P15": "p15",
    "P20": "p20",
    "P25": "p25",
    "P30": "p30",
    "P35": "p35",
    "C35": "c35",
    "C30": "c30",
    "C25": "c25",
    "C20": "c20",
    "C15": "c15",
    "C10": "c10",
}

EXPIRY_CANDIDATES = ["expiry", "expiration", "exp", "exp_date", "expdate", "expirationdate"]
TS_CANDIDATES = ["ts", "timestamp", "minute", "quoteDate"]
SPOT_CANDIDATES = ["underlying", "spot", "price", "stockprice", "underlyingprice"]

def _find_col(df: pd.DataFrame, names: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None

def _normalize_smile_row(row: pd.Series) -> dict:
    """Map EXACT dashboard columns → payload keys (p10..c10, atm)."""
    smile = {}
    cols_lower = {c.lower(): c for c in row.index}
    for src, key in SMILE_COLS.items():
        c = cols_lower.get(src.lower())
        if c is None:
            continue
        v = pd.to_numeric(row[c], errors="coerce")
        if pd.notna(v):
            smile[key] = float(v)
    return smile

# ---------- Preferred path: use the dashboard's minutes-for-day function ----------

def _payload_from_dashboard_day(df_day: pd.DataFrame, ticker: str, when: dt.datetime | None) -> dict:
    """Build one-minute payload using the same columns the dashboard plots."""
    if df_day is None or df_day.empty:
        raise RuntimeError("Dashboard day DataFrame is empty.")

    # timestamp column
    ts_col = _find_col(df_day, TS_CANDIDATES)
    if not ts_col:
        raise RuntimeError(f"Couldn't find a timestamp column among {TS_CANDIDATES}. Got: {list(df_day.columns)}")

    # expiry column
    exp_col = _find_col(df_day, EXPIRY_CANDIDATES)
    if not exp_col:
        raise RuntimeError(f"Couldn't find an expiry column among {EXPIRY_CANDIDATES}. Got: {list(df_day.columns)}")

    # spot column (optional)
    spot_col = _find_col(df_day, SPOT_CANDIDATES)

    df = df_day.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")

    # choose a minute: latest if not provided
    if when is None:
        when = pd.Timestamp(df[ts_col].max()).to_pydatetime()
    # take last <= when per expiry
    at_or_before = df[df[ts_col] <= pd.Timestamp(when, tz="UTC")]
    if at_or_before.empty:
        raise RuntimeError("No rows at or before the requested minute.")

    # collapse to one row per expiry at that minute
    per_exp = (
        at_or_before.sort_values([exp_col, ts_col])
        .groupby(exp_col, as_index=False)
        .tail(1)
    )

    # UTC minute string
    minute_utc = pd.Timestamp(when).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ") \
        if pd.Timestamp(when).tz is not None else pd.Timestamp(when, tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")

    # underlying (spot) — choose first non-null if available
    underlying = None
    if spot_col and spot_col in per_exp.columns:
        try:
            s = pd.to_numeric(per_exp[spot_col], errors="coerce")
            if s.notna().any():
                underlying = float(s.dropna().iloc[0])
        except Exception:
            pass

    # Build expirations smiles
    expirations = []
    for _, row in per_exp.iterrows():
        # parse expiry date
        exp_val = row[exp_col]
        exp_date = pd.to_datetime(exp_val, errors="coerce").date() if not isinstance(exp_val, dt.date) else exp_val
        # smile from EXACT columns
        smile = _normalize_smile_row(row)
        # dte relative to trade date (use the minute's date in ET/NY if you prefer; UTC is fine for ingest)
        trade_date = pd.Timestamp(when).tz_convert("UTC").date() if pd.Timestamp(when).tz is not None else pd.Timestamp(when, tz="UTC").date()
        dte = (exp_date - trade_date).days if isinstance(exp_date, dt.date) else 0

        expirations.append({
            "expiry_date": exp_date.isoformat(),
            "dte": int(max(dte, 0)),
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

# ---------- Fallback path: one-minute CSV (usually ATM-only) ----------

try:
    from zoneinfo import ZoneInfo
    ET_TZ = ZoneInfo("America/New_York")
except Exception:
    class _ET(dt.tzinfo):
        def utcoffset(self, d): return dt.timedelta(hours=-5)
        def dst(self, d): return dt.timedelta(0)
        def tzname(self, d): return "ET"
    ET_TZ = _ET()

def _last_rth_minute(now_et: dt.datetime) -> dt.datetime:
    open_t, close_t = dt.time(9,30), dt.time(16,0)
    d = now_et.date()
    t = now_et.time()
    if t >= close_t:
        return dt.datetime.combine(d, dt.time(15,59), tzinfo=ET_TZ)
    if t < open_t:
        # previous weekday 15:59 (naive; ignores holidays)
        prev = d - dt.timedelta(days=1)
        while prev.weekday() >= 5:
            prev -= dt.timedelta(days=1)
        return dt.datetime.combine(prev, dt.time(15,59), tzinfo=ET_TZ)
    return now_et.replace(second=0, microsecond=0) - dt.timedelta(minutes=1)

def _payload_from_one_minute_csv(ticker: str) -> dict | None:
    ts_et = _last_rth_minute(dt.datetime.now(ET_TZ))
    df = _FALLBACK_FETCH(ts_et, ticker=ticker, expiry_iso=None)
    if df is None or df.empty:
        return None

    # minute in UTC Z-format
    ts = pd.Timestamp(ts_et)
    if ts.tz is None: ts = ts.tz_localize(ET_TZ)
    minute_utc = ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")

    # try to detect expiry & ATM-only
    exp_col = _find_col(df, EXPIRY_CANDIDATES)
    if exp_col is None:
        # if the CSV is per-expiry file, caller may have set expiry_iso; otherwise, skip
        # store as one pseudo-expiry (minute-only ingest)
        return {
            "ticker": ticker,
            "minute": minute_utc,
            "underlying": None,
            "rf_rate": None,
            "div_yield": None,
            "expirations": [],
        }

    df = df.copy()
    df[exp_col] = pd.to_datetime(df[exp_col], errors="coerce").dt.date
    expirations = []
    for exp in df[exp_col].dropna().unique():
        row = df[df[exp_col] == exp].iloc[0]
        atm_col = _find_col(df, ["ATM", "atm", "ATMIV", "iv_atm", "atm_iv"])
        atm = float(pd.to_numeric(row[atm_col], errors="coerce")) if atm_col else None
        trade_date = ts_et.date()
        dte = (exp - trade_date).days if isinstance(exp, dt.date) else 0
        expirations.append({"expiry_date": exp.isoformat(), "dte": int(max(dte,0)), "forward": None, "smile": {"atm": atm} if atm is not None else {}})

    return {"ticker": ticker, "minute": minute_utc, "underlying": None, "rf_rate": None, "div_yield": None, "expirations": expirations}

# ---------- main ----------

def main():
    ticker = os.getenv("TICKER", "SPX")
    trade_date = dt.date.today()  # use today's date for the smoke test
    init_orats_monies_schema()

    if _DASH_FETCH is not None:
        # Preferred: fetch the same day-level frame your dashboard uses
        df_day = _DASH_FETCH(ticker=ticker, trade_date=trade_date, expiry=None)  # adjust args if your signature differs
        if df_day is None or len(df_day) == 0:
            print("Dashboard fetch returned no rows; falling back to one-minute CSV (likely ATM-only).")
            payload = _payload_from_one_minute_csv(ticker)
            if not payload:
                print("No data available from fallback either.")
                return
        else:
            # Build payload from the LAST minute present in the dashboard data
            # If you want a specific minute, set `when=...`
            payload = _payload_from_dashboard_day(df_day, ticker=ticker, when=None)
    else:
        print("Dashboard day-fetch function not found; using one-minute CSV fallback (ATM-only).")
        payload = _payload_from_one_minute_csv(ticker)
        if not payload:
            print("No data available from fallback.")
            return

    # Upsert and read back
    n_rows = upsert_minute_payload(payload)
    minute_str = payload.get("minute")
    today = trade_date
    out = read_minutes_df(today, ticker=ticker)
    print(f"[OK] Upserted {n_rows} rows for {ticker} at {minute_str}")
    print(f"[OK] DB now has {len(out)} rows for {ticker} on {today}")

    if not out.empty:
        cols = [c for c in ["minute_ts","expiry_date","underlying","atm_iv","smile"] if c in out.columns]
        print(out.tail(5)[cols].to_string(index=False))

if __name__ == "__main__":
    # Make sure Python can see your repo packages when running directly
    if os.getcwd() not in sys.path:
        sys.path.insert(0, os.getcwd())
    main()
