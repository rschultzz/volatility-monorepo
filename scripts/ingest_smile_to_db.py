# pragma: no cover
# Ingest a window of *monies smile* minutes into Postgres.
# - Uses your Live Intraday "hist/live/one-minute" endpoint via
#   packages.shared.options_orats.fetch_intraday_implied_monies_history
# - Writes EXACT smile dicts (p10..c10 + atm) per (minute, expiry)
# - Works after-hours by clamping to the RTH session.

import os
import sys
import datetime as dt
import pandas as pd

# ensure repo root on path
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from packages.shared.options_orats import ET_TZ, fetch_intraday_implied_monies_history
from packages.shared.utils.data_io import init_orats_monies_schema
from packages.shared.ingest.monies_ingest import upsert_minute_payload

# --- mapping helpers ---------------------------------------------------------
SMILE_BUCKETS = ["P10","P15","P20","P25","P30","P35","ATM","C35","C30","C25","C20","C15","C10"]

# If the CSV already has monies buckets (P10..C10), prefer those.
# Otherwise, map ORATS volXX (call-delta vols) → monies buckets:
#   C10 <- vol10, C15 <- vol15, ..., C35 <- vol35
#   P10 <- vol90, P15 <- vol85, ..., P35 <- vol65
VOL_MAP = {
    "P10": [90], "P15": [85], "P20": [80], "P25": [75], "P30": [70], "P35": [65],
    "C35": [35], "C30": [30], "C25": [25], "C20": [20], "C15": [15], "C10": [10],
}

ATM_ALIASES = ["atmiv","mwVol","mwvol","ATM","atm","iv_atm","atm_iv"]
UNDERLYING_CANDS = ["stockPrice","underlying","underlyingprice","price","spot","stockprice"]
TS_CANDS = ["ts","quoteDate","quotedate","timestamp","time","minute"]
EXP_CANDS = ["expiry","expiration","exp","exp_date","expdate","expirationdate","expirDate"]
RF_CANDS = ["riskFreeRate","rf","rf_rate"]
DIV_CANDS = ["yieldRate","dividendYield","div_yield","divYield"]


def _find_col(df: pd.DataFrame, names: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None


def _vol_aliases(delta_int: int) -> list[str]:
    d = str(delta_int)
    return [f"vol {d}", f"vol{d}", f"vol_{d}"]


# --- window selection that works pre/after-hours -----------------------------

def _prev_bd(d: dt.date) -> dt.date:
    while d.weekday() >= 5:
        d -= dt.timedelta(days=1)
    return d


def _compute_window(range_minutes: int) -> tuple[dt.datetime, dt.datetime]:
    now_et = pd.Timestamp.now(tz=ET_TZ).to_pydatetime()
    open_et  = dt.datetime.combine(now_et.date(), dt.time(9, 30), tzinfo=ET_TZ)
    close_et = dt.datetime.combine(now_et.date(), dt.time(16, 0), tzinfo=ET_TZ)

    if now_et < open_et:
        y = _prev_bd(now_et.date() - dt.timedelta(days=1))
        open_et  = dt.datetime.combine(y, dt.time(9, 30), tzinfo=ET_TZ)
        close_et = dt.datetime.combine(y, dt.time(16, 0), tzinfo=ET_TZ)
        end_et   = close_et
    elif now_et > close_et:
        end_et = close_et
    else:
        end_et = now_et

    start_et = max(open_et, end_et - dt.timedelta(minutes=range_minutes))
    return start_et, end_et


# --- transform + ingest ------------------------------------------------------

def _build_payloads(df_day: pd.DataFrame, ticker: str) -> list[dict]:
    if df_day is None or df_day.empty:
        return []

    # normalize ts/expiry if caller didn't already
    ts_col = _find_col(df_day, TS_CANDS)
    exp_col = _find_col(df_day, EXP_CANDS)
    if not ts_col or not exp_col:
        return []

    df = df_day.copy()
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.dropna(subset=[ts_col, exp_col])
    df[exp_col] = pd.to_datetime(df[exp_col], errors="coerce").dt.date

    spot_col = _find_col(df, UNDERLYING_CANDS)
    rf_col = _find_col(df, RF_CANDS)
    div_col = _find_col(df, DIV_CANDS)

    # Prefer true monies buckets if present
    lower = {c.lower(): c for c in df.columns}
    monies_present = {name: lower.get(name.lower()) for name in SMILE_BUCKETS if lower.get(name.lower())}

    payloads = []
    for ts_utc, block in df.groupby(df[ts_col]):
        per_exp = block.sort_values([exp_col, ts_col]).groupby(exp_col, as_index=False).tail(1)

        underlying = None
        if spot_col and spot_col in per_exp.columns:
            s = pd.to_numeric(per_exp[spot_col], errors="coerce")
            if s.notna().any():
                underlying = float(s.dropna().iloc[0])

        rf_rate = None
        if rf_col and rf_col in per_exp.columns:
            r = pd.to_numeric(per_exp[rf_col], errors="coerce")
            if r.notna().any():
                rf_rate = float(r.dropna().iloc[0])

        div_yield = None
        if div_col and div_col in per_exp.columns:
            y = pd.to_numeric(per_exp[div_col], errors="coerce")
            if y.notna().any():
                div_yield = float(y.dropna().iloc[0])

        trade_date = pd.Timestamp(ts_utc).tz_convert("UTC").date()
        minute = pd.Timestamp(ts_utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        expirations = []
        for _, row in per_exp.iterrows():
            exp_date = row[exp_col]
            dte = int(max((exp_date - trade_date).days, 0))

            smile = {}
            # ATM first
            atm_col = _find_col(per_exp, ATM_ALIASES)
            if atm_col:
                v = pd.to_numeric(row[atm_col], errors="coerce")
                if pd.notna(v):
                    smile["atm"] = float(v)

            if monies_present:
                # direct monies columns (P/C labels)
                for name, orig in monies_present.items():
                    if name.upper() == "ATM":
                        continue  # already handled
                    v = pd.to_numeric(row[orig], errors="coerce")
                    if pd.notna(v):
                        smile[name.lower()] = float(v)
            else:
                # derive from volXX columns
                row_lower = {c.lower(): c for c in row.index}
                for name, deltas in VOL_MAP.items():
                    for d in deltas:
                        col = None
                        for alias in _vol_aliases(d):
                            if alias.lower() in row_lower:
                                col = row_lower[alias.lower()]
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
            "rf_rate": rf_rate,
            "div_yield": div_yield,
            "expirations": expirations,
        })

    return payloads


def main():
    ticker = os.getenv("TICKER", "SPX")
    range_minutes = int(os.getenv("RANGE_MINUTES", "60"))
    dry = os.getenv("DRY_RUN", "0") == "1"

    start_et, end_et = _compute_window(range_minutes)
    print(f"[INFO] Fetching {ticker} range: {start_et} → {end_et} (ET), ~{range_minutes} minutes")

    init_orats_monies_schema()

    df = fetch_intraday_implied_monies_history(ticker, start_et, end_et)
    if df is None or df.empty:
        print("[ERROR] No data returned from ORATS for this window.")
        return

    # Normalize obvious differences from ORATS (expirDate → expiry)
    if "expirDate" in df.columns and "expiry" not in df.columns:
        df = df.rename(columns={"expirDate": "expiry"})

    # Also ensure we have a timestamp column named 'ts' for grouping
    if "ts" not in df.columns:
        ts_col = _find_col(df, TS_CANDS)
        if ts_col:
            df["ts"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")

    n_rows = len(df)
    n_minutes = df["ts"].nunique() if "ts" in df.columns else "?"
    n_exps = df["expiry"].nunique() if "expiry" in df.columns else "?"
    print(f"[INFO] Got {n_rows} rows | minutes={n_minutes} | expiries={n_exps} | cols={list(df.columns)[:14]}")

    payloads = _build_payloads(df, ticker)
    if dry:
        print(f"[DRY_RUN] Would upsert {len(payloads)} minute payloads.")
        return

    total = 0
    for i, payload in enumerate(payloads, 1):
        total += upsert_minute_payload(payload)
        if i % 25 == 0:
            print(f"[INGEST] {i}/{len(payloads)} minutes processed…")

    print(f"[OK] Upserted {total} rows across expiries for {ticker}.")


if __name__ == "__main__":
    main()
