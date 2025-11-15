#!/usr/bin/env python3
from __future__ import annotations

import os, sys, io, time, argparse, datetime as dt
from typing import Any, Dict, Optional, List

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Ensure "packages/..." imports work when run as a script
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Project imports
try:
    from packages.shared.ingest.monies_ingest import upsert_minute_payload
    from packages.shared.options_orats import ET_TZ
except Exception:
    from ingest.monies_ingest import upsert_minute_payload  # type: ignore
    from options_orats import ET_TZ  # type: ignore

ORATS_API_KEY = os.getenv("ORATS_API_KEY")
BASE = "https://api.orats.io"

def _session() -> requests.Session:
    s = requests.Session()
    s.mount(
        "https://",
        HTTPAdapter(
            max_retries=Retry(
                total=6,
                backoff_factor=0.5,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=frozenset(["GET"]),
                raise_on_status=False,
            )
        ),
    )
    return s

_SES = _session()

def _to_utc_iso_from_et(ts_like: Any) -> str:
    """Force ET -> UTC Z string."""
    ts = pd.Timestamp(ts_like)
    ts = ts.tz_localize(ET_TZ) if ts.tz is None else ts.tz_convert(ET_TZ)
    ts = ts.tz_convert("UTC")
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")

def _fetch_hist_minutes_json_chunk(
    ticker: str,
    start_et: dt.datetime,
    end_et: dt.datetime,
    timeout: int = 60,
) -> pd.DataFrame:
    """
    JSON version of the history call (more reliable than CSV right now).
    Returns a DataFrame with at least: tradeDate, expirDate, stockPrice, riskFreeRate, yieldRate, volXX...
    """
    if not ORATS_API_KEY:
        raise RuntimeError("ORATS_API_KEY not set in environment.")

    url = f"{BASE}/datav2/hist/monies/implied"
    params = {
        "token": ORATS_API_KEY,
        "ticker": ticker,
        "startDateTime": start_et.strftime("%Y%m%d%H%M"),
        "endDateTime":   end_et.strftime("%Y%m%d%H%M"),
        # No fields param required for JSON; it returns the full record.
    }
    r = _SES.get(url, params=params, headers={"User-Agent": "backfill-day-expiry/0.3"}, timeout=timeout)
    r.raise_for_status()

    # Some infra occasionally returns CSV text even on the JSON path; handle defensively.
    try:
        data = r.json()
        if not isinstance(data, list):
            return pd.DataFrame()
        return pd.DataFrame(data)
    except ValueError:
        txt = r.text.strip()
        if not txt or txt.startswith("<"):
            return pd.DataFrame()
        # Fallback: parse as CSV if that’s what we actually got.
        return pd.read_csv(io.StringIO(txt))

def _pc13_from_row(row: pd.Series) -> Dict[str, float]:
    """Canonical 13-bucket smile from volXX columns to match your chart 1:1."""
    out: Dict[str, float] = {}
    def num(c: str) -> Optional[float]:
        if c not in row.index: return None
        v = pd.to_numeric(row[c], errors="coerce")
        return None if pd.isna(v) else float(v)

    # puts
    for k, col in {"p10":"vol90","p15":"vol85","p20":"vol80","p25":"vol75","p30":"vol70","p35":"vol65"}.items():
        v = num(col)
        if v is not None: out[k] = v
    # atm
    v_atm = num("vol50")
    if v_atm is not None: out["atm"] = v_atm
    # calls
    for k, col in {"c35":"vol35","c30":"vol30","c25":"vol25","c20":"vol20","c15":"vol15","c10":"vol10"}.items():
        v = num(col)
        if v is not None: out[k] = v
    return out

def _chunk_iter(trade_date: dt.date, chunk_minutes: int = 120) -> List[tuple[dt.datetime, dt.datetime]]:
    """Build ET chunk windows across RTH 09:30–16:00 ET."""
    start = dt.datetime.combine(trade_date, dt.time(9,30)).replace(tzinfo=ET_TZ)
    end   = dt.datetime.combine(trade_date, dt.time(16,0)).replace(tzinfo=ET_TZ)
    chunks = []
    cur = start
    step = dt.timedelta(minutes=chunk_minutes)
    while cur < end:
        nxt = min(cur + step, end)
        chunks.append((cur, nxt))
        cur = nxt
    return chunks

def main():
    ap = argparse.ArgumentParser(description="Backfill one trade_date+expiry from ORATS history into orats_monies_minute.")
    ap.add_argument("--ticker", required=True)
    ap.add_argument("--date",   required=True, help="YYYY-MM-DD trade date")
    ap.add_argument("--expiry", required=True, help="YYYY-MM-DD target expiry (Friday; Saturday roll will be included)")
    ap.add_argument("--chunk-mins", type=int, default=120, help="Chunk size (minutes). Try 60/30 if you see 502s.")
    ap.add_argument("--sleep", type=float, default=0.5, help="Sleep between chunk requests (seconds).")
    ap.add_argument("--timeout", type=int, default=60, help="HTTP timeout per chunk.")
    args = ap.parse_args()

    ticker = args.ticker.upper()
    trade_date = dt.date.fromisoformat(args.date)
    expiry = dt.date.fromisoformat(args.expiry)
    expiry_plus = expiry + dt.timedelta(days=1)  # ORATS often stores monthlies on Saturday

    print(f"[backfill] fetching {ticker} {trade_date} for expiry {expiry} (and {expiry_plus}) in {args.chunk_mins}m chunks (JSON)")

    # 1) fetch all chunks via JSON → concat
    dfs = []
    for i, (a,b) in enumerate(_chunk_iter(trade_date, args.chunk_mins), start=1):
        try:
            dfc = _fetch_hist_minutes_json_chunk(ticker, a, b, timeout=args.timeout)
        except Exception as e:
            print(f"[warn] chunk {i} {a.strftime('%H:%M')}–{b.strftime('%H:%M')} failed: {e}")
            dfc = pd.DataFrame()
        if not dfc.empty:
            dfs.append(dfc)
            print(f"[chunk] {i:02d} {a.strftime('%H:%M')}–{b.strftime('%H:%M')}: {len(dfc)} rows")
        time.sleep(args.sleep)

    if not dfs:
        print("[backfill] no data returned (all chunks empty).")
        return

    df = pd.concat(dfs, ignore_index=True)

    # Normalize column names across JSON/CSV responses
    # JSON may use 'tradeDate', 'expirDate', 'stockPrice', 'riskFreeRate', 'yieldRate', 'volXX'...
    # Ensure the expected keys exist
    for need in ["tradeDate","expirDate"]:
        if need not in df.columns:
            raise RuntimeError(f"Expected column '{need}' missing in response. Have: {list(df.columns)}")

    # 2) filter to requested expiry (Friday or Saturday)
    df["expirDate"] = pd.to_datetime(df["expirDate"], errors="coerce").dt.date
    df = df[df["expirDate"].isin({expiry, expiry_plus})].copy()
    if df.empty:
        print("[backfill] no rows for that expiry/day after filtering.")
        return

    # 3) upsert each minute (smile → canonical p/c/atm buckets) using strict UTC minute
    n = 0
    for _, row in df.iterrows():
        # tradeDate usually ET with no tz; convert to UTC Z
        minute_iso = _to_utc_iso_from_et(row["tradeDate"])

        def fnum(col: str) -> Optional[float]:
            if col not in df.columns: return None
            v = pd.to_numeric(row[col], errors="coerce")
            return None if pd.isna(v) else float(v)

        smile = _pc13_from_row(row)
        payload = {
            "ticker": ticker,
            "minute": minute_iso,  # strict UTC Z string
            "underlying": fnum("stockPrice"),
            "rf_rate":   fnum("riskFreeRate"),
            "div_yield": fnum("yieldRate"),
            "expirations": [{
                "expiry_date": expiry.isoformat(),  # pin to Friday for consistency
                "dte": max((expiry - pd.Timestamp(minute_iso).tz_convert("UTC").date()).days, 0),
                "forward": None,
                "smile": smile,
            }],
        }
        n += upsert_minute_payload(payload, grid="delta")

    print(f"[backfill] upserted {n} minutes for {ticker} {trade_date} {expiry}")

if __name__ == "__main__":
    main()
