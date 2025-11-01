from __future__ import annotations
import datetime as dt
from typing import Optional

# ET tz for the monies endpoint timestamp
try:
    from zoneinfo import ZoneInfo
    ET_TZ = ZoneInfo("America/New_York")
except Exception:
    class _ET(dt.tzinfo):
        def utcoffset(self, d): return dt.timedelta(hours=-5)
        def dst(self, d): return dt.timedelta(0)
        def tzname(self, d): return "ET"
    ET_TZ = _ET()

from shared.options_orats import fetch_one_minute_monies

def run(date: Optional[str] = None) -> None:
    """
    Minimal live fetch to prove ORATS connectivity.
    No DB writes yet—just logs the number of rows fetched.
    """
    # If a date is provided we could convert it to a timestamp;
    # for now, use "now minus 1 minute" in ET to align to a completed minute.
    now_et = dt.datetime.now(ET_TZ).replace(second=0, microsecond=0)
    ts_et = now_et - dt.timedelta(minutes=1)

    ticker = "SPX"
    expiry_iso = None   # keep None for now; we can add selection later

    df = fetch_one_minute_monies(ts_et, ticker, expiry_iso)
    if df is None or df.empty:
        print(f"[ingest] ORATS returned no data for {ticker} at {ts_et.isoformat()}")
        return

    print(f"[ingest] Fetched {len(df):,} rows for {ticker} at {ts_et.isoformat()}")
