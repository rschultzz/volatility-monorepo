# options_orats.py — minimal monies client (hist only) + PT→ET helper
from __future__ import annotations
import os, io, datetime as dt
from typing import Optional
import pandas as pd
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

ORATS_API_KEY = os.getenv("ORATS_API_KEY")
BASE = "https://api.orats.io"

try:
    from zoneinfo import ZoneInfo
    PT_TZ = ZoneInfo("America/Los_Angeles")
    ET_TZ = ZoneInfo("America/New_York")
except Exception:
    class _PT(dt.tzinfo):
        def utcoffset(self, d): return dt.timedelta(hours=-8)
        def dst(self, d): return dt.timedelta(0)
        def tzname(self, d): return "PT"
    class _ET(dt.tzinfo):
        def utcoffset(self, d): return dt.timedelta(hours=-5)
        def dst(self, d): return dt.timedelta(0)
        def tzname(self, d): return "ET"
    PT_TZ, ET_TZ = _PT(), _ET()

def _session() -> requests.Session:
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=Retry(total=2, backoff_factor=0.3, status_forcelist=[429,500,502,503,504])))
    return s

_SES = _session()

def pt_minute_to_et(trade_date_iso: str, hhmm_pt: str) -> dt.datetime:
    h, m = map(int, hhmm_pt.strip().split(":"))
    ts_pt = dt.datetime.combine(dt.date.fromisoformat(trade_date_iso), dt.time(h, m, tzinfo=PT_TZ))
    return ts_pt.astimezone(ET_TZ).replace(second=0, microsecond=0)

def fetch_one_minute_monies(ts_et: dt.datetime, ticker: str, expiry_iso: Optional[str]) -> Optional[pd.DataFrame]:
    if not ORATS_API_KEY: return None
    params = {
        "ticker": ticker,
        "tradeDate": ts_et.strftime("%Y%m%d%H%M"),
        "token": ORATS_API_KEY,
    }
    if expiry_iso:
        params["expiry"] = expiry_iso
    url = f"{BASE}/datav2/hist/live/one-minute/monies/implied.csv"
    r = _SES.get(url, params=params, headers={"User-Agent":"monies-min/0.1"}, timeout=30)
    try:
        r.raise_for_status()
    except requests.exceptions.HTTPError:
        return None
    txt = r.text.strip()
    if not txt or txt.startswith("<"):
        return None
    try:
        df = pd.read_csv(io.StringIO(txt))
    except Exception:
        return None
    if df.empty: return None
    if "quoteDate" in df.columns:
        qd = pd.to_datetime(df["quoteDate"], utc=True, errors="coerce")
        df["quoteDate"] = qd.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    if "expiry" not in df.columns and expiry_iso:
        df["expiry"] = expiry_iso
    return df
