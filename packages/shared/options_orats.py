# options_orats.py — monies client helpers (live one‑minute + day/minute full‑smile)
# Drop this file at: packages/shared/options_orats.py
from __future__ import annotations

import os
import io
import datetime as dt
from typing import Optional

import pandas as pd
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
ORATS_API_KEY = os.getenv("ORATS_API_KEY")
BASE = "https://api.orats.io"

# Optional override for a day/minute CSV endpoint (if your account provides one)
# Example: export ORATS_MONIES_DAY_URL="https://api.orats.io/datav2/hist/intraday/monies/implied.csv"
ORATS_MONIES_DAY_URL = os.getenv("ORATS_MONIES_DAY_URL")

# -----------------------------------------------------------------------------
# Timezones
# -----------------------------------------------------------------------------
try:
    from zoneinfo import ZoneInfo
    PT_TZ = ZoneInfo("America/Los_Angeles")
    ET_TZ = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover (older Pythons)
    class _PT(dt.tzinfo):
        def utcoffset(self, d): return dt.timedelta(hours=-8)
        def dst(self, d): return dt.timedelta(0)
        def tzname(self, d): return "PT"
    class _ET(dt.tzinfo):
        def utcoffset(self, d): return dt.timedelta(hours=-5)
        def dst(self, d): return dt.timedelta(0)
        def tzname(self, d): return "ET"
    PT_TZ, ET_TZ = _PT(), _ET()

# -----------------------------------------------------------------------------
# HTTP Session with retries
# -----------------------------------------------------------------------------

def _session() -> requests.Session:
    s = requests.Session()
    s.mount(
        "https://",
        HTTPAdapter(
            max_retries=Retry(
                total=3,
                backoff_factor=0.25,
                status_forcelist=[429, 500, 502, 503, 504],
            )
        ),
    )
    return s

_SES = _session()

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def pt_minute_to_et(trade_date_iso: str, hhmm_pt: str) -> dt.datetime:
    """Convert a PT hh:mm on a given trade_date (YYYY-MM-DD) to an ET minute (tz-aware)."""
    h, m = map(int, hhmm_pt.strip().split(":"))
    ts_pt = dt.datetime.combine(dt.date.fromisoformat(trade_date_iso), dt.time(h, m, tzinfo=PT_TZ))
    return ts_pt.astimezone(ET_TZ).replace(second=0, microsecond=0)

def _iso_utc_minute(ts: dt.datetime) -> str:
    """Return an ISO8601 Z string for the minute, ensuring UTC tz-awareness."""
    t = pd.Timestamp(ts)
    if t.tz is None:
        # assume ET input for one-minute helper fallback
        t = t.tz_localize(ET_TZ)
    return t.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")

# -----------------------------------------------------------------------------
# LIVE one‑minute monies (kept EXACTLY as your app expects)
# -----------------------------------------------------------------------------

def fetch_one_minute_monies(ts_et: dt.datetime, ticker: str, expiry_iso: Optional[str]) -> Optional[pd.DataFrame]:
    """
    Fetch a single ET minute CSV for monies implied vols.
    Returns a DataFrame as delivered by ORATS. If 'quoteDate' is present, it is normalized to
    UTC Z-string for convenience. If 'expiry' is absent and you passed expiry_iso, a column is added.

    This function is left UNCHANGED in signature/behavior so your live dash keeps working.
    """
    if not ORATS_API_KEY:
        return None

    params = {
        "ticker": ticker,
        "tradeDate": ts_et.strftime("%Y%m%d%H%M"),  # ET minute stamp
        "token": ORATS_API_KEY,
    }
    if expiry_iso:
        params["expiry"] = expiry_iso

    url = f"{BASE}/datav2/hist/live/one-minute/monies/implied.csv"
    r = _SES.get(url, params=params, headers={"User-Agent": "monies-min/0.1"}, timeout=30)
    try:
        r.raise_for_status()
    except requests.exceptions.HTTPError:
        return None

    txt = (r.text or "").strip()
    if not txt or txt.startswith("<"):
        return None

    try:
        df = pd.read_csv(io.StringIO(txt))
    except Exception:
        return None

    if df.empty:
        return None

    if "quoteDate" in df.columns:
        qd = pd.to_datetime(df["quoteDate"], utc=True, errors="coerce")
        df["quoteDate"] = qd.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    if "expiry" not in df.columns and expiry_iso:
        df["expiry"] = expiry_iso

    return df

# -----------------------------------------------------------------------------
# Live Intraday API — one‑minute history over a RANGE (for DB builds/backfills)
# -----------------------------------------------------------------------------

def fetch_intraday_implied_monies_history(
    ticker: str,
    start_et: dt.datetime,
    end_et: dt.datetime,
) -> Optional[pd.DataFrame]:
    """
    Live Intraday API: historical one-minute implied monies (range request).
    tradeDate must be ET, formatted "YYYYMMDDHHMM,YYYYMMDDHHMM".
    Endpoint path family: /datav2/hist/live/one-minute/monies/implied

    Returns a DataFrame with at least columns: ['ts' (UTC Timestamp), 'expiry' (date)],
    plus whatever smile columns your plan exposes (ATM and/or bucket vols like vol10/vol90, etc.).
    """
    if not ORATS_API_KEY:
        return None

    # ensure ET and minute precision
    def _et_floor_min(x: dt.datetime) -> dt.datetime:
        t = pd.Timestamp(x)
        if t.tz is None:
            t = t.tz_localize(ET_TZ)
        return t.tz_convert(ET_TZ).replace(second=0, microsecond=0).to_pydatetime()

    s = _et_floor_min(start_et)
    e = _et_floor_min(end_et)
    trade_range = f"{s.strftime('%Y%m%d%H%M')},{e.strftime('%Y%m%d%H%M')}"

    url = f"{BASE}/datav2/hist/live/one-minute/monies/implied"
    params = {"token": ORATS_API_KEY, "ticker": ticker, "tradeDate": trade_range}

    r = _SES.get(url, params=params, headers={"User-Agent": "monies-hist/0.1"}, timeout=60)
    try:
        r.raise_for_status()
    except requests.exceptions.HTTPError:
        return None

    txt = (r.text or "").strip()
    if not txt or txt.startswith("<"):
        return None

    try:
        df = pd.read_csv(io.StringIO(txt))
    except Exception:
        return None

    if df.empty:
        return None

    # normalize timestamp → 'ts' (UTC)
    ts_col = None
    for name in ("ts", "quoteDate", "quotedate", "timestamp", "time", "minute"):
        if name in df.columns:
            ts_col = name
            break
    if not ts_col:
        return None
    df["ts"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")

    # expiry → date
    if "expiry" in df.columns:
        df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce").dt.date
    else:
        for alt in ("expiration", "exp", "exp_date", "expdate", "expirationdate", "expirDate"):
            if alt in df.columns:
                df["expiry"] = pd.to_datetime(df[alt], errors="coerce").dt.date
                break
    if "expiry" not in df.columns:
        return None

    return df

# -----------------------------------------------------------------------------
# Utility: list ET minutes for an RTH session (used by fallback day builder)
# -----------------------------------------------------------------------------

def _rth_minutes_for_date(d: dt.date, now_et: dt.datetime | None = None) -> list[dt.datetime]:
    """Return ET minute stamps for 9:30→16:00 on date d (or up to now if today)."""
    open_t, close_t = dt.time(9, 30), dt.time(16, 0)
    start = dt.datetime.combine(d, open_t, tzinfo=ET_TZ)
    end   = dt.datetime.combine(d, close_t, tzinfo=ET_TZ)
    if now_et and d == now_et.date():
        end = min(end, now_et.replace(second=0, microsecond=0))
    mins = []
    t = start
    while t <= end:
        mins.append(t)
        t += dt.timedelta(minutes=1)
    return mins

# -----------------------------------------------------------------------------
# DAY/MINUTE builders used by the dash and by DB ingestion
# -----------------------------------------------------------------------------

def fetch_monies_minutes_for_day_via_one_minute(
    ticker: str,
    trade_date: dt.date,
    expiry: Optional[dt.date] = None,
    max_minutes: int | None = None,
) -> Optional[pd.DataFrame]:
    """
    Build a *day* DataFrame by iterating the live one-minute endpoint you already use.
    Note: Most plans only guarantee "today" here; older dates may not return.
    Returns a DataFrame containing at least ['ts','expiry'] and whatever other columns were provided.
    """
    if not ORATS_API_KEY:
        return None

    now_et = dt.datetime.now(ET_TZ)
    minute_list = _rth_minutes_for_date(trade_date, now_et=now_et)
    if max_minutes is not None and max_minutes > 0:
        minute_list = minute_list[-max_minutes:]

    frames: list[pd.DataFrame] = []
    exp_iso = expiry.isoformat() if expiry else None

    for ts_et in minute_list:
        df = fetch_one_minute_monies(ts_et, ticker=ticker, expiry_iso=exp_iso)
        if df is None or df.empty:
            continue
        # Normalize timestamp → 'ts' (UTC)
        if "quoteDate" in df.columns:
            ts = pd.to_datetime(df["quoteDate"], utc=True, errors="coerce")
            df["ts"] = ts
        elif "ts" in df.columns:
            df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
        else:
            df["ts"] = pd.to_datetime(_iso_utc_minute(ts_et), utc=True, errors="coerce")
        # Ensure expiry present
        if "expiry" in df.columns:
            df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce").dt.date
        elif expiry is not None:
            df["expiry"] = expiry
        else:
            for alt in ("expiration", "exp", "exp_date", "expdate", "expirationdate"):
                if alt in df.columns:
                    df["expiry"] = pd.to_datetime(df[alt], errors="coerce").dt.date
                    break
            if "expiry" not in df.columns:
                continue
        frames.append(df)

    if not frames:
        return None

    out = pd.concat(frames, ignore_index=True)
    if "ts" not in out.columns or "expiry" not in out.columns:
        return None
    return out


def fetch_monies_minutes_for_day(
    ticker: str,
    trade_date: dt.date,
    expiry: Optional[dt.date] = None,
) -> Optional[pd.DataFrame]:
    """
    Preferred day/minute fetcher for your dash/ingest.
    Behavior:
      1) If ORATS_MONIES_DAY_URL is set, try that CSV endpoint (some accounts have a direct day CSV).
      2) Else try Live Intraday API range endpoint (/datav2/hist/live/one-minute/monies/implied).
      3) Else fallback to iterating one-minute calls across the session.

    Always normalizes to columns: 'ts' (UTC Timestamp) and 'expiry' (date).
    """
    if not ORATS_API_KEY:
        return None

    # 1) explicit day URL override (CSV)
    if ORATS_MONIES_DAY_URL:
        params = {"ticker": ticker, "tradeDate": trade_date.strftime("%Y%m%d"), "token": ORATS_API_KEY}
        if expiry is not None:
            params["expiry"] = expiry.isoformat()
        try:
            r = _SES.get(ORATS_MONIES_DAY_URL, params=params, headers={"User-Agent": "monies-day/0.2"}, timeout=60)
            r.raise_for_status()
            txt = (r.text or "").strip()
            if txt and not txt.startswith("<"):
                df = pd.read_csv(io.StringIO(txt))
                if df is not None and not df.empty:
                    # normalize
                    ts_col = None
                    for name in ("ts", "quoteDate", "quotedate", "timestamp", "time", "minute"):
                        if name in df.columns:
                            ts_col = name
                            break
                    if ts_col:
                        df["ts"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
                        if "expiry" in df.columns:
                            df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce").dt.date
                        elif expiry is not None:
                            df["expiry"] = expiry
                        else:
                            for alt in ("expiration", "exp", "exp_date", "expdate", "expirationdate"):
                                if alt in df.columns:
                                    df["expiry"] = pd.to_datetime(df[alt], errors="coerce").dt.date
                                    break
                        if "expiry" in df.columns:
                            return df
        except Exception:
            pass  # fall through to range/loop

    # 2) Live Intraday range endpoint for the session window (9:30 ET to 16:00 or now)
    try:
        now_et = dt.datetime.now(ET_TZ)
        open_et = dt.datetime.combine(trade_date, dt.time(9, 30), tzinfo=ET_TZ)
        end_et = min(now_et, dt.datetime.combine(trade_date, dt.time(16, 0), tzinfo=ET_TZ))
        if end_et >= open_et:
            df_range = fetch_intraday_implied_monies_history(ticker, open_et, end_et)
            if df_range is not None and not df_range.empty:
                return df_range
    except Exception:
        pass

    # 3) Final fallback: iterate one-minute calls across the session
    return fetch_monies_minutes_for_day_via_one_minute(ticker, trade_date, expiry, max_minutes=None)

# -----------------------------------------------------------------------------
# (Optional) Historical intraday helper returning a whole‑day DataFrame by string
# -----------------------------------------------------------------------------

def fetch_historical_day_monies_df(
    date_yyyymmdd: str,
    ticker: str = "SPX",
    expiry: Optional[dt.date] = None,
) -> Optional[pd.DataFrame]:
    """
    Convenience wrapper identical to fetch_monies_minutes_for_day but taking a YYYYMMDD string.
    """
    try:
        d = dt.datetime.strptime(date_yyyymmdd, "%Y%m%d").date()
    except ValueError:
        d = dt.date.fromisoformat(date_yyyymmdd)
    return fetch_monies_minutes_for_day(ticker=ticker, trade_date=d, expiry=expiry)
