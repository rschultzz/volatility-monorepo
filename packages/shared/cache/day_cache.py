from __future__ import annotations

import os
import json
import datetime as dt
from typing import Dict, Tuple, Any, List, Optional

import pandas as pd
from sqlalchemy import text

# Works for both legacy and monorepo paths
try:
    from utils.data_io import tx  # noqa: F401
except ImportError:
    from packages.shared.utils.data_io import tx  # noqa: F401

# ----------------------------------------------------------------------------
# Simple in‑memory day cache (DB → DataFrame with volXX columns)
# ----------------------------------------------------------------------------

# Canonical p/c/atm → volXX mapping (17 buckets)
_PC_TO_VOLXX: Dict[str, str] = {
    # put side
    "p45": "vol55", "p40": "vol60", "p35": "vol65", "p30": "vol70",
    "p25": "vol75", "p20": "vol80", "p15": "vol85", "p10": "vol90",
    # atm
    "atm": "vol50",
    # call side
    "c45": "vol45", "c40": "vol40", "c35": "vol35", "c30": "vol30",
    "c25": "vol25", "c20": "vol20", "c15": "vol15", "c10": "vol10",
}

# Vol columns order useful for plotting
_VOL_COLS_ORDER: List[str] = [
    "vol90", "vol85", "vol80", "vol75", "vol70", "vol65", "vol60", "vol55",
    "vol50",
    "vol45", "vol40", "vol35", "vol30", "vol25", "vol20", "vol15", "vol10",
]

# In‑memory cache
_MAX_ENTRIES = int(os.getenv("DAY_CACHE_MAX", "12"))
_DAY_CACHE: Dict[Tuple[str, str, str], pd.DataFrame] = {}
_LRU: List[Tuple[str, str, str]] = []

_DEBUG = os.getenv("DEBUG_DAYCACHE", "0") == "1"


def _log(msg: str) -> None:
    if _DEBUG:
        print(f"[daycache] {msg}")


def _key(ticker: str, trade_date: str | dt.date, expiry: str | dt.date) -> Tuple[str, str, str]:
    t = str(ticker).upper()
    d = trade_date if isinstance(trade_date, str) else trade_date.isoformat()
    e = expiry if isinstance(expiry, str) else expiry.isoformat()
    return (t, d, e)


def _touch(k: Tuple[str, str, str]) -> None:
    if k in _LRU:
        _LRU.remove(k)
    _LRU.append(k)
    while len(_LRU) > _MAX_ENTRIES:
        old = _LRU.pop(0)
        _DAY_CACHE.pop(old, None)
        _log(f"evicted {old}")


# ----------------------------------------------------------------------------
# DB → records → DataFrame helpers
# ----------------------------------------------------------------------------

def _ensure_utc(ts_like: Any) -> pd.Timestamp:
    ts = pd.Timestamp(ts_like)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _expand_smile(smile: Any) -> Dict[str, float]:
    """Return a dict of volXX columns from canonical p/c/atm keys in `smile`."""
    if smile is None:
        return {}
    if isinstance(smile, str):
        try:
            smile = json.loads(smile)
        except Exception:
            return {}
    out: Dict[str, float] = {}
    for k, v in (smile or {}).items():
        if v is None:
            continue
        kk = str(k).lower()
        vol = _PC_TO_VOLXX.get(kk)
        if vol:
            try:
                out[vol] = float(v)
            except Exception:
                pass
    return out


def _records_to_day_df(recs: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for r in recs:
        minute_ts = _ensure_utc(r["minute_ts"]).floor("min")
        base = {
            "minute_ts": minute_ts,
            "underlying": r.get("underlying"),
            "expiry_date": r.get("expiry_date"),
        }
        base.update(_expand_smile(r.get("smile")))
        rows.append(base)

    if not rows:
        return pd.DataFrame(columns=["minute_ts", "underlying", "expiry_date", *_VOL_COLS_ORDER])

    df = pd.DataFrame(rows).sort_values("minute_ts").reset_index(drop=True)
    # Ensure all vol columns exist
    for c in _VOL_COLS_ORDER:
        if c not in df.columns:
            df[c] = pd.NA
    # Keep tidy column order
    cols = ["minute_ts", "underlying", "expiry_date", *_VOL_COLS_ORDER]
    return df[cols]


# packages/shared/cache/day_cache.py


# packages/shared/cache/day_cache.py

def _load_day_from_db(ticker: str, trade_date_iso: str, expiry_iso: str) -> pd.DataFrame:
    from sqlalchemy import text
    try:
        from packages.shared.utils.data_io import tx
    except Exception:
        from utils.data_io import tx

    d = dt.date.fromisoformat(trade_date_iso)
    e = dt.date.fromisoformat(expiry_iso)
    e_plus = e + dt.timedelta(days=1)  # handle Saturday monthlies

    sql = text("""
        SELECT minute_ts, expiry_date, underlying, rf_rate, div_yield, smile
          FROM orats_monies_minute
         WHERE ticker = :t
           AND trade_date = :d
           AND (expiry_date = :e OR expiry_date = :e_plus)
           AND (grid = 'delta' OR grid IS NULL)
         ORDER BY minute_ts
    """)
    with tx() as c:
        recs = c.execute(sql, {"t": ticker, "d": d, "e": e, "e_plus": e_plus}).mappings().all()

    df = pd.DataFrame(recs)
    if not df.empty and not pd.api.types.is_datetime64tz_dtype(df["minute_ts"]):
        df["minute_ts"] = pd.to_datetime(df["minute_ts"], utc=True, errors="coerce")

    # helpful debug: show which expiry_dates we actually loaded
    loaded_expiries = sorted({str(x) for x in df.get("expiry_date", [])}) if not df.empty else []
    print(f"[daycache] loaded {len(df)} rows (expiries={loaded_expiries}) for {ticker} {trade_date_iso} {expiry_iso}")
    return df




# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def get_day_df(ticker: str, trade_date: str | dt.date, expiry: str | dt.date) -> pd.DataFrame:
    """DB‑first: return the entire day DataFrame for (ticker, trade_date, expiry).
    Caches the DataFrame in memory. No API calls here; we'll add fallback later.
    """
    k = _key(ticker, trade_date, expiry)
    if k in _DAY_CACHE:
        _touch(k)
        return _DAY_CACHE[k]

    df = _load_day_from_db(*k)
    _DAY_CACHE[k] = df
    _touch(k)
    return df


def refresh_today_if_needed(ticker: str, trade_date: str | dt.date, expiry: str | dt.date) -> Tuple[pd.DataFrame, int]:
    """For today: append any new minutes that have appeared in the DB since the cached max ts.
    Returns (updated_df, n_appended). If the key isn't cached, it loads it first.
    """
    k = _key(ticker, trade_date, expiry)
    df = get_day_df(*k)  # ensures cached
    if df.empty:
        return df, 0

    last_ts = df["minute_ts"].max()
    sql = text(
        """
        SELECT minute_ts, expiry_date, underlying, smile
          FROM orats_monies_minute
         WHERE ticker = :t
           AND trade_date = :d
           AND expiry_date = :e
           AND minute_ts > :last_ts
         ORDER BY minute_ts
        """
    )
    with tx() as conn:
        recs = conn.execute(sql, {"t": k[0], "d": k[1], "e": k[2], "last_ts": last_ts.to_pydatetime()}).mappings().all()

    add_df = _records_to_day_df(list(recs))
    if add_df.empty:
        return df, 0

    new_df = pd.concat([df, add_df], ignore_index=True)
    new_df = new_df.drop_duplicates(subset=["minute_ts"], keep="last").sort_values("minute_ts").reset_index(drop=True)
    _DAY_CACHE[k] = new_df
    _touch(k)
    _log(f"appended {len(add_df)} new rows; total {len(new_df)}")
    return new_df, len(add_df)


def clear_day_cache(ticker: Optional[str] = None, trade_date: Optional[str] = None, expiry: Optional[str] = None) -> None:
    """Clear the entire cache, or a specific key if all three parts are provided."""
    if ticker and trade_date and expiry:
        k = _key(ticker, trade_date, expiry)
        _DAY_CACHE.pop(k, None)
        if k in _LRU:
            _LRU.remove(k)
        _log(f"cleared {k}")
    else:
        _DAY_CACHE.clear(); _LRU.clear()
        _log("cleared all")


def cache_info() -> Dict[str, Any]:
    return {
        "size": len(_DAY_CACHE),
        "keys": list(_LRU),
        "max_entries": _MAX_ENTRIES,
    }
