# packages/shared/ingest/monies_ingest.py
from __future__ import annotations

import os
import json
import datetime as dt
from typing import Dict, Any, Iterable, List, Optional

import pandas as pd
from sqlalchemy import text

# ---- project helpers (works for both legacy and monorepo paths) ----
try:
    from utils.data_io import tx, init_orats_monies_schema, get_engine  # noqa: F401
except ImportError:  # monorepo path
    from packages.shared.utils.data_io import tx, init_orats_monies_schema, get_engine  # noqa: F401

# time helpers from your ORATS client
from packages.shared.options_orats import pt_minute_to_et, ET_TZ


# ============================================================================
# NORMALIZE ONE-MINUTE PAYLOAD (exactly one minute, many expiries)
# ============================================================================

def _to_utc_py(ts_like: Any) -> dt.datetime:
    ts = pd.Timestamp(ts_like)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.to_pydatetime()


def _normalize_orats_minute(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert a single ORATS 'monies' minute dict into DB rows (one per expiry).

    Expected shape (keys are tolerant; tweak here once if needed):
        {
          "ticker": "SPX",
          "minute": "2025-11-04T14:31:00Z",        # sometimes 'ts'
          "underlying": 5987.12,
          "rf_rate": 0.0451,
          "div_yield": 0.0132,
          "expirations": [
            { "expiry_date": "2025-12-20", "dte": 46, "forward": 5989.5,
              "smile": {
                  "p10":0.192, ..., "atm":0.158, "c10":0.159,
                  "vol90":0.192, ..., "vol10":0.159
              }
            }, ...
          ]
        }

    We *do not* touch the smile values here; whatever is in the 'smile' dict
    (p10/c10/atm and/or volXX) is stored verbatim as JSON.
    """
    # timestamp
    minute_key = "minute" if "minute" in payload else "ts"
    minute_ts = _to_utc_py(payload[minute_key])

    # common fields
    ticker = payload.get("ticker", "SPX")
    underlying = payload.get("underlying")
    rf_rate = payload.get("rf_rate")
    div_yield = payload.get("div_yield")

    # expirations list (allow alternate key)
    expirations = payload.get("expirations") or payload.get("exp_list") or []
    rows: List[Dict[str, Any]] = []

    for e in expirations:
        exp_key = "expiry_date" if "expiry_date" in e else "expiry"
        expiry_date = pd.Timestamp(e[exp_key]).date()
        dte = int(e.get("dte", max((expiry_date - minute_ts.date()).days, 0)))
        smile: Dict[str, Any] = e.get("smile") or {}
        rows.append({
            "ticker": ticker,
            "trade_date": minute_ts.date(),
            "minute_ts": minute_ts,
            "expiry_date": expiry_date,
            "dte": dte,
            "underlying": underlying,
            "forward": e.get("forward"),
            "rf_rate": rf_rate,
            "div_yield": div_yield,
            "atm_iv": smile.get("atm"),
            # store smile as JSON text → CAST to jsonb in SQL
            "smile_json": json.dumps(smile) if not isinstance(smile, str) else smile,
        })

    return rows


# ============================================================================
# UPSERTS
# ============================================================================

UPSERT_SQL = """
INSERT INTO orats_monies_minute
(ticker, trade_date, minute_ts, expiry_date, dte,
 underlying, forward, rf_rate, div_yield, atm_iv, smile, grid, updated_at)
VALUES
(:ticker, :trade_date, :minute_ts, :expiry_date, :dte,
 :underlying, :forward, :rf_rate, :div_yield, :atm_iv, CAST(:smile_json AS jsonb), :grid, now())
ON CONFLICT (ticker, expiry_date, minute_ts) DO UPDATE SET
  dte        = EXCLUDED.dte,
  underlying = EXCLUDED.underlying,
  forward    = EXCLUDED.forward,
  rf_rate    = EXCLUDED.rf_rate,
  div_yield  = EXCLUDED.div_yield,
  atm_iv     = EXCLUDED.atm_iv,
  smile      = EXCLUDED.smile,
  grid       = EXCLUDED.grid,   -- ensure delta label overwrites
  updated_at = now();
"""


def upsert_minute_payload(payload: Dict[str, Any], *, grid: str = "delta") -> int:
    rows = _normalize_orats_minute(payload)
    if not rows:
        return 0
    for r in rows:
        r["grid"] = grid
    init_orats_monies_schema()
    with tx() as conn:
        conn.execute(text(UPSERT_SQL), rows)
    return len(rows)


# ============================================================================
# INGEST FROM DASHBOARD DATAFRAME (canonicalize + store volXX)
# ============================================================================

def upsert_from_dashboard_minute(
    df_minute: pd.DataFrame,
    ticker: str,
    *,
    minute_col_candidates=("ts", "quoteDate", "quotedate", "timestamp", "time", "minute"),
    expiry_col_candidates=("expiry", "expiration", "exp", "exp_date", "expdate", "expirationdate", "expirDate"),
    spot_col_candidates=("underlying", "underlyingprice", "stockPrice", "spot", "price", "stockprice"),
    store_volxx: Optional[bool] = None,  # default True: keep volXX for smile chart
) -> int:
    """
    Normalize a per-minute DataFrame (one row per expiry).

    We build a canonical smile dict with:
      - p10,p15,p20,p25,p30,p35,atm,c35,c30,c25,c20,c15,c10  (if available)
      - AND all volXX/mwVol columns (vol90..vol10, mwVol) from the DataFrame.

    The DB 'smile' JSON will therefore contain both the canonical P/C/ATM keys
    and the raw volXX keys. When reading back for the smile chart we prefer
    the volXX keys so the curve matches the live ORATS API exactly.
    """
    if df_minute is None or df_minute.empty:
        return 0

    # Default: store volXX unless explicitly disabled
    if store_volxx is None:
        store_volxx = bool(int(os.getenv("STORE_VOLXX", "1")))

    # case-insensitive lookup
    lower = {c.lower(): c for c in df_minute.columns}

    def pick(cands):
        for c in cands:
            if c.lower() in lower:
                return lower[c.lower()]
        return None

    minute_col = pick(minute_col_candidates)
    expiry_col = pick(expiry_col_candidates)
    spot_col   = pick(spot_col_candidates)
    if not minute_col or not expiry_col:
        raise RuntimeError(f"required columns missing; have: {list(df_minute.columns)}")

    # minute + trade_date
    ts0 = pd.to_datetime(df_minute[minute_col].iloc[0], utc=True, errors="coerce")
    if pd.isna(ts0):
        raise RuntimeError("invalid minute timestamp")
    minute_utc = pd.Timestamp(ts0).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    trade_date = pd.Timestamp(ts0).tz_convert("UTC").date()

    # underlying
    underlying = None
    if spot_col and spot_col in df_minute.columns:
        s = pd.to_numeric(df_minute[spot_col], errors="coerce")
        if s.notna().any():
            underlying = float(s.dropna().iloc[0])

    # maps to derive the canonical smile (P/C/ATM)
    pc_keys = ["p10", "p15", "p20", "p25", "p30", "p35", "atm",
               "c35", "c30", "c25", "c20", "c15", "c10"]
    vol_to_pc = {
        "vol90": "p10", "vol85": "p15", "vol80": "p20", "vol75": "p25",
        "vol70": "p30", "vol65": "p35",
        "vol50": "atm",
        "vol35": "c35", "vol30": "c30", "vol25": "c25", "vol20": "c20",
        "vol15": "c15", "vol10": "c10",
    }

    def get_val(row, *cands):
        """Try multiple column names (case-insensitive); return float or None."""
        for name in cands:
            col = lower.get(str(name).lower())
            if col is not None:
                v = pd.to_numeric(row[col], errors="coerce")
                if pd.notna(v):
                    return float(v)
        return None

    df = df_minute.copy()
    df[expiry_col] = pd.to_datetime(df[expiry_col], errors="coerce").dt.date

    expirations: List[Dict[str, Any]] = []
    for _, row in df.dropna(subset=[expiry_col]).iterrows():
        exp_date = row[expiry_col]
        dte = int(max((exp_date - trade_date).days, 0)) if isinstance(exp_date, dt.date) else 0

        # Build a canonical smile dict with P/C/ATM keys...
        smile: Dict[str, float] = {}

        # 1) prefer P/C/ATM columns if present
        for k in pc_keys:
            v = get_val(row, k)  # e.g., "p10", "atm", "c10"
            if v is not None:
                smile[k] = v

        # 2) fill any missing from volXX columns
        for vol_col, pc_key in vol_to_pc.items():
            if pc_key not in smile:
                v = get_val(row, vol_col)
                if v is not None:
                    smile[pc_key] = v

        # 3) also store raw volXX/mwVol alongside (preferred for smile chart)
        if store_volxx:
            for c in row.index:
                lc = str(c).lower()
                if lc == "mwvol" or lc.startswith("vol"):
                    v = pd.to_numeric(row[c], errors="coerce")
                    if pd.notna(v):
                        # e.g., "vol90", "vol50", "mwVol"
                        smile[str(c)] = float(v)

        expirations.append({
            "expiry_date": exp_date.isoformat(),
            "dte": dte,
            "forward": None,
            "smile": smile,
        })

    payload = {
        "ticker": ticker,
        "minute": minute_utc,
        "underlying": underlying,
        "rf_rate": None,
        "div_yield": None,
        "expirations": expirations,
    }
    return upsert_minute_payload(payload, grid="delta")


# ============================================================================
# BULK UPSERT (e.g., historical iterator)
# ============================================================================

def upsert_many_minutes(payloads: Iterable[Dict[str, Any]], *, grid: str = "delta") -> int:
    """Upsert a sequence of minute payloads. Returns total rows affected."""
    init_orats_monies_schema()
    total = 0
    with tx() as conn:
        for p in payloads:
            rows = _normalize_orats_minute(p)
            if rows:
                for r in rows:
                    r["grid"] = grid
                conn.execute(text(UPSERT_SQL), rows)
                total += len(rows)
    return total


# ============================================================================
# SIMPLE READERS (for charts/tests)
# ============================================================================

def read_minutes_df(
    trade_date: dt.date,
    ticker: str,
    expiry_date: Optional[dt.date] = None,
    start: Optional[dt.datetime] = None,
    end: Optional[dt.datetime] = None,
) -> pd.DataFrame:
    """Return DataFrame with: minute_ts, expiry_date, underlying, atm_iv, smile (json/dict)."""
    where = ["ticker = :t", "trade_date = :d"]
    params: Dict[str, Any] = {"t": ticker, "d": trade_date}
    if expiry_date:
        where.append("expiry_date = :e")
        params["e"] = expiry_date
    if start:
        where.append("minute_ts >= :s")
        params["s"] = start
    if end:
        where.append("minute_ts <= :u")
        params["u"] = end

    sql = f"""
      SELECT minute_ts, expiry_date, underlying, atm_iv, smile
        FROM orats_monies_minute
       WHERE {' AND '.join(where)}
       ORDER BY minute_ts;
    """

    with tx() as conn:
        recs = conn.execute(text(sql), params).mappings().all()

    return pd.DataFrame(recs)


# Map DB smile -> volXX so your callback can keep using row["vol50"], etc.
_DB_TO_VOLXX = {
    "atm": "vol50",
    "p10": "vol90", "p15": "vol85", "p20": "vol80", "p25": "vol75",
    "p30": "vol70", "p35": "vol65",
    "c35": "vol35", "c30": "vol30", "c25": "vol25", "c20": "vol20",
    "c15": "vol15", "c10": "vol10",
}


def read_minute_expiry_df_from_db(
    ticker: str,
    trade_date_iso: str,   # "YYYY-MM-DD"
    expiration_iso: str,   # "YYYY-MM-DD"
    hhmm_pt: str,          # "HH:MM" in PT
) -> pd.DataFrame:
    """
    Return a single-row DataFrame for (minute, expiry) with columns like vol50..vol10.

    Behavior:
      - If the DB 'smile' JSON contains volXX keys (vol90..vol10, vol50), we use
        those directly — this matches the live ORATS API smile 1:1.
      - If NOT, but it has canonical P/C/ATM keys, we map those to volXX and
        mark the row as '__legacy_smile' = True so the callback can choose to
        re-fetch from the API for *today's* date.
      - As a last resort, we look for dedicated volXX columns in the row.
    """
    # lazy imports to avoid cycles
    from packages.shared.cache.day_cache import get_day_df, refresh_today_if_needed

    # Convert PT minute -> ET tz-aware -> UTC window [floor, floor+60s)
    ts_et = pt_minute_to_et(trade_date_iso, hhmm_pt)
    ts = pd.Timestamp(ts_et)
    if ts.tz is None:
        ts = ts.tz_localize(ET_TZ)
    ts_floor_utc = ts.tz_convert("UTC").floor("min")
    ts_end_utc = ts_floor_utc + pd.Timedelta(minutes=1)

    # If it's today, allow an incremental refresh; else just read cached day.
    is_today = (trade_date_iso == dt.date.today().isoformat())
    day_df = refresh_today_if_needed(ticker, trade_date_iso, expiration_iso)[0] if is_today \
             else get_day_df(ticker, trade_date_iso, expiration_iso)

    if day_df is None or day_df.empty:
        return pd.DataFrame()

    # Slice the cached day in memory
    mask = (day_df["minute_ts"] >= ts_floor_utc) & (day_df["minute_ts"] < ts_end_utc)
    sel = day_df.loc[mask].tail(1)
    if sel.empty:
        return pd.DataFrame()
    r = sel.iloc[0]

    # Base fields
    ts_db = pd.Timestamp(r["minute_ts"])
    ts_db = ts_db.tz_convert("UTC") if ts_db.tz is not None else ts_db.tz_localize("UTC")

    out: Dict[str, Any] = {
        "quoteDate": ts_db.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "expiry": expiration_iso,
        "underlying": float(r["underlying"]) if pd.notna(r.get("underlying")) else None,
        "__src": "db-cache",
        "__legacy_smile": False,
    }

    # Decode smile JSON if present
    smile = r.get("smile")
    if isinstance(smile, str):
        try:
            smile = json.loads(smile)
        except Exception:
            smile = None

    vol_keys = [
        "vol90", "vol85", "vol80", "vol75", "vol70", "vol65",
        "vol50",
        "vol35", "vol30", "vol25", "vol20", "vol15", "vol10",
    ]

    wrote = 0
    legacy_smile = False

    if isinstance(smile, dict) and smile:
        # normalize keys to lowercase for lookup
        lower_smile = {str(k).lower(): v for k, v in smile.items()}

        # Does JSON already have volXX keys?
        has_volxx = any(k.lower() in lower_smile for k in vol_keys)

        # Does JSON have canonical P/C/ATM keys?
        pc_keys_lower = [
            "atm", "p10", "p15", "p20", "p25", "p30", "p35",
            "c35", "c30", "c25", "c20", "c15", "c10",
        ]
        has_pc = any(pk in lower_smile for pk in pc_keys_lower)

        # Legacy definition: canonical keys present but no volXX in JSON
        legacy_smile = (not has_volxx) and has_pc

        if has_volxx:
            # 1) Prefer raw volXX keys if any are present
            for vk in vol_keys:
                v = lower_smile.get(vk.lower())
                if v is not None:
                    try:
                        out[vk] = float(v)
                        wrote += 1
                    except Exception:
                        pass
        else:
            # 2) Fallback: map canonical P/C/ATM keys -> volXX
            for pc_key, vol_key in _DB_TO_VOLXX.items():
                v = lower_smile.get(pc_key)
                if v is not None:
                    try:
                        out[vol_key] = float(v)
                        wrote += 1
                    except Exception:
                        pass

    out["__legacy_smile"] = legacy_smile

    # 3) Last-resort fallback: copy any dedicated volXX columns from the row
    if wrote == 0:
        for c in vol_keys:
            if c in sel.columns and pd.notna(r.get(c)):
                out[c] = float(r[c])

    return pd.DataFrame([out])
