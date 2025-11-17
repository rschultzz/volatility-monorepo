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
              "smile": {  # we store this dict VERBATIM as JSON
                "p10": 0.19, "p15": 0.18, ..., "atm": 0.16, "c10": 0.17,
                "vol90": 0.31, "vol85": ..., "vol10": 0.32, "mwVol": 0.27
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
    # tag rows
    for r in rows:
        r["grid"] = grid
    init_orats_monies_schema()
    with tx() as conn:
        conn.execute(text(UPSERT_SQL), rows)
    return len(rows)


# ============================================================================
# INGEST FROM DASHBOARD DATAFRAME
#   → build smile dict DIRECTLY from ORATS columns (no P/C → volXX remapping)
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
    Normalize a per-minute DataFrame (one row per expiry) coming from the
    ORATS 'monies' minute endpoint.

    IMPORTANT:
      * Each row becomes a 'smile' dict that:
          - copies any P/C/ATM columns directly (p10, p15, ..., atm, c10, etc)
          - copies ALL volXX and mwVol columns directly (vol100..vol0, mwVol)
      * There is NO mapping from P/C/ATM → volXX here. The volXX keys in the
        DB always come straight from the ORATS vol grid.

    Also IMPORTANT:
      The ORATS DataFrame for a single expiry can have MANY rows. We only want
      ONE smile per (minute, expiry), so we collapse to *one row per expiry*
      before building the smile.
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

    # minute + trade_date (UTC)
    ts0 = pd.to_datetime(df_minute[minute_col].iloc[0], utc=True, errors="coerce")
    if pd.isna(ts0):
        raise RuntimeError("invalid minute timestamp")
    ts0 = pd.Timestamp(ts0).tz_convert("UTC")
    minute_utc = ts0.strftime("%Y-%m-%dT%H:%M:%SZ")
    trade_date = ts0.date()

    # underlying
    underlying = None
    if spot_col and spot_col in df_minute.columns:
        s = pd.to_numeric(df_minute[spot_col], errors="coerce")
        if s.notna().any():
            underlying = float(s.dropna().iloc[0])

    # Canonical P/C/ATM names we care about
    PC_CANON = {
        "p10", "p15", "p20", "p25", "p30", "p35",
        "atm",
        "c35", "c30", "c25", "c20", "c15", "c10",
    }

    # --- collapse to ONE row per expiry ---
    df = df_minute.copy()
    df[expiry_col] = pd.to_datetime(df[expiry_col], errors="coerce").dt.date
    df = df.dropna(subset=[expiry_col])

    # For each expiry_date, keep the *first* row — this matches your debug use of head(1)
    df = df.sort_values(by=[expiry_col])
    df = df.drop_duplicates(subset=[expiry_col], keep="first")

    expirations: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        exp_date = row[expiry_col]
        dte = int(max((exp_date - trade_date).days, 0)) if isinstance(exp_date, dt.date) else 0

        smile: Dict[str, float] = {}

        # 1) Copy P/C/ATM columns directly (canonicalised to lowercase)
        for col in row.index:
            lc = str(col).lower()
            if lc in PC_CANON:
                v = pd.to_numeric(row[col], errors="coerce")
                if pd.notna(v):
                    smile[lc] = float(v)

        # 2) Copy volXX + mwVol columns directly from ORATS (if requested)
        if store_volxx:
            for col in row.index:
                lc = str(col).lower()
                if lc == "mwvol" or lc.startswith("vol"):
                    v = pd.to_numeric(row[col], errors="coerce")
                    if pd.notna(v):
                        # keep the volXX key name as-is (lowercase)
                        smile[lc] = float(v)

        # 3) Ensure we have an 'atm' key: if missing but vol50 exists, use that
        if "atm" not in smile and "vol50" in smile:
            smile["atm"] = smile["vol50"]

        # 4) Optional debug: see exactly what we're writing for each minute/expiry
        if os.getenv("DEBUG_MONIES_UPSERT") == "1":
            try:
                key_vols = {k: smile.get(k) for k in ("p10", "atm", "c10", "vol90", "vol50", "vol10")}
                print(
                    "[monies_upsert]",
                    "minute", minute_utc,
                    "exp", exp_date,
                    "key_vols", key_vols,
                )
            except Exception:
                # never let debug printing break ingestion
                pass

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
    # tag this as delta-grid data
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
        where.append("expiry_date = :e"); params["e"] = expiry_date
    if start:
        where.append("minute_ts >= :s"); params["s"] = start
    if end:
        where.append("minute_ts <= :u"); params["u"] = end

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
    "p10": "vol90", "p15": "vol85", "p20": "vol80", "p25": "vol75", "p30": "vol70", "p35": "vol65",
    "c35": "vol35", "c30": "vol30", "c25": "vol25", "c20": "vol20", "c15": "vol15", "c10": "vol10",
}


def read_minute_expiry_df_from_db(
    ticker: str,
    trade_date_iso: str,   # "YYYY-MM-DD"
    expiration_iso: str,   # "YYYY-MM-DD"
    hhmm_pt: str,          # "HH:MM" in PT
) -> pd.DataFrame:
    """
    Return a single-row DataFrame for (minute, expiry) with columns like vol50..vol10.

    BEHAVIOR:
      - We query the DB *directly* (no day-cache) and pick the row with
        minute_ts closest to the requested minute.
      - If the DB 'smile' JSON contains volXX keys (vol90..vol10, vol50), we use
        those directly — this matches the live ORATS API smile 1:1.
      - Otherwise, we fall back to canonical P/C/ATM keys (p10,p15,...,atm,c10)
        and map them to volXX using _DB_TO_VOLXX.
    """
    # Convert PT minute -> ET tz-aware -> UTC
    ts_et = pt_minute_to_et(trade_date_iso, hhmm_pt)
    ts = pd.Timestamp(ts_et)
    if ts.tz is None:
        ts = ts.tz_localize(ET_TZ)
    ts_utc = ts.tz_convert("UTC")

    trade_date = dt.date.fromisoformat(trade_date_iso)
    expiry_date = dt.date.fromisoformat(expiration_iso)

    # Query the closest minute from DB directly
    sql = """
      SELECT minute_ts, expiry_date, underlying, atm_iv, smile
        FROM orats_monies_minute
       WHERE ticker = :t
         AND trade_date = :d
         AND expiry_date = :e
       ORDER BY ABS(EXTRACT(EPOCH FROM (minute_ts - :ts)))
       LIMIT 1;
    """

    with tx() as conn:
        rec = conn.execute(
            text(sql),
            {"t": ticker, "d": trade_date, "e": expiry_date, "ts": ts_utc.to_pydatetime()},
        ).mappings().first()

    if not rec:
        return pd.DataFrame()

    r = rec

    # Base fields
    ts_db = pd.Timestamp(r["minute_ts"])
    ts_db = ts_db.tz_convert("UTC") if ts_db.tz is not None else ts_db.tz_localize("UTC")

    out: Dict[str, Any] = {
        "quoteDate": ts_db.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "expiry": expiration_iso,
        "underlying": float(r["underlying"]) if r.get("underlying") is not None else None,
        "__src": "db-direct",
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
    if isinstance(smile, dict) and smile:
        # normalize keys to lowercase for lookup
        lower_smile = {str(k).lower(): v for k, v in smile.items()}

        # 1) Prefer raw volXX keys if any are present
        has_volxx = any(k.lower() in lower_smile for k in vol_keys)
        if has_volxx:
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

    # 3) Last-resort fallback: nothing in smile → just return ATM if present
    if wrote == 0 and r.get("atm_iv") is not None:
        try:
            out["vol50"] = float(r["atm_iv"])
        except Exception:
            pass

    return pd.DataFrame([out])
