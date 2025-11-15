# apps/web/modules/Smile/callbacks.py
from __future__ import annotations

import datetime as dt
import math
import os
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import Input, Output

from packages.shared.options_orats import fetch_one_minute_monies, pt_minute_to_et, PT_TZ, ET_TZ
from packages.shared.surface_compare import k_for_abs_delta
from packages.shared.ingest.monies_ingest import upsert_from_dashboard_minute, read_minute_expiry_df_from_db
from packages.shared.cache.day_cache import get_day_df, refresh_today_if_needed

# ---- App IDs ----
TRADE_DATE_ID = "trade-date"
EXPIRATION_ID = "expiration-date-pick"
SMILE_TIME_INPUT = "smile-time-input"
SMILE_GRAPH = "SMILE_GRAPH"
EXPECTED_TOGGLE_ID = "expected-ss-toggle"
CLOCK_ID = "CLOCK"

TICKER = "SPX"
EPS_T = 1e-4
DEBUG = os.getenv("DEBUG_SMILE", "0") == "1"

COLORWAY = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"
]

# Map live P/C/ATM -> volXX when API rows don’t include volXX
_PC_TO_VOL = {
    "p10": "vol90", "p15": "vol85", "p20": "vol80", "p25": "vol75", "p30": "vol70", "p35": "vol65",
    "atm": "vol50",
    "c35": "vol35", "c30": "vol30", "c25": "vol25", "c20": "vol20", "c15": "vol15", "c10": "vol10",
}

def _log(msg: str) -> None:
    if DEBUG:
        print(f"[smile] {msg}")

# -----------------------------------------------------------------------------
# Minute LRU cache (process-local) to make repeated clicks instant
# -----------------------------------------------------------------------------
_MINUTE_CACHE: Dict[tuple, tuple[pd.DataFrame, float]] = {}
_MINUTE_CACHE_MAX = 600  # entries
_MINUTE_CACHE_TTL = 300  # seconds

def _cache_key(trade_date_iso: str, expiration_iso: str, hhmm_pt: str) -> tuple:
    # Use the actual minute floor UTC so key is stable across TZs
    ts_et = pt_minute_to_et(trade_date_iso, hhmm_pt)
    ts = pd.Timestamp(ts_et)
    if ts.tz is None:
        ts = ts.tz_localize(ET_TZ)
    ts_floor_utc = ts.tz_convert("UTC").floor("min")
    return (TICKER, trade_date_iso, expiration_iso, ts_floor_utc.to_pydatetime())

def _cache_get(k: tuple) -> Optional[pd.DataFrame]:
    rec = _MINUTE_CACHE.get(k)
    if not rec:
        return None
    df, ts = rec
    if (dt.datetime.now().timestamp() - ts) > _MINUTE_CACHE_TTL:
        try:
            del _MINUTE_CACHE[k]
        except Exception:
            pass
        return None
    return df

def _cache_put(k: tuple, df: pd.DataFrame) -> None:
    if len(_MINUTE_CACHE) >= _MINUTE_CACHE_MAX:
        # Drop an arbitrary item (simple LRU-ish)
        _MINUTE_CACHE.pop(next(iter(_MINUTE_CACHE)))
    _MINUTE_CACHE[k] = (df, dt.datetime.now().timestamp())

# ---------- helpers ----------
def _ensure_volxx(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    lower = {c.lower(): c for c in out.columns}
    def get(name: str):
        col = lower.get(name.lower())
        return col if col in out.columns else None
    for pc, vol in _PC_TO_VOL.items():
        if vol not in out.columns:
            src = get(pc)
            if src is not None:
                out[vol] = out[src]
    return out

def _available_buckets(row: pd.Series) -> List[int]:
    bucks = []
    for c in row.index:
        if isinstance(c, str) and c.startswith("vol") and c[3:].isdigit():
            n = int(c[3:])
            if 1 <= n <= 99:
                bucks.append(n)
    return sorted(bucks, reverse=True)

def _bucket_labels_order(buckets: List[int]) -> Tuple[List[int], List[str]]:
    puts = [n for n in buckets if n >= 50]
    calls = [n for n in buckets if n < 50]
    order = puts + calls
    labels = []
    for n in order:
        if n > 50:
            labels.append(f"P{100-n}")
        elif n == 50:
            labels.append("ATM")
        else:
            labels.append(f"C{n}")
    return order, labels

def _years_to_exp(ts_et: dt.datetime, expiration_iso: str) -> float:
    exp_date = dt.date.fromisoformat(expiration_iso)
    rem = dt.datetime.combine(exp_date, dt.time(0, 0)) - ts_et.replace(tzinfo=None)
    T = max(0.0, rem.days/365.0 + rem.seconds/(365.0*24*3600))
    return max(T, EPS_T)

def _k_grid_for_row(row: pd.Series, T: float) -> Tuple[np.ndarray, np.ndarray]:
    buckets = _available_buckets(row)
    if not buckets or 50 not in buckets:
        raise ValueError("row missing buckets/ATM")
    atm = float(row["vol50"])
    k_list, s_list = [], []
    for n in buckets:
        if n == 50:
            k = 0.0
        else:
            p, is_put = ((100 - n) / 100.0, True) if n > 50 else (n / 100.0, False)
            k = k_for_abs_delta(p, is_put=is_put, sigma=atm, T=T)
        k_list.append(k)
        s_list.append(float(row[f"vol{n}"]))
    k = np.array(k_list, float)
    s = np.array(s_list, float)
    mask = np.concatenate(([True], np.diff(k) > 1e-12))
    return k[mask], s[mask]

def _interp_linear_extrap(x: float, xs: np.ndarray, ys: np.ndarray) -> float:
    if x <= xs[0]:
        x0, x1, y0, y1 = xs[0], xs[1], ys[0], ys[1]
        return float(y0 + (y1 - y0) * (x - x0) / (x1 - x0))
    if x >= xs[-1]:
        x0, x1, y0, y1 = xs[-2], xs[-1], ys[-2], ys[-1]
        return float(y1 + (y1 - y0) * (x - x1) / (x1 - x0))
    return float(np.interp(x, xs, ys))

def _expected_curve_shifted(prev_row: pd.Series,
                            prev_T: float,
                            prev_stock: float,
                            now_row: pd.Series,
                            now_T: float,
                            now_stock: float) -> Tuple[List[str], np.ndarray, float]:
    k_prev, s_prev = _k_grid_for_row(prev_row, prev_T)
    k_shift = math.log(now_stock / prev_stock) if (prev_stock and now_stock) else 0.0

    buckets = _available_buckets(now_row)
    buckets = [n for n in buckets if n not in (95, 5)]
    if not buckets or 50 not in buckets:
        raise ValueError("now row missing buckets/ATM")
    order, labels = _bucket_labels_order(buckets)

    atm_now = float(now_row["vol50"])
    shape_vals = []
    for n in order:
        if n == 50:
            k_now = 0.0
        else:
            p, is_put = ((100 - n) / 100.0, True) if n > 50 else (n / 100.0, False)
            k_now = k_for_abs_delta(p, is_put=is_put, sigma=atm_now, T=now_T)
        shape_vals.append(_interp_linear_extrap(k_now + k_shift, k_prev, s_prev))
    shape = np.array(shape_vals, float)

    exp_atm_shape = _interp_linear_extrap(k_shift, k_prev, s_prev)
    ret_frac = (now_stock - prev_stock) / prev_stock
    level_shift_pp = max(-6.0, min(6.0, (-ret_frac) * 100.0 * 4.5))
    atm_exp = exp_atm_shape + level_shift_pp / 100.0
    expected = shape + (atm_exp - exp_atm_shape)
    return labels, expected * 100.0, atm_exp * 100.0

# ---------- unified minute fetcher (CACHE -> DB/day-cache -> API) ----------
def _minute_df(trade_date_iso: str, expiration_iso: str, hhmm_pt: str) -> tuple[pd.DataFrame, str]:
    """
    Returns a 1-row DataFrame with volXX columns and a source tag:
      'db'  -> from day-cache/DB OR from in-process minute cache
      'api' -> live API fallback (only if DB miss)
      'none'-> nothing found
    """
    key = _cache_key(trade_date_iso, expiration_iso, hhmm_pt)
    cached = _cache_get(key)
    if cached is not None and not cached.empty:
        return _ensure_volxx(cached), "db"  # treat memory as DB for legend clarity

    # Try day-cache / DB
    df_db = read_minute_expiry_df_from_db(TICKER, trade_date_iso, expiration_iso, hhmm_pt)
    if df_db is not None and not df_db.empty:
        df_db = _ensure_volxx(df_db)
        _cache_put(key, df_db)
        return df_db, "db"

    # API -> upsert -> re-read
    ts_et = pt_minute_to_et(trade_date_iso, hhmm_pt)
    df_api = fetch_one_minute_monies(ts_et, TICKER, expiration_iso)
    if df_api is not None and not df_api.empty:
        try:
            upsert_from_dashboard_minute(df_api, ticker=TICKER)
            df_db2 = read_minute_expiry_df_from_db(TICKER, trade_date_iso, expiration_iso, hhmm_pt)
            if df_db2 is not None and not df_db2.empty:
                df_db2 = _ensure_volxx(df_db2)
                _cache_put(key, df_db2)
                return df_db2, "db"
        except Exception as e:
            _log(f"upsert error @ {hhmm_pt}: {e}")
        # last resort: plot from API and cache briefly
        df_api = _ensure_volxx(df_api)
        _cache_put(key, df_api)
        return df_api, "api"

    return pd.DataFrame(), "none"

# ----------------- main callback -----------------
def register_callbacks(app):
    @app.callback(
        Output(SMILE_GRAPH, "figure"),
        Input(TRADE_DATE_ID, "date"),
        Input(EXPIRATION_ID, "date"),
        Input(SMILE_TIME_INPUT, "value"),
        Input(EXPECTED_TOGGLE_ID, "value"),
        Input(CLOCK_ID, "n_intervals"),
    )
    def render_smile(trade_date_iso, expiration_iso, times_pt, expected_value, _tick):
        expected_on = (expected_value != "off")

        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=40),
            title=f"ORATS Smile Grid — {trade_date_iso or ''} (Exp: {expiration_iso or ''})",
            xaxis_title="Bucket (P10 … ATM … C10)",
            yaxis_title="IV (%)",
            legend=dict(orientation="v", x=1.02, y=1.0, bgcolor="rgba(0,0,0,0)"),
            colorway=COLORWAY,
        )

        if not trade_date_iso or not expiration_iso:
            return fig

        # ensure a “now” for today
        if not times_pt:
            times_pt = ["06:31"]
        now_pt = dt.datetime.now(PT_TZ)
        if trade_date_iso == now_pt.date().isoformat():
            hhmm = now_pt.strftime("%H:%M")
            if "06:30" <= hhmm <= "13:00":
                times_pt = sorted(set(times_pt + [hhmm]))
        times_sorted = sorted(times_pt)

        prev_row = None
        prev_stock = None
        prev_T = None
        traces = 0

        for i, hhmm_pt in enumerate(times_sorted):
            df_now, source = _minute_df(trade_date_iso, expiration_iso, hhmm_pt)
            if df_now is None or df_now.empty:
                _log(f"{hhmm_pt} PT: no data from DB/API")
                continue

            row_now = df_now.iloc[0]
            if "vol50" not in row_now.index or pd.isna(row_now["vol50"]):
                _log(f"{hhmm_pt} PT: missing vol50")
                continue

            color = COLORWAY[i % len(COLORWAY)]
            buckets_now = _available_buckets(row_now)
            buckets_now = [n for n in buckets_now if n not in (95, 5)]
            if not buckets_now:
                _log(f"{hhmm_pt} PT: no volXX buckets to plot")
                continue

            order_now, labels_now = _bucket_labels_order(buckets_now)
            y_now = [float(row_now.get(f"vol{n}")) * 100.0 for n in order_now]

            fig.add_trace(go.Scatter(
                x=labels_now, y=y_now,
                mode="lines+markers",
                name=f"{hhmm_pt} PT" + (" (DB)" if source == "db" else " (API)"),
                line=dict(width=2, color=color),
                marker=dict(size=5, color=color),
            ))
            traces += 1

            # expected (if we have prior row + stock)
            stock_now = None
            for sname in ("underlying", "stockPrice"):
                if sname in row_now.index and pd.notna(row_now[sname]):
                    stock_now = float(row_now[sname]); break

            ts_et = pt_minute_to_et(trade_date_iso, hhmm_pt)
            if expected_on and prev_row is not None and prev_T is not None and prev_stock is not None and stock_now is not None:
                try:
                    labels_exp, expected_y, atm_exp_pct = _expected_curve_shifted(
                        prev_row, prev_T, prev_stock,
                        row_now, _years_to_exp(ts_et, expiration_iso), stock_now
                    )
                    fig.add_trace(go.Scatter(
                        x=labels_exp, y=expected_y,
                        mode="lines",
                        name=f"Expected (SS) — {hhmm_pt}",
                        line=dict(width=2, dash="dot", color=color),
                    ))
                    fig.add_trace(go.Scatter(
                        x=["ATM"], y=[atm_exp_pct],
                        mode="markers",
                        marker=dict(symbol="triangle-up", size=9, color=color),
                        name="ATM exp (SS)",
                        showlegend=False,
                    ))
                except Exception as e:
                    _log(f"expected err {hhmm_pt}: {e}")

            prev_row = row_now
            prev_stock = stock_now
            prev_T = _years_to_exp(ts_et, expiration_iso)

        if traces == 0:
            _log("no traces rendered")
        return fig
