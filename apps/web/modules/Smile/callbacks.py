from __future__ import annotations
import datetime as dt
import math
from typing import List, Optional, Tuple
import os

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import Input, Output
from sqlalchemy import create_engine, text

# ---- App IDs ----
TRADE_DATE_ID = "trade-date"
EXPIRATION_ID = "expiration-date-pick"
SMILE_TIME_INPUT = "smile-time-input"
SMILE_GRAPH = "SMILE_GRAPH"
EXPECTED_TOGGLE_ID = "expected-ss-toggle"
CLOCK_ID = "CLOCK"

# ---- Database Configuration ----
DATABASE_URL = os.getenv("DATABASE_URL")  # or "CURVE_DB_URL" if you prefer a custom name

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is not set")

DB_TABLE_NAME = "orats_monies_minute"
DB_ENGINE = create_engine(DATABASE_URL)

# ---- Plotting Constants ----
TICKER = "SPX"
EPS_T = 1e-4
BETA_VOLPTS_PER_1PCT = 4.5
BETA_MAX_SHIFT_PP = 6.0
COLORWAY = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"
]

# ----------------- Data Fetching -----------------
def fetch_data_from_db(trade_date_iso: str, expiration_iso: str, times_pt: List[str]) -> pd.DataFrame:
    """
    Fetches all necessary data for a given trade date, expiration, and time slices
    from the database with a single query.
    """
    if not trade_date_iso or not expiration_iso or not times_pt:
        return pd.DataFrame()

    time_filters = [f"'{trade_date_iso} {hhmm}:00'" for hhmm in times_pt]
    query = text(f"""
        SELECT *
        FROM "{DB_TABLE_NAME}"
        WHERE
            trade_date = :trade_date AND
            expir_date = :expir_date AND
            snapshot_pt IN ({','.join(time_filters)})
        ORDER BY snapshot_pt;
    """)

    try:
        with DB_ENGINE.connect() as connection:
            df = pd.read_sql(query, connection, params={
                "trade_date": trade_date_iso,
                "expir_date": expiration_iso,
            })
        return df
    except Exception as e:
        print(f"Database query failed: {e}")
        return pd.DataFrame()

# ----------------- Plotting Utils -----------------
def _years_to_exp(ts_utc: dt.datetime, expiration_iso: str) -> float:
    exp_date = dt.date.fromisoformat(expiration_iso)
    rem = dt.datetime.combine(exp_date, dt.time(16, 0), tzinfo=dt.timezone.utc) - ts_utc
    T = max(0.0, rem.total_seconds() / (365.0 * 24 * 3600))
    return max(T, EPS_T)

def _available_buckets(row: pd.Series) -> List[int]:
    out = []
    for c in row.index:
        if c.startswith("vol") and c[3:].isdigit():
            n = int(c[3:])
            if 1 <= n <= 99:
                out.append(n)
    return sorted(out, reverse=True)

def _bucket_labels_order(buckets: List[int]) -> Tuple[List[int], List[str]]:
    puts = [n for n in buckets if n >= 50]
    calls = [n for n in buckets if n < 50]
    order = puts + calls
    labels = [f"P{100-n}" if n > 50 else "ATM" if n == 50 else f"C{n}" for n in order]
    return order, labels

from packages.shared.surface_compare import k_for_abs_delta
def _k_grid_for_row(row: pd.Series, T: float) -> Tuple[np.ndarray, np.ndarray]:
    buckets = _available_buckets(row)
    if not buckets or 'vol50' not in row or pd.isna(row['vol50']):
        return np.array([]), np.array([])
    atm = float(row["vol50"])
    k_list, s_list = [], []
    for n in buckets:
        if n == 50: k = 0.0
        else:
            p, is_put = ((100 - n) / 100.0, True) if n > 50 else (n / 100.0, False)
            k = k_for_abs_delta(p, is_put=is_put, sigma=atm, T=T)
        k_list.append(k)
        s_list.append(float(row[f"vol{n}"]))
    k, s = np.array(k_list, float), np.array(s_list, float)
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

def _expected_curve_shifted(prev_row, prev_T, prev_stock, now_row, now_T, now_stock):
    k_prev, s_prev = _k_grid_for_row(prev_row, prev_T)
    if k_prev.size == 0: return [], np.array([]), 0.0
    k_shift = math.log(now_stock / prev_stock) if (prev_stock and now_stock) else 0.0
    buckets = [n for n in _available_buckets(now_row) if n not in (95, 5)]
    if not buckets or 'vol50' not in now_row or pd.isna(now_row['vol50']):
        return [], np.array([]), 0.0
    order, labels = _bucket_labels_order(buckets)
    atm_now = float(now_row["vol50"])
    shape_vals = []
    for n in order:
        if n == 50: k_now = 0.0
        else:
            p, is_put = ((100 - n) / 100.0, True) if n > 50 else (n / 100.0, False)
            k_now = k_for_abs_delta(p, is_put=is_put, sigma=atm_now, T=now_T)
        shape_vals.append(_interp_linear_extrap(k_now + k_shift, k_prev, s_prev))
    shape = np.array(shape_vals, float)
    exp_atm_shape = _interp_linear_extrap(k_shift, k_prev, s_prev)
    ret_frac = (now_stock - prev_stock) / prev_stock if prev_stock else 0.0
    level_shift_pp = max(-BETA_MAX_SHIFT_PP, min(BETA_MAX_SHIFT_PP, (-ret_frac) * 100.0 * BETA_VOLPTS_PER_1PCT))
    atm_exp = exp_atm_shape + level_shift_pp / 100.0
    shift = atm_exp - exp_atm_shape
    expected = shape + shift
    return labels, expected * 100.0, atm_exp * 100.0

# ----------------- Main Callback -----------------
def register_callbacks(app):
    @app.callback(
        Output(SMILE_GRAPH, "figure"),
        [Input(TRADE_DATE_ID, "date"),
         Input(EXPIRATION_ID, "date"),
         Input(SMILE_TIME_INPUT, "value"),
         Input(EXPECTED_TOGGLE_ID, "value")]
    )
    def render_smile(trade_date_iso, expiration_iso, times_pt, expected_value):
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=40),
            title=f"Smile Grid — {trade_date_iso or ''} (Exp: {expiration_iso or ''})",
            xaxis_title="Bucket (P10 … ATM … C10)",
            yaxis_title="IV (%)",
            legend=dict(orientation="v", x=1.02, y=1.0, bgcolor="rgba(0,0,0,0)"),
            colorway=COLORWAY,
        )

        if not all([trade_date_iso, expiration_iso, times_pt]):
            return fig

        df = fetch_data_from_db(trade_date_iso, expiration_iso, sorted(times_pt))
        if df.empty:
            return fig

        prev_row, prev_stock, prev_T = None, None, None

        for i, (snapshot, row_now) in enumerate(df.groupby('snapshot_pt')):
            row_now = row_now.iloc[0]
            hhmm_pt = snapshot.strftime("%H:%M")
            color = COLORWAY[i % len(COLORWAY)]

            buckets_now = [n for n in _available_buckets(row_now) if n not in (95, 5)]
            order_now, labels_now = _bucket_labels_order(buckets_now)
            y_now = [float(row_now.get(f"vol{n}", 0)) * 100.0 for n in order_now]
            fig.add_trace(go.Scatter(
                x=labels_now, y=y_now, mode="lines+markers", name=f"{hhmm_pt} PT",
                line=dict(width=2, color=color), marker=dict(size=5, color=color)
            ))

            stock_now = float(row_now.get("stock_price", 0))
            
            # FIX: Use correct 'snap_shot_date' column and handle missing values
            ts_utc_val = row_now.get("snap_shot_date")
            now_T = 0
            if pd.notna(ts_utc_val):
                ts_utc = pd.to_datetime(ts_utc_val, utc=True)
                now_T = _years_to_exp(ts_utc, expiration_iso)

            if expected_value == "on" and prev_row is not None and prev_stock is not None and prev_T is not None and now_T > 0:
                try:
                    labels_exp, y_exp, atm_exp_pct = _expected_curve_shifted(
                        prev_row, prev_T, prev_stock, row_now, now_T, stock_now
                    )
                    if labels_exp:
                        fig.add_trace(go.Scatter(
                            x=labels_exp, y=y_exp, mode="lines", name=f"Expected — {hhmm_pt}",
                            line=dict(width=2, dash="dot", color=color)
                        ))
                        fig.add_trace(go.Scatter(
                            x=["ATM"], y=[atm_exp_pct], mode="markers",
                            marker=dict(symbol="triangle-up", size=9, color=color), showlegend=False
                        ))
                except Exception as e:
                    print(f"Could not calculate expected curve for {hhmm_pt}: {e}")

            prev_row, prev_stock, prev_T = row_now, stock_now, now_T

        return fig
