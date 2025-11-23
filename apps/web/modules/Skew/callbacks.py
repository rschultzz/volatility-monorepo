from __future__ import annotations
import datetime as dt
import math
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import Input, Output
from sqlalchemy import create_engine, text

# ---- Dash IDs ----
TRADE_DATE_ID = "trade-date"
EXPIRATION_ID = "expiration-date-pick"
SMILE_TIME_INPUT = "smile-time-input"
SKEW_TABLE = "SKEW_TABLE"
CLOCK_ID = "CLOCK"
EXPECTED_TOGGLE_ID = "expected-ss-toggle"

# ---- Database Configuration ----
DATABASE_URL = "postgresql+psycopg://rschultz:5hUHvSVPDyVXhz7acgJZvlvnj7nFMDap@dpg-d38sm515pdvs738rknj0-a.oregon-postgres.render.com/curve_trading?sslmode=require"
DB_TABLE_NAME = "orats_monies_minute"
DB_ENGINE = create_engine(DATABASE_URL)

# ---- Constants ----
TICKER = "SPX"
EPS_T = 1e-4
MIN_SKEW_DENOM_PP = 0.25
BETA_VOLPTS_PER_1PCT = 4.5
BETA_MAX_SHIFT_PP = 6.0

# ----------------- Data Fetching -----------------
def fetch_data_from_db(trade_date_iso: str, expiration_iso: str, times_pt: List[str]) -> pd.DataFrame:
    if not trade_date_iso or not expiration_iso or not times_pt:
        return pd.DataFrame()
    time_filters = [f"'{trade_date_iso} {hhmm}:00'" for hhmm in times_pt]
    query = text(f"""
        SELECT * FROM "{DB_TABLE_NAME}"
        WHERE trade_date = :trade_date AND expir_date = :expir_date
          AND snapshot_pt IN ({','.join(time_filters)})
        ORDER BY snapshot_pt;
    """)
    try:
        with DB_ENGINE.connect() as connection:
            return pd.read_sql(query, connection, params={"trade_date": trade_date_iso, "expir_date": expiration_iso})
    except Exception as e:
        print(f"Skew DB query failed: {e}")
        return pd.DataFrame()

# ----------------- Calculation Utils -----------------
def _skews_from_row(row: pd.Series) -> Tuple[float, float, float]:
    atm = float(pd.to_numeric(row.get("vol50"), errors="coerce"))
    c25 = float(pd.to_numeric(row.get("vol25"), errors="coerce"))
    p25 = float(pd.to_numeric(row.get("vol75"), errors="coerce"))
    return atm, (c25 - atm) * 100.0, (p25 - atm) * 100.0

def _pct_change_frac(curr: Optional[float], base: Optional[float]) -> Optional[float]:
    if base in (None, 0) or curr is None: return None
    return (curr - base) / abs(base) * 100.0

def _pct_change_pp(curr_pp: Optional[float], base_pp: Optional[float]) -> Optional[float]:
    if curr_pp is None or base_pp is None: return None
    denom = max(abs(base_pp), MIN_SKEW_DENOM_PP)
    return (curr_pp - base_pp) / denom * 100.0

def _years_to_exp(ts_utc: dt.datetime, expiration_iso: str) -> float:
    exp_date = dt.date.fromisoformat(expiration_iso)
    rem = dt.datetime.combine(exp_date, dt.time(16, 0), tzinfo=dt.timezone.utc) - ts_utc
    T = max(0.0, rem.total_seconds() / (365.0 * 24 * 3600))
    return max(T, EPS_T)

from packages.shared.surface_compare import k_for_abs_delta
def _k_grid_for_row(row: pd.Series, T: float) -> Tuple[np.ndarray, np.ndarray]:
    buckets = sorted([int(c[3:]) for c in row.index if c.startswith("vol") and c[3:].isdigit()], reverse=True)
    if not buckets or 'vol50' not in row or pd.isna(row['vol50']): return np.array([]), np.array([])
    atm = float(row["vol50"])
    k_list, s_list = [], []
    for n in buckets:
        if n == 50: k = 0.0
        else:
            p, is_put = ((100 - n) / 100.0, True) if n > 50 else (n / 100.0, False)
            k = k_for_abs_delta(p, is_put=is_put, sigma=atm, T=T)
        k_list.append(k); s_list.append(float(row[f"vol{n}"]))
    k, s = np.array(k_list, float), np.array(s_list, float)
    mask = np.concatenate(([True], np.diff(k) > 1e-12)); return k[mask], s[mask]

def _interp_linear_extrap(kq: float, k_grid: np.ndarray, s_grid: np.ndarray) -> float:
    if kq <= k_grid[0]:
        x0,x1,y0,y1 = k_grid[0],k_grid[1],s_grid[0],s_grid[1]; return float(y0 + (y1-y0)*(kq-x0)/(x1-x0))
    if kq >= k_grid[-1]:
        x0,x1,y0,y1 = k_grid[-2],k_grid[-1],s_grid[-2],s_grid[-1]; return float(y1 + (y1-y0)*(kq-x1)/(x1-x0))
    return float(np.interp(kq, k_grid, s_grid))

# ----------------- Main Callback -----------------
def register_callbacks(app):
    @app.callback(
        Output(SKEW_TABLE, "figure"),
        [Input(TRADE_DATE_ID, "date"), Input(EXPIRATION_ID, "date"),
         Input(SMILE_TIME_INPUT, "value"), Input(EXPECTED_TOGGLE_ID, "value")]
    )
    def render_skew_table(trade_date_iso, expiration_iso, times_pt, expected_value):
        expected_on = (expected_value == "on")
        base_cols = ["Time (PT)", "Stock", "Δ Stock %", "ATM IV %", "Call Skew", "Put Skew", "Δ ATM IV %", "Δ Call Skew %", "Δ Put Skew %"]
        exp_cols = ["ATM exp (SS) %", "ATM residual (bp)"]
        
        if not all([trade_date_iso, expiration_iso, times_pt]):
            cols = base_cols if not expected_on else base_cols[:4] + exp_cols + base_cols[4:]
            return go.Figure(data=[go.Table(header=dict(values=cols), cells=dict(values=[[] for _ in cols]))]).update_layout(template="plotly_dark", title="Skew — Select Date & Expiration")

        df_data = fetch_data_from_db(trade_date_iso, expiration_iso, sorted(times_pt))
        if df_data.empty:
            return go.Figure().update_layout(template="plotly_dark", title=f"Skew — No data for {trade_date_iso} / {expiration_iso}")

        table_rows = []
        prev_row, prev_stock, prev_T = None, None, None
        prev_stock_actual, prev_atm_actual, prev_call_skew_pp_actual, prev_put_skew_pp_actual = None, None, None, None

        # FIX: Group by snapshot_pt to handle potential duplicates (e.g., AM/PM settlements)
        for _, group in df_data.groupby('snapshot_pt'):
            row_now = group.iloc[0] # Take the first row for this timestamp
            
            stock_now = float(row_now.get("stock_price", 0))
            atm_now, call_skew_pp_now, put_skew_pp_now = _skews_from_row(row_now)
            ts_utc_val = row_now.get("snap_shot_date"); now_T = 0
            if pd.notna(ts_utc_val): now_T = _years_to_exp(pd.to_datetime(ts_utc_val, utc=True), expiration_iso)

            d_stock_pct = _pct_change_frac(stock_now, prev_stock_actual)
            d_atm_pct = _pct_change_frac(atm_now, prev_atm_actual)
            d_call_pct = _pct_change_pp(call_skew_pp_now, prev_call_skew_pp_actual)
            d_put_pct = _pct_change_pp(put_skew_pp_now, prev_put_skew_pp_actual)
            atm_exp_pct, atm_res_bp = None, None

            if expected_on and prev_row is not None and prev_T is not None and prev_stock is not None and now_T > 0:
                try:
                    k_prev, s_prev = _k_grid_for_row(prev_row, prev_T)
                    k_shift = math.log(stock_now / prev_stock) if prev_stock else 0.0
                    exp_atm_shape = _interp_linear_extrap(k_shift, k_prev, s_prev)
                    ret_frac = (stock_now - prev_stock) / prev_stock if prev_stock else 0.0
                    level_shift_pp = max(-BETA_MAX_SHIFT_PP, min(BETA_MAX_SHIFT_PP, (-ret_frac) * 100.0 * BETA_VOLPTS_PER_1PCT))
                    atm_exp = exp_atm_shape + (level_shift_pp / 100.0)
                    
                    k_c25 = k_for_abs_delta(0.25, is_put=False, sigma=atm_now, T=now_T)
                    k_p25 = k_for_abs_delta(0.25, is_put=True, sigma=atm_now, T=now_T)
                    exp_c25_shape = _interp_linear_extrap(k_c25 + k_shift, k_prev, s_prev)
                    exp_p25_shape = _interp_linear_extrap(k_p25 + k_shift, k_prev, s_prev)
                    shift_frac = atm_exp - exp_atm_shape
                    exp_c25, exp_p25 = exp_c25_shape + shift_frac, exp_p25_shape + shift_frac
                    exp_call_skew_pp, exp_put_skew_pp = (exp_c25 - atm_exp) * 100.0, (exp_p25 - atm_exp) * 100.0
                    
                    d_atm_pct = _pct_change_frac(atm_now, atm_exp)
                    d_call_pct = _pct_change_pp(call_skew_pp_now, exp_call_skew_pp)
                    d_put_pct = _pct_change_pp(put_skew_pp_now, exp_put_skew_pp)
                    atm_exp_pct, atm_res_bp = round(atm_exp * 100.0, 2), int(round((atm_now - atm_exp) * 10000.0))
                except Exception as e:
                    print(f"Could not calculate expected skew for {row_now.snapshot_pt.strftime('%H:%M')}: {e}")

            table_rows.append({
                "Time (PT)": row_now.snapshot_pt.strftime("%H:%M"), "Stock": round(stock_now, 2), "Δ Stock %": round(d_stock_pct, 2) if d_stock_pct is not None else None,
                "ATM IV %": round(atm_now * 100.0, 2), "ATM exp (SS) %": atm_exp_pct, "ATM residual (bp)": atm_res_bp,
                "Call Skew": round(call_skew_pp_now, 2), "Put Skew": round(put_skew_pp_now, 2),
                "Δ ATM IV %": round(d_atm_pct, 2) if d_atm_pct is not None else None, "Δ Call Skew %": round(d_call_pct, 2) if d_call_pct is not None else None, "Δ Put Skew %": round(d_put_pct, 2) if d_put_pct is not None else None,
            })
            prev_row, prev_stock, prev_T = row_now, stock_now, now_T
            prev_stock_actual, prev_atm_actual, prev_call_skew_pp_actual, prev_put_skew_pp_actual = stock_now, atm_now, call_skew_pp_now, put_skew_pp_now

        df_table = pd.DataFrame(table_rows)
        ordered_cols = base_cols[:4] + exp_cols + base_cols[4:] if expected_on else base_cols
        df_table = df_table[ordered_cols]
        
        cell_colors = [['green' if v is not None and v > 0 else 'red' if v is not None and v < 0 else 'black' for v in df_table[c]] if c in {'Δ Stock %','Δ ATM IV %','Δ Call Skew %','Δ Put Skew %','ATM residual (bp)'} else ['black'] * len(df_table) for c in df_table.columns]
        
        fig = go.Figure(data=[go.Table(header=dict(values=list(df_table.columns)), cells=dict(values=[df_table[c] for c in df_table.columns], fill_color=cell_colors, align="left", font=dict(color="white")))])
        fig.update_layout(template="plotly_dark", title=f"Skew (DB) — {trade_date_iso}   Exp: {expiration_iso}", margin=dict(l=0, r=0, t=36, b=0))
        return fig
