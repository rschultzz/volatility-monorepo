from __future__ import annotations
import datetime as dt
import math
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import Input, Output

# ---- Dash IDs ----
TRADE_DATE_ID = "trade-date"
EXPIRATION_ID = "expiration-date-pick"
SMILE_TIME_INPUT = "smile-time-input"
SKEW_TABLE = "SKEW_TABLE"
CLOCK_ID = "CLOCK"
EXPECTED_TOGGLE_ID = "expected-ss-toggle"

# ---- Data helpers ----
from packages.shared.options_orats import fetch_one_minute_monies, pt_minute_to_et, PT_TZ
from packages.shared.surface_compare import k_for_abs_delta

TICKER = "SPX"
NEEDED_COLS = ["vol25", "vol50", "vol75"]
EPS_T = 1e-4

# keep skew % deltas sane when expected/previous ≈ 0 pp
MIN_SKEW_DENOM_PP = 0.25

# leverage add-on to match Smile overlay ATM marker
BETA_VOLPTS_PER_1PCT = 4.5
BETA_MAX_SHIFT_PP = 6.0


# ---------- utilities ----------
def _get_row(df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    row = df.iloc[0]
    if not all(c in row.index for c in NEEDED_COLS):
        return None
    return row

def _skews_from_row(row: pd.Series) -> Tuple[float, float, float]:
    """Return (atm_frac, call_skew_pp, put_skew_pp)."""
    atm = float(pd.to_numeric(row["vol50"], errors="coerce"))
    c25 = float(pd.to_numeric(row["vol25"], errors="coerce"))
    p25 = float(pd.to_numeric(row["vol75"], errors="coerce"))
    return atm, (c25 - atm) * 100.0, (p25 - atm) * 100.0

def _pct_change_frac(curr: Optional[float], base: Optional[float]) -> Optional[float]:
    if base in (None, 0) or curr is None:
        return None
    return (curr - base) / abs(base) * 100.0

def _pct_change_pp(curr_pp: Optional[float], base_pp: Optional[float]) -> Optional[float]:
    if curr_pp is None or base_pp is None:
        return None
    denom = max(abs(base_pp), MIN_SKEW_DENOM_PP)
    return (curr_pp - base_pp) / denom * 100.0

def _years_to_exp(ts_et: dt.datetime, expiration_iso: str) -> float:
    exp_date = dt.date.fromisoformat(expiration_iso)
    remaining = dt.datetime.combine(exp_date, dt.time(0,0)) - ts_et.replace(tzinfo=None)
    T = max(0.0, remaining.days/365.0 + remaining.seconds/(365.0*24*3600))
    return max(T, EPS_T)

def _available_buckets(row: pd.Series) -> List[int]:
    buckets = []
    for c in row.index:
        if c.startswith("vol") and c[3:].isdigit():
            n = int(c[3:])
            if 1 <= n <= 99:
                buckets.append(n)
    puts  = sorted([n for n in buckets if n >= 50], reverse=True)
    calls = sorted([n for n in buckets if n < 50],  reverse=True)
    out, seen = [], set()
    for n in puts + calls:
        if n not in seen:
            seen.add(n); out.append(n)
    return out

def _abs_delta_is_put(bucket: int) -> Tuple[float, bool]:
    if bucket == 50: return 0.50, False
    if bucket > 50:  return (100 - bucket)/100.0, True
    return bucket/100.0, False

def _prev_smile_interp(prev_row: pd.Series, T_prev: float):
    """
    Build previous minute’s (k_prev, sigma_prev) grid using its ATM for delta→k.
    """
    if "vol50" not in prev_row:
        raise ValueError("prev row missing ATM")
    atm_prev = float(prev_row["vol50"])
    buckets_prev = _available_buckets(prev_row)
    if len(buckets_prev) < 4:
        raise ValueError("prev row has too few buckets")
    k_prev, s_prev = [], []
    for n in buckets_prev:
        if n == 50:
            k = 0.0
        else:
            p, is_put = _abs_delta_is_put(n)
            k = k_for_abs_delta(p, is_put=is_put, sigma=atm_prev, T=T_prev)
        k_prev.append(k); s_prev.append(float(prev_row[f"vol{n}"]))
    k_np = np.array(k_prev, float); s_np = np.array(s_prev, float)
    mask = np.concatenate(([True], np.diff(k_np) > 1e-12))
    k_np, s_np = k_np[mask], s_np[mask]
    if k_np.size < 3: raise ValueError("prev k-grid degenerate")
    return k_np, s_np

def _interp_linear_extrap(kq: float, k_grid: np.ndarray, s_grid: np.ndarray) -> float:
    if kq <= k_grid[0]:
        x0,x1,y0,y1 = k_grid[0],k_grid[1],s_grid[0],s_grid[1]
        return float(y0 + (y1-y0)*(kq-x0)/(x1-x0))
    if kq >= k_grid[-1]:
        x0,x1,y0,y1 = k_grid[-2],k_grid[-1],s_grid[-2],s_grid[-1]
        return float(y1 + (y1-y0)*(kq-x1)/(x1-x0))
    return float(np.interp(kq, k_grid, s_grid))


# ---------- main callback ----------
def register_callbacks(app):
    @app.callback(
        Output(SKEW_TABLE, "figure"),
        Input(TRADE_DATE_ID, "date"),
        Input(EXPIRATION_ID, "date"),
        Input(SMILE_TIME_INPUT, "value"),
        Input(EXPECTED_TOGGLE_ID, "value"),
        Input(CLOCK_ID, "n_intervals"),
    )
    def render_skew_table(trade_date_iso, expiration_iso, times_pt, expected_value, _tick):
        expected_on = (expected_value != "off")

        # Column order requested:
        # Time, Stock, Δ Stock %, ATM IV %, [ATM exp (SS) %, ATM residual (bp)], Call Skew, Put Skew, Δ ATM IV %, Δ Call Skew %, Δ Put Skew %
        base_cols = [
            "Time (PT)", "Stock", "Δ Stock %", "ATM IV %",
            "Call Skew", "Put Skew",
            "Δ ATM IV %", "Δ Call Skew %", "Δ Put Skew %",
        ]
        exp_cols = ["ATM exp (SS) %", "ATM residual (bp)"]

        if not trade_date_iso or not expiration_iso:
            cols = base_cols if not expected_on else ["Time (PT)", "Stock", "Δ Stock %", "ATM IV %"] + exp_cols + ["Call Skew","Put Skew","Δ ATM IV %","Δ Call Skew %","Δ Put Skew %"]
            return go.Figure(data=[go.Table(
                header=dict(values=cols, fill_color="#444", font=dict(color="white"), align="left"),
                cells=dict(values=[[] for _ in cols], fill_color="black", font=dict(color="white"), align="left"),
            )]).update_layout(template="plotly_dark", title="Skew — select Trade Date & Expiration")

        # ensure "now" minute is present on today's date
        if not times_pt:
            times_pt = ["06:31"]
        now_pt = dt.datetime.now(PT_TZ)
        if trade_date_iso == now_pt.date().isoformat():
            hhmm_now = now_pt.strftime("%H:%M")
            if "06:30" <= hhmm_now <= "13:00":
                times_pt = sorted(set(times_pt + [hhmm_now]))
        times_sorted = sorted(times_pt)

        rows = []

        prev_row = None
        prev_stock = None
        prev_T = None

        # OFF mode deltas
        prev_stock_actual = None
        prev_atm_actual = None
        prev_call_skew_pp_actual = None
        prev_put_skew_pp_actual = None

        for hhmm_pt in times_sorted:
            ts_et = pt_minute_to_et(trade_date_iso, hhmm_pt)
            df_now = fetch_one_minute_monies(ts_et, TICKER, expiration_iso)
            row_now = _get_row(df_now)
            if row_now is None:
                continue

            stock_now = float(pd.to_numeric(df_now["stockPrice"], errors="coerce").median()) if "stockPrice" in df_now.columns else None
            atm_now, call_skew_pp_now, put_skew_pp_now = _skews_from_row(row_now)
            T_now = _years_to_exp(ts_et, expiration_iso)

            atm_exp_pct = None
            atm_res_bp  = None
            d_atm_pct   = None
            d_call_pct  = None
            d_put_pct   = None

            if expected_on and prev_row is not None and prev_T is not None and prev_stock is not None and stock_now is not None:
                try:
                    # previous surface and SS k-shift
                    k_prev, s_prev = _prev_smile_interp(prev_row, prev_T)
                    k_shift = math.log(stock_now / prev_stock) if (prev_stock and stock_now) else 0.0

                    # --- 1) Expected ATM anchor (match Smile overlay marker) ---
                    exp_atm_shape = _interp_linear_extrap(k_shift, k_prev, s_prev)  # fraction
                    # leverage add-on:
                    ret_frac = (stock_now - prev_stock) / prev_stock
                    level_shift_pp = max(-BETA_MAX_SHIFT_PP,
                                         min(BETA_MAX_SHIFT_PP, (-ret_frac) * 100.0 * BETA_VOLPTS_PER_1PCT))
                    atm_exp = exp_atm_shape + (level_shift_pp / 100.0)  # fraction
                    atm_exp_pct = round(atm_exp * 100.0, 2)
                    atm_res_bp  = int(round((atm_now - atm_exp) * 10000.0))

                    # --- 2) Expected curve SHAPE (same as Smile dotted line) ---
                    # Use current ATM to map Δ->k, then read from previous surface at (k_now + k_shift)
                    k_c25_now = k_for_abs_delta(0.25, is_put=False, sigma=atm_now, T=T_now)
                    k_p25_now = k_for_abs_delta(0.25, is_put=True,  sigma=atm_now, T=T_now)
                    exp_c25_shape = _interp_linear_extrap(k_c25_now + k_shift, k_prev, s_prev)  # fraction
                    exp_p25_shape = _interp_linear_extrap(k_p25_now + k_shift, k_prev, s_prev)  # fraction

                    # --- 3) Vertical shift so curve ATM == atm_exp ---
                    shift_frac = atm_exp - exp_atm_shape
                    exp_c25 = exp_c25_shape + shift_frac
                    exp_p25 = exp_p25_shape + shift_frac

                    # Expected skews after the vertical shift (equivalently vs shape ATM)
                    exp_call_skew_pp = (exp_c25 - atm_exp) * 100.0
                    exp_put_skew_pp  = (exp_p25 - atm_exp) * 100.0

                    # --- 4) Table % deltas vs shifted expected curve ---
                    d_call_pct = _pct_change_pp(call_skew_pp_now, exp_call_skew_pp)
                    d_put_pct  = _pct_change_pp(put_skew_pp_now,  exp_put_skew_pp)
                    d_atm_pct  = _pct_change_frac(atm_now, atm_exp)

                except Exception:
                    # graceful fallback to previous actual slice
                    d_call_pct = _pct_change_pp(call_skew_pp_now, prev_call_skew_pp_actual)
                    d_put_pct  = _pct_change_pp(put_skew_pp_now,  prev_put_skew_pp_actual)
                    d_atm_pct  = _pct_change_frac(atm_now,        prev_atm_actual)
            else:
                # OFF mode: compare against previous actual slice
                d_call_pct = _pct_change_pp(call_skew_pp_now, prev_call_skew_pp_actual)
                d_put_pct  = _pct_change_pp(put_skew_pp_now,  prev_put_skew_pp_actual)
                d_atm_pct  = _pct_change_frac(atm_now,        prev_atm_actual)

            d_stock_pct = None
            if prev_stock_actual:
                d_stock_pct = (stock_now - prev_stock_actual) / abs(prev_stock_actual) * 100.0 if stock_now is not None else None

            rows.append({
                "Time (PT)": hhmm_pt,
                "Stock": None if stock_now is None else round(stock_now, 2),
                "Δ Stock %": None if d_stock_pct is None else round(d_stock_pct, 2),
                "ATM IV %": round(atm_now * 100.0, 2),
                "ATM exp (SS) %": atm_exp_pct,
                "ATM residual (bp)": atm_res_bp,
                "Call Skew": round(call_skew_pp_now, 2),
                "Put Skew":  round(put_skew_pp_now,  2),
                "Δ ATM IV %": None if d_atm_pct  is None else round(d_atm_pct, 2),
                "Δ Call Skew %": None if d_call_pct is None else round(d_call_pct, 2),
                "Δ Put Skew %":  None if d_put_pct  is None else round(d_put_pct,  2),
            })

            # advance previous-actual pointers
            prev_row, prev_stock, prev_T = row_now, stock_now, T_now
            prev_stock_actual, prev_atm_actual = stock_now, atm_now
            prev_call_skew_pp_actual, prev_put_skew_pp_actual = call_skew_pp_now, put_skew_pp_now

        if not rows:
            cols = base_cols if not expected_on else ["Time (PT)", "Stock", "Δ Stock %", "ATM IV %"] + exp_cols + ["Call Skew","Put Skew","Δ ATM IV %","Δ Call Skew %","Δ Put Skew %"]
            return go.Figure(data=[go.Table(
                header=dict(values=cols, fill_color="#444", font=dict(color="white"), align="left"),
                cells=dict(values=[[] for _ in cols], fill_color="black", font=dict(color="white"), align="left"),
            )]).update_layout(template="plotly_dark", title=f"Skew — no data for {trade_date_iso} / {expiration_iso}")

        df = pd.DataFrame(rows)
        ordered_cols = (["Time (PT)", "Stock", "Δ Stock %", "ATM IV %", "ATM exp (SS) %", "ATM residual (bp)", "Call Skew", "Put Skew", "Δ ATM IV %", "Δ Call Skew %", "Δ Put Skew %"]
                        if expected_on else
                        ["Time (PT)", "Stock", "Δ Stock %", "ATM IV %", "Call Skew", "Put Skew", "Δ ATM IV %", "Δ Call Skew %", "Δ Put Skew %"])
        df = df[ordered_cols]

        color_cols = {'Δ Stock %','Δ ATM IV %','Δ Call Skew %','Δ Put Skew %','ATM residual (bp)'}
        cell_colors = []
        for col in df.columns:
            if col in color_cols:
                colors = ['green' if (v is not None and v > 0) else 'red' if (v is not None and v < 0) else 'black'
                          for v in df[col]]
                cell_colors.append(colors)
            else:
                cell_colors.append(['black'] * len(df))

        fig = go.Figure(data=[go.Table(
            header=dict(values=list(df.columns), fill_color="#444", font=dict(color="white"), align="left"),
            cells=dict(values=[df[c] for c in df.columns], fill_color=cell_colors, align="left", font=dict(color="white")),
        )])
        fig.update_layout(
            template="plotly_dark",
            title=f"Skew (50Δ → 25Δ) — {trade_date_iso}   Exp: {expiration_iso}",
            margin=dict(l=0, r=0, t=36, b=0),
        )
        return fig
