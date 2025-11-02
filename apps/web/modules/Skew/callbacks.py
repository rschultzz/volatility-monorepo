# apps/web/modules/Skew/callbacks.py — Skew table with ATM expected vs actual + Δ skews restored
from __future__ import annotations
import datetime as dt
from typing import Tuple
import pandas as pd
import plotly.graph_objs as go
from dash import Input, Output

# IDs reused from the main app
TRADE_DATE_ID = "trade-date"
EXPIRATION_ID = "expiration-date-pick"
SMILE_TIME_INPUT = "smile-time-input"
SKEW_TABLE = "SKEW_TABLE"
CLOCK_ID = "CLOCK"

from packages.shared.options_orats import fetch_one_minute_monies, pt_minute_to_et, PT_TZ
from packages.shared.surface_compare import TimeSlice, expected_vs_actual_atm, k_for_abs_delta

TICKER = "SPX"

# We need C25, ATM, P25 from the monies grid:
#   ATM   -> vol50
#   C25   -> vol25
#   P25   -> vol75
NEEDED_COLS = ["vol25", "vol50", "vol75"]

def _get_row(df: pd.DataFrame) -> pd.Series | None:
    if df is None or df.empty:
        return None
    row = df.iloc[0]
    if not all(c in row.index for c in NEEDED_COLS):
        return None
    return row

def _skews_from_row(row: pd.Series) -> Tuple[float, float, float]:
    """Return (atm_sigma, call_skew_pp, put_skew_pp). Skews in IV percentage points."""
    atm = float(pd.to_numeric(row["vol50"], errors="coerce"))
    c25 = float(pd.to_numeric(row["vol25"], errors="coerce"))
    p25 = float(pd.to_numeric(row["vol75"], errors="coerce"))
    call_skew = (c25 - atm) * 100.0
    put_skew  = (p25 - atm) * 100.0
    return atm, call_skew, put_skew

def _pct_change(curr: float | None, prev: float | None) -> float | None:
    if prev is None or prev == 0 or curr is None:
        return None
    return (curr - prev) / abs(prev) * 100.0

def _years_to_exp(ts_et: dt.datetime, expiration_iso: str) -> float:
    exp_date = dt.date.fromisoformat(expiration_iso)
    # Use midnight ET for simplicity; minute-level error is negligible for intraday ATM comparison
    remaining = dt.datetime.combine(exp_date, dt.time(0, 0)) - ts_et.replace(tzinfo=None)
    return max(0.0, remaining.days / 365.0 + remaining.seconds / (365.0 * 24 * 3600))

def _estimate_slope(atm_sigma: float, T: float, c25_sigma: float, p25_sigma: float) -> float:
    """
    Estimate dσ/dk at ATM using symmetric 25Δ wings.
    Uses constant-vol deltas to get k25 (good enough intraday).
    """
    if T <= 0:
        return 0.0
    k_put  = k_for_abs_delta(0.25, is_put=True,  sigma=atm_sigma, T=T)   # negative
    k_call = k_for_abs_delta(0.25, is_put=False, sigma=atm_sigma, T=T)   # positive
    # Central slope around k=0
    return (c25_sigma - p25_sigma) / (k_call - k_put)

def register_callbacks(app):
    @app.callback(
        Output(SKEW_TABLE, "figure"),
        Input(TRADE_DATE_ID, "date"),
        Input(EXPIRATION_ID, "date"),
        Input(SMILE_TIME_INPUT, "value"),   # list of "HH:MM" PT
        Input(CLOCK_ID, "n_intervals"),     # tick every minute
    )
    def render_skew_table(trade_date_iso, expiration_iso, times_pt, _tick):
        # Guards
        if not trade_date_iso or not expiration_iso:
            return go.Figure(data=[go.Table(
                header=dict(values=["Time (PT)", "Stock", "ATM IV %", "Call Skew", "Put Skew",
                                    "Δ Call Skew %", "Δ Put Skew %", "Δ Stock %", "Δ ATM IV %",
                                    "ATM exp (SS) %", "ATM residual (bp)"],
                            fill_color="#444", font=dict(color="white")),
                cells=dict(values=[[] for _ in range(11)], fill_color="black", font=dict(color="white"))
            )]).update_layout(template="plotly_dark", title="Skew — select Trade Date & Expiration")

        if not times_pt:
            times_pt = ["06:31"]

        # If today's trade date, include the current PT minute during RTH
        now_pt = dt.datetime.now(PT_TZ)
        if trade_date_iso == now_pt.date().isoformat():
            now_hhmm = now_pt.strftime("%H:%M")
            if "06:30" <= now_hhmm <= "13:00":
                times_pt = sorted(set(times_pt + [now_hhmm]))

        # Sort times chronologically (PT)
        times_sorted = sorted(times_pt)

        rows = []
        prev_stock = None
        prev_atm = None
        prev_slope = None
        prev_T = None
        prev_call_skew_pp = None
        prev_put_skew_pp = None

        for hhmm_pt in times_sorted:
            ts_et = pt_minute_to_et(trade_date_iso, hhmm_pt)
            df = fetch_one_minute_monies(ts_et, TICKER, expiration_iso)
            row = _get_row(df)
            if row is None:
                continue

            # Representative stock for the minute (median if multiple quotes)
            stock = float(pd.to_numeric(df["stockPrice"], errors="coerce").median()) if "stockPrice" in df.columns else None

            # Skews & ATM
            atm, call_skew_pp, put_skew_pp = _skews_from_row(row)

            # Compute time to expiry and local slope at ATM from 25Δ wings
            T_years = _years_to_exp(ts_et, expiration_iso)
            c25_sigma = atm + call_skew_pp/100.0
            p25_sigma = atm + put_skew_pp/100.0
            slope = _estimate_slope(atm, T_years, c25_sigma, p25_sigma)

            # Expected ATM under sticky-strike using previous slice as the baseline
            atm_exp_ss = None
            atm_res_bp = None
            if prev_atm is not None and prev_stock is not None and stock is not None and prev_T is not None:
                prev_slice = TimeSlice(S=prev_stock, r=0.0, q=0.0, T=prev_T, sigma0=prev_atm, slope=prev_slope, curvature=0.0)
                new_slice  = TimeSlice(S=stock,      r=0.0, q=0.0, T=T_years, sigma0=atm,     slope=slope,     curvature=0.0)
                cmp = expected_vs_actual_atm(prev_slice, new_slice)
                atm_exp_ss = cmp["pred_atm_sticky_strike"] * 100.0
                atm_res_bp = (cmp["atm_actual"] - cmp["pred_atm_sticky_strike"]) * 10000.0

            # % changes vs previous slice
            d_stock       = _pct_change(stock,        prev_stock)
            d_atm         = _pct_change(atm,          prev_atm)
            d_call_skew   = _pct_change(call_skew_pp, prev_call_skew_pp)
            d_put_skew    = _pct_change(put_skew_pp,  prev_put_skew_pp)

            rows.append({
                "Time (PT)": hhmm_pt,
                "Stock": None if stock is None else round(stock, 2),
                "ATM IV %": round(atm * 100.0, 2),
                "Call Skew": round(call_skew_pp, 2),
                "Put Skew":  round(put_skew_pp,  2),
                "Δ Call Skew %": None if d_call_skew is None else round(d_call_skew, 2),
                "Δ Put Skew %":  None if d_put_skew  is None else round(d_put_skew,  2),
                "Δ Stock %":     None if d_stock     is None else round(d_stock,     2),
                "Δ ATM IV %":    None if d_atm       is None else round(d_atm,       2),
                "ATM exp (SS) %": None if atm_exp_ss is None else round(atm_exp_ss,  2),
                "ATM residual (bp)": None if atm_res_bp is None else int(round(atm_res_bp)),
            })

            prev_stock = stock
            prev_atm   = atm
            prev_slope = slope
            prev_T     = T_years
            prev_call_skew_pp = call_skew_pp
            prev_put_skew_pp  = put_skew_pp

        # Build the table figure
        headers = ["Time (PT)", "Stock", "ATM IV %", "Call Skew", "Put Skew",
                   "Δ Call Skew %", "Δ Put Skew %", "Δ Stock %", "Δ ATM IV %",
                   "ATM exp (SS) %", "ATM residual (bp)"]

        if not rows:
            return go.Figure(data=[go.Table(
                header=dict(values=headers, fill_color="#444", font=dict(color="white")),
                cells=dict(values=[[] for _ in headers], fill_color="black", font=dict(color="white"))
            )]).update_layout(template="plotly_dark", title=f"Skew — no data for {trade_date_iso} / {expiration_iso}")

        tdf = pd.DataFrame(rows)

        # Conditional backgrounds for change/residual columns
        color_cols = {'Δ Call Skew %', 'Δ Put Skew %', 'Δ Stock %', 'Δ ATM IV %', 'ATM residual (bp)'}
        cell_colors = []
        for col in tdf.columns:
            if col in color_cols:
                colors = [
                    'green' if (v is not None and v > 0) else
                    'red'   if (v is not None and v < 0) else
                    'black'
                    for v in tdf[col]
                ]
                cell_colors.append(colors)
            else:
                cell_colors.append(['black'] * len(tdf))

        fig = go.Figure(data=[go.Table(
            header=dict(values=list(tdf.columns), fill_color="#444", font=dict(color="white"), align="left"),
            cells=dict(values=[tdf[c] for c in tdf.columns], fill_color=cell_colors, align="left", font=dict(color="white"))
        )]).update_layout(template="plotly_dark", title=f"Skew (50Δ→25Δ) — {trade_date_iso}  Exp: {expiration_iso}")

        return fig
