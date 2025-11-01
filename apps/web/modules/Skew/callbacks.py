# modules/Skew/callbacks.py — Skew table with live "now" minute + 1-min refresh
from __future__ import annotations
import datetime as dt
from typing import List, Tuple
import pandas as pd
import plotly.graph_objs as go
from dash import Input, Output

# IDs reused from the main app
TRADE_DATE_ID = "trade-date"
EXPIRATION_ID = "expiration-date-pick"
SMILE_TIME_INPUT = "smile-time-input"
SKEW_TABLE = "SKEW_TABLE"
CLOCK_ID = "CLOCK"

from shared.options_orats import fetch_one_minute_monies, pt_minute_to_et, PT_TZ

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
    """Return (stock, call_skew, put_skew) in IV percentage points: (C25−ATM, P25−ATM)*100."""
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
                header=dict(values=["Time (PT)", "Stock", "Call Skew", "Put Skew", "Δ Call Skew %", "Δ Put Skew %", "Δ Stock %", "Δ ATM IV %"],
                            fill_color="#444", font=dict(color="white")),
                cells=dict(values=[[], [], [], [], [], [], [], []], fill_color="black", font=dict(color="white"))
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
        prev_call = None
        prev_put  = None
        prev_atm = None

        for hhmm_pt in times_sorted:
            ts_et = pt_minute_to_et(trade_date_iso, hhmm_pt)
            df = fetch_one_minute_monies(ts_et, TICKER, expiration_iso)
            row = _get_row(df)
            if row is None:
                continue

            # Try to get a representative stock for the minute (median if multiple)
            if df is not None and "stockPrice" in df.columns:
                stock = float(pd.to_numeric(df["stockPrice"], errors="coerce").median())
            else:
                stock = None

            atm, call_skew, put_skew = _skews_from_row(row)

            d_stock = _pct_change(stock, prev_stock)
            d_call  = _pct_change(call_skew, prev_call)
            d_put   = _pct_change(put_skew,  prev_put)
            d_atm   = _pct_change(atm, prev_atm)

            rows.append({
                "Time (PT)": hhmm_pt,
                "Stock": None if stock is None else round(stock, 2),
                "Call Skew": round(call_skew, 2),
                "Put Skew":  round(put_skew,  2),
                "Δ Call Skew %": None if d_call  is None else round(d_call,  2),
                "Δ Put Skew %":  None if d_put   is None else round(d_put,   2),
                "Δ Stock %":     None if d_stock is None else round(d_stock, 2),
                "Δ ATM IV %":    None if d_atm   is None else round(d_atm,   2),
            })

            prev_stock = stock
            prev_call, prev_put = call_skew, put_skew
            prev_atm = atm

        # Build the table figure
        if not rows:
            return go.Figure(data=[go.Table(
                header=dict(values=["Time (PT)", "Stock", "Call Skew", "Put Skew", "Δ Call Skew %", "Δ Put Skew %", "Δ Stock %", "Δ ATM IV %"],
                            fill_color="#444", font=dict(color="white")),
                cells=dict(values=[[], [], [], [], [], [], [], []], fill_color="black", font=dict(color="white"))
            )]).update_layout(template="plotly_dark", title=f"Skew — no data for {trade_date_iso} / {expiration_iso}")

        tdf = pd.DataFrame(rows)

        # Conditional cell backgrounds for Δ columns
        cell_colors = []
        for col in tdf.columns:
            if col in ['Δ Call Skew %', 'Δ Put Skew %', 'Δ Stock %', 'Δ ATM IV %']:
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
