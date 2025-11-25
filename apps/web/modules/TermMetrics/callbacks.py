from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go
from typing import List
import numpy as np
import os
from io import StringIO

from packages.shared.utils import fetch_term_metrics_data, is_market_hours

# ---- App IDs ----
TRADE_DATE_ID = "trade-date"
SMILE_TIME_INPUT = "smile-time-input"
LIVE_DATA_STORE_ID = "live-data-store"

# ---- Constants ----
LIVE_COLOR = "#FFD700"

def _pct_change(curr: float | None, prev: float | None) -> float | None:
    if prev in (None, 0) or curr is None or pd.isna(prev) or pd.isna(curr):
        return None
    return (curr - prev) / abs(prev) * 100.0

def calculate_term_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    df["expir_date"] = pd.to_datetime(df["expir_date"])
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["dte"] = (df["expir_date"] - df["trade_date"]).dt.days

    def get_closest_iv(group, dte_target):
        if group.empty:
            return np.nan
        closest_row = group.iloc[(group["dte"] - dte_target).abs().argmin()]
        return closest_row["vol50"]

    iv_3d = get_closest_iv(df, 3)
    iv_30d = get_closest_iv(df, 30)
    iv_90d = get_closest_iv(df, 90)

    front_back_spread = (iv_3d - iv_30d) if not (pd.isna(iv_3d) or pd.isna(iv_30d)) else np.nan
    front_back_ratio = (iv_3d / iv_30d) if not (pd.isna(iv_3d) or pd.isna(iv_30d) or iv_30d == 0) else np.nan
    slope_30_90 = (iv_30d - iv_90d) if not (pd.isna(iv_30d) or pd.isna(iv_90d)) else np.nan

    return pd.DataFrame([{
        "front_back_spread": front_back_spread,
        "front_back_ratio": front_back_ratio,
        "slope_30_90": slope_30_90,
    }])

def register_callbacks(app):
    @app.callback(
        Output("term-metrics-table", "figure"),
        [
            Input(TRADE_DATE_ID, "date"),
            Input(SMILE_TIME_INPUT, "value"),
            Input(LIVE_DATA_STORE_ID, "data"),
        ],
    )
    def update_term_metrics(trade_date_iso, times_pt, live_data_json):
        cols = [
            "Time (PT)", "3D-30D Spread", "Δ 3D-30D Spread %",
            "3D/30D Ratio", "Δ 3D/30D Ratio %", "30D-90D Slope", "Δ 30D-90D Slope %",
        ]
        delta_cols = {"Δ 3D-30D Spread %", "Δ 3D/30D Ratio %", "Δ 30D-90D Slope %"}

        if not trade_date_iso:
            empty_values = [[] for _ in cols]
            fig = go.Figure(data=[go.Table(header=dict(values=cols), cells=dict(values=empty_values, align="left", font=dict(color="white"), fill_color="black"))])
            fig.update_layout(template="plotly_dark", title="Term Metrics", margin=dict(l=0, r=0, t=36, b=0))
            return fig

        table_rows = []
        prev_spread = prev_ratio = prev_slope = None

        if times_pt:
            df = fetch_term_metrics_data(trade_date_iso, sorted(times_pt))
            if not df.empty:
                for _, row in df.iterrows():
                    spread = float(row["front_back_spread"]) if pd.notna(row["front_back_spread"]) else None
                    ratio = float(row["front_back_ratio"]) if pd.notna(row["front_back_ratio"]) else None
                    slope = float(row["slope_30_90"]) if pd.notna(row["slope_30_90"]) else None

                    d_spread = _pct_change(spread, prev_spread)
                    d_ratio = _pct_change(ratio, prev_ratio)
                    d_slope = _pct_change(slope, prev_slope)

                    table_rows.append({
                        "Time (PT)": pd.to_datetime(row["snapshot_pt"]).strftime("%H:%M"),
                        "3D-30D Spread": round(spread, 4) if spread is not None else None,
                        "Δ 3D-30D Spread %": round(d_spread, 2) if d_spread is not None else None,
                        "3D/30D Ratio": round(ratio, 4) if ratio is not None else None,
                        "Δ 3D/30D Ratio %": round(d_ratio, 2) if d_ratio is not None else None,
                        "30D-90D Slope": round(slope, 4) if slope is not None else None,
                        "Δ 30D-90D Slope %": round(d_slope, 2) if d_slope is not None else None,
                    })
                    prev_spread, prev_ratio, prev_slope = spread, ratio, slope

        if live_data_json and is_market_hours():
            df_live = pd.read_json(StringIO(live_data_json), orient="split")
            if not df_live.empty:
                live_metrics = calculate_term_metrics(df_live)
                if not live_metrics.empty:
                    live_row = live_metrics.iloc[0]
                    spread = float(live_row["front_back_spread"]) if pd.notna(live_row["front_back_spread"]) else None
                    ratio = float(live_row["front_back_ratio"]) if pd.notna(live_row["front_back_ratio"]) else None
                    slope = float(live_row["slope_30_90"]) if pd.notna(live_row["slope_30_90"]) else None

                    d_spread = _pct_change(spread, prev_spread)
                    d_ratio = _pct_change(ratio, prev_ratio)
                    d_slope = _pct_change(slope, prev_slope)

                    table_rows.append({
                        "Time (PT)": "Live",
                        "3D-30D Spread": round(spread, 4) if spread is not None else None,
                        "Δ 3D-30D Spread %": round(d_spread, 2) if d_spread is not None else None,
                        "3D/30D Ratio": round(ratio, 4) if ratio is not None else None,
                        "Δ 3D/30D Ratio %": round(d_ratio, 2) if d_ratio is not None else None,
                        "30D-90D Slope": round(slope, 4) if slope is not None else None,
                        "Δ 30D-90D Slope %": round(d_slope, 2) if d_slope is not None else None,
                    })

        if not table_rows:
            return go.Figure(data=[go.Table(header=dict(values=cols), cells=dict(values=[[] for _ in cols]))]).update_layout(template="plotly_dark", title=f"Term Metrics — No data for {trade_date_iso}", margin=dict(l=0, r=0, t=36, b=0))

        df_table = pd.DataFrame(table_rows, columns=cols)
        cell_colors = [
            ["green" if (col in delta_cols and v is not None and v > 0) else "red" if (col in delta_cols and v is not None and v < 0) else "black" for v in df_table[col]]
            for col in df_table.columns
        ]
        
        fig = go.Figure(data=[go.Table(header=dict(values=cols), cells=dict(values=[df_table[c] for c in cols], align="left", font=dict(color='white'), fill_color=cell_colors))])
        fig.update_layout(template="plotly_dark", title="Term Metrics", margin=dict(l=0, r=0, t=36, b=0))
        return fig
