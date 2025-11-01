# modules/Smile/callbacks.py — plot all buckets P10 → ATM → C10 for selected PT times
from __future__ import annotations
from typing import List, Tuple
import pandas as pd
import plotly.graph_objs as go
from dash import Input, Output

import ids
from options_orats import fetch_one_minute_monies, pt_minute_to_et

TICKER = "SPX"

# Call-delta buckets from 10Δ puts through ATM to 10Δ calls
CALL_DELTAS = [90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10]
VOL_COLS = [f"vol{d}" for d in CALL_DELTAS]

def _delta_label(call_delta: int) -> str:
    if call_delta > 50:      # put side
        return f"P{100 - call_delta}"   # e.g., 90 -> P10
    if call_delta == 50:
        return "ATM"
    return f"C{call_delta}"             # e.g., 25 -> C25

def _row_to_full_bucket_line(row: pd.Series) -> Tuple[List[str], List[float]]:
    """Convert a monies grid row into (labels, iv%) from P10 … ATM … C10."""
    labels, ivs = [], []
    for d in CALL_DELTAS:
        col = f"vol{d}"
        if col not in row.index:
            continue
        v = pd.to_numeric(row.get(col), errors="coerce")
        if pd.notna(v):
            labels.append(_delta_label(d))
            ivs.append(float(v) * 100.0)
    return labels, ivs

def register_callbacks(app):
    @app.callback(
        Output("SMILE_GRAPH", "figure"),
        Output("SMILE_TABLE", "figure"),
        Input(ids.TRADE_DATE_PICK, "date"),
        Input(ids.EXPIRATION_DATE_PICK, "date"),
        Input(ids.SMILE_TIME_INPUT, "value"),   # list of "HH:MM" (PT)
    )
    def render_smile_all_buckets(trade_date_iso, expiration_iso, times_pt):
        # guards
        if not trade_date_iso or not expiration_iso:
            fig = go.Figure().update_layout(
                template="plotly_dark",
                title="ORATS Smile Grid — select a Trade Date & Expiration",
                xaxis_title="Bucket (P10 … ATM … C10)", yaxis_title="IV (%)"
            )
            tbl = go.Figure(data=[go.Table(
                header=dict(values=["Time (PT)", "P10", "ATM", "C10"],
                            fill_color="black", font=dict(color="white"))
            )]).update_layout(template="plotly_dark", title="Selected Buckets")
            return fig, tbl

        if not times_pt:
            times_pt = ["06:35", "07:00", "08:00"]

        traces = []
        table_rows = []

        for hhmm_pt in times_pt:
            # Convert PT → ET and fetch one-minute monies (hist endpoint inside fetcher)
            ts_et = pt_minute_to_et(trade_date_iso, hhmm_pt)
            df = fetch_one_minute_monies(ts_et, TICKER, expiration_iso)
            if df is None or df.empty:
                continue

            row0 = df.iloc[0]
            labels, ivs = _row_to_full_bucket_line(row0)
            if not labels:
                continue

            traces.append(go.Scatter(x=labels, y=ivs, mode="lines+markers", name=f"{hhmm_pt} PT"))

            # Small table: P10 / ATM / C10
            label_to_iv = {l: v for l, v in zip(labels, ivs)}
            stock = float(pd.to_numeric(df.get("stockPrice"), errors="coerce").median()) if "stockPrice" in df.columns else None
            table_rows.append({
                "Time (PT)": hhmm_pt,
                "Stock": stock,
                "P10": label_to_iv.get("P10"),
                "ATM": label_to_iv.get("ATM"),
                "C10": label_to_iv.get("C10"),
            })

        if not traces:
            fig = go.Figure().update_layout(
                template="plotly_dark",
                title=f"No monies data for {trade_date_iso} / {expiration_iso} at selected times",
                xaxis_title="Bucket (P10 … ATM … C10)", yaxis_title="IV (%)"
            )
            tbl = go.Figure(data=[go.Table(
                header=dict(values=["Time (PT)", "Stock", "P10", "ATM", "C10"],
                            fill_color="black", font=dict(color="white"))
            )]).update_layout(template="plotly_dark", title="Selected Buckets")
            return fig, tbl

        fig = go.Figure(data=traces).update_layout(
            template="plotly_dark",
            title=f"ORATS Smile Grid — {trade_date_iso} (Exp: {expiration_iso})",
            xaxis_title="Bucket (P10 … ATM … C10)",
            yaxis_title="IV (%)"
        )

        tdf = pd.DataFrame(table_rows)
        table_fig = go.Figure(data=[go.Table(
            header=dict(values=list(tdf.columns), fill_color="black", font=dict(color="white"), align="left"),
            cells=dict(values=[tdf[c] for c in tdf.columns], align="left")
        )]).update_layout(template="plotly_dark", title="Selected Buckets")

        return fig, table_fig
