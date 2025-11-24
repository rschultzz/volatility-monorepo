from __future__ import annotations
import pandas as pd
import plotly.graph_objs as go
from dash import Input, Output
from typing import List

from packages.shared.utils import fetch_term_structure_data

# ---- App IDs ----
TRADE_DATE_ID = "trade-date"
SMILE_TIME_INPUT = "smile-time-input"
TERM_STRUCTURE_GRAPH_ID = "term-structure-graph"
LIVE_DATA_STORE_ID = "live-data-store"

# ---- Plotting Constants ----
COLORWAY = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"
]
LIVE_COLOR = "#FFD700"

# ----------------- Main Callback -----------------
def register_callbacks(app):
    @app.callback(
        Output(TERM_STRUCTURE_GRAPH_ID, "figure"),
        [Input(TRADE_DATE_ID, "date"),
         Input(SMILE_TIME_INPUT, "value"),
         Input(LIVE_DATA_STORE_ID, "data")]
    )
    def render_term_structure_graph(trade_date_iso, times_pt, live_data_json):
        fig = go.Figure(layout={"uirevision": trade_date_iso})

        fig.update_layout(
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=40),
            xaxis_title="DTE",
            yaxis_title="ATM IV (%)",
            legend=dict(orientation="v", x=1.02, y=1.0, bgcolor="rgba(0,0,0,0)"),
            colorway=COLORWAY,
            xaxis=dict(
                rangeslider=dict(visible=True),
                range=[0, 90]
            )
        )

        if not trade_date_iso:
            fig.update_layout(title="Term Structure — Select a Trade Date")
            return fig

        fig.update_layout(title=f"ATM Term Structure — {trade_date_iso}")
        
        if times_pt:
            df = fetch_term_structure_data(trade_date_iso, sorted(times_pt))
            if not df.empty:
                df['expir_date'] = pd.to_datetime(df['expir_date'])
                df['trade_date'] = pd.to_datetime(df['trade_date'])
                df['dte'] = (df['expir_date'] - df['trade_date']).dt.days
                
                for i, (snapshot, group) in enumerate(df.groupby('snapshot_pt')):
                    hhmm_pt = snapshot.strftime("%H:%M")
                    color = COLORWAY[i % len(COLORWAY)]
                    
                    fig.add_trace(go.Scatter(
                        x=group['dte'],
                        y=group['vol50'] * 100.0,
                        mode='lines+markers',
                        name=f"{hhmm_pt} PT",
                        line=dict(width=2, color=color),
                        marker=dict(size=6)
                    ))

        if live_data_json:
            df_live = pd.read_json(live_data_json, orient="split")
            if not df_live.empty:
                df_live['expir_date'] = pd.to_datetime(df_live['expir_date'])
                df_live['trade_date'] = pd.to_datetime(df_live['trade_date'])
                df_live['dte'] = (df_live['expir_date'] - df_live['trade_date']).dt.days
                
                fig.add_trace(go.Scatter(
                    x=df_live['dte'],
                    y=df_live['vol50'] * 100.0,
                    mode='lines+markers',
                    name="Live",
                    line=dict(width=3, color=LIVE_COLOR),
                    marker=dict(size=6)
                ))

        return fig
