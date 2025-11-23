from __future__ import annotations
import pandas as pd
import plotly.graph_objs as go
from dash import Input, Output
from sqlalchemy import create_engine, text
from typing import List

# ---- App IDs ----
TRADE_DATE_ID = "trade-date"
SMILE_TIME_INPUT = "smile-time-input"
TERM_STRUCTURE_GRAPH_ID = "term-structure-graph"

# ---- Database Configuration ----
DATABASE_URL = "postgresql+psycopg://rschultz:5hUHvSVPDyVXhz7acgJZvlvnj7nFMDap@dpg-d38sm515pdvs738rknj0-a.oregon-postgres.render.com/curve_trading?sslmode=require"
DB_TABLE_NAME = "orats_monies_minute"
DB_ENGINE = create_engine(DATABASE_URL)

# ---- Plotting Constants ----
COLORWAY = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"
]

# ----------------- Data Fetching -----------------
def fetch_term_structure_data(trade_date_iso: str, times_pt: List[str]) -> pd.DataFrame:
    if not trade_date_iso or not times_pt:
        return pd.DataFrame()
    time_filters = [f"'{trade_date_iso} {hhmm}:00'" for hhmm in times_pt]
    query = text(f"""
        SELECT snapshot_pt, trade_date, expir_date, vol50
        FROM "{DB_TABLE_NAME}"
        WHERE trade_date = :trade_date AND snapshot_pt IN ({','.join(time_filters)})
        ORDER BY snapshot_pt, expir_date;
    """)
    try:
        with DB_ENGINE.connect() as connection:
            return pd.read_sql(query, connection, params={"trade_date": trade_date_iso})
    except Exception as e:
        print(f"Term Structure DB query failed: {e}")
        return pd.DataFrame()

# ----------------- Main Callback -----------------
def register_callbacks(app):
    @app.callback(
        Output(TERM_STRUCTURE_GRAPH_ID, "figure"),
        [Input(TRADE_DATE_ID, "date"),
         Input(SMILE_TIME_INPUT, "value")]
    )
    def render_term_structure_graph(trade_date_iso, times_pt):
        # uirevision preserves user interactions (like zoom) as long as the key doesn't change.
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
                # Set the initial range, but allow the user to change it by zooming.
                # uirevision will preserve their changes.
                range=[0, 90]
            )
            # By not specifying yaxis.range or yaxis.autorange, we allow Plotly's
            # default behavior, which includes interactive zooming on the axis.
        )

        if not trade_date_iso or not times_pt:
            fig.update_layout(title="Term Structure — Select a Trade Date and Time")
            return fig

        fig.update_layout(title=f"ATM Term Structure — {trade_date_iso}")
        
        df = fetch_term_structure_data(trade_date_iso, sorted(times_pt))
        
        if df.empty:
            return fig

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

        return fig
