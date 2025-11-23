
from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go
from sqlalchemy import create_engine, text
from typing import List

# ---- App IDs ----
TRADE_DATE_ID = "trade-date"
SMILE_TIME_INPUT = "smile-time-input"

# ---- Database Configuration ----
DATABASE_URL = "postgresql+psycopg://rschultz:5hUHvSVPDyVXhz7acgJZvlvnj7nFMDap@dpg-d38sm515pdvs738rknj0-a.oregon-postgres.render.com/curve_trading?sslmode=require"
DB_TABLE_NAME = "orats_monies_minute"
DB_ENGINE = create_engine(DATABASE_URL)

def fetch_term_metrics_data(trade_date_iso: str, times_pt: List[str]) -> pd.DataFrame:
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
            df = pd.read_sql(query, connection, params={"trade_date": trade_date_iso})
            if df.empty:
                return pd.DataFrame()

            df['expir_date'] = pd.to_datetime(df['expir_date'])
            df['trade_date'] = pd.to_datetime(df['trade_date'])
            df['dte'] = (df['expir_date'] - df['trade_date']).dt.days
            
            def get_closest_iv(group, dte_target):
                return group.iloc[(group['dte'] - dte_target).abs().argmin()]

            metrics = []
            for snapshot, group in df.groupby('snapshot_pt'):
                iv_3d = get_closest_iv(group, 3)['vol50']
                iv_30d = get_closest_iv(group, 30)['vol50']
                iv_90d = get_closest_iv(group, 90)['vol50']
                
                metrics.append({
                    "snapshot_pt": snapshot,
                    "front_back_spread": iv_3d - iv_30d,
                    "front_back_ratio": iv_3d / iv_30d,
                    "slope_30_90": iv_30d - iv_90d,
                })
            
            return pd.DataFrame(metrics)

    except Exception as e:
        print(f"Term Metrics DB query failed: {e}")
        return pd.DataFrame()

def register_callbacks(app):
    @app.callback(
        [
            Output("term-metrics-table", "data"),
            Output("term-metrics-graph", "figure"),
        ],
        [
            Input(TRADE_DATE_ID, "date"),
            Input(SMILE_TIME_INPUT, "value"),
        ],
    )
    def update_term_metrics(trade_date_iso, times_pt):
        if not trade_date_iso or not times_pt:
            return [], go.Figure(layout={"template": "plotly_dark", "title": "Select Trade Date and Time"})

        df = fetch_term_metrics_data(trade_date_iso, sorted(times_pt))

        if df.empty:
            return [], go.Figure(layout={"template": "plotly_dark", "title": "No Data Available"})

        latest_metrics = df.iloc[-1]
        table_data = [
            {"metric": "Front vs 1-month Spread", "value": f"{latest_metrics['front_back_spread']:.4f}"},
            {"metric": "Front vs 1-month Ratio", "value": f"{latest_metrics['front_back_ratio']:.4f}"},
            {"metric": "1M vs 3M Slope", "value": f"{latest_metrics['slope_30_90']:.4f}"},
        ]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['snapshot_pt'], y=df['front_back_spread'], name='Front/Back Spread'))
        fig.add_trace(go.Scatter(x=df['snapshot_pt'], y=df['slope_30_90'], name='30/90 Slope'))
        
        fig.update_layout(
            template="plotly_dark",
            title="Term Metrics Over Time",
            xaxis_title="Time",
            yaxis_title="Value",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        return table_data, fig
