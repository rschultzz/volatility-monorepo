# apps/web/modules/Ironbeam/callbacks.py
import os
import datetime as dt
import pandas as pd
from dash import Input, Output, State
from sqlalchemy import create_engine, text
import plotly.graph_objects as go
from zoneinfo import ZoneInfo

DB_TABLE_NAME = os.environ.get("IRONBEAM_BARS_TABLE", "ironbeam_es_1m_bars")

def _get_db_url() -> str:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set in the environment")
    return db_url

# Create the engine once when the module is loaded.
# pool_pre_ping=True tells the engine to check if connections are alive before using them.
db_url = _get_db_url()
engine = create_engine(db_url, pool_pre_ping=True)

def register_ironbeam_callbacks(app):
    @app.callback(
        Output('ironbeam-chart', 'figure'),
        Input('ironbeam-interval', 'n_intervals'),
        State('ironbeam-chart', 'relayoutData')  # Read the current zoom/pan state
    )
    def update_chart(n, relayout_data):
        # Fetch the last 24 hours of data
        query = text(f"""
            SELECT * FROM {DB_TABLE_NAME}
            WHERE datetime >= (NOW() AT TIME ZONE 'utc') - INTERVAL '24 hours'
            ORDER BY datetime ASC
        """)
        try:
            with engine.connect() as connection:
                df = pd.read_sql(query, connection, parse_dates=['datetime'])
        except Exception as e:
            print(f"Error fetching data: {e}")
            return go.Figure(layout_title_text="Database error.")

        if df.empty:
            return go.Figure(layout_title_text="No data available for the last 24 hours.")

        df['datetime_pt'] = df['datetime'].dt.tz_localize('UTC').dt.tz_convert('America/Los_Angeles')

        fig = go.Figure(data=[go.Candlestick(
            x=df['datetime_pt'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='ES'
        )])
        fig.update_layout(
            title='ES Front Month - 1m OHLC (Last 24 Hours)',
            xaxis_title='Time (Pacific Time)',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False,
            template="plotly_dark"
        )

        # If the user has zoomed or panned, the relayout_data will contain the new ranges.
        # We check for 'xaxis.autorange' to know if the user has double-clicked to reset.
        if relayout_data and 'xaxis.autorange' not in relayout_data:
            if 'xaxis.range[0]' in relayout_data:
                fig.update_layout(
                    xaxis_range=[relayout_data['xaxis.range[0]'], relayout_data['xaxis.range[1]']]
                )
            if 'yaxis.range[0]' in relayout_data:
                fig.update_layout(
                    yaxis_range=[relayout_data['yaxis.range[0]'], relayout_data['yaxis.range[1]']]
                )

        return fig
