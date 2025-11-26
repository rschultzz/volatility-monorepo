# apps/web/modules/Ironbeam/callbacks.py
import os
import datetime as dt
import pandas as pd
from dash import Input, Output, State, no_update, Patch
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
        Output('ironbeam-latest-ts-store', 'data'),
        Input('ironbeam-interval', 'n_intervals'),
        State('ironbeam-latest-ts-store', 'data')
    )
    def update_chart(n, latest_ts_str):
        # On the first load, latest_ts_str will be None. This is a more robust
        # way to detect the initial load.
        if not latest_ts_str:
            query = text(f"""
                SELECT * FROM {DB_TABLE_NAME}
                WHERE datetime >= (NOW() AT TIME ZONE 'utc') - INTERVAL '1 hour'
                ORDER BY datetime ASC
            """)
            try:
                df = pd.read_sql(query, engine, parse_dates=['datetime'])
            except Exception as e:
                print(f"Error on initial data load: {e}")
                return go.Figure(layout_title_text="Database error on initial load."), no_update

            if df.empty:
                return go.Figure(layout_title_text="No data available for the last hour."), None

            df['datetime_pt'] = df['datetime'].dt.tz_localize('UTC').dt.tz_convert('America/Los_Angeles')
            new_latest_ts = df['datetime'].iloc[-1].isoformat()

            fig = go.Figure(data=[go.Candlestick(
                x=df['datetime_pt'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='ES'
            )])
            fig.update_layout(
                title='ES Front Month - 1m OHLC (Live)',
                xaxis_title='Time (Pacific Time)',
                yaxis_title='Price',
                xaxis_rangeslider_visible=False,
                template="plotly_dark"
            )
            return fig, new_latest_ts

        # For all subsequent updates, fetch only new data and patch the figure
        query = text(f"""
            SELECT * FROM {DB_TABLE_NAME}
            WHERE datetime > :latest_ts
            ORDER BY datetime ASC
        """)
        try:
            df_new = pd.read_sql(query, engine, params={'latest_ts': latest_ts_str}, parse_dates=['datetime'])
        except Exception as e:
            print(f"Error fetching new data: {e}")
            return no_update, no_update

        if df_new.empty:
            return no_update, no_update

        df_new['datetime_pt'] = df_new['datetime'].dt.tz_localize('UTC').dt.tz_convert('America/Los_Angeles')
        new_latest_ts = df_new['datetime'].iloc[-1].isoformat()

        # Create a Patch object to extend the existing trace.
        # It's crucial to convert the pandas Series to lists with .tolist()
        patched_figure = Patch()
        patched_figure['data'][0]['x'].extend(df_new['datetime_pt'].tolist())
        patched_figure['data'][0]['open'].extend(df_new['open'].tolist())
        patched_figure['data'][0]['high'].extend(df_new['high'].tolist())
        patched_figure['data'][0]['low'].extend(df_new['low'].tolist())
        patched_figure['data'][0]['close'].extend(df_new['close'].tolist())

        return patched_figure, new_latest_ts
