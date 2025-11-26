# apps/web/modules/Ironbeam/callbacks.py
import os
import datetime as dt
import pandas as pd
from dash import Input, Output
from sqlalchemy import create_engine
import plotly.graph_objects as go

DB_TABLE_NAME = os.environ.get("IRONBEAM_BARS_TABLE", "ironbeam_es_1m_bars")

def _get_db_url() -> str:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set in the environment")
    return db_url

def register_ironbeam_callbacks(app):
    @app.callback(
        Output('ironbeam-chart', 'figure'),
        Input('ironbeam-interval', 'n_intervals')
    )
    def update_chart(n):
        db_url = _get_db_url()
        engine = create_engine(db_url)

        # Fetch the last 1 hour of data
        query = f"""
        SELECT * FROM {DB_TABLE_NAME}
        WHERE datetime >= NOW() - INTERVAL '1 hour'
        ORDER BY datetime ASC
        """
        try:
            df = pd.read_sql(query, engine)
        except Exception as e:
            print(f"Error fetching data: {e}")
            return go.Figure()

        if df.empty:
            return go.Figure(layout_title_text="No data available for the last hour.")

        fig = go.Figure(data=[go.Candlestick(
            x=df['datetime'],
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close']
        )])

        fig.update_layout(
            title='ES Front Month - 1m OHLC (Last Hour)',
            xaxis_title='Time',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False,
            template="plotly_dark"
        )
        return fig
