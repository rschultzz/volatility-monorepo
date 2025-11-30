# apps/web/modules/Ironbeam/callbacks.py
import os
import datetime as dt
import pandas as pd
from dash import Input, Output, State
from sqlalchemy import create_engine, text
import plotly.graph_objects as go
from zoneinfo import ZoneInfo

DB_TABLE_NAME = os.environ.get("IRONBEAM_BARS_TABLE", "ironbeam_es_1m_bars")
PUT_COLOR = os.getenv("GEX_PUT_COLOR", "#E5E7EB")
CALL_COLOR = os.getenv("GEX_CALL_COLOR", "#334155")


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
        Output("ironbeam-chart", "figure"),
        [Input("trade-date", "date"), Input("ironbeam-interval", "n_intervals")],
        State("ironbeam-chart", "relayoutData"),  # Read the current zoom/pan state
    )
    def update_chart(trade_date, n, relayout_data):
        if not trade_date:
            # Initially, no date might be selected
            return go.Figure(layout_title_text="Select a trade date to view chart.")

        # Parse the date and define the full day in Pacific Time
        try:
            selected_date = dt.datetime.strptime(trade_date, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            return go.Figure(layout_title_text="Invalid date format.")

        pt_tz = ZoneInfo("America/Los_Angeles")
        start_of_day_pt = dt.datetime.combine(selected_date, dt.time.min, tzinfo=pt_tz)
        # End of day is start of next day, for a non-inclusive upper bound
        end_of_day_pt = dt.datetime.combine(
            selected_date + dt.timedelta(days=1), dt.time.min, tzinfo=pt_tz
        )

        # Convert to UTC for the database query
        start_utc = start_of_day_pt.astimezone(ZoneInfo("UTC"))
        end_utc = end_of_day_pt.astimezone(ZoneInfo("UTC"))

        # Fetch data for the selected date
        query = text(
            f"""
            SELECT * FROM {DB_TABLE_NAME}
            WHERE datetime >= :start_date AND datetime < :end_date
            ORDER BY datetime ASC
        """
        )
        params = {"start_date": start_utc, "end_date": end_utc}

        try:
            with engine.connect() as connection:
                df = pd.read_sql(
                    query, connection, params=params, parse_dates=["datetime"]
                )
        except Exception as e:
            print(f"Error fetching data: {e}")
            return go.Figure(layout_title_text="Database error.")

        if df.empty:
            return go.Figure(
                layout_title_text=f"No data available for {selected_date.strftime('%Y-%m-%d')}."
            )

        df["datetime_pt"] = (
            df["datetime"].dt.tz_localize("UTC").dt.tz_convert("America/Los_Angeles")
        )

        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=df["datetime_pt"],
                    open=df["open"],
                    high=df["high"],
                    low=df["low"],
                    close=df["close"],
                    name="ES",
                    increasing_line_color=CALL_COLOR,
                    decreasing_line_color=PUT_COLOR,
                )
            ]
        )
        fig.update_layout(
            title=f"ES Front Month - 1m OHLC ({selected_date.strftime('%Y-%m-%d')})",
            xaxis_title="Time (Pacific Time)",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            hovermode="closest",
        )

        # Add crosshairs (spikes) that track the cursor and show values on axes
        fig.update_xaxes(
            showspikes=True,
            spikedash="dot",
            spikemode="across",
            spikesnap="cursor",
            hoverformat="%H:%M:%S",  # Format for time display on axis
        )
        fig.update_yaxes(
            showspikes=True,
            spikedash="dot",
            spikemode="across",
            spikesnap="cursor",
            hoverformat="%.2f",  # Format for price display on axis
        )

        # If the user has zoomed or panned, the relayout_data will contain the new ranges.
        # We check for 'xaxis.autorange' to know if the user has double-clicked to reset.
        if relayout_data and "xaxis.autorange" not in relayout_data:
            if "xaxis.range[0]" in relayout_data:
                fig.update_layout(
                    xaxis_range=[
                        relayout_data["xaxis.range[0]"],
                        relayout_data["xaxis.range[1]"],
                    ]
                )
            if "yaxis.range[0]" in relayout_data:
                fig.update_layout(
                    yaxis_range=[
                        relayout_data["yaxis.range[0]"],
                        relayout_data["yaxis.range[1]"],
                    ]
                )

        return fig
