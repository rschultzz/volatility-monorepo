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
TICKER = os.getenv("GEX_TICKER", "SPX")


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
            return go.Figure(layout_title_text="Select a trade date to view chart.")

        try:
            selected_date = dt.datetime.strptime(trade_date, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            return go.Figure(layout_title_text="Invalid date format.")

        # --- Data Fetching ---
        try:
            pt_tz = ZoneInfo("America/Los_Angeles")
            start_of_day_pt = dt.datetime.combine(
                selected_date, dt.time.min, tzinfo=pt_tz
            )
            end_of_day_pt = dt.datetime.combine(
                selected_date + dt.timedelta(days=1), dt.time.min, tzinfo=pt_tz
            )
            start_utc = start_of_day_pt.astimezone(ZoneInfo("UTC"))
            end_utc = end_of_day_pt.astimezone(ZoneInfo("UTC"))

            # Use a single connection for both queries
            with engine.connect() as connection:
                # Fetch OHLC data
                query_ohlc = text(
                    f"SELECT * FROM {DB_TABLE_NAME} WHERE datetime >= :start_date AND datetime < :end_date ORDER BY datetime ASC"
                )
                params_ohlc = {"start_date": start_utc, "end_date": end_utc}
                df = pd.read_sql(
                    query_ohlc, connection, params=params_ohlc, parse_dates=["datetime"]
                )

                # Fetch Gamma data
                dialect = connection.dialect.name
                level_expr = (
                    "ROUND(discounted_level)::INT"
                    if dialect == "postgresql"
                    else "CAST(ROUND(discounted_level) AS INTEGER)"
                )
                where = ["trade_date = :d", "discounted_level IS NOT NULL"]
                params_gex: dict[str, object] = {"d": selected_date.isoformat()}
                if TICKER:
                    where.append("ticker = :tkr")
                    params_gex["tkr"] = TICKER

                query_gex = text(
                    f"""
                    SELECT {level_expr} AS level, COALESCE(SUM(gex_call), 0) + COALESCE(SUM(gex_put), 0) AS total_gamma
                    FROM orats_oi_gamma WHERE {" AND ".join(where)}
                    GROUP BY {level_expr} ORDER BY {level_expr}
                """
                )
                gex_df = pd.read_sql(
                    query_gex, connection, params=params_gex, index_col="level"
                )
                gex_series = (
                    gex_df["total_gamma"]
                    if not gex_df.empty
                    else pd.Series(dtype="float64")
                )

        except Exception as e:
            print(f"Error fetching data for {trade_date}: {e}")
            return go.Figure(layout_title_text="Database error.")

        # --- Figure Creation ---
        if df.empty:
            return go.Figure(
                layout_title_text=f"No OHLC data for {selected_date.strftime('%Y-%m-%d')}."
            )

        df["datetime_pt"] = (
            df["datetime"].dt.tz_localize("UTC").dt.tz_convert("America/Los_Angeles")
        )

        fig = go.Figure()

        # Add Candlestick trace
        fig.add_trace(
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
        )

        # Add Gamma Heatmap trace
        if not gex_series.empty:
            z_data = []
            for gamma_value in gex_series.values:
                z_data.append([gamma_value] * len(df["datetime_pt"]))

            fig.add_trace(
                go.Heatmap(
                    x=df["datetime_pt"],
                    y=gex_series.index,
                    z=z_data,
                    colorscale="gray",
                    reversescale=True,
                    showscale=False,
                    name="Gamma",
                    opacity=0.7,
                )
            )

        # --- Layout and Axes ---
        fig.update_layout(
            title=f"ES Front Month - 1m OHLC with Gamma Overlay ({selected_date.strftime('%Y-%m-%d')})",
            xaxis_title="Time (Pacific Time)",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False,
            template="plotly_dark",
            hovermode="closest",
        )

        fig.update_xaxes(
            showspikes=True,
            spikedash="dot",
            spikemode="across",
            spikesnap="cursor",
            hoverformat="%H:%M:%S",
        )
        fig.update_yaxes(
            showspikes=True,
            spikedash="dot",
            spikemode="across",
            spikesnap="cursor",
            hoverformat="%.2f",
        )

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
