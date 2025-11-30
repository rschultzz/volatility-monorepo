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


def _fetch_gex_for_heatmap(trade_date: dt.date) -> pd.Series:
    """
    Returns a pandas Series with level as the index and total_gamma as the values.

    Source table:
      orats_oi_gamma with columns: trade_date, ticker, discounted_level, gex_call, gex_put
    """
    dialect = engine.dialect.name
    level_expr = (
        "ROUND(discounted_level)::INT"
        if dialect == "postgresql"
        else "CAST(ROUND(discounted_level) AS INTEGER)"
    )

    where = ["trade_date = :d", "discounted_level IS NOT NULL"]
    params: dict[str, object] = {"d": trade_date.isoformat()}
    if TICKER:
        where.append("ticker = :tkr")
        params["tkr"] = TICKER

    sql = f"""
        SELECT
            {level_expr} AS level,
            COALESCE(SUM(gex_call), 0) + COALESCE(SUM(gex_put), 0) AS total_gamma
        FROM orats_oi_gamma
        WHERE {" AND ".join(where)}
        GROUP BY {level_expr}
        ORDER BY {level_expr}
    """

    with engine.connect() as con:
        df = pd.read_sql(text(sql), con, params=params, index_col="level")

    if df.empty:
        return pd.Series(dtype="float64", name="total_gamma")
    return df["total_gamma"]


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

            query = text(
                f"""
                SELECT * FROM {DB_TABLE_NAME}
                WHERE datetime >= :start_date AND datetime < :end_date
                ORDER BY datetime ASC
            """
            )
            params = {"start_date": start_utc, "end_date": end_utc}

            with engine.connect() as connection:
                df = pd.read_sql(
                    query, connection, params=params, parse_dates=["datetime"]
                )

            gex_series = _fetch_gex_for_heatmap(selected_date)

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

        fig = go.Figure()

        # Add Candlestick chart
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

        # Add Gamma Heatmap
        if not gex_series.empty:
            z_data = [
                [gamma_val] * len(df["datetime_pt"])
                for gamma_val in gex_series.values
            ]
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
