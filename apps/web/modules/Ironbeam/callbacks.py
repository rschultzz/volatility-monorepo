# apps/web/modules/Ironbeam/callbacks.py
import os
import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State
from sqlalchemy import create_engine, text

# ---------- Config ----------
DB_TABLE_NAME = os.environ.get("IRONBEAM_BARS_TABLE", "ironbeam_es_1m_bars")

# Candle colors (reuse GEX colors so it all feels consistent)
PUT_COLOR = os.getenv("GEX_PUT_COLOR", "#E5E7EB")   # used for down candles
CALL_COLOR = os.getenv("GEX_CALL_COLOR", "#334155") # used for up candles

# Ticker for the GEX table
TICKER = os.getenv("GEX_TICKER", "SPX")

# How far above/below price we keep GEX levels (index points)
GEX_LEVEL_PADDING = float(os.getenv("GEX_LEVEL_PADDING", "150"))

# Only show heatmap where |net_gamma| >= this threshold (default: 10B)
GEX_ABS_THRESHOLD = float(os.getenv("GEX_ABS_THRESHOLD", "1e10"))

# Colorscale tuned for dark background:
# - strong negatives: bright cyan/blue
# - near zero: black (so background stays dark)
# - strong positives: bright orange/yellow
GEX_HEATMAP_COLORSCALE = [
    [0.0,  "#00f5ff"],  # strong negative
    [0.25, "#0099ff"],
    [0.5,  "#000000"],  # near zero
    [0.75, "#ff9b00"],
    [1.0,  "#ffe600"],  # strong positive
]


# ---------- DB engine ----------
def _get_db_url() -> str:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set in the environment")
    return db_url


db_url = _get_db_url()
engine = create_engine(db_url, pool_pre_ping=True)


# ---------- GEX helper ----------
def _fetch_gex_grouped_by_level(trade_date: dt.date) -> pd.DataFrame:
    """
    Mirror of gamma/callbacks logic:

    Returns columns:
      level (float), net_gamma (float)

    Source table:
      orats_oi_gamma with columns: trade_date, ticker,
      discounted_level, gex_call, gex_put

    Convention:
      - Sum gex_call and gex_put by rounded discounted_level
      - Puts are treated as negative (to match your gamma chart)
      - net_gamma = call_gamma + put_gamma
    """
    dialect = engine.dialect.name

    # Postgres vs SQLite rounding syntax
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
            COALESCE(SUM(gex_call), 0) AS call_gamma_raw,
            COALESCE(SUM(gex_put),  0) AS put_gamma_raw
        FROM orats_oi_gamma
        WHERE {" AND ".join(where)}
        GROUP BY {level_expr}
        ORDER BY {level_expr}
    """

    with engine.connect() as con:
        df = pd.read_sql(text(sql), con, params=params)

    if df.empty:
        return pd.DataFrame(
            columns=["level", "net_gamma"]
        ).astype({"level": "float64", "net_gamma": "float64"})

    # Match gamma/callbacks convention: puts negative, calls positive
    df["call_gamma"] = df["call_gamma_raw"].astype(float)
    df["put_gamma"] = -df["put_gamma_raw"].abs().astype(float)
    df["net_gamma"] = df["call_gamma"] + df["put_gamma"]

    df["level"] = df["level"].astype(float)

    return df[["level", "net_gamma"]]


# ---------- Dash callback registration ----------
def register_ironbeam_callbacks(app):
    @app.callback(
        Output("ironbeam-chart", "figure"),
        [Input("trade-date", "date"), Input("ironbeam-interval", "n_intervals")],
        State("ironbeam-chart", "relayoutData"),
    )
    def update_chart(trade_date, n, relayout_data):
        if not trade_date:
            return go.Figure(layout_title_text="Select a trade date to view chart.")

        # ----- Parse selected trade date (this is the SPX/ORATS trade_date) -----
        try:
            selected_date = dt.datetime.strptime(trade_date, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            return go.Figure(layout_title_text="Invalid date format.")

        pt_tz = ZoneInfo("America/Los_Angeles")

        # ----- Time window in PT:
        # For trade_date D, show ES from:
        #   D-1 15:00 PT  -->  D 13:00 PT
        # and plot GEX for trade_date = D across that whole window.
        start_pt = dt.datetime.combine(
            selected_date - dt.timedelta(days=1),
            dt.time(15, 0),
            tzinfo=pt_tz,
        )
        end_pt = dt.datetime.combine(
            selected_date,
            dt.time(13, 0),
            tzinfo=pt_tz,
        )

        # Convert to UTC for DB query
        start_utc = start_pt.astimezone(ZoneInfo("UTC"))
        end_utc = end_pt.astimezone(ZoneInfo("UTC"))

        # ----- Fetch 1m ES bars -----
        bars_query = text(
            f"""
            SELECT * FROM {DB_TABLE_NAME}
            WHERE datetime >= :start_date AND datetime < :end_date
            ORDER BY datetime ASC
        """
        )
        bars_params = {"start_date": start_utc, "end_date": end_utc}

        try:
            with engine.connect() as connection:
                df_bars = pd.read_sql(
                    bars_query, connection, params=bars_params, parse_dates=["datetime"]
                )
        except Exception as e:
            print(f"Error fetching bar data: {e}")
            return go.Figure(layout_title_text="Database error (bars).")

        if df_bars.empty:
            return go.Figure(
                layout_title_text=(
                    f"No bar data available for window "
                    f"{start_pt.strftime('%Y-%m-%d %H:%M')} – {end_pt.strftime('%Y-%m-%d %H:%M')} PT."
                )
            )

        # Convert to PT for display on x-axis
        df_bars["datetime_pt"] = (
            df_bars["datetime"].dt.tz_localize("UTC").dt.tz_convert("America/Los_Angeles")
        )

        # ----- Default axis ranges from price only -----
        y_min_price = float(df_bars["low"].min())
        y_max_price = float(df_bars["high"].max())
        y_pad = 0.01 * (y_max_price - y_min_price) if y_max_price > y_min_price else 1.0
        default_y_range = [y_min_price - y_pad, y_max_price + y_pad]

        x_min_time = df_bars["datetime_pt"].min()
        x_max_time = df_bars["datetime_pt"].max()
        default_x_range = [x_min_time, x_max_time]

        # ----- Persist zoom / pan from relayoutData -----
        x_range = None
        y_range = None

        if relayout_data:
            x_auto = relayout_data.get("xaxis.autorange", False)
            y_auto = relayout_data.get("yaxis.autorange", False)

            if (
                not x_auto
                and "xaxis.range[0]" in relayout_data
                and "xaxis.range[1]" in relayout_data
            ):
                x_range = [
                    relayout_data["xaxis.range[0]"],
                    relayout_data["xaxis.range[1]"],
                ]

            if (
                not y_auto
                and "yaxis.range[0]" in relayout_data
                and "yaxis.range[1]" in relayout_data
            ):
                y_range = [
                    relayout_data["yaxis.range[0]"],
                    relayout_data["yaxis.range[1]"],
                ]

        if x_range is None:
            x_range = default_x_range
        if y_range is None:
            y_range = default_y_range

        # ----- Fetch GEX levels for the same trade_date (D) -----
        try:
            df_gex = _fetch_gex_grouped_by_level(selected_date)
        except Exception as e:
            print(f"Error fetching GEX data: {e}")
            df_gex = pd.DataFrame(columns=["level", "net_gamma"])

        # ----- Filter GEX to a band around price to avoid clutter -----
        if not df_gex.empty:
            band_min = y_min_price - GEX_LEVEL_PADDING
            band_max = y_max_price + GEX_LEVEL_PADDING
            df_gex = df_gex[(df_gex["level"] >= band_min) & (df_gex["level"] <= band_max)]

        # ---------- Build figure ----------
        fig = go.Figure()

        # 1) GEX heatmap (tomorrow's GEX, applied from 15:00 prev day to 13:00 today)
        if not df_gex.empty:
            times = df_bars["datetime_pt"].values
            levels = df_gex["level"].values
            net_gex = df_gex["net_gamma"].values

            # Tile net GEX horizontally across all timestamps
            z = np.tile(net_gex.reshape(-1, 1), (1, len(times)))

            # Apply magnitude threshold: values below |threshold| become NaN (transparent)
            if GEX_ABS_THRESHOLD > 0:
                mask = np.abs(z) < GEX_ABS_THRESHOLD
                z = np.where(mask, np.nan, z)
                strong = net_gex[np.abs(net_gex) >= GEX_ABS_THRESHOLD]
            else:
                strong = net_gex

            # Use strongest values for cmin/cmax so big walls pop
            if strong.size:
                max_abs_gamma = float(np.nanmax(np.abs(strong)))
            else:
                max_abs_gamma = float(np.nanmax(np.abs(net_gex))) if net_gex.size else 0.0

            fig.add_trace(
                go.Heatmap(
                    x=times,
                    y=levels,
                    z=z,
                    coloraxis="coloraxis",
                    opacity=0.45,  # colors visible but candles still readable
                    hovertemplate=(
                        "Time=%{x|%H:%M}<br>"
                        "Level=%{y}<br>"
                        "Net GEX=%{z:.3g}<extra></extra>"
                    ),
                    zauto=False,
                )
            )

            if max_abs_gamma > 0:
                fig.update_layout(
                    coloraxis=dict(
                        colorscale=GEX_HEATMAP_COLORSCALE,
                        cmin=-max_abs_gamma,
                        cmax=max_abs_gamma,
                        colorbar_title="Net GEX",
                    )
                )
            else:
                fig.update_layout(
                    coloraxis=dict(
                        colorscale=GEX_HEATMAP_COLORSCALE,
                        colorbar_title="Net GEX",
                    )
                )
        else:
            fig.add_annotation(
                text="No GEX data for this trade date",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.99,
                y=0.99,
                xanchor="right",
                yanchor="top",
                font=dict(size=10),
            )

        # 2) ES candlesticks on top
        fig.add_trace(
            go.Candlestick(
                x=df_bars["datetime_pt"],
                open=df_bars["open"],
                high=df_bars["high"],
                low=df_bars["low"],
                close=df_bars["close"],
                name="ES",
                increasing=dict(line=dict(color=CALL_COLOR, width=1.2)),
                decreasing=dict(line=dict(color=PUT_COLOR, width=1.2)),
            )
        )

        fig.update_layout(
            title=(
                "ES Front Month - 1m OHLC with Net GEX Heatmap "
                f"(GEX for {selected_date.isoformat()}, "
                f"window {start_pt.strftime('%Y-%m-%d %H:%M')}–{end_pt.strftime('%Y-%m-%d %H:%M')} PT)"
            ),
            xaxis_title="Time (Pacific Time)",
            yaxis_title="Price / Discounted Level",
            template="plotly_dark",
            hovermode="closest",
            # X-axis rangeslider: Zoom slider (only exists for x in Plotly)
            xaxis=dict(
                range=x_range,
                rangeslider=dict(visible=True),
                showspikes=True,
                spikedash="dot",
                spikemode="across",
                spikesnap="cursor",
                hoverformat="%H:%M:%S",
            ),
            yaxis=dict(
                range=y_range,
                showspikes=True,
                spikedash="dot",
                spikemode="across",
                spikesnap="cursor",
                hoverformat="%.2f",
            ),
        )

        return fig
