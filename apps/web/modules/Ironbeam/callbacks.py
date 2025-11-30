# apps/web/modules/Ironbeam/callbacks.py
import os
import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State
from dash.exceptions import PreventUpdate
from sqlalchemy import create_engine, text

# ---------- Config ----------
DB_TABLE_NAME = os.environ.get("IRONBEAM_BARS_TABLE", "ironbeam_es_1m_bars")

# Candle colors (reuse GEX colors so it all feels consistent)
PUT_COLOR = os.getenv("GEX_PUT_COLOR", "#E5E7EB")   # down candles
CALL_COLOR = os.getenv("GEX_CALL_COLOR", "#60a5fa") # up candles

# Highlight color for selected slices
HIGHLIGHT_COLOR = os.getenv("IRONBEAM_HIGHLIGHT_COLOR", "#ef4444")  # red

# Ticker for the GEX table
TICKER = os.getenv("GEX_TICKER", "SPX")

# How far above/below price we keep GEX levels (index points)
GEX_LEVEL_PADDING = float(os.getenv("GEX_LEVEL_PADDING", "150"))

# Default minimum |GEX| for plotting (in raw units, not billions).
# Slider overrides this.
GEX_ABS_THRESHOLD_DEFAULT = float(os.getenv("GEX_ABS_THRESHOLD", "1e10"))

# Color span logic:
# - If GEX_COLOR_ABS_MAX > 0, we clamp colors to ±that value
# - Else we use the given percentile of |net_gamma| (default: 95)
GEX_COLOR_ABS_MAX = float(os.getenv("GEX_COLOR_ABS_MAX", "0"))
GEX_COLOR_PERCENTILE = float(os.getenv("GEX_COLOR_PERCENTILE", "95"))

# Colorscale tuned for dark background:
#   - strong negatives: blue
#   - near zero: dark slate (almost invisible)
#   - strong positives: green
GEX_HEATMAP_COLORSCALE = [
    [0.0,  "#1d4ed8"],  # strong negative (deep blue)
    [0.25, "#60a5fa"],  # medium negative (lighter blue)
    [0.5,  "#020617"],  # near zero (very dark slate / bg)
    [0.75, "#22c55e"],  # medium positive (green)
    [1.0,  "#bbf7d0"],  # strong positive (pale green)
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
    Returns columns:
      level, call_gamma, put_gamma, net_gamma

    Source table:
      orats_oi_gamma with columns: trade_date, ticker,
      discounted_level, gex_call, gex_put

    Convention:
      - puts are negative
      - net_gamma = call_gamma + put_gamma
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
            columns=["level", "call_gamma", "put_gamma", "net_gamma"]
        ).astype(
            {
                "level": "float64",
                "call_gamma": "float64",
                "put_gamma": "float64",
                "net_gamma": "float64",
            }
        )

    df["call_gamma"] = df["call_gamma_raw"].astype(float)
    df["put_gamma"] = -df["put_gamma_raw"].abs().astype(float)  # puts negative
    df["net_gamma"] = df["call_gamma"] + df["put_gamma"]
    df["level"] = df["level"].astype(float)

    return df[["level", "call_gamma", "put_gamma", "net_gamma"]]


# ---------- Dash callback registration ----------
def register_ironbeam_callbacks(app):
    # ---- Main ES + GEX chart ----
    @app.callback(
        Output("ironbeam-chart", "figure"),
        [
            Input("trade-date", "date"),
            Input("ironbeam-interval", "n_intervals"),
            Input("gex-threshold-billions", "value"),  # slider value in billions
            Input("smile-time-input", "value"),        # selected PT slices
            Input("ironbeam-y-zoom", "value"),         # vertical zoom factor
        ],
    )
    def update_chart(trade_date, n, threshold_billions, selected_times_pt, y_zoom):
        if not trade_date:
            return go.Figure(layout_title_text="Select a trade date to view chart.")

        # Normalize selected times to a list of strings like ["06:31", "10:15"]
        if selected_times_pt is None:
            selected_times = []
        elif isinstance(selected_times_pt, list):
            selected_times = [str(t) for t in selected_times_pt]
        else:
            selected_times = [str(selected_times_pt)]

        # Convert slider value (billions) to raw units
        if threshold_billions is None:
            current_threshold = GEX_ABS_THRESHOLD_DEFAULT
        else:
            current_threshold = float(threshold_billions) * 1e9

        # Y-zoom factor: <1 = zoom in (more stretched), >1 = zoom out
        zoom_factor = float(y_zoom) if y_zoom is not None else 1.0
        if zoom_factor <= 0:
            zoom_factor = 1.0

        # ----- Parse selected trade date -----
        try:
            selected_date = dt.datetime.strptime(trade_date, "%Y-%m-%d").date()
        except (ValueError, TypeError):
            return go.Figure(layout_title_text="Invalid date format.")

        pt_tz = ZoneInfo("America/Los_Angeles")

        # Time window in PT: D-1 15:00 → D 13:00
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

        # RTH window for background shading: 06:30–13:00 PT on the trade date
        rth_start_pt = dt.datetime.combine(
            selected_date,
            dt.time(6, 30),
            tzinfo=pt_tz,
        )
        rth_end_pt = dt.datetime.combine(
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
        # Convenient HH:MM PT string for matching against Smile/Skew time slices
        df_bars["time_hhmm_pt"] = df_bars["datetime_pt"].dt.strftime("%H:%M")

        # Underlying full-session low/high (for GEX band)
        underlying_low = float(df_bars["low"].min())
        underlying_high = float(df_bars["high"].max())

        # --- Session for y-range: previous day 15:00 → trade date 13:00 PT ---
        dt_pt = df_bars["datetime_pt"]
        session_start_pt = start_pt
        session_end_pt = end_pt

        mask_session = (dt_pt >= session_start_pt) & (dt_pt <= session_end_pt)
        df_session = df_bars[mask_session]
        ref_df = df_session if not df_session.empty else df_bars

        day_low = float(ref_df["low"].min())
        day_high = float(ref_df["high"].max())

        if day_high > 0 and day_high > day_low:
            base_low = day_low * 0.99
            base_high = day_high * 1.01
            center = 0.5 * (base_low + base_high)
            half_span = 0.5 * (base_high - base_low)
            half_span *= zoom_factor
            y_min_price = center - half_span
            y_max_price = center + half_span
        else:
            # Safety fallback if prices are weird
            y_pad = 0.01 * (day_high - day_low) if day_high > day_low else 1.0
            y_min_price = day_low - y_pad
            y_max_price = day_high + y_pad

        y_range = [y_min_price, y_max_price]

        # ----- Fetch GEX levels for the same trade_date (D) -----
        try:
            df_gex = _fetch_gex_grouped_by_level(selected_date)
        except Exception as e:
            print(f"Error fetching GEX data: {e}")
            df_gex = pd.DataFrame(columns=["level", "call_gamma", "put_gamma", "net_gamma"])

        # ----- Filter GEX to a band around price to avoid clutter -----
        if not df_gex.empty:
            band_min = underlying_low - GEX_LEVEL_PADDING
            band_max = underlying_high + GEX_LEVEL_PADDING
            df_gex = df_gex[(df_gex["level"] >= band_min) & (df_gex["level"] <= band_max)]

        # ---------- Build figure ----------
        fig = go.Figure()

        # 1) GEX heatmap
        if not df_gex.empty:
            time_index = pd.date_range(
                start=start_pt,
                end=end_pt,
                freq="1min",
                inclusive="left",
            )
            times = time_index.to_pydatetime()

            levels = df_gex["level"].values
            call_gex = df_gex["call_gamma"].values
            put_gex = df_gex["put_gamma"].values
            net_gex = df_gex["net_gamma"].values

            # Tile net GEX horizontally across all timestamps (what we color by)
            z = np.tile(net_gex.reshape(-1, 1), (1, len(times)))

            # ---- Threshold: keep rows where either side is big ----
            mag = np.abs(call_gex) + np.abs(put_gex)
            if current_threshold > 0:
                mag_z = np.tile(mag.reshape(-1, 1), (1, len(times)))
                mask = mag_z < current_threshold
                z = np.where(mask, np.nan, z)
                color_base = net_gex[mag >= current_threshold]
            else:
                color_base = net_gex

            if color_base.size == 0:
                color_base = net_gex

            # ---- Color span: cater to the cluster, not monsters ----
            if color_base.size:
                if GEX_COLOR_ABS_MAX > 0:
                    color_span = GEX_COLOR_ABS_MAX
                else:
                    color_span = float(
                        np.nanpercentile(np.abs(color_base), GEX_COLOR_PERCENTILE)
                    )
                    if not np.isfinite(color_span) or color_span <= 0:
                        color_span = float(np.nanmax(np.abs(color_base))) or 1.0
            else:
                color_span = 1.0

            fig.add_trace(
                go.Heatmap(
                    x=times,
                    y=levels,
                    z=z,
                    coloraxis="coloraxis",
                    opacity=0.35,
                    zsmooth="best",
                    hovertemplate=(
                        "Time=%{x|%H:%M}<br>"
                        "Level=%{y}<br>"
                        "Net GEX=%{z:.3g}<extra></extra>"
                    ),
                    zauto=False,
                )
            )

            fig.update_layout(
                coloraxis=dict(
                    colorscale=GEX_HEATMAP_COLORSCALE,
                    cmin=-color_span,
                    cmax=color_span,
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

        # 2) ES candlesticks (base layer)
        fig.add_trace(
            go.Candlestick(
                x=df_bars["datetime_pt"],
                open=df_bars["open"],
                high=df_bars["high"],
                low=df_bars["low"],
                close=df_bars["close"],
                name="ES",
                increasing=dict(line=dict(color=CALL_COLOR, width=1.0)),
                decreasing=dict(line=dict(color=PUT_COLOR, width=1.0)),
                showlegend=True,
            )
        )

        # 3) Overlay highlighted candles for selected time slices
        if selected_times:
            mask_selected = df_bars["time_hhmm_pt"].isin(selected_times)
            df_sel = df_bars[mask_selected]
            if not df_sel.empty:
                fig.add_trace(
                    go.Candlestick(
                        x=df_sel["datetime_pt"],
                        open=df_sel["open"],
                        high=df_sel["high"],
                        low=df_sel["low"],
                        close=df_sel["close"],
                        name="Selected slices",
                        increasing=dict(line=dict(color=HIGHLIGHT_COLOR, width=2.0)),
                        decreasing=dict(line=dict(color=HIGHLIGHT_COLOR, width=2.0)),
                        showlegend=False,
                    )
                )

        # 4) Layout + RTH shading
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
            uirevision="ironbeam-gex",  # keeps zoom / rangeslider state
            clickmode="event",          # click events only; no selection/fade
            xaxis=dict(
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
            shapes=[
                dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=rth_start_pt,
                    x1=rth_end_pt,
                    y0=0,
                    y1=1,
                    fillcolor="#020617",
                    opacity=0.85,
                    layer="below",
                    line=dict(width=0),
                )
            ],
        )

        return fig

    # ---- Click on ES bar -> toggle PT time in Smile/Skew time slices ----
    @app.callback(
        Output("smile-time-input", "value", allow_duplicate=True),
        Input("ironbeam-chart", "clickData"),
        State("smile-time-input", "value"),
        prevent_initial_call=True,
    )
    def add_smile_time_from_price_click(click_data, current_values):
        """
        Click a bar to toggle its PT minute in the Smile/Skew time-slice list.
        - If not present: add it
        - If already present: remove it
        """
        if not click_data or not click_data.get("points"):
            raise PreventUpdate

        point = click_data["points"][0]
        x_val = point.get("x")
        if x_val is None:
            raise PreventUpdate

        # x_val is typically an ISO timestamp string; let pandas parse it
        ts = pd.to_datetime(x_val)

        # Treat as PT; the figure x-axis is in PT
        if ts.tzinfo is None:
            ts = ts.tz_localize("America/Los_Angeles")
        ts_pt = ts.tz_convert("America/Los_Angeles")

        hhmm = ts_pt.strftime("%H:%M")  # e.g. "06:31"

        # Normalize existing value into a list
        if current_values is None:
            current_values = []
        elif not isinstance(current_values, list):
            current_values = [current_values]

        # Toggle behavior
        if hhmm in current_values:
            new_values = [t for t in current_values if t != hhmm]
        else:
            new_values = current_values + [hhmm]

        return new_values
