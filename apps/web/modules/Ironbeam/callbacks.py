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
DB_TRADES_TABLE = os.environ.get("IRONBEAM_TRADES_TABLE", "ironbeam_es_trades")

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
#   - strong negatives (puts dominating): bright orange
#   - near zero: bright yellow band
#   - strong positives (calls dominating): green

GEX_HEATMAP_COLORSCALE = [
    [0.0,  "#ea580c"],  # strong negative (deep orange)
    [0.25, "#fb923c"],  # medium negative (lighter orange)
    [0.5,  "#facc15"],  # near zero (bright yellow)
    [0.75, "#22c55e"],  # medium positive (green)
    [1.0,  "#bbf7d0"],  # strong positive (pale green)
]

# Colorscale tuned for dark background:
#   - strong negatives: blue
#   - near zero: dark slate (almost invisible)
#   - strong positives: green

# GEX_HEATMAP_COLORSCALE = [
#     [0.0,  "#1d4ed8"],  # strong negative (deep blue)
#     [0.25, "#60a5fa"],  # medium negative (lighter blue)
#     [0.5,  "#020617"],  # near zero (very dark slate / bg)
#     [0.75, "#22c55e"],  # medium positive (green)
#     [1.0,  "#bbf7d0"],  # strong positive (pale green)
# ]



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


# ---------- Bar resampling helper ----------
def _resample_ohlc(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Resample a 1-minute OHLC dataframe to a coarser frequency (e.g. '5min').

    Expects columns: datetime, open, high, low, close (and optionally volume).
    """
    if df.empty or "datetime" not in df.columns:
        return pd.DataFrame(columns=df.columns)

    df = df.sort_values("datetime").copy()
    df = df.set_index("datetime")

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    # Carry volume if present
    for vol_col in ["volume", "vol", "size"]:
        if vol_col in df.columns:
            agg[vol_col] = "sum"

    df_resampled = df.resample(freq, label="right", closed="right").agg(agg)
    # Drop bars that are completely empty
    df_resampled = df_resampled.dropna(subset=["open", "high", "low", "close"])

    df_resampled = df_resampled.reset_index()
    return df_resampled


# ---------- Dash callback registration ----------
def register_ironbeam_callbacks(app):
    # ---- Main ES + GEX chart ----

    @app.callback(
        Output("ironbeam-chart", "figure"),
        [
            Input("trade-date", "date"),
            Input("ironbeam-interval", "n_intervals"),
            Input("gex-threshold-billions", "value"),   # slider value in billions
            Input("smile-time-input", "value"),         # selected PT slices
            Input("ironbeam-bar-interval", "value"),    # '1min' or '5min'
        ],
    )
    def update_chart(trade_date, n, threshold_billions, selected_times_pt, bar_interval):
        if not trade_date:
            return go.Figure(layout_title_text="Select a trade to view chart.")

        # Normalize selected times to a list of strings like ["06:31", "10:15"]
        if selected_times_pt is None:
            selected_times: list[str] = []
        elif isinstance(selected_times_pt, list):
            selected_times = [str(t) for t in selected_times_pt]
        else:
            selected_times = [str(selected_times_pt)]

        # Normalize bar interval
        interval = bar_interval or "1min"

        # Convert slider value (billions) to raw units
        if threshold_billions is None:
            current_threshold = GEX_ABS_THRESHOLD_DEFAULT
        else:
            current_threshold = float(threshold_billions) * 1e9

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

        # ----- Fetch 1m ES bars plus trades over the same window -----
        bars_query = text(
            f"""
            SELECT * FROM {DB_TABLE_NAME}
            WHERE datetime >= :start_date AND datetime < :end_date
            ORDER BY datetime ASC
        """
        )
        trades_query = text(
            f"""
            SELECT ts_utc, price, size
            FROM {DB_TRADES_TABLE}
            WHERE ts_utc >= :start_date AND ts_utc < :end_date
            ORDER BY ts_utc ASC
        """
        )
        params = {"start_date": start_utc, "end_date": end_utc}

        try:
            with engine.connect() as connection:
                df_bars_1m = pd.read_sql(
                    bars_query, connection, params=params, parse_dates=["datetime"]
                )
                try:
                    df_trades = pd.read_sql(
                        trades_query,
                        connection,
                        params=params,
                        parse_dates=["ts_utc"],
                    )
                except Exception as e:
                    print(f"[Ironbeam] Error fetching trades data: {e}")
                    df_trades = pd.DataFrame()
        except Exception as e:
            print(f"[Ironbeam] Error fetching bar data: {e}")
            return go.Figure(layout_title_text="Database error (bars/trades).")

        print(
            f"[Ironbeam] update_chart: {len(df_bars_1m)} bars, "
            f"{len(df_trades)} trades in window {start_utc} → {end_utc}"
        )

        # ---------- Build canonical 1m series: DB bars + at most ONE live minute ----------
        df_all_1m = df_bars_1m.copy()

        if not df_trades.empty and not df_bars_1m.empty:
            # ts_utc from DB is UTC; keep it tz-aware, then floor to minute and strip tz
            df_trades["ts_utc"] = pd.to_datetime(df_trades["ts_utc"], utc=True)
            df_trades["minute"] = df_trades["ts_utc"].dt.floor("min")
            df_trades["minute"] = df_trades["minute"].dt.tz_localize(None)

            # Last completed bar from Ironbeam (naive UTC)
            last_bar_dt = df_bars_1m["datetime"].max()
            last_bar = df_bars_1m[df_bars_1m["datetime"] == last_bar_dt].iloc[0]
            last_close = float(last_bar["close"])

            latest_trade_minute = df_trades["minute"].max()

            print(
                f"[Ironbeam] last_bar_dt={last_bar_dt}, "
                f"latest_trade_minute={latest_trade_minute}"
            )

            # Only build a live bar if trades exist in a *new* minute beyond DB bars
            if latest_trade_minute is not None and latest_trade_minute > last_bar_dt:
                trades_live = df_trades[df_trades["minute"] == latest_trade_minute].copy()

                if not trades_live.empty:
                    # Aggregate trades in that latest minute
                    prices = trades_live["price"].astype(float)
                    sizes = trades_live["size"].fillna(0).astype(float)

                    t_high = float(prices.max())
                    t_low = float(prices.min())
                    t_close = float(prices.iloc[-1])
                    volume = float(sizes.sum())

                    # Anchor open at last official close to keep continuity
                    live_open = last_close
                    live_high = max(live_open, t_high)
                    live_low = min(live_open, t_low)
                    live_close = t_close

                    live_df = pd.DataFrame(
                        [
                            {
                                "datetime": latest_trade_minute,  # naive UTC
                                "open": live_open,
                                "high": live_high,
                                "low": live_low,
                                "close": live_close,
                                "volume": volume,
                            }
                        ]
                    )

                    print(
                        "[Ironbeam] live anchored bar:",
                        live_df[["datetime", "open", "high", "low", "close"]]
                        .to_dict("records")[0],
                    )

                    older_bars = df_bars_1m[
                        df_bars_1m["datetime"] <= last_bar_dt
                    ].copy()

                    df_all_1m = (
                        pd.concat([older_bars, live_df], ignore_index=True)
                        .sort_values("datetime")
                    )
            else:
                print("[Ironbeam] DB already has latest minute; no live bar appended.")

        # If for some reason we still have nothing, bail
        if df_all_1m.empty:
            return go.Figure(
                layout_title_text=(
                    f"No bar or trade data available for window "
                    f"{start_pt.strftime('%Y-%m-%d %H:%M')} – {end_pt.strftime('%Y-%m-%d %H:%M')} PT."
                )
            )

        # Optional debug: what is the last bar we are actually plotting?
        last_bar_dbg = df_all_1m.sort_values("datetime").tail(1)
        print(
            "[Ironbeam] final last bar:",
            last_bar_dbg[["datetime", "open", "high", "low", "close"]]
            .to_dict("records")[0],
        )

        # ----- Compute envelope and GEX band from combined 1m data -----
        # Convert combined to PT for envelope calc
        df_all_1m_pt = df_all_1m.copy()
        df_all_1m_pt["datetime_pt"] = (
            df_all_1m_pt["datetime"]
            .dt.tz_localize("UTC")
            .dt.tz_convert("America/Los_Angeles")
        )

        # Underlying full-session low/high (for GEX band)
        underlying_low = float(df_all_1m["low"].min())
        underlying_high = float(df_all_1m["high"].max())

        # Session for default y-envelope: previous day 15:00 → trade date 13:00 PT
        dt_pt_all = df_all_1m_pt["datetime_pt"]
        mask_session = (dt_pt_all >= start_pt) & (dt_pt_all <= end_pt)
        df_session = df_all_1m_pt[mask_session]
        ref_df = df_session if not df_session.empty else df_all_1m_pt

        day_low = float(ref_df["low"].min())
        day_high = float(ref_df["high"].max())

        if day_high > 0 and day_high > day_low:
            y_min_price = day_low * 0.99
            y_max_price = day_high * 1.01
        else:
            # Safety fallback if prices are weird
            y_pad = 0.01 * (day_high - day_low) if day_high > day_low else 1.0
            y_min_price = day_low - y_pad
            y_max_price = day_high + y_pad

        # ----- Apply bar-interval resampling for plotting (with fallback) -----
        df_bars = df_all_1m.copy()
        if interval == "5min":
            df_resampled = _resample_ohlc(df_all_1m, "5min")
            if df_resampled is not None and not df_resampled.empty:
                df_bars = df_resampled
            else:
                # Fallback: keep 1m bars if resample failed
                print(
                    f"[Ironbeam] Resample to 5min returned empty for {trade_date}; "
                    "using 1min bars."
                )

        # If for some reason df_bars ended up empty, fall back again
        if df_bars.empty:
            df_bars = df_all_1m.copy()

        # Convert df_bars to PT for display on x-axis
        df_bars["datetime_pt"] = (
            df_bars["datetime"]
            .dt.tz_localize("UTC")
            .dt.tz_convert("America/Los_Angeles")
        )
        # Convenient HH:MM PT string for matching against Smile/Skew time slices
        df_bars["time_hhmm_pt"] = df_bars["datetime_pt"].dt.strftime("%H:%M")

        # ----- Fetch GEX levels for the same trade_date (D) -----
        try:
            df_gex = _fetch_gex_grouped_by_level(selected_date)
        except Exception as e:
            print(f"Error fetching GEX data: {e}")
            df_gex = pd.DataFrame(
                columns=["level", "call_gamma", "put_gamma", "net_gamma"]
            )

        # ----- Filter GEX to a band around price to avoid clutter -----
        if not df_gex.empty:
            band_min = underlying_low - GEX_LEVEL_PADDING
            band_max = underlying_high + GEX_LEVEL_PADDING
            df_gex = df_gex[
                (df_gex["level"] >= band_min) & (df_gex["level"] <= band_max)
            ]

        # ---------- Build figure ----------
        fig = go.Figure()

        # 1) GEX heatmap (stays on 1m time grid)
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

        # 2) ES candlesticks (base layer, 1m or 5min depending on interval)
        fig.add_trace(
            go.Candlestick(
                x=df_bars["datetime_pt"],
                open=df_bars["open"],
                high=df_bars["high"],
                low=df_bars["low"],
                close=df_bars["close"],
                name=f"ES ({interval})",
                increasing=dict(
                    line=dict(color=CALL_COLOR, width=1.0),
                    fillcolor=CALL_COLOR,
                ),
                decreasing=dict(
                    line=dict(color=PUT_COLOR, width=1.0),
                    fillcolor=PUT_COLOR,
                ),
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
                        increasing=dict(
                            line=dict(color=HIGHLIGHT_COLOR, width=2.0),
                            fillcolor=HIGHLIGHT_COLOR,
                        ),
                        decreasing=dict(
                            line=dict(color=HIGHLIGHT_COLOR, width=2.0),
                            fillcolor=HIGHLIGHT_COLOR,
                        ),
                        showlegend=False,
                    )
                )

        # 4) Invisible anchors so *default* autorange sees the 1% envelope as extremes
        fig.add_trace(
            go.Scatter(
                x=[
                    df_bars["datetime_pt"].min(),
                    df_bars["datetime_pt"].max(),
                ],
                y=[
                    y_min_price,
                    y_max_price,
                ],
                mode="markers",
                marker=dict(size=0, opacity=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # 5) Layout + RTH shading (NO yaxis.range / autorange here)
        fig.update_layout(
            title=(
                "ES Front Month "
                f"({interval}) - OHLC with Net GEX Heatmap "
                f"(GEX for {selected_date.isoformat()}, "
                f"window {start_pt.strftime('%Y-%m-%d %H:%M')}–"
                f"{end_pt.strftime('%Y-%m-%d %H:%M')} PT)"
            ),
            xaxis_title="Time (Pacific Time)",
            yaxis_title="Price / Discounted Level",
            template="plotly_dark",
            hovermode="closest",
            dragmode="pan",             # default tool = Pan
            uirevision="ironbeam-gex",
            clickmode="event",          # click events only; no selection/fade
            xaxis=dict(
                rangeslider=dict(visible=False),
                showspikes=True,
                spikedash="dot",
                spikemode="across",
                spikesnap="cursor",
                hoverformat="%H:%M:%S",
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

        # Style-only tweaks for Y (no range/autorange), and explicitly unlock zoom
        fig.update_yaxes(
            showgrid=False,
            fixedrange=False,
            showspikes=True,
            spikedash="dot",
            spikemode="across",
            spikesnap="cursor",
            hoverformat="%.2f",
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
