import os
import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pc
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
# - Else we use a percentile of |net_gamma|
GEX_COLOR_ABS_MAX = float(os.getenv("GEX_COLOR_ABS_MAX", "0"))
GEX_COLOR_PERCENTILE = float(os.getenv("GEX_COLOR_PERCENTILE", "95"))

# Background colors for ETH vs RTH (more contrast, still dark theme)
ETH_BG_COLOR = os.getenv("IRONBEAM_ETH_BG_COLOR", "#1f2937")  # dark gray (outside RTH)
RTH_BG_COLOR = os.getenv("IRONBEAM_RTH_BG_COLOR", "#4b5563")  # medium gray (RTH)

# How many days of ES data to show on either side of the selected trade date
DAYS_EITHER_SIDE = int(os.getenv("IRONBEAM_DAYS_EITHER_SIDE", "5"))

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

GEX_HEATMAP_COLORSCALE = [
    [0.0,  "#1d4ed8"],  # strong negative (deep blue)
    [0.25, "#60a5fa"],  # medium negative (lighter blue)
    [0.50, "#4b5563"],  # near zero: medium gray so it pops on dark bg
    [0.75, "#22c55e"],  # medium positive (green)
    [1.0,  "#bbf7d0"],  # strong positive (pale green)
]


# ---------- DB engine ----------
def _get_db_url() -> str:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set in the environment")

    # normalize for SQLAlchemy
    if db_url.startswith("postgres://"):
        db_url = "postgresql://" + db_url[len("postgres://"):]
    if db_url.startswith("postgresql://") and "+psycopg" not in db_url:
        db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)

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

    We bin levels to 1.0 index-point increments so levels line up on whole points.
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
    for vol_col in ["volume", "vol", "size"]:
        if vol_col in df.columns:
            agg[vol_col] = "sum"

    df_resampled = df.resample(freq, label="right", closed="right").agg(agg)
    df_resampled = df_resampled.dropna(subset=["open", "high", "low", "close"])
    df_resampled = df_resampled.reset_index()
    return df_resampled


# ---------- Color helper ----------
def _color_for_net_gex(net_val: float, cmin: float, cmax: float) -> str:
    """
    Map a single net GEX value into a color on GEX_HEATMAP_COLORSCALE
    using the same cmin/cmax as the coloraxis.
    """
    if not np.isfinite(net_val):
        return pc.sample_colorscale(GEX_HEATMAP_COLORSCALE, 0.5)[0]

    span = cmax - cmin
    if span <= 0:
        t = 0.5
    else:
        t = (np.clip(net_val, cmin, cmax) - cmin) / span
    return pc.sample_colorscale(GEX_HEATMAP_COLORSCALE, float(t))[0]


# ---------- Dash callback registration ----------
def register_ironbeam_callbacks(app):
    # ---- Main ES + GEX chart ----
    @app.callback(
        Output("ironbeam-chart", "figure"),
        [
            Input("trade-date", "date"),
            Input("gex-threshold-billions", "value"),   # slider value in billions
            Input("smile-time-input", "value"),         # selected PT slices
            Input("ironbeam-bar-interval", "value"),    # '1min' or '5min'
            Input("ironbeam-interval", "n_intervals"),  # auto-refresh
        ],
        [
            State("ironbeam-chart", "relayoutData"),
            State("ironbeam-chart", "figure"),
        ],
    )
    def update_chart(
        trade_date,
        threshold_billions,
        selected_times_pt,
        bar_interval,
        n_intervals,
        relayout,
        prev_fig,
    ):
        if not trade_date:
            return go.Figure(layout_title_text="Select a trade date to view chart.")

        # ---- prior meta: keep skip_dates + locked ranges across refreshes ----
        prev_meta = {}
        if isinstance(prev_fig, dict):
            prev_meta = (prev_fig.get("layout") or {}).get("meta") or {}
        prev_skip_dates = prev_meta.get("gex_skip_dates", [])
        locked_y_range = prev_meta.get("locked_y_range")
        locked_x_range = prev_meta.get("locked_x_range")

        # ---- update locks from relayoutData (user zoom/pan/reset) ----
        if isinstance(relayout, dict):
            # y-axis zoom
            y0 = relayout.get("yaxis.range[0]")
            y1 = relayout.get("yaxis.range[1]")
            if y0 is None or y1 is None:
                y0 = relayout.get("yaxis2.range[0]", y0)
                y1 = relayout.get("yaxis2.range[1]", y1)
            if y0 is not None and y1 is not None:
                locked_y_range = [y0, y1]

            # y-axis reset (double-click)
            if relayout.get("yaxis.autorange") or relayout.get("yaxis2.autorange"):
                locked_y_range = None

            # x-axis zoom
            x0 = relayout.get("xaxis.range[0]")
            x1 = relayout.get("xaxis.range[1]")
            if x0 is not None and x1 is not None:
                locked_x_range = [x0, x1]

            # x-axis reset
            if relayout.get("xaxis.autorange"):
                locked_x_range = None

        # ---- normalize time-slice input ----
        if selected_times_pt is None:
            selected_times = []
        elif isinstance(selected_times_pt, list):
            selected_times = [str(t) for t in selected_times_pt]
        else:
            selected_times = [str(selected_times_pt)]

        interval = bar_interval or "1min"

        # Slider is in billions → raw units
        if threshold_billions is None:
            current_threshold = GEX_ABS_THRESHOLD_DEFAULT
        else:
            current_threshold = float(threshold_billions) * 1e9

        # ---- Parse trade date ----
        try:
            selected_date = dt.datetime.strptime(trade_date, "%Y-%m-%d").date()
        except (TypeError, ValueError):
            return go.Figure(layout_title_text="Invalid trade date.")

        pt_tz = ZoneInfo("America/Los_Angeles")

        # ---- Multi-day price window: ±DAYS_EITHER_SIDE around trade date ----
        days_before = DAYS_EITHER_SIDE
        days_after = DAYS_EITHER_SIDE

        full_start_pt = dt.datetime.combine(
            selected_date - dt.timedelta(days=days_before),
            dt.time(0, 0),
            tzinfo=pt_tz,
        )
        full_end_pt = dt.datetime.combine(
            selected_date + dt.timedelta(days=days_after),
            dt.time(23, 59),
            tzinfo=pt_tz,
        )

        # Default visible window: D-1 15:00 → D 13:00 PT
        default_start_pt = dt.datetime.combine(
            selected_date - dt.timedelta(days=1),
            dt.time(15, 0),
            tzinfo=pt_tz,
        )
        default_end_pt = dt.datetime.combine(
            selected_date,
            dt.time(13, 0),
            tzinfo=pt_tz,
        )

        # ---- Query 1m bars in UTC ----
        start_utc = full_start_pt.astimezone(ZoneInfo("UTC"))
        end_utc = full_end_pt.astimezone(ZoneInfo("UTC"))

        bars_query = text(
            f"""
            SELECT * FROM {DB_TABLE_NAME}
            WHERE datetime >= :start_date AND datetime < :end_date
            ORDER BY datetime ASC
        """
        )
        params = {"start_date": start_utc, "end_date": end_utc}

        try:
            with engine.connect() as conn:
                df_bars_1m = pd.read_sql(
                    bars_query, conn, params=params, parse_dates=["datetime"]
                )
        except Exception as e:
            print(f"[Ironbeam] Error fetching bar data: {e}")
            return go.Figure(layout_title_text="Database error when loading bars.")

        if df_bars_1m.empty:
            return go.Figure(
                layout_title_text=(
                    f"No ES bar data for "
                    f"{full_start_pt.strftime('%Y-%m-%d %H:%M')} – "
                    f"{full_end_pt.strftime('%Y-%m-%d %H:%M')} PT."
                )
            )

        df_all_1m = df_bars_1m.sort_values("datetime").copy()

        # ---- Convert bars to PT ----
        df_all_1m_pt = df_all_1m.copy()
        df_all_1m_pt["datetime_pt"] = (
            df_all_1m_pt["datetime"]
            .dt.tz_localize("UTC")
            .dt.tz_convert(pt_tz)
        )

        # Envelope over default session
        dt_pt_all = df_all_1m_pt["datetime_pt"]
        mask_session = (dt_pt_all >= default_start_pt) & (dt_pt_all <= default_end_pt)
        df_session = df_all_1m_pt[mask_session]
        ref_df = df_session if not df_session.empty else df_all_1m_pt

        day_low = float(ref_df["low"].min())
        day_high = float(ref_df["high"].max())

        if day_high > 0 and day_high > day_low:
            y_min_price = day_low * 0.99
            y_max_price = day_high * 1.01
        else:
            pad = 0.01 * (day_high - day_low) if day_high > day_low else 1.0
            y_min_price = day_low - pad
            y_max_price = day_high + pad

        # Underlying low/high over full window (for GEX level band)
        underlying_low = float(df_all_1m["low"].min())
        underlying_high = float(df_all_1m["high"].max())
        band_min = underlying_low - GEX_LEVEL_PADDING
        band_max = underlying_high + GEX_LEVEL_PADDING

        # ---- Resample if needed ----
        df_bars = df_all_1m.copy()
        if interval == "5min":
            df_resampled = _resample_ohlc(df_all_1m, "5min")
            if df_resampled is not None and not df_resampled.empty:
                df_bars = df_resampled

        if df_bars.empty:
            df_bars = df_all_1m.copy()

        # PT time for bars
        df_bars["datetime_pt"] = (
            df_bars["datetime"]
            .dt.tz_localize("UTC")
            .dt.tz_convert(pt_tz)
        )
        df_bars["time_hhmm_pt"] = df_bars["datetime_pt"].dt.strftime("%H:%M")

        # ---- Fetch GEX for selected trade date ----
        try:
            df_gex = _fetch_gex_grouped_by_level(selected_date)
        except Exception as e:
            print(f"[Ironbeam] Error fetching GEX: {e}")
            df_gex = pd.DataFrame(
                columns=["level", "call_gamma", "put_gamma", "net_gamma"]
            )

        fig = go.Figure()

        # ---- Thin GEX lines for selected date ----
        if not df_gex.empty:
            df_gex = df_gex[
                (df_gex["level"] >= band_min)
                & (df_gex["level"] <= band_max)
            ].copy()

            if not df_gex.empty:
                levels = df_gex["level"].to_numpy(dtype=float)
                call_g = df_gex["call_gamma"].to_numpy(dtype=float)
                put_g = df_gex["put_gamma"].to_numpy(dtype=float)
                net_g = df_gex["net_gamma"].to_numpy(dtype=float)

                # Time grid: D-1 15:00 → D 13:00 PT
                day_start_pt = default_start_pt
                day_end_pt = default_end_pt

                time_index = pd.date_range(
                    start=day_start_pt,
                    end=day_end_pt,
                    freq="1min",
                    inclusive="left",
                )
                if not time_index.empty:
                    times = time_index.to_pydatetime()

                    # color span: compress so typical levels (e.g. ~10B) get real color
                    if GEX_COLOR_ABS_MAX > 0:
                        # hard cap from env if you want, e.g. 1e10
                        color_span = GEX_COLOR_ABS_MAX
                    else:
                        base = net_g[np.isfinite(net_g)]
                        if base.size:
                            abs_base = np.abs(base)
                            # use percentile so a few giant levels don't wash everything out
                            try:
                                p = float(np.nanpercentile(abs_base, GEX_COLOR_PERCENTILE))
                            except Exception:
                                p = 0.0
                            max_abs = float(np.nanmax(abs_base))
                            if p > 0:
                                color_span = p
                            elif max_abs > 0:
                                color_span = max_abs
                            else:
                                color_span = 1.0
                        else:
                            color_span = 1.0

                    cmin = -color_span
                    cmax = color_span

                    # invisible heatmap as coloraxis host (for colorbar on left)
                    fig.add_trace(
                        go.Heatmap(
                            x=[day_start_pt, day_end_pt],
                            y=[band_min, band_max],
                            z=[[0, 0], [0, 0]],
                            showscale=True,
                            opacity=0.0,
                            hoverinfo="skip",
                            coloraxis="coloraxis",
                        )
                    )

                    fig.update_layout(
                        coloraxis=dict(
                            colorscale=GEX_HEATMAP_COLORSCALE,
                            cmin=cmin,
                            cmax=cmax,
                            colorbar=dict(
                                title="Net GEX",
                                x=-0.06,
                                xanchor="right",
                                y=0.5,
                                len=0.9,
                            ),
                        )
                    )

                    mag = np.abs(call_g) + np.abs(put_g)

                    for lvl, net_val, mag_val in zip(levels, net_g, mag):
                        if current_threshold > 0 and mag_val < current_threshold:
                            continue

                        color = _color_for_net_gex(net_val, cmin, cmax)

                        fig.add_trace(
                            go.Scatter(
                                x=times,
                                y=[lvl] * len(times),
                                mode="lines",
                                line=dict(
                                    color=color,
                                    width=1.5,
                                ),
                                opacity=1.0,  # <<< fade the gamma bands
                                name=f"GEX {selected_date.isoformat()}",
                                showlegend=False,
                                hovertemplate=(
                                    "Time=%{x|%Y-%m-%d %H:%M}<br>"
                                    f"Level={lvl:.0f}<br>"
                                    f"Net GEX={net_val:.3g}<extra></extra>"
                                ),
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

        # ---- ES candles over full multi-day window (right axis) ----
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
                yaxis="y2",
            )
        )

        # Highlight selected time slices
        if selected_times:
            mask_sel = df_bars["time_hhmm_pt"].isin(selected_times)
            df_sel = df_bars[mask_sel]
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
                        yaxis="y2",
                    )
                )

        # Invisible anchors so default autorange sees the 1% envelope
        fig.add_trace(
            go.Scatter(
                x=[df_bars["datetime_pt"].min(), df_bars["datetime_pt"].max()],
                y=[y_min_price, y_max_price],
                mode="markers",
                marker=dict(size=0, opacity=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # ---- RTH shading for each day in the multi-day window ----
        shapes = []
        for offset in range(-days_before, days_after + 1):
            d = selected_date + dt.timedelta(days=offset)
            rth_start_pt = dt.datetime.combine(
                d,
                dt.time(6, 30),
                tzinfo=pt_tz,
            )
            rth_end_pt = dt.datetime.combine(
                d,
                dt.time(13, 0),
                tzinfo=pt_tz,
            )

            if rth_end_pt <= full_start_pt or rth_start_pt >= full_end_pt:
                continue

            shapes.append(
                dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=rth_start_pt,
                    x1=rth_end_pt,
                    y0=0,
                    y1=1,
                    fillcolor=RTH_BG_COLOR,
                    opacity=0.5,
                    layer="below",
                    line=dict(width=0),
                )
            )

        # ---- meta with persisted state (band + skip_dates + locked ranges) ----
        meta = dict(
            gex_skip_dates=list(prev_skip_dates),
            band_min=float(band_min),
            band_max=float(band_max),
        )
        if locked_y_range is not None:
            meta["locked_y_range"] = locked_y_range
        if locked_x_range is not None:
            meta["locked_x_range"] = locked_x_range

        fig.update_layout(
            title=(
                "ES Front Month "
                f"({interval}) - OHLC with Net GEX Lines "
                f"(centered on {selected_date.isoformat()}, "
                f"default {default_start_pt.strftime('%Y-%m-%d %H:%M')}–"
                f"{default_end_pt.strftime('%Y-%m-%d %H:%M')} PT, "
                f"data window {full_start_pt.strftime('%Y-%m-%d %H:%M')}–"
                f"{full_end_pt.strftime('%Y-%m-%d %H:%M')} PT)"
            ),
            xaxis_title="Time (Pacific Time)",
            yaxis_title="Discounted Level (GEX)",
            yaxis=dict(
                showticklabels=False,
                ticks="",
            ),
            yaxis2=dict(
                title="ES Price",
                overlaying="y",
                side="right",
                matches="y",
            ),
            xaxis=dict(
                rangeslider=dict(visible=False),
                showspikes=True,
                spikedash="dot",
                spikemode="across",
                spikesnap="cursor",
                hoverformat="%H:%M:%S",
                range=[default_start_pt, default_end_pt],
                domain=[0.0, 1.0],
            ),
            template="plotly_dark",
            hovermode="closest",
            dragmode="pan",
            uirevision=f"ironbeam-gex-{selected_date.isoformat()}",
            clickmode="event",
            plot_bgcolor=ETH_BG_COLOR,
            paper_bgcolor=ETH_BG_COLOR,
            height=1200,
            shapes=shapes,
            meta=meta,
            margin=dict(l=90, r=80, t=80, b=80),
        )

        fig.update_yaxes(
            showgrid=False,
            fixedrange=False,
            showspikes=True,
            spikedash="dot",
            spikemode="across",
            spikesnap="cursor",
            hoverformat="%.2f",
        )

        # ---- apply locked ranges (if any) ----
        if locked_y_range is not None:
            try:
                fig.update_yaxes(range=locked_y_range, autorange=False)
                if "yaxis2" in fig.layout:
                    fig.layout["yaxis2"]["range"] = locked_y_range
                    fig.layout["yaxis2"]["autorange"] = False
            except Exception as e:
                print(f"[Ironbeam] warning: could not apply locked y-range: {e}")

        if locked_x_range is not None:
            try:
                fig.update_xaxes(range=locked_x_range, autorange=False)
            except Exception as e:
                print(f"[Ironbeam] warning: could not apply locked x-range: {e}")

        return fig

    # ---- Progressive multi-day GEX loader ----
    @app.callback(
        Output("ironbeam-chart", "figure", allow_duplicate=True),
        Input("ironbeam-interval", "n_intervals"),
        State("ironbeam-chart", "figure"),
        State("trade-date", "date"),
        State("gex-threshold-billions", "value"),
        prevent_initial_call=True,
    )
    def progressively_add_gex(n_intervals, fig, trade_date, threshold_billions):
        """
        After the main chart is drawn, progressively add GEX lines
        for other days in the ±DAYS_EITHER_SIDE window around the
        selected trade date.

        Each interval tick adds at most ONE new day's GEX.
        """
        if fig is None or not trade_date:
            raise PreventUpdate

        print(f"[GEX loader] tick={n_intervals}, trade_date={trade_date}")

        # Parse selected date
        try:
            selected_date = dt.datetime.strptime(trade_date, "%Y-%m-%d").date()
        except (TypeError, ValueError):
            print("[GEX loader] invalid trade_date, abort")
            raise PreventUpdate

        pt_tz = ZoneInfo("America/Los_Angeles")
        days_before = DAYS_EITHER_SIDE
        days_after = DAYS_EITHER_SIDE

        # Threshold from slider
        if threshold_billions is None:
            current_threshold = GEX_ABS_THRESHOLD_DEFAULT
        else:
            current_threshold = float(threshold_billions) * 1e9

        # Recompute full PT window (must match update_chart)
        full_start_pt = dt.datetime.combine(
            selected_date - dt.timedelta(days=days_before),
            dt.time(0, 0),
            tzinfo=pt_tz,
        )
        full_end_pt = dt.datetime.combine(
            selected_date + dt.timedelta(days=days_after),
            dt.time(23, 59),
            tzinfo=pt_tz,
        )

        data = fig.get("data", [])
        if not data:
            print("[GEX loader] no data in figure yet, abort")
            raise PreventUpdate

        layout = fig.get("layout", {})
        meta = layout.get("meta") or {}

        # band & locked ranges from meta
        band_min = meta.get("band_min")
        band_max = meta.get("band_max")
        if band_min is None or band_max is None:
            print("[GEX loader] band_min/band_max missing from meta, abort")
            raise PreventUpdate

        band_min = float(band_min)
        band_max = float(band_max)
        print(f"[GEX loader] price band from meta: {band_min:.1f}–{band_max:.1f}")

        skip_dates_list = meta.get("gex_skip_dates", [])
        if not isinstance(skip_dates_list, list):
            skip_dates_list = []
        skip_dates = set(skip_dates_list)

        # which dates already have GEX
        loaded_dates = set(skip_dates)
        for tr in data:
            if tr.get("type") in ("heatmap", "scatter"):
                name = tr.get("name", "")
                if isinstance(name, str) and name.startswith("GEX "):
                    ds = name[4:]
                    loaded_dates.add(ds)

        print(f"[GEX loader] already handled dates: {sorted(loaded_dates)}")

        # candidate dates around selected_date, excluding selected_date itself
        candidate_dates = []
        for offset in range(-days_before, days_after + 1):
            d = selected_date + dt.timedelta(days=offset)
            if d == selected_date:
                continue
            candidate_dates.append(d)

        print(f"[GEX loader] candidates: {[d.isoformat() for d in candidate_dates]}")

        remaining = [d for d in candidate_dates if d.isoformat() not in loaded_dates]
        print(f"[GEX loader] remaining: {[d.isoformat() for d in remaining]}")

        if not remaining:
            print("[GEX loader] nothing remaining to load, abort")
            raise PreventUpdate

        remaining.sort(key=lambda d: abs((d - selected_date).days))
        target_date = remaining[0]
        target_str = target_date.isoformat()
        print(f"[GEX loader] target_date={target_str}")

        # ---- Fetch GEX for target_date ----
        try:
            df_gex = _fetch_gex_grouped_by_level(target_date)
        except Exception as e:
            print(f"[GEX loader] Error fetching GEX for {target_date}: {e}")
            skip_dates.add(target_str)
            meta["gex_skip_dates"] = sorted(skip_dates)
            layout["meta"] = meta
            fig["layout"] = layout
            return fig

        if df_gex.empty:
            print(f"[GEX loader] no GEX rows for {target_str}, skipping")
            skip_dates.add(target_str)
            meta["gex_skip_dates"] = sorted(skip_dates)
            layout["meta"] = meta
            fig["layout"] = layout
            return fig

        df_gex = df_gex[
            (df_gex["level"] >= band_min)
            & (df_gex["level"] <= band_max)
        ].copy()
        if df_gex.empty:
            print(f"[GEX loader] all GEX levels for {target_str} are outside band, skipping")
            skip_dates.add(target_str)
            meta["gex_skip_dates"] = sorted(skip_dates)
            layout["meta"] = meta
            fig["layout"] = layout
            return fig

        levels = df_gex["level"].to_numpy(dtype=float)
        call_g = df_gex["call_gamma"].to_numpy(dtype=float)
        put_g = df_gex["put_gamma"].to_numpy(dtype=float)
        net_g = df_gex["net_gamma"].to_numpy(dtype=float)

        # Time window: D-1 15:00 → D 13:00 PT, clipped to full window
        day_start_pt = dt.datetime.combine(
            target_date - dt.timedelta(days=1),
            dt.time(15, 0),
            tzinfo=pt_tz,
        )
        day_end_pt = dt.datetime.combine(
            target_date,
            dt.time(13, 0),
            tzinfo=pt_tz,
        )

        if day_end_pt <= full_start_pt or day_start_pt >= full_end_pt:
            print(f"[GEX loader] time window {day_start_pt}–{day_end_pt} outside full window, skipping")
            skip_dates.add(target_str)
            meta["gex_skip_dates"] = sorted(skip_dates)
            layout["meta"] = meta
            fig["layout"] = layout
            return fig

        day_start_clip = max(day_start_pt, full_start_pt)
        day_end_clip = min(day_end_pt, full_end_pt)

        time_index = pd.date_range(
            start=day_start_clip,
            end=day_end_clip,
            freq="1min",
            inclusive="left",
        )
        if time_index.empty:
            print(f"[GEX loader] empty time_index for {target_str}, skipping")
            skip_dates.add(target_str)
            meta["gex_skip_dates"] = sorted(skip_dates)
            layout["meta"] = meta
            fig["layout"] = layout
            return fig

        times = time_index.to_pydatetime()

        # color range from main chart
        coloraxis = layout.get("coloraxis", {})
        cmin = coloraxis.get("cmin")
        cmax = coloraxis.get("cmax")

        if cmin is None or cmax is None:
            if net_g.size:
                max_abs = float(np.nanmax(np.abs(net_g))) or 1.0
            else:
                max_abs = 1.0
            cmin = -max_abs
            cmax = max_abs
            layout.setdefault("coloraxis", {})
            layout["coloraxis"]["cmin"] = cmin
            layout["coloraxis"]["cmax"] = cmax
            fig["layout"] = layout

        print(f"[GEX loader] adding GEX lines for {target_str} with {len(levels)} levels, {len(times)} times")

        mag = np.abs(call_g) + np.abs(put_g)

        for lvl, net_val, mag_val in zip(levels, net_g, mag):
            if current_threshold > 0 and mag_val < current_threshold:
                continue
            color = _color_for_net_gex(net_val, cmin, cmax)

            fig["data"].append(
                dict(
                    type="scatter",
                    mode="lines",
                    x=list(times),
                    y=[float(lvl)] * len(times),
                    line=dict(
                        color=color,
                        width=1.5,
                    ),
                    name=f"GEX {target_str}",
                    hovertemplate=(
                        "Trade date="
                        + target_str
                        + "<br>"
                        "Time=%{x|%Y-%m-%d %H:%M}<br>"
                        f"Level={float(lvl):.0f}<br>"
                        f"Net GEX={float(net_val):.3g}<extra></extra>"
                    ),
                    showlegend=False,
                )
            )

        skip_dates.add(target_str)
        meta["gex_skip_dates"] = sorted(skip_dates)
        layout["meta"] = meta
        fig["layout"] = layout

        return fig

    # ---- Click on ES bar -> toggle PT time in Smile/Skew time slices ----
    @app.callback(
        Output("smile-time-input", "value", allow_duplicate=True),
        Input("ironbeam-chart", "clickData"),
        State("smile-time-input", "value"),
        prevent_initial_call=True,
    )
    def add_smile_time_from_price_click(click_data, current_values):
        if not click_data or not click_data.get("points"):
            raise PreventUpdate

        point = click_data["points"][0]
        x_val = point.get("x")
        if x_val is None:
            raise PreventUpdate

        ts = pd.to_datetime(x_val)
        if ts.tzinfo is None:
            ts = ts.tz_localize("America/Los_Angeles")
        ts_pt = ts.tz_convert("America/Los_Angeles")

        hhmm = ts_pt.strftime("%H:%M")

        if current_values is None:
            current_values = []
        elif not isinstance(current_values, list):
            current_values = [current_values]

        if hhmm in current_values:
            new_values = [t for t in current_values if t != hhmm]
        else:
            new_values = current_values + [hhmm]

        return new_values
