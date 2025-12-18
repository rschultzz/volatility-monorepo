# apps/web/modules/Ironbeam/callbacks.py

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
PUT_COLOR = os.getenv("GEX_PUT_COLOR", "#E5E7EB")  # down candles
CALL_COLOR = os.getenv("GEX_CALL_COLOR", "#60a5fa")  # up candles

# Highlight color for selected slices
HIGHLIGHT_COLOR = os.getenv("IRONBEAM_HIGHLIGHT_COLOR", "#ef4444")  # red

# Ticker for the GEX table
TICKER = os.getenv("GEX_TICKER", "SPX")

# How far above/below price we keep GEX levels (index points)
GEX_LEVEL_PADDING = float(os.getenv("GEX_LEVEL_PADDING", "150"))

# Default minimum |GEX| for plotting (in raw units, not billions). Slider overrides this.
GEX_ABS_THRESHOLD_DEFAULT = float(os.getenv("GEX_ABS_THRESHOLD", "1e10"))

# Color span logic:
# - If GEX_COLOR_ABS_MAX > 0, clamp colors to ±that value
# - Else use percentile of |net_gamma| (default: 95)
GEX_COLOR_ABS_MAX = float(os.getenv("GEX_COLOR_ABS_MAX", "0"))
GEX_COLOR_PERCENTILE = float(os.getenv("GEX_COLOR_PERCENTILE", "95"))

# How many days of data to show around the selected trade date
DAYS_EITHER_SIDE = int(os.getenv("IRONBEAM_DAYS_EITHER_SIDE", "5"))

# Progressive loader: how many dates to add per interval tick
MULTI_LOAD_DAYS_PER_TICK = int(os.getenv("IRONBEAM_MULTI_LOAD_DAYS_PER_TICK", "1"))

# Reduce visual/trace load
GEX_MAX_LEVELS_PER_DAY = int(os.getenv("GEX_MAX_LEVELS_PER_DAY", "80"))
GEX_MIN_LEVEL_SPACING = float(os.getenv("GEX_MIN_LEVEL_SPACING", "5"))  # index points
GEX_LEVEL_BUCKET = float(os.getenv("GEX_LEVEL_BUCKET", "1"))  # index points
# GEX line styling
GEX_LEVEL_LINE_WIDTH = float(os.getenv("GEX_LEVEL_LINE_WIDTH", "2.0"))  # base width for all GEX level lines
GEX_LEVEL_LINE_WIDTH_SCALE = float(os.getenv("GEX_LEVEL_LINE_WIDTH_SCALE", "1.5"))  # extra width at max |GEX|
GEX_LEVEL_LINE_WIDTH_MAX = float(os.getenv("GEX_LEVEL_LINE_WIDTH_MAX", "4.0"))  # clamp
GEX_LEVEL_LINE_OPACITY = float(os.getenv("GEX_LEVEL_LINE_OPACITY", "0.90"))  # base opacity

# Background colors for ETH vs RTH (dark theme)
ETH_BG_COLOR = os.getenv("IRONBEAM_ETH_BG_COLOR", "#1f2937")  # outside RTH
RTH_BG_COLOR = os.getenv("IRONBEAM_RTH_BG_COLOR", "#4b5563")  # RTH shading

# Colorscale tuned for dark background (more contrast across magnitudes)
# Negative (puts) runs indigo -> blue -> cyan; near zero is gray; positive (calls) runs green -> lime -> yellow.
GEX_HEATMAP_COLORSCALE = [
    [0.00, "#312e81"],  # strong negative (deep indigo)
    [0.15, "#1d4ed8"],  # negative (blue)
    [0.30, "#38bdf8"],  # negative (cyan)
    [0.45, "#bae6fd"],  # weak negative (pale cyan)
    [0.50, "#4b5563"],  # near zero (gray)
    [0.55, "#bbf7d0"],  # weak positive (pale green)
    [0.70, "#4ade80"],  # positive (green)
    [0.85, "#a3e635"],  # positive (lime)
    [1.00, "#fef08a"],  # strong positive (pale yellow)
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


engine = create_engine(_get_db_url(), pool_pre_ping=True)


# ---------- Helpers ----------
def _session_window_pt(trade_date: dt.date, pt_tz: ZoneInfo) -> tuple[dt.datetime, dt.datetime]:
    """Trading-day window used throughout the app: D-1 15:00 → D 13:00 PT."""
    start_pt = dt.datetime.combine(trade_date - dt.timedelta(days=1), dt.time(15, 0), tzinfo=pt_tz)
    end_pt = dt.datetime.combine(trade_date, dt.time(13, 0), tzinfo=pt_tz)
    return start_pt, end_pt


def _resample_ohlc(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if df.empty or "datetime" not in df.columns:
        return pd.DataFrame(columns=df.columns)

    df = df.sort_values("datetime").copy().set_index("datetime")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    for vol_col in ["volume", "vol", "size"]:
        if vol_col in df.columns:
            agg[vol_col] = "sum"

    out = df.resample(freq, label="right", closed="right").agg(agg)
    out = out.dropna(subset=["open", "high", "low", "close"]).reset_index()
    return out


def _fetch_bars_pt(start_pt: dt.datetime, end_pt: dt.datetime, interval: str, pt_tz: ZoneInfo) -> pd.DataFrame:
    start_utc = start_pt.astimezone(ZoneInfo("UTC"))
    end_utc = end_pt.astimezone(ZoneInfo("UTC"))

    q = text(
        f"""
        SELECT * FROM {DB_TABLE_NAME}
        WHERE datetime >= :start_date AND datetime < :end_date
        ORDER BY datetime ASC
        """
    )

    with engine.connect() as conn:
        df_1m = pd.read_sql(q, conn, params={"start_date": start_utc, "end_date": end_utc}, parse_dates=["datetime"])

    if df_1m.empty:
        return df_1m

    df = df_1m.sort_values("datetime").copy()
    if interval == "5min":
        df5 = _resample_ohlc(df, "5min")
        if df5 is not None and not df5.empty:
            df = df5

    df["datetime_pt"] = df["datetime"].dt.tz_localize("UTC").dt.tz_convert(pt_tz)
    df["time_hhmm_pt"] = df["datetime_pt"].dt.strftime("%H:%M")
    return df


def _fetch_available_trade_dates(center: dt.date, days_back: int = 45, days_fwd: int = 45) -> list[dt.date]:
    """Pull real trade_date values from orats_oi_gamma so weekends/holidays drop out.

    NOTE: We *avoid* patterns like '(:tkr IS NULL OR ticker = :tkr)' because psycopg/Postgres can
    fail to infer the bind parameter type in that form. Instead we conditionally add the filter.
    """
    start = (center - dt.timedelta(days=days_back)).isoformat()
    end = (center + dt.timedelta(days=days_fwd)).isoformat()

    if TICKER:
        sql = text(
            """
            SELECT DISTINCT trade_date
            FROM orats_oi_gamma
            WHERE trade_date >= :start
              AND trade_date <= :end
              AND ticker = :tkr
            ORDER BY trade_date
            """
        )
        params = {"start": start, "end": end, "tkr": TICKER}
    else:
        sql = text(
            """
            SELECT DISTINCT trade_date
            FROM orats_oi_gamma
            WHERE trade_date >= :start
              AND trade_date <= :end
            ORDER BY trade_date
            """
        )
        params = {"start": start, "end": end}

    with engine.connect() as con:
        df = pd.read_sql(sql, con, params=params)

    if df.empty or "trade_date" not in df.columns:
        return []
    # pandas may give Timestamp; normalize to date
    out: list[dt.date] = []
    for v in df["trade_date"].tolist():
        if isinstance(v, dt.date) and not isinstance(v, dt.datetime):
            out.append(v)
        else:
            out.append(pd.to_datetime(v).date())
    return out


def _sanitize_figure_dict(fig: dict) -> dict:
    """Remove invalid properties that can persist in Dash figure state across callbacks."""
    try:
        layout = fig.get('layout') or {}
        if isinstance(layout, dict):
            for xk, xv in list(layout.items()):
                if not (isinstance(xk, str) and xk.startswith('xaxis')):
                    continue
                if not isinstance(xv, dict):
                    continue
                rs = xv.get('rangeslider')
                if isinstance(rs, dict):
                    for k in list(rs.keys()):
                        # rangeslider supports 'yaxis' but not 'yaxis2', 'yaxis3', etc.
                        if isinstance(k, str) and k.startswith('yaxis') and k != 'yaxis':
                            rs.pop(k, None)
    except Exception:
        pass
    return fig


def _window_trade_dates(center: dt.date, n_each_side: int) -> list[dt.date]:
    dates = _fetch_available_trade_dates(center)
    if not dates:
        # fallback: weekdays only (best-effort)
        dates = [d.date() for d in pd.bdate_range(center - dt.timedelta(days=30), center + dt.timedelta(days=30))]

    if center in dates:
        idx = dates.index(center)
    else:
        # insert center in-order
        dates = sorted(set(dates + [center]))
        idx = dates.index(center)

    left = dates[max(0, idx - n_each_side):idx]
    right = dates[idx + 1: idx + 1 + n_each_side]
    return left + [center] + right


def _roll_forward_to_weekday(d: dt.date) -> dt.date:
    """Roll a date forward to Monday-Friday."""
    while d.weekday() >= 5:
        d += dt.timedelta(days=1)
    return d


def _next_trade_date(d: dt.date, pt_tz: ZoneInfo) -> dt.date:
    """Best-effort next trading date.

    Prefers DB-derived trade_date values (skips holidays), falls back to next weekday.
    """
    try:
        dates = _fetch_available_trade_dates(d, days_back=2, days_fwd=14)
        dates = sorted({x for x in dates if isinstance(x, dt.date)})
        for x in dates:
            if x > d:
                return x
    except Exception:
        pass

    nd = d + dt.timedelta(days=1)
    return _roll_forward_to_weekday(nd)


def _current_session_trade_date(pt_tz: ZoneInfo) -> dt.date:
    """Trade date whose session (D-1 15:00 -> D 13:00 PT) is currently active."""
    now_pt = dt.datetime.now(tz=pt_tz)
    d = now_pt.date()
    if now_pt.time() >= dt.time(15, 0):
        d = _next_trade_date(d, pt_tz)
    return _roll_forward_to_weekday(d)


def _effective_trade_date(selected_date: dt.date, pt_tz: ZoneInfo) -> tuple[dt.date, str | None]:
    """If user selected today but it's after 15:00 PT, show the overnight session.

    After 15:00 PT, the new session belongs to the next trade date.
    """
    try:
        now_pt = dt.datetime.now(tz=pt_tz)
        if selected_date == now_pt.date() and now_pt.time() >= dt.time(15, 0):
            eff = _next_trade_date(selected_date, pt_tz)
            return eff, f"After 15:00 PT, overnight session rolls to {eff.isoformat()}"
    except Exception:
        pass
    return selected_date, None


def _fetch_gex_grouped_by_level(trade_date: dt.date) -> pd.DataFrame:
    """
    Returns columns: level, call_gamma, put_gamma, net_gamma
    Source: orats_oi_gamma (trade_date, ticker, discounted_level, gex_call, gex_put)
    Convention: puts are negative; net_gamma = call + put
    """
    dialect = engine.dialect.name

    bucket = max(float(GEX_LEVEL_BUCKET), 1.0)
    if dialect == "postgresql":
        level_expr = f"(ROUND(discounted_level / {bucket}) * {bucket})::INT"
    else:
        level_expr = f"CAST(ROUND(discounted_level / {bucket}) * {bucket} AS INTEGER)"

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
        return pd.DataFrame(columns=["level", "call_gamma", "put_gamma", "net_gamma"]).astype(
            {"level": "float64", "call_gamma": "float64", "put_gamma": "float64", "net_gamma": "float64"}
        )

    df["call_gamma"] = df["call_gamma_raw"].astype(float)
    df["put_gamma"] = -df["put_gamma_raw"].abs().astype(float)
    df["net_gamma"] = df["call_gamma"] + df["put_gamma"]
    df["level"] = df["level"].astype(float)
    return df[["level", "call_gamma", "put_gamma", "net_gamma"]]


def _color_for_net_gex(net_val: float, cmin: float, cmax: float) -> str:
    if not np.isfinite(net_val):
        return pc.sample_colorscale(GEX_HEATMAP_COLORSCALE, 0.5)[0]
    span = float(cmax - cmin)
    t = 0.5 if span <= 0 else (np.clip(net_val, cmin, cmax) - cmin) / span
    return pc.sample_colorscale(GEX_HEATMAP_COLORSCALE, float(t))[0]


def _compute_color_span(net_g: np.ndarray) -> tuple[float, float]:
    if GEX_COLOR_ABS_MAX > 0:
        span = float(GEX_COLOR_ABS_MAX)
    else:
        base = net_g[np.isfinite(net_g)]
        if base.size:
            abs_base = np.abs(base)
            try:
                p = float(np.nanpercentile(abs_base, float(GEX_COLOR_PERCENTILE)))
            except Exception:
                p = 0.0
            max_abs = float(np.nanmax(abs_base)) if abs_base.size else 0.0
            span = p if p > 0 else (max_abs if max_abs > 0 else 1.0)
        else:
            span = 1.0
    return -span, span


def _select_levels(df_gex: pd.DataFrame, band_min: float, band_max: float, threshold: float) -> pd.DataFrame:
    if df_gex.empty:
        return df_gex
    df = df_gex[(df_gex["level"] >= band_min) & (df_gex["level"] <= band_max)].copy()
    if df.empty:
        return df
    df["mag"] = np.abs(df["call_gamma"].to_numpy(dtype=float)) + np.abs(df["put_gamma"].to_numpy(dtype=float))
    if threshold and threshold > 0:
        df = df[df["mag"] >= threshold].copy()
    if df.empty:
        return df
    df = df.sort_values("mag", ascending=False)

    # Enforce min spacing so we don't draw a solid block of lines.
    keep_rows = []
    kept_levels: list[float] = []
    min_space = max(float(GEX_MIN_LEVEL_SPACING), 0.0)
    for _, r in df.iterrows():
        lvl = float(r["level"])
        if min_space <= 0:
            keep_rows.append(r)
            continue
        if all(abs(lvl - kl) >= min_space for kl in kept_levels):
            keep_rows.append(r)
            kept_levels.append(lvl)
        if len(keep_rows) >= max(int(GEX_MAX_LEVELS_PER_DAY), 1):
            break

    out = pd.DataFrame(keep_rows) if keep_rows else df.head(max(int(GEX_MAX_LEVELS_PER_DAY), 1)).copy()
    return out[["level", "call_gamma", "put_gamma", "net_gamma"]]


# ---------- Dash callback registration ----------
def register_ironbeam_callbacks(app):
    # ---- Base chart (selected date only) ----
    @app.callback(
        Output("ironbeam-chart", "figure"),
        [
            Input("trade-date", "date"),
            Input("gex-threshold-billions", "value"),
            Input("smile-time-input", "value"),
            Input("ironbeam-bar-interval", "value"),
        ],
        [State("ironbeam-chart", "figure")],
    )
    def update_chart(trade_date, threshold_billions, selected_times_pt, bar_interval, prev_fig):
        if not trade_date:
            return go.Figure(layout_title_text="Select a trade date to view chart.")

        # Normalize selected times
        if selected_times_pt is None:
            selected_times: list[str] = []
        elif isinstance(selected_times_pt, list):
            selected_times = [str(t) for t in selected_times_pt]
        else:
            selected_times = [str(selected_times_pt)]

        interval = bar_interval or "1min"

        # Slider (billions) → raw
        current_threshold = GEX_ABS_THRESHOLD_DEFAULT if threshold_billions is None else float(threshold_billions) * 1e9

        # Parse selected trade date
        try:
            selected_date = dt.datetime.strptime(trade_date, "%Y-%m-%d").date()
        except (TypeError, ValueError):
            return go.Figure(layout_title_text="Invalid trade date.")

        pt_tz = ZoneInfo("America/Los_Angeles")

        ui_date = selected_date
        session_date, rollover_note = _effective_trade_date(ui_date, pt_tz)

        # Keep any existing zoom locks when rebuilding
        prev_meta = {}
        if isinstance(prev_fig, dict):
            prev_meta = (prev_fig.get("layout") or {}).get("meta") or {}
        locked_y_range = prev_meta.get("locked_y_range")
        locked_x_range = prev_meta.get("locked_x_range")

        # Target window of *trade dates* (real, excludes weekends/holidays)
        target_dates = _window_trade_dates(session_date, DAYS_EITHER_SIDE)
        target_dates_str = [d.isoformat() for d in target_dates]

        # Selected day session window (what we show first)
        day_start_pt, day_end_pt = _session_window_pt(session_date, pt_tz)

        # Full multi-day window (earliest session start → latest session end)
        try:
            window_start_pt, _ = _session_window_pt(target_dates[0], pt_tz)
            _, window_end_pt = _session_window_pt(target_dates[-1], pt_tz)
        except Exception:
            window_start_pt, window_end_pt = day_start_pt, day_end_pt

        # Pull price bars for selected date only
        try:
            df_bars = _fetch_bars_pt(day_start_pt, day_end_pt, interval, pt_tz)
        except Exception as e:
            print(f"[Ironbeam] Error fetching bar data: {e}")
            return go.Figure(layout_title_text="Database error when loading bars.")

        if df_bars.empty:
            return go.Figure(layout_title_text=f"No ES bar data for {selected_date.isoformat()} session.")

        # Price envelope for selected day
        low = float(df_bars["low"].min())
        high = float(df_bars["high"].max())
        band_min = low - GEX_LEVEL_PADDING
        band_max = high + GEX_LEVEL_PADDING

        fig = go.Figure()

        # --- GEX for selected date ---
        try:
            df_gex = _fetch_gex_grouped_by_level(session_date)
        except Exception as e:
            print(f"[Ironbeam] Error fetching GEX: {e}")
            df_gex = pd.DataFrame(columns=["level", "call_gamma", "put_gamma", "net_gamma"])

        if not df_gex.empty:
            df_gex_day = _select_levels(df_gex, band_min, band_max, current_threshold)
            if not df_gex_day.empty:
                net_g = df_gex_day["net_gamma"].to_numpy(dtype=float)
                cmin, cmax = _compute_color_span(net_g)

                # Invisible trace to host the continuous colorscale & colorbar
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
                        colorbar=dict(title="Net GEX", x=-0.06, xanchor="right", y=0.5, len=0.9),
                    )
                )

                for _, r in df_gex_day.iterrows():
                    lvl = float(r["level"])
                    net_val = float(r["net_gamma"])
                    color = _color_for_net_gex(net_val, cmin, cmax)
                    # Make magnitude pop: slightly thicker + more opaque for larger |GEX|
                    max_abs = float(max(abs(cmin), abs(cmax), 1e-9))
                    norm = float(min(1.0, abs(net_val) / max_abs))
                    line_width = min(GEX_LEVEL_LINE_WIDTH_MAX, GEX_LEVEL_LINE_WIDTH + norm * GEX_LEVEL_LINE_WIDTH_SCALE)
                    line_opacity = float(min(1.0, max(0.15, GEX_LEVEL_LINE_OPACITY * (0.40 + 0.60 * norm))))
                    fig.add_trace(
                        go.Scatter(
                            x=[day_start_pt, day_end_pt],
                            y=[lvl, lvl],
                            mode="lines",
                            line=dict(color=color, width=line_width),
                            opacity=line_opacity,
                            name=f"GEX {session_date.isoformat()}",
                            showlegend=False,
                            hovertemplate=(
                                f"Trade date={session_date.isoformat()}<br>"
                                f"Level={lvl:.0f}<br>Net GEX={net_val:.3g}<extra></extra>"
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

        # Ensure we always have a coloraxis so progressive days can reuse it
        if "coloraxis" not in fig.layout:
            fig.update_layout(
                coloraxis=dict(
                    colorscale=GEX_HEATMAP_COLORSCALE,
                    cmin=-1.0,
                    cmax=1.0,
                    colorbar=dict(title="Net GEX", x=-0.06, xanchor="right", y=0.5, len=0.9),
                )
            )

        # --- Price candles for selected date ---
        fig.add_trace(
            go.Candlestick(
                x=df_bars["datetime_pt"],
                open=df_bars["open"],
                high=df_bars["high"],
                low=df_bars["low"],
                close=df_bars["close"],
                name=f"ES {session_date.isoformat()} ({interval})",
                increasing=dict(line=dict(color=CALL_COLOR, width=1.0), fillcolor=CALL_COLOR),
                decreasing=dict(line=dict(color=PUT_COLOR, width=1.0), fillcolor=PUT_COLOR),
                showlegend=True,
                yaxis="y2",
            )
        )

        # Highlight selected time slices (only within this session)
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
                        increasing=dict(line=dict(color=HIGHLIGHT_COLOR, width=2.0), fillcolor=HIGHLIGHT_COLOR),
                        decreasing=dict(line=dict(color=HIGHLIGHT_COLOR, width=2.0), fillcolor=HIGHLIGHT_COLOR),
                        showlegend=False,
                        yaxis="y2",
                    )
                )

        # --- RTH shading for each target date ---
        shapes = []
        for d in target_dates:
            rth_start_pt = dt.datetime.combine(d, dt.time(6, 30), tzinfo=pt_tz)
            rth_end_pt = dt.datetime.combine(d, dt.time(13, 0), tzinfo=pt_tz)
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

        meta = dict(
            multi_target_dates=target_dates_str,
            multi_loaded_dates=[session_date.isoformat()],
            multi_effective_date=session_date.isoformat(),
            multi_ui_date=ui_date.isoformat(),
            multi_skip_dates=[],
        )
        if locked_y_range is not None:
            meta["locked_y_range"] = locked_y_range
        if locked_x_range is not None:
            meta["locked_x_range"] = locked_x_range

        # Default auto-zoom to the selected session window (can be overridden by user zoom)
        if locked_x_range is None:
            meta["locked_x_range"] = [day_start_pt, day_end_pt]
            locked_x_range = meta["locked_x_range"]
        fig.update_layout(
            title=(
                f"ES (front month) + Net GEX Lines (multi-day; center={session_date.isoformat()})"
            ),
            xaxis_title="Time (Pacific Time)",
            yaxis_title="Discounted Level (GEX)",
            yaxis=dict(showticklabels=False, ticks=""),
            yaxis2=dict(title="ES Price", overlaying="y", side="right", matches="y"),
            xaxis=dict(
                rangeslider=dict(visible=False),
                showspikes=True,
                spikedash="dot",
                spikemode="across",
                spikesnap="cursor",
                hoverformat="%H:%M:%S",
                range=[day_start_pt, day_end_pt],
                domain=[0.0, 1.0],
            ),
            template="plotly_dark",
            hovermode="closest",
            dragmode="pan",
            uirevision=f"ironbeam-multi-{session_date.isoformat()}-{interval}",
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

        # Apply any locked ranges we already know about
        if locked_y_range is not None:
            try:
                fig.update_yaxes(range=locked_y_range, autorange=False)
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

    # ---- Persist zoom locks into figure.meta (so they survive progressive updates) ----
    @app.callback(
        Output("ironbeam-chart", "figure", allow_duplicate=True),
        Input("ironbeam-chart", "relayoutData"),
        State("ironbeam-chart", "figure"),
        prevent_initial_call=True,
    )
    def persist_zoom(relayout, fig):
        if not isinstance(fig, dict) or not isinstance(relayout, dict):
            raise PreventUpdate

        layout = fig.get("layout", {})
        meta = layout.get("meta") or {}

        # y-axis zoom
        y0 = relayout.get("yaxis.range[0]")
        y1 = relayout.get("yaxis.range[1]")
        if y0 is None or y1 is None:
            y0 = relayout.get("yaxis2.range[0]", y0)
            y1 = relayout.get("yaxis2.range[1]", y1)
        if y0 is not None and y1 is not None:
            meta["locked_y_range"] = [y0, y1]

        # y reset
        if relayout.get("yaxis.autorange") or relayout.get("yaxis2.autorange"):
            meta.pop("locked_y_range", None)

        # x-axis zoom
        x0 = relayout.get("xaxis.range[0]")
        x1 = relayout.get("xaxis.range[1]")
        if x0 is not None and x1 is not None:
            meta["locked_x_range"] = [x0, x1]
        if relayout.get("xaxis.autorange"):
            meta.pop("locked_x_range", None)

        layout["meta"] = meta
        fig["layout"] = layout
        return fig

    # ---- Progressive multi-day loader: adds price + GEX for nearby trade dates ----
    @app.callback(
        Output("ironbeam-chart", "figure", allow_duplicate=True),
        Input("ironbeam-interval", "n_intervals"),
        State("ironbeam-chart", "figure"),
        State("trade-date", "date"),
        State("gex-threshold-billions", "value"),
        State("ironbeam-bar-interval", "value"),
        prevent_initial_call=True,
    )
    def progressively_add_days(n_intervals, fig, trade_date, threshold_billions, bar_interval):
        # Dash supplies figure state as a plain dict
        if not isinstance(fig, dict) or not trade_date:
            raise PreventUpdate

        try:
            selected_date = dt.datetime.strptime(trade_date, "%Y-%m-%d").date()
        except (TypeError, ValueError):
            raise PreventUpdate

        interval = bar_interval or "1min"
        current_threshold = GEX_ABS_THRESHOLD_DEFAULT if threshold_billions is None else float(threshold_billions) * 1e9

        pt_tz = ZoneInfo("America/Los_Angeles")

        # Convert dict -> Figure so we can safely call add_trace / update_layout
        fig = _sanitize_figure_dict(fig)
        fig_obj = go.Figure(fig)

        # Pull meta that was set by the base (selected-date) render
        meta = {}
        try:
            meta = (fig.get("layout") or {}).get("meta") or {}
        except Exception:
            meta = {}

        # Use the effective (session) date stored in meta if available
        session_str = meta.get("multi_effective_date") or selected_date.isoformat()
        try:
            session_date = dt.datetime.strptime(session_str, "%Y-%m-%d").date()
        except Exception:
            session_date = selected_date

        target_dates = meta.get("multi_target_dates")
        if not isinstance(target_dates, list) or not target_dates:
            raise PreventUpdate

        loaded = meta.get("multi_loaded_dates", [])
        skipped = meta.get("multi_skip_dates", [])
        if not isinstance(loaded, list):
            loaded = []
        if not isinstance(skipped, list):
            skipped = []

        did_anything = False

        # ---------- Live refresh: update the selected day's candle trace as new bars arrive ----------
        is_live_day = False
        try:
            now_pt = dt.datetime.now(tz=pt_tz)
            is_live_day = (session_date == _current_session_trade_date(pt_tz))
        except Exception:
            is_live_day = False

        if is_live_day:
            day_start_pt, day_end_pt = _session_window_pt(session_date, pt_tz)
            try:
                df_live = _fetch_bars_pt(day_start_pt, day_end_pt, interval, pt_tz)
            except Exception as e:
                print(f"[Ironbeam live] error fetching bars: {e}")
                df_live = pd.DataFrame()

            if df_live is not None and not df_live.empty:
                trace_name = f"ES {session_date.isoformat()} ({interval})"

                idx_trace = None
                old_last = None
                for i, tr in enumerate(fig_obj.data):
                    try:
                        if getattr(tr, "type", None) == "candlestick" and getattr(tr, "name", None) == trace_name:
                            idx_trace = i
                            xs = list(getattr(tr, "x", []) or [])
                            if xs:
                                old_last = str(xs[-1])
                            break
                    except Exception:
                        continue

                new_last = str(df_live["datetime_pt"].iloc[-1])

                if old_last != new_last:
                    x_list = df_live["datetime_pt"].astype(str).tolist()
                    o_list = df_live["open"].astype(float).tolist()
                    h_list = df_live["high"].astype(float).tolist()
                    l_list = df_live["low"].astype(float).tolist()
                    c_list = df_live["close"].astype(float).tolist()

                    if idx_trace is not None:
                        # Update in-place to preserve styling/legend settings
                        tr = fig_obj.data[idx_trace]
                        tr.x = x_list
                        setattr(tr, "open", o_list)
                        tr.high = h_list
                        tr.low = l_list
                        tr.close = c_list
                    else:
                        # Fallback: add the selected-day trace if missing
                        fig_obj.add_trace(
                            go.Candlestick(
                                x=x_list,
                                open=o_list,
                                high=h_list,
                                low=l_list,
                                close=c_list,
                                name=trace_name,
                                increasing=dict(line=dict(color=CALL_COLOR, width=1.0), fillcolor=CALL_COLOR),
                                decreasing=dict(line=dict(color=PUT_COLOR, width=1.0), fillcolor=PUT_COLOR),
                                showlegend=True,
                                yaxis="y2",
                            )
                        )

                    did_anything = True

        # ---------- Progressive add: load nearby dates after initial render ----------
        remaining = [
            d for d in target_dates
            if d != session_date.isoformat() and d not in loaded and d not in skipped
        ]

        batch: list[str] = []
        if remaining:
            # load nearest dates first: -1, +1, -2, +2 ... (best-effort)
            remaining_dates: list[dt.date] = []
            for s in remaining:
                try:
                    remaining_dates.append(dt.datetime.strptime(s, "%Y-%m-%d").date())
                except Exception:
                    continue
            remaining_dates.sort(key=lambda d: (abs((d - session_date).days), (d - session_date).days))
            batch = [d.isoformat() for d in remaining_dates[: max(int(MULTI_LOAD_DAYS_PER_TICK), 1)]]

        # pull existing coloraxis cmin/cmax (fallback safe)
        cmin, cmax = -1.0, 1.0
        try:
            caxis = getattr(fig_obj.layout, "coloraxis", None)
            if caxis is not None and getattr(caxis, "cmin", None) is not None and getattr(caxis, "cmax", None) is not None:
                cmin, cmax = float(caxis.cmin), float(caxis.cmax)
        except Exception:
            pass

        for target_str in batch:
            try:
                target_date = dt.datetime.strptime(target_str, "%Y-%m-%d").date()
            except Exception:
                skipped.append(target_str)
                did_anything = True
                continue

            day_start_pt, day_end_pt = _session_window_pt(target_date, pt_tz)

            try:
                df_bars = _fetch_bars_pt(day_start_pt, day_end_pt, interval, pt_tz)
            except Exception as e:
                print(f"[Ironbeam multi] bars error for {target_str}: {e}")
                df_bars = pd.DataFrame()

            if df_bars is None or df_bars.empty:
                skipped.append(target_str)
                did_anything = True
                continue

            # ---- Price candles for that day ----
            fig_obj.add_trace(
                go.Candlestick(
                    x=df_bars["datetime_pt"].astype(str).tolist(),
                    open=df_bars["open"].astype(float).tolist(),
                    high=df_bars["high"].astype(float).tolist(),
                    low=df_bars["low"].astype(float).tolist(),
                    close=df_bars["close"].astype(float).tolist(),
                    name=f"ES {target_str} ({interval})",
                    increasing=dict(line=dict(color=CALL_COLOR, width=1.0), fillcolor=CALL_COLOR),
                    decreasing=dict(line=dict(color=PUT_COLOR, width=1.0), fillcolor=PUT_COLOR),
                    showlegend=False,
                    yaxis="y2",
                )
            )

            # ---- GEX lines for that day ----
            low = float(df_bars["low"].min())
            high = float(df_bars["high"].max())
            band_min = low - GEX_LEVEL_PADDING
            band_max = high + GEX_LEVEL_PADDING

            try:
                df_gex = _fetch_gex_grouped_by_level(target_date)
            except Exception as e:
                print(f"[Ironbeam multi] gex error for {target_str}: {e}")
                df_gex = pd.DataFrame(columns=["level", "call_gamma", "put_gamma", "net_gamma"])

            if not df_gex.empty:
                df_gex_day = _select_levels(df_gex, band_min, band_max, current_threshold)
                for _, r in df_gex_day.iterrows():
                    lvl = float(r["level"])
                    net_val = float(r["net_gamma"])
                    color = _color_for_net_gex(net_val, float(cmin), float(cmax))
                    fig_obj.add_trace(
                        go.Scattergl(
                            x=[day_start_pt, day_end_pt],
                            y=[lvl, lvl],
                            mode="lines",
                            line=dict(color=color, width=1.0),
                            opacity=1.0,
                            name=f"GEX {target_str}",
                            showlegend=False,
                            hovertemplate=(
                                    f"Trade date={target_str}<br>"
                                    + f"Level={lvl:.0f}<br>Net GEX={net_val:.3g}<extra></extra>"
                            ),
                        )
                    )

            loaded.append(target_str)
            did_anything = True

        # persist meta
        meta["multi_loaded_dates"] = sorted(set(loaded))
        meta["multi_skip_dates"] = sorted(set(skipped))
        fig_obj.update_layout(meta=meta)

        # Re-apply zoom locks if present
        locked_y_range = meta.get("locked_y_range")
        locked_x_range = meta.get("locked_x_range")
        if locked_y_range is not None:
            try:
                fig_obj.update_layout(
                    yaxis=dict(range=locked_y_range, autorange=False),
                    yaxis2=dict(range=locked_y_range, autorange=False),
                )
            except Exception:
                pass
        if locked_x_range is not None:
            try:
                fig_obj.update_layout(xaxis=dict(range=locked_x_range, autorange=False))
            except Exception:
                pass

        if not did_anything:
            raise PreventUpdate

        return fig_obj

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
            return [t for t in current_values if t != hhmm]
        return current_values + [hhmm]