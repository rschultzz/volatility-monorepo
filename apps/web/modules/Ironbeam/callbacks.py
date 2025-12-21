
# apps/web/modules/Ironbeam/callbacks.py

import os
import math
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

# Candle colors
PUT_COLOR = os.getenv("GEX_PUT_COLOR", "#E5E7EB")
CALL_COLOR = os.getenv("GEX_CALL_COLOR", "#60a5fa")

HIGHLIGHT_COLOR = os.getenv("IRONBEAM_HIGHLIGHT_COLOR", "#ef4444")

# Ticker for the GEX table
TICKER = os.getenv("GEX_TICKER", "SPX")

GEX_LEVEL_PADDING = float(os.getenv("GEX_LEVEL_PADDING", "150"))
GEX_ABS_THRESHOLD_DEFAULT = float(os.getenv("GEX_ABS_THRESHOLD", "1e10"))

GEX_COLOR_ABS_MAX = float(os.getenv("GEX_COLOR_ABS_MAX", "0"))
GEX_COLOR_PERCENTILE = float(os.getenv("GEX_COLOR_PERCENTILE", "95"))

DAYS_EITHER_SIDE = int(os.getenv("IRONBEAM_DAYS_EITHER_SIDE", "5"))
MULTI_LOAD_DAYS_PER_TICK = int(os.getenv("IRONBEAM_MULTI_LOAD_DAYS_PER_TICK", "1"))

GEX_MAX_LEVELS_PER_DAY = int(os.getenv("GEX_MAX_LEVELS_PER_DAY", "80"))
GEX_MIN_LEVEL_SPACING = float(os.getenv("GEX_MIN_LEVEL_SPACING", "5"))
GEX_LEVEL_BUCKET = float(os.getenv("GEX_LEVEL_BUCKET", "1"))

GEX_LEVEL_LINE_WIDTH = float(os.getenv("GEX_LEVEL_LINE_WIDTH", "2.0"))
GEX_LEVEL_LINE_WIDTH_SCALE = float(os.getenv("GEX_LEVEL_LINE_WIDTH_SCALE", "1.5"))
GEX_LEVEL_LINE_WIDTH_MAX = float(os.getenv("GEX_LEVEL_LINE_WIDTH_MAX", "4.0"))
GEX_LEVEL_LINE_OPACITY = float(os.getenv("GEX_LEVEL_LINE_OPACITY", "0.90"))

# Hover grid (tooltip everywhere)
HOVERGRID_MAX_POINTS = int(os.getenv("IRONBEAM_HOVERGRID_MAX_POINTS", "300000"))
HOVERGRID_Y_POINTS = int(os.getenv("IRONBEAM_HOVERGRID_Y_POINTS", "70"))
HOVERGRID_OPACITY = float(os.getenv("IRONBEAM_HOVERGRID_OPACITY", "0.001"))
GEX_HOVER_TOLERANCE = float(os.getenv("GEX_HOVER_TOLERANCE", "2.0"))

# Crosshair styling (transparent white dashed)
SPIKE_COLOR = os.getenv("IRONBEAM_SPIKE_COLOR", "rgba(255,255,255,0.55)")
SPIKE_WIDTH = float(os.getenv("IRONBEAM_SPIKE_WIDTH", "1"))

# Background colors
ETH_BG_COLOR = os.getenv("IRONBEAM_ETH_BG_COLOR", "#1f2937")
RTH_BG_COLOR = os.getenv("IRONBEAM_RTH_BG_COLOR", "#4b5563")

# High-contrast diverging colorscale for dark bg
GEX_HEATMAP_COLORSCALE = [
    [0.00, "#312e81"],
    [0.15, "#1d4ed8"],
    [0.30, "#38bdf8"],
    [0.45, "#bae6fd"],
    [0.50, "#4b5563"],
    [0.55, "#bbf7d0"],
    [0.70, "#4ade80"],
    [0.85, "#a3e635"],
    [1.00, "#fef08a"],
]


# ---------- DB engine ----------
def _get_db_url() -> str:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set in the environment")

    # Render often provides "postgres://"
    if db_url.startswith("postgres://"):
        db_url = "postgresql://" + db_url[len("postgres://") :]

    # Prefer psycopg driver
    if db_url.startswith("postgresql://") and "+psycopg" not in db_url:
        db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)

    return db_url


engine = create_engine(_get_db_url(), pool_pre_ping=True)


# ---------- Helpers ----------
def _session_window_pt(trade_date: dt.date, pt_tz: ZoneInfo) -> tuple[dt.datetime, dt.datetime]:
    """
    ES session window in PT:
      - starts prior day 15:00 PT (ETH open)
      - ends trade_date 13:00 PT (RTH close)
    """
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

    # DB datetimes are stored UTC naive -> localize then convert
    # If they come in tz-aware for any reason, handle safely.
    dtcol = df["datetime"]
    if getattr(dtcol.dt, "tz", None) is None:
        df["datetime_pt"] = dtcol.dt.tz_localize("UTC").dt.tz_convert(pt_tz)
    else:
        df["datetime_pt"] = dtcol.dt.tz_convert(pt_tz)

    df["time_hhmm_pt"] = df["datetime_pt"].dt.strftime("%H:%M")
    return df


def _fetch_available_trade_dates(center: dt.date, days_back: int = 45, days_fwd: int = 45) -> list[dt.date]:
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

    out: list[dt.date] = []
    for v in df["trade_date"].tolist():
        if isinstance(v, dt.date) and not isinstance(v, dt.datetime):
            out.append(v)
        else:
            out.append(pd.to_datetime(v).date())
    return out


def _sanitize_figure_dict(fig: dict) -> dict:
    """
    Plotly sometimes sticks invalid keys under rangeslider when figures are round-tripped as dicts.
    This keeps Dash from exploding when we rehydrate the dict into a go.Figure.
    """
    try:
        layout = fig.get("layout") or {}
        if isinstance(layout, dict):
            for xk, xv in list(layout.items()):
                if not (isinstance(xk, str) and xk.startswith("xaxis")):
                    continue
                if not isinstance(xv, dict):
                    continue
                rs = xv.get("rangeslider")
                if isinstance(rs, dict):
                    for k in list(rs.keys()):
                        if isinstance(k, str) and k.startswith("yaxis") and k != "yaxis":
                            rs.pop(k, None)
    except Exception:
        pass
    return fig


def _window_trade_dates(center: dt.date, n_each_side: int) -> list[dt.date]:
    dates = _fetch_available_trade_dates(center)
    if not dates:
        dates = [d.date() for d in pd.bdate_range(center - dt.timedelta(days=30), center + dt.timedelta(days=30))]

    if center in dates:
        idx = dates.index(center)
    else:
        dates = sorted(set(dates + [center]))
        idx = dates.index(center)

    left = dates[max(0, idx - n_each_side) : idx]
    right = dates[idx + 1 : idx + 1 + n_each_side]
    return left + [center] + right


def _roll_forward_to_weekday(d: dt.date) -> dt.date:
    while d.weekday() >= 5:
        d += dt.timedelta(days=1)
    return d


def _next_trade_date(d: dt.date, pt_tz: ZoneInfo) -> dt.date:
    try:
        dates = _fetch_available_trade_dates(d, days_back=2, days_fwd=14)
        dates = sorted({x for x in dates if isinstance(x, dt.date)})
        for x in dates:
            if x > d:
                return x
    except Exception:
        pass
    return _roll_forward_to_weekday(d + dt.timedelta(days=1))


def _current_session_trade_date(pt_tz: ZoneInfo) -> dt.date:
    now_pt = dt.datetime.now(tz=pt_tz)
    d = now_pt.date()
    if now_pt.time() >= dt.time(15, 0):
        d = _next_trade_date(d, pt_tz)
    return _roll_forward_to_weekday(d)


def _effective_trade_date(selected_date: dt.date, pt_tz: ZoneInfo) -> tuple[dt.date, str | None]:
    """
    After ~15:00 PT, the "current" ES session has effectively rolled to the next trade date.
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

    keep_rows = []
    kept_levels: list[float] = []
    min_space = max(float(GEX_MIN_LEVEL_SPACING), 0.0)
    for _, r in df.iterrows():
        lvl = float(r["level"])
        if min_space <= 0 or all(abs(lvl - kl) >= min_space for kl in kept_levels):
            keep_rows.append(r)
            kept_levels.append(lvl)
        if len(keep_rows) >= max(int(GEX_MAX_LEVELS_PER_DAY), 1):
            break

    out = pd.DataFrame(keep_rows) if keep_rows else df.head(max(int(GEX_MAX_LEVELS_PER_DAY), 1)).copy()
    return out[["level", "call_gamma", "put_gamma", "net_gamma"]]


def _remove_traces_by_name_prefix(fig_obj: go.Figure, prefix: str) -> None:
    keep = []
    for tr in fig_obj.data:
        nm = getattr(tr, "name", "") or ""
        if isinstance(nm, str) and nm.startswith(prefix):
            continue
        keep.append(tr)
    fig_obj.data = tuple(keep)


def _infer_price_range_from_fig(fig_obj: go.Figure) -> tuple[float | None, float | None]:
    lows: list[float] = []
    highs: list[float] = []
    for tr in fig_obj.data:
        try:
            if getattr(tr, "type", None) == "candlestick":
                lo = getattr(tr, "low", None)
                hi = getattr(tr, "high", None)
                if lo is not None:
                    lows.extend([float(x) for x in lo if x is not None])
                if hi is not None:
                    highs.extend([float(x) for x in hi if x is not None])
        except Exception:
            continue

    if not lows or not highs:
        return None, None
    return float(np.nanmin(lows)), float(np.nanmax(highs))


def _build_hovergrid_traces(
    pt_tz: ZoneInfo,
    target_dates_str: list[str],
    gex_levels_by_day: dict,
    y_min: float,
    y_max: float,
    x_min: dt.datetime | str | None = None,
    x_max: dt.datetime | str | None = None,
) -> tuple[go.Scattergl, go.Scattergl]:
    """
    Two invisible hover grids on y2 (for "tooltip anywhere"):

      - base: Time + Price
      - gex : same tooltip (NO GEX fields), but kept separate so we can preserve
              the "partition near levels" logic without changing upstream code.

    If x_min/x_max are provided, we ONLY build points inside that viewport
    (big perf win, reduces tooltip-vs-crosshair mismatch).
    """

    def _to_pt_datetime(v: dt.datetime | str | None) -> dt.datetime | None:
        if v is None:
            return None
        try:
            ts = pd.to_datetime(v)
        except Exception:
            return None

        if isinstance(ts, pd.Timestamp):
            if ts.tzinfo is None:
                ts = ts.tz_localize(pt_tz)
            else:
                ts = ts.tz_convert(pt_tz)
            return ts.to_pydatetime()

        if isinstance(ts, dt.datetime):
            if ts.tzinfo is None:
                return ts.replace(tzinfo=pt_tz)
            return ts.astimezone(pt_tz)

        return None

    y_points_req = max(20, int(HOVERGRID_Y_POINTS))
    max_points = max(5000, int(HOVERGRID_MAX_POINTS))

    if y_min >= y_max:
        y_min, y_max = (y_max - 1.0), y_max

    if not target_dates_str:
        empty = go.Scattergl(
            x=[],
            y=[],
            mode="markers",
            marker=dict(size=6, color="rgba(0,0,0,0)"),
            opacity=HOVERGRID_OPACITY,
            name="__hovergrid__empty",
            showlegend=False,
            yaxis="y2",
            customdata=[],
            hovertemplate="Time=%{x|%Y-%m-%d %H:%M:%S}<br>Price=%{customdata:.2f}<extra></extra>",
        )
        return empty, empty

    view_start = _to_pt_datetime(x_min)
    view_end = _to_pt_datetime(x_max)
    if view_start and view_end and view_start > view_end:
        view_start, view_end = view_end, view_start

    overlaps: list[tuple[str, dt.datetime, dt.datetime]] = []
    total_minutes = 0

    for d_str in target_dates_str:
        try:
            d = dt.datetime.strptime(d_str, "%Y-%m-%d").date()
        except Exception:
            continue

        sess_start, sess_end = _session_window_pt(d, pt_tz)

        if view_start and view_end:
            lo = max(sess_start, view_start)
            hi = min(sess_end, view_end)
        else:
            lo, hi = sess_start, sess_end

        if lo >= hi:
            continue

        overlaps.append((d_str, lo, hi))
        total_minutes += int((hi - lo).total_seconds() // 60) + 1

    if not overlaps:
        empty = go.Scattergl(
            x=[],
            y=[],
            mode="markers",
            marker=dict(size=6, color="rgba(0,0,0,0)"),
            opacity=HOVERGRID_OPACITY,
            name="__hovergrid__empty",
            showlegend=False,
            yaxis="y2",
            customdata=[],
            hovertemplate="Time=%{x|%Y-%m-%d %H:%M:%S}<br>Price=%{customdata:.2f}<extra></extra>",
        )
        return empty, empty

    # Prefer 1-minute X spacing; shrink Y points first to stay under max_points.
    y_points_eff = min(y_points_req, max(20, int(max_points // max(1, total_minutes))))
    step_min = 1

    # If still too many points, increase X step.
    if total_minutes * y_points_eff > max_points:
        step_min = int(math.ceil((total_minutes * y_points_eff) / max_points))

    y_vec = np.linspace(float(y_min), float(y_max), int(y_points_eff)).astype(float)
    y_step = float(y_vec[1] - y_vec[0]) if len(y_vec) > 1 else 1.0
    tol = float(max(GEX_HOVER_TOLERANCE, 0.60 * y_step))

    x_base_parts, y_base_parts = [], []
    x_gex_parts, y_gex_parts = [], []

    for (d_str, lo, hi) in overlaps:
        x_day = pd.date_range(lo, hi, freq=f"{step_min}min", inclusive="both").to_pydatetime().tolist()
        if not x_day:
            continue

        levels_list = gex_levels_by_day.get(d_str) if isinstance(gex_levels_by_day, dict) else None

        if not isinstance(levels_list, list) or not levels_list:
            x_rep = np.repeat(x_day, len(y_vec))
            y_rep = np.tile(y_vec, len(x_day))
            x_base_parts.append(x_rep)
            y_base_parts.append(y_rep)
            continue

        try:
            levels = np.array([float(p[0]) for p in levels_list], dtype=float)
        except Exception:
            levels = np.array([], dtype=float)

        if levels.size == 0:
            x_rep = np.repeat(x_day, len(y_vec))
            y_rep = np.tile(y_vec, len(x_day))
            x_base_parts.append(x_rep)
            y_base_parts.append(y_rep)
            continue

        idx = np.abs(levels.reshape(-1, 1) - y_vec.reshape(1, -1)).argmin(axis=0)
        nearest_lvl = levels[idx]
        mask = np.abs(y_vec - nearest_lvl) <= tol

        y_base = y_vec[~mask]
        y_gex = y_vec[mask]

        if y_base.size > 0:
            x_rep_b = np.repeat(x_day, len(y_base))
            y_rep_b = np.tile(y_base, len(x_day))
            x_base_parts.append(x_rep_b)
            y_base_parts.append(y_rep_b)

        if y_gex.size > 0:
            x_rep_g = np.repeat(x_day, len(y_gex))
            y_rep_g = np.tile(y_gex, len(x_day))
            x_gex_parts.append(x_rep_g)
            y_gex_parts.append(y_rep_g)

    x_base = np.concatenate(x_base_parts) if x_base_parts else np.array([], dtype=object)
    y_base = np.concatenate(y_base_parts) if y_base_parts else np.array([], dtype=float)

    x_gex = np.concatenate(x_gex_parts) if x_gex_parts else np.array([], dtype=object)
    y_gex = np.concatenate(y_gex_parts) if y_gex_parts else np.array([], dtype=float)

    # Small hoverlabel offset: shift invisible points a touch (label isn't directly under cursor),
    # but show TRUE price via customdata.
    hover_offset_steps = float(os.getenv("IRONBEAM_HOVER_OFFSET_STEPS", "0.15"))
    y_offset = float(hover_offset_steps * y_step)

    hover_tpl = "Time=%{x|%Y-%m-%d %H:%M:%S}<br>Price=%{customdata:.2f}<extra></extra>"

    base_trace = go.Scattergl(
        x=x_base,
        y=y_base + y_offset,
        mode="markers",
        marker=dict(size=6, color="rgba(0,0,0,0)"),
        opacity=HOVERGRID_OPACITY,
        name="__hovergrid__base",
        showlegend=False,
        yaxis="y2",
        customdata=y_base,  # TRUE price
        hovertemplate=hover_tpl,
    )

    gex_trace = go.Scattergl(
        x=x_gex,
        y=y_gex + y_offset,
        mode="markers",
        marker=dict(size=6, color="rgba(0,0,0,0)"),
        opacity=HOVERGRID_OPACITY,
        name="__hovergrid__gex",
        showlegend=False,
        yaxis="y2",
        customdata=y_gex,  # TRUE price
        hovertemplate=hover_tpl,
    )

    return base_trace, gex_trace


# ---------- Dash callback registration ----------
def register_ironbeam_callbacks(app):
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

        if selected_times_pt is None:
            selected_times: list[str] = []
        elif isinstance(selected_times_pt, list):
            selected_times = [str(t) for t in selected_times_pt]
        else:
            selected_times = [str(selected_times_pt)]

        interval = bar_interval or "1min"
        current_threshold = GEX_ABS_THRESHOLD_DEFAULT if threshold_billions is None else float(threshold_billions) * 1e9

        try:
            selected_date = dt.datetime.strptime(trade_date, "%Y-%m-%d").date()
        except (TypeError, ValueError):
            return go.Figure(layout_title_text="Invalid trade date.")

        pt_tz = ZoneInfo("America/Los_Angeles")
        ui_date = selected_date
        session_date, _ = _effective_trade_date(ui_date, pt_tz)

        # ---- read previous zoom locks ----
        prev_meta = {}
        if isinstance(prev_fig, dict):
            prev_meta = (prev_fig.get("layout") or {}).get("meta") or {}

        locked_y_range = prev_meta.get("locked_y_range")
        locked_x_range = prev_meta.get("locked_x_range")

        # If user changed date or interval, do NOT keep old locks.
        prev_effective = prev_meta.get("multi_effective_date")
        prev_interval = prev_meta.get("bar_interval")
        if prev_effective and str(prev_effective) != session_date.isoformat():
            locked_x_range = None
            locked_y_range = None
        if prev_interval and str(prev_interval) != interval:
            locked_x_range = None
            locked_y_range = None

        # ---- multi-day targets ----
        target_dates = _window_trade_dates(session_date, DAYS_EITHER_SIDE)
        target_dates_str = [d.isoformat() for d in target_dates]

        day_start_pt, day_end_pt = _session_window_pt(session_date, pt_tz)

        # RTH window (PT) for the selected session day (default x-zoom)
        rth_start_pt_center = dt.datetime.combine(session_date, dt.time(6, 30), tzinfo=pt_tz)
        rth_end_pt_center = dt.datetime.combine(session_date, dt.time(13, 0), tzinfo=pt_tz)

        try:
            df_bars = _fetch_bars_pt(day_start_pt, day_end_pt, interval, pt_tz)
        except Exception as e:
            print(f"[Ironbeam] Error fetching bar data: {e}")
            return go.Figure(layout_title_text="Database error when loading bars.")

        if df_bars.empty:
            return go.Figure(layout_title_text=f"No ES bar data for {selected_date.isoformat()} session.")

        # Full-session price range (for GEX band selection)
        full_low = float(df_bars["low"].min())
        full_high = float(df_bars["high"].max())

        # RTH-only range (for tighter default y-zoom)
        df_rth = df_bars[(df_bars["datetime_pt"] >= rth_start_pt_center) & (df_bars["datetime_pt"] <= rth_end_pt_center)]
        if df_rth.empty:
            df_rth = df_bars

        rth_low = float(df_rth["low"].min())
        rth_high = float(df_rth["high"].max())
        rth_rng = max(1e-6, (rth_high - rth_low))
        y_pad = min(25.0, max(3.0, 0.12 * rth_rng))
        default_y_range = [rth_low - y_pad, rth_high + y_pad]

        low = full_low
        high = full_high
        band_min = low - GEX_LEVEL_PADDING
        band_max = high + GEX_LEVEL_PADDING

        fig = go.Figure()

        # --- GEX (selected day only; other days added progressively) ---
        gex_levels_by_day: dict[str, list[list[float]]] = {}

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

                # colorbar host (invisible heatmap)
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

                gex_levels_by_day[session_date.isoformat()] = [
                    [float(r["level"]), float(r["net_gamma"]) / 1e9] for _, r in df_gex_day.iterrows()
                ]

                max_abs = float(max(abs(cmin), abs(cmax), 1e-9))
                for _, r in df_gex_day.iterrows():
                    lvl = float(r["level"])
                    net_val = float(r["net_gamma"])
                    color = _color_for_net_gex(net_val, cmin, cmax)

                    norm = float(min(1.0, abs(net_val) / max_abs))
                    line_width = min(GEX_LEVEL_LINE_WIDTH_MAX, GEX_LEVEL_LINE_WIDTH + norm * GEX_LEVEL_LINE_WIDTH_SCALE)
                    line_opacity = float(min(1.0, max(0.12, GEX_LEVEL_LINE_OPACITY * (0.40 + 0.60 * norm))))

                    fig.add_trace(
                        go.Scattergl(
                            x=[day_start_pt, day_end_pt],
                            y=[lvl, lvl],
                            mode="lines",
                            line=dict(color=color, width=line_width),
                            opacity=line_opacity,
                            name=f"GEX {session_date.isoformat()}",
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )

        if "coloraxis" not in fig.layout:
            fig.update_layout(
                coloraxis=dict(
                    colorscale=GEX_HEATMAP_COLORSCALE,
                    cmin=-1.0,
                    cmax=1.0,
                    colorbar=dict(title="Net GEX", x=-0.06, xanchor="right", y=0.5, len=0.9),
                )
            )

        # --- Price candles (selected day) ---
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
                hoverinfo="skip",
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
                        increasing=dict(line=dict(color=HIGHLIGHT_COLOR, width=2.0), fillcolor=HIGHLIGHT_COLOR),
                        decreasing=dict(line=dict(color=HIGHLIGHT_COLOR, width=2.0), fillcolor=HIGHLIGHT_COLOR),
                        showlegend=False,
                        yaxis="y2",
                        hoverinfo="skip",
                    )
                )

        # RTH shading for each target date
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

        meta: dict = dict(
            bar_interval=interval,
            multi_target_dates=target_dates_str,
            multi_loaded_dates=[session_date.isoformat()],
            multi_effective_date=session_date.isoformat(),
            multi_ui_date=ui_date.isoformat(),
            multi_skip_dates=[],
            gex_levels_by_day=gex_levels_by_day,
        )
        if locked_y_range is not None:
            meta["locked_y_range"] = locked_y_range
        if locked_x_range is not None:
            meta["locked_x_range"] = locked_x_range

        # Default zoom only if user hasn't already zoomed
        if locked_x_range is None:
            meta["locked_x_range"] = [rth_start_pt_center, rth_end_pt_center]
            locked_x_range = meta["locked_x_range"]
        if locked_y_range is None:
            meta["locked_y_range"] = default_y_range
            locked_y_range = meta["locked_y_range"]

        # Hover grid (Time + cursor Price anywhere) — build ONLY for current viewport & only loaded days
        x0, x1 = (locked_x_range if locked_x_range is not None else [rth_start_pt_center, rth_end_pt_center])
        y0, y1 = (locked_y_range if locked_y_range is not None else default_y_range)
        pad_y = max(5.0, 0.04 * (float(y1) - float(y0)))

        hover_days = meta.get("multi_loaded_dates") or [session_date.isoformat()]
        if not isinstance(hover_days, list) or not hover_days:
            hover_days = [session_date.isoformat()]

        hover_base, hover_gex = _build_hovergrid_traces(
            pt_tz=pt_tz,
            target_dates_str=hover_days,
            gex_levels_by_day=gex_levels_by_day,
            y_min=float(y0) - pad_y,
            y_max=float(y1) + pad_y,
            x_min=x0,
            x_max=x1,
        )
        fig.add_trace(hover_base)
        fig.add_trace(hover_gex)

        fig.update_layout(
            title=f"ES (front month) + Net GEX Lines (multi-day; center={session_date.isoformat()})",
            xaxis_title="Time (Pacific Time)",
            yaxis_title="Discounted Level (GEX)",
            yaxis=dict(showticklabels=False, ticks=""),
            yaxis2=dict(title="ES Price", overlaying="y", side="right", matches="y"),
            xaxis=dict(
                rangeslider=dict(visible=False),
                showspikes=True,
                spikecolor=SPIKE_COLOR,
                spikethickness=SPIKE_WIDTH,
                spikedash="dash",
                spikemode="across",
                spikesnap="cursor",
                hoverformat="%H:%M:%S",
                range=(locked_x_range if locked_x_range is not None else [rth_start_pt_center, rth_end_pt_center]),
                domain=[0.0, 1.0],
            ),
            template="plotly_dark",
            hovermode="closest",
            hoverdistance=-1,
            spikedistance=-1,
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
            spikecolor=SPIKE_COLOR,
            spikethickness=SPIKE_WIDTH,
            spikedash="dash",
            spikemode="across",
            spikesnap="cursor",
            hoverformat="%.2f",
        )

        # Apply locked ranges explicitly
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

    # ---- Persist zoom locks (and prevent stale relayout overwrites) ----
    @app.callback(
        Output("ironbeam-chart", "figure", allow_duplicate=True),
        Input("ironbeam-chart", "relayoutData"),
        State("ironbeam-chart", "figure"),
        State("trade-date", "date"),
        State("ironbeam-bar-interval", "value"),
        prevent_initial_call=True,
    )
    def persist_zoom(relayout, fig, trade_date, bar_interval):
        if not isinstance(fig, dict) or not isinstance(relayout, dict) or not trade_date:
            raise PreventUpdate

        # Ignore relayout events that aren't actual zoom/pan/autorange changes.
        interesting = any(
            k in relayout
            for k in (
                "xaxis.range[0]",
                "xaxis.range[1]",
                "xaxis.autorange",
                "yaxis.range[0]",
                "yaxis.range[1]",
                "yaxis.autorange",
                "yaxis2.range[0]",
                "yaxis2.range[1]",
                "yaxis2.autorange",
            )
        )
        if not interesting:
            raise PreventUpdate

        layout = fig.get("layout", {})
        meta = layout.get("meta") or {}

        # Stale guard: if this relayout belongs to an older figure/date, drop it.
        pt_tz = ZoneInfo("America/Los_Angeles")
        try:
            selected_date = dt.datetime.strptime(trade_date, "%Y-%m-%d").date()
        except Exception:
            raise PreventUpdate
        eff_date, _ = _effective_trade_date(selected_date, pt_tz)

        meta_ui = meta.get("multi_ui_date")
        meta_eff = meta.get("multi_effective_date")
        meta_int = meta.get("bar_interval")
        interval = bar_interval or "1min"

        if isinstance(meta_ui, str) and meta_ui and meta_ui != selected_date.isoformat():
            raise PreventUpdate
        if isinstance(meta_eff, str) and meta_eff and meta_eff != eff_date.isoformat():
            raise PreventUpdate
        if isinstance(meta_int, str) and meta_int and meta_int != interval:
            raise PreventUpdate

        # Store Y lock
        y0 = relayout.get("yaxis.range[0]")
        y1 = relayout.get("yaxis.range[1]")
        if y0 is None or y1 is None:
            y0 = relayout.get("yaxis2.range[0]", y0)
            y1 = relayout.get("yaxis2.range[1]", y1)

        if y0 is not None and y1 is not None:
            meta["locked_y_range"] = [y0, y1]
        if relayout.get("yaxis.autorange") or relayout.get("yaxis2.autorange"):
            meta.pop("locked_y_range", None)

        # Store X lock
        x0 = relayout.get("xaxis.range[0]")
        x1 = relayout.get("xaxis.range[1]")
        if x0 is not None and x1 is not None:
            meta["locked_x_range"] = [x0, x1]
        if relayout.get("xaxis.autorange"):
            meta.pop("locked_x_range", None)

        layout["meta"] = meta
        fig["layout"] = layout
        return fig

    # ---- Progressive loader + live refresh (and prevent stale interval overwrites) ----
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
        if not isinstance(fig, dict) or not trade_date:
            raise PreventUpdate

        try:
            selected_date = dt.datetime.strptime(trade_date, "%Y-%m-%d").date()
        except (TypeError, ValueError):
            raise PreventUpdate

        interval = bar_interval or "1min"
        current_threshold = GEX_ABS_THRESHOLD_DEFAULT if threshold_billions is None else float(threshold_billions) * 1e9
        pt_tz = ZoneInfo("America/Los_Angeles")

        # ---- stale guard: drop in-flight ticks from previous date/figure ----
        meta0 = (fig.get("layout") or {}).get("meta") or {}
        meta_ui = meta0.get("multi_ui_date")
        if isinstance(meta_ui, str) and meta_ui and meta_ui != selected_date.isoformat():
            raise PreventUpdate

        eff_date, _ = _effective_trade_date(selected_date, pt_tz)
        meta_eff = meta0.get("multi_effective_date")
        if isinstance(meta_eff, str) and meta_eff and meta_eff != eff_date.isoformat():
            raise PreventUpdate

        base_interval = meta0.get("bar_interval")
        if isinstance(base_interval, str) and base_interval and base_interval != interval:
            raise PreventUpdate

        fig = _sanitize_figure_dict(fig)
        fig_obj = go.Figure(fig)
        meta = (fig.get("layout") or {}).get("meta") or {}

        session_str = meta.get("multi_effective_date") or eff_date.isoformat()
        try:
            session_date = dt.datetime.strptime(session_str, "%Y-%m-%d").date()
        except Exception:
            session_date = eff_date

        target_dates = meta.get("multi_target_dates")
        if not isinstance(target_dates, list) or not target_dates:
            raise PreventUpdate

        loaded = meta.get("multi_loaded_dates", [])
        skipped = meta.get("multi_skip_dates", [])
        if not isinstance(loaded, list):
            loaded = []
        if not isinstance(skipped, list):
            skipped = []

        gex_levels_by_day = meta.get("gex_levels_by_day") or {}
        if not isinstance(gex_levels_by_day, dict):
            gex_levels_by_day = {}

        did_anything = False
        loaded_changed = False

        # ---- live refresh (selected session candle trace only) ----
        is_live_day = False
        try:
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
                if old_last != new_last and idx_trace is not None:
                    tr = fig_obj.data[idx_trace]
                    tr.x = df_live["datetime_pt"].astype(str).tolist()
                    setattr(tr, "open", df_live["open"].astype(float).tolist())
                    tr.high = df_live["high"].astype(float).tolist()
                    tr.low = df_live["low"].astype(float).tolist()
                    tr.close = df_live["close"].astype(float).tolist()
                    did_anything = True

        # ---- progressive add of other days ----
        remaining = [d for d in target_dates if d != session_date.isoformat() and d not in loaded and d not in skipped]
        batch: list[str] = []
        if remaining:
            remaining_dates: list[dt.date] = []
            for s in remaining:
                try:
                    remaining_dates.append(dt.datetime.strptime(s, "%Y-%m-%d").date())
                except Exception:
                    continue
            remaining_dates.sort(key=lambda d: (abs((d - session_date).days), (d - session_date).days))
            batch = [d.isoformat() for d in remaining_dates[: max(int(MULTI_LOAD_DAYS_PER_TICK), 1)]]

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
                    hoverinfo="skip",
                )
            )

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
                if not df_gex_day.empty:
                    gex_levels_by_day[target_str] = [
                        [float(r["level"]), float(r["net_gamma"]) / 1e9] for _, r in df_gex_day.iterrows()
                    ]
                    max_abs = float(max(abs(cmin), abs(cmax), 1e-9))
                    for _, r in df_gex_day.iterrows():
                        lvl = float(r["level"])
                        net_val = float(r["net_gamma"])
                        color = pc.sample_colorscale(
                            GEX_HEATMAP_COLORSCALE,
                            0.5 if (cmax - cmin) <= 0 else (np.clip(net_val, cmin, cmax) - cmin) / (cmax - cmin),
                        )[0]
                        norm = float(min(1.0, abs(net_val) / max_abs))
                        line_width = min(GEX_LEVEL_LINE_WIDTH_MAX, GEX_LEVEL_LINE_WIDTH + norm * GEX_LEVEL_LINE_WIDTH_SCALE)
                        line_opacity = float(min(1.0, max(0.12, GEX_LEVEL_LINE_OPACITY * (0.40 + 0.60 * norm))))

                        fig_obj.add_trace(
                            go.Scattergl(
                                x=[day_start_pt, day_end_pt],
                                y=[lvl, lvl],
                                mode="lines",
                                line=dict(color=color, width=line_width),
                                opacity=line_opacity,
                                name=f"GEX {target_str}",
                                showlegend=False,
                                hoverinfo="skip",
                            )
                        )

            loaded.append(target_str)
            loaded_changed = True
            did_anything = True

        meta["multi_loaded_dates"] = sorted(set(loaded))
        meta["multi_skip_dates"] = sorted(set(skipped))
        meta["gex_levels_by_day"] = gex_levels_by_day
        fig_obj.update_layout(meta=meta)

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

        # Rebuild hover grid only when we add days (avoid heavy work every tick)
        if loaded_changed:
            _remove_traces_by_name_prefix(fig_obj, "__hovergrid__")

            # Use locked ranges if available (viewport-limited hovergrid)
            x_min = None
            x_max = None
            if locked_x_range is not None and isinstance(locked_x_range, (list, tuple)) and len(locked_x_range) == 2:
                x_min, x_max = locked_x_range[0], locked_x_range[1]

            y_min = None
            y_max = None
            if locked_y_range is not None and isinstance(locked_y_range, (list, tuple)) and len(locked_y_range) == 2:
                try:
                    y_min = float(locked_y_range[0])
                    y_max = float(locked_y_range[1])
                except Exception:
                    y_min, y_max = None, None

            if y_min is None or y_max is None:
                y_min, y_max = _infer_price_range_from_fig(fig_obj)
            if y_min is None or y_max is None:
                y_min, y_max = 0.0, 1.0

            pad = max(5.0, 0.04 * (y_max - y_min))

            hover_days = meta.get("multi_loaded_dates") or []
            if not isinstance(hover_days, list) or not hover_days:
                hover_days = [session_date.isoformat()]

            hb, hg = _build_hovergrid_traces(
                pt_tz=pt_tz,
                target_dates_str=hover_days,
                gex_levels_by_day=gex_levels_by_day,
                y_min=float(y_min) - pad,
                y_max=float(y_max) + pad,
                x_min=x_min,
                x_max=x_max,
            )
            fig_obj.add_trace(hb)
            fig_obj.add_trace(hg)

        if not did_anything:
            raise PreventUpdate

        return fig_obj

    # ---- Click on ES bar -> toggle PT time ----
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
