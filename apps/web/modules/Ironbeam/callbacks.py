# apps/web/modules/Ironbeam/callbacks.py

import os
import re
import json
import math
import datetime as dt
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots

from dash import Input, Output, State, html, dcc, ctx, no_update, MATCH, ALL, Patch, ClientsideFunction
from dash.exceptions import PreventUpdate

# Indicator plugin registry (Step 6)
from modules.Ironbeam.indicators.registry import PLUGIN_MAP as IB_PLUGIN_MAP, options as ib_indicator_options
from modules.Ironbeam.indicators.gex_overlay import build_gex_overlay_traces as _build_gex_overlay_traces_plugin
from sqlalchemy import create_engine, text
from flask import jsonify, request


def _indicator_state_token(state: object) -> str:
    """Stable token for guarding against stale-figure overwrites when indicator state changes."""
    if not isinstance(state, dict):
        return ""
    enabled = state.get("enabled") or []
    if enabled is None:
        enabled = []
    elif not isinstance(enabled, list):
        enabled = [enabled]
    enabled = sorted(str(x) for x in enabled if x is not None)

    cfg_all = state.get("cfg") if isinstance(state.get("cfg"), dict) else {}
    gex_cfg = cfg_all.get("gex_overlay") if isinstance(cfg_all.get("gex_overlay"), dict) else {}
    min_abs_b = gex_cfg.get("min_abs_b")
    return f"enabled={','.join(enabled)}|gex_min_abs_b={min_abs_b}"


# ---------- Config ----------
DB_TABLE_NAME = os.environ.get("IRONBEAM_BARS_TABLE", "ironbeam_es_1m_bars")
DB_TRADES_TABLE = os.environ.get("IRONBEAM_TRADES_TABLE", "ironbeam_es_trades")

TRADES_SYMBOL = os.environ.get("IRONBEAM_TRADES_SYMBOL") or os.environ.get("IRONBEAM_SYMBOL", "XCME:ES.H26")

LIVE_TRADES_LOOKBACK_MIN = int(os.getenv("IRONBEAM_LIVE_TRADES_LOOKBACK_MIN", "15"))
LIVE_TRADES_MAX_BARS = int(os.getenv("IRONBEAM_LIVE_TRADES_MAX_BARS", "12"))

# Candle colors
PUT_COLOR = os.getenv("GEX_PUT_COLOR", "#E5E7EB")  # down candles
CALL_COLOR = os.getenv("GEX_CALL_COLOR", "#60a5fa")  # up candles

LIVE_TRADES_TRACE_PREFIX = "Live (trades)"
# Match the rest of the plot
LIVE_UP_COLOR = os.getenv("IRONBEAM_LIVE_UP_COLOR", CALL_COLOR)
LIVE_DOWN_COLOR = os.getenv("IRONBEAM_LIVE_DOWN_COLOR", PUT_COLOR)

HIGHLIGHT_COLOR = os.getenv("IRONBEAM_HIGHLIGHT_COLOR", "#ef4444")

# Flow aggregation table (1s) for aggressor-based indicators
FLOW_TABLE_NAME = os.environ.get("IRONBEAM_FLOW_TABLE", "ironbeam_es_flow_1s")
FLOW_SYMBOL = os.environ.get("IRONBEAM_FLOW_SYMBOL", TRADES_SYMBOL)

FLOW_EMA_LEN = int(os.getenv("IRONBEAM_FLOW_EMA_LEN", "840"))  # smoothing length
FLOW_RESAMPLE = os.getenv("IRONBEAM_FLOW_RESAMPLE", "1s").lower()  # 1s, 5s, 15s, 1m
FLOW_SESSION = os.getenv("IRONBEAM_FLOW_SESSION", "RTH").upper()  # RTH or FULL

# Defaults match price chart colors
FLOW_POS_COLOR = os.getenv("IRONBEAM_FLOW_POS_COLOR", CALL_COLOR)
FLOW_NEG_COLOR = os.getenv("IRONBEAM_FLOW_NEG_COLOR", PUT_COLOR)
FLOW_STRENGTH_COLOR = os.getenv("IRONBEAM_FLOW_STRENGTH_COLOR", "rgba(156,163,175,0.75)")
FLOW_FILL_ALPHA = float(os.getenv("IRONBEAM_FLOW_FILL_ALPHA", "0.18"))

# Ticker for the GEX table
TICKER = os.getenv("GEX_TICKER", "SPX")

GEX_LEVEL_PADDING = float(os.getenv("GEX_LEVEL_PADDING", "150"))
GEX_ABS_THRESHOLD_DEFAULT = float(os.getenv("GEX_ABS_THRESHOLD", "1e10"))

GEX_COLOR_ABS_MAX = float(os.getenv("GEX_COLOR_ABS_MAX", "0"))
GEX_COLOR_PERCENTILE = float(os.getenv("GEX_COLOR_PERCENTILE", "95"))

LEGACY_CLASSIC_DAYS_EITHER_SIDE = int(os.getenv("IRONBEAM_DAYS_EITHER_SIDE", "2"))
REACT_DAYS_EITHER_SIDE_1MIN = int(os.getenv("IRONBEAM_REACT_DAYS_EITHER_SIDE_1MIN", "1"))
REACT_DAYS_EITHER_SIDE_5MIN = int(os.getenv("IRONBEAM_REACT_DAYS_EITHER_SIDE_5MIN", "10"))
MULTI_LOAD_DAYS_PER_TICK = int(os.getenv("IRONBEAM_MULTI_LOAD_DAYS_PER_TICK", "1"))


def _react_days_either_side_for_interval(interval: str) -> int:
    iv = str(interval or "1min").strip().lower()
    return max(0, REACT_DAYS_EITHER_SIDE_5MIN if iv == "5min" else REACT_DAYS_EITHER_SIDE_1MIN)

# --- Smoothness toggles (render-only; safe defaults) ---
USE_HOVERGRID = os.getenv("IRONBEAM_USE_HOVERGRID", "1").strip().lower() in ("1", "true", "yes", "y")
HOVERLINE_MAX_POINTS = int(os.getenv("IRONBEAM_HOVERLINE_MAX_POINTS", "60000"))
MULTIDAY_PREFETCH = os.getenv("IRONBEAM_MULTIDAY_PREFETCH", "0").strip().lower() in ("1", "true", "yes", "y")
ZOOM_COOLDOWN_MS = int(float(os.getenv("IRONBEAM_ZOOM_COOLDOWN_MS", "450")))

GEX_MAX_LEVELS_PER_DAY = int(os.getenv("GEX_MAX_LEVELS_PER_DAY", "80"))
GEX_MIN_LEVEL_SPACING = float(os.getenv("GEX_MIN_LEVEL_SPACING", "5"))
GEX_LEVEL_BUCKET = float(os.getenv("GEX_LEVEL_BUCKET", "1"))

GEX_LEVEL_LINE_WIDTH = float(os.getenv("GEX_LEVEL_LINE_WIDTH", "2.0"))
GEX_LEVEL_LINE_WIDTH_SCALE = float(os.getenv("GEX_LEVEL_LINE_WIDTH_SCALE", "1.5"))
GEX_LEVEL_LINE_WIDTH_MAX = float(os.getenv("GEX_LEVEL_LINE_WIDTH_MAX", "4.0"))
GEX_LEVEL_LINE_OPACITY = float(os.getenv("GEX_LEVEL_LINE_OPACITY", "0.90"))

# Hover grid (tooltip everywhere)
HOVERGRID_MAX_POINTS = int(os.getenv("IRONBEAM_HOVERGRID_MAX_POINTS", "60000"))
HOVERGRID_Y_POINTS = int(os.getenv("IRONBEAM_HOVERGRID_Y_POINTS", "45"))
HOVERGRID_X_SECONDS = int(os.getenv("IRONBEAM_HOVERGRID_X_SECONDS", "15"))
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
        db_url = "postgresql://" + db_url[len("postgres://"):]

    # Prefer psycopg driver (psycopg v3)
    if db_url.startswith("postgresql://") and "+psycopg" not in db_url:
        db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)

    return db_url


engine = create_engine(_get_db_url(), pool_pre_ping=True)


# ---------- Helpers ----------

def _fetch_trades_utc(start_utc: dt.datetime, end_utc: dt.datetime, symbol: str) -> pd.DataFrame:
    q = text(
        f"""
        SELECT ts_utc, symbol, price, size
        FROM {DB_TRADES_TABLE}
        WHERE symbol = :sym
          AND ts_utc >= :start_utc
          AND ts_utc <  :end_utc
        ORDER BY ts_utc ASC
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql(
            q,
            conn,
            params={"sym": symbol, "start_utc": start_utc, "end_utc": end_utc},
            parse_dates=["ts_utc"],
        )
    return df


def _hex_to_rgba(color: str, alpha: float) -> str:
    """
    Convert '#RRGGBB' to 'rgba(r,g,b,a)'. If input is already rgba/other, return as-is.
    """
    c = (color or "").strip()
    if c.startswith("rgba") or c.startswith("rgb"):
        return c
    if not c.startswith("#") or len(c) != 7:
        return c
    r = int(c[1:3], 16)
    g = int(c[3:5], 16)
    b = int(c[5:7], 16)
    a = max(0.0, min(1.0, float(alpha)))
    return f"rgba({r},{g},{b},{a})"


def _fetch_flow_utc(start_utc: dt.datetime, end_utc: dt.datetime, symbol: str) -> pd.DataFrame:
    """
    Fetch flow data. If 'symbol' returns no rows, discovery logic finds the 
    dominant symbol in the time range to support historical contracts.
    """
    def _do_query(s: str | None) -> pd.DataFrame:
        params = {"start_utc": start_utc, "end_utc": end_utc}
        where_sym = ""
        if s:
            where_sym = "AND symbol = :sym"
            params["sym"] = s
            
        q = text(
            f"""
            SELECT ts_utc, symbol, buy_vol, sell_vol, unknown_vol
            FROM {FLOW_TABLE_NAME}
            WHERE ts_utc >= :start_utc
              AND ts_utc <  :end_utc
              {where_sym}
            ORDER BY ts_utc ASC
            """
        )
        with engine.connect() as conn:
            return pd.read_sql(q, conn, params=params, parse_dates=["ts_utc"])

    # 1. Try with provided symbol (usually current front)
    df = _do_query(symbol)
    if not df.empty:
        return df
        
    # 2. If empty, discovery: find whatever symbol is there
    df_all = _do_query(None)
    if df_all.empty:
        return df_all
        
    # 3. Pick the dominant symbol to avoid roll overlap issues
    top_sym = df_all['symbol'].value_counts().idxmax()
    return df_all[df_all['symbol'] == top_sym].copy()


def _floor_to_sec(ts: dt.datetime) -> dt.datetime:
    return ts.replace(microsecond=0)


def _trades_to_ohlc(
        df_trades: pd.DataFrame,
        freq: str,
        pt_tz: ZoneInfo,
        *,
        label_mode: str = "left",
) -> pd.DataFrame:
    """
    Convert trades -> OHLC bars (tz-aware PT in datetime_pt).

    `label_mode`:
      - "left": bars are timestamped at the period START (recommended for matching 1m bar tables)
      - "right": bars are timestamped at the period END (useful when matching resampled 5m bars)

    The returned dataframe has columns: datetime_pt, open, high, low, close, volume
    """
    if df_trades is None or df_trades.empty:
        return pd.DataFrame(columns=["datetime_pt", "open", "high", "low", "close", "volume"])

    label_mode = (label_mode or "left").lower().strip()
    if label_mode not in ("left", "right"):
        label_mode = "left"

    df = df_trades.sort_values("ts_utc").copy()

    ts = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
    df = df.loc[ts.notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=["datetime_pt", "open", "high", "low", "close", "volume"])

    df["ts_utc"] = ts
    df["ts_pt"] = df["ts_utc"].dt.tz_convert(pt_tz)

    # Build bucket labels that match the desired convention
    if label_mode == "left":
        df["bucket_pt"] = df["ts_pt"].dt.floor(freq)
    else:
        # ceil() returns the same timestamp if already on a boundary
        df["bucket_pt"] = df["ts_pt"].dt.ceil(freq)

    g = df.groupby("bucket_pt", sort=True)

    out = pd.DataFrame(
        {
            "open": g["price"].first(),
            "high": g["price"].max(),
            "low": g["price"].min(),
            "close": g["price"].last(),
            "volume": g["size"].sum(),
        }
    ).reset_index()

    out = out.rename(columns={"bucket_pt": "datetime_pt"})
    out = out.dropna(subset=["open", "high", "low", "close"])
    return out


def _apply_live_trades_overlay_patch(patch_obj: Patch, fig_dict: dict, df_base_bars: pd.DataFrame, interval: str, pt_tz: ZoneInfo) -> bool:
    """
    Adds/refreshes a separate candlestick trace built from trades using Patch.
    """
    if df_base_bars is None or df_base_bars.empty:
        return False
    if interval not in ("1min", "5min"):
        return False

    if interval == "1min":
        freq = "1min"
        label_mode = "left"
    else:
        freq = "5min"
        label_mode = "right"

    # Check if overlay exists in fig_dict
    overlay_idx = None
    if "data" in fig_dict and isinstance(fig_dict["data"], list):
        for i, tr in enumerate(fig_dict["data"]):
            if tr.get("name", "").startswith(LIVE_TRADES_TRACE_PREFIX):
                overlay_idx = i
                break

    base_ts = pd.to_datetime(df_base_bars.get("datetime_pt"), errors="coerce")
    base_ts = base_ts.dropna()
    if base_ts.empty:
        if overlay_idx is not None:
            del patch_obj.data[overlay_idx]
            return True
        return False

    try:
        if base_ts.dt.tz is None:
            base_ts = base_ts.dt.tz_localize(pt_tz)
        else:
            base_ts = base_ts.dt.tz_convert(pt_tz)
    except Exception:
        return False

    base_ts = base_ts.sort_values()
    last_base_ts = base_ts.iloc[-1]

    now_pt = dt.datetime.now(tz=pt_tz)
    now_ts = pd.Timestamp(now_pt)

    if label_mode == "left":
        current_bucket = now_ts.floor(freq)
    else:
        current_bucket = now_ts.ceil(freq)

    step = pd.Timedelta(freq)
    start_bucket = last_base_ts + step

    if start_bucket > current_bucket:
        if overlay_idx is not None:
            del patch_obj.data[overlay_idx]
            return True
        return False

    expected = pd.date_range(start=start_bucket, end=current_bucket, freq=freq)

    buffer = max(dt.timedelta(seconds=10), dt.timedelta(minutes=1) if freq == "1min" else dt.timedelta(minutes=5))
    fetch_start_pt = (expected[0].to_pydatetime() - buffer)
    fetch_end_pt = (expected[-1].to_pydatetime() + buffer)

    start_utc = fetch_start_pt.astimezone(ZoneInfo("UTC"))
    end_utc = fetch_end_pt.astimezone(ZoneInfo("UTC"))

    try:
        df_tr = _fetch_trades_utc(start_utc, end_utc, symbol=TRADES_SYMBOL)
    except Exception as e:
        print(f"[live trades] fetch error: {e}")
        return False

    bars_tr = _trades_to_ohlc(df_tr, freq=freq, pt_tz=pt_tz, label_mode=label_mode)
    if bars_tr.empty:
        if overlay_idx is not None:
            del patch_obj.data[overlay_idx]
            return True
        return False

    bars_tr["datetime_pt"] = pd.to_datetime(bars_tr["datetime_pt"], errors="coerce")
    bars_tr = bars_tr.dropna(subset=["datetime_pt"]).copy()

    if bars_tr["datetime_pt"].dt.tz is None:
        bars_tr["datetime_pt"] = bars_tr["datetime_pt"].dt.tz_localize(pt_tz)
    else:
        bars_tr["datetime_pt"] = bars_tr["datetime_pt"].dt.tz_convert(pt_tz)

    bars_tr = bars_tr[bars_tr["datetime_pt"].isin(expected)].copy()

    if bars_tr.empty:
        if overlay_idx is not None:
            del patch_obj.data[overlay_idx]
            return True
        return False

    base_set = set(pd.to_datetime(base_ts).to_list())
    bars_tr = bars_tr[~bars_tr["datetime_pt"].isin(base_set)].copy()

    if bars_tr.empty:
        if overlay_idx is not None:
            del patch_obj.data[overlay_idx]
            return True
        return False

    bars_tr = bars_tr.sort_values("datetime_pt").tail(max(1, LIVE_TRADES_MAX_BARS))

    x_new = bars_tr["datetime_pt"].tolist()
    o_new = bars_tr["open"].astype(float).tolist()
    h_new = bars_tr["high"].astype(float).tolist()
    l_new = bars_tr["low"].astype(float).tolist()
    c_new = bars_tr["close"].astype(float).tolist()

    if overlay_idx is not None:
        # Check if update is needed
        tr = fig_dict["data"][overlay_idx]
        old_x = tr.get("x", [])
        old_c = tr.get("close", [])

        # Simple check: if last point matches, assume no change (optimization)
        if old_x and old_c and len(old_x) == len(x_new):
            if str(old_x[-1]) == str(x_new[-1]) and float(old_c[-1]) == float(c_new[-1]):
                return False

        patch_obj.data[overlay_idx].x = x_new
        patch_obj.data[overlay_idx].open = o_new
        patch_obj.data[overlay_idx].high = h_new
        patch_obj.data[overlay_idx].low = l_new
        patch_obj.data[overlay_idx].close = c_new
        return True
    else:
        # Add new trace
        new_trace = dict(
            type="candlestick",
            x=x_new,
            open=o_new,
            high=h_new,
            low=l_new,
            close=c_new,
            name=f"{LIVE_TRADES_TRACE_PREFIX} {TRADES_SYMBOL}",
            increasing=dict(line=dict(color=LIVE_UP_COLOR, width=2.0), fillcolor="rgba(96,165,250,0.18)"),
            decreasing=dict(line=dict(color=LIVE_DOWN_COLOR, width=2.0), fillcolor="rgba(229,231,235,0.18)"),
            showlegend=False,
            yaxis="y2",
            hovertemplate="<extra></extra>",
        )
        patch_obj.data.append(new_trace)
        return True


def _apply_live_trades_overlay(fig_obj: go.Figure, df_base_bars: pd.DataFrame, interval: str, pt_tz: ZoneInfo) -> bool:
    """
    Adds/refreshes a separate candlestick trace built from trades.
    (Legacy version for full figure rebuilds)
    """
    if df_base_bars is None or df_base_bars.empty:
        return False
    if interval not in ("1min", "5min"):
        return False

    if interval == "1min":
        freq = "1min"
        label_mode = "left"
    else:
        freq = "5min"
        label_mode = "right"

    had_overlay = False
    try:
        for tr in fig_obj.data:
            if getattr(tr, "name", "") and str(tr.name).startswith(LIVE_TRADES_TRACE_PREFIX):
                had_overlay = True
                break
    except Exception:
        had_overlay = False

    base_ts = pd.to_datetime(df_base_bars.get("datetime_pt"), errors="coerce")
    base_ts = base_ts.dropna()
    if base_ts.empty:
        if had_overlay:
            _remove_traces_by_name_prefix(fig_obj, LIVE_TRADES_TRACE_PREFIX)
            return True
        return False

    try:
        if base_ts.dt.tz is None:
            base_ts = base_ts.dt.tz_localize(pt_tz)
        else:
            base_ts = base_ts.dt.tz_convert(pt_tz)
    except Exception:
        return False

    base_ts = base_ts.sort_values()
    last_base_ts = base_ts.iloc[-1]

    now_pt = dt.datetime.now(tz=pt_tz)
    now_ts = pd.Timestamp(now_pt)

    if label_mode == "left":
        current_bucket = now_ts.floor(freq)
    else:
        current_bucket = now_ts.ceil(freq)

    step = pd.Timedelta(freq)
    start_bucket = last_base_ts + step

    if start_bucket > current_bucket:
        if had_overlay:
            _remove_traces_by_name_prefix(fig_obj, LIVE_TRADES_TRACE_PREFIX)
            return True
        return False

    expected = pd.date_range(start=start_bucket, end=current_bucket, freq=freq)

    buffer = max(dt.timedelta(seconds=10), dt.timedelta(minutes=1) if freq == "1min" else dt.timedelta(minutes=5))
    fetch_start_pt = (expected[0].to_pydatetime() - buffer)
    fetch_end_pt = (expected[-1].to_pydatetime() + buffer)

    start_utc = fetch_start_pt.astimezone(ZoneInfo("UTC"))
    end_utc = fetch_end_pt.astimezone(ZoneInfo("UTC"))

    try:
        df_tr = _fetch_trades_utc(start_utc, end_utc, symbol=TRADES_SYMBOL)
    except Exception as e:
        print(f"[live trades] fetch error: {e}")
        return False

    bars_tr = _trades_to_ohlc(df_tr, freq=freq, pt_tz=pt_tz, label_mode=label_mode)
    if bars_tr.empty:
        if had_overlay:
            _remove_traces_by_name_prefix(fig_obj, LIVE_TRADES_TRACE_PREFIX)
            return True
        return False

    bars_tr["datetime_pt"] = pd.to_datetime(bars_tr["datetime_pt"], errors="coerce")
    bars_tr = bars_tr.dropna(subset=["datetime_pt"]).copy()

    if bars_tr["datetime_pt"].dt.tz is None:
        bars_tr["datetime_pt"] = bars_tr["datetime_pt"].dt.tz_localize(pt_tz)
    else:
        bars_tr["datetime_pt"] = bars_tr["datetime_pt"].dt.tz_convert(pt_tz)

    bars_tr = bars_tr[bars_tr["datetime_pt"].isin(expected)].copy()

    if bars_tr.empty:
        if had_overlay:
            _remove_traces_by_name_prefix(fig_obj, LIVE_TRADES_TRACE_PREFIX)
            return True
        return False

    base_set = set(pd.to_datetime(base_ts).to_list())
    bars_tr = bars_tr[~bars_tr["datetime_pt"].isin(base_set)].copy()

    if bars_tr.empty:
        if had_overlay:
            _remove_traces_by_name_prefix(fig_obj, LIVE_TRADES_TRACE_PREFIX)
            return True
        return False

    bars_tr = bars_tr.sort_values("datetime_pt").tail(max(1, LIVE_TRADES_MAX_BARS))

    overlay_idx = None
    try:
        for i, tr in enumerate(fig_obj.data):
            if getattr(tr, "name", "") and str(tr.name).startswith(LIVE_TRADES_TRACE_PREFIX):
                overlay_idx = i
                break
    except Exception:
        overlay_idx = None

    x_new = bars_tr["datetime_pt"].tolist()
    o_new = bars_tr["open"].astype(float).tolist()
    h_new = bars_tr["high"].astype(float).tolist()
    l_new = bars_tr["low"].astype(float).tolist()
    c_new = bars_tr["close"].astype(float).tolist()

    if overlay_idx is not None:
        try:
            tr = fig_obj.data[overlay_idx]
            old_x = list(getattr(tr, "x", []) or [])
            old_c = list(getattr(tr, "close", []) or [])
            if old_x and old_c and str(old_x[-1]) == str(x_new[-1]) and float(old_c[-1]) == float(c_new[-1]) and len(old_x) == len(x_new):
                return False

            tr.x = x_new
            setattr(tr, "open", o_new)
            tr.high = h_new
            tr.low = l_new
            tr.close = c_new
            return True
        except Exception:
            _remove_traces_by_name_prefix(fig_obj, LIVE_TRADES_TRACE_PREFIX)
            overlay_idx = None

    fig_obj.add_trace(
        go.Candlestick(
            x=x_new,
            open=o_new,
            high=h_new,
            low=l_new,
            close=c_new,
            name=f"{LIVE_TRADES_TRACE_PREFIX} {TRADES_SYMBOL}",
            increasing=dict(line=dict(color=LIVE_UP_COLOR, width=2.0), fillcolor="rgba(96,165,250,0.18)"),
            decreasing=dict(line=dict(color=LIVE_DOWN_COLOR, width=2.0), fillcolor="rgba(229,231,235,0.18)"),
            showlegend=False,
            yaxis="y2",
            hovertemplate="<extra></extra>",
        )
    )

    return True


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

    left = dates[max(0, idx - n_each_side): idx]
    right = dates[idx + 1: idx + 1 + n_each_side]
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
    Map the user-selected trade date to the session date used for loading bars/GEX.

    IMPORTANT:
    - By default we do **NOT** roll the selected date forward after 15:00 PT.
      (That roll-forward makes it look like "today" data disappeared after the close.)
    - If you *do* want ETH behavior (15:00 PT roll), set:
        IRONBEAM_ROLL_SESSION_AFTER_15PT=1
    """

    roll = str(os.getenv('IRONBEAM_ROLL_SESSION_AFTER_15PT', '0')).lower() in ('1', 'true', 'yes')
    if not roll:
        return selected_date, None

    # Optional: roll forward after ~15:00 PT to follow the CME/ETH session boundary.
    try:
        now_pt = dt.datetime.now(tz=pt_tz)
        if selected_date == now_pt.date() and now_pt.time() >= dt.time(15, 0):
            eff = _next_trade_date(selected_date, pt_tz)
            return eff, f'After 15:00 PT, overnight session rolls to {eff.isoformat()}'
    except Exception:
        pass
    return selected_date, None

def _bars_df_to_payload_rows(
        df_bars: pd.DataFrame,
        pt_tz: ZoneInfo,
        *,
        session_date: dt.date,
        is_center: bool,
) -> list[dict]:
    if df_bars is None or df_bars.empty:
        return []

    if "datetime" in df_bars.columns:
        ts_utc = pd.to_datetime(df_bars["datetime"], utc=True, errors="coerce")
    else:
        ts_pt = pd.to_datetime(df_bars["datetime_pt"], errors="coerce")
        try:
            if getattr(ts_pt.dt, "tz", None) is None:
                ts_utc = ts_pt.dt.tz_localize(pt_tz).dt.tz_convert("UTC")
            else:
                ts_utc = ts_pt.dt.tz_convert("UTC")
        except Exception:
            ts_utc = pd.to_datetime([], utc=True)

    rows: list[dict] = []
    for i, row in df_bars.reset_index(drop=True).iterrows():
        try:
            ts = ts_utc.iloc[i]
        except Exception:
            continue
        if pd.isna(ts):
            continue

        try:
            rows.append({
                "time": int(pd.Timestamp(ts).timestamp()),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "session_date": session_date.isoformat(),
                "is_center": bool(is_center),
            })
        except Exception:
            continue

    return rows
def _build_gex_segments_for_session(
        session_date: dt.date,
        df_bars: pd.DataFrame,
        pt_tz: ZoneInfo,
        *,
        gex_min_abs_b: float | None = None,
) -> list[dict]:
    if df_bars is None or df_bars.empty:
        return []

    try:
        df_gex = _fetch_gex_grouped_by_level(session_date)
    except Exception:
        return []

    if df_gex is None or df_gex.empty:
        return []

    try:
        band_min = float(df_bars["low"].min()) - float(GEX_LEVEL_PADDING)
        band_max = float(df_bars["high"].max()) + float(GEX_LEVEL_PADDING)
    except Exception:
        return []

    threshold_abs = (
        float(gex_min_abs_b) * 1e9
        if gex_min_abs_b is not None
        else float(GEX_ABS_THRESHOLD_DEFAULT)
    )

    df_day = _select_levels(df_gex, band_min, band_max, threshold_abs)
    if df_day is None or df_day.empty:
        return []

    net_g = df_day["net_gamma"].to_numpy(dtype=float)
    cmin, cmax = _compute_color_span(net_g)
    denom = float(max(abs(cmin), abs(cmax), 1.0))

    start_pt, end_pt = _session_window_pt(session_date, pt_tz)
    start_utc = int(start_pt.astimezone(ZoneInfo("UTC")).timestamp())
    end_utc = int(end_pt.astimezone(ZoneInfo("UTC")).timestamp())

    out: list[dict] = []
    for _, r in df_day.iterrows():
        lvl = float(r["level"])
        net_val = float(r["net_gamma"])
        color = _color_for_net_gex(net_val, cmin, cmax)
        norm = float(min(1.0, abs(net_val) / denom))
        line_width = float(
            min(
                GEX_LEVEL_LINE_WIDTH_MAX,
                GEX_LEVEL_LINE_WIDTH + norm * GEX_LEVEL_LINE_WIDTH_SCALE,
            )
        )
        line_opacity = float(
            min(1.0, max(0.12, GEX_LEVEL_LINE_OPACITY * (0.40 + 0.60 * norm)))
        )

        out.append({
            "session_date": session_date.isoformat(),
            "start_time": start_utc,
            "end_time": end_utc,
            "level": lvl,
            "net_gamma": net_val,
            "call_gamma": float(r["call_gamma"]),
            "put_gamma": float(r["put_gamma"]),
            "color": _hex_to_rgba(color, line_opacity),
            "line_width": line_width,
            "opacity": line_opacity,
        })

    return out

def _build_react_preview_bars_payload(
        trade_date: str | None,
        interval: str | None,
        *,
        gex_enabled: bool = True,
        gex_min_abs_b: float | None = None,
        phase: str = "center",
        days_either_side: int | None = None,
) -> tuple[dict, int]:
    interval = str(interval or "1min").strip()
    if interval not in ("1min", "5min"):
        interval = "1min"

    phase = str(phase or "center").strip().lower()
    if phase not in {"center", "multi"}:
        phase = "center"

    if not trade_date:
        return {
            "error": "trade_date is required",
            "bars": [],
            "gex_levels": [],
            "gex_segments": [],
            "interval": interval,
            "phase": phase,
        }, 400

    try:
        selected_date = dt.datetime.strptime(str(trade_date), "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return {
            "error": "Invalid trade date",
            "bars": [],
            "gex_levels": [],
            "gex_segments": [],
            "interval": interval,
            "phase": phase,
        }, 400

    pt_tz = ZoneInfo("America/Los_Angeles")
    session_date, session_note = _effective_trade_date(selected_date, pt_tz)

    if phase == "multi":
        n_each_side = _react_days_either_side_for_interval(interval) if days_either_side is None else max(0, int(days_either_side))
        target_dates = [d for d in _window_trade_dates(session_date, n_each_side) if d != session_date]
    else:
        n_each_side = 0
        target_dates = [session_date]

    loaded_frames: list[tuple[dt.date, pd.DataFrame]] = []

    for load_date in target_dates:
        day_start_pt, day_end_pt = _session_window_pt(load_date, pt_tz)
        try:
            df_day = _fetch_bars_pt(day_start_pt, day_end_pt, interval, pt_tz)
        except Exception as e:
            return {
                "error": f"Database error when loading bars: {e}",
                "bars": [],
                "gex_levels": [],
                "gex_segments": [],
                "trade_date": str(trade_date),
                "effective_trade_date": session_date.isoformat(),
                "interval": interval,
                "phase": phase,
                "days_either_side": n_each_side,
            }, 500

        if df_day is not None and not df_day.empty:
            loaded_frames.append((load_date, df_day))

    bars: list[dict] = []
    for load_date, df_day in loaded_frames:
        bars.extend(
            _bars_df_to_payload_rows(
                df_day,
                pt_tz,
                session_date=load_date,
                is_center=(load_date == session_date),
            )
        )
    bars.sort(key=lambda x: x["time"])

    gex_segments: list[dict] = []
    if gex_enabled:
        for load_date, df_day in loaded_frames:
            try:
                gex_segments.extend(
                    _build_gex_segments_for_session(
                        load_date,
                        df_day,
                        pt_tz,
                        gex_min_abs_b=gex_min_abs_b,
                    )
                )
            except Exception:
                continue

    # Keep center-day gex_levels for backward compatibility if needed
    gex_levels: list[dict] = []
    if phase == "center" and gex_enabled and loaded_frames:
        df_bars = loaded_frames[0][1]
        try:
            df_gex = _fetch_gex_grouped_by_level(session_date)
            if df_gex is not None and not df_gex.empty:
                band_min = float(df_bars["low"].min()) - float(GEX_LEVEL_PADDING)
                band_max = float(df_bars["high"].max()) + float(GEX_LEVEL_PADDING)
                threshold_abs = (
                    float(gex_min_abs_b) * 1e9
                    if gex_min_abs_b is not None
                    else float(GEX_ABS_THRESHOLD_DEFAULT)
                )
                df_day = _select_levels(df_gex, band_min, band_max, threshold_abs)
                if df_day is not None and not df_day.empty:
                    net_g = df_day["net_gamma"].to_numpy(dtype=float)
                    cmin, cmax = _compute_color_span(net_g)
                    denom = float(max(abs(cmin), abs(cmax), 1.0))
                    for _, r in df_day.iterrows():
                        lvl = float(r["level"])
                        net_val = float(r["net_gamma"])
                        color = _color_for_net_gex(net_val, cmin, cmax)
                        norm = float(min(1.0, abs(net_val) / denom))
                        line_width = float(
                            min(
                                GEX_LEVEL_LINE_WIDTH_MAX,
                                GEX_LEVEL_LINE_WIDTH + norm * GEX_LEVEL_LINE_WIDTH_SCALE,
                            )
                        )
                        line_opacity = float(
                            min(1.0, max(0.12, GEX_LEVEL_LINE_OPACITY * (0.40 + 0.60 * norm)))
                        )
                        gex_levels.append({
                            "level": lvl,
                            "net_gamma": net_val,
                            "call_gamma": float(r["call_gamma"]),
                            "put_gamma": float(r["put_gamma"]),
                            "color": _hex_to_rgba(color, line_opacity),
                            "line_width": line_width,
                            "opacity": line_opacity,
                        })
        except Exception:
            gex_levels = []

    return {
        "bars": bars,
        "gex_levels": gex_levels,
        "gex_segments": gex_segments,
        "trade_date": str(trade_date),
        "effective_trade_date": session_date.isoformat(),
        "interval": interval,
        "session_note": session_note,
        "symbol": TRADES_SYMBOL,
        "gex_enabled": bool(gex_enabled),
        "gex_min_abs_b": float(gex_min_abs_b) if gex_min_abs_b is not None else None,
        "phase": phase,
        "days_either_side": n_each_side,
        "requested_dates": [d.isoformat() for d in target_dates],
        "loaded_dates": [d.isoformat() for d, _ in loaded_frames],
    }, 200

def _build_react_live_trades_overlay_payload(
        trade_date: str | None,
        interval: str | None,
) -> tuple[dict, int]:
    interval = str(interval or "1min").strip()
    if interval not in ("1min", "5min"):
        interval = "1min"

    if not trade_date:
        return {
            "error": "trade_date is required",
            "bars": [],
            "interval": interval,
            "is_live_day": False,
        }, 400

    try:
        selected_date = dt.datetime.strptime(str(trade_date), "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return {
            "error": "Invalid trade date",
            "bars": [],
            "interval": interval,
            "is_live_day": False,
        }, 400

    pt_tz = ZoneInfo("America/Los_Angeles")
    session_date, session_note = _effective_trade_date(selected_date, pt_tz)

    try:
        is_live_day = (session_date == _current_session_trade_date(pt_tz))
    except Exception:
        is_live_day = False

    if not is_live_day:
        return {
            "bars": [],
            "trade_date": str(trade_date),
            "effective_trade_date": session_date.isoformat(),
            "interval": interval,
            "session_note": session_note,
            "symbol": TRADES_SYMBOL,
            "is_live_day": False,
            "source": "live_trades_overlay",
        }, 200

    day_start_pt, day_end_pt = _session_window_pt(session_date, pt_tz)

    try:
        df_base_bars = _fetch_bars_pt(day_start_pt, day_end_pt, interval, pt_tz)
    except Exception as e:
        return {
            "error": f"Database error when loading base bars for live overlay: {e}",
            "bars": [],
            "trade_date": str(trade_date),
            "effective_trade_date": session_date.isoformat(),
            "interval": interval,
            "session_note": session_note,
            "symbol": TRADES_SYMBOL,
            "is_live_day": True,
            "source": "live_trades_overlay",
        }, 500

    if df_base_bars is None or df_base_bars.empty:
        return {
            "bars": [],
            "trade_date": str(trade_date),
            "effective_trade_date": session_date.isoformat(),
            "interval": interval,
            "session_note": session_note,
            "symbol": TRADES_SYMBOL,
            "is_live_day": True,
            "source": "live_trades_overlay",
        }, 200

    if interval == "1min":
        freq = "1min"
        label_mode = "left"
    else:
        freq = "5min"
        label_mode = "right"

    base_ts = pd.to_datetime(df_base_bars.get("datetime_pt"), errors="coerce").dropna()
    if base_ts.empty:
        return {
            "bars": [],
            "trade_date": str(trade_date),
            "effective_trade_date": session_date.isoformat(),
            "interval": interval,
            "session_note": session_note,
            "symbol": TRADES_SYMBOL,
            "is_live_day": True,
            "source": "live_trades_overlay",
        }, 200

    try:
        if base_ts.dt.tz is None:
            base_ts = base_ts.dt.tz_localize(pt_tz)
        else:
            base_ts = base_ts.dt.tz_convert(pt_tz)
    except Exception as e:
        return {
            "error": f"Could not normalize base-bar timestamps for live overlay: {e}",
            "bars": [],
            "trade_date": str(trade_date),
            "effective_trade_date": session_date.isoformat(),
            "interval": interval,
            "session_note": session_note,
            "symbol": TRADES_SYMBOL,
            "is_live_day": True,
            "source": "live_trades_overlay",
        }, 500

    base_ts = base_ts.sort_values()
    last_base_ts = base_ts.iloc[-1]

    now_pt = dt.datetime.now(tz=pt_tz)
    now_ts = pd.Timestamp(now_pt)
    if label_mode == "left":
        current_bucket = now_ts.floor(freq)
    else:
        current_bucket = now_ts.ceil(freq)

    step = pd.Timedelta(freq)
    start_bucket = last_base_ts + step

    if start_bucket > current_bucket:
        return {
            "bars": [],
            "trade_date": str(trade_date),
            "effective_trade_date": session_date.isoformat(),
            "interval": interval,
            "session_note": session_note,
            "symbol": TRADES_SYMBOL,
            "is_live_day": True,
            "source": "live_trades_overlay",
        }, 200

    expected = pd.date_range(start=start_bucket, end=current_bucket, freq=freq)
    if expected.empty:
        return {
            "bars": [],
            "trade_date": str(trade_date),
            "effective_trade_date": session_date.isoformat(),
            "interval": interval,
            "session_note": session_note,
            "symbol": TRADES_SYMBOL,
            "is_live_day": True,
            "source": "live_trades_overlay",
        }, 200

    buffer = max(
        dt.timedelta(seconds=10),
        dt.timedelta(minutes=1) if freq == "1min" else dt.timedelta(minutes=5),
    )
    fetch_start_pt = expected[0].to_pydatetime() - buffer
    fetch_end_pt = expected[-1].to_pydatetime() + buffer

    start_utc = fetch_start_pt.astimezone(ZoneInfo("UTC"))
    end_utc = fetch_end_pt.astimezone(ZoneInfo("UTC"))

    try:
        df_trades = _fetch_trades_utc(start_utc, end_utc, symbol=TRADES_SYMBOL)
    except Exception as e:
        return {
            "error": f"Database error when loading live trades: {e}",
            "bars": [],
            "trade_date": str(trade_date),
            "effective_trade_date": session_date.isoformat(),
            "interval": interval,
            "session_note": session_note,
            "symbol": TRADES_SYMBOL,
            "is_live_day": True,
            "source": "live_trades_overlay",
        }, 500

    overlay_df = _trades_to_ohlc(df_trades, freq=freq, pt_tz=pt_tz, label_mode=label_mode)
    if overlay_df is None or overlay_df.empty:
        return {
            "bars": [],
            "trade_date": str(trade_date),
            "effective_trade_date": session_date.isoformat(),
            "interval": interval,
            "session_note": session_note,
            "symbol": TRADES_SYMBOL,
            "is_live_day": True,
            "source": "live_trades_overlay",
        }, 200

    overlay_df["datetime_pt"] = pd.to_datetime(overlay_df["datetime_pt"], errors="coerce")
    overlay_df = overlay_df.dropna(subset=["datetime_pt"]).copy()
    if overlay_df.empty:
        return {
            "bars": [],
            "trade_date": str(trade_date),
            "effective_trade_date": session_date.isoformat(),
            "interval": interval,
            "session_note": session_note,
            "symbol": TRADES_SYMBOL,
            "is_live_day": True,
            "source": "live_trades_overlay",
        }, 200

    if overlay_df["datetime_pt"].dt.tz is None:
        overlay_df["datetime_pt"] = overlay_df["datetime_pt"].dt.tz_localize(pt_tz)
    else:
        overlay_df["datetime_pt"] = overlay_df["datetime_pt"].dt.tz_convert(pt_tz)

    overlay_df = overlay_df[overlay_df["datetime_pt"].isin(expected)].copy()
    if overlay_df.empty:
        return {
            "bars": [],
            "trade_date": str(trade_date),
            "effective_trade_date": session_date.isoformat(),
            "interval": interval,
            "session_note": session_note,
            "symbol": TRADES_SYMBOL,
            "is_live_day": True,
            "source": "live_trades_overlay",
        }, 200

    base_set = set(pd.to_datetime(base_ts).to_list())
    overlay_df = overlay_df[~overlay_df["datetime_pt"].isin(base_set)].copy()
    if overlay_df.empty:
        return {
            "bars": [],
            "trade_date": str(trade_date),
            "effective_trade_date": session_date.isoformat(),
            "interval": interval,
            "session_note": session_note,
            "symbol": TRADES_SYMBOL,
            "is_live_day": True,
            "source": "live_trades_overlay",
        }, 200

    overlay_df = overlay_df.sort_values("datetime_pt").tail(max(1, LIVE_TRADES_MAX_BARS)).copy()

    rows = _bars_df_to_payload_rows(
        overlay_df,
        pt_tz,
        session_date=session_date,
        is_center=True,
    )
    for row in rows:
        row["is_live_overlay"] = True

    return {
        "bars": rows,
        "trade_date": str(trade_date),
        "effective_trade_date": session_date.isoformat(),
        "interval": interval,
        "session_note": session_note,
        "symbol": TRADES_SYMBOL,
        "is_live_day": True,
        "source": "live_trades_overlay",
        "max_bars": int(LIVE_TRADES_MAX_BARS),
    }, 200

def _build_react_flow_payload(
        trade_date: str | None,
        *,
        session: str | None = None,
        resample: str | None = None,
        ema_len: int | None = None,
) -> tuple[dict, int]:
    if not trade_date:
        return {
            "error": "trade_date is required",
            "flow_points": [],
        }, 400

    try:
        selected_date = dt.datetime.strptime(str(trade_date), "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return {
            "error": "Invalid trade date",
            "flow_points": [],
        }, 400

    pt_tz = ZoneInfo("America/Los_Angeles")
    session_date, session_note = _effective_trade_date(selected_date, pt_tz)

    session_mode = str(session or FLOW_SESSION or "RTH").upper().strip()
    if session_mode not in {"RTH", "FULL"}:
        session_mode = "RTH"

    resample_mode = str(resample or FLOW_RESAMPLE or "1s").lower().strip()
    if resample_mode not in {"1s", "5s", "15s", "1m", "60s"}:
        resample_mode = "1s"

    try:
        span = max(1, int(ema_len if ema_len is not None else FLOW_EMA_LEN))
    except Exception:
        span = FLOW_EMA_LEN

    full_start_pt, full_end_pt = _session_window_pt(session_date, pt_tz)

    if session_mode == "RTH":
        view_start_pt = dt.datetime.combine(session_date, dt.time(6, 30), tzinfo=pt_tz)
        view_end_pt = dt.datetime.combine(session_date, dt.time(13, 0), tzinfo=pt_tz)
    else:
        view_start_pt = full_start_pt
        view_end_pt = full_end_pt

    start_utc = view_start_pt.astimezone(ZoneInfo("UTC"))
    end_utc = view_end_pt.astimezone(ZoneInfo("UTC"))

    try:
        df = _fetch_flow_utc(start_utc, end_utc, symbol=FLOW_SYMBOL)
    except Exception as e:
        return {
            "error": f"Database error when loading flow: {e}",
            "flow_points": [],
            "trade_date": str(trade_date),
            "effective_trade_date": session_date.isoformat(),
            "session": session_mode,
            "resample": resample_mode,
            "ema_len": span,
        }, 500

    if df is None or df.empty:
        return {
            "flow_points": [],
            "trade_date": str(trade_date),
            "effective_trade_date": session_date.isoformat(),
            "session_note": session_note,
            "session": session_mode,
            "resample": resample_mode,
            "ema_len": span,
            "symbol": FLOW_SYMBOL,
            "point_count": 0,
        }, 200

    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts_utc"]).set_index("ts_utc").sort_index()

    for col in ["buy_vol", "sell_vol", "unknown_vol"]:
        if col not in df.columns:
            df[col] = 0.0

    full_idx = pd.date_range(
        start=start_utc,
        end=end_utc,
        freq="1s",
        tz="UTC",
        inclusive="left",
    )
    df = df.reindex(full_idx)
    df[["buy_vol", "sell_vol", "unknown_vol"]] = (
        df[["buy_vol", "sell_vol", "unknown_vol"]].fillna(0.0).astype(float)
    )

    rule_map = {"1s": "1s", "5s": "5s", "15s": "15s", "1m": "1min", "60s": "1min"}
    rule = rule_map.get(resample_mode, "1s")
    if rule != "1s":
        df = df.resample(rule).sum()

    buy = df["buy_vol"].astype(float)
    sell = df["sell_vol"].astype(float)

    ema_buy = buy.ewm(span=span, adjust=False).mean()
    ema_sell = sell.ewm(span=span, adjust=False).mean()
    diff = (ema_buy - ema_sell).astype(float)

    flow_points: list[dict] = []
    for ts, val in diff.items():
        try:
            if pd.isna(ts) or not np.isfinite(float(val)):
                continue
            flow_points.append({
                "time": int(pd.Timestamp(ts).timestamp()),
                "value": float(val),
            })
        except Exception:
            continue

    return {
        "flow_points": flow_points,
        "trade_date": str(trade_date),
        "effective_trade_date": session_date.isoformat(),
        "session_note": session_note,
        "session": session_mode,
        "resample": resample_mode,
        "ema_len": span,
        "symbol": FLOW_SYMBOL,
        "point_count": len(flow_points),
    }, 200


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
            marker=dict(size=10, color="rgba(0,0,0,0)"),
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
    total_seconds = 0

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
        total_seconds += int((hi - lo).total_seconds()) + 1

    if not overlaps:
        empty = go.Scattergl(
            x=[],
            y=[],
            mode="markers",
            marker=dict(size=10, color="rgba(0,0,0,0)"),
            opacity=HOVERGRID_OPACITY,
            name="__hovergrid__empty",
            showlegend=False,
            yaxis="y2",
            customdata=[],
            hovertemplate="Time=%{x|%Y-%m-%d %H:%M:%S}<br>Price=%{customdata:.2f}<extra></extra>",
        )
        return empty, empty

        # Prefer fine X spacing (seconds) for "cursor anywhere"; shrink Y first to stay under max_points.
    x_step_req = max(1, int(HOVERGRID_X_SECONDS))
    # Approximate requested x points across all overlaps
    x_points_req = max(1, int(math.ceil(total_seconds / x_step_req)) + len(overlaps))
    y_points_eff = min(y_points_req, max(20, int(max_points // max(1, x_points_req))))

    # Choose an effective X step so (x_points_eff * y_points_eff) <= max_points
    allowed_x_points = max(100, int(max_points // max(1, y_points_eff)))
    step_sec = max(x_step_req, int(math.ceil(total_seconds / max(1, allowed_x_points))))

    y_vec = np.linspace(float(y_min), float(y_max), int(y_points_eff)).astype(float)
    y_step = float(y_vec[1] - y_vec[0]) if len(y_vec) > 1 else 1.0
    tol = float(max(GEX_HOVER_TOLERANCE, 0.60 * y_step))

    x_base_parts, y_base_parts = [], []
    x_gex_parts, y_gex_parts = [], []

    for (d_str, lo, hi) in overlaps:
        x_day = pd.date_range(lo, hi, freq=f"{step_sec}s", inclusive="both").to_pydatetime().tolist()
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
        customdata=y_base,
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
        customdata=y_gex,
        hovertemplate=hover_tpl,
    )

    return base_trace, gex_trace


def _empty_hoverline(name: str) -> go.Scattergl:
    return go.Scattergl(
        x=[],
        y=[],
        mode="markers",
        marker=dict(size=6, color="rgba(0,0,0,0)"),
        opacity=float(os.getenv("IRONBEAM_HOVERLINE_OPACITY", "0.01")),
        name=name,
        showlegend=False,
        yaxis="y2",
        customdata=[],
        hovertemplate="Time=%{x|%Y-%m-%d %H:%M:%S}<br>Price=%{customdata:.2f}<extra></extra>",
    )


def _build_hoverline_from_df(df_bars: pd.DataFrame) -> go.Scattergl:
    """Lightweight hover trigger: one invisible point per bar (much faster than a 2D hover grid)."""
    if df_bars is None or df_bars.empty or "datetime_pt" not in df_bars.columns:
        return _empty_hoverline("__hoverline__")

    xs = df_bars["datetime_pt"].tolist()
    ys = df_bars["close"].astype(float).tolist() if "close" in df_bars.columns else [0.0] * len(xs)

    # Downsample to keep interactions snappy
    max_pts = max(5000, int(HOVERLINE_MAX_POINTS))
    if len(xs) > max_pts:
        step = int(math.ceil(len(xs) / max_pts))
        xs = xs[::step]
        ys = ys[::step]

    return go.Scattergl(
        x=xs,
        y=ys,
        mode="markers",
        marker=dict(size=6, color="rgba(0,0,0,0)"),
        opacity=float(os.getenv("IRONBEAM_HOVERLINE_OPACITY", "0.01")),
        name="__hoverline__",
        showlegend=False,
        yaxis="y2",
        customdata=ys,
        hovertemplate="Time=%{x|%Y-%m-%d %H:%M:%S}<br>Price=%{customdata:.2f}<extra></extra>",
    )


def _build_hoverline_from_fig(fig_obj: go.Figure, pt_tz: ZoneInfo) -> go.Scattergl:
    xs: list = []
    ys: list[float] = []

    try:
        for tr in fig_obj.data:
            if getattr(tr, "type", None) != "candlestick":
                continue
            nm = str(getattr(tr, "name", "") or "")
            if not nm.startswith("ES "):
                continue
            x = list(getattr(tr, "x", []) or [])
            c = list(getattr(tr, "close", []) or [])
            if not x or not c:
                continue
            # align lengths
            n = min(len(x), len(c))
            xs.extend(x[:n])
            ys.extend([float(v) for v in c[:n]])
    except Exception:
        xs, ys = [], []

    if not xs:
        return _empty_hoverline("__hoverline__")

    # Sort by time (best effort)
    try:
        ts = pd.to_datetime(xs, errors="coerce")
        if isinstance(ts, pd.DatetimeIndex) or isinstance(ts, pd.Series):
            m = ~pd.isna(ts)
            ts_valid = ts[m]
            xs_valid = [xs[i] for i, ok in enumerate(m) if ok]
            ys_valid = [ys[i] for i, ok in enumerate(m) if ok]
            order = list(ts_valid.argsort())
            xs = [xs_valid[i] for i in order]
            ys = [ys_valid[i] for i in order]
    except Exception:
        pass

    max_pts = max(5000, int(HOVERLINE_MAX_POINTS))
    if len(xs) > max_pts:
        step = int(math.ceil(len(xs) / max_pts))
        xs = xs[::step]
        ys = ys[::step]

    return go.Scattergl(
        x=xs,
        y=ys,
        mode="markers",
        marker=dict(size=6, color="rgba(0,0,0,0)"),
        opacity=float(os.getenv("IRONBEAM_HOVERLINE_OPACITY", "0.01")),
        name="__hoverline__",
        showlegend=False,
        yaxis="y2",
        customdata=ys,
        hovertemplate="Time=%{x|%Y-%m-%d %H:%M:%S}<br>Price=%{customdata:.2f}<extra></extra>",
    )


def build_aggressor_flow_figure(trade_date, indicator_state, shared_xrange):
    """Build the Aggressor Flow panel figure (used by dynamic indicator panels)."""
    if not trade_date:
        return go.Figure(layout_title_text="Select a trade date to view Aggressor Flow.")

    # Indicator enable + settings (from sidebar)
    enabled = []
    if isinstance(indicator_state, dict):
        enabled = indicator_state.get("enabled") or []
    if not isinstance(enabled, list):
        enabled = [enabled] if enabled else []

    if "aggressor_flow" not in enabled:
        return go.Figure(layout_title_text="Aggressor Flow is disabled (enable it in the Indicators panel).")

    cfg = {}
    if isinstance(indicator_state, dict):
        cfg = (indicator_state.get("cfg") or {}).get("aggressor_flow") or {}

    session_mode = str(cfg.get("session", FLOW_SESSION)).upper()
    resample_mode = str(cfg.get("resample", FLOW_RESAMPLE)).lower()
    ema_len = int(cfg.get("ema_len", FLOW_EMA_LEN))

    panel_height = cfg.get("panel_height", 330)
    try:
        panel_height = int(float(panel_height))
    except Exception:
        panel_height = 330
    panel_height = max(120, min(700, panel_height))

    pos_line = str(cfg.get("pos_color", FLOW_POS_COLOR))
    neg_line = str(cfg.get("neg_color", FLOW_NEG_COLOR))

    hist_alpha = cfg.get("hist_alpha", float(os.getenv("IRONBEAM_FLOW_HIST_ALPHA", "0.30")))
    try:
        hist_alpha = float(hist_alpha)
    except Exception:
        hist_alpha = float(os.getenv("IRONBEAM_FLOW_HIST_ALPHA", "0.30"))

    pt_tz = ZoneInfo("America/Los_Angeles")
    session_date = dt.date.fromisoformat(trade_date)
    now_pt = dt.datetime.now(pt_tz)

    # Session window in PT
    if session_mode == "RTH":
        start_pt = dt.datetime.combine(session_date, dt.time(6, 30), tzinfo=pt_tz)
        end_pt = dt.datetime.combine(session_date, dt.time(13, 0), tzinfo=pt_tz)
    else:  # FULL
        start_pt = dt.datetime.combine(session_date, dt.time(0, 0), tzinfo=pt_tz)
        end_pt = dt.datetime.combine(session_date, dt.time(23, 59, 59), tzinfo=pt_tz)

    if session_date == now_pt.date():
        end_pt = min(end_pt, now_pt)

    start_utc = _floor_to_sec(start_pt.astimezone(ZoneInfo("UTC")))
    end_utc = _floor_to_sec(end_pt.astimezone(ZoneInfo("UTC")))
    if end_utc <= start_utc:
        return go.Figure(layout_title_text="Aggressor Flow: empty time range.")

    try:
        df = _fetch_flow_utc(start_utc, end_utc, symbol=FLOW_SYMBOL)
    except Exception as e:
        return go.Figure(layout_title_text=f"Aggressor Flow DB error: {e}")

    if df.empty:
        return go.Figure(layout_title_text="Aggressor Flow: no data for this window (yet).")

    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts_utc"]).set_index("ts_utc").sort_index()

    for col in ["buy_vol", "sell_vol", "unknown_vol"]:
        if col not in df.columns:
            df[col] = 0.0

    # Continuous 1s index (fills missing seconds with 0)
    full_idx = pd.date_range(start=start_utc, end=end_utc, freq="1s", tz="UTC", inclusive="left")
    df = df.reindex(full_idx)
    df[["buy_vol", "sell_vol", "unknown_vol"]] = (
        df[["buy_vol", "sell_vol", "unknown_vol"]].fillna(0.0).astype(float)
    )

    # Optional resample
    rule_map = {"1s": "1s", "5s": "5s", "15s": "15s", "1m": "1min", "60s": "1min"}
    rule = rule_map.get((resample_mode or "1s").lower(), "1s")
    if rule != "1s":
        df = df.resample(rule).sum()

    buy = df["buy_vol"].astype(float)
    sell = df["sell_vol"].astype(float)

    span = max(1, int(ema_len))
    ema_buy = buy.ewm(span=span, adjust=False).mean()
    ema_sell = sell.ewm(span=span, adjust=False).mean()

    # Signed histogram
    diff = (ema_buy - ema_sell).astype(float)

    x = ema_buy.index.tz_convert(pt_tz)
    x_list = list(x)

    pos_hist_fill = _hex_to_rgba(pos_line, hist_alpha)
    neg_hist_fill = _hex_to_rgba(neg_line, hist_alpha)

    fig = go.Figure()

    # ---- Histogram fill in contiguous segments (prevents any wrong-side fill) ----
    diff_arr = diff.to_numpy(dtype=float)

    def _add_hist_segments(mask: np.ndarray, line_color: str, fill_color: str):
        idx = np.flatnonzero(mask & np.isfinite(diff_arr))
        if idx.size == 0:
            return

        start = int(idx[0])
        prev = int(idx[0])

        for k in idx[1:]:
            k = int(k)
            if k != prev + 1:
                if prev - start >= 1:
                    xs = x_list[start: prev + 1]
                    ys = diff_arr[start: prev + 1].tolist()
                    fig.add_trace(
                        go.Scatter(
                            x=xs,
                            y=ys,
                            mode="lines",
                            line=dict(color=line_color, width=1.5),
                            fill="tozeroy",
                            fillcolor=fill_color,
                            showlegend=False,
                            hovertemplate="Diff=%{y:.2f}<extra></extra>",
                            connectgaps=False,
                        )
                    )
                start = k
            prev = k

        if prev - start >= 1:
            xs = x_list[start: prev + 1]
            ys = diff_arr[start: prev + 1].tolist()
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color=line_color, width=1.5),
                    fill="tozeroy",
                    fillcolor=fill_color,
                    showlegend=False,
                    hovertemplate="Diff=%{y:.2f}<extra></extra>",
                    connectgaps=False,
                )
            )

    _add_hist_segments(diff_arr >= 0, pos_line, pos_hist_fill)
    _add_hist_segments(diff_arr < 0, neg_line, neg_hist_fill)

    # Zero line
    fig.add_hline(y=0, line_width=1, line_dash="solid", line_color="rgba(255,255,255,0.25)")

    # ---- Center 0 in the middle: symmetric y-range around 0 ----
    finite = diff_arr[np.isfinite(diff_arr)]
    if finite.size == 0:
        max_abs = 1.0
    else:
        max_abs = float(np.nanmax(np.abs(finite)))
        if not np.isfinite(max_abs) or max_abs <= 0:
            max_abs = 1.0

    pad = max(1e-6, 0.10 * max_abs)
    y_range = [-max_abs - pad, max_abs + pad]

    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor=ETH_BG_COLOR,
        paper_bgcolor=ETH_BG_COLOR,
        margin=dict(l=90, r=80, t=55, b=50, autoexpand=False),
        height=panel_height,
        showlegend=False,
        uirevision=f"ironbeam-flow-{trade_date}-{resample_mode}-{span}-{session_mode}-histonly",
        dragmode="pan",
        hovermode="x unified",
    )

    fig.update_xaxes(rangeslider=dict(visible=False))
    fig.update_yaxes(
        title_text="Aggression",
        showgrid=True,
        fixedrange=False,
        zeroline=False,
        range=y_range,
        autorange=False,
    )

    # Apply shared x-range if available
    def _parse_to_pt(v):
        """Coerce x values into PT tz-aware pandas Timestamps.

        Plotly/Dash sometimes emits naive strings (no timezone). Those are in the
        chart's display timezone (PT for this app), *not* UTC.
        """
        if v is None:
            return None
        try:
            ts = pd.to_datetime(v, errors="coerce")
        except Exception:
            return None
        if ts is pd.NaT:
            return None

        try:
            ts = pd.Timestamp(ts)
        except Exception:
            return None

        try:
            if ts.tzinfo is None:
                ts = ts.tz_localize(pt_tz)
            else:
                ts = ts.tz_convert(pt_tz)
        except Exception:
            return None
        return ts

    if isinstance(shared_xrange, dict):
        x0 = _parse_to_pt(shared_xrange.get("x0"))
        x1 = _parse_to_pt(shared_xrange.get("x1"))
        if x0 is not None and x1 is not None:
            fig.update_xaxes(range=[x0, x1], autorange=False)

        hx = _parse_to_pt(shared_xrange.get("hover_x"))
        if hx is not None:
            fig.add_vline(
                x=hx,
                line_width=1,
                line_dash="dash",
                line_color="rgba(255,255,255,0.35)",
            )

    # Check if y-range is locked in shared_xrange (we reuse this store for y-locks too)
    if isinstance(shared_xrange, dict):
        y0 = shared_xrange.get(f"y0_{indicator_state.get('enabled', [])[0] if indicator_state.get('enabled') else ''}")
        y1 = shared_xrange.get(f"y1_{indicator_state.get('enabled', [])[0] if indicator_state.get('enabled') else ''}")
        # Actually, we need to know the specific panel ID. But here we are building for 'aggressor_flow'.
        # Let's use a specific key for flow y-range.
        flow_y0 = shared_xrange.get("flow_y0")
        flow_y1 = shared_xrange.get("flow_y1")

        if flow_y0 is not None and flow_y1 is not None:
            try:
                fig.update_yaxes(range=[float(flow_y0), float(flow_y1)], autorange=False)
            except Exception:
                pass

    return fig


# ---------- Dash callback registration ----------
def register_ironbeam_callbacks(app):
    # -------------------------
    # React Preview: lightweight bars API for the standalone React chart
    # -------------------------
    if not getattr(app.server, "_ironbeam_react_bars_route_registered", False):

        @app.server.route("/api/ironbeam/bars", methods=["GET"])
        def ironbeam_react_bars_api():
            trade_date = request.args.get("trade_date")
            interval = request.args.get("interval")
            phase = (request.args.get("phase") or "center").strip().lower()

            gex_enabled_raw = (request.args.get("gex_enabled") or "1").strip().lower()
            gex_enabled = gex_enabled_raw not in {"0", "false", "no", "off"}

            gex_min_abs_b_raw = request.args.get("gex_min_abs_b")
            try:
                gex_min_abs_b = float(gex_min_abs_b_raw) if gex_min_abs_b_raw not in (None, "") else None
            except Exception:
                gex_min_abs_b = None

            days_either_side_raw = request.args.get("days_either_side")
            try:
                days_either_side = int(days_either_side_raw) if days_either_side_raw not in (None, "") else None
            except Exception:
                days_either_side = None

            payload, status = _build_react_preview_bars_payload(
                trade_date,
                interval,
                gex_enabled=gex_enabled,
                gex_min_abs_b=gex_min_abs_b,
                phase=phase,
                days_either_side=days_either_side,
            )
            resp = jsonify(payload)

            origin = request.headers.get("Origin")

            allowed = {
                "http://localhost:5173",
                "http://127.0.0.1:5173",
                "http://0.0.0.0:5173",
            }

            extra_allowed = os.getenv("IRONBEAM_REACT_ALLOWED_ORIGINS", "")
            for item in extra_allowed.split(","):
                item = item.strip()
                if item:
                    allowed.add(item)

            if origin in allowed:
                resp.headers["Access-Control-Allow-Origin"] = origin
                resp.headers["Access-Control-Allow-Credentials"] = "true"
                resp.headers["Vary"] = "Origin"

            return resp, status

        app.server._ironbeam_react_bars_route_registered = True

        if not getattr(app.server, "_ironbeam_react_live_trades_overlay_route_registered", False):

            @app.server.route("/api/ironbeam/live-trades-overlay", methods=["GET"])
            def ironbeam_react_live_trades_overlay_api():
                trade_date = request.args.get("trade_date")
                interval = request.args.get("interval")

                payload, status = _build_react_live_trades_overlay_payload(
                    trade_date,
                    interval,
                )

                resp = jsonify(payload)

                origin = request.headers.get("Origin")
                allowed = {
                    "http://localhost:5173",
                    "http://127.0.0.1:5173",
                    "http://0.0.0.0:5173",
                }

                extra_allowed = os.getenv("IRONBEAM_REACT_ALLOWED_ORIGINS", "")
                for item in extra_allowed.split(","):
                    item = item.strip()
                    if item:
                        allowed.add(item)

                if origin in allowed:
                    resp.headers["Access-Control-Allow-Origin"] = origin
                    resp.headers["Access-Control-Allow-Credentials"] = "true"
                    resp.headers["Vary"] = "Origin"

                return resp, status

            app.server._ironbeam_react_live_trades_overlay_route_registered = True

        if not getattr(app.server, "_ironbeam_react_flow_route_registered", False):

            @app.server.route("/api/ironbeam/flow", methods=["GET"])
            def ironbeam_react_flow_api():
                trade_date = request.args.get("trade_date")
                session = request.args.get("session", FLOW_SESSION)
                resample = request.args.get("resample", FLOW_RESAMPLE)
                raw_ema_len = request.args.get("ema_len", str(FLOW_EMA_LEN))

                try:
                    ema_len = max(1, int(raw_ema_len))
                except Exception:
                    ema_len = FLOW_EMA_LEN

                payload, status = _build_react_flow_payload(
                    trade_date,
                    session=session,
                    resample=resample,
                    ema_len=ema_len,
                )

                resp = jsonify(payload)

                origin = request.headers.get("Origin")
                allowed = {
                    "http://localhost:5173",
                    "http://127.0.0.1:5173",
                    "http://0.0.0.0:5173",
                }
                if origin in allowed:
                    resp.headers["Access-Control-Allow-Origin"] = origin
                    resp.headers["Access-Control-Allow-Credentials"] = "true"
                    resp.headers["Vary"] = "Origin"

                return resp, status

            app.server._ironbeam_react_flow_route_registered = True

    # ---- Clientside Sync: Crosshair & Zoom ----
    app.clientside_callback(
        ClientsideFunction(namespace="ironbeam", function_name="sync_crosshair_and_zoom"),
        Output({"type": "ib-indicator-panel", "id": MATCH}, "figure", allow_duplicate=True),
        [Input("ironbeam-chart", "relayoutData"), Input("ironbeam-chart", "hoverData")],
        [State({"type": "ib-indicator-panel", "id": MATCH}, "figure")],
        prevent_initial_call=True,
    )

    # ---- Capture Indicator Panel Zoom (Y-axis) ----
    @app.callback(
        Output("ib-shared-xrange", "data", allow_duplicate=True),
        Input({"type": "ib-indicator-panel", "id": ALL}, "relayoutData"),
        State("ib-shared-xrange", "data"),
        State({"type": "ib-indicator-panel", "id": ALL}, "id"),
        prevent_initial_call=True,
    )
    def capture_indicator_zoom(relayouts, current_shared, panel_ids):
        if not ctx.triggered:
            raise PreventUpdate

        trig_id = ctx.triggered_id
        if not trig_id or trig_id.get("type") != "ib-indicator-panel":
            raise PreventUpdate

        pid = trig_id.get("id")

        # Find the relayout data for this specific ID
        try:
            # panel_ids is a list of dicts like {'id': 'aggressor_flow', 'type': 'ib-indicator-panel'}
            # We need to find the index where the dict matches trig_id
            idx = panel_ids.index(trig_id)
            relayout = relayouts[idx]
        except (ValueError, IndexError, AttributeError):
            raise PreventUpdate

        if not relayout or not isinstance(relayout, dict):
            raise PreventUpdate

        # Only process for aggressor_flow for now
        if pid != "aggressor_flow":
            raise PreventUpdate

        data = dict(current_shared or {})
        changed = False

        # Check for Y-axis zoom/pan
        y0 = relayout.get("yaxis.range[0]")
        y1 = relayout.get("yaxis.range[1]")

        # Handle autorange (reset zoom)
        if relayout.get("yaxis.autorange") is True:
            if "flow_y0" in data:
                del data["flow_y0"]
                changed = True
            if "flow_y1" in data:
                del data["flow_y1"]
                changed = True
        elif y0 is not None and y1 is not None:
            if data.get("flow_y0") != y0 or data.get("flow_y1") != y1:
                data["flow_y0"] = y0
                data["flow_y1"] = y1
                changed = True

        if not changed:
            raise PreventUpdate

        return data

    # ---- Clientside Highlight: Click on Candle ----
    app.clientside_callback(
        ClientsideFunction(namespace="ironbeam", function_name="highlight_candle"),
        Output("ironbeam-chart", "figure", allow_duplicate=True),
        [Input("ironbeam-chart", "clickData")],
        [State("ironbeam-chart", "figure")],
        prevent_initial_call=True,
    )

    @app.callback(
        Output("ironbeam-chart", "figure"),
        [
            Input("trade-date", "date"),
            Input("smile-time-input", "value"),
            Input("ironbeam-bar-interval", "value"),
            Input("ib-indicator-state", "data"),
        ],
        [State("ironbeam-chart", "figure")],
    )
    def update_chart(trade_date, selected_times_pt, bar_interval, indicator_state, prev_fig):
        if not trade_date:
            return go.Figure(layout_title_text="Select a trade date to view chart.")

        if selected_times_pt is None:
            selected_times: list[str] = []
        elif isinstance(selected_times_pt, list):
            selected_times = [str(t) for t in selected_times_pt]
        else:
            selected_times = [str(selected_times_pt)]

        interval = bar_interval or "1min"
        # Decide whether the GEX overlay is enabled + what threshold to use.
        indicator_state = indicator_state if isinstance(indicator_state, dict) else {}
        enabled_list = indicator_state.get("enabled") or []
        if not isinstance(enabled_list, list):
            enabled_list = [enabled_list]
        gex_enabled = ("gex_overlay" in enabled_list)

        # Fallback: old top-level threshold control (billions)
        fallback_threshold = GEX_ABS_THRESHOLD_DEFAULT

        # Preferred: plugin config (Min |GEX| in billions)
        gex_plugin = IB_PLUGIN_MAP.get("gex_overlay")
        gex_defaults = (getattr(gex_plugin, "default_config", lambda: {})() or {}) if gex_plugin else {}
        gex_cfg_all = (indicator_state.get("cfg") or {}) if isinstance(indicator_state.get("cfg"), dict) else {}
        gex_cfg = gex_cfg_all.get("gex_overlay") if isinstance(gex_cfg_all.get("gex_overlay"), dict) else {}
        min_abs_b = gex_cfg.get("min_abs_b", gex_defaults.get("min_abs_b"))
        current_threshold = (float(min_abs_b) * 1e9) if (min_abs_b is not None) else fallback_threshold

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
        target_dates = _window_trade_dates(session_date, LEGACY_CLASSIC_DAYS_EITHER_SIDE)
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
            return go.Figure(layout_title_text=f"No ES bar data for session {session_date.isoformat()}.")

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
        if gex_enabled:
            try:
                df_gex = _fetch_gex_grouped_by_level(session_date)
            except Exception as e:
                print(f"[Ironbeam] Error fetching GEX: {e}")
                df_gex = pd.DataFrame(columns=["level", "call_gamma", "put_gamma", "net_gamma"])
        else:
            df_gex = pd.DataFrame(columns=["level", "call_gamma", "put_gamma", "net_gamma"])

        if gex_enabled and (not df_gex.empty):
            # selection for hover-grid + consistent colorbar span
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

                # store selected levels (for hover-grid / debugging)
                gex_levels_by_day[session_date.isoformat()] = [
                    [float(r["level"]), float(r["net_gamma"]) / 1e9] for _, r in df_gex_day.iterrows()
                ]

                # overlay lines are now built by the plugin helper (keeps callbacks smaller)
                for tr in _build_gex_overlay_traces_plugin(
                        df_gex=df_gex,
                        x_start=day_start_pt,
                        x_end=day_end_pt,
                        band_min=band_min,
                        band_max=band_max,
                        threshold_abs=current_threshold,
                        name_prefix=f"GEX {session_date.isoformat()}",
                ):
                    fig.add_trace(tr)

        if gex_enabled and "coloraxis" not in fig.layout:
            # Ensure a coloraxis exists (for multi-day updates) even if no levels plotted yet.
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
                showlegend=False,
                yaxis="y2",
                hovertemplate="<extra></extra>",
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
                        hovertemplate="<extra></extra>",
                    )
                )

        # --- Live trades overlay (only when chart is on current session day) ---
        try:
            is_live_day = (session_date == _current_session_trade_date(pt_tz))
        except Exception:
            is_live_day = False

        if is_live_day:
            try:
                _apply_live_trades_overlay(fig, df_bars, interval, pt_tz)
            except Exception as e:
                print(f"[Ironbeam] live overlay error: {e}")

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
        meta["indicator_state_token"] = _indicator_state_token(indicator_state)
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
        # Hover trigger (kept lightweight by default; hovergrid is optional)
        if USE_HOVERGRID:
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
        else:
            # One invisible point per bar is enough to keep crosshairs/hover smooth.
            fig.add_trace(_build_hoverline_from_df(df_bars))

        fig.update_layout(
            # title=f"ES (front month) + Net GEX Lines (multi-day; center={session_date.isoformat()})",
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
            height=None,
            autosize=True,
            shapes=shapes,
            meta=meta,
            margin=dict(l=90, r=80, t=80, b=80, autoexpand=False),
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
        State("smile-time-input", "value"),
        State("ironbeam-bar-interval", "value"),
        State("ib-indicator-state", "data"),
        State("ib-shared-xrange", "data"),
        prevent_initial_call=True,
    )
    def persist_zoom(relayout, fig, trade_date, selected_times_pt, bar_interval, indicator_state, shared_xrange):
        if not isinstance(fig, dict) or not isinstance(relayout, dict) or not trade_date:
            raise PreventUpdate

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
        # Guard against stale relayout events overwriting a freshly-rendered figure after indicator toggles/settings changes.
        current_token = _indicator_state_token(indicator_state)
        fig_token = meta.get("indicator_state_token")
        if fig_token is not None and current_token and fig_token != current_token:
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

        meta["indicator_state_token"] = _indicator_state_token(indicator_state)
        meta["last_relayout_ms"] = int(time.time() * 1000)

        # Use Patch to update meta without sending full figure
        p = Patch()
        p.layout.meta = meta
        return p

    # ---- Progressive loader + live refresh (and prevent stale interval overwrites) ----
    @app.callback(
        Output("ironbeam-chart", "figure", allow_duplicate=True),
        Input("ironbeam-interval", "n_intervals"),
        State("ironbeam-chart", "figure"),
        State("trade-date", "date"),
        State("smile-time-input", "value"),
        State("ironbeam-bar-interval", "value"),
        State("ib-indicator-state", "data"),
        State("ib-shared-xrange", "data"),
        prevent_initial_call=True,
    )
    def progressively_add_days(n_intervals, fig, trade_date, selected_times_pt, bar_interval, indicator_state, shared_xrange):
        if not isinstance(fig, dict) or not trade_date:
            raise PreventUpdate

        try:
            selected_date = dt.datetime.strptime(trade_date, "%Y-%m-%d").date()
        except (TypeError, ValueError):
            raise PreventUpdate

        interval = bar_interval or "1min"
        # Guard against stale interval overwrites after indicator toggles.
        current_token = _indicator_state_token(indicator_state)
        meta = (fig.get("layout") or {}).get("meta") if isinstance(fig, dict) else {}
        fig_token = meta.get("indicator_state_token") if isinstance(meta, dict) else None
        if fig_token is not None and current_token and fig_token != current_token:
            raise PreventUpdate

        # Respect the GEX overlay enabled state + plugin threshold.
        indicator_state = indicator_state if isinstance(indicator_state, dict) else {}
        enabled_list = indicator_state.get("enabled") or []
        if not isinstance(enabled_list, list):
            enabled_list = [enabled_list]
        gex_enabled = ("gex_overlay" in enabled_list)
        if not gex_enabled:
            raise PreventUpdate

        fallback_threshold = GEX_ABS_THRESHOLD_DEFAULT
        gex_plugin = IB_PLUGIN_MAP.get("gex_overlay")
        gex_defaults = (getattr(gex_plugin, "default_config", lambda: {})() or {}) if gex_plugin else {}
        gex_cfg_all = (indicator_state.get("cfg") or {}) if isinstance(indicator_state.get("cfg"), dict) else {}
        gex_cfg = gex_cfg_all.get("gex_overlay") if isinstance(gex_cfg_all.get("gex_overlay"), dict) else {}
        min_abs_b = gex_cfg.get("min_abs_b", gex_defaults.get("min_abs_b"))
        current_threshold = (float(min_abs_b) * 1e9) if (min_abs_b is not None) else fallback_threshold

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

        # Use Patch() to avoid resetting client-side zoom state
        p = Patch()

        # We still need to read from 'fig' to check existing traces
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
        df_live = pd.DataFrame()  # default; may be replaced when session is live
        loaded_changed = False

        # ---- live refresh (selected session candle trace + trades overlay) ----
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

                # Find trace index in existing figure
                if "data" in fig and isinstance(fig["data"], list):
                    for i, tr in enumerate(fig["data"]):
                        if tr.get("type") == "candlestick" and tr.get("name") == trace_name:
                            idx_trace = i
                            xs = tr.get("x", [])
                            if xs:
                                old_last = str(xs[-1])
                            break

                new_last = str(df_live["datetime_pt"].iloc[-1])
                if old_last != new_last and idx_trace is not None:
                    # Update existing trace using Patch
                    p.data[idx_trace].x = df_live["datetime_pt"].astype(str).tolist()
                    p.data[idx_trace].open = df_live["open"].astype(float).tolist()
                    p.data[idx_trace].high = df_live["high"].astype(float).tolist()
                    p.data[idx_trace].low = df_live["low"].astype(float).tolist()
                    p.data[idx_trace].close = df_live["close"].astype(float).tolist()
                    did_anything = True

                # Refresh live overlay EVERY tick
                try:
                    changed = _apply_live_trades_overlay_patch(p, fig, df_live, interval, pt_tz)
                    if changed:
                        did_anything = True
                except Exception as e:
                    print(f"[Ironbeam live] overlay error: {e}")

        # ---- progressive add of other days ----
        view_start = None
        view_end = None
        locked_x_range = meta.get("locked_x_range") if isinstance(meta, dict) else None
        if isinstance(locked_x_range, (list, tuple)) and len(locked_x_range) == 2:
            try:
                vs = pd.to_datetime(locked_x_range[0])
                ve = pd.to_datetime(locked_x_range[1])
                if isinstance(vs, pd.Timestamp):
                    if vs.tzinfo is None:
                        vs = vs.tz_localize(pt_tz)
                    else:
                        vs = vs.tz_convert(pt_tz)
                if isinstance(ve, pd.Timestamp):
                    if ve.tzinfo is None:
                        ve = ve.tz_localize(pt_tz)
                    else:
                        ve = ve.tz_convert(pt_tz)
                view_start = vs.to_pydatetime()
                view_end = ve.to_pydatetime()
                if view_start > view_end:
                    view_start, view_end = view_end, view_start
            except Exception:
                view_start, view_end = None, None

        # Optional cooldown after a relayout event (avoid updates mid-drag).
        cooldown_active = False
        try:
            last_ms = meta.get("last_relayout_ms") if isinstance(meta, dict) else None
            if last_ms is not None:
                now_ms = int(time.time() * 1000)
                cooldown_active = (now_ms - int(last_ms)) < int(ZOOM_COOLDOWN_MS)
        except Exception:
            cooldown_active = False

        # Optional cooldown after hover updates (avoid resetting crosshair mid-hover).
        hover_cooldown_active = False
        try:
            if isinstance(shared_xrange, dict):
                hm = shared_xrange.get("last_hover_ms")
                if hm is not None:
                    now_ms = int(time.time() * 1000)
                    hover_cooldown_active = (now_ms - int(hm)) < int(HOVER_COOLDOWN_MS)
        except Exception:
            hover_cooldown_active = False

        # If hovering, skip all figure mutations for this tick to prevent crosshair flicker.
        if hover_cooldown_active:
            raise PreventUpdate

        needed: list[str] = []
        if (not MULTIDAY_PREFETCH) and view_start and view_end:
            for d_str in target_dates:
                if d_str == session_date.isoformat():
                    continue
                try:
                    d = dt.datetime.strptime(d_str, "%Y-%m-%d").date()
                except Exception:
                    continue
                sess_start, sess_end = _session_window_pt(d, pt_tz)
                if sess_end >= view_start and sess_start <= view_end:
                    needed.append(d_str)

        if MULTIDAY_PREFETCH or not needed:
            remaining = [d for d in target_dates if d != session_date.isoformat() and d not in loaded and d not in skipped]
        else:
            remaining = [d for d in needed if d not in loaded and d not in skipped]

        batch: list[str] = []
        if remaining and not cooldown_active:
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
            caxis = (fig.get("layout") or {}).get("coloraxis")
            if caxis and caxis.get("cmin") is not None and caxis.get("cmax") is not None:
                cmin, cmax = float(caxis["cmin"]), float(caxis["cmax"])
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

            # Add trace via Patch
            new_trace = dict(
                type="candlestick",
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
                hovertemplate="<extra></extra>",
            )
            p.data.append(new_trace)

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

                        gex_trace = dict(
                            type="scattergl",
                            x=[day_start_pt, day_end_pt],
                            y=[lvl, lvl],
                            mode="lines",
                            line=dict(color=color, width=line_width),
                            opacity=line_opacity,
                            name=f"GEX {target_str}",
                            showlegend=False,
                            hoverinfo="skip",
                        )
                        p.data.append(gex_trace)

            loaded.append(target_str)
            loaded_changed = True
            did_anything = True

        p.layout.meta["multi_loaded_dates"] = sorted(set(loaded))
        p.layout.meta["multi_skip_dates"] = sorted(set(skipped))
        p.layout.meta["gex_levels_by_day"] = gex_levels_by_day

        locked_y_range = meta.get("locked_y_range")
        # We do NOT update locked_x_range here to avoid resetting user zoom.
        # The client maintains the x-range.

        if locked_y_range is not None:
            # We can update Y range if needed, but usually better to leave it unless we really need to enforce it.
            # If we don't touch it, client keeps it.
            pass

        # Rebuild hover trigger only when we add days (avoid heavy work every tick)
        if loaded_changed:
            # Remove old hover traces
            indices_to_remove = []
            if "data" in fig and isinstance(fig["data"], list):
                for i, tr in enumerate(fig["data"]):
                    nm = tr.get("name", "")
                    if nm.startswith("__hovergrid__") or nm.startswith("__hoverline__"):
                        indices_to_remove.append(i)

            for i in sorted(indices_to_remove, reverse=True):
                del p.data[i]

            if USE_HOVERGRID:
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
                    # Infer from fig data
                    lows, highs = [], []
                    for tr in fig.get("data", []):
                        if tr.get("type") == "candlestick":
                            if tr.get("low"): lows.extend(tr["low"])
                            if tr.get("high"): highs.extend(tr["high"])
                    if lows and highs:
                        y_min, y_max = float(np.nanmin(lows)), float(np.nanmax(highs))
                    else:
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
                # Convert go objects to dicts for Patch
                p.data.append(hb.to_plotly_json())
                p.data.append(hg.to_plotly_json())
            else:
                pass

        # ---- re-apply selected time-slice highlight so it persists across interval refresh/tab switches ----
        try:
            if selected_times_pt:
                # Remove any existing highlight trace(s)
                indices_to_remove = []
                if "data" in fig and isinstance(fig["data"], list):
                    for i, tr in enumerate(fig["data"]):
                        if tr.get("name", "").startswith("Selected slices"):
                            indices_to_remove.append(i)
                for i in sorted(indices_to_remove, reverse=True):
                    del p.data[i]

                # Prefer already-fetched live bars for the session day; otherwise refetch the day bars
                df_sel = None
                if isinstance(df_live, pd.DataFrame) and (not df_live.empty):
                    df_sel = df_live
                else:
                    # Recompute day window for the effective session date in PT
                    day_start_pt = dt.datetime.combine(session_date, dt.time(0, 0), tzinfo=pt_tz)
                    day_end_pt = day_start_pt + dt.timedelta(days=1)
                    df_sel = _fetch_bars_pt(day_start_pt, day_end_pt, interval, pt_tz)

                if isinstance(df_sel, pd.DataFrame) and (not df_sel.empty) and 'time_hhmm_pt' in df_sel.columns:
                    mask_sel = df_sel['time_hhmm_pt'].isin(selected_times_pt if isinstance(selected_times_pt, list) else [selected_times_pt])
                    df_h = df_sel.loc[mask_sel]
                    if not df_h.empty:
                        p.data.append(
                            dict(
                                type="candlestick",
                                x=df_h['datetime_pt'],
                                open=df_h['open'],
                                high=df_h['high'],
                                low=df_h['low'],
                                close=df_h['close'],
                                name="Selected slices",
                                increasing=dict(line=dict(color=HIGHLIGHT_COLOR, width=2.0), fillcolor=HIGHLIGHT_COLOR),
                                decreasing=dict(line=dict(color=HIGHLIGHT_COLOR, width=2.0), fillcolor=HIGHLIGHT_COLOR),
                                showlegend=False,
                                yaxis='y2',
                                hovertemplate='<extra></extra>',
                            )
                        )
                        did_anything = True
        except Exception:
            # Never break live refresh due to highlight issues
            pass

        if not did_anything:
            raise PreventUpdate

        return p

    # -------------------------------------------------------------------------
    # Step 4: Wire Indicators sidebar -> Store (enabled + settings) and behavior
    # -------------------------------------------------------------------------

    @app.callback(
        Output("ib-indicator-enabled", "options"),
        Output("ib-settings-indicator", "options"),
        Input("ib-indicator-state", "data"),
        prevent_initial_call=False,
    )
    def ib_populate_indicator_options(state):
        """Populate sidebar indicator options from the plugin registry."""
        opts = ib_indicator_options()

        # Keep the enabled checklist stable. For sidebar settings, Aggressor Flow is now edited
        # from the React panel gear instead of the left settings form.
        settings_opts = [o for o in opts if isinstance(o, dict) and o.get("value") != "aggressor_flow"]
        return opts, settings_opts

    @app.callback(
        Output("ib-indicator-state", "data", allow_duplicate=True),
        Input("ib-indicator-enabled", "options"),
        State("ib-indicator-state", "data"),
        State("ib-shared-xrange", "data"),
        prevent_initial_call="initial_duplicate",
    )
    def ib_migrate_default_enabled(options, state, shared_xrange):
        """One-time migration: keep legacy behavior by default-enabling GEX overlay when first introduced."""
        state = state if isinstance(state, dict) else {}
        migrations = state.get("migrations") if isinstance(state.get("migrations"), dict) else {}
        if migrations.get("default_enable_gex_overlay") is True:
            return no_update

        has_gex = any(isinstance(o, dict) and o.get("value") == "gex_overlay" for o in (options or []))
        if not has_gex:
            return no_update

        enabled = state.get("enabled") or []
        if not isinstance(enabled, list):
            enabled = [enabled]
        if "gex_overlay" not in enabled:
            enabled = list(enabled) + ["gex_overlay"]

        migrations = dict(migrations)
        migrations["default_enable_gex_overlay"] = True
        cfg_all = state.get("cfg") or {}
        if not isinstance(cfg_all, dict):
            cfg_all = {}
        extras = {k: v for k, v in state.items() if k not in ("enabled", "cfg", "migrations")}
        return {"enabled": enabled, "cfg": cfg_all, "migrations": migrations, **extras}

    @app.callback(
        Output("ib-indicator-enabled", "value"),
        Output("ib-settings-indicator", "value"),
        Input("ib-indicator-state", "data"),
        State("ib-settings-indicator", "value"),
        State("ib-indicator-enabled", "value"),
        prevent_initial_call=False,
    )
    def ib_sync_indicator_sidebar(state, current_settings_selection, current_enabled_value):
        """Keep the sidebar controls in sync with the persisted indicator state."""
        enabled = []
        if isinstance(state, dict):
            enabled = state.get("enabled") or []
        if enabled is None:
            enabled = []
        elif not isinstance(enabled, list):
            enabled = [enabled]

        # Filter to known plugins (in case localStorage contains old ids)
        enabled = [pid for pid in enabled if pid in IB_PLUGIN_MAP]

        settings_enabled = [pid for pid in enabled if pid != "aggressor_flow"]

        sel = current_settings_selection
        if sel not in settings_enabled:
            sel = settings_enabled[0] if settings_enabled else None

        # If only configs changed (not enabled list / selection), don't push updates.
        cur_enabled = current_enabled_value
        if cur_enabled is None:
            cur_enabled = []
        elif not isinstance(cur_enabled, list):
            cur_enabled = [cur_enabled]

        if cur_enabled == enabled and current_settings_selection == sel:
            return no_update, no_update

        return enabled, sel

    @app.callback(
        Output("ib-indicator-state", "data", allow_duplicate=True),
        Input("ib-indicator-enabled", "value"),
        State("ib-indicator-state", "data"),
        State("ib-shared-xrange", "data"),
        prevent_initial_call=True,
    )
    def ib_update_enabled_indicators(enabled_value, state, shared_xrange):
        """Persist the enabled indicators list (preserve configs even if disabled)."""
        state = state if isinstance(state, dict) else {}
        cfg_all = state.get("cfg") or {}

        if enabled_value is None:
            enabled = []
        elif isinstance(enabled_value, list):
            enabled = enabled_value
        else:
            enabled = [enabled_value]

        extras = {k: v for k, v in (state.items() if isinstance(state, dict) else []) if k not in ("enabled", "cfg")}
        return {"enabled": enabled, "cfg": cfg_all, **extras}

    @app.callback(
        Output("ib-settings-form", "children"),
        Input("ib-settings-indicator", "value"),
        State("ib-indicator-state", "data"),
        State("ib-shared-xrange", "data"),
        prevent_initial_call=False,
    )
    def ib_render_settings_form(selected_indicator, state, shared_xrange):
        """Render settings controls for the selected indicator (schema-driven)."""
        if not selected_indicator or selected_indicator not in IB_PLUGIN_MAP:
            return html.Div(
                "Aggressor Flow settings now live in the panel gear inside React Preview. Select another indicator to edit sidebar settings.",
                style={"color": "#9ca3af", "fontSize": "12px", "marginTop": "6px"},
            )

        plugin = IB_PLUGIN_MAP[selected_indicator]

        if selected_indicator == "aggressor_flow":
            return html.Div(
                "Aggressor Flow settings now live in the panel gear inside React Preview.",
                style={"color": "#9ca3af", "fontSize": "12px", "lineHeight": "1.5"},
            )

        if selected_indicator == "gex_overlay":
            return html.Div(
                "GEX overlay settings now live in the price chart gear inside React Preview.",
                style={"color": "#9ca3af", "fontSize": "12px", "lineHeight": "1.5"},
            )

        state = state if isinstance(state, dict) else {}
        cfg_all = state.get("cfg") or {}
        persisted = cfg_all.get(selected_indicator) or {}

        # Merge defaults -> persisted
        cfg = dict(getattr(plugin, "default_config", lambda: {})() or {})
        if isinstance(persisted, dict):
            cfg.update(persisted)
        schema = dict(getattr(plugin, "schema", lambda: {})() or {})

        # Back-compat: keep your existing flow controls, even if the plugin schema is minimal.
        if selected_indicator == "aggressor_flow":
            schema.setdefault("ema_len", {"type": "int", "min": 1, "max": 5000, "step": 1, "label": "EMA length"})
            schema.setdefault(
                "resample",
                {"type": "select", "options": ["1s", "5s", "15s", "1m"], "label": "Resample"},
            )
            schema.setdefault("session", {"type": "select", "options": ["RTH", "FULL"], "label": "Session"},
            )
            schema.setdefault(
                "hist_alpha",
                {"type": "float", "min": 0.05, "max": 1.0, "step": 0.05, "label": "Histogram opacity"},
            )
            schema.setdefault("panel_height", {"type": "int", "min": 140, "max": 520, "step": 10, "label": "Panel height (px)"})

            # Fill any missing defaults from env vars (keeps old behavior)
            cfg.setdefault("ema_len", int(os.getenv("IRONBEAM_FLOW_EMA_LEN", "840")))
            cfg.setdefault("resample", str(os.getenv("IRONBEAM_FLOW_RESAMPLE", "1s")))
            cfg.setdefault("session", str(os.getenv("IRONBEAM_FLOW_SESSION", "RTH")))
            cfg.setdefault("hist_alpha", float(os.getenv("IRONBEAM_FLOW_HIST_ALPHA", "0.30")))
            cfg.setdefault("panel_height", int(os.getenv("IRONBEAM_FLOW_PANEL_HEIGHT", "260")))

        label_style = {"color": "#e5e7eb", "fontSize": "12px", "marginBottom": "4px", "marginTop": "8px"}
        input_style = {
            "width": "100%",
            "backgroundColor": "#0b1220",
            "color": "white",
            "border": "1px solid #1f2937",
            "borderRadius": "8px",
            "padding": "6px",
        }

        def _control(field: str):
            meta = schema.get(field) or {}
            ftype = meta.get("type")
            label = meta.get("label", field)

            # ID mapping (keep old ids for Aggressor Flow so Step 4 callbacks keep working)
            if selected_indicator == "aggressor_flow":
                id_map = {
                    "ema_len": "ib-flow-ema-len",
                    "resample": "ib-flow-resample",
                    "session": "ib-flow-session",
                    "hist_alpha": "ib-flow-hist-alpha",
                    "panel_height": "ib-flow-panel-height",
                }
                cid = id_map.get(field, f"ib-flow-{field}")
            else:
                cid = f"ib-{selected_indicator}-{field}"

            value = cfg.get(field)

            if ftype in ("int", "float", "number"):
                # Special: Aggressor Flow panel height as a slider
                if selected_indicator == "aggressor_flow" and field == "panel_height":
                    vmin = meta.get("min", 140)
                    vmax = meta.get("max", 520)
                    vstep = meta.get("step", 10)
                    # Simple marks so it stays readable
                    marks = {int(vmin): str(int(vmin)), int((vmin + vmax) // 2): str(int((vmin + vmax) // 2)), int(vmax): str(int(vmax))}
                    return html.Div(
                        [
                            html.Div(label, style=label_style),
                            dcc.Slider(
                                id=cid,
                                min=vmin,
                                max=vmax,
                                step=vstep,
                                value=value,
                                updatemode="mouseup",
                                marks=marks,
                            ),
                        ]
                    )

                step = meta.get("step", 1 if ftype == "int" else 0.1)
                return html.Div(
                    [
                        html.Div(label, style=label_style),
                        dcc.Input(
                            id=cid,
                            type="number",
                            min=meta.get("min"),
                            max=meta.get("max"),
                            step=step,
                            value=value,
                            debounce=True,
                            style=input_style,
                        ),
                    ]
                )

            if ftype == "select":
                return html.Div(
                    [
                        html.Div(label, style=label_style),
                        dcc.Dropdown(
                            id=cid,
                            options=[{"label": str(o), "value": o} for o in (meta.get("options") or [])],
                            value=value,
                            clearable=False,
                            style={
                                "backgroundColor": "#0b1220",
                                "borderRadius": "8px",
                            },
                        ),
                    ]
                )

            # default: text
            return html.Div(
                [
                    html.Div(label, style=label_style),
                    dcc.Input(
                        id=cid,
                        type="text",
                        value="" if value is None else str(value),
                        style=input_style,
                    ),
                ]
            )

        # For now, only Aggressor Flow has settings wired end-to-end.
        if selected_indicator != "aggressor_flow":
            return html.Div(
                "No settings UI is wired for this indicator yet.",
                style={"color": "#9ca3af", "fontSize": "12px", "marginTop": "6px"},
            )

        # Ordered fields for the current flow indicator
        fields = ["ema_len", "resample", "session", "hist_alpha", "panel_height"]
        return html.Div([_control(f) for f in fields])

    @app.callback(
        Output("ib-indicator-state", "data", allow_duplicate=True),
        Input("ib-flow-ema-len", "value"),
        Input("ib-flow-resample", "value"),
        Input("ib-flow-session", "value"),
        Input("ib-flow-hist-alpha", "value"),
        Input("ib-flow-panel-height", "value"),
        State("ib-indicator-state", "data"),
        State("ib-shared-xrange", "data"),
        prevent_initial_call=True,
    )
    def ib_persist_flow_settings(ema_len, resample_mode, session_mode, hist_alpha, panel_height, state, shared_xrange):
        """Persist Aggressor Flow settings into ib-indicator-state.cfg.aggressor_flow."""
        state = state if isinstance(state, dict) else {}
        enabled = state.get("enabled") or []
        if not isinstance(enabled, list):
            enabled = [enabled] if enabled else []

        cfg_all = state.get("cfg") or {}
        flow_cfg = cfg_all.get("aggressor_flow") or {}

        if ema_len is not None:
            try:
                flow_cfg["ema_len"] = int(ema_len)
            except Exception:
                pass
        if resample_mode:
            flow_cfg["resample"] = str(resample_mode).lower()
        if session_mode:
            flow_cfg["session"] = str(session_mode).upper()
        if hist_alpha is not None:
            try:
                flow_cfg["hist_alpha"] = float(hist_alpha)
            except Exception:
                pass

        if panel_height is not None:
            try:
                flow_cfg["panel_height"] = int(panel_height)
            except Exception:
                pass

        cfg_all["aggressor_flow"] = flow_cfg
        extras = {k: v for k, v in (state.items() if isinstance(state, dict) else []) if k not in ("enabled", "cfg")}
        return {"enabled": enabled, "cfg": cfg_all, **extras}

    @app.callback(
        Output("ib-indicator-state", "data", allow_duplicate=True),
        Input("ib-gex-min-abs-b", "value"),
        State("ib-indicator-state", "data"),
        State("ib-shared-xrange", "data"),
        prevent_initial_call=True,
    )
    def ib_persist_gex_settings(min_abs_b, state, shared_xrange):
        """Persist GEX overlay settings into ib-indicator-state.cfg[gex_overlay]."""
        state = state if isinstance(state, dict) else {}
        enabled = state.get("enabled") or []
        if not isinstance(enabled, list):
            enabled = [enabled]
        cfg_all = state.get("cfg") or {}
        if not isinstance(cfg_all, dict):
            cfg_all = {}
        gex_cfg = cfg_all.get("gex_overlay")
        if not isinstance(gex_cfg, dict):
            gex_cfg = {}
        if min_abs_b is not None:
            try:
                gex_cfg["min_abs_b"] = float(min_abs_b)
            except Exception:
                pass
        cfg_all = dict(cfg_all)
        cfg_all["gex_overlay"] = gex_cfg
        extras = {k: v for k, v in state.items() if k not in ("enabled", "cfg")}
        return {"enabled": enabled, "cfg": cfg_all, **extras}

    @app.callback(
        Output("ib-indicator-panels", "children"),
        Input("ib-indicator-state", "data"),
        prevent_initial_call=False,
    )
    def ib_render_indicator_panels(state):
        enabled = []
        if isinstance(state, dict):
            enabled = state.get('enabled') or []
        if not isinstance(enabled, list):
            enabled = [enabled] if enabled else []

        # De-duplicate while preserving order
        seen = set()
        enabled = [x for x in enabled if x and (x not in seen and not seen.add(x))]

        children = []
        for pid in enabled:
            plugin = IB_PLUGIN_MAP.get(pid)
            if not plugin:
                continue
            kind = getattr(plugin, 'kind', 'panel')
            if kind != 'panel':
                continue

            # Panel height (per-indicator config; defaults to 260px)
            height_px = 260
            try:
                cfg_all = (state or {}).get("cfg") or {}
                cfg_pid = cfg_all.get(pid) or {}
                # Aggressor Flow uses panel_height setting
                if isinstance(cfg_pid, dict) and cfg_pid.get("panel_height") is not None:
                    height_px = int(cfg_pid.get("panel_height"))
            except Exception:
                height_px = 260

            children.append(
                dcc.Graph(
                    id={'type': 'ib-indicator-panel', 'id': pid},
                    style={'height': f'{height_px}px', 'marginTop': '10px'},
                    config={'displayModeBar': True, 'scrollZoom': True, 'displaylogo': False, 'responsive': True},
                )
            )
        return children

    @app.callback(
        Output({'type': 'ib-indicator-panel', 'id': MATCH}, 'figure'),
        [
            Input('trade-date', 'date'),
            Input('smile-time-input', 'value'),  # heartbeat (value not used)
            Input('ib-indicator-state', 'data'),
            Input('ib-shared-xrange', 'data'),
            Input('ironbeam-interval', 'n_intervals'),
            State({'type': 'ib-indicator-panel', 'id': MATCH}, 'id'),
            State({'type': 'ib-indicator-panel', 'id': MATCH}, 'figure'),
        ],
        prevent_initial_call=False,
    )
    def ib_update_indicator_panel(trade_date, _heartbeat, indicator_state, shared_xrange, n_intervals, panel_id, prev_panel_fig):
        pid = panel_id.get('id') if isinstance(panel_id, dict) else None
        if not pid:
            raise PreventUpdate

        # If disabled (race), just show an empty figure
        enabled = []
        if isinstance(indicator_state, dict):
            enabled = indicator_state.get('enabled') or []
        if not isinstance(enabled, list):
            enabled = [enabled] if enabled else []
        if pid not in enabled:
            return go.Figure()

        triggered_ids = [t['prop_id'] for t in ctx.triggered]

        # If triggered by interval, check if we are on the live date.
        if 'ironbeam-interval.n_intervals' in triggered_ids:
            try:
                pt_tz = ZoneInfo("America/Los_Angeles")
                selected_date = dt.datetime.strptime(trade_date, "%Y-%m-%d").date()
                curr_session = _current_session_trade_date(pt_tz)
                eff_date, _ = _effective_trade_date(selected_date, pt_tz)

                if eff_date != curr_session:
                    return no_update
            except Exception:
                return no_update

        # Determine if we can use Patch optimization (only x-range changed)
        # We check if 'trade-date' or 'ib-indicator-state' triggered the update.
        is_config_change = any('trade-date' in t or 'ib-indicator-state' in t for t in triggered_ids)

        if not is_config_change and 'ib-shared-xrange.data' in triggered_ids and 'ironbeam-interval.n_intervals' not in triggered_ids:
            if pid == 'aggressor_flow':
                if isinstance(shared_xrange, dict):
                    x0 = shared_xrange.get("x0")
                    x1 = shared_xrange.get("x1")
                    if x0 and x1:
                        p = Patch()
                        p.layout.xaxis.range = [x0, x1]
                        p.layout.xaxis.autorange = False
                        return p
                return no_update

        if pid == 'aggressor_flow':
            return build_aggressor_flow_figure(trade_date, indicator_state, shared_xrange)

        # Unknown panel plugin: return empty
        return go.Figure()

    @app.callback(
        Output("ironbeam-chart", "style"),
        Input("ib-indicator-state", "data"),
        State("ironbeam-chart", "style"),
        prevent_initial_call=False,
    )
    def ib_resize_price_when_flow_hidden(state, current_style):
        """Grow the price chart when the Aggressor Flow panel is disabled."""
        base_style = current_style if isinstance(current_style, dict) else {}

        enabled = []
        if isinstance(state, dict):
            enabled = state.get("enabled") or []
        if not isinstance(enabled, list):
            enabled = [enabled] if enabled else []

        s = dict(base_style)

        # The original layout reserved ~panel_height px for the flow chart + ~10px margin.
        # When flow is hidden, reclaim that vertical space.
        base_reserved = 250  # header/tabs/padding, etc.
        flow_height = 260
        try:
            cfg_all = (state or {}).get("cfg") or {}
            flow_cfg = cfg_all.get("aggressor_flow") or {}
            if isinstance(flow_cfg, dict) and flow_cfg.get("panel_height") is not None:
                flow_height = int(flow_cfg.get("panel_height"))
        except Exception:
            flow_height = 260

        if "aggressor_flow" in enabled:
            s["height"] = f"calc(100vh - {base_reserved + flow_height + 10}px)"
        else:
            s["height"] = f"calc(100vh - {base_reserved}px)"
        return s

    @app.callback(
        Output("ib-shared-xrange", "data"),
        Input("ironbeam-chart", "relayoutData"),
        State("ib-shared-xrange", "data"),
        prevent_initial_call=True,
    )
    def capture_shared_xrange(relayout, current):
        """
        Stores:
          - x0/x1 from relayout (zoom/pan)

        Note: Hover crosshair sync is now handled clientside to avoid latency.
        """
        data = dict(current or {})

        pt_tz = ZoneInfo("America/Los_Angeles")

        def _to_pt_iso(v):
            """Normalize x values to PT ISO strings (with offset)."""
            if v is None:
                return None
            try:
                ts = pd.to_datetime(v, errors="coerce")
            except Exception:
                return None
            if ts is pd.NaT:
                return None
            ts = pd.Timestamp(ts)
            try:
                if ts.tzinfo is None:
                    ts = ts.tz_localize(pt_tz)
                else:
                    ts = ts.tz_convert(pt_tz)
            except Exception:
                return None
            return ts.isoformat()

        # We only have relayoutData now
        if not isinstance(relayout, dict):
            raise PreventUpdate

        changed = False

        # Reset (double click)
        if relayout.get("xaxis.autorange") or relayout.get("xaxis.autorange") is True:
            if "x0" in data or "x1" in data:
                data.pop("x0", None)
                data.pop("x1", None)
                changed = True
        else:
            # Most common keys
            x0 = relayout.get("xaxis.range[0]")
            x1 = relayout.get("xaxis.range[1]")
            rng = relayout.get("xaxis.range")

            # Some plotly updates use list-style range
            if (x0 is None or x1 is None) and isinstance(rng, (list, tuple)) and len(rng) == 2:
                x0, x1 = rng[0], rng[1]

            # Fallback: support xaxis2/xaxis3 if present
            if x0 is None or x1 is None:
                for ax in ("xaxis", "xaxis2", "xaxis3"):
                    x0 = relayout.get(f"{ax}.range[0]")
                    x1 = relayout.get(f"{ax}.range[1]")
                    rng = relayout.get(f"{ax}.range")
                    if (x0 is None or x1 is None) and isinstance(rng, (list, tuple)) and len(rng) == 2:
                        x0, x1 = rng[0], rng[1]
                    if x0 is not None and x1 is not None:
                        break

            x0n = _to_pt_iso(x0)
            x1n = _to_pt_iso(x1)
            if x0n is not None and x1n is not None:
                if data.get("x0") != x0n or data.get("x1") != x1n:
                    data["x0"] = x0n
                    data["x1"] = x1n
                    changed = True

        if not changed:
            raise PreventUpdate

        return data or None

    # ---- Click a price bar to set global Time Slices (PT) ----
    # This restores the old behavior: clicking a candle sets the "Time Slices (PT)" dropdown
    # (id="smile-time-input") and the selected bar is highlighted in red by the main chart builder.
    @app.callback(
        Output("smile-time-input", "value", allow_duplicate=True),
        Input("ironbeam-chart", "clickData"),
        State("smile-time-input", "value"),
        State("ironbeam-bar-interval", "value"),
        prevent_initial_call=True,
    )
    def ib_click_bar_sets_time_slices(click_data, current_value, bar_interval):
        if not isinstance(click_data, dict):
            raise PreventUpdate
        pts = click_data.get("points") or []
        if not pts:
            raise PreventUpdate

        x = pts[0].get("x")
        if x is None:
            raise PreventUpdate

        # Parse x (can be ISO string / datetime)
        try:
            ts = pd.to_datetime(x, utc=False)
        except Exception:
            raise PreventUpdate

        # Convert to PT and format to match dropdown options (HH:MM)
        try:
            if getattr(ts, "tzinfo", None) is None:
                ts = ts.tz_localize(ZoneInfo("America/Los_Angeles"))
            else:
                ts = ts.tz_convert(ZoneInfo("America/Los_Angeles"))
        except Exception:
            # if pandas returns python datetime already
            pass

        # Adjust for 1-minute bars: select the next minute (end of bar)
        # This fixes the "selecting bars a minute behind" issue.
        if (bar_interval or "1min") == "1min":
            ts = ts + pd.Timedelta(minutes=1)

        try:
            hhmm = ts.strftime("%H:%M")
        except Exception:
            try:
                hhmm = pd.Timestamp(ts).strftime("%H:%M")
            except Exception:
                raise PreventUpdate
        # Build a MULTI-select value list (append, keep unique)
        if current_value is None:
            existing = []
        elif isinstance(current_value, list):
            existing = current_value
        else:
            existing = [current_value]

        if hhmm in existing:
            # Toggle off: remove the clicked time
            new_val = [t for t in existing if t != hhmm]
        else:
            # Toggle on: append
            new_val = existing + [hhmm]

        # Avoid unnecessary updates (prevents UI jitter)
        if isinstance(existing, list) and existing == new_val:
            raise PreventUpdate
        return new_val

    # -------------------------
    # UI: Sidebar hidden in React-preview-first layout
    # -------------------------
    @app.callback(
        Output("ib-ui-state", "data"),
        Input("ib-sidebar-toggle", "n_clicks"),
        State("ib-ui-state", "data"),
        prevent_initial_call=True,
    )
    def ib_toggle_sidebar(n_clicks, ui_state):
        raise PreventUpdate

    @app.callback(
        Output("ib-indicator-sidebar", "style"),
        Output("ib-sidebar-content", "style"),
        Output("ib-sidebar-toggle", "children"),
        Output("ib-ironbeam-row", "style"),
        Output("ib-sidebar-toggle", "style"),
        Input("ib-ui-state", "data"),
    )
    def ib_apply_sidebar_styles(ui_state):
        sidebar_style = {
            "width": "0px",
            "minWidth": "0px",
            "padding": "0px",
            "border": "0px",
            "overflow": "hidden",
            "display": "none",
        }
        content_style = {"display": "none"}
        btn = "«"
        row_style = {"display": "flex", "alignItems": "stretch", "gap": "0px", "width": "100%", "minWidth": 0, "flex": "1 1 auto", "minHeight": 0}
        toggle_style = {"display": "none"}
        return sidebar_style, content_style, btn, row_style, toggle_style

    # -------------------------
    # UI: Chart mode toggle (Classic vs React Preview)
    # NOTE: do not round-trip toggle -> store -> toggle, because Dash treats that as a
    # circular dependency. For step 1 we let the radio value drive visibility directly.
    # -------------------------

    @app.callback(
        Output("ib-react-preview-frame", "src"),
        Input("trade-date", "date"),
        Input("smile-time-input", "value"),
        Input("ironbeam-bar-interval", "value"),
        Input("ib-chart-mode-toggle", "value"),
        Input("ib-indicator-state", "data"),
    )
    def ib_update_react_preview_src(trade_date, selected_times_pt, bar_interval, chart_mode, indicator_state):
        base = os.getenv("IRONBEAM_REACT_PREVIEW_URL", "/react-preview").rstrip("/")
        td = trade_date or dt.date.today().isoformat()
        interval = (bar_interval or "1min").strip()
        if interval not in {"1min", "5min"}:
            interval = "1min"

        enabled = []
        gex_min_abs_b = None
        if isinstance(indicator_state, dict):
            enabled = indicator_state.get("enabled") or []
            if not isinstance(enabled, list):
                enabled = [enabled] if enabled else []

            cfg_all = indicator_state.get("cfg") if isinstance(indicator_state.get("cfg"), dict) else {}
            gex_cfg = cfg_all.get("gex_overlay") if isinstance(cfg_all.get("gex_overlay"), dict) else {}
            raw_min_abs_b = gex_cfg.get("min_abs_b")
            try:
                if raw_min_abs_b not in (None, ""):
                    gex_min_abs_b = float(raw_min_abs_b)
            except Exception:
                gex_min_abs_b = None

        gex_enabled = "gex_overlay" in enabled if enabled else True

        # normalize selected time slices
        if selected_times_pt is None:
            raw_times = []
        elif isinstance(selected_times_pt, list):
            raw_times = selected_times_pt
        else:
            raw_times = [selected_times_pt]

        cleaned_times = []
        seen = set()
        for item in raw_times:
            s = str(item or "").strip()
            if not s:
                continue
            if not re.match(r"^\d{2}:\d{2}$", s):
                continue
            if s in seen:
                continue
            seen.add(s)
            cleaned_times.append(s)

        params = [
            f"trade_date={td}",
            f"interval={interval}",
            f"gex_enabled={1 if gex_enabled else 0}",
            f"days_either_side={_react_days_either_side_for_interval(interval)}",
        ]

        if gex_min_abs_b is not None:
            params.append(f"gex_min_abs_b={gex_min_abs_b}")

        if cleaned_times:
            params.append(f"selected_times={','.join(cleaned_times)}")

        return f"{base}?{'&'.join(params)}"

    @app.callback(
        Output("smile-time-input", "value", allow_duplicate=True),
        Input("ib-react-timeslice-bridge", "value"),
        State("smile-time-input", "value"),
        prevent_initial_call=True,
    )
    def ib_apply_react_timeslice_bridge(raw_value, current_value):
        if raw_value in (None, ""):
            raise PreventUpdate

        try:
            payload = json.loads(raw_value)
        except Exception:
            raise PreventUpdate

        if isinstance(payload, dict):
            times = payload.get("times") or []
        else:
            times = payload

        if not isinstance(times, list):
            raise PreventUpdate

        cleaned = []
        seen = set()
        for item in times:
            s = str(item or "").strip()
            if not s:
                continue
            if not re.match(r"^\d{2}:\d{2}$", s):
                continue
            if s not in seen:
                cleaned.append(s)
                seen.add(s)

        current = current_value or []
        if not isinstance(current, list):
            current = [current] if current else []

        if cleaned == current:
            raise PreventUpdate

        return cleaned

    @app.callback(
        Output("ib-react-timeslice-parent", "value"),
        Input("smile-time-input", "value"),
    )
    def ib_publish_parent_timeslices_to_react(times_value):
        times = times_value or []
        if not isinstance(times, list):
            times = [times] if times else []

        cleaned = []
        seen = set()
        for item in times:
            s = str(item or "").strip()
            if not s:
                continue
            if not re.match(r"^\d{2}:\d{2}$", s):
                continue
            if s not in seen:
                cleaned.append(s)
                seen.add(s)

        return json.dumps({"times": cleaned})

    @app.callback(
        Output("ib-classic-chart-wrap", "style"),
        Output("ib-react-preview-wrap", "style"),
        Input("ib-chart-mode-toggle", "value"),
    )
    def ib_apply_chart_mode(chart_mode):
        mode = (chart_mode or "react_preview").strip()
        if mode == "react_preview":
            return (
                {"display": "none", "width": "100%", "minWidth": 0, "minHeight": 0},
                {
                    "display": "flex",
                    "flexDirection": "column",
                    "flex": "1 1 auto",
                    "width": "100%",
                    "minWidth": 0,
                    "minHeight": 0,
                    "height": "100%",
                },
            )

        return (
            {
                "display": "flex",
                "flexDirection": "column",
                "flex": "1 1 auto",
                "width": "100%",
                "minWidth": 0,
                "minHeight": 0,
                "height": "100%",
            },
            {"display": "none", "width": "100%", "minWidth": 0, "minHeight": 0},
        )


    @app.callback(
        Output("ib-chart-mode-toggle", "value", allow_duplicate=True),
        Input("ib-react-preview-frame", "id"),
        prevent_initial_call="initial_duplicate",
    )
    def ib_default_chart_mode(_frame_id):
        return "react_preview"

    @app.callback(
        Output("ironbeam-bar-interval", "style"),
        Output("ib-chart-mode-toggle", "style"),
        Input("ib-chart-mode-toggle", "value"),
    )
    def ib_toggle_external_chart_controls(chart_mode):
        mode = (chart_mode or "react_preview").strip()
        if mode == "react_preview":
            hidden = {"display": "none"}
            return hidden, hidden
        return {}, {}
