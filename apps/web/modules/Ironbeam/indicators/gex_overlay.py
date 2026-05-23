# apps/web/modules/Ironbeam/indicators/gex_overlay.py
#
# Step 14a:
# - Implement the *actual* GEX overlay trace-building logic inside the plugin.
# - This file is SAFE to drop in now: Step 13 still draws GEX from callbacks.py.
# - Step 14b will switch the main price figure to call this plugin for overlays.

from __future__ import annotations

import os
import datetime as dt
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Callable, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pc

from .base import IndicatorPlugin, IndicatorSpec

# -------- Defaults (match callbacks.py) --------
GEX_LEVEL_PADDING = float(os.getenv("GEX_LEVEL_PADDING", "150"))

# Threshold is absolute $ gamma; plugin config uses "billions"
GEX_ABS_THRESHOLD_DEFAULT = float(os.getenv("GEX_ABS_THRESHOLD", "1e10"))

GEX_COLOR_ABS_MAX = float(os.getenv("GEX_COLOR_ABS_MAX", "0"))
GEX_COLOR_PERCENTILE = float(os.getenv("GEX_COLOR_PERCENTILE", "95"))

GEX_MAX_LEVELS_PER_DAY = int(os.getenv("GEX_MAX_LEVELS_PER_DAY", "80"))
GEX_MIN_LEVEL_SPACING = float(os.getenv("GEX_MIN_LEVEL_SPACING", "5"))

GEX_LEVEL_LINE_WIDTH = float(os.getenv("GEX_LEVEL_LINE_WIDTH", "2.0"))
GEX_LEVEL_LINE_WIDTH_SCALE = float(os.getenv("GEX_LEVEL_LINE_WIDTH_SCALE", "1.5"))
GEX_LEVEL_LINE_WIDTH_MAX = float(os.getenv("GEX_LEVEL_LINE_WIDTH_MAX", "4.0"))
GEX_LEVEL_LINE_OPACITY = float(os.getenv("GEX_LEVEL_LINE_OPACITY", "0.90"))

# High-contrast diverging colorscale for dark bg (same as callbacks.py)
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


def _compute_color_span(net_g: np.ndarray) -> tuple[float, float]:
    """Return (cmin, cmax) for coloring net_gamma."""
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


def _color_for_net_gex(net_val: float, cmin: float, cmax: float) -> str:
    if not np.isfinite(net_val):
        return pc.sample_colorscale(GEX_HEATMAP_COLORSCALE, 0.5)[0]
    span = float(cmax - cmin)
    t = 0.5 if span <= 0 else (np.clip(net_val, cmin, cmax) - cmin) / span
    return pc.sample_colorscale(GEX_HEATMAP_COLORSCALE, float(t))[0]


def _select_levels(df_gex: pd.DataFrame, band_min: float, band_max: float, threshold_abs: float) -> pd.DataFrame:
    """Match the level selection logic in callbacks.py."""
    if df_gex is None or df_gex.empty:
        return pd.DataFrame(columns=["level", "call_gamma", "put_gamma", "net_gamma"])

    df = df_gex[(df_gex["level"] >= band_min) & (df_gex["level"] <= band_max)].copy()
    if df.empty:
        return df

    # magnitude = |call| + |put|
    df["mag"] = np.abs(df["call_gamma"].to_numpy(dtype=float)) + np.abs(df["put_gamma"].to_numpy(dtype=float))

    if threshold_abs and threshold_abs > 0:
        df = df[df["mag"] >= float(threshold_abs)].copy()
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


def build_gex_overlay_traces(
    df_gex: pd.DataFrame,
    x_start: dt.datetime,
    x_end: dt.datetime,
    band_min: float,
    band_max: float,
    threshold_abs: float,
    name_prefix: str = "GEX",
) -> list[go.Scattergl]:
    """Return plotly traces for the GEX overlay."""
    df_day = _select_levels(df_gex, band_min, band_max, threshold_abs)
    if df_day.empty:
        return []

    net_g = df_day["net_gamma"].to_numpy(dtype=float)
    cmin, cmax = _compute_color_span(net_g)

    denom = float(max(abs(cmin), abs(cmax), 1.0))

    traces: list[go.Scattergl] = []
    for _, r in df_day.iterrows():
        lvl = float(r["level"])
        net_val = float(r["net_gamma"])
        color = _color_for_net_gex(net_val, cmin, cmax)

        norm = float(min(1.0, abs(net_val) / denom))
        line_width = float(min(GEX_LEVEL_LINE_WIDTH_MAX, GEX_LEVEL_LINE_WIDTH + norm * GEX_LEVEL_LINE_WIDTH_SCALE))
        line_opacity = float(min(1.0, max(0.12, GEX_LEVEL_LINE_OPACITY * (0.40 + 0.60 * norm))))

        traces.append(
            go.Scattergl(
                x=[x_start, x_end],
                y=[lvl, lvl],
                mode="lines",
                line=dict(color=color, width=line_width),
                opacity=line_opacity,
                name=name_prefix,
                showlegend=False,
                hoverinfo="skip",
            )
        )

    return traces


class GexOverlay(IndicatorPlugin):
    """GEX levels plotted as overlays on the main price chart."""

    id = "gex_overlay"
    name = "GEX Levels"
    kind = "overlay"

    def default_config(self) -> Dict[str, Any]:
        # Env var is absolute $ gamma (default 1e10 == 10B). Store config in "billions".
        env_abs = float(os.getenv("GEX_ABS_THRESHOLD", str(GEX_ABS_THRESHOLD_DEFAULT)))
        return {
            "min_abs_b": env_abs / 1e9,  # minimum |GEX| (B) to plot
        }

    def schema(self) -> Dict[str, Any]:
        return {
            "min_abs_b": {
                "type": "float",
                "min": 0.0,
                "max": 200.0,
                "step": 0.5,
                "label": "Min |GEX| (B)",
            },
        }

    def required_datasets(self) -> List[str]:
        return ["gex"]

    def build(self, ctx: Dict[str, Any], cfg: Dict[str, Any]) -> IndicatorSpec:
        """
        ctx requirements (Step 14b will provide these):
          - session_date: date or 'YYYY-MM-DD'
          - df_gex: DataFrame with columns [level, call_gamma, put_gamma, net_gamma]
            OR fetch_gex_grouped_by_level: callable(date)->DataFrame
          - band_min/band_max OR df_bars (so we can compute from low/high)
          - x_start/x_end (preferred) OR pt_tz (so we can create a day span)
        """
        # --- date / tz ---
        pt_tz = ctx.get("pt_tz") or ZoneInfo("America/Los_Angeles")
        session_date = ctx.get("session_date")
        if isinstance(session_date, str):
            session_date = dt.date.fromisoformat(session_date)
        if not isinstance(session_date, dt.date):
            # not enough context; return nothing (safe)
            return IndicatorSpec(kind="overlay", traces=[])

        # --- x span ---
        x_start = ctx.get("x_start")
        x_end = ctx.get("x_end")
        if not isinstance(x_start, dt.datetime) or not isinstance(x_end, dt.datetime):
            # default to full calendar day in PT (caller can override)
            x_start = dt.datetime.combine(session_date, dt.time(0, 0), tzinfo=pt_tz)
            x_end = dt.datetime.combine(session_date, dt.time(23, 59), tzinfo=pt_tz)

        # --- band selection ---
        band_min = ctx.get("band_min")
        band_max = ctx.get("band_max")
        if band_min is None or band_max is None:
            df_bars = ctx.get("df_bars")
            if isinstance(df_bars, pd.DataFrame) and (not df_bars.empty) and ("low" in df_bars.columns) and ("high" in df_bars.columns):
                low = float(df_bars["low"].min())
                high = float(df_bars["high"].max())
                band_min = low - GEX_LEVEL_PADDING
                band_max = high + GEX_LEVEL_PADDING
            else:
                # fallback: wide band around last price if provided
                last_px = float(ctx.get("last_price") or 0.0)
                band_min = last_px - 500.0
                band_max = last_px + 500.0

        # --- threshold ---
        min_abs_b = float((cfg or {}).get("min_abs_b") or (GEX_ABS_THRESHOLD_DEFAULT / 1e9))
        threshold_abs = float(max(0.0, min_abs_b)) * 1e9

        # --- df_gex ---
        df_gex = ctx.get("df_gex")
        if not isinstance(df_gex, pd.DataFrame):
            fetcher: Optional[Callable[[dt.date], pd.DataFrame]] = ctx.get("fetch_gex_grouped_by_level")
            if callable(fetcher):
                try:
                    df_gex = fetcher(session_date)
                except Exception:
                    df_gex = pd.DataFrame(columns=["level", "call_gamma", "put_gamma", "net_gamma"])
            else:
                df_gex = pd.DataFrame(columns=["level", "call_gamma", "put_gamma", "net_gamma"])

        traces = build_gex_overlay_traces(
            df_gex=df_gex,
            x_start=x_start,
            x_end=x_end,
            band_min=float(band_min),
            band_max=float(band_max),
            threshold_abs=float(threshold_abs),
            name_prefix=f"GEX {session_date.isoformat()}",
        )

        return IndicatorSpec(kind="overlay", traces=traces)
