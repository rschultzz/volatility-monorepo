from __future__ import annotations

import datetime as dt
import os

import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback
from sqlalchemy import create_engine, text
from sqlalchemy.engine.url import make_url

from .live_proxy import fetch_live_proxy_grouped_by_level

ZOOM_PCT = float(os.getenv("GEX_ZOOM_PCT", "0"))
PUT_COLOR = os.getenv("GEX_PUT_COLOR", "#E5E7EB")
CALL_COLOR = os.getenv("GEX_CALL_COLOR", "#60a5fa")
LIVE_PUT_COLOR = os.getenv("GEX_LIVE_PUT_COLOR", "#9CA3AF")
LIVE_CALL_COLOR = os.getenv("GEX_LIVE_CALL_COLOR", "#34D399")
VOLUME_PUT_COLOR = os.getenv("GEX_LIVE_VOLUME_PUT_COLOR", LIVE_PUT_COLOR)
VOLUME_CALL_COLOR = os.getenv("GEX_LIVE_VOLUME_CALL_COLOR", LIVE_CALL_COLOR)
COMBINED_NET_COLOR = os.getenv("GEX_COMBINED_NET_COLOR", "#F59E0B")
TICKER = os.getenv("GEX_TICKER", "SPX")

try:
    from shared.utils.data_io import get_engine  # type: ignore
except Exception:
    def get_engine():
        raw = (
            os.getenv("DATABASE_URL")
            or os.getenv("POSTGRES_URL")
            or os.getenv("POSTGRES_CONNECTION_STRING")
            or "postgresql://localhost/postgres"
        )
        url = make_url(raw)
        if url.get_backend_name() == "postgresql" and url.get_driver_name() in (None, "", "psycopg2"):
            url = url.set(drivername="postgresql+psycopg")
        return create_engine(url, pool_pre_ping=True, pool_recycle=300)


def _empty_base_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["level", "call_gamma", "put_gamma", "net_gamma"]).astype(
        {"level": "int64", "call_gamma": "float64", "put_gamma": "float64", "net_gamma": "float64"}
    )


def _empty_live_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "level",
            "call_live",
            "put_live",
            "net_live",
            "call_volume_live",
            "put_volume_live",
            "net_volume_live",
            "contracts_touched",
        ]
    ).astype(
        {
            "level": "int64",
            "call_live": "float64",
            "put_live": "float64",
            "net_live": "float64",
            "call_volume_live": "float64",
            "put_volume_live": "float64",
            "net_volume_live": "float64",
            "contracts_touched": "int64",
        }
    )


def _fetch_gex_grouped_by_level(trade_date: dt.date) -> pd.DataFrame:
    eng = get_engine()
    dialect = eng.dialect.name
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
            COALESCE(SUM(gex_call), 0) AS call_gamma,
            COALESCE(SUM(gex_put),  0) AS put_gamma
        FROM orats_oi_gamma
        WHERE {' AND '.join(where)}
        GROUP BY {level_expr}
        ORDER BY {level_expr}
    """

    with eng.connect() as con:
        df = pd.read_sql(text(sql), con, params=params)

    if df.empty:
        return _empty_base_df()

    df["put_gamma"] = -df["put_gamma"].abs()
    df["net_gamma"] = df["call_gamma"] + df["put_gamma"]
    df["level"] = df["level"].astype(int)
    return df


def _merge_base_and_live(base_df: pd.DataFrame, live_df: pd.DataFrame) -> pd.DataFrame:
    base = base_df.copy() if base_df is not None and not base_df.empty else _empty_base_df()
    live = live_df.copy() if live_df is not None and not live_df.empty else _empty_live_df()

    if base.empty and live.empty:
        return pd.DataFrame(
            columns=[
                "level", "call_gamma", "put_gamma", "net_gamma",
                "call_live", "put_live", "net_live",
                "call_volume_live", "put_volume_live", "net_volume_live",
                "call_combined", "put_combined", "net_combined",
            ]
        )

    merged = pd.merge(base, live, how="outer", on="level")
    for col in [
        "call_gamma", "put_gamma", "net_gamma",
        "call_live", "put_live", "net_live",
        "call_volume_live", "put_volume_live", "net_volume_live",
    ]:
        if col not in merged.columns:
            merged[col] = 0.0
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    merged["call_combined"] = merged["call_gamma"] + merged["call_live"]
    merged["put_combined"] = merged["put_gamma"] + merged["put_live"]
    merged["net_combined"] = merged["call_combined"] + merged["put_combined"]
    merged["level"] = pd.to_numeric(merged["level"], errors="coerce")
    merged = merged.dropna(subset=["level"]).sort_values("level").reset_index(drop=True)
    merged["level"] = merged["level"].astype(int)
    return merged


def _max_abs_for_mode(df: pd.DataFrame, mode: str) -> float:
    if df.empty:
        return 0.0
    cols_by_mode = {
        "gex": ["put_gamma", "call_gamma"],
        "volume": ["put_volume_live", "call_volume_live"],
        "live": ["put_live", "call_live"],
        "combined": ["put_gamma", "call_gamma", "put_live", "call_live", "put_combined", "call_combined", "net_combined"],
    }
    vals: list[float] = []
    for col in cols_by_mode.get(mode, cols_by_mode["gex"]):
        if col in df.columns:
            vals.append(float(df[col].abs().max() or 0.0))
    return max(vals) if vals else 0.0


def _build_gex_figure(df: pd.DataFrame, trade_date_str: str, *, mode: str, live_meta: dict[str, object] | None = None) -> go.Figure:
    live_meta = live_meta or {}
    mode = (mode or "gex").lower()

    if mode == "volume":
        title = f"Live Volume by Discounted Level — {trade_date_str}"
        xaxis_title = "Cumulative day volume by rounded discounted level"
    elif mode == "live":
        title = f"Live Volume/Gamma Proxy by Discounted Level — {trade_date_str}"
        xaxis_title = "Live proxy (cumulative day volume × latest gamma proxy)"
    elif mode == "combined":
        title = f"GEX + Live Proxy by Discounted Level — {trade_date_str}"
        xaxis_title = "Gamma / live proxy"
    else:
        title = f"GEX by Discounted Level — {trade_date_str}"
        xaxis_title = "Gamma (Σ by rounded discounted_level)"

    if TICKER:
        title += f" ({TICKER})"
    as_of_label = live_meta.get("as_of_label")
    if mode in {"volume", "live", "combined"} and as_of_label:
        title += f" — {as_of_label}"

    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title="Discounted Level (rounded)",
        margin=dict(l=90, r=30, t=84, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        barmode="overlay",
    )

    if df.empty:
        msg_map = {
            "gex": "No gamma data for the selected date",
            "volume": "No live volume data for the selected date",
            "live": "No live proxy data for the selected date",
            "combined": "No combined data for the selected date",
        }
        msg = msg_map.get(mode, "No data for the selected date")
        if live_meta.get("error"):
            msg = f"{msg}<br>{live_meta['error']}"
        fig.add_annotation(text=msg, showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5)
        fig.add_vline(x=0, line_width=1, line_dash="dot")
        return fig

    y = df["level"].astype(str)
    if mode == "gex":
        fig.add_bar(x=df["put_gamma"], y=y, orientation="h", name="Puts", marker=dict(color=PUT_COLOR, line=dict(width=0)), hovertemplate="Level=%{y}<br>Puts=%{x:.4g}<extra></extra>")
        fig.add_bar(x=df["call_gamma"], y=y, orientation="h", name="Calls", marker=dict(color=CALL_COLOR, line=dict(width=0)), hovertemplate="Level=%{y}<br>Calls=%{x:.4g}<extra></extra>")
    elif mode == "volume":
        fig.add_bar(x=df["put_volume_live"], y=y, orientation="h", name="Volume Puts", marker=dict(color=VOLUME_PUT_COLOR, line=dict(width=0)), opacity=0.9, hovertemplate="Level=%{y}<br>Put volume=%{x:.0f}<extra></extra>")
        fig.add_bar(x=df["call_volume_live"], y=y, orientation="h", name="Volume Calls", marker=dict(color=VOLUME_CALL_COLOR, line=dict(width=0)), opacity=0.9, hovertemplate="Level=%{y}<br>Call volume=%{x:.0f}<extra></extra>")
    elif mode == "live":
        fig.add_bar(x=df["put_live"], y=y, orientation="h", name="Live Puts", marker=dict(color=LIVE_PUT_COLOR, line=dict(width=0)), opacity=0.90, hovertemplate="Level=%{y}<br>Live puts=%{x:.4g}<extra></extra>")
        fig.add_bar(x=df["call_live"], y=y, orientation="h", name="Live Calls", marker=dict(color=LIVE_CALL_COLOR, line=dict(width=0)), opacity=0.90, hovertemplate="Level=%{y}<br>Live calls=%{x:.4g}<extra></extra>")
    else:
        fig.add_bar(x=df["put_gamma"], y=y, orientation="h", name="Base Puts", marker=dict(color=PUT_COLOR, line=dict(width=0)), opacity=0.95, width=0.82, hovertemplate="Level=%{y}<br>Base puts=%{x:.4g}<extra></extra>")
        fig.add_bar(x=df["call_gamma"], y=y, orientation="h", name="Base Calls", marker=dict(color=CALL_COLOR, line=dict(width=0)), opacity=0.95, width=0.82, hovertemplate="Level=%{y}<br>Base calls=%{x:.4g}<extra></extra>")
        fig.add_bar(x=df["put_live"], y=y, orientation="h", name="Live Puts", marker=dict(color=LIVE_PUT_COLOR, line=dict(width=0)), opacity=0.45, width=0.42, hovertemplate="Level=%{y}<br>Live puts=%{x:.4g}<extra></extra>")
        fig.add_bar(x=df["call_live"], y=y, orientation="h", name="Live Calls", marker=dict(color=LIVE_CALL_COLOR, line=dict(width=0)), opacity=0.45, width=0.42, hovertemplate="Level=%{y}<br>Live calls=%{x:.4g}<extra></extra>")
        fig.add_scatter(x=df["net_combined"], y=y, mode="markers", name="Combined Net", marker=dict(size=6, color=COMBINED_NET_COLOR), hovertemplate="Level=%{y}<br>Combined net=%{x:.4g}<extra></extra>")

    max_abs = _max_abs_for_mode(df, mode)
    if ZOOM_PCT > 0 and max_abs > 0:
        span = max_abs * ZOOM_PCT
        fig.update_xaxes(range=[-span, span])
    elif max_abs > 0:
        fig.update_xaxes(range=[-max_abs * 1.05, max_abs * 1.05])

    fig.add_vline(x=0, line_width=1, line_dash="dot", opacity=0.6)

    if mode in {"volume", "live", "combined"}:
        bits = []
        if live_meta.get("rows_used") is not None:
            bits.append(f"rows used: {int(live_meta['rows_used'])}")
        if live_meta.get("dte_min_used") is not None and live_meta.get("dte_max_used") is not None:
            bits.append(f"DTE: {int(float(live_meta['dte_min_used']))} to {int(float(live_meta['dte_max_used']))}")
        if live_meta.get("selected_time_pt"):
            bits.append(f"PT selected: {live_meta['selected_time_pt']}")
        if live_meta.get("selected_target_et"):
            bits.append(f"target ET: {live_meta['selected_target_et']}")
        if live_meta.get("candidate_trade_dates"):
            bits.append(f"requested: {', '.join(map(str, live_meta['candidate_trade_dates'][:2]))}")
        if live_meta.get("error"):
            bits.append(f"note: {live_meta['error']}")
        if bits:
            fig.add_annotation(text=" | ".join(bits), showarrow=False, xref="paper", yref="paper", x=0.5, y=1.12, font=dict(size=11))

    return fig


@callback(
    Output("GEX_GRAPH", "figure"),
    Input("trade-date", "date"),
    Input("smile-time-input", "value"),
    Input("gex-display-mode", "value"),
    Input("gex-live-interval", "n_intervals"),
)
def render_gex(
    trade_date_iso: str | None,
    selected_times_pt: str | list[str] | None,
    mode: str | None,
    _n_intervals: int | None,
):
    mode = (mode or "gex").lower()
    if not trade_date_iso:
        return _build_gex_figure(pd.DataFrame(), "—", mode=mode, live_meta={})

    trade_date = dt.date.fromisoformat(trade_date_iso)
    base_df = _fetch_gex_grouped_by_level(trade_date)
    live_df = _empty_live_df()
    live_meta: dict[str, object] = {}

    if mode in {"volume", "live", "combined"}:
        live_df, live_meta = fetch_live_proxy_grouped_by_level(
            trade_date,
            ticker=TICKER,
            selected_times_pt=selected_times_pt,
        )

    if mode == "gex":
        plot_df = base_df
    elif mode == "volume":
        plot_df = live_df
    elif mode == "live":
        plot_df = live_df
    else:
        plot_df = _merge_base_and_live(base_df, live_df)

    return _build_gex_figure(plot_df, trade_date_iso, mode=mode, live_meta=live_meta)
