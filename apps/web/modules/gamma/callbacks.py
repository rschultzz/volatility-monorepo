# modules/gex/callbacks.py
from __future__ import annotations
import os
import datetime as dt
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback

# How wide to zoom: percent of max call_gamma (default 3%)
ZOOM_PCT = float(os.getenv("GEX_ZOOM_PCT", "0.03"))


# === Add these near the top (after imports) ===
PUT_COLOR  = os.getenv("GEX_PUT_COLOR",  "#E5E7EB")  # light gray (old dash)
CALL_COLOR = os.getenv("GEX_CALL_COLOR", "#334155")  # slate/dark blue (old dash)


# If you want to target a different symbol, change this (or set env GEX_TICKER)
TICKER = os.getenv("GEX_TICKER", "SPX")

# ----- DB engine -----
try:
    # Prefer your project's shared engine if it exists
    from shared.utils.data_io import get_engine  # type: ignore
except Exception:
    from sqlalchemy import create_engine
    def get_engine():
        url = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL")
        if not url:
            raise RuntimeError("DATABASE_URL not set; add it to your env or .env")
        return create_engine(url, pool_pre_ping=True, pool_recycle=300)

from sqlalchemy import text

# ----- Query: group by rounded discounted_level -----
def _fetch_gex_grouped_by_level(trade_date: dt.date) -> pd.DataFrame:
    """
    Returns columns: level(int), call_gamma, put_gamma, net_gamma
    From table: orats_oi_gamma
    WHERE trade_date = :d [AND ticker = :tkr]
    GROUP BY ROUND(discounted_level) (nearest 1)
    """
    eng = get_engine()
    dialect = eng.dialect.name
    level_expr = "ROUND(discounted_level)::INT" if dialect == "postgresql" else "CAST(ROUND(discounted_level) AS INTEGER)"

    where = ["trade_date = :d", "discounted_level IS NOT NULL"]
    params = {"d": trade_date.isoformat()}
    if TICKER:
        where.append("ticker = :tkr")
        params["tkr"] = TICKER

    sql = f"""
        SELECT
            {level_expr} AS level,
            COALESCE(SUM(gex_call), 0) AS call_gamma,
            COALESCE(SUM(gex_put), 0)  AS put_gamma
        FROM orats_oi_gamma
        WHERE {" AND ".join(where)}
        GROUP BY {level_expr}
        ORDER BY {level_expr}
    """

    with eng.connect() as con:
        df = pd.read_sql(text(sql), con, params=params)

    if df.empty:
        return pd.DataFrame(columns=["level", "call_gamma", "put_gamma", "net_gamma"]).astype(
            {"level": "int64", "call_gamma": "float64", "put_gamma": "float64", "net_gamma": "float64"}
        )

    # Convention: puts to the left (negative)
    df["put_gamma"] = -df["put_gamma"].abs()
    df["net_gamma"] = df["call_gamma"] + df["put_gamma"]
    df["level"] = df["level"].astype(int)
    return df

# ----- Figure -----
def _build_gex_figure(df: pd.DataFrame, trade_date_str: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        title=f"GEX by Discounted Level — {trade_date_str}" + (f" ({TICKER})" if TICKER else ""),
        xaxis_title="Gamma (Σ by rounded discounted_level)",
        yaxis_title="Discounted Level (rounded)",
        margin=dict(l=90, r=30, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    if df.empty:
        fig.add_annotation(text="No gamma data for the selected date", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.5)
        fig.add_vline(x=0, line_width=1, line_dash="dot")
        return fig

    y = df["level"].astype(str)

    fig.add_bar(
        x=df["put_gamma"], y=y, orientation="h", name="Puts",
        marker=dict(color=PUT_COLOR, line=dict(width=0)),
        hovertemplate="Level=%{y}<br>Puts=%{x:.4g}<extra></extra>",
    )
    fig.add_bar(
        x=df["call_gamma"], y=y, orientation="h", name="Calls",
        marker=dict(color=CALL_COLOR, line=dict(width=0)),
        hovertemplate="Level=%{y}<br>Calls=%{x:.4g}<extra></extra>",
    )
    # === Auto-zoom to ±(ZOOM_PCT * max_call_gamma) ===
    try:
        max_call = float(df["call_gamma"].abs().max())
    except Exception:
        max_call = 0.0

    if max_call > 0:
        span = max_call * ZOOM_PCT
        # symmetric zoom around 0
        fig.update_xaxes(range=[-span, span])


    fig.update_layout(barmode="relative")
    fig.add_vline(x=0, line_width=1, line_dash="dot")
    fig.update_yaxes(categoryorder="array", categoryarray=y.tolist())
    return fig

# ----- Dash callback (date-only) -----
@callback(
    Output("GEX_GRAPH", "figure"),
    Input("trade-date", "date"),
)
def render_gex(trade_date_iso: str | None):
    if not trade_date_iso:
        return _build_gex_figure(pd.DataFrame(), "—")
    trade_date = dt.date.fromisoformat(trade_date_iso)
    df = _fetch_gex_grouped_by_level(trade_date)
    return _build_gex_figure(df, trade_date_iso)
