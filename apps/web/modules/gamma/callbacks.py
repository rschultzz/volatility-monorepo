# apps/web/modules/gamma/callbacks.py
from __future__ import annotations
import os
import datetime as dt
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback
from sqlalchemy import text
from sqlalchemy import create_engine
from sqlalchemy.engine.url import make_url

# ---------- DB engine ----------
def _get_db_url() -> str:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set in the environment")

    # make sure SQLAlchemy driver is compatible
    if db_url.startswith("postgres://"):
        db_url = "postgresql://" + db_url[len("postgres://"):]
    if db_url.startswith("postgresql://") and "+psycopg" not in db_url:
        db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)

    return db_url


db_url = _get_db_url()
engine = create_engine(db_url, pool_pre_ping=True)

# ---------- Config ----------
# Optional zoom: set GEX_ZOOM_PCT to a float like "0.03" for ±3% of max call gamma.
# Leave 0 (default) to show the full width with a 5% padding.
ZOOM_PCT = float(os.getenv("GEX_ZOOM_PCT", "0"))

# Colors
PUT_COLOR  = os.getenv("GEX_PUT_COLOR",  "#E5E7EB")  # down candles
CALL_COLOR = os.getenv("GEX_CALL_COLOR", "#60a5fa") # up candles

# Filter by symbol if your table contains more than SPX
TICKER = os.getenv("GEX_TICKER", "SPX")


# ---------- DB engine ----------
try:
    # Prefer the shared engine helper in the monorepo
    from shared.utils.data_io import get_engine  # type: ignore
except Exception:
    # Basic fallback if shared import isn't available
    from sqlalchemy import create_engine


    def get_engine():
        raw = (
                os.getenv("DATABASE_URL")
                or os.getenv("POSTGRES_URL")
                or os.getenv("POSTGRES_CONNECTION_STRING")
                or "postgresql://localhost/postgres"
        )
        url = make_url(raw)
        # If the URL doesn't specify a driver or says psycopg2, force psycopg (v3)
        if url.get_backend_name() == "postgresql" and url.get_driver_name() in (None, "", "psycopg2"):
            url = url.set(drivername="postgresql+psycopg")
        # If you’re using read-only roles, keep your existing pool args
        return create_engine(url, pool_pre_ping=True, pool_recycle=300)

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

    Levels are binned to 1.0 index-point increments (whole points).
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


# ---------- Query ----------



# ---------- Figure ----------
def _build_gex_figure(df: pd.DataFrame, trade_date_str: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        title=f"GEX by Discounted Level — {trade_date_str}" + (f" ({TICKER})" if TICKER else ""),
        xaxis_title="Gamma (Σ by rounded discounted_level)",
        yaxis_title="Discounted Level (rounded)",
        margin=dict(l=90, r=30, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        barmode="overlay",
    )

    if df.empty:
        fig.add_annotation(
            text="No gamma data for the selected date",
            showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5
        )
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

    # --- Width handling (ALL inside the function) ---
    try:
        max_call = float(df["call_gamma"].abs().max() or 0.0)
    except Exception:
        max_call = 0.0

    max_abs = max(
        abs(float(df["put_gamma"].min() or 0.0)),
        abs(float(df["call_gamma"].max() or 0.0)),
    )

    if ZOOM_PCT > 0 and max_call > 0:
        # Optional tight zoom: ±(ZOOM_PCT * max call gamma)
        span = max_call * ZOOM_PCT
        fig.update_xaxes(range=[-span, span])
    elif max_abs > 0:
        # Default: show full width (+5% padding)
        fig.update_xaxes(range=[-max_abs * 1.05, max_abs * 1.05])

    # Zero line
    fig.add_vline(x=0, line_width=1, line_dash="dot", opacity=0.6)

    return fig


# ---------- Dash callback ----------
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
