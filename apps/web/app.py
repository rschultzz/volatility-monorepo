# apps/web/app.py

from __future__ import annotations

# load local environment variables from .env
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

# --- make repo root importable so `packages.*` works on Render ---
import sys
from pathlib import Path as _P

REPO_ROOT = _P(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# -----------------------------------------------------------------

import os
import datetime as dt
from typing import List
from functools import lru_cache

import pandas as pd
from sqlalchemy import create_engine, text

from dash import Dash, html, dcc, Input, Output, State, no_update
from dash import dash_table
from dash.exceptions import PreventUpdate
import dash_auth
from zoneinfo import ZoneInfo
from flask import send_from_directory



# ===== Modules =====
from modules.Skew.components import make_skew_block
from modules.Skew.callbacks import register_callbacks as register_skew
from modules.gamma.components import gex_block
from modules.gamma import callbacks as _gex_callbacks  # noqa: F401
from modules.Smile.callbacks import register_callbacks as register_smile
from modules.TermStructure.components import make_term_structure_block
from modules.TermStructure.callbacks import register_callbacks as register_term_structure
from modules.TermMetrics.components import make_term_metrics_block
from modules.TermMetrics.callbacks import register_callbacks as register_term_metrics
from modules.Ironbeam.components import ironbeam_layout
from modules.Ironbeam.callbacks import register_ironbeam_callbacks

# ===== Backtests (existing) =====
from modules.Backtests.components import make_backtests_tab
from modules.Backtests.callbacks import register_callbacks as register_backtests

# ===== IDs =====
CLOCK_ID = "CLOCK"
TRADE_DATE_PICK = "trade-date"
EXPIRATION_DATE_PICK = "expiration-date-pick"
SMILE_TIME_INPUT = "smile-time-input"
SMILE_GRAPH = "SMILE_GRAPH"
EXPECTED_TOGGLE_ID = "expected-ss-toggle"
LIVE_DATA_STORE_ID = "live-data-store"
LIVE_UPDATE_TIMER_ID = "live-update-timer"

# Tabs (selector only; containers below stay mounted)
MAIN_TABS_ID = "main-tabs"
TAB_DASHBOARD = "tab-dashboard"
TAB_PRICE_CHART = "tab-price-chart"
TAB_BACKTESTS = "tab-backtests"
TAB_BACKTESTS_V2 = "tab-backtests-v2"

# Backtests v2 view name (you renamed it)
BT_VIEW_NAME = os.getenv("BT_VIEW_NAME", "es_minutes_with_features_bt")

# ---- Tabs styling (match backtest section cards) ----
TABS_WRAP_STYLE = {
    "backgroundColor": "#0b1220",
    "border": "1px solid #1f2937",
    "borderRadius": "14px",
    "padding": "6px",
    "marginBottom": "12px",
}

TABS_STYLE = {
    "backgroundColor": "transparent",
    "borderBottom": "0px",
    "height": "44px",
}

TAB_STYLE = {
    "backgroundColor": "transparent",
    "border": "0px",
    "padding": "10px 16px",
    "borderRadius": "12px",
    "color": "#93c5fd",  # light blue like section headers
    "fontWeight": "700",
    "fontSize": "13px",
}

TAB_SELECTED_STYLE = {
    "backgroundColor": "#111827",  # slightly lighter than card bg
    "border": "1px solid #60a5fa",  # blue outline
    "padding": "10px 16px",
    "borderRadius": "12px",
    "color": "#bfdbfe",
    "fontWeight": "800",
    "fontSize": "13px",
}

CARD_STYLE = {
    "backgroundColor": "#0b1220",
    "border": "1px solid #1f2937",
    "borderRadius": "14px",
    "padding": "12px",
}

LABEL_STYLE = {"color": "#e5e7eb", "fontSize": "12px", "fontWeight": "600", "marginBottom": "6px"}


# ===== UI Helpers =====
def get_default_trade_date() -> dt.date:
    """
    Return 'today' if it's Monday–Friday.
    If it's Saturday, return Friday.
    If it's Sunday, return Friday.
    """
    today = dt.date.today()
    if today.weekday() == 5:
        return today - dt.timedelta(days=1)  # Sat -> Fri
    if today.weekday() == 6:
        return today - dt.timedelta(days=2)  # Sun -> Fri
    return today


def pt_time_options(start="06:30", end="13:00", step_min=1) -> List[dict]:
    t0, t1 = dt.datetime.strptime(start, "%H:%M"), dt.datetime.strptime(end, "%H:%M")
    out, cur = [], t0
    while cur <= t1:
        hhmm = cur.strftime("%H:%M")
        out.append({"label": f"{hhmm} PT", "value": hhmm})
        cur += dt.timedelta(minutes=step_min)
    return out


# ===== DB helpers (Backtests v2 only) =====
@lru_cache(maxsize=1)
def _get_engine():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set")
    return create_engine(db_url, pool_pre_ping=True)

def _to_iso_utc(x) -> str:
    ts = pd.to_datetime(x, utc=True)
    return ts.isoformat()



def _bt2_fetch_rows(start_date: str, end_date: str, rth_only: bool, limit_rows: int):
    engine = _get_engine()

    # Convert Dash date strings -> Python date objects (cleanest for Postgres)
    start_d = dt.date.fromisoformat(start_date)
    end_d = dt.date.fromisoformat(end_date)

    q = text(
        f"""
        SELECT
          ts_pt,
          ts_utc,
          trade_date,
          is_rth,
          open,
          high,
          low,
          close,
          volume,
          net_gex,
          gex_wall_above,
          gex_wall_above_gex,
          gex_wall_below,
          gex_wall_below_gex,
          gex_strong_wall_above,
          gex_strong_wall_above_gex,
          gex_strong_wall_below,
          gex_strong_wall_below_gex,
          stock_price,
          atmiv,
          put_skew_pp_primary,
          call_skew_pp_primary,
          smile_vol25_primary,
          smile_vol50_primary,
          smile_vol75_primary
        FROM public.{BT_VIEW_NAME}
        WHERE trade_date >= :start_date
          AND trade_date <= :end_date
          AND (:rth_only = FALSE OR is_rth = TRUE)
        ORDER BY ts_pt
        LIMIT :limit_rows
        """
    )

    with engine.connect() as conn:
        df = pd.read_sql(
            q,
            conn,
            params={
                "start_date": start_d,
                "end_date": end_d,
                "rth_only": bool(rth_only),
                "limit_rows": int(limit_rows),
            },
        )

    return df

def _safe_ident(name: str) -> str:
    # allow only letters, numbers, underscore
    ok = all(ch.isalnum() or ch == "_" for ch in name)
    if not ok:
        raise ValueError(f"Unsafe identifier: {name}")
    return name


def _ensure_strategy_id(strategy_name: str) -> int:
    """
    Create strategy if missing; return id.
    """
    engine = _get_engine()
    strategy_name = (strategy_name or "").strip()
    if not strategy_name:
        raise ValueError("Strategy name is required")

    with engine.begin() as conn:
        row = conn.execute(
            text("SELECT id FROM bt_strategies WHERE name = :name"),
            {"name": strategy_name},
        ).fetchone()
        if row:
            return int(row[0])

        new_id = conn.execute(
            text("INSERT INTO bt_strategies (name, feature_version) VALUES (:name, 'v1') RETURNING id"),
            {"name": strategy_name},
        ).fetchone()[0]
        return int(new_id)


def _save_strategy_instance(strategy_id: int, anchor_ts_utc: str, entry_ts_utc: str, label: int = 1) -> int:
    """
    Save anchor+entry rows from es_minutes_with_features_bt and a small feature JSON.
    Returns inserted instance id.
    """
    engine = _get_engine()
    view_name = _safe_ident(os.getenv("BT_VIEW_NAME", "es_minutes_with_features_bt"))

    sql = text(
        f"""
        WITH
        a AS (
          SELECT *
          FROM public.{view_name}
          WHERE ts_utc = :anchor_ts_utc
          LIMIT 1
        ),
        e AS (
          SELECT *
          FROM public.{view_name}
          WHERE ts_utc = :entry_ts_utc
          LIMIT 1
        ),
        f AS (
          SELECT jsonb_build_object(
            'px_chg',          (e.close - a.close),
            'ret',             CASE WHEN a.close IS NULL OR a.close = 0 THEN NULL ELSE (e.close / a.close - 1) END,

            'atmiv_chg',       (e.atmiv - a.atmiv),
            'put_skew_pp_chg', (e.put_skew_pp_primary - a.put_skew_pp_primary),
            'call_skew_pp_chg',(e.call_skew_pp_primary - a.call_skew_pp_primary),

            'net_gex',         e.net_gex,
            'is_rth',          e.is_rth,

            'dist_wall_above', CASE WHEN e.gex_wall_above IS NULL THEN NULL ELSE (e.gex_wall_above - e.close) END,
            'dist_wall_below', CASE WHEN e.gex_wall_below IS NULL THEN NULL ELSE (e.close - e.gex_wall_below) END,

            'es_close_entry',  e.close,
            'spx_stock_entry', e.stock_price
          ) AS features
          FROM a CROSS JOIN e
        )
        INSERT INTO bt_strategy_instances (
          strategy_id,
          anchor_ts_utc,
          entry_ts_utc,
          trade_date,
          is_rth,
          anchor_row,
          entry_row,
          features,
          feature_version,
          label,
          labeled_at
        )
        SELECT
          :strategy_id,
          (SELECT ts_utc FROM a),
          (SELECT ts_utc FROM e),
          (SELECT trade_date FROM e),
          (SELECT is_rth FROM e),
          (SELECT to_jsonb(a) FROM a),
          (SELECT to_jsonb(e) FROM e),
          (SELECT features FROM f),
          'v1',
          :label,
          CASE WHEN :label = 0 THEN NULL ELSE now() END
        RETURNING id;
        """
    )

    with engine.begin() as conn:
        row = conn.execute(
            sql,
            {
                "strategy_id": int(strategy_id),
                "anchor_ts_utc": anchor_ts_utc,
                "entry_ts_utc": entry_ts_utc,
                "label": int(label),
            },
        ).fetchone()

    if not row:
        raise ValueError("Could not save instance — missing anchor or entry row for those timestamps.")
    return int(row[0])






def _bt2_list_strategy_options() -> List[dict]:
    """Return strategies for dropdown as [{'label': name, 'value': name}, ...]."""
    try:
        engine = _get_engine()
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT name FROM bt_strategies ORDER BY name")).fetchall()
        opts = [{"label": r[0], "value": r[0]} for r in rows if r and r[0]]
        return opts if opts else [{"label": "Strategy A", "value": "Strategy A"}]
    except Exception:
        # Table may not exist yet in a fresh DB
        return [{"label": "Strategy A", "value": "Strategy A"}]

# ===== App setup =====
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

server.secret_key = os.getenv("SECRET_KEY", "dev-secret-key")

VALID_USERNAME_PASSWORD_PAIRS = {
    "ryan": "ChangeThisPassword123!",
    "noah": "ChangeThisPassword123!",
    "sara": "ChangeThisPassword123!",
}
auth = dash_auth.BasicAuth(app, VALID_USERNAME_PASSWORD_PAIRS)


REACT_PREVIEW_DIST_DIR = (REPO_ROOT / "react_price_preview" / "dist").resolve()


def _react_preview_build_ready() -> bool:
    return REACT_PREVIEW_DIST_DIR.exists() and (REACT_PREVIEW_DIST_DIR / "index.html").exists()


@server.route("/react-preview")
@server.route("/react-preview/")
def react_preview_index():
    if not _react_preview_build_ready():
        return (
            "React preview build not found. Build react_price_preview/dist before starting Dash.",
            503,
        )
    return send_from_directory(str(REACT_PREVIEW_DIST_DIR), "index.html")


@server.route("/react-preview/<path:path>")
def react_preview_assets(path):
    if not _react_preview_build_ready():
        return (
            "React preview build not found. Build react_price_preview/dist before starting Dash.",
            503,
        )

    candidate = (REACT_PREVIEW_DIST_DIR / path).resolve()
    if candidate.exists() and candidate.is_file():
        return send_from_directory(str(REACT_PREVIEW_DIST_DIR), path)

    return send_from_directory(str(REACT_PREVIEW_DIST_DIR), "index.html")


def serve_layout():
    """Layout factory so defaults are recomputed on each page load."""
    default_trade_date = get_default_trade_date()
    time_options = pt_time_options()

    # Backtests (existing) default range
    bt_end = default_trade_date
    bt_start = default_trade_date - dt.timedelta(days=10)

    # Backtests v2 default range
    bt2_end = default_trade_date
    bt2_start = default_trade_date - dt.timedelta(days=3)

    # Backtests v2 strategies
    bt2_strategy_options = _bt2_list_strategy_options()
    bt2_strategy_default = bt2_strategy_options[0]["value"] if bt2_strategy_options else "Strategy A"

    # ----- DASHBOARD BODY (this matches your original layout) -----
    dashboard_children = [
        # ===== Global controls (date / expiration / time slices / expected) =====
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Trade Date", style={"color": "white", "marginBottom": "4px"}),
                        dcc.DatePickerSingle(
                            id=TRADE_DATE_PICK,
                            disabled=False,
                            display_format="YYYY-MM-DD",
                            date=default_trade_date,
                        ),
                    ],
                    style={"marginRight": "16px", "display": "flex", "flexDirection": "column"},
                ),
                html.Div(
                    [
                        html.Label("Expiration", style={"color": "white", "marginBottom": "4px"}),
                        dcc.DatePickerSingle(
                            id=EXPIRATION_DATE_PICK,
                            disabled=False,
                            display_format="YYYY-MM-DD",
                            date=default_trade_date,
                        ),
                    ],
                    style={"marginRight": "16px", "display": "flex", "flexDirection": "column"},
                ),
                html.Div(
                    [
                        html.Label("Time Slices (PT)", style={"color": "white", "marginBottom": "4px"}),
                        # IMPORTANT: multi=True and list value (Smile/Skew callbacks expect list)
                        dcc.Dropdown(
                            id=SMILE_TIME_INPUT,
                            options=time_options,
                            value=["06:31"],
                            multi=True,
                            style={"minWidth": "320px"},
                        ),
                    ],
                    style={"marginRight": "16px", "display": "flex", "flexDirection": "column"},
                ),
                html.Div(
                    [
                        html.Label("Compare to Expected (SS)", style={"color": "white", "marginBottom": "4px"}),
                        dcc.RadioItems(
                            id=EXPECTED_TOGGLE_ID,
                            options=[{"label": "ON", "value": "on"}, {"label": "OFF", "value": "off"}],
                            value="on",
                            inline=True,
                            labelStyle={"marginRight": "16px", "color": "white", "fontSize": "13px"},
                            inputStyle={"marginRight": "6px"},
                        ),
                    ],
                    style={"marginRight": "16px", "display": "flex", "flexDirection": "column"},
                ),
            ],
            style={
                "display": "flex",
                "flexWrap": "wrap",
                "alignItems": "center",
                "gap": "24px",
                "marginBottom": "8px",
            },
        ),
        html.Hr(style={"borderColor": "#444"}),
        # ===== Smile + GEX block =====
        html.Div(
            [
                html.Div(dcc.Graph(id=SMILE_GRAPH, style={"height": "100%"}), style={"minWidth": 0, "flex": "2"}),
                html.Div(gex_block(), style={"minWidth": 0, "flex": "1"}),
            ],
            style={"display": "flex", "gap": "16px", "alignItems": "stretch", "height": "calc(60vh - 100px)"},
        ),
        html.Hr(style={"borderColor": "#333"}),
        # ===== Term structure / Skew / Term metrics =====
        html.Div(
            [
                html.Div(make_term_structure_block(), style={"flex": "1"}),
                html.Div(
                    [make_skew_block(), make_term_metrics_block()],
                    style={"flex": "1", "display": "flex", "flexDirection": "column", "gap": "16px"},
                ),
            ],
            style={"display": "flex", "gap": "16px", "alignItems": "stretch", "height": "calc(40vh - 100px)"},
        ),
    ]

    # ----- IRONBEAM BODY -----
    ironbeam_children = [
        html.Div(
            [
                # Controls row
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label(
                                    "Bar Interval",
                                    style={
                                        "color": "white",
                                        "marginBottom": "6px",
                                        "fontSize": "13px",
                                        "fontWeight": "500",
                                    },
                                ),
                                dcc.RadioItems(
                                    id="ironbeam-bar-interval",
                                    options=[
                                        {"label": "1 min", "value": "1min"},
                                        {"label": "5 min", "value": "5min"},
                                    ],
                                    value="1min",
                                    inline=True,
                                    labelStyle={"marginRight": "16px", "color": "white", "fontSize": "13px"},
                                    inputStyle={"marginRight": "6px"},
                                    style={"paddingTop": "4px"},
                                ),
                            ],
                            style={
                                "minWidth": "180px",
                                "flex": "0 0 auto",
                                "display": "flex",
                                "flexDirection": "column",
                            },
                        ),
                        html.Div(
                            [
                                html.Label(
                                    "Chart Mode",
                                    style={
                                        "color": "white",
                                        "marginBottom": "6px",
                                        "fontSize": "13px",
                                        "fontWeight": "500",
                                    },
                                ),
                                dcc.RadioItems(
                                    id="ib-chart-mode-toggle",
                                    options=[
                                        {"label": "Classic", "value": "classic"},
                                        {"label": "React Preview", "value": "react_preview"},
                                    ],
                                    value="classic",
                                    inline=True,
                                    labelStyle={"marginRight": "16px", "color": "white", "fontSize": "13px"},
                                    inputStyle={"marginRight": "6px"},
                                    style={"paddingTop": "4px"},
                                ),
                            ],
                            style={
                                "minWidth": "220px",
                                "flex": "0 0 auto",
                                "display": "flex",
                                "flexDirection": "column",
                            },
                        ),
                    ],
                    style={
                        "display": "flex",
                        "gap": "18px",
                        "alignItems": "flex-end",
                        "flexWrap": "wrap",
                        "marginBottom": "10px",
                    },
                ),
                # Ironbeam chart block
                ironbeam_layout(),
            ]
        )
    ]

    # ----- BACKTESTS V2 BODY (NEW) -----
    backtests_v2_children = [
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("Date Range (Backtests v2)", style=LABEL_STYLE),
                                dcc.DatePickerRange(
                                    id="bt2-date-range",
                                    start_date=bt2_start,
                                    end_date=bt2_end,
                                    display_format="YYYY-MM-DD",
                                ),
                            ],
                            style={"flex": "0 0 auto", "minWidth": "320px"},
                        ),
                        html.Div(
                            [
                                html.Label("RTH only", style=LABEL_STYLE),
                                dcc.RadioItems(
                                    id="bt2-rth-only",
                                    options=[{"label": "Yes", "value": "yes"}, {"label": "No", "value": "no"}],
                                    value="yes",
                                    inline=True,
                                    labelStyle={"marginRight": "16px", "color": "white", "fontSize": "13px"},
                                    inputStyle={"marginRight": "6px"},
                                ),
                            ],
                            style={"flex": "0 0 auto", "minWidth": "200px"},
                        ),
                        html.Div(
                            [
                                html.Label("Max rows", style=LABEL_STYLE),
                                dcc.Input(
                                    id="bt2-limit-rows",
                                    type="number",
                                    min=50,
                                    max=10000,
                                    step=50,
                                    value=800,
                                    style={
                                        "width": "120px",
                                        "padding": "6px 8px",
                                        "borderRadius": "8px",
                                        "border": "1px solid #1f2937",
                                        "backgroundColor": "#0b1220",
                                        "color": "white",
                                    },
                                ),
                            ],
                            style={"flex": "0 0 auto", "minWidth": "140px"},
                        ),
                        html.Div(
                            [
                                html.Button(
                                    "Load",
                                    id="bt2-load-btn",
                                    n_clicks=0,
                                    style={
                                        "backgroundColor": "#111827",
                                        "border": "1px solid #60a5fa",
                                        "color": "#bfdbfe",
                                        "fontWeight": "800",
                                        "borderRadius": "10px",
                                        "padding": "8px 14px",
                                        "cursor": "pointer",
                                        "marginTop": "20px",
                                    },
                                ),
                            ],
                            style={"flex": "0 0 auto"},
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div("Source view:", style={"color": "#94a3b8", "fontSize": "12px"}),
                                        html.Div(BT_VIEW_NAME, style={"color": "#e5e7eb", "fontSize": "13px", "fontWeight": "700"}),
                                    ],
                                    style={"marginTop": "18px"},
                                )
                            ],
                            style={"flex": "1"},
                        ),
                    ],
                    style={"display": "flex", "gap": "14px", "alignItems": "flex-start", "flexWrap": "wrap"},
                ),
                html.Hr(style={"borderColor": "#1f2937", "margin": "12px 0"}),

                # Stores live INSIDE the tab/card
                dcc.Store(id="bt2-anchor-ts-utc"),
                dcc.Store(id="bt2-entry-ts-utc"),

                html.Div(
                    id="bt2-summary",
                    style={"color": "#e5e7eb", "fontSize": "13px", "marginBottom": "10px"},
                ),

                # ✅ UI controls go HERE (above the table)
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("Strategy", style=LABEL_STYLE),
                                dcc.Dropdown(
                                    id="bt2-strategy-name",
                                    options=bt2_strategy_options,
                                    value=bt2_strategy_default,
                                    clearable=False,
                                    searchable=True,
                                    style={"minWidth": "260px"},
                                ),
                                html.Div(
                                    [
                                        dcc.Input(
                                            id="bt2-new-strategy-name",
                                            type="text",
                                            placeholder="New strategy name…",
                                            style={
                                                "width": "220px",
                                                "padding": "6px 8px",
                                                "borderRadius": "8px",
                                                "border": "1px solid #1f2937",
                                                "backgroundColor": "#0b1220",
                                                "color": "white",
                                                "marginTop": "8px",
                                            },
                                        ),
                                        html.Button(
                                            "Create",
                                            id="bt2-create-strategy",
                                            n_clicks=0,
                                            style={
                                                "backgroundColor": "#111827",
                                                "border": "1px solid #60a5fa",
                                                "color": "#bfdbfe",
                                                "fontWeight": "900",
                                                "borderRadius": "10px",
                                                "padding": "6px 10px",
                                                "cursor": "pointer",
                                                "marginLeft": "10px",
                                                "marginTop": "8px",
                                            },
                                        ),
                                    ],
                                    style={"display": "flex", "alignItems": "center"},
                                ),
                                html.Div(
                                    id="bt2-strategy-status",
                                    style={"color": "#94a3b8", "fontSize": "12px", "marginTop": "6px"},
                                ),
                            ],
                            style={"marginRight": "12px"},
                        ),
                        html.Button(
                            "Set Anchor (selected row)",
                            id="bt2-set-anchor",
                            n_clicks=0,
                            style={
                                "backgroundColor": "#111827",
                                "border": "1px solid #60a5fa",
                                "color": "#bfdbfe",
                                "fontWeight": "800",
                                "borderRadius": "10px",
                                "padding": "8px 14px",
                                "cursor": "pointer",
                                "marginTop": "20px",
                                "marginRight": "10px",
                            },
                        ),
                        html.Button(
                            "Set Entry (selected row)",
                            id="bt2-set-entry",
                            n_clicks=0,
                            style={
                                "backgroundColor": "#111827",
                                "border": "1px solid #60a5fa",
                                "color": "#bfdbfe",
                                "fontWeight": "800",
                                "borderRadius": "10px",
                                "padding": "8px 14px",
                                "cursor": "pointer",
                                "marginTop": "20px",
                                "marginRight": "10px",
                            },
                        ),
                        html.Button(
                            "Save Example",
                            id="bt2-save-example",
                            n_clicks=0,
                            style={
                                "backgroundColor": "#0b1220",
                                "border": "1px solid #22c55e",
                                "color": "#bbf7d0",
                                "fontWeight": "900",
                                "borderRadius": "10px",
                                "padding": "8px 14px",
                                "cursor": "pointer",
                                "marginTop": "20px",
                            },
                        ),
                        html.Div(
                            id="bt2-anchor-entry-status",
                            style={"color": "#e5e7eb", "marginTop": "24px", "marginLeft": "14px"},
                        ),
                    ],
                    style={"display": "flex", "alignItems": "flex-start", "flexWrap": "wrap"},
                ),

                html.Div(id="bt2-save-status", style={"color": "#e5e7eb", "marginTop": "8px"}),

                # --- Find Similar (v1) controls + candidates table ---
                html.Hr(style={"borderColor": "#1f2937", "margin": "12px 0"}),

                dcc.Store(id="bt2-cands-store"),

                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("Lookahead (minutes)", style=LABEL_STYLE),
                                dcc.Input(
                                    id="bt2-lookahead-min",
                                    type="number",
                                    min=1,
                                    max=240,
                                    step=1,
                                    value=5,
                                    style={
                                        "width": "120px",
                                        "padding": "6px 8px",
                                        "borderRadius": "8px",
                                        "border": "1px solid #1f2937",
                                        "backgroundColor": "#0b1220",
                                        "color": "white",
                                    },
                                ),
                            ],
                            style={"marginRight": "12px"},
                        ),
                        html.Div(
                            [
                                html.Label("Stride (minutes)", style=LABEL_STYLE),
                                dcc.Input(
                                    id="bt2-stride-min",
                                    type="number",
                                    min=1,
                                    max=60,
                                    step=1,
                                    value=1,
                                    style={
                                        "width": "120px",
                                        "padding": "6px 8px",
                                        "borderRadius": "8px",
                                        "border": "1px solid #1f2937",
                                        "backgroundColor": "#0b1220",
                                        "color": "white",
                                    },
                                ),
                            ],
                            style={"marginRight": "12px"},
                        ),
                        html.Div(
                            [
                                html.Label("Top N", style=LABEL_STYLE),
                                dcc.Input(
                                    id="bt2-topn",
                                    type="number",
                                    min=5,
                                    max=500,
                                    step=5,
                                    value=50,
                                    style={
                                        "width": "120px",
                                        "padding": "6px 8px",
                                        "borderRadius": "8px",
                                        "border": "1px solid #1f2937",
                                        "backgroundColor": "#0b1220",
                                        "color": "white",
                                    },
                                ),
                            ],
                            style={"marginRight": "12px"},
                        ),
                        html.Button(
                            "Find Similar",
                            id="bt2-find-similar",
                            n_clicks=0,
                            style={
                                "backgroundColor": "#111827",
                                "border": "1px solid #f59e0b",
                                "color": "#fde68a",
                                "fontWeight": "900",
                                "borderRadius": "10px",
                                "padding": "8px 14px",
                                "cursor": "pointer",
                                "marginTop": "20px",
                                "marginRight": "10px",
                            },
                        ),
                        html.Button(
                            "Accept",
                            id="bt2-accept-cand",
                            n_clicks=0,
                            style={
                                "backgroundColor": "#0b1220",
                                "border": "1px solid #22c55e",
                                "color": "#bbf7d0",
                                "fontWeight": "900",
                                "borderRadius": "10px",
                                "padding": "8px 14px",
                                "cursor": "pointer",
                                "marginTop": "20px",
                                "marginRight": "10px",
                            },
                        ),
                        html.Button(
                            "Reject",
                            id="bt2-reject-cand",
                            n_clicks=0,
                            style={
                                "backgroundColor": "#0b1220",
                                "border": "1px solid #ef4444",
                                "color": "#fecaca",
                                "fontWeight": "900",
                                "borderRadius": "10px",
                                "padding": "8px 14px",
                                "cursor": "pointer",
                                "marginTop": "20px",
                            },
                        ),
                        html.Button(
                            "Load Candidate on Dashboard",
                            id="bt2-load-cand-to-dash",
                            n_clicks=0,
                            style={
                                "backgroundColor": "#111827",
                                "border": "1px solid #60a5fa",
                                "color": "#bfdbfe",
                                "fontWeight": "900",
                                "borderRadius": "10px",
                                "padding": "8px 14px",
                                "cursor": "pointer",
                                "marginTop": "20px",
                                "marginLeft": "10px",
                            },
                        ),
                        html.Div(id="bt2-find-status", style={"color": "#e5e7eb", "marginTop": "24px", "marginLeft": "14px"}),
                    ],
                    style={"display": "flex", "alignItems": "flex-start", "flexWrap": "wrap"},
                ),

                dash_table.DataTable(
                    id="bt2-cands-table",
                    data=[],
                    columns=[],
                    page_size=15,
                    row_selectable="single",
                    selected_rows=[],
                    sort_action="native",
                    filter_action="native",
                    style_table={"overflowX": "auto", "maxHeight": "40vh", "overflowY": "auto"},
                    style_header={
                        "backgroundColor": "#111827",
                        "color": "white",
                        "fontWeight": "700",
                        "border": "1px solid #1f2937",
                    },
                    style_cell={
                        "backgroundColor": "#0b1220",
                        "color": "white",
                        "border": "1px solid #1f2937",
                        "fontSize": "12px",
                        "padding": "6px",
                        "whiteSpace": "nowrap",
                    },
                ),

                html.Div(id="bt2-cand-action-status", style={"color": "#e5e7eb", "marginTop": "8px"}),
                html.Div(id="bt2-load-cand-status", style={"color": "#e5e7eb", "marginTop": "8px"}),
                html.Hr(style={"borderColor": "#1f2937", "margin": "12px 0"}),

                html.Div(
                    [
                        html.Div("Saved Examples", style={"color": "#e5e7eb", "fontSize": "14px", "fontWeight": "800"}),

                        html.Button(
                            "Refresh Saved",
                            id="bt2-refresh-saved",
                            n_clicks=0,
                            style={
                                "backgroundColor": "#111827",
                                "border": "1px solid #60a5fa",
                                "color": "#bfdbfe",
                                "fontWeight": "900",
                                "borderRadius": "10px",
                                "padding": "8px 14px",
                                "cursor": "pointer",
                            },
                        ),
                        html.Button(
                            "Load on Dashboard",
                            id="bt2-load-saved-to-dash",
                            n_clicks=0,
                            style={
                                "backgroundColor": "#0b1220",
                                "border": "1px solid #22c55e",
                                "color": "#bbf7d0",
                                "fontWeight": "900",
                                "borderRadius": "10px",
                                "padding": "8px 14px",
                                "cursor": "pointer",
                                "marginLeft": "10px",
                            },
                        ),
                        html.Div(id="bt2-saved-status", style={"color": "#e5e7eb", "marginLeft": "14px"}),
                    ],
                    style={"display": "flex", "alignItems": "center", "gap": "10px", "flexWrap": "wrap"},
                ),

                dash_table.DataTable(
                    id="bt2-saved-table",
                    data=[],
                    columns=[],
                    page_size=10,
                    row_selectable="single",
                    selected_rows=[],
                    sort_action="native",
                    filter_action="native",
                    style_table={"overflowX": "auto", "maxHeight": "35vh", "overflowY": "auto"},
                    style_header={
                        "backgroundColor": "#111827",
                        "color": "white",
                        "fontWeight": "700",
                        "border": "1px solid #1f2937",
                    },
                    style_cell={
                        "backgroundColor": "#0b1220",
                        "color": "white",
                        "border": "1px solid #1f2937",
                        "fontSize": "12px",
                        "padding": "6px",
                        "whiteSpace": "nowrap",
                    },
                ),

                html.Div(id="bt2-load-dash-status", style={"color": "#e5e7eb", "marginTop": "8px"}),

                dash_table.DataTable(
                    id="bt2-table",
                    data=[],
                    columns=[],
                    page_size=25,
                    row_selectable="single",
                    selected_rows=[],
                    sort_action="native",
                    filter_action="native",
                    style_table={"overflowX": "auto", "maxHeight": "70vh", "overflowY": "auto"},
                    style_header={
                        "backgroundColor": "#111827",
                        "color": "white",
                        "fontWeight": "700",
                        "border": "1px solid #1f2937",
                    },
                    style_cell={
                        "backgroundColor": "#0b1220",
                        "color": "white",
                        "border": "1px solid #1f2937",
                        "fontSize": "12px",
                        "padding": "6px",
                        "whiteSpace": "nowrap",
                    },
                ),

            ],
            style=CARD_STYLE,
        )
    ]

    return html.Div(
        [
            dcc.Store(id=LIVE_DATA_STORE_ID),
            dcc.Interval(id=LIVE_UPDATE_TIMER_ID, interval=15 * 1000, n_intervals=0),
            # ===== Top bar =====
            html.Div(
                [
                    html.Div(
                        "Surface Dynamics",
                        style={
                            "fontWeight": "600",
                            "fontSize": "20px",
                            "color": "#e5e7eb",
                            "textAlign": "center",
                            "width": "100%",
                        },
                    ),
                    html.A(
                        "Home",
                        href="https://blog.surfacedynamics.io",
                        style={
                            "color": "#93c5fd",
                            "textDecoration": "none",
                            "fontWeight": "500",
                            "padding": "4px 10px",
                            "borderRadius": "6px",
                            "border": "1px solid #1f2937",
                            "position": "absolute",
                            "right": "16px",
                            "top": "50%",
                            "transform": "translateY(-50%)",
                        },
                    ),
                ],
                style={
                    "position": "relative",
                    "padding": "8px 16px",
                    "borderBottom": "1px solid #1f2937",
                    "marginBottom": "8px",
                },
            ),
            dcc.Interval(id=CLOCK_ID, interval=60_000, n_intervals=0),
            # ===== Tabs selector (styled) =====
            html.Div(
                [
                    dcc.Tabs(
                        id=MAIN_TABS_ID,
                        value=TAB_DASHBOARD,
                        colors={"border": "#1f2937", "primary": "#60a5fa", "background": "#0b1220"},
                        style=TABS_STYLE,
                        children=[
                            dcc.Tab(label="Dashboard", value=TAB_DASHBOARD, style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                            dcc.Tab(label="Price Chart", value=TAB_PRICE_CHART, style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                            dcc.Tab(label="Backtests", value=TAB_BACKTESTS, style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                            dcc.Tab(label="Backtests v2", value=TAB_BACKTESTS_V2, style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                        ],
                    )
                ],
                style=TABS_WRAP_STYLE,
            ),
            # ===== Dashboard container =====
            html.Div(id="dashboard-container", children=dashboard_children, style={"display": "block"}),
            # ===== Ironbeam container =====
            html.Div(id="ironbeam-container", children=ironbeam_children, style={"display": "none"}),
            # ===== Backtests container (existing) =====
            html.Div(id="backtests-container", children=make_backtests_tab(bt_start, bt_end), style={"display": "none"}),
            # ===== Backtests v2 container (new) =====
            html.Div(id="backtests-v2-container", children=backtests_v2_children, style={"display": "none"}),
        ],
        style={"backgroundColor": "black", "color": "white", "minHeight": "100vh", "padding": "0 80px 30px"},
    )


app.layout = serve_layout


@app.callback(
    Output(EXPIRATION_DATE_PICK, "date"),
    Input(TRADE_DATE_PICK, "date"),
)
def sync_expiration_with_trade(trade_date):
    return trade_date


@app.callback(
    Output("dashboard-container", "style"),
    Output("ironbeam-container", "style"),
    Output("backtests-container", "style"),
    Output("backtests-v2-container", "style"),
    Input(MAIN_TABS_ID, "value"),
)
def _switch_main_tab(tab_value):
    if tab_value == TAB_BACKTESTS:
        return {"display": "none"}, {"display": "none"}, {"display": "block"}, {"display": "none"}
    if tab_value == TAB_BACKTESTS_V2:
        return {"display": "none"}, {"display": "none"}, {"display": "none"}, {"display": "block"}
    if tab_value == TAB_PRICE_CHART:
        return {"display": "none"}, {"display": "block"}, {"display": "none"}, {"display": "none"}
    return {"display": "block"}, {"display": "none"}, {"display": "none"}, {"display": "none"}


# ===== Backtests v2 loader callback (NEW) =====
@app.callback(
    Output("bt2-table", "data"),
    Output("bt2-table", "columns"),
    Output("bt2-summary", "children"),
    Input("bt2-load-btn", "n_clicks"),
    State("bt2-date-range", "start_date"),
    State("bt2-date-range", "end_date"),
    State("bt2-rth-only", "value"),
    State("bt2-limit-rows", "value"),
)
def _bt2_load_rows(n_clicks, start_date, end_date, rth_only_value, limit_rows):
    if not n_clicks:
        # show instructions until the first click
        return [], [], "Click Load to pull rows from the new backtest view."

    if not start_date or not end_date:
        return [], [], "Pick a start + end date, then click Load."

    rth_only = (rth_only_value == "yes")
    limit_rows = int(limit_rows or 800)

    try:
        df = _bt2_fetch_rows(start_date, end_date, rth_only, limit_rows)
    except Exception as e:
        return [], [], f"Error loading data from {BT_VIEW_NAME}: {e}"

    if df.empty:
        return [], [], f"No rows returned from {BT_VIEW_NAME} for {start_date} → {end_date} (rth_only={rth_only})."

    cols = [{"name": c, "id": c} for c in df.columns]
    summary = (
        f"Loaded {len(df):,} rows from {BT_VIEW_NAME} | "
        f"{df['ts_pt'].min()} → {df['ts_pt'].max()} | "
        f"trade_date range: {start_date} → {end_date} | rth_only={rth_only}"
    )
    return df.to_dict("records"), cols, summary
def _bt2_fetch_training_features(strategy_id: int) -> pd.DataFrame:
    """
    Load accepted (label=1) training features for a strategy into a numeric dataframe.
    """
    engine = _get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            text(
                """
                SELECT features
                FROM bt_strategy_instances
                WHERE strategy_id = :sid AND label = 1
                ORDER BY id
                """
            ),
            {"sid": int(strategy_id)},
        ).fetchall()

    feats = [r[0] for r in rows]  # JSONB -> dict
    if not feats:
        return pd.DataFrame()

    df = pd.DataFrame(feats)

    # force numeric where possible
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _bt2_compute_feature_dict(anchor_row: dict, entry_row: dict) -> dict:
    """
    Compute the same v1 features we store when you Save Example.
    anchor_row / entry_row are dict rows from the view.
    """
    a_close = anchor_row.get("close")
    e_close = entry_row.get("close")

    # safe return
    ret = None
    if a_close not in (None, 0) and e_close is not None:
        try:
            ret = (float(e_close) / float(a_close)) - 1.0
        except Exception:
            ret = None

    out = {
        "px_chg": None if (a_close is None or e_close is None) else (float(e_close) - float(a_close)),
        "ret": ret,
        "atmiv_chg": None
        if (anchor_row.get("atmiv") is None or entry_row.get("atmiv") is None)
        else (float(entry_row["atmiv"]) - float(anchor_row["atmiv"])),
        "put_skew_pp_chg": None
        if (anchor_row.get("put_skew_pp_primary") is None or entry_row.get("put_skew_pp_primary") is None)
        else (float(entry_row["put_skew_pp_primary"]) - float(anchor_row["put_skew_pp_primary"])),
        "call_skew_pp_chg": None
        if (anchor_row.get("call_skew_pp_primary") is None or entry_row.get("call_skew_pp_primary") is None)
        else (float(entry_row["call_skew_pp_primary"]) - float(anchor_row["call_skew_pp_primary"])),
        "net_gex": entry_row.get("net_gex"),
        "is_rth": entry_row.get("is_rth"),
        "dist_wall_above": None
        if (entry_row.get("gex_wall_above") is None or e_close is None)
        else (float(entry_row["gex_wall_above"]) - float(e_close)),
        "dist_wall_below": None
        if (entry_row.get("gex_wall_below") is None or e_close is None)
        else (float(e_close) - float(entry_row["gex_wall_below"])),
        "es_close_entry": e_close,
        "spx_stock_entry": entry_row.get("stock_price"),
    }
    return out


@app.callback(
    Output("bt2-cands-table", "data"),
    Output("bt2-cands-table", "columns"),
    Output("bt2-cands-store", "data"),
    Output("bt2-find-status", "children"),
    Input("bt2-find-similar", "n_clicks"),
    State("bt2-strategy-name", "value"),
    State("bt2-date-range", "start_date"),
    State("bt2-date-range", "end_date"),
    State("bt2-rth-only", "value"),
    State("bt2-lookahead-min", "value"),
    State("bt2-stride-min", "value"),
    State("bt2-topn", "value"),
)
def _bt2_find_similar(n, strategy_name, start_date, end_date, rth_only_value, lookahead_min, stride_min, topn):
    if not n:
        raise PreventUpdate

    try:
        if not start_date or not end_date:
            return [], [], [], "Pick a date range first."

        sid = _ensure_strategy_id(strategy_name)

        train_df = _bt2_fetch_training_features(sid)
        if train_df.empty:
            return [], [], [], "No saved examples yet for this strategy. Save a couple first."

        feat_cols = ["ret", "atmiv_chg", "put_skew_pp_chg", "call_skew_pp_chg", "dist_wall_above", "dist_wall_below"]
        feat_cols = [c for c in feat_cols if c in train_df.columns]
        REQUIRED = ["atmiv_chg", "put_skew_pp_chg", "call_skew_pp_chg"]

        if not feat_cols:
            return [], [], [], "Training examples exist, but no usable feature columns were found."

        mu = train_df[feat_cols].mean(numeric_only=True)
        sigma = train_df[feat_cols].std(numeric_only=True, ddof=0).replace(0, 1.0)

        # If training data can't support required fields, stop early with a clear message.
        for k in REQUIRED:
            if k not in mu.index or pd.isna(mu[k]):
                return [], [], [], f"Strategy has accepted examples, but required feature '{k}' is missing/NULL. Save examples with valid skew + ATM IV first."

        rth_only = (rth_only_value == "yes")
        lookahead_min = int(lookahead_min or 5)
        stride_min = int(stride_min or 1)
        topn = int(topn or 50)

        df = _bt2_fetch_rows(start_date, end_date, rth_only, limit_rows=50000)
        if df.empty:
            return [], [], [], "No rows in that range."

        # Normalize to UTC minute timestamps so entry lookup works
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True).dt.floor("min")
        df = df.sort_values("ts_utc").reset_index(drop=True)

        rows = df.to_dict("records")
        by_ts = {r["ts_utc"]: r for r in rows}

        cands = []
        for i in range(0, len(rows), stride_min):
            a = rows[i]
            a_ts = a["ts_utc"]
            e_ts = (a_ts + pd.Timedelta(minutes=lookahead_min)).floor("min")

            e = by_ts.get(e_ts)
            if not e:
                continue

            feats = _bt2_compute_feature_dict(a, e)

            # Require ATM IV change + both skew changes
            missing_req = [k for k in REQUIRED if feats.get(k) is None or pd.isna(feats.get(k))]
            if missing_req:
                continue

            zsum = 0.0
            used = 0
            for c in feat_cols:
                v = feats.get(c)
                if v is None or pd.isna(v) or pd.isna(mu[c]):
                    continue
                z = (float(v) - float(mu[c])) / float(sigma[c])
                zsum += z * z
                used += 1
            if used == 0:
                continue

            dist = (zsum ** 0.5)
            score = 100.0 / (1.0 + dist)

            cands.append(
                {
                    "score": round(score, 2),
                    "anchor_ts_utc": a_ts.isoformat(),
                    "entry_ts_utc": e_ts.isoformat(),
                    "ret": feats.get("ret"),
                    "atmiv_chg": feats.get("atmiv_chg"),
                    "put_skew_pp_chg": feats.get("put_skew_pp_chg"),
                    "call_skew_pp_chg": feats.get("call_skew_pp_chg"),
                    "dist_wall_above": feats.get("dist_wall_above"),
                    "dist_wall_below": feats.get("dist_wall_below"),
                }
            )

        if not cands:
            return [], [], [], "No candidates found. Try stride=1, smaller lookahead, or widen date range."

        cands = sorted(cands, key=lambda x: x["score"], reverse=True)[:topn]
        cand_df = pd.DataFrame(cands)

        cols = [{"name": c, "id": c} for c in cand_df.columns]
        status = f"Found {len(cands)} candidates | strategy_id={sid} | lookahead={lookahead_min}m | stride={stride_min}m"

        return cand_df.to_dict("records"), cols, cands, status

    except Exception as ex:
        return [], [], [], f"Find Similar error: {ex}"

_PT = ZoneInfo("America/Los_Angeles")


def _utc_to_pt_hhmm(ts_utc_val) -> str:
    ts = pd.to_datetime(ts_utc_val, utc=True)
    ts_pt = ts.tz_convert(_PT)
    return ts_pt.strftime("%H:%M")

def _utc_to_pt_date(ts_utc_val) -> str:
    ts = pd.to_datetime(ts_utc_val, utc=True).tz_convert(_PT)
    return ts.date().isoformat()



def _bt2_fetch_saved_instances(strategy_id: int, limit_rows: int = 100) -> pd.DataFrame:
    engine = _get_engine()
    with engine.connect() as conn:
        df = pd.read_sql(
            text(
                """
                SELECT
                  id,
                  label,
                  trade_date,
                  anchor_ts_utc,
                  entry_ts_utc,
                  created_at,
                  features->>'ret' AS ret,
                  features->>'atmiv_chg' AS atmiv_chg,
                  features->>'put_skew_pp_chg' AS put_skew_pp_chg,
                  features->>'call_skew_pp_chg' AS call_skew_pp_chg,
                  features->>'dist_wall_above' AS dist_wall_above,
                  features->>'dist_wall_below' AS dist_wall_below
                FROM bt_strategy_instances
                WHERE strategy_id = :sid AND label IN (0, 1)
                ORDER BY id DESC
                LIMIT :lim
                """
            ),
            conn,
            params={"sid": int(strategy_id), "lim": int(limit_rows)},
        )
    return df

@app.callback(
    Output("bt2-strategy-name", "options"),
    Output("bt2-strategy-name", "value"),
    Output("bt2-strategy-status", "children"),
    Input("bt2-create-strategy", "n_clicks"),
    State("bt2-new-strategy-name", "value"),
    prevent_initial_call=True,
)
def _bt2_create_strategy(n, new_name):
    if not n:
        raise PreventUpdate

    name = (new_name or "").strip()
    if not name:
        return no_update, no_update, "Enter a name and click Create."

    try:
        _ensure_strategy_id(name)
        opts = _bt2_list_strategy_options()
        return opts, name, f"Selected strategy: {name}"
    except Exception as ex:
        return no_update, no_update, f"Create strategy error: {ex}"




@app.callback(
    Output("bt2-saved-table", "data"),
    Output("bt2-saved-table", "columns"),
    Output("bt2-saved-status", "children"),
    Input("bt2-strategy-name", "value"),
    Input("bt2-refresh-saved", "n_clicks"),
)
def _bt2_refresh_saved(strategy_name, n):
    try:
        if not (strategy_name or '').strip():
            return [], [], "Select a strategy."
        sid = _ensure_strategy_id(strategy_name)
        df = _bt2_fetch_saved_instances(sid, limit_rows=200)
        if df.empty:
            return [], [], f"No saved instances yet for '{strategy_name}'."

        cols = [{"name": c, "id": c} for c in df.columns]
        return df.to_dict("records"), cols, f"Loaded {len(df)} saved instances for '{strategy_name}' (strategy_id={sid})."
    except Exception as ex:
        return [], [], f"Refresh Saved error: {ex}"


@app.callback(
    Output(MAIN_TABS_ID, "value", allow_duplicate=True),
    Output(TRADE_DATE_PICK, "date", allow_duplicate=True),
    Output(SMILE_TIME_INPUT, "value", allow_duplicate=True),
    Output("bt2-load-dash-status", "children"),
    Input("bt2-load-saved-to-dash", "n_clicks"),
    State("bt2-saved-table", "data"),
    State("bt2-saved-table", "selected_rows"),
    prevent_initial_call=True,
)
def _bt2_load_saved_to_dashboard(n, rows, selected_rows):
    # If no row is selected, show a message (don't silently do nothing)
    if not rows or not selected_rows:
        return no_update, no_update, no_update, "Select a saved example row first, then click Load on Dashboard."

    try:
        r = rows[selected_rows[0]]

        trade_date = r.get("trade_date")
        anchor_ts = r.get("anchor_ts_utc")
        entry_ts = r.get("entry_ts_utc")

        if not trade_date or not anchor_ts or not entry_ts:
            return no_update, no_update, no_update, "Selected saved row is missing trade_date/anchor/entry."

        trade_date_str = pd.to_datetime(trade_date).date().isoformat()

        a_hhmm = _utc_to_pt_hhmm(anchor_ts)
        e_hhmm = _utc_to_pt_hhmm(entry_ts)

        times = sorted(list({a_hhmm, e_hhmm}))

        status = f"Loaded saved id={r.get('id')} → trade_date={trade_date_str}, times={times} (PT)."
        return TAB_DASHBOARD, trade_date_str, times, status

    except Exception as ex:
        return no_update, no_update, no_update, f"Load on Dashboard error: {ex}"


@app.callback(
    Output(MAIN_TABS_ID, "value", allow_duplicate=True),
    Output(TRADE_DATE_PICK, "date", allow_duplicate=True),
    Output(SMILE_TIME_INPUT, "value", allow_duplicate=True),
    Output("bt2-load-cand-status", "children"),
    Input("bt2-load-cand-to-dash", "n_clicks"),
    State("bt2-cands-table", "data"),
    State("bt2-cands-table", "selected_rows"),
    prevent_initial_call=True,
)
def _bt2_load_candidate_to_dashboard(n, rows, selected_rows):
    if not rows or not selected_rows:
        return no_update, no_update, no_update, "Select a candidate row first, then click Load Candidate on Dashboard."

    try:
        r = rows[selected_rows[0]]
        anchor_ts = r.get("anchor_ts_utc")
        entry_ts = r.get("entry_ts_utc")
        if not anchor_ts or not entry_ts:
            return no_update, no_update, no_update, "Selected candidate missing anchor_ts_utc/entry_ts_utc."

        # Derive trade date from anchor timestamp in PT
        trade_date_str = _utc_to_pt_date(anchor_ts)

        a_hhmm = _utc_to_pt_hhmm(anchor_ts)
        e_hhmm = _utc_to_pt_hhmm(entry_ts)

        times = sorted(list({a_hhmm, e_hhmm}))

        status = f"Loaded candidate → trade_date={trade_date_str}, times={times} (PT)."
        return TAB_DASHBOARD, trade_date_str, times, status

    except Exception as ex:
        return no_update, no_update, no_update, f"Load Candidate error: {ex}"






# Register module callbacks
register_smile(app)
register_skew(app)
register_term_structure(app)
register_term_metrics(app)
register_ironbeam_callbacks(app)
register_backtests(app, expected_toggle_id=EXPECTED_TOGGLE_ID)

from dash import callback_context  # if not already imported


@app.callback(
    Output("bt2-cand-action-status", "children"),
    Input("bt2-accept-cand", "n_clicks"),
    Input("bt2-reject-cand", "n_clicks"),
    State("bt2-cands-table", "data"),
    State("bt2-cands-table", "selected_rows"),
    State("bt2-strategy-name", "value"),
    prevent_initial_call=True,
)
def _bt2_accept_reject_candidate(n_acc, n_rej, rows, selected_rows, strategy_name):
    trig = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else None
    if not rows or not selected_rows:
        return "Select a candidate row first."

    r = rows[selected_rows[0]]
    anchor_ts = r.get("anchor_ts_utc")
    entry_ts = r.get("entry_ts_utc")
    if not anchor_ts or not entry_ts:
        return "Selected candidate missing anchor_ts_utc/entry_ts_utc."

    # Decide label
    if trig == "bt2-accept-cand":
        label = 1
    elif trig == "bt2-reject-cand":
        label = -1
    else:
        raise PreventUpdate

    try:
        sid = _ensure_strategy_id(strategy_name)

        # Ensure ordering
        a = pd.to_datetime(anchor_ts, utc=True)
        e = pd.to_datetime(entry_ts, utc=True)
        if e < a:
            a, e = e, a
        anchor_ts = a.isoformat()
        entry_ts = e.isoformat()

        new_id = _save_strategy_instance(sid, anchor_ts, entry_ts, label=label)
        action = "ACCEPTED ✅" if label == 1 else "REJECTED ❌"
        return f"{action} candidate saved as instance_id={new_id} (strategy_id={sid})."

    except Exception as ex:
        return f"Accept/Reject error: {ex}"



@app.callback(
    Output("bt2-anchor-ts-utc", "data"),
    Output("bt2-entry-ts-utc", "data"),
    Output("bt2-anchor-entry-status", "children"),
    Input("bt2-set-anchor", "n_clicks"),
    Input("bt2-set-entry", "n_clicks"),
    State("bt2-table", "data"),
    State("bt2-table", "selected_rows"),
    State("bt2-anchor-ts-utc", "data"),
    State("bt2-entry-ts-utc", "data"),
    prevent_initial_call=True,
)
def _bt2_set_anchor_or_entry(n_anchor, n_entry, rows, selected_rows, anchor_ts, entry_ts):
    # Which button fired?
    trig = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else None

    if not rows or not selected_rows:
        msg = "Select a row in the table first."
        return anchor_ts, entry_ts, msg

    r = rows[selected_rows[0]]
    ts_utc = r.get("ts_utc")
    if ts_utc is None:
        return anchor_ts, entry_ts, "Selected row has no ts_utc."

    ts_utc = _to_iso_utc(ts_utc)

    if trig == "bt2-set-anchor":
        anchor_ts = ts_utc
    elif trig == "bt2-set-entry":
        entry_ts = ts_utc
    else:
        raise PreventUpdate

    msg = f"Anchor: {anchor_ts or '(not set)'} | Entry: {entry_ts or '(not set)'}"
    return anchor_ts, entry_ts, msg



@app.callback(
    Output("bt2-save-status", "children"),
    Input("bt2-save-example", "n_clicks"),
    State("bt2-strategy-name", "value"),
    State("bt2-anchor-ts-utc", "data"),
    State("bt2-entry-ts-utc", "data"),
)
def _bt2_save_example(n, strategy_name, anchor_ts, entry_ts):
    if not n:
        raise PreventUpdate
    if not anchor_ts or not entry_ts:
        return "Set both Anchor and Entry first."

    # Ensure ordering (if user picked them backwards, auto-fix)
    a = pd.to_datetime(anchor_ts, utc=True)
    e = pd.to_datetime(entry_ts, utc=True)
    if e < a:
        a, e = e, a
        anchor_ts = a.isoformat()
        entry_ts = e.isoformat()

    try:
        sid = _ensure_strategy_id(strategy_name)
        instance_id = _save_strategy_instance(sid, anchor_ts, entry_ts, label=1)
        return f"✅ Saved instance {instance_id} to strategy '{strategy_name}' (strategy_id={sid})."
    except Exception as ex:
        return f"Error saving instance: {ex}"



if __name__ == "__main__":
    import socket

    def _choose_port(preferred=8050, tries=50):
        p = os.getenv("PORT")
        if p:
            return int(p)
        for port in [preferred] + list(range(preferred + 1, preferred + tries)):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("0.0.0.0", port))
                    return port
                except OSError:
                    continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    app.run(host="0.0.0.0", port=_choose_port(), debug=True)
