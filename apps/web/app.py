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
import re
import socket
import datetime as dt
from typing import List

import dash_auth
from dash import Dash, html, dcc, Input, Output, State, no_update
from dash.exceptions import PreventUpdate
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
from modules.BacktestsV2.components import make_backtests_v2_tab
from modules.BacktestsV2.routes import (
    register_backtests_v2_routes,
    get_backtests_v2_selection_since,
)

# ===== IDs =====
CLOCK_ID = "CLOCK"
TRADE_DATE_PICK = "trade-date"
EXPIRATION_DATE_PICK = "expiration-date-pick"
SMILE_TIME_INPUT = "smile-time-input"
SMILE_GRAPH = "SMILE_GRAPH"
EXPECTED_TOGGLE_ID = "expected-ss-toggle"
LIVE_DATA_STORE_ID = "live-data-store"
LIVE_UPDATE_TIMER_ID = "live-update-timer"

BT2_SELECTION_POLL_ID = "bt2-selection-poll"
BT2_SELECTION_SEQ_ID = "bt2-selection-seq"

# Tabs (selector only; containers below stay mounted)
MAIN_TABS_ID = "main-tabs"
TAB_DASHBOARD = "tab-dashboard"
TAB_PRICE_CHART = "tab-price-chart"
TAB_BACKTESTS = "tab-backtests"

# ---- Tabs styling ----
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
    "color": "#93c5fd",
    "fontWeight": "700",
    "fontSize": "13px",
}

TAB_SELECTED_STYLE = {
    "backgroundColor": "#111827",
    "border": "1px solid #60a5fa",
    "padding": "10px 16px",
    "borderRadius": "12px",
    "color": "#bfdbfe",
    "fontWeight": "800",
    "fontSize": "13px",
}


def get_default_trade_date() -> dt.date:
    """
    Return 'today' if it's Monday–Friday.
    If it's Saturday/Sunday, return Friday.
    """
    today = dt.date.today()
    if today.weekday() == 5:
        return today - dt.timedelta(days=1)
    if today.weekday() == 6:
        return today - dt.timedelta(days=2)
    return today


def pt_time_options(start: str = "06:30", end: str = "13:00", step_min: int = 1) -> List[dict]:
    t0, t1 = dt.datetime.strptime(start, "%H:%M"), dt.datetime.strptime(end, "%H:%M")
    out, cur = [], t0
    while cur <= t1:
        hhmm = cur.strftime("%H:%M")
        out.append({"label": f"{hhmm} PT", "value": hhmm})
        cur += dt.timedelta(minutes=step_min)
    return out


def _parse_hhmm(value: object) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    m = re.search(r"(\d{1,2}):(\d{2})", s)
    if not m:
        return None
    hh = int(m.group(1))
    mm = m.group(2)
    if hh < 0 or hh > 23:
        return None
    return f"{hh:02d}:{mm}"


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

# Mount the React Backtests app + API routes
register_backtests_v2_routes(server, REPO_ROOT)

# Mount the React Price Chart preview
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

    dashboard_children = [
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
        html.Div(
            [
                html.Div(dcc.Graph(id=SMILE_GRAPH, style={"height": "100%"}), style={"minWidth": 0, "flex": "2"}),
                html.Div(gex_block(), style={"minWidth": 0, "flex": "1"}),
            ],
            style={"display": "flex", "gap": "16px", "alignItems": "stretch", "height": "calc(60vh - 100px)"},
        ),
        html.Hr(style={"borderColor": "#333"}),
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

    ironbeam_children = [
        html.Div(
            [
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
                                    value="react_preview",
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
                ironbeam_layout(),
            ],
            style={
                "display": "flex",
                "flexDirection": "column",
                "height": "100%",
                "width": "100%",
                "minHeight": 0,
            },
        )
    ]

    return html.Div(
        [
            dcc.Store(id=LIVE_DATA_STORE_ID),
            dcc.Interval(id=LIVE_UPDATE_TIMER_ID, interval=15 * 1000, n_intervals=0),
            dcc.Store(id=BT2_SELECTION_SEQ_ID, data=0),
            dcc.Interval(id=BT2_SELECTION_POLL_ID, interval=750, n_intervals=0),
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
                        ],
                    )
                ],
                style=TABS_WRAP_STYLE,
            ),
            html.Div(
                id="dashboard-container",
                children=dashboard_children,
                style={"display": "block", "flex": "1 1 auto", "minHeight": 0, "overflowY": "auto"},
            ),
            html.Div(
                id="ironbeam-container",
                children=ironbeam_children,
                style={"display": "none", "flex": "1 1 auto", "minHeight": 0, "width": "100%"},
            ),
            html.Div(
                id="backtests-container",
                children=[make_backtests_v2_tab()],
                style={"display": "none", "flex": "1 1 auto", "minHeight": 0, "overflow": "hidden"},
            ),
        ],
        style={
            "backgroundColor": "black",
            "color": "white",
            "height": "100vh",
            "minHeight": "100vh",
            "padding": "0 24px 12px",
            "display": "flex",
            "flexDirection": "column",
            "overflow": "hidden",
            "width": "100%",
        },
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
    Input(MAIN_TABS_ID, "value"),
)
def _switch_main_tab(tab_value):
    hidden = {"display": "none"}
    scrollable = {"display": "block", "flex": "1 1 auto", "minHeight": 0, "overflowY": "auto"}
    price_chart = {"display": "flex", "flex": "1 1 auto", "minHeight": 0, "overflow": "hidden", "width": "100%"}
    backtests = {"display": "block", "flex": "1 1 auto", "minHeight": 0, "overflow": "hidden"}

    if tab_value == TAB_BACKTESTS:
        return hidden, hidden, backtests
    if tab_value == TAB_PRICE_CHART:
        return hidden, price_chart, hidden
    return scrollable, hidden, hidden


@app.callback(
    Output(BT2_SELECTION_SEQ_ID, "data"),
    Output(TRADE_DATE_PICK, "date"),
    Output(SMILE_TIME_INPUT, "value"),
    Output(MAIN_TABS_ID, "value"),
    Input(BT2_SELECTION_POLL_ID, "n_intervals"),
    State(BT2_SELECTION_SEQ_ID, "data"),
    prevent_initial_call=True,
)
def apply_backtests_selection(_n_intervals, last_seq):
    seq, payload = get_backtests_v2_selection_since(last_seq)

    if payload is None:
        raise PreventUpdate

    trade_date = payload.get("trade_date")
    start_time = _parse_hhmm(payload.get("start_ts_pt"))
    target_time = _parse_hhmm(payload.get("target_ts_pt"))

    times = []
    for t in [start_time, target_time]:
        if t and t not in times:
            times.append(t)

    out_trade_date = trade_date if trade_date else no_update
    out_times = times if times else no_update

    return seq, out_trade_date, out_times, TAB_PRICE_CHART


# Register module callbacks
register_smile(app)
register_skew(app)
register_term_structure(app)
register_term_metrics(app)
register_ironbeam_callbacks(app)


if __name__ == "__main__":
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
