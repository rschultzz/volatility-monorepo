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
from dash import Dash, html, dcc, Input, Output
import dash_auth

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

# ===== Backtests (NEW) =====
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


# ===== App setup =====
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

server.secret_key = os.getenv("SECRET_KEY", "dev-secret-key")

VALID_USERNAME_PASSWORD_PAIRS = {
    "ryan": "ChangeThisPassword123!",
    "sara": "ChangeThisPassword123!",
}
auth = dash_auth.BasicAuth(app, VALID_USERNAME_PASSWORD_PAIRS)


def serve_layout():
    """Layout factory so defaults are recomputed on each page load."""
    default_trade_date = get_default_trade_date()
    time_options = pt_time_options()

    # Backtests default range
    bt_end = default_trade_date
    bt_start = default_trade_date - dt.timedelta(days=10)

    # ----- DASHBOARD BODY (this matches your original layout) -----
    dashboard_children = [
        # ===== Global controls (date / expiration / time slices / expected) =====
        html.Div(
            [
                html.Div(
                    [
                        html.Label(
                            "Trade Date",
                            style={"color": "white", "marginBottom": "4px"},
                        ),
                        dcc.DatePickerSingle(
                            id=TRADE_DATE_PICK,
                            disabled=False,
                            display_format="YYYY-MM-DD",
                            date=default_trade_date,
                        ),
                    ],
                    style={
                        "marginRight": "16px",
                        "display": "flex",
                        "flexDirection": "column",
                    },
                ),
                html.Div(
                    [
                        html.Label(
                            "Expiration",
                            style={"color": "white", "marginBottom": "4px"},
                        ),
                        dcc.DatePickerSingle(
                            id=EXPIRATION_DATE_PICK,
                            disabled=False,
                            display_format="YYYY-MM-DD",
                            date=default_trade_date,
                        ),
                    ],
                    style={
                        "marginRight": "16px",
                        "display": "flex",
                        "flexDirection": "column",
                    },
                ),
                html.Div(
                    [
                        html.Label(
                            "Time Slices (PT)",
                            style={"color": "white", "marginBottom": "4px"},
                        ),
                        # IMPORTANT: multi=True and list value (Smile/Skew callbacks expect list)
                        dcc.Dropdown(
                            id=SMILE_TIME_INPUT,
                            options=time_options,
                            value=["06:31"],
                            multi=True,
                            style={"minWidth": "320px"},
                        ),
                    ],
                    style={
                        "marginRight": "16px",
                        "display": "flex",
                        "flexDirection": "column",
                    },
                ),
                html.Div(
                    [
                        html.Label(
                            "Compare to Expected (SS)",
                            style={"color": "white", "marginBottom": "4px"},
                        ),
                        dcc.RadioItems(
                            id=EXPECTED_TOGGLE_ID,
                            options=[
                                {"label": "ON", "value": "on"},
                                {"label": "OFF", "value": "off"},
                            ],
                            value="on",
                            inline=True,
                            labelStyle={
                                "marginRight": "16px",
                                "color": "white",
                                "fontSize": "13px",
                            },
                            inputStyle={"marginRight": "6px"},
                        ),
                    ],
                    style={
                        "marginRight": "16px",
                        "display": "flex",
                        "flexDirection": "column",
                    },
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
                html.Div(
                    dcc.Graph(id=SMILE_GRAPH, style={"height": "840px"}),
                    style={"minWidth": 0},
                ),
                html.Div(gex_block(), style={"minWidth": 0}),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "2fr 1fr",
                "gap": "16px",
                "alignItems": "stretch",
            },
        ),
        html.Hr(style={"borderColor": "#333"}),

        # ===== Term structure / Skew / Term metrics =====
        html.Div(
            [
                make_term_structure_block(),
                html.Div(
                    [
                        make_skew_block(),
                        make_term_metrics_block(),
                    ]
                ),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr 1fr",
                "gap": "16px",
                "alignItems": "stretch",
            },
        ),
    ]

    # ----- IRONBEAM BODY -----
    ironbeam_children = [
        # ===== Ironbeam section + GEX threshold + Bar Interval toggle =====
        html.Div(
            [
                # Top row: GEX slider + Bar Interval toggle side-by-side
                html.Div(
                    [
                        # GEX threshold slider
                        html.Div(
                            [
                                html.Label(
                                    "Min |Net GEX| (Billions)",
                                    style={
                                        "color": "white",
                                        "marginBottom": "6px",
                                        "fontSize": "13px",
                                        "fontWeight": "500",
                                    },
                                ),
                                dcc.Slider(
                                    id="gex-threshold-billions",
                                    min=0,
                                    max=300,
                                    step=10,
                                    value=100,
                                    marks={
                                        0: {"label": "0", "style": {"fontSize": "11px"}},
                                        50: {"label": "50", "style": {"fontSize": "11px"}},
                                        100: {"label": "100", "style": {"fontSize": "11px"}},
                                        150: {"label": "150", "style": {"fontSize": "11px"}},
                                        200: {"label": "200", "style": {"fontSize": "11px"}},
                                        250: {"label": "250", "style": {"fontSize": "11px"}},
                                        300: {"label": "300", "style": {"fontSize": "11px"}},
                                    },
                                ),
                            ],
                            style={"minWidth": "320px", "flex": "1 1 auto"},
                        ),
                        # Bar interval toggle
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
                                    labelStyle={
                                        "marginRight": "16px",
                                        "color": "white",
                                        "fontSize": "13px",
                                    },
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
        ),
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
                        colors={
                            "border": "#1f2937",
                            "primary": "#60a5fa",
                            "background": "#0b1220",
                        },
                        style=TABS_STYLE,
                        children=[
                            dcc.Tab(
                                label="Dashboard",
                                value=TAB_DASHBOARD,
                                style=TAB_STYLE,
                                selected_style=TAB_SELECTED_STYLE,
                            ),
                            dcc.Tab(
                                label="Price Chart",
                                value=TAB_PRICE_CHART,
                                style=TAB_STYLE,
                                selected_style=TAB_SELECTED_STYLE,
                            ),
                            dcc.Tab(
                                label="Backtests",
                                value=TAB_BACKTESTS,
                                style=TAB_STYLE,
                                selected_style=TAB_SELECTED_STYLE,
                            ),
                        ],
                    )
                ],
                style=TABS_WRAP_STYLE,
            ),

            # ===== Dashboard container =====
            html.Div(id="dashboard-container", children=dashboard_children, style={"display": "block"}),

            # ===== Ironbeam container =====
            html.Div(id="ironbeam-container", children=ironbeam_children, style={"display": "none"}),

            # ===== Backtests container =====
            html.Div(
                id="backtests-container",
                children=make_backtests_tab(bt_start, bt_end),
                style={"display": "none"},
            ),
        ],
        style={
            "backgroundColor": "black",
            "color": "white",
            "minHeight": "100vh",
            "padding": "0 80px 30px",
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
    if tab_value == TAB_BACKTESTS:
        return {"display": "none"}, {"display": "none"}, {"display": "block"}
    if tab_value == TAB_PRICE_CHART:
        return {"display": "none"}, {"display": "block"}, {"display": "none"}
    return {"display": "block"}, {"display": "none"}, {"display": "none"}


# Register module callbacks
register_smile(app)
register_skew(app)
register_term_structure(app)
register_term_metrics(app)
register_ironbeam_callbacks(app)
register_backtests(app, expected_toggle_id=EXPECTED_TOGGLE_ID)


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
