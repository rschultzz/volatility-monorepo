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

# ===== IDs =====
CLOCK_ID = "CLOCK"
TRADE_DATE_PICK = "trade-date"
EXPIRATION_DATE_PICK = "expiration-date-pick"
SMILE_TIME_INPUT = "smile-time-input"
SMILE_GRAPH = "SMILE_GRAPH"
EXPECTED_TOGGLE_ID = "expected-ss-toggle"
LIVE_DATA_STORE_ID = "live-data-store"
LIVE_UPDATE_TIMER_ID = "live-update-timer"
TAB_SWITCH_ID = "main-tab-switch"


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


def third_friday_of_month(year, month):
    for day in range(15, 22):
        d = dt.date(year, month, day)
        if d.weekday() == 4:
            return d
    return None


def third_friday_next_month() -> dt.date:
    today = dt.date.today()
    y, m = (today.year, today.month + 1)
    if m > 12:
        y, m = y + 1, 1
    return third_friday_of_month(y, m)


def pt_time_options(start="06:30", end="13:00", step_min=1) -> List[dict]:
    t0, t1 = dt.datetime.strptime(start, "%H:%M"), dt.datetime.strptime(end, "%H:%M")
    out, cur = [], t0
    while cur <= t1:
        hhmm = cur.strftime("%H:%M")
        out.append({"label": f"{hhmm} PT", "value": hhmm})
        cur += dt.timedelta(minutes=step_min)
    return out


# ===== App =====
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# --- Flask secret key for sessions (required by dash_auth and session storage) ---
server.secret_key = os.getenv("SECRET_KEY", "dev-secret-key")
# In production (Render), set SECRET_KEY as an environment variable.

VALID_USERNAME_PASSWORD_PAIRS = {
    "ryan": "ChangeThisPassword123!",
    "sara": "ChangeThisPassword123!",
}
auth = dash_auth.BasicAuth(app, VALID_USERNAME_PASSWORD_PAIRS)


def serve_layout():
    """Layout factory so defaults are recomputed on each page load."""
    # Weekday -> today, Weekend -> Friday
    default_trade_date = get_default_trade_date()
    time_options = pt_time_options()

    # ===== Analytics section (Smile / GEX / Term / Skew / Term metrics) =====
    analytics_section = html.Div(
        id="analytics-section",
        children=[
            # Smile + GEX block
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

            # Term structure / Skew / Term metrics
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
        ],
    )

    # ===== Price section (Ironbeam + GEX threshold + bar interval) =====
    price_section = html.Div(
        id="price-section",
        hidden=True,  # start hidden; tab switch callback toggles this
        children=[
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
                                        max=300,  # 0–300B
                                        step=10,
                                        value=100,
                                        marks={
                                            0: {
                                                "label": "0",
                                                "style": {"fontSize": "11px"},
                                            },
                                            50: {
                                                "label": "50",
                                                "style": {"fontSize": "11px"},
                                            },
                                            100: {
                                                "label": "100",
                                                "style": {"fontSize": "11px"},
                                            },
                                            150: {
                                                "label": "150",
                                                "style": {"fontSize": "11px"},
                                            },
                                            200: {
                                                "label": "200",
                                                "style": {"fontSize": "11px"},
                                            },
                                            250: {
                                                "label": "250",
                                                "style": {"fontSize": "11px"},
                                            },
                                        },
                                        tooltip={
                                            "placement": "bottom",
                                            "always_visible": False,
                                        },
                                    ),
                                ],
                                style={
                                    "minWidth": "260px",
                                    "flex": "0 0 33%",
                                    "maxWidth": "33%",
                                },
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
                                        style={
                                            "paddingTop": "4px",
                                        },
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
                            "flexWrap": "wrap",
                            "alignItems": "center",
                            "gap": "24px",
                            "marginBottom": "8px",
                        },
                    ),

                    # Ironbeam chart block (interval + graph)
                    ironbeam_layout(),
                ],
                style={"marginTop": "8px"},
            )
        ],
    )

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
                                # Default to same as trade date (0DTE)
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
                                "Expected (SS)",
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
                                inputStyle={"marginRight": "6px"},
                                labelStyle={"marginRight": "12px"},
                                style={"color": "white"},
                            ),
                        ],
                        style={
                            "display": "flex",
                            "alignItems": "flex-start",
                            "gap": "16px",
                            "flexWrap": "wrap",
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "alignItems": "flex-start",
                    "gap": "16px",
                    "flexWrap": "wrap",
                },
            ),

            html.Hr(style={"borderColor": "#444"}),

            # ===== "Tabs" switch =====
            dcc.RadioItems(
                id=TAB_SWITCH_ID,
                options=[
                    {
                        "label": "Smile / Skew / Term / GEX",
                        "value": "analytics",
                    },
                    {
                        "label": "Price",
                        "value": "price",
                    },
                ],
                value="analytics",
                inline=True,
                labelStyle={
                    "marginRight": "16px",
                    "padding": "6px 12px",
                    "border": "1px solid #1f2937",
                    "borderRadius": "6px 6px 0 0",
                    "cursor": "pointer",
                },
                inputStyle={"marginRight": "6px"},
                style={
                    "marginBottom": "8px",
                    "color": "white",
                },
            ),

            # ===== Tab contents =====
            analytics_section,
            price_section,
        ],
        style={
            "backgroundColor": "black",
            "color": "white",
            "minHeight": "100vh",
            "padding": "0 80px 30px",
        },
    )


# Use function-based layout so defaults aren't frozen at deploy time
app.layout = serve_layout


# Keep expiration in sync with trade date (0DTE default)
@app.callback(
    Output(EXPIRATION_DATE_PICK, "date"),
    Input(TRADE_DATE_PICK, "date"),
)
def sync_expiration_with_trade(trade_date):
    return trade_date


# Toggle which section is visible based on "tab" switch
@app.callback(
    Output("analytics-section", "hidden"),
    Output("price-section", "hidden"),
    Input(TAB_SWITCH_ID, "value"),
)
def toggle_tab_visibility(active):
    if active == "price":
        # hide analytics, show price
        return True, False
    # default: show analytics, hide price
    return False, True


# Register all module callbacks
register_smile(app)
register_skew(app)
register_term_structure(app)
register_term_metrics(app)
register_ironbeam_callbacks(app)

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
