# apps/web/app.py

from __future__ import annotations

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import sys
from pathlib import Path as _P

REPO_ROOT = _P(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import os
import datetime as dt
from typing import List, Optional, Dict, Any

import pandas as pd
from sqlalchemy import create_engine, text

from dash import Dash, html, dcc, Input, Output, State, dash_table, ctx
from dash.exceptions import PreventUpdate
import dash_auth

from packages.backtests.gex_fade import GexFadeParams, run_gex_fade_backtest  # noqa: E402

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

CLOCK_ID = "CLOCK"
TRADE_DATE_PICK = "trade-date"
EXPIRATION_DATE_PICK = "expiration-date-pick"
SMILE_TIME_INPUT = "smile-time-input"
SMILE_GRAPH = "SMILE_GRAPH"
EXPECTED_TOGGLE_ID = "expected-ss-toggle"
LIVE_DATA_STORE_ID = "live-data-store"
LIVE_UPDATE_TIMER_ID = "live-update-timer"

MAIN_TABS_ID = "main-tabs"
TAB_DASHBOARD = "tab-dashboard"
TAB_BACKTESTS = "tab-backtests"

BT_DATE_RANGE_ID = "bt-date-range"
BT_RUN_BTN_ID = "bt-run-btn"
BT_STATUS_ID = "bt-status"
BT_SUMMARY_ID = "bt-summary"
BT_TRADES_TABLE_ID = "bt-trades-table"
BT_TRADES_STORE_ID = "bt-trades-store"
BT_DOWNLOAD_BTN_ID = "bt-download-btn"
BT_DOWNLOAD_ID = "bt-download"

BT_ENTRY_PROX_ID = "bt-entry-proximity-max"
BT_WALL_GEX_MIN_B_ID = "bt-wall-gex-min-b"
BT_NET_GEX_MIN_B_ID = "bt-net-gex-min-b"
BT_MIN_BAR_ID = "bt-min-bar-index"
BT_MAX_BAR_ID = "bt-max-bar-index"
BT_REQUIRE_RTH_ID = "bt-require-rth"
BT_MIN_ABS_SKEW_ID = "bt-min-abs-skew"

BT_MIN_MINUTES_BETWEEN_TESTS_ID = "bt-min-minutes-between-tests"
BT_MIN_SKEW_DROP_PCT_ID = "bt-min-skew-drop-pct"

# NEW option controls
BT_REQUIRE_RESET_ID = "bt-require-reset"
BT_RESET_BUFFER_ID = "bt-reset-buffer-points"

BT_STOP_POINTS_ID = "bt-stop-loss-points"
BT_TARGET_RR_ID = "bt-target-rr"
BT_MAX_BARS_TRADE_ID = "bt-max-bars-in-trade"
BT_MAX_TRADES_DAY_ID = "bt-max-trades-per-day"

# Backtest Mode Toggle
BT_MODE_TOGGLE_ID = "bt-mode-toggle"


def get_default_trade_date() -> dt.date:
    today = dt.date.today()
    if today.weekday() == 5:
        return today - dt.timedelta(days=1)
    if today.weekday() == 6:
        return today - dt.timedelta(days=2)
    return today


def pt_time_options(start="06:30", end="13:00", step_min=1) -> List[dict]:
    t0, t1 = dt.datetime.strptime(start, "%H:%M"), dt.datetime.strptime(end, "%H:%M")
    out, cur = [], t0
    while cur <= t1:
        hhmm = cur.strftime("%H:%M")
        out.append({"label": f"{hhmm} PT", "value": hhmm})
        cur += dt.timedelta(minutes=step_min)
    return out


def _engine() -> Optional[Any]:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        return None
    return create_engine(db_url)


def _load_es_minutes_with_features(eng, start_date: Optional[str], end_date: Optional[str]) -> pd.DataFrame:
    base_query = """
        SELECT
            trade_date,
            ts_utc,
            ts_pt,
            bar_index,
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
            put_skew_pp_primary,
            smile_dte_primary,
            smile_expir_primary
        FROM es_minutes_with_features
    """
    conditions = []
    params: Dict[str, object] = {}
    if start_date:
        conditions.append("trade_date >= :start_date")
        params["start_date"] = start_date
    if end_date:
        conditions.append("trade_date <= :end_date")
        params["end_date"] = end_date
    if conditions:
        base_query += " WHERE " + " AND ".join(conditions)
    base_query += " ORDER BY trade_date, ts_utc"
    with eng.connect() as conn:
        return pd.read_sql_query(text(base_query), conn, params=params)


def _summary_row(label: str, value: str) -> html.Div:
    return html.Div(
        [
            html.Div(label, style={"color": "#9ca3af", "fontSize": "12px"}),
            html.Div(value, style={"color": "#e5e7eb", "fontSize": "18px", "fontWeight": "600"}),
        ],
        style={"padding": "10px 12px", "border": "1px solid #1f2937", "borderRadius": "10px", "backgroundColor": "#0b0f19", "minWidth": "140px"},
    )


def _num_input(label: str, _id: str, value: float | int, step: float | int = 1, min_: float | int | None = None):
    return html.Div(
        [
            html.Label(label, style={"color": "white", "marginBottom": "6px", "fontSize": "13px", "fontWeight": "500"}),
            dcc.Input(
                id=_id,
                type="number",
                value=value,
                step=step,
                min=min_,
                debounce=True,
                style={"width": "100%", "backgroundColor": "#0b0f19", "color": "#e5e7eb", "border": "1px solid #1f2937", "borderRadius": "8px", "padding": "8px 10px"},
            ),
        ],
        style={"display": "flex", "flexDirection": "column", "gap": "4px"},
    )


app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server
server.secret_key = os.getenv("SECRET_KEY", "dev-secret-key")

VALID_USERNAME_PASSWORD_PAIRS = {"ryan": "ChangeThisPassword123!", "sara": "ChangeThisPassword123!"}
auth = dash_auth.BasicAuth(app, VALID_USERNAME_PASSWORD_PAIRS)


def _dashboard_tab(default_trade_date: dt.date) -> html.Div:
    time_options = pt_time_options()
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [html.Label("Trade Date", style={"color": "white", "marginBottom": "4px"}),
                         dcc.DatePickerSingle(id=TRADE_DATE_PICK, display_format="YYYY-MM-DD", date=default_trade_date)],
                        style={"marginRight": "16px", "display": "flex", "flexDirection": "column"},
                    ),
                    html.Div(
                        [html.Label("Expiration", style={"color": "white", "marginBottom": "4px"}),
                         dcc.DatePickerSingle(id=EXPIRATION_DATE_PICK, display_format="YYYY-MM-DD", date=default_trade_date)],
                        style={"marginRight": "16px", "display": "flex", "flexDirection": "column"},
                    ),
                    html.Div(
                        [html.Label("Time Slices (PT)", style={"color": "white", "marginBottom": "4px"}),
                         dcc.Dropdown(id=SMILE_TIME_INPUT, options=time_options, value=["06:31"], multi=True, style={"minWidth": "320px"})],
                        style={"marginRight": "16px", "display": "flex", "flexDirection": "column"},
                    ),
                    html.Div(
                        [html.Label("Expected (SS)", style={"color": "white", "marginBottom": "4px"}),
                         dcc.RadioItems(
                             id=EXPECTED_TOGGLE_ID,
                             options=[{"label": "ON", "value": "on"}, {"label": "OFF", "value": "off"}],
                             value="on",
                             inline=True,
                             inputStyle={"marginRight": "6px"},
                             labelStyle={"marginRight": "12px"},
                             style={"color": "white"},
                         )],
                        style={"display": "flex", "alignItems": "flex-start", "gap": "16px", "flexWrap": "wrap"},
                    ),
                ],
                style={"display": "flex", "alignItems": "flex-start", "gap": "16px", "flexWrap": "wrap"},
            ),
            html.Hr(style={"borderColor": "#444"}),

            html.Div(
                [
                    html.Div(dcc.Graph(id=SMILE_GRAPH, style={"height": "840px"}), style={"minWidth": 0}),
                    html.Div(gex_block(), style={"minWidth": 0}),
                ],
                style={"display": "grid", "gridTemplateColumns": "2fr 1fr", "gap": "16px", "alignItems": "stretch"},
            ),

            html.Hr(style={"borderColor": "#333"}),

            html.Div(
                [make_term_structure_block(), html.Div([make_skew_block(), make_term_metrics_block()])],
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px", "alignItems": "stretch"},
            ),

            html.Hr(style={"borderColor": "#333"}),

            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Label("Min |Net GEX| (Billions)", style={"color": "white", "marginBottom": "6px", "fontSize": "13px", "fontWeight": "500"}),
                                    dcc.Slider(id="gex-threshold-billions", min=0, max=300, step=10, value=100,
                                               marks={0: {"label": "0"}, 50: {"label": "50"}, 100: {"label": "100"}, 150: {"label": "150"}, 200: {"label": "200"}, 250: {"label": "250"}},
                                               tooltip={"placement": "bottom", "always_visible": False}),
                                ],
                                style={"minWidth": "260px", "flex": "0 0 33%", "maxWidth": "33%"},
                            ),
                            html.Div(
                                [
                                    html.Label("Bar Interval", style={"color": "white", "marginBottom": "6px", "fontSize": "13px", "fontWeight": "500"}),
                                    dcc.RadioItems(
                                        id="ironbeam-bar-interval",
                                        options=[{"label": "1 min", "value": "1min"}, {"label": "5 min", "value": "5min"}],
                                        value="1min",
                                        inline=True,
                                        labelStyle={"marginRight": "16px", "color": "white", "fontSize": "13px"},
                                        inputStyle={"marginRight": "6px"},
                                        style={"paddingTop": "4px"},
                                    ),
                                ],
                                style={"minWidth": "180px", "flex": "0 0 auto", "display": "flex", "flexDirection": "column"},
                            ),
                        ],
                        style={"display": "flex", "flexWrap": "wrap", "alignItems": "center", "gap": "24px", "marginBottom": "8px"},
                    ),
                    ironbeam_layout(),
                ],
                style={"marginTop": "8px"},
            ),
        ],
        style={"paddingTop": "8px"},
    )


def _backtests_tab(default_start: dt.date, default_end: dt.date) -> html.Div:
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Backtests", style={"fontSize": "22px", "fontWeight": "700", "color": "#e5e7eb"}),
                            html.Div("Run gex_fade and inspect every trade.", style={"color": "#9ca3af", "marginTop": "4px"}),
                        ],
                        style={"marginBottom": "10px"},
                    ),
                    html.Div(
                        [
                            html.Label("Date Range (trade_date)", style={"color": "white", "marginBottom": "6px", "fontSize": "13px"}),
                            dcc.DatePickerRange(id=BT_DATE_RANGE_ID, start_date=default_start, end_date=default_end, display_format="YYYY-MM-DD", minimum_nights=0),
                        ],
                        style={"display": "flex", "flexDirection": "column", "gap": "4px"},
                    ),
                    html.Div(
                        [
                            html.Label("Backtest Mode", style={"color": "white", "marginBottom": "6px", "fontSize": "13px"}),
                            dcc.RadioItems(
                                id=BT_MODE_TOGGLE_ID,
                                options=[{"label": "ON", "value": "on"}, {"label": "OFF", "value": "off"}],
                                value="off",
                                inline=True,
                                labelStyle={"marginRight": "14px", "color": "white", "fontSize": "13px"},
                                inputStyle={"marginRight": "6px"},
                            ),
                        ],
                        style={"display": "flex", "flexDirection": "column", "gap": "4px", "marginLeft": "20px"},
                    ),
                ],
                style={"border": "1px solid #1f2937", "borderRadius": "12px", "padding": "14px", "backgroundColor": "#0b0f19", "display": "flex", "alignItems": "center", "justifyContent": "space-between"},
            ),

            html.Div(style={"height": "12px"}),

            html.Div(
                [
                    _num_input("Entry proximity max (pts below wall)", BT_ENTRY_PROX_ID, 2.0, step=0.25, min_=0),
                    _num_input("Min |Wall GEX| (Billions)", BT_WALL_GEX_MIN_B_ID, 50, step=5, min_=0),
                    _num_input("Min Net GEX (Billions)", BT_NET_GEX_MIN_B_ID, 0, step=10),
                    _num_input("Min bar index", BT_MIN_BAR_ID, 30, step=1, min_=0),
                    _num_input("Max bar index", BT_MAX_BAR_ID, 350, step=1, min_=0),

                    html.Div(
                        [
                            html.Label("Require RTH", style={"color": "white", "marginBottom": "6px", "fontSize": "13px", "fontWeight": "500"}),
                            dcc.RadioItems(
                                id=BT_REQUIRE_RTH_ID,
                                options=[{"label": "Yes", "value": "yes"}, {"label": "No", "value": "no"}],
                                value="yes",
                                inline=True,
                                labelStyle={"marginRight": "14px", "color": "white", "fontSize": "13px"},
                                inputStyle={"marginRight": "6px"},
                            ),
                        ],
                        style={"display": "flex", "flexDirection": "column", "gap": "4px"},
                    ),

                    _num_input("Min |put skew| (pp) for tests", BT_MIN_ABS_SKEW_ID, 0.5, step=0.1, min_=0),
                    _num_input("Min minutes between tests", BT_MIN_MINUTES_BETWEEN_TESTS_ID, 30, step=1, min_=0),
                    _num_input("Min skew drop (%) from anchor", BT_MIN_SKEW_DROP_PCT_ID, 50, step=1, min_=0),

                    # NEW: reset option
                    html.Div(
                        [
                            html.Label("Require reset between tests", style={"color": "white", "marginBottom": "6px", "fontSize": "13px", "fontWeight": "500"}),
                            dcc.RadioItems(
                                id=BT_REQUIRE_RESET_ID,
                                options=[{"label": "No", "value": "no"}, {"label": "Yes", "value": "yes"}],
                                value="no",
                                inline=True,
                                labelStyle={"marginRight": "14px", "color": "white", "fontSize": "13px"},
                                inputStyle={"marginRight": "6px"},
                            ),
                        ],
                        style={"display": "flex", "flexDirection": "column", "gap": "4px"},
                    ),
                    _num_input("Reset buffer (pts) beyond proximity zone", BT_RESET_BUFFER_ID, 2.0, step=0.25, min_=0),

                    _num_input("Stop loss (ES points)", BT_STOP_POINTS_ID, 2.0, step=0.25, min_=0),
                    _num_input("Target R:R", BT_TARGET_RR_ID, 2.0, step=0.25, min_=0),
                    _num_input("Max bars in trade", BT_MAX_BARS_TRADE_ID, 60, step=1, min_=1),
                    _num_input("Max trades per day", BT_MAX_TRADES_DAY_ID, 8, step=1, min_=0),
                ],
                style={"display": "grid", "gridTemplateColumns": "repeat(3, minmax(260px, 1fr))", "gap": "12px"},
            ),

            html.Div(style={"height": "12px"}),

            html.Div(
                [
                    html.Button("Run Backtest", id=BT_RUN_BTN_ID, n_clicks=0,
                                style={"backgroundColor": "#2563eb", "color": "white", "border": "none", "borderRadius": "10px", "padding": "10px 14px", "fontWeight": "700", "cursor": "pointer"}),
                    html.Button("Download CSV", id=BT_DOWNLOAD_BTN_ID, n_clicks=0,
                                style={"backgroundColor": "#111827", "color": "#e5e7eb", "border": "1px solid #1f2937", "borderRadius": "10px", "padding": "10px 14px", "fontWeight": "700", "cursor": "pointer"}),
                    dcc.Download(id=BT_DOWNLOAD_ID),
                    html.Div(id=BT_STATUS_ID, style={"color": "#9ca3af", "marginLeft": "12px"}),
                ],
                style={"display": "flex", "alignItems": "center", "gap": "10px"},
            ),

            html.Div(style={"height": "12px"}),
            dcc.Store(id=BT_TRADES_STORE_ID),
            html.Div(id=BT_SUMMARY_ID, style={"display": "flex", "gap": "10px", "flexWrap": "wrap"}),
            html.Div(style={"height": "12px"}),

            dcc.Loading(
                dash_table.DataTable(
                    id=BT_TRADES_TABLE_ID,
                    data=[],
                    columns=[],
                    page_size=25,
                    sort_action="native",
                    filter_action="native",
                    row_selectable=False,
                    style_table={"overflowX": "auto"},
                    style_header={"backgroundColor": "#111827", "color": "#e5e7eb", "border": "1px solid #1f2937", "fontWeight": "700", "fontSize": "12px"},
                    style_cell={"backgroundColor": "#0b0f19", "color": "#e5e7eb", "border": "1px solid #111827", "fontSize": "12px", "padding": "8px",
                                "whiteSpace": "nowrap", "maxWidth": "520px", "overflow": "hidden", "textOverflow": "ellipsis"},
                    style_data_conditional=[
                        {"if": {"row_index": "odd"}, "backgroundColor": "#0a0d14"},
                        {"if": {"state": "selected"}, "backgroundColor": "#1e3a8a", "color": "white", "border": "1px solid #2563eb"},
                    ],
                ),
                type="default",
            ),
        ],
        style={"paddingTop": "8px"},
    )


def serve_layout():
    default_trade_date = get_default_trade_date()
    bt_end = default_trade_date
    bt_start = bt_end - dt.timedelta(days=60)

    return html.Div(
        [
            dcc.Store(id=LIVE_DATA_STORE_ID),
            dcc.Interval(id=LIVE_UPDATE_TIMER_ID, interval=15 * 1000, n_intervals=0),
            dcc.Interval(id=CLOCK_ID, interval=60_000, n_intervals=0),

            html.Div(
                [
                    html.Div("Surface Dynamics", style={"fontWeight": "600", "fontSize": "20px", "color": "#e5e7eb", "textAlign": "center", "width": "100%"}),
                    html.A("Home", href="https://blog.surfacedynamics.io",
                           style={"color": "#93c5fd", "textDecoration": "none", "fontWeight": "500", "padding": "4px 10px", "borderRadius": "6px",
                                  "border": "1px solid #1f2937", "position": "absolute", "right": "16px", "top": "50%", "transform": "translateY(-50%)"}),
                ],
                style={"position": "relative", "padding": "8px 16px", "borderBottom": "1px solid #1f2937", "marginBottom": "8px"},
            ),

            dcc.Tabs(
                id=MAIN_TABS_ID,
                value=TAB_DASHBOARD,
                children=[
                    dcc.Tab(label="Dashboard", value=TAB_DASHBOARD, children=_dashboard_tab(default_trade_date),
                            style={"backgroundColor": "black", "color": "#e5e7eb", "border": "1px solid #1f2937"},
                            selected_style={"backgroundColor": "#0b0f19", "color": "white", "border": "1px solid #1f2937"}),
                    dcc.Tab(label="Backtests", value=TAB_BACKTESTS, children=_backtests_tab(bt_start, bt_end),
                            style={"backgroundColor": "black", "color": "#e5e7eb", "border": "1px solid #1f2937"},
                            selected_style={"backgroundColor": "#0b0f19", "color": "white", "border": "1px solid #1f2937"}),
                ],
                parent_style={"marginTop": "6px"},
                style={"backgroundColor": "black"},
            ),
        ],
        style={"backgroundColor": "black", "color": "white", "minHeight": "100vh", "padding": "0 80px 30px"},
    )


app.layout = serve_layout


@app.callback(Output(EXPIRATION_DATE_PICK, "date", allow_duplicate=True), Input(TRADE_DATE_PICK, "date"), prevent_initial_call=True)
def sync_expiration_with_trade(trade_date):
    return trade_date


@app.callback(
    Output(BT_TRADES_TABLE_ID, "data"),
    Output(BT_TRADES_TABLE_ID, "columns"),
    Output(BT_SUMMARY_ID, "children"),
    Output(BT_STATUS_ID, "children"),
    Output(BT_TRADES_STORE_ID, "data"),
    Input(BT_RUN_BTN_ID, "n_clicks"),
    State(BT_DATE_RANGE_ID, "start_date"),
    State(BT_DATE_RANGE_ID, "end_date"),
    State(BT_ENTRY_PROX_ID, "value"),
    State(BT_WALL_GEX_MIN_B_ID, "value"),
    State(BT_NET_GEX_MIN_B_ID, "value"),
    State(BT_MIN_BAR_ID, "value"),
    State(BT_MAX_BAR_ID, "value"),
    State(BT_REQUIRE_RTH_ID, "value"),
    State(BT_MIN_ABS_SKEW_ID, "value"),
    State(BT_MIN_MINUTES_BETWEEN_TESTS_ID, "value"),
    State(BT_MIN_SKEW_DROP_PCT_ID, "value"),
    State(BT_REQUIRE_RESET_ID, "value"),
    State(BT_RESET_BUFFER_ID, "value"),
    State(BT_STOP_POINTS_ID, "value"),
    State(BT_TARGET_RR_ID, "value"),
    State(BT_MAX_BARS_TRADE_ID, "value"),
    State(BT_MAX_TRADES_DAY_ID, "value"),
)
def run_backtest_cb(
    n_clicks,
    start_date,
    end_date,
    entry_prox,
    wall_gex_min_b,
    net_gex_min_b,
    min_bar,
    max_bar,
    require_rth,
    min_abs_skew,
    min_minutes_between_tests,
    min_skew_drop_pct,
    require_reset,
    reset_buffer_points,
    stop_points,
    target_rr,
    max_bars_in_trade,
    max_trades_per_day,
):
    if not n_clicks:
        raise PreventUpdate

    eng = _engine()
    if eng is None:
        return [], [], [], "DATABASE_URL is not set (Render env vars or local .env).", None

    entry_prox_f = float(entry_prox) if entry_prox is not None else 2.0
    wall_gex_min = float(wall_gex_min_b) * 1e9 if wall_gex_min_b is not None else 50e9
    net_gex_min = float(net_gex_min_b) * 1e9 if net_gex_min_b is not None else 0.0
    min_bar_i = int(min_bar) if min_bar is not None else 30
    max_bar_i = int(max_bar) if max_bar is not None else 350
    require_rth_b = (require_rth == "yes")
    min_abs_skew_f = float(min_abs_skew) if min_abs_skew is not None else 0.0

    min_gap_i = int(min_minutes_between_tests) if min_minutes_between_tests is not None else 30
    min_drop_frac = (float(min_skew_drop_pct) / 100.0) if min_skew_drop_pct is not None else 0.50

    require_reset_b = (require_reset == "yes")
    reset_buffer_f = float(reset_buffer_points) if reset_buffer_points is not None else 2.0

    stop_points_f = float(stop_points) if stop_points is not None else 2.0
    target_rr_f = float(target_rr) if target_rr is not None else 2.0
    max_bars_i = int(max_bars_in_trade) if max_bars_in_trade is not None else 60
    max_trades_i = int(max_trades_per_day) if max_trades_per_day is not None else 8

    params = GexFadeParams(
        entry_proximity_max=entry_prox_f,
        gex_wall_min=wall_gex_min,
        gex_net_min=net_gex_min,
        min_bar_index=min_bar_i,
        max_bar_index=max_bar_i,
        require_rth=require_rth_b,
        min_abs_skew=min_abs_skew_f,
        min_minutes_between_tests=min_gap_i,
        min_put_skew_drop_frac=min_drop_frac,
        require_reset_between_tests=require_reset_b,
        reset_buffer_points=reset_buffer_f,
        stop_loss_points=stop_points_f,
        target_rr=target_rr_f,
        max_bars_in_trade=max_bars_i,
        max_trades_per_day=max_trades_i,
    )

    try:
        df = _load_es_minutes_with_features(eng, start_date, end_date)
    except Exception as e:
        return [], [], [], f"DB load failed: {e}", None

    if df.empty:
        return [], [], [], "No rows returned for that date range.", None

    try:
        trades_df, summary = run_gex_fade_backtest(df, params)
    except Exception as e:
        return [], [], [], f"Backtest crashed: {e}", None

    n_trades = summary.get("n_trades", 0)
    win_rate = summary.get("win_rate", 0.0)
    avg_r = summary.get("avg_r", 0.0)
    total_r = summary.get("total_r", 0.0)

    summary_children = [
        _summary_row("Trades", f"{n_trades:,}"),
        _summary_row("Win rate", f"{win_rate:.1%}"),
        _summary_row("Avg R", f"{avg_r:.2f}"),
        _summary_row("Total R", f"{total_r:.1f}"),
    ]

    if trades_df is None or trades_df.empty:
        return [], [], summary_children, "No trades generated with current parameters.", None

    preferred_cols = [
        "trade_date",
        "anchor_test_ts_pt",
        "confirm_test_ts_pt",
        "minutes_between_tests",
        "reset_required",
        "reset_seen",
        "anchor_put_skew_pp",
        "confirm_put_skew_pp",
        "put_skew_drop_pct",
        "entry_ts_pt",
        "exit_ts_pt",
        "entry_price",
        "exit_price",
        "exit_reason",
        "pnl_points",
        "r_mult",
        "wall_level",
        "dist_to_wall_at_entry",
        "put_skew_entry_pp",
        "net_gex",
        "wall_gex",
        "smile_dte_primary",
        "smile_expir_primary",
    ]
    cols = [c for c in preferred_cols if c in trades_df.columns] + [c for c in trades_df.columns if c not in preferred_cols]
    trades_df = trades_df[cols].copy()

    columns = [{"name": c, "id": c} for c in trades_df.columns]
    data = trades_df.to_dict("records")
    store_payload = trades_df.to_json(orient="split", date_format="iso")

    return data, columns, summary_children, f"Done. Trades: {len(trades_df):,}", store_payload


@app.callback(
    Output(BT_DOWNLOAD_ID, "data"),
    Input(BT_DOWNLOAD_BTN_ID, "n_clicks"),
    State(BT_TRADES_STORE_ID, "data"),
    prevent_initial_call=True,
)
def download_trades_cb(n, store_json):
    if not n or not store_json:
        raise PreventUpdate
    df = pd.read_json(store_json, orient="split")
    return dcc.send_data_frame(df.to_csv, "gex_fade_trades.csv", index=False)


# --- NEW: Toggle row selection mode ---
@app.callback(
    Output(BT_TRADES_TABLE_ID, "row_selectable"),
    Input(BT_MODE_TOGGLE_ID, "value"),
)
def toggle_row_selection(mode_value):
    if mode_value == "on":
        return "single"
    return False


# --- NEW: Handle row selection -> Update Dashboard ---
@app.callback(
    Output(MAIN_TABS_ID, "value"),
    Output(TRADE_DATE_PICK, "date"),
    Output(EXPIRATION_DATE_PICK, "date", allow_duplicate=True),
    Output(SMILE_TIME_INPUT, "value"),
    Input(BT_TRADES_TABLE_ID, "selected_rows"),
    State(BT_TRADES_TABLE_ID, "data"),
    State(BT_MODE_TOGGLE_ID, "value"),
    prevent_initial_call=True,
)
def on_trade_selected(selected_rows, data, mode_value):
    if mode_value != "on" or not selected_rows or not data:
        raise PreventUpdate

    row_idx = selected_rows[0]
    if row_idx >= len(data):
        raise PreventUpdate

    row = data[row_idx]
    
    # Extract fields
    trade_date = row.get("trade_date")
    expiration = row.get("smile_expir_primary")
    
    # Timeslices: anchor_test_ts_pt and confirm_test_ts_pt
    # Note: These might be formatted as HH:MM strings already if they come from the table
    # If they are missing, fallback to entry_ts_pt
    
    times = []
    if row.get("anchor_test_ts_pt"):
        times.append(row["anchor_test_ts_pt"])
    if row.get("confirm_test_ts_pt"):
        times.append(row["confirm_test_ts_pt"])
        
    # If no anchor/confirm times (e.g. old backtest logic), maybe just use entry time?
    # But user specifically asked for anchor/confirm.
    # If the list is empty, let's default to entry time if available
    if not times and row.get("entry_ts_pt"):
        times.append(row["entry_ts_pt"])

    # Ensure unique
    times = list(set(times))
    
    # If we have valid data, switch tab and update inputs
    if trade_date:
        # Return: Switch to Dashboard tab, set trade date, set expiration, set time slices
        # Note: expiration date picker might need YYYY-MM-DD. 
        # smile_expir_primary is likely a date string or object.
        return TAB_DASHBOARD, trade_date, expiration, times

    raise PreventUpdate


# --- NEW: Highlight selected row ---
@app.callback(
    Output(BT_TRADES_TABLE_ID, "style_data_conditional"),
    Input(BT_TRADES_TABLE_ID, "selected_rows"),
)
def update_table_style(selected_rows):
    base_style = [
        {"if": {"row_index": "odd"}, "backgroundColor": "#0a0d14"},
    ]
    if selected_rows:
        row_idx = selected_rows[0]
        base_style.append({
            "if": {"row_index": row_idx},
            "backgroundColor": "#1e3a8a",
            "color": "white",
            "border": "1px solid #2563eb",
        })
    return base_style


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
