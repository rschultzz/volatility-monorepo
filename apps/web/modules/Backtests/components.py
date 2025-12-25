# apps/web/modules/Backtests/components.py
from __future__ import annotations

import datetime as dt
from typing import List

from dash import dcc, html, dash_table

from packages.backtests.registry import get_strategies, ParamSpec

# ===== IDs (Backtests Tab) =====
BT_STRATEGY_ID = "bt-strategy"
BT_EXPECTED_TOGGLE_ID = "bt-expected-ss-toggle"
BT_MODE_TOGGLE_ID = "bt-mode-toggle"

BT_DATE_RANGE_ID = "bt-date-range"
BT_PARAM_PANEL_ID = "bt-param-panel"
BT_RUN_BTN_ID = "bt-run-btn"
BT_SUMMARY_ID = "bt-summary"
BT_TRADES_STORE_ID = "bt-trades-store"
BT_TRADES_TABLE_ID = "bt-trades-table"
BT_DOWNLOAD_BTN_ID = "bt-download-btn"
BT_DOWNLOAD_ID = "bt-download"


def _param_control(strategy_key: str, spec: ParamSpec):
    pid = {"type": "bt-param", "strategy": strategy_key, "name": spec.name}

    label = html.Div(
        [
            html.Div(spec.label, style={"color": "#e5e7eb", "fontSize": "12px", "fontWeight": "600"}),
            html.Div(spec.help, style={"color": "#9ca3af", "fontSize": "11px"}) if spec.help else None,
        ]
    )

    if spec.kind == "bool":
        ctrl = dcc.RadioItems(
            id=pid,
            options=[{"label": "ON", "value": True}, {"label": "OFF", "value": False}],
            value=bool(spec.default),
            inline=True,
            inputStyle={"marginRight": "6px"},
            labelStyle={"marginRight": "12px"},
            style={"color": "white"},
        )
    elif spec.kind == "select":
        ctrl = dcc.Dropdown(
            id=pid,
            options=spec.options or [],
            value=spec.default,
            clearable=False,
            style={"minWidth": "240px"},
        )
    else:
        ctrl = dcc.Input(
            id=pid,
            type="number",
            value=spec.default,
            min=spec.min,
            max=spec.max,
            step=spec.step,
            debounce=True,
            style={
                "width": "220px",
                "backgroundColor": "#111827",
                "border": "1px solid #374151",
                "color": "white",
                "borderRadius": "8px",
                "padding": "6px 10px",
            },
        )

    return html.Div([label, ctrl], style={"display": "flex", "flexDirection": "column", "gap": "6px"})


def render_params(strategy_key: str) -> List[html.Div]:
    strategies = get_strategies()
    spec = strategies[strategy_key]

    groups = {}
    for p in spec.params:
        groups.setdefault(p.group, []).append(p)

    out: List[html.Div] = []
    for gname, plist in groups.items():
        out.append(
            html.Div(
                [
                    html.Div(gname, style={"color": "#93c5fd", "fontWeight": "700", "marginBottom": "8px"}),
                    html.Div(
                        [_param_control(strategy_key, p) for p in plist],
                        style={"display": "grid", "gridTemplateColumns": "repeat(2, minmax(260px, 1fr))", "gap": "14px"},
                    ),
                ],
                style={
                    "backgroundColor": "#0b1220",
                    "border": "1px solid #1f2937",
                    "borderRadius": "12px",
                    "padding": "12px",
                    "marginBottom": "12px",
                },
            )
        )
    return out


def make_backtests_tab(default_start: dt.date, default_end: dt.date) -> html.Div:
    strategies = get_strategies()
    default_strategy = list(strategies.keys())[0]

    return html.Div(
        [
            dcc.Store(id=BT_TRADES_STORE_ID),

            # Top controls
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Strategy", style={"color": "#9ca3af", "fontSize": "12px"}),
                            dcc.Dropdown(
                                id=BT_STRATEGY_ID,
                                options=[{"label": s.label, "value": k} for k, s in strategies.items()],
                                value=default_strategy,
                                clearable=False,
                                style={"minWidth": "320px"},
                            ),
                        ],
                        style={"display": "flex", "flexDirection": "column", "gap": "6px"},
                    ),
                    html.Div(
                        [
                            html.Div("Compare to Expected (SS)", style={"color": "#9ca3af", "fontSize": "12px"}),
                            dcc.RadioItems(
                                id=BT_EXPECTED_TOGGLE_ID,
                                options=[{"label": "ON", "value": "on"}, {"label": "OFF", "value": "off"}],
                                value="on",
                                inline=True,
                                inputStyle={"marginRight": "6px"},
                                labelStyle={"marginRight": "12px"},
                                style={"color": "white"},
                            ),
                        ],
                        style={"display": "flex", "flexDirection": "column", "gap": "6px"},
                    ),
                    html.Div(
                        [
                            html.Div("Backtest Mode", style={"color": "#9ca3af", "fontSize": "12px"}),
                            dcc.RadioItems(
                                id=BT_MODE_TOGGLE_ID,
                                options=[{"label": "ON", "value": "on"}, {"label": "OFF", "value": "off"}],
                                value="off",
                                inline=True,
                                inputStyle={"marginRight": "6px"},
                                labelStyle={"marginRight": "12px"},
                                style={"color": "white"},
                            ),
                        ],
                        style={"display": "flex", "flexDirection": "column", "gap": "6px"},
                    ),
                    html.Div(
                        [
                            html.Div("Date range", style={"color": "#9ca3af", "fontSize": "12px"}),
                            dcc.DatePickerRange(
                                id=BT_DATE_RANGE_ID,
                                start_date=default_start,
                                end_date=default_end,
                                display_format="YYYY-MM-DD",
                                minimum_nights=0,
                                style={"backgroundColor": "#111827"},
                            ),
                        ],
                        style={"display": "flex", "flexDirection": "column", "gap": "6px"},
                    ),
                    html.Button(
                        "Run",
                        id=BT_RUN_BTN_ID,
                        n_clicks=0,
                        style={
                            "marginTop": "18px",
                            "backgroundColor": "#2563eb",
                            "color": "white",
                            "border": "none",
                            "borderRadius": "10px",
                            "padding": "10px 16px",
                            "fontWeight": "700",
                            "cursor": "pointer",
                            "height": "40px",
                        },
                    ),
                    html.Button(
                        "Download CSV",
                        id=BT_DOWNLOAD_BTN_ID,
                        n_clicks=0,
                        style={
                            "marginTop": "18px",
                            "backgroundColor": "#111827",
                            "color": "white",
                            "border": "1px solid #374151",
                            "borderRadius": "10px",
                            "padding": "10px 16px",
                            "fontWeight": "700",
                            "cursor": "pointer",
                            "height": "40px",
                        },
                    ),
                    dcc.Download(id=BT_DOWNLOAD_ID),
                ],
                style={
                    "display": "flex",
                    "flexWrap": "wrap",
                    "gap": "16px",
                    "alignItems": "flex-end",
                    "marginBottom": "12px",
                },
            ),

            html.Div(
                id="bt-strategy-desc",
                style={"color": "#9ca3af", "fontSize": "12px", "marginBottom": "10px"},
            ),

            html.Div(id=BT_PARAM_PANEL_ID, children=render_params(default_strategy)),

            html.Div(id=BT_SUMMARY_ID, style={"marginTop": "10px", "marginBottom": "10px"}),

            dash_table.DataTable(
                id=BT_TRADES_TABLE_ID,
                data=[],
                columns=[],
                page_size=30,
                sort_action="native",
                filter_action="native",
                # IMPORTANT: we will toggle this between None and "single" in callbacks
                row_selectable=None,
                selected_rows=[],
                style_table={"overflowX": "auto", "border": "1px solid #1f2937", "borderRadius": "12px"},
                style_header={"backgroundColor": "#111827", "color": "white", "fontWeight": "700"},
                style_cell={
                    "backgroundColor": "#0b1220",
                    "color": "white",
                    "border": "1px solid #1f2937",
                    "fontSize": "12px",
                    "padding": "6px",
                    "minWidth": "90px",
                    "maxWidth": "260px",
                    "whiteSpace": "normal",
                },
            ),
        ],
        style={"padding": "12px 0"},
    )
