# apps/web/modules/Ironbeam/components.py
#
# Step 2:
# - Keeps Classic vs React Preview toggle.
# - Replaces the placeholder preview card with an iframe that points to a local React dev server.
# - The iframe URL is configurable with IRONBEAM_REACT_PREVIEW_URL.

from dash import html, dcc
import os


REACT_PREVIEW_URL = os.getenv("IRONBEAM_REACT_PREVIEW_URL", "/react-preview")


def ironbeam_layout():
    return html.Div(
        [
            dcc.Interval(id="ironbeam-interval", interval=10000, n_intervals=0),
            dcc.Store(
                id="ib-indicator-state",
                storage_type="local",
                data={"enabled": ["aggressor_flow", "gex_overlay"], "cfg": {}},
            ),
            dcc.Store(id="ib-shared-xrange", storage_type="memory"),
            dcc.Store(
                id="ib-ui-state",
                storage_type="local",
                data={"sidebar_collapsed": False},
            ),
            dcc.Input(
                id="ib-react-timeslice-bridge",
                type="text",
                value="",
                style={"display": "none"},
            ),
            dcc.Input(
                id="ib-react-timeslice-parent",
                type="text",
                value="",
                style={"display": "none"},
            ),
            html.Button(
                "bridge",
                id="ib-react-timeslice-trigger",
                n_clicks=0,
                style={"display": "none"},
            ),
            html.Button(
                "«",
                id="ib-sidebar-toggle",
                n_clicks=0,
                title="Collapse/expand indicators",
                style={
                    "position": "absolute",
                    "left": "303px",
                    "top": "10px",
                    "zIndex": 2000,
                    "width": "34px",
                    "height": "34px",
                    "borderRadius": "10px",
                    "border": "1px solid #1f2937",
                    "backgroundColor": "#111827",
                    "color": "#bfdbfe",
                    "cursor": "pointer",
                    "fontWeight": "900",
                    "lineHeight": "32px",
                    "transition": "left 0.18s ease",
                },
            ),
            html.Div(
                id="ib-ironbeam-row",
                children=[
                    html.Div(
                        [
                            html.Div(
                                "Indicators",
                                style={
                                    "color": "white",
                                    "fontWeight": "800",
                                    "fontSize": "14px",
                                    "marginBottom": "10px",
                                    "marginLeft": "6px",
                                },
                            ),
                            html.Div(
                                id="ib-sidebar-content",
                                children=[
                                    dcc.Checklist(
                                        id="ib-indicator-enabled",
                                        options=[],
                                        value=["aggressor_flow", "gex_overlay"],
                                        inputStyle={"marginRight": "8px"},
                                        labelStyle={
                                            "display": "block",
                                            "color": "white",
                                            "fontSize": "13px",
                                            "marginBottom": "6px",
                                        },
                                        style={"marginBottom": "10px"},
                                    ),
                                    html.Div(
                                        style={
                                            "height": "1px",
                                            "backgroundColor": "#1f2937",
                                            "margin": "10px 0",
                                        }
                                    ),
                                    html.Div(
                                        "Settings",
                                        style={
                                            "color": "white",
                                            "fontWeight": "800",
                                            "fontSize": "14px",
                                            "marginBottom": "8px",
                                        },
                                    ),
                                    dcc.Dropdown(
                                        id="ib-settings-indicator",
                                        options=[],
                                        value="aggressor_flow",
                                        clearable=False,
                                        style={"backgroundColor": "#0b1220", "color": "black"},
                                    ),
                                    html.Div(id="ib-settings-form", style={"marginTop": "8px"}),
                                ],
                                style={"display": "block"},
                            ),
                        ],
                        id="ib-indicator-sidebar",
                        style={
                            "width": "320px",
                            "minWidth": "320px",
                            "border": "1px solid #1f2937",
                            "borderRadius": "12px",
                            "padding": "12px",
                            "backgroundColor": "#0b1220",
                            "transition": "width 0.18s ease, min-width 0.18s ease, padding 0.18s ease",
                            "overflow": "hidden",
                        },
                    ),
                    html.Div(
                        [
                            html.Div(
                                id="ib-classic-chart-wrap",
                                children=[
                                    dcc.Graph(
                                        id="ironbeam-chart",
                                        clear_on_unhover=True,
                                        style={
                                            "flex": "1 1 auto",
                                            "height": "100%",
                                            "minHeight": "260px",
                                        },
                                        config={
                                            "displaylogo": False,
                                            "scrollZoom": True,
                                            "modeBarButtonsToRemove": ["autoScale2d"],
                                            "responsive": True,
                                        },
                                    ),
                                    html.Div(
                                        id="ib-indicator-panels",
                                        style={"flex": "0 0 auto", "marginTop": "10px"},
                                    ),
                                ],
                                style={
                                    "display": "flex",
                                    "flexDirection": "column",
                                    "flex": "1 1 auto",
                                    "minHeight": 0,
                                },
                            ),
                            html.Div(
                                id="ib-react-preview-wrap",
                                children=[
                                    html.Iframe(
                                        id="ib-react-preview-frame",
                                        src=REACT_PREVIEW_URL,
                                        style={
                                            "width": "100%",
                                            "height": "calc(100vh - 265px)",
                                            "minHeight": "620px",
                                            "border": "1px solid #1f2937",
                                            "borderRadius": "14px",
                                            "backgroundColor": "#020617",
                                        },
                                    ),
                                ],
                                style={"display": "none"},
                            ),
                        ],
                        id="ib-chart-area",
                        style={
                            "flex": "1",
                            "minWidth": 0,
                            "height": "calc(100vh - 190px)",
                            "display": "flex",
                            "flexDirection": "column",
                        },
                    ),
                ],
                style={"display": "flex", "gap": "12px", "alignItems": "flex-start"},
            ),
        ],
        style={"marginTop": "4px", "position": "relative", "width": "100%"},
    )
