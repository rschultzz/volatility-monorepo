# apps/web/modules/Ironbeam/components.py
#
# Collapsible indicator sidebar (LEFT) with a floating toggle button.
# - Sidebar collapses completely (width 0).
# - Toggle button docks to the sidebar edge when open, and moves to the left margin when collapsed.
# - Charts reclaim the full width when sidebar is collapsed.
# - Existing IDs are preserved for callbacks.

from dash import html, dcc


def ironbeam_layout():
    return html.Div(
        [
            dcc.Interval(id="ironbeam-interval", interval=10000, n_intervals=0),

            # Indicator selection + configs (persisted)
            dcc.Store(
                id="ib-indicator-state",
                storage_type="local",
                data={"enabled": ["aggressor_flow", "gex_overlay"], "cfg": {}},
            ),

            # Shared x-range for panel alignment
            dcc.Store(id="ib-shared-xrange", storage_type="memory"),

            # UI state (sidebar collapsed/open)
            dcc.Store(
                id="ib-ui-state",
                storage_type="local",
                data={"sidebar_collapsed": False},
            ),

            # Floating collapse toggle (always visible)
            html.Button(
                "«",
                id="ib-sidebar-toggle",
                n_clicks=0,
                title="Collapse/expand indicators",
                style={
                    "position": "absolute",
                    "left": "303px",  # will be overridden by callback based on collapsed/open
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

            # Row: sidebar + charts
            html.Div(
                id="ib-ironbeam-row",
                children=[
                    # ===== INDICATORS SIDEBAR (LEFT) =====
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
                            # Content wrapper so we can hide it when collapsed
                            html.Div(
                                id="ib-sidebar-content",
                                children=[
                                    dcc.Checklist(
                                        id="ib-indicator-enabled",
                                        options=[],  # populated from registry in callbacks
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
                                        options=[],  # populated from registry in callbacks
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

                    # ===== CHART AREA (RIGHT) =====
                    html.Div(
                        [
                            dcc.Graph(
                                id="ironbeam-chart",
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
