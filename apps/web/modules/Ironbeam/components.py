# apps/web/modules/Ironbeam/components.py
#
# Step 10:
# - Remove the hidden legacy placeholder `ironbeam-flow-chart` entirely.
# - Layout is now fully plugin-panel based: Aggressor Flow renders inside `ib-indicator-panels`.
# - Stores + sidebar IDs unchanged.

from dash import html, dcc


def ironbeam_layout():
    return html.Div(
        [
            # Auto-refresh timer for live bars / overlays
            dcc.Interval(
                id="ironbeam-interval",
                interval=10000,  # 10s
                n_intervals=0,
            ),

            # Indicator selection + configs (persisted across refresh)
            dcc.Store(
                id="ib-indicator-state",
                storage_type="local",
                data={
                    "enabled": ["aggressor_flow"],
                    "cfg": {},
                },
            ),

            # Shared x-range for keeping all panels aligned
            dcc.Store(id="ib-shared-xrange", storage_type="memory"),

            html.Div(
                [
                    # ===== MAIN CHARTS (left) =====
                    html.Div(
                        [
                            dcc.Graph(
                                id="ironbeam-chart",
                                # Default height; callbacks may override to reclaim space
                                style={"height": "calc(100vh - 250px)"},
                                config={
                                    "displaylogo": False,
                                    "scrollZoom": True,
                                    "modeBarButtonsToRemove": ["autoScale2d"],
                                    "responsive": True,
                                },
                            ),

                            # Dynamic plugin panels render here (Aggressor Flow is now one of them)
                            html.Div(id="ib-indicator-panels", style={"marginTop": "10px"}),
                        ],
                        style={"flex": "1", "minWidth": 0},
                    ),

                    # ===== INDICATORS SIDEBAR (right) =====
                    html.Div(
                        [
                            html.Div(
                                "Indicators",
                                style={
                                    "color": "white",
                                    "fontWeight": "700",
                                    "fontSize": "14px",
                                    "marginBottom": "8px",
                                },
                            ),

                            dcc.Checklist(
                                id="ib-indicator-enabled",
                                options=[],  # populated from registry in callbacks
                                value=["aggressor_flow"],
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
                                    "fontWeight": "700",
                                    "fontSize": "14px",
                                    "marginBottom": "8px",
                                },
                            ),

                            dcc.Dropdown(
                                id="ib-settings-indicator",
                                options=[],  # populated from registry in callbacks
                                value="aggressor_flow",
                                clearable=False,
                                style={
                                    "backgroundColor": "#0b1220",
                                    "color": "black",
                                },
                            ),

                            html.Div(
                                id="ib-settings-form",
                                style={"marginTop": "6px"},
                            ),
                        ],
                        style={
                            "width": "320px",
                            "minWidth": "320px",
                            "marginLeft": "12px",
                            "border": "1px solid #1f2937",
                            "borderRadius": "12px",
                            "padding": "12px",
                            "backgroundColor": "#0b1220",
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "gap": "12px",
                    "alignItems": "flex-start",
                },
            ),
        ],
        style={"marginTop": "4px"},
    )
