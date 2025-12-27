# apps/web/modules/Ironbeam/components.py

from dash import html, dcc


def ironbeam_layout():
    """
    Ironbeam block: auto-refresh timer + main ES/GEX chart + aggressor flow chart.

    Step 3 (SAFE UI scaffolding):
    - Adds an "indicators" sidebar UI (enable list + settings selector).
    - Keeps ALL existing chart IDs unchanged so current callbacks keep working:
        - ironbeam-chart
        - ironbeam-flow-chart
    - Keeps the Stores from Step 1:
        - ib-indicator-state (persisted)
        - ib-shared-xrange (memory)
    - Does NOT yet wire the sidebar to anything (that’s Step 4).
    """
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
                    # purely a default; Step 4 will sync this with the checklist
                    "enabled": ["aggressor_flow"],
                    "cfg": {},
                },
            ),

            # Shared x-range for keeping all panels aligned (Step 2 uses this)
            dcc.Store(id="ib-shared-xrange", storage_type="memory"),

            html.Div(
                [
                    # ===== MAIN CHARTS (left) =====
                    html.Div(
                        [
                            dcc.Graph(
                                id="ironbeam-chart",
                                style={"height": "calc(100vh - 520px)"},
                                config={
                                    "displaylogo": False,
                                    "scrollZoom": True,
                                    "modeBarButtonsToRemove": ["autoScale2d"],
                                    "responsive": True,
                                },
                            ),

                            dcc.Graph(
                                id="ironbeam-flow-chart",
                                style={"height": "260px", "marginTop": "10px"},
                                config={
                                    "displayModeBar": True,
                                    "scrollZoom": True,
                                    "displaylogo": False,
                                    "responsive": True,
                                },
                            ),

                            # Placeholder: future dynamic plugin panels will render here
                            html.Div(id="ib-indicator-panels", style={"marginTop": "10px"}),
                        ],
                        style={"flex": "1", "minWidth": 0},
                    ),

                    # ===== INDICATORS SIDEBAR (right) =====
                    html.Div(
                        [
                            html.Div(
                                "indicators",
                                style={
                                    "color": "white",
                                    "fontWeight": "700",
                                    "fontSize": "14px",
                                    "marginBottom": "8px",
                                },
                            ),

                            dcc.Checklist(
                                id="ib-indicator-enabled",
                                options=[
                                    {"label": "Aggressor Flow (current bottom chart)", "value": "aggressor_flow"},
                                ],
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
                                options=[
                                    {"label": "Aggressor Flow", "value": "aggressor_flow"},
                                ],
                                value="aggressor_flow",
                                clearable=False,
                                style={
                                    "backgroundColor": "#0b1220",
                                    "color": "black",
                                },
                            ),

                            # Settings form placeholder (Step 4 will populate and bind)
                            html.Div(
                                id="ib-settings-form",
                                children=html.Div(
                                    "Step 4 will wire these settings to the indicator and persist them.",
                                    style={
                                        "color": "#9ca3af",
                                        "fontSize": "12px",
                                        "marginTop": "10px",
                                        "lineHeight": "16px",
                                    },
                                ),
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
