# apps/web/modules/Ironbeam/components.py

from dash import html, dcc


def ironbeam_layout():
    """
    Ironbeam block: bar-interval toggle + auto-refresh + main ES/GEX chart.
    """
    return html.Div(
        [
            # --- Bar interval toggle ---
            html.Div(
                [
                    html.Label(
                        "Bar Interval",
                        style={
                            "color": "white",
                            "marginBottom": "4px",
                            "fontSize": "12px",
                        },
                    ),
                    dcc.RadioItems(
                        id="ironbeam-bar-interval",
                        options=[
                            {"label": "1 min", "value": "1min"},
                            {"label": "5 min", "value": "5min"},
                        ],
                        value="1min",   # default to 1-minute bars
                        inline=True,
                        labelStyle={"marginRight": "12px", "color": "white"},
                        inputStyle={"marginRight": "4px"},
                    ),
                ],
                style={
                    "marginBottom": "8px",
                    "display": "flex",
                    "flexDirection": "column",
                },
            ),

            # --- Auto-refresh timer for live bars ---
            dcc.Interval(
                id="ironbeam-interval",
                interval=60 * 1000,  # 1 minute refresh
                n_intervals=0,
            ),

            # --- Main ES + GEX chart ---
            dcc.Graph(
                id="ironbeam-chart",
                style={"height": "1500px"},
                config={
                    "displaylogo": False,
                    "scrollZoom": True,           # wheel zoom on hover
                    "modeBarButtonsToRemove": [
                        "autoScale2d",
                    ],
                },
            ),
        ],
        style={"marginTop": "4px"},
    )
