# apps/web/modules/Ironbeam/components.py

from dash import html, dcc


def ironbeam_layout():
    """
    Ironbeam block: auto-refresh timer + main ES/GEX chart + aggressor flow chart.

    Notes:
    - The main chart remains `ironbeam-chart`.
    - The new indicator chart is `ironbeam-flow-chart` (used by the new flow callback).
    """
    return html.Div(
        [
            # Auto-refresh timer for live bars / overlays
            dcc.Interval(
                id="ironbeam-interval",
                interval=10000,  # 10s (adjust if you want snappier updates)
                n_intervals=0,
            ),

            # Main ES + GEX chart
            dcc.Graph(
                id="ironbeam-chart",
                style={"height": "calc(100vh - 520px)"},
                config={
                    "displaylogo": False,
                    "scrollZoom": True,  # wheel zoom on hover
                    "modeBarButtonsToRemove": [
                        "autoScale2d",
                    ],
                    "responsive": True,
                },
            ),

            # Aggressor Flow / CVD-style indicator chart
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
        ],
        style={"marginTop": "4px"},
    )
