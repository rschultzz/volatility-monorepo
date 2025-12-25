# apps/web/modules/Ironbeam/components.py

from dash import html, dcc


def ironbeam_layout():
    """
    Ironbeam block: auto-refresh timer + main ES/GEX chart.
    (Bar-interval toggle now lives in app.py next to the GEX slider.)
    """
    return html.Div(
        [
            # Auto-refresh timer for live bars
            dcc.Interval(
                id="ironbeam-interval",
                interval=10000,
                n_intervals=0,
            ),

            # Main ES + GEX chart
            dcc.Graph(
                id="ironbeam-chart",
                style={"height": "calc(100vh - 240px)"},
                config={
                    "displaylogo": False,
                    "scrollZoom": True,           # wheel zoom on hover
                    "modeBarButtonsToRemove": [
                        "autoScale2d",
                    ],
                    "responsive": True,
                },
            ),
        ],
        style={"marginTop": "4px"},
    )
