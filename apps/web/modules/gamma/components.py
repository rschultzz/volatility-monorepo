# modules/gex/components.py
from dash import html, dcc

# CR-DH: Levels placement switch, same labels as the React price chart's. Default Corrected,
# remembered per browser; independent of the price chart's own switch.
GEX_LEVELS_MODE_ID = "gex-levels-mode"
GEX_LEVELS_ROW_PX = 26


def gex_block():
    return html.Div(
        id="gex-block",
        className="p-2",
        style={"height": "100%"},
        children=[

            html.Div(
                [
                    html.Span("Levels:", style={"color": "white", "fontSize": "13px", "marginRight": "10px"}),
                    dcc.RadioItems(
                        id=GEX_LEVELS_MODE_ID,
                        options=[
                            {"label": "Current", "value": "current"},
                            {"label": "Corrected", "value": "corrected"},
                        ],
                        value="corrected",
                        persistence=True,
                        persistence_type="local",
                        inline=True,
                        labelStyle={"marginRight": "16px", "color": "white", "fontSize": "13px"},
                        inputStyle={"marginRight": "6px"},
                    ),
                ],
                style={"display": "flex", "alignItems": "center", "height": f"{GEX_LEVELS_ROW_PX}px"},
            ),

            dcc.Graph(
                id="GEX_GRAPH",
                config={"displayModeBar": True, "responsive": True},
                style={"height": f"calc(100% - {GEX_LEVELS_ROW_PX}px)"}
            ),

        ],
    )
