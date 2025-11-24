# modules/gex/components.py
from dash import html, dcc

def gex_block():
    return html.Div(
        id="gex-block",
        className="p-2",
        children=[

            dcc.Graph(
                id="GEX_GRAPH",
                config={"displayModeBar": True, "responsive": True},
                style={"height": "840px"}
            ),

        ],
    )
