# modules/gex/components.py
from dash import html, dcc

def gex_block():
    return html.Div(
        id="gex-block",
        className="p-2",
        style={"height": "100%"},
        children=[

            dcc.Graph(
                id="GEX_GRAPH",
                config={"displayModeBar": True, "responsive": True},
                style={"height": "100%"}
            ),

        ],
    )
