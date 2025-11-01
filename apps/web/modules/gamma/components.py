# modules/gex/components.py
from dash import html, dcc

def gex_block():
    return html.Div(
        id="gex-block",
        className="p-2",
        children=[
            html.H3("Gamma by Discounted Level", className="mb-1"),
            dcc.Graph(
                id="GEX_GRAPH",
                config={"displayModeBar": True, "responsive": True},
                style={"height": "780px"}
            ),
            html.Small("Calls → right, Puts → left; grouped by rounded discounted level"),
        ],
    )
