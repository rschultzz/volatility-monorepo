from __future__ import annotations
from dash import dcc, html

TERM_STRUCTURE_GRAPH_ID = "term-structure-graph"

def make_term_structure_block() -> html.Div:
    """
    Returns the HTML block for the Term Structure graph.
    """
    return html.Div(
        [
            html.H3( style={"textAlign": "center"}),
            dcc.Graph(
                id=TERM_STRUCTURE_GRAPH_ID,
                style={"height": "100%"}
            ),
        ],
        style={"minWidth": 0, "height": "100%"},
    )
