import dash_bootstrap_components as dbc
from dash import dcc, html


def make_term_metrics_block():
    return html.Div(
        [
            # Just the graph; the figure's title ("Term Metrics") will be shown inside it
            dcc.Graph(
                id="term-metrics-table",
                style={"height": "160px"},
            ),
        ],
        className="mb-4",
    )
