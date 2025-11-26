# apps/web/modules/Ironbeam/components.py
from dash import dcc, html

def ironbeam_layout():
    return html.Div([
        html.H2("Ironbeam ES 1-Minute Bars"),
        # Hidden store to track the timestamp of the latest bar
        dcc.Store(id='ironbeam-latest-ts-store'),
        dcc.Graph(id='ironbeam-chart'),
        dcc.Interval(
            id='ironbeam-interval',
            interval=1 * 1000,  # in milliseconds
            n_intervals=0
        )
    ])
