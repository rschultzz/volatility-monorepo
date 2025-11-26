# apps/web/modules/Ironbeam/components.py
from dash import dcc, html

def ironbeam_layout():
    return html.Div([
        html.H2("Ironbeam ES 1-Minute Bars"),
        dcc.Graph(id='ironbeam-chart'),
        dcc.Interval(
            id='ironbeam-interval',
            interval=1 * 1000,  # in milliseconds
            n_intervals=0
        )
    ])
