# apps/web/modules/Ironbeam/components.py
from dash import dcc, html

def ironbeam_layout():
    return html.Div([
        html.H2("Ironbeam ES 1-Minute Bars"),
        dcc.Graph(
            id='ironbeam-chart',
            style={'height': '1440px'},
            config={
                "scrollZoom": True,  # mouse wheel zooms in/out
                "doubleClick": "autosize",  # double-click to reset axes
            },
        ),
        dcc.Interval(
            id='ironbeam-interval',
            interval=5 * 1000,  # in milliseconds
            n_intervals=0
        )
    ])
