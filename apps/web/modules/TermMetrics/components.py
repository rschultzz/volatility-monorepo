import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html

def make_term_metrics_block():
    term_metrics_table = dbc.Col(
        [
            dbc.Label("Term Metrics", html_for="term-metrics-table"),
            dash_table.DataTable(
                id="term-metrics-table",
                columns=[
                    {"name": "Metric", "id": "metric"},
                    {"name": "Value", "id": "value"},
                ],
                style_as_list_view=True,
                style_cell={"padding": "5px"},
                style_header={
                    "backgroundColor": "rgb(30, 30, 30)",
                    "color": "white",
                    "fontWeight": "bold",
                },
                style_data={"backgroundColor": "rgb(50, 50, 50)", "color": "white"},
            ),
        ],
        width=12,
        className="mb-3",
    )

    term_metrics_graph = dbc.Col(
        [
            dbc.Label("Term Metrics Over Time", html_for="term-metrics-graph"),
            dcc.Graph(id="term-metrics-graph"),
        ],
        width=12,
    )

    return dbc.Row([term_metrics_table, term_metrics_graph])
