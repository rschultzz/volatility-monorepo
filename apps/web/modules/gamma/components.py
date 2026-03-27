from dash import dcc, html


def gex_block():
    return html.Div(
        id="gex-block",
        className="p-2",
        style={"height": "100%", "display": "flex", "flexDirection": "column", "gap": "8px"},
        children=[
            html.Div(
                style={
                    "display": "flex",
                    "justifyContent": "space-between",
                    "alignItems": "center",
                    "gap": "12px",
                    "flexWrap": "wrap",
                },
                children=[
                    html.Div(
                        [
                            html.Div(
                                "Gamma Display",
                                style={"color": "#d1d5db", "fontSize": "12px", "fontWeight": "600", "marginBottom": "4px"},
                            ),
                            dcc.RadioItems(
                                id="gex-display-mode",
                                options=[
                                    {"label": "GEX", "value": "gex"},
                                    {"label": "Live Volume", "value": "volume"},
                                    {"label": "Live Proxy", "value": "live"},
                                    {"label": "Combined", "value": "combined"},
                                ],
                                value="gex",
                                inline=True,
                                labelStyle={"marginRight": "14px", "color": "white", "fontSize": "13px"},
                                inputStyle={"marginRight": "6px"},
                            ),
                        ]
                    ),
                    html.Div(
                        "Live modes refresh every 60s",
                        style={"color": "#9ca3af", "fontSize": "11px"},
                    ),
                ],
            ),
            dcc.Interval(id="gex-live-interval", interval=60 * 1000, n_intervals=0),
            dcc.Graph(
                id="GEX_GRAPH",
                config={"displayModeBar": True, "responsive": True},
                style={"height": "100%", "minHeight": 0, "flex": "1 1 auto"},
            ),
        ],
    )
