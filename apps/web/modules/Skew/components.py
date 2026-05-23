# modules/Skew/components.py — Skew (50Δ → 25Δ) table
from dash import html, dcc

# Local IDs (keep it self-contained)
SKEW_TABLE = "SKEW_TABLE"

def make_skew_block():
    return html.Div([

        dcc.Graph(id=SKEW_TABLE, style={"height": "100%"}),
    ], id="skew-block", style={"marginTop": "16px", "height": "100%"})
