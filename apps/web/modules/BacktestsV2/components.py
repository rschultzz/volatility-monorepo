from __future__ import annotations

from dash import html


CARD_STYLE = {
    "backgroundColor": "#0b1220",
    "border": "1px solid #1f2937",
    "borderRadius": "14px",
    "padding": "0",
    "height": "100%",
    "minHeight": 0,
    "overflow": "hidden",
}


IFRAME_STYLE = {
    "width": "100%",
    "height": "100%",
    "border": "0",
    "display": "block",
    "backgroundColor": "#020617",
}


def make_backtests_v2_tab() -> html.Div:
    return html.Div(
        [
            html.Iframe(
                src="/backtests-v2-preview/",
                style=IFRAME_STYLE,
            )
        ],
        style=CARD_STYLE,
    )
