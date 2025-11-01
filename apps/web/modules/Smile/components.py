# modules/Smile/components.py — Smile block with PST time selector
from datetime import datetime, timedelta
from dash import html, dcc
import ids

def _pt_times(start="06:30", end="13:00", step_min=5):
    """Return [("HH:MM","HH:MM"), ...] in PT at step_min increments."""
    t0 = datetime.strptime(start, "%H:%M")
    t1 = datetime.strptime(end,   "%H:%M")
    out = []
    cur = t0
    while cur <= t1:
        hhmm = cur.strftime("%H:%M")
        out.append((hhmm, hhmm))  # (label value)
        cur += timedelta(minutes=step_min)
    return out

def make_smile_block():
    # Options: label "HH:MM PT", value "HH:MM" (plain for easy parsing)
    time_opts = [{"label": f"{hhmm} PT", "value": hhmm} for hhmm, _ in _pt_times()]
    default_times = ["06:35", "07:00", "08:00"]

    return html.Div([
        html.H3("ORATS Monies — Smile", style={"marginBottom": "8px"}),

        html.Div([
            html.Label("Time Slices (PT)", style={"marginRight": "8px"}),
            dcc.Dropdown(
                id=ids.SMILE_TIME_INPUT,
                options=time_opts,
                value=default_times,
                multi=True,
                placeholder="Select PT times…",
                style={"minWidth": "320px"}
            ),
        ], style={"display": "flex", "alignItems": "center", "gap": "8px", "marginBottom": "8px"}),

        dcc.Graph(id="SMILE_GRAPH", style={"height": "520px"}),
        dcc.Graph(id="SMILE_TABLE", style={"height": "320px", "marginTop": "8px"}),
    ], id="smile-block", style={"marginTop": "12px"})
