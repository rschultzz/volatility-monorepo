# apps/web/app.py

from __future__ import annotations

# load local environment variables from .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --- make repo root importable so `packages.*` works on Render ---
import sys
from pathlib import Path as _P

REPO_ROOT = _P(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# -----------------------------------------------------------------

import datetime as dt
from typing import List
from dash import Dash, html, dcc
import dash_auth

# ===== Modules =====
from modules.Skew.components import make_skew_block
from modules.Skew.callbacks import register_callbacks as register_skew
from modules.gamma.components import gex_block
from modules.gamma import callbacks as _gex_callbacks
from modules.Smile.callbacks import register_callbacks as register_smile
from modules.TermStructure.components import make_term_structure_block
from modules.TermStructure.callbacks import register_callbacks as register_term_structure

# ===== IDs =====
CLOCK_ID = "CLOCK"
TRADE_DATE_PICK = "trade-date"
EXPIRATION_DATE_PICK = "expiration-date-pick"
SMILE_TIME_INPUT = "smile-time-input"
SMILE_GRAPH = "SMILE_GRAPH"
EXPECTED_TOGGLE_ID = "expected-ss-toggle"

# ===== UI Helpers =====
def get_default_trade_date() -> dt.date:
    today = dt.date.today()
    if today.weekday() == 5: return today - dt.timedelta(days=1) # Sat -> Fri
    if today.weekday() == 6: return today - dt.timedelta(days=2) # Sun -> Fri
    return today

def third_friday_of_month(year, month):
    for day in range(15, 22):
        d = dt.date(year, month, day)
        if d.weekday() == 4: return d
    return None

def third_friday_next_month() -> dt.date:
    today = dt.date.today()
    y, m = (today.year, today.month + 1)
    if m > 12: y, m = y + 1, 1
    return third_friday_of_month(y, m)

def pt_time_options(start="06:30", end="13:00", step_min=1) -> List[dict]:
    t0, t1 = dt.datetime.strptime(start, "%H:%M"), dt.datetime.strptime(end, "%H:%M")
    out, cur = [], t0
    while cur <= t1:
        hhmm = cur.strftime("%H:%M")
        out.append({"label": f"{hhmm} PT", "value": hhmm})
        cur += dt.timedelta(minutes=step_min)
    return out

# ===== App =====
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server

VALID_USERNAME_PASSWORD_PAIRS = {"ryan": "ChangeThisPassword123!", "sara": "ChangeThisPassword123!"}
auth = dash_auth.BasicAuth(app, VALID_USERNAME_PASSWORD_PAIRS)

app.layout = html.Div(
    [
        html.Div([
            html.Div("Surface Dynamics", style={"fontWeight": "600", "fontSize": "20px", "color": "#e5e7eb", "textAlign": "center", "width": "100%"}),
            html.A("Home", href="https://blog.surfacedynamics.io", style={"color": "#93c5fd", "textDecoration": "none", "fontWeight": "500", "padding": "4px 10px", "borderRadius": "6px", "border": "1px solid #1f2937", "position": "absolute", "right": "16px", "top": "50%", "transform": "translateY(-50%)"}),
        ], style={"position": "relative", "padding": "8px 16px", "borderBottom": "1px solid #1f2937", "marginBottom": "8px"}),
        
        dcc.Interval(id=CLOCK_ID, interval=60_000, n_intervals=0),
        
        html.Div([
            html.Div([
                html.Label("Trade Date", style={"color": "white", "marginBottom": "4px"}),
                dcc.DatePickerSingle(id=TRADE_DATE_PICK, disabled=False, display_format="YYYY-MM-DD", date=get_default_trade_date()),
            ], style={"marginRight": "16px", "display": "flex", "flexDirection": "column"}),
            html.Div([
                html.Label("Expiration", style={"color": "white", "marginBottom": "4px"}),
                dcc.DatePickerSingle(id=EXPIRATION_DATE_PICK, disabled=False, display_format="YYYY-MM-DD", date=third_friday_next_month()),
            ], style={"marginRight": "16px", "display": "flex", "flexDirection": "column"}),
            html.Div([
                html.Label("Time Slices (PT)", style={"color": "white", "marginBottom": "4px"}),
                dcc.Dropdown(id=SMILE_TIME_INPUT, options=pt_time_options(), value=["06:31"], multi=True, style={"minWidth": "320px"}),
            ], style={"marginRight": "16px", "display": "flex", "flexDirection": "column"}),
            html.Div([
                html.Label("Expected (SS)", style={"color": "white", "marginBottom": "4px"}),
                dcc.RadioItems(id=EXPECTED_TOGGLE_ID, options=[{"label": "ON", "value": "on"}, {"label": "OFF", "value": "off"}], value="on", inline=True, inputStyle={"marginRight": "6px"}, labelStyle={"marginRight": "12px"}, style={"color": "white"}),
            ], style={"display": "flex", "alignItems": "flex-start", "gap": "16px", "flexWrap": "wrap"}),
        ], style={"display": "flex", "alignItems": "flex-start", "gap": "16px", "flexWrap": "wrap"}),
        
        html.Hr(style={"borderColor": "#444"}),
        
        html.Div([
            html.Div(dcc.Graph(id=SMILE_GRAPH, style={"height": "840px"}), style={"minWidth": 0}),
            html.Div(gex_block(), style={"minWidth": 0}),
        ], style={"display": "grid", "gridTemplateColumns": "2fr 1fr", "gap": "16px", "alignItems": "stretch"}),
        
        html.Hr(style={"borderColor": "#333"}),
        
        # --- New Term Structure and Skew Row ---
        html.Div([
            make_term_structure_block(),
            make_skew_block(),
        ], style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "16px", "alignItems": "stretch"}),

    ], style={"backgroundColor": "black", "color": "white", "minHeight": "100vh", "padding": "0 16px 16px"}
)

# Register all module callbacks
register_smile(app)
register_skew(app)
register_term_structure(app)

if __name__ == "__main__":
    import os, socket
    def _choose_port(preferred=8050, tries=50):
        p = os.getenv("PORT")
        if p: return int(p)
        for port in [preferred] + list(range(preferred + 1, preferred + tries)):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try: s.bind(("0.0.0.0", port)); return port
                except OSError: continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0)); return s.getsockname()[1]
    app.run(host="0.0.0.0", port=_choose_port(), debug=True)
