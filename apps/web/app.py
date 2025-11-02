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
REPO_ROOT = _P(__file__).resolve().parents[2]  # .../volatility-monorepo
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# -----------------------------------------------------------------

import datetime as dt
from typing import List, Tuple

import pandas as pd
import plotly.graph_objs as go
from dash import Dash, html, dcc

# ===== Existing Skew module =====
from modules.Skew.components import make_skew_block
from modules.Skew.callbacks import register_callbacks as register_skew

# ===== GEX module =====
from modules.gamma.components import gex_block
from modules.gamma import callbacks as _gex_callbacks  # registers the GEX callback

# ===== ORATS helpers for Smile =====
from packages.shared.options_orats import fetch_one_minute_monies, pt_minute_to_et, PT_TZ

# ===== IDs =====
CLOCK_ID = "CLOCK"
TRADE_DATE_PICK = "trade-date"
EXPIRATION_DATE_PICK = "expiration-date-pick"
SMILE_TIME_INPUT = "smile-time-input"
SMILE_GRAPH = "SMILE_GRAPH"
EXPECTED_TOGGLE_ID = "expected-ss-toggle"   # <-- NEW external toggle

# ===== Smile constants =====
TICKER = "SPX"
CALL_DELTAS = [90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10]

def delta_label(call_delta: int) -> str:
    if call_delta > 50:
        return f"P{100 - call_delta}"
    if call_delta == 50:
        return "ATM"
    return f"C{call_delta}"

def row_to_full_bucket_line(row: pd.Series) -> Tuple[List[str], List[float]]:
    labels, ivs = [], []
    for d in CALL_DELTAS:
        col = f"vol{d}"
        if col not in row.index:
            continue
        v = pd.to_numeric(row.get(col), errors="coerce")
        if pd.notna(v):
            labels.append(delta_label(d))
            ivs.append(float(v) * 100.0)
    return labels, ivs

def get_default_trade_date() -> dt.date:
    today = dt.date.today()
    if today.weekday() == 5:  # Sat
        return today - dt.timedelta(days=1)
    if today.weekday() == 6:  # Sun
        return today - dt.timedelta(days=2)
    return today

def third_friday_of_month(year, month):
    for day in range(15, 22):
        d = dt.date(year, month, day)
        if d.weekday() == 4:
            return d
    return None

def third_friday_next_month() -> dt.date:
    today = dt.date.today()
    y, m = today.year, today.month + 1
    if m > 12:
        y, m = y + 1, 1
    return third_friday_of_month(y, m)

def pt_time_options(start="06:30", end="13:00", step_min=1) -> List[dict]:
    t0 = dt.datetime.strptime(start, "%H:%M")
    t1 = dt.datetime.strptime(end, "%H:%M")
    out = []
    cur = t0
    while cur <= t1:
        hhmm = cur.strftime("%H:%M")
        out.append({"label": f"{hhmm} PT", "value": hhmm})
        cur += dt.timedelta(minutes=step_min)
    return out

# ===== App =====
app = Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div(
    [
        # 1-minute heartbeat
        dcc.Interval(id=CLOCK_ID, interval=60_000, n_intervals=0),

        html.H3("Volatility Dash — Smile + Gamma", style={"color": "white"}),

        # Controls row
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Trade Date", style={"color": "white"}),
                        dcc.DatePickerSingle(
                            id=TRADE_DATE_PICK,
                            disabled=False,
                            display_format="YYYY-MM-DD",
                            date=get_default_trade_date(),
                        ),
                    ],
                    style={"marginRight": "16px"},
                ),
                html.Div(
                    [
                        html.Label("Expiration", style={"color": "white"}),
                        dcc.DatePickerSingle(
                            id=EXPIRATION_DATE_PICK,
                            disabled=False,
                            display_format="YYYY-MM-DD",
                            date=third_friday_next_month(),
                        ),
                    ],
                ),
                html.Div(
                    [
                        html.Label("Time Slices (PT) — Smile only", style={"color": "white"}),
                        dcc.Dropdown(
                            id=SMILE_TIME_INPUT,
                            options=pt_time_options(),
                            value=["06:31"],
                            multi=True,
                            style={"minWidth": "320px"},
                        ),
                        html.Small("(GEX ignores time; uses trade-date only)", style={"color": "#aaa"}),
                    ],
                    style={"marginLeft": "16px"},
                ),
                html.Div(
                    [
                        html.Label("Expected (SS)", style={"color": "white"}),
                        dcc.RadioItems(
                            id=EXPECTED_TOGGLE_ID,
                            options=[
                                {"label": "ON",  "value": "on"},
                                {"label": "OFF", "value": "off"},
                            ],
                            value="on",
                            inline=True,
                            inputStyle={"marginRight": "6px"},
                            labelStyle={"marginRight": "12px"},
                            style={"color": "white"},
                        ),
                    ],
                    style={"marginLeft": "24px"},
                ),
            ],
            style={"display": "flex", "alignItems": "center", "gap": "16px", "flexWrap": "wrap"},
        ),

        html.Hr(style={"borderColor": "#444"}),

        # ===== Two-column content row: Smile (2fr) | GEX (1fr) =====
        html.Div(
            [
                html.Div(
                    dcc.Graph(id=SMILE_GRAPH, style={"height": "840px"}),
                    style={"minWidth": 0},
                ),
                html.Div(
                    gex_block(),
                    style={"minWidth": 0},
                ),
            ],
            style={"display": "grid", "gridTemplateColumns": "2fr 1fr", "gap": "16px", "alignItems": "stretch"},
        ),

        html.Hr(style={"borderColor": "#333"}),

        # Skew block
        make_skew_block(),
    ],
    style={"backgroundColor": "black", "color": "white", "minHeight": "100vh", "padding": "16px"},
)

# Register callbacks
register_skew(app)
from modules.Smile.callbacks import register_callbacks as register_smile
register_smile(app)

server = app.server

# --- pick a free port locally ---
import os, socket
def _choose_port(preferred=8050, tries=50):
    p = os.getenv("PORT")
    if p:
        try: return int(p)
        except ValueError: pass
    for port in [preferred] + list(range(preferred + 1, preferred + tries)):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=_choose_port(), debug=True)
