# app_modular.py — Smile (time slices) on left (2/3) + GEX (date-only) on right (1/3)
from __future__ import annotations
# load local environment variables from .env
try:
    from dotenv import load_dotenv
    load_dotenv()  # reads .env in the project root
except Exception:
    pass
# Load local env vars explicitly from project root
from pathlib import Path
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
except Exception:
    pass


import datetime as dt
from typing import List, Tuple

import pandas as pd
import plotly.graph_objs as go
from dash import Dash, html, dcc
from dash import Input, Output

# ===== Existing Skew module (unchanged) =====
from modules.Skew.components import make_skew_block
from modules.Skew.callbacks import register_callbacks as register_skew

# ===== GEX module (components + callbacks) =====
from modules.gamma.components import gex_block
from modules.gamma import callbacks as _gex_callbacks  # registers the GEX callback

# ===== ORATS helpers for Smile (unchanged) =====
from shared.options_orats import fetch_one_minute_monies, pt_minute_to_et, PT_TZ

# ===== IDs =====
CLOCK_ID = "CLOCK"
TRADE_DATE_PICK = "trade-date"
EXPIRATION_DATE_PICK = "expiration-date-pick"
SMILE_TIME_INPUT = "smile-time-input"
SMILE_GRAPH = "SMILE_GRAPH"

# ===== Smile constants (unchanged) =====
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


# ===== UI helpers (unchanged) =====
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
        # 1-minute heartbeat (keeps *today* fresh; GEX ignores time selection)
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
            ],
            style={"display": "flex", "alignItems": "center", "gap": "16px"},
        ),

        html.Hr(style={"borderColor": "#444"}),

        # ===== Two-column content row: Smile (2fr) | GEX (1fr) =====
        html.Div(
            [
                # Left: Smile graph (2/3 width)
                html.Div(
                    dcc.Graph(id=SMILE_GRAPH, style={"height": "840px"}),
                    style={"minWidth": 0},  # prevents overflow in flex/grid
                ),
                # Right: GEX block (1/3 width)
                html.Div(
                    gex_block(),
                    style={"minWidth": 0},
                ),
            ],
            style={
                "display": "grid",
                "gridTemplateColumns": "2fr 1fr",  # 2/3 vs 1/3
                "gap": "16px",
                "alignItems": "stretch",
            },
        ),

        html.Hr(style={"borderColor": "#333"}),

        # Skew block (unchanged, full width)
        make_skew_block(),
    ],
    style={"backgroundColor": "black", "color": "white", "minHeight": "100vh", "padding": "16px"},
)

# Register Skew callbacks (unchanged)
register_skew(app)


# ===== Smile callback (unchanged logic) =====
@app.callback(
    Output(SMILE_GRAPH, "figure"),
    Input(TRADE_DATE_PICK, "date"),
    Input(EXPIRATION_DATE_PICK, "date"),
    Input(SMILE_TIME_INPUT, "value"),
    Input(CLOCK_ID, "n_intervals"),
)
def render_smile(trade_date_iso, expiration_iso, times_pt, _tick):
    if not trade_date_iso or not expiration_iso:
        return go.Figure().update_layout(
            template="plotly_dark",
            title="Select Trade Date & Expiration",
            xaxis_title="Bucket (P10 … ATM … C10)",
            yaxis_title="IV (%)",
        )

    if not times_pt:
        times_pt = ["06:31"]

    now_pt = dt.datetime.now(PT_TZ)
    if trade_date_iso == now_pt.date().isoformat():
        now_hhmm = now_pt.strftime("%H:%M")
        if "06:30" <= now_hhmm <= "13:00":
            times_pt = sorted(set(times_pt + [now_hhmm]))

    traces = []
    for hhmm_pt in sorted(times_pt):
        ts_et = pt_minute_to_et(trade_date_iso, hhmm_pt)
        df = fetch_one_minute_monies(ts_et, TICKER, expiration_iso)
        if df is None or df.empty:
            continue

        row0 = df.iloc[0]
        labels, ivs = row_to_full_bucket_line(row0)
        if not labels:
            continue

        traces.append(go.Scatter(x=labels, y=ivs, mode="lines+markers", name=f"{hhmm_pt} PT"))

    if not traces:
        return go.Figure().update_layout(
            template="plotly_dark",
            title=f"No monies data for {trade_date_iso} / {expiration_iso} at selected times",
            xaxis_title="Bucket (P10 … ATM … C10)",
            yaxis_title="IV (%)",
        )

    return go.Figure(data=traces).update_layout(
        template="plotly_dark",
        title=f"ORATS Smile Grid — {trade_date_iso} (Exp: {expiration_iso})",
        xaxis_title="Bucket (P10 … ATM … C10)",
        yaxis_title="IV (%)",
    )


server = app.server

if __name__ == "__main__":
    app.run(debug=True, port=0, use_reloader=False)
