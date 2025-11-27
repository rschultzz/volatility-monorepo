from __future__ import annotations
import datetime as dt
import math
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import Input, Output
import pytz

from packages.shared.utils import fetch_live_orats_data, fetch_data_from_db
from packages.shared.surface_compare import k_for_abs_delta
from packages.shared.options_orats import pt_minute_to_et

# ---- App IDs ----
TRADE_DATE_ID = "trade-date"
EXPIRATION_ID = "expiration-date-pick"
SMILE_TIME_INPUT = "smile-time-input"
SMILE_GRAPH = "SMILE_GRAPH"
EXPECTED_TOGGLE_ID = "expected-ss-toggle"
CLOCK_ID = "CLOCK"
LIVE_DATA_STORE_ID = "live-data-store"
LIVE_UPDATE_TIMER_ID = "live-update-timer"

# ---- Plotting Constants ----
TICKER = "SPX"
EPS_T = 1e-4
BETA_VOLPTS_PER_1PCT = 4.5
BETA_MAX_SHIFT_PP = 6.0
COLORWAY = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
]
LIVE_COLOR = "#FFD700"  # Gold for live

# ---- Market Hours ----
MARKET_OPEN = dt.time(9, 30)
MARKET_CLOSE = dt.time(16, 0)
MARKET_TIMEZONE = pytz.timezone("US/Eastern")


# ----------------- Time / bucket utils -----------------
def _years_to_exp(ts_et: dt.datetime, expiration_iso: str) -> float:
    exp_date = dt.date.fromisoformat(expiration_iso)
    rem = dt.datetime.combine(exp_date, dt.time(0, 0)) - ts_et.replace(tzinfo=None)
    T = max(0.0, rem.days / 365.0 + rem.seconds / (365.0 * 24 * 3600))
    return max(T, EPS_T)


def _available_buckets(row: pd.Series) -> List[int]:
    out: List[int] = []
    for c in row.index:
        if c.startswith("vol") and c[3:].isdigit() and not pd.isna(row[c]):
            n = int(c[3:])
            if 1 <= n <= 99:
                out.append(n)
    return sorted(out, reverse=True)


def _bucket_labels_order(buckets: List[int]) -> Tuple[List[int], List[str]]:
    puts = [n for n in buckets if n >= 50]
    calls = [n for n in buckets if n < 50]
    order = puts + calls
    labels: List[str] = []
    for n in order:
        if n > 50:
            labels.append(f"P{100-n}")
        elif n == 50:
            labels.append("ATM")
        else:
            labels.append(f"C{n}")
    return order, labels

# ----------------- Main Callbacks -----------------
def register_callbacks(app):
    @app.callback(
        Output(LIVE_DATA_STORE_ID, "data"),
        Input(LIVE_UPDATE_TIMER_ID, "n_intervals"),
    )
    def update_live_data(n):
        df_live = fetch_live_orats_data()
        if df_live is not None and not df_live.empty:
            return df_live.to_json(orient="split")
        return None

    @app.callback(
        Output(SMILE_GRAPH, "figure"),
        [
            Input(TRADE_DATE_ID, "date"),
            Input(EXPIRATION_ID, "date"),
            Input(SMILE_TIME_INPUT, "value"),
            Input(EXPECTED_TOGGLE_ID, "value"),
            Input(LIVE_DATA_STORE_ID, "data"),
        ],
    )
    def render_smile(trade_date_iso, expiration_iso, times_pt, expected_value, live_data_json):
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=40),
            title=f"Smile Grid — {trade_date_iso or ''} (Exp: {expiration_iso or ''})",
            xaxis_title="Bucket (P10 … ATM … C10)",
            yaxis_title="IV (%)",
            legend=dict(orientation="v", x=1.02, y=1.0, bgcolor="rgba(0,0,0,0)"),
            colorway=COLORWAY,
        )

        if not trade_date_iso or not expiration_iso:
            return fig

        # --------- Historical slices from DB ---------
        if times_pt:
            times_sorted = sorted(times_pt)
            df = fetch_data_from_db(trade_date_iso, expiration_iso, times_sorted)
            
            # --- DIAGNOSTIC LOGGING ---
            print(f"--- Smile Callback Diagnostics (Date: {trade_date_iso}) ---")
            if df is not None and not df.empty:
                print(f"DataFrame shape: {df.shape}")
                print("DataFrame head:")
                print(df.head())
            else:
                print("DataFrame is empty or None.")
            print("----------------------------------------------------")
            # --- END DIAGNOSTIC LOGGING ---

            if df is not None and not df.empty:
                df = df.copy()
                df["snapshot_pt_time"] = pd.to_datetime(df["snapshot_pt"]).dt.strftime("%H:%M")
                for i, hhmm_pt in enumerate(times_sorted):
                    rows = df[df["snapshot_pt_time"] == hhmm_pt]
                    if rows.empty:
                        continue
                    
                    row_now = rows.iloc[0]
                    color = COLORWAY[i % len(COLORWAY)]

                    buckets_now = _available_buckets(row_now)
                    buckets_now = [n for n in buckets_now if n not in (95, 5)]
                    if not buckets_now or 50 not in buckets_now:
                        continue
                    
                    order_now, labels_now = _bucket_labels_order(buckets_now)
                    
                    try:
                        y_now = [float(row_now[f"vol{n}"]) * 100.0 for n in order_now]
                    except (KeyError, TypeError):
                        continue

                    if y_now and not all(pd.isna(y_now)):
                        fig.add_trace(
                            go.Scatter(
                                x=labels_now,
                                y=y_now,
                                mode="lines+markers",
                                name=f"{hhmm_pt} PT",
                                line=dict(width=2, color=color),
                                marker=dict(size=5, color=color),
                            )
                        )

        # --------- Live slice from ORATS API ---------
        now_et = dt.datetime.now(MARKET_TIMEZONE)
        is_market_hours = MARKET_OPEN <= now_et.time() <= MARKET_CLOSE and now_et.weekday() < 5

        if live_data_json and is_market_hours:
            df_live = pd.read_json(live_data_json, orient="split")
            if df_live is not None and not df_live.empty:
                live_row_df = df_live[df_live["expir_date"] == expiration_iso]
                if not live_row_df.empty:
                    live_row = live_row_df.iloc[0]

                    buckets_live = _available_buckets(live_row)
                    buckets_live = [n for n in buckets_live if n not in (95, 5)]
                    if buckets_live and 50 in buckets_live:
                        order_live, labels_live = _bucket_labels_order(buckets_live)
                        try:
                            y_live = [float(live_row[f"vol{n}"]) * 100.0 for n in order_live]
                        except (KeyError, TypeError):
                            y_live = []

                        if y_live and not all(pd.isna(y_live)):
                            fig.add_trace(
                                go.Scatter(
                                    x=labels_live,
                                    y=y_live,
                                    mode="lines+markers",
                                    name="Live",
                                    line=dict(width=3, color=LIVE_COLOR),
                                    marker=dict(size=6, color=LIVE_COLOR),
                                )
                            )
        return fig
