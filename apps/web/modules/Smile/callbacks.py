##Direct From API

from __future__ import annotations
import datetime as dt
import math
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import Input, Output

# ---- App IDs ----
TRADE_DATE_ID = "trade-date"
EXPIRATION_ID = "expiration-date-pick"
SMILE_TIME_INPUT = "smile-time-input"
SMILE_GRAPH = "SMILE_GRAPH"
EXPECTED_TOGGLE_ID = "expected-ss-toggle"
CLOCK_ID = "CLOCK"

# ---- Data helpers ----
from packages.shared.options_orats import fetch_one_minute_monies, pt_minute_to_et, PT_TZ
from packages.shared.surface_compare import k_for_abs_delta

TICKER = "SPX"
EPS_T = 1e-4
NEEDED_COLS = ["vol25", "vol50", "vol75"]

# leverage add-on (must match what you want on the Smile)
BETA_VOLPTS_PER_1PCT = 4.5       # vol points per 1% spot down
BETA_MAX_SHIFT_PP = 6.0          # clamp add-on in +/- vol points

# fixed colorway so expected traces can reuse the same color as their actual slice
COLORWAY = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52"
]

# ---------------- utils ----------------
def _get_row(df: pd.DataFrame) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    row = df.iloc[0]
    if not all(c in row.index for c in NEEDED_COLS):
        return None
    return row

def _years_to_exp(ts_et: dt.datetime, expiration_iso: str) -> float:
    exp_date = dt.date.fromisoformat(expiration_iso)
    rem = dt.datetime.combine(exp_date, dt.time(0, 0)) - ts_et.replace(tzinfo=None)
    T = max(0.0, rem.days/365.0 + rem.seconds/(365.0*24*3600))
    return max(T, EPS_T)

def _available_buckets(row: pd.Series) -> List[int]:
    """
    Return available ORATS 'volNN' buckets as ints (e.g., 95,90,...,5).
    """
    out = []
    for c in row.index:
        if c.startswith("vol") and c[3:].isdigit():
            n = int(c[3:])
            if 1 <= n <= 99:
                out.append(n)
    return sorted(out, reverse=True)

def _bucket_labels_order(buckets: List[int]) -> Tuple[List[int], List[str]]:
    """
    Order for plotting:
      P side: 95 -> 50, then ATM, then C side: 45 -> 5
    Labels:
      n>50 -> P{100-n}, n=50 -> ATM, n<50 -> C{n}
    """
    puts = [n for n in buckets if n >= 50]
    calls = [n for n in buckets if n < 50]
    order = puts + calls
    labels = []
    for n in order:
        if n > 50:
            labels.append(f"P{100-n}")
        elif n == 50:
            labels.append("ATM")
        else:
            labels.append(f"C{n}")
    return order, labels

def _k_grid_for_row(row: pd.Series, T: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (k, sigma) grid for a row using its *own* ATM for Δ->k mapping.
    """
    buckets = _available_buckets(row)
    if not buckets or 50 not in buckets:
        raise ValueError("row missing buckets/ATM")
    atm = float(row["vol50"])
    k_list, s_list = [], []
    for n in buckets:
        if n == 50:
            k = 0.0
        else:
            # convert volNN to absolute delta:
            if n > 50:
                p, is_put = (100 - n) / 100.0, True
            else:
                p, is_put = n / 100.0, False
            k = k_for_abs_delta(p, is_put=is_put, sigma=atm, T=T)
        k_list.append(k)
        s_list.append(float(row[f"vol{n}"]))
    k = np.array(k_list, float)
    s = np.array(s_list, float)
    # ensure strict monotonic k
    mask = np.concatenate(([True], np.diff(k) > 1e-12))
    return k[mask], s[mask]

def _interp_linear_extrap(x: float, xs: np.ndarray, ys: np.ndarray) -> float:
    if x <= xs[0]:
        x0, x1, y0, y1 = xs[0], xs[1], ys[0], ys[1]
        return float(y0 + (y1 - y0) * (x - x0) / (x1 - x0))
    if x >= xs[-1]:
        x0, x1, y0, y1 = xs[-2], xs[-1], ys[-2], ys[-1]
        return float(y1 + (y1 - y0) * (x - x1) / (x1 - x0))
    return float(np.interp(x, xs, ys))

# --------------- expected curve ----------------
def _expected_curve_shifted(prev_row: pd.Series,
                            prev_T: float,
                            prev_stock: float,
                            now_row: pd.Series,
                            now_T: float,
                            now_stock: float) -> Tuple[List[str], np.ndarray, float]:
    """
    Build the dotted expected curve for the current minute as:
      1) SHAPE: sticky-strike from previous surface using *current* σ_now for Δ->k,
         evaluate prev surface at (k_now + k_shift).
      2) ATM anchor: expected ATM = prev surface at k_shift + leverage add-on.
      3) VERTICAL SHIFT: shift entire shape by (atm_exp - shape_atm) so ATM lines up.

    Returns:
      labels (x axis), expected_y (percent), atm_exp_percent
    """
    # previous surface in its own k-space
    k_prev, s_prev = _k_grid_for_row(prev_row, prev_T)
    # k shift from spot change
    k_shift = math.log(now_stock / prev_stock) if (prev_stock and now_stock) else 0.0

    # buckets/x-labels to display (use what's in "now" row so x matches live line)
    buckets = _available_buckets(now_row)
    # ---- filter out 5Δ (P5/C5): drop 95 and 5
    buckets = [n for n in buckets if n not in (95, 5)]
    if not buckets or 50 not in buckets:
        raise ValueError("now row missing buckets/ATM")
    order, labels = _bucket_labels_order(buckets)

    # current ATM (fraction) for Δ->k mapping
    atm_now = float(now_row["vol50"])

    # --- shape values from previous surface at (k_now + k_shift)
    shape_vals = []
    for n in order:
        if n == 50:
            k_now = 0.0
        else:
            if n > 50:
                p, is_put = (100 - n) / 100.0, True
            else:
                p, is_put = n / 100.0, False
            k_now = k_for_abs_delta(p, is_put=is_put, sigma=atm_now, T=now_T)
        shape_vals.append(_interp_linear_extrap(k_now + k_shift, k_prev, s_prev))
    shape = np.array(shape_vals, float)

    # --- expected ATM anchor: previous surface at k_shift + leverage add-on
    exp_atm_shape = _interp_linear_extrap(k_shift, k_prev, s_prev)
    ret_frac = (now_stock - prev_stock) / prev_stock
    level_shift_pp = max(-BETA_MAX_SHIFT_PP,
                         min(BETA_MAX_SHIFT_PP, (-ret_frac) * 100.0 * BETA_VOLPTS_PER_1PCT))
    atm_exp = exp_atm_shape + level_shift_pp / 100.0

    # --- vertical shift of entire curve so ATM equals atm_exp
    shift = atm_exp - exp_atm_shape
    expected = shape + shift

    return labels, expected * 100.0, atm_exp * 100.0  # return in percent for plotting

# ----------------- main callback -----------------
def register_callbacks(app):
    @app.callback(
        Output(SMILE_GRAPH, "figure"),
        Input(TRADE_DATE_ID, "date"),
        Input(EXPIRATION_ID, "date"),
        Input(SMILE_TIME_INPUT, "value"),
        Input(EXPECTED_TOGGLE_ID, "value"),
        Input(CLOCK_ID, "n_intervals"),
    )
    def render_smile(trade_date_iso, expiration_iso, times_pt, expected_value, _tick):
        expected_on = (expected_value != "off")

        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=40),
            title=f"ORATS Smile Grid — {trade_date_iso or ''} (Exp: {expiration_iso or ''})",
            xaxis_title="Bucket (P10 … ATM … C10)",
            yaxis_title="IV (%)",
            legend=dict(orientation="v", x=1.02, y=1.0, bgcolor="rgba(0,0,0,0)"),
            colorway=COLORWAY,
        )

        if not trade_date_iso or not expiration_iso:
            return fig

        # ensure a “now” point is present for today
        if not times_pt:
            times_pt = ["06:31"]
        now_pt = dt.datetime.now(PT_TZ)
        if trade_date_iso == now_pt.date().isoformat():
            hhmm = now_pt.strftime("%H:%M")
            if "06:30" <= hhmm <= "13:00":
                times_pt = sorted(set(times_pt + [hhmm]))
        times_sorted = sorted(times_pt)

        prev_row = None
        prev_stock = None
        prev_T = None

        # plot each selected time
        for i, hhmm_pt in enumerate(times_sorted):
            ts_et = pt_minute_to_et(trade_date_iso, hhmm_pt)
            df_now = fetch_one_minute_monies(ts_et, TICKER, expiration_iso)
            row_now = _get_row(df_now)
            if row_now is None:
                continue

            # choose the color for this timeslice and reuse for expected
            color = COLORWAY[i % len(COLORWAY)]

            # live line (actual)
            buckets_now = _available_buckets(row_now)
            # ---- filter out 5Δ (P5/C5): drop 95 and 5
            buckets_now = [n for n in buckets_now if n not in (95, 5)]
            order_now, labels_now = _bucket_labels_order(buckets_now)
            y_now = [float(row_now[f"vol{n}"]) * 100.0 for n in order_now]

            fig.add_trace(go.Scatter(
                x=labels_now, y=y_now,
                mode="lines+markers",
                name=f"{hhmm_pt} PT",
                line=dict(width=2, color=color),
                marker=dict(size=5, color=color),
            ))

            # expected dotted curve for slices after the first, if toggle ON
            stock_now = float(pd.to_numeric(df_now["stockPrice"], errors="coerce").median()) if "stockPrice" in df_now.columns else None
            if expected_on and prev_row is not None and prev_T is not None and prev_stock is not None and stock_now is not None:
                try:
                    labels_exp, expected_y, atm_exp_pct = _expected_curve_shifted(
                        prev_row, prev_T, prev_stock,
                        row_now, _years_to_exp(ts_et, expiration_iso), stock_now
                    )

                    # dotted expected line — same color as actual
                    fig.add_trace(go.Scatter(
                        x=labels_exp, y=expected_y,
                        mode="lines",
                        name=f"Expected (SS) — {hhmm_pt}",
                        line=dict(width=2, dash="dot", color=color),
                    ))

                    # triangle marker for ATM expected (no legend) — same color
                    fig.add_trace(go.Scatter(
                        x=["ATM"], y=[atm_exp_pct],
                        mode="markers",
                        marker=dict(symbol="triangle-up", size=9, color=color),
                        name="ATM exp (SS)",
                        showlegend=False,
                    ))
                except Exception:
                    pass

            # advance previous pointers
            prev_row = row_now
            prev_stock = stock_now
            prev_T = _years_to_exp(ts_et, expiration_iso)

        return fig
