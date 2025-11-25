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
    """
    Match the original smile callback: time to expiry in years using an ET
    timestamp and midnight on the expiration date.
    """
    exp_date = dt.date.fromisoformat(expiration_iso)
    rem = dt.datetime.combine(exp_date, dt.time(0, 0)) - ts_et.replace(tzinfo=None)
    T = max(0.0, rem.days / 365.0 + rem.seconds / (365.0 * 24 * 3600))
    return max(T, EPS_T)


def _available_buckets(row: pd.Series) -> List[int]:
    """
    Return available ORATS 'volNN' buckets as ints (e.g., 95, 90, ..., 5).
    """
    out: List[int] = []
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
    labels: List[str] = []
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
    Mirrors the original implementation so the sticky-strike logic matches.
    """
    buckets = _available_buckets(row)
    if not buckets or 50 not in buckets or "vol50" not in row or pd.isna(row["vol50"]):
        raise ValueError("row missing buckets/ATM for k-grid")

    atm = float(row["vol50"])
    k_list: List[float] = []
    s_list: List[float] = []
    for n in buckets:
        if n == 50:
            k = 0.0
        else:
            if n > 50:
                p, is_put = (100 - n) / 100.0, True
            else:
                p, is_put = n / 100.0, False
            k = k_for_abs_delta(p, is_put=is_put, sigma=atm, T=T)
        k_list.append(k)
        s_list.append(float(row[f"vol{n}"]))
    k = np.array(k_list, float)
    s = np.array(s_list, float)

    # ensure strict monotonic k (important for interpolation)
    mask = np.concatenate(([True], np.diff(k) > 1e-12))
    return k[mask], s[mask]


def _interp_linear_extrap(x: float, xs: np.ndarray, ys: np.ndarray) -> float:
    """
    Linear interpolation with straight-line extrapolation at the wings,
    same as in the original callback.
    """
    if xs.size == 0 or ys.size == 0:
        return float("nan")
    if xs.size == 1:
        return float(ys[0])

    if x <= xs[0]:
        x0, x1, y0, y1 = xs[0], xs[1], ys[0], ys[1]
        return float(y0 + (y1 - y0) * (x - x0) / (x1 - x0))
    if x >= xs[-1]:
        x0, x1, y0, y1 = xs[-2], xs[-1], ys[-2], ys[-1]
        return float(y1 + (y1 - y0) * (x - x1) / (x1 - x0))
    return float(np.interp(x, xs, ys))


def _expected_curve_shifted(
    prev_row: pd.Series,
    prev_T: float,
    prev_stock: float,
    now_row: pd.Series,
    now_T: float,
    now_stock: float,
) -> Tuple[List[str], np.ndarray, float]:
    """
    Sticky-strike expected curve, copied from the working API version.

    Build the dotted expected curve for the current minute as:
      1) SHAPE: sticky-strike from previous surface using *current* σ_now for Δ->k
         mapping, evaluate prev surface at (k_now + k_shift).
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
    # filter out 5Δ (P5/C5): drop 95 and 5
    buckets = [n for n in buckets if n not in (95, 5)]
    if not buckets or 50 not in buckets:
        raise ValueError("now row missing buckets/ATM")
    order, labels = _bucket_labels_order(buckets)

    # current ATM (fraction) for Δ->k mapping
    atm_now = float(now_row["vol50"])

    # --- shape values from previous surface at (k_now + k_shift)
    shape_vals: List[float] = []
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
    level_shift_pp = max(
        -BETA_MAX_SHIFT_PP,
        min(BETA_MAX_SHIFT_PP, (-ret_frac) * 100.0 * BETA_VOLPTS_PER_1PCT),
    )
    atm_exp = exp_atm_shape + level_shift_pp / 100.0

    # --- vertical shift of entire curve so ATM equals atm_exp
    shift = atm_exp - exp_atm_shape
    expected = shape + shift

    return labels, expected * 100.0, atm_exp * 100.0


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
        # Same semantics as original: anything except "off" shows expected
        expected_on = (expected_value != "off")

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

        prev_row: Optional[pd.Series] = None
        prev_stock: Optional[float] = None
        prev_T: Optional[float] = None

        # reference slice for live expected (last good historical slice)
        ref_row: Optional[pd.Series] = None
        ref_stock: Optional[float] = None
        ref_T: Optional[float] = None

        # --------- Historical slices from DB ---------
        if times_pt:
            times_sorted = sorted(times_pt)  # e.g. ["06:31", "07:00"]
            df = fetch_data_from_db(trade_date_iso, expiration_iso, times_sorted)

            if df is not None and not df.empty:
                for i, hhmm_pt in enumerate(times_sorted):
                    # Fetch the row for this PT time
                    rows = df[df["snapshot_pt"] == hhmm_pt]
                    if rows.empty:
                        continue
                    row_now = rows.iloc[0]
                    color = COLORWAY[i % len(COLORWAY)]

                    # Actual line (same bucket handling as original)
                    buckets_now = _available_buckets(row_now)
                    buckets_now = [n for n in buckets_now if n not in (95, 5)]
                    if not buckets_now or 50 not in buckets_now:
                        continue
                    order_now, labels_now = _bucket_labels_order(buckets_now)
                    try:
                        y_now = [float(row_now[f"vol{n}"]) * 100.0 for n in order_now]
                    except KeyError:
                        continue

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

                    # stock + T for this slice
                    stock_val = row_now.get("stock_price")
                    stock_now = (
                        float(stock_val)
                        if stock_val is not None and not pd.isna(stock_val)
                        else None
                    )

                    ts_et = pt_minute_to_et(trade_date_iso, hhmm_pt)
                    now_T = _years_to_exp(ts_et, expiration_iso)

                    # expected dotted curve relative to previous slice
                    if (
                        expected_on
                        and prev_row is not None
                        and prev_stock is not None
                        and prev_T is not None
                        and stock_now is not None
                    ):
                        try:
                            labels_exp, y_exp, atm_exp_pct = _expected_curve_shifted(
                                prev_row, prev_T, prev_stock,
                                row_now, now_T, stock_now,
                            )

                            fig.add_trace(
                                go.Scatter(
                                    x=labels_exp,
                                    y=y_exp,
                                    mode="lines",
                                    name=f"Expected (SS) — {hhmm_pt}",
                                    line=dict(width=2, dash="dot", color=color),
                                )
                            )
                            fig.add_trace(
                                go.Scatter(
                                    x=["ATM"],
                                    y=[atm_exp_pct],
                                    mode="markers",
                                    marker=dict(
                                        symbol="triangle-up", size=9, color=color
                                    ),
                                    name="ATM exp (SS)",
                                    showlegend=False,
                                )
                            )
                        except Exception as e:
                            print(f"Could not calculate expected curve for {hhmm_pt}: {e}")

                    # advance previous pointers
                    prev_row, prev_stock, prev_T = row_now, stock_now, now_T

                    # keep a reference slice for live expected if k-grid is valid
                    try:
                        k_now_ref, _ = _k_grid_for_row(row_now, now_T)
                    except Exception:
                        k_now_ref = np.array([])
                    if (
                        k_now_ref.size >= 2
                        and stock_now is not None
                    ):
                        ref_row, ref_stock, ref_T = row_now, stock_now, now_T

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
                            y_live = [
                                float(live_row[f"vol{n}"]) * 100.0 for n in order_live
                            ]
                        except KeyError:
                            y_live = []

                        if y_live:
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

                            stock_live_val = live_row.get("stock_price")
                            stock_live = (
                                float(stock_live_val)
                                if stock_live_val is not None and not pd.isna(stock_live_val)
                                else None
                            )

                            ts_live_val = live_row.get("snapshot_pt")
                            # snapshot_pt is PT hh:mm for live, same as historical
                            if isinstance(ts_live_val, str):
                                hhmm_live = ts_live_val
                            else:
                                # fallback if we stored a time/datetime
                                try:
                                    hhmm_live = pd.to_datetime(ts_live_val).strftime("%H:%M")
                                except Exception:
                                    hhmm_live = "06:31"

                            ts_et_live = pt_minute_to_et(trade_date_iso, hhmm_live)
                            live_T = _years_to_exp(ts_et_live, expiration_iso)

                            if (
                                expected_on
                                and ref_row is not None
                                and ref_stock is not None
                                and ref_T is not None
                                and stock_live is not None
                            ):
                                try:
                                    labels_exp_live, y_exp_live, atm_exp_pct_live = (
                                        _expected_curve_shifted(
                                            ref_row,
                                            ref_T,
                                            ref_stock,
                                            live_row,
                                            live_T,
                                            stock_live,
                                        )
                                    )

                                    fig.add_trace(
                                        go.Scatter(
                                            x=labels_exp_live,
                                            y=y_exp_live,
                                            mode="lines",
                                            name="Expected (SS) — Live",
                                            line=dict(
                                                width=2, dash="dot", color=LIVE_COLOR
                                            ),
                                        )
                                    )
                                    fig.add_trace(
                                        go.Scatter(
                                            x=["ATM"],
                                            y=[atm_exp_pct_live],
                                            mode="markers",
                                            marker=dict(
                                                symbol="triangle-up",
                                                size=9,
                                                color=LIVE_COLOR,
                                            ),
                                            name="ATM exp (SS)",
                                            showlegend=False,
                                        )
                                    )
                                except Exception as e:
                                    print(
                                        f"Could not calculate live expected curve: {e}"
                                    )

        return fig
