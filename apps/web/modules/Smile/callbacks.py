# apps/web/modules/Smile/callbacks.py
import datetime as dt
import math
from typing import List, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import Input, Output

# Dash IDs
TRADE_DATE_ID = "trade-date"
EXPIRATION_ID = "expiration-date-pick"
SMILE_TIME_INPUT = "smile-time-input"
SMILE_GRAPH = "SMILE_GRAPH"
CLOCK_ID = "CLOCK"

from packages.shared.options_orats import fetch_one_minute_monies, pt_minute_to_et
from packages.shared.surface_compare import k_for_abs_delta

TICKER = "SPX"
EPS_T = 1e-4  # ~52 min in years

COLORWAY = [
    "#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A",
    "#19D3F3", "#FF6692", "#B6E880", "#FF97FF", "#FECB52",
]

def _years_to_exp(ts_et: dt.datetime, expiration_iso: str) -> float:
    exp = dt.date.fromisoformat(expiration_iso)
    rem = dt.datetime.combine(exp, dt.time(0, 0)) - ts_et.replace(tzinfo=None)
    T = max(0.0, rem.days / 365.0 + rem.seconds / (365.0 * 24 * 3600))
    return max(T, EPS_T)

def _extract_row(df: pd.DataFrame) -> pd.Series | None:
    return None if df is None or df.empty else df.iloc[0]

def _stock(df: pd.DataFrame) -> float | None:
    if df is None or df.empty or "stockPrice" not in df.columns:
        return None
    s = pd.to_numeric(df["stockPrice"], errors="coerce")
    return float(s.median()) if s.notna().any() else None

def _available_buckets(row: pd.Series) -> List[int]:
    """P90..P50 then C45..C10 (call side descending)."""
    buckets = []
    for c in row.index:
        if c.startswith("vol") and c[3:].isdigit():
            n = int(c[3:])
            if 1 <= n <= 99:
                buckets.append(n)
    puts  = sorted([n for n in buckets if n >= 50], reverse=True)
    calls = sorted([n for n in buckets if n < 50],  reverse=True)
    out, seen = [], set()
    for n in puts + calls:
        if n not in seen:
            seen.add(n); out.append(n)
    return out

def _label(n: int) -> str:
    if n == 50: return "ATM"
    if n > 50:  return f"P{100 - n}"
    return f"C{n}"

def _abs_delta_is_put(n: int) -> Tuple[float, bool]:
    if n == 50: return 0.50, False
    if n > 50:  return (100 - n) / 100.0, True
    return n / 100.0, False

def _prev_smile_interp(prev_row: pd.Series, T_prev: float):
    """Return (k_prev asc, sigma_prev) from previous minute using σ=ATM(prev)."""
    if "vol50" not in prev_row:
        raise ValueError("prev row missing ATM")
    atm_prev = float(prev_row["vol50"])
    buckets_prev = _available_buckets(prev_row)
    if len(buckets_prev) < 4:
        raise ValueError("prev row has too few buckets")
    k_prev, s_prev = [], []
    for n in buckets_prev:
        if n == 50:
            k = 0.0
        else:
            p, is_put = _abs_delta_is_put(n)
            k = k_for_abs_delta(p, is_put=is_put, sigma=atm_prev, T=T_prev)
        k_prev.append(k)
        s_prev.append(float(prev_row[f"vol{n}"]))
    k_np = np.array(k_prev, float)
    s_np = np.array(s_prev, float)
    mask = np.concatenate(([True], np.diff(k_np) > 1e-12))  # strictly increasing
    k_np, s_np = k_np[mask], s_np[mask]
    if k_np.size < 3:
        raise ValueError("prev k-grid degenerate")
    return k_np, s_np

def _interp_with_linear_extrap(kq: float, k_grid: np.ndarray, s_grid: np.ndarray) -> float:
    if kq <= k_grid[0]:
        x0, x1, y0, y1 = k_grid[0], k_grid[1], s_grid[0], s_grid[1]
        return float(y0 + (y1 - y0) * (kq - x0) / (x1 - x0))
    if kq >= k_grid[-1]:
        x0, x1, y0, y1 = k_grid[-2], k_grid[-1], s_grid[-2], s_grid[-1]
        return float(y1 + (y1 - y0) * (kq - x1) / (x1 - x0))
    return float(np.interp(kq, k_grid, s_grid))

def register_callbacks(app):
    @app.callback(
        Output(SMILE_GRAPH, "figure"),
        Input(TRADE_DATE_ID, "date"),
        Input(EXPIRATION_ID, "date"),
        Input(SMILE_TIME_INPUT, "value"),
        Input(CLOCK_ID, "n_intervals"),
    )
    def render_smile(trade_date_iso, expiration_iso, times_pt, _tick):
        fig = go.Figure().update_layout(
            template="plotly_dark",
            title=f"ORATS Smile Grid — {trade_date_iso or ''} (Exp: {expiration_iso or ''})",
            xaxis_title="Bucket (P10 … ATM … C10)",
            yaxis_title="IV (%)",
            margin=dict(l=40, r=160, t=60, b=40),  # room for right legend
            legend=dict(
                orientation="v",
                y=1, yanchor="top",
                x=1.02, xanchor="left",
                traceorder="normal",
            ),
        )
        if not trade_date_iso or not expiration_iso:
            return fig

        # normalize times
        if not times_pt:
            times_pt = ["06:31"]
        if not isinstance(times_pt, list):
            times_pt = [str(times_pt)]
        times_pt = sorted({str(t) for t in times_pt})

        expected_trace_mask: List[bool] = []

        for idx, hhmm in enumerate(times_pt):
            color = COLORWAY[idx % len(COLORWAY)]
            ts_et = pt_minute_to_et(trade_date_iso, hhmm)

            df_now = fetch_one_minute_monies(ts_et, TICKER, expiration_iso)
            row_now = _extract_row(df_now)
            if row_now is None or "vol50" not in row_now:
                continue

            # actual curve
            buckets = _available_buckets(row_now)
            xs = [_label(n) for n in buckets]
            ys = [float(row_now[f"vol{n}"]) * 100.0 for n in buckets]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines+markers",
                name=f"{hhmm} PT", legendgroup=f"{hhmm}",
                hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
                line=dict(width=2, color=color),
                marker=dict(color=color),
            ))
            expected_trace_mask.append(False)

            # skip overlay for first slice
            if idx == 0:
                continue

            # previous SELECTED slice as reference; fallback ±1..3 mins
            prev_hhmm = times_pt[idx - 1]
            ts_prev_sel = pt_minute_to_et(trade_date_iso, prev_hhmm)
            df_prev = fetch_one_minute_monies(ts_prev_sel, TICKER, expiration_iso)
            prev_row = _extract_row(df_prev)
            used_prev = ts_prev_sel
            prev_stock = _stock(df_prev)
            if prev_row is None or "vol50" not in prev_row:
                prev_row = None; prev_stock = None; used_prev = None
                for m in (1, 2, 3):
                    ts_prev = ts_et - dt.timedelta(minutes=m)
                    df_p = fetch_one_minute_monies(ts_prev, TICKER, expiration_iso)
                    r = _extract_row(df_p)
                    if r is not None and "vol50" in r:
                        prev_row = r; prev_stock = _stock(df_p); used_prev = ts_prev; break
                if prev_row is None:
                    print(f"[SMILE] overlay skipped for {hhmm}: no data for prior slice {prev_hhmm}")
                    continue

            try:
                T_now  = _years_to_exp(ts_et,      expiration_iso)
                T_prev = _years_to_exp(used_prev,  expiration_iso)
                k_prev, s_prev = _prev_smile_interp(prev_row, T_prev)

                stock_prev = prev_stock or float(prev_row.get("stockPrice", 0.0))
                stock_now  = _stock(df_now)
                k_shift = math.log(stock_now / stock_prev) if (stock_prev and stock_now) else 0.0

                atm_now = float(row_now["vol50"])

                # expected curve
                exp_curve = []
                for n in buckets:
                    if n == 50:
                        k_now = 0.0
                    else:
                        p, is_put = _abs_delta_is_put(n)
                        k_now = k_for_abs_delta(p, is_put=is_put, sigma=atm_now, T=T_now)
                    exp_curve.append(_interp_with_linear_extrap(k_now + k_shift, k_prev, s_prev) * 100.0)

                fig.add_trace(go.Scatter(
                    x=xs, y=exp_curve, mode="lines+markers",
                    name=f"Expected (SS) — {hhmm}", legendgroup=f"{hhmm}",
                    line=dict(dash="dot", width=2, color=color),
                    marker=dict(color=color),
                    hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
                ))
                expected_trace_mask.append(True)

                # ATM expected dot — keep on chart, hide in legend
                atm_exp = _interp_with_linear_extrap(k_shift, k_prev, s_prev) * 100.0
                fig.add_trace(go.Scatter(
                    x=["ATM"], y=[atm_exp],
                    mode="markers",
                    name=f"ATM exp (SS) — {hhmm}",
                    legendgroup=f"{hhmm}",
                    marker=dict(symbol="triangle-up", size=10, color=color),
                    showlegend=False,  # << hide from legend
                ))
                expected_trace_mask.append(True)
            except Exception as e:
                print(f"[SMILE] overlay skipped for {hhmm}: {e}")

        # Expected ON/OFF toggle
        total_traces = len(fig.data)
        visible_all = [True] * total_traces
        visible_no_expected = []
        exp_iter = iter(expected_trace_mask)
        for _tr in fig.data:
            try:
                is_expected = next(exp_iter)
            except StopIteration:
                is_expected = False
            visible_no_expected.append(False if is_expected else True)

        fig.update_layout(
            updatemenus=[dict(
                type="buttons",
                direction="right",
                x=1.02, xanchor="left",
                y=1.08, yanchor="top",
                pad=dict(r=4, t=2, b=2),
                buttons=[
                    dict(label="Expected: ON",  method="update", args=[{"visible": visible_all}]),
                    dict(label="Expected: OFF", method="update", args=[{"visible": visible_no_expected}]),
                ],
                showactive=True,
            )]
        )

        return fig
