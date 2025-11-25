from __future__ import annotations
import datetime as dt
import math
from typing import List, Tuple, Optional
from io import StringIO

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import Input, Output

from packages.shared.utils import fetch_skew_data, is_market_hours
from packages.shared.surface_compare import k_for_abs_delta

# ---- Dash IDs ----
TRADE_DATE_ID = "trade-date"
EXPIRATION_ID = "expiration-date-pick"
SMILE_TIME_INPUT = "smile-time-input"
SKEW_TABLE = "SKEW_TABLE"
CLOCK_ID = "CLOCK"              # still wired but unused; safe to keep
EXPECTED_TOGGLE_ID = "expected-ss-toggle"
LIVE_DATA_STORE_ID = "live-data-store"

# ---- Constants ----
TICKER = "SPX"
EPS_T = 1e-4

# keep skew % deltas sane when expected/previous ≈ 0 pp
MIN_SKEW_DENOM_PP = 0.25

# leverage add-on (must match Smile expected overlay)
BETA_VOLPTS_PER_1PCT = 4.5
BETA_MAX_SHIFT_PP = 6.0


# ----------------- Helpers -----------------
def _skews_from_row(row: pd.Series) -> Tuple[float, float, float]:
    """Return (atm_frac, call_skew_pp, put_skew_pp)."""
    atm = float(pd.to_numeric(row.get("vol50"), errors="coerce"))
    c25 = float(pd.to_numeric(row.get("vol25"), errors="coerce"))
    p25 = float(pd.to_numeric(row.get("vol75"), errors="coerce"))
    return atm, (c25 - atm) * 100.0, (p25 - atm) * 100.0


def _pct_change_frac(curr: Optional[float], base: Optional[float]) -> Optional[float]:
    if base in (None, 0) or curr is None:
        return None
    return (curr - base) / abs(base) * 100.0


def _pct_change_pp(curr_pp: Optional[float], base_pp: Optional[float]) -> Optional[float]:
    if curr_pp is None or base_pp is None:
        return None
    denom = max(abs(base_pp), MIN_SKEW_DENOM_PP)
    return (curr_pp - base_pp) / denom * 100.0


def _years_to_exp(ts_utc: dt.datetime, expiration_iso: str) -> float:
    """
    Time to expiry in years using a UTC timestamp and 16:00 UTC expiry.
    (Close enough to the original ET/midnight version for our purposes.)
    """
    exp_date = dt.date.fromisoformat(expiration_iso)
    exp_dt_utc = dt.datetime.combine(exp_date, dt.time(16, 0), tzinfo=dt.timezone.utc)
    rem = exp_dt_utc - ts_utc
    T = max(0.0, rem.total_seconds() / (365.0 * 24 * 3600))
    return max(T, EPS_T)


def _T_from_row_snapshot(row: pd.Series, expiration_iso: str) -> Optional[float]:
    ts_utc_val = row.get("snap_shot_date")
    if ts_utc_val is None or pd.isna(ts_utc_val):
        return None
    ts_utc = pd.to_datetime(ts_utc_val, utc=True).to_pydatetime()
    return _years_to_exp(ts_utc, expiration_iso)


def _available_buckets(row: pd.Series) -> List[int]:
    buckets: List[int] = []
    for c in row.index:
        if c.startswith("vol") and c[3:].isdigit():
            n = int(c[3:])
            if 1 <= n <= 99:
                buckets.append(n)
    # same ordering logic as original: puts (≥50) then calls (<50), both descending,
    # with no duplicates
    puts = sorted([n for n in buckets if n >= 50], reverse=True)
    calls = sorted([n for n in buckets if n < 50], reverse=True)
    out: List[int] = []
    seen = set()
    for n in puts + calls:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _abs_delta_is_put(bucket: int) -> Tuple[float, bool]:
    if bucket == 50:
        return 0.50, False
    if bucket > 50:
        return (100 - bucket) / 100.0, True
    return bucket / 100.0, False


def _prev_smile_interp(prev_row: pd.Series, T_prev: float):
    """
    Build previous surface’s (k_prev, sigma_prev) grid using its ATM for Δ→k.
    This matches the original skew logic and the Smile expected overlay.
    """
    if "vol50" not in prev_row:
        raise ValueError("prev row missing ATM")
    atm_prev = float(prev_row["vol50"])
    buckets_prev = _available_buckets(prev_row)
    if len(buckets_prev) < 4:
        raise ValueError("prev row has too few buckets")

    k_prev: List[float] = []
    s_prev: List[float] = []
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
    mask = np.concatenate(([True], np.diff(k_np) > 1e-12))
    k_np, s_np = k_np[mask], s_np[mask]
    if k_np.size < 3:
        raise ValueError("prev k-grid degenerate")
    return k_np, s_np


def _interp_linear_extrap(kq: float, k_grid: np.ndarray, s_grid: np.ndarray) -> float:
    if kq <= k_grid[0]:
        x0, x1, y0, y1 = k_grid[0], k_grid[1], s_grid[0], s_grid[1]
        return float(y0 + (y1 - y0) * (kq - x0) / (x1 - x0))
    if kq >= k_grid[-1]:
        x0, x1, y0, y1 = k_grid[-2], k_grid[-1], s_grid[-2], s_grid[-1]
        return float(y1 + (y1 - y0) * (kq - x1) / (x1 - x0))
    return float(np.interp(kq, k_grid, s_grid))


# ----------------- Main Callback -----------------
def register_callbacks(app):
    @app.callback(
        Output(SKEW_TABLE, "figure"),
        [
            Input(TRADE_DATE_ID, "date"),
            Input(EXPIRATION_ID, "date"),
            Input(SMILE_TIME_INPUT, "value"),
            Input(EXPECTED_TOGGLE_ID, "value"),
            Input(LIVE_DATA_STORE_ID, "data"),
        ],
    )
    def render_skew_table(trade_date_iso, expiration_iso, times_pt, expected_value, live_data_json):
        # Same semantics as original: anything except "off" means we use Expected SS.
        expected_on = (expected_value != "off")

        base_cols = [
            "Time (PT)",
            "Stock",
            "Δ Stock %",
            "ATM IV %",
            "Call Skew",
            "Put Skew",
            "Δ ATM IV %",
            "Δ Call Skew %",
            "Δ Put Skew %",
        ]
        exp_cols = ["ATM exp (SS) %", "ATM residual (bp)"]

        if not trade_date_iso or not expiration_iso:
            cols = (
                base_cols
                if not expected_on
                else ["Time (PT)", "Stock", "Δ Stock %", "ATM IV %"]
                + exp_cols
                + ["Call Skew", "Put Skew", "Δ ATM IV %", "Δ Call Skew %", "Δ Put Skew %"]
            )
            fig = go.Figure(
                data=[
                    go.Table(
                        header=dict(
                            values=cols,
                            font=dict(color="white"),
                            align="left",
                        ),
                        cells=dict(
                            values=[[] for _ in cols],
                            fill_color="black",
                            font=dict(color="white"),
                            align="left",
                        ),
                    )
                ]
            )
            fig.update_layout(
                template="plotly_dark",
                title="Skew — select Trade Date & Expiration",
            )
            return fig

        rows: List[dict] = []

        prev_row: Optional[pd.Series] = None
        prev_stock: Optional[float] = None
        prev_T: Optional[float] = None

        # OFF-mode “previous actual” references
        prev_stock_actual: Optional[float] = None
        prev_atm_actual: Optional[float] = None
        prev_call_skew_pp_actual: Optional[float] = None
        prev_put_skew_pp_actual: Optional[float] = None

        # --------- Historical slices from DB ---------
        if times_pt:
            df_data = fetch_skew_data(trade_date_iso, expiration_iso, sorted(times_pt))
            if df_data is not None and not df_data.empty:
                # Ensure chronological order by snapshot_pt
                df_data = df_data.sort_values("snapshot_pt")
                for snapshot, group in df_data.groupby("snapshot_pt"):
                    row_now = group.iloc[0]
                    hhmm_pt = (
                        snapshot.strftime("%H:%M")
                        if hasattr(snapshot, "strftime")
                        else str(snapshot)
                    )

                    stock_val = row_now.get("stock_price")
                    stock_now = (
                        float(stock_val)
                        if stock_val is not None and not pd.isna(stock_val)
                        else None
                    )
                    atm_now, call_skew_pp_now, put_skew_pp_now = _skews_from_row(row_now)
                    T_now = _T_from_row_snapshot(row_now, expiration_iso)

                    # Defaults: OFF-mode deltas vs previous actual slice
                    d_stock_pct = _pct_change_frac(stock_now, prev_stock_actual)
                    d_atm_pct = _pct_change_frac(atm_now, prev_atm_actual)
                    d_call_pct = _pct_change_pp(
                        call_skew_pp_now, prev_call_skew_pp_actual
                    )
                    d_put_pct = _pct_change_pp(
                        put_skew_pp_now, prev_put_skew_pp_actual
                    )
                    atm_exp_pct: Optional[float] = None
                    atm_res_bp: Optional[int] = None

                    # Expected SS mode: compare vs *expected* curve from previous surface
                    if (
                        expected_on
                        and prev_row is not None
                        and prev_T is not None
                        and prev_stock is not None
                        and stock_now is not None
                        and T_now is not None
                    ):
                        try:
                            # previous surface and SS k-shift
                            k_prev, s_prev = _prev_smile_interp(prev_row, prev_T)
                            k_shift = (
                                math.log(stock_now / prev_stock)
                                if (prev_stock and stock_now)
                                else 0.0
                            )

                            # --- 1) Expected ATM anchor (matches Smile ATM marker) ---
                            exp_atm_shape = _interp_linear_extrap(
                                k_shift, k_prev, s_prev
                            )  # fraction
                            ret_frac = (stock_now - prev_stock) / prev_stock
                            level_shift_pp = max(
                                -BETA_MAX_SHIFT_PP,
                                min(
                                    BETA_MAX_SHIFT_PP,
                                    (-ret_frac) * 100.0 * BETA_VOLPTS_PER_1PCT,
                                ),
                            )
                            atm_exp = exp_atm_shape + (level_shift_pp / 100.0)
                            atm_exp_pct = round(atm_exp * 100.0, 2)
                            atm_res_bp = int(
                                round((atm_now - atm_exp) * 10000.0)
                            )

                            # --- 2) Expected curve SHAPE (same as Smile dotted line) ---
                            k_c25_now = k_for_abs_delta(
                                0.25, is_put=False, sigma=atm_now, T=T_now
                            )
                            k_p25_now = k_for_abs_delta(
                                0.25, is_put=True, sigma=atm_now, T=T_now
                            )
                            exp_c25_shape = _interp_linear_extrap(
                                k_c25_now + k_shift, k_prev, s_prev
                            )
                            exp_p25_shape = _interp_linear_extrap(
                                k_p25_now + k_shift, k_prev, s_prev
                            )

                            # --- 3) Vertical shift so curve ATM == atm_exp ---
                            shift_frac = atm_exp - exp_atm_shape
                            exp_c25 = exp_c25_shape + shift_frac
                            exp_p25 = exp_p25_shape + shift_frac

                            exp_call_skew_pp = (exp_c25 - atm_exp) * 100.0
                            exp_put_skew_pp = (exp_p25 - atm_exp) * 100.0

                            # --- 4) Table % deltas vs shifted expected curve ---
                            d_call_pct = _pct_change_pp(
                                call_skew_pp_now, exp_call_skew_pp
                            )
                            d_put_pct = _pct_change_pp(
                                put_skew_pp_now, exp_put_skew_pp
                            )
                            d_atm_pct = _pct_change_frac(atm_now, atm_exp)
                        except Exception as e:
                            # graceful fallback to previous actual slice
                            print(
                                f"Could not calculate expected skew for {hhmm_pt}: {e}"
                            )

                    rows.append(
                        {
                            "Time (PT)": hhmm_pt,
                            "Stock": None
                            if stock_now is None
                            else round(stock_now, 2),
                            "Δ Stock %": None
                            if d_stock_pct is None
                            else round(d_stock_pct, 2),
                            "ATM IV %": round(atm_now * 100.0, 2),
                            "ATM exp (SS) %": atm_exp_pct,
                            "ATM residual (bp)": atm_res_bp,
                            "Call Skew": round(call_skew_pp_now, 2),
                            "Put Skew": round(put_skew_pp_now, 2),
                            "Δ ATM IV %": None
                            if d_atm_pct is None
                            else round(d_atm_pct, 2),
                            "Δ Call Skew %": None
                            if d_call_pct is None
                            else round(d_call_pct, 2),
                            "Δ Put Skew %": None
                            if d_put_pct is None
                            else round(d_put_pct, 2),
                        }
                    )

                    # advance pointers
                    prev_row, prev_stock, prev_T = row_now, stock_now, T_now
                    prev_stock_actual, prev_atm_actual = stock_now, atm_now
                    (
                        prev_call_skew_pp_actual,
                        prev_put_skew_pp_actual,
                    ) = (call_skew_pp_now, put_skew_pp_now)

        # --------- Live row from live ORATS data ---------
        if live_data_json and is_market_hours():
            df_live = pd.read_json(StringIO(live_data_json), orient="split")
            live_row_df = df_live[df_live["expir_date"] == expiration_iso]
            if live_row_df is not None and not live_row_df.empty:
                live_row = live_row_df.iloc[0]
                stock_live_val = live_row.get("stock_price")
                stock_live = (
                    float(stock_live_val)
                    if stock_live_val is not None and not pd.isna(stock_live_val)
                    else None
                )
                atm_live, call_skew_pp_live, put_skew_pp_live = _skews_from_row(
                    live_row
                )
                T_live = _T_from_row_snapshot(live_row, expiration_iso)

                # defaults: OFF-mode deltas vs previous actual slice
                d_stock_pct = _pct_change_frac(stock_live, prev_stock_actual)
                d_atm_pct = _pct_change_frac(atm_live, prev_atm_actual)
                d_call_pct = _pct_change_pp(
                    call_skew_pp_live, prev_call_skew_pp_actual
                )
                d_put_pct = _pct_change_pp(
                    put_skew_pp_live, prev_put_skew_pp_actual
                )
                atm_exp_pct: Optional[float] = None
                atm_res_bp: Optional[int] = None

                if (
                    expected_on
                    and prev_row is not None
                    and prev_T is not None
                    and prev_stock is not None
                    and stock_live is not None
                    and T_live is not None
                ):
                    try:
                        k_prev, s_prev = _prev_smile_interp(prev_row, prev_T)
                        k_shift = (
                            math.log(stock_live / prev_stock)
                            if (prev_stock and stock_live)
                            else 0.0
                        )

                        exp_atm_shape = _interp_linear_extrap(
                            k_shift, k_prev, s_prev
                        )
                        ret_frac = (stock_live - prev_stock) / prev_stock
                        level_shift_pp = max(
                            -BETA_MAX_SHIFT_PP,
                            min(
                                BETA_MAX_SHIFT_PP,
                                (-ret_frac) * 100.0 * BETA_VOLPTS_PER_1PCT,
                            ),
                        )
                        atm_exp = exp_atm_shape + (level_shift_pp / 100.0)
                        atm_exp_pct = round(atm_exp * 100.0, 2)
                        atm_res_bp = int(
                            round((atm_live - atm_exp) * 10000.0)
                        )

                        k_c25_live = k_for_abs_delta(
                            0.25, is_put=False, sigma=atm_live, T=T_live
                        )
                        k_p25_live = k_for_abs_delta(
                            0.25, is_put=True, sigma=atm_live, T=T_live
                        )
                        exp_c25_shape = _interp_linear_extrap(
                            k_c25_live + k_shift, k_prev, s_prev
                        )
                        exp_p25_shape = _interp_linear_extrap(
                            k_p25_live + k_shift, k_prev, s_prev
                        )
                        shift_frac = atm_exp - exp_atm_shape
                        exp_c25 = exp_c25_shape + shift_frac
                        exp_p25 = exp_p25_shape + shift_frac
                        exp_call_skew_pp = (exp_c25 - atm_exp) * 100.0
                        exp_put_skew_pp = (exp_p25 - atm_exp) * 100.0

                        d_call_pct = _pct_change_pp(
                            call_skew_pp_live, exp_call_skew_pp
                        )
                        d_put_pct = _pct_change_pp(
                            put_skew_pp_live, exp_put_skew_pp
                        )
                        d_atm_pct = _pct_change_frac(atm_live, atm_exp)
                    except Exception as e:
                        print(f"Could not calculate expected skew for Live: {e}")

                rows.append(
                    {
                        "Time (PT)": "Live",
                        "Stock": None
                        if stock_live is None
                        else round(stock_live, 2),
                        "Δ Stock %": None
                        if d_stock_pct is None
                        else round(d_stock_pct, 2),
                        "ATM IV %": round(atm_live * 100.0, 2),
                        "ATM exp (SS) %": atm_exp_pct,
                        "ATM residual (bp)": atm_res_bp,
                        "Call Skew": round(call_skew_pp_live, 2),
                        "Put Skew": round(put_skew_pp_live, 2),
                        "Δ ATM IV %": None
                        if d_atm_pct is None
                        else round(d_atm_pct, 2),
                        "Δ Call Skew %": None
                        if d_call_pct is None
                        else round(d_call_pct, 2),
                        "Δ Put Skew %": None
                        if d_put_pct is None
                        else round(d_put_pct, 2),
                    }
                )

        if not rows:
            cols = (
                base_cols
                if not expected_on
                else ["Time (PT)", "Stock", "Δ Stock %", "ATM IV %"]
                + exp_cols
                + ["Call Skew", "Put Skew", "Δ ATM IV %", "Δ Call Skew %", "Δ Put Skew %"]
            )
            fig = go.Figure(
                data=[
                    go.Table(
                        header=dict(
                            values=cols,
                            font=dict(color="white"),
                            align="left",
                        ),
                        cells=dict(
                            values=[[] for _ in cols],
                            fill_color="black",
                            font=dict(color="white"),
                            align="left",
                        ),
                    )
                ]
            )
            fig.update_layout(
                template="plotly_dark",
                title=f"Skew — no data for {trade_date_iso} / {expiration_iso}",
            )
            return fig

        df = pd.DataFrame(rows)
        ordered_cols = (
            [
                "Time (PT)",
                "Stock",
                "Δ Stock %",
                "ATM IV %",
                "ATM exp (SS) %",
                "ATM residual (bp)",
                "Call Skew",
                "Put Skew",
                "Δ ATM IV %",
                "Δ Call Skew %",
                "Δ Put Skew %",
            ]
            if expected_on
            else [
                "Time (PT)",
                "Stock",
                "Δ Stock %",
                "ATM IV %",
                "Call Skew",
                "Put Skew",
                "Δ ATM IV %",
                "Δ Call Skew %",
                "Δ Put Skew %",
            ]
        )
        df = df[ordered_cols]

        color_cols = {
            "Δ Stock %",
            "Δ ATM IV %",
            "Δ Call Skew %",
            "Δ Put Skew %",
            "ATM residual (bp)",
        }
        cell_colors: List[List[str]] = []
        for col in df.columns:
            if col in color_cols:
                colors = [
                    "green"
                    if (v is not None and v > 0)
                    else "red"
                    if (v is not None and v < 0)
                    else "black"
                    for v in df[col]
                ]
            else:
                colors = ["black"] * len(df)
            cell_colors.append(colors)

        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=list(df.columns),
                        font=dict(color="white"),
                        align="left",
                    ),
                    cells=dict(
                        values=[df[c] for c in df.columns],
                        fill_color=cell_colors,
                        align="left",
                        font=dict(color="white"),
                    ),
                )
            ]
        )
        fig.update_layout(
            template="plotly_dark",
            title=f"Skew (50Δ → 25Δ) — {trade_date_iso}   Exp: {expiration_iso}",
            margin=dict(l=0, r=0, t=36, b=0),
        )
        return fig
