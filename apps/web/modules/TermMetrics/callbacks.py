from dash.dependencies import Input, Output
import pandas as pd
import plotly.graph_objs as go
from sqlalchemy import create_engine, text
from typing import List
import numpy as np
import os

# ---- App IDs ----
TRADE_DATE_ID = "trade-date"
SMILE_TIME_INPUT = "smile-time-input"

# ---- Database Configuration ----
DATABASE_URL = os.getenv("DATABASE_URL")  # or "CURVE_DB_URL" if you prefer a custom name

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL environment variable is not set")

DB_TABLE_NAME = "orats_monies_minute"
DB_ENGINE = create_engine(DATABASE_URL)


def _pct_change(curr: float | None, prev: float | None) -> float | None:
    """
    Percent change vs previous value, similar to Skew table:
    (curr - prev) / abs(prev) * 100, guarding for None / zero.
    """
    if prev in (None, 0) or curr is None or pd.isna(prev) or pd.isna(curr):
        return None
    return (curr - prev) / abs(prev) * 100.0


def fetch_term_metrics_data(trade_date_iso: str, times_pt: List[str]) -> pd.DataFrame:
    if not trade_date_iso or not times_pt:
        return pd.DataFrame()

    # Ensure times_pt are sorted for consistent query results
    sorted_times_pt = sorted(times_pt)
    time_filters = [f"'{trade_date_iso} {hhmm}:00'" for hhmm in sorted_times_pt]

    query = text(f"""
        SELECT snapshot_pt, trade_date, expir_date, vol50
        FROM "{DB_TABLE_NAME}"
        WHERE trade_date = :trade_date AND snapshot_pt IN ({','.join(time_filters)})
        ORDER BY snapshot_pt, expir_date;
    """)

    try:
        with DB_ENGINE.connect() as connection:
            df = pd.read_sql(query, connection, params={"trade_date": trade_date_iso})
            if df.empty:
                return pd.DataFrame()

            df["expir_date"] = pd.to_datetime(df["expir_date"])
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            df["dte"] = (df["expir_date"] - df["trade_date"]).dt.days

            def get_closest_iv(group, dte_target):
                if group.empty:
                    return np.nan
                # Find the row with the DTE closest to the target
                closest_row = group.iloc[(group["dte"] - dte_target).abs().argmin()]
                return closest_row["vol50"]

            metrics_list = []
            for snapshot, group in df.groupby("snapshot_pt"):
                iv_3d = get_closest_iv(group, 3)
                iv_30d = get_closest_iv(group, 30)
                iv_90d = get_closest_iv(group, 90)

                # Calculate metrics, handling potential NaN values
                front_back_spread = (
                    iv_3d - iv_30d
                    if not (pd.isna(iv_3d) or pd.isna(iv_30d))
                    else np.nan
                )
                front_back_ratio = (
                    iv_3d / iv_30d
                    if not (pd.isna(iv_3d) or pd.isna(iv_30d) or iv_30d == 0)
                    else np.nan
                )
                slope_30_90 = (
                    iv_30d - iv_90d
                    if not (pd.isna(iv_30d) or pd.isna(iv_90d))
                    else np.nan
                )

                metrics_list.append(
                    {
                        "snapshot_pt": snapshot,
                        "front_back_spread": front_back_spread,
                        "front_back_ratio": front_back_ratio,
                        "slope_30_90": slope_30_90,
                    }
                )

            metrics_df = pd.DataFrame(metrics_list)
            # Sort by snapshot_pt to ensure consistent ordering
            metrics_df = metrics_df.sort_values("snapshot_pt").reset_index(drop=True)
            return metrics_df

    except Exception as e:
        print(f"Term Metrics DB query failed: {e}")
        return pd.DataFrame()


def register_callbacks(app):
    @app.callback(
        Output("term-metrics-table", "figure"),
        [
            Input(TRADE_DATE_ID, "date"),
            Input(SMILE_TIME_INPUT, "value"),
        ],
    )
    def update_term_metrics(trade_date_iso, times_pt):
        # Column order: base metric, then its Δ%, for each metric
        cols = [
            "Time (PT)",
            "3D-30D Spread",
            "Δ 3D-30D Spread %",
            "3D/30D Ratio",
            "Δ 3D/30D Ratio %",
            "30D-90D Slope",
            "Δ 30D-90D Slope %",
        ]
        delta_cols = {
            "Δ 3D-30D Spread %",
            "Δ 3D/30D Ratio %",
            "Δ 30D-90D Slope %",
        }

        # No inputs yet: empty dark table
        if not trade_date_iso or not times_pt:
            empty_values = [[] for _ in cols]
            fig = go.Figure(
                data=[
                    go.Table(
                        header=dict(values=cols),
                        cells=dict(
                            values=empty_values,
                            align="left",
                            font=dict(color="white"),
                            fill_color="black",
                        ),
                    )
                ]
            )
            fig.update_layout(
                template="plotly_dark",
                title="Term Metrics",
                margin=dict(l=0, r=0, t=36, b=0),
            )
            return fig

        df = fetch_term_metrics_data(trade_date_iso, sorted(times_pt))
        if df.empty:
            empty_values = [[] for _ in cols]
            fig = go.Figure(
                data=[
                    go.Table(
                        header=dict(values=cols),
                        cells=dict(
                            values=empty_values,
                            align="left",
                            font=dict(color="white"),
                            fill_color="black",
                        ),
                    )
                ]
            )
            fig.update_layout(
                template="plotly_dark",
                title=f"Term Metrics — No data for {trade_date_iso}",
                margin=dict(l=0, r=0, t=36, b=0),
            )
            return fig

        # Build display table with % change vs previous row
        table_rows = []
        prev_spread = prev_ratio = prev_slope = None

        for _, row in df.iterrows():
            spread = (
                float(row["front_back_spread"])
                if pd.notna(row["front_back_spread"])
                else None
            )
            ratio = (
                float(row["front_back_ratio"])
                if pd.notna(row["front_back_ratio"])
                else None
            )
            slope = (
                float(row["slope_30_90"])
                if pd.notna(row["slope_30_90"])
                else None
            )

            d_spread = _pct_change(spread, prev_spread)
            d_ratio = _pct_change(ratio, prev_ratio)
            d_slope = _pct_change(slope, prev_slope)

            table_rows.append(
                {
                    "Time (PT)": pd.to_datetime(row["snapshot_pt"]).strftime("%H:%M"),
                    "3D-30D Spread": round(spread, 4) if spread is not None else None,
                    "Δ 3D-30D Spread %": (
                        round(d_spread, 2) if d_spread is not None else None
                    ),
                    "3D/30D Ratio": round(ratio, 4) if ratio is not None else None,
                    "Δ 3D/30D Ratio %": (
                        round(d_ratio, 2) if d_ratio is not None else None
                    ),
                    "30D-90D Slope": round(slope, 4) if slope is not None else None,
                    "Δ 30D-90D Slope %": (
                        round(d_slope, 2) if d_slope is not None else None
                    ),
                }
            )

            prev_spread, prev_ratio, prev_slope = spread, ratio, slope

        df_table = pd.DataFrame(table_rows, columns=cols)

        # Color matrix: green for positive deltas, red for negative, black otherwise
        cell_colors = [
            [
                (
                    "green"
                    if (col in delta_cols and v is not None and v > 0)
                    else "red"
                    if (col in delta_cols and v is not None and v < 0)
                    else "black"
                )
                for v in df_table[col]
            ]
            for col in df_table.columns
        ]

        fig = go.Figure(
            data=[
                go.Table(
                    header=dict(values=cols),
                    cells=dict(
                        values=[df_table[c] for c in cols],
                        align="left",
                        font=dict(color="white"),
                        fill_color=cell_colors,
                    ),
                )
            ]
        )
        fig.update_layout(
            template="plotly_dark",
            title="Term Metrics",
            margin=dict(l=0, r=0, t=36, b=0),
        )
        return fig
