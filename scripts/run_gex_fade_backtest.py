#!/usr/bin/env python
"""
Run the GEX fade backtest over es_minute_features and print basic stats.

Usage:
    python scripts/run_gex_fade_backtest.py

You can later extend this to accept CLI arguments or date ranges.
"""

import os
import datetime as dt

import pandas as pd
from sqlalchemy import create_engine, text

# Make repo root importable (same trick as app.py)
import sys
from pathlib import Path as _P

REPO_ROOT = _P(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load .env
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except Exception:
    pass

from packages.backtests.gex_fade import run_gex_fade_backtest, GexFadeParams


def main():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set in environment or .env")

    engine = create_engine(db_url)

    # Pull the columns we need from es_minute_features
    query = text("""
        SELECT
            trade_date,
            ts_utc,
            ts_pt,
            bar_index,
            is_rth,
            close,
            high,
            low,
            net_gex,
            gex_wall_above,
            gex_wall_above_gex,
            dist_to_wall_above_pts,
            put_skew_pp_primary,
            smile_dte_primary
        FROM es_minute_features
        ORDER BY trade_date, ts_utc
    """)

    print("[run_gex_fade_backtest] Loading features from es_minute_features...")
    df = pd.read_sql_query(query, engine)
    print(f"[run_gex_fade_backtest] Loaded {len(df)} rows")

    # Optional: restrict date range while testing
    # start_date = dt.date(2025, 11, 24)
    # end_date = dt.date(2025, 12, 10)
    # params = GexFadeParams(...)
    # trades_df, summary = run_gex_fade_backtest(df, params, start_date, end_date)

    # Looser params for exploration so we see more trades
    params = GexFadeParams(
        entry_proximity_max=1.5,  # tests within ~1.5 pts of wall
        touch_proximity_max=0.5,  # "touch" within ~1 pt
        min_test_gap_bars=15,
        put_skew_increase_min=0.3,  # your ~68% skew change passes
        min_baseline_skew=0.2,
        gex_net_min=0.0,
        gex_wall_min=1e11,
        bar_index_min=30,
        bar_index_max=350,
        stop_loss_points=2.0,
        r_mult=4.0,
        max_bars_in_trade=60,
        max_trades_per_day=8,
        rth_skew_min_offset_bars=15,
        use_lower_gex_target=True,
        lower_gex_min_abs=1e11,
        entry_min_range_frac=0.7,  # 🔹 only short near top 30% of day's range
    )

    trades_df, summary = run_gex_fade_backtest(df, params)

    print("\n=== Backtest Summary ===")
    for k, v in summary.items():
        if k == "params":
            continue
        print(f"{k}: {v}")

    print("\n=== Params Used ===")
    for k, v in summary.get("params", {}).items():
        print(f"{k}: {v}")

    print("\n=== All Trades ===")
    if not trades_df.empty:
        cols = [
            "trade_date",
            "entry_ts",
            "entry_ts_pt",
            "exit_ts",
            "exit_ts_pt",
            "entry_price",
            "exit_price",
            "pnl_points",
            "exit_reason",
            "wall_key",
            "gex_wall_above",
            "gex_wall_above_gex",
            "put_skew_base",
            "put_skew_entry",
            "put_skew_change_pct",
            "put_skew_base_ts",
            "put_skew_entry_ts",
            "put_skew_base_ts_pt",
            "put_skew_entry_ts_pt",
            "smile_dte_primary",
        ]
        cols = [c for c in cols if c in trades_df.columns]
        print(trades_df[cols].to_string(index=False))

        cols = [c for c in cols if c in trades_df.columns]
        print(trades_df[cols].to_string(index=False))
    else:
        print("No trades generated.")


if __name__ == "__main__":
    main()
