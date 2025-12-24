#!/usr/bin/env python
"""
Run the GEX bounce (long support) backtest over es_minute_features and print basic stats.

Usage:
    python scripts/run_gex_bounce_backtest.py
"""

import os
import pandas as pd
from sqlalchemy import create_engine, text

# Make repo root importable (same trick as app.py)
import sys
from pathlib import Path as _P

from sqlalchemy import create_engine, text
import pandas as pd


REPO_ROOT = _P(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Load .env
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())
except Exception:
    pass

from packages.backtests.gex_bounce import run_gex_bounce_backtest, GexBounceParams





def main():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set in environment or .env")

    engine = create_engine(db_url)

    query = text("""
                 SELECT trade_date,
                        ts_utc,
                        ts_pt,
                        bar_index,
                        is_rth, close, high, low, net_gex, gex_wall_below, gex_wall_below_gex, dist_to_wall_below_pts, gex_wall_above, gex_wall_above_gex, put_skew_pp_primary, smile_dte_primary, smile_expir_primary -- 🔹 NEW
                 FROM es_minute_features
                 ORDER BY trade_date, ts_utc
                 """)

    print("[run_gex_bounce_backtest] Loading features from es_minute_features...")
    df = pd.read_sql_query(query, engine)
    print(f"[run_gex_bounce_backtest] Loaded {len(df)} rows")

    params = GexBounceParams(
        # be generous about "near" and "touch"
        entry_proximity_max=2.0,
        touch_proximity_max=1.0,
        min_test_gap_bars=5,

        # 🔻 turn off skew requirements for now
        put_skew_decrease_min=0.0,  # accept any later test (we'll still record change_pct)
        min_baseline_skew=0.0,

        # 🔻 don't filter on net_gex sign, and allow smaller walls
        gex_net_min=-1e12,  # effectively no net_gex filter
        gex_wall_min=5e10,  # allow smaller support GEX

        bar_index_min=30,
        bar_index_max=350,
        stop_loss_points=2.0,
        r_mult=2.0,
        max_bars_in_trade=60,
        max_trades_per_day=8,
        rth_skew_min_offset_bars=15,

        # 🔻 disable "bottom X% of range" filter for now
        entry_max_range_frac=0.0,

        # keep this off until entries look right
        use_upper_gex_target=False,
    )

    trades_df, summary = run_gex_bounce_backtest(df, params)


    print("\n=== Backtest Summary (GEX Bounce) ===")
    for k, v in summary.items():
        if k == "params":
            continue
        print(f"{k}: {v}")

    print("\n=== Sample Trades (first 10) ===")
    if not trades_df.empty:
        print(trades_df.head(10).to_string(index=False))
    else:
        print("No trades generated.")


if __name__ == "__main__":
    main()
