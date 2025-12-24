#!/usr/bin/env python
"""
Run the simplified GEX fade backtest against the `es_minutes_with_features` view.

All tunable parameters live in the PARAMS object so it's easy to wire them
into Dash controls later.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path as _P

import pandas as pd
from sqlalchemy import create_engine, text

# --- Make repo root importable so `packages.*` works ---
REPO_ROOT = _P(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- Load .env (same pattern as other scripts) ---
try:
    from dotenv import load_dotenv, find_dotenv

    load_dotenv(find_dotenv())
except Exception:
    pass

from packages.backtests.gex_fade import GexFadeParams, run_gex_fade_backtest


# ---------------------------------------------------------------------------
# Tunable parameters
# ---------------------------------------------------------------------------

PARAMS = GexFadeParams(
    # How close to the upper wall do we need to be to consider it a "test"?
    entry_proximity_max=2.0,      # points from wall for a candidate cluster
    touch_proximity_max=1.0,      # points from wall to count as a true "touch"

    # Structure of tests / bars we care about
    min_test_gap_bars=5,          # separate tests by at least this many bars
    bar_index_min=60,             # roughly 7:30 PT if bar_index starts at 0
    bar_index_max=360,            # roughly 13:00 PT

    # Gamma filters
    gex_wall_min=5e10,            # minimum |gex_wall_above_gex| in the cluster
    gex_net_min=-1e12,            # minimum net_gex at entry (very loose by default)

    # Optional skew filter (0 = disabled)
    min_entry_skew_abs=0.0,       # require |put_skew_pp_primary| >= this at entry

    # Risk management
    stop_loss_points=2.0,         # stop above entry (short)
    r_mult=2.0,                   # target = entry - stop_loss_points * r_mult
    max_bars_in_trade=60,         # time stop in bars
    max_trades_per_day=8,         # safety cap

    # Misc
    rth_only=True,                # trade only bars flagged as RTH
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_minutes(engine) -> pd.DataFrame:
    """
    Load the joined minute/gamma/smile data from the view.

    We intentionally only select columns that we know exist on
    `es_minutes_with_features`. Distance-to-wall columns, if missing,
    are computed in `gex_fade.run_gex_fade_backtest`.
    """
    query = text(
        """
        SELECT
            trade_date,
            ts_utc,
            ts_pt,
            bar_index,
            is_rth,
            open,
            high,
            low,
            close,
            volume,
            net_gex,
            gex_wall_above,
            gex_wall_above_gex,
            gex_wall_below,
            gex_wall_below_gex,
            put_skew_pp_primary,
            smile_dte_primary,
            smile_expir_primary
        FROM es_minutes_with_features
        ORDER BY trade_date, ts_utc
        """
    )

    df = pd.read_sql_query(
        query,
        engine,
        parse_dates=["trade_date", "ts_utc", "ts_pt"],
    )
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL environment variable is not set")

    print("[run_gex_fade_backtest] Loading data from es_minutes_with_features...")
    engine = create_engine(database_url)

    df = load_minutes(engine)
    print(f"[run_gex_fade_backtest] Loaded {len(df):,} rows")

    trades_df, summary = run_gex_fade_backtest(df, PARAMS)
    print(f"[run_gex_fade_backtest] Simulated {len(trades_df):,} trades")

    # --- Persist outputs next to the script ---
    out_trades = "gex_fade_trades.csv"
    out_summary = "gex_fade_summary.json"

    trades_df.to_csv(out_trades, index=False)

    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"[run_gex_fade_backtest] Wrote trades to {out_trades}")
    print("[run_gex_fade_backtest] Summary:")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
