from __future__ import annotations

# Load local environment variables from .env (for DATABASE_URL)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import os
import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

# Make the repo root importable so that `packages.*` works
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from packages.backtests.gex_fade import GexFadeParams, run_gex_fade_backtest  # noqa: E402


# ===================== Tunable parameters =====================

# Date range for the backtest (inclusive).
# Use None for either end to leave it open.
START_DATE = "2025-11-01"
END_DATE = "2025-12-31"

# Strategy parameters. You can tweak these and re-run.
PARAMS = GexFadeParams(
    entry_proximity_max=2.0,      # points below wall
    gex_wall_min=5e10,            # minimum |GEX| at the wall
    gex_net_min=0.0,              # require net_gex >= this
    min_bar_index=30,             # don't trade in first 30 minutes
    max_bar_index=350,            # don't trade into the close
    require_rth=True,             # only trade RTH minutes
    min_abs_skew=0.5,             # require at least 0.5pp of skew (abs)
    stop_loss_points=2.0,         # stop size in ES points
    target_rr=2.0,                # target = stop * R
    max_bars_in_trade=60,         # max holding time
    max_trades_per_day=8,         # daily trade cap
)


def main() -> None:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL environment variable is not set")

    engine = create_engine(db_url)

    print("[run_gex_fade_backtest] Loading data from es_minutes_with_features...")

    base_query = """
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
    """

    conditions = []
    params: dict[str, object] = {}

    if START_DATE is not None:
        conditions.append("trade_date >= :start_date")
        params["start_date"] = START_DATE
    if END_DATE is not None:
        conditions.append("trade_date <= :end_date")
        params["end_date"] = END_DATE

    if conditions:
        base_query += " WHERE " + " AND ".join(conditions)

    base_query += " ORDER BY trade_date, ts_utc"

    with engine.connect() as conn:
        df = pd.read_sql_query(text(base_query), conn, params=params)

    print(f"[run_gex_fade_backtest] Loaded {len(df):,} rows")

    trades_df, summary = run_gex_fade_backtest(df, PARAMS)

    print("\n[run_gex_fade_backtest] Summary:")
    print(f"  n_trades : {summary['n_trades']}")
    print(f"  win_rate : {summary['win_rate']:.1%}")
    print(f"  avg R    : {summary['avg_r']:.2f}")
    print(f"  total R  : {summary['total_r']:.1f}")

    if not trades_df.empty:
        cols_to_show = [
            "trade_date",
            "entry_ts_pt",
            "entry_bar_index",
            "entry_price",
            "wall_level",
            "dist_to_wall_at_entry",
            "put_skew_entry_pp",
            "exit_ts_pt",
            "exit_bar_index",
            "exit_price",
            "exit_reason",
            "pnl_points",
            "r_mult",
        ]
        cols_to_show = [c for c in cols_to_show if c in trades_df.columns]

        print("\n[run_gex_fade_backtest] First 50 trades:")
        print(trades_df[cols_to_show].head(50).to_string(index=False))

        out_path = REPO_ROOT / "gex_fade_trades.csv"
        trades_df.to_csv(out_path, index=False)
        print(f"\n[run_gex_fade_backtest] Saved trades to {out_path}")
    else:
        print("\n[run_gex_fade_backtest] No trades generated with current parameters.")


if __name__ == "__main__":
    main()
