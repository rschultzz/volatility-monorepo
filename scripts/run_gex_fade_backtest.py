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

# Make repo root importable so `packages.*` works
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from packages.backtests.gex_fade import GexFadeParams, run_gex_fade_backtest  # noqa: E402


START_DATE = "2025-11-01"
END_DATE = "2025-12-31"

# ---- NEW: SS expected toggle (Dash will control this later) ----
# False = use anchor skew baseline (current behavior)
# True  = compare confirm skew vs SS-expected baseline (once gex_fade supports it)
USE_EXPECTED_SS = False

PARAMS = GexFadeParams(
    entry_proximity_max=2.0,
    gex_wall_min=5e10,
    gex_net_min=0.0,
    min_bar_index=30,
    max_bar_index=350,
    require_rth=True,
    min_abs_skew=0.5,

    min_minutes_between_tests=30,
    min_put_skew_drop_frac=0.50,

    # Reset option
    require_reset_between_tests=False,  # set True to require "leave zone" before confirm
    reset_buffer_points=2.0,            # extra distance beyond proximity zone

    stop_loss_points=2.0,
    target_rr=2.0,
    max_bars_in_trade=60,
    max_trades_per_day=8,
)

# If/when gex_fade adds this param, this will wire in automatically.
# This avoids breaking today if your local GexFadeParams doesn’t yet include it.
if hasattr(PARAMS, "compare_put_skew_to_expected_ss"):
    setattr(PARAMS, "compare_put_skew_to_expected_ss", bool(USE_EXPECTED_SS))


def main() -> None:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL environment variable is not set")

    engine = create_engine(db_url)

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

            -- existing skew/smile fields
            put_skew_pp_primary,
            smile_dte_primary,
            smile_expir_primary,

            -- NEW: required for SS-expected behavior (appended in your view)
            stock_price,
            atmiv,
            call_skew_pp_primary,

            -- NEW: raw smile buckets (Skew module expected-SS logic scans volNN columns)
            vol10, vol15, vol20, vol25, vol30, vol35, vol40, vol45,
            vol50,
            vol55, vol60, vol65, vol70, vol75, vol80, vol85, vol90

        FROM es_minutes_with_features
    """

    conditions = []
    qparams: dict[str, object] = {}

    if START_DATE:
        conditions.append("trade_date >= :start_date")
        qparams["start_date"] = START_DATE
    if END_DATE:
        conditions.append("trade_date <= :end_date")
        qparams["end_date"] = END_DATE

    if conditions:
        base_query += " WHERE " + " AND ".join(conditions)

    base_query += " ORDER BY trade_date, ts_utc"

    with engine.connect() as conn:
        df = pd.read_sql_query(text(base_query), conn, params=qparams)

    trades_df, summary = run_gex_fade_backtest(df, PARAMS)

    print("\nSummary:")
    print(f"  n_trades : {summary['n_trades']}")
    print(f"  win_rate : {summary['win_rate']:.1%}")
    print(f"  avg R    : {summary['avg_r']:.2f}")
    print(f"  total R  : {summary['total_r']:.1f}")

    if trades_df.empty:
        print("\nNo trades.")
        return

    cols_to_show = [
        "trade_date",
        "anchor_test_ts_pt",
        "confirm_test_ts_pt",
        "minutes_between_tests",
        "reset_required",
        "reset_seen",
        "anchor_put_skew_pp",
        "confirm_put_skew_pp",
        "put_skew_drop_pct",
        "entry_ts_pt",
        "exit_ts_pt",
        "entry_price",
        "exit_price",
        "exit_reason",
        "pnl_points",
        "r_mult",
    ]
    cols_to_show = [c for c in cols_to_show if c in trades_df.columns]
    print("\nFirst 50 trades:")
    print(trades_df[cols_to_show].head(50).to_string(index=False))

    out_path = REPO_ROOT / "gex_fade_trades.csv"
    trades_df.to_csv(out_path, index=False)
    print(f"\nSaved trades to {out_path}")


if __name__ == "__main__":
    main()
