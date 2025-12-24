from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class GexFadeParams:
    """Parameters for the GEX fade (short at call wall) backtest.

    Notes
    -----
    - We always trade from the SHORT side, fading an overhead GEX wall.
    - Entry happens when price is within `entry_proximity_max` points
      *below* the wall and skew conditions are satisfied.
    - Exit is either at a fixed stop, a fixed R-multiple target, or a
      max holding time in bars.
    """

    # GEX / location filters
    entry_proximity_max: float = 2.0   # max points below wall to enter
    gex_wall_min: float = 5e10         # min |gex_wall_above_gex| to be tradable
    gex_net_min: float = 0.0           # require net_gex >= this (use >0 for 'call heavy')

    # Time-of-day / structure filters
    min_bar_index: int = 30
    max_bar_index: int = 350
    require_rth: bool = True          # only trade during RTH minutes

    # Skew filter
    min_abs_skew: float = 0.0         # minimum |put_skew_pp_primary| at entry

    # Risk management
    stop_loss_points: float = 2.0     # points
    target_rr: float = 2.0            # target = stop_loss_points * target_rr
    max_bars_in_trade: int = 60
    max_trades_per_day: int = 8


def _compute_dist_to_wall(row: pd.Series) -> float:
    wall = row.get("gex_wall_above")
    close = row.get("close")
    if pd.isna(wall) or pd.isna(close):
        return np.nan
    # Distance in *points* from price up to the wall.
    # If price is above the wall this will be negative.
    return float(wall) - float(close)


def run_gex_fade_backtest(
    df: pd.DataFrame,
    params: GexFadeParams,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run the GEX fade backtest on a minute-level dataframe.

    Expected columns in `df` (from es_minutes_with_features):

    - trade_date (date)
    - ts_utc (timestamp, tz-aware)
    - ts_pt (timestamp, naive or tz-naive Pacific)
    - bar_index (int)
    - is_rth (bool)
    - open, high, low, close, volume (floats)
    - net_gex, gex_wall_above, gex_wall_above_gex
    - gex_wall_below, gex_wall_below_gex (not strictly required)
    - put_skew_pp_primary
    - smile_dte_primary
    - smile_expir_primary
    """

    if df.empty:
        return df.copy(), {
            "n_trades": 0,
            "win_rate": 0.0,
            "avg_r": 0.0,
            "total_r": 0.0,
        }

    # Ensure sorted
    df = df.sort_values(["trade_date", "ts_utc"]).reset_index(drop=True)

    # Pre-compute distance to the overhead wall (in points)
    df = df.copy()
    df["dist_to_wall_above_pts"] = df.apply(_compute_dist_to_wall, axis=1)

    trades: list[Dict[str, Any]] = []

    current_trade_date = None
    trades_today = 0
    position: Dict[str, Any] | None = None

    stop = params.stop_loss_points
    target_points = params.stop_loss_points * params.target_rr

    for idx, row in df.iterrows():
        trade_date = row["trade_date"]

        # New day: reset per-day counters / position
        if current_trade_date is None or trade_date != current_trade_date:
            # If we were still in a trade from the prior day, exit at prior close
            if position is not None:
                exit_price = float(position["last_close"])
                pnl_points = -(exit_price - position["entry_price"])  # short
                trades.append(
                    {
                        "trade_date": current_trade_date,
                        "entry_ts": position["entry_ts"],
                        "entry_ts_pt": position["entry_ts_pt"],
                        "exit_ts": position["last_ts_utc"],
                        "exit_ts_pt": position["last_ts_pt"],
                        "entry_bar_index": position["entry_bar_index"],
                        "exit_bar_index": position["last_bar_index"],
                        "entry_price": position["entry_price"],
                        "exit_price": exit_price,
                        "pnl_points": pnl_points,
                        "stop_price": position["stop_price"],
                        "target_price": position["target_price"],
                        "exit_reason": "eod",
                        "wall_level": position["wall_level"],
                        "wall_gex": position["wall_gex"],
                        "net_gex": position["net_gex"],
                        "dist_to_wall_at_entry": position["dist_to_wall_at_entry"],
                        "put_skew_entry_pp": position["put_skew_entry_pp"],
                        "smile_dte_primary": position["smile_dte_primary"],
                        "smile_expir_primary": position["smile_expir_primary"],
                        "r_mult": pnl_points / stop if stop != 0 else np.nan,
                    }
                )
                position = None

            current_trade_date = trade_date
            trades_today = 0

        # Update trailing info even if we don't trade at this bar
        last_close = float(row["close"])
        last_ts_utc = row["ts_utc"]
        last_ts_pt = row.get("ts_pt")
        last_bar_index = int(row["bar_index"])

        if position is not None:
            position["last_close"] = last_close
            position["last_ts_utc"] = last_ts_utc
            position["last_ts_pt"] = last_ts_pt
            position["last_bar_index"] = last_bar_index

        # Skip non-RTH if required
        if params.require_rth and not bool(row.get("is_rth", False)):
            continue

        # Skip if we've already hit our trade limit for the day
        if trades_today >= params.max_trades_per_day:
            continue

        bar_index = int(row["bar_index"])

        # ----- Manage open position first -----
        if position is not None and position["is_open"]:
            # Hard max holding time
            if bar_index - position["entry_bar_index"] >= params.max_bars_in_trade:
                exit_reason = "max_bars"
                exit_price = last_close
            else:
                # For a short: stop if high >= stop_price; target if low <= target_price
                stop_price = position["stop_price"]
                target_price = position["target_price"]
                bar_high = float(row["high"])
                bar_low = float(row["low"])

                exit_reason = None
                exit_price = None

                if bar_high >= stop_price:
                    exit_reason = "stop"
                    exit_price = stop_price
                elif bar_low <= target_price:
                    exit_reason = "target"
                    exit_price = target_price

            if exit_reason is not None:
                pnl_points = -(exit_price - position["entry_price"])
                trades.append(
                    {
                        "trade_date": trade_date,
                        "entry_ts": position["entry_ts"],
                        "entry_ts_pt": position["entry_ts_pt"],
                        "exit_ts": row["ts_utc"],
                        "exit_ts_pt": row.get("ts_pt"),
                        "entry_bar_index": position["entry_bar_index"],
                        "exit_bar_index": bar_index,
                        "entry_price": position["entry_price"],
                        "exit_price": exit_price,
                        "pnl_points": pnl_points,
                        "stop_price": position["stop_price"],
                        "target_price": position["target_price"],
                        "exit_reason": exit_reason,
                        "wall_level": position["wall_level"],
                        "wall_gex": position["wall_gex"],
                        "net_gex": position["net_gex"],
                        "dist_to_wall_at_entry": position["dist_to_wall_at_entry"],
                        "put_skew_entry_pp": position["put_skew_entry_pp"],
                        "smile_dte_primary": position["smile_dte_primary"],
                        "smile_expir_primary": position["smile_expir_primary"],
                        "r_mult": pnl_points / stop if stop != 0 else np.nan,
                    }
                )
                trades_today += 1
                position = None

            # If still in the trade, don't try to open another
            if position is not None:
                continue

        # ----- No open position: look for a new short fade -----
        # Basic time-of-day filter
        if bar_index < params.min_bar_index or bar_index > params.max_bar_index:
            continue

        # GEX wall + regime checks
        gex_wall_above = row.get("gex_wall_above")
        gex_wall_above_gex = row.get("gex_wall_above_gex")
        net_gex = row.get("net_gex")
        dist_to_wall = row.get("dist_to_wall_above_pts", np.inf)

        if (
            pd.isna(gex_wall_above)
            or pd.isna(gex_wall_above_gex)
            or pd.isna(net_gex)
        ):
            continue

        if abs(float(gex_wall_above_gex)) < params.gex_wall_min:
            continue

        if float(net_gex) < params.gex_net_min:
            continue

        # We want price just below the wall (distance >= 0 but not too big)
        if not (0.0 <= float(dist_to_wall) <= params.entry_proximity_max):
            continue

        # Skew filter — this is now *required*, so trades can’t fire without skew
        put_skew = row.get("put_skew_pp_primary")
        if pd.isna(put_skew):
            continue

        if abs(float(put_skew)) < params.min_abs_skew:
            continue

        # All conditions satisfied: open a new short
        entry_price = float(row["close"])
        stop_price = entry_price + stop
        target_price = entry_price - target_points

        position = {
            "is_open": True,
            "entry_ts": row["ts_utc"],
            "entry_ts_pt": row.get("ts_pt"),
            "entry_bar_index": bar_index,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "wall_level": float(gex_wall_above),
            "wall_gex": float(gex_wall_above_gex),
            "net_gex": float(net_gex),
            "dist_to_wall_at_entry": float(dist_to_wall),
            "put_skew_entry_pp": float(put_skew),
            "smile_dte_primary": row.get("smile_dte_primary"),
            "smile_expir_primary": row.get("smile_expir_primary"),
            "last_close": last_close,
            "last_ts_utc": last_ts_utc,
            "last_ts_pt": last_ts_pt,
            "last_bar_index": last_bar_index,
        }

    # If we end the loop with an open position, close it at the last seen price
    if position is not None:
        exit_price = float(position["last_close"])
        pnl_points = -(exit_price - position["entry_price"])
        trades.append(
            {
                "trade_date": current_trade_date,
                "entry_ts": position["entry_ts"],
                "entry_ts_pt": position["entry_ts_pt"],
                "exit_ts": position["last_ts_utc"],
                "exit_ts_pt": position["last_ts_pt"],
                "entry_bar_index": position["entry_bar_index"],
                "exit_bar_index": position["last_bar_index"],
                "entry_price": position["entry_price"],
                "exit_price": exit_price,
                "pnl_points": pnl_points,
                "stop_price": position["stop_price"],
                "target_price": position["target_price"],
                "exit_reason": "eod",
                "wall_level": position["wall_level"],
                "wall_gex": position["wall_gex"],
                "net_gex": position["net_gex"],
                "dist_to_wall_at_entry": position["dist_to_wall_at_entry"],
                "put_skew_entry_pp": position["put_skew_entry_pp"],
                "smile_dte_primary": position["smile_dte_primary"],
                "smile_expir_primary": position["smile_expir_primary"],
                "r_mult": pnl_points / stop if stop != 0 else np.nan,
            }
        )

    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        summary = {
            "n_trades": 0,
            "win_rate": 0.0,
            "avg_r": 0.0,
            "total_r": 0.0,
            "params": params,
        }
        return trades_df, summary

    # Basic summary stats
    trades_df["r_mult"] = trades_df["pnl_points"] / stop if stop != 0 else np.nan

    n_trades = len(trades_df)
    wins = (trades_df["pnl_points"] > 0).sum()
    win_rate = wins / n_trades if n_trades > 0 else 0.0
    avg_r = trades_df["r_mult"].mean()
    total_r = trades_df["r_mult"].sum()

    summary = {
        "n_trades": n_trades,
        "win_rate": win_rate,
        "avg_r": avg_r,
        "total_r": total_r,
        "params": params,
    }

    return trades_df, summary
