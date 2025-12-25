from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


@dataclass
class GexFadeParams:
    """
    Updated entry logic:
    - Must observe an "anchor" test of the overhead wall (price within proximity).
    - Must observe a second (or later) test of the SAME wall at least
      `min_minutes_between_tests` minutes after the anchor test.
    - On the confirm test, |put skew| must have dropped by at least
      `min_put_skew_drop_frac` vs the anchor test.

    Optional enhancement:
    - If `require_reset_between_tests=True`, then between anchor and confirm
      the price must "leave the wall zone" before the confirm counts.
      We define leaving the zone as:
        dist_to_wall >= entry_proximity_max + reset_buffer_points
      (i.e., price pulls back farther below the wall than the test zone).
    """

    # GEX / location filters
    entry_proximity_max: float = 2.0   # max points below wall to count as a "test"
    gex_wall_min: float = 5e10         # min |gex_wall_above_gex| to be tradable
    gex_net_min: float = 0.0           # require net_gex >= this

    # Time-of-day / structure filters
    min_bar_index: int = 30
    max_bar_index: int = 350
    require_rth: bool = True

    # Skew filter (required for "tests" and entry)
    min_abs_skew: float = 0.0

    # Multi-test + skew-drop logic
    min_minutes_between_tests: int = 30
    min_put_skew_drop_frac: float = 0.50  # 0.50 => 50% drop in |skew|

    # NEW: optional reset requirement
    require_reset_between_tests: bool = False
    reset_buffer_points: float = 2.0  # extra distance beyond proximity zone to count as "reset"

    # Risk management
    stop_loss_points: float = 2.0
    target_rr: float = 2.0
    max_bars_in_trade: int = 60
    max_trades_per_day: int = 8


def _compute_dist_to_wall(row: pd.Series) -> float:
    wall = row.get("gex_wall_above")
    close = row.get("close")
    if pd.isna(wall) or pd.isna(close):
        return np.nan
    return float(wall) - float(close)  # negative means price above wall


def _fmt_time(ts: Any) -> str:
    if pd.isna(ts):
        return ""
    if hasattr(ts, "strftime"):
        return ts.strftime("%H:%M")
    return str(ts)


def _fmt_billions(val: float) -> str:
    if pd.isna(val):
        return ""
    return f"{val / 1e9:.2f}B"


def _safe_minutes_diff(ts_late: Any, ts_early: Any) -> float:
    try:
        if pd.isna(ts_late) or pd.isna(ts_early):
            return float("nan")
        delta = ts_late - ts_early
        return float(delta.total_seconds() / 60.0)
    except Exception:
        return float("nan")


def run_gex_fade_backtest(
    df: pd.DataFrame,
    params: GexFadeParams,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if df.empty:
        return df.copy(), {"n_trades": 0, "win_rate": 0.0, "avg_r": 0.0, "total_r": 0.0}

    df = df.sort_values(["trade_date", "ts_utc"]).reset_index(drop=True)
    df = df.copy()
    df["dist_to_wall_above_pts"] = df.apply(_compute_dist_to_wall, axis=1)

    trades: list[Dict[str, Any]] = []

    current_trade_date = None
    trades_today = 0
    position: Dict[str, Any] | None = None

    # Track anchor tests per wall within the day
    # key: rounded wall level, value: anchor info (plus reset_seen)
    anchors_by_wall: Dict[float, Dict[str, Any]] = {}

    stop = float(params.stop_loss_points)
    target_points = float(params.stop_loss_points) * float(params.target_rr)

    # Precompute reset threshold distance beyond the test zone
    reset_trigger_dist = float(params.entry_proximity_max) + float(params.reset_buffer_points)

    for _, row in df.iterrows():
        trade_date = row["trade_date"]

        # New day resets
        if current_trade_date is None or trade_date != current_trade_date:
            # If still in trade from prior day, exit at last close
            if position is not None:
                exit_price = float(position["last_close"])
                pnl_points = -(exit_price - position["entry_price"])  # short

                trades.append(
                    {
                        "trade_date": current_trade_date,
                        "entry_ts_pt": _fmt_time(position["entry_ts_pt"]),
                        "exit_ts_pt": _fmt_time(position["last_ts_pt"]),
                        "entry_price": position["entry_price"],
                        "exit_price": exit_price,
                        "pnl_points": pnl_points,
                        "stop_price": position["stop_price"],
                        "target_price": position["target_price"],
                        "exit_reason": "eod",
                        "wall_level": position["wall_level"],
                        "wall_gex": _fmt_billions(position["wall_gex"]),
                        "net_gex": _fmt_billions(position["net_gex"]),
                        "dist_to_wall_at_entry": position["dist_to_wall_at_entry"],
                        "put_skew_entry_pp": round(position["put_skew_entry_pp"], 2),
                        "smile_dte_primary": position["smile_dte_primary"],
                        "smile_expir_primary": position["smile_expir_primary"],
                        "r_mult": pnl_points / stop if stop != 0 else np.nan,
                        # Test metadata
                        "anchor_test_ts_pt": _fmt_time(position.get("anchor_test_ts_pt")),
                        "confirm_test_ts_pt": _fmt_time(position.get("confirm_test_ts_pt")),
                        "anchor_put_skew_pp": round(float(position.get("anchor_put_skew_pp", np.nan)), 2),
                        "confirm_put_skew_pp": round(float(position.get("confirm_put_skew_pp", np.nan)), 2),
                        "put_skew_drop_pct": round(float(position.get("put_skew_drop_pct", np.nan)), 1),
                        "minutes_between_tests": position.get("minutes_between_tests", np.nan),
                        "reset_required": bool(position.get("reset_required", False)),
                        "reset_seen": bool(position.get("reset_seen", False)),
                    }
                )
                position = None

            current_trade_date = trade_date
            trades_today = 0
            anchors_by_wall = {}

        # Update trailing info
        last_close = float(row["close"])
        last_ts_utc = row.get("ts_utc")
        last_ts_pt = row.get("ts_pt")
        last_bar_index = int(row["bar_index"])

        if position is not None:
            position["last_close"] = last_close
            position["last_ts_utc"] = last_ts_utc
            position["last_ts_pt"] = last_ts_pt
            position["last_bar_index"] = last_bar_index

        # RTH filter
        if params.require_rth and not bool(row.get("is_rth", False)):
            continue

        # Per-day trade limit
        if trades_today >= params.max_trades_per_day:
            continue

        bar_index = int(row["bar_index"])

        # ----- Manage open position -----
        if position is not None and position["is_open"]:
            if bar_index - position["entry_bar_index"] >= params.max_bars_in_trade:
                exit_reason = "max_bars"
                exit_price = last_close
            else:
                stop_price = float(position["stop_price"])
                target_price = float(position["target_price"])
                bar_high = float(row["high"])
                bar_low = float(row["low"])

                exit_reason = None
                exit_price = None

                # Conservative: stop checked first
                if bar_high >= stop_price:
                    exit_reason = "stop"
                    exit_price = stop_price
                elif bar_low <= target_price:
                    exit_reason = "target"
                    exit_price = target_price

            if exit_reason is not None:
                pnl_points = -(float(exit_price) - float(position["entry_price"]))
                trades.append(
                    {
                        "trade_date": trade_date,
                        "entry_ts_pt": _fmt_time(position["entry_ts_pt"]),
                        "exit_ts_pt": _fmt_time(row.get("ts_pt")),
                        "entry_price": position["entry_price"],
                        "exit_price": float(exit_price),
                        "pnl_points": pnl_points,
                        "stop_price": position["stop_price"],
                        "target_price": position["target_price"],
                        "exit_reason": exit_reason,
                        "wall_level": position["wall_level"],
                        "wall_gex": _fmt_billions(position["wall_gex"]),
                        "net_gex": _fmt_billions(position["net_gex"]),
                        "dist_to_wall_at_entry": position["dist_to_wall_at_entry"],
                        "put_skew_entry_pp": round(position["put_skew_entry_pp"], 2),
                        "smile_dte_primary": position["smile_dte_primary"],
                        "smile_expir_primary": position["smile_expir_primary"],
                        "r_mult": pnl_points / stop if stop != 0 else np.nan,
                        # Test metadata
                        "anchor_test_ts_pt": _fmt_time(position.get("anchor_test_ts_pt")),
                        "confirm_test_ts_pt": _fmt_time(position.get("confirm_test_ts_pt")),
                        "anchor_put_skew_pp": round(float(position.get("anchor_put_skew_pp", np.nan)), 2),
                        "confirm_put_skew_pp": round(float(position.get("confirm_put_skew_pp", np.nan)), 2),
                        "put_skew_drop_pct": round(float(position.get("put_skew_drop_pct", np.nan)), 1),
                        "minutes_between_tests": position.get("minutes_between_tests", np.nan),
                        "reset_required": bool(position.get("reset_required", False)),
                        "reset_seen": bool(position.get("reset_seen", False)),
                    }
                )
                trades_today += 1
                position = None

            if position is not None:
                continue

        # ----- No open position: scan for anchor/confirm tests -----
        if bar_index < params.min_bar_index or bar_index > params.max_bar_index:
            continue

        gex_wall_above = row.get("gex_wall_above")
        gex_wall_above_gex = row.get("gex_wall_above_gex")
        net_gex = row.get("net_gex")
        dist_to_wall = row.get("dist_to_wall_above_pts", np.nan)

        if pd.isna(gex_wall_above) or pd.isna(gex_wall_above_gex) or pd.isna(net_gex) or pd.isna(dist_to_wall):
            continue

        if abs(float(gex_wall_above_gex)) < float(params.gex_wall_min):
            continue

        if float(net_gex) < float(params.gex_net_min):
            continue

        wall_level = float(gex_wall_above)
        wall_key = round(wall_level, 2)

        # If we already have an anchor, update reset_seen whenever price pulls back far enough
        if wall_key in anchors_by_wall:
            if float(dist_to_wall) >= reset_trigger_dist:
                anchors_by_wall[wall_key]["reset_seen"] = True

        # A "test" bar must be within proximity zone
        if not (0.0 <= float(dist_to_wall) <= float(params.entry_proximity_max)):
            continue

        # A test also requires skew to be present (and pass min_abs_skew)
        put_skew = row.get("put_skew_pp_primary")
        if pd.isna(put_skew):
            continue
        if abs(float(put_skew)) < float(params.min_abs_skew):
            continue

        # If no anchor yet, record anchor and move on
        if wall_key not in anchors_by_wall:
            anchors_by_wall[wall_key] = {
                "ts_utc": row.get("ts_utc"),
                "ts_pt": row.get("ts_pt"),
                "bar_index": bar_index,
                "put_skew_pp": float(put_skew),
                "reset_seen": False,
            }
            continue

        # Anchor exists: enforce min time and optional reset requirement
        anchor = anchors_by_wall[wall_key]
        minutes_between = _safe_minutes_diff(row.get("ts_utc"), anchor.get("ts_utc"))
        if not np.isfinite(minutes_between) or minutes_between < float(params.min_minutes_between_tests):
            continue

        if params.require_reset_between_tests and not bool(anchor.get("reset_seen", False)):
            # We haven't "left the wall zone" yet, so this test cannot confirm
            continue

        anchor_abs = abs(float(anchor.get("put_skew_pp", np.nan)))
        confirm_abs = abs(float(put_skew))
        if not np.isfinite(anchor_abs) or anchor_abs <= 0:
            continue

        drop_frac = (anchor_abs - confirm_abs) / anchor_abs
        if drop_frac < float(params.min_put_skew_drop_frac):
            continue

        # Entry: short at close of confirm test bar
        entry_price = float(row["close"])
        stop_price = entry_price + stop
        target_price = entry_price - target_points

        position = {
            "is_open": True,
            "entry_ts": row.get("ts_utc"),
            "entry_ts_pt": row.get("ts_pt"),
            "entry_bar_index": bar_index,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "target_price": target_price,
            "wall_level": wall_level,
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
            # Test metadata
            "anchor_test_ts_pt": anchor.get("ts_pt"),
            "confirm_test_ts_pt": row.get("ts_pt"),
            "anchor_put_skew_pp": float(anchor.get("put_skew_pp")),
            "confirm_put_skew_pp": float(put_skew),
            "put_skew_drop_pct": float(drop_frac * 100.0),
            "minutes_between_tests": int(round(minutes_between)),
            "reset_required": bool(params.require_reset_between_tests),
            "reset_seen": bool(anchor.get("reset_seen", False)),
        }

        # After taking a trade, clear anchors so we don't reuse them immediately
        anchors_by_wall = {}

    # Close any open position at end
    if position is not None:
        exit_price = float(position["last_close"])
        pnl_points = -(exit_price - position["entry_price"])
        trades.append(
            {
                "trade_date": current_trade_date,
                "entry_ts_pt": _fmt_time(position["entry_ts_pt"]),
                "exit_ts_pt": _fmt_time(position["last_ts_pt"]),
                "entry_price": position["entry_price"],
                "exit_price": exit_price,
                "pnl_points": pnl_points,
                "stop_price": position["stop_price"],
                "target_price": position["target_price"],
                "exit_reason": "eod",
                "wall_level": position["wall_level"],
                "wall_gex": _fmt_billions(position["wall_gex"]),
                "net_gex": _fmt_billions(position["net_gex"]),
                "dist_to_wall_at_entry": position["dist_to_wall_at_entry"],
                "put_skew_entry_pp": round(position["put_skew_entry_pp"], 2),
                "smile_dte_primary": position["smile_dte_primary"],
                "smile_expir_primary": position["smile_expir_primary"],
                "r_mult": pnl_points / stop if stop != 0 else np.nan,
                "anchor_test_ts_pt": _fmt_time(position.get("anchor_test_ts_pt")),
                "confirm_test_ts_pt": _fmt_time(position.get("confirm_test_ts_pt")),
                "anchor_put_skew_pp": round(float(position.get("anchor_put_skew_pp", np.nan)), 2),
                "confirm_put_skew_pp": round(float(position.get("confirm_put_skew_pp", np.nan)), 2),
                "put_skew_drop_pct": round(float(position.get("put_skew_drop_pct", np.nan)), 1),
                "minutes_between_tests": position.get("minutes_between_tests", np.nan),
                "reset_required": bool(position.get("reset_required", False)),
                "reset_seen": bool(position.get("reset_seen", False)),
            }
        )

    trades_df = pd.DataFrame(trades)

    if trades_df.empty:
        summary = {"n_trades": 0, "win_rate": 0.0, "avg_r": 0.0, "total_r": 0.0, "params": params}
        return trades_df, summary

    trades_df["r_mult"] = trades_df["pnl_points"] / stop if stop != 0 else np.nan

    n_trades = len(trades_df)
    wins = (trades_df["pnl_points"] > 0).sum()
    win_rate = wins / n_trades if n_trades else 0.0
    avg_r = trades_df["r_mult"].mean()
    total_r = trades_df["r_mult"].sum()

    summary = {"n_trades": n_trades, "win_rate": win_rate, "avg_r": avg_r, "total_r": total_r, "params": params}
    return trades_df, summary
