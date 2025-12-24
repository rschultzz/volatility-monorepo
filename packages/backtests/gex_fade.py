# packages/backtests/gex_fade.py
"""
Simplified GEX fade (short upper wall) backtest that works directly on the
`es_minutes_with_features` view.

The view is expected to provide at least:

    trade_date          (date)
    ts_utc              (timestamp with time zone)
    ts_pt               (timestamp or timestamptz)
    bar_index           (int)
    is_rth              (bool)
    open, high, low, close, volume  (floats)
    net_gex             (float, total gamma exposure)
    gex_wall_above      (float, price of nearest large wall above)
    gex_wall_above_gex  (float, gamma size of that wall)
    gex_wall_below      (float, price of nearest large wall below)
    gex_wall_below_gex  (float, gamma size of the lower wall)
    put_skew_pp_primary (float, 0DTE skew metric)
    smile_dte_primary   (float, DTE of that smile (should be ~0 for 0DTE))
    smile_expir_primary (date, actual ORATS expiry date)

Distances to the walls (in points) will be computed here if they are not
already present:

    dist_to_wall_above_pts = gex_wall_above - close
    dist_to_wall_below_pts = close - gex_wall_below
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


@dataclass
class GexFadeParams:
    """
    Parameters for the fade-at-upper-wall backtest.

    All of these are intended to be easy to surface as tunable controls
    in the Dash UI later.
    """

    # --- Entry / test detection ---
    entry_proximity_max: float = 2.0      # max distance (pts) to be considered "near" the upper wall
    touch_proximity_max: float = 1.0      # max distance (pts) to count as a "touch" of the wall
    min_test_gap_bars: int = 5            # minimum bar gap between separate tests
    bar_index_min: int = 30               # ignore very early bars
    bar_index_max: int = 350              # ignore very late bars

    # --- Gamma filters ---
    gex_wall_min: float = 5e10            # minimum |gex_wall_above_gex| to trade
    gex_net_min: float = -1e12            # minimum net_gex (effectively no filter by default)

    # --- Skew filters (optional) ---
    min_entry_skew_abs: float = 0.0       # require |put_skew_pp_primary| >= this at entry (0 = no filter)

    # --- Risk management ---
    stop_loss_points: float = 2.0         # hard stop distance (pts) above entry
    r_mult: float = 2.0                   # target = entry - stop_loss_points * r_mult
    max_bars_in_trade: int = 60           # time stop in bars
    max_trades_per_day: int = 8           # cap number of trades per day

    # --- Misc ---
    rth_only: bool = True                 # only trade bars with is_rth = True


def _ensure_distance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add dist_to_wall_above_pts / dist_to_wall_below_pts if missing."""
    out = df.copy()

    if "dist_to_wall_above_pts" not in out.columns:
        out["dist_to_wall_above_pts"] = np.where(
            out["gex_wall_above"].notna(),
            out["gex_wall_above"] - out["close"],
            np.nan,
        )

    if "dist_to_wall_below_pts" not in out.columns:
        out["dist_to_wall_below_pts"] = np.where(
            out["gex_wall_below"].notna(),
            out["close"] - out["gex_wall_below"],
            np.nan,
        )

    return out


def run_gex_fade_backtest(
    minutes: pd.DataFrame,
    params: GexFadeParams,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Main entry point.

    Parameters
    ----------
    minutes : DataFrame
        Full history from es_minutes_with_features.
    params : GexFadeParams
        Strategy knobs.

    Returns
    -------
    trades_df : DataFrame
        One row per simulated trade.
    summary : dict
        Simple performance summary with the params echo'd back.
    """
    if minutes.empty:
        return pd.DataFrame(), {"params": asdict(params), "n_trades": 0}

    df = minutes.copy()
    df = df.sort_values(["trade_date", "ts_utc"]).reset_index(drop=True)
    df = _ensure_distance_columns(df)

    all_trades: List[Dict] = []

    for trade_date, day in df.groupby("trade_date", sort=True):
        day_trades = _run_day(day, params)
        all_trades.extend(day_trades)

    trades_df = pd.DataFrame(all_trades)
    summary = _summarize_trades(trades_df, params)
    return trades_df, summary


# ---------------------------------------------------------------------------
# Per-day logic
# ---------------------------------------------------------------------------

def _run_day(day: pd.DataFrame, params: GexFadeParams) -> List[Dict]:
    """Simulate all trades for a single trade_date."""
    if params.rth_only:
        day = day[day["is_rth"]].copy()

    if day.empty:
        return []

    # Require some gamma structure for the day
    if (day["gex_wall_above_gex"].abs() >= params.gex_wall_min).sum() == 0:
        return []

    # Build tests of the upper wall
    tests = _build_tests_for_day(day, params)
    if not tests:
        return []

    trades: List[Dict] = []
    last_entry_bar_index: Optional[int] = None

    for test in tests:
        if last_entry_bar_index is not None:
            if test["touch_bar_index"] - last_entry_bar_index < params.min_test_gap_bars:
                continue

        if len(trades) >= params.max_trades_per_day:
            break

        entry = _make_entry_from_test(day, test, params)
        if entry is None:
            continue

        trade = _simulate_trade(day, entry, params)
        trades.append(trade)
        last_entry_bar_index = entry["entry_bar_index"]

    return trades


def _build_tests_for_day(day: pd.DataFrame, params: GexFadeParams) -> List[Dict]:
    """
    Identify tests of the upper GEX wall for a single day.

    A "test" is a contiguous cluster of bars where:
      - price is near the upper wall: 0 <= dist_to_wall_above_pts <= entry_proximity_max
      - the wall is "large enough": |gex_wall_above_gex| >= gex_wall_min
      - bar_index is between bar_index_min and bar_index_max

    For each cluster we also track the first bar that truly "touches" the wall:
      - 0 <= dist_to_wall_above_pts <= touch_proximity_max

    Returns
    -------
    list of dicts with keys:
        start_ts_utc, start_bar_index,
        touch_ts_utc, touch_bar_index
    """
    day = day.copy()

    mask = (
        day["gex_wall_above"].notna()
        & day["gex_wall_above_gex"].abs().ge(params.gex_wall_min)
        & day["dist_to_wall_above_pts"].ge(0)
        & day["dist_to_wall_above_pts"].le(params.entry_proximity_max)
        & day["bar_index"].between(params.bar_index_min, params.bar_index_max)
    )

    candidates = day.loc[mask].copy()
    if candidates.empty:
        return []

    tests: List[Dict] = []
    current: Optional[Dict] = None

    for row in candidates.itertuples():
        if current is None:
            current = {
                "start_ts_utc": row.ts_utc,
                "start_bar_index": row.bar_index,
                "touch_ts_utc": None,
                "touch_bar_index": None,
                "last_bar_index": row.bar_index,
            }
        else:
            # Start a new test if we have a gap in bar_index
            if row.bar_index > current["last_bar_index"] + 1:
                tests.append(current)
                current = {
                    "start_ts_utc": row.ts_utc,
                    "start_bar_index": row.bar_index,
                    "touch_ts_utc": None,
                    "touch_bar_index": None,
                    "last_bar_index": row.bar_index,
                }
            else:
                current["last_bar_index"] = row.bar_index

        # Record the first true "touch" in this cluster
        if (
            row.dist_to_wall_above_pts >= 0
            and row.dist_to_wall_above_pts <= params.touch_proximity_max
            and current["touch_ts_utc"] is None
        ):
            current["touch_ts_utc"] = row.ts_utc
            current["touch_bar_index"] = row.bar_index

    if current is not None:
        tests.append(current)

    # Keep only tests that actually have a touch
    tests = [t for t in tests if t["touch_ts_utc"] is not None]
    return tests


def _make_entry_from_test(
    day: pd.DataFrame,
    test: Dict,
    params: GexFadeParams,
) -> Optional[Dict]:
    """
    Turn a test (cluster) into a concrete short entry at the touch bar.
    """
    touch_idx = day.index[day["ts_utc"] == test["touch_ts_utc"]]
    if touch_idx.empty:
        return None

    idx = touch_idx[0]
    row = day.loc[idx]

    # Basic sanity filters
    if not (params.bar_index_min <= row["bar_index"] <= params.bar_index_max):
        return None

    if row["gex_wall_above_gex"] is None or np.isnan(row["gex_wall_above_gex"]):
        return None

    if abs(row["gex_wall_above_gex"]) < params.gex_wall_min:
        return None

    if row["net_gex"] < params.gex_net_min:
        return None

    if params.min_entry_skew_abs > 0 and (
        row.get("put_skew_pp_primary") is None
        or np.isnan(row.get("put_skew_pp_primary"))
        or abs(row.get("put_skew_pp_primary")) < params.min_entry_skew_abs
    ):
        return None

    entry_price = float(row["close"])
    stop_loss = entry_price + params.stop_loss_points
    target = entry_price - params.stop_loss_points * params.r_mult

    entry = {
        "trade_date": row["trade_date"],
        "entry_ts_utc": row["ts_utc"],
        "entry_ts_pt": row.get("ts_pt"),
        "entry_bar_index": int(row["bar_index"]),
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "target_price": target,
        "gex_wall_above": row.get("gex_wall_above"),
        "gex_wall_above_gex": row.get("gex_wall_above_gex"),
        "dist_to_wall_above_pts": row.get("dist_to_wall_above_pts"),
        "net_gex": row.get("net_gex"),
        "put_skew_pp_primary": row.get("put_skew_pp_primary"),
        "smile_dte_primary": row.get("smile_dte_primary"),
        "smile_expir_primary": row.get("smile_expir_primary"),
    }
    return entry


def _simulate_trade(
    day: pd.DataFrame,
    entry: Dict,
    params: GexFadeParams,
) -> Dict:
    """
    Walk forward bar-by-bar from the entry and decide how the trade exits.

    Because we don't have tick-level data we use OHLC logic:

      - If a future bar's high >= stop_loss before low <= target, we assume
        the stop was hit first (conservative).
      - If low <= target and high < stop_loss, target is hit.
      - If neither is hit within `max_bars_in_trade` bars, we exit at close.
    """
    # Work on the same trade_date, sorted
    day = day.sort_values("ts_utc").reset_index(drop=True)

    start_idx = day.index[day["ts_utc"] == entry["entry_ts_utc"]][0]

    exit_reason = "timeout"
    exit_price = float(day.loc[start_idx, "close"])
    exit_ts_utc = day.loc[start_idx, "ts_utc"]
    exit_ts_pt = day.loc[start_idx, "ts_pt"]
    exit_bar_index = int(day.loc[start_idx, "bar_index"])

    for step, (_, bar) in enumerate(day.iloc[start_idx + 1 :].iterrows(), start=1):
        if step > params.max_bars_in_trade:
            break

        high = float(bar["high"])
        low = float(bar["low"])

        stop_hit = high >= entry["stop_loss"]
        target_hit = low <= entry["target_price"]

        # Conservative ordering: assume stop hits first if both touched
        if stop_hit:
            exit_reason = "stop"
            exit_price = entry["stop_loss"]
            exit_ts_utc = bar["ts_utc"]
            exit_ts_pt = bar["ts_pt"]
            exit_bar_index = int(bar["bar_index"])
            break
        elif target_hit:
            exit_reason = "target"
            exit_price = entry["target_price"]
            exit_ts_utc = bar["ts_utc"]
            exit_ts_pt = bar["ts_pt"]
            exit_bar_index = int(bar["bar_index"])
            break
        else:
            # update last seen bar for timeout exit
            exit_price = float(bar["close"])
            exit_ts_utc = bar["ts_utc"]
            exit_ts_pt = bar["ts_pt"]
            exit_bar_index = int(bar["bar_index"])

    # Short trade: P&L in points is entry_price - exit_price
    pnl_pts = entry["entry_price"] - exit_price
    r_multiple = pnl_pts / params.stop_loss_points if params.stop_loss_points != 0 else 0.0

    trade = {
        **entry,
        "exit_ts_utc": exit_ts_utc,
        "exit_ts_pt": exit_ts_pt,
        "exit_bar_index": exit_bar_index,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "pnl_pts": pnl_pts,
        "r_multiple": r_multiple,
    }
    return trade


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _summarize_trades(trades_df: pd.DataFrame, params: GexFadeParams) -> Dict[str, float]:
    """Compute basic performance stats."""
    summary: Dict[str, float] = {"params": asdict(params)}

    if trades_df.empty:
        summary.update(
            {
                "n_trades": 0,
                "win_rate": 0.0,
                "avg_r": 0.0,
                "total_r": 0.0,
                "best_r": 0.0,
                "worst_r": 0.0,
            }
        )
        return summary

    n_trades = len(trades_df)
    wins = trades_df["r_multiple"] > 0
    win_rate = float(wins.mean())

    avg_r = float(trades_df["r_multiple"].mean())
    total_r = float(trades_df["r_multiple"].sum())
    best_r = float(trades_df["r_multiple"].max())
    worst_r = float(trades_df["r_multiple"].min())

    summary.update(
        {
            "n_trades": n_trades,
            "win_rate": win_rate,
            "avg_r": avg_r,
            "total_r": total_r,
            "best_r": best_r,
            "worst_r": worst_r,
        }
    )
    return summary
