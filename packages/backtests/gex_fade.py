from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd


@dataclass
class GexFadeParams:
    """
    Parameters for the 'fade upper GEX wall using put skew' strategy.
    Designed so you can wire these directly to Dash sliders / inputs later.
    """

    # How close price has to be to be considered "near" the wall (for clusters/entries)
    entry_proximity_max: float = 0.75

    # How close price has to be to be considered a true "touch" of the wall
    # when we pick the skew time for each test.
    touch_proximity_max: float = 0.25

    # How many 1m bars between near-wall clusters to treat them as separate tests
    min_test_gap_bars: int = 20

    # Minimum % increase in put skew between baseline (Test 1) and later test
    put_skew_increase_min: float = 0.7

    # Minimum absolute baseline skew to be usable as reference
    min_baseline_skew: float = 0.25

    # GEX regime / wall size filters
    gex_net_min: float = 0.0
    gex_wall_min: float = 1e11

    # When we’re allowed to open trades intraday
    bar_index_min: int = 30
    bar_index_max: int = 330

    # Risk management
    stop_loss_points: float = 2.0
    r_mult: float = 2.0
    max_bars_in_trade: int = 60
    max_trades_per_day: int = 3
    close_cutoff_bar_index: Optional[int] = None

    # Don't use the first N RTH minutes for skew baselines / comparisons
    rth_skew_min_offset_bars: int = 15

    # Only trade walls within this many points of the highest wall for the day.
    # (Currently unused; here for future experiments.)
    primary_wall_tolerance_pts: float = 3.0

    # Require entries to be in the top portion of the day's RTH range.
    # 0.0 disables the filter; 0.7 means "only short if price is in the top 70% of the day's range".
    entry_min_range_frac: float = 0.0

    # Optional: for shorts, target the next big GEX wall *below* price
    # instead of a fixed R-multiple.
    use_lower_gex_target: bool = False
    lower_gex_min_abs: float = 1e11  # min |gex_wall_below_gex| to trust as a real level


def run_gex_fade_backtest(
    df_features: pd.DataFrame,
    params: Optional[GexFadeParams] = None,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run the 'fade upper GEX wall' backtest on es_minute_features data.
    """
    if params is None:
        params = GexFadeParams()

    df = df_features.copy()

    # Normalize trade_date to a plain date
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date

    if start_date is not None:
        start_date = pd.to_datetime(start_date).date()
        df = df[df["trade_date"] >= start_date]

    if end_date is not None:
        end_date = pd.to_datetime(end_date).date()
        df = df[df["trade_date"] <= end_date]

    if df.empty:
        return pd.DataFrame(), _summarize_trades(pd.DataFrame(), params)

    # Ensure sorted
    df = df.sort_values(["trade_date", "ts_utc"]).reset_index(drop=True)

    trades: List[Dict[str, Any]] = []

    # Loop by day
    for trade_date, day in df.groupby("trade_date", sort=True):
        day_trades = _run_day(day.copy(), trade_date, params)
        trades.extend(day_trades)

    trades_df = pd.DataFrame(trades)
    summary = _summarize_trades(trades_df, params)
    return trades_df, summary


def _run_day(day: pd.DataFrame, trade_date, params: GexFadeParams) -> List[Dict[str, Any]]:
    """
    Run a single trading day.
    """
    day = day.sort_values("ts_utc").reset_index(drop=True)
    if "bar_index" in day.columns:
        day["bar_index"] = day["bar_index"].astype("Int64")

    # RTH only
    day_rth = day[day["is_rth"].fillna(False)].copy()
    if day_rth.empty:
        return []

    # Build tests and candidates
    tests_df = _build_tests_for_day(day_rth, params)
    if tests_df.empty:
        return []

    candidates_by_ts = _compute_candidates_for_day(day_rth, tests_df, params)

    # Simulate trades
    trades = _simulate_day(day_rth, candidates_by_ts, params, trade_date)
    return trades


def _build_tests_for_day(day_rth: pd.DataFrame, params: GexFadeParams) -> pd.DataFrame:
    """
    Identify tests of the upper GEX wall for a single day.

    - Tests are clusters of bars that get *near* the wall
      (dist_to_wall_above_pts <= entry_proximity_max).
    - We DO NOT require an actual touch for a test to exist.
    - For each test we store:
        * start_ts_utc / start_bar_index  -> where we measure skew
        * touch_ts_utc / touch_bar_index  -> first true tag of the wall (if any),
                                             defined by dist_to_wall_above_pts <= touch_proximity_max
    - We also ignore the first `rth_skew_min_offset_bars` minutes of RTH
      when building tests (so the open noise isn't used as baseline).
    """
    day = day_rth.copy()

    # Determine the first RTH bar index and the minimum index allowed
    if "bar_index" in day.columns and not day["bar_index"].isna().all():
        rth_open_idx = int(day["bar_index"].min())
        skew_min_idx = rth_open_idx + params.rth_skew_min_offset_bars
    else:
        skew_min_idx = None

    # "Near wall" condition for clustering
    cond = (
        day["gex_wall_above"].notna()
        & day["gex_wall_above_gex"].abs().ge(params.gex_wall_min)
        & day["net_gex"].gt(params.gex_net_min)
        & day["dist_to_wall_above_pts"].between(0, params.entry_proximity_max)
    )

    if skew_min_idx is not None:
        cond = cond & day["bar_index"].ge(skew_min_idx)

    nw = day[cond].copy()
    if nw.empty:
        return pd.DataFrame(
            columns=[
                "wall_key",
                "test_seq",
                "start_bar_index",
                "end_bar_index",
                "start_ts_utc",
                "put_skew_start",
                "touch_ts_utc",
                "touch_bar_index",
            ]
        )

    # Group near-wall bars by rounded wall price
    nw["wall_key"] = np.round(nw["gex_wall_above"]).astype("float")

    tests: List[Dict[str, Any]] = []

    for wall_key, g in nw.groupby("wall_key"):
        g = g.sort_values("bar_index").reset_index(drop=False)
        g["bar_index"] = g["bar_index"].astype(int)

        # New test whenever gap in bar_index > min_test_gap_bars
        gaps = g["bar_index"].diff().fillna(params.min_test_gap_bars + 1)
        g["local_test_id"] = (gaps > params.min_test_gap_bars).cumsum()

        # Each (wall_key, local_test_id) is a test cluster
        for j, (_, gt) in enumerate(g.groupby("local_test_id"), start=1):
            # This cluster *is* a test even if it never truly touches the wall.
            # Test "start" = first near-wall bar in the cluster.
            start_row = gt.iloc[0]

            start_bar_index = int(start_row["bar_index"])
            start_ts_utc = start_row["ts_utc"]
            if pd.notna(start_row.get("put_skew_pp_primary", np.nan)):
                put_skew_start = float(start_row["put_skew_pp_primary"])
            else:
                put_skew_start = np.nan

            end_bar_index = int(gt["bar_index"].max())

            # Within this cluster, find the first true touch (if any)
            touch_mask = gt["dist_to_wall_above_pts"].le(params.touch_proximity_max)
            if touch_mask.any():
                touch_row = gt[touch_mask].iloc[0]
                touch_ts_utc = touch_row["ts_utc"]
                touch_bar_index = int(touch_row["bar_index"])
            else:
                touch_ts_utc = pd.NaT
                touch_bar_index = None

            tests.append(
                {
                    "wall_key": float(wall_key),
                    "test_seq": int(j),
                    "start_bar_index": start_bar_index,
                    "end_bar_index": end_bar_index,
                    "start_ts_utc": start_ts_utc,
                    "put_skew_start": put_skew_start,
                    "touch_ts_utc": touch_ts_utc,
                    "touch_bar_index": touch_bar_index,
                }
            )

    tests_df = pd.DataFrame(tests)
    return tests_df


def _compute_candidates_for_day(
    day_rth: pd.DataFrame,
    tests_df: pd.DataFrame,
    params: GexFadeParams,
) -> Dict[pd.Timestamp, Dict[str, Any]]:
    """
    Decide which tests become entry candidates.

    For each wall_key in this day:

    - Baseline = first test with |skew| >= min_baseline_skew.
    - For every *later* test:
        * compute skew change vs baseline,
        * if change_pct >= put_skew_increase_min AND test has a touch_ts_utc,
          create a short entry candidate at that touch time.

    So you get at most ONE candidate per test, and only for tests where:
      - skew is meaningfully more bid than baseline, and
      - price actually tagged the wall in that cluster.
    """
    if tests_df.empty:
        return {}

    # Use ts_utc as index for quick lookups
    day = day_rth.sort_values("ts_utc").copy()
    day["ts_utc"] = pd.to_datetime(day["ts_utc"])
    day = day.set_index("ts_utc", drop=False)

    candidates_by_ts: Dict[pd.Timestamp, Dict[str, Any]] = {}

    for wall_key, wtests in tests_df.groupby("wall_key"):
        # Ensure chronological order by test start time
        wtests = wtests.sort_values("start_ts_utc").reset_index(drop=True)

        # ---- Find baseline test ----
        baseline_idx = None
        for idx, row in wtests.iterrows():
            skew = row["put_skew_start"]
            if pd.notna(skew) and abs(float(skew)) >= params.min_baseline_skew:
                baseline_idx = idx
                break

        if baseline_idx is None:
            continue

        baseline_row = wtests.iloc[baseline_idx]
        base_skew = float(baseline_row["put_skew_start"])
        base_ts_utc = pd.to_datetime(baseline_row["start_ts_utc"])
        if base_ts_utc not in day.index:
            continue
        base_ts_pt = day.loc[base_ts_utc].get("ts_pt")

        # ---- For each later test, see if skew is sufficiently higher ----
        for idx in range(baseline_idx + 1, len(wtests)):
            test_row = wtests.iloc[idx]

            curr_skew = test_row["put_skew_start"]
            if pd.isna(curr_skew):
                continue
            curr_skew = float(curr_skew)

            change_pct = (curr_skew - base_skew) / abs(base_skew)
            if change_pct < params.put_skew_increase_min:
                continue

            # We need a true touch in this cluster to actually enter
            touch_ts_utc = test_row.get("touch_ts_utc")
            if pd.isna(touch_ts_utc):
                continue

            entry_ts = pd.to_datetime(touch_ts_utc)
            if entry_ts not in day.index:
                continue

            entry_row = day.loc[entry_ts]
            entry_ts_pt = entry_row.get("ts_pt")

            # Double-check regime and near-wall condition at the touch bar
            gex_wall_above = entry_row.get("gex_wall_above")
            gex_wall_above_gex = entry_row.get("gex_wall_above_gex", 0.0)
            net_gex = entry_row.get("net_gex", 0.0)
            dist_to_wall = entry_row.get("dist_to_wall_above_pts", np.inf)

            if (
                pd.isna(gex_wall_above)
                or round(float(gex_wall_above)) != round(float(wall_key))
                or abs(float(gex_wall_above_gex)) < params.gex_wall_min
                or float(net_gex) <= params.gex_net_min
                or not (0 <= float(dist_to_wall) <= params.entry_proximity_max)
            ):
                continue

            cand = {
                "wall_key": float(wall_key),
                "entry_ts": entry_ts,
                "entry_bar_index": int(entry_row["bar_index"]),
                "put_skew_base": base_skew,
                "put_skew_entry": curr_skew,
                "put_skew_change_pct": float(change_pct),
                "put_skew_base_ts": base_ts_utc,
                "put_skew_entry_ts": pd.to_datetime(test_row["start_ts_utc"]),
                "put_skew_base_ts_pt": base_ts_pt,
                "put_skew_entry_ts_pt": entry_ts_pt,  # PT for entry test (touch) time
            }

            # If multiple candidates share the same entry_ts (e.g. overlapping walls),
            # keep the one with the largest |skew change|.
            prev = candidates_by_ts.get(entry_ts)
            if prev is None or abs(cand["put_skew_change_pct"]) > abs(prev["put_skew_change_pct"]):
                candidates_by_ts[entry_ts] = cand

    return candidates_by_ts


def _simulate_day(
    day_rth: pd.DataFrame,
    candidates_by_ts: Dict[pd.Timestamp, Dict[str, Any]],
    params: GexFadeParams,
    trade_date,
) -> List[Dict[str, Any]]:
    """
    Given intraday bars and entry candidates, simulate trades for one day.
    """
    trades: List[Dict[str, Any]] = []

    day = day_rth.sort_values("ts_utc").reset_index(drop=True)
    close_cutoff = params.close_cutoff_bar_index or params.bar_index_max

    # Day's RTH range for "top of range" filtering
    day_high = float(day["high"].max())
    day_low = float(day["low"].min())
    day_range = max(day_high - day_low, 1e-6)  # avoid divide-by-zero

    position: Optional[Dict[str, Any]] = None
    trades_today = 0

    for _, row in day.iterrows():
        bar_index = int(row["bar_index"]) if not pd.isna(row["bar_index"]) else None
        if bar_index is None:
            continue

        ts = pd.to_datetime(row["ts_utc"])
        ts_pt = (
            pd.to_datetime(row["ts_pt"])
            if ("ts_pt" in row and not pd.isna(row["ts_pt"]))
            else None
        )
        high = float(row["high"])
        low = float(row["low"])
        close_price = float(row["close"])

        # --- Manage existing position first ---
        if position is not None:
            position["bars_in_trade"] += 1
            stop_price = position["stop_price"]
            target_price = position["target_price"]

            stop_hit = high >= stop_price
            target_hit = low <= target_price

            exit_trade = False
            exit_reason = None
            exit_price = None

            if stop_hit and target_hit:
                exit_price = stop_price
                exit_reason = "stop_and_target_same_bar"
                exit_trade = True
            elif stop_hit:
                exit_price = stop_price
                exit_reason = "stop"
                exit_trade = True
            elif target_hit:
                exit_price = target_price
                exit_reason = "target"
                exit_trade = True
            elif (
                position["bars_in_trade"] >= params.max_bars_in_trade
                or bar_index >= close_cutoff
            ):
                exit_price = close_price
                exit_reason = "time"
                exit_trade = True

            if exit_trade:
                pnl_points = position["entry_price"] - exit_price  # short
                trades.append(
                    {
                        "trade_date": trade_date,
                        "entry_ts": position["entry_ts"],
                        "entry_ts_pt": position.get("entry_ts_pt"),
                        "exit_ts": ts,
                        "exit_ts_pt": ts_pt,
                        "entry_bar_index": position["entry_bar_index"],
                        "exit_bar_index": bar_index,
                        "entry_price": position["entry_price"],
                        "exit_price": exit_price,
                        "pnl_points": pnl_points,
                        "stop_price": position["stop_price"],
                        "target_price": position["target_price"],
                        "exit_reason": exit_reason,
                        "wall_key": position["wall_key"],
                        "gex_wall_above": position["gex_wall_above"],
                        "gex_wall_above_gex": position["gex_wall_above_gex"],
                        "put_skew_base": position["put_skew_base"],
                        "put_skew_entry": position["put_skew_entry"],
                        "put_skew_change_pct": position["put_skew_change_pct"],
                        "put_skew_base_ts": position.get("put_skew_base_ts"),
                        "put_skew_entry_ts": position.get("put_skew_entry_ts"),
                        "put_skew_base_ts_pt": position.get("put_skew_base_ts_pt"),
                        "put_skew_entry_ts_pt": position.get("put_skew_entry_ts_pt"),
                        "smile_dte_primary": position.get("smile_dte_primary"),
                    }
                )
                position = None

        # --- Consider new entry if flat ---
        if position is None:
            if trades_today >= params.max_trades_per_day:
                continue
            if not (params.bar_index_min <= bar_index <= params.bar_index_max):
                continue

            cand = candidates_by_ts.get(ts)
            if cand is None:
                continue

            # Double-check wall + regime at the actual bar
            gex_wall_above = row.get("gex_wall_above")
            gex_wall_above_gex = row.get("gex_wall_above_gex", 0.0)
            net_gex = row.get("net_gex", 0.0)
            dist_to_wall = row.get("dist_to_wall_above_pts", np.inf)

            if (
                pd.isna(gex_wall_above)
                or abs(float(gex_wall_above_gex)) < params.gex_wall_min
                or float(net_gex) <= params.gex_net_min
                or not (0 <= float(dist_to_wall) <= params.entry_proximity_max)
            ):
                continue

            # Entry price = bar close
            entry_price = close_price

            # Only short near the top of the day's range, if requested
            entry_frac = (entry_price - day_low) / day_range
            if params.entry_min_range_frac > 0.0 and entry_frac < params.entry_min_range_frac:
                continue

            # --- Risk & target ---
            stop_price = entry_price + params.stop_loss_points
            target_price = entry_price - params.r_mult * params.stop_loss_points

            # Optional: override target with lower GEX wall if it's big enough
            if params.use_lower_gex_target:
                wall_below = row.get("gex_wall_below")
                wall_below_gex = row.get("gex_wall_below_gex", 0.0)

                if (
                    wall_below is not None
                    and not pd.isna(wall_below)
                    and float(wall_below) < entry_price  # below us
                    and abs(float(wall_below_gex)) >= params.lower_gex_min_abs
                ):
                    target_price = float(wall_below)

            position = {
                "entry_ts": ts,
                "entry_ts_pt": ts_pt,
                "entry_bar_index": bar_index,
                "entry_price": entry_price,
                "stop_price": stop_price,
                "target_price": target_price,
                "bars_in_trade": 0,
                "trade_date": trade_date,
                "wall_key": cand["wall_key"],
                "gex_wall_above": float(gex_wall_above),
                "gex_wall_above_gex": float(gex_wall_above_gex),
                "put_skew_base": cand["put_skew_base"],
                "put_skew_entry": cand["put_skew_entry"],
                "put_skew_change_pct": cand["put_skew_change_pct"],
                "put_skew_base_ts": cand.get("put_skew_base_ts"),
                "put_skew_entry_ts": cand.get("put_skew_entry_ts"),
                "put_skew_base_ts_pt": cand.get("put_skew_base_ts_pt"),
                "put_skew_entry_ts_pt": cand.get("put_skew_entry_ts_pt"),
                "smile_dte_primary": row.get("smile_dte_primary"),
            }
            trades_today += 1

    return trades


def _summarize_trades(trades_df: pd.DataFrame, params: GexFadeParams) -> Dict[str, Any]:
    """
    Basic summary stats for the trade list.
    """
    summary: Dict[str, Any] = {
        "total_trades": 0,
        "win_trades": 0,
        "loss_trades": 0,
        "flat_trades": 0,
        "win_rate": None,
        "avg_pnl": None,
        "avg_win": None,
        "avg_loss": None,
        "expectancy": None,
        "params": asdict(params),
    }

    if trades_df is None or trades_df.empty:
        return summary

    summary["total_trades"] = int(len(trades_df))

    wins = trades_df["pnl_points"] > 0
    losses = trades_df["pnl_points"] < 0
    flats = trades_df["pnl_points"] == 0

    summary["win_trades"] = int(wins.sum())
    summary["loss_trades"] = int(losses.sum())
    summary["flat_trades"] = int(flats.sum())

    if summary["total_trades"] > 0:
        summary["win_rate"] = float(summary["win_trades"]) / summary["total_trades"]
        summary["avg_pnl"] = float(trades_df["pnl_points"].mean())

        if wins.any():
            summary["avg_win"] = float(trades_df.loc[wins, "pnl_points"].mean())
        if losses.any():
            summary["avg_loss"] = float(trades_df.loc[losses, "pnl_points"].mean())

        summary["expectancy"] = summary["avg_pnl"]

    return summary
