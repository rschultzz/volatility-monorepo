from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Any, Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

from packages.shared.surface_compare import k_for_abs_delta


# ---- Expected (SS) constants (match Skew module) ----
EPS_T = 1e-4

# keep skew % deltas sane when expected/previous ≈ 0 pp
MIN_SKEW_DENOM_PP = 0.25

# leverage add-on (must match Smile expected overlay)
BETA_VOLPTS_PER_1PCT = 4.5
BETA_MAX_SHIFT_PP = 6.0


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
    anchor_in_top_range_pct: float = 30.0 # anchor must be in top X% of day's range

    # Skew filter
    min_abs_skew: float = 0.0         # minimum |put_skew_pp_primary| at entry

    # Multi-test + skew-increase logic
    min_minutes_between_tests: int = 30
    min_put_skew_increase_frac: float = 0.50  # 0.50 => 50% increase in skew

    # NEW: optional reset requirement
    require_reset_between_tests: bool = False
    reset_buffer_points: float = 2.0  # extra distance beyond proximity zone to count as "reset"

    # NEW: compare confirm skew vs SS-expected baseline (synced to dashboard toggle)
    compare_put_skew_to_expected_ss: bool = False

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


def _pct_change_pp(curr_pp: float, base_pp: float) -> float:
    """
    Matches logic in apps/web/modules/Skew/callbacks.py:
    (curr - base) / max(abs(base), 0.25) * 100.0
    """
    denom = max(abs(base_pp), MIN_SKEW_DENOM_PP)
    return (curr_pp - base_pp) / denom * 100.0


def _to_utc_datetime(ts: Any) -> Optional[dt.datetime]:
    if ts is None or pd.isna(ts):
        return None
    ts2 = pd.to_datetime(ts, utc=True, errors="coerce")
    if ts2 is None or pd.isna(ts2):
        return None
    # pandas Timestamp -> python datetime
    return ts2.to_pydatetime()


def _exp_iso(expiration: Any) -> Optional[str]:
    if expiration is None or pd.isna(expiration):
        return None
    try:
        d = pd.to_datetime(expiration).date()
        return d.isoformat()
    except Exception:
        try:
            return str(expiration)
        except Exception:
            return None


def _years_to_exp(ts_utc: dt.datetime, expiration_iso: str) -> float:
    """
    Time to expiry in years using a UTC timestamp and 16:00 UTC expiry.
    (Mirrors Skew module helper.)
    """
    exp_date = dt.date.fromisoformat(expiration_iso)
    exp_dt_utc = dt.datetime.combine(exp_date, dt.time(16, 0), tzinfo=dt.timezone.utc)
    rem = exp_dt_utc - ts_utc
    T = max(0.0, rem.total_seconds() / (365.0 * 24 * 3600))
    return max(T, EPS_T)


def _available_buckets(row: pd.Series) -> List[int]:
    buckets: List[int] = []
    for c in row.index:
        if c.startswith("vol") and c[3:].isdigit():
            n = int(c[3:])
            if 1 <= n <= 99:
                buckets.append(n)
    # puts (>=50) then calls (<50), both descending, no duplicates
    puts = sorted([n for n in buckets if n >= 50], reverse=True)
    calls = sorted([n for n in buckets if n < 50], reverse=True)
    out: List[int] = []
    seen = set()
    for n in puts + calls:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def _abs_delta_is_put(bucket: int) -> Tuple[float, bool]:
    if bucket == 50:
        return 0.50, False
    if bucket > 50:
        return (100 - bucket) / 100.0, True
    return bucket / 100.0, False


def _interp_linear_extrap(kq: float, k_grid: np.ndarray, s_grid: np.ndarray) -> float:
    if kq <= k_grid[0]:
        x0, x1, y0, y1 = k_grid[0], k_grid[1], s_grid[0], s_grid[1]
        return float(y0 + (y1 - y0) * (kq - x0) / (x1 - x0))
    if kq >= k_grid[-1]:
        x0, x1, y0, y1 = k_grid[-2], k_grid[-1], s_grid[-2], s_grid[-1]
        return float(y1 + (y1 - y0) * (kq - x1) / (x1 - x0))
    return float(np.interp(kq, k_grid, s_grid))


def _prev_smile_interp(prev_row: pd.Series, T_prev: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build previous surface’s (k_prev, sigma_prev) grid using its ATM for Δ→k.
    Mirrors Skew module behavior.
    """
    if "vol50" not in prev_row or pd.isna(prev_row["vol50"]):
        raise ValueError("prev row missing ATM")
    atm_prev = float(prev_row["vol50"])

    buckets_prev = _available_buckets(prev_row)
    if len(buckets_prev) < 4:
        raise ValueError("prev row has too few buckets")

    k_prev: List[float] = []
    s_prev: List[float] = []
    for n in buckets_prev:
        col = f"vol{n}"
        v = prev_row.get(col)
        if v is None or pd.isna(v):
            continue

        if n == 50:
            k = 0.0
        else:
            p, is_put = _abs_delta_is_put(n)
            k = k_for_abs_delta(p, is_put=is_put, sigma=atm_prev, T=T_prev)

        k_prev.append(float(k))
        s_prev.append(float(v))

    k_np = np.array(k_prev, float)
    s_np = np.array(s_prev, float)

    if k_np.size < 3:
        raise ValueError("prev k-grid too small")

    order = np.argsort(k_np)
    k_np, s_np = k_np[order], s_np[order]

    # de-dupe
    mask = np.concatenate(([True], np.diff(k_np) > 1e-12))
    k_np, s_np = k_np[mask], s_np[mask]
    if k_np.size < 3:
        raise ValueError("prev k-grid degenerate")

    return k_np, s_np


def _expected_put_skew_pp_from_prev(
    prev_row: pd.Series,
    prev_ts_utc: dt.datetime,
    prev_stock: float,
    now_ts_utc: dt.datetime,
    now_stock: float,
    expiration_iso: str,
    atm_now: float,
) -> Tuple[float, Optional[float], Optional[int], Optional[float]]:
    """
    Returns:
      exp_put_skew_pp, atm_exp_pct, atm_residual_bp, k_shift
    """
    T_prev = _years_to_exp(prev_ts_utc, expiration_iso)
    T_now = _years_to_exp(now_ts_utc, expiration_iso)

    k_prev, s_prev = _prev_smile_interp(prev_row, T_prev)

    k_shift = math.log(now_stock / prev_stock)
    exp_atm_shape = _interp_linear_extrap(k_shift, k_prev, s_prev)

    ret_frac = (now_stock - prev_stock) / prev_stock
    level_shift_pp = max(
        -BETA_MAX_SHIFT_PP,
        min(BETA_MAX_SHIFT_PP, (-ret_frac) * 100.0 * BETA_VOLPTS_PER_1PCT),
    )
    atm_exp = exp_atm_shape + (level_shift_pp / 100.0)
    atm_exp_pct = round(atm_exp * 100.0, 2)
    atm_residual_bp = int(round((atm_now - atm_exp) * 10000.0))

    k_p25_now = k_for_abs_delta(0.25, is_put=True, sigma=atm_now, T=T_now)
    exp_p25_shape = _interp_linear_extrap(k_p25_now + k_shift, k_prev, s_prev)

    # vertical shift so curve ATM == atm_exp
    shift_frac = atm_exp - exp_atm_shape
    exp_p25 = exp_p25_shape + shift_frac

    exp_put_skew_pp = (exp_p25 - atm_exp) * 100.0
    return float(exp_put_skew_pp), float(atm_exp_pct), int(atm_residual_bp), float(k_shift)


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

    If params.compare_put_skew_to_expected_ss is True, also expects:
    - stock_price
    - vol50 (and volNN buckets used for prev surface grid)
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

    day_high = -np.inf
    day_low = np.inf

    # Track anchor tests per wall within the day
    # key: rounded wall level, value: anchor info (plus reset_seen)
    anchors_by_wall: Dict[float, Dict[str, Any]] = {}

    stop = params.stop_loss_points
    target_points = params.stop_loss_points * params.target_rr

    # Precompute reset threshold distance beyond the test zone
    reset_trigger_dist = float(params.entry_proximity_max) + float(params.reset_buffer_points)

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
                        "put_skew_change_pct": round(float(position.get("put_skew_change_pct", np.nan)), 1),
                        "minutes_between_tests": position.get("minutes_between_tests", np.nan),
                        "reset_required": bool(position.get("reset_required", False)),
                        "reset_seen": bool(position.get("reset_seen", False)),
                        # NEW: SS compare debug
                        "compare_to_expected_ss": bool(position.get("compare_to_expected_ss", False)),
                        "expected_put_skew_pp": position.get("expected_put_skew_pp", np.nan),
                        "atm_exp_ss_pct": position.get("atm_exp_ss_pct", np.nan),
                        "atm_residual_bp": position.get("atm_residual_bp", np.nan),
                        "k_shift": position.get("k_shift", np.nan),
                    }
                )
                position = None

            current_trade_date = trade_date
            trades_today = 0
            anchors_by_wall = {}
            day_high = -np.inf
            day_low = np.inf

        # Update daily range
        day_high = max(day_high, row["high"])
        day_low = min(day_low, row["low"])

        # Update trailing info even if we don't trade at this bar
        last_close = float(row["close"])
        last_ts_utc = row.get("ts_utc")
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
                        "put_skew_change_pct": round(float(position.get("put_skew_change_pct", np.nan)), 1),
                        "minutes_between_tests": position.get("minutes_between_tests", np.nan),
                        "reset_required": bool(position.get("reset_required", False)),
                        "reset_seen": bool(position.get("reset_seen", False)),
                        # NEW: SS compare debug
                        "compare_to_expected_ss": bool(position.get("compare_to_expected_ss", False)),
                        "expected_put_skew_pp": position.get("expected_put_skew_pp", np.nan),
                        "atm_exp_ss_pct": position.get("atm_exp_ss_pct", np.nan),
                        "atm_residual_bp": position.get("atm_residual_bp", np.nan),
                        "k_shift": position.get("k_shift", np.nan),
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

        wall_level = float(gex_wall_above)
        wall_key = round(wall_level, 2)

        # If we already have an anchor, update reset_seen whenever price pulls back far enough
        if wall_key in anchors_by_wall:
            if float(dist_to_wall) >= reset_trigger_dist:
                anchors_by_wall[wall_key]["reset_seen"] = True

        # We want price just below the wall (distance >= 0 but not too big)
        if not (0.0 <= float(dist_to_wall) <= params.entry_proximity_max):
            continue

        # Skew filter — this is now *required*, so trades can’t fire without skew
        put_skew = row.get("put_skew_pp_primary")
        if pd.isna(put_skew):
            continue

        if abs(float(put_skew)) < params.min_abs_skew:
            continue

        # If no anchor yet, record anchor and move on
        if wall_key not in anchors_by_wall:
            # NEW: Check if anchor is in top X% of day's range
            day_range = day_high - day_low
            if day_range > 0:
                price_pct_of_range = (row["close"] - day_low) / day_range
                required_pct = (100.0 - params.anchor_in_top_range_pct) / 100.0
                if price_pct_of_range < required_pct:
                    continue

            # Store full prev surface vols + stock for expected SS
            prev_surface = {
                c: float(row[c])
                for c in row.index
                if c.startswith("vol") and c[3:].isdigit() and pd.notna(row[c])
            }
            prev_stock = row.get("stock_price")
            prev_stock_f = float(prev_stock) if (prev_stock is not None and not pd.isna(prev_stock)) else np.nan

            anchors_by_wall[wall_key] = {
                "ts_utc": row.get("ts_utc"),
                "ts_pt": row.get("ts_pt"),
                "bar_index": bar_index,
                "put_skew_pp": float(put_skew),
                "reset_seen": False,
                "prev_surface": prev_surface,
                "prev_stock": prev_stock_f,
                "smile_expir_primary": row.get("smile_expir_primary"),
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

        anchor_skew = float(anchor.get("put_skew_pp", np.nan))
        confirm_skew = float(put_skew)

        # Default baseline: anchor actual skew
        baseline_pp = anchor_skew
        expected_put_skew_pp = np.nan
        atm_exp_ss_pct = np.nan
        atm_residual_bp = np.nan
        k_shift_val = np.nan
        used_expected_ss = False

        # If enabled, use SS-expected baseline derived from anchor surface
        if params.compare_put_skew_to_expected_ss:
            try:
                prev_ts = _to_utc_datetime(anchor.get("ts_utc"))
                now_ts = _to_utc_datetime(row.get("ts_utc"))
                exp_iso = _exp_iso(anchor.get("smile_expir_primary") or row.get("smile_expir_primary"))
                prev_stock = float(anchor.get("prev_stock", np.nan))
                now_stock = row.get("stock_price")
                now_stock = float(now_stock) if (now_stock is not None and not pd.isna(now_stock)) else np.nan

                # current ATM vol (prefer vol50 raw bucket)
                atm_now_val = row.get("vol50")
                if atm_now_val is None or pd.isna(atm_now_val):
                    atm_now_val = row.get("smile_vol50_primary")
                atm_now = float(atm_now_val) if (atm_now_val is not None and not pd.isna(atm_now_val)) else np.nan

                prev_surface = anchor.get("prev_surface", {})
                prev_row = pd.Series(prev_surface)

                if (
                    prev_ts is not None
                    and now_ts is not None
                    and exp_iso
                    and np.isfinite(prev_stock)
                    and np.isfinite(now_stock)
                    and np.isfinite(atm_now)
                    and (prev_surface is not None and len(prev_surface) >= 4)
                ):
                    exp_put_skew_pp, atm_exp_ss_pct, atm_residual_bp, k_shift_val = _expected_put_skew_pp_from_prev(
                        prev_row=prev_row,
                        prev_ts_utc=prev_ts,
                        prev_stock=prev_stock,
                        now_ts_utc=now_ts,
                        now_stock=now_stock,
                        expiration_iso=exp_iso,
                        atm_now=atm_now,
                    )
                    expected_put_skew_pp = exp_put_skew_pp
                    baseline_pp = exp_put_skew_pp
                    used_expected_ss = True
            except Exception:
                # Fallback to anchor baseline if expected calc fails
                pass

        # Calculate % change exactly like Skew/callbacks.py
        pct_change = _pct_change_pp(confirm_skew, baseline_pp)

        # We want an INCREASE of at least X%.
        required_increase_pct = float(params.min_put_skew_increase_frac) * 100.0
        if pct_change < required_increase_pct:
            continue

        # All conditions satisfied: open a new short
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
            "anchor_put_skew_pp": anchor_skew,
            "confirm_put_skew_pp": confirm_skew,
            "put_skew_change_pct": pct_change,
            "minutes_between_tests": int(round(minutes_between)),
            "reset_required": bool(params.require_reset_between_tests),
            "reset_seen": bool(anchor.get("reset_seen", False)),
            # NEW: SS compare debug
            "compare_to_expected_ss": used_expected_ss,
            "expected_put_skew_pp": expected_put_skew_pp,
            "atm_exp_ss_pct": atm_exp_ss_pct,
            "atm_residual_bp": atm_residual_bp,
            "k_shift": k_shift_val,
        }

        # After taking a trade, clear anchors so we don't reuse them immediately
        anchors_by_wall = {}

    # If we end the loop with an open position, close it at the last seen price
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
                "put_skew_change_pct": round(float(position.get("put_skew_change_pct", np.nan)), 1),
                "minutes_between_tests": position.get("minutes_between_tests", np.nan),
                "reset_required": bool(position.get("reset_required", False)),
                "reset_seen": bool(position.get("reset_seen", False)),
                # NEW: SS compare debug
                "compare_to_expected_ss": bool(position.get("compare_to_expected_ss", False)),
                "expected_put_skew_pp": position.get("expected_put_skew_pp", np.nan),
                "atm_exp_ss_pct": position.get("atm_exp_ss_pct", np.nan),
                "atm_residual_bp": position.get("atm_residual_bp", np.nan),
                "k_shift": position.get("k_shift", np.nan),
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
