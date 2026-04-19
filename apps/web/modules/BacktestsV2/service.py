from __future__ import annotations

import os
import math
import datetime as dt
from functools import lru_cache
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import pytz
from sqlalchemy import create_engine, text

from packages.shared.utils import fetch_skew_data
from packages.shared.surface_compare import k_for_abs_delta

DEFAULT_SOURCE_VIEW = os.getenv("BT2_SOURCE_VIEW", os.getenv("BT_VIEW_NAME", "es_minutes_with_features_bt"))

MARKET_TIMEZONE = pytz.timezone("US/Eastern")
EPS_T = 1e-4
MIN_SKEW_DENOM_PP = 0.25
BETA_VOLPTS_PER_1PCT = 4.5
BETA_MAX_SHIFT_PP = 6.0
THETA_ATM_PP_PER_SQRT_YEAR = -638

MIN_TARGET_ACCEPTANCE_BARS = 3
MAX_BARS_OUTSIDE_DURING_ACCEPTANCE = 1


@lru_cache(maxsize=1)
def get_engine():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set")
    return create_engine(db_url, pool_pre_ping=True)


def safe_ident(name: str) -> str:
    if not name or not all(ch.isalnum() or ch == "_" for ch in name):
        raise ValueError(f"Unsafe identifier: {name!r}")
    return name


def parse_date(value: Any) -> dt.date:
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value))


def gex_to_bn(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except Exception:
        return None
    if math.isnan(v):
        return None
    if abs(v) >= 1_000_000:
        return v / 1_000_000_000.0
    return v


def load_source_rows(start_date: str, end_date: str, source_view: str | None = None) -> pd.DataFrame:
    source_view = safe_ident(source_view or DEFAULT_SOURCE_VIEW)
    start_d = parse_date(start_date)
    end_d = parse_date(end_date)

    sql = text(
        f"""
        SELECT
            trade_date,
            ts_pt,
            ts_utc,
            is_rth,
            open,
            high,
            low,
            close,
            gex_wall_above,
            gex_wall_above_gex,
            gex_wall_below,
            gex_wall_below_gex,
            gex_strong_wall_above,
            gex_strong_wall_above_gex,
            gex_strong_wall_below,
            gex_strong_wall_below_gex
        FROM public.{source_view}
        WHERE trade_date >= :start_date
          AND trade_date <= :end_date
          AND is_rth = TRUE
        ORDER BY trade_date, ts_utc
        """
    )

    with get_engine().connect() as conn:
        df = pd.read_sql(sql, conn, params={"start_date": start_d, "end_date": end_d})

    if df.empty:
        return df

    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
    return df.sort_values(["trade_date", "ts_utc"]).reset_index(drop=True)


LEVEL_DEFS: List[Tuple[str, str, str, str]] = [
    ("primary", "wall_above", "gex_wall_above", "gex_wall_above_gex"),
    ("primary", "wall_below", "gex_wall_below", "gex_wall_below_gex"),
    ("strong", "strong_wall_above", "gex_strong_wall_above", "gex_strong_wall_above_gex"),
    ("strong", "strong_wall_below", "gex_strong_wall_below", "gex_strong_wall_below_gex"),
]


def _selected_level_defs(level_family: str) -> List[Tuple[str, str, str, str]]:
    if level_family == "primary":
        return [x for x in LEVEL_DEFS if x[0] == "primary"]
    if level_family == "strong":
        return [x for x in LEVEL_DEFS if x[0] == "strong"]
    return LEVEL_DEFS[:]


def _row_levels(row: Dict[str, Any], level_family: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for family, label, level_col, gex_col in _selected_level_defs(level_family):
        level = row.get(level_col)
        gex_bn = gex_to_bn(row.get(gex_col))
        if level is None or gex_bn is None:
            continue
        try:
            level_v = float(level)
        except Exception:
            continue
        out.append(
            {
                "family": family,
                "label": label,
                "level": level_v,
                "gex_bn": float(gex_bn),
                "level_col": level_col,
                "gex_col": gex_col,
            }
        )
    return out


def _qualifying_levels(row: Dict[str, Any], level_family: str, min_level_gex_bn: float) -> List[Dict[str, Any]]:
    return [item for item in _row_levels(row, level_family) if abs(item["gex_bn"]) >= float(min_level_gex_bn)]


def _collect_day_levels(day_rows: List[Dict[str, Any]], level_family: str, min_level_gex_bn: float) -> List[Dict[str, Any]]:
    by_level: Dict[float, Dict[str, Any]] = {}

    for row in day_rows:
        for item in _qualifying_levels(row, level_family, min_level_gex_bn):
            key = round(float(item["level"]), 4)
            existing = by_level.get(key)
            if existing is None:
                by_level[key] = {
                    "level": float(item["level"]),
                    "max_abs_gex_bn": abs(float(item["gex_bn"])),
                    "families": {str(item["family"])},
                    "labels": {str(item["label"])},
                }
            else:
                existing["max_abs_gex_bn"] = max(existing["max_abs_gex_bn"], abs(float(item["gex_bn"])))
                existing["families"].add(str(item["family"]))
                existing["labels"].add(str(item["label"]))

    out = list(by_level.values())
    out.sort(key=lambda x: x["level"])
    for item in out:
        item["families"] = sorted(item["families"])
        item["labels"] = sorted(item["labels"])
    return out


def _make_zone(items: List[Dict[str, Any]], zone_id: int) -> Dict[str, Any]:
    sorted_items = sorted(items, key=lambda x: float(x["level"]))
    low = float(sorted_items[0]["level"])
    high = float(sorted_items[-1]["level"])
    levels = [float(x["level"]) for x in sorted_items]
    return {
        "zone_id": int(zone_id),
        "low": low,
        "high": high,
        "width": round(high - low, 2),
        "count": len(sorted_items),
        "levels": levels,
        "levels_text": ", ".join(f"{x:.0f}" if float(x).is_integer() else f"{x:.2f}" for x in levels),
        "max_abs_gex_bn": round(max(float(x["max_abs_gex_bn"]) for x in sorted_items), 2),
        "items": sorted_items,
    }


def _build_zones(levels: List[Dict[str, Any]], zone_merge_distance_pts: float) -> List[Dict[str, Any]]:
    if not levels:
        return []

    zones: List[Dict[str, Any]] = []
    current_items: List[Dict[str, Any]] = [levels[0]]

    for item in levels[1:]:
        prev = current_items[-1]
        if float(item["level"]) - float(prev["level"]) <= float(zone_merge_distance_pts):
            current_items.append(item)
        else:
            zones.append(_make_zone(current_items, len(zones)))
            current_items = [item]

    zones.append(_make_zone(current_items, len(zones)))
    return zones


def _in_zone_close(row: Dict[str, Any], zone: Dict[str, Any], max_zone_breach_pts: float) -> bool:
    close_v = row.get("close")
    if close_v is None:
        return False
    try:
        close_f = float(close_v)
    except Exception:
        return False
    return (float(zone["low"]) - float(max_zone_breach_pts)) <= close_f <= (float(zone["high"]) + float(max_zone_breach_pts))


def _true_segments(mask: List[bool]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    start_idx: int | None = None
    for i, flag in enumerate(mask):
        if flag and start_idx is None:
            start_idx = i
        elif (not flag) and start_idx is not None:
            out.append((start_idx, i - 1))
            start_idx = None
    if start_idx is not None:
        out.append((start_idx, len(mask) - 1))
    return out


def _find_last_pivot(day_rows: List[Dict[str, Any]], start_idx: int, end_idx: int, direction: str, pivot_strength_bars: int) -> Tuple[int, float]:
    look = max(1, int(pivot_strength_bars))

    if direction == "up":
        candidate_idx: int | None = None
        for i in range(end_idx, start_idx - 1, -1):
            left = max(start_idx, i - look)
            right = min(end_idx, i + look)
            center_v = float(day_rows[i]["low"])
            window_vals = [float(day_rows[k]["low"]) for k in range(left, right + 1)]
            if center_v <= min(window_vals):
                candidate_idx = i
                break
        if candidate_idx is None:
            min_v = min(float(day_rows[i]["low"]) for i in range(start_idx, end_idx + 1))
            for i in range(end_idx, start_idx - 1, -1):
                if float(day_rows[i]["low"]) == min_v:
                    candidate_idx = i
                    break
        assert candidate_idx is not None
        return candidate_idx, float(day_rows[candidate_idx]["low"])

    candidate_idx = None
    for i in range(end_idx, start_idx - 1, -1):
        left = max(start_idx, i - look)
        right = min(end_idx, i + look)
        center_v = float(day_rows[i]["high"])
        window_vals = [float(day_rows[k]["high"]) for k in range(left, right + 1)]
        if center_v >= max(window_vals):
            candidate_idx = i
            break
    if candidate_idx is None:
        max_v = max(float(day_rows[i]["high"]) for i in range(start_idx, end_idx + 1))
        for i in range(end_idx, start_idx - 1, -1):
            if float(day_rows[i]["high"]) == max_v:
                candidate_idx = i
                break
    assert candidate_idx is not None
    return candidate_idx, float(day_rows[candidate_idx]["high"])


def _first_target_hit(
    day_rows: List[Dict[str, Any]],
    *,
    start_scan_idx: int,
    direction: str,
    zone: Dict[str, Any],
    target_level: float,
    target_proximity_pts: float,
    max_zone_breach_pts: float,
) -> int | None:
    for j in range(start_scan_idx, len(day_rows)):
        row = day_rows[j]

        if _in_zone_close(row, zone, max_zone_breach_pts):
            return None

        try:
            high_v = float(row["high"])
            low_v = float(row["low"])
        except Exception:
            return None

        if direction == "up":
            if low_v < float(zone["low"]) - float(max_zone_breach_pts):
                return None
            if high_v >= float(target_level) - float(target_proximity_pts):
                return j
        else:
            if high_v > float(zone["high"]) + float(max_zone_breach_pts):
                return None
            if low_v <= float(target_level) + float(target_proximity_pts):
                return j

    return None


def _normalize_pt_label(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass

    if hasattr(value, "strftime"):
        try:
            return value.strftime("%H:%M")
        except Exception:
            pass

    s = str(value).strip()
    if not s:
        return ""
    if len(s) >= 5 and s[2] == ":":
        return s[:5]

    try:
        return pd.to_datetime(s).strftime("%H:%M")
    except Exception:
        return s


def _skews_from_row(row: pd.Series) -> Tuple[float, float, float]:
    atm = float(pd.to_numeric(row.get("vol50"), errors="coerce"))
    c25 = float(pd.to_numeric(row.get("vol25"), errors="coerce"))
    p25 = float(pd.to_numeric(row.get("vol75"), errors="coerce"))
    return atm, (c25 - atm) * 100.0, (p25 - atm) * 100.0


def _pct_change_frac(curr: Optional[float], base: Optional[float]) -> Optional[float]:
    if base in (None, 0) or curr is None:
        return None
    return (curr - base) / abs(base) * 100.0


def _pct_change_pp(curr_pp: Optional[float], base_pp: Optional[float]) -> Optional[float]:
    if curr_pp is None or base_pp is None:
        return None
    denom = max(abs(base_pp), MIN_SKEW_DENOM_PP)
    return (curr_pp - base_pp) / denom * 100.0


def _years_to_exp(ts_et: dt.datetime, expiration_iso: str) -> float:
    if ts_et.tzinfo is None:
        ts_et = MARKET_TIMEZONE.localize(ts_et)
    else:
        ts_et = ts_et.astimezone(MARKET_TIMEZONE)

    exp_date = dt.date.fromisoformat(expiration_iso)
    exp_dt_et = MARKET_TIMEZONE.localize(dt.datetime.combine(exp_date, dt.time(16, 0)))
    rem = exp_dt_et - ts_et
    T = max(0.0, rem.total_seconds() / (365.0 * 24 * 3600))
    return max(T, EPS_T)


def _T_from_row_snapshot(row: pd.Series, expiration_iso: str) -> Optional[float]:
    ts_utc_val = row.get("snap_shot_date")
    if ts_utc_val is None or pd.isna(ts_utc_val):
        return None
    ts_et = pd.to_datetime(ts_utc_val, utc=True).tz_convert(MARKET_TIMEZONE).to_pydatetime()
    return _years_to_exp(ts_et, expiration_iso)


def _available_buckets(row: pd.Series) -> List[int]:
    buckets: List[int] = []
    for c in row.index:
        if c.startswith("vol") and c[3:].isdigit():
            n = int(c[3:])
            if 1 <= n <= 99:
                buckets.append(n)
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


def _prev_smile_interp(prev_row: pd.Series, T_prev: float):
    if "vol50" not in prev_row:
        raise ValueError("prev row missing ATM")

    atm_prev = float(prev_row["vol50"])
    buckets_prev = _available_buckets(prev_row)
    if len(buckets_prev) < 4:
        raise ValueError("prev row has too few buckets")

    k_prev: List[float] = []
    s_prev: List[float] = []
    for n in buckets_prev:
        if n == 50:
            k = 0.0
        else:
            p, is_put = _abs_delta_is_put(n)
            k = k_for_abs_delta(p, is_put=is_put, sigma=atm_prev, T=T_prev)
        k_prev.append(k)
        s_prev.append(float(prev_row[f"vol{n}"]))

    k_np = np.array(k_prev, float)
    s_np = np.array(s_prev, float)
    mask = np.concatenate(([True], np.diff(k_np) > 1e-12))
    k_np, s_np = k_np[mask], s_np[mask]
    if k_np.size < 3:
        raise ValueError("prev k-grid degenerate")
    return k_np, s_np


def _interp_linear_extrap(kq: float, k_grid: np.ndarray, s_grid: np.ndarray) -> float:
    if kq <= k_grid[0]:
        x0, x1, y0, y1 = k_grid[0], k_grid[1], s_grid[0], s_grid[1]
        return float(y0 + (y1 - y0) * (kq - x0) / (x1 - x0))
    if kq >= k_grid[-1]:
        x0, x1, y0, y1 = k_grid[-2], k_grid[-1], s_grid[-2], s_grid[-1]
        return float(y1 + (y1 - y0) * (kq - x1) / (x1 - x0))
    return float(np.interp(kq, k_grid, s_grid))


def _expected_skew_deltas_from_entry(entry_row: pd.Series, curr_row: pd.Series, expiration_iso: str) -> Dict[str, Optional[float]]:
    entry_stock_val = entry_row.get("stock_price")
    curr_stock_val = curr_row.get("stock_price")

    entry_stock = float(entry_stock_val) if entry_stock_val is not None and not pd.isna(entry_stock_val) else None
    curr_stock = float(curr_stock_val) if curr_stock_val is not None and not pd.isna(curr_stock_val) else None

    if entry_stock is None or curr_stock is None:
        return {"delta_atm_iv_pct": None, "delta_call_skew_pct": None, "delta_put_skew_pct": None}

    entry_T = _T_from_row_snapshot(entry_row, expiration_iso)
    curr_T = _T_from_row_snapshot(curr_row, expiration_iso)
    if entry_T is None or curr_T is None:
        return {"delta_atm_iv_pct": None, "delta_call_skew_pct": None, "delta_put_skew_pct": None}

    atm_now, call_skew_pp_now, put_skew_pp_now = _skews_from_row(curr_row)

    try:
        k_prev, s_prev = _prev_smile_interp(entry_row, entry_T)
        k_shift = math.log(curr_stock / entry_stock) if entry_stock and curr_stock else 0.0

        exp_atm_shape = _interp_linear_extrap(k_shift, k_prev, s_prev)

        ret_frac = (curr_stock - entry_stock) / entry_stock
        level_shift_pp = max(
            -BETA_MAX_SHIFT_PP,
            min(BETA_MAX_SHIFT_PP, (-ret_frac) * 100.0 * BETA_VOLPTS_PER_1PCT),
        )

        droot = max(0.0, math.sqrt(max(entry_T, EPS_T)) - math.sqrt(max(curr_T, EPS_T)))
        atm_theta_pp = THETA_ATM_PP_PER_SQRT_YEAR * droot

        atm_exp = exp_atm_shape + (level_shift_pp / 100.0) + (atm_theta_pp / 100.0)

        k_c25_now = k_for_abs_delta(0.25, is_put=False, sigma=atm_now, T=curr_T)
        k_p25_now = k_for_abs_delta(0.25, is_put=True, sigma=atm_now, T=curr_T)

        exp_c25_shape = _interp_linear_extrap(k_c25_now + k_shift, k_prev, s_prev)
        exp_p25_shape = _interp_linear_extrap(k_p25_now + k_shift, k_prev, s_prev)

        shift_frac = atm_exp - exp_atm_shape
        exp_c25 = exp_c25_shape + shift_frac
        exp_p25 = exp_p25_shape + shift_frac

        exp_call_skew_pp = (exp_c25 - atm_exp) * 100.0
        exp_put_skew_pp = (exp_p25 - atm_exp) * 100.0

        return {
            "delta_atm_iv_pct": _pct_change_frac(atm_now, atm_exp),
            "delta_call_skew_pct": _pct_change_pp(call_skew_pp_now, exp_call_skew_pp),
            "delta_put_skew_pct": _pct_change_pp(put_skew_pp_now, exp_put_skew_pp),
        }
    except Exception:
        return {"delta_atm_iv_pct": None, "delta_call_skew_pct": None, "delta_put_skew_pct": None}


def _fetch_skew_rows_for_times(trade_date: str, expiration_iso: str, times_pt: List[str]) -> pd.DataFrame:
    clean_times = sorted({t for t in times_pt if t})
    if not clean_times:
        return pd.DataFrame()

    df = fetch_skew_data(trade_date, expiration_iso, clean_times)
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    out["snapshot_pt_label"] = out["snapshot_pt"].apply(_normalize_pt_label)
    out = out.sort_values("snapshot_pt")
    return out


def _close_near_target(row: Dict[str, Any], target_level: float, band_pts: float) -> bool:
    close_v = row.get("close")
    if close_v is None:
        return False
    try:
        return abs(float(close_v) - float(target_level)) <= float(band_pts)
    except Exception:
        return False


def _find_target_acceptance_rows(
    *,
    day_rows: List[Dict[str, Any]],
    target_hit_idx: int,
    target_level: float,
    target_proximity_pts: float,
    consolidation_window_minutes: int,
) -> List[Dict[str, Any]]:
    max_bars = max(1, int(consolidation_window_minutes))
    end_idx = min(len(day_rows) - 1, target_hit_idx + max_bars - 1)

    candidate_rows = day_rows[target_hit_idx : end_idx + 1]
    band = float(target_proximity_pts)

    best_cluster: List[Dict[str, Any]] = []
    current_cluster: List[Dict[str, Any]] = []
    outside_count = 0

    for row in candidate_rows:
        inside = _close_near_target(row, target_level, band)

        if inside:
            current_cluster.append(row)
            outside_count = 0
        else:
            if current_cluster:
                outside_count += 1
                if outside_count <= MAX_BARS_OUTSIDE_DURING_ACCEPTANCE:
                    continue
                if len(current_cluster) >= len(best_cluster):
                    best_cluster = current_cluster[:]
                current_cluster = []
                outside_count = 0

    if len(current_cluster) >= len(best_cluster):
        best_cluster = current_cluster[:]

    if len(best_cluster) >= MIN_TARGET_ACCEPTANCE_BARS:
        return best_cluster

    fallback = [row for row in candidate_rows if _close_near_target(row, target_level, band)]
    if len(fallback) >= MIN_TARGET_ACCEPTANCE_BARS:
        return fallback

    return []


def _observe_consolidation_range(
    *,
    day_rows: List[Dict[str, Any]],
    start_idx: int,
    consolidation_window_minutes: int,
    wall_high: float,
    max_zone_breach_pts: float,
    pivot_price: float,
    move_points: float,
    max_move_loss_pct: float,
) -> Dict[str, Any]:
    """
    Observe bars from start_idx for up to consolidation_window_minutes, building
    the true consolidation range freely. No proximity-to-level constraint.

    Three outcomes per bar:
      - PROMOTE: close > wall_high + max_zone_breach_pts
                 Price pushed above the wall — range discarded, caller should
                 advance to next GEX wall and restart with a fresh clock.
      - INVALIDATE: close < pivot_price + move_points * (1 - max_move_loss_pct)
                    75%+ of the move has been given back — setup is dead.
      - CONTINUE: neither trigger — keep building range.

    After consolidation_window_minutes bars with no trigger, range is confirmed.

    Returns dict with:
      status        — 'confirmed' | 'promoted' | 'invalidated' | 'insufficient'
      rows          — all bars observed
      range_high    — highest high seen
      range_low     — lowest low seen
      bars_observed — count
      promote_idx   — bar index in day_rows where promotion triggered (or None)
    """
    max_bars = max(1, int(consolidation_window_minutes))
    end_idx = min(len(day_rows) - 1, start_idx + max_bars - 1)

    invalidation_floor = float(pivot_price) + float(move_points) * (1.0 - float(max_move_loss_pct))
    breach_ceiling = float(wall_high) + float(max_zone_breach_pts)

    observed: List[Dict[str, Any]] = []
    range_high: Optional[float] = None
    range_low: Optional[float] = None

    for i in range(start_idx, end_idx + 1):
        row = day_rows[i]
        high_v = row.get("high")
        low_v = row.get("low")
        close_v = row.get("close")

        if high_v is not None and low_v is not None:
            h = float(high_v)
            lo = float(low_v)
            range_high = h if range_high is None else max(range_high, h)
            range_low = lo if range_low is None else min(range_low, lo)

        observed.append(row)

        if close_v is not None:
            c = float(close_v)
            if c > breach_ceiling:
                return {
                    "status": "promoted",
                    "rows": observed,
                    "range_high": range_high,
                    "range_low": range_low,
                    "bars_observed": len(observed),
                    "promote_idx": i,
                }
            if c < invalidation_floor:
                return {
                    "status": "invalidated",
                    "rows": observed,
                    "range_high": range_high,
                    "range_low": range_low,
                    "bars_observed": len(observed),
                    "promote_idx": None,
                }

    if len(observed) < max_bars:
        return {
            "status": "insufficient",
            "rows": observed,
            "range_high": range_high,
            "range_low": range_low,
            "bars_observed": len(observed),
            "promote_idx": None,
        }

    return {
        "status": "confirmed",
        "rows": observed,
        "range_high": range_high,
        "range_low": range_low,
        "bars_observed": len(observed),
        "promote_idx": None,
    }


def _observe_consolidation_range_down(
    *,
    day_rows: List[Dict[str, Any]],
    start_idx: int,
    consolidation_window_minutes: int,
    wall_high: float,
    wall_low: float,
    max_zone_breach_pts: float,
    pivot_price: float,
    move_points: float,
    max_move_loss_pct: float,
) -> Dict[str, Any]:
    """
    Mirror of _observe_consolidation_range for DOWN moves → LONG trades.

    PROMOTE:    close < wall_high - max_zone_breach_pts
                Price pushed through the top of the target zone — discard,
                advance to next zone down.
    INVALIDATE: close > pivot_price - move_points * (1 - max_move_loss_pct)
                75%+ of the down move given back — setup dead.
    CONFIRM:    window elapsed without either trigger.
    """
    max_bars = max(1, int(consolidation_window_minutes))
    end_idx  = min(len(day_rows) - 1, start_idx + max_bars - 1)

    invalidation_ceiling = float(pivot_price) - float(move_points) * (1.0 - float(max_move_loss_pct))
    breach_floor         = float(wall_high) - float(max_zone_breach_pts)

    observed:   List[Dict[str, Any]] = []
    range_high: Optional[float]      = None
    range_low:  Optional[float]      = None

    for i in range(start_idx, end_idx + 1):
        row     = day_rows[i]
        high_v  = row.get("high")
        low_v   = row.get("low")
        close_v = row.get("close")

        if high_v is not None and low_v is not None:
            h  = float(high_v)
            lo = float(low_v)
            range_high = h  if range_high is None else max(range_high, h)
            range_low  = lo if range_low  is None else min(range_low,  lo)

            # PROMOTE: bar low pushed below the wall — price is through,
            # advance to next zone down regardless of where it closed.
            if lo < breach_floor:
                return {"status": "promoted", "rows": observed, "range_high": range_high, "range_low": range_low, "bars_observed": len(observed), "promote_idx": i}

        observed.append(row)

        # INVALIDATE: close gave back too much of the down move
        if close_v is not None:
            c = float(close_v)
            if c > invalidation_ceiling:
                return {"status": "invalidated", "rows": observed, "range_high": range_high, "range_low": range_low, "bars_observed": len(observed), "promote_idx": None}

    if len(observed) < max_bars:
        return {"status": "insufficient", "rows": observed, "range_high": range_high, "range_low": range_low, "bars_observed": len(observed), "promote_idx": None}

    return {"status": "confirmed", "rows": observed, "range_high": range_high, "range_low": range_low, "bars_observed": len(observed), "promote_idx": None}


def _make_default_trade_fields(trade_reason: str = "not_evaluated") -> Dict[str, Any]:
    return {
        "trade_entry_found": False,
        "trade_entry_ts_pt": None,
        "trade_entry_price": None,
        "trade_entry_reason": trade_reason,
        "trade_range_high_at_entry": None,
        "trade_range_low_at_entry": None,
        "trade_entry_band_floor": None,
        "trade_initial_stop_price": None,
        "trade_take_profit_price": None,
        "trade_trailing_active": False,
        "trade_trailing_stop_price": None,
        "trade_exit_ts_pt": None,
        "trade_exit_price": None,
        "trade_exit_reason": None,
        "trade_realized_points": None,
        "trade_mfe_points": None,
        "trade_mae_points": None,
        "trade_outcome": None,
    }


def _simulate_short_trade_from_signal(
    *,
    day_rows: List[Dict[str, Any]],
    signal_idx: int,
    seed_rows: List[Dict[str, Any]],
    confirmed_range_high: Optional[float],
    consolidation_end_idx: int,
    entry_within_top_pts: float,
    entry_search_window_minutes: int,
    initial_stop_pts: float,
    trail_activate_profit_pts: float,
    trailing_stop_pts: float,
    take_profit_pts: float,
    max_minutes_before_close: int = 45,
) -> Dict[str, Any]:
    default = _make_default_trade_fields("no_entry_window")

    if signal_idx < 0 or signal_idx >= len(day_rows):
        default["trade_entry_reason"] = "invalid_signal_index"
        return default

    seed_high = max(float(r["high"]) for r in seed_rows if r.get("high") is not None)
    seed_low = min(float(r["low"]) for r in seed_rows if r.get("low") is not None)
    # Use the confirmed full-range high if provided — this ensures the entry band
    # is anchored to the true range top even if the skew signal fired early.
    range_high = confirmed_range_high if confirmed_range_high is not None else seed_high
    range_low = seed_low

    # Entry search begins after consolidation window closes, not at signal time.
    # Signal may fire during consolidation but entry must wait until range is proven.
    entry_search_start = max(signal_idx, consolidation_end_idx + 1)
    search_end_idx = min(len(day_rows) - 1, entry_search_start + max(1, int(entry_search_window_minutes)) - 1)

    entry_idx: Optional[int] = None
    entry_price: Optional[float] = None
    entry_band_floor: Optional[float] = None
    range_high_at_entry: Optional[float] = None
    range_low_at_entry: Optional[float] = None

    for j in range(entry_search_start, search_end_idx + 1):
        row = day_rows[j]
        if row.get("high") is None or row.get("low") is None:
            continue

        # Block entry within max_minutes_before_close of 13:00 PT close
        if max_minutes_before_close > 0:
            ts_pt = _normalize_pt_label(row.get("ts_pt"))
            if ts_pt:
                try:
                    eh, em = int(ts_pt[:2]), int(ts_pt[3:5])
                    minutes_to_close = (13 - eh) * 60 - em
                    if minutes_to_close <= int(max_minutes_before_close):
                        break
                except Exception:
                    pass

        high_v = float(row["high"])
        low_v = float(row["low"])
        range_high = max(range_high, high_v)
        range_low = min(range_low, low_v)

        band_floor = range_high - float(entry_within_top_pts)

        if high_v >= band_floor:
            entry_idx = j
            entry_price = round(band_floor, 2)
            entry_band_floor = round(band_floor, 2)
            range_high_at_entry = round(range_high, 2)
            range_low_at_entry = round(range_low, 2)
            break

    if entry_idx is None or entry_price is None:
        return default

    initial_stop_price = round(entry_price + float(initial_stop_pts), 2)
    take_profit_price = round(entry_price - float(take_profit_pts), 2)

    lowest_low_since_entry = entry_price
    trailing_active = False
    trailing_stop_price: Optional[float] = None
    mfe_points = 0.0
    mae_points = 0.0

    exit_ts_pt: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None

    for k in range(entry_idx + 1, len(day_rows)):
        row = day_rows[k]
        if row.get("high") is None or row.get("low") is None:
            continue

        high_v = float(row["high"])
        low_v = float(row["low"])

        lowest_low_since_entry = min(lowest_low_since_entry, low_v)
        mfe_points = max(mfe_points, entry_price - low_v)
        mae_points = max(mae_points, high_v - entry_price)

        if (not trailing_active) and mfe_points >= float(trail_activate_profit_pts):
            trailing_active = True

        active_stop_price = initial_stop_price
        if trailing_active:
            trailing_stop_price = round(lowest_low_since_entry + float(trailing_stop_pts), 2)
            active_stop_price = trailing_stop_price

        # Conservative assumption: if both stop and target are hit in same bar,
        # the stop wins.
        if high_v >= active_stop_price:
            exit_ts_pt = _normalize_pt_label(row.get("ts_pt"))
            exit_price = round(active_stop_price, 2)
            exit_reason = "trailing_stop" if trailing_active else "initial_stop"
            break

        if low_v <= take_profit_price:
            exit_ts_pt = _normalize_pt_label(row.get("ts_pt"))
            exit_price = round(take_profit_price, 2)
            exit_reason = "take_profit"
            break

    if exit_price is None:
        last_row = day_rows[-1]
        last_close = last_row.get("close")
        if last_close is not None:
            exit_ts_pt = _normalize_pt_label(last_row.get("ts_pt"))
            exit_price = round(float(last_close), 2)
            exit_reason = "end_of_day"
        else:
            exit_reason = "open_no_exit"

    realized = None if exit_price is None else round(entry_price - float(exit_price), 2)

    if realized is None:
        outcome = None
    elif realized > 0:
        outcome = "win"
    elif realized < 0:
        outcome = "loss"
    else:
        outcome = "flat"

    return {
        "trade_entry_found": True,
        "trade_entry_ts_pt": _normalize_pt_label(day_rows[entry_idx].get("ts_pt")),
        "trade_entry_price": entry_price,
        "trade_entry_reason": "entry_band_hit",
        "trade_range_high_at_entry": range_high_at_entry,
        "trade_range_low_at_entry": range_low_at_entry,
        "trade_entry_band_floor": entry_band_floor,
        "trade_initial_stop_price": initial_stop_price,
        "trade_take_profit_price": take_profit_price,
        "trade_trailing_active": trailing_active,
        "trade_trailing_stop_price": trailing_stop_price,
        "trade_exit_ts_pt": exit_ts_pt,
        "trade_exit_price": exit_price,
        "trade_exit_reason": exit_reason,
        "trade_realized_points": realized,
        "trade_mfe_points": round(mfe_points, 2),
        "trade_mae_points": round(mae_points, 2),
        "trade_outcome": outcome,
    }


def _compute_prior_move_context(
    day_rows: List[Dict[str, Any]],
    pivot_idx: int,
    pivot_price: float,
    up_move_pts: float,
) -> Dict[str, Any]:
    """
    Look back from the start of the up move (pivot_idx) across the session so far
    and compute three context features that characterise prior trend:

    prior_session_down_pts
        The largest peak-to-trough down move that completed *before* pivot_idx.
        Computed as the maximum (peak_high - subsequent_low) over a rolling window
        of all RTH bars up to and including pivot_idx.

    prior_down_vs_up_ratio
        prior_session_down_pts / up_move_pts.
        > 1.0 means the prior down move was larger than the up move we are about
        to fade — the "bounce off the lows" warning signal.

    start_pct_of_session_range
        Where pivot_price sits within the session high/low range observed up to
        pivot_idx.  0.0 = at the session low, 1.0 = at the session high.
        A low value (e.g. < 0.25) means we are still near the bottom of the day's
        range — another bounce indicator.
    """
    empty: Dict[str, Any] = {
        "prior_session_down_pts": None,
        "prior_down_vs_up_ratio": None,
        "start_pct_of_session_range": None,
    }

    if pivot_idx <= 0 or up_move_pts <= 0:
        return empty

    prior_rows = day_rows[: pivot_idx + 1]

    # Session range up to pivot
    try:
        session_high = max(float(r["high"]) for r in prior_rows if r.get("high") is not None)
        session_low = min(float(r["low"]) for r in prior_rows if r.get("low") is not None)
    except (ValueError, TypeError):
        return empty

    session_range = session_high - session_low
    if session_range <= 0:
        start_pct = None
    else:
        start_pct = round((pivot_price - session_low) / session_range, 3)

    # Largest prior down move: rolling peak → trough
    max_down = 0.0
    running_peak = float(prior_rows[0]["high"]) if prior_rows[0].get("high") is not None else None

    for r in prior_rows:
        h = r.get("high")
        lo = r.get("low")
        if h is None or lo is None:
            continue
        h_f = float(h)
        lo_f = float(lo)
        if running_peak is None:
            running_peak = h_f
        else:
            running_peak = max(running_peak, h_f)
        drawdown = running_peak - lo_f
        if drawdown > max_down:
            max_down = drawdown

    prior_down_pts = round(max_down, 2) if max_down > 0 else None
    ratio = round(max_down / up_move_pts, 3) if (prior_down_pts is not None and up_move_pts > 0) else None

    return {
        "prior_session_down_pts": prior_down_pts,
        "prior_down_vs_up_ratio": ratio,
        "start_pct_of_session_range": start_pct,
    }


def _evaluate_up_short_setup(
    *,
    trade_date: dt.date,
    day_rows: List[Dict[str, Any]],
    entry_idx: int,
    target_hit_idx: int,
    target_level: float,
    consolidation_window_minutes: int,
    short_put_skew_increase_pct: float,
    short_call_skew_max_pct: float,
    entry_within_top_pts: float,
    entry_search_window_minutes: int,
    initial_stop_pts: float,
    trail_activate_profit_pts: float,
    trailing_stop_pts: float,
    take_profit_pts: float,
    prior_ctx: Dict[str, Any],
    max_prior_down_up_ratio: float,
    max_start_pct_of_range: float,
    observed_rows: List[Dict[str, Any]],
    confirmed_range_high: Optional[float],
    max_minutes_before_close: int = 45,
) -> Dict[str, Any]:
    default = {
        "consolidation_minutes_observed": 0,
        "consolidation_end_ts_pt": None,
        "short_setup_found": False,
        "short_signal_ts_pt": None,
        "short_signal_price": None,
        "short_signal_delta_atm_iv_pct": None,
        "short_signal_delta_call_skew_pct": None,
        "short_signal_delta_put_skew_pct": None,
        "short_setup_reason": "not_evaluated",
        **_make_default_trade_fields("setup_not_hit"),
    }

    # --- Prior context filter ---
    ratio = prior_ctx.get("prior_down_vs_up_ratio")
    start_pct = prior_ctx.get("start_pct_of_session_range")
    ratio_breached = ratio is not None and float(ratio) > float(max_prior_down_up_ratio)
    pct_breached = start_pct is not None and float(start_pct) < float(max_start_pct_of_range)
    if ratio_breached and pct_breached:
        default["short_setup_reason"] = "prior_context_invalidated"
        default["trade_entry_reason"] = "prior_context_invalidated"
        return default

    # Range is pre-built by _observe_consolidation_range in _day_zone_results
    if not observed_rows:
        default["short_setup_reason"] = "no_target_consolidation"
        default["trade_entry_reason"] = "no_setup"
        return default

    entry_ts_pt = _normalize_pt_label(day_rows[entry_idx].get("ts_pt"))
    acceptance_times = [_normalize_pt_label(r.get("ts_pt")) for r in observed_rows]
    skew_times = [entry_ts_pt] + acceptance_times

    skew_df = _fetch_skew_rows_for_times(str(trade_date), str(trade_date), skew_times)
    if skew_df.empty:
        default["consolidation_minutes_observed"] = len(observed_rows)
        default["consolidation_end_ts_pt"] = _normalize_pt_label(observed_rows[-1].get("ts_pt"))
        default["short_setup_reason"] = "no_skew_data"
        default["trade_entry_reason"] = "no_setup"
        return default

    by_label: Dict[str, pd.Series] = {}
    for _, row in skew_df.iterrows():
        label = str(row.get("snapshot_pt_label") or "")
        if label and label not in by_label:
            by_label[label] = row

    entry_skew_row = by_label.get(entry_ts_pt)
    if entry_skew_row is None:
        default["consolidation_minutes_observed"] = len(observed_rows)
        default["consolidation_end_ts_pt"] = _normalize_pt_label(observed_rows[-1].get("ts_pt"))
        default["short_setup_reason"] = "missing_entry_skew"
        default["trade_entry_reason"] = "no_setup"
        return default

    latest_metrics: Dict[str, Optional[float]] = {
        "delta_atm_iv_pct": None,
        "delta_call_skew_pct": None,
        "delta_put_skew_pct": None,
    }

    for i, row in enumerate(observed_rows):
        label = _normalize_pt_label(row.get("ts_pt"))
        curr_skew_row = by_label.get(label)
        if curr_skew_row is None:
            continue

        metrics = _expected_skew_deltas_from_entry(entry_skew_row, curr_skew_row, str(trade_date))
        latest_metrics = metrics

        d_put = metrics.get("delta_put_skew_pct")
        d_call = metrics.get("delta_call_skew_pct")

        if d_put is None or d_call is None:
            continue

        if d_put >= float(short_put_skew_increase_pct) and d_call <= float(short_call_skew_max_pct):
            signal_price = round(float(row["close"]), 2) if row.get("close") is not None else None
            signal_label = label
            signal_day_idx = next(
                (
                    idx
                    for idx in range(target_hit_idx, len(day_rows))
                    if _normalize_pt_label(day_rows[idx].get("ts_pt")) == signal_label
                ),
                -1,
            )

            seed_rows = observed_rows[: i + 1]

            # Find the day_rows index of the last consolidation bar so the
            # entry search starts after the window closes.
            last_consol_ts = _normalize_pt_label(observed_rows[-1].get("ts_pt"))
            consolidation_end_idx = next(
                (
                    idx
                    for idx in range(target_hit_idx, len(day_rows))
                    if _normalize_pt_label(day_rows[idx].get("ts_pt")) == last_consol_ts
                ),
                signal_day_idx,
            )

            trade_eval = _simulate_short_trade_from_signal(
                day_rows=day_rows,
                signal_idx=signal_day_idx,
                seed_rows=seed_rows,
                confirmed_range_high=confirmed_range_high,
                consolidation_end_idx=consolidation_end_idx,
                entry_within_top_pts=entry_within_top_pts,
                entry_search_window_minutes=entry_search_window_minutes,
                initial_stop_pts=initial_stop_pts,
                trail_activate_profit_pts=trail_activate_profit_pts,
                trailing_stop_pts=trailing_stop_pts,
                take_profit_pts=take_profit_pts,
                max_minutes_before_close=max_minutes_before_close,
            )

            return {
                "consolidation_minutes_observed": len(observed_rows),
                "consolidation_end_ts_pt": _normalize_pt_label(observed_rows[-1].get("ts_pt")),
                "short_setup_found": True,
                "short_signal_ts_pt": signal_label,
                "short_signal_price": signal_price,
                "short_signal_delta_atm_iv_pct": None if metrics["delta_atm_iv_pct"] is None else round(float(metrics["delta_atm_iv_pct"]), 2),
                "short_signal_delta_call_skew_pct": round(float(d_call), 2),
                "short_signal_delta_put_skew_pct": round(float(d_put), 2),
                "short_setup_reason": "threshold_hit",
                **trade_eval,
            }

    return {
        "consolidation_minutes_observed": len(observed_rows),
        "consolidation_end_ts_pt": _normalize_pt_label(observed_rows[-1].get("ts_pt")),
        "short_setup_found": False,
        "short_signal_ts_pt": None,
        "short_signal_price": None,
        "short_signal_delta_atm_iv_pct": None if latest_metrics["delta_atm_iv_pct"] is None else round(float(latest_metrics["delta_atm_iv_pct"]), 2),
        "short_signal_delta_call_skew_pct": None if latest_metrics["delta_call_skew_pct"] is None else round(float(latest_metrics["delta_call_skew_pct"]), 2),
        "short_signal_delta_put_skew_pct": None if latest_metrics["delta_put_skew_pct"] is None else round(float(latest_metrics["delta_put_skew_pct"]), 2),
        "short_setup_reason": "threshold_not_met",
        **_make_default_trade_fields("no_setup"),
    }


# ─────────────────────────────────────────────────────────────
#  DOWN MOVE → LONG  (mirror of up-move short)
# ─────────────────────────────────────────────────────────────

def _compute_prior_move_context_down(
    day_rows: List[Dict[str, Any]],
    pivot_idx: int,
    pivot_price: float,
    down_move_pts: float,
) -> Dict[str, Any]:
    """
    Mirror of _compute_prior_move_context for down moves.
    Measures the largest prior UP move before the down move started,
    and where the pivot sits within the session range.
    """
    empty: Dict[str, Any] = {
        "prior_session_up_pts": None,
        "prior_up_vs_down_ratio": None,
        "start_pct_of_session_range": None,
    }

    if pivot_idx <= 0 or down_move_pts <= 0:
        return empty

    prior_rows = day_rows[: pivot_idx + 1]

    try:
        session_high = max(float(r["high"]) for r in prior_rows if r.get("high") is not None)
        session_low  = min(float(r["low"])  for r in prior_rows if r.get("low")  is not None)
    except (ValueError, TypeError):
        return empty

    session_range = session_high - session_low
    start_pct = round((pivot_price - session_low) / session_range, 3) if session_range > 0 else None

    # Largest prior up move: rolling trough → peak
    max_up = 0.0
    running_trough: Optional[float] = None
    for r in prior_rows:
        h  = r.get("high")
        lo = r.get("low")
        if h is None or lo is None:
            continue
        h_f  = float(h)
        lo_f = float(lo)
        running_trough = lo_f if running_trough is None else min(running_trough, lo_f)
        rally = h_f - running_trough
        if rally > max_up:
            max_up = rally

    prior_up_pts = round(max_up, 2) if max_up > 0 else None
    ratio = round(max_up / down_move_pts, 3) if (prior_up_pts is not None and down_move_pts > 0) else None

    return {
        "prior_session_up_pts": prior_up_pts,
        "prior_up_vs_down_ratio": ratio,
        "start_pct_of_session_range": start_pct,
    }


def _simulate_long_trade_from_signal(
    *,
    day_rows: List[Dict[str, Any]],
    signal_idx: int,
    seed_rows: List[Dict[str, Any]],
    confirmed_range_low: Optional[float],
    consolidation_end_idx: int,
    entry_within_bottom_pts: float,
    entry_search_window_minutes: int,
    long_initial_stop_pts: float,
    long_trail_activate_profit_pts: float,
    long_trailing_stop_pts: float,
    long_take_profit_pts: float,
    max_minutes_before_close: int = 45,
) -> Dict[str, Any]:
    """Mirror of _simulate_short_trade_from_signal for LONG trades."""
    default = _make_default_trade_fields("no_entry_window")

    if signal_idx < 0 or signal_idx >= len(day_rows):
        default["trade_entry_reason"] = "invalid_signal_index"
        return default

    seed_low  = min(float(r["low"])  for r in seed_rows if r.get("low")  is not None)
    seed_high = max(float(r["high"]) for r in seed_rows if r.get("high") is not None)
    # Anchor entry band to the confirmed full-range low
    range_low  = confirmed_range_low if confirmed_range_low is not None else seed_low
    range_high = seed_high

    # For longs, start entry search from the signal bar itself — unlike shorts
    # where we wait for consolidation to close before looking for a re-test,
    # long entries should be taken at or near the signal when price is still
    # near the lows, not after the bounce has already begun.
    entry_search_start = signal_idx
    search_end_idx = min(len(day_rows) - 1, entry_search_start + max(1, int(entry_search_window_minutes)) - 1)

    entry_idx: Optional[int]   = None
    entry_price: Optional[float] = None
    entry_band_ceiling: Optional[float] = None
    range_high_at_entry: Optional[float] = None
    range_low_at_entry:  Optional[float] = None

    for j in range(entry_search_start, search_end_idx + 1):
        row = day_rows[j]
        if row.get("high") is None or row.get("low") is None:
            continue

        # Block entry within max_minutes_before_close of 13:00 PT close
        if max_minutes_before_close > 0:
            ts_pt = _normalize_pt_label(row.get("ts_pt"))
            if ts_pt:
                try:
                    eh, em = int(ts_pt[:2]), int(ts_pt[3:5])
                    minutes_to_close = (13 - eh) * 60 - em
                    if minutes_to_close <= int(max_minutes_before_close):
                        break
                except Exception:
                    pass

        high_v = float(row["high"])
        low_v  = float(row["low"])
        range_low  = min(range_low,  low_v)
        range_high = max(range_high, high_v)

        band_ceiling = range_low + float(entry_within_bottom_pts)
        if low_v <= band_ceiling:
            entry_idx          = j
            entry_price        = round(band_ceiling, 2)
            entry_band_ceiling = round(band_ceiling, 2)
            range_high_at_entry = round(range_high, 2)
            range_low_at_entry  = round(range_low,  2)
            break

    if entry_idx is None or entry_price is None:
        return default

    initial_stop_price = round(entry_price - float(long_initial_stop_pts), 2)
    take_profit_price  = round(entry_price + float(long_take_profit_pts),  2)

    highest_high_since_entry = entry_price
    trailing_active  = False
    trailing_stop_price: Optional[float] = None
    mfe_points = 0.0
    mae_points = 0.0

    exit_ts_pt:  Optional[str]   = None
    exit_price:  Optional[float] = None
    exit_reason: Optional[str]   = None

    for k in range(entry_idx + 1, len(day_rows)):
        row = day_rows[k]
        if row.get("high") is None or row.get("low") is None:
            continue
        high_v = float(row["high"])
        low_v  = float(row["low"])

        highest_high_since_entry = max(highest_high_since_entry, high_v)
        mfe_points = max(mfe_points, high_v - entry_price)
        mae_points = max(mae_points, entry_price - low_v)

        if (not trailing_active) and mfe_points >= float(long_trail_activate_profit_pts):
            trailing_active = True

        active_stop_price = initial_stop_price
        if trailing_active:
            trailing_stop_price = round(highest_high_since_entry - float(long_trailing_stop_pts), 2)
            active_stop_price   = trailing_stop_price

        if low_v <= active_stop_price:
            exit_ts_pt  = _normalize_pt_label(row.get("ts_pt"))
            exit_price  = round(active_stop_price, 2)
            exit_reason = "trailing_stop" if trailing_active else "initial_stop"
            break

        if high_v >= take_profit_price:
            exit_ts_pt  = _normalize_pt_label(row.get("ts_pt"))
            exit_price  = round(take_profit_price, 2)
            exit_reason = "take_profit"
            break

    if exit_price is None:
        last_row   = day_rows[-1]
        last_close = last_row.get("close")
        if last_close is not None:
            exit_ts_pt  = _normalize_pt_label(last_row.get("ts_pt"))
            exit_price  = round(float(last_close), 2)
            exit_reason = "end_of_day"
        else:
            exit_reason = "open_no_exit"

    realized = None if exit_price is None else round(float(exit_price) - entry_price, 2)
    if realized is None:
        outcome = None
    elif realized > 0:
        outcome = "win"
    elif realized < 0:
        outcome = "loss"
    else:
        outcome = "flat"

    return {
        "trade_entry_found":       True,
        "trade_entry_ts_pt":       _normalize_pt_label(day_rows[entry_idx].get("ts_pt")),
        "trade_entry_price":       entry_price,
        "trade_entry_reason":      "entry_band_hit",
        "trade_range_high_at_entry": range_high_at_entry,
        "trade_range_low_at_entry":  range_low_at_entry,
        "trade_entry_band_floor":  entry_band_ceiling,
        "trade_initial_stop_price": initial_stop_price,
        "trade_take_profit_price":  take_profit_price,
        "trade_trailing_active":    trailing_active,
        "trade_trailing_stop_price": trailing_stop_price,
        "trade_exit_ts_pt":         exit_ts_pt,
        "trade_exit_price":         exit_price,
        "trade_exit_reason":        exit_reason,
        "trade_realized_points":    realized,
        "trade_mfe_points":         round(mfe_points, 2),
        "trade_mae_points":         round(mae_points, 2),
        "trade_outcome":            outcome,
    }


def _evaluate_down_long_setup(
    *,
    trade_date: dt.date,
    day_rows: List[Dict[str, Any]],
    entry_idx: int,
    target_hit_idx: int,
    target_level: float,
    consolidation_window_minutes: int,
    long_put_skew_min_decrease_pct: float,
    long_call_skew_min_increase_pct: float,
    entry_within_bottom_pts: float,
    entry_search_window_minutes: int,
    long_initial_stop_pts: float,
    long_trail_activate_profit_pts: float,
    long_trailing_stop_pts: float,
    long_take_profit_pts: float,
    prior_ctx: Dict[str, Any],
    max_prior_down_up_ratio: float,
    max_start_pct_of_range: float,
    source_zone_low: float,
    source_zone_high: float,
    wall_low: float,
    max_move_loss_pct: float,
    pivot_price: float,
    move_points: float,
    observed_rows: List[Dict[str, Any]],
    confirmed_range_low: Optional[float],
    max_minutes_before_close: int = 45,
) -> Dict[str, Any]:
    """
    Evaluate DOWN move → LONG trade from pre-built consolidation range.
    The observation window is run by the caller (_day_zone_results) which
    handles promotion to lower zones. This function receives the confirmed
    observed_rows and confirmed_range_low directly.

    Skew signal: delta_put_skew_pct <= -long_put_skew_min_decrease_pct (put fear unwinding)
                 delta_call_skew_pct >=  long_call_skew_min_increase_pct (calls being bid)
    Entry: within entry_within_bottom_pts of confirmed range low after consolidation.
    """
    default = {
        "consolidation_minutes_observed": 0,
        "consolidation_end_ts_pt":        None,
        "long_setup_found":               False,
        "long_signal_ts_pt":              None,
        "long_signal_price":              None,
        "long_signal_delta_atm_iv_pct":   None,
        "long_signal_delta_call_skew_pct": None,
        "long_signal_delta_put_skew_pct":  None,
        "long_setup_reason":              "not_evaluated",
        **_make_default_trade_fields("setup_not_hit"),
    }

    # Prior context filter
    ratio     = prior_ctx.get("prior_up_vs_down_ratio")
    start_pct = prior_ctx.get("start_pct_of_session_range")
    ratio_breached = ratio     is not None and float(ratio)     > float(max_prior_down_up_ratio)
    pct_breached   = start_pct is not None and float(start_pct) > (1.0 - float(max_start_pct_of_range))
    if ratio_breached and pct_breached:
        default["long_setup_reason"]  = "prior_context_invalidated"
        default["trade_entry_reason"] = "prior_context_invalidated"
        return default

    # Range is pre-built by _observe_consolidation_range_down in _day_zone_results
    if not observed_rows:
        default["long_setup_reason"]  = "no_target_consolidation"
        default["trade_entry_reason"] = "no_setup"
        return default

    entry_ts_pt      = _normalize_pt_label(day_rows[entry_idx].get("ts_pt"))
    acceptance_times = [_normalize_pt_label(r.get("ts_pt")) for r in observed_rows]
    skew_times       = [entry_ts_pt] + acceptance_times

    skew_df = _fetch_skew_rows_for_times(str(trade_date), str(trade_date), skew_times)
    if skew_df.empty:
        default["consolidation_minutes_observed"] = len(observed_rows)
        default["consolidation_end_ts_pt"]        = _normalize_pt_label(observed_rows[-1].get("ts_pt"))
        default["long_setup_reason"]              = "no_skew_data"
        default["trade_entry_reason"]             = "no_setup"
        return default

    by_label: Dict[str, pd.Series] = {}
    for _, row in skew_df.iterrows():
        label = str(row.get("snapshot_pt_label") or "")
        if label and label not in by_label:
            by_label[label] = row

    entry_skew_row = by_label.get(entry_ts_pt)
    if entry_skew_row is None:
        default["consolidation_minutes_observed"] = len(observed_rows)
        default["consolidation_end_ts_pt"]        = _normalize_pt_label(observed_rows[-1].get("ts_pt"))
        default["long_setup_reason"]              = "missing_entry_skew"
        default["trade_entry_reason"]             = "no_setup"
        return default

    latest_metrics: Dict[str, Optional[float]] = {
        "delta_atm_iv_pct":   None,
        "delta_call_skew_pct": None,
        "delta_put_skew_pct":  None,
    }

    for i, row in enumerate(observed_rows):
        label = _normalize_pt_label(row.get("ts_pt"))
        curr_skew_row = by_label.get(label)
        if curr_skew_row is None:
            continue

        metrics = _expected_skew_deltas_from_entry(entry_skew_row, curr_skew_row, str(trade_date))
        latest_metrics = metrics

        d_put  = metrics.get("delta_put_skew_pct")
        d_call = metrics.get("delta_call_skew_pct")
        if d_put is None or d_call is None:
            continue

        # Long signal: put skew decreasing (fear unwinding) AND calls being bid
        if d_put <= -float(long_put_skew_min_decrease_pct) and d_call >= float(long_call_skew_min_increase_pct):
            signal_price = round(float(row["close"]), 2) if row.get("close") is not None else None
            signal_label = label
            signal_day_idx = next(
                (idx for idx in range(target_hit_idx, len(day_rows))
                 if _normalize_pt_label(day_rows[idx].get("ts_pt")) == signal_label),
                -1,
            )

            seed_rows = observed_rows[: i + 1]

            last_consol_ts = _normalize_pt_label(observed_rows[-1].get("ts_pt"))
            consolidation_end_idx = next(
                (idx for idx in range(target_hit_idx, len(day_rows))
                 if _normalize_pt_label(day_rows[idx].get("ts_pt")) == last_consol_ts),
                signal_day_idx,
            )

            trade_eval = _simulate_long_trade_from_signal(
                day_rows=day_rows,
                signal_idx=signal_day_idx,
                seed_rows=seed_rows,
                confirmed_range_low=confirmed_range_low,
                consolidation_end_idx=consolidation_end_idx,
                entry_within_bottom_pts=entry_within_bottom_pts,
                entry_search_window_minutes=entry_search_window_minutes,
                long_initial_stop_pts=long_initial_stop_pts,
                long_trail_activate_profit_pts=long_trail_activate_profit_pts,
                long_trailing_stop_pts=long_trailing_stop_pts,
                long_take_profit_pts=long_take_profit_pts,
                max_minutes_before_close=max_minutes_before_close,
            )

            return {
                "consolidation_minutes_observed":  len(observed_rows),
                "consolidation_end_ts_pt":         _normalize_pt_label(observed_rows[-1].get("ts_pt")),
                "long_setup_found":                True,
                "long_signal_ts_pt":               signal_label,
                "long_signal_price":               signal_price,
                "long_signal_delta_atm_iv_pct":    None if metrics["delta_atm_iv_pct"] is None else round(float(metrics["delta_atm_iv_pct"]), 2),
                "long_signal_delta_call_skew_pct": round(float(d_call), 2),
                "long_signal_delta_put_skew_pct":  round(float(d_put),  2),
                "long_setup_reason":               "threshold_hit",
                **trade_eval,
            }

    return {
        "consolidation_minutes_observed":  len(observed_rows),
        "consolidation_end_ts_pt":         _normalize_pt_label(observed_rows[-1].get("ts_pt")),
        "long_setup_found":                False,
        "long_signal_ts_pt":               None,
        "long_signal_price":               None,
        "long_signal_delta_atm_iv_pct":    None if latest_metrics["delta_atm_iv_pct"] is None else round(float(latest_metrics["delta_atm_iv_pct"]), 2),
        "long_signal_delta_call_skew_pct": None if latest_metrics["delta_call_skew_pct"] is None else round(float(latest_metrics["delta_call_skew_pct"]), 2),
        "long_signal_delta_put_skew_pct":  None if latest_metrics["delta_put_skew_pct"]  is None else round(float(latest_metrics["delta_put_skew_pct"]),  2),
        "long_setup_reason":               "threshold_not_met",
        **_make_default_trade_fields("no_setup"),
    }


def _day_zone_results(
    trade_date: dt.date,
    day_rows: List[Dict[str, Any]],
    *,
    level_family: str,
    min_level_gex_bn: float,
    zone_merge_distance_pts: float,
    min_clean_move_points: float,
    target_proximity_pts: float,
    max_zone_breach_pts: float,
    pivot_strength_bars: int,
    max_results: int,
    consolidation_window_minutes: int,
    short_put_skew_increase_pct: float,
    short_call_skew_max_pct: float,
    entry_within_top_pts: float,
    entry_search_window_minutes: int,
    initial_stop_pts: float,
    trail_activate_profit_pts: float,
    trailing_stop_pts: float,
    take_profit_pts: float,
    max_prior_down_up_ratio: float,
    max_start_pct_of_range: float,
    max_move_loss_pct: float,
    min_minutes_after_open: int,
    long_put_skew_min_decrease_pct: float,
    long_call_skew_min_increase_pct: float,
    max_minutes_before_close: int,
    long_initial_stop_pts: float,
    long_trail_activate_profit_pts: float,
    long_trailing_stop_pts: float,
    long_take_profit_pts: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    qualifying_levels = _collect_day_levels(day_rows, level_family, min_level_gex_bn)
    zones = _build_zones(qualifying_levels, zone_merge_distance_pts)

    diagnostics = {
        "qualifying_levels": len(qualifying_levels),
        "zones_total": len(zones),
        "source_zones_with_clean_targets": 0,
        "zone_episodes_considered": 0,
        "valid_instances": 0,
        "up_short_setups_found": 0,
        "actual_trades_found": 0,
        "winning_trades": 0,
        "sample_zones": [
            {
                "range": f"{zone['low']:.2f} – {zone['high']:.2f}",
                "levels": zone["levels_text"],
                "width": round(zone["width"], 2),
                "count": int(zone["count"]),
                "max_abs_gex_bn": round(float(zone["max_abs_gex_bn"]), 2),
            }
            for zone in zones[:8]
        ],
        "sample_short_setups": [],
        "sample_trades": [],
    }

    if not zones:
        return [], diagnostics

    results: List[Dict[str, Any]] = []

    for zone_idx, zone in enumerate(zones):
        masks = [_in_zone_close(row, zone, max_zone_breach_pts) for row in day_rows]
        episodes = _true_segments(masks)

        if zone_idx < len(zones) - 1:
            # Gather all zones above as potential targets
            candidate_target_zone_indices = list(range(zone_idx + 1, len(zones)))

            for seg_start, seg_end in episodes:
                diagnostics["zone_episodes_considered"] += 1
                seg_min_low = min(float(day_rows[i]["low"]) for i in range(seg_start, seg_end + 1))
                if seg_min_low < float(zone["low"]) - float(max_zone_breach_pts):
                    continue

                pivot_idx, pivot_price = _find_last_pivot(day_rows, seg_start, seg_end, "up", pivot_strength_bars)

                # Skip episodes whose pivot falls within min_minutes_after_open
                # of the RTH open (06:30 PT). Early-session price is often noisy
                # and this filter prevents chasing moves that start at the bell.
                if min_minutes_after_open > 0:
                    pivot_pt = _normalize_pt_label(day_rows[pivot_idx].get("ts_pt"))
                    if pivot_pt:
                        try:
                            ph, pm = int(pivot_pt[:2]), int(pivot_pt[3:5])
                            pivot_minutes_since_open = (ph - 6) * 60 + pm - 30
                            if pivot_minutes_since_open < int(min_minutes_after_open):
                                continue
                        except Exception:
                            pass

                # Walk up through candidate target zones, promoting if price
                # pushes through a wall before consolidating.
                scan_from_idx = seg_end + 1
                for target_zone_offset, t_zone_idx in enumerate(candidate_target_zone_indices):
                    target_zone_up = zones[t_zone_idx]
                    clean_space_up = float(target_zone_up["low"]) - float(zone["high"])
                    if clean_space_up < float(min_clean_move_points):
                        continue

                    if target_zone_offset == 0:
                        diagnostics["source_zones_with_clean_targets"] += 1

                    hit_idx = _first_target_hit(
                        day_rows,
                        start_scan_idx=scan_from_idx,
                        direction="up",
                        zone=zone,
                        target_level=float(target_zone_up["low"]),
                        target_proximity_pts=target_proximity_pts,
                        max_zone_breach_pts=max_zone_breach_pts,
                    )
                    if hit_idx is None:
                        break

                    move_points = float(target_zone_up["low"]) - float(pivot_price)

                    # Observe the consolidation window from the moment of target hit.
                    # If price promotes (pushes above the wall), advance to the next
                    # GEX zone and restart with a fresh clock from promote_idx.
                    obs = _observe_consolidation_range(
                        day_rows=day_rows,
                        start_idx=hit_idx,
                        consolidation_window_minutes=consolidation_window_minutes,
                        wall_high=float(target_zone_up["high"]),
                        max_zone_breach_pts=float(max_zone_breach_pts),
                        pivot_price=float(pivot_price),
                        move_points=float(move_points),
                        max_move_loss_pct=float(max_move_loss_pct),
                    )

                    if obs["status"] == "promoted":
                        # Price pushed through this wall — try next zone up.
                        # Reset scan start to the bar that triggered promotion.
                        scan_from_idx = obs["promote_idx"]
                        continue

                    if obs["status"] in ("invalidated", "insufficient"):
                        # Move gave back too much or ran out of day — no setup.
                        break

                    # status == "confirmed" — range is locked, evaluate setup.
                    observed_rows = obs["rows"]
                    start_row = day_rows[pivot_idx]
                    target_row = day_rows[hit_idx]
                    start_open = start_row.get("open")
                    target_open = target_row.get("open")

                    prior_ctx = _compute_prior_move_context(
                        day_rows,
                        pivot_idx=pivot_idx,
                        pivot_price=pivot_price,
                        up_move_pts=float(move_points),
                    )

                    setup_eval = _evaluate_up_short_setup(
                        trade_date=trade_date,
                        day_rows=day_rows,
                        entry_idx=pivot_idx,
                        target_hit_idx=hit_idx,
                        target_level=float(target_zone_up["low"]),
                        consolidation_window_minutes=consolidation_window_minutes,
                        short_put_skew_increase_pct=short_put_skew_increase_pct,
                        short_call_skew_max_pct=short_call_skew_max_pct,
                        entry_within_top_pts=entry_within_top_pts,
                        entry_search_window_minutes=entry_search_window_minutes,
                        initial_stop_pts=initial_stop_pts,
                        trail_activate_profit_pts=trail_activate_profit_pts,
                        trailing_stop_pts=trailing_stop_pts,
                        take_profit_pts=take_profit_pts,
                        prior_ctx=prior_ctx,
                        max_prior_down_up_ratio=max_prior_down_up_ratio,
                        max_start_pct_of_range=max_start_pct_of_range,
                        observed_rows=observed_rows,
                        confirmed_range_high=obs["range_high"],
                        max_minutes_before_close=max_minutes_before_close,
                    )

                    row_out = {
                        "trade_date": str(trade_date),
                        "direction": "up",
                        "source_zone_low": round(float(zone["low"]), 2),
                        "source_zone_high": round(float(zone["high"]), 2),
                        "source_zone_width": round(float(zone["width"]), 2),
                        "source_zone_levels": zone["levels_text"],
                        "target_level": round(float(target_zone_up["low"]), 2),
                        "target_zone_range": f"{target_zone_up['low']:.2f} – {target_zone_up['high']:.2f}",
                        "clean_space_points": round(float(clean_space_up), 2),
                        "start_ts_pt": str(start_row.get("ts_pt")),
                        "start_ts_utc": pd.Timestamp(start_row.get("ts_utc")).isoformat(),
                        "start_open": round(float(start_open), 2) if start_open is not None else None,
                        "start_pivot_price": round(float(pivot_price), 2),
                        "start_context": "last pivot low in source zone",
                        "target_ts_pt": str(target_row.get("ts_pt")),
                        "target_ts_utc": pd.Timestamp(target_row.get("ts_utc")).isoformat(),
                        "target_open": round(float(target_open), 2) if target_open is not None else None,
                        "target_trigger_price": round(float(target_zone_up["low"]) - float(target_proximity_pts), 2),
                        "move_points": round(float(move_points), 2),
                        "elapsed_bars": int(hit_idx - pivot_idx),
                        **prior_ctx,
                        **setup_eval,
                    }
                    results.append(row_out)
                    diagnostics["valid_instances"] += 1

                    if setup_eval.get("short_setup_found"):
                        diagnostics["up_short_setups_found"] += 1
                        if len(diagnostics["sample_short_setups"]) < 8:
                            diagnostics["sample_short_setups"].append(
                                {
                                    "trade_date": str(trade_date),
                                    "start_ts_pt": str(start_row.get("ts_pt")),
                                    "target_ts_pt": str(target_row.get("ts_pt")),
                                    "signal_ts_pt": setup_eval.get("short_signal_ts_pt"),
                                    "target_level": round(float(target_zone_up["low"]), 2),
                                    "delta_put_skew_pct": setup_eval.get("short_signal_delta_put_skew_pct"),
                                    "delta_call_skew_pct": setup_eval.get("short_signal_delta_call_skew_pct"),
                                }
                            )

                    if setup_eval.get("trade_entry_found"):
                        diagnostics["actual_trades_found"] += 1
                        if setup_eval.get("trade_outcome") == "win":
                            diagnostics["winning_trades"] += 1
                        if len(diagnostics["sample_trades"]) < 8:
                            diagnostics["sample_trades"].append(
                                {
                                    "trade_date": str(trade_date),
                                    "signal_ts_pt": setup_eval.get("short_signal_ts_pt"),
                                    "entry_ts_pt": setup_eval.get("trade_entry_ts_pt"),
                                    "entry_price": setup_eval.get("trade_entry_price"),
                                    "exit_ts_pt": setup_eval.get("trade_exit_ts_pt"),
                                    "exit_price": setup_eval.get("trade_exit_price"),
                                    "exit_reason": setup_eval.get("trade_exit_reason"),
                                    "realized_points": setup_eval.get("trade_realized_points"),
                                }
                            )

                    if len(results) >= int(max_results):
                        return results, diagnostics

                    # Range confirmed and result recorded — stop walking up zones.
                    break

        if zone_idx > 0:
            # Gather all zones below as potential targets (walking downward)
            candidate_target_zone_indices_down = list(range(zone_idx - 1, -1, -1))

            for seg_start, seg_end in episodes:
                diagnostics["zone_episodes_considered"] += 1
                seg_max_high = max(float(day_rows[i]["high"]) for i in range(seg_start, seg_end + 1))
                if seg_max_high > float(zone["high"]) + float(max_zone_breach_pts):
                    continue

                pivot_idx, pivot_price = _find_last_pivot(day_rows, seg_start, seg_end, "down", pivot_strength_bars)

                # min_minutes_after_open filter for down moves
                if min_minutes_after_open > 0:
                    pivot_pt = _normalize_pt_label(day_rows[pivot_idx].get("ts_pt"))
                    if pivot_pt:
                        try:
                            ph, pm = int(pivot_pt[:2]), int(pivot_pt[3:5])
                            pivot_minutes_since_open = (ph - 6) * 60 + pm - 30
                            if pivot_minutes_since_open < int(min_minutes_after_open):
                                continue
                        except Exception:
                            pass

                # Walk down through candidate target zones, promoting if price
                # pushes through a wall before consolidating.
                scan_from_idx = seg_end + 1
                final_hit_idx = None
                for target_zone_offset, t_zone_idx in enumerate(candidate_target_zone_indices_down):
                    target_zone_down = zones[t_zone_idx]
                    clean_space_down = float(zone["low"]) - float(target_zone_down["high"])
                    if clean_space_down < float(min_clean_move_points):
                        continue

                    if target_zone_offset == 0:
                        diagnostics["source_zones_with_clean_targets"] += 1

                    hit_idx = _first_target_hit(
                        day_rows,
                        start_scan_idx=scan_from_idx,
                        direction="down",
                        zone=zone,
                        target_level=float(target_zone_down["high"]),
                        target_proximity_pts=target_proximity_pts,
                        max_zone_breach_pts=max_zone_breach_pts,
                    )
                    if hit_idx is None:
                        break

                    final_hit_idx = hit_idx
                    move_points = float(pivot_price) - float(target_zone_down["high"])

                    # Observe consolidation from target hit. If price promotes
                    # (pushes below wall low), advance to next zone down.
                    obs = _observe_consolidation_range_down(
                        day_rows=day_rows,
                        start_idx=hit_idx,
                        consolidation_window_minutes=consolidation_window_minutes,
                        wall_high=float(target_zone_down["high"]),
                        wall_low=float(target_zone_down["low"]),
                        max_zone_breach_pts=float(max_zone_breach_pts),
                        pivot_price=float(pivot_price),
                        move_points=float(move_points),
                        max_move_loss_pct=float(max_move_loss_pct),
                    )

                    if obs["status"] == "promoted":
                        scan_from_idx = obs["promote_idx"]
                        continue

                    if obs["status"] in ("invalidated", "insufficient"):
                        break

                    # Confirmed — evaluate setup
                    observed_rows = obs["rows"]
                    start_row   = day_rows[pivot_idx]
                    target_row  = day_rows[final_hit_idx]
                    start_open  = start_row.get("open")
                    target_open = target_row.get("open")

                    prior_ctx = _compute_prior_move_context_down(
                        day_rows,
                        pivot_idx=pivot_idx,
                        pivot_price=pivot_price,
                        down_move_pts=float(move_points),
                    )

                    setup_eval = _evaluate_down_long_setup(
                        trade_date=trade_date,
                        day_rows=day_rows,
                        entry_idx=pivot_idx,
                        target_hit_idx=final_hit_idx,
                        target_level=float(target_zone_down["high"]),
                        consolidation_window_minutes=consolidation_window_minutes,
                        long_put_skew_min_decrease_pct=long_put_skew_min_decrease_pct,
                        long_call_skew_min_increase_pct=long_call_skew_min_increase_pct,
                        entry_within_bottom_pts=entry_within_top_pts,
                        entry_search_window_minutes=entry_search_window_minutes,
                        long_initial_stop_pts=long_initial_stop_pts,
                        long_trail_activate_profit_pts=long_trail_activate_profit_pts,
                        long_trailing_stop_pts=long_trailing_stop_pts,
                        long_take_profit_pts=long_take_profit_pts,
                        prior_ctx=prior_ctx,
                        max_prior_down_up_ratio=max_prior_down_up_ratio,
                        max_start_pct_of_range=max_start_pct_of_range,
                        source_zone_low=float(zone["low"]),
                        source_zone_high=float(zone["high"]),
                        wall_low=float(target_zone_down["low"]),
                        max_move_loss_pct=max_move_loss_pct,
                        pivot_price=float(pivot_price),
                        move_points=float(move_points),
                        observed_rows=observed_rows,
                        confirmed_range_low=obs["range_low"],
                        max_minutes_before_close=max_minutes_before_close,
                    )

                    results.append(
                        {
                            "trade_date":           str(trade_date),
                            "direction":            "down",
                            "source_zone_low":      round(float(zone["low"]),  2),
                            "source_zone_high":     round(float(zone["high"]), 2),
                            "source_zone_width":    round(float(zone["width"]),2),
                            "source_zone_levels":   zone["levels_text"],
                            "target_level":         round(float(target_zone_down["high"]), 2),
                            "target_zone_range":    f"{target_zone_down['low']:.2f} – {target_zone_down['high']:.2f}",
                            "clean_space_points":   round(float(clean_space_down), 2),
                            "start_ts_pt":          str(start_row.get("ts_pt")),
                            "start_ts_utc":         pd.Timestamp(start_row.get("ts_utc")).isoformat(),
                            "start_open":           round(float(start_open), 2) if start_open is not None else None,
                            "start_pivot_price":    round(float(pivot_price), 2),
                            "start_context":        "last pivot high in source zone",
                            "target_ts_pt":         str(target_row.get("ts_pt")),
                            "target_ts_utc":        pd.Timestamp(target_row.get("ts_utc")).isoformat(),
                            "target_open":          round(float(target_open), 2) if target_open is not None else None,
                            "target_trigger_price": round(float(target_zone_down["high"]) + float(target_proximity_pts), 2),
                            "move_points":          round(float(move_points), 2),
                            "elapsed_bars":         int(final_hit_idx - pivot_idx),
                            **prior_ctx,
                            **setup_eval,
                        }
                    )
                    diagnostics["valid_instances"] += 1

                    if setup_eval.get("long_setup_found"):
                        diagnostics["up_short_setups_found"] += 1

                    if setup_eval.get("trade_entry_found"):
                        diagnostics["actual_trades_found"] += 1
                        if setup_eval.get("trade_outcome") == "win":
                            diagnostics["winning_trades"] += 1

                    if len(results) >= int(max_results):
                        return results, diagnostics

                    break

    # Deduplicate down-move rows: if multiple source zones produced a result
    # with the same target level and target bar, keep only the one with the
    # largest move_points (highest source zone = cleanest move).
    seen_down: dict = {}
    deduped: List[Dict[str, Any]] = []
    for row in results:
        if row.get("direction") != "down":
            deduped.append(row)
            continue
        key = (row.get("trade_date"), row.get("target_level"), row.get("target_ts_pt"))
        if key not in seen_down or (row.get("move_points") or 0) > (seen_down[key].get("move_points") or 0):
            seen_down[key] = row
    for row in results:
        if row.get("direction") == "down":
            key = (row.get("trade_date"), row.get("target_level"), row.get("target_ts_pt"))
            if seen_down.get(key) is row:
                deduped.append(row)

    return deduped, diagnostics


def scan_gex_level_moves(
    *,
    start_date: str,
    end_date: str,
    min_level_gex_bn: float,
    zone_merge_distance_pts: float,
    min_clean_move_points: float,
    target_proximity_pts: float,
    max_zone_breach_pts: float,
    pivot_strength_bars: int,
    level_family: str,
    max_results: int,
    consolidation_window_minutes: int,
    short_put_skew_increase_pct: float,
    short_call_skew_max_pct: float,
    entry_within_top_pts: float,
    entry_search_window_minutes: int,
    initial_stop_pts: float,
    trail_activate_profit_pts: float,
    trailing_stop_pts: float,
    take_profit_pts: float,
    max_prior_down_up_ratio: float = 2.0,
    max_start_pct_of_range: float = 0.20,
    max_move_loss_pct: float = 0.75,
    min_minutes_after_open: int = 15,
    long_put_skew_min_decrease_pct: float = 80.0,
    long_call_skew_min_increase_pct: float = 30.0,
    max_minutes_before_close: int = 45,
    long_initial_stop_pts: float = 10.0,
    long_trail_activate_profit_pts: float = 20.0,
    long_trailing_stop_pts: float = 10.0,
    long_take_profit_pts: float = 35.0,
    source_view: str | None = None,
) -> Dict[str, Any]:
    level_family = (level_family or "primary").strip().lower()
    if level_family not in {"primary", "strong", "both"}:
        raise ValueError("level_family must be one of: primary, strong, both")

    df = load_source_rows(start_date=start_date, end_date=end_date, source_view=source_view)
    source_view = safe_ident(source_view or DEFAULT_SOURCE_VIEW)

    if df.empty:
        return {
            "rows": [],
            "summary": {
                "source_view": source_view,
                "days_scanned": 0,
                "bars_scanned": 0,
                "instances_found": 0,
                "zones_total": 0,
                "up_short_setups_found": 0,
                "actual_trades_found": 0,
                "winning_trades": 0,
            },
            "diagnostics": {
                "bars_total": 0,
                "days_total": 0,
                "qualifying_levels_seen": 0,
                "zones_total": 0,
                "source_zones_with_clean_targets": 0,
                "zone_episodes_considered": 0,
                "valid_instances": 0,
                "up_short_setups_found": 0,
                "actual_trades_found": 0,
                "winning_trades": 0,
                "sample_zones": [],
                "sample_results": [],
                "sample_short_setups": [],
                "sample_trades": [],
            },
        }

    results: List[Dict[str, Any]] = []
    total_qualifying_levels = 0
    total_zones = 0
    total_source_zones = 0
    total_zone_episodes = 0
    total_short_setups = 0
    total_actual_trades = 0
    total_winning_trades = 0
    sample_zones: List[Dict[str, Any]] = []
    sample_short_setups: List[Dict[str, Any]] = []
    sample_trades: List[Dict[str, Any]] = []

    for trade_date, day_df in df.groupby("trade_date", sort=True):
        day_rows = day_df.to_dict("records")
        day_results, day_diag = _day_zone_results(
            trade_date,
            day_rows,
            level_family=level_family,
            min_level_gex_bn=float(min_level_gex_bn),
            zone_merge_distance_pts=float(zone_merge_distance_pts),
            min_clean_move_points=float(min_clean_move_points),
            target_proximity_pts=float(target_proximity_pts),
            max_zone_breach_pts=float(max_zone_breach_pts),
            pivot_strength_bars=int(pivot_strength_bars),
            max_results=max(0, int(max_results) - len(results)),
            consolidation_window_minutes=int(consolidation_window_minutes),
            short_put_skew_increase_pct=float(short_put_skew_increase_pct),
            short_call_skew_max_pct=float(short_call_skew_max_pct),
            entry_within_top_pts=float(entry_within_top_pts),
            entry_search_window_minutes=int(entry_search_window_minutes),
            initial_stop_pts=float(initial_stop_pts),
            trail_activate_profit_pts=float(trail_activate_profit_pts),
            trailing_stop_pts=float(trailing_stop_pts),
            take_profit_pts=float(take_profit_pts),
            max_prior_down_up_ratio=float(max_prior_down_up_ratio),
            max_start_pct_of_range=float(max_start_pct_of_range),
            max_move_loss_pct=float(max_move_loss_pct),
            min_minutes_after_open=int(min_minutes_after_open),
            long_put_skew_min_decrease_pct=float(long_put_skew_min_decrease_pct),
            long_call_skew_min_increase_pct=float(long_call_skew_min_increase_pct),
            max_minutes_before_close=int(max_minutes_before_close),
            long_initial_stop_pts=float(long_initial_stop_pts),
            long_trail_activate_profit_pts=float(long_trail_activate_profit_pts),
            long_trailing_stop_pts=float(long_trailing_stop_pts),
            long_take_profit_pts=float(long_take_profit_pts),
        )
        results.extend(day_results)

        total_qualifying_levels += int(day_diag["qualifying_levels"])
        total_zones += int(day_diag["zones_total"])
        total_source_zones += int(day_diag["source_zones_with_clean_targets"])
        total_zone_episodes += int(day_diag["zone_episodes_considered"])
        total_short_setups += int(day_diag["up_short_setups_found"])
        total_actual_trades += int(day_diag["actual_trades_found"])
        total_winning_trades += int(day_diag["winning_trades"])

        for zone in day_diag["sample_zones"]:
            if len(sample_zones) >= 8:
                break
            sample_zones.append({"trade_date": str(trade_date), **zone})

        for item in day_diag.get("sample_short_setups", []):
            if len(sample_short_setups) >= 8:
                break
            sample_short_setups.append(item)

        for item in day_diag.get("sample_trades", []):
            if len(sample_trades) >= 8:
                break
            sample_trades.append(item)

        if len(results) >= int(max_results):
            break

    diagnostics = {
        "bars_total": int(len(df)),
        "days_total": int(df["trade_date"].nunique()),
        "qualifying_levels_seen": int(total_qualifying_levels),
        "zones_total": int(total_zones),
        "source_zones_with_clean_targets": int(total_source_zones),
        "zone_episodes_considered": int(total_zone_episodes),
        "valid_instances": int(len(results)),
        "up_short_setups_found": int(total_short_setups),
        "actual_trades_found": int(total_actual_trades),
        "winning_trades": int(total_winning_trades),
        "sample_zones": sample_zones,
        "sample_results": [
            {
                "trade_date": row["trade_date"],
                "direction": row["direction"],
                "source_zone": f"{row['source_zone_low']:.2f} – {row['source_zone_high']:.2f}",
                "target_level": row["target_level"],
                "start_ts_pt": row["start_ts_pt"],
                "target_ts_pt": row["target_ts_pt"],
                "clean_space_points": row["clean_space_points"],
            }
            for row in results[:8]
        ],
        "sample_short_setups": sample_short_setups,
        "sample_trades": sample_trades,
    }

    return {
        "rows": results,
        "summary": {
            "source_view": source_view,
            "days_scanned": int(df["trade_date"].nunique()),
            "bars_scanned": int(len(df)),
            "instances_found": int(len(results)),
            "zones_total": int(total_zones),
            "up_short_setups_found": int(total_short_setups),
            "actual_trades_found": int(total_actual_trades),
            "winning_trades": int(total_winning_trades),
        },
        "diagnostics": diagnostics,
    }