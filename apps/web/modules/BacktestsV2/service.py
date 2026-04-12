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


def _evaluate_up_short_setup(
    *,
    trade_date: dt.date,
    day_rows: List[Dict[str, Any]],
    entry_idx: int,
    target_hit_idx: int,
    target_level: float,
    target_proximity_pts: float,
    consolidation_window_minutes: int,
    short_put_skew_increase_pct: float,
    short_call_skew_max_pct: float,
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
    }

    max_bars = max(1, int(consolidation_window_minutes))
    end_idx = min(len(day_rows) - 1, target_hit_idx + max_bars - 1)

    consolidation_rows: List[Dict[str, Any]] = []
    for i in range(target_hit_idx, end_idx + 1):
        row = day_rows[i]
        close_v = row.get("close")
        if close_v is None:
            break
        try:
            close_f = float(close_v)
        except Exception:
            break

        if abs(close_f - float(target_level)) <= float(target_proximity_pts):
            consolidation_rows.append(row)
        else:
            break

    if not consolidation_rows:
        default["short_setup_reason"] = "no_target_consolidation"
        return default

    entry_ts_pt = _normalize_pt_label(day_rows[entry_idx].get("ts_pt"))
    consolidation_times = [_normalize_pt_label(r.get("ts_pt")) for r in consolidation_rows]
    skew_times = [entry_ts_pt] + consolidation_times

    skew_df = _fetch_skew_rows_for_times(str(trade_date), str(trade_date), skew_times)
    if skew_df.empty:
        default["consolidation_minutes_observed"] = len(consolidation_rows)
        default["consolidation_end_ts_pt"] = _normalize_pt_label(consolidation_rows[-1].get("ts_pt"))
        default["short_setup_reason"] = "no_skew_data"
        return default

    by_label: Dict[str, pd.Series] = {}
    for _, row in skew_df.iterrows():
        label = str(row.get("snapshot_pt_label") or "")
        if label and label not in by_label:
            by_label[label] = row

    entry_skew_row = by_label.get(entry_ts_pt)
    if entry_skew_row is None:
        default["consolidation_minutes_observed"] = len(consolidation_rows)
        default["consolidation_end_ts_pt"] = _normalize_pt_label(consolidation_rows[-1].get("ts_pt"))
        default["short_setup_reason"] = "missing_entry_skew"
        return default

    latest_metrics: Dict[str, Optional[float]] = {
        "delta_atm_iv_pct": None,
        "delta_call_skew_pct": None,
        "delta_put_skew_pct": None,
    }

    for row in consolidation_rows:
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
            return {
                "consolidation_minutes_observed": len(consolidation_rows),
                "consolidation_end_ts_pt": _normalize_pt_label(consolidation_rows[-1].get("ts_pt")),
                "short_setup_found": True,
                "short_signal_ts_pt": label,
                "short_signal_price": round(float(row["close"]), 2) if row.get("close") is not None else None,
                "short_signal_delta_atm_iv_pct": None if metrics["delta_atm_iv_pct"] is None else round(float(metrics["delta_atm_iv_pct"]), 2),
                "short_signal_delta_call_skew_pct": round(float(d_call), 2),
                "short_signal_delta_put_skew_pct": round(float(d_put), 2),
                "short_setup_reason": "threshold_hit",
            }

    return {
        "consolidation_minutes_observed": len(consolidation_rows),
        "consolidation_end_ts_pt": _normalize_pt_label(consolidation_rows[-1].get("ts_pt")),
        "short_setup_found": False,
        "short_signal_ts_pt": None,
        "short_signal_price": None,
        "short_signal_delta_atm_iv_pct": None if latest_metrics["delta_atm_iv_pct"] is None else round(float(latest_metrics["delta_atm_iv_pct"]), 2),
        "short_signal_delta_call_skew_pct": None if latest_metrics["delta_call_skew_pct"] is None else round(float(latest_metrics["delta_call_skew_pct"]), 2),
        "short_signal_delta_put_skew_pct": None if latest_metrics["delta_put_skew_pct"] is None else round(float(latest_metrics["delta_put_skew_pct"]), 2),
        "short_setup_reason": "threshold_not_met",
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
    }

    if not zones:
        return [], diagnostics

    results: List[Dict[str, Any]] = []

    for zone_idx, zone in enumerate(zones):
        masks = [_in_zone_close(row, zone, max_zone_breach_pts) for row in day_rows]
        episodes = _true_segments(masks)

        if zone_idx < len(zones) - 1:
            target_zone_up = zones[zone_idx + 1]
            clean_space_up = float(target_zone_up["low"]) - float(zone["high"])
            if clean_space_up >= float(min_clean_move_points):
                diagnostics["source_zones_with_clean_targets"] += 1
                for seg_start, seg_end in episodes:
                    diagnostics["zone_episodes_considered"] += 1
                    seg_min_low = min(float(day_rows[i]["low"]) for i in range(seg_start, seg_end + 1))
                    if seg_min_low < float(zone["low"]) - float(max_zone_breach_pts):
                        continue

                    pivot_idx, pivot_price = _find_last_pivot(day_rows, seg_start, seg_end, "up", pivot_strength_bars)
                    hit_idx = _first_target_hit(
                        day_rows,
                        start_scan_idx=seg_end + 1,
                        direction="up",
                        zone=zone,
                        target_level=float(target_zone_up["low"]),
                        target_proximity_pts=target_proximity_pts,
                        max_zone_breach_pts=max_zone_breach_pts,
                    )
                    if hit_idx is None:
                        continue

                    start_row = day_rows[pivot_idx]
                    target_row = day_rows[hit_idx]
                    start_open = start_row.get("open")
                    target_open = target_row.get("open")
                    move_points = float(target_zone_up["low"]) - float(pivot_price)

                    setup_eval = _evaluate_up_short_setup(
                        trade_date=trade_date,
                        day_rows=day_rows,
                        entry_idx=pivot_idx,
                        target_hit_idx=hit_idx,
                        target_level=float(target_zone_up["low"]),
                        target_proximity_pts=target_proximity_pts,
                        consolidation_window_minutes=consolidation_window_minutes,
                        short_put_skew_increase_pct=short_put_skew_increase_pct,
                        short_call_skew_max_pct=short_call_skew_max_pct,
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

                    if len(results) >= int(max_results):
                        return results, diagnostics

        if zone_idx > 0:
            target_zone_down = zones[zone_idx - 1]
            clean_space_down = float(zone["low"]) - float(target_zone_down["high"])
            if clean_space_down >= float(min_clean_move_points):
                diagnostics["source_zones_with_clean_targets"] += 1
                for seg_start, seg_end in episodes:
                    diagnostics["zone_episodes_considered"] += 1
                    seg_max_high = max(float(day_rows[i]["high"]) for i in range(seg_start, seg_end + 1))
                    if seg_max_high > float(zone["high"]) + float(max_zone_breach_pts):
                        continue

                    pivot_idx, pivot_price = _find_last_pivot(day_rows, seg_start, seg_end, "down", pivot_strength_bars)
                    hit_idx = _first_target_hit(
                        day_rows,
                        start_scan_idx=seg_end + 1,
                        direction="down",
                        zone=zone,
                        target_level=float(target_zone_down["high"]),
                        target_proximity_pts=target_proximity_pts,
                        max_zone_breach_pts=max_zone_breach_pts,
                    )
                    if hit_idx is None:
                        continue

                    start_row = day_rows[pivot_idx]
                    target_row = day_rows[hit_idx]
                    start_open = start_row.get("open")
                    target_open = target_row.get("open")
                    move_points = float(pivot_price) - float(target_zone_down["high"])

                    results.append(
                        {
                            "trade_date": str(trade_date),
                            "direction": "down",
                            "source_zone_low": round(float(zone["low"]), 2),
                            "source_zone_high": round(float(zone["high"]), 2),
                            "source_zone_width": round(float(zone["width"]), 2),
                            "source_zone_levels": zone["levels_text"],
                            "target_level": round(float(target_zone_down["high"]), 2),
                            "target_zone_range": f"{target_zone_down['low']:.2f} – {target_zone_down['high']:.2f}",
                            "clean_space_points": round(float(clean_space_down), 2),
                            "start_ts_pt": str(start_row.get("ts_pt")),
                            "start_ts_utc": pd.Timestamp(start_row.get("ts_utc")).isoformat(),
                            "start_open": round(float(start_open), 2) if start_open is not None else None,
                            "start_pivot_price": round(float(pivot_price), 2),
                            "start_context": "last pivot high in source zone",
                            "target_ts_pt": str(target_row.get("ts_pt")),
                            "target_ts_utc": pd.Timestamp(target_row.get("ts_utc")).isoformat(),
                            "target_open": round(float(target_open), 2) if target_open is not None else None,
                            "target_trigger_price": round(float(target_zone_down["high"]) + float(target_proximity_pts), 2),
                            "move_points": round(float(move_points), 2),
                            "elapsed_bars": int(hit_idx - pivot_idx),
                            "consolidation_minutes_observed": 0,
                            "consolidation_end_ts_pt": None,
                            "short_setup_found": False,
                            "short_signal_ts_pt": None,
                            "short_signal_price": None,
                            "short_signal_delta_atm_iv_pct": None,
                            "short_signal_delta_call_skew_pct": None,
                            "short_signal_delta_put_skew_pct": None,
                            "short_setup_reason": "down_move_not_evaluated",
                        }
                    )
                    diagnostics["valid_instances"] += 1
                    if len(results) >= int(max_results):
                        return results, diagnostics

    return results, diagnostics


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
                "sample_zones": [],
                "sample_results": [],
                "sample_short_setups": [],
            },
        }

    results: List[Dict[str, Any]] = []
    total_qualifying_levels = 0
    total_zones = 0
    total_source_zones = 0
    total_zone_episodes = 0
    total_short_setups = 0
    sample_zones: List[Dict[str, Any]] = []
    sample_short_setups: List[Dict[str, Any]] = []

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
        )
        results.extend(day_results)

        total_qualifying_levels += int(day_diag["qualifying_levels"])
        total_zones += int(day_diag["zones_total"])
        total_source_zones += int(day_diag["source_zones_with_clean_targets"])
        total_zone_episodes += int(day_diag["zone_episodes_considered"])
        total_short_setups += int(day_diag["up_short_setups_found"])

        for zone in day_diag["sample_zones"]:
            if len(sample_zones) >= 8:
                break
            sample_zones.append({"trade_date": str(trade_date), **zone})

        for item in day_diag.get("sample_short_setups", []):
            if len(sample_short_setups) >= 8:
                break
            sample_short_setups.append(item)

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
        },
        "diagnostics": diagnostics,
    }