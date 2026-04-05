from __future__ import annotations

import os
import math
import datetime as dt
from functools import lru_cache
from typing import Any, Dict, List, Tuple

import pandas as pd
from sqlalchemy import create_engine, text

DEFAULT_SOURCE_VIEW = os.getenv("BT2_SOURCE_VIEW", os.getenv("BT_VIEW_NAME", "es_minutes_with_features_bt"))


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
    # Supports both raw-dollar GEX and already-normalized BN values.
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
    out: List[Dict[str, Any]] = []
    for item in _row_levels(row, level_family):
        if abs(item["gex_bn"]) >= float(min_level_gex_bn):
            out.append(item)
    return out


def _touched_levels(
    row: Dict[str, Any],
    level_family: str,
    min_level_gex_bn: float,
    touch_buffer_points: float,
) -> List[Dict[str, Any]]:
    low = row.get("low")
    high = row.get("high")
    if low is None or high is None:
        return []

    try:
        low_v = float(low)
        high_v = float(high)
    except Exception:
        return []

    out: List[Dict[str, Any]] = []
    for item in _qualifying_levels(row, level_family, min_level_gex_bn):
        level_v = float(item["level"])
        if (low_v - touch_buffer_points) <= level_v <= (high_v + touch_buffer_points):
            out.append(item)
    return out


def _nearest_qualifying_distance(row: Dict[str, Any], level_family: str, min_level_gex_bn: float) -> float | None:
    close_v = row.get("close")
    if close_v is None:
        return None
    try:
        close_f = float(close_v)
    except Exception:
        return None

    levels = _qualifying_levels(row, level_family, min_level_gex_bn)
    if not levels:
        return None
    return min(abs(float(item["level"]) - close_f) for item in levels)


def _diagnostics_payload(
    df: pd.DataFrame,
    *,
    level_family: str,
    min_level_gex_bn: float,
    touch_buffer_points: float,
) -> Dict[str, Any]:
    diag: Dict[str, Any] = {
        "bars_total": int(len(df)),
        "days_total": int(df["trade_date"].nunique()) if not df.empty else 0,
        "rows_with_any_level": 0,
        "rows_with_any_qualifying_level": 0,
        "bars_touching_qualifying_level": 0,
        "bars_within_0_5_pts": 0,
        "bars_within_1_pt": 0,
        "bars_within_2_pts": 0,
        "bars_within_5_pts": 0,
        "bars_within_10_pts": 0,
        "start_candidates": 0,
        "sample_qualifying_rows": [],
        "column_stats": {},
    }

    selected_defs = _selected_level_defs(level_family)
    for _family, label, _level_col, _gex_col in selected_defs:
        diag["column_stats"][label] = {
            "rows_with_level": 0,
            "rows_meeting_gex_threshold": 0,
            "max_abs_gex_bn": None,
        }

    if df.empty:
        return diag

    day_start_candidates = 0
    sample_rows: List[Dict[str, Any]] = []

    for trade_date, day_df in df.groupby("trade_date", sort=True):
        prev_touch_keys: set[str] = set()
        for row in day_df.to_dict("records"):
            all_levels = _row_levels(row, level_family)
            if all_levels:
                diag["rows_with_any_level"] += 1

            for item in all_levels:
                stat = diag["column_stats"][item["label"]]
                stat["rows_with_level"] += 1
                max_abs = abs(float(item["gex_bn"]))
                current_max = stat["max_abs_gex_bn"]
                if current_max is None or max_abs > current_max:
                    stat["max_abs_gex_bn"] = max_abs
                if max_abs >= float(min_level_gex_bn):
                    stat["rows_meeting_gex_threshold"] += 1

            qualifying = _qualifying_levels(row, level_family, min_level_gex_bn)
            if qualifying:
                diag["rows_with_any_qualifying_level"] += 1
                if len(sample_rows) < 8:
                    nearest_dist = _nearest_qualifying_distance(row, level_family, min_level_gex_bn)
                    sample_rows.append(
                        {
                            "trade_date": str(trade_date),
                            "ts_pt": str(row.get("ts_pt")),
                            "close": round(float(row["close"]), 2) if row.get("close") is not None else None,
                            "nearest_distance": round(float(nearest_dist), 2) if nearest_dist is not None else None,
                            "levels": [
                                f"{item['label']} @ {item['level']:.2f} ({item['gex_bn']:.2f} BN)"
                                for item in qualifying[:3]
                            ],
                        }
                    )

            nearest_dist = _nearest_qualifying_distance(row, level_family, min_level_gex_bn)
            if nearest_dist is not None:
                if nearest_dist <= 0.5:
                    diag["bars_within_0_5_pts"] += 1
                if nearest_dist <= 1.0:
                    diag["bars_within_1_pt"] += 1
                if nearest_dist <= 2.0:
                    diag["bars_within_2_pts"] += 1
                if nearest_dist <= 5.0:
                    diag["bars_within_5_pts"] += 1
                if nearest_dist <= 10.0:
                    diag["bars_within_10_pts"] += 1

            touched = _touched_levels(row, level_family, min_level_gex_bn, touch_buffer_points)
            if touched:
                diag["bars_touching_qualifying_level"] += 1
                current_keys = {f"{item['label']}|{item['level']:.2f}" for item in touched}
                day_start_candidates += sum(1 for key in current_keys if key not in prev_touch_keys)
                prev_touch_keys = current_keys
            else:
                prev_touch_keys = set()

    diag["start_candidates"] = int(day_start_candidates)
    diag["sample_qualifying_rows"] = sample_rows

    for stat in diag["column_stats"].values():
        if stat["max_abs_gex_bn"] is not None:
            stat["max_abs_gex_bn"] = round(float(stat["max_abs_gex_bn"]), 2)

    return diag


def scan_gex_level_moves(
    *,
    start_date: str,
    end_date: str,
    min_level_gex_bn: float,
    min_move_points: float,
    touch_buffer_points: float,
    level_family: str,
    max_results: int,
    source_view: str | None = None,
) -> Dict[str, Any]:
    level_family = (level_family or "primary").strip().lower()
    if level_family not in {"primary", "strong", "both"}:
        raise ValueError("level_family must be one of: primary, strong, both")

    df = load_source_rows(start_date=start_date, end_date=end_date, source_view=source_view)
    source_view = safe_ident(source_view or DEFAULT_SOURCE_VIEW)

    diagnostics = _diagnostics_payload(
        df,
        level_family=level_family,
        min_level_gex_bn=float(min_level_gex_bn),
        touch_buffer_points=float(touch_buffer_points),
    )

    if df.empty:
        return {
            "rows": [],
            "summary": {
                "source_view": source_view,
                "days_scanned": 0,
                "bars_scanned": 0,
                "instances_found": 0,
            },
            "diagnostics": diagnostics,
        }

    results: List[Dict[str, Any]] = []

    for trade_date, day_df in df.groupby("trade_date", sort=True):
        day_rows = day_df.to_dict("records")
        touched_by_idx: List[List[Dict[str, Any]]] = []

        for row in day_rows:
            touched_by_idx.append(
                _touched_levels(
                    row=row,
                    level_family=level_family,
                    min_level_gex_bn=min_level_gex_bn,
                    touch_buffer_points=touch_buffer_points,
                )
            )

        prev_touch_keys: set[str] = set()
        for i, row in enumerate(day_rows):
            touched_here = touched_by_idx[i]
            if not touched_here:
                prev_touch_keys = set()
                continue

            start_candidates: List[Dict[str, Any]] = []
            current_keys = set()
            for lvl in touched_here:
                key = f"{lvl['label']}|{lvl['level']:.2f}"
                current_keys.add(key)
                if key not in prev_touch_keys:
                    start_candidates.append(lvl)

            prev_touch_keys = current_keys
            if not start_candidates:
                continue

            for start in start_candidates:
                for j in range(i + 1, len(day_rows)):
                    target_candidates = []
                    for target in touched_by_idx[j]:
                        move_points = abs(float(target["level"]) - float(start["level"]))
                        if move_points < float(min_move_points):
                            continue
                        if abs(float(target["level"]) - float(start["level"])) < 1e-9:
                            continue
                        target_candidates.append((target, move_points))

                    if not target_candidates:
                        continue

                    target, move_points = max(target_candidates, key=lambda x: x[1])
                    start_open = row.get("open")
                    target_open = day_rows[j].get("open")

                    results.append(
                        {
                            "trade_date": str(trade_date),
                            "start_ts_pt": str(row.get("ts_pt")),
                            "start_ts_utc": pd.Timestamp(row.get("ts_utc")).isoformat(),
                            "start_open": round(float(start_open), 2) if start_open is not None else None,
                            "start_level_type": start["label"],
                            "start_level": round(float(start["level"]), 2),
                            "start_level_gex_bn": round(float(start["gex_bn"]), 2),
                            "target_ts_pt": str(day_rows[j].get("ts_pt")),
                            "target_ts_utc": pd.Timestamp(day_rows[j].get("ts_utc")).isoformat(),
                            "target_open": round(float(target_open), 2) if target_open is not None else None,
                            "target_level_type": target["label"],
                            "target_level": round(float(target["level"]), 2),
                            "target_level_gex_bn": round(float(target["gex_bn"]), 2),
                            "move_points": round(float(move_points), 2),
                            "elapsed_bars": int(j - i),
                        }
                    )
                    break

                if len(results) >= int(max_results):
                    break

            if len(results) >= int(max_results):
                break

        if len(results) >= int(max_results):
            break

    diagnostics["instances_found"] = int(len(results))

    return {
        "rows": results,
        "summary": {
            "source_view": source_view,
            "days_scanned": int(df["trade_date"].nunique()),
            "bars_scanned": int(len(df)),
            "instances_found": int(len(results)),
        },
        "diagnostics": diagnostics,
    }
