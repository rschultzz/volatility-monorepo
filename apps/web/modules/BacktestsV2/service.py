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
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    qualifying_levels = _collect_day_levels(day_rows, level_family, min_level_gex_bn)
    zones = _build_zones(qualifying_levels, zone_merge_distance_pts)

    diagnostics = {
        "qualifying_levels": len(qualifying_levels),
        "zones_total": len(zones),
        "source_zones_with_clean_targets": 0,
        "zone_episodes_considered": 0,
        "valid_instances": 0,
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

                    results.append(
                        {
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
                        }
                    )
                    diagnostics["valid_instances"] += 1
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
            },
            "diagnostics": {
                "bars_total": 0,
                "days_total": 0,
                "qualifying_levels_seen": 0,
                "zones_total": 0,
                "source_zones_with_clean_targets": 0,
                "zone_episodes_considered": 0,
                "valid_instances": 0,
                "sample_zones": [],
                "sample_results": [],
            },
        }

    results: List[Dict[str, Any]] = []
    total_qualifying_levels = 0
    total_zones = 0
    total_source_zones = 0
    total_zone_episodes = 0
    sample_zones: List[Dict[str, Any]] = []

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
        )
        results.extend(day_results)

        total_qualifying_levels += int(day_diag["qualifying_levels"])
        total_zones += int(day_diag["zones_total"])
        total_source_zones += int(day_diag["source_zones_with_clean_targets"])
        total_zone_episodes += int(day_diag["zone_episodes_considered"])

        for zone in day_diag["sample_zones"]:
            if len(sample_zones) >= 8:
                break
            sample_zones.append({"trade_date": str(trade_date), **zone})

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
    }

    return {
        "rows": results,
        "summary": {
            "source_view": source_view,
            "days_scanned": int(df["trade_date"].nunique()),
            "bars_scanned": int(len(df)),
            "instances_found": int(len(results)),
            "zones_total": int(total_zones),
        },
        "diagnostics": diagnostics,
    }
