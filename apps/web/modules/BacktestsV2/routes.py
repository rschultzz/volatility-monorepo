from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import jsonify, request, send_from_directory
from sqlalchemy import create_engine, text

from .service import scan_gex_level_moves, DEFAULT_SOURCE_VIEW
from .strategy_registry import build_strategy_registry, serialize_strategy


_SELECTION_STATE: dict[str, Any] = {
    "seq": 0,
    "payload": None,
}


def _get_registry():
    return build_strategy_registry(scan_gex_level_moves)


def _default_strategy_key() -> str:
    return "up_move_short"


def _engine():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("DATABASE_URL is not set")
    return create_engine(db_url, pool_pre_ping=True)


def _normalize_payload_aliases(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(payload or {})

    if "entryRangeTopPts" in out and "entryWithinTopPts" not in out:
        out["entryWithinTopPts"] = out["entryRangeTopPts"]

    if "trailActivationProfitPts" in out and "trailActivateProfitPts" not in out:
        out["trailActivateProfitPts"] = out["trailActivationProfitPts"]

    return out


def _slugify(value: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (value or "").strip().lower())
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def _json_safe_value(value: Any) -> Any:
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)


def _sanitize_saved_params(params: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    clean: Dict[str, Any] = {}
    allowed_keys = set(defaults.keys())

    for key, value in (params or {}).items():
        if key not in allowed_keys:
            continue
        # Preserve arrays (like bypassFilters) as-is
        if isinstance(value, list):
            clean[key] = value
        else:
            clean[key] = _json_safe_value(value)

    return clean


def _normalize_notes(value: Any) -> str:
    return str(value or "").strip()


def _normalize_name(value: Any, fallback_key: str) -> str:
    name = str(value or "").strip()
    return name or fallback_key.replace("_", " ").title()


def _load_strategy_rows() -> Dict[str, Dict[str, Any]]:
    sql = text(
        """
        SELECT
            id,
            strategy_key,
            base_strategy_key,
            name,
            feature_version,
            tags,
            notes,
            saved_params
        FROM public.bt_strategies
        WHERE strategy_key IS NOT NULL
        """
    )

    out: Dict[str, Dict[str, Any]] = {}
    with _engine().connect() as conn:
        rows = conn.execute(sql).mappings().all()

    for row in rows:
        key = str(row.get("strategy_key") or "").strip()
        if not key:
            continue
        out[key] = dict(row)

    return out


def _legacy_base_registry_key(strategy_key: str, registry: Dict[str, Any]) -> str | None:
    key = str(strategy_key or "").strip()

    if key in registry:
        return key

    if key.startswith("up_move_short"):
        return "up_move_short" if "up_move_short" in registry else None

    if key.startswith("down_move"):
        return "down_move_scan" if "down_move_scan" in registry else None

    return None


def _resolve_strategy_payloads():
    registry = _get_registry()
    db_rows = _load_strategy_rows()

    payloads: Dict[str, Dict[str, Any]] = {}
    all_keys = set(registry.keys()) | set(db_rows.keys())

    for key in sorted(all_keys):
        db_row = db_rows.get(key) or {}
        explicit_base_key = str(db_row.get("base_strategy_key") or "").strip()
        base_registry_key = explicit_base_key or _legacy_base_registry_key(key, registry)

        if not base_registry_key or base_registry_key not in registry:
            continue

        spec = registry[base_registry_key]
        saved_params = db_row.get("saved_params") or {}
        if not isinstance(saved_params, dict):
            saved_params = {}

        merged_defaults = {**spec.defaults, **_sanitize_saved_params(saved_params, spec.defaults)}

        serialized = {
            **serialize_strategy(spec, defaults_override=merged_defaults),
            "key": key,
            "strategyId": db_row.get("id"),
            "savedParams": saved_params,
            "featureVersion": db_row.get("feature_version"),
            "dbName": db_row.get("name"),
            "displayName": db_row.get("name") or spec.label,
            "strategyKey": key,
            "baseStrategyKey": base_registry_key,
            "notes": db_row.get("notes") or "",
            "tags": db_row.get("tags") or [],
        }

        payloads[key] = {
            "spec": spec,
            "db_row": db_row,
            "base_registry_key": base_registry_key,
            "serialized": serialized,
        }

    return payloads


def _save_strategy_defaults(strategy_key: str, params: Dict[str, Any], *, name: Any = None, notes: Any = None) -> Dict[str, Any]:
    resolved = _resolve_strategy_payloads()
    item = resolved.get(strategy_key)
    if item is None:
        raise ValueError(f"Unknown strategy key: {strategy_key}")

    spec = item["spec"]
    defaults = spec.defaults
    clean = _sanitize_saved_params(params, defaults)
    normalized_name = _normalize_name(name, strategy_key)
    normalized_notes = _normalize_notes(notes)

    sql = text(
        """
        UPDATE public.bt_strategies
        SET
            name = :name,
            notes = :notes,
            base_strategy_key = :base_strategy_key,
            saved_params = CAST(:saved_params AS jsonb)
        WHERE strategy_key = :strategy_key
        RETURNING id, strategy_key, base_strategy_key, name, feature_version, tags, notes, saved_params
        """
    )

    with _engine().begin() as conn:
        row = conn.execute(
            sql,
            {
                "strategy_key": strategy_key,
                "name": normalized_name,
                "notes": normalized_notes,
                "base_strategy_key": item["base_registry_key"],
                "saved_params": json.dumps(clean),
            },
        ).mappings().first()

    if row is None:
        raise ValueError(f"Could not save defaults for strategy_key={strategy_key}")

    merged_defaults = {**defaults, **clean}
    return {
        **serialize_strategy(spec, defaults_override=merged_defaults),
        "key": row.get("strategy_key"),
        "strategyId": row.get("id"),
        "savedParams": row.get("saved_params") or {},
        "featureVersion": row.get("feature_version"),
        "dbName": row.get("name"),
        "displayName": row.get("name") or spec.label,
        "strategyKey": row.get("strategy_key"),
        "baseStrategyKey": row.get("base_strategy_key") or item["base_registry_key"],
        "notes": row.get("notes") or "",
        "tags": row.get("tags") or [],
    }


def _create_strategy_from_existing(base_strategy_key: str, new_name: Any, new_strategy_key: Any, params: Dict[str, Any], notes: Any = None) -> Dict[str, Any]:
    resolved = _resolve_strategy_payloads()
    item = resolved.get(base_strategy_key)
    if item is None:
        raise ValueError(f"Unknown base strategy key: {base_strategy_key}")

    spec = item["spec"]
    base_registry_key = item["base_registry_key"]

    candidate_key = str(new_strategy_key or "").strip()
    if not candidate_key:
        candidate_key = _slugify(str(new_name or ""))

    if not candidate_key:
        raise ValueError("A strategy key is required")

    if candidate_key in resolved:
        raise ValueError(f"Strategy key already exists: {candidate_key}")

    clean = _sanitize_saved_params(params, spec.defaults)
    normalized_name = _normalize_name(new_name, candidate_key)
    normalized_notes = _normalize_notes(notes)

    sql = text(
        """
        INSERT INTO public.bt_strategies
            (name, strategy_key, base_strategy_key, feature_version, tags, notes, saved_params)
        VALUES
            (:name, :strategy_key, :base_strategy_key, :feature_version, :tags, :notes, CAST(:saved_params AS jsonb))
        RETURNING id, strategy_key, base_strategy_key, name, feature_version, tags, notes, saved_params
        """
    )

    with _engine().begin() as conn:
        row = conn.execute(
            sql,
            {
                "name": normalized_name,
                "strategy_key": candidate_key,
                "base_strategy_key": base_registry_key,
                "feature_version": "v2",
                "tags": ["backtests_v2"],
                "notes": normalized_notes,
                "saved_params": json.dumps(clean),
            },
        ).mappings().first()

    merged_defaults = {**spec.defaults, **clean}
    return {
        **serialize_strategy(spec, defaults_override=merged_defaults),
        "key": row.get("strategy_key"),
        "strategyId": row.get("id"),
        "savedParams": row.get("saved_params") or {},
        "featureVersion": row.get("feature_version"),
        "dbName": row.get("name"),
        "displayName": row.get("name") or spec.label,
        "strategyKey": row.get("strategy_key"),
        "baseStrategyKey": row.get("base_strategy_key") or base_registry_key,
        "notes": row.get("notes") or "",
        "tags": row.get("tags") or [],
    }


def _strategy_direction(base_strategy_key: str) -> Optional[str]:
    """Canonical mapping from a base strategy key to its direction.
    Uses the same prefix-based matching as _legacy_base_registry_key so
    user-cloned strategy keys like 'up_move_short_v1_2' resolve correctly.
    Returns 'up', 'down', or None (non-directional)."""
    key = str(base_strategy_key or "").strip().lower()
    if not key:
        return None
    if key.startswith("up_move_short"):
        return "up"
    if key.startswith("down_move"):
        return "down"
    return None


def _row_matches_strategy(row: Dict[str, Any], base_strategy_key: str) -> bool:
    direction = str(row.get("direction") or "").strip().lower()
    expected = _strategy_direction(base_strategy_key)
    if expected is None:
        return True   # non-directional strategy — accept all rows
    return direction == expected


def _filter_funnel_for_direction(funnel: List[dict], direction: Optional[str]) -> List[dict]:
    """Reshape each stage so the flat keys (candidates_in, kept, dropped, drop_reasons)
    reflect the selected direction. Preserves existing frontend contract."""
    if not funnel:
        return []

    out = []
    for stage in funnel:
        scope = stage.get("scope", "shared")
        if scope == "shared":
            bucket = stage.get("shared", {}) or {}
        elif direction in ("up", "down"):
            bucket = stage.get(direction, {}) or {}
        else:
            # Unknown direction — sum up+down as a combined view
            up = stage.get("up", {}) or {}
            down = stage.get("down", {}) or {}
            combined_reasons = Counter(up.get("drop_reasons", {}) or {})
            combined_reasons.update(down.get("drop_reasons", {}) or {})
            bucket = {
                "candidates_in": (up.get("candidates_in", 0) or 0) + (down.get("candidates_in", 0) or 0),
                "kept":          (up.get("kept", 0) or 0)          + (down.get("kept", 0) or 0),
                "dropped":       (up.get("dropped", 0) or 0)       + (down.get("dropped", 0) or 0),
                "drop_reasons":  dict(combined_reasons),
            }

        out.append({
            "key":           stage.get("key"),
            "label":         stage.get("label"),
            "kind":          stage.get("kind"),
            "scope":         scope,
            "bypassed":      stage.get("bypassed", False),
            "candidates_in": bucket.get("candidates_in", 0),
            "kept":          bucket.get("kept", 0),
            "dropped":       bucket.get("dropped", 0),
            "drop_reasons":  bucket.get("drop_reasons", {}) or {},
        })
    return out


def _filtered_rows(rows: List[Dict[str, Any]], base_strategy_key: str) -> List[Dict[str, Any]]:
    return [row for row in (rows or []) if _row_matches_strategy(row, base_strategy_key)]


def _rebuild_summary(rows: List[Dict[str, Any]], strategy_key: str, original_summary: Dict[str, Any] | None) -> Dict[str, Any]:
    base = dict(original_summary or {})
    filtered = rows or []

    base["instances_found"] = len(filtered)
    base["up_short_setups_found"] = sum(1 for row in filtered if bool(row.get("short_setup_found")))
    base["executed_short_trades"] = sum(1 for row in filtered if bool(row.get("trade_entry_found")))
    base["winning_trades"] = sum(1 for row in filtered if str(row.get("trade_outcome") or "") == "win")
    base["strategy_key"] = strategy_key
    return base


def _rebuild_diagnostics(rows: List[Dict[str, Any]], strategy_key: str, original_diagnostics: Dict[str, Any] | None) -> Dict[str, Any]:
    base = dict(original_diagnostics or {})
    filtered = rows or []

    base["valid_instances"] = len(filtered)
    base["up_short_setups_found"] = sum(1 for row in filtered if bool(row.get("short_setup_found")))
    base["actual_trades_found"] = sum(1 for row in filtered if bool(row.get("trade_entry_found")))
    base["winning_trades"] = sum(1 for row in filtered if str(row.get("trade_outcome") or "") == "win")
    base["strategy_key"] = strategy_key
    base["sample_results"] = filtered[:8]
    return base


def get_backtests_v2_selection_since(last_seq: int | None):
    current_seq = int(_SELECTION_STATE.get("seq") or 0)
    if last_seq is not None and current_seq <= int(last_seq):
        return current_seq, None
    return current_seq, _SELECTION_STATE.get("payload")


def register_backtests_v2_routes(server, repo_root: Path) -> None:
    dist_dir = (Path(repo_root) / "react_backtests_v2" / "dist").resolve()

    def build_ready() -> bool:
        return dist_dir.exists() and (dist_dir / "index.html").exists()

    def index():
        if not build_ready():
            return (
                "Backtests v2 React build not found. Build react_backtests_v2/dist before starting Dash.",
                503,
            )
        return send_from_directory(str(dist_dir), "index.html")

    def assets(path: str):
        if not build_ready():
            return (
                "Backtests v2 React build not found. Build react_backtests_v2/dist before starting Dash.",
                503,
            )

        candidate = (dist_dir / path).resolve()
        if candidate.exists() and candidate.is_file():
            return send_from_directory(str(dist_dir), path)

        return send_from_directory(str(dist_dir), "index.html")

    def list_strategies_api():
        resolved = _resolve_strategy_payloads()
        default_key = _default_strategy_key()

        return jsonify(
            {
                "ok": True,
                "defaultStrategyKey": default_key,
                "strategies": [item["serialized"] for item in resolved.values()],
            }
        )

    def save_strategy_defaults_api():
        raw_payload = _normalize_payload_aliases(request.get_json(silent=True) or {})
        strategy_key = str(raw_payload.get("strategyKey") or "").strip()
        params = raw_payload.get("params") or {}
        name = raw_payload.get("name")
        notes = raw_payload.get("notes")

        if not strategy_key:
            return jsonify({"ok": False, "error": "strategyKey is required"}), 400

        if not isinstance(params, dict):
            return jsonify({"ok": False, "error": "params must be an object"}), 400

        try:
            strategy = _save_strategy_defaults(strategy_key, params, name=name, notes=notes)
            return jsonify({"ok": True, "strategy": strategy})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    def create_strategy_api():
        raw_payload = _normalize_payload_aliases(request.get_json(silent=True) or {})
        base_strategy_key = str(raw_payload.get("baseStrategyKey") or "").strip()
        new_name = raw_payload.get("name")
        new_strategy_key = raw_payload.get("strategyKey")
        params = raw_payload.get("params") or {}
        notes = raw_payload.get("notes")

        if not base_strategy_key:
            return jsonify({"ok": False, "error": "baseStrategyKey is required"}), 400

        if not isinstance(params, dict):
            return jsonify({"ok": False, "error": "params must be an object"}), 400

        try:
            strategy = _create_strategy_from_existing(
                base_strategy_key=base_strategy_key,
                new_name=new_name,
                new_strategy_key=new_strategy_key,
                params=params,
                notes=notes,
            )
            return jsonify({"ok": True, "strategy": strategy})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    def scan_gex_moves_api():
        raw_payload = _normalize_payload_aliases(request.get_json(silent=True) or {})
        resolved = _resolve_strategy_payloads()

        strategy_key = str(raw_payload.get("strategyKey") or _default_strategy_key()).strip()
        item = resolved.get(strategy_key)
        if item is None:
            return jsonify(
                {
                    "ok": False,
                    "error": f"Unknown strategyKey: {strategy_key}",
                    "availableStrategyKeys": list(resolved.keys()),
                }
            ), 400

        strategy = item["spec"]
        strategy_meta = item["serialized"]
        settings = {**strategy_meta["defaults"], **raw_payload}
        settings["strategyKey"] = strategy_key

        try:
            data = strategy.runner(
                start_date=settings["startDate"],
                end_date=settings["endDate"],
                min_level_gex_bn=float(settings["minLevelGexBn"]),
                zone_merge_distance_pts=float(settings["zoneMergeDistancePts"]),
                min_clean_move_points=float(settings["minCleanMovePoints"]),
                target_proximity_pts=float(settings["targetProximityPts"]),
                max_zone_breach_pts=float(settings["maxZoneBreachPts"]),
                pivot_strength_bars=int(settings["pivotStrengthBars"]),
                level_family=str(settings["levelFamily"]),
                max_results=int(settings["maxResults"]),
                consolidation_window_minutes=int(settings["consolidationWindowMinutes"]),
                short_put_skew_increase_pct=float(settings["shortPutSkewIncreasePct"]),
                short_call_skew_max_pct=float(settings["shortCallSkewMaxPct"]),
                entry_within_top_pts=float(settings["entryWithinTopPts"]),
                entry_search_window_minutes=int(settings["entrySearchWindowMinutes"]),
                initial_stop_pts=float(settings["initialStopPts"]),
                trail_activate_profit_pts=float(settings["trailActivateProfitPts"]),
                trailing_stop_pts=float(settings["trailingStopPts"]),
                take_profit_pts=float(settings["takeProfitPts"]),
                max_prior_down_up_ratio=float(settings.get("maxPriorDownUpRatio", 2.0)),
                max_start_pct_of_range=float(settings.get("maxStartPctOfRange", 0.20)),
                max_move_loss_pct=float(settings.get("maxMoveLossPct", 0.75)),
                min_minutes_after_open=int(settings.get("minMinutesAfterOpen", 15)),
                long_put_skew_min_decrease_pct=float(settings.get("longPutSkewMinDecreasePct", 80.0)),
                long_call_skew_min_increase_pct=float(settings.get("longCallSkewMinIncreasePct", 30.0)),
                max_minutes_before_close=int(settings.get("maxMinutesBeforeClose", 45)),
                source_view=DEFAULT_SOURCE_VIEW,
                bypass_filters=tuple(settings.get("bypassFilters") or ()),
            )

            base_strategy_key = strategy_meta.get("baseStrategyKey") or item["base_registry_key"]
            raw_rows = data.get("rows") or []
            rows = _filtered_rows(raw_rows, base_strategy_key)
            summary = _rebuild_summary(rows, strategy_key, data.get("summary"))
            diagnostics = _rebuild_diagnostics(rows, strategy_key, data.get("diagnostics"))

            # Filter funnel by direction
            funnel_raw = data.get("funnel", [])
            direction = _strategy_direction(base_strategy_key)
            funnel_filtered = _filter_funnel_for_direction(funnel_raw, direction)

            return jsonify(
                {
                    "ok": True,
                    "strategy": strategy_meta,
                    "settings": settings,
                    "sourceView": DEFAULT_SOURCE_VIEW,
                    "rows": rows,
                    "summary": summary,
                    "diagnostics": diagnostics,
                    "funnel": funnel_filtered,
                    "funnelRaw": funnel_raw,
                }
            )
        except Exception as exc:
            return jsonify(
                {
                    "ok": False,
                    "error": str(exc),
                    "strategy": strategy_meta,
                    "settings": settings,
                    "sourceView": DEFAULT_SOURCE_VIEW,
                }
            ), 400

    def select_trade_api():
        payload = request.get_json(silent=True) or {}

        trade_date = str(payload.get("trade_date") or "").strip()
        start_ts_pt = str(payload.get("start_ts_pt") or "").strip()
        target_ts_pt = str(payload.get("target_ts_pt") or "").strip()
        signal_ts_pt = str(payload.get("signal_ts_pt") or "").strip()
        trade_entry_ts_pt = str(payload.get("trade_entry_ts_pt") or "").strip()
        trade_exit_ts_pt = str(payload.get("trade_exit_ts_pt") or "").strip()

        if not trade_date:
            return jsonify({"ok": False, "error": "trade_date is required"}), 400

        _SELECTION_STATE["seq"] = int(_SELECTION_STATE.get("seq") or 0) + 1
        _SELECTION_STATE["payload"] = {
            "trade_date": trade_date,
            "start_ts_pt": start_ts_pt,
            "target_ts_pt": target_ts_pt,
            "signal_ts_pt": signal_ts_pt,
            "trade_entry_ts_pt": trade_entry_ts_pt,
            "trade_exit_ts_pt": trade_exit_ts_pt,
        }

        return jsonify(
            {
                "ok": True,
                "seq": _SELECTION_STATE["seq"],
                "selection": _SELECTION_STATE["payload"],
            }
        )

    def scan_signals_api():
        """
        Scan for setups on a given date and persist results to bt_signals.
        Uses the same runner as scan_gex_moves_api but also:
        - Computes a settings_hash so stale results can be identified
        - Upserts rows to bt_signals (preserves existing labels)
        - Returns signal rows shaped for SignalPanel
        """
        import hashlib, datetime as _dt, uuid as _uuid

        raw_payload = _normalize_payload_aliases(request.get_json(silent=True) or {})
        resolved = _resolve_strategy_payloads()

        strategy_key = str(raw_payload.get("strategyKey") or _default_strategy_key()).strip()
        item = resolved.get(strategy_key)
        if item is None:
            return jsonify({"ok": False, "error": f"Unknown strategyKey: {strategy_key}"}), 400

        strategy = item["spec"]
        strategy_meta = item["serialized"]
        settings = {**strategy_meta["defaults"], **raw_payload}
        settings["strategyKey"] = strategy_key

        trade_date = str(raw_payload.get("tradeDate") or raw_payload.get("startDate") or "").strip()
        if not trade_date:
            return jsonify({"ok": False, "error": "tradeDate is required"}), 400

        # Settings hash — identifies this exact parameter set
        hash_keys = [
            "minLevelGexBn", "zoneMergeDistancePts", "minCleanMovePoints",
            "targetProximityPts", "maxZoneBreachPts", "pivotStrengthBars",
            "levelFamily", "consolidationWindowMinutes",
            "shortPutSkewIncreasePct", "shortCallSkewMaxPct",
            "entryWithinTopPts", "entrySearchWindowMinutes",
            "initialStopPts", "trailActivateProfitPts", "trailingStopPts", "takeProfitPts",
            "maxPriorDownUpRatio", "maxStartPctOfRange", "maxMoveLossPct",
            "minMinutesAfterOpen", "longPutSkewMinDecreasePct", "longCallSkewMinIncreasePct",
            "maxMinutesBeforeClose", "longInitialStopPts", "longTrailActivateProfitPts",
            "longTrailingStopPts", "longTakeProfitPts",
        ]
        hash_str = json.dumps({k: settings.get(k) for k in hash_keys}, sort_keys=True)
        settings_hash = hashlib.md5(hash_str.encode()).hexdigest()[:12]

        try:
            data = strategy.runner(
                start_date=trade_date,
                end_date=trade_date,
                min_level_gex_bn=float(settings["minLevelGexBn"]),
                zone_merge_distance_pts=float(settings["zoneMergeDistancePts"]),
                min_clean_move_points=float(settings["minCleanMovePoints"]),
                target_proximity_pts=float(settings["targetProximityPts"]),
                max_zone_breach_pts=float(settings["maxZoneBreachPts"]),
                pivot_strength_bars=int(settings["pivotStrengthBars"]),
                level_family=str(settings["levelFamily"]),
                max_results=int(settings.get("maxResults", 2500)),
                consolidation_window_minutes=int(settings["consolidationWindowMinutes"]),
                short_put_skew_increase_pct=float(settings["shortPutSkewIncreasePct"]),
                short_call_skew_max_pct=float(settings["shortCallSkewMaxPct"]),
                entry_within_top_pts=float(settings["entryWithinTopPts"]),
                entry_search_window_minutes=int(settings["entrySearchWindowMinutes"]),
                initial_stop_pts=float(settings["initialStopPts"]),
                trail_activate_profit_pts=float(settings["trailActivateProfitPts"]),
                trailing_stop_pts=float(settings["trailingStopPts"]),
                take_profit_pts=float(settings["takeProfitPts"]),
                max_prior_down_up_ratio=float(settings.get("maxPriorDownUpRatio", 2.0)),
                max_start_pct_of_range=float(settings.get("maxStartPctOfRange", 0.20)),
                max_move_loss_pct=float(settings.get("maxMoveLossPct", 0.75)),
                min_minutes_after_open=int(settings.get("minMinutesAfterOpen", 15)),
                long_put_skew_min_decrease_pct=float(settings.get("longPutSkewMinDecreasePct", 80.0)),
                long_call_skew_min_increase_pct=float(settings.get("longCallSkewMinIncreasePct", 30.0)),
                max_minutes_before_close=int(settings.get("maxMinutesBeforeClose", 45)),
                long_initial_stop_pts=float(settings.get("longInitialStopPts", 10.0)),
                long_trail_activate_profit_pts=float(settings.get("longTrailActivateProfitPts", 20.0)),
                long_trailing_stop_pts=float(settings.get("longTrailingStopPts", 10.0)),
                long_take_profit_pts=float(settings.get("longTakeProfitPts", 35.0)),
                source_view=DEFAULT_SOURCE_VIEW,
                bypass_filters=tuple(settings.get("bypassFilters") or ()),
            )

            base_strategy_key = strategy_meta.get("baseStrategyKey") or item["base_registry_key"]
            raw_rows = data.get("rows") or []
            rows = _filtered_rows(raw_rows, base_strategy_key)

            # Shape rows into signal objects and persist to bt_signals
            engine = _engine()
            signal_out = []
            with engine.begin() as conn:
                for row in rows:
                    direction = row.get("direction", "up")
                    has_setup = row.get("short_setup_found") or row.get("long_setup_found")
                    has_trade = row.get("trade_entry_found")
                    status = "completed" if has_trade else ("signal_fired" if has_setup else "expired")

                    signal_ts_pt = row.get("short_signal_ts_pt") or row.get("long_signal_ts_pt")
                    put_skew = row.get("short_signal_delta_put_skew_pct") or row.get("long_signal_delta_put_skew_pct")
                    call_skew = row.get("short_signal_delta_call_skew_pct") or row.get("long_signal_delta_call_skew_pct")

                    # Check if this exact row already exists (match on date + direction + target + signal_ts)
                    existing = conn.execute(text("""
                        SELECT signal_id, label, label_note
                        FROM bt_signals
                        WHERE trade_date = :td
                          AND strategy_key = :sk
                          AND settings_hash = :sh
                          AND direction = :dir
                          AND target_level = :tl
                          AND signal_ts_pt IS NOT DISTINCT FROM :stp
                        LIMIT 1
                    """), {
                        "td": trade_date, "sk": strategy_key, "sh": settings_hash,
                        "dir": direction,
                        "tl": row.get("target_level"),
                        "stp": signal_ts_pt,
                    }).fetchone()

                    if existing:
                        signal_id = str(existing[0])
                        label = existing[1]
                        label_note = existing[2]
                        # Update outcome/status in case it changed
                        conn.execute(text("""
                            UPDATE bt_signals SET
                                status = :status,
                                entry_price = :ep,
                                initial_stop = :st,
                                take_profit = :tp,
                                realized_pts = :rp,
                                outcome = :oc,
                                raw_result = CAST(:rr AS jsonb)
                            WHERE signal_id = :sid
                        """), {
                            "status": status, "ep": row.get("trade_entry_price"),
                            "st": row.get("trade_initial_stop_price"),
                            "tp": row.get("trade_take_profit_price"),
                            "rp": row.get("trade_realized_points"),
                            "oc": row.get("trade_outcome"),
                            "rr": json.dumps({k: _json_safe_value(v) for k, v in row.items()}),
                            "sid": signal_id,
                        })
                    else:
                        signal_id = str(_uuid.uuid4())
                        label = None
                        label_note = None
                        conn.execute(text("""
                            INSERT INTO bt_signals (
                                signal_id, trade_date, strategy_key, settings_hash,
                                direction, status,
                                source_zone_low, source_zone_high, target_level,
                                entry_price, initial_stop, take_profit, trailing_stop,
                                signal_ts_pt, entry_ts_pt, exit_ts_pt,
                                put_skew, call_skew,
                                realized_pts, outcome,
                                raw_result
                            ) VALUES (
                                :sid, :td, :sk, :sh,
                                :dir, :status,
                                :szl, :szh, :tl,
                                :ep, :ist, :tp, :ts,
                                :stp, :etp, :xtp,
                                :put, :call,
                                :rp, :oc,
                                CAST(:rr AS jsonb)
                            )
                        """), {
                            "sid": signal_id, "td": trade_date, "sk": strategy_key, "sh": settings_hash,
                            "dir": direction, "status": status,
                            "szl": row.get("source_zone_low"), "szh": row.get("source_zone_high"),
                            "tl": row.get("target_level"),
                            "ep": row.get("trade_entry_price"),
                            "ist": row.get("trade_initial_stop_price"),
                            "tp": row.get("trade_take_profit_price"),
                            "ts": row.get("trade_trailing_stop_price"),
                            "stp": signal_ts_pt,
                            "etp": row.get("trade_entry_ts_pt"),
                            "xtp": row.get("trade_exit_ts_pt"),
                            "put": put_skew, "call": call_skew,
                            "rp": row.get("trade_realized_points"),
                            "oc": row.get("trade_outcome"),
                            "rr": json.dumps({k: _json_safe_value(v) for k, v in row.items()}),
                        })

                    signal_out.append({
                        "signal_id": signal_id,
                        "trade_date": trade_date,
                        "strategy_key": strategy_key,
                        "settings_hash": settings_hash,
                        "direction": direction,
                        "status": status,
                        "source_zone_low": row.get("source_zone_low"),
                        "source_zone_high": row.get("source_zone_high"),
                        "target_level": row.get("target_level"),
                        "entry_price": row.get("trade_entry_price"),
                        "initial_stop": row.get("trade_initial_stop_price"),
                        "take_profit": row.get("trade_take_profit_price"),
                        "signal_ts_pt": signal_ts_pt,
                        "entry_ts_pt": row.get("trade_entry_ts_pt"),
                        "exit_ts_pt": row.get("trade_exit_ts_pt"),
                        "put_skew": put_skew,
                        "call_skew": call_skew,
                        "realized_pts": row.get("trade_realized_points"),
                        "outcome": row.get("trade_outcome"),
                        "label": label,
                        "label_note": label_note,
                    })

            return jsonify({"ok": True, "signals": signal_out, "settingsHash": settings_hash})

        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    def label_signal_api():
        """Update the thumbs up/down label on a signal."""
        payload = request.get_json(silent=True) or {}
        signal_id = str(payload.get("signalId") or "").strip()
        label = payload.get("label")  # +1, -1, 0, or null
        note = payload.get("note", "")

        if not signal_id:
            return jsonify({"ok": False, "error": "signalId is required"}), 400

        if label is not None and label not in (-1, 0, 1):
            return jsonify({"ok": False, "error": "label must be -1, 0, 1, or null"}), 400

        try:
            engine = _engine()
            with engine.begin() as conn:
                result = conn.execute(text("""
                    UPDATE bt_signals
                    SET label = :label,
                        label_note = :note,
                        labeled_at = now()
                    WHERE signal_id = :sid::uuid
                    RETURNING signal_id, label, label_note, labeled_at
                """), {"label": label, "note": note or None, "sid": signal_id})
                row = result.fetchone()
                if row is None:
                    return jsonify({"ok": False, "error": "signal not found"}), 404
            return jsonify({"ok": True, "signalId": str(row[0]), "label": row[1]})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 400

    # ─────────────────────────────────────────────────────────────────────
    # Saved Scan Cache
    # ─────────────────────────────────────────────────────────────────────
    # Persists full scan results (rows + diagnostics + funnel + params) to
    # bt2_scan_cache table for fast load/filter without rescanning.
    #
    # Workflow:
    #   1. POST /api/backtests-v2/saved-scans/run       — runs a scan with
    #      maximally-permissive filters and saves the result. Returns scan_id.
    #   2. GET  /api/backtests-v2/saved-scans            — list all saved scans
    #   3. GET  /api/backtests-v2/saved-scans/<scan_id>  — load a saved scan
    #      (rows + diagnostics + funnel)
    #   4. DELETE /api/backtests-v2/saved-scans/<scan_id> — remove a saved scan

    def _ensure_scan_cache_table_exists(conn):
        """Idempotent table creation. Cheap to call on every request."""
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS public.bt2_scan_cache (
              scan_id        BIGSERIAL PRIMARY KEY,
              created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
              label          VARCHAR(200),
              direction      VARCHAR(8) NOT NULL,
              start_date     DATE NOT NULL,
              end_date       DATE NOT NULL,
              params         JSONB NOT NULL,
              funnel         JSONB,
              diagnostics    JSONB,
              rows           JSONB NOT NULL,
              row_count      INT NOT NULL,
              notes          TEXT
            )
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_bt2_scan_cache_created
              ON public.bt2_scan_cache (created_at DESC)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_bt2_scan_cache_dir_dates
              ON public.bt2_scan_cache (direction, start_date, end_date)
        """))

    def saved_scans_run_api():
        """
        Run a scan with permissive filters and persist the full result.
        Body: {
          direction: 'up' | 'down',
          start_date, end_date,
          label?, notes?,
          params?: {...overrides for the scan...}
        }
        """
        raw_payload = _normalize_payload_aliases(request.get_json(silent=True) or {})
        direction = str(raw_payload.get("direction") or "").strip().lower()
        if direction not in ("up", "down"):
            return jsonify({"ok": False, "error": "direction must be 'up' or 'down'"}), 400

        start_date = str(raw_payload.get("startDate") or raw_payload.get("start_date") or "").strip()
        end_date   = str(raw_payload.get("endDate")   or raw_payload.get("end_date")   or "").strip()
        if not start_date or not end_date:
            return jsonify({"ok": False, "error": "startDate and endDate are required"}), 400

        label = _normalize_name(raw_payload.get("label"), f"{direction.upper()} {start_date} to {end_date}")
        notes = _normalize_notes(raw_payload.get("notes"))

        # Pick the right base strategy and resolve full defaults
        resolved = _resolve_strategy_payloads()
        base_key = "up_move_short" if direction == "up" else "down_move_scan"
        item = None
        for v in resolved.values():
            if v.get("base_registry_key") == base_key:
                item = v
                break
        if item is None:
            return jsonify({"ok": False, "error": f"No registered strategy for direction={direction}"}), 500

        strategy = item["spec"]
        strategy_meta = item["serialized"]

        # Build maximally-permissive settings for the cached scan.
        # Skew bypass + study mode on by default — gives the broadest set of rows
        # so that downstream filters can subset without rescanning.
        permissive = dict(strategy_meta["defaults"])
        permissive["startDate"] = start_date
        permissive["endDate"]   = end_date
        permissive["executionMode"] = "study_target_hits"
        # Bypass the skew gate so we capture every target-touch event,
        # regardless of skew configuration.
        existing_bypass = list(permissive.get("bypassFilters") or [])
        if "skew_signal_fired" not in existing_bypass:
            existing_bypass.append("skew_signal_fired")
        permissive["bypassFilters"] = existing_bypass

        # Apply any user overrides on top
        overrides = raw_payload.get("params") or {}
        if isinstance(overrides, dict):
            permissive.update(overrides)

        try:
            data = strategy.runner(
                start_date=permissive["startDate"],
                end_date=permissive["endDate"],
                min_level_gex_bn=float(permissive["minLevelGexBn"]),
                zone_merge_distance_pts=float(permissive["zoneMergeDistancePts"]),
                min_clean_move_points=float(permissive["minCleanMovePoints"]),
                target_proximity_pts=float(permissive["targetProximityPts"]),
                max_zone_breach_pts=float(permissive["maxZoneBreachPts"]),
                pivot_strength_bars=int(permissive["pivotStrengthBars"]),
                level_family=str(permissive["levelFamily"]),
                max_results=int(permissive["maxResults"]),
                consolidation_window_minutes=int(permissive["consolidationWindowMinutes"]),
                short_put_skew_increase_pct=float(permissive["shortPutSkewIncreasePct"]),
                short_call_skew_max_pct=float(permissive["shortCallSkewMaxPct"]),
                entry_within_top_pts=float(permissive["entryWithinTopPts"]),
                entry_search_window_minutes=int(permissive["entrySearchWindowMinutes"]),
                initial_stop_pts=float(permissive["initialStopPts"]),
                trail_activate_profit_pts=float(permissive["trailActivateProfitPts"]),
                trailing_stop_pts=float(permissive["trailingStopPts"]),
                take_profit_pts=float(permissive["takeProfitPts"]),
                max_prior_down_up_ratio=float(permissive.get("maxPriorDownUpRatio", 2.0)),
                max_start_pct_of_range=float(permissive.get("maxStartPctOfRange", 0.20)),
                max_move_loss_pct=float(permissive.get("maxMoveLossPct", 0.75)),
                min_minutes_after_open=int(permissive.get("minMinutesAfterOpen", 15)),
                long_put_skew_min_decrease_pct=float(permissive.get("longPutSkewMinDecreasePct", 80.0)),
                long_call_skew_min_increase_pct=float(permissive.get("longCallSkewMinIncreasePct", 30.0)),
                max_minutes_before_close=int(permissive.get("maxMinutesBeforeClose", 45)),
                source_view=DEFAULT_SOURCE_VIEW,
                bypass_filters=tuple(permissive.get("bypassFilters") or ()),
                execution_mode=str(permissive.get("executionMode") or "study_target_hits"),
                forward_horizons_minutes=tuple(
                    int(x) for x in (permissive.get("forwardHorizonsMinutes") or [30, 60, 90, 120, 180])
                    if str(x).strip().lstrip('-').isdigit() and int(x) > 0
                ) or (30, 60, 90, 120, 180),
                condor_wing_width_pts=float(permissive.get("condorWingWidthPts") or 10.0),
            )
        except Exception as exc:
            return jsonify({"ok": False, "error": f"scan failed: {exc}"}), 400

        raw_rows = data.get("rows") or []
        rows = _filtered_rows(raw_rows, base_key)
        diagnostics = _rebuild_diagnostics(rows, strategy_meta["key"], data.get("diagnostics"))
        funnel = _filter_funnel_for_direction(data.get("funnel", []), direction)

        # Persist
        try:
            engine = _engine()
            with engine.begin() as conn:
                _ensure_scan_cache_table_exists(conn)
                result = conn.execute(text("""
                    INSERT INTO public.bt2_scan_cache
                      (label, direction, start_date, end_date, params, funnel, diagnostics, rows, row_count, notes)
                    VALUES
                      (:label, :direction, :start_date, :end_date,
                       CAST(:params AS JSONB),
                       CAST(:funnel AS JSONB),
                       CAST(:diagnostics AS JSONB),
                       CAST(:rows AS JSONB),
                       :row_count,
                       :notes)
                    RETURNING scan_id, created_at
                """), {
                    "label": label,
                    "direction": direction,
                    "start_date": start_date,
                    "end_date": end_date,
                    "params": json.dumps(permissive),
                    "funnel": json.dumps(funnel),
                    "diagnostics": json.dumps(diagnostics),
                    "rows": json.dumps(rows),
                    "row_count": len(rows),
                    "notes": notes or None,
                })
                row = result.fetchone()
                scan_id = int(row[0])
                created_at = row[1].isoformat() if row[1] is not None else None
        except Exception as exc:
            return jsonify({"ok": False, "error": f"persist failed: {exc}"}), 500

        return jsonify({
            "ok": True,
            "scan_id": scan_id,
            "created_at": created_at,
            "label": label,
            "direction": direction,
            "start_date": start_date,
            "end_date": end_date,
            "row_count": len(rows),
        })

    def saved_scans_list_api():
        """
        List metadata for all saved scans (no rows).
        Optional query params: direction=up|down
        """
        direction_filter = str(request.args.get("direction") or "").strip().lower()
        try:
            engine = _engine()
            with engine.begin() as conn:
                _ensure_scan_cache_table_exists(conn)
                if direction_filter in ("up", "down"):
                    result = conn.execute(text("""
                        SELECT scan_id, created_at, label, direction, start_date, end_date,
                               row_count, notes,
                               (params->>'executionMode') AS execution_mode
                        FROM public.bt2_scan_cache
                        WHERE direction = :direction
                        ORDER BY created_at DESC
                    """), {"direction": direction_filter})
                else:
                    result = conn.execute(text("""
                        SELECT scan_id, created_at, label, direction, start_date, end_date,
                               row_count, notes,
                               (params->>'executionMode') AS execution_mode
                        FROM public.bt2_scan_cache
                        ORDER BY created_at DESC
                    """))
                items = []
                for r in result:
                    items.append({
                        "scan_id": int(r[0]),
                        "created_at": r[1].isoformat() if r[1] is not None else None,
                        "label": r[2],
                        "direction": r[3],
                        "start_date": r[4].isoformat() if r[4] is not None else None,
                        "end_date": r[5].isoformat() if r[5] is not None else None,
                        "row_count": int(r[6]) if r[6] is not None else 0,
                        "notes": r[7],
                        "execution_mode": r[8],
                    })
            return jsonify({"ok": True, "scans": items})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

    def saved_scans_load_api(scan_id):
        """Return the full saved scan: rows + diagnostics + funnel."""
        try:
            sid = int(scan_id)
        except Exception:
            return jsonify({"ok": False, "error": "scan_id must be integer"}), 400

        try:
            engine = _engine()
            with engine.begin() as conn:
                _ensure_scan_cache_table_exists(conn)
                result = conn.execute(text("""
                    SELECT scan_id, created_at, label, direction, start_date, end_date,
                           params, funnel, diagnostics, rows, row_count, notes
                    FROM public.bt2_scan_cache
                    WHERE scan_id = :scan_id
                """), {"scan_id": sid})
                row = result.fetchone()
                if row is None:
                    return jsonify({"ok": False, "error": "scan not found"}), 404
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

        return jsonify({
            "ok": True,
            "scan_id": int(row[0]),
            "created_at": row[1].isoformat() if row[1] is not None else None,
            "label": row[2],
            "direction": row[3],
            "start_date": row[4].isoformat() if row[4] is not None else None,
            "end_date": row[5].isoformat() if row[5] is not None else None,
            "params": row[6],          # already a dict via JSONB → SQLAlchemy
            "funnel": row[7] or [],
            "diagnostics": row[8] or {},
            "rows": row[9] or [],
            "row_count": int(row[10]) if row[10] is not None else 0,
            "notes": row[11],
        })

    def saved_scans_delete_api(scan_id):
        """Delete a saved scan."""
        try:
            sid = int(scan_id)
        except Exception:
            return jsonify({"ok": False, "error": "scan_id must be integer"}), 400

        try:
            engine = _engine()
            with engine.begin() as conn:
                _ensure_scan_cache_table_exists(conn)
                result = conn.execute(text("""
                    DELETE FROM public.bt2_scan_cache
                    WHERE scan_id = :scan_id
                    RETURNING scan_id
                """), {"scan_id": sid})
                row = result.fetchone()
                if row is None:
                    return jsonify({"ok": False, "error": "scan not found"}), 404
            return jsonify({"ok": True, "scan_id": sid})
        except Exception as exc:
            return jsonify({"ok": False, "error": str(exc)}), 500

    if "backtests_v2_preview_index" not in server.view_functions:
        server.add_url_rule("/backtests-v2-preview", endpoint="backtests_v2_preview_index", view_func=index)

    if "backtests_v2_preview_index_slash" not in server.view_functions:
        server.add_url_rule("/backtests-v2-preview/", endpoint="backtests_v2_preview_index_slash", view_func=index)

    if "backtests_v2_preview_assets" not in server.view_functions:
        server.add_url_rule("/backtests-v2-preview/<path:path>", endpoint="backtests_v2_preview_assets", view_func=assets)

    if "backtests_v2_list_strategies" not in server.view_functions:
        server.add_url_rule(
            "/api/backtests-v2/strategies",
            endpoint="backtests_v2_list_strategies",
            view_func=list_strategies_api,
            methods=["GET"],
        )

    if "backtests_v2_save_strategy_defaults" not in server.view_functions:
        server.add_url_rule(
            "/api/backtests-v2/strategy-defaults",
            endpoint="backtests_v2_save_strategy_defaults",
            view_func=save_strategy_defaults_api,
            methods=["POST"],
        )

    if "backtests_v2_create_strategy" not in server.view_functions:
        server.add_url_rule(
            "/api/backtests-v2/strategy-create",
            endpoint="backtests_v2_create_strategy",
            view_func=create_strategy_api,
            methods=["POST"],
        )

    if "backtests_v2_scan_gex_moves" not in server.view_functions:
        server.add_url_rule(
            "/api/backtests-v2/gex-moves",
            endpoint="backtests_v2_scan_gex_moves",
            view_func=scan_gex_moves_api,
            methods=["POST"],
        )

    if "backtests_v2_select_trade" not in server.view_functions:
        server.add_url_rule(
            "/api/backtests-v2/select-trade",
            endpoint="backtests_v2_select_trade",
            view_func=select_trade_api,
            methods=["POST"],
        )

    if "backtests_v2_scan_signals" not in server.view_functions:
        server.add_url_rule(
            "/api/backtests-v2/signals/scan",
            endpoint="backtests_v2_scan_signals",
            view_func=scan_signals_api,
            methods=["POST"],
        )

    if "backtests_v2_label_signal" not in server.view_functions:
        server.add_url_rule(
            "/api/backtests-v2/signals/label",
            endpoint="backtests_v2_label_signal",
            view_func=label_signal_api,
            methods=["POST"],
        )

    # ── Saved Scan Cache ──
    if "backtests_v2_saved_scans_run" not in server.view_functions:
        server.add_url_rule(
            "/api/backtests-v2/saved-scans/run",
            endpoint="backtests_v2_saved_scans_run",
            view_func=saved_scans_run_api,
            methods=["POST"],
        )

    if "backtests_v2_saved_scans_list" not in server.view_functions:
        server.add_url_rule(
            "/api/backtests-v2/saved-scans",
            endpoint="backtests_v2_saved_scans_list",
            view_func=saved_scans_list_api,
            methods=["GET"],
        )

    if "backtests_v2_saved_scans_load" not in server.view_functions:
        server.add_url_rule(
            "/api/backtests-v2/saved-scans/<scan_id>",
            endpoint="backtests_v2_saved_scans_load",
            view_func=saved_scans_load_api,
            methods=["GET"],
        )

    if "backtests_v2_saved_scans_delete" not in server.view_functions:
        server.add_url_rule(
            "/api/backtests-v2/saved-scans/<scan_id>",
            endpoint="backtests_v2_saved_scans_delete",
            view_func=saved_scans_delete_api,
            methods=["DELETE"],
        )
