from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List

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


def _row_matches_strategy(row: Dict[str, Any], base_strategy_key: str) -> bool:
    direction = str(row.get("direction") or "").strip().lower()

    if base_strategy_key == "up_move_short":
        return direction == "up"

    if base_strategy_key == "down_move_scan":
        return direction == "down"

    return True


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
                source_view=DEFAULT_SOURCE_VIEW,
            )

            base_strategy_key = strategy_meta.get("baseStrategyKey") or item["base_registry_key"]
            raw_rows = data.get("rows") or []
            rows = _filtered_rows(raw_rows, base_strategy_key)
            summary = _rebuild_summary(rows, strategy_key, data.get("summary"))
            diagnostics = _rebuild_diagnostics(rows, strategy_key, data.get("diagnostics"))

            return jsonify(
                {
                    "ok": True,
                    "strategy": strategy_meta,
                    "settings": settings,
                    "sourceView": DEFAULT_SOURCE_VIEW,
                    "rows": rows,
                    "summary": summary,
                    "diagnostics": diagnostics,
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

        if not trade_date:
            return jsonify({"ok": False, "error": "trade_date is required"}), 400

        _SELECTION_STATE["seq"] = int(_SELECTION_STATE.get("seq") or 0) + 1
        _SELECTION_STATE["payload"] = {
            "trade_date": trade_date,
            "start_ts_pt": start_ts_pt,
            "target_ts_pt": target_ts_pt,
            "signal_ts_pt": signal_ts_pt,
            "trade_entry_ts_pt": trade_entry_ts_pt,
        }

        return jsonify(
            {
                "ok": True,
                "seq": _SELECTION_STATE["seq"],
                "selection": _SELECTION_STATE["payload"],
            }
        )

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
