from __future__ import annotations

from pathlib import Path
from typing import Any

from flask import jsonify, request, send_from_directory

from .service import scan_gex_level_moves, DEFAULT_SOURCE_VIEW


DEFAULT_SETTINGS = {
    "startDate": None,
    "endDate": None,
    "minLevelGexBn": 50,
    "zoneMergeDistancePts": 10,
    "minCleanMovePoints": 20,
    "targetProximityPts": 5,
    "maxZoneBreachPts": 5,
    "pivotStrengthBars": 3,
    "levelFamily": "primary",
    "maxResults": 2500,
    "consolidationWindowMinutes": 15,
    "shortPutSkewIncreasePct": 80,
    "shortCallSkewMaxPct": 30,
    "entryWithinTopPts": 2,
    "entrySearchWindowMinutes": 30,
    "initialStopPts": 6,
    "trailActivateProfitPts": 10,
    "trailingStopPts": 6,
    "takeProfitPts": 20,
}

_SELECTION_STATE: dict[str, Any] = {
    "seq": 0,
    "payload": None,
}


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

    def scan_gex_moves_api():
        payload = request.get_json(silent=True) or {}
        settings = {**DEFAULT_SETTINGS, **payload}

        try:
            data = scan_gex_level_moves(
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
            return jsonify({
                "ok": True,
                "settings": settings,
                "sourceView": DEFAULT_SOURCE_VIEW,
                **data,
            })
        except Exception as exc:
            return jsonify({
                "ok": False,
                "error": str(exc),
                "settings": settings,
                "sourceView": DEFAULT_SOURCE_VIEW,
            }), 400

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

        return jsonify({
            "ok": True,
            "seq": _SELECTION_STATE["seq"],
            "selection": _SELECTION_STATE["payload"],
        })

    if "backtests_v2_preview_index" not in server.view_functions:
        server.add_url_rule(
            "/backtests-v2-preview",
            endpoint="backtests_v2_preview_index",
            view_func=index,
        )

    if "backtests_v2_preview_index_slash" not in server.view_functions:
        server.add_url_rule(
            "/backtests-v2-preview/",
            endpoint="backtests_v2_preview_index_slash",
            view_func=index,
        )

    if "backtests_v2_preview_assets" not in server.view_functions:
        server.add_url_rule(
            "/backtests-v2-preview/<path:path>",
            endpoint="backtests_v2_preview_assets",
            view_func=assets,
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