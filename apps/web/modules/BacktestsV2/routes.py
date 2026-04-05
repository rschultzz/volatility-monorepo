from __future__ import annotations

from pathlib import Path

from flask import send_from_directory


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
