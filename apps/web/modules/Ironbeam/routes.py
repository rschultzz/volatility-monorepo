"""Ironbeam GEX-landscape Flask route (CR-016 hotfix 5).

Extracted from callbacks.py so it can be unit-tested without instantiating
the full Dash application.  Registered via:

    register_gex_landscape_route(app.server, engine)

where `engine` is a SQLAlchemy Engine (same one callbacks.py already holds).

Spot-resolution order (mirrors TodaySetup / Analogues):
  1. spot= param present → use it (explicit wins, existing behaviour).
  2. spot= absent → query ironbeam_es_1m_bars for the RTH open
     (first bar 06:30–13:00 PT on trade_date).
  3. No bars found → 400 with a clear error asking the caller to pass
     spot= explicitly.
"""
from __future__ import annotations

import datetime as dt

from flask import jsonify, request
from sqlalchemy import text

from packages.shared.gex_landscape_api import build_gex_landscape_response

# RTH open query — mirrors Bars.service._OPEN_SQL but expressed as SA text()
# so it runs inside the engine.connect() context already open for the
# landscape fetch (single connection, no type-mismatch gymnastics).
_RTH_OPEN_SQL = text("""
    SELECT open
    FROM ironbeam_es_1m_bars
    WHERE (datetime AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')::date = :d
      AND (datetime AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')::time
          BETWEEN '06:30:00' AND '13:00:00'
    ORDER BY datetime ASC
    LIMIT 1
""")

_CORS_ORIGINS = {
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://0.0.0.0:5173",
}


def _add_cors(resp, origin: str | None):
    if origin in _CORS_ORIGINS:
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Access-Control-Allow-Credentials"] = "true"
        resp.headers["Vary"] = "Origin"
    return resp


def register_gex_landscape_route(server, engine) -> None:
    """Wire GET /api/gex-landscape onto *server* using *engine* for DB access.

    Guard against double-registration (Dash hot-reload or test setUp calling
    register_ironbeam_callbacks multiple times).
    """
    if getattr(server, "_ironbeam_react_gex_landscape_route_registered", False):
        return

    @server.route("/api/gex-landscape", methods=["GET"])
    def ironbeam_react_gex_landscape_api():
        ticker = (request.args.get("ticker") or "").strip()
        date_str = (request.args.get("date") or "").strip()
        if not ticker or not date_str:
            return jsonify({"error": "ticker and date are required"}), 400

        # Parse date early — needed for spot resolution below.
        try:
            trade_date_obj = dt.date.fromisoformat(date_str)
        except (TypeError, ValueError):
            return jsonify({"error": "date must be YYYY-MM-DD"}), 400

        spot_raw = request.args.get("spot")

        # iv and implied_move are optional; pass through as-is so the
        # builder owns the numeric parsing + mutual-exclusion check.
        iv_raw = request.args.get("iv")
        move_raw = request.args.get("implied_move")
        iv_arg = iv_raw if (iv_raw is not None and iv_raw != "") else None
        move_arg = move_raw if (move_raw is not None and move_raw != "") else None

        # accuracy is optional; builder owns validation.
        accuracy_arg = request.args.get("accuracy")

        with engine.connect() as conn:
            # ── Spot resolution (CR-016 hotfix 5) ─────────────────────────
            # When spot is omitted, resolve from the first RTH bar on the
            # requested date.  This lets react_today_setup call the endpoint
            # without a spot param (the frontend no longer holds DEFAULT_SPOT).
            if spot_raw is None or spot_raw == "":
                row = conn.execute(_RTH_OPEN_SQL, {"d": trade_date_obj}).fetchone()
                if not row or row[0] is None:
                    return jsonify({
                        "error": (
                            f"no RTH bars found for {date_str}; "
                            "pass spot= explicitly to override"
                        )
                    }), 400
                spot_raw = str(float(row[0]))

            payload, status = build_gex_landscape_response(
                conn, ticker, date_str, spot_raw,
                iv=iv_arg, implied_move=move_arg,
                accuracy=accuracy_arg,
            )

        resp = jsonify(payload)
        return _add_cors(resp, request.headers.get("Origin")), status

    server._ironbeam_react_gex_landscape_route_registered = True
