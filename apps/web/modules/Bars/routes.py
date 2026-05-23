"""Bars routes — Flask wiring for GET /api/bars (CR-016).

Endpoint:
    GET /api/bars?date=YYYY-MM-DD[&ticker=SPX][&session=rth]

Returns a JSON array of { time, open, high, low, close } lightweight-charts
candlestick objects for the RTH session on the requested date. Empty array
when no bars exist — 200, not 404, so the mini chart can empty-state cleanly.

The `session` param is accepted for forward-compatibility but currently
only `rth` (the default) is supported.
"""
from __future__ import annotations

import datetime as dt
import os
from typing import Optional

import psycopg
from flask import jsonify, request

from .service import fetch_rth_bars


def _normalize_db_url(url: str) -> str:
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    if url.startswith("postgresql+"):
        url = "postgresql://" + url.split("://", 1)[1]
    if "sslmode=" not in url:
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}sslmode=require"
    return url


def _conn():
    raw = os.getenv("DATABASE_URL", "").strip()
    if not raw:
        raise RuntimeError("DATABASE_URL is not set")
    return psycopg.connect(_normalize_db_url(raw))


def _parse_date(s: str) -> Optional[dt.date]:
    try:
        return dt.date.fromisoformat(s)
    except (TypeError, ValueError):
        return None


def register_bars_routes(server) -> None:
    """Wire GET /api/bars onto the Flask server."""
    if "bars_rth" in server.view_functions:
        return

    def bars_rth():
        date_s = (request.args.get("date") or "").strip()
        trade_date = _parse_date(date_s)
        if not trade_date:
            return jsonify({"ok": False, "error": "date is required (YYYY-MM-DD)"}), 400

        ticker = (request.args.get("ticker") or "SPX").strip() or "SPX"
        # session param accepted but only 'rth' supported.

        try:
            conn = _conn()
        except Exception as e:
            return jsonify({"ok": False, "error": f"db connect failed: {e}"}), 500

        try:
            bars = fetch_rth_bars(conn, ticker, trade_date)
            return jsonify(bars)
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500
        finally:
            try:
                conn.close()
            except Exception:
                pass

    server.add_url_rule(
        "/api/bars",
        endpoint="bars_rth",
        view_func=bars_rth,
        methods=["GET"],
    )
