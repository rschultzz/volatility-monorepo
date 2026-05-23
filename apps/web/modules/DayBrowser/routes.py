"""DayBrowser routes — Flask wiring for GET /api/days (CR-016).

Endpoint:
    GET /api/days?regime=pin[&from=YYYY-MM-DD][&to=YYYY-MM-DD][&ticker=SPX]

Returns all corpus days whose effective regime matches the requested label,
within the date range. Effective regime applies promoted audit overrides.

Defaults:
  from  — 30 days back from today
  to    — today
  ticker — SPX
"""
from __future__ import annotations

import datetime as dt
import os
from typing import Optional

import psycopg
from flask import jsonify, request

from .service import query_days_by_regime

_VALID_REGIMES = frozenset({
    "magnetic-pin", "magnet-above", "magnet-below",
    "bounded", "amplification", "untethered", "broken-magnet",
    "pinned",  # corrected-regime alias
})


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


def _parse_date(s) -> Optional[dt.date]:
    try:
        return dt.date.fromisoformat(str(s))
    except (TypeError, ValueError):
        return None


def register_day_browser_routes(server) -> None:
    """Wire GET /api/days onto the Flask server."""
    if "day_browser_list" in server.view_functions:
        return

    def day_browser_list():
        regime = (request.args.get("regime") or "").strip()
        if not regime:
            return jsonify({"ok": False, "error": "regime is required"}), 400

        today = dt.date.today()
        date_from = _parse_date(request.args.get("from")) or (today - dt.timedelta(days=30))
        date_to = _parse_date(request.args.get("to")) or today
        ticker = (request.args.get("ticker") or "SPX").strip() or "SPX"

        if date_from > date_to:
            return jsonify({"ok": False, "error": "from must be ≤ to"}), 400

        try:
            conn = _conn()
        except Exception as e:
            return jsonify({"ok": False, "error": f"db connect failed: {e}"}), 500

        try:
            days = query_days_by_regime(conn, ticker, regime, date_from, date_to)
            return jsonify({
                "ok": True,
                "regime": regime,
                "ticker": ticker,
                "from": date_from.isoformat(),
                "to": date_to.isoformat(),
                "count": len(days),
                "days": days,
            })
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500
        finally:
            try:
                conn.close()
            except Exception:
                pass

    server.add_url_rule(
        "/api/days",
        endpoint="day_browser_list",
        view_func=day_browser_list,
        methods=["GET"],
    )
