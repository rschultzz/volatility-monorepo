"""AuditFlags routes — Flask wiring for /api/audit-flags (CR-016).

Endpoints:
    POST   /api/audit-flags                  — create a flag
    DELETE /api/audit-flags/<flag_id>        — remove a flag
    POST   /api/audit-flags/<flag_id>/promote — promote a regime_wrong flag
    POST   /api/audit-flags/<flag_id>/demote  — demote a regime_wrong flag
    GET    /api/audit-flags?date=&ticker=     — list flags for a date
"""
from __future__ import annotations

import datetime as dt
import os
from typing import Optional

import psycopg
from flask import jsonify, request

from .service import (
    create_flag,
    delete_flag,
    demote_flag,
    list_flags_for_date,
    promote_flag,
)


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


def register_audit_flags_routes(server) -> None:
    """Wire all /api/audit-flags endpoints onto the Flask server."""
    if "audit_flags_create" in server.view_functions:
        return

    # ── POST /api/audit-flags ────────────────────────────────────────────────
    def audit_flags_create():
        body = request.get_json(silent=True) or {}
        flag_type = body.get("flag_type", "").strip()
        ticker = (body.get("ticker") or "SPX").strip() or "SPX"
        trade_date = _parse_date(body.get("trade_date"))
        analogue_date = _parse_date(body.get("analogue_date")) if body.get("analogue_date") else None
        corrected_regime = (body.get("corrected_regime") or "").strip() or None
        note = body.get("note") or None

        if not trade_date:
            return jsonify({"ok": False, "error": "trade_date is required (YYYY-MM-DD)"}), 400

        try:
            conn = _conn()
        except Exception as e:
            return jsonify({"ok": False, "error": f"db connect failed: {e}"}), 500

        try:
            flag = create_flag(
                conn, flag_type, ticker, trade_date,
                analogue_date=analogue_date,
                corrected_regime=corrected_regime,
                note=note,
            )
            conn.commit()
            return jsonify({"ok": True, "flag": flag}), 201
        except ValueError as e:
            return jsonify({"ok": False, "error": str(e)}), 400
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500
        finally:
            try:
                conn.close()
            except Exception:
                pass

    server.add_url_rule("/api/audit-flags", endpoint="audit_flags_create",
                        view_func=audit_flags_create, methods=["POST"])

    # ── DELETE /api/audit-flags/<flag_id> ────────────────────────────────────
    def audit_flags_delete(flag_id):
        try:
            fid = int(flag_id)
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "flag_id must be an integer"}), 400

        try:
            conn = _conn()
        except Exception as e:
            return jsonify({"ok": False, "error": f"db connect failed: {e}"}), 500

        try:
            deleted = delete_flag(conn, fid)
            conn.commit()
            if not deleted:
                return jsonify({"ok": False, "error": f"flag {fid} not found"}), 404
            return jsonify({"ok": True})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500
        finally:
            try:
                conn.close()
            except Exception:
                pass

    server.add_url_rule("/api/audit-flags/<flag_id>", endpoint="audit_flags_delete",
                        view_func=audit_flags_delete, methods=["DELETE"])

    # ── POST /api/audit-flags/<flag_id>/promote ──────────────────────────────
    def audit_flags_promote(flag_id):
        try:
            fid = int(flag_id)
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "flag_id must be an integer"}), 400

        try:
            conn = _conn()
        except Exception as e:
            return jsonify({"ok": False, "error": f"db connect failed: {e}"}), 500

        try:
            flag = promote_flag(conn, fid)
            conn.commit()
            return jsonify({"ok": True, "flag": flag})
        except ValueError as e:
            return jsonify({"ok": False, "error": str(e)}), 404
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500
        finally:
            try:
                conn.close()
            except Exception:
                pass

    server.add_url_rule("/api/audit-flags/<flag_id>/promote", endpoint="audit_flags_promote",
                        view_func=audit_flags_promote, methods=["POST"])

    # ── POST /api/audit-flags/<flag_id>/demote ───────────────────────────────
    def audit_flags_demote(flag_id):
        try:
            fid = int(flag_id)
        except (TypeError, ValueError):
            return jsonify({"ok": False, "error": "flag_id must be an integer"}), 400

        try:
            conn = _conn()
        except Exception as e:
            return jsonify({"ok": False, "error": f"db connect failed: {e}"}), 500

        try:
            flag = demote_flag(conn, fid)
            conn.commit()
            return jsonify({"ok": True, "flag": flag})
        except ValueError as e:
            return jsonify({"ok": False, "error": str(e)}), 404
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500
        finally:
            try:
                conn.close()
            except Exception:
                pass

    server.add_url_rule("/api/audit-flags/<flag_id>/demote", endpoint="audit_flags_demote",
                        view_func=audit_flags_demote, methods=["POST"])

    # ── GET /api/audit-flags?date=&ticker= ───────────────────────────────────
    def audit_flags_list():
        date_s = (request.args.get("date") or "").strip()
        trade_date = _parse_date(date_s)
        if not trade_date:
            return jsonify({"ok": False, "error": "date is required (YYYY-MM-DD)"}), 400

        ticker = (request.args.get("ticker") or "SPX").strip() or "SPX"

        try:
            conn = _conn()
        except Exception as e:
            return jsonify({"ok": False, "error": f"db connect failed: {e}"}), 500

        try:
            flags = list_flags_for_date(conn, ticker, trade_date)
            return jsonify({"ok": True, "flags": flags})
        except Exception as e:
            return jsonify({"ok": False, "error": str(e)}), 500
        finally:
            try:
                conn.close()
            except Exception:
                pass

    server.add_url_rule("/api/audit-flags", endpoint="audit_flags_list",
                        view_func=audit_flags_list, methods=["GET"])
