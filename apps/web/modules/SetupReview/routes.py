"""SetupReview routes — Flask wiring for the CR-CC setup review queue (L4).

Endpoints (all JSON):
    GET  /api/setup-review/candidates?cr=&set=&cell=&status=unreviewed|reviewed|all
         list without outcomes, plus the known sets and reason chips
    GET  /api/setup-review/candidates/<id>
         detail; `outcome` only once the candidate has a review
    POST /api/setup-review/candidates/<id>/review   {label, reasons[], comment}
         insert-only. Before reveal: the blind label (label required, first save only;
         409 label_locked after). After reveal: a post_reveal note (label optional, copies
         the blind label when absent). Readers of labels use the FIRST pre-reveal review
         per candidate; post-reveal rows are notes only (CR-DE).
    POST /api/setup-review/candidates/<id>/reveal   sets revealed_at, returns the outcome
    GET  /api/setup-review/summary?cr=&set=

DB errors are mapped, not raised (A1.5): a missing privilege returns 403 with
`error: "permission"` and the connected role; missing tables return 503; the
insert-only guard trigger returns 409.
"""
from __future__ import annotations

import logging
import os
import threading

import psycopg
from flask import jsonify, request

from . import service

log = logging.getLogger(__name__)

PREFIX = "/api/setup-review"


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


def log_db_role_once() -> None:
    """Log the web service's DB role name once (never the URL), in the background (A1.5)."""

    def _run():
        try:
            with _conn() as conn:
                log.warning("CR-CC setup-review: web DB role = %s", service.current_role(conn))
        except Exception as e:  # startup must not depend on the DB
            log.warning("CR-CC setup-review: could not read web DB role (%s)", type(e).__name__)

    threading.Thread(target=_run, name="setup-review-role", daemon=True).start()


def _db_error(e: Exception, conn=None):
    if isinstance(e, psycopg.errors.InsufficientPrivilege):
        role = None
        try:
            if conn is not None:
                conn.rollback()
                role = service.current_role(conn)
        except Exception:
            pass
        return jsonify({
            "ok": False,
            "error": "permission",
            "message": f"DB role {role or '(unknown)'} lacks a privilege on the setup review tables "
                       "(needs SELECT, INSERT on setup_candidates / setup_reviews and UPDATE (revealed_at) "
                       "on setup_reviews; see scripts/cr_cc_ddl.sql)",
            "role": role,
        }), 403
    if isinstance(e, psycopg.errors.UndefinedTable):
        return jsonify({"ok": False, "error": "missing_tables",
                        "message": "setup review tables are not installed (scripts/cr_cc_ddl.sql)"}), 503
    if isinstance(e, psycopg.errors.RaiseException):
        return jsonify({"ok": False, "error": "insert_only", "message": str(e).splitlines()[0]}), 409
    log.exception("setup-review DB error")
    return jsonify({"ok": False, "error": "db", "message": str(e)}), 500


def _run(fn, *, write=False):
    """Open a connection, run fn(conn), commit on write, map DB errors."""
    try:
        conn = _conn()
    except Exception as e:
        return jsonify({"ok": False, "error": "db_connect", "message": str(e)}), 500
    try:
        out = fn(conn)
        if write:
            conn.commit()
        return out
    except service.ReviewError as e:
        conn.rollback()
        return jsonify({"ok": False, "error": e.code, "message": str(e)}), e.status
    except psycopg.Error as e:
        return _db_error(e, conn)
    finally:
        try:
            conn.close()
        except Exception:
            pass


def register_setup_review_routes(server) -> None:
    """Wire the /api/setup-review/* routes onto the Flask server."""
    if "setup_review_list" in server.view_functions:
        return

    def setup_review_list():
        status = (request.args.get("status") or "all").strip().lower()
        if status not in service.STATUSES:
            return jsonify({"ok": False, "error": "bad_request",
                            "message": f"status must be one of {service.STATUSES}"}), 400
        args = dict(
            cr=(request.args.get("cr") or "").strip() or None,
            set_name=(request.args.get("set") or "").strip() or None,
            cell=(request.args.get("cell") or "").strip() or None,
        )

        def fn(conn):
            return jsonify({
                "ok": True,
                "candidates": service.list_candidates(conn, status=status, **args),
                "sets": service.list_sets(conn),
                "reasons": service.known_reasons(conn),
            })

        return _run(fn)

    def setup_review_detail(candidate_id):
        def fn(conn):
            cand = service.get_candidate(conn, candidate_id)
            if cand is None:
                return jsonify({"ok": False, "error": "not_found"}), 404
            return jsonify({"ok": True, "candidate": cand})

        return _run(fn)

    def setup_review_add(candidate_id):
        body = request.get_json(silent=True) or {}
        return _run(lambda conn: jsonify({"ok": True, "review": service.add_review(conn, candidate_id, body)}),
                    write=True)

    def setup_review_reveal(candidate_id):
        return _run(lambda conn: jsonify({"ok": True, **service.reveal(conn, candidate_id)}), write=True)

    def setup_review_summary():
        args = dict(
            cr=(request.args.get("cr") or "").strip() or None,
            set_name=(request.args.get("set") or "").strip() or None,
        )
        return _run(lambda conn: jsonify({"ok": True, "groups": service.summary(conn, **args)}))

    server.add_url_rule(f"{PREFIX}/candidates", endpoint="setup_review_list",
                        view_func=setup_review_list, methods=["GET"])
    server.add_url_rule(f"{PREFIX}/candidates/<path:candidate_id>/review", endpoint="setup_review_add",
                        view_func=setup_review_add, methods=["POST"])
    server.add_url_rule(f"{PREFIX}/candidates/<path:candidate_id>/reveal", endpoint="setup_review_reveal",
                        view_func=setup_review_reveal, methods=["POST"])
    server.add_url_rule(f"{PREFIX}/candidates/<path:candidate_id>", endpoint="setup_review_detail",
                        view_func=setup_review_detail, methods=["GET"])
    server.add_url_rule(f"{PREFIX}/summary", endpoint="setup_review_summary",
                        view_func=setup_review_summary, methods=["GET"])
