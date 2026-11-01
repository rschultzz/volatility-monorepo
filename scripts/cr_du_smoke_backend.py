"""CR-DU smoke backend: auth-free local Flask app for the price chart (Flow v3, the v1-engine inputs route) — port 8068.
Everything read-only: the Ironbeam routes (incl. /api/ironbeam/v1-engine/inputs), the trade-log and setup-review
routes all run on the backfill role with default_transaction_read_only=on (no PATCH / POST can write).
The Ironbeam engine is capped at 3 + 2 connections (Render DB has 103 slots; the 2026-10-28 starvation note).

    /Users/ryan/code/volatility-monorepo/apps/web/.venv/bin/python -u scripts/cr_du_smoke_backend.py
"""
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for line in open(os.path.join(REPO, ".env")):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())
for p in (REPO, os.path.join(REPO, "apps", "web")):
    sys.path.insert(0, p)

import dash  # noqa: E402
import sqlalchemy  # noqa: E402
from flask import request  # noqa: E402

import psycopg  # noqa: E402

from packages.shared.backfill_safety import connect_redacted  # noqa: E402

import modules.Ironbeam.callbacks as cb  # noqa: E402
from modules.SetupReview import routes as review_routes  # noqa: E402
from modules.TradeLog.routes import trade_log_bp  # noqa: E402

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.layout = dash.html.Div()
os.environ["DATABASE_URL"] = os.environ["BACKFILL_DATABASE_URL"]   # everything: backfill role, read-only
os.environ["PGOPTIONS"] = "-c default_transaction_read_only=on"
_ib_url = cb._get_db_url()
cb.engine = sqlalchemy.create_engine(_ib_url, pool_pre_ping=True, pool_size=3, max_overflow=2,
                                     connect_args={"options": "-c default_transaction_read_only=on"})
cb.register_ironbeam_callbacks(app)
app.server.register_blueprint(trade_log_bp)


# setup-review routes for the existing specs (as CR-DP's smoke backend): backfill role, read-only — any review POST fails
def _review_conn():
    return connect_redacted(os.environ["BACKFILL_DATABASE_URL"], options="-c default_transaction_read_only=on")


review_routes._conn = _review_conn
review_routes.register_setup_review_routes(app.server)


@app.server.after_request
def _smoke_cors(resp):
    o = request.headers.get("Origin")
    if o and o.startswith(("http://localhost:517", "http://localhost:518", "http://localhost:519")) and "Access-Control-Allow-Origin" not in resp.headers:
        resp.headers["Access-Control-Allow-Origin"] = o
        resp.headers["Access-Control-Allow-Credentials"] = "true"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
        resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PATCH, OPTIONS"
    return resp


@app.server.route("/api/trade-log/<path:_p>", methods=["OPTIONS"])
def _preflight(_p):
    return "", 204


@app.server.route("/api/setup-review/<path:_p>", methods=["OPTIONS"])
def _preflight_review(_p):
    return "", 204


if __name__ == "__main__":
    app.server.run(host="127.0.0.1", port=int(os.environ.get("CRDU_PORT", "8068")), debug=False, threaded=True)
