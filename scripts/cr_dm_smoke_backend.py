"""CR-DM smoke backend (copy of CR-DK's): auth-free local Flask app for the price chart with Flow v3. Port 8065.

The Ironbeam engine is capped at 3 + 2 connections (Render DB has 103 slots; see the 2026-10-28 starvation note).

    apps/web/.venv/bin/python -u scripts/cr_dm_smoke_backend.py

Ironbeam routes (incl. /api/ironbeam/flow/cells) and the setup-review routes, ALL on the backfill
role with default_transaction_read_only=on: the smoke only GETs candidates; the chip test aborts the review POST in the browser (any review POST
fails at the DB). No writes.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for line in open(os.path.join(REPO, ".env")):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())
os.environ["DATABASE_URL"] = os.environ["BACKFILL_DATABASE_URL"]
os.environ["PGOPTIONS"] = "-c default_transaction_read_only=on"
for p in (REPO, os.path.join(REPO, "apps", "web")):
    sys.path.insert(0, p)

import dash  # noqa: E402
import psycopg  # noqa: E402
import sqlalchemy  # noqa: E402
from flask import request  # noqa: E402

import modules.Ironbeam.callbacks as cb  # noqa: E402
from modules.SetupReview import routes as review_routes  # noqa: E402

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.layout = dash.html.Div()
cb.engine = sqlalchemy.create_engine(cb._get_db_url(), pool_pre_ping=True, pool_size=3, max_overflow=2)
cb.register_ironbeam_callbacks(app)


def _review_conn():
    return psycopg.connect(review_routes._normalize_db_url(os.environ["BACKFILL_DATABASE_URL"]),
                           options="-c default_transaction_read_only=on")


review_routes._conn = _review_conn
review_routes.register_setup_review_routes(app.server)


@app.server.after_request
def _smoke_cors(resp):
    o = request.headers.get("Origin")
    if o and o.startswith(("http://localhost:517", "http://localhost:518")) and "Access-Control-Allow-Origin" not in resp.headers:
        resp.headers["Access-Control-Allow-Origin"] = o
        resp.headers["Access-Control-Allow-Credentials"] = "true"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.server.route("/api/setup-review/<path:_p>", methods=["OPTIONS"])
def _preflight(_p):
    return "", 204


if __name__ == "__main__":
    app.server.run(host="127.0.0.1", port=int(os.environ.get("CRDF_PORT", "8065")), debug=False, threaded=True)
