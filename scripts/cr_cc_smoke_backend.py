"""CR-CC smoke backend: auth-free local Flask app for the price chart + setup review queue. Port 8061.

    apps/web/.venv/bin/python -u scripts/cr_cc_smoke_backend.py

- Ironbeam routes (/api/ironbeam/*, /api/gex-landscape) and the other read routes run on the
  backfill role with default_transaction_read_only=on (as scripts/cr_bv_smoke_backend.py).
- /api/setup-review/* runs on DATABASE_URL (the web role) with read-only switched off for its
  own connections, because Save / Reveal write setup_reviews (insert-only; the guard trigger
  refuses every other change).
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for line in open(os.path.join(REPO, ".env")):
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())
WEB_DATABASE_URL = os.environ["DATABASE_URL"]
os.environ["DATABASE_URL"] = os.environ["BACKFILL_DATABASE_URL"]
os.environ["PGOPTIONS"] = "-c default_transaction_read_only=on"
for p in (REPO, os.path.join(REPO, "apps", "web")):
    sys.path.insert(0, p)

import dash  # noqa: E402
import psycopg  # noqa: E402
from flask import request  # noqa: E402

import modules.Ironbeam.callbacks as cb  # noqa: E402
from modules.Analogues.routes import register_analogues_routes  # noqa: E402
from modules.SetupReview import routes as review_routes  # noqa: E402

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.layout = dash.html.Div()
cb.register_ironbeam_callbacks(app)
register_analogues_routes(app.server)


def _review_conn():
    return psycopg.connect(review_routes._normalize_db_url(WEB_DATABASE_URL),
                           options="-c default_transaction_read_only=off")


review_routes._conn = _review_conn
review_routes.register_setup_review_routes(app.server)


@app.server.after_request
def _smoke_cors(resp):
    o = request.headers.get("Origin")
    if o and o.startswith("http://localhost:517") and "Access-Control-Allow-Origin" not in resp.headers:
        resp.headers["Access-Control-Allow-Origin"] = o
        resp.headers["Access-Control-Allow-Credentials"] = "true"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.server.route("/api/setup-review/<path:_p>", methods=["OPTIONS"])
def _preflight(_p):
    return "", 204


if __name__ == "__main__":
    review_routes.log_db_role_once()
    app.server.run(host="127.0.0.1", port=8061, debug=False)
