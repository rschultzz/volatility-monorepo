"""CR-CK smoke backend: auth-free local Flask app for the price chart (Overview panel + Stickiness route). Port 8064.

    apps/web/.venv/bin/python -u scripts/cr_ck_smoke_backend.py

Ironbeam routes (/api/ironbeam/*, /api/gex-landscape) only, on the backfill role with
default_transaction_read_only=on (as scripts/cr_bv_smoke_backend.py). No writes.
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
from flask import request  # noqa: E402

import modules.Ironbeam.callbacks as cb  # noqa: E402

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.layout = dash.html.Div()
cb.register_ironbeam_callbacks(app)


@app.server.after_request
def _smoke_cors(resp):
    o = request.headers.get("Origin")
    if o and o.startswith("http://localhost:517") and "Access-Control-Allow-Origin" not in resp.headers:
        resp.headers["Access-Control-Allow-Origin"] = o
        resp.headers["Access-Control-Allow-Credentials"] = "true"
    return resp


if __name__ == "__main__":
    app.server.run(host="127.0.0.1", port=int(os.environ.get("CRCK_PORT", "8064")), debug=False, threaded=True)
