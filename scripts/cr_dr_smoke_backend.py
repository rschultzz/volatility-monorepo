"""CR-DR smoke backend: auth-free local Flask app for the price chart (Flow v3) and the Trade Log routes. Port 8069.

The Ironbeam engine is capped at 3 + 2 connections (Render DB has 103 slots; see the 2026-10-28 starvation note).
Ironbeam routes run on the backfill role with default_transaction_read_only=on. The trade-log routes use the
owner role (DATABASE_URL) because the smoke saves chips / checklist / notes on Ryan's trades (PATCH only).

    /Users/ryan/code/volatility-monorepo/apps/web/.venv/bin/python -u scripts/cr_dr_smoke_backend.py
    ... --dump OUT_DIR TRADE_DATE UNTIL      (no server: write the three until-replay payloads as JSON via the test client)
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

import modules.Ironbeam.callbacks as cb  # noqa: E402
from modules.TradeLog.routes import trade_log_bp  # noqa: E402

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.layout = dash.html.Div()
_owner_url = os.environ["DATABASE_URL"]
os.environ["DATABASE_URL"] = os.environ["BACKFILL_DATABASE_URL"]   # Ironbeam: backfill role, read-only
_ib_url = cb._get_db_url()
os.environ["DATABASE_URL"] = _owner_url                              # trade log: owner (PATCH review fields)
cb.engine = sqlalchemy.create_engine(_ib_url, pool_pre_ping=True, pool_size=3, max_overflow=2,
                                     connect_args={"options": "-c default_transaction_read_only=on"})
cb.register_ironbeam_callbacks(app)
app.server.register_blueprint(trade_log_bp)


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


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--dump":
        out, day, until = sys.argv[2], sys.argv[3], sys.argv[4]
        c = app.server.test_client()
        reqs = {
            "cells": f"/api/ironbeam/flow/cells?trade_date={day}&session=FULL&a_k=3&a_m=3&until={until}",
            "v2": f"/api/ironbeam/flow?trade_date={day}&session=FULL&mode=v2&until={until}&effort=1",
            "bars": f"/api/ironbeam/bars?trade_date={day}&interval=1min&gex_enabled=0&phase=center&until={until}",
        }
        for k, url in reqs.items():
            r = c.get(url)
            print(k, r.status_code)
            with open(os.path.join(out, f"{k}.json"), "w") as fh:
                json.dump(r.get_json(), fh)
        sys.exit(0)
    app.server.run(host="127.0.0.1", port=int(os.environ.get("CRDR_PORT", "8069")), debug=False, threaded=True)
