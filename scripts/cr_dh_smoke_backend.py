"""CR-DH smoke backend (from CR-DF's): auth-free local app with the Dashboard GEX block at / and the price
chart API (Ironbeam + setup-review routes). Port 8067.

    apps/web/.venv/bin/python -u scripts/cr_dh_smoke_backend.py

ALL DB access on the backfill role with default_transaction_read_only=on; engines capped at 3 + 2
connections (Render DB has 103 slots). No writes.

Dashboard page: /?date=YYYY-MM-DD drives the real `trade-date` → `render_gex` callback (modules.gamma),
with the real `gex_block()` (Levels switch, local persistence). The live timer never fires.
/smoke/origin-figure?date=… renders origin/main's modules/gamma/callbacks.py (read with `git show`)
in-process on the same DB, for the Current byte-identity check.
"""
import datetime as dt
import importlib.util
import os
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

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
from dash import Input, Output, dcc, html  # noqa: E402
from flask import jsonify, request  # noqa: E402

import modules.Ironbeam.callbacks as cb  # noqa: E402
from modules.SetupReview import routes as review_routes  # noqa: E402
from modules.gamma import callbacks as gamma  # noqa: E402  (registers render_gex)
from modules.gamma.components import gex_block  # noqa: E402

app = dash.Dash(__name__, suppress_callback_exceptions=True)
cb.engine = sqlalchemy.create_engine(cb._get_db_url(), pool_pre_ping=True, pool_size=3, max_overflow=2)
cb.register_ironbeam_callbacks(app)
# gamma's fallback get_engine() builds a new engine per call; share one small engine here.
gamma.get_engine = lambda: cb.engine

app.layout = html.Div(
    [
        dcc.Location(id="url"),
        dcc.DatePickerSingle(id="trade-date", date=None),
        dcc.Interval(id="live-update-timer", interval=24 * 3600 * 1000, n_intervals=0),
        html.Div(gex_block(), style={"height": "70vh", "width": "560px"}),
    ],
    style={"background": "#111", "padding": "8px"},
)


@dash.callback(Output("trade-date", "date"), Input("url", "search"))
def _date_from_url(search):
    from urllib.parse import parse_qs
    return (parse_qs((search or "").lstrip("?")).get("date") or [None])[0]


_ORIGIN = None


def _origin_gamma():
    global _ORIGIN
    if _ORIGIN is None:
        src = subprocess.run(["git", "show", "origin/main:apps/web/modules/gamma/callbacks.py"],
                             cwd=REPO, check=True, capture_output=True, text=True).stdout
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(src)
        spec = importlib.util.spec_from_file_location("origin_gamma_callbacks", f.name)
        mod = importlib.util.module_from_spec(spec)
        with patch.object(dash, "callback", lambda *a, **k: (lambda fn: fn)):
            spec.loader.exec_module(mod)
        os.unlink(f.name)
        mod.get_engine = lambda: cb.engine
        _ORIGIN = mod
    return _ORIGIN


@app.server.route("/smoke/origin-figure")
def _origin_figure():
    d = request.args.get("date")
    with patch.object(dash, "ctx", SimpleNamespace(triggered=[], triggered_id=None)):
        fig = _origin_gamma().render_gex(d, 0, None)
    # Plotly 6 base64-encodes arrays in to_json(); the browser compares plain lists.
    lay = fig.layout
    return jsonify({
        "data": [{"name": t.name, "x": [float(v) for v in t.x], "y": [float(v) for v in t.y]} for t in fig.data],
        "layout": {
            "title": {"text": lay.title.text},
            "xaxis": {"title": {"text": lay.xaxis.title.text}, "range": list(lay.xaxis.range) if lay.xaxis.range else None},
            "yaxis": {"title": {"text": lay.yaxis.title.text}, "range": list(lay.yaxis.range) if lay.yaxis.range else None},
        },
    })


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
    app.server.run(host="127.0.0.1", port=int(os.environ.get("CRDH_PORT", "8067")), debug=False, threaded=True)
