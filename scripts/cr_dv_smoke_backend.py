"""CR-DV smoke backend (copy of CR-DP's): auth-free local Flask app for the price chart with Flow v3. Port 8068.

Ironbeam routes on the backfill role with default_transaction_read_only=on; engine capped at 3 + 2 connections. No writes.

Simulated live session (weekend smoke): `GET /__crdv/sim?now=2026-10-30T15:40:15Z[&stall_from=...]` makes
/api/ironbeam/flow/cells for 2026-10-30 answer as if now were `now`: every flow_v3 read is cut to what existed then
(trades by received_at, es_flow_price_1m by CR-DD-live run completion, 1-s flow / bar closes by time) and, with
`stall_from`, no trade at or after it exists (a fixture gap for the feed-stall flag). `GET /__crdv/sim` clears it.

    /Users/ryan/code/volatility-monorepo/apps/web/.venv/bin/python -u scripts/cr_dv_smoke_backend.py
"""
import datetime as dt
import json
import os
import sys
import threading

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
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sqlalchemy  # noqa: E402
from flask import jsonify, request  # noqa: E402

import modules.Ironbeam.callbacks as cb  # noqa: E402
from modules.Ironbeam import flow_v2, flow_v3  # noqa: E402

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.layout = dash.html.Div()
cb.engine = sqlalchemy.create_engine(cb._get_db_url(), pool_pre_ping=True, pool_size=3, max_overflow=2)
cb.register_ironbeam_callbacks(app)

# ------------------------------------------------------------------------------ simulated now (2026-10-30 only)
U = dt.timezone.utc
SIM_DATE = dt.date(2026, 10, 30)
SYM = "XCME:ES.Z26"
A, B = dt.datetime(2026, 10, 30, 13, 30, tzinfo=U), dt.datetime(2026, 10, 31, 0, 0, tzinfo=U)
SIM = {"now": None, "stall_from": None, "on": False}
_DATA = {}
_LOCK = threading.RLock()


def _load():
    with _LOCK:
        if _DATA:
            return _DATA
        with cb.engine.connect() as c:
            tr = pd.read_sql(sqlalchemy.text(
                "SELECT extract(epoch FROM ts_utc)::float8 ts, extract(epoch FROM received_at)::float8 rcv, price, size, aggressor_side side "
                "FROM ironbeam_es_trades WHERE symbol = :s AND ts_utc >= :a AND ts_utc < :b"), c, params={"s": SYM, "a": A, "b": B})
            p1m = pd.read_sql(sqlalchemy.text(
                "SELECT extract(epoch FROM minute_utc)::bigint t, price, buy_vol::float8 buy, sell_vol::float8 sell "
                "FROM es_flow_price_1m WHERE symbol = :s AND source = 'ironbeam' AND minute_utc >= :a AND minute_utc < :b"),
                c, params={"s": SYM, "a": A, "b": B})
        tr["t"] = (tr["ts"] // 60 * 60).astype(np.int64)
        lags = json.load(open(os.path.join(REPO, "scripts/logs/cr-dv/q2_lags.json")))["lags"]
        rth = sorted(t for t in p1m["t"].unique() if A.timestamp() <= t < A.timestamp() + 390 * 60)
        readable = dict(zip(rth, [t + 60 + l for t, l in zip(rth, lags)]))   # q2_lag.py: the 390 RTH minutes
        p1m["rcv"] = p1m["t"].map(lambda t: readable.get(t, t + 60 + 600))
        _DATA.update(tr=tr, p1m=p1m)
        print(f"[crdv-sim] loaded {len(tr)} trades, {len(p1m)} price_1m rows", flush=True)
        return _DATA


def _cut_ts():
    T = SIM["now"].timestamp()
    st = SIM["stall_from"].timestamp() if SIM["stall_from"] else None
    return T, st


_orig = {n: getattr(flow_v3, n) for n in ("fetch_trade_cells", "fetch_price_1m", "_trades_last", "build_cells_payload")}
_orig_flow, _orig_bars = flow_v2.fetch_flow_minutes, flow_v2.fetch_bar_closes


def f_trade_cells(engine, symbol, a, b):
    if not SIM["on"] or symbol != SYM:
        return _orig["fetch_trade_cells"](engine, symbol, a, b)
    tr = _load()["tr"]
    T, st = _cut_ts()
    k = (tr["ts"] >= a.timestamp()) & (tr["ts"] < b.timestamp()) & (tr["rcv"] <= T)
    if st is not None:
        k &= tr["ts"] < st
    return tr[k].groupby(["t", "price", "side"], as_index=False)["size"].sum().astype({"size": float})[["t", "price", "side", "size"]]


def f_price_1m(engine, symbol, source, a, b):
    if not SIM["on"] or symbol != SYM:
        return _orig["fetch_price_1m"](engine, symbol, source, a, b)
    p = _load()["p1m"]
    T, st = _cut_ts()
    k = (p["rcv"] <= T) & (p["t"] >= a.timestamp()) & (p["t"] < b.timestamp())
    if st is not None:
        k &= p["t"] + 60 <= st
    return p[k][["t", "price", "buy", "sell"]].reset_index(drop=True)


def f_trades_last(engine, symbol):
    if not SIM["on"] or symbol != SYM:
        return _orig["_trades_last"](engine, symbol)
    tr = _load()["tr"]
    T, st = _cut_ts()
    k = tr["rcv"] <= T
    if st is not None:
        k &= tr["ts"] < st
    return dt.datetime.fromtimestamp(float(tr.loc[k, "ts"].max()), U) if k.any() else None


def f_flow(engine, table, ranges, now_utc=None):
    if not SIM["on"]:
        return _orig_flow(engine, table, ranges, now_utc=now_utc)
    cut = SIM["now"] - dt.timedelta(seconds=2)
    if SIM["stall_from"]:
        cut = min(cut, SIM["stall_from"])
    return _orig_flow(engine, table, [(a, min(b, cut)) for a, b in ranges if a < cut], now_utc=SIM["now"])


def f_bars(engine, a, b):
    if not SIM["on"]:
        return _orig_bars(engine, a, b)
    return _orig_bars(engine, a, min(b, flow_v2._floor_min(SIM["now"])))


def f_build(engine, trade_date, *, session, until_pt, params, session_date, session_note, now_utc=None):
    if SIM["now"] is not None and session_date == SIM_DATE and until_pt is None:
        with _LOCK:   # one simulated request at a time (the read wrappers share SIM)
            SIM["on"] = True
            try:
                flow_v3._CACHE.clear()
                return _orig["build_cells_payload"](engine, trade_date, session=session, until_pt=None, params=params,
                                                    session_date=session_date, session_note=session_note, now_utc=SIM["now"])
            finally:
                SIM["on"] = False
    return _orig["build_cells_payload"](engine, trade_date, session=session, until_pt=until_pt, params=params,
                                        session_date=session_date, session_note=session_note, now_utc=now_utc)


flow_v3.fetch_trade_cells, flow_v3.fetch_price_1m, flow_v3._trades_last = f_trade_cells, f_price_1m, f_trades_last
flow_v3.build_cells_payload = f_build
flow_v2.fetch_flow_minutes, flow_v2.fetch_bar_closes = f_flow, f_bars


@app.server.route("/__crdv/sim", methods=["GET"])
def _sim():
    def iso(v):
        return dt.datetime.fromisoformat(v.replace("Z", "+00:00")).astimezone(U) if v else None
    SIM["now"], SIM["stall_from"] = iso(request.args.get("now")), iso(request.args.get("stall_from"))
    if SIM["now"] is not None:
        _load()
    return jsonify({k: (v.isoformat() if isinstance(v, dt.datetime) else v) for k, v in SIM.items()})


DIST = os.environ.get("CRDV_DIST")   # optional: a `vite build --base=/crdv/` of react_price_preview, served at /crdv/


@app.server.route("/crdv/", defaults={"path": "index.html"})
@app.server.route("/crdv/<path:path>")
def _dist(path):
    from flask import send_from_directory
    if not DIST:
        return "CRDV_DIST not set", 404
    return send_from_directory(DIST, path if os.path.exists(os.path.join(DIST, path)) else "index.html")


@app.server.after_request
def _smoke_cors(resp):
    o = request.headers.get("Origin")
    if o and o.startswith(("http://localhost:517", "http://localhost:518", "http://localhost:519")) and "Access-Control-Allow-Origin" not in resp.headers:
        resp.headers["Access-Control-Allow-Origin"] = o
        resp.headers["Access-Control-Allow-Credentials"] = "true"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


if __name__ == "__main__":
    app.server.run(host="127.0.0.1", port=int(os.environ.get("CRDV_PORT", "8068")), debug=False, threaded=True)
