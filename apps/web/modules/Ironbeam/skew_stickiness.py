"""Skew stickiness rows for the price chart's Overview panel (CR-CK Part 3). Read-only.

GET /api/ironbeam/skew-stickiness?trade_date=YYYY-MM-DD&expiration_date=YYYY-MM-DD&times=HH:MM,...[&until=...]

For each selected time after the first, over the window previous selected time -> this time (the panel's other Δ
columns compare against the previous selected time), the A14 stickiness read from `packages/shared/skew_expected.py`:
actual (OLS trend) Δ put skew, sticky-strike expected Δ for the same ES move, the ratio (|ES move| >= 0.15 %) and the
noise flag. 0DTE PM only. Put skew from `orats_monies_minute`; ES = `ironbeam_es_1m_bars` close at each snapshot's
quote time (CR-CJ A1). The route is fetched only when the panel's "Stickiness" setting is on; `/api/ironbeam/skew-data`
is untouched.

Registered via register_skew_stickiness_route(app.server, engine).
"""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd
from flask import jsonify, request
from sqlalchemy import text

from packages.shared import skew_expected as se

PT = ZoneInfo("America/Los_Angeles")
QUOTE_MAX_AGE_S = 300                 # as CR-CJ A1: snapshots whose quotes are > 5 min from the label are dropped
VOL_BUCKETS = tuple(range(5, 100, 5))

_SKEW_SQL = text(f"""
    SELECT snapshot_pt, snap_shot_date, quote_date, {", ".join(f"vol{n}" for n in VOL_BUCKETS)}
    FROM orats_monies_minute
    WHERE ticker = 'SPX' AND trade_date = :d AND expir_date = :d AND expiry_tod = 'pm'
      AND snapshot_pt >= :a AND snapshot_pt <= :b
    ORDER BY snapshot_pt
""")
_ES_SQL = text("""
    SELECT datetime, close FROM ironbeam_es_1m_bars
    WHERE datetime >= :a AND datetime < :b ORDER BY datetime
""")

_CORS_ORIGINS = {"http://localhost:5173", "http://127.0.0.1:5173", "http://0.0.0.0:5173"}


def _minute(hhmm: str) -> int | None:
    try:
        h, m = str(hhmm).strip().split(":")[:2]
        return int(h) * 60 + int(m)
    except (ValueError, TypeError):
        return None


def _hhmm(m: int) -> str:
    return f"{m // 60:02d}:{m % 60:02d}"


def _r(x, nd=2):
    return None if x is None else round(float(x), nd)


def build_skew_stickiness_rows(times: list[str], snaps: pd.DataFrame, es_close: dict) -> list[dict]:
    """Pure part. `snaps`: one row per snapshot with `minute` (PT minute of day of the label), `quote_minute`
    (quote time rounded to the minute) and vol5..vol95 (fractions). `es_close`: {bar start minute PT: close}.
    Returns one dict per selected time, in time order."""
    mins = sorted({m for m in (_minute(t) for t in times) if m is not None})
    by_min = snaps.drop_duplicates("minute").set_index("minute") if len(snaps) else snaps
    put = {}
    for m, row in by_min.iterrows():
        v75, v50 = row.get("vol75"), row.get("vol50")
        if pd.notna(v75) and pd.notna(v50):
            put[int(m)] = (float(v75) - float(v50)) * 100.0

    def es_at(m):
        if m not in by_min.index:
            return None
        q = by_min.at[m, "quote_minute"]
        return None if pd.isna(q) else es_close.get(int(q) - 1)     # close of the bar starting Q - 1

    out = []
    for i, m in enumerate(mins):
        row = {"time": _hhmm(m), "window": None, "actual": None, "expected": None, "ratio": None,
               "reason": "first time", "band": None, "beyond_noise": None, "es_move_pct": None, "late_day": False}
        if i > 0:
            a = mins[i - 1]
            row["window"] = f"{_hhmm(a)}-{_hhmm(m)}"
            row["late_day"] = m > se.LATE_DAY_MIN
            es0, es1 = es_at(a), es_at(m)
            if a not in put or m not in put or es0 is None or es1 is None:
                row["reason"] = "insufficient data"
            else:
                start = by_min.loc[a]
                vols = {n: float(start[f"vol{n}"]) for n in VOL_BUCKETS if pd.notna(start.get(f"vol{n}"))}
                ts = [k for k in sorted(put) if a <= k <= m]
                rd = se.stickiness_read(ts, [put[k] for k in ts], a, m, vols, es0, es1)
                row.update(actual=_r(rd.actual_pp), expected=_r(rd.expected_pp), ratio=_r(rd.ratio),
                           reason=rd.reason, band=_r(rd.band_pp), beyond_noise=rd.beyond_noise,
                           es_move_pct=_r(rd.es_move_pct, 3), late_day=rd.late_day)
        out.append(row)
    return out


def build_skew_stickiness_payload(conn, trade_date: str | None, expiration_date: str | None, times: list[str],
                                  until_pt: dt.datetime | None = None) -> tuple[dict, int]:
    if not trade_date or not expiration_date:
        return {"error": "Missing trade_date or expiration_date"}, 400
    try:
        d = dt.date.fromisoformat(trade_date)
    except ValueError:
        return {"error": "trade_date must be YYYY-MM-DD"}, 400
    mins = sorted({m for m in (_minute(t) for t in times) if m is not None})
    if until_pt is not None:
        cap = until_pt.hour * 60 + until_pt.minute
        mins = [m for m in mins if m <= cap]
    if trade_date != expiration_date:
        return {"stickiness": [{"time": _hhmm(m), "reason": "0DTE only"} for m in mins]}, 200
    if len(mins) < 2:
        return {"stickiness": build_skew_stickiness_rows([_hhmm(m) for m in mins], pd.DataFrame(), {})}, 200

    a = dt.datetime.combine(d, dt.time()) + dt.timedelta(minutes=mins[0])
    b = dt.datetime.combine(d, dt.time()) + dt.timedelta(minutes=mins[-1])
    snaps = pd.DataFrame(conn.execute(_SKEW_SQL, {"d": trade_date, "a": a, "b": b}).mappings().all())
    if snaps.empty:
        return {"stickiness": build_skew_stickiness_rows([_hhmm(m) for m in mins], snaps, {})}, 200
    lab = pd.to_datetime(snaps["snap_shot_date"], utc=True, errors="coerce")
    qts = pd.to_datetime(snaps["quote_date"], utc=True, errors="coerce")
    snaps = snaps[((lab - qts).dt.total_seconds().abs() <= QUOTE_MAX_AGE_S).to_numpy()].copy()
    qts = qts[snaps.index]
    snaps["minute"] = [t.hour * 60 + t.minute for t in pd.to_datetime(snaps["snapshot_pt"])]
    q_pt = qts.dt.tz_convert(PT)
    q_round = (q_pt + pd.Timedelta(seconds=30)).dt.floor("min")
    snaps["quote_minute"] = (q_round.dt.hour * 60 + q_round.dt.minute).to_numpy()
    for n in VOL_BUCKETS:
        snaps[f"vol{n}"] = pd.to_numeric(snaps[f"vol{n}"], errors="coerce")

    # `ironbeam_es_1m_bars.datetime` is naive UTC (bar start)
    utc = dt.timezone.utc
    es_a = (dt.datetime.combine(d, dt.time(), tzinfo=PT) + dt.timedelta(minutes=mins[0] - 5)).astimezone(utc)
    es_b = (dt.datetime.combine(d, dt.time(), tzinfo=PT) + dt.timedelta(minutes=mins[-1] + 1)).astimezone(utc)
    es_a, es_b = es_a.replace(tzinfo=None), es_b.replace(tzinfo=None)
    es = {}
    for r in conn.execute(_ES_SQL, {"a": es_a, "b": es_b}).mappings().all():
        t = pd.Timestamp(r["datetime"])
        t = (t.tz_localize("UTC") if t.tzinfo is None else t).tz_convert(PT)
        es[t.hour * 60 + t.minute] = float(r["close"])
    return {"stickiness": build_skew_stickiness_rows([_hhmm(m) for m in mins], snaps, es)}, 200


def register_skew_stickiness_route(server, engine, parse_until) -> None:
    if getattr(server, "_ironbeam_react_skew_stickiness_route_registered", False):
        return

    @server.route("/api/ironbeam/skew-stickiness", methods=["GET"])
    def ironbeam_react_skew_stickiness_api():
        times_str = request.args.get("times")
        times = times_str.split(",") if times_str else []
        with engine.connect() as conn:
            payload, status = build_skew_stickiness_payload(
                conn, request.args.get("trade_date"), request.args.get("expiration_date"), times,
                until_pt=parse_until(request.args.get("until"), PT),
            )
        resp = jsonify(payload)
        origin = request.headers.get("Origin")
        if origin in _CORS_ORIGINS:
            resp.headers["Access-Control-Allow-Origin"] = origin
            resp.headers["Access-Control-Allow-Credentials"] = "true"
            resp.headers["Vary"] = "Origin"
        return resp, status

    server._ironbeam_react_skew_stickiness_route_registered = True
