"""CR-DC — `GET /api/ironbeam/flow/cells`: Flow v3 price cells and absorption marks (read-only).

Per session, the same two segments as Flow v2 (`flow_v2.py`), each from one source and one contract:
  - ETH  [prior 15:00, 06:30) PT (FULL view only): Ironbeam;
  - RTH  [06:30, 13:00) PT: Databento for sessions 2024-04-03 .. 2026-06-05 it covers (B4 / CR-CD A1.1),
    otherwise Ironbeam. The contract is v2's (A1.4: the one whose minute closes match the chart's bars).
Each segment's cells come from the first tier that covers it (spec A1.3 / DC17):
  1. `es_flow_price_1m` (exact; CR-DD) — skipped while the table does not exist;
  2. `ironbeam_es_trades` (exact) — Ironbeam segments, when the oldest retained trade <= segment start;
  3. the 1-second table (approximate: each second's sell at its low, buy at its high).
Absorption marks (DC9) are computed for the RTH segment on exact tiers only (A1.2). Nothing after `until`
is read. The DB is read only through the `fetch_*` / `_exists_price_1m` / `_trades_oldest` functions, so
tests can record and replay them; completed segments are cached in-process, live ones incrementally (DC14).
"""
from __future__ import annotations

import datetime as dt
import threading
import time

import numpy as np
import pandas as pd
from sqlalchemy import text

from modules.Ironbeam import flow_v2
from packages.shared import flow_heat as fh

PRICE_1M_TABLE = "es_flow_price_1m"
TRADES_TABLE = "ironbeam_es_trades"
PRICE_1M_COVERAGE = 0.99            # A1 readings: minutes in price_1m / minutes with trades in the 1-minute flow
LIVE_RECHECK_MIN = 2                # DC14: a live poll re-reads from the last cached minute - 2 min
TRADES_OLDEST_TTL_S = 60.0
TABLE_RECHECK_S = 300.0
CACHE_MAX = 500

_CACHE: dict[tuple, object] = {}
_CACHE_LOCK = threading.Lock()
_STATE: dict[str, tuple[float, object]] = {}

BADGES = {"price_1m": "exact: es_flow_price_1m", "trades": "exact: trades", "seconds": "approximate: 1-s data"}


# ------------------------------------------------------------------------------------------ DB reads
def _utc(x: dt.datetime) -> dt.datetime:
    return x.astimezone(flow_v2.UTC)


def _exists_price_1m(engine) -> bool:
    with engine.connect() as conn:
        return bool(conn.execute(text(f"SELECT to_regclass('{PRICE_1M_TABLE}') IS NOT NULL")).scalar())


def _trades_oldest(engine, symbol: str) -> dt.datetime | None:
    """Oldest retained trade of `symbol` (index `(symbol, ts_utc DESC)`; a full min() scans the table)."""
    with engine.connect() as conn:
        return conn.execute(text(f"SELECT ts_utc FROM {TRADES_TABLE} WHERE symbol = :s ORDER BY ts_utc ASC LIMIT 1"),
                            {"s": symbol}).scalar()


def fetch_price_1m(engine, symbol: str, source: str, a: dt.datetime, b: dt.datetime) -> pd.DataFrame:
    q = text(f"""SELECT extract(epoch FROM minute_utc)::bigint AS t, price, buy_vol::float8 AS buy, sell_vol::float8 AS sell
                 FROM {PRICE_1M_TABLE} WHERE symbol = :s AND source = :src AND minute_utc >= :a AND minute_utc < :b""")
    with engine.connect() as conn:
        return pd.read_sql(q, conn, params={"s": symbol, "src": source, "a": _utc(a), "b": _utc(b)})


def fetch_trade_cells(engine, symbol: str, a: dt.datetime, b: dt.datetime) -> pd.DataFrame:
    """Trades summed per (minute, price, aggressor side) in SQL — the same sums `cells_from_trades` makes."""
    q = text(f"""SELECT (floor(extract(epoch FROM ts_utc) / 60) * 60)::bigint AS t, price, aggressor_side AS side,
                        sum(size)::float8 AS size
                 FROM {TRADES_TABLE} WHERE symbol = :s AND ts_utc >= :a AND ts_utc < :b GROUP BY 1, 2, 3""")
    with engine.connect() as conn:
        return pd.read_sql(q, conn, params={"s": symbol, "a": _utc(a), "b": _utc(b)})


def fetch_seconds(engine, table: str, symbol: str, a: dt.datetime, b: dt.datetime) -> pd.DataFrame:
    q = text(f"""SELECT extract(epoch FROM ts_utc)::bigint AS ts, high, low, buy_vol::float8 AS buy, sell_vol::float8 AS sell
                 FROM {table} WHERE symbol = :s AND ts_utc >= :a AND ts_utc < :b""")
    with engine.connect() as conn:
        return pd.read_sql(q, conn, params={"s": symbol, "a": _utc(a), "b": _utc(b)})


def _cached(key: str, ttl: float, fn):
    mono = time.monotonic()
    hit = _STATE.get(key)
    if hit is not None and mono - hit[0] <= ttl:
        return hit[1]
    val = fn()
    _STATE[key] = (mono, val)
    return val


# ------------------------------------------------------------------------------------------ tiers
def _cells_for(engine, tier: str, table: str, symbol: str, source: str, a: dt.datetime, b: dt.datetime) -> pd.DataFrame:
    if tier == "price_1m":
        r = fetch_price_1m(engine, symbol, source, a, b)
        return fh.cells_from_price_1m(r["t"], r["price"], r["buy"], r["sell"])
    if tier == "trades":
        r = fetch_trade_cells(engine, symbol, a, b)
        return fh.cells_from_trades(r["t"], r["price"], r["size"], r["side"])
    r = fetch_seconds(engine, table, symbol, a, b)
    return fh.cells_from_seconds(r["ts"], r["high"], r["low"], r["buy"], r["sell"])


def _segment_cells(engine, tier, table, symbol, source, a, b, complete, now_utc) -> pd.DataFrame:
    """Completed segments: cached whole. Live: cached up to a cut, re-read from cut - 2 min (DC14)."""
    if len(_CACHE) > CACHE_MAX:
        with _CACHE_LOCK:
            _CACHE.clear()
    if complete:
        key = ("cells", tier, symbol, source, a.isoformat(), b.isoformat())
        if key not in _CACHE:
            c = _cells_for(engine, tier, table, symbol, source, a, b)
            with _CACHE_LOCK:
                _CACHE[key] = c
        return _CACHE[key]
    key = ("live", tier, symbol, source, a.isoformat())
    hit = _CACHE.get(key)
    if hit is not None and hit[0] > a:
        cut, old = hit
        start = min(cut, b)
        c = old[old["t"] < flow_v2._epoch(start)]
        if start < b:
            c = pd.concat([c, _cells_for(engine, tier, table, symbol, source, start, b)], ignore_index=True)
    else:
        c = _cells_for(engine, tier, table, symbol, source, a, b)
    new_cut = min(b, flow_v2._floor_min(now_utc) - dt.timedelta(minutes=LIVE_RECHECK_MIN))
    if new_cut > a:
        with _CACHE_LOCK:
            _CACHE[key] = (new_cut, c[c["t"] < flow_v2._epoch(new_cut)].reset_index(drop=True))
    return c[c["t"] < flow_v2._epoch(b)].reset_index(drop=True)


def choose_tier(engine, *, source: str, symbol: str, a: dt.datetime, b: dt.datetime, minutes: pd.DataFrame) -> str:
    """DC17 order: es_flow_price_1m if it covers the segment, then trades (Ironbeam only), then 1-s."""
    if _cached("price_1m_exists", TABLE_RECHECK_S, lambda: _exists_price_1m(engine)):
        have = minutes.index[minutes["vol"] > 0] if minutes is not None and len(minutes) else []
        if len(have):
            r = fetch_price_1m(engine, symbol, source, a, b)
            if r["t"].nunique() >= PRICE_1M_COVERAGE * len(have):
                return "price_1m"
    if source == "ironbeam":
        oldest = _cached(f"trades_oldest:{symbol}", TRADES_OLDEST_TTL_S, lambda: _trades_oldest(engine, symbol))
        if oldest is not None and oldest <= a:
            return "trades"
    return "seconds"


def parse_absorb_params(args) -> fh.AbsorbParams:
    """⚙ K and M from the query string (`a_k`, `a_m`), clamped; absent / invalid = defaults (K 3, M 3)."""
    dflt = fh.AbsorbParams()

    def num(name, lo, hi, default, cast=float):
        try:
            v = cast(args.get(name))
        except (TypeError, ValueError):
            return default
        return default if not np.isfinite(v) else min(hi, max(lo, v))

    return fh.AbsorbParams(k=num("a_k", 0.5, 50.0, dflt.k), m=int(num("a_m", 1, 30, dflt.m, cast=int)))


def _num(x):
    return None if x is None else float(x)


def _marks_json(marks: list[dict]) -> list[dict]:
    keys = ("kind", "time", "price", "confirm_time", "aggressor", "opposite", "merged_vol", "median",
            "break_time", "held_min")
    return [{k: (_num(mk[k]) if isinstance(mk[k], (float, np.floating)) else mk[k]) for k in keys} for mk in marks]


# ------------------------------------------------------------------------------------------ payload
def build_cells_payload(engine, trade_date: str | None, *, session: str | None, until_pt: dt.datetime | None,
                        params: fh.AbsorbParams, session_date: dt.date, session_note: str | None,
                        now_utc: dt.datetime | None = None) -> tuple[dict, int]:
    shared = flow_v2._SharedConnection(engine)
    try:
        return _build(shared, trade_date, session=session, until_pt=until_pt, params=params,
                      session_date=session_date, session_note=session_note, now_utc=now_utc)
    finally:
        shared.close()


def _build(engine, trade_date, *, session, until_pt, params, session_date, session_note, now_utc=None):
    now_utc = now_utc or dt.datetime.now(flow_v2.UTC)
    session_mode = str(session or "RTH").upper().strip()
    if session_mode not in {"RTH", "FULL"}:
        session_mode = "RTH"
    d = session_date
    eth_a = dt.datetime.combine(d - dt.timedelta(days=1), dt.time(15, 0), tzinfo=flow_v2.PT_TZ)
    rth_a, rth_b = flow_v2._pt(d, 6, 30), flow_v2._pt(d, 13, 0)
    end = rth_b if until_pt is None else max(eth_a, min(rth_b, until_pt))
    complete = end.astimezone(flow_v2.UTC) <= now_utc - dt.timedelta(minutes=2)
    payload = {"mode": "cells", "trade_date": str(trade_date), "effective_trade_date": d.isoformat(),
               "session_note": session_note, "session": session_mode, "tick": fh.TICK,
               "absorb_params": {"k": params.k, "m": params.m}}
    if until_pt is not None:
        payload["until"] = until_pt.replace(tzinfo=None).isoformat()
    notes, segments = [], []
    bars_from = eth_a if session_mode == "FULL" else rth_a
    all_bars = flow_v2.fetch_bar_closes(engine, bars_from.astimezone(flow_v2.UTC), end.astimezone(flow_v2.UTC)) \
        if end > bars_from else {}

    plan = []
    if session_mode == "FULL" and end > eth_a:
        plan.append(("eth", eth_a, min(rth_a, end)))
    if end > rth_a:
        plan.append(("rth", rth_a, end))
    for name, a_pt, b_pt in plan:
        a, b = a_pt.astimezone(flow_v2.UTC), b_pt.astimezone(flow_v2.UTC)
        bar_closes = flow_v2._bars_in(all_bars, a, b)
        source, table, sym, m = "ironbeam", flow_v2.IRONBEAM_TABLE, None, None
        if name == "rth" and flow_v2.DATABENTO_FIRST <= d <= flow_v2.DATABENTO_LAST_SERVED:
            sym, _, m = flow_v2._segment_minutes(engine, flow_v2.DATABENTO_TABLE, a, b, bar_closes, complete)
            if sym is not None:
                source, table = "databento", flow_v2.DATABENTO_TABLE
        if source == "ironbeam":
            sym, _, m = flow_v2._segment_minutes(engine, flow_v2.IRONBEAM_TABLE, a, b, bar_closes, complete)
        if sym is None or m is None or m.empty:
            notes.append(f"No {name.upper()} flow rows for this session.")
            continue
        tier = choose_tier(engine, source=source, symbol=sym, a=a, b=b, minutes=m)
        cells = _segment_cells(engine, tier, table, sym, source, a, b, complete, now_utc)
        seg = {"name": name, "source": source, "symbol": sym, "tier": tier, "exact": tier != "seconds",
               "badge": BADGES[tier], "start": flow_v2._epoch(a), "end": flow_v2._epoch(b),
               "cells": {"t": cells["t"].astype(int).tolist(), "p": cells["p"].astype(int).tolist(),
                         "buy": cells["buy"].astype(float).tolist(), "sell": cells["sell"].astype(float).tolist()},
               "marks": None, "marks_note": None}
        if name == "rth":
            if tier == "seconds":
                seg["marks_note"] = "Absorbed: exact data only"       # A1.2
            else:
                seg["marks"] = _marks_json(fh.absorption_marks(cells, m[["h", "l"]], seg["start"], seg["end"], params))
        else:
            seg["marks_note"] = "Absorbed: RTH only"
        segments.append(seg)
    if d > flow_v2.DATABENTO_LAST_SERVED:
        notes.append("Ironbeam source: Databento flow is served only through 2026-06-05 (holdout protection).")
    payload.update(segments=segments, notes=notes)
    return payload, 200
