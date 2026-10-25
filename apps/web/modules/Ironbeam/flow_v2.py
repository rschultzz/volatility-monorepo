"""CR-CD — `/api/ironbeam/flow?mode=v2`: the Flow v2 pane payload (L2, L4, L5, L7, L9; A1, A2).

Per session, two segments, each from one source:
  - ETH  [prior 15:00, 06:30) PT: Ironbeam (`ironbeam_es_flow_1s`), CVD anchored at 15:00; delta, CVD and
    delta share only (A1 D10: absorption and markers are RTH-only).
  - RTH  [06:30, 13:00) PT: Databento (`es_flow_1s_databento`) for sessions 2024-04-03 .. 2026-06-05 that it
    covers (A1.1, holdout protection), otherwise Ironbeam; CVD anchored at 06:30; absorption against the
    prior-20-session baseline from the same source (A1 D4); markers per A2 (MarkerParams from the request).
In two-contract sessions the contract whose minute closes match the chart's bars is used (A1.4); the
baseline sessions use their highest-volume contract (bars are not fetched for them).

The DB is read only through `fetch_flow_minutes` and `fetch_bar_closes` (1-minute rows on `ts_utc`
ranges; the baseline in multi-range queries). Since CR-CD A4 `fetch_flow_minutes` reads the stored
1-minute tables for whole minutes before their cutoff and aggregates the rest on the fly, so tests can record and replay them. Completed
sessions' minutes are cached in-process (A1.6 / D9); L1 allows no table writes.
"""
from __future__ import annotations

import datetime as dt
import threading
import time
from contextlib import contextmanager
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sqlalchemy import text

from packages.shared import flow_1m
from packages.shared import flow_features as ff
from packages.shared.trading_calendar import is_trading_day

PT_TZ = ZoneInfo("America/Los_Angeles")
UTC = dt.timezone.utc

DATABENTO_TABLE = "es_flow_1s_databento"
IRONBEAM_TABLE = "ironbeam_es_flow_1s"
DATABENTO_FIRST = dt.date(2024, 4, 3)
DATABENTO_LAST_SERVED = dt.date(2026, 6, 5)        # A1.1: holdout begins after this session
IRONBEAM_FIRST = dt.date(2025, 12, 26)
BAR_MATCH_MIN = 0.9                                # A1.4: share of equal minute closes
BASELINE_LOOKBACK_DAYS = 45                        # trading days searched for 20 prior sessions
WARMUP_MIN = 5                                     # RTH look-back for the first 5-min absorption window
BASELINE_CHUNK = 5                                 # sessions per baseline query (22 ranges in one OR: 17 s on Ironbeam)
CACHE_MAX = 4000                                   # entries; the cache is cleared when it grows past this
FRESH_1S = dt.timedelta(minutes=5)                 # A4.4: the last 5 minutes always come from the 1-second table
WATERMARK_TTL_S = 30.0                             # seconds a 1-minute table's last stored minute is reused
TABLE_RECHECK_S = 300.0                            # seconds before re-checking a missing 1-minute table

_CACHE: dict[tuple, object] = {}
_CACHE_LOCK = threading.Lock()


# ------------------------------------------------------------------------------------------ DB reads
_FLOW_SQL_1S = """
        SELECT symbol, (floor(extract(epoch FROM ts_utc) / 60) * 60)::bigint AS t,
               (array_agg(open ORDER BY ts_utc))[1] AS o, max(high) AS h, min(low) AS l,
               (array_agg(close ORDER BY ts_utc DESC))[1] AS c,
               sum(buy_vol)::float8 AS buy, sum(sell_vol)::float8 AS sell, sum(volume)::float8 AS vol,
               count(*) AS n
        FROM {table}
        WHERE {where}
        GROUP BY 1, 2"""
_FLOW_COLS = ["symbol", "t", "o", "h", "l", "c", "buy", "sell", "vol", "n"]
_TABLE_STATE: dict[str, tuple[float, bool]] = {}
_WATERMARK: dict[str, tuple[float, dt.datetime | None]] = {}


def _where(col: str, ranges) -> tuple[str, dict]:
    where = " OR ".join(f"({col} >= :a{k} AND {col} < :b{k})" for k in range(len(ranges)))
    params = {}
    for k, (a, b) in enumerate(ranges):
        params[f"a{k}"], params[f"b{k}"] = a, b
    return where, params


def _floor_min(x: dt.datetime) -> dt.datetime:
    return x.replace(second=0, microsecond=0)


def _ceil_min(x: dt.datetime) -> dt.datetime:
    f = _floor_min(x)
    return f if f == x else f + dt.timedelta(minutes=1)


def split_ranges(ranges, cut: dt.datetime | None):
    """A4.4: whole minutes before `cut` -> the 1-minute table; ragged partial-minute edges and
    anything at or after `cut` -> the 1-second table. Returns (ranges_1m, ranges_1s)."""
    r1m, r1s = [], []
    for a, b in ranges:
        a1 = _ceil_min(a)
        b1 = _floor_min(b) if cut is None else min(_floor_min(b), cut)
        if cut is None or a1 >= b1:
            r1s.append((a, b))
            continue
        r1m.append((a1, b1))
        if a < a1:
            r1s.append((a, a1))
        if b1 < b:
            r1s.append((b1, b))
    return r1m, r1s


def _watermark_1m(engine, table_1m: str) -> dt.datetime | None:
    """The 1-minute table's last stored minute (None if the table is absent or empty); cached briefly.

    Both 1-minute tables are checked together (one existence query, one max query), so a cold
    payload pays these round trips once, not per table.
    """
    mono = time.monotonic()
    tables = sorted(flow_1m.TABLE_1M.values())
    state = _TABLE_STATE.get(table_1m)
    if state is None or (not state[1] and mono - state[0] > TABLE_RECHECK_S):
        with engine.connect() as conn:
            row = conn.execute(text("SELECT " + ", ".join(
                f"to_regclass('{t}') IS NOT NULL" for t in tables))).fetchone()
        for t, ok in zip(tables, row):
            _TABLE_STATE[t] = (mono, bool(ok))
        state = _TABLE_STATE[table_1m]
    if not state[1]:
        return None
    wm = _WATERMARK.get(table_1m)
    if wm is None or mono - wm[0] > WATERMARK_TTL_S:
        present = [t for t in tables if _TABLE_STATE.get(t, (0, False))[1]]
        with engine.connect() as conn:
            row = conn.execute(text("SELECT " + ", ".join(
                f"(SELECT max(minute_utc) FROM {t})" for t in present))).fetchone()
        for t, last in zip(present, row):
            _WATERMARK[t] = (mono, last)
        wm = _WATERMARK[table_1m]
    return wm[1]


def _cutoff_1m(engine, table_1m: str, now_utc: dt.datetime) -> dt.datetime | None:
    """min(last stored minute + 1 min, now floored to the minute - 5 min); None if absent/empty."""
    wm = _watermark_1m(engine, table_1m)
    if wm is None:
        return None
    return min(wm.astimezone(UTC) + dt.timedelta(minutes=1), _floor_min(now_utc) - FRESH_1S)


class _SharedConnection:
    """Engine stand-in for one payload build: every `connect()` yields the same connection, opened on
    first use (so a build costs one pool checkout, one pre-ping and one rollback, not one per query)."""

    def __init__(self, engine):
        self._engine, self._conn = engine, None

    @contextmanager
    def connect(self):
        if self._conn is None:
            self._conn = self._engine.connect()
        yield self._conn

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None


def _bars_in(bars: dict[int, float], a: dt.datetime, b: dt.datetime) -> dict[int, float]:
    lo, hi = _epoch(a), _epoch(b)
    return {t: c for t, c in bars.items() if lo <= t < hi}


def _read(engine, sql: str, params: dict) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)


def fetch_flow_minutes(engine, table: str, ranges: list[tuple[dt.datetime, dt.datetime]],
                       now_utc: dt.datetime | None = None) -> pd.DataFrame:
    """Flow-built 1-minute bars per contract over the union of [start, end) ranges.

    Columns: symbol, t (bar-start epoch s), o, h, l, c, buy, sell, vol, n (1-second rows in the minute).
    CR-CD A4.4: whole minutes before the cutoff are read from the stored 1-minute table (one query),
    the rest is aggregated on the fly from the 1-second table (one query); same rows either way
    (identity-checked by scripts/cr_cd_a4_identity.py).
    """
    table_1m = flow_1m.TABLE_1M.get(table)
    cut = _cutoff_1m(engine, table_1m, now_utc or dt.datetime.now(UTC)) if table_1m else None
    r1m, r1s = split_ranges(ranges, cut)
    frames = []
    if r1m:
        where, params = _where("minute_utc", r1m)
        frames.append(_read(engine, flow_1m.v2_select(table_1m, where), params))
    if r1s:
        where, params = _where("ts_utc", r1s)
        frames.append(_read(engine, _FLOW_SQL_1S.format(table=table, where=where), params))
    if not frames:
        return pd.DataFrame({c: pd.Series(dtype=float) for c in _FLOW_COLS}).astype(
            {"symbol": str, "t": np.int64, "n": int})
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    df = df[_FLOW_COLS].sort_values(["symbol", "t"], kind="mergesort").reset_index(drop=True)
    cols = list(ff.COLUMNS)
    df[cols] = df[cols].astype(float)
    df["t"] = df["t"].astype(np.int64)
    df["n"] = df["n"].astype(int)
    df["symbol"] = df["symbol"].astype(str)
    return df


def fetch_bar_closes(engine, start_utc: dt.datetime, end_utc: dt.datetime) -> dict[int, float]:
    """The chart's minute closes (`ironbeam_es_1m_bars`, naive-UTC bar start)."""
    q = text("""SELECT extract(epoch FROM datetime AT TIME ZONE 'UTC')::bigint AS t, close
                FROM ironbeam_es_1m_bars WHERE datetime >= :a AND datetime < :b""")
    with engine.connect() as conn:
        rows = conn.execute(q, {"a": start_utc.replace(tzinfo=None), "b": end_utc.replace(tzinfo=None)}).fetchall()
    return {int(r[0]): float(r[1]) for r in rows}


# ------------------------------------------------------------------------------------------ helpers
def _pt(d: dt.date, hh: int, mm: int = 0) -> dt.datetime:
    return dt.datetime.combine(d, dt.time(hh, mm), tzinfo=PT_TZ)


def _epoch(x: dt.datetime) -> int:
    return int(x.timestamp())


def rth_source(session: dt.date, databento_volumes: dict | None) -> str:
    """A1.1: Databento for covered sessions 2024-04-03 .. 2026-06-05, Ironbeam otherwise."""
    if DATABENTO_FIRST <= session <= DATABENTO_LAST_SERVED and databento_volumes:
        return "databento"
    return "ironbeam"


def choose_symbol(volumes: dict[str, float], minute_closes: dict[str, pd.Series], bar_closes: dict[int, float]):
    """A1.4: the contract whose minute closes equal the chart's bars on >= 90 % of shared minutes;
    else the highest-volume contract. Returns (symbol, how, match share or None)."""
    if not volumes:
        return None, "none", None
    by_vol = max(volumes, key=volumes.get)
    if len(volumes) == 1 or not bar_closes:
        return by_vol, "only" if len(volumes) == 1 else "volume", None
    best, best_share = None, -1.0
    bars = pd.Series(bar_closes)
    for sym, closes in minute_closes.items():
        common = closes.index.intersection(bars.index)
        if len(common) == 0:
            continue
        share = float((np.abs(closes.loc[common].to_numpy() - bars.loc[common].to_numpy()) < 0.01).mean())
        if share > best_share:
            best, best_share = sym, share
    if best is not None and best_share >= BAR_MATCH_MIN:
        return best, "bars", best_share
    return by_vol, "volume", best_share if best is not None else None


def _pick(rows: pd.DataFrame, bar_closes: dict[int, float] | None):
    """(symbol, how, minutes indexed by t) from one window's rows; A1.4 when bar closes are given."""
    if rows.empty:
        return None, "none", None
    vols = rows.groupby("symbol")["vol"].sum().to_dict()
    per = {sym: g.set_index("t") for sym, g in rows.groupby("symbol")}
    sym, how, _ = choose_symbol(vols, {k: v["c"] for k, v in per.items()}, bar_closes or {})
    return sym, how, per[sym].drop(columns="symbol")


def _segment_minutes(engine, table, start_utc, end_utc, bar_closes, cacheable):
    """(symbol, how, minutes) for one window, contract chosen by A1.4; cached when the window is complete."""
    key = ("seg", table, start_utc.isoformat(), end_utc.isoformat())
    if cacheable and key in _CACHE:
        return _CACHE[key]
    out = _pick(fetch_flow_minutes(engine, table, [(start_utc, end_utc)]), bar_closes)
    if cacheable:
        with _CACHE_LOCK:
            _CACHE[key] = out
    return out


def _prior_sessions(session: dt.date, source: str):
    d, out = session, []
    for _ in range(BASELINE_LOOKBACK_DAYS * 2):
        d -= dt.timedelta(days=1)
        if source == "databento" and d < DATABENTO_FIRST:
            break
        if source == "ironbeam" and d < IRONBEAM_FIRST:
            break
        if is_trading_day(d):
            out.append(d)
        if len(out) >= BASELINE_LOOKBACK_DAYS:
            break
    return out


def _baseline(engine, session: dt.date, source: str) -> dict | None:
    """Prior-20-session absorption baseline from the same source (A1 D4).

    Each prior session's RTH minutes (its highest-volume contract) are fetched in one multi-range
    query for all sessions not yet cached, then cached per session for the life of the process.
    """
    key = ("base", source, session.isoformat())
    if key in _CACHE:
        return _CACHE[key]
    if len(_CACHE) > CACHE_MAX:
        with _CACHE_LOCK:
            _CACHE.clear()
    table = DATABENTO_TABLE if source == "databento" else IRONBEAM_TABLE
    candidates = _prior_sessions(session, source)
    frames, used = [], []
    pos = 0
    while len(frames) < ff.BASE_N and pos < len(candidates):
        batch = candidates[pos:pos + (ff.BASE_N - len(frames)) + 2]
        pos += len(batch)
        windows = {d: (_pt(d, 6, 30).astimezone(UTC), _pt(d, 13, 0).astimezone(UTC)) for d in batch}
        todo = [d for d in batch if ("rthmin", table, d.isoformat()) not in _CACHE]
        # A4: from the stored 1-minute table the whole batch is one cheap query; the 5-session
        # chunks are only for the on-the-fly 1-second fallback (22 ranges in one OR: 17 s there).
        wm = _watermark_1m(engine, flow_1m.TABLE_1M[table])
        step = (len(todo) if todo and wm is not None and wm + dt.timedelta(minutes=1) >= windows[todo[0]][1]
                else BASELINE_CHUNK)
        for k in range(0, len(todo), max(step, 1)):
            chunk = todo[k:k + max(step, 1)]
            rows = fetch_flow_minutes(engine, table, [windows[d] for d in chunk])
            local = rows["t"].map(lambda t: dt.datetime.fromtimestamp(int(t), PT_TZ).date())
            with _CACHE_LOCK:
                for d in chunk:
                    _, _, m = _pick(rows[local == d], None)
                    _CACHE[("rthmin", table, d.isoformat())] = m
        for d in batch:
            m = _CACHE[("rthmin", table, d.isoformat())]
            if m is None or m.empty or len(frames) >= ff.BASE_N:
                continue
            a, b = windows[d]
            frames.append(ff.complete_minutes(m[list(ff.COLUMNS)], _epoch(a), _epoch(b)))
            used.append(d)
    out = None
    if len(frames) >= ff.BASE_MIN:
        out = ff.absorption_baseline(frames)
        out.update(source=source, sessions=[used[-1].isoformat(), used[0].isoformat()])
    with _CACHE_LOCK:
        _CACHE[key] = out
    return out


def _num(x):
    if x is None:
        return None
    x = float(x)
    return None if not np.isfinite(x) else x


def _bars(series: pd.DataFrame, seg: str, counts: pd.Series) -> list[dict]:
    out = []
    for t, r in series.iterrows():
        out.append({"time": int(t), "seg": seg, "buy": _num(r["buy"]), "sell": _num(r["sell"]),
                    "vol": _num(r["vol"]), "delta": _num(r["delta"]), "cvd": _num(r["cvd"]),
                    "share": _num(r["share"]), "imp_rel": _num(r["imp_rel"]), "absorb": bool(r["absorb"]),
                    "n": int(counts.get(t, 0))})
    return out


def _gaps(counts: pd.Series, index) -> list[dict]:
    """Runs of >= 2 consecutive minutes with no 1-second rows (feed holes)."""
    out, run = [], []
    for t in index:
        if counts.get(t, 0) == 0:
            run.append(int(t))
            continue
        if len(run) >= 2:
            out.append({"start": run[0], "end": run[-1] + 60, "minutes": len(run)})
        run = []
    if len(run) >= 2:
        out.append({"start": run[0], "end": run[-1] + 60, "minutes": len(run)})
    return out


def parse_marker_params(args) -> ff.MarkerParams:
    """A2.1: marker settings from the query string, clamped; absent/invalid = defaults."""
    dflt = ff.MarkerParams()

    def num(name, lo, hi, default, cast=float):
        try:
            v = cast(args.get(name))
        except (TypeError, ValueError):
            return default
        return default if not np.isfinite(v) else min(hi, max(lo, v))

    return ff.MarkerParams(swing=num("m_swing", 1.0, 50.0, dflt.swing),
                           min_prev_push=num("m_push", 0.0, 100000.0, dflt.min_prev_push),
                           min_div_cvd=num("m_div", 0.0, 100000.0, dflt.min_div_cvd),
                           fade_min_share=num("m_fade", 0.0, 1.0, dflt.fade_min_share),
                           gap_min=int(num("m_gap", 1, 390, dflt.gap_min, cast=int)))


# ------------------------------------------------------------------------------------------ payload
def build_flow_v2_payload(engine, trade_date: str | None, **kw) -> tuple[dict, int]:
    """The v2 payload for one session, built on a single DB connection (A4 / L9)."""
    shared = _SharedConnection(engine)
    try:
        return _build_flow_v2_payload(shared, trade_date, **kw)
    finally:
        shared.close()


def _build_flow_v2_payload(engine, trade_date: str | None, *, session: str | None, until_pt: dt.datetime | None,
                          params: ff.MarkerParams, session_date: dt.date, session_note: str | None,
                          now_utc: dt.datetime | None = None) -> tuple[dict, int]:
    now_utc = now_utc or dt.datetime.now(UTC)
    session_mode = str(session or "RTH").upper().strip()
    if session_mode not in {"RTH", "FULL"}:
        session_mode = "RTH"
    d = session_date
    eth_a = dt.datetime.combine(d - dt.timedelta(days=1), dt.time(15, 0), tzinfo=PT_TZ)
    rth_a, rth_b = _pt(d, 6, 30), _pt(d, 13, 0)
    end = rth_b if until_pt is None else max(eth_a, min(rth_b, until_pt))
    notes = []
    payload = {"mode": "v2", "trade_date": str(trade_date), "effective_trade_date": d.isoformat(),
               "session_note": session_note, "session": session_mode,
               "marker_params": {"swing": params.swing, "min_prev_push": params.min_prev_push,
                                 "min_div_cvd": params.min_div_cvd, "fade_min_share": params.fade_min_share,
                                 "gap_min": params.gap_min}}
    if until_pt is not None:
        payload["until"] = until_pt.replace(tzinfo=None).isoformat()
    segments, bars, markers, gaps = [], [], [], []
    complete = end.astimezone(UTC) <= now_utc - dt.timedelta(minutes=2) and until_pt is None
    # the chart's bar closes for the whole loaded window, once (contract choice, A1.4)
    bars_from = eth_a if session_mode == "FULL" else rth_a - dt.timedelta(minutes=WARMUP_MIN)
    all_bars = fetch_bar_closes(engine, bars_from.astimezone(UTC), end.astimezone(UTC)) if end > bars_from else {}

    # ETH (FULL only): Ironbeam, CVD anchored at 15:00, no absorption or markers.
    if session_mode == "FULL" and end > eth_a:
        a, b = eth_a.astimezone(UTC), min(rth_a, end).astimezone(UTC)
        sym, how, m = _segment_minutes(engine, IRONBEAM_TABLE, a, b, _bars_in(all_bars, a, b), complete)
        if sym is not None and m is not None and not m.empty:
            mm = ff.complete_minutes(m[list(ff.COLUMNS)], _epoch(a), _epoch(b))
            s = ff.pane_series(mm, _epoch(a), None, absorption=False)
            bars += _bars(s, "eth", m["n"])
            gaps += _gaps(m["n"], mm.index)
            segments.append({"name": "eth", "source": "ironbeam", "symbol": sym, "symbol_choice": how,
                             "anchor": eth_a.replace(tzinfo=None).isoformat(), "start": _epoch(a), "end": _epoch(b)})
        elif d < IRONBEAM_FIRST:
            notes.append("No ETH flow before 2025-12-26 (Ironbeam starts then; Databento covers 06:15-13:00 PT only).")

    # RTH
    if end > rth_a:
        a, b = rth_a.astimezone(UTC), end.astimezone(UTC)
        warm = (rth_a - dt.timedelta(minutes=WARMUP_MIN)).astimezone(UTC)
        bar_closes = _bars_in(all_bars, warm, b)
        sym, how, m = (None, "none", None)
        source = "ironbeam"
        if DATABENTO_FIRST <= d <= DATABENTO_LAST_SERVED:
            sym, how, m = _segment_minutes(engine, DATABENTO_TABLE, warm, b, bar_closes, complete)
            source = rth_source(d, {sym: 1.0} if sym else {})
        elif d > DATABENTO_LAST_SERVED:
            notes.append("Ironbeam source: Databento flow is served only through 2026-06-05 (holdout protection).")
        if source == "ironbeam":
            sym, how, m = _segment_minutes(engine, IRONBEAM_TABLE, warm, b, bar_closes, complete)
        if sym is not None and m is not None and not m.empty:
            mm = ff.complete_minutes(m[list(ff.COLUMNS)], _epoch(warm), _epoch(b))
            base = _baseline(engine, d, source)
            if base is None:
                notes.append("Absorption off: fewer than 5 prior sessions for the baseline.")
            s = ff.pane_series(mm, _epoch(a), base, absorption=base is not None)
            bars += _bars(s, "rth", m["n"])
            gaps += _gaps(m["n"], mm.index[mm.index >= _epoch(a)])
            markers = ff.pane_markers(mm, s, _epoch(a), params)
            segments.append({"name": "rth", "source": source, "symbol": sym, "symbol_choice": how,
                             "anchor": rth_a.replace(tzinfo=None).isoformat(), "start": _epoch(a), "end": _epoch(b)})
            payload["baseline"] = base
        else:
            notes.append("No RTH flow rows for this session.")

    rth_seg = next((s for s in segments if s["name"] == "rth"), None)
    payload.update(source=(rth_seg or (segments[0] if segments else {})).get("source"),
                   symbol=(rth_seg or (segments[0] if segments else {})).get("symbol"),
                   segments=segments, bars=bars, markers=markers, gaps=gaps, notes=notes,
                   point_count=len(bars))
    payload.setdefault("baseline", None)
    return payload, 200
