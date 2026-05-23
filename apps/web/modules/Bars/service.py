"""Bars service — fetch RTH 1-min OHLC from ironbeam_es_1m_bars (CR-016).

RTH session: first bar whose datetime (converted to PT) falls on trade_date
through 13:00 PT. This matches the convention used by _fetch_session_outcomes
in the Analogues module: filter by calendar date in PT, which captures the
~06:30–13:00 PT RTH window.

Returns a list of dicts suitable for lightweight-charts CandlestickData:
  { time: int (UTC epoch seconds), open: float, high: float,
    low: float, close: float }

Returns [] if no bars exist for the date — callers should empty-state
rather than 404.
"""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

_UTC = ZoneInfo("UTC")
_PT = ZoneInfo("America/Los_Angeles")

_SQL = """
    SELECT
        EXTRACT(EPOCH FROM datetime AT TIME ZONE 'UTC')::bigint AS time,
        open, high, low, close
    FROM ironbeam_es_1m_bars
    WHERE (datetime AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')::date = %s
      AND (datetime AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')::time
          BETWEEN '06:30:00' AND '13:00:00'
    ORDER BY datetime ASC
"""

_OPEN_SQL = """
    SELECT open
    FROM ironbeam_es_1m_bars
    WHERE (datetime AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')::date = %s
      AND (datetime AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')::time
          BETWEEN '06:30:00' AND '13:00:00'
    ORDER BY datetime ASC
    LIMIT 1
"""


def fetch_rth_bars(conn, ticker: str, trade_date: dt.date) -> list[dict]:
    """Return RTH 1-min bars for trade_date as lightweight-charts-compatible dicts."""
    with conn.cursor() as cur:
        cur.execute(_SQL, (trade_date,))
        rows = cur.fetchall()
    if not rows:
        return []
    return [
        {
            "time": int(r[0]),
            "open": float(r[1]),
            "high": float(r[2]),
            "low": float(r[3]),
            "close": float(r[4]),
        }
        for r in rows
        if all(v is not None for v in r)
    ]


def fetch_rth_open(conn, trade_date: dt.date) -> float | None:
    """Return the open price of the first RTH bar on trade_date, or None."""
    with conn.cursor() as cur:
        cur.execute(_OPEN_SQL, (trade_date,))
        row = cur.fetchone()
    if not row or row[0] is None:
        return None
    return float(row[0])
