"""One session definition for everything that reads ES RTH bars per day (CR-BI A1.1).

Session window: bars whose bar-open timestamp is 06:30:00–13:00:00 America/Los_Angeles,
inclusive (391 bars; DST handled by the zone, not by a fixed UTC offset). This is the window
the t0 outcome columns have always used (cr_b_backfill_outcomes). CR-G / CR-I used a fixed
13:30–20:00 UTC window — 05:30–11:59 PT in PST months — which this module retires.

Sessions: NYSE trading days from packages.shared.trading_calendar, not bar presence. ES prints
RTH-window bars on market holidays (Globex) and the table holds stray weekend bars; neither is
a session.

Public API
----------
RTH_BARS_SQL
fetch_es_daily_bars(conn, date_from, date_to) -> DataFrame[date -> open, high, low, close]
trading_sessions_only(daily) -> same frame, NYSE trading days only
session_ohlc_at(daily, trade_date, timeframes) -> {session_{open,high,low,close}_tN: float | None}
post_touch_positions(daily, trade_date, days_to_reach, drift_target, tolerance, timeframes) -> {N: -1|0|1|None}
"""
from __future__ import annotations

import datetime as dt
from typing import Optional

import pandas as pd

from packages.shared.trading_calendar import is_trading_day

SESSION_TZ = "America/Los_Angeles"
SESSION_START_PT = dt.time(6, 30)
SESSION_END_PT = dt.time(13, 0)      # inclusive, bar-open timestamp
TIMEFRAMES = (1, 5, 15)

# Aggregate 1-minute RTH bars to daily OHLC.
# open = first bar's open, high = max(high), low = min(low), close = last bar's close.
RTH_BARS_SQL = """
    WITH rth AS (
        SELECT
            (datetime AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')::date AS session_date,
            open, high, low, close,
            ROW_NUMBER() OVER (
                PARTITION BY
                    (datetime AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')::date
                ORDER BY datetime ASC
            ) AS rn_asc,
            ROW_NUMBER() OVER (
                PARTITION BY
                    (datetime AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')::date
                ORDER BY datetime DESC
            ) AS rn_desc
        FROM ironbeam_es_1m_bars
        WHERE
            (datetime AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')::time
                BETWEEN '06:30:00' AND '13:00:00'
          AND (datetime AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')::date
                BETWEEN %s AND %s
    )
    SELECT
        session_date,
        MAX(CASE WHEN rn_asc  = 1 THEN open  END) AS open,
        MAX(high)                                   AS high,
        MIN(low)                                    AS low,
        MAX(CASE WHEN rn_desc = 1 THEN close END)  AS close
    FROM rth
    GROUP BY session_date
    ORDER BY session_date
"""


def trading_sessions_only(daily: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose date is not an NYSE trading day (holiday Globex sessions, weekend strays)."""
    if daily.empty:
        return daily
    return daily[[is_trading_day(d) for d in daily.index]]


def fetch_es_daily_bars(conn, date_from: dt.date, date_to: dt.date,
                        trading_days_only: bool = True) -> pd.DataFrame:
    """ES RTH daily OHLC over [date_from, date_to]; index = date objects."""
    with conn.cursor() as cur:
        cur.execute(RTH_BARS_SQL, (date_from, date_to))
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close"])
    df = pd.DataFrame(rows, columns=["session_date", "open", "high", "low", "close"]).set_index("session_date")
    df.index = [d for d in df.index]        # keep as date objects
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return trading_sessions_only(df) if trading_days_only else df


def session_ohlc_at(daily: pd.DataFrame, trade_date: dt.date,
                    timeframes: tuple[int, ...] = TIMEFRAMES) -> dict[str, Optional[float]]:
    """Session OHLC at the Nth session after trade_date (CR-G columns). None beyond the frame
    or when trade_date itself is not a session in the frame (CR-G's rule)."""
    out: dict[str, Optional[float]] = {f"session_{k}_t{n}": None
                                       for n in timeframes for k in ("open", "high", "low", "close")}
    if daily.empty or trade_date not in daily.index:
        return out
    later = daily[daily.index > trade_date].sort_index()
    for n in timeframes:
        if len(later) >= n:
            b = later.iloc[n - 1]
            if not any(pd.isna(b[k]) for k in ("open", "high", "low", "close")):
                for k in ("open", "high", "low", "close"):
                    out[f"session_{k}_t{n}"] = float(b[k])
    return out


def post_touch_positions(daily: pd.DataFrame, trade_date: dt.date, days_to_reach: Optional[int],
                         drift_target: Optional[float], tolerance: Optional[float],
                         timeframes: tuple[int, ...] = TIMEFRAMES) -> dict[int, Optional[int]]:
    """position_tN_post_touch (CR-I columns): session close at days_to_reach + N vs target ± tolerance."""
    from packages.shared.probability import classify_post_touch_positions

    if days_to_reach is None or drift_target is None or not tolerance or tolerance <= 0:
        return {n: None for n in timeframes}
    forward = daily[daily.index >= trade_date].sort_index().dropna(subset=["high", "low", "close"])
    return classify_post_touch_positions(
        days_to_reach=int(days_to_reach), horizon_bars=forward, drift_target=float(drift_target),
        tolerance=float(tolerance), timeframes_sessions=timeframes)
