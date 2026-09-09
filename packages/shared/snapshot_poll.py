"""Clock-independent wait for the day's 06:33 PT SPX monies snapshot (CR-AU
decision 1, generalised for CR-BD decision 2).

Render cron schedules are UTC. After the DST change (2026-11-01) the 13:35 /
13:40 UTC jobs fire at 05:35 / 05:40 PT — before the 06:33 PT open-straddle pin
exists — and every job that assumed the clock went dark. Instead of assuming,
each job polls `orats_monies_minute` for the snapshot it needs, up to a budget
(120 min by default), and exits 0 with a logged "no open snapshot" when none
appears (holiday, ingest outage).

`poll_until` takes an injectable clock and sleep so the timeout is testable;
`wait_for_open_snapshot` is the DB-backed probe the three crons share.
"""
from __future__ import annotations

import time as _time
from datetime import date, datetime, time
from typing import Callable, Optional
from zoneinfo import ZoneInfo

MAX_WAIT_MIN = 120.0
POLL_SECONDS = 60.0
OPEN_SNAPSHOT_FLOOR = time(6, 33)
_PT = ZoneInfo("America/Los_Angeles")

# trade_date is TEXT in orats_monies_minute — pass the ISO string
_SPX_OPEN_SQL = """
SELECT spot_price, snapshot_pt FROM orats_monies_minute
WHERE ticker = %s AND trade_date = %s AND snapshot_pt >= %s AND spot_price IS NOT NULL
ORDER BY snapshot_pt ASC, dte ASC LIMIT 1
"""


def now_pt() -> datetime:
    """Naive wall-clock time in America/Los_Angeles."""
    return datetime.now(_PT).replace(tzinfo=None)


def poll_until(probe: Callable[[], object], *, max_wait_min: float, poll_s: float = POLL_SECONDS,
               now_fn: Callable[[], datetime] = now_pt, sleep_fn: Callable[[float], None] = _time.sleep,
               label: str = "condition", log: Callable[[str], None] = print) -> tuple[object, float, int]:
    """Call `probe` until it returns a truthy value or `max_wait_min` minutes have
    elapsed on `now_fn`'s clock. Returns (value_or_None, waited_minutes, attempts).
    max_wait_min = 0 → exactly one probe; the last sleep is clipped to the budget."""
    start = now_fn()
    attempts = 0
    while True:
        attempts += 1
        value = probe()
        elapsed = (now_fn() - start).total_seconds() / 60.0
        if value:
            return value, round(elapsed, 2), attempts
        remaining_s = max_wait_min * 60.0 - elapsed * 60.0
        if remaining_s <= 0:
            return None, round(elapsed, 2), attempts
        log(f"  waiting for {label}: attempt {attempts}, {elapsed:.1f} min elapsed of {max_wait_min:g}; "
            f"next probe in {min(poll_s, remaining_s):.0f}s")
        sleep_fn(min(poll_s, remaining_s))


def fetch_open_snapshot(conn, ticker: str, trade_date: date):
    """The first orats_monies_minute snapshot ≥ 06:33 PT for trade_date: (spot_price, snapshot_pt) or None."""
    return conn.execute(_SPX_OPEN_SQL, (ticker, trade_date.isoformat(),
                                        datetime.combine(trade_date, OPEN_SNAPSHOT_FLOOR))).fetchone()


def poll_budget_for(trade_date: date, today: date, max_wait_min: float = MAX_WAIT_MIN) -> float:
    """A past date's snapshot either exists or never will → one probe; today's may still be arriving."""
    return max_wait_min if trade_date >= today else 0.0


def wait_for_open_snapshot(conn, ticker: str, trade_date: date, *, max_wait_min: Optional[float] = None,
                           poll_s: float = POLL_SECONDS, now_fn: Callable[[], datetime] = now_pt,
                           sleep_fn: Callable[[float], None] = _time.sleep,
                           log: Callable[[str], None] = print) -> tuple[Optional[tuple], float, int, float]:
    """Poll for trade_date's 06:33 PT snapshot. Budget defaults to `poll_budget_for`
    (120 min when trade_date is today or later on the PT clock, one probe for a
    past date). Returns (row_or_None, waited_minutes, attempts, budget_minutes)."""
    budget = poll_budget_for(trade_date, now_fn().date()) if max_wait_min is None else float(max_wait_min)
    row, waited, attempts = poll_until(lambda: fetch_open_snapshot(conn, ticker, trade_date), max_wait_min=budget,
                                       poll_s=poll_s, now_fn=now_fn, sleep_fn=sleep_fn,
                                       label=f"{trade_date} 06:33 PT {ticker} snapshot in orats_monies_minute", log=log)
    return row, waited, attempts, budget


def no_snapshot_line(ticker: str, trade_date: date, waited: float, attempts: int, budget: float) -> str:
    """The one log line every cron prints before exiting 0 without a snapshot."""
    return (f"no open snapshot: no orats_monies_minute row ≥ 06:33 PT for {ticker} {trade_date} after {waited:g} min / "
            f"{attempts} probe(s) (budget {budget:g} min) — holiday, DST drift or ingest outage. Nothing written.")
