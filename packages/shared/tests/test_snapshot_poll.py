"""CR-BD decision 2 / G1 — shared snapshot poll: DST-drift case, 120-min timeout,
past-date single probe, budget selection."""
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[3])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from packages.shared.snapshot_poll import (
    MAX_WAIT_MIN, POLL_SECONDS, no_snapshot_line, poll_budget_for, poll_until, wait_for_open_snapshot,
)


class _Clock:
    def __init__(self, start):
        self.now = start; self.sleeps = []
    def now_fn(self):
        return self.now
    def sleep_fn(self, s):
        self.sleeps.append(s); self.now += timedelta(seconds=s)


class _Conn:
    """Fake psycopg connection: the snapshot 'appears' once the clock reaches `appears_at`."""
    def __init__(self, clock, appears_at=None, row=(7705.27, datetime(2026, 9, 8, 6, 33))):
        self.clock, self.appears_at, self.row, self.queries = clock, appears_at, row, []
    def execute(self, sql, params):
        self.queries.append(params)
        conn = self
        class _Cur:
            def fetchone(_):
                if conn.appears_at is not None and conn.clock.now >= conn.appears_at:
                    return conn.row
                return None
        return _Cur()


def test_dst_drift_case_cron_fires_an_hour_early_and_waits_for_the_pin():
    # after 2026-11-01, 13:35 UTC is 05:35 PT: the job must wait for the 06:33 snapshot instead of failing
    clk = _Clock(datetime(2026, 11, 2, 5, 35))
    conn = _Conn(clk, appears_at=datetime(2026, 11, 2, 6, 34), row=(6900.0, datetime(2026, 11, 2, 6, 33)))
    row, waited, attempts, budget = wait_for_open_snapshot(conn, "SPX", date(2026, 11, 2), now_fn=clk.now_fn,
                                                           sleep_fn=clk.sleep_fn, log=lambda m: None)
    assert row == (6900.0, datetime(2026, 11, 2, 6, 33))
    assert budget == MAX_WAIT_MIN and waited == 59.0 and attempts == 60
    assert all(s == POLL_SECONDS for s in clk.sleeps)
    assert conn.queries[0] == ("SPX", "2026-11-02", datetime(2026, 11, 2, 6, 33))      # trade_date passed as ISO text


def test_holiday_times_out_after_120_minutes_and_exits_with_the_logged_line():
    clk = _Clock(datetime(2026, 9, 7, 6, 50))
    conn = _Conn(clk, appears_at=None)
    row, waited, attempts, budget = wait_for_open_snapshot(conn, "SPX", date(2026, 9, 7), now_fn=clk.now_fn,
                                                           sleep_fn=clk.sleep_fn, log=lambda m: None)
    assert row is None and waited == 120.0 and attempts == 121 and budget == 120.0
    line = no_snapshot_line("SPX", date(2026, 9, 7), waited, attempts, budget)
    assert line.startswith("no open snapshot:") and "121 probe(s)" in line and "Nothing written" in line


def test_past_date_is_a_single_probe():
    clk = _Clock(datetime(2026, 9, 8, 21, 5))
    conn = _Conn(clk, appears_at=None)
    row, waited, attempts, budget = wait_for_open_snapshot(conn, "SPX", date(2026, 9, 4), now_fn=clk.now_fn,
                                                           sleep_fn=clk.sleep_fn, log=lambda m: None)
    assert row is None and attempts == 1 and budget == 0.0 and clk.sleeps == []


def test_budget_selection_and_override():
    assert poll_budget_for(date(2026, 9, 8), date(2026, 9, 8)) == 120.0
    assert poll_budget_for(date(2026, 9, 4), date(2026, 9, 8)) == 0.0
    assert poll_budget_for(date(2026, 9, 9), date(2026, 9, 8)) == 120.0
    clk = _Clock(datetime(2026, 9, 8, 6, 50))
    conn = _Conn(clk, appears_at=None)
    _, _, attempts, budget = wait_for_open_snapshot(conn, "SPX", date(2026, 9, 8), max_wait_min=0, now_fn=clk.now_fn,
                                                    sleep_fn=clk.sleep_fn, log=lambda m: None)
    assert attempts == 1 and budget == 0.0


def test_poll_until_last_sleep_is_clipped_to_the_budget():
    clk = _Clock(datetime(2026, 9, 4, 6, 0))
    v, waited, attempts = poll_until(lambda: None, max_wait_min=2.5, poll_s=60, now_fn=clk.now_fn, sleep_fn=clk.sleep_fn, log=lambda m: None)
    assert v is None and clk.sleeps == [60, 60, 30] and waited == 2.5 and attempts == 4
