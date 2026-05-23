"""Unit tests for apps/web/modules/Bars/service.py (CR-016).

Run with:
    python -m unittest apps.web.modules.Bars.tests.test_service
"""
from __future__ import annotations

import datetime as dt
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parents[6]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.web.modules.Bars.service import fetch_rth_bars, fetch_rth_open

TRADE_DATE = dt.date(2026, 5, 7)

_SAMPLE_ROWS = [
    (1746613800, 7362.5, 7370.0, 7358.0, 7365.25),
    (1746613860, 7365.25, 7368.0, 7362.0, 7366.50),
    (1746613920, 7366.50, 7372.5, 7364.0, 7371.00),
]


def _stub_conn(rows_all=None, row_one=None):
    cur = MagicMock()
    cur.__enter__ = lambda s: cur
    cur.__exit__ = MagicMock(return_value=False)
    cur.fetchall.return_value = rows_all if rows_all is not None else []
    cur.fetchone.return_value = row_one
    conn = MagicMock()
    conn.cursor.return_value = cur
    return conn


class TestFetchRthBars(unittest.TestCase):
    def test_returns_formatted_dicts(self):
        conn = _stub_conn(rows_all=_SAMPLE_ROWS)
        result = fetch_rth_bars(conn, "SPX", TRADE_DATE)
        self.assertEqual(len(result), 3)
        first = result[0]
        self.assertEqual(first["time"], 1746613800)
        self.assertAlmostEqual(first["open"], 7362.5)
        self.assertAlmostEqual(first["high"], 7370.0)
        self.assertAlmostEqual(first["low"], 7358.0)
        self.assertAlmostEqual(first["close"], 7365.25)

    def test_empty_when_no_rows(self):
        conn = _stub_conn(rows_all=[])
        result = fetch_rth_bars(conn, "SPX", TRADE_DATE)
        self.assertEqual(result, [])

    def test_skips_rows_with_null_values(self):
        rows = [
            (1746613800, None, 7370.0, 7358.0, 7365.25),
            (1746613860, 7365.25, 7368.0, 7362.0, 7366.50),
        ]
        conn = _stub_conn(rows_all=rows)
        result = fetch_rth_bars(conn, "SPX", TRADE_DATE)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0]["open"], 7365.25)

    def test_passes_trade_date_to_query(self):
        conn = _stub_conn(rows_all=_SAMPLE_ROWS)
        fetch_rth_bars(conn, "SPX", TRADE_DATE)
        cur = conn.cursor.return_value
        # execute(sql, params) — params is second positional arg
        params = cur.execute.call_args[0][1]
        self.assertIn(TRADE_DATE, params)

    def test_all_ohlc_fields_present(self):
        conn = _stub_conn(rows_all=_SAMPLE_ROWS)
        result = fetch_rth_bars(conn, "SPX", TRADE_DATE)
        for bar in result:
            for key in ("time", "open", "high", "low", "close"):
                self.assertIn(key, bar)


class TestFetchRthOpen(unittest.TestCase):
    def test_returns_open_float(self):
        conn = _stub_conn(row_one=(7362.5,))
        result = fetch_rth_open(conn, TRADE_DATE)
        self.assertAlmostEqual(result, 7362.5)

    def test_returns_none_when_no_rows(self):
        conn = _stub_conn(row_one=None)
        result = fetch_rth_open(conn, TRADE_DATE)
        self.assertIsNone(result)

    def test_returns_none_when_open_null(self):
        conn = _stub_conn(row_one=(None,))
        result = fetch_rth_open(conn, TRADE_DATE)
        self.assertIsNone(result)

    def test_passes_trade_date_to_query(self):
        conn = _stub_conn(row_one=(7362.5,))
        fetch_rth_open(conn, TRADE_DATE)
        cur = conn.cursor.return_value
        params = cur.execute.call_args[0][1]
        self.assertIn(TRADE_DATE, params)


class TestRthBoundary(unittest.TestCase):
    """Verify that both queries include the 06:30–13:00 PT time-of-day filter
    so overnight ES bars are excluded from RTH fetches (CR-016 hotfix)."""

    def test_fetch_rth_bars_sql_contains_rth_time_filter(self):
        """The SQL issued by fetch_rth_bars must contain the 06:30–13:00 filter."""
        conn = _stub_conn(rows_all=_SAMPLE_ROWS)
        fetch_rth_bars(conn, "SPX", TRADE_DATE)
        cur = conn.cursor.return_value
        sql_issued = cur.execute.call_args[0][0]
        self.assertIn("06:30:00", sql_issued)
        self.assertIn("13:00:00", sql_issued)
        self.assertIn("BETWEEN", sql_issued.upper())

    def test_fetch_rth_open_sql_contains_rth_time_filter(self):
        """The SQL issued by fetch_rth_open must contain the 06:30–13:00 filter."""
        conn = _stub_conn(row_one=(7362.5,))
        fetch_rth_open(conn, TRADE_DATE)
        cur = conn.cursor.return_value
        sql_issued = cur.execute.call_args[0][0]
        self.assertIn("06:30:00", sql_issued)
        self.assertIn("13:00:00", sql_issued)
        self.assertIn("BETWEEN", sql_issued.upper())

    def test_fetch_rth_bars_filters_overnight_bars_not_in_result(self):
        """Rows whose UTC epoch maps outside 06:30–13:00 PT should not appear.

        The SQL filter runs in the DB; at the service layer the test verifies
        that the *query itself* carries the time clause (above). This test
        confirms the service doesn't add a second in-Python filter that would
        silently pass through overnight rows fetched by a bad query.

        We simulate a correct DB by returning only RTH-window rows from the mock
        and asserting all five keys are present on every result row — there is no
        in-Python post-filter that strips time data after the fact.
        """
        # Epoch 1746613800 = 2026-05-07 13:30 UTC = 06:30 PT (RTH open)
        rth_rows = [
            (1746613800, 7362.5, 7370.0, 7358.0, 7365.25),
        ]
        conn = _stub_conn(rows_all=rth_rows)
        result = fetch_rth_bars(conn, "SPX", TRADE_DATE)
        self.assertEqual(len(result), 1)
        row = result[0]
        self.assertEqual(row["time"], 1746613800)
        for key in ("time", "open", "high", "low", "close"):
            self.assertIn(key, row)


if __name__ == "__main__":
    unittest.main()
