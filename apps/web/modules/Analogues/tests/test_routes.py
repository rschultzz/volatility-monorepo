"""Unit tests for apps/web/modules/Analogues/routes.py (CR-014).

Focused on _load_candidates — the function that selects candidate days
for similarity ranking. CR-013 originally gated this on bt_signals
labels; CR-014 removed that gate so every day with a stored feature
vector at the requested feature_version is a candidate.

Uses a stub connection/cursor to capture the executed SQL — avoids a
real DB while still exercising the routes-layer behavior end-to-end.

Run with:
    python -m unittest apps.web.modules.Analogues.tests.test_routes
"""
from __future__ import annotations

import datetime as dt
import unittest
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.web.modules.Analogues.routes import (
    _fetch_session_outcomes,
    _load_candidates,
)


class _StubCursor:
    def __init__(self, rows):
        self._rows = rows
        self.last_sql = None
        self.last_params = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def execute(self, sql, params):
        self.last_sql = sql
        self.last_params = params

    def fetchall(self):
        return self._rows


class _StubConn:
    def __init__(self, rows):
        self._cursor = _StubCursor(rows)

    def cursor(self):
        return self._cursor


class TestLoadCandidates(unittest.TestCase):

    def test_returns_all_feature_rows_regardless_of_labels(self):
        """CR-014: every day with a feature vector is a candidate. The
        function does not see, and should not depend on, label state."""
        rows = [
            (dt.date(2026, 5, 6), {"implied_move_1d": 40.0}),
            (dt.date(2026, 5, 7), {"implied_move_1d": 42.0}),
            (dt.date(2026, 5, 18), {"implied_move_1d": 38.0}),
            (dt.date(2026, 5, 20), {"implied_move_1d": 41.0}),
            (dt.date(2026, 5, 21), {"implied_move_1d": 43.0}),
            (dt.date(2026, 5, 22), {"implied_move_1d": 45.0}),
        ]
        conn = _StubConn(rows)
        result = _load_candidates(conn, "SPX", "v0.5.0")
        self.assertEqual(len(result), 6)
        dates = [d for (d, _) in result]
        self.assertEqual(
            dates,
            ["2026-05-06", "2026-05-07", "2026-05-18",
             "2026-05-20", "2026-05-21", "2026-05-22"],
        )
        # Feature vectors carried through unchanged
        for (_, vec), (_, expected_vec) in zip(result, rows):
            self.assertEqual(vec, expected_vec)

    def test_query_has_no_label_gate(self):
        """The candidate-selection SQL must NOT join on bt_signals or
        filter on label IS NOT NULL — that was CR-013 behavior and CR-014
        removes it. Guards against regression."""
        conn = _StubConn([])
        _load_candidates(conn, "SPX", "v0.5.0")
        sql = conn._cursor.last_sql or ""
        self.assertNotIn("bt_signals", sql.lower())
        self.assertNotIn("label", sql.lower())
        # Sanity: still queries the right table on the right filters
        self.assertIn("bt_daily_features", sql.lower())
        self.assertIn("feature_version", sql.lower())

    def test_empty_corpus_returns_empty_list(self):
        conn = _StubConn([])
        result = _load_candidates(conn, "SPX", "v0.5.0")
        self.assertEqual(result, [])

    def test_passes_ticker_and_version_as_params(self):
        conn = _StubConn([])
        _load_candidates(conn, "ES", "v0.5.1")
        self.assertEqual(conn._cursor.last_params, ("ES", "v0.5.1"))


class TestFetchSessionOutcomes(unittest.TestCase):
    """Hotfix for CR-013 bug: query was referencing nonexistent `ts_pt`
    column. Verifies the corrected query uses `datetime`, applies the
    UTC→PT conversion in the WHERE clause, and returns the expected
    outcome shape with PT-converted session timestamps."""

    def _bars(self):
        # 5/22/2026 PT session — 06:30 open through 13:00 close, naive
        # UTC datetimes (matches the column's storage convention).
        # 06:30 PT = 13:30 UTC; 13:00 PT = 20:00 UTC.
        return [
            (dt.datetime(2026, 5, 22, 13, 30), 5290.00, 5295.00, 5288.00, 5293.00),
            (dt.datetime(2026, 5, 22, 14, 00), 5293.00, 5302.50, 5292.00, 5301.00),
            (dt.datetime(2026, 5, 22, 16, 30), 5301.00, 5310.00, 5285.00, 5288.50),
            (dt.datetime(2026, 5, 22, 19, 00), 5288.50, 5296.00, 5280.00, 5283.00),
            (dt.datetime(2026, 5, 22, 20, 00), 5283.00, 5286.00, 5278.00, 5279.00),
        ]

    def test_returns_populated_outcome_block(self):
        conn = _StubConn(self._bars())
        result = _fetch_session_outcomes(conn, "2026-05-22")

        self.assertIn("eod_return_pts", result)
        self.assertAlmostEqual(result["eod_return_pts"], 5279.00 - 5290.00)
        self.assertAlmostEqual(result["intraday_range_pts"], 5310.00 - 5278.00)
        self.assertAlmostEqual(result["mfe_above_open_pts"], 5310.00 - 5290.00)
        self.assertAlmostEqual(result["mfe_below_open_pts"], 5278.00 - 5290.00)
        # Session timestamps converted to PT for frontend friendliness.
        # 13:30 UTC -> 06:30 PT, 20:00 UTC -> 13:00 PT.
        self.assertEqual(result["session_start"], "2026-05-22T06:30:00-07:00")
        self.assertEqual(result["session_end"], "2026-05-22T13:00:00-07:00")

    def test_empty_corpus_returns_empty_dict(self):
        conn = _StubConn([])
        result = _fetch_session_outcomes(conn, "2026-05-22")
        self.assertEqual(result, {})

    def test_query_uses_datetime_column_not_ts_pt(self):
        """Regression guard for the original CR-013 column-name bug."""
        conn = _StubConn([])
        _fetch_session_outcomes(conn, "2026-05-22")
        sql = (conn._cursor.last_sql or "").lower()
        self.assertNotIn("ts_pt", sql)
        self.assertIn("datetime", sql)
        # Confirm the UTC→PT conversion is in the WHERE clause — naive
        # UTC must be tagged before being converted to PT.
        self.assertIn("at time zone 'utc'", sql)
        self.assertIn("at time zone 'america/los_angeles'", sql)


if __name__ == "__main__":
    unittest.main()
