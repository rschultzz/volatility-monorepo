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

from apps.web.modules.Analogues.routes import _load_candidates


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


if __name__ == "__main__":
    unittest.main()
