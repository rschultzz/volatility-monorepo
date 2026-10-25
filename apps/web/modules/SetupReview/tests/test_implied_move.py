"""CR-CD A3 — `service.implied_move`: the canonical IM for the replay price-axis fit (no DB).

Run with:
    apps/web/.venv/bin/python -m unittest apps.web.modules.SetupReview.tests.test_implied_move
"""
from __future__ import annotations

import datetime as dt
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

REPO_ROOT = Path(__file__).resolve().parents[5]
for p in (str(REPO_ROOT), str(REPO_ROOT / "apps" / "web")):
    if p not in sys.path:
        sys.path.insert(0, p)

from modules.SetupReview import service  # noqa: E402


def conn_returning(value):
    cur = MagicMock()
    cur.fetchone.return_value = None if value is None else (value,)
    conn = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cur
    return conn, cur


class TestImpliedMove(unittest.TestCase):
    def test_returns_canonical_value(self):
        conn, cur = conn_returning(41.5)
        self.assertEqual(service.implied_move(conn, dt.date(2026, 5, 1), dt.datetime(2026, 5, 1, 7, 39)), 41.5)
        sql, params = cur.execute.call_args[0]
        self.assertIn("implied_move_1d", sql)
        self.assertEqual(params, (service.CANONICAL_FEATURE_VERSION, dt.date(2026, 5, 1)))

    def test_entry_before_0633_gets_none_without_a_query(self):
        conn, cur = conn_returning(41.5)
        self.assertIsNone(service.implied_move(conn, dt.date(2026, 5, 1), dt.datetime(2026, 5, 1, 6, 32)))
        cur.execute.assert_not_called()
        self.assertEqual(service.implied_move(conn, dt.date(2026, 5, 1), dt.datetime(2026, 5, 1, 6, 33)), 41.5)

    def test_missing_or_nonpositive(self):
        for v in (None, 0.0, -3.0):
            conn, _ = conn_returning(v)
            self.assertIsNone(service.implied_move(conn, dt.date(2026, 5, 1), dt.datetime(2026, 5, 1, 9, 0)))


if __name__ == "__main__":
    unittest.main()
