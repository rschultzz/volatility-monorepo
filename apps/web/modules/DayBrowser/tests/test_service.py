"""Unit tests for DayBrowser service (CR-016).

Run with:
    python -m unittest apps.web.modules.DayBrowser.tests.test_service
"""
from __future__ import annotations

import datetime as dt
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parents[6]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.web.modules.DayBrowser.service import query_days_by_regime

DATE_A = dt.date(2026, 5, 7)
DATE_B = dt.date(2026, 5, 14)
DATE_FROM = dt.date(2026, 4, 1)
DATE_TO = dt.date(2026, 5, 23)

_FEATURE_VEC = {"is_pin_day": 1, "implied_move_1d": 55.0}

_DB_ROWS = [
    (DATE_A, _FEATURE_VEC, "magnetic-pin", [], 7362.5),
    (DATE_B, _FEATURE_VEC, "magnet-below", [], 7400.0),
]

_OUTCOMES_ROW = (1746613800, 7362.5, 7365.0, 7370.0, 7358.0)


def _stub_conn(main_rows, outcomes_row=None):
    cur = MagicMock()
    cur.__enter__ = lambda s: cur
    cur.__exit__ = MagicMock(return_value=False)
    cur.fetchall.return_value = main_rows
    cur.fetchone.return_value = outcomes_row or _OUTCOMES_ROW
    conn = MagicMock()
    conn.cursor.return_value = cur
    return conn


def _patch_effective_regimes(override: dict):
    return patch(
        "apps.web.modules.DayBrowser.service.get_effective_regimes",
        return_value=override,
    )


class TestQueryDaysByRegime(unittest.TestCase):
    def test_returns_only_matching_regime(self):
        conn = _stub_conn(_DB_ROWS)
        effective = {DATE_A: "magnetic-pin", DATE_B: "magnet-below"}
        with _patch_effective_regimes(effective):
            result = query_days_by_regime(conn, "SPX", "magnetic-pin", DATE_FROM, DATE_TO)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["trade_date"], "2026-05-07")
        self.assertEqual(result[0]["regime"], "magnetic-pin")

    def test_empty_when_no_matching_regime(self):
        conn = _stub_conn(_DB_ROWS)
        effective = {DATE_A: "magnetic-pin", DATE_B: "magnet-below"}
        with _patch_effective_regimes(effective):
            result = query_days_by_regime(conn, "SPX", "bounded", DATE_FROM, DATE_TO)
        self.assertEqual(result, [])

    def test_promoted_override_applied(self):
        conn = _stub_conn(_DB_ROWS)
        # DATE_B stored as magnet-below but promoted to magnetic-pin
        effective = {DATE_A: "magnetic-pin", DATE_B: "magnetic-pin"}
        with _patch_effective_regimes(effective):
            result = query_days_by_regime(conn, "SPX", "magnetic-pin", DATE_FROM, DATE_TO)
        self.assertEqual(len(result), 2)

    def test_empty_corpus_returns_empty(self):
        conn = _stub_conn([])
        result = query_days_by_regime(conn, "SPX", "magnetic-pin", DATE_FROM, DATE_TO)
        self.assertEqual(result, [])

    def test_row_has_expected_keys(self):
        conn = _stub_conn([_DB_ROWS[0]])
        effective = {DATE_A: "magnetic-pin"}
        with _patch_effective_regimes(effective):
            result = query_days_by_regime(conn, "SPX", "magnetic-pin", DATE_FROM, DATE_TO)
        self.assertTrue(len(result) > 0)
        row = result[0]
        for key in ("trade_date", "regime", "auto_regime", "feature_vector", "outcomes"):
            self.assertIn(key, row)

    def test_outcomes_populated(self):
        conn = _stub_conn([_DB_ROWS[0]], outcomes_row=_OUTCOMES_ROW)
        effective = {DATE_A: "magnetic-pin"}
        with _patch_effective_regimes(effective):
            result = query_days_by_regime(conn, "SPX", "magnetic-pin", DATE_FROM, DATE_TO)
        outcomes = result[0]["outcomes"]
        self.assertIn("open_px", outcomes)
        self.assertAlmostEqual(outcomes["open_px"], 7362.5)


if __name__ == "__main__":
    unittest.main()
