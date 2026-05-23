"""Unit tests for AuditFlags service (CR-016).

Run with:
    python -m unittest apps.web.modules.AuditFlags.tests.test_service
"""
from __future__ import annotations

import datetime as dt
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

REPO_ROOT = Path(__file__).resolve().parents[6]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.web.modules.AuditFlags.service import (
    _row_to_dict,
    create_flag,
    delete_flag,
    demote_flag,
    list_flags_for_date,
    promote_flag,
)

TRADE_DATE = dt.date(2026, 5, 7)
ANALOGUE_DATE = dt.date(2026, 5, 14)
_NOW = dt.datetime(2026, 5, 23, 12, 0, 0, tzinfo=dt.timezone.utc)

_REGIME_ROW = ("magnetic-pin",)

_REGIME_FLAG_ROW = (
    1, "regime_wrong", "SPX", TRADE_DATE, None,
    "magnetic-pin", "pinned", False, None, _NOW,
)

_PAIR_FLAG_ROW = (
    2, "not_a_true_analogue", "SPX", TRADE_DATE, ANALOGUE_DATE,
    "magnetic-pin", None, False, None, _NOW,
)


def _stub_conn(fetchone_side=None, fetchall_return=None):
    cur = MagicMock()
    cur.__enter__ = lambda s: cur
    cur.__exit__ = MagicMock(return_value=False)
    if fetchone_side is not None:
        cur.fetchone.side_effect = fetchone_side
    cur.fetchall.return_value = fetchall_return or []
    conn = MagicMock()
    conn.cursor.return_value = cur
    return conn


class TestRowToDict(unittest.TestCase):
    def test_basic_serialization(self):
        d = _row_to_dict(_REGIME_FLAG_ROW)
        self.assertEqual(d["flag_id"], 1)
        self.assertEqual(d["flag_type"], "regime_wrong")
        self.assertEqual(d["trade_date"], "2026-05-07")
        self.assertIsNone(d["analogue_date"])
        self.assertEqual(d["auto_regime"], "magnetic-pin")
        self.assertEqual(d["corrected_regime"], "pinned")
        self.assertFalse(d["promoted"])

    def test_pair_flag_serializes_analogue_date(self):
        d = _row_to_dict(_PAIR_FLAG_ROW)
        self.assertEqual(d["analogue_date"], "2026-05-14")


class TestCreateFlag(unittest.TestCase):
    def test_regime_wrong_creates_row(self):
        conn = _stub_conn(fetchone_side=[_REGIME_ROW, _REGIME_FLAG_ROW])
        result = create_flag(
            conn, "regime_wrong", "SPX", TRADE_DATE,
            corrected_regime="pinned",
        )
        self.assertEqual(result["flag_id"], 1)
        self.assertEqual(result["flag_type"], "regime_wrong")

    def test_pair_flag_creates_row(self):
        conn = _stub_conn(fetchone_side=[_REGIME_ROW, _PAIR_FLAG_ROW])
        result = create_flag(
            conn, "not_a_true_analogue", "SPX", TRADE_DATE,
            analogue_date=ANALOGUE_DATE,
        )
        self.assertEqual(result["flag_id"], 2)
        self.assertEqual(result["flag_type"], "not_a_true_analogue")

    def test_raises_on_invalid_flag_type(self):
        conn = _stub_conn()
        with self.assertRaises(ValueError):
            create_flag(conn, "invalid_type", "SPX", TRADE_DATE, corrected_regime="x")

    def test_raises_on_missing_corrected_regime_for_regime_wrong(self):
        conn = _stub_conn()
        with self.assertRaises(ValueError):
            create_flag(conn, "regime_wrong", "SPX", TRADE_DATE)

    def test_raises_on_missing_analogue_date_for_pair_flag(self):
        conn = _stub_conn()
        with self.assertRaises(ValueError):
            create_flag(conn, "not_a_true_analogue", "SPX", TRADE_DATE)

    def test_auto_regime_captured_from_features(self):
        conn = _stub_conn(fetchone_side=[("magnet-below",), _REGIME_FLAG_ROW])
        create_flag(conn, "regime_wrong", "SPX", TRADE_DATE, corrected_regime="pinned")
        cur = conn.cursor.return_value
        # First execute call is the auto_regime lookup
        first_call_sql = cur.execute.call_args_list[0][0][0]
        self.assertIn("bt_daily_features", first_call_sql)

    def test_auto_regime_none_when_feature_row_missing(self):
        conn = _stub_conn(fetchone_side=[None, _REGIME_FLAG_ROW])
        create_flag(conn, "regime_wrong", "SPX", TRADE_DATE, corrected_regime="pinned")
        # Should not raise; auto_regime will be None passed to INSERT
        cur = conn.cursor.return_value
        insert_args = cur.execute.call_args_list[1][0][1]
        # 5th positional param is auto_regime
        self.assertIsNone(insert_args[4])


class TestDeleteFlag(unittest.TestCase):
    def test_returns_true_when_deleted(self):
        conn = _stub_conn(fetchone_side=[(1,)])
        result = delete_flag(conn, 1)
        self.assertTrue(result)

    def test_returns_false_when_not_found(self):
        conn = _stub_conn(fetchone_side=[None])
        result = delete_flag(conn, 999)
        self.assertFalse(result)


class TestPromoteDemoteFlag(unittest.TestCase):
    def test_promote_returns_updated_flag(self):
        promoted_row = tuple(_REGIME_FLAG_ROW[:7]) + (True,) + _REGIME_FLAG_ROW[8:]
        conn = _stub_conn(fetchone_side=[promoted_row])
        result = promote_flag(conn, 1)
        self.assertTrue(result["promoted"])

    def test_promote_raises_when_flag_not_found(self):
        conn = _stub_conn(fetchone_side=[None])
        with self.assertRaises(ValueError):
            promote_flag(conn, 999)

    def test_demote_returns_updated_flag(self):
        conn = _stub_conn(fetchone_side=[_REGIME_FLAG_ROW])
        result = demote_flag(conn, 1)
        self.assertFalse(result["promoted"])

    def test_demote_raises_when_flag_not_found(self):
        conn = _stub_conn(fetchone_side=[None])
        with self.assertRaises(ValueError):
            demote_flag(conn, 999)


class TestListFlagsForDate(unittest.TestCase):
    def test_returns_list_of_dicts(self):
        conn = _stub_conn(fetchall_return=[_REGIME_FLAG_ROW, _PAIR_FLAG_ROW])
        result = list_flags_for_date(conn, "SPX", TRADE_DATE)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["flag_type"], "regime_wrong")
        self.assertEqual(result[1]["flag_type"], "not_a_true_analogue")

    def test_empty_when_no_flags(self):
        conn = _stub_conn(fetchall_return=[])
        result = list_flags_for_date(conn, "SPX", TRADE_DATE)
        self.assertEqual(result, [])

    def test_passes_ticker_and_date_to_query(self):
        conn = _stub_conn(fetchall_return=[])
        list_flags_for_date(conn, "SPX", TRADE_DATE)
        cur = conn.cursor.return_value
        params = cur.execute.call_args[0][1]
        self.assertIn("SPX", params)
        self.assertIn(TRADE_DATE, params)


if __name__ == "__main__":
    unittest.main()
