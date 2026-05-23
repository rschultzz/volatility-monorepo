"""Tests for apps/web/modules/TodaySetup/routes.py (CR-015).

Uses stub connection/cursor pattern from Analogues/tests/test_routes.py to
exercise routes-layer helpers without a real DB.

Run with:
    python -m unittest apps.web.modules.TodaySetup.tests.test_routes
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.web.modules.TodaySetup.routes import (
    _build_context,
    _normalize_db_url,
    _parse_date,
    _parse_float,
)
import datetime as dt


class TestNormalizeDbUrl(unittest.TestCase):
    def test_postgres_scheme_rewritten(self):
        url = _normalize_db_url("postgres://user:pass@host/db")
        self.assertTrue(url.startswith("postgresql://"))

    def test_postgresql_plus_scheme_rewritten(self):
        url = _normalize_db_url("postgresql+psycopg://user:pass@host/db")
        self.assertTrue(url.startswith("postgresql://"))
        self.assertNotIn("+psycopg", url)

    def test_sslmode_added_when_absent(self):
        url = _normalize_db_url("postgresql://user:pass@host/db")
        self.assertIn("sslmode=require", url)

    def test_sslmode_not_duplicated_when_present(self):
        url = _normalize_db_url("postgresql://user:pass@host/db?sslmode=require")
        self.assertEqual(url.count("sslmode=require"), 1)


class TestParseDate(unittest.TestCase):
    def test_valid_iso(self):
        d = _parse_date("2026-05-07")
        self.assertEqual(d, dt.date(2026, 5, 7))

    def test_invalid_returns_none(self):
        self.assertIsNone(_parse_date("not-a-date"))
        self.assertIsNone(_parse_date(""))
        self.assertIsNone(_parse_date(None))


class TestParseFloat(unittest.TestCase):
    def test_valid_float(self):
        self.assertAlmostEqual(_parse_float("7444.3"), 7444.3)

    def test_invalid_returns_none(self):
        self.assertIsNone(_parse_float("abc"))

    def test_none_returns_none(self):
        self.assertIsNone(_parse_float(None))

    def test_nan_returns_none(self):
        self.assertIsNone(_parse_float("nan"))


class TestBuildContext(unittest.TestCase):
    def _simple_payload(self, regime="magnetic-pin", n_clusters=1):
        clusters = [
            {
                "center_price": 7400.0 + i * 50,
                "quality": "pin",
                "max_gex": 712e9,
                "avg_fwhm": 100.0,
                "bucket": "8-30",
            }
            for i in range(n_clusters)
        ]
        return {
            "regime": {"regime": regime},
            "confluences": clusters,
            "bucket_summary": {"primary_bucket": "8-30"},
        }

    def test_date_and_ticker_in_context(self):
        payload = self._simple_payload()
        ctx = _build_context(dt.date(2026, 5, 7), "SPX", 7362.0, 50.0, payload)
        self.assertEqual(ctx["date"], "2026-05-07")
        self.assertEqual(ctx["ticker"], "SPX")

    def test_spot_and_implied_move(self):
        payload = self._simple_payload()
        ctx = _build_context(dt.date(2026, 5, 7), "SPX", 7362.0, 50.0, payload)
        self.assertAlmostEqual(ctx["spot"], 7362.0)
        self.assertAlmostEqual(ctx["implied_move"], 50.0)

    def test_regime_extracted(self):
        payload = self._simple_payload(regime="magnet-above")
        ctx = _build_context(dt.date(2026, 5, 22), "SPX", 7444.0, 50.0, payload)
        self.assertEqual(ctx["regime"], "magnet-above")

    def test_top_cluster_is_highest_gex(self):
        payload = self._simple_payload(n_clusters=2)
        # Second cluster has higher max_gex by construction:
        payload["confluences"][1]["max_gex"] = 900e9
        ctx = _build_context(dt.date(2026, 5, 7), "SPX", 7362.0, 50.0, payload)
        self.assertAlmostEqual(ctx["top_cluster"]["center_price"], 7450.0)

    def test_clusters_list_length(self):
        payload = self._simple_payload(n_clusters=3)
        ctx = _build_context(dt.date(2026, 5, 7), "SPX", 7362.0, 50.0, payload)
        self.assertEqual(len(ctx["clusters"]), 3)

    def test_no_clusters_top_cluster_is_none(self):
        payload = {"regime": {"regime": "untethered"}, "confluences": [],
                   "bucket_summary": {}}
        ctx = _build_context(dt.date(2026, 5, 18), "SPX", 7421.0, 50.0, payload)
        self.assertIsNone(ctx["top_cluster"])
        self.assertEqual(ctx["clusters"], [])


if __name__ == "__main__":
    unittest.main()
