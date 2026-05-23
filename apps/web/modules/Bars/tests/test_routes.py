"""Integration tests for GET /api/bars (CR-016).

Run with:
    python -m unittest apps.web.modules.Bars.tests.test_routes
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parents[6]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import flask
from apps.web.modules.Bars.routes import register_bars_routes, _normalize_db_url, _parse_date

_SAMPLE_BARS = [
    {"time": 1746613800, "open": 7362.5, "high": 7370.0, "low": 7358.0, "close": 7365.25},
    {"time": 1746613860, "open": 7365.25, "high": 7368.0, "low": 7362.0, "close": 7366.50},
]


def _make_app():
    app = flask.Flask(__name__)
    register_bars_routes(app)
    return app


def _patch_bars(bars):
    return patch("apps.web.modules.Bars.routes.fetch_rth_bars", return_value=bars)


def _patch_conn():
    return patch("apps.web.modules.Bars.routes._conn", return_value=MagicMock())


class TestNormalizeDbUrl(unittest.TestCase):
    def test_postgres_scheme_rewritten(self):
        url = _normalize_db_url("postgres://user:pass@host/db")
        self.assertTrue(url.startswith("postgresql://"))

    def test_sslmode_added(self):
        url = _normalize_db_url("postgresql://user:pass@host/db")
        self.assertIn("sslmode=require", url)


class TestParseDate(unittest.TestCase):
    def test_valid_iso(self):
        import datetime as dt
        self.assertEqual(_parse_date("2026-05-07"), dt.date(2026, 5, 7))

    def test_invalid_returns_none(self):
        self.assertIsNone(_parse_date("notadate"))

    def test_none_returns_none(self):
        self.assertIsNone(_parse_date(None))


class TestBarsRoute(unittest.TestCase):
    def setUp(self):
        self.app = _make_app()
        self.client = self.app.test_client()

    def test_missing_date_returns_400(self):
        r = self.client.get("/api/bars")
        self.assertEqual(r.status_code, 400)
        body = json.loads(r.data)
        self.assertIn("date", body.get("error", "").lower())

    def test_invalid_date_returns_400(self):
        r = self.client.get("/api/bars?date=notadate")
        self.assertEqual(r.status_code, 400)

    def test_returns_bars_array(self):
        with _patch_conn(), _patch_bars(_SAMPLE_BARS):
            r = self.client.get("/api/bars?date=2026-05-07&ticker=SPX")
        self.assertEqual(r.status_code, 200)
        data = json.loads(r.data)
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 2)
        self.assertAlmostEqual(data[0]["open"], 7362.5)

    def test_returns_empty_array_when_no_bars(self):
        with _patch_conn(), _patch_bars([]):
            r = self.client.get("/api/bars?date=2026-05-07")
        self.assertEqual(r.status_code, 200)
        data = json.loads(r.data)
        self.assertEqual(data, [])

    def test_default_ticker_spx(self):
        with _patch_conn(), _patch_bars(_SAMPLE_BARS) as mock_svc:
            self.client.get("/api/bars?date=2026-05-07")
        mock_svc.assert_called_once()
        _, ticker, _ = mock_svc.call_args[0]
        self.assertEqual(ticker, "SPX")

    def test_session_param_accepted(self):
        with _patch_conn(), _patch_bars(_SAMPLE_BARS):
            r = self.client.get("/api/bars?date=2026-05-07&session=rth")
        self.assertEqual(r.status_code, 200)

    def test_register_idempotent(self):
        app2 = _make_app()
        register_bars_routes(app2)
        with _patch_conn(), _patch_bars(_SAMPLE_BARS):
            r = app2.test_client().get("/api/bars?date=2026-05-07")
        self.assertEqual(r.status_code, 200)


if __name__ == "__main__":
    unittest.main()
