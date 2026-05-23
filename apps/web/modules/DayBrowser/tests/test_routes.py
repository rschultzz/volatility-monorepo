"""Integration tests for GET /api/days (CR-016).

Run with:
    python -m unittest apps.web.modules.DayBrowser.tests.test_routes
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parents[6]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import flask
from apps.web.modules.DayBrowser.routes import register_day_browser_routes

_SAMPLE_DAYS = [
    {
        "trade_date": "2026-05-07",
        "regime": "magnetic-pin",
        "auto_regime": "magnetic-pin",
        "feature_vector": {},
        "outcomes": {"open_px": 7362.5, "eod_return_pts": 2.5},
    }
]


def _make_app():
    app = flask.Flask(__name__)
    register_day_browser_routes(app)
    return app


def _patch_conn():
    return patch("apps.web.modules.DayBrowser.routes._conn", return_value=MagicMock())


def _patch_days(days):
    return patch("apps.web.modules.DayBrowser.routes.query_days_by_regime", return_value=days)


class TestDayBrowserRoute(unittest.TestCase):
    def setUp(self):
        self.client = _make_app().test_client()

    def test_missing_regime_returns_400(self):
        r = self.client.get("/api/days?from=2026-04-01&to=2026-05-23")
        self.assertEqual(r.status_code, 400)
        body = json.loads(r.data)
        self.assertIn("regime", body.get("error", "").lower())

    def test_from_after_to_returns_400(self):
        with _patch_conn(), _patch_days(_SAMPLE_DAYS):
            r = self.client.get("/api/days?regime=magnetic-pin&from=2026-05-23&to=2026-04-01")
        self.assertEqual(r.status_code, 400)

    def test_returns_days_array(self):
        with _patch_conn(), _patch_days(_SAMPLE_DAYS):
            r = self.client.get("/api/days?regime=magnetic-pin&from=2026-04-01&to=2026-05-23")
        self.assertEqual(r.status_code, 200)
        body = json.loads(r.data)
        self.assertTrue(body["ok"])
        self.assertEqual(len(body["days"]), 1)
        self.assertEqual(body["count"], 1)
        self.assertEqual(body["regime"], "magnetic-pin")

    def test_empty_days_returns_200(self):
        with _patch_conn(), _patch_days([]):
            r = self.client.get("/api/days?regime=bounded")
        self.assertEqual(r.status_code, 200)
        body = json.loads(r.data)
        self.assertEqual(body["days"], [])
        self.assertEqual(body["count"], 0)

    def test_default_ticker_spx(self):
        with _patch_conn(), _patch_days(_SAMPLE_DAYS) as mock_svc:
            self.client.get("/api/days?regime=magnetic-pin")
        args = mock_svc.call_args[0]
        self.assertEqual(args[1], "SPX")

    def test_custom_ticker(self):
        with _patch_conn(), _patch_days(_SAMPLE_DAYS) as mock_svc:
            self.client.get("/api/days?regime=magnetic-pin&ticker=ES")
        args = mock_svc.call_args[0]
        self.assertEqual(args[1], "ES")

    def test_response_includes_date_range(self):
        with _patch_conn(), _patch_days(_SAMPLE_DAYS):
            r = self.client.get("/api/days?regime=magnetic-pin&from=2026-04-01&to=2026-05-23")
        body = json.loads(r.data)
        self.assertEqual(body["from"], "2026-04-01")
        self.assertEqual(body["to"], "2026-05-23")

    def test_register_idempotent(self):
        app2 = _make_app()
        register_day_browser_routes(app2)
        with _patch_conn(), _patch_days(_SAMPLE_DAYS):
            r = app2.test_client().get("/api/days?regime=magnetic-pin")
        self.assertEqual(r.status_code, 200)


if __name__ == "__main__":
    unittest.main()
