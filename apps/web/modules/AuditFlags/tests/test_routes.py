"""Integration tests for /api/audit-flags endpoints (CR-016).

Run with:
    python -m unittest apps.web.modules.AuditFlags.tests.test_routes
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
from apps.web.modules.AuditFlags.routes import register_audit_flags_routes

TRADE_DATE = dt.date(2026, 5, 7)
_NOW = dt.datetime(2026, 5, 23, 12, 0, 0, tzinfo=dt.timezone.utc)

_REGIME_FLAG = {
    "flag_id": 1,
    "flag_type": "regime_wrong",
    "ticker": "SPX",
    "trade_date": "2026-05-07",
    "analogue_date": None,
    "auto_regime": "magnetic-pin",
    "corrected_regime": "pinned",
    "promoted": False,
    "note": None,
    "created_at": _NOW.isoformat(),
}


def _make_app():
    app = flask.Flask(__name__)
    register_audit_flags_routes(app)
    return app


def _patch_conn():
    return patch("apps.web.modules.AuditFlags.routes._conn", return_value=MagicMock())


class TestCreateFlag(unittest.TestCase):
    def setUp(self):
        self.client = _make_app().test_client()

    def test_missing_trade_date_returns_400(self):
        with _patch_conn():
            r = self.client.post("/api/audit-flags",
                                 json={"flag_type": "regime_wrong", "corrected_regime": "pinned"})
        self.assertEqual(r.status_code, 400)

    def test_invalid_flag_type_returns_400(self):
        with _patch_conn(), \
             patch("apps.web.modules.AuditFlags.routes.create_flag",
                   side_effect=ValueError("Invalid flag_type")):
            r = self.client.post("/api/audit-flags", json={
                "flag_type": "bad_type", "trade_date": "2026-05-07",
            })
        self.assertEqual(r.status_code, 400)

    def test_valid_regime_wrong_returns_201(self):
        with _patch_conn(), \
             patch("apps.web.modules.AuditFlags.routes.create_flag", return_value=_REGIME_FLAG):
            r = self.client.post("/api/audit-flags", json={
                "flag_type": "regime_wrong",
                "trade_date": "2026-05-07",
                "corrected_regime": "pinned",
            })
        self.assertEqual(r.status_code, 201)
        body = json.loads(r.data)
        self.assertTrue(body["ok"])
        self.assertEqual(body["flag"]["flag_id"], 1)


class TestDeleteFlag(unittest.TestCase):
    def setUp(self):
        self.client = _make_app().test_client()

    def test_delete_found_returns_200(self):
        with _patch_conn(), \
             patch("apps.web.modules.AuditFlags.routes.delete_flag", return_value=True):
            r = self.client.delete("/api/audit-flags/1")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(json.loads(r.data)["ok"])

    def test_delete_not_found_returns_404(self):
        with _patch_conn(), \
             patch("apps.web.modules.AuditFlags.routes.delete_flag", return_value=False):
            r = self.client.delete("/api/audit-flags/999")
        self.assertEqual(r.status_code, 404)

    def test_invalid_flag_id_returns_400(self):
        r = self.client.delete("/api/audit-flags/notanint")
        self.assertEqual(r.status_code, 400)


class TestPromoteDemote(unittest.TestCase):
    def setUp(self):
        self.client = _make_app().test_client()

    def test_promote_returns_200(self):
        promoted = dict(_REGIME_FLAG, promoted=True)
        with _patch_conn(), \
             patch("apps.web.modules.AuditFlags.routes.promote_flag", return_value=promoted):
            r = self.client.post("/api/audit-flags/1/promote")
        self.assertEqual(r.status_code, 200)
        self.assertTrue(json.loads(r.data)["flag"]["promoted"])

    def test_promote_not_found_returns_404(self):
        with _patch_conn(), \
             patch("apps.web.modules.AuditFlags.routes.promote_flag",
                   side_effect=ValueError("not found")):
            r = self.client.post("/api/audit-flags/999/promote")
        self.assertEqual(r.status_code, 404)

    def test_demote_returns_200(self):
        with _patch_conn(), \
             patch("apps.web.modules.AuditFlags.routes.demote_flag", return_value=_REGIME_FLAG):
            r = self.client.post("/api/audit-flags/1/demote")
        self.assertEqual(r.status_code, 200)
        self.assertFalse(json.loads(r.data)["flag"]["promoted"])


class TestListFlags(unittest.TestCase):
    def setUp(self):
        self.client = _make_app().test_client()

    def test_missing_date_returns_400(self):
        r = self.client.get("/api/audit-flags?ticker=SPX")
        self.assertEqual(r.status_code, 400)

    def test_valid_request_returns_flags(self):
        with _patch_conn(), \
             patch("apps.web.modules.AuditFlags.routes.list_flags_for_date",
                   return_value=[_REGIME_FLAG]):
            r = self.client.get("/api/audit-flags?date=2026-05-07&ticker=SPX")
        self.assertEqual(r.status_code, 200)
        body = json.loads(r.data)
        self.assertTrue(body["ok"])
        self.assertEqual(len(body["flags"]), 1)

    def test_empty_flags_list(self):
        with _patch_conn(), \
             patch("apps.web.modules.AuditFlags.routes.list_flags_for_date", return_value=[]):
            r = self.client.get("/api/audit-flags?date=2026-05-07")
        self.assertEqual(r.status_code, 200)
        body = json.loads(r.data)
        self.assertEqual(body["flags"], [])


if __name__ == "__main__":
    unittest.main()
