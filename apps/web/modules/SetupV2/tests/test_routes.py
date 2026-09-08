"""Tests for apps/web/modules/SetupV2/routes.py (CR-AW).

Route wiring only — build_card is patched; DB helpers are stubbed.

Run with:
    python -m pytest apps/web/modules/SetupV2/tests -q
"""
from __future__ import annotations

import datetime as dt
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flask import Flask

from apps.web.modules.SetupV2 import routes as r

_MODULE = "apps.web.modules.SetupV2.routes"


def _app():
    app = Flask(__name__)
    r.register_setup_v2_routes(app)
    r.register_setup_v2_routes(app)   # idempotent
    return app


class TestCardRoute(unittest.TestCase):
    def test_missing_date_is_400(self):
        with _app().test_client() as c:
            resp = c.get("/api/setup-v2/card")
        self.assertEqual(resp.status_code, 400)
        self.assertFalse(resp.get_json()["ok"])

    def test_bad_date_is_400(self):
        with _app().test_client() as c:
            resp = c.get("/api/setup-v2/card?date=nope")
        self.assertEqual(resp.status_code, 400)

    def test_db_connect_failure_is_500(self):
        with patch(f"{_MODULE}._conn", side_effect=RuntimeError("DATABASE_URL is not set")):
            with _app().test_client() as c:
                resp = c.get("/api/setup-v2/card?date=2026-09-03")
        self.assertEqual(resp.status_code, 500)
        self.assertIn("db connect failed", resp.get_json()["error"])

    def test_payload_passthrough_and_connection_closed(self):
        conn = MagicMock()
        payload = {"ok": True, "date": "2026-09-03", "verdict": {"code": "enter"}}
        with patch(f"{_MODULE}._conn", return_value=conn), \
             patch(f"{_MODULE}.build_card", return_value=(payload, 200)) as bc:
            with _app().test_client() as c:
                resp = c.get("/api/setup-v2/card?date=2026-09-03&ticker=SPX")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.get_json()["verdict"]["code"], "enter")
        bc.assert_called_once_with(conn, "SPX", dt.date(2026, 9, 3))
        conn.close.assert_called_once()

    def test_404_from_build_card_propagates(self):
        with patch(f"{_MODULE}._conn", return_value=MagicMock()), \
             patch(f"{_MODULE}.build_card", return_value=({"ok": False, "error": "no landscape"}, 404)):
            with _app().test_client() as c:
                resp = c.get("/api/setup-v2/card?date=2026-09-03")
        self.assertEqual(resp.status_code, 404)

    def test_unexpected_error_is_500(self):
        with patch(f"{_MODULE}._conn", return_value=MagicMock()), \
             patch(f"{_MODULE}.build_card", side_effect=ValueError("boom")):
            with _app().test_client() as c:
                resp = c.get("/api/setup-v2/card?date=2026-09-03")
        self.assertEqual(resp.status_code, 500)
        self.assertEqual(resp.get_json()["error"], "boom")


class TestPageRoute(unittest.TestCase):
    def test_503_when_build_missing(self):
        with patch(f"{_MODULE}._v2_build_ready", return_value=False):
            with _app().test_client() as c:
                resp = c.get("/setup-v2")
                resp2 = c.get("/setup-v2/")
        self.assertEqual(resp.status_code, 503)
        self.assertEqual(resp2.status_code, 503)
        self.assertIn("Setup v2 build not found", resp.get_data(as_text=True))

    def test_serves_the_v2_entry_when_built(self):
        with patch(f"{_MODULE}._v2_build_ready", return_value=True), \
             patch(f"{_MODULE}.send_from_directory", return_value="<html>v2</html>") as sfd:
            with _app().test_client() as c:
                resp = c.get("/setup-v2")
        self.assertEqual(resp.status_code, 200)
        sfd.assert_called_once_with(str(r.V2_DIST_DIR), "setup-v2.html")


class TestProposalLegs(unittest.TestCase):
    def test_pl_data_leg_shape_and_calendar_expiry(self):
        proposal = {
            "expiry_dte_target": 15,
            "legs": [{"side": "long", "type": "call", "strike": 7796.175, "quantity": 1},
                     {"side": "short", "type": "call", "strike": 7806.175, "quantity": 1}],
        }
        legs, exp = r._proposal_legs_for_pricing(proposal, dt.date(2026, 9, 3))
        self.assertEqual(exp, dt.date(2026, 9, 18))
        self.assertEqual(legs[0], {"strike": 7796.175, "expiration": dt.date(2026, 9, 18), "flag": "c", "side": "long", "qty": 1})
        self.assertEqual(legs[1]["side"], "short")


class TestFairValueSign(unittest.TestCase):
    """A1: d = close − wall; positive d pays in full, d ≤ −width pays nothing."""

    def test_sign_convention_through_the_helper(self):
        from packages.shared.probability import analogue_fair_value
        fv = analogue_fair_value([+31.5, -19.4, -134.5, +3.6], 10.0, seed=20260903)
        # values 10, 0, 0, 10 → fair 5.0; full payout 2/4; any payout 2/4
        self.assertAlmostEqual(fv["fair"], 5.0)
        self.assertAlmostEqual(fv["full_payout_rate"], 0.5)
        self.assertAlmostEqual(fv["any_payout_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
