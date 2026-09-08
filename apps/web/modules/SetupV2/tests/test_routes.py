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


class TestT15Derivation(unittest.TestCase):
    """A2: target = close_at_horizon − final_close_distance; d15 = target − session_close_t15."""

    def _o(self, td, status="computed", end=dt.date(2026, 9, 1), fcd=-165.675, c15=7700.0):
        return {"trade_date": td, "outcome_status": status, "horizon_end_date": end,
                "final_close_distance_from_target": fcd, "session_close_t15": c15}

    def test_distances_and_exclusion_counts(self):
        closes = {dt.date(2026, 9, 1): 7642.25, dt.date(2026, 9, 3): 7751.25}
        outcomes = [
            self._o("a"),                                                    # wall 7807.925; close_t15 7700 → d15 +107.925
            self._o("b", end=dt.date(2026, 9, 3), fcd=-52.825, c15=7810.0),  # wall 7804.075; closed above → d15 −5.925
            self._o("c", c15=None),                                          # no T+15 close → excluded
            self._o("d", end=dt.date(2026, 9, 9)),                           # no horizon close → excluded
            self._o("e", status="pending_history"),                          # not computed → ignored
        ]
        d15, counts = r.t15_distances_below_target(outcomes, closes)
        self.assertEqual(len(d15), 2)
        self.assertAlmostEqual(d15[0], 107.925, places=3)
        self.assertAlmostEqual(d15[1], -5.925, places=3)
        self.assertEqual(counts, {"n_computed": 4, "n_valued": 2, "n_no_t15_close": 1, "n_no_horizon_close": 1})

    def test_values_through_the_helper(self):
        from packages.shared.probability import analogue_fair_value
        fv = analogue_fair_value([107.925, -5.925], 10.0, seed=20260903)
        self.assertAlmostEqual(fv["fair"], 5.0)               # 0 and 10
        self.assertAlmostEqual(fv["full_payout_rate"], 0.5)
        self.assertAlmostEqual(fv["any_payout_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
