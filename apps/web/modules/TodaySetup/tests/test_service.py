"""Tests for apps/web/modules/TodaySetup/service.py (CR-015).

Run with:
    python -m unittest apps.web.modules.TodaySetup.tests.test_service
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.web.modules.TodaySetup.service import (
    build_proposals_response,
    _leg_to_dict,
    _proposal_to_dict,
)
from packages.shared.strategy_templates import Leg, TradeProposal


def _make_leg(side="short", type_="call", strike=7500.0, quantity=1):
    return Leg(side=side, type=type_, strike=strike, quantity=quantity)


def _make_proposal(**kwargs):
    defaults = dict(
        template_id="pin_butterfly_tight",
        template_kind="butterfly",
        anchor_strategy="cluster_centered",
        rationale="Test rationale",
        legs=[_make_leg("long", "call", 7380.0), _make_leg("short", "call", 7400.0, 2),
              _make_leg("long", "call", 7420.0)],
        expiry_dte_target=15,
        expiry_dte_bucket="8-30",
        source={"type": "cluster", "regime": "magnetic-pin"},
        wing_distance_recipe="half_fwhm",
    )
    defaults.update(kwargs)
    return TradeProposal(**defaults)


def _pin_payload():
    return {
        "regime": {
            "regime": "magnetic-pin",
            "drift_target": 7400.0,
        },
        "confluences": [
            {"center_price": 7400.0, "quality": "pin", "avg_fwhm": 100.0,
             "max_gex": 712e9, "bucket": "8-30"},
        ],
        "bucket_summary": {"primary_bucket": "8-30"},
    }


def _magnet_payload():
    return {
        "regime": {
            "regime": "magnet-above",
            "drift_target": 7524.0,
            "drift_direction": "up",
            "dominant_wall": {"price": 7524.0, "gex": 492e9},
        },
        "confluences": [
            {"center_price": 7524.0, "quality": "feature", "avg_fwhm": 160.0,
             "max_gex": 492e9, "bucket": "8-30"},
        ],
        "bucket_summary": {"primary_bucket": "8-30"},
    }


class TestLegToDict(unittest.TestCase):
    def test_fields_present(self):
        leg = _make_leg("long", "put", 7380.0, 2)
        d = _leg_to_dict(leg)
        self.assertEqual(d["side"], "long")
        self.assertEqual(d["type"], "put")
        self.assertEqual(d["strike"], 7380.0)
        self.assertEqual(d["quantity"], 2)

    def test_strike_spx_omitted_by_default(self):
        """Without strike_spx kwarg, the key must not appear in the dict."""
        leg = _make_leg()
        d = _leg_to_dict(leg)
        self.assertNotIn("strike_spx", d)

    def test_strike_spx_included_when_provided(self):
        """Explicit strike_spx=5200 must appear as-is."""
        leg = _make_leg(strike=5207.5)
        d = _leg_to_dict(leg, strike_spx=5205)
        self.assertEqual(d["strike_spx"], 5205)


class TestProposalToDict(unittest.TestCase):
    def test_required_fields(self):
        p = _make_proposal()
        d = _proposal_to_dict(p)
        for key in ("template_id", "template_kind", "anchor_strategy",
                    "rationale", "legs", "expiry_dte_target",
                    "expiry_dte_bucket", "source"):
            self.assertIn(key, d)

    def test_legs_serialised(self):
        p = _make_proposal()
        d = _proposal_to_dict(p)
        self.assertIsInstance(d["legs"], list)
        self.assertEqual(len(d["legs"]), 3)
        for leg in d["legs"]:
            self.assertIn("strike", leg)

    def test_strike_spx_absent_without_carry_rates(self):
        """Without r/q kwargs, strike_spx must not appear on any leg."""
        p = _make_proposal()
        d = _proposal_to_dict(p)
        for leg in d["legs"]:
            self.assertNotIn("strike_spx", leg)

    def test_strike_spx_present_with_carry_rates(self):
        """With r=5%, q=1.5%, each leg gains a strike_spx that is a multiple of 5."""
        p = _make_proposal()
        d = _proposal_to_dict(p, risk_free_rate=0.05, yield_rate=0.015)
        for leg in d["legs"]:
            self.assertIn("strike_spx", leg)
            self.assertIsInstance(leg["strike_spx"], int)
            self.assertEqual(leg["strike_spx"] % 5, 0)

    def test_strike_spx_direction(self):
        """r > q → ES level > SPX level (carry discounts SPX above ES)."""
        p = _make_proposal(legs=[_make_leg(strike=7400.0)])
        d = _proposal_to_dict(p, risk_free_rate=0.05, yield_rate=0.015)
        strike_spx = d["legs"][0]["strike_spx"]
        # SPX should be less than ES level when net carry is positive
        self.assertLess(strike_spx, 7400)

    def test_wing_distance_recipe_included_for_butterfly(self):
        p = _make_proposal(template_kind="butterfly", wing_distance_recipe="half_fwhm")
        d = _proposal_to_dict(p)
        self.assertIn("wing_distance_recipe", d)
        self.assertEqual(d["wing_distance_recipe"], "half_fwhm")

    def test_wing_distance_recipe_omitted_for_spread(self):
        p = _make_proposal(
            template_id="directional_spread_to_target",
            template_kind="spread",
            wing_distance_recipe="",
            legs=[_make_leg("short", "call", 7524.0), _make_leg("long", "call", 7534.0)],
        )
        d = _proposal_to_dict(p)
        self.assertNotIn("wing_distance_recipe", d)


class TestBuildProposalsResponse(unittest.TestCase):
    def setUp(self):
        self.context = {
            "date": "2026-05-07",
            "ticker": "SPX",
            "spot": 7362.0,
            "implied_move": 50.0,
            "regime": "magnetic-pin",
        }

    def test_ok_true(self):
        resp = build_proposals_response(_pin_payload(), 7362.0, 50.0, self.context)
        self.assertTrue(resp["ok"])

    def test_context_carried_through(self):
        resp = build_proposals_response(_pin_payload(), 7362.0, 50.0, self.context)
        self.assertEqual(resp["context"], self.context)

    def test_proposals_list_present(self):
        resp = build_proposals_response(_pin_payload(), 7362.0, 50.0, self.context)
        self.assertIsInstance(resp["proposals"], list)
        self.assertGreater(len(resp["proposals"]), 0)

    def test_pin_day_yields_three_butterflies(self):
        resp = build_proposals_response(_pin_payload(), 7362.0, 50.0, self.context)
        kinds = [p["template_kind"] for p in resp["proposals"]]
        self.assertEqual(kinds.count("butterfly"), 3)

    def test_magnet_day_yields_spread(self):
        ctx = dict(self.context, regime="magnet-above")
        resp = build_proposals_response(_magnet_payload(), 7444.0, 50.0, ctx)
        kinds = [p["template_kind"] for p in resp["proposals"]]
        self.assertIn("spread", kinds)

    def test_all_proposals_have_required_keys(self):
        resp = build_proposals_response(_pin_payload(), 7362.0, 50.0, self.context)
        required = {"template_id", "template_kind", "anchor_strategy",
                    "rationale", "legs", "expiry_dte_target",
                    "expiry_dte_bucket", "source"}
        for p in resp["proposals"]:
            for key in required:
                self.assertIn(key, p, f"missing key {key!r} in proposal {p['template_id']!r}")

    def test_strike_spx_in_proposals_when_carry_rates_provided(self):
        """build_proposals_response with r/q propagates strike_spx to all legs."""
        resp = build_proposals_response(
            _pin_payload(), 7362.0, 50.0, self.context,
            risk_free_rate=0.05, yield_rate=0.015,
        )
        for p in resp["proposals"]:
            for leg in p["legs"]:
                self.assertIn("strike_spx", leg)
                self.assertEqual(leg["strike_spx"] % 5, 0)

    def test_strike_spx_absent_when_carry_rates_omitted(self):
        """build_proposals_response without r/q must not emit strike_spx."""
        resp = build_proposals_response(_pin_payload(), 7362.0, 50.0, self.context)
        for p in resp["proposals"]:
            for leg in p["legs"]:
                self.assertNotIn("strike_spx", leg)


if __name__ == "__main__":
    unittest.main()
