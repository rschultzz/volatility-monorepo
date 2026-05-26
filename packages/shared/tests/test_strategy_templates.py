"""Tests for packages/shared/strategy_templates.py (CR-015).

Run with:
    python -m unittest packages.shared.tests.test_strategy_templates
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from packages.shared.strategy_templates import (
    DTE_TARGET_BY_BUCKET,
    TEMPLATES,
    Leg,
    TradeProposal,
    generate_proposals,
)

# ── Fixture helpers ───────────────────────────────────────────────────────────

def _pin_cluster(center=7400.0, avg_fwhm=100.0, max_gex=712e9, bucket="8-30"):
    return {"center_price": center, "quality": "pin", "avg_fwhm": avg_fwhm,
            "max_gex": max_gex, "bucket": bucket}


def _feature_cluster(center=7516.0, avg_fwhm=160.0, max_gex=492e9, bucket="8-30"):
    return {"center_price": center, "quality": "feature", "avg_fwhm": avg_fwhm,
            "max_gex": max_gex, "bucket": bucket}


def _pin_payload(spot=7362.0, implied_move=50.0, clusters=None):
    clusters = clusters or [_pin_cluster()]
    return {
        "regime": {
            "regime": "magnetic-pin",
            "drift_target": clusters[0]["center_price"],
            "drift_direction": "pin",
        },
        "confluences": clusters,
        "bucket_summary": {"primary_bucket": "8-30"},
    }


def _magnet_above_payload(spot=7444.0, implied_move=50.0, drift_target=7524.0):
    return {
        "regime": {
            "regime": "magnet-above",
            "drift_target": drift_target,
            "drift_direction": "up",
            "dominant_wall": {"price": drift_target, "gex": 492e9},
        },
        "confluences": [_feature_cluster(center=drift_target)],
        "bucket_summary": {"primary_bucket": "8-30"},
    }


def _magnet_below_payload(spot=7500.0, implied_move=50.0, drift_target=7380.0):
    return {
        "regime": {
            "regime": "magnet-below",
            "drift_target": drift_target,
            "drift_direction": "down",
            "dominant_wall": {"price": drift_target, "gex": 520e9},
        },
        "confluences": [_feature_cluster(center=drift_target)],
        "bucket_summary": {"primary_bucket": "8-30"},
    }


def _bounded_payload(spot=7505.0, lower=7502.0, upper=7543.0):
    return {
        "regime": {
            "regime": "bounded",
            "containment_zone": {
                "lower_price": lower,
                "upper_price": upper,
                "lower_gex": 1694e9,
                "upper_gex": 1652e9,
            },
        },
        "confluences": [_feature_cluster(center=lower + (upper - lower) / 2)],
        "bucket_summary": {"primary_bucket": "8-30"},
    }


def _untethered_payload(spot=7421.0, implied_move=50.0):
    return {
        "regime": {"regime": "untethered"},
        "confluences": [_feature_cluster()],
        "bucket_summary": {"primary_bucket": "8-30"},
    }


def _amplification_payload(spot=7357.0):
    return {
        "regime": {"regime": "amplification"},
        "confluences": [_feature_cluster()],
        "bucket_summary": {"primary_bucket": "8-30"},
    }


# ── DTE target mapping ────────────────────────────────────────────────────────

class TestDteMappings(unittest.TestCase):
    def test_all_buckets_present(self):
        for bucket in ("0DTE", "1-7", "8-30", "30+"):
            self.assertIn(bucket, DTE_TARGET_BY_BUCKET)

    def test_values_ordered(self):
        self.assertLess(DTE_TARGET_BY_BUCKET["0DTE"], DTE_TARGET_BY_BUCKET["1-7"])
        self.assertLess(DTE_TARGET_BY_BUCKET["1-7"], DTE_TARGET_BY_BUCKET["8-30"])
        self.assertLess(DTE_TARGET_BY_BUCKET["8-30"], DTE_TARGET_BY_BUCKET["30+"])


# ── Pin butterfly templates ───────────────────────────────────────────────────

class TestPinButterflyTemplates(unittest.TestCase):
    def setUp(self):
        self.payload = _pin_payload()
        self.spot = 7362.0
        self.im = 50.0
        self.proposals = generate_proposals(self.payload, self.spot, self.im)

    def _butterfly_proposals(self):
        return [p for p in self.proposals if p.template_kind == "butterfly"]

    def test_three_butterfly_variants(self):
        butterflies = self._butterfly_proposals()
        self.assertEqual(len(butterflies), 3)

    def test_template_ids(self):
        ids = {p.template_id for p in self._butterfly_proposals()}
        self.assertEqual(ids, {
            "pin_butterfly_tight",
            "pin_butterfly_medium",
            "pin_butterfly_wide",
        })

    def test_each_has_three_legs(self):
        for p in self._butterfly_proposals():
            self.assertEqual(len(p.legs), 3)

    def test_body_at_cluster_center(self):
        for p in self._butterfly_proposals():
            body_leg = next(l for l in p.legs if l.quantity == 2)
            self.assertAlmostEqual(body_leg.strike, 7400.0)

    def test_wings_symmetric(self):
        for p in self._butterfly_proposals():
            body_leg = next(l for l in p.legs if l.quantity == 2)
            wing_legs = [l for l in p.legs if l.quantity == 1]
            self.assertEqual(len(wing_legs), 2)
            lo = min(l.strike for l in wing_legs)
            hi = max(l.strike for l in wing_legs)
            self.assertAlmostEqual(body_leg.strike - lo, hi - body_leg.strike)

    def test_tight_wings_narrower_than_medium(self):
        tight = next(p for p in self._butterfly_proposals() if p.template_id == "pin_butterfly_tight")
        medium = next(p for p in self._butterfly_proposals() if p.template_id == "pin_butterfly_medium")
        tight_hi = max(l.strike for l in tight.legs)
        medium_hi = max(l.strike for l in medium.legs)
        self.assertLess(tight_hi, medium_hi)

    def test_wing_distances_positive(self):
        for p in self._butterfly_proposals():
            body_leg = next(l for l in p.legs if l.quantity == 2)
            hi = max(l.strike for l in p.legs)
            self.assertGreater(hi - body_leg.strike, 0)

    def test_anchor_strategy_populated(self):
        for p in self._butterfly_proposals():
            self.assertEqual(p.anchor_strategy, "cluster_centered")

    def test_expiry_bucket_populated(self):
        for p in self._butterfly_proposals():
            self.assertIn(p.expiry_dte_bucket, DTE_TARGET_BY_BUCKET)

    def test_no_butterfly_from_feature_cluster(self):
        payload = _magnet_above_payload()
        proposals = generate_proposals(payload, 7444.0, 50.0)
        butterflies = [p for p in proposals if p.template_kind == "butterfly"]
        self.assertEqual(len(butterflies), 0)

    def test_two_pin_clusters_produce_six_butterflies(self):
        two_clusters = _pin_payload(clusters=[
            _pin_cluster(center=7353.0),
            _pin_cluster(center=7372.0),
        ])
        proposals = generate_proposals(two_clusters, 7362.0, 50.0)
        butterflies = [p for p in proposals if p.template_kind == "butterfly"]
        self.assertEqual(len(butterflies), 6)


# ── Directional spread template ───────────────────────────────────────────────

class TestDirectionalSpreadTemplate(unittest.TestCase):
    def test_magnet_above_fires_call_spread(self):
        # CR-I: both credit (directional_spread_to_target) and debit (debit_spread_to_target)
        # are emitted for magnet regimes; qualification filters them at the service layer.
        payload = _magnet_above_payload(spot=7444.0, drift_target=7524.0)
        proposals = generate_proposals(payload, 7444.0, 50.0)
        spreads = [p for p in proposals if p.template_kind == "spread"]
        self.assertEqual(len(spreads), 2)
        # Credit spread: short at drift_target, long further OTM
        credit = next(p for p in spreads if p.template_id == "directional_spread_to_target")
        short_leg = next(l for l in credit.legs if l.side == "short")
        self.assertAlmostEqual(short_leg.strike, 7524.0)
        self.assertEqual(short_leg.type, "call")
        long_leg = next(l for l in credit.legs if l.side == "long")
        self.assertAlmostEqual(long_leg.strike, 7534.0)

    def test_magnet_below_fires_put_spread(self):
        payload = _magnet_below_payload(spot=7500.0, drift_target=7380.0)
        proposals = generate_proposals(payload, 7500.0, 50.0)
        spreads = [p for p in proposals if p.template_kind == "spread"]
        self.assertEqual(len(spreads), 2)
        credit = next(p for p in spreads if p.template_id == "directional_spread_to_target")
        short_leg = next(l for l in credit.legs if l.side == "short")
        self.assertEqual(short_leg.type, "put")
        long_leg = next(l for l in credit.legs if l.side == "long")
        self.assertLess(long_leg.strike, short_leg.strike)

    def test_no_spread_on_pin_day(self):
        payload = _pin_payload()
        proposals = generate_proposals(payload, 7362.0, 50.0)
        spreads = [p for p in proposals if p.template_kind == "spread"]
        self.assertEqual(len(spreads), 0)

    def test_no_spread_on_bounded_day(self):
        payload = _bounded_payload()
        proposals = generate_proposals(payload, 7505.0, 50.0)
        spreads = [p for p in proposals if p.template_kind == "spread"]
        self.assertEqual(len(spreads), 0)

    def test_spread_anchor_strategy_populated(self):
        payload = _magnet_above_payload()
        proposals = generate_proposals(payload, 7444.0, 50.0)
        spread = next(p for p in proposals if p.template_kind == "spread")
        self.assertEqual(spread.anchor_strategy, "cluster_centered")

    def test_spread_template_id(self):
        payload = _magnet_above_payload()
        proposals = generate_proposals(payload, 7444.0, 50.0)
        spread = next(p for p in proposals if p.template_kind == "spread")
        self.assertEqual(spread.template_id, "directional_spread_to_target")

    def test_no_spread_without_drift_target(self):
        payload = {
            "regime": {"regime": "magnet-above"},  # missing drift_target
            "confluences": [],
        }
        proposals = generate_proposals(payload, 7444.0, 50.0)
        spreads = [p for p in proposals if p.template_kind == "spread"]
        self.assertEqual(len(spreads), 0)


# ── Debit spread to target template (CR-025 / CR-I) ──────────────────────────

class TestDebitToTargetTemplate(unittest.TestCase):
    def test_magnet_above_emits_debit_call_spread(self):
        payload = _magnet_above_payload(spot=7444.0, drift_target=7524.0)
        proposals = generate_proposals(payload, 7444.0, 50.0)
        debit = next(
            (p for p in proposals if p.template_id == "debit_spread_to_target"), None
        )
        self.assertIsNotNone(debit)
        self.assertEqual(debit.template_kind, "spread")
        # Long call 10pt inside target (toward spot)
        long_leg = next(l for l in debit.legs if l.side == "long")
        self.assertAlmostEqual(long_leg.strike, 7514.0)
        self.assertEqual(long_leg.type, "call")
        # Short call at drift_target
        short_leg = next(l for l in debit.legs if l.side == "short")
        self.assertAlmostEqual(short_leg.strike, 7524.0)
        self.assertEqual(short_leg.type, "call")

    def test_magnet_below_emits_debit_put_spread(self):
        payload = _magnet_below_payload(spot=7500.0, drift_target=7380.0)
        proposals = generate_proposals(payload, 7500.0, 50.0)
        debit = next(
            (p for p in proposals if p.template_id == "debit_spread_to_target"), None
        )
        self.assertIsNotNone(debit)
        # Long put 10pt inside target (toward spot, so higher strike for puts)
        long_leg = next(l for l in debit.legs if l.side == "long")
        self.assertAlmostEqual(long_leg.strike, 7390.0)
        self.assertEqual(long_leg.type, "put")
        short_leg = next(l for l in debit.legs if l.side == "short")
        self.assertAlmostEqual(short_leg.strike, 7380.0)
        self.assertEqual(short_leg.type, "put")

    def test_no_debit_spread_on_pin_day(self):
        payload = _pin_payload()
        proposals = generate_proposals(payload, 7362.0, 50.0)
        debit_spreads = [p for p in proposals if p.template_id == "debit_spread_to_target"]
        self.assertEqual(len(debit_spreads), 0)

    def test_no_debit_spread_without_drift_target(self):
        payload = {"regime": {"regime": "magnet-above"}, "confluences": []}
        proposals = generate_proposals(payload, 7444.0, 50.0)
        debit_spreads = [p for p in proposals if p.template_id == "debit_spread_to_target"]
        self.assertEqual(len(debit_spreads), 0)


# ── Bounded iron condor template ──────────────────────────────────────────────

class TestBoundedCondorTemplate(unittest.TestCase):
    def test_bounded_fires_condor(self):
        payload = _bounded_payload(spot=7505.0, lower=7502.0, upper=7543.0)
        proposals = generate_proposals(payload, 7505.0, 50.0)
        condors = [p for p in proposals if p.template_kind == "condor"]
        self.assertEqual(len(condors), 1)

    def test_condor_leg_structure(self):
        payload = _bounded_payload(spot=7505.0, lower=7502.0, upper=7543.0)
        proposals = generate_proposals(payload, 7505.0, 50.0)
        condor = next(p for p in proposals if p.template_kind == "condor")
        self.assertEqual(len(condor.legs), 4)
        sides = [l.side for l in condor.legs]
        self.assertEqual(sides.count("long"), 2)
        self.assertEqual(sides.count("short"), 2)

    def test_condor_short_put_at_lower_wall(self):
        payload = _bounded_payload(spot=7505.0, lower=7502.0, upper=7543.0)
        proposals = generate_proposals(payload, 7505.0, 50.0)
        condor = next(p for p in proposals if p.template_kind == "condor")
        short_put = next(l for l in condor.legs if l.side == "short" and l.type == "put")
        self.assertAlmostEqual(short_put.strike, 7502.0)

    def test_condor_short_call_at_upper_wall(self):
        payload = _bounded_payload(spot=7505.0, lower=7502.0, upper=7543.0)
        proposals = generate_proposals(payload, 7505.0, 50.0)
        condor = next(p for p in proposals if p.template_kind == "condor")
        short_call = next(l for l in condor.legs if l.side == "short" and l.type == "call")
        self.assertAlmostEqual(short_call.strike, 7543.0)

    def test_condor_longs_are_10pt_outside_shorts(self):
        payload = _bounded_payload(spot=7505.0, lower=7502.0, upper=7543.0)
        proposals = generate_proposals(payload, 7505.0, 50.0)
        condor = next(p for p in proposals if p.template_kind == "condor")
        long_put = next(l for l in condor.legs if l.side == "long" and l.type == "put")
        long_call = next(l for l in condor.legs if l.side == "long" and l.type == "call")
        self.assertAlmostEqual(long_put.strike, 7492.0)
        self.assertAlmostEqual(long_call.strike, 7553.0)

    def test_no_condor_on_magnet_day(self):
        payload = _magnet_above_payload()
        proposals = generate_proposals(payload, 7444.0, 50.0)
        condors = [p for p in proposals if p.template_kind == "condor"]
        self.assertEqual(len(condors), 0)

    def test_no_condor_without_containment_zone(self):
        payload = {
            "regime": {"regime": "bounded"},  # missing containment_zone
            "confluences": [],
        }
        proposals = generate_proposals(payload, 7505.0, 50.0)
        condors = [p for p in proposals if p.template_kind == "condor"]
        self.assertEqual(len(condors), 0)


# ── Feature no-trade template ─────────────────────────────────────────────────

class TestFeatureNoTradeTemplate(unittest.TestCase):
    def test_untethered_fires_no_trade(self):
        payload = _untethered_payload()
        proposals = generate_proposals(payload, 7421.0, 50.0)
        no_trades = [p for p in proposals if p.template_kind == "no_trade"]
        self.assertEqual(len(no_trades), 1)

    def test_amplification_fires_no_trade(self):
        payload = _amplification_payload()
        proposals = generate_proposals(payload, 7357.0, 50.0)
        no_trades = [p for p in proposals if p.template_kind == "no_trade"]
        self.assertEqual(len(no_trades), 1)

    def test_no_trade_has_empty_legs(self):
        payload = _untethered_payload()
        proposals = generate_proposals(payload, 7421.0, 50.0)
        no_trade = next(p for p in proposals if p.template_kind == "no_trade")
        self.assertEqual(no_trade.legs, [])

    def test_no_trade_rationale_mentions_regime(self):
        payload = _untethered_payload()
        proposals = generate_proposals(payload, 7421.0, 50.0)
        no_trade = next(p for p in proposals if p.template_kind == "no_trade")
        self.assertIn("untethered", no_trade.rationale)

    def test_no_no_trade_on_pin_day(self):
        payload = _pin_payload()
        proposals = generate_proposals(payload, 7362.0, 50.0)
        no_trades = [p for p in proposals if p.template_kind == "no_trade"]
        self.assertEqual(len(no_trades), 0)

    def test_no_no_trade_on_magnet_day(self):
        payload = _magnet_above_payload()
        proposals = generate_proposals(payload, 7444.0, 50.0)
        no_trades = [p for p in proposals if p.template_kind == "no_trade"]
        self.assertEqual(len(no_trades), 0)


# ── generate_proposals integration ───────────────────────────────────────────

class TestGenerateProposals(unittest.TestCase):
    def test_pin_day_only_butterflies(self):
        proposals = generate_proposals(_pin_payload(), 7362.0, 50.0)
        kinds = {p.template_kind for p in proposals}
        self.assertEqual(kinds, {"butterfly"})

    def test_magnet_above_only_spread(self):
        proposals = generate_proposals(_magnet_above_payload(), 7444.0, 50.0)
        kinds = {p.template_kind for p in proposals}
        self.assertEqual(kinds, {"spread"})

    def test_bounded_only_condor(self):
        proposals = generate_proposals(_bounded_payload(), 7505.0, 50.0)
        kinds = {p.template_kind for p in proposals}
        self.assertEqual(kinds, {"condor"})

    def test_untethered_only_no_trade(self):
        proposals = generate_proposals(_untethered_payload(), 7421.0, 50.0)
        kinds = {p.template_kind for p in proposals}
        self.assertEqual(kinds, {"no_trade"})

    def test_empty_regime_block_returns_empty(self):
        payload = {"regime": {}, "confluences": []}
        proposals = generate_proposals(payload, 7400.0, 50.0)
        self.assertEqual(proposals, [])

    def test_all_proposals_are_trade_proposal_instances(self):
        for payload, spot, im in [
            (_pin_payload(), 7362.0, 50.0),
            (_magnet_above_payload(), 7444.0, 50.0),
            (_bounded_payload(), 7505.0, 50.0),
            (_untethered_payload(), 7421.0, 50.0),
        ]:
            for p in generate_proposals(payload, spot, im):
                self.assertIsInstance(p, TradeProposal)

    def test_all_proposals_have_valid_source_dict(self):
        for payload, spot, im in [
            (_pin_payload(), 7362.0, 50.0),
            (_magnet_above_payload(), 7444.0, 50.0),
            (_bounded_payload(), 7505.0, 50.0),
            (_untethered_payload(), 7421.0, 50.0),
        ]:
            for p in generate_proposals(payload, spot, im):
                self.assertIsInstance(p.source, dict)
                self.assertIn("regime", p.source)


if __name__ == "__main__":
    unittest.main()
