"""Tests for packages/shared/strike_anchors.py (CR-015).

Run with:
    python -m unittest packages.shared.tests.test_strike_anchors
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from packages.shared.strike_anchors import (
    ANCHOR_STRATEGIES,
    AnchorStrategy,
    ClusterCenteredAnchor,
)

_CLUSTER = {"center_price": 7400.0, "avg_fwhm": 100.0}
_IMPLIED_MOVE = 60.0


class TestButterflyStrikes(unittest.TestCase):
    anchor = ClusterCenteredAnchor()

    def test_half_fwhm(self):
        lo, body, hi = self.anchor.butterfly_strikes(_CLUSTER, "half_fwhm", _IMPLIED_MOVE)
        self.assertEqual(body, 7400.0)
        self.assertAlmostEqual(lo, 7400.0 - 50.0)
        self.assertAlmostEqual(hi, 7400.0 + 50.0)

    def test_full_fwhm(self):
        lo, body, hi = self.anchor.butterfly_strikes(_CLUSTER, "full_fwhm", _IMPLIED_MOVE)
        self.assertEqual(body, 7400.0)
        self.assertAlmostEqual(lo, 7400.0 - 100.0)
        self.assertAlmostEqual(hi, 7400.0 + 100.0)

    def test_sigma_1x(self):
        lo, body, hi = self.anchor.butterfly_strikes(_CLUSTER, "sigma_1x", _IMPLIED_MOVE)
        self.assertEqual(body, 7400.0)
        self.assertAlmostEqual(lo, 7400.0 - 60.0)
        self.assertAlmostEqual(hi, 7400.0 + 60.0)

    def test_half_wings_narrower_than_full(self):
        lo_h, _, hi_h = self.anchor.butterfly_strikes(_CLUSTER, "half_fwhm", _IMPLIED_MOVE)
        lo_f, _, hi_f = self.anchor.butterfly_strikes(_CLUSTER, "full_fwhm", _IMPLIED_MOVE)
        self.assertLess(hi_h, hi_f)
        self.assertGreater(lo_h, lo_f)

    def test_zero_fwhm_half_gives_zero_spread(self):
        cluster_no_fwhm = {"center_price": 7400.0, "avg_fwhm": 0.0}
        lo, body, hi = self.anchor.butterfly_strikes(cluster_no_fwhm, "half_fwhm", 50.0)
        self.assertEqual(lo, body)
        self.assertEqual(hi, body)

    def test_sigma_1x_ignores_fwhm(self):
        cluster_big_fwhm = {"center_price": 7400.0, "avg_fwhm": 500.0}
        lo, _, hi = self.anchor.butterfly_strikes(cluster_big_fwhm, "sigma_1x", 30.0)
        self.assertAlmostEqual(hi - 7400.0, 30.0)

    def test_missing_avg_fwhm_treated_as_zero(self):
        cluster_no_key = {"center_price": 7400.0}
        lo, body, hi = self.anchor.butterfly_strikes(cluster_no_key, "half_fwhm", 50.0)
        self.assertEqual(lo, body)
        self.assertEqual(hi, body)

    def test_invalid_recipe_raises(self):
        with self.assertRaises(ValueError):
            self.anchor.butterfly_strikes(_CLUSTER, "bad_recipe", _IMPLIED_MOVE)


class TestSpreadStrikes(unittest.TestCase):
    anchor = ClusterCenteredAnchor()

    def test_above_spot_is_call(self):
        short, long_, direction = self.anchor.spread_strikes(7524.0, 7444.0, 10.0)
        self.assertEqual(direction, "call")
        self.assertEqual(short, 7524.0)
        self.assertAlmostEqual(long_, 7534.0)

    def test_below_spot_is_put(self):
        short, long_, direction = self.anchor.spread_strikes(7380.0, 7444.0, 10.0)
        self.assertEqual(direction, "put")
        self.assertEqual(short, 7380.0)
        self.assertAlmostEqual(long_, 7370.0)

    def test_call_long_higher_than_short(self):
        short, long_, direction = self.anchor.spread_strikes(7524.0, 7444.0, 10.0)
        self.assertEqual(direction, "call")
        self.assertGreater(long_, short)

    def test_put_long_lower_than_short(self):
        short, long_, direction = self.anchor.spread_strikes(7380.0, 7444.0, 10.0)
        self.assertEqual(direction, "put")
        self.assertLess(long_, short)


class TestCondorStrikes(unittest.TestCase):
    anchor = ClusterCenteredAnchor()

    def test_structure(self):
        lp, sp, sc, lc = self.anchor.condor_strikes(7502.0, 7543.0, 10.0)
        self.assertEqual(sp, 7502.0)
        self.assertEqual(sc, 7543.0)
        self.assertAlmostEqual(lp, 7492.0)
        self.assertAlmostEqual(lc, 7553.0)

    def test_ordering(self):
        lp, sp, sc, lc = self.anchor.condor_strikes(7500.0, 7540.0, 10.0)
        self.assertLess(lp, sp)
        self.assertLess(sp, sc)
        self.assertLess(sc, lc)

    def test_wing_widths(self):
        lp, sp, sc, lc = self.anchor.condor_strikes(7500.0, 7540.0, 15.0)
        self.assertAlmostEqual(sp - lp, 15.0)
        self.assertAlmostEqual(lc - sc, 15.0)


class TestRegistry(unittest.TestCase):
    def test_cluster_centered_in_registry(self):
        self.assertIn("cluster_centered", ANCHOR_STRATEGIES)

    def test_registry_values_implement_protocol(self):
        for strategy in ANCHOR_STRATEGIES.values():
            self.assertIsInstance(strategy, AnchorStrategy)

    def test_registry_name_matches_key(self):
        for key, strategy in ANCHOR_STRATEGIES.items():
            self.assertEqual(strategy.name, key)


if __name__ == "__main__":
    unittest.main()
