"""Tests for packages/shared/forward_math.py.

Run with:
    python -m pytest packages/shared/tests/test_forward_math.py -v
"""
from __future__ import annotations

import sys
import unittest
from math import exp
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from packages.shared.forward_math import compute_spx_strike


class TestComputeSpxStrike(unittest.TestCase):

    def test_identity_at_zero_carry(self):
        """r == q → carry = 1.0 → strike_spx ≈ strike_es (modulo rounding)."""
        # 5200 ES, r=q=5%, dte=15: no net carry
        result = compute_spx_strike(5200.0, 15, 0.05, 0.05)
        self.assertEqual(result, 5200)

    def test_zero_rates(self):
        """r=0, q=0 → carry = 1.0 → identity mapping."""
        result = compute_spx_strike(4000.0, 15, 0.0, 0.0)
        self.assertEqual(result, 4000)

    def test_typical_positive_carry(self):
        """r > q → ES level > SPX level (discounting pushes SPX below ES)."""
        # strike_es = spx * exp((r-q)*t)
        dte, r, q = 15, 0.05, 0.015
        t = (dte + 1) / 252.0
        spx_exact = 5200.0 / exp((r - q) * t)
        expected = round(spx_exact / 5) * 5
        result = compute_spx_strike(5200.0, dte, r, q)
        self.assertEqual(result, expected)
        # direction check: SPX < ES when r > q
        self.assertLess(result, 5200)

    def test_negative_net_carry(self):
        """q > r → carry < 1 → SPX level > ES level."""
        dte, r, q = 15, 0.01, 0.04
        t = (dte + 1) / 252.0
        spx_exact = 4000.0 / exp((r - q) * t)
        expected = round(spx_exact / 5) * 5
        result = compute_spx_strike(4000.0, dte, r, q)
        self.assertEqual(result, expected)

    def test_rounding_to_nearest_5(self):
        """Result is always a multiple of 5."""
        for dte in (1, 7, 15, 30, 45):
            result = compute_spx_strike(5137.5, dte, 0.053, 0.018)
            self.assertEqual(result % 5, 0, f"Not multiple of 5 for dte={dte}: {result}")

    def test_returns_int(self):
        """Return type must be int."""
        result = compute_spx_strike(4200.0, 15, 0.05, 0.02)
        self.assertIsInstance(result, int)

    def test_roundtrip_consistency(self):
        """compute_discounted_level inverse gives back original SPX within 1 pt."""
        spx_ref = 4200.0
        dte, r, q = 15, 0.05, 0.02
        t = (dte + 1) / 252.0
        es_level = spx_ref * exp((r - q) * t)
        recovered = compute_spx_strike(es_level, dte, r, q)
        # Recovered should be within 5 of original (rounding to nearest 5)
        self.assertAlmostEqual(recovered, spx_ref, delta=5)

    def test_large_dte(self):
        """45-day DTE produces a valid multiple-of-5 result."""
        result = compute_spx_strike(5000.0, 45, 0.05, 0.015)
        self.assertEqual(result % 5, 0)

    def test_zero_dte(self):
        """0-DTE: t = 1/252 ≈ 0 → strike_spx ≈ strike_es (tiny carry)."""
        result = compute_spx_strike(4000.0, 0, 0.05, 0.02)
        self.assertEqual(result % 5, 0)
        # With dte=0, t=1/252; carry is tiny — result very close to 4000
        self.assertAlmostEqual(result, 4000, delta=5)


if __name__ == "__main__":
    unittest.main()
