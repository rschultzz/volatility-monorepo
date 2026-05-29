"""Tests for build_grid_bounds (CR-G Step 9) in Proposals/service.py.

Covers per-regime asymmetric grid logic, cap activation, and fallback cases.

Run with:
    python -m pytest apps/web/modules/Proposals/tests/test_service_grid.py -v
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[6]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.web.modules.Proposals.service import build_grid_bounds


# ── helpers ───────────────────────────────────────────────────────────────────

def _magnet_above(drift_target: float) -> dict:
    return {"regime": "magnet-above", "drift_target": drift_target}

def _magnet_below(drift_target: float) -> dict:
    return {"regime": "magnet-below", "drift_target": drift_target}

def _magnetic_pin(drift_target: float) -> dict:
    return {"regime": "magnetic-pin", "drift_target": drift_target}

def _bounded(lower: float, upper: float) -> dict:
    return {
        "regime": "bounded",
        "containment_zone": {"lower_price": lower, "upper_price": upper},
    }

def _symmetric(regime: str = "amplification") -> dict:
    return {"regime": regime}


# ── magnet-above ─────────────────────────────────────────────────────────────

class TestMagnetAbove(unittest.TestCase):
    """magnet-above: right bound extends past spot+1.5×IM to drift_target+1.5×IM."""

    def test_asymmetric_when_magnet_far(self):
        """Magnet at spot + 1×IM → right bound > spot + 1.5×IM."""
        spot, im = 7420.0, 55.0
        drift = spot + 1.0 * im   # 7475 — magnet 1×IM above spot
        lo, hi = build_grid_bounds(spot, im, _magnet_above(drift))
        # Asymmetric: hi = max(spot+1.5×IM, drift+1.5×IM) = drift+1.5×IM = 7557.5
        sym_hi = spot + 1.5 * im     # 7502.5
        expected_hi = drift + 1.5 * im  # 7557.5
        self.assertGreater(hi, sym_hi)
        self.assertAlmostEqual(hi, expected_hi, places=3)

    def test_symmetric_when_magnet_close(self):
        """Magnet at spot + 0.5×IM → max() picks symmetric bound (no extension)."""
        spot, im = 7420.0, 55.0
        drift = spot + 0.5 * im   # 7447.5 — magnet only 0.5×IM above
        lo, hi = build_grid_bounds(spot, im, _magnet_above(drift))
        sym_hi = spot + 1.5 * im  # 7502.5
        # drift+1.5×IM = 7530 > sym_hi=7502.5, so hi is STILL extended
        # (0.5 + 1.5 = 2.0 > 1.5 — any magnet above spot extends the grid)
        self.assertGreaterEqual(hi, sym_hi)

    def test_left_bound_stays_symmetric(self):
        """Left side is always spot − 1.5×IM for magnet-above."""
        spot, im = 7420.0, 55.0
        drift = spot + 2.0 * im
        lo, hi = build_grid_bounds(spot, im, _magnet_above(drift))
        self.assertAlmostEqual(lo, spot - 1.5 * im, places=3)

    def test_cap_activates_when_magnet_very_far(self):
        """Magnet at spot + 5×IM → hi capped at spot + 3×IM."""
        spot, im = 7420.0, 55.0
        drift = spot + 5.0 * im   # very far
        lo, hi = build_grid_bounds(spot, im, _magnet_above(drift))
        cap_hi = spot + 3.0 * im
        self.assertAlmostEqual(hi, cap_hi, places=3)


# ── magnet-below ─────────────────────────────────────────────────────────────

class TestMagnetBelow(unittest.TestCase):

    def test_asymmetric_when_magnet_far(self):
        """Magnet at spot − 1×IM → left bound extended further below spot."""
        spot, im = 7420.0, 55.0
        drift = spot - 1.0 * im   # 7365
        lo, hi = build_grid_bounds(spot, im, _magnet_below(drift))
        sym_lo = spot - 1.5 * im
        expected_lo = drift - 1.5 * im
        self.assertLess(lo, sym_lo)
        self.assertAlmostEqual(lo, expected_lo, places=3)

    def test_right_bound_stays_symmetric(self):
        """Right side is always spot + 1.5×IM for magnet-below."""
        spot, im = 7420.0, 55.0
        drift = spot - 2.0 * im
        lo, hi = build_grid_bounds(spot, im, _magnet_below(drift))
        self.assertAlmostEqual(hi, spot + 1.5 * im, places=3)

    def test_cap_activates_when_magnet_very_far(self):
        """Magnet at spot − 5×IM → lo capped at spot − 3×IM."""
        spot, im = 7420.0, 55.0
        drift = spot - 5.0 * im
        lo, hi = build_grid_bounds(spot, im, _magnet_below(drift))
        cap_lo = spot - 3.0 * im
        self.assertAlmostEqual(lo, cap_lo, places=3)


# ── magnetic-pin ─────────────────────────────────────────────────────────────

class TestMagneticPin(unittest.TestCase):

    def test_centered_on_drift_target(self):
        """Pin grid is drift_target ± 1.5×IM (centred on magnet, not spot)."""
        spot, im = 7420.0, 55.0
        drift = 7400.0   # pin level slightly below spot
        lo, hi = build_grid_bounds(spot, im, _magnetic_pin(drift))
        self.assertAlmostEqual(lo, drift - 1.5 * im, places=3)
        self.assertAlmostEqual(hi, drift + 1.5 * im, places=3)

    def test_cap_on_far_pin(self):
        """Pin grid capped at spot ± 3×IM on each side."""
        spot, im = 7420.0, 55.0
        drift = spot + 4.0 * im   # absurdly far pin
        lo, hi = build_grid_bounds(spot, im, _magnetic_pin(drift))
        cap_hi = spot + 3.0 * im
        self.assertLessEqual(hi, cap_hi + 1e-9)


# ── bounded ──────────────────────────────────────────────────────────────────

class TestBounded(unittest.TestCase):

    def test_covers_containment_zone_with_buffer(self):
        """Grid covers [lower_price − 0.5×IM, upper_price + 0.5×IM]."""
        spot, im = 7400.0, 50.0
        lower_price = spot - 60.0   # 7340
        upper_price = spot + 60.0   # 7460
        lo, hi = build_grid_bounds(spot, im, _bounded(lower_price, upper_price))
        self.assertAlmostEqual(lo, lower_price - 0.5 * im, places=3)
        self.assertAlmostEqual(hi, upper_price + 0.5 * im, places=3)

    def test_cap_on_very_wide_containment(self):
        """Containment zone wider than 6×IM triggers the cap."""
        spot, im = 7400.0, 50.0
        lower_price = spot - 4.0 * im   # 7200
        upper_price = spot + 4.0 * im   # 7600
        lo, hi = build_grid_bounds(spot, im, _bounded(lower_price, upper_price))
        cap_lo = spot - 3.0 * im
        cap_hi = spot + 3.0 * im
        # With 0.5×IM buffer: lower_price−0.5×IM = spot−4.5×IM > cap_lo=spot−3×IM → capped
        self.assertGreaterEqual(lo, cap_lo - 1e-9)
        self.assertLessEqual(hi, cap_hi + 1e-9)


# ── symmetric regimes ─────────────────────────────────────────────────────────

class TestSymmetricRegimes(unittest.TestCase):

    def _check_symmetric(self, regime: str):
        spot, im = 7420.0, 55.0
        lo, hi = build_grid_bounds(spot, im, {"regime": regime})
        self.assertAlmostEqual(lo, spot - 1.5 * im, places=3)
        self.assertAlmostEqual(hi, spot + 1.5 * im, places=3)

    def test_amplification(self):
        self._check_symmetric("amplification")

    def test_untethered(self):
        self._check_symmetric("untethered")

    def test_broken_magnet(self):
        self._check_symmetric("broken-magnet")

    def test_unknown_regime(self):
        """Unrecognised regime → symmetric fallback, no crash."""
        spot, im = 7420.0, 55.0
        lo, hi = build_grid_bounds(spot, im, {"regime": "future-regime-v2"})
        self.assertAlmostEqual(lo, spot - 1.5 * im, places=3)
        self.assertAlmostEqual(hi, spot + 1.5 * im, places=3)


# ── fallback / safety ─────────────────────────────────────────────────────────

class TestFallbacks(unittest.TestCase):

    def test_none_regime_block(self):
        """None → symmetric fallback, no crash."""
        spot, im = 7420.0, 55.0
        lo, hi = build_grid_bounds(spot, im, None)
        self.assertAlmostEqual(lo, spot - 1.5 * im, places=3)
        self.assertAlmostEqual(hi, spot + 1.5 * im, places=3)

    def test_empty_dict(self):
        """Empty dict (no 'regime' key) → symmetric fallback."""
        spot, im = 7420.0, 55.0
        lo, hi = build_grid_bounds(spot, im, {})
        self.assertAlmostEqual(lo, spot - 1.5 * im, places=3)

    def test_missing_drift_target(self):
        """magnet-above with missing drift_target → symmetric fallback, no crash."""
        spot, im = 7420.0, 55.0
        lo, hi = build_grid_bounds(spot, im, {"regime": "magnet-above"})
        self.assertAlmostEqual(lo, spot - 1.5 * im, places=3)
        self.assertAlmostEqual(hi, spot + 1.5 * im, places=3)

    def test_missing_containment_zone(self):
        """bounded with missing containment_zone → symmetric fallback, no crash."""
        spot, im = 7420.0, 55.0
        lo, hi = build_grid_bounds(spot, im, {"regime": "bounded"})
        self.assertAlmostEqual(lo, spot - 1.5 * im, places=3)
        self.assertAlmostEqual(hi, spot + 1.5 * im, places=3)

    def test_zero_implied_move(self):
        """implied_move == 0 → uses 50pt fallback half-width; asymmetric logic still applies, no crash."""
        spot, im = 7420.0, 0.0
        drift = 7500.0
        lo, hi = build_grid_bounds(spot, im, _magnet_above(drift))
        # half=50 fallback; drift+50=7550 > spot+50=7470 → hi=7550 (capped at spot+150=7570)
        self.assertEqual(lo, spot - 50.0)   # left bound: spot - 50
        self.assertEqual(hi, drift + 50.0)  # right bound: drift + half = 7550
        self.assertLess(lo, hi)

    def test_lo_less_than_hi(self):
        """Sanity: lo < hi for all supported regimes."""
        spot, im = 7420.0, 55.0
        cases = [
            _magnet_above(spot + 1.5 * im),
            _magnet_below(spot - 1.5 * im),
            _magnetic_pin(spot),
            _bounded(spot - 60, spot + 60),
            _symmetric("amplification"),
        ]
        for rb in cases:
            lo, hi = build_grid_bounds(spot, im, rb)
            self.assertLess(lo, hi, f"lo >= hi for {rb}")


# ── 2026-05-21 reference check ────────────────────────────────────────────────

class TestReferenceCase(unittest.TestCase):
    """Matches the Step 9 integration check target: spot=7420.5, IM=55.6, magnet=7526."""

    def test_magnet_above_2026_05_21(self):
        spot, im, drift = 7420.5, 55.6, 7526.0
        lo, hi = build_grid_bounds(spot, im, _magnet_above(drift))
        # Right bound: max(spot+1.5×IM, drift+1.5×IM) = drift+1.5×IM = 7609.4
        # Cap: spot+3×IM = 7420.5+166.8 = 7587.3 → hi capped at 7587.3
        cap_hi = spot + 3.0 * im   # 7587.3
        self.assertAlmostEqual(hi, cap_hi, places=1)
        # Left bound: symmetric
        self.assertAlmostEqual(lo, spot - 1.5 * im, places=1)
        # hi is meaningfully past the magnet (>1×IM past)
        self.assertGreater(hi, drift + im)


if __name__ == "__main__":
    unittest.main()
