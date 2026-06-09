"""Tests for VerticalPlugin — smokes #4 and #5 (three-zone payoff) + is_touch."""
import sys
from pathlib import Path

_ROOT = str(Path(__file__).parent.parent.parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import unittest

from packages.shared.strategy_templates import Leg
from packages.shared.backtest.plugins.vertical import VerticalPlugin


def _credit_call_spread(short_k: float, width: float = 10.0) -> list:
    """Short K, long K+width — classic credit call spread."""
    return [
        Leg(side="short", type="call", strike=short_k),
        Leg(side="long",  type="call", strike=short_k + width),
    ]


class TestVerticalPluginPayoff(unittest.TestCase):
    """Three-zone payoff for credit call spread (short K=5100, long K=5110)."""

    def setUp(self):
        self.plugin = VerticalPlugin()
        self.legs = _credit_call_spread(5100.0, width=10.0)

    # ── Smoke #4: max-profit zone ────────────────────────────────────────────

    def test_payoff_below_short_strike(self):
        """S < short strike → both OTM → payoff = 0 (max profit for credit)."""
        self.assertAlmostEqual(self.plugin.payoff(self.legs, 5050.0), 0.0)

    def test_payoff_at_short_strike(self):
        """S = short strike → short call exactly ATM → payoff = 0."""
        self.assertAlmostEqual(self.plugin.payoff(self.legs, 5100.0), 0.0)

    # ── Smoke #5: max-loss zone ──────────────────────────────────────────────

    def test_payoff_above_long_strike(self):
        """S ≥ long strike → both ITM → payoff = -width (max loss for credit)."""
        self.assertAlmostEqual(self.plugin.payoff(self.legs, 5110.0), -10.0)

    def test_payoff_well_above_long_strike(self):
        """S >> long strike → payoff still -width (spread is bounded)."""
        self.assertAlmostEqual(self.plugin.payoff(self.legs, 5200.0), -10.0)

    # ── Between zone ─────────────────────────────────────────────────────────

    def test_payoff_halfway_between(self):
        """S = 5105 → short ITM by 5, long OTM → payoff = -5."""
        self.assertAlmostEqual(self.plugin.payoff(self.legs, 5105.0), -5.0)

    def test_payoff_one_point_above_short(self):
        """S = 5101 → payoff = -1."""
        self.assertAlmostEqual(self.plugin.payoff(self.legs, 5101.0), -1.0)

    def test_payoff_one_point_below_long(self):
        """S = 5109 → payoff = -9."""
        self.assertAlmostEqual(self.plugin.payoff(self.legs, 5109.0), -9.0)

    # ── P&L formula: entry_credit + payoff ──────────────────────────────────

    def test_pnl_max_profit_zone(self):
        """entry_credit + payoff(S ≤ short_K) = entry_credit."""
        entry_credit = 3.0
        payoff = self.plugin.payoff(self.legs, 5050.0)
        self.assertAlmostEqual(entry_credit + payoff, 3.0)

    def test_pnl_max_loss_zone(self):
        """entry_credit + payoff(S ≥ long_K) = entry_credit - width."""
        entry_credit = 3.0
        payoff = self.plugin.payoff(self.legs, 5120.0)
        self.assertAlmostEqual(entry_credit + payoff, -7.0)

    def test_pnl_partial_loss_zone(self):
        """entry_credit + payoff(S between) = credit - (S - short_K)."""
        entry_credit = 3.0
        payoff = self.plugin.payoff(self.legs, 5105.0)
        self.assertAlmostEqual(entry_credit + payoff, -2.0)


class TestVerticalPluginIsTouch(unittest.TestCase):
    """is_touch tests — underlying-only event, not option-quote-based."""

    def setUp(self):
        self.plugin = VerticalPlugin()

    def test_touch_when_at_target(self):
        """ES price exactly at drift_target → touched."""
        self.assertTrue(self.plugin.is_touch(5100.0, 5100.0))

    def test_touch_when_above_target(self):
        """ES price above drift_target → touched."""
        self.assertTrue(self.plugin.is_touch(5100.0, 5105.0))

    def test_no_touch_when_below_target(self):
        """ES price below drift_target → not touched."""
        self.assertFalse(self.plugin.is_touch(5100.0, 5099.99))

    def test_no_touch_well_below(self):
        self.assertFalse(self.plugin.is_touch(5100.0, 4900.0))


class TestVerticalPluginNetPrice(unittest.TestCase):
    """net_price delegation to net_price_from_real_quotes."""

    def setUp(self):
        self.plugin = VerticalPlugin()

    def test_returns_position_value(self):
        legs = [
            Leg(side="short", type="call", strike=5100.0),
            Leg(side="long",  type="call", strike=5110.0),
        ]
        qmap = {(5100.0, "C"): 5.0, (5110.0, "C"): 2.0}
        self.assertAlmostEqual(self.plugin.net_price(legs, qmap), -3.0)

    def test_returns_none_when_leg_missing(self):
        legs = [
            Leg(side="short", type="call", strike=5100.0),
            Leg(side="long",  type="call", strike=5110.0),
        ]
        qmap = {(5100.0, "C"): 5.0}
        self.assertIsNone(self.plugin.net_price(legs, qmap))


if __name__ == "__main__":
    unittest.main()
