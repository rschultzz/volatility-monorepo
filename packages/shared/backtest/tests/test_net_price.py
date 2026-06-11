"""Tests for net_price_from_real_quotes — smokes #1 and #2."""
import sys
import os
from pathlib import Path

# Repo root on sys.path so packages.shared resolves without install
_ROOT = str(Path(__file__).parent.parent.parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import unittest

from packages.shared.strategy_templates import Leg
from packages.shared.backtest.net_price import net_price_from_real_quotes


class TestNetPrice(unittest.TestCase):

    def _credit_spread_legs(self):
        """Short K=5100 call, long K=5110 call — credit call spread."""
        return [
            Leg(side="short", type="call", strike=5100.0),
            Leg(side="long",  type="call", strike=5110.0),
        ]

    def _quote_map(self, c1, c2):
        """quote_map with mids c1 for K=5100 call and c2 for K=5110 call."""
        return {
            (5100.0, "C"): c1,
            (5110.0, "C"): c2,
        }

    # ── Smoke #1: complete quote_map returns correct position value ──────────

    def test_credit_spread_position_value(self):
        """net_price is negative for a credit spread (we NET received cash)."""
        legs = self._credit_spread_legs()
        qmap = self._quote_map(c1=5.0, c2=2.0)
        result = net_price_from_real_quotes(legs, qmap)
        # sign convention: short=-1, long=+1
        # -1*5.0 + 1*2.0 = -3.0  (negative = credit received)
        self.assertAlmostEqual(result, -3.0)

    def test_entry_credit_is_negated_position_value(self):
        """entry_credit = -net_price; positive for credit spread."""
        legs = self._credit_spread_legs()
        qmap = self._quote_map(c1=5.0, c2=2.0)
        pos_val = net_price_from_real_quotes(legs, qmap)
        entry_credit = -pos_val
        self.assertAlmostEqual(entry_credit, 3.0)

    def test_debit_spread_positive_position_value(self):
        """For a debit spread (long low, short high), position value is positive."""
        legs = [
            Leg(side="long",  type="call", strike=5100.0),
            Leg(side="short", type="call", strike=5110.0),
        ]
        qmap = self._quote_map(c1=5.0, c2=2.0)
        result = net_price_from_real_quotes(legs, qmap)
        # +1*5.0 + -1*2.0 = +3.0  (positive = we paid net)
        self.assertAlmostEqual(result, 3.0)

    def test_quantity_respected(self):
        """Leg.quantity=2 doubles the contribution."""
        legs = [
            Leg(side="short", type="call", strike=5100.0, quantity=2),
            Leg(side="long",  type="call", strike=5110.0, quantity=1),
        ]
        qmap = self._quote_map(c1=5.0, c2=2.0)
        result = net_price_from_real_quotes(legs, qmap)
        # -2*5.0 + 1*2.0 = -8.0
        self.assertAlmostEqual(result, -8.0)

    # ── Smoke #2: any missing leg → None ────────────────────────────────────

    def test_missing_short_leg_returns_none(self):
        """Missing short-leg quote returns None."""
        legs = self._credit_spread_legs()
        qmap = {(5110.0, "C"): 2.0}  # 5100 call absent
        result = net_price_from_real_quotes(legs, qmap)
        self.assertIsNone(result)

    def test_missing_long_leg_returns_none(self):
        """Missing long-leg quote returns None."""
        legs = self._credit_spread_legs()
        qmap = {(5100.0, "C"): 5.0}  # 5110 call absent
        result = net_price_from_real_quotes(legs, qmap)
        self.assertIsNone(result)

    def test_empty_quote_map_returns_none(self):
        """Empty quote_map returns None."""
        legs = self._credit_spread_legs()
        result = net_price_from_real_quotes(legs, {})
        self.assertIsNone(result)

    def test_option_type_mapping(self):
        """Leg.type='put' maps to 'P' key, not 'C'."""
        legs = [
            Leg(side="short", type="put", strike=5000.0),
            Leg(side="long",  type="put", strike=4990.0),
        ]
        qmap = {(5000.0, "P"): 4.0, (4990.0, "P"): 2.0}
        result = net_price_from_real_quotes(legs, qmap)
        # -4.0 + 2.0 = -2.0
        self.assertAlmostEqual(result, -2.0)

    def test_wrong_option_type_key_returns_none(self):
        """'call' key (instead of 'C') is not found → None."""
        legs = self._credit_spread_legs()
        qmap = {("call", 5100.0): 5.0, ("call", 5110.0): 2.0}
        result = net_price_from_real_quotes(legs, qmap)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
