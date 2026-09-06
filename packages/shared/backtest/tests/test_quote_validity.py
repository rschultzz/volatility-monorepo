"""CR-AN decision 2–7: quote-sanity rules and their effect in the harness."""
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

_ROOT = str(Path(__file__).parent.parent.parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import unittest

from packages.shared.strategy_templates import Leg
from packages.shared.backtest.models import TradeInput
from packages.shared.backtest.harness import BacktestHarness
from packages.shared.backtest.plugins.vertical import VerticalPlugin
from packages.shared.backtest.plugins.debit_vertical import DebitVerticalPlugin
from packages.shared.backtest.quote_validity import (
    build_quote_map,
    expected_position_sign,
    leg_quote_is_valid,
    spread_value_is_valid,
)

_CREDIT_LEGS = [Leg(side="short", type="call", strike=5100.0), Leg(side="long", type="call", strike=5110.0)]
_DEBIT_LEGS  = [Leg(side="long",  type="call", strike=5090.0), Leg(side="short", type="call", strike=5100.0)]
_WIDTH = 10.0


def _ts(m):
    return datetime(2025, 6, 1, 6, 30, 0) + timedelta(minutes=m)


def _input(legs, prob=0.60):
    return TradeInput(signal_date=date(2025, 6, 1), partition="train", distance_band="near",
                      drift_target_sigma=1.2, structural_prob=prob, spread_width=_WIDTH, legs=legs)


class TestLegQuoteRule(unittest.TestCase):
    def test_valid_quote_passes(self):
        self.assertTrue(leg_quote_is_valid(5.0, 5.2))
        qmap, bad = build_quote_map([(5100.0, "C", 5.0, 5.2), (5110.0, "C", 2.0, 2.4)])
        self.assertEqual(bad, 0)
        self.assertAlmostEqual(qmap[(5100.0, "C")], 5.1)
        self.assertAlmostEqual(qmap[(5110.0, "C")], 2.2)

    def test_bid_above_ask_rejected(self):
        self.assertFalse(leg_quote_is_valid(5.5, 5.0))
        qmap, bad = build_quote_map([(5100.0, "C", 5.5, 5.0)])
        self.assertEqual((qmap, bad), ({}, 1))

    def test_negative_bid_rejected(self):
        self.assertFalse(leg_quote_is_valid(-0.05, 5.0))

    def test_one_sided_book_rejected(self):
        self.assertFalse(leg_quote_is_valid(0.0, 0.0))    # ask must be > 0
        self.assertFalse(leg_quote_is_valid(None, 5.0))
        self.assertFalse(leg_quote_is_valid(5.0, None))
        self.assertTrue(leg_quote_is_valid(0.0, 0.05))    # zero bid, positive ask is a real book


class TestSpreadRangeRule(unittest.TestCase):
    def test_signs(self):
        self.assertEqual(expected_position_sign(_DEBIT_LEGS), 1)
        self.assertEqual(expected_position_sign(_CREDIT_LEGS), -1)
        puts = [Leg(side="long", type="put", strike=5100.0), Leg(side="short", type="put", strike=5090.0)]
        self.assertEqual(expected_position_sign(puts), 1)
        self.assertIsNone(expected_position_sign(_DEBIT_LEGS + _CREDIT_LEGS))

    def test_value_above_width_rejected(self):
        self.assertFalse(spread_value_is_valid(42.25, _WIDTH, _DEBIT_LEGS))
        self.assertFalse(spread_value_is_valid(-26.8, _WIDTH, _CREDIT_LEGS))

    def test_value_below_zero_rejected(self):
        # a debit call spread can never be worth less than 0 (inverted legs)
        self.assertFalse(spread_value_is_valid(-3.0, _WIDTH, _DEBIT_LEGS))
        # a credit spread can never be worth more than 0 in the harness's signed convention
        self.assertFalse(spread_value_is_valid(3.0, _WIDTH, _CREDIT_LEGS))

    def test_values_in_range_pass(self):
        self.assertTrue(spread_value_is_valid(3.0, _WIDTH, _DEBIT_LEGS))
        self.assertTrue(spread_value_is_valid(-3.0, _WIDTH, _CREDIT_LEGS))
        self.assertTrue(spread_value_is_valid(0.0, _WIDTH, _DEBIT_LEGS))
        self.assertTrue(spread_value_is_valid(10.0, _WIDTH, _DEBIT_LEGS))
        self.assertTrue(spread_value_is_valid(-10.0, _WIDTH, _CREDIT_LEGS))

    def test_none_rejected(self):
        self.assertFalse(spread_value_is_valid(None, _WIDTH, _DEBIT_LEGS))


class TestHarnessAppliesRule(unittest.TestCase):
    def _run(self, legs, plugin, scan):
        h = BacktestHarness(plugin=plugin, edge_threshold=0.0, split_date=date(2025, 8, 12))
        return h.run_trade(_input(legs), scan, None, [], None)

    def test_one_leg_missing_minute_is_invalid(self):
        scan = [(_ts(0), {(5090.0, "C"): 6.0}),                      # short leg missing
                (_ts(1), {(5090.0, "C"): 6.0, (5100.0, "C"): 2.5})]   # valid: +3.5 debit
        r = self._run(_DEBIT_LEGS, DebitVerticalPlugin(), scan)
        self.assertEqual(len(r.entry_scan), 2)
        self.assertFalse(r.entry_scan[0].quote_valid)
        self.assertTrue(r.entry_scan[1].quote_valid)
        self.assertEqual((r.n_minutes_total, r.n_minutes_valid), (2, 1))
        self.assertTrue(r.had_invalid_quote)

    def test_baseline_is_first_valid_minute(self):
        # minutes 0–2 invalid: over width, inverted (negative for a debit), missing leg
        scan = [(_ts(0), {(5090.0, "C"): 50.0, (5100.0, "C"): 2.0}),   # +48 > width
                (_ts(1), {(5090.0, "C"): 1.0,  (5100.0, "C"): 4.0}),   # −3 for a debit
                (_ts(2), {(5100.0, "C"): 2.0}),                        # long leg missing
                (_ts(3), {(5090.0, "C"): 6.0,  (5100.0, "C"): 2.5}),   # +3.5 valid
                (_ts(4), {(5090.0, "C"): 6.2,  (5100.0, "C"): 2.5})]   # +3.7 valid
        r = self._run(_DEBIT_LEGS, DebitVerticalPlugin(), scan)
        self.assertEqual(r.baseline_minute_offset, 3)
        self.assertAlmostEqual(r.baseline_net_credit, -3.5)
        self.assertEqual(r.fill_time, _ts(3))            # gated fill also skips minutes 0–2
        self.assertEqual([m.quote_valid for m in r.entry_scan], [False, False, False, True, True])
        self.assertIsNone(r.excluded_reason)

    def test_zero_valid_minutes_excludes_trade(self):
        scan = [(_ts(0), {(5100.0, "C"): 50.0, (5110.0, "C"): 2.0}),   # credit spread worth −48
                (_ts(1), {(5100.0, "C"): 2.0,  (5110.0, "C"): 5.0}),   # +3 for a credit (inverted)
                (_ts(2), {})]
        r = self._run(_CREDIT_LEGS, VerticalPlugin(), scan)
        self.assertEqual(r.excluded_reason, "no_valid_entry_minute")
        self.assertFalse(r.filled)
        self.assertIsNone(r.baseline_net_credit)
        self.assertEqual((r.n_minutes_total, r.n_minutes_valid), (3, 0))
        self.assertEqual(len(r.entry_scan), 3)           # invariant #3 still holds

    def test_touch_exit_skips_invalid_minute(self):
        entry = [(_ts(0), {(5090.0, "C"): 6.0, (5100.0, "C"): 2.5})]   # fill at +3.5 (credit −3.5)
        touch_scan = [(_ts(30), {(5090.0, "C"): 90.0, (5100.0, "C"): 2.0}),   # +88 invalid
                      (_ts(31), {(5090.0, "C"): 9.0,  (5100.0, "C"): 0.5})]   # +8.5 valid
        h = BacktestHarness(plugin=DebitVerticalPlugin(), edge_threshold=0.0, split_date=date(2025, 8, 12))
        r = h.run_trade(_input(_DEBIT_LEGS), entry, _ts(30), touch_scan, None)
        self.assertEqual(r.touch_exit_minute, _ts(31))
        self.assertAlmostEqual(r.touch_exit_pnl, -3.5 + 8.5)


if __name__ == "__main__":
    unittest.main()
