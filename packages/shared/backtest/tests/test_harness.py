"""Tests for BacktestHarness — invariant #3 + smokes #9 + fill/touch/close logic."""
import sys
from datetime import date, datetime
from pathlib import Path

_ROOT = str(Path(__file__).parent.parent.parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import unittest

from packages.shared.strategy_templates import Leg
from packages.shared.backtest.models import QuoteMap, TradeInput
from packages.shared.backtest.harness import BacktestHarness
from packages.shared.backtest.plugins.vertical import VerticalPlugin
from packages.shared.backtest.plugins.debit_vertical import DebitVerticalPlugin
from packages.shared.backtest.plugins.stub_condor import StubCondorPlugin
from packages.shared.backtest.plugins.protocol import BacktestPlugin


# ── Shared fixtures ──────────────────────────────────────────────────────────

_SPLIT_DATE = date(2025, 8, 12)
_LEGS = [
    Leg(side="short", type="call", strike=5100.0),
    Leg(side="long",  type="call", strike=5110.0),
]
_WIDTH = 10.0


def _make_input(signal_date=date(2025, 6, 1), structural_prob=0.60, partition="train"):
    return TradeInput(
        signal_date=signal_date,
        partition=partition,
        distance_band="near",
        drift_target_sigma=1.2,
        structural_prob=structural_prob,
        spread_width=_WIDTH,
        legs=_LEGS,
    )


def _ts(h, m):
    """Helper: build a datetime at given hour:minute."""
    return datetime(2025, 6, 1, h, m, 0)


def _complete_qmap(c1=5.0, c2=2.0):
    """Both legs quoted."""
    return {(5100.0, "C"): c1, (5110.0, "C"): c2}


def _empty_qmap():
    """Neither leg quoted (simulates a missing-quote minute)."""
    return {}


def _harness(threshold=0.0, plugin=None):
    return BacktestHarness(
        plugin=plugin or VerticalPlugin(),
        edge_threshold=threshold,
        split_date=_SPLIT_DATE,
    )


def _debit_harness(threshold=0.0):
    """Harness with DebitVerticalPlugin — use for all touch-exit P&L tests."""
    return BacktestHarness(
        plugin=DebitVerticalPlugin(),
        edge_threshold=threshold,
        split_date=_SPLIT_DATE,
    )


# ── Tests ────────────────────────────────────────────────────────────────────

class TestEntryScanAllMinutesPersisted(unittest.TestCase):
    """Invariant #3: ALL entry-day minutes in entry_scan, not just quoted ones."""

    def test_missing_quote_minutes_still_in_scan(self):
        harness = _harness()
        entry_day = [
            (_ts(13, 31), _empty_qmap()),    # minute 1 — no quotes
            (_ts(13, 32), _complete_qmap()), # minute 2 — quoted
            (_ts(13, 33), _empty_qmap()),    # minute 3 — no quotes
        ]
        result = harness.run_trade(_make_input(), entry_day, None, [], None)
        self.assertEqual(len(result.entry_scan), 3)

    def test_missing_quote_rows_have_none_fields(self):
        harness = _harness()
        entry_day = [
            (_ts(13, 31), _empty_qmap()),
            (_ts(13, 32), _complete_qmap()),
        ]
        result = harness.run_trade(_make_input(), entry_day, None, [], None)
        self.assertIsNone(result.entry_scan[0].net_position_value)
        self.assertIsNone(result.entry_scan[0].net_credit)
        self.assertIsNone(result.entry_scan[0].edge)

    def test_quoted_rows_have_values(self):
        harness = _harness()
        entry_day = [(_ts(13, 31), _complete_qmap(c1=5.0, c2=2.0))]
        result = harness.run_trade(_make_input(structural_prob=0.60), entry_day, None, [], None)
        row = result.entry_scan[0]
        self.assertAlmostEqual(row.net_position_value, -3.0)
        self.assertAlmostEqual(row.net_credit, 3.0)
        # edge = 0.60 - (3.0 / 10.0) = 0.30
        self.assertAlmostEqual(row.edge, 0.30)

    def test_scan_timestamp_matches_input(self):
        harness = _harness()
        entry_day = [
            (_ts(13, 31), _empty_qmap()),
            (_ts(13, 45), _complete_qmap()),
        ]
        result = harness.run_trade(_make_input(), entry_day, None, [], None)
        self.assertEqual(result.entry_scan[0].snapshot_pt, _ts(13, 31))
        self.assertEqual(result.entry_scan[1].snapshot_pt, _ts(13, 45))


class TestFillLogic(unittest.TestCase):
    """Fill = first minute where edge >= threshold."""

    def test_fills_at_first_threshold_minute(self):
        # threshold=0.10; edge = 0.60 - credit/10
        # minute 1: credit=1.0 → edge=0.50 ≥ 0.10 → fill
        # minute 2 would also fill but we take first
        harness = _harness(threshold=0.10)
        entry_day = [
            (_ts(13, 31), _complete_qmap(c1=6.0, c2=5.0)),  # credit=1.0, edge=0.50
            (_ts(13, 32), _complete_qmap(c1=7.0, c2=5.0)),  # credit=2.0, edge=0.40
        ]
        result = harness.run_trade(_make_input(), entry_day, None, [], None)
        self.assertTrue(result.filled)
        self.assertEqual(result.fill_time, _ts(13, 31))
        self.assertAlmostEqual(result.fill_net_credit, 1.0)

    def test_no_fill_when_all_below_threshold(self):
        # threshold=0.40; credit=1.0 → edge=0.50 ≥ 0.40 → fills
        # Actually need below: threshold=0.60; credit=5.0 → edge=0.60-0.50=0.10 < 0.60
        harness = _harness(threshold=0.60)
        entry_day = [
            (_ts(13, 31), _complete_qmap(c1=6.0, c2=1.0)),  # credit=5.0, edge=0.10
        ]
        result = harness.run_trade(_make_input(structural_prob=0.60), entry_day, None, [], None)
        self.assertFalse(result.filled)
        self.assertIsNone(result.fill_time)
        self.assertIsNone(result.fill_net_credit)

    def test_fill_skips_missing_quote_minutes(self):
        # First minute has no quotes; fill should land on second minute
        harness = _harness(threshold=0.0)
        entry_day = [
            (_ts(13, 31), _empty_qmap()),           # no quotes
            (_ts(13, 32), _complete_qmap(c1=5.0, c2=2.0)),
        ]
        result = harness.run_trade(_make_input(), entry_day, None, [], None)
        self.assertTrue(result.filled)
        self.assertEqual(result.fill_time, _ts(13, 32))

    def test_empty_entry_day_not_filled(self):
        harness = _harness()
        result = harness.run_trade(_make_input(), [], None, [], None)
        self.assertFalse(result.filled)


class TestTouchExitOutcome(unittest.TestCase):
    """Touch-exit P&L = entry_credit + position_value_at_close.

    Uses DebitVerticalPlugin (touch_exit_is_meaningful=True). For credit (VerticalPlugin),
    touch-exit P&L is None by design (decision #6).
    """

    def _run_with_touch(self, fill_credit, close_pos_val, touch_ts=None, close_ts=None):
        """Run a filled debit trade with touch at touch_ts, close quote at close_ts."""
        touch_ts = touch_ts or _ts(14, 0)
        close_ts = close_ts or _ts(14, 0)
        # entry credit: -pos_val → pos_val = -fill_credit
        c1 = fill_credit + 2.0  # short at c1, long at 2.0 → net = -c1+2 → credit = c1-2
        c2 = 2.0
        entry_day = [(_ts(13, 31), _complete_qmap(c1=c1, c2=c2))]
        # close qmap: need pos_val = close_pos_val = -c1_close + c2_close
        # pick c1_close s.t. -c1_close + 2.0 = close_pos_val
        c1_close = 2.0 - close_pos_val
        touch_window = [(close_ts, _complete_qmap(c1=c1_close, c2=2.0))]
        harness = _debit_harness(threshold=0.0)
        trade = _make_input(structural_prob=0.60)
        return harness.run_trade(trade, entry_day, touch_ts, touch_window, None)

    def test_touch_exit_pnl_credit_minus_cost(self):
        # entry_credit=3.0, close_pos_val=-1.0 → touch_exit_pnl = 3.0 + (-1.0) = 2.0
        result = self._run_with_touch(fill_credit=3.0, close_pos_val=-1.0)
        self.assertAlmostEqual(result.touch_exit_pnl, 2.0)

    def test_touch_exit_pnl_at_breakeven(self):
        # entry_credit=3.0, close_pos_val=-3.0 → pnl = 0.0
        result = self._run_with_touch(fill_credit=3.0, close_pos_val=-3.0)
        self.assertAlmostEqual(result.touch_exit_pnl, 0.0)

    def test_no_touch_exit_pnl_when_no_touch(self):
        harness = _debit_harness(threshold=0.0)
        entry_day = [(_ts(13, 31), _complete_qmap())]
        result = harness.run_trade(_make_input(), entry_day, None, [], 5095.0)
        self.assertIsNone(result.touch_exit_pnl)
        self.assertIsNone(result.touch_exit_minute)

    def test_touch_exit_minute_at_or_after_touch(self):
        """First option-quote minute AT or AFTER touch_datetime is used."""
        harness = _debit_harness(threshold=0.0)
        entry_day = [(_ts(13, 31), _complete_qmap(c1=5.0, c2=2.0))]
        touch_ts = _ts(14, 30)
        touch_window = [
            (_ts(14, 25), _complete_qmap(c1=4.0, c2=1.5)),  # BEFORE touch — skipped
            (_ts(14, 30), _complete_qmap(c1=3.5, c2=1.0)),  # AT touch — used
            (_ts(14, 35), _complete_qmap(c1=3.0, c2=0.8)),  # after — not used
        ]
        result = harness.run_trade(_make_input(), entry_day, touch_ts, touch_window, None)
        self.assertEqual(result.touch_exit_minute, _ts(14, 30))

    def test_touch_found_flag_set(self):
        harness = _debit_harness(threshold=0.0)
        entry_day = [(_ts(13, 31), _complete_qmap())]
        result = harness.run_trade(_make_input(), entry_day, _ts(14, 0), [(_ts(14, 0), _complete_qmap())], None)
        self.assertTrue(result.touch_found)

    def test_touch_found_false_when_no_touch(self):
        harness = _debit_harness(threshold=0.0)
        entry_day = [(_ts(13, 31), _complete_qmap())]
        result = harness.run_trade(_make_input(), entry_day, None, [], 5095.0)
        self.assertFalse(result.touch_found)


class TestCloseOutcome(unittest.TestCase):
    """Close P&L and zone from settlement price."""

    def test_close_max_profit_zone(self):
        harness = _harness(threshold=0.0)
        entry_day = [(_ts(13, 31), _complete_qmap(c1=5.0, c2=2.0))]
        result = harness.run_trade(_make_input(), entry_day, None, [], 5050.0)
        # entry_credit=3.0, payoff=0 → pnl=3.0
        self.assertAlmostEqual(result.close_pnl, 3.0)
        self.assertEqual(result.close_zone, "max_profit")

    def test_close_max_loss_zone(self):
        harness = _harness(threshold=0.0)
        entry_day = [(_ts(13, 31), _complete_qmap(c1=5.0, c2=2.0))]
        result = harness.run_trade(_make_input(), entry_day, None, [], 5120.0)
        # entry_credit=3.0, payoff=-10 → pnl=-7.0
        self.assertAlmostEqual(result.close_pnl, -7.0)
        self.assertEqual(result.close_zone, "max_loss")

    def test_close_partial_loss_zone(self):
        harness = _harness(threshold=0.0)
        entry_day = [(_ts(13, 31), _complete_qmap(c1=5.0, c2=2.0))]
        result = harness.run_trade(_make_input(), entry_day, None, [], 5105.0)
        # payoff = -5.0 → pnl = 3.0 - 5.0 = -2.0
        self.assertAlmostEqual(result.close_pnl, -2.0)
        self.assertEqual(result.close_zone, "partial_loss")

    def test_close_pnl_none_when_no_fill(self):
        harness = _harness(threshold=0.99)  # very high threshold
        entry_day = [(_ts(13, 31), _complete_qmap())]
        result = harness.run_trade(_make_input(structural_prob=0.10), entry_day, None, [], 5095.0)
        self.assertIsNone(result.close_pnl)
        self.assertIsNone(result.close_zone)

    def test_close_pnl_none_when_no_settlement(self):
        harness = _harness(threshold=0.0)
        entry_day = [(_ts(13, 31), _complete_qmap())]
        result = harness.run_trade(_make_input(), entry_day, None, [], None)
        self.assertIsNone(result.close_pnl)


class TestBaselineOutcome(unittest.TestCase):
    """Baseline fills at first-quoted minute with no edge gate."""

    def test_baseline_fills_first_quoted_minute(self):
        harness = _harness(threshold=0.99)  # edge gate prevents edge fill
        entry_day = [
            (_ts(13, 31), _complete_qmap(c1=5.0, c2=2.0)),
            (_ts(13, 32), _complete_qmap(c1=6.0, c2=3.0)),
        ]
        result = harness.run_trade(_make_input(structural_prob=0.10), entry_day, None, [], 5050.0)
        self.assertFalse(result.filled)
        self.assertAlmostEqual(result.baseline_net_credit, 3.0)
        self.assertAlmostEqual(result.baseline_close_pnl, 3.0)  # pnl=3.0 at max_profit

    def test_baseline_independent_of_fill(self):
        """Baseline and edge fill can differ (fill at minute 2, baseline at minute 1)."""
        harness = _harness(threshold=0.20)
        # min1: credit=1.0, edge=0.50 → fills? 0.50 ≥ 0.20 yes, fills at min1
        # But with threshold=0.50: min1 edge=0.50 → fills
        # Let's use threshold=-1.0 so both fill at minute 1 for this test
        harness2 = BacktestHarness(
            plugin=VerticalPlugin(), edge_threshold=-1.0, split_date=_SPLIT_DATE
        )
        entry_day = [
            (_ts(13, 31), _complete_qmap(c1=5.0, c2=2.0)),  # credit=3.0
        ]
        result = harness2.run_trade(_make_input(), entry_day, None, [], 5095.0)
        self.assertAlmostEqual(result.baseline_net_credit, result.fill_net_credit)


class TestStubCondorPlugin(unittest.TestCase):
    """Smoke #9: StubCondorPlugin satisfies BacktestPlugin protocol and harness runs it."""

    def test_stub_condor_satisfies_protocol(self):
        plugin = StubCondorPlugin()
        self.assertIsInstance(plugin, BacktestPlugin)

    def test_harness_accepts_stub_condor(self):
        """Harness runs end-to-end with StubCondorPlugin — no harness edits needed."""
        harness = BacktestHarness(
            plugin=StubCondorPlugin(),
            edge_threshold=0.0,
            split_date=_SPLIT_DATE,
        )
        legs = [
            Leg(side="short", type="put",  strike=5000.0),
            Leg(side="long",  type="put",  strike=4990.0),
            Leg(side="short", type="call", strike=5100.0),
            Leg(side="long",  type="call", strike=5110.0),
        ]
        trade = TradeInput(
            signal_date=date(2025, 6, 1),
            partition="train",
            distance_band="near",
            drift_target_sigma=1.2,
            structural_prob=0.60,
            spread_width=10.0,
            legs=legs,
        )
        qmap = {
            (5000.0, "P"): 2.5,
            (4990.0, "P"): 1.0,
            (5100.0, "C"): 2.5,
            (5110.0, "C"): 1.0,
        }
        entry_day = [(_ts(13, 31), qmap)]
        result = harness.run_trade(trade, entry_day, None, [], 5050.0)
        self.assertTrue(result.filled)
        self.assertIsNotNone(result.close_pnl)

    def test_stub_condor_net_price_returns_none_when_leg_missing(self):
        plugin = StubCondorPlugin()
        legs = [Leg(side="short", type="call", strike=5100.0)]
        result = plugin.net_price(legs, {})
        self.assertIsNone(result)

    def test_stub_condor_payoff_is_zero(self):
        plugin = StubCondorPlugin()
        self.assertEqual(plugin.payoff([], 5050.0), 0.0)


class TestDualOutcomeIndependence(unittest.TestCase):
    """touch_exit_pnl and close_pnl are independent — both computed from fill_net_credit.

    Uses DebitVerticalPlugin (touch_exit_is_meaningful=True). Decision #6.
    """

    def test_both_outcomes_computed(self):
        harness = _debit_harness(threshold=0.0)
        entry_day = [(_ts(13, 31), _complete_qmap(c1=5.0, c2=2.0))]  # credit=3.0
        touch_ts = _ts(14, 0)
        touch_window = [(_ts(14, 0), _complete_qmap(c1=4.0, c2=1.5))]  # pos_val=-2.5
        # touch_exit_pnl = 3.0 + (-2.5) = 0.5
        result = _debit_harness().run_trade(_make_input(), entry_day, touch_ts, touch_window, 5095.0)
        # close_pnl: DebitVerticalPlugin, settlement=5095 < 5100 (long) → max_loss zone
        # payoff at 5095 for debit (long 5100, short 5110): long 5100C = max(5095-5100,0)=0 → payoff=0
        # close_pnl = 3.0 + 0 = 3.0; zone = max_loss (debit: settle ≤ lower)
        self.assertAlmostEqual(result.touch_exit_pnl, 0.5)
        self.assertAlmostEqual(result.close_pnl, 3.0)
        self.assertEqual(result.close_zone, "max_loss")

    def test_one_outcome_none_doesnt_affect_other(self):
        """No settlement → close_pnl is None; touch outcome unaffected."""
        harness = _debit_harness(threshold=0.0)
        entry_day = [(_ts(13, 31), _complete_qmap(c1=5.0, c2=2.0))]
        touch_ts = _ts(14, 0)
        touch_window = [(_ts(14, 0), _complete_qmap(c1=4.0, c2=1.5))]
        result = harness.run_trade(_make_input(), entry_day, touch_ts, touch_window, None)
        self.assertAlmostEqual(result.touch_exit_pnl, 0.5)
        self.assertIsNone(result.close_pnl)


class TestStructureAsymmetricOutcomes(unittest.TestCase):
    """Decision #6: debit computes touch-exit P&L; credit (VerticalPlugin) does not."""

    def test_debit_computes_touch_exit_pnl(self):
        """DebitVerticalPlugin: touch_exit_pnl is populated when touch occurs."""
        harness = _debit_harness(threshold=0.0)
        entry_day = [(_ts(13, 31), _complete_qmap(c1=5.0, c2=2.0))]
        touch_ts = _ts(14, 0)
        touch_window = [(_ts(14, 0), _complete_qmap(c1=4.0, c2=1.5))]
        result = harness.run_trade(_make_input(), entry_day, touch_ts, touch_window, None)
        self.assertTrue(result.touch_found)
        self.assertIsNotNone(result.touch_exit_pnl)

    def test_credit_touch_exit_pnl_is_none(self):
        """VerticalPlugin (credit): touch_exit_pnl is None even when touch occurs."""
        harness = _harness(threshold=0.0)  # default = VerticalPlugin (credit)
        entry_day = [(_ts(13, 31), _complete_qmap(c1=5.0, c2=2.0))]
        touch_ts = _ts(14, 0)
        touch_window = [(_ts(14, 0), _complete_qmap(c1=4.0, c2=1.5))]
        result = harness.run_trade(_make_input(), entry_day, touch_ts, touch_window, None)
        self.assertTrue(result.touch_found)
        self.assertIsNone(result.touch_exit_pnl,
            msg="Credit plugin must not produce touch-exit P&L (breach diagnostic only)")

    def test_credit_baseline_touch_exit_pnl_is_none(self):
        """Credit baseline also has no touch-exit P&L."""
        harness = _harness(threshold=0.99)  # high threshold → no edge fill
        entry_day = [(_ts(13, 31), _complete_qmap(c1=5.0, c2=2.0))]
        touch_ts = _ts(14, 0)
        touch_window = [(_ts(14, 0), _complete_qmap(c1=4.0, c2=1.5))]
        result = harness.run_trade(_make_input(structural_prob=0.10), entry_day, touch_ts, touch_window, 5050.0)
        self.assertIsNone(result.baseline_touch_exit_pnl,
            msg="Credit baseline touch-exit P&L must be None")
        self.assertIsNotNone(result.baseline_close_pnl)


if __name__ == "__main__":
    unittest.main()
