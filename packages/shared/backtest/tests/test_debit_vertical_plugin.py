"""Tests for DebitVerticalPlugin — hand-check (Step 3 mirror of Step 1 credit verification).

Debit spread under test (all classes):
    LONG  4190 call @ 3.00  (long leg = target−10)
    SHORT 4200 call @ 1.50  (short leg = target)
    net debit paid   = 3.00 − 1.50 = 1.50  (pos_val = +1.50, net_credit = −1.50)
    spread width     = 4200 − 4190 = 10

Payoff (intrinsic at expiry, per-leg sign convention):
    settle ≤ 4190  → long 4190C = 0, short 4200C = 0 → payoff = 0
    settle = 4195  → long 4190C = 5, short 4200C = 0 → payoff = +5
    settle = 4200  → long 4190C = 10, short 4200C = 0 → payoff = +10
    settle > 4200  → long 4190C = settle−4190, short 4200C = settle−4200
                   → payoff = +(settle−4190) − +(settle−4200) = +10 (bounded)

P&L = net_credit + payoff = −1.50 + payoff:
    settle ≤ 4190  → −1.50  (max loss = −debit)
    settle = 4195  → +3.50  (partial profit)
    settle = 4200  → +8.50  (max profit = width − debit)
    settle > 4200  → +8.50  (bounded)

These numbers are derived by hand and must match the plugin output exactly.
"""
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
from packages.shared.backtest.plugins.debit_vertical import DebitVerticalPlugin
from packages.shared.backtest.plugins.protocol import BacktestPlugin


# ── Spread fixtures ──────────────────────────────────────────────────────────

LONG_K  = 4190.0   # long leg: target − 10
SHORT_K = 4200.0   # short leg: target
WIDTH   = SHORT_K - LONG_K   # 10.0

LONG_MID  = 3.00   # long 4190 call mid
SHORT_MID = 1.50   # short 4200 call mid
# pos_val = +1*3.00 + -1*1.50 = +1.50  (positive = paid net debit)
# net_credit = -pos_val = -1.50         (negative = debit paid)

_LEGS = [
    Leg(side="long",  type="call", strike=LONG_K),
    Leg(side="short", type="call", strike=SHORT_K),
]

_SPLIT = date(2025, 8, 12)


def _trade(structural_prob=0.60):
    return TradeInput(
        signal_date=date(2025, 6, 1),
        partition="train",
        distance_band="near",
        drift_target_sigma=1.2,
        structural_prob=structural_prob,
        spread_width=WIDTH,
        legs=_LEGS,
    )


def _harness():
    return BacktestHarness(
        plugin=DebitVerticalPlugin(),
        edge_threshold=0.0,
        split_date=_SPLIT,
    )


def _entry_qmap():
    return {(LONG_K, "C"): LONG_MID, (SHORT_K, "C"): SHORT_MID}


def _ts(h, m):
    return datetime(2025, 6, 1, h, m, 0)


# ── Protocol conformance ─────────────────────────────────────────────────────

class TestDebitPluginProtocol(unittest.TestCase):
    """DebitVerticalPlugin satisfies BacktestPlugin protocol."""

    def test_satisfies_protocol(self):
        self.assertIsInstance(DebitVerticalPlugin(), BacktestPlugin)

    def test_touch_exit_is_meaningful_true(self):
        self.assertTrue(DebitVerticalPlugin.touch_exit_is_meaningful)


# ── Payoff hand-check (five points) ─────────────────────────────────────────

class TestDebitPayoffHandCheck(unittest.TestCase):
    """Decision #5: mirrored three-zone payoff, verified by hand.

    All expected payoffs derived from first principles above.
    These are the five canonical points the task requires.
    """

    def setUp(self):
        self.plugin = DebitVerticalPlugin()

    # Point 1: well below long strike (max loss zone)
    def test_payoff_below_long_strike(self):
        """settle = 4180 < 4190 (long): payoff = 0  (both OTM)."""
        actual = self.plugin.payoff(_LEGS, 4180.0)
        print(f"\n[HAND] settle=4180  payoff={actual:+.2f}  expected=0.00")
        self.assertAlmostEqual(actual, 0.0, places=10)

    # Point 2: exactly at long strike (boundary)
    def test_payoff_at_long_strike(self):
        """settle = 4190 (exactly at long): payoff = 0.  P&L = −1.50 (max loss)."""
        actual = self.plugin.payoff(_LEGS, 4190.0)
        print(f"\n[HAND] settle=4190  payoff={actual:+.2f}  expected=0.00")
        self.assertAlmostEqual(actual, 0.0, places=10)

    # Point 3: between strikes (partial profit zone)
    def test_payoff_between_strikes(self):
        """settle = 4195 (between 4190 and 4200): payoff = +5.  P&L = +3.50."""
        actual = self.plugin.payoff(_LEGS, 4195.0)
        print(f"\n[HAND] settle=4195  payoff={actual:+.2f}  expected=+5.00")
        self.assertAlmostEqual(actual, 5.0, places=10)

    # Point 4: exactly at short strike (boundary — target reached)
    def test_payoff_at_short_strike(self):
        """settle = 4200 (exactly at short = target): payoff = +10.  P&L = +8.50."""
        actual = self.plugin.payoff(_LEGS, 4200.0)
        print(f"\n[HAND] settle=4200  payoff={actual:+.2f}  expected=+10.00")
        self.assertAlmostEqual(actual, 10.0, places=10)

    # Point 5: above short strike (max profit zone, bounded)
    def test_payoff_above_short_strike(self):
        """settle = 4210 > 4200 (above short): payoff = +10 (bounded by width)."""
        actual = self.plugin.payoff(_LEGS, 4210.0)
        print(f"\n[HAND] settle=4210  payoff={actual:+.2f}  expected=+10.00")
        self.assertAlmostEqual(actual, 10.0, places=10)


# ── P&L hand-check via harness ────────────────────────────────────────────────

class TestDebitPnLHandCheck(unittest.TestCase):
    """Full P&L via harness: entry_credit + payoff at each of the five points.

    net_credit = −1.50, so P&L = −1.50 + payoff.
    """

    def _run_close(self, settlement):
        entry_day = [(_ts(13, 31), _entry_qmap())]
        return _harness().run_trade(_trade(), entry_day, None, [], settlement)

    def test_pnl_below_long_strike_max_loss(self):
        """settle = 4180: P&L = −1.50 + 0 = −1.50  (max loss = −debit)."""
        result = self._run_close(4180.0)
        actual = result.close_pnl
        print(f"\n[HAND P&L] settle=4180  expected=−1.50  actual={actual:+.2f}")
        self.assertAlmostEqual(actual, -1.50, places=10)
        self.assertEqual(result.close_zone, "max_loss")

    def test_pnl_at_long_strike_boundary(self):
        """settle = 4190: P&L = −1.50 + 0 = −1.50  (max loss at boundary)."""
        result = self._run_close(4190.0)
        actual = result.close_pnl
        print(f"\n[HAND P&L] settle=4190  expected=−1.50  actual={actual:+.2f}")
        self.assertAlmostEqual(actual, -1.50, places=10)
        self.assertEqual(result.close_zone, "max_loss")

    def test_pnl_between_strikes_partial(self):
        """settle = 4195: P&L = −1.50 + 5 = +3.50  (partial profit)."""
        result = self._run_close(4195.0)
        actual = result.close_pnl
        print(f"\n[HAND P&L] settle=4195  expected=+3.50  actual={actual:+.2f}")
        self.assertAlmostEqual(actual, 3.50, places=10)
        self.assertEqual(result.close_zone, "partial_loss")

    def test_pnl_at_short_strike_max_profit(self):
        """settle = 4200 (target reached): P&L = −1.50 + 10 = +8.50  (max profit)."""
        result = self._run_close(4200.0)
        actual = result.close_pnl
        print(f"\n[HAND P&L] settle=4200  expected=+8.50  actual={actual:+.2f}")
        self.assertAlmostEqual(actual, 8.50, places=10)
        self.assertEqual(result.close_zone, "max_profit")

    def test_pnl_above_short_strike_bounded(self):
        """settle = 4210: P&L = −1.50 + 10 = +8.50  (capped at max profit)."""
        result = self._run_close(4210.0)
        actual = result.close_pnl
        print(f"\n[HAND P&L] settle=4210  expected=+8.50  actual={actual:+.2f}")
        self.assertAlmostEqual(actual, 8.50, places=10)
        self.assertEqual(result.close_zone, "max_profit")


# ── Touch-exit P&L (debit: touch = WIN) ────────────────────────────────────

class TestDebitTouchExitPnL(unittest.TestCase):
    """Debit touch-exit: harness computes touch-exit P&L (touch = WIN).

    At the touch minute (ES >= 4200), the spread should trade near its max value.
    Set quotes at touch to: long 4190C = 11.00, short 4200C = 1.00
        pos_val_at_touch = +11.00 − 1.00 = +10.00
        touch_exit_pnl   = −1.50 + 10.00 = +8.50  (close to max profit)
    """

    TOUCH_LONG_MID  = 11.00
    TOUCH_SHORT_MID = 1.00
    # pos_val = +11 − 1 = +10; touch_exit_pnl = −1.50 + 10 = +8.50

    def _run(self):
        entry_day  = [(_ts(13, 31), _entry_qmap())]
        touch_ts   = _ts(14, 0)
        touch_qmap = {(LONG_K, "C"): self.TOUCH_LONG_MID,
                      (SHORT_K, "C"): self.TOUCH_SHORT_MID}
        touch_window = [(_ts(14, 0), touch_qmap)]
        return _harness().run_trade(_trade(), entry_day, touch_ts, touch_window, None)

    def test_touch_exit_pnl_is_computed(self):
        """touch_exit_pnl is not None for debit (touch = WIN)."""
        result = self._run()
        self.assertIsNotNone(result.touch_exit_pnl)

    def test_touch_exit_pnl_value(self):
        """touch_exit_pnl = net_credit + pos_val_at_touch = −1.50 + 10.00 = +8.50."""
        result = self._run()
        actual = result.touch_exit_pnl
        print(f"\n[TOUCH EXIT] expected=+8.50  actual={actual:+.2f}")
        self.assertAlmostEqual(actual, 8.50, places=10)

    def test_touch_exit_minute_recorded(self):
        result = self._run()
        self.assertEqual(result.touch_exit_minute, _ts(14, 0))


# ── is_touch ─────────────────────────────────────────────────────────────────

class TestDebitIsTouch(unittest.TestCase):
    """is_touch is the same underlying event as credit: ES >= drift_target."""

    def setUp(self):
        self.plugin = DebitVerticalPlugin()

    def test_touch_at_target(self):
        self.assertTrue(self.plugin.is_touch(4200.0, 4200.0))

    def test_touch_above_target(self):
        self.assertTrue(self.plugin.is_touch(4200.0, 4205.0))

    def test_no_touch_below_target(self):
        self.assertFalse(self.plugin.is_touch(4200.0, 4199.99))


# ── net_price ─────────────────────────────────────────────────────────────────

class TestDebitNetPrice(unittest.TestCase):
    """net_price returns positive for a debit spread (paid net cash)."""

    def setUp(self):
        self.plugin = DebitVerticalPlugin()

    def test_debit_positive_position_value(self):
        """pos_val = +1*3.00 + -1*1.50 = +1.50 (positive = paid net debit)."""
        qmap = {(LONG_K, "C"): LONG_MID, (SHORT_K, "C"): SHORT_MID}
        result = self.plugin.net_price(_LEGS, qmap)
        self.assertAlmostEqual(result, 1.50)

    def test_net_credit_is_negative(self):
        """net_credit = −pos_val = −1.50 for debit spread."""
        qmap = {(LONG_K, "C"): LONG_MID, (SHORT_K, "C"): SHORT_MID}
        pos_val = self.plugin.net_price(_LEGS, qmap)
        net_credit = -pos_val
        self.assertAlmostEqual(net_credit, -1.50)

    def test_returns_none_when_leg_missing(self):
        qmap = {(LONG_K, "C"): LONG_MID}
        self.assertIsNone(self.plugin.net_price(_LEGS, qmap))


# ── close_zone ───────────────────────────────────────────────────────────────

class TestDebitCloseZone(unittest.TestCase):
    """Debit close_zone: MIRRORED from credit (decision #5).

    max_profit  → settle ≥ upper (target reached)
    partial_loss → between
    max_loss    → settle ≤ lower (both OTM; lost the debit)
    """

    def setUp(self):
        self.plugin = DebitVerticalPlugin()

    def test_max_loss_below_long(self):
        self.assertEqual(self.plugin.close_zone(_LEGS, 4180.0), "max_loss")

    def test_max_loss_at_long(self):
        self.assertEqual(self.plugin.close_zone(_LEGS, 4190.0), "max_loss")

    def test_partial_between(self):
        self.assertEqual(self.plugin.close_zone(_LEGS, 4195.0), "partial_loss")

    def test_max_profit_at_short(self):
        self.assertEqual(self.plugin.close_zone(_LEGS, 4200.0), "max_profit")

    def test_max_profit_above_short(self):
        self.assertEqual(self.plugin.close_zone(_LEGS, 4210.0), "max_profit")


# ── Edge formula for debit ───────────────────────────────────────────────────

class TestDebitEdgeFormula(unittest.TestCase):
    """Edge = structural_prob − |net_credit| / spread_width.

    For debit, net_credit < 0; abs() normalises: edge = structural_prob − debit/width.
    """

    def test_edge_debit_structural_minus_implied(self):
        """entry: long 4190C @ 3.00, short 4200C @ 1.50 → debit=1.50, width=10
        edge = 0.60 − 1.50/10 = 0.45
        """
        harness = BacktestHarness(
            plugin=DebitVerticalPlugin(),
            edge_threshold=0.0,
            split_date=_SPLIT,
        )
        entry_day = [(_ts(13, 31), _entry_qmap())]
        result = harness.run_trade(_trade(structural_prob=0.60), entry_day, None, [], None)
        row = result.entry_scan[0]
        # pos_val = +1.50; net_credit = -1.50; abs = 1.50; edge = 0.60 - 0.15 = 0.45
        print(f"\n[EDGE] expected=0.45  actual={row.edge:+.4f}  net_credit={row.net_credit:+.2f}")
        self.assertAlmostEqual(row.net_credit, -1.50, places=10)
        self.assertAlmostEqual(row.edge, 0.45, places=10)


if __name__ == "__main__":
    unittest.main(verbosity=2)
