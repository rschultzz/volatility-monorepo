"""Verification tests — hand-computed proof of sign convention and crawl invariant.

All expected values are derived below from first principles; the test compares
the harness's actual output to these values without rounding or tolerance games
(results are exact multiples of 0.50 — no floating-point ambiguity).

Spread under test (all four tests):
    SHORT 4200 call, LONG 4210 call
    Entry mids: c1=3.00 (short), c2=1.50 (long)
    net_pos_val = -3.00 + 1.50 = -1.50   (negative = credit received)
    entry_credit = -pos_val = +1.50
    spread_width = 4210 - 4200 = 10

Units are raw ES points, no ×100 multiplier applied by the harness.
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
from packages.shared.backtest.plugins.vertical import VerticalPlugin
from packages.shared.backtest.plugins.debit_vertical import DebitVerticalPlugin


# ── Shared spread fixture ────────────────────────────────────────────────────

SHORT_K = 4200.0
LONG_K  = 4210.0
WIDTH   = LONG_K - SHORT_K          # 10.0

ENTRY_C1 = 3.00   # short strike mid
ENTRY_C2 = 1.50   # long  strike mid
# net_pos_val = -3.00 + 1.50 = -1.50 → entry_credit = 1.50

_LEGS = [
    Leg(side="short", type="call", strike=SHORT_K),
    Leg(side="long",  type="call", strike=LONG_K),
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
        plugin=VerticalPlugin(),
        edge_threshold=0.0,   # fill on first quoted minute, always
        split_date=_SPLIT,
    )


def _debit_harness():
    """Harness with DebitVerticalPlugin — for touch-exit reconciliation tests."""
    return BacktestHarness(
        plugin=DebitVerticalPlugin(),
        edge_threshold=0.0,
        split_date=_SPLIT,
    )


def _entry_qmap(c1=ENTRY_C1, c2=ENTRY_C2):
    return {(SHORT_K, "C"): c1, (LONG_K, "C"): c2}


def _ts(h, m):
    return datetime(2025, 6, 1, h, m, 0)


# ── TEST A: Credit-spread P&L hand-check ────────────────────────────────────

class TestA_CreditSpreadPnL(unittest.TestCase):
    """Hand-computed P&L for all three close zones. No × multiplier — raw ES points.

    entry_credit = 1.50, width = 10

    Zone 1 (max profit):  S ≤ 4200  → payoff = 0           → pnl = 1.50
    Zone 2 (max loss):    S ≥ 4210  → payoff = −10          → pnl = 1.50 − 10 = −8.50
    Zone 3 (partial):     S = 4205  → payoff = −(4205−4200) = −5 → pnl = 1.50 − 5 = −3.50
    """

    def _run_close(self, settlement):
        entry_day = [(_ts(13, 31), _entry_qmap())]
        return _harness().run_trade(_trade(), entry_day, None, [], settlement)

    def test_A1_below_short_strike_max_profit(self):
        """S = 4190 (below short strike 4200) → close_pnl = +1.50."""
        result = self._run_close(4190.0)
        # Formula: entry_credit + payoff = 1.50 + 0 = 1.50
        actual = result.close_pnl
        print(f"\n[TEST A1] S=4190  expected=+1.50  actual={actual:+.2f}")
        self.assertAlmostEqual(actual, 1.50, places=10,
            msg=f"Max-profit zone failed. Got {actual}, expected 1.50")
        self.assertEqual(result.close_zone, "max_profit")

    def test_A2_above_long_strike_max_loss(self):
        """S = 4220 (above long strike 4210) → close_pnl = −8.50."""
        result = self._run_close(4220.0)
        # Formula: entry_credit + payoff = 1.50 + (−10) = −8.50
        actual = result.close_pnl
        print(f"\n[TEST A2] S=4220  expected=-8.50  actual={actual:+.2f}")
        self.assertAlmostEqual(actual, -8.50, places=10,
            msg=f"Max-loss zone failed. Got {actual}, expected -8.50")
        self.assertEqual(result.close_zone, "max_loss")

    def test_A3_between_strikes_partial_loss(self):
        """S = 4205 (between 4200 and 4210) → close_pnl = −3.50."""
        result = self._run_close(4205.0)
        # Formula: entry_credit + payoff = 1.50 + (−(4205−4200)) = 1.50 − 5 = −3.50
        actual = result.close_pnl
        print(f"\n[TEST A3] S=4205  expected=-3.50  actual={actual:+.2f}")
        self.assertAlmostEqual(actual, -3.50, places=10,
            msg=f"Partial-loss zone failed. Got {actual}, expected -3.50")
        self.assertEqual(result.close_zone, "partial_loss")


# ── TEST B: Boundary cases at exact strike prices ────────────────────────────

class TestB_StrikeBoundaries(unittest.TestCase):
    """Payoff and close_pnl at exactly the short and long strikes.

    At S = 4200 (short strike, lower boundary):
        Short call intrinsic: max(4200−4200, 0) = 0  (exactly OTM / ATM → 0)
        Long  call intrinsic: max(4200−4210, 0) = 0
        payoff = 0 → close_pnl = 1.50
        zone  = 'max_profit' (≤ low threshold, inequality is ≤)

    At S = 4210 (long strike, upper boundary):
        Short call intrinsic: max(4210−4200, 0) = 10
        Long  call intrinsic: max(4210−4210, 0) = 0
        payoff = −10 + 0 = −10 → close_pnl = −8.50
        zone  = 'max_loss' (≥ high threshold, inequality is ≥)
    """

    def _run_close(self, settlement):
        entry_day = [(_ts(13, 31), _entry_qmap())]
        return _harness().run_trade(_trade(), entry_day, None, [], settlement)

    def test_B1_exactly_at_short_strike(self):
        """S = 4200 (exactly at short strike) → payoff = 0, close_pnl = +1.50."""
        result = self._run_close(4200.0)
        actual_pnl = result.close_pnl
        actual_zone = result.close_zone
        print(f"\n[TEST B1] S=4200 (short strike)  pnl={actual_pnl:+.2f}  zone={actual_zone}")
        self.assertAlmostEqual(actual_pnl, 1.50, places=10,
            msg=f"At short strike: expected close_pnl=1.50, got {actual_pnl}")
        self.assertEqual(actual_zone, "max_profit",
            msg=f"At short strike: expected zone=max_profit, got {actual_zone}")

    def test_B2_exactly_at_long_strike(self):
        """S = 4210 (exactly at long strike) → payoff = −10, close_pnl = −8.50."""
        result = self._run_close(4210.0)
        actual_pnl = result.close_pnl
        actual_zone = result.close_zone
        print(f"\n[TEST B2] S=4210 (long strike)   pnl={actual_pnl:+.2f}  zone={actual_zone}")
        self.assertAlmostEqual(actual_pnl, -8.50, places=10,
            msg=f"At long strike: expected close_pnl=-8.50, got {actual_pnl}")
        self.assertEqual(actual_zone, "max_loss",
            msg=f"At long strike: expected zone=max_loss, got {actual_zone}")

    def test_B3_continuity_just_inside_boundaries(self):
        """One tick inside each boundary confirms no jump discontinuity."""
        r_just_below = self._run_close(4199.99)
        r_just_above  = self._run_close(4210.01)
        print(f"\n[TEST B3] S=4199.99 → pnl={r_just_below.close_pnl:+.4f}  zone={r_just_below.close_zone}")
        print(f"[TEST B3] S=4210.01 → pnl={r_just_above.close_pnl:+.4f}  zone={r_just_above.close_zone}")
        self.assertEqual(r_just_below.close_zone, "max_profit")
        self.assertEqual(r_just_above.close_zone, "max_loss")
        self.assertAlmostEqual(r_just_below.close_pnl, 1.50, places=10)
        self.assertAlmostEqual(r_just_above.close_pnl, -8.50, places=1)


# ── TEST C: Touch-path vs close-path reconciliation ─────────────────────────

class TestC_TouchCloseReconciliation(unittest.TestCase):
    """Confirms touch_exit_pnl and close_pnl use a consistent basis.

    Uses DebitVerticalPlugin (touch_exit_is_meaningful=True, decision #6).
    The credit plugin (VerticalPlugin) has no touch-exit P&L by design.

    Set up a trade where the option market at the touch minute prices the spread
    at EXACTLY its intrinsic value at S=4205 (between strikes):

        Intrinsic at S=4205:
            Short 4200 call intrinsic = 4205 − 4200 = 5.00  (cost to holder)
            Long  4210 call intrinsic = max(4205−4210, 0) = 0.00
            payoff = −5 + 0 = −5.00

        Touch qmap: set c1_touch=6.00, c2_touch=1.00 so that
            pos_val_at_touch = −6.00 + 1.00 = −5.00   (matches intrinsic)

        touch_exit_pnl = entry_credit + pos_val_at_touch
                       = 1.50 + (−5.00) = −3.50

        close_pnl (S=4205) = entry_credit + payoff(legs, 4205)
                            = 1.50 + (−5.00) = −3.50

    Both must equal −3.50. If they differ, the two second terms use a different
    basis (one being position-value from-holder, the other being payoff at expiry)
    and the sign chain is broken somewhere.

    Note: The spread here is short 4200 / long 4210 (credit orientation), but
    DebitVerticalPlugin is used to exercise the touch-exit harness path. The
    per-leg payoff formula is structure-agnostic; the reconciliation holds for
    any spread where option market prices reflect intrinsic value at the moment.
    """

    TOUCH_C1 = 6.00   # short 4200 mid at touch — makes pos_val = intrinsic at S=4205
    TOUCH_C2 = 1.00   # long  4210 mid at touch

    # Expected derived values (hand-computed):
    EXPECTED_ENTRY_CREDIT  = 1.50    # = 3.00 − 1.50
    EXPECTED_POS_VAL_TOUCH = -5.00   # = −6.00 + 1.00
    EXPECTED_TOUCH_EXIT    = -3.50   # = 1.50 + (−5.00)
    EXPECTED_CLOSE         = -3.50   # = 1.50 + payoff(4205) = 1.50 + (−5.00)

    def _run(self):
        entry_day = [(_ts(13, 31), _entry_qmap())]
        touch_ts  = _ts(14, 0)
        touch_window = [(_ts(14, 0), {(SHORT_K, "C"): self.TOUCH_C1,
                                       (LONG_K,  "C"): self.TOUCH_C2})]
        return _debit_harness().run_trade(_trade(), entry_day, touch_ts, touch_window, 4205.0)

    def test_C1_touch_exit_equals_expected(self):
        result = self._run()
        actual = result.touch_exit_pnl
        print(f"\n[TEST C1] touch_exit_pnl  expected={self.EXPECTED_TOUCH_EXIT:+.2f}  actual={actual:+.2f}")
        self.assertAlmostEqual(actual, self.EXPECTED_TOUCH_EXIT, places=10,
            msg=f"touch_exit_pnl mismatch: expected {self.EXPECTED_TOUCH_EXIT}, got {actual}")

    def test_C2_close_pnl_equals_expected(self):
        result = self._run()
        actual = result.close_pnl
        print(f"\n[TEST C2] close_pnl       expected={self.EXPECTED_CLOSE:+.2f}  actual={actual:+.2f}")
        self.assertAlmostEqual(actual, self.EXPECTED_CLOSE, places=10,
            msg=f"close_pnl mismatch: expected {self.EXPECTED_CLOSE}, got {actual}")

    def test_C3_both_reconcile(self):
        """The key assertion: both paths give the same number at the shared point."""
        result = self._run()
        print(f"\n[TEST C3] reconciliation: touch_exit_pnl={result.touch_exit_pnl:+.2f}  "
              f"close_pnl={result.close_pnl:+.2f}  delta={result.touch_exit_pnl - result.close_pnl:+.2f}")
        self.assertAlmostEqual(result.touch_exit_pnl, result.close_pnl, places=10,
            msg=f"Two P&L paths disagree: touch={result.touch_exit_pnl}, close={result.close_pnl}. "
                f"Check pos_val sign vs payoff sign.")


# ── TEST D: Crawl-level missing-leg skip ─────────────────────────────────────

class TestD_CrawlMissingLegSkip(unittest.TestCase):
    """Invariant #1 at crawl level: absent leg at one minute does not bleed into neighbours.

    Minute layout (all 09:4x):
        09:41 — both legs present (c1=3.00, c2=1.50) → net_price = −1.50
        09:42 — SHORT 4200 call absent, LONG 4210 call present (c2=1.50 only)
        09:43 — both legs present (c1=2.50, c2=1.20) → net_price = −1.30

    Assertions:
        1. entry_scan has 3 rows (all minutes recorded, invariant #3).
        2. 09:42 row has net_position_value = None, net_credit = None, edge = None.
        3. 09:42 does NOT use c2=1.50 from the adjacent minute paired with a missing c1.
        4. 09:41 and 09:43 have correct, independent net_position_value values.
    """

    def _build_scan(self):
        return [
            (_ts(9, 41), {(SHORT_K, "C"): 3.00, (LONG_K, "C"): 1.50}),  # both present
            (_ts(9, 42), {(LONG_K, "C"): 1.50}),                          # short absent
            (_ts(9, 43), {(SHORT_K, "C"): 2.50, (LONG_K, "C"): 1.20}),  # both present
        ]

    def _run(self):
        return _harness().run_trade(_trade(), self._build_scan(), None, [], None)

    def test_D1_all_three_minutes_in_scan(self):
        """All 3 minutes present in entry_scan (invariant #3 cross-check)."""
        result = self._run()
        print(f"\n[TEST D1] entry_scan length: {len(result.entry_scan)} (expected 3)")
        self.assertEqual(len(result.entry_scan), 3)

    def test_D2_missing_leg_minute_has_none(self):
        """09:42 (absent short leg) → net_position_value=None, net_credit=None, edge=None."""
        result = self._run()
        row_42 = result.entry_scan[1]
        print(f"\n[TEST D2] 09:42 row: net_position_value={row_42.net_position_value}  "
              f"net_credit={row_42.net_credit}  edge={row_42.edge}")
        self.assertEqual(row_42.snapshot_pt, _ts(9, 42))
        self.assertIsNone(row_42.net_position_value,
            msg="09:42 should be None (absent short leg), not a fabricated price")
        self.assertIsNone(row_42.net_credit)
        self.assertIsNone(row_42.edge)

    def test_D3_adjacent_minutes_unaffected(self):
        """09:41 and 09:43 have independent, correct net_position_value values."""
        result = self._run()
        row_41 = result.entry_scan[0]
        row_43 = result.entry_scan[2]
        # 09:41: pos_val = −3.00 + 1.50 = −1.50
        # 09:43: pos_val = −2.50 + 1.20 = −1.30
        print(f"\n[TEST D3] 09:41 net_position_value={row_41.net_position_value} (expected −1.50)")
        print(f"[TEST D3] 09:43 net_position_value={row_43.net_position_value} (expected −1.30)")
        self.assertAlmostEqual(row_41.net_position_value, -1.50, places=10,
            msg="09:41 pos_val should be −1.50 (not contaminated by 09:42)")
        self.assertAlmostEqual(row_43.net_position_value, -1.30, places=10,
            msg="09:43 pos_val should be −1.30 (not contaminated by 09:42)")

    def test_D4_fill_lands_on_first_complete_minute_not_gap(self):
        """Fill lands at 09:41 (first complete minute), not on the absent-leg minute."""
        result = self._run()
        print(f"\n[TEST D4] fill_time={result.fill_time}  (expected 09:41, not 09:42)")
        self.assertTrue(result.filled)
        self.assertEqual(result.fill_time, _ts(9, 41),
            msg=f"Fill should be at 09:41. Got {result.fill_time}.")
        # Confirm fill credit = 1.50 (from 09:41, not a None or 09:43 value)
        self.assertAlmostEqual(result.fill_net_credit, 1.50, places=10)

    def test_D5_no_cross_minute_quote_pairing(self):
        """The absent-leg minute (09:42) has None — no stale/carried-forward quote used."""
        result = self._run()
        row_42 = result.entry_scan[1]
        # If the harness accidentally looked at 09:41's c1=3.00 with 09:42's c2=1.50,
        # it would produce pos_val=−1.50. We assert this did NOT happen.
        self.assertIsNone(row_42.net_position_value,
            msg="Cross-minute pairing detected: 09:42 has a value but short leg was absent. "
                "The harness must not carry forward quotes from adjacent minutes.")


if __name__ == "__main__":
    # Run with -v to see the print statements
    unittest.main(verbosity=2)
