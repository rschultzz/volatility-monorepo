"""Unit tests for packages/shared/outcomes.py and packages/shared/buckets.py (CR-B).

All tests use synthetic bar DataFrames — no DB required.

Run with:
    python -m unittest packages.shared.tests.test_outcomes
"""
from __future__ import annotations

import unittest
from datetime import date, timedelta

import pandas as pd

from packages.shared.buckets import bucket_sessions
from packages.shared.outcomes import compute_outcome

# ─── helpers ─────────────────────────────────────────────────────────────────

_MONDAY = date(2023, 6, 5)  # known Monday for deterministic test dates
_EM     = 40.0              # expected_move used across most tests
_TOL    = 0.25 * _EM        # = 10.0


def _bars(rows: list[tuple], start: date = _MONDAY) -> pd.DataFrame:
    """Build a daily OHLC DataFrame from (open, high, low, close) tuples.

    Index = consecutive calendar dates starting from `start`; no weekend
    skipping (tests don't need a real trading calendar — compute_outcome only
    counts sequential rows, not calendar gaps).
    """
    idx = [start + timedelta(days=i) for i in range(len(rows))]
    return pd.DataFrame(rows, index=idx, columns=["open", "high", "low", "close"])


def _outcome(regime, drift_target=4300.0, dominant_bucket="1-7 DTE",
             expected_move=_EM, bars=None, trade_date=_MONDAY):
    """Call compute_outcome with sensible defaults so tests only specify deltas."""
    if bars is None:
        bars = _bars([
            (4100, 4150, 4080, 4130),
            (4130, 4200, 4110, 4180),
            (4180, 4310, 4160, 4290),
            (4290, 4320, 4270, 4300),
            (4300, 4320, 4280, 4295),
        ])
    return compute_outcome(trade_date, regime, drift_target, dominant_bucket,
                           expected_move, bars)


# ─── bucket_sessions ─────────────────────────────────────────────────────────

class TestBucketSessions(unittest.TestCase):

    def test_all_four_labels_return_correct_counts(self):
        cases = [
            ("0DTE",     1),
            ("1-7 DTE",  5),
            ("8-30 DTE", 20),
            ("30+ DTE",  60),
        ]
        for label, expected in cases:
            with self.subTest(label=label):
                self.assertEqual(bucket_sessions(label), expected)

    def test_unknown_label_raises_key_error(self):
        for bad in ["1-7", "0dte", "30+", "", "8-30DTE"]:
            with self.subTest(label=bad):
                with self.assertRaises(KeyError):
                    bucket_sessions(bad)


# ─── na_regime paths ─────────────────────────────────────────────────────────

class TestNaRegime(unittest.TestCase):

    def _assert_na_regime(self, regime: str):
        result = _outcome(regime)
        self.assertEqual(result["outcome_status"], "na_regime")
        for metric in ("reached_touch", "reached_close", "days_to_reach",
                       "max_excursion_in_direction", "final_close_distance_from_target",
                       "actual_realized_em_pct"):
            self.assertIsNone(result[metric], msg=f"{metric} should be None for na_regime")

    def test_bounded_is_na_regime(self):
        self._assert_na_regime("bounded")

    def test_amplification_is_na_regime(self):
        self._assert_na_regime("amplification")

    def test_untethered_is_na_regime(self):
        self._assert_na_regime("untethered")

    def test_empty_string_regime_is_na_regime(self):
        self._assert_na_regime("")

    def test_unknown_regime_is_na_regime(self):
        self._assert_na_regime("broken-magnet")

    def test_regime_kind_at_classification_preserved(self):
        result = _outcome("bounded")
        self.assertEqual(result["regime_kind_at_classification"], "bounded")


# ─── na_data paths (Lesson 2: skip-and-log, no silent fallback) ─────────────

class TestNaData(unittest.TestCase):

    def _assert_na_data(self, **kwargs):
        result = _outcome("magnet-above", **kwargs)
        self.assertEqual(result["outcome_status"], "na_data")
        for metric in ("reached_touch", "reached_close", "days_to_reach",
                       "max_excursion_in_direction", "final_close_distance_from_target",
                       "actual_realized_em_pct", "horizon_end_date"):
            self.assertIsNone(result[metric], msg=f"{metric} should be None for na_data")

    def test_drift_target_none(self):
        self._assert_na_data(drift_target=None)

    def test_expected_move_zero(self):
        self._assert_na_data(expected_move=0.0)

    def test_expected_move_negative(self):
        self._assert_na_data(expected_move=-5.0)

    def test_expected_move_none(self):
        self._assert_na_data(expected_move=None)

    def test_empty_bars_dataframe(self):
        self._assert_na_data(bars=pd.DataFrame(
            columns=["open", "high", "low", "close"]
        ))

    def test_bars_all_ohlc_nan(self):
        import math
        nan = float("nan")
        self._assert_na_data(bars=_bars([
            (nan, nan, nan, nan),
            (nan, nan, nan, nan),
        ]))

    def test_unknown_dominant_bucket_label(self):
        result = compute_outcome(
            _MONDAY, "magnet-above", 4300.0, "bad-label", _EM,
            _bars([(4100, 4500, 4080, 4300)] * 5),
        )
        self.assertEqual(result["outcome_status"], "na_data")

    def test_horizon_sessions_set_even_for_na_data_drift_target_none(self):
        # horizon_sessions should be set even when we return na_data,
        # so that diagnostic queries can aggregate by bucket.
        result = _outcome("magnet-above", drift_target=None, dominant_bucket="1-7 DTE")
        self.assertEqual(result["horizon_sessions"], 5)


# ─── pending_history path ─────────────────────────────────────────────────────

class TestPendingHistory(unittest.TestCase):

    def test_fewer_bars_than_horizon_returns_pending(self):
        # 1-7 DTE → 5 sessions required; provide only 3
        result = _outcome(
            "magnet-above",
            dominant_bucket="1-7 DTE",
            bars=_bars([(4100, 4150, 4080, 4130)] * 3),
        )
        self.assertEqual(result["outcome_status"], "pending_history")

    def test_pending_metrics_are_null(self):
        result = _outcome(
            "magnet-above",
            dominant_bucket="1-7 DTE",
            bars=_bars([(4100, 4150, 4080, 4130)] * 3),
        )
        for metric in ("reached_touch", "reached_close", "days_to_reach",
                       "max_excursion_in_direction", "final_close_distance_from_target",
                       "actual_realized_em_pct", "horizon_end_date"):
            self.assertIsNone(result[metric])

    def test_horizon_sessions_set_for_pending(self):
        result = _outcome(
            "magnet-above",
            dominant_bucket="30+ DTE",
            bars=_bars([(4100, 4150, 4080, 4130)] * 10),
        )
        self.assertEqual(result["outcome_status"], "pending_history")
        self.assertEqual(result["horizon_sessions"], 60)

    def test_exactly_n_minus_1_bars_is_pending(self):
        # 5 sessions required; 4 bars → pending
        result = _outcome(
            "magnet-above",
            dominant_bucket="1-7 DTE",
            bars=_bars([(4100, 4150, 4080, 4130)] * 4),
        )
        self.assertEqual(result["outcome_status"], "pending_history")

    def test_exactly_n_bars_is_computed(self):
        # 5 sessions required; 5 bars → computed
        result = _outcome(
            "magnet-above",
            drift_target=4500.0,  # above all highs → not reached
            dominant_bucket="1-7 DTE",
            bars=_bars([(4100, 4150, 4080, 4130)] * 5),
        )
        self.assertEqual(result["outcome_status"], "computed")


# ─── magnet-above happy paths ────────────────────────────────────────────────

class TestMagnetAbove(unittest.TestCase):

    def test_target_reached_on_third_session(self):
        # Bar 0: high=4150, Bar 1: high=4200, Bar 2: high=4310 >= 4300 → touch
        bars = _bars([
            (4100, 4150, 4080, 4130),  # idx 0 — no touch
            (4130, 4200, 4110, 4180),  # idx 1 — no touch
            (4180, 4310, 4160, 4290),  # idx 2 — TOUCH (first)
            (4290, 4320, 4270, 4300),  # idx 3
            (4300, 4320, 4280, 4295),  # idx 4 — horizon end
        ])
        result = compute_outcome(_MONDAY, "magnet-above", 4300.0, "1-7 DTE", _EM, bars)
        self.assertEqual(result["outcome_status"], "computed")
        self.assertTrue(result["reached_touch"])
        self.assertEqual(result["days_to_reach"], 2)  # 0-indexed: bar at position 2

    def test_target_not_reached(self):
        bars = _bars([(4100, 4150, 4080, 4130)] * 5)
        result = compute_outcome(_MONDAY, "magnet-above", 4300.0, "1-7 DTE", _EM, bars)
        self.assertEqual(result["outcome_status"], "computed")
        self.assertFalse(result["reached_touch"])
        self.assertIsNone(result["days_to_reach"])

    def test_reached_close_true_when_final_close_within_tolerance(self):
        # tolerance = 0.25 * 40 = 10; final close = 4295, |4295-4300| = 5 <= 10
        bars = _bars([
            (4100, 4150, 4080, 4130),
            (4130, 4200, 4110, 4180),
            (4180, 4310, 4160, 4290),
            (4290, 4320, 4270, 4300),
            (4300, 4320, 4280, 4295),  # close=4295, within tol
        ])
        result = compute_outcome(_MONDAY, "magnet-above", 4300.0, "1-7 DTE", _EM, bars)
        self.assertTrue(result["reached_close"])

    def test_reached_close_false_when_final_close_outside_tolerance(self):
        # final close = 4130, |4130-4300| = 170 >> 10
        bars = _bars([(4100, 4150, 4080, 4130)] * 5)
        result = compute_outcome(_MONDAY, "magnet-above", 4300.0, "1-7 DTE", _EM, bars)
        self.assertFalse(result["reached_close"])

    def test_max_excursion_is_max_high_minus_first_open(self):
        bars = _bars([
            (4100, 4150, 4080, 4130),
            (4130, 4320, 4110, 4300),  # highest high
            (4300, 4310, 4280, 4290),
            (4290, 4295, 4270, 4280),
            (4280, 4290, 4260, 4270),
        ])
        result = compute_outcome(_MONDAY, "magnet-above", 5000.0, "1-7 DTE", _EM, bars)
        # max_high = 4320, first_open = 4100 → excursion = 220.0
        self.assertAlmostEqual(result["max_excursion_in_direction"], 220.0)

    def test_final_close_distance_is_signed(self):
        # final close = 4295; drift_target = 4300 → distance = -5.0
        bars = _bars([
            (4100, 4150, 4080, 4130),
            (4130, 4200, 4110, 4180),
            (4180, 4310, 4160, 4290),
            (4290, 4320, 4270, 4300),
            (4300, 4320, 4280, 4295),
        ])
        result = compute_outcome(_MONDAY, "magnet-above", 4300.0, "1-7 DTE", _EM, bars)
        self.assertAlmostEqual(result["final_close_distance_from_target"], -5.0)

    def test_actual_realized_em_pct_is_range_over_expected_move(self):
        # max_high=4320, min_low=4080, EM=40 → (4320-4080)/40 = 6.0
        bars = _bars([
            (4100, 4150, 4080, 4130),
            (4130, 4200, 4110, 4180),
            (4180, 4310, 4160, 4290),
            (4290, 4320, 4270, 4300),
            (4300, 4320, 4280, 4295),
        ])
        result = compute_outcome(_MONDAY, "magnet-above", 4300.0, "1-7 DTE", _EM, bars)
        self.assertAlmostEqual(result["actual_realized_em_pct"], (4320 - 4080) / _EM)

    def test_horizon_end_date_is_last_session_in_horizon(self):
        bars = _bars([(4100, 4150, 4080, 4130)] * 5)
        result = compute_outcome(_MONDAY, "magnet-above", 5000.0, "1-7 DTE", _EM, bars)
        expected_end = _MONDAY + timedelta(days=4)
        self.assertEqual(result["horizon_end_date"], expected_end)

    def test_horizon_sessions_set_to_bucket_count(self):
        bars = _bars([(4100, 4150, 4080, 4130)] * 5)
        result = compute_outcome(_MONDAY, "magnet-above", 5000.0, "1-7 DTE", _EM, bars)
        self.assertEqual(result["horizon_sessions"], 5)


# ─── magnet-below happy paths (synthetic — no corpus data) ───────────────────

class TestMagnetBelow(unittest.TestCase):
    """magnet-below has 0 rows in v0.5.0-rebuilt corpus; tested synthetically."""

    def test_target_reached_on_second_session(self):
        # drift_target = 4050; Bar 0 low=4080 (no touch); Bar 1 low=4040 <= 4050 → touch
        bars = _bars([
            (4200, 4220, 4080, 4195),  # idx 0 — low=4080, no touch
            (4195, 4200, 4040, 4060),  # idx 1 — low=4040, TOUCH
            (4060, 4080, 4020, 4070),
            (4070, 4090, 4050, 4080),
            (4080, 4100, 4060, 4090),
        ])
        result = compute_outcome(_MONDAY, "magnet-below", 4050.0, "1-7 DTE", _EM, bars)
        self.assertEqual(result["outcome_status"], "computed")
        self.assertTrue(result["reached_touch"])
        self.assertEqual(result["days_to_reach"], 1)

    def test_target_not_reached(self):
        # drift_target = 3800; all lows > 3800
        bars = _bars([(4200, 4220, 4100, 4195)] * 5)
        result = compute_outcome(_MONDAY, "magnet-below", 3800.0, "1-7 DTE", _EM, bars)
        self.assertEqual(result["outcome_status"], "computed")
        self.assertFalse(result["reached_touch"])
        self.assertIsNone(result["days_to_reach"])

    def test_max_excursion_is_first_open_minus_min_low(self):
        # first_open=4200, min_low=4040 → excursion = 160.0
        bars = _bars([
            (4200, 4220, 4080, 4195),
            (4195, 4200, 4040, 4060),  # min low here
            (4060, 4080, 4055, 4070),
            (4070, 4090, 4060, 4080),
            (4080, 4100, 4070, 4090),
        ])
        result = compute_outcome(_MONDAY, "magnet-below", 3800.0, "1-7 DTE", _EM, bars)
        self.assertAlmostEqual(result["max_excursion_in_direction"], 4200 - 4040)


# ─── magnetic-pin happy paths ─────────────────────────────────────────────────

class TestMagneticPin(unittest.TestCase):
    # drift_target=4200, EM=40, tolerance=10 → band [4190, 4210]

    def test_reached_close_true_when_final_close_within_tolerance(self):
        # final close = 4195, |4195-4200| = 5 <= 10 → reached_close=True
        bars = _bars([
            (4250, 4260, 4230, 4245),  # above band, no touch
            (4245, 4255, 4225, 4240),
            (4240, 4250, 4220, 4235),
            (4235, 4245, 4215, 4220),
            (4220, 4230, 4185, 4195),  # range crosses band; close=4195 in tol
        ])
        result = compute_outcome(_MONDAY, "magnetic-pin", 4200.0, "1-7 DTE", _EM, bars)
        self.assertEqual(result["outcome_status"], "computed")
        self.assertTrue(result["reached_close"])

    def test_reached_close_false_when_final_close_outside_tolerance(self):
        # All bars well above band; final close = 4280
        bars = _bars([(4250, 4290, 4240, 4280)] * 5)
        result = compute_outcome(_MONDAY, "magnetic-pin", 4200.0, "1-7 DTE", _EM, bars)
        self.assertFalse(result["reached_close"])

    def test_reached_touch_true_when_bar_range_overlaps_band(self):
        # Bar with high=4195 >= 4190 AND low=4185 <= 4210 → overlaps [4190,4210]
        bars = _bars([
            (4250, 4260, 4230, 4245),
            (4245, 4255, 4225, 4240),
            (4240, 4250, 4220, 4235),
            (4235, 4215, 4185, 4195),  # high=4215>=4190, low=4185<=4210 → touch
            (4195, 4210, 4180, 4200),
        ])
        result = compute_outcome(_MONDAY, "magnetic-pin", 4200.0, "1-7 DTE", _EM, bars)
        self.assertTrue(result["reached_touch"])

    def test_reached_touch_false_when_bars_entirely_above_band(self):
        # All bars: low > 4210 → range never overlaps [4190, 4210]
        bars = _bars([(4250, 4290, 4220, 4260)] * 5)
        result = compute_outcome(_MONDAY, "magnetic-pin", 4200.0, "1-7 DTE", _EM, bars)
        self.assertFalse(result["reached_touch"])


# ─── edge cases ──────────────────────────────────────────────────────────────

class TestEdgeCases(unittest.TestCase):

    def test_max_excursion_clamped_to_zero_when_price_never_moved_favorably(self):
        # magnet-above: all highs below or equal to first bar's open
        # open=4200; high=4195 (below open) on all bars → excursion < 0 → clamped to 0
        bars = _bars([
            (4200, 4195, 4150, 4160),  # high < open
            (4160, 4155, 4110, 4120),
            (4120, 4115, 4070, 4080),
            (4080, 4075, 4030, 4040),
            (4040, 4035, 3990, 4000),
        ])
        result = compute_outcome(_MONDAY, "magnet-above", 5000.0, "1-7 DTE", _EM, bars)
        self.assertAlmostEqual(result["max_excursion_in_direction"], 0.0)

    def test_days_to_reach_zero_when_target_hit_on_trade_date(self):
        # Bar 0 (trade_date) high >= drift_target → days_to_reach = 0
        bars = _bars([
            (4100, 4350, 4080, 4300),  # idx 0: high=4350 >= 4300 → immediate touch
            (4300, 4320, 4280, 4310),
            (4310, 4315, 4290, 4305),
            (4305, 4310, 4295, 4300),
            (4300, 4310, 4290, 4295),
        ])
        result = compute_outcome(_MONDAY, "magnet-above", 4300.0, "1-7 DTE", _EM, bars)
        self.assertTrue(result["reached_touch"])
        self.assertEqual(result["days_to_reach"], 0)

    def test_actual_realized_em_pct_uses_max_high_minus_min_low(self):
        # Explicit formula verification: NOT (high - open) or (open - low)
        # max_high=4250 (bar 1), min_low=4080 (bar 0), EM=40 → 170/40 = 4.25
        bars = _bars([
            (4100, 4150, 4080, 4130),  # low=4080 is the min
            (4130, 4250, 4110, 4200),  # high=4250 is the max
            (4200, 4220, 4180, 4210),
            (4210, 4230, 4190, 4220),
            (4220, 4240, 4200, 4230),
        ])
        result = compute_outcome(_MONDAY, "magnet-above", 5000.0, "1-7 DTE", _EM, bars)
        expected = (4250 - 4080) / _EM  # = 170 / 40 = 4.25
        self.assertAlmostEqual(result["actual_realized_em_pct"], expected)

    def test_0dte_bucket_produces_1_session_horizon(self):
        bars = _bars([(4100, 4150, 4080, 4130)] * 3)
        result = compute_outcome(_MONDAY, "magnet-above", 5000.0, "0DTE", _EM, bars)
        self.assertEqual(result["horizon_sessions"], 1)
        # Only first bar used even though 3 bars provided
        self.assertEqual(result["horizon_end_date"], _MONDAY)

    def test_30plus_dte_bucket_produces_60_session_horizon(self):
        bars = _bars([(4100, 4150, 4080, 4130)] * 60)
        result = compute_outcome(_MONDAY, "magnet-above", 5000.0, "30+ DTE", _EM, bars)
        self.assertEqual(result["horizon_sessions"], 60)
        self.assertEqual(result["horizon_end_date"], _MONDAY + timedelta(days=59))

    def test_bars_before_trade_date_are_ignored(self):
        # Provide bars for 10 days but trade_date is partway in
        # — bars before trade_date must not count toward the horizon
        all_bars = _bars([(4100, 4150, 4080, 4130)] * 10)
        trade_date = _MONDAY + timedelta(days=3)  # 4th row
        # Only 7 rows remain from trade_date → enough for 5-session horizon
        result = compute_outcome(
            trade_date, "magnet-above", 5000.0, "1-7 DTE", _EM, all_bars
        )
        self.assertEqual(result["outcome_status"], "computed")
        self.assertEqual(result["horizon_sessions"], 5)

    def test_regime_and_bucket_recorded_in_all_status_paths(self):
        for status, kwargs in [
            ("na_regime",       {"regime": "bounded"}),
            ("na_data",         {"drift_target": None}),
            ("pending_history", {"bars": _bars([(4100, 4150, 4080, 4130)] * 2)}),
        ]:
            with self.subTest(status=status):
                result = _outcome("magnet-above" if "regime" not in kwargs else kwargs.pop("regime"),
                                  **kwargs)
                self.assertEqual(result["regime_kind_at_classification"],
                                 "bounded" if status == "na_regime" else "magnet-above")
                self.assertEqual(result["dominant_bucket_at_classification"], "1-7 DTE")


if __name__ == "__main__":
    unittest.main()
