"""Unit tests for packages/shared/stats.py and packages/shared/probability.py (CR-C + CR-I).

All tests use synthetic data — no DB required.
_aggregate_outcomes, classify_post_touch_positions, and aggregate_post_touch_distribution
are pure functions; the DB-layer (_rank_analogues_with_outcomes,
compute_structural_probability) is exercised in Step 2 smoke tests against the
live endpoint.

Run with:
    python -m unittest packages.shared.tests.test_probability -v
"""
from __future__ import annotations

import unittest

from packages.shared.stats import wilson_ci
from packages.shared.probability import (
    _aggregate_outcomes,
    classify_post_touch_positions,
    aggregate_post_touch_distribution,
)


# ─── helpers ─────────────────────────────────────────────────────────────────

def _row(
    outcome_status="computed",
    reached_touch=True,
    reached_close=False,
    days_to_reach=1,
    max_excursion=50.0,
    implied_move_1d=30.0,
    distance=1.5,
    trade_date="2023-01-02",
):
    """Build a synthetic outcome row with sensible defaults."""
    return {
        "trade_date":                 trade_date,
        "distance":                   distance,
        "outcome_status":             outcome_status,
        "reached_touch":              reached_touch,
        "reached_close":              reached_close,
        "days_to_reach":              days_to_reach,
        "max_excursion_in_direction": max_excursion,
        "implied_move_1d":            implied_move_1d,
    }


def _computed(**kw):
    """Convenience wrapper — always outcome_status='computed'."""
    return _row(outcome_status="computed", **kw)


class _FakeBars:
    """Minimal iloc-supporting sequence for classify_post_touch_positions tests.

    Mimics the DataFrame interface used by the function:
      len(bars)           → number of sessions
      bars.iloc[i]        → dict with 'close' key
      bars.iloc[i]['close'] → float
    """

    def __init__(self, closes):
        self._rows = [{"close": c} for c in closes]

    def __len__(self):
        return len(self._rows)

    @property
    def iloc(self):
        return self._rows


def _touch_row(
    bucket="8-30 DTE",
    t1=1,
    t5=1,
    t15=1,
    outcome_status="computed",
    reached_touch=True,
):
    """Build a synthetic analogue row for aggregate_post_touch_distribution tests.

    All position_tN values default to +1 (above). Pass None to simulate a
    missing timeframe (pre-backfill or horizon truncation).
    """
    return {
        "outcome_status":                    outcome_status,
        "reached_touch":                     reached_touch,
        "dominant_bucket_at_classification": bucket,
        "position_t1_post_touch":            t1,
        "position_t5_post_touch":            t5,
        "position_t15_post_touch":           t15,
    }


# ─── wilson_ci ───────────────────────────────────────────────────────────────

class TestWilsonCI(unittest.TestCase):

    def test_n_zero_returns_none_none(self):
        lower, upper = wilson_ci(0, 0)
        self.assertIsNone(lower)
        self.assertIsNone(upper)

    def test_symmetric_near_half(self):
        # 50/100: symmetric around 0.50, approximately (0.404, 0.596)
        lower, upper = wilson_ci(50, 100)
        self.assertAlmostEqual(lower, 0.404, delta=0.01)
        self.assertAlmostEqual(upper, 0.596, delta=0.01)

    def test_spec_example_12_of_20(self):
        # Spec example: 12/20 → approximately (0.387, 0.781)
        lower, upper = wilson_ci(12, 20)
        self.assertAlmostEqual(lower, 0.387, delta=0.01)
        self.assertAlmostEqual(upper, 0.781, delta=0.01)

    def test_rare_event_1_of_100(self):
        lower, upper = wilson_ci(1, 100)
        self.assertGreater(lower, 0.0)
        self.assertLess(lower, 0.01)
        self.assertLess(upper, 0.055)

    def test_near_certain_99_of_100(self):
        lower, upper = wilson_ci(99, 100)
        self.assertGreater(lower, 0.945)
        self.assertLess(upper, 1.0)

    def test_zero_successes_n_positive(self):
        # 0/20: lower must be 0.0, upper must be a small positive value
        lower, upper = wilson_ci(0, 20)
        self.assertEqual(lower, 0.0)
        self.assertGreater(upper, 0.0)
        self.assertLess(upper, 0.20)

    def test_bounds_always_in_unit_interval(self):
        for s, n in [(0, 1), (1, 1), (0, 50), (50, 50), (7, 10)]:
            lower, upper = wilson_ci(s, n)
            self.assertGreaterEqual(lower, 0.0, f"lower < 0 for {s}/{n}")
            self.assertLessEqual(upper,   1.0,  f"upper > 1 for {s}/{n}")
            self.assertLessEqual(lower,   upper, f"lower > upper for {s}/{n}")


# ─── _aggregate_outcomes ─────────────────────────────────────────────────────

class TestAggregateOutcomes(unittest.TestCase):

    def test_happy_path_15_of_20_touched(self):
        rows = (
            [_computed(reached_touch=True,  reached_close=True,  days_to_reach=1,
                       distance=i * 0.1) for i in range(1, 16)]
            + [_computed(reached_touch=False, reached_close=False, days_to_reach=None,
                         distance=i * 0.1) for i in range(16, 21)]
        )
        r = _aggregate_outcomes(rows)
        self.assertEqual(r["outcome_status"], "ok")
        self.assertEqual(r["k_with_outcomes"], 20)
        self.assertAlmostEqual(r["touch_rate"], 0.75, places=4)
        self.assertAlmostEqual(r["close_rate"], 0.75, places=4)
        self.assertIsNotNone(r["touch_ci_lower"])
        self.assertIsNotNone(r["touch_ci_upper"])
        self.assertLess(r["touch_ci_lower"], 0.75)
        self.assertGreater(r["touch_ci_upper"], 0.75)
        self.assertAlmostEqual(r["mean_days_to_reach"], 1.0, places=2)

    def test_all_na_regime_returns_no_data(self):
        rows = [_row(outcome_status="na_regime", reached_touch=None,
                     reached_close=None, days_to_reach=None) for _ in range(20)]
        r = _aggregate_outcomes(rows)
        self.assertEqual(r["outcome_status"], "no_data")
        self.assertEqual(r["k_with_outcomes"], 0)
        self.assertIsNone(r["touch_rate"])
        self.assertIsNone(r["touch_ci_lower"])
        self.assertIn("na_regime", r["note"])

    def test_mixed_statuses_filters_to_computed(self):
        rows = (
            [_computed(reached_touch=True) for _ in range(10)]
            + [_row(outcome_status="na_regime",       reached_touch=None,
                    reached_close=None, days_to_reach=None) for _ in range(5)]
            + [_row(outcome_status="pending_history", reached_touch=None,
                    reached_close=None, days_to_reach=None) for _ in range(3)]
            + [_row(outcome_status="na_data",         reached_touch=None,
                    reached_close=None, days_to_reach=None) for _ in range(2)]
        )
        r = _aggregate_outcomes(rows)
        self.assertEqual(r["outcome_status"], "ok")
        self.assertEqual(r["k_with_outcomes"], 10)
        self.assertAlmostEqual(r["touch_rate"], 1.0, places=4)
        self.assertIn("na_regime",       r["note"])
        self.assertIn("pending_history", r["note"])
        self.assertIn("na_data",         r["note"])

    def test_no_touches_mean_days_is_none(self):
        rows = [_computed(reached_touch=False, reached_close=False,
                          days_to_reach=None) for _ in range(20)]
        r = _aggregate_outcomes(rows)
        self.assertEqual(r["outcome_status"], "ok")
        self.assertAlmostEqual(r["touch_rate"], 0.0, places=4)
        self.assertIsNone(r["mean_days_to_reach"])

    def test_zero_implied_move_excluded_from_excursion(self):
        rows = (
            [_computed(max_excursion=60.0, implied_move_1d=30.0) for _ in range(10)]
            + [_computed(max_excursion=60.0, implied_move_1d=0.0) for _ in range(10)]
        )
        r = _aggregate_outcomes(rows)
        self.assertEqual(r["outcome_status"], "ok")
        # Only the 10 valid rows contribute; ratio = 60/30 = 2.0
        self.assertAlmostEqual(r["mean_excursion_pct"], 2.0, places=4)
        self.assertIn("excluded from excursion mean", r["note"])

    def test_all_invalid_implied_move_excursion_is_none(self):
        rows = [_computed(max_excursion=50.0, implied_move_1d=0.0) for _ in range(10)]
        r = _aggregate_outcomes(rows)
        self.assertEqual(r["outcome_status"], "ok")
        self.assertIsNone(r["mean_excursion_pct"])

    def test_k_equals_1_produces_valid_aggregate(self):
        rows = [_computed(reached_touch=True, days_to_reach=2,
                          max_excursion=40.0, implied_move_1d=25.0, distance=0.8)]
        r = _aggregate_outcomes(rows)
        self.assertEqual(r["outcome_status"], "ok")
        self.assertEqual(r["k_with_outcomes"], 1)
        self.assertAlmostEqual(r["touch_rate"], 1.0, places=4)
        self.assertIsNotNone(r["touch_ci_lower"])
        self.assertIsNotNone(r["touch_ci_upper"])
        self.assertAlmostEqual(r["mean_excursion_pct"], 40.0 / 25.0, places=4)

    def test_distance_range_in_note(self):
        rows = [
            _computed(distance=0.5),
            _computed(distance=1.0),
            _computed(distance=2.5),
        ]
        r = _aggregate_outcomes(rows)
        self.assertIn("Distance range", r["note"])
        self.assertIn("0.50σ", r["note"])
        self.assertIn("2.50σ", r["note"])

    def test_mean_days_uses_only_touched_rows(self):
        rows = [
            _computed(reached_touch=True,  days_to_reach=0),
            _computed(reached_touch=True,  days_to_reach=2),
            _computed(reached_touch=True,  days_to_reach=4),
            _computed(reached_touch=False, days_to_reach=None),
            _computed(reached_touch=False, days_to_reach=None),
        ]
        r = _aggregate_outcomes(rows)
        # Mean of [0, 2, 4] = 2.0; untouched rows must not contribute
        self.assertAlmostEqual(r["mean_days_to_reach"], 2.0, places=2)

    def test_none_implied_move_excluded_from_excursion(self):
        rows = (
            [_computed(max_excursion=30.0, implied_move_1d=20.0) for _ in range(5)]
            + [_computed(max_excursion=30.0, implied_move_1d=None) for _ in range(5)]
        )
        r = _aggregate_outcomes(rows)
        # Only the 5 valid rows contribute; ratio = 30/20 = 1.5
        self.assertAlmostEqual(r["mean_excursion_pct"], 1.5, places=4)

    def test_empty_rows_returns_no_data(self):
        r = _aggregate_outcomes([])
        self.assertEqual(r["outcome_status"], "no_data")
        self.assertEqual(r["k_with_outcomes"], 0)


# ─── classify_post_touch_positions ──────────────────────────────────────────


class TestClassifyPostTouchPositions(unittest.TestCase):

    def test_above_below_at_basic(self):
        """Above / at / below are classified correctly at distinct closes."""
        # drift=100, tolerance=5 → band [95, 105]
        # days_to_reach=0 → T+1 at idx 1, T+2 at idx 2, T+3 at idx 3
        bars = _FakeBars([
            100,   # idx 0: touch session
            110,   # idx 1: T+1 → above (110 > 105)
            100,   # idx 2: T+2 → at   (abs(100-100)=0 ≤ 5)
            90,    # idx 3: T+3 → below (90 < 95)
        ])
        r = classify_post_touch_positions(
            days_to_reach=0,
            horizon_bars=bars,
            drift_target=100.0,
            tolerance=5.0,
            timeframes_sessions=(1, 2, 3),
        )
        self.assertEqual(r[1],  1)
        self.assertEqual(r[2],  0)
        self.assertEqual(r[3], -1)

    def test_exact_boundary_counts_as_at(self):
        """Close exactly at drift_target ± tolerance → inclusive, classified as 'at'."""
        bars = _FakeBars([100, 105, 95])   # idx 0 = touch; tf=1 → 105; tf=2 → 95
        r = classify_post_touch_positions(
            days_to_reach=0,
            horizon_bars=bars,
            drift_target=100.0,
            tolerance=5.0,
            timeframes_sessions=(1, 2),
        )
        self.assertEqual(r[1], 0)   # 105 = upper bound → "at"
        self.assertEqual(r[2], 0)   # 95  = lower bound → "at"

    def test_out_of_bounds_returns_none(self):
        """Bar beyond sequence length returns None without raising."""
        bars = _FakeBars([100, 110])   # indices 0, 1 only
        r = classify_post_touch_positions(
            days_to_reach=0,
            horizon_bars=bars,
            drift_target=100.0,
            tolerance=5.0,
            timeframes_sessions=(1, 5, 15),
        )
        self.assertEqual(r[1],  1)       # idx 1 = 110 > 105 → above
        self.assertIsNone(r[5])          # idx 5 out of range
        self.assertIsNone(r[15])         # idx 15 out of range

    def test_days_to_reach_offset_applied(self):
        """T+N is at iloc[days_to_reach + N], not iloc[N]."""
        # 10 bars; days_to_reach=3 → T+1 at idx 4, T+5 at idx 8
        closes = [100, 100, 100, 100, 50, 100, 100, 100, 200, 100]
        bars = _FakeBars(closes)
        r = classify_post_touch_positions(
            days_to_reach=3,
            horizon_bars=bars,
            drift_target=100.0,
            tolerance=5.0,
            timeframes_sessions=(1, 5),
        )
        self.assertEqual(r[1], -1)   # idx 4 = 50 < 95 → below
        self.assertEqual(r[5],  1)   # idx 8 = 200 > 105 → above


# ─── aggregate_post_touch_distribution ──────────────────────────────────────
#
# Each test maps to one of the 12 Step-1 test cases specified in the CR-I
# kickoff prompt.  Tests 1-6 exercise the pattern-label decision tree;
# tests 7-12 exercise bucket filtering, pool fallback, and CI properties.


class TestAggregatePostTouchDistribution(unittest.TestCase):

    # ── Pattern label decision tree ──────────────────────────────────────────

    def test_01_stepping_stone_all_above(self):
        """All touchers close above tolerance at T+1/T+5/T+15 → stepping-stone."""
        rows = [_touch_row(t1=1, t5=1, t15=1) for _ in range(10)]
        r = aggregate_post_touch_distribution(rows, "8-30 DTE")
        self.assertEqual(r["filter_mode"], "strict")
        self.assertEqual(r["pattern_label"], "stepping-stone")
        self.assertAlmostEqual(r["fractions"]["t1"]["above"], 1.0, places=4)
        self.assertAlmostEqual(r["fractions"]["t5"]["above"], 1.0, places=4)
        self.assertAlmostEqual(r["fractions"]["t15"]["above"], 1.0, places=4)

    def test_02_touch_and_pin_all_at(self):
        """All touchers close within tolerance at every timeframe → touch-and-pin."""
        rows = [_touch_row(t1=0, t5=0, t15=0) for _ in range(10)]
        r = aggregate_post_touch_distribution(rows, "8-30 DTE")
        self.assertEqual(r["pattern_label"], "touch-and-pin")
        self.assertAlmostEqual(r["fractions"]["t1"]["at"], 1.0, places=4)

    def test_03_touch_and_reject_all_below(self):
        """All touchers close below tolerance at every timeframe → touch-and-reject."""
        rows = [_touch_row(t1=-1, t5=-1, t15=-1) for _ in range(10)]
        r = aggregate_post_touch_distribution(rows, "8-30 DTE")
        self.assertEqual(r["pattern_label"], "touch-and-reject")
        self.assertAlmostEqual(r["fractions"]["t15"]["below"], 1.0, places=4)

    def test_04_overshoot_then_revert(self):
        """Above at T+1 and T+5; below at T+15 → overshoot-then-revert."""
        rows = [_touch_row(t1=1, t5=1, t15=-1) for _ in range(10)]
        r = aggregate_post_touch_distribution(rows, "8-30 DTE")
        self.assertEqual(r["pattern_label"], "overshoot-then-revert")

    def test_05_slow_revert_monotone_decrease(self):
        """above_t1=0.80, above_t5=0.50, above_t15=0.20 → slow-revert.

        Monotonically decreasing above-fraction, above_t1 > 0.50, and
        above_t5 does NOT exceed 0.50 (strict >) so overshoot branch is skipped.
        """
        # 10 same-bucket rows constructed so counts are exactly 8/5/2 above.
        rows = (
            [_touch_row(t1=-1, t5=-1, t15=-1) for _ in range(2)]
            + [_touch_row(t1=1,  t5=-1, t15=-1) for _ in range(3)]
            + [_touch_row(t1=1,  t5=1,  t15=-1) for _ in range(3)]
            + [_touch_row(t1=1,  t5=1,  t15=1)  for _ in range(2)]
        )
        r = aggregate_post_touch_distribution(rows, "8-30 DTE")
        self.assertEqual(r["pattern_label"], "slow-revert")
        self.assertAlmostEqual(r["fractions"]["t1"]["above"],  0.8, places=3)
        self.assertAlmostEqual(r["fractions"]["t5"]["above"],  0.5, places=3)
        self.assertAlmostEqual(r["fractions"]["t15"]["above"], 0.2, places=3)

    def test_06_mixed_equal_three_way_split(self):
        """Equal 1/3 split at all timeframes → no dominant signal → mixed."""
        rows = (
            [_touch_row(t1=1,  t5=1,  t15=1)  for _ in range(4)]
            + [_touch_row(t1=0,  t5=0,  t15=0)  for _ in range(4)]
            + [_touch_row(t1=-1, t5=-1, t15=-1) for _ in range(4)]
        )
        r = aggregate_post_touch_distribution(rows, "8-30 DTE")
        self.assertEqual(r["pattern_label"], "mixed")
        # All three fractions equal (0.333…); at is NOT strictly largest
        self.assertAlmostEqual(r["fractions"]["t1"]["above"], 1 / 3, places=3)
        self.assertAlmostEqual(r["fractions"]["t1"]["at"],    1 / 3, places=3)
        self.assertAlmostEqual(r["fractions"]["t1"]["below"], 1 / 3, places=3)

    # ── Bucket filtering ─────────────────────────────────────────────────────

    def test_07_strict_filter_triggers(self):
        """12 same-bucket of 17 total touchers → strict mode; denominator = 12."""
        rows = (
            [_touch_row(bucket="8-30 DTE", t1=1, t5=1, t15=1) for _ in range(12)]
            + [_touch_row(bucket="1-7 DTE", t1=1, t5=1, t15=1) for _ in range(5)]
        )
        r = aggregate_post_touch_distribution(rows, "8-30 DTE")
        self.assertEqual(r["filter_mode"],    "strict")
        self.assertEqual(r["same_bucket_n"],  12)
        self.assertEqual(r["total_touchers"], 17)
        self.assertEqual(r["denominator_t1"], 12)   # pool = same-bucket only
        self.assertEqual(r["denominator_t5"], 12)
        self.assertEqual(r["denominator_t15"], 12)

    def test_08_pooled_fallback_triggers(self):
        """3 same-bucket of 8 total touchers → pooled-fallback; denominator = 8."""
        rows = (
            [_touch_row(bucket="8-30 DTE", t1=1, t5=1, t15=1) for _ in range(3)]
            + [_touch_row(bucket="1-7 DTE", t1=1, t5=1, t15=1) for _ in range(5)]
        )
        r = aggregate_post_touch_distribution(rows, "8-30 DTE")
        self.assertEqual(r["filter_mode"],    "pooled-fallback")
        self.assertEqual(r["same_bucket_n"],  3)
        self.assertEqual(r["total_touchers"], 8)
        self.assertEqual(r["denominator_t1"], 8)   # pool = all touchers
        self.assertIsNotNone(r["fractions"])
        self.assertIsNotNone(r["pattern_label"])

    def test_09_insufficient_sample(self):
        """3 total touchers < pooled_minimum=4 → insufficient; no aggregation."""
        rows = [_touch_row(bucket="8-30 DTE") for _ in range(3)]
        r = aggregate_post_touch_distribution(rows, "8-30 DTE")
        self.assertEqual(r["filter_mode"],    "insufficient")
        self.assertEqual(r["total_touchers"], 3)
        self.assertIsNone(r["fractions"])
        self.assertIsNone(r["wilson_cis"])
        self.assertIsNone(r["pattern_label"])
        self.assertEqual(r["denominator_t1"], 0)

    def test_10_zero_dte_pre_check(self):
        """anchor_bucket='0DTE' → immediate zero_dte_corpus_insufficient return."""
        rows = [_touch_row(bucket="0DTE", t1=1, t5=1, t15=1) for _ in range(20)]
        r = aggregate_post_touch_distribution(rows, "0DTE")
        self.assertEqual(r["filter_mode"],    "zero_dte_corpus_insufficient")
        self.assertEqual(r["total_touchers"], 20)   # counts are still computed
        self.assertEqual(r["denominator_t1"], 0)
        self.assertIsNone(r["fractions"])
        self.assertIsNone(r["pattern_label"])

    def test_11_null_handling_per_timeframe(self):
        """17 touchers where 4 lack T+15 position → denominator_t15=13, t1/t5=17."""
        rows = (
            [_touch_row(t1=1, t5=1, t15=1)    for _ in range(13)]
            + [_touch_row(t1=1, t5=1, t15=None) for _ in range(4)]
        )
        r = aggregate_post_touch_distribution(rows, "8-30 DTE")
        self.assertEqual(r["denominator_t1"],  17)
        self.assertEqual(r["denominator_t5"],  17)
        self.assertEqual(r["denominator_t15"], 13)
        # All available positions are +1 → pattern still computable
        self.assertIsNotNone(r["pattern_label"])
        self.assertEqual(r["pattern_label"], "stepping-stone")

    def test_12_wilson_ci_width_vs_sample_size(self):
        """Wilson CIs narrow at large N and wide at small N.

        Large-N case: N=50 pool with 40/5/5 below/at/above split at T+1 →
        CI for 'below' should be tight (width < 0.25).

        Small-N case: 47 touchers have t1=None, leaving 3 for T+1 denominator.
        CI for 'below' at N=3 should have lower bound well below 1.0 (< 0.75).
        """
        # ── Large N (50) ─────────────────────────────────────────────────────
        rows_50 = (
            [_touch_row(t1=-1, t5=-1, t15=-1) for _ in range(40)]   # below
            + [_touch_row(t1=0,  t5=-1, t15=-1) for _ in range(5)]   # at
            + [_touch_row(t1=1,  t5=-1, t15=-1) for _ in range(5)]   # above
        )
        r50 = aggregate_post_touch_distribution(rows_50, "8-30 DTE")
        self.assertEqual(r50["denominator_t1"], 50)
        ci_below_50 = r50["wilson_cis"]["t1"]["below"]   # 40/50 = 0.80
        self.assertIsNotNone(ci_below_50[0])
        ci_width_50 = ci_below_50[1] - ci_below_50[0]
        self.assertGreater(ci_width_50, 0.0)     # non-degenerate
        self.assertLess(ci_width_50, 0.25)       # tight at N=50

        # ── Small N (3 via NULL handling) ────────────────────────────────────
        rows_3 = (
            [_touch_row(t1=-1, t5=-1, t15=-1)   for _ in range(3)]    # 3 with t1
            + [_touch_row(t1=None, t5=-1, t15=-1) for _ in range(47)]  # 47 missing t1
        )
        r3 = aggregate_post_touch_distribution(rows_3, "8-30 DTE")
        self.assertEqual(r3["denominator_t1"], 3)
        ci_below_3 = r3["wilson_cis"]["t1"]["below"]   # 3/3
        self.assertIsNotNone(ci_below_3[0])
        self.assertLess(ci_below_3[0], 0.75)   # lower bound well below 1.0 at N=3


if __name__ == "__main__":
    unittest.main()
