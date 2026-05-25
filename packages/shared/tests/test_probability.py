"""Unit tests for packages/shared/stats.py and packages/shared/probability.py (CR-C).

All tests use synthetic data — no DB required.
_aggregate_outcomes and wilson_ci are pure functions; the DB-layer
(_rank_analogues_with_outcomes, compute_structural_probability) is exercised
in Step 2 smoke tests against the live endpoint.

Run with:
    python -m unittest packages.shared.tests.test_probability -v
"""
from __future__ import annotations

import unittest

from packages.shared.stats import wilson_ci
from packages.shared.probability import _aggregate_outcomes


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


if __name__ == "__main__":
    unittest.main()
