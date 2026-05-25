"""Unit tests for packages/shared/vol_features.py (CR-D).

All tests use synthetic DataFrames — no DB required.

Run with:
    python -m unittest packages.shared.tests.test_vol_features
"""
from __future__ import annotations

import math
import unittest
from datetime import date, timedelta

import pandas as pd

from packages.shared.vol_features import (
    compute_atm_iv_percentile,
    compute_realized_vol_20d,
    compute_skew_percentile,
    compute_smile_convexity,
    compute_term_structure_slope,
    compute_vol_risk_premium,
)

# ─── helpers ─────────────────────────────────────────────────────────────────

_TODAY = date(2024, 6, 3)   # arbitrary anchor; Monday


def _iv_hist(n_prior: int, prior_val: float, current_val: float) -> pd.DataFrame:
    """Build [trade_date, atm_iv] with n_prior prior rows + 1 current row."""
    prior_dates = [_TODAY - timedelta(days=n_prior - i) for i in range(n_prior)]
    dates = prior_dates + [_TODAY]
    vals = [prior_val] * n_prior + [current_val]
    return pd.DataFrame({'trade_date': dates, 'atm_iv': vals})


def _skew_hist(n_prior: int, prior_val: float, current_val: float) -> pd.DataFrame:
    prior_dates = [_TODAY - timedelta(days=n_prior - i) for i in range(n_prior)]
    dates = prior_dates + [_TODAY]
    vals = [prior_val] * n_prior + [current_val]
    return pd.DataFrame({'trade_date': dates, 'raw_skew': vals})


def _conv_hist(n_prior: int, prior_val: float, current_val: float) -> pd.DataFrame:
    prior_dates = [_TODAY - timedelta(days=n_prior - i) for i in range(n_prior)]
    dates = prior_dates + [_TODAY]
    vals = [prior_val] * n_prior + [current_val]
    return pd.DataFrame({'trade_date': dates, 'convexity': vals})


def _slope_hist(n_prior: int, val: float) -> pd.DataFrame:
    """Prior-only slope history (no current day row)."""
    prior_dates = [_TODAY - timedelta(days=n_prior - i) for i in range(n_prior)]
    return pd.DataFrame({'trade_date': prior_dates, 'slope': [val] * n_prior})


# ─── TestAtmIvPercentile ─────────────────────────────────────────────────────

class TestAtmIvPercentile(unittest.TestCase):

    def test_constant_history_gives_50(self):
        # All 60 prior values = 0.20; current = 0.20 → percentile=50 (kind='mean')
        hist = _iv_hist(60, 0.20, 0.20)
        result = compute_atm_iv_percentile(_TODAY, hist)
        self.assertAlmostEqual(result, 50.0, places=5)

    def test_current_above_all_history_gives_100(self):
        # All prior values = 0.20; current = 0.25 (above every prior) → 100.0
        hist = _iv_hist(60, 0.20, 0.25)
        result = compute_atm_iv_percentile(_TODAY, hist)
        self.assertAlmostEqual(result, 100.0, places=5)

    def test_current_below_all_history_gives_0(self):
        # All prior values = 0.20; current = 0.15 (below every prior) → 0.0
        hist = _iv_hist(60, 0.20, 0.15)
        result = compute_atm_iv_percentile(_TODAY, hist)
        self.assertAlmostEqual(result, 0.0, places=5)

    def test_rising_iv_later_dates_get_higher_percentile(self):
        # Build an ascending IV series; a higher current value scores higher
        n = 60
        prior_vals = [0.10 + i * 0.001 for i in range(n)]  # 0.10 → 0.169
        prior_dates = [_TODAY - timedelta(days=n - i) for i in range(n)]

        # Low current: 0.05 (below all prior) → near 0
        df_low = pd.DataFrame({
            'trade_date': prior_dates + [_TODAY],
            'atm_iv': prior_vals + [0.05],
        })
        pct_low = compute_atm_iv_percentile(_TODAY, df_low)

        # High current: 0.25 (above all prior) → near 100
        df_high = pd.DataFrame({
            'trade_date': prior_dates + [_TODAY],
            'atm_iv': prior_vals + [0.25],
        })
        pct_high = compute_atm_iv_percentile(_TODAY, df_high)

        self.assertIsNotNone(pct_low)
        self.assertIsNotNone(pct_high)
        self.assertLess(pct_low, pct_high)

    def test_insufficient_history_returns_none(self):
        # Only 59 prior rows (< 60) → None
        hist = _iv_hist(59, 0.20, 0.20)
        self.assertIsNone(compute_atm_iv_percentile(_TODAY, hist))

    def test_zero_prior_rows_returns_none(self):
        hist = _iv_hist(0, 0.20, 0.20)
        self.assertIsNone(compute_atm_iv_percentile(_TODAY, hist))

    def test_exactly_60_prior_rows_computes(self):
        hist = _iv_hist(60, 0.20, 0.20)
        result = compute_atm_iv_percentile(_TODAY, hist)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 100.0)


# ─── TestSkewPercentile ──────────────────────────────────────────────────────

class TestSkewPercentile(unittest.TestCase):

    def test_constant_skew_gives_50(self):
        hist = _skew_hist(60, 0.05, 0.05)
        result = compute_skew_percentile(_TODAY, hist)
        self.assertAlmostEqual(result, 50.0, places=5)

    def test_insufficient_history_returns_none(self):
        hist = _skew_hist(59, 0.05, 0.05)
        self.assertIsNone(compute_skew_percentile(_TODAY, hist))

    def test_negative_skew_handled(self):
        # Negative skew (vol25 > vol75) is unusual but must not raise
        hist = _skew_hist(60, -0.01, -0.01)
        result = compute_skew_percentile(_TODAY, hist)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 50.0, places=5)

    def test_current_above_all_history_gives_100(self):
        hist = _skew_hist(60, 0.03, 0.10)
        result = compute_skew_percentile(_TODAY, hist)
        self.assertAlmostEqual(result, 100.0, places=5)

    def test_current_below_all_history_gives_0(self):
        hist = _skew_hist(60, 0.05, 0.001)
        result = compute_skew_percentile(_TODAY, hist)
        self.assertAlmostEqual(result, 0.0, places=5)


# ─── TestTermStructureSlope ──────────────────────────────────────────────────

class TestTermStructureSlope(unittest.TestCase):

    def test_backwardation_gives_positive_slope(self):
        # front > back → positive raw_slope
        hist = _slope_hist(60, 0.01)
        raw, pct = compute_term_structure_slope(_TODAY, 0.20, 0.18, hist)
        self.assertAlmostEqual(raw, 0.02, places=6)
        self.assertIsNotNone(pct)

    def test_contango_gives_negative_slope(self):
        # front < back → negative raw_slope
        hist = _slope_hist(60, -0.01)
        raw, pct = compute_term_structure_slope(_TODAY, 0.15, 0.18, hist)
        self.assertAlmostEqual(raw, -0.03, places=6)
        self.assertIsNotNone(pct)

    def test_missing_front_iv_returns_none_none(self):
        hist = _slope_hist(60, 0.01)
        raw, pct = compute_term_structure_slope(_TODAY, None, 0.18, hist)
        self.assertIsNone(raw)
        self.assertIsNone(pct)

    def test_missing_back_iv_returns_none_none(self):
        hist = _slope_hist(60, 0.01)
        raw, pct = compute_term_structure_slope(_TODAY, 0.20, None, hist)
        self.assertIsNone(raw)
        self.assertIsNone(pct)

    def test_insufficient_history_returns_none_none(self):
        hist = _slope_hist(59, 0.01)
        raw, pct = compute_term_structure_slope(_TODAY, 0.20, 0.18, hist)
        self.assertIsNone(raw)
        self.assertIsNone(pct)

    def test_constant_slope_history_gives_percentile_50(self):
        # Use 0.50 - 0.25 = 0.25 (exact in binary float; both powers of 2).
        # All prior slopes = 0.25; current slope = 0.25 → percentile=50
        hist = _slope_hist(60, 0.25)
        raw, pct = compute_term_structure_slope(_TODAY, 0.50, 0.25, hist)
        self.assertAlmostEqual(raw, 0.25, places=6)
        self.assertAlmostEqual(pct, 50.0, places=5)

    def test_percentile_in_valid_range(self):
        hist = _slope_hist(60, 0.00)
        raw, pct = compute_term_structure_slope(_TODAY, 0.20, 0.18, hist)
        self.assertIsNotNone(pct)
        self.assertGreaterEqual(pct, 0.0)
        self.assertLessEqual(pct, 100.0)


# ─── TestSmileConvexity ──────────────────────────────────────────────────────

class TestSmileConvexity(unittest.TestCase):

    def test_constant_convexity_gives_50(self):
        hist = _conv_hist(60, 0.005, 0.005)
        result = compute_smile_convexity(_TODAY, hist)
        self.assertAlmostEqual(result, 50.0, places=5)

    def test_flat_smile_zero_convexity_handled(self):
        # convexity = 0.0 (wing IV exactly equals ATM average)
        hist = _conv_hist(60, 0.005, 0.0)
        result = compute_smile_convexity(_TODAY, hist)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 0.0, places=5)

    def test_insufficient_history_returns_none(self):
        hist = _conv_hist(59, 0.005, 0.005)
        self.assertIsNone(compute_smile_convexity(_TODAY, hist))


# ─── TestVolRiskPremium ──────────────────────────────────────────────────────

class TestVolRiskPremium(unittest.TestCase):

    def test_realized_greater_than_implied_gives_positive_vrp(self):
        result = compute_vol_risk_premium(_TODAY, 0.20, 0.15)
        self.assertAlmostEqual(result, 0.05, places=6)

    def test_realized_less_than_implied_gives_negative_vrp(self):
        result = compute_vol_risk_premium(_TODAY, 0.10, 0.18)
        self.assertAlmostEqual(result, -0.08, places=6)

    def test_realized_equals_implied_gives_zero_vrp(self):
        result = compute_vol_risk_premium(_TODAY, 0.15, 0.15)
        self.assertAlmostEqual(result, 0.0, places=6)

    def test_realized_none_returns_none(self):
        self.assertIsNone(compute_vol_risk_premium(_TODAY, None, 0.15))

    def test_current_atm_iv_none_returns_none(self):
        self.assertIsNone(compute_vol_risk_premium(_TODAY, 0.20, None))

    def test_current_atm_iv_zero_returns_none(self):
        # Defensive guard: atm_iv=0 is invalid (would likely be a data error)
        self.assertIsNone(compute_vol_risk_premium(_TODAY, 0.20, 0.0))

    def test_current_atm_iv_negative_returns_none(self):
        self.assertIsNone(compute_vol_risk_premium(_TODAY, 0.20, -0.01))


# ─── TestComputeRealizedVol ──────────────────────────────────────────────────

class TestComputeRealizedVol(unittest.TestCase):

    def test_constant_closes_gives_zero_realized_vol(self):
        # All prices equal → all log-returns = 0 → std = 0 → realized vol = 0
        closes = [100.0] * 21
        result = compute_realized_vol_20d(closes)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result, 0.0, places=6)

    def test_fewer_than_21_closes_returns_none(self):
        self.assertIsNone(compute_realized_vol_20d([100.0] * 20))
        self.assertIsNone(compute_realized_vol_20d([]))
        self.assertIsNone(compute_realized_vol_20d([100.0] * 10))

    def test_exactly_21_closes_computes(self):
        closes = [100.0 + i * 0.5 for i in range(21)]
        result = compute_realized_vol_20d(closes)
        self.assertIsNotNone(result)
        self.assertGreater(result, 0.0)

    def test_annualization_factor(self):
        # 10 pairs of alternating +/-0.01 log-returns → 20 returns total.
        # Sample std (ddof=1) of alternating ±x series: x * sqrt(n/(n-1))
        # = 0.01 * sqrt(20/19). Annualized = that * sqrt(252).
        alt_rets = [0.01 if i % 2 == 0 else -0.01 for i in range(20)]
        alt_closes = [100.0]
        for r in alt_rets:
            alt_closes.append(alt_closes[-1] * math.exp(r))
        result = compute_realized_vol_20d(alt_closes)
        self.assertIsNotNone(result)
        expected = 0.01 * math.sqrt(20 / 19) * math.sqrt(252)
        self.assertAlmostEqual(result, expected, places=6)

    def test_uses_last_21_closes(self):
        # Provide 25 closes; first 4 are outliers; result must use only last 21
        outliers = [1.0, 1.0, 1.0, 1.0]   # would produce huge returns if included
        normal = [100.0 + i * 0.1 for i in range(21)]
        closes = outliers + normal
        result_full = compute_realized_vol_20d(closes)
        result_normal = compute_realized_vol_20d(normal)
        self.assertAlmostEqual(result_full, result_normal, places=6)


if __name__ == '__main__':
    unittest.main()
