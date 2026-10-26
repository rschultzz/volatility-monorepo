"""Tests for packages/shared/skew_expected.py (CR-CJ).

Run with:
    python -m unittest packages.shared.tests.test_skew_expected -v

Bucket accounting, the minute-weighted beta, drift summation, the recommended flat put-skew curve (residual = raw
change), the ATM curve's sign, A6 divergence reads, domain checks and input validation.
"""
from __future__ import annotations

import sys
import unittest
from datetime import datetime, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from packages.shared import skew_expected as se  # noqa: E402


def params(beta_up, beta_down=None, theta=0.0):
    up = beta_up if isinstance(beta_up, tuple) else (beta_up,) * se.N_BUCKETS
    dn = up if beta_down is None else (beta_down if isinstance(beta_down, tuple) else (beta_down,) * se.N_BUCKETS)
    th = theta if isinstance(theta, tuple) else (theta,) * se.N_BUCKETS
    return se.CurveParams(beta_up=up, beta_down=dn, theta=th)


class TestBuckets(unittest.TestCase):
    def test_within_one_bucket(self):
        w = se.minutes_by_bucket(time(7, 5), time(7, 20))
        self.assertEqual(w[1], 15)
        self.assertEqual(sum(w), 15)

    def test_straddles_buckets(self):
        w = se.minutes_by_bucket(time(7, 50), time(8, 10))
        self.assertEqual((w[2], w[3], sum(w)), (10, 10, 20))     # 07:30-08:00 is bucket 2

    def test_clipped_to_rth(self):
        w = se.minutes_by_bucket(time(6, 0), time(6, 40))
        self.assertEqual((w[0], sum(w)), (10, 10))
        self.assertEqual(sum(se.minutes_by_bucket(time(12, 50), time(14, 0))), 10)

    def test_accepts_datetime_and_int(self):
        self.assertEqual(se.minutes_by_bucket(datetime(2026, 5, 1, 9, 0), 9 * 60 + 30)[5], 30)

    def test_reversed_raises(self):
        with self.assertRaises(ValueError):
            se.minutes_by_bucket(time(9, 0), time(8, 0))

    def test_param_length_checked(self):
        with self.assertRaises(ValueError):
            se.CurveParams(beta_up=(1.0,), beta_down=(1.0,), theta=(0.0,))


class TestExpected(unittest.TestCase):
    def test_beta_times_move(self):
        self.assertAlmostEqual(se.expected_change_pp(0.5, time(8, 0), time(8, 20), params(-2.0)), -1.0)

    def test_up_down_slopes(self):
        p = params(-4.0, 1.0)
        self.assertAlmostEqual(se.expected_change_pp(0.25, time(8, 0), time(8, 15), p), -1.0)
        self.assertAlmostEqual(se.expected_change_pp(-0.25, time(8, 0), time(8, 15), p), -0.25)

    def test_minute_weighted_beta(self):
        betas = tuple(float(i) for i in range(se.N_BUCKETS))       # bucket b has beta b
        p = params(betas)
        # 07:20-07:50: 10 min in bucket 1 (beta 1) and 20 min in bucket 2 (beta 2): mean beta 5/3
        self.assertAlmostEqual(se.expected_change_pp(0.3, time(7, 20), time(7, 50), p), 0.3 * 5.0 / 3.0)

    def test_drift_sums_per_minute(self):
        theta = tuple(0.01 * i for i in range(se.N_BUCKETS))
        p = params(0.0, theta=theta)
        self.assertAlmostEqual(se.expected_change_pp(0.0, time(7, 20), time(7, 50), p), 10 * 0.01 + 20 * 0.02)

    def test_empty_window(self):
        self.assertEqual(se.expected_change_pp(1.0, time(9, 0), time(9, 0), params(-3.0)), 0.0)


class TestRecommended(unittest.TestCase):
    def test_put_skew_residual_is_raw_change(self):
        r = se.put_skew_residual_pp(4.2, 5.0, 6800.0, 6830.0, time(8, 0), time(8, 25))
        self.assertAlmostEqual(r, 0.8)

    def test_atm_expects_vol_up_on_selloff(self):
        e = se.expected_change_pp(-1.0, time(9, 0), time(9, 30), se.ATM_PARAMS)
        self.assertGreater(e, 6.0)
        self.assertLess(se.expected_change_pp(1.0, time(9, 0), time(9, 30), se.ATM_PARAMS), -6.0)

    def test_atm_residual(self):
        # 1 % rally over 09:00-09:30: expected about -7 vol pts, so a flat ATM leaves a residual of about +7
        r = se.atm_residual_pp(20.0, 20.0, 6800.0, 6868.0, time(9, 0), time(9, 30))
        exp = se.expected_change_pp(1.0, time(9, 0), time(9, 30), se.ATM_PARAMS)
        self.assertAlmostEqual(r, -exp)

    def test_lad_reference_differs_from_recommended(self):
        self.assertNotEqual(se.PUT_SKEW_LAD_A2_PARAMS, se.PUT_SKEW_PARAMS)


class TestReadsAndDomain(unittest.TestCase):
    def test_divergence(self):
        self.assertEqual(se.divergence_read(0.4, 10.0), "divergent")      # price up, put skew up vs expected
        self.assertEqual(se.divergence_read(-0.4, -10.0), "divergent")    # price down, put skew down
        self.assertEqual(se.divergence_read(-0.4, 10.0), "confirming")
        self.assertEqual(se.divergence_read(0.0, 10.0), "none")
        self.assertEqual(se.divergence_read(0.4, 0.0), "none")
        self.assertEqual(se.divergence_read(float("nan"), 5.0), "none")

    def test_domain(self):
        self.assertTrue(se.in_domain(time(7, 0), time(12, 45)))
        self.assertFalse(se.in_domain(time(6, 45), time(7, 30)))
        self.assertFalse(se.in_domain(time(12, 30), time(12, 50)))

    def test_es_move_pct(self):
        self.assertAlmostEqual(se.es_move_pct(6800.0, 6868.0), 1.0)


if __name__ == "__main__":
    unittest.main()
