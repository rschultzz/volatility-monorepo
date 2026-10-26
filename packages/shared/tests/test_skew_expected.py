"""Tests for packages/shared/skew_expected.py (CR-CJ; stickiness functions CR-CK).

Run with:
    python -m unittest packages.shared.tests.test_skew_expected -v

Bucket accounting, the minute-weighted beta, drift summation, the recommended flat put-skew curve (residual = raw
change), the ATM curve's sign, A6 divergence reads, domain checks and input validation.
"""
from __future__ import annotations

import math
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


# ---------------------------------------------------------------------------------------------- CR-CK stickiness
def smile(atm=0.20, slope=-0.6, curv=0.0, T=None):
    """A smile {call-delta bucket: vol} built from sigma(k) = atm + slope * k + curv * k^2 on the module's own k map."""
    T = se.years_to_close_0dte(time(8, 0)) if T is None else T
    out = {}
    for n in range(5, 100, 5):
        if n == 50:
            k = 0.0
        else:
            p, is_put = se._abs_delta_is_put(n)
            k = se.k_for_abs_delta(p, is_put=is_put, sigma=atm, T=T)
        out[n] = atm + slope * k + curv * k * k
    return out, T


class TestStickySmile(unittest.TestCase):
    def test_flat_smile_no_change(self):
        v, T = smile(slope=0.0)
        self.assertAlmostEqual(se.sticky_strike_put_skew_change_pp(v, T, 0.5), 0.0, places=10)

    def test_linear_smile_no_change(self):
        # a straight line in k: put skew = slope * k_p25 at any shift, so sticky strike moves it by nothing
        v, T = smile(slope=-0.8)
        self.assertAlmostEqual(se.sticky_strike_put_skew_change_pp(v, T, 0.5), 0.0, places=8)

    def test_zero_move_zero_change(self):
        v, T = smile(slope=-0.8, curv=40.0)
        self.assertAlmostEqual(se.sticky_strike_put_skew_change_pp(v, T, 0.0), 0.0, places=12)

    def test_convex_smile_rally_flattens_put_skew(self):
        # convex smile (curvature > 0): a rally reads the old smile further right, where it is flatter, so the new
        # 25-delta put sits less far above the new ATM -> put skew falls, and the reverse on a sell-off
        v, T = smile(slope=-0.8, curv=40.0)
        up = se.sticky_strike_put_skew_change_pp(v, T, 0.5)
        dn = se.sticky_strike_put_skew_change_pp(v, T, -0.5)
        self.assertLess(up, 0.0)
        self.assertGreater(dn, 0.0)

    def test_matches_hand_quadratic(self):
        # inside the grid, interpolation of a quadratic is close to the quadratic: change = -2 * curv * k_p25 * shift
        v, T = smile(slope=-0.8, curv=40.0)
        k_p25 = se.k_for_abs_delta(0.25, is_put=True, sigma=0.20, T=T)
        shift = math.log1p(0.003)
        want = 2.0 * 40.0 * k_p25 * shift * 100.0               # sigma(kp+x) - sigma(x) - (sigma(kp) - sigma(0)) = 2c kp x
        self.assertAlmostEqual(se.sticky_strike_put_skew_change_pp(v, T, 0.3), want, delta=abs(want) * 0.1)

    def test_grid_matches_skew_module_rule(self):
        v, T = smile(slope=-0.8, curv=40.0)
        k, s = se.smile_k_grid(v, T)
        self.assertEqual(len(k), 19)
        self.assertTrue(all(b > a for a, b in zip(k, k[1:])))
        self.assertEqual(k[9], 0.0)                              # vol50 at k = 0
        self.assertAlmostEqual(se.interp_linear_extrap(0.0, k, s), v[50])
        self.assertAlmostEqual(se.interp_linear_extrap(k[0] - 0.01, k, s),
                               s[0] - (s[1] - s[0]) * 0.01 / (k[1] - k[0]))

    def test_needs_atm(self):
        v, T = smile()
        del v[50]
        with self.assertRaises(ValueError):
            se.sticky_strike_put_skew_change_pp(v, T, 0.3)


class TestTrend(unittest.TestCase):
    def test_exact_line(self):
        ts = list(range(450, 471))
        ys = [1.0 + 0.05 * (t - 450) for t in ts]
        self.assertAlmostEqual(se.trend_change_pp(ts, ys, 450, 470), 1.0)

    def test_trend_ignores_endpoint_spike(self):
        ts = list(range(450, 471))
        ys = [0.0] * 21
        ys[-1] = 3.0                                              # endpoint jump: endpoint change 3, trend much less
        tr = se.trend_change_pp(ts, ys, 450, 470)
        self.assertLess(abs(tr), 1.5)

    def test_coverage_and_points(self):
        ts = [450, 451, 452, 470]
        self.assertIsNone(se.trend_change_pp(ts, [0, 1, 2, 3], 450, 470))       # 4 of 21 minutes
        self.assertIsNone(se.trend_change_pp(list(range(450, 454)), [0, 1, 2, 3], 450, 453))   # 4 points < 5
        self.assertIsNone(se.trend_change_pp([450], [1.0], 450, 450))

    def test_nan_dropped_and_outside_window_ignored(self):
        ts = list(range(440, 481))
        ys = [float(t - 450) * 0.1 for t in ts]
        ys[15] = float("nan")
        self.assertAlmostEqual(se.trend_change_pp(ts, ys, 450, 470), 2.0)


class TestNoiseBand(unittest.TestCase):
    TABLE = {b: tuple(0.5 + 0.01 * L for L in se.NOISE_BAND_LENGTHS) for b in range(1, 13)}

    def test_grid_points_and_interp(self):
        self.assertAlmostEqual(se.noise_band_pp(15, time(8, 0), self.TABLE), 0.65)
        self.assertAlmostEqual(se.noise_band_pp(25, time(8, 0), self.TABLE), 0.75)

    def test_clamped(self):
        self.assertAlmostEqual(se.noise_band_pp(2, time(8, 0), self.TABLE), 0.55)
        self.assertAlmostEqual(se.noise_band_pp(300, time(8, 0), self.TABLE), 1.70)

    def test_unmeasured_bucket(self):
        self.assertIsNone(se.noise_band_pp(15, time(6, 40), self.TABLE))     # 06:30 bucket not measured
        self.assertIsNone(se.noise_band_pp(15, time(8, 0), {}))


class TestStickinessRead(unittest.TestCase):
    def setUp(self):
        self.v, _ = smile(slope=-0.8, curv=40.0, T=se.years_to_close_0dte(time(7, 30)))
        self.ts = list(range(450, 471))                          # 07:30 .. 07:50

    def test_ratio_and_sign(self):
        exp = se.sticky_strike_put_skew_change_pp(self.v, se.years_to_close_0dte(time(7, 30)), 0.4)
        ys = [exp * (t - 450) / 20.0 for t in self.ts]           # put skew moved exactly as sticky strike says
        r = se.stickiness_read(self.ts, ys, time(7, 30), time(7, 50), self.v, 7000.0, 7028.0,
                               band_table=TestNoiseBand.TABLE)
        self.assertEqual(r.reason, "ok")
        self.assertAlmostEqual(r.ratio, 1.0, places=6)
        ys = [-y for y in ys]                                    # put skew steepened into a rally
        r = se.stickiness_read(self.ts, ys, time(7, 30), time(7, 50), self.v, 7000.0, 7028.0)
        self.assertAlmostEqual(r.ratio, -1.0, places=6)

    def test_move_too_small(self):
        r = se.stickiness_read(self.ts, [0.1] * 21, time(7, 30), time(7, 50), self.v, 7000.0, 7007.0)
        self.assertEqual(r.reason, "move too small")             # 0.10 % < 0.15 %
        self.assertIsNone(r.ratio)
        self.assertIsNotNone(r.expected_pp)

    def test_noise_flag(self):
        ys = [0.05 * (t - 450) for t in self.ts]                 # actual +1.0 over 20 min
        r = se.stickiness_read(self.ts, ys, time(7, 30), time(7, 50), self.v, 7000.0, 7028.0,
                               band_table=TestNoiseBand.TABLE)
        self.assertAlmostEqual(r.band_pp, 0.70)
        self.assertTrue(r.beyond_noise)
        r = se.stickiness_read(self.ts, [y / 4 for y in ys], time(7, 30), time(7, 50), self.v, 7000.0, 7028.0,
                               band_table=TestNoiseBand.TABLE)
        self.assertFalse(r.beyond_noise)

    def test_late_day_flag(self):
        r = se.stickiness_read(self.ts, [0.0] * 21, time(7, 30), time(7, 50), self.v, 7000.0, 7028.0)
        self.assertFalse(r.late_day)
        ts = list(range(640, 661))                               # 10:40 .. 11:00: ends at 11:00, not after
        r = se.stickiness_read(ts, [0.0] * 21, time(10, 40), time(11, 0), self.v, 7000.0, 7028.0)
        self.assertFalse(r.late_day)
        ts = list(range(641, 662))                               # 10:41 .. 11:01
        r = se.stickiness_read(ts, [0.0] * 21, time(10, 41), time(11, 1), self.v, 7000.0, 7028.0)
        self.assertTrue(r.late_day)
        r = se.stickiness_read([700, 720], [0.0, 1.0], time(11, 40), time(12, 0), self.v, 7000.0, 7028.0)
        self.assertEqual((r.reason, r.late_day), ("insufficient data", True))

    def test_insufficient_data(self):
        r = se.stickiness_read([450, 470], [0.0, 1.0], time(7, 30), time(7, 50), self.v, 7000.0, 7028.0)
        self.assertEqual(r.reason, "insufficient data")
        self.assertIsNone(r.ratio)


if __name__ == "__main__":
    unittest.main()
