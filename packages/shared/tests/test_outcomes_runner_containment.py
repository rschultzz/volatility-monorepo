"""CR-AQ: compute_session_containment / nearest_walls (synthetic bars, no DB)."""
from __future__ import annotations

import unittest
from datetime import date

import pandas as pd

from packages.shared.outcomes_runner import (
    CONTAINMENT_COLUMNS,
    compute_session_containment,
    nearest_walls,
)

_D = date(2026, 5, 6)


def _bars(o, h, l, c, d=_D):
    return pd.DataFrame([(o, h, l, c)], index=[d], columns=["open", "high", "low", "close"])


def _walls(*prices):
    return [{"price": p, "gex": 1.0, "sign": 1 if i % 2 == 0 else -1} for i, p in enumerate(prices)]


class TestNearestWalls(unittest.TestCase):
    def test_nearest_each_side_any_sign(self):
        self.assertEqual(nearest_walls(_walls(7200, 7250, 7300, 7350), 7271.2), (7250.0, 7300.0))

    def test_missing_side(self):
        self.assertEqual(nearest_walls(_walls(7306.2), 7271.2), (None, 7306.2))
        self.assertEqual(nearest_walls(_walls(7200), 7271.2), (7200.0, None))
        self.assertEqual(nearest_walls([], 7271.2), (None, None))

    def test_wall_at_reference_price_is_neither_side(self):
        self.assertEqual(nearest_walls(_walls(7271.2, 7300), 7271.2), (None, 7300.0))


class TestContainment(unittest.TestCase):
    W = _walls(7250, 7300)   # band 50 wide around an open of 7271.2

    def test_contained_true(self):
        r = compute_session_containment(_D, self.W, _bars(7271.2, 7290, 7260, 7280), 40.0)
        self.assertTrue(r["contained_close"]); self.assertTrue(r["contained_range"])
        self.assertIsNone(r["breach_side"])
        self.assertAlmostEqual(r["close_pos_in_band"], 0.6)
        self.assertAlmostEqual(r["range_over_im"], 0.75)
        self.assertAlmostEqual(r["close_move_over_im"], 0.22)
        self.assertEqual((r["session_high_t0"], r["session_low_t0"], r["session_close_t0"]), (7290.0, 7260.0, 7280.0))
        self.assertEqual((r["wall_below_price"], r["wall_above_price"]), (7250.0, 7300.0))

    def test_breach_above_close_outside(self):
        r = compute_session_containment(_D, self.W, _bars(7271.2, 7320, 7265, 7315), 40.0)
        self.assertFalse(r["contained_close"]); self.assertFalse(r["contained_range"])
        self.assertEqual(r["breach_side"], "above")
        self.assertGreater(r["close_pos_in_band"], 1.0)

    def test_breach_below_but_close_back_inside(self):
        r = compute_session_containment(_D, self.W, _bars(7271.2, 7280, 7240, 7260), 40.0)
        self.assertTrue(r["contained_close"]); self.assertFalse(r["contained_range"])
        self.assertEqual(r["breach_side"], "below")

    def test_breach_both(self):
        r = compute_session_containment(_D, self.W, _bars(7271.2, 7310, 7240, 7270), 40.0)
        self.assertEqual(r["breach_side"], "both"); self.assertFalse(r["contained_range"])

    def test_close_pos_at_band_edges(self):
        r0 = compute_session_containment(_D, self.W, _bars(7271.2, 7290, 7250, 7250), 40.0)
        self.assertAlmostEqual(r0["close_pos_in_band"], 0.0); self.assertFalse(r0["contained_close"])
        r1 = compute_session_containment(_D, self.W, _bars(7271.2, 7300, 7260, 7300), 40.0)
        self.assertAlmostEqual(r1["close_pos_in_band"], 1.0); self.assertFalse(r1["contained_close"])

    def test_null_wall_side_gives_null_containment_but_keeps_ohlc(self):
        r = compute_session_containment(_D, _walls(7306.2), _bars(7271.2, 7330, 7265, 7325), 40.0)
        self.assertIsNone(r["contained_close"]); self.assertIsNone(r["contained_range"])
        self.assertIsNone(r["close_pos_in_band"]); self.assertIsNone(r["breach_side"])
        self.assertIsNone(r["wall_below_price"]); self.assertEqual(r["wall_above_price"], 7306.2)
        self.assertEqual(r["session_close_t0"], 7325.0)
        self.assertAlmostEqual(r["range_over_im"], 65 / 40)

    def test_range_over_im_null_when_im_nonpositive_or_missing(self):
        for im in (0.0, -5.0, None, "x"):
            r = compute_session_containment(_D, self.W, _bars(7271.2, 7290, 7260, 7280), im)
            self.assertIsNone(r["range_over_im"]); self.assertIsNone(r["close_move_over_im"])
            self.assertTrue(r["contained_close"])   # containment does not depend on IM

    def test_no_bar_row_gives_all_none(self):
        r = compute_session_containment(date(2026, 5, 7), self.W, _bars(7271.2, 7290, 7260, 7280), 40.0)
        self.assertEqual(set(r), set(CONTAINMENT_COLUMNS))
        self.assertTrue(all(v is None for v in r.values()))


if __name__ == "__main__":
    unittest.main()
