"""CR-AO decisions 1–3: listed-strike snapping; CR-AR decisions 1–2: pair snapping with width cap."""
import sys
from datetime import date
from pathlib import Path

_ROOT = str(Path(__file__).parent.parent.parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import unittest

from packages.shared.options_cache.strikes import (
    StrikeNotListed,
    StructureNotListed,
    listed_strikes,
    snap_spread_to_listed,
    snap_to_candidates,
    snap_to_listed_strike,
    snap_vertical_legs,
    snap_vertical_pair,
)


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0] if self._rows else None


class _FakeConn:
    """Answers the two chain queries: prior close and listed strikes per (close, expiry)."""

    def __init__(self, prior: date, chain: dict):
        self.prior = prior
        self.chain = chain          # {(prior, expiry): [strikes]}
        self.calls = []

    def execute(self, sql, params):
        self.calls.append((sql, params))
        if "max(trade_date)" in sql:
            return _Result([(self.prior,)])
        _, pc, exp = params
        return _Result([(k,) for k in self.chain.get((pc, exp), [])])


class _SQLAlchemyLikeConn(_FakeConn):
    def exec_driver_sql(self, sql, params):
        return self.execute(sql, params)

    def execute(self, sql, params=None):  # would need text() for real SQLAlchemy; must not be used
        if not hasattr(self, "_via_driver"):
            self._via_driver = True
        return super().execute(sql, params)


_EXP = date(2026, 8, 3)
_TD = date(2026, 7, 13)
_PC = date(2026, 7, 10)
_GRID_15D = [7500, 7550, 7600, 7610, 7620, 7650, 7700, 7750]   # coarse grid, no 7655


class TestSnapToCandidates(unittest.TestCase):
    def test_nearest(self):
        self.assertEqual(snap_to_candidates(7655, _GRID_15D), 7650)
        self.assertEqual(snap_to_candidates(7648, _GRID_15D), 7650)
        self.assertEqual(snap_to_candidates(7699, _GRID_15D), 7700)

    def test_tie_goes_toward_spot(self):
        # 7675 is equidistant from 7650 and 7700
        self.assertEqual(snap_to_candidates(7675, _GRID_15D, toward=7580), 7650)   # spot below → lower
        self.assertEqual(snap_to_candidates(7675, _GRID_15D, toward=7800), 7700)   # spot above → upper

    def test_tie_without_toward_takes_lower(self):
        self.assertEqual(snap_to_candidates(7675, _GRID_15D), 7650)

    def test_empty_raises(self):
        with self.assertRaises(StrikeNotListed):
            snap_to_candidates(7655, [])


class TestSnapToListedStrike(unittest.TestCase):
    def test_uses_prior_close_chain(self):
        conn = _FakeConn(_PC, {(_PC, _EXP): _GRID_15D})
        self.assertEqual(snap_to_listed_strike(7655, _EXP, _TD, conn, toward=7580), 7650)
        pc, cands = listed_strikes(conn, _EXP, _TD)
        self.assertEqual(pc, _PC)
        self.assertEqual(cands[0], 7500.0)

    def test_unlisted_expiry_raises(self):
        conn = _FakeConn(_PC, {})
        with self.assertRaises(StrikeNotListed):
            snap_to_listed_strike(7655, _EXP, _TD, conn)

    def test_sqlalchemy_style_connection(self):
        conn = _SQLAlchemyLikeConn(_PC, {(_PC, _EXP): _GRID_15D})
        self.assertEqual(snap_to_listed_strike(7655, _EXP, _TD, conn), 7650)
        self.assertTrue(getattr(conn, "_via_driver", False))


class TestSnapVerticalLegs(unittest.TestCase):
    def test_debit_legs_and_width_actual(self):
        # debit: anchor at target, other leg 10 below → nearest listed below 7650 is 7620 → width 30
        conn = _FakeConn(_PC, {(_PC, _EXP): _GRID_15D})
        v = snap_vertical_legs(7655, -10, _EXP, _TD, conn, toward=7580)
        self.assertEqual((v.anchor, v.other), (7650, 7620))
        self.assertEqual(v.width_actual, 30.0)
        self.assertEqual(v.width_nominal, 10.0)
        self.assertEqual(v.prior_close, _PC)

    def test_credit_legs_when_grid_is_complete(self):
        full = list(range(7600, 7705, 5))
        conn = _FakeConn(_PC, {(_PC, _EXP): full})
        v = snap_vertical_legs(7655, +10, _EXP, _TD, conn)
        self.assertEqual((v.anchor, v.other, v.width_actual), (7655, 7665, 10.0))

    def test_other_leg_is_strictly_beyond_anchor(self):
        # anchor 7650; +10 side candidates are 7700, 7750 → 7700 (never 7650 itself)
        conn = _FakeConn(_PC, {(_PC, _EXP): _GRID_15D})
        v = snap_vertical_legs(7650, +10, _EXP, _TD, conn)
        self.assertEqual((v.anchor, v.other, v.width_actual), (7650, 7700, 50.0))

    def test_no_strike_on_required_side_raises(self):
        conn = _FakeConn(_PC, {(_PC, _EXP): [7650]})
        with self.assertRaises(StrikeNotListed):
            snap_vertical_legs(7650, -10, _EXP, _TD, conn)


class TestSnapSpreadToListed(unittest.TestCase):
    """CR-AR decisions 1–2 (G1): pair snapping with a hard 2× width cap."""

    _FULL = list(range(5650, 5755, 5))                        # complete 5-point grid
    _TENS = [5650, 5660, 5670, 5680, 5690, 5700, 5710, 5720, 5730, 5740, 5750]
    _TWENTIES = [5660, 5680, 5700, 5720, 5740]                # only 2× pairs exist
    _25S = [5650, 5675, 5700, 5725, 5750]                     # only 2.5× pairs exist → unlistable
    _FIFTIES = [5650, 5700, 5750]                             # CR-AP's 2024-09-17 credit (50-wide)

    def test_nominal_pair_found_on_full_grid(self):
        v = snap_spread_to_listed(5710, 10, "credit", None, self._FULL, toward=5680)
        self.assertEqual((v.anchor, v.other, v.width_actual), (5710, 5720, 10.0))
        self.assertEqual((v.k_low, v.k_high), (5710, 5720))
        self.assertFalse(v.narrower_than_nominal); self.assertFalse(v.widened); self.assertFalse(v.at_cap)

    def test_nominal_pair_found_on_ten_point_grid_snaps_anchor(self):
        # target 5713 → anchor 5710 (nearest listed), wing 5720
        v = snap_spread_to_listed(5713, 10, "credit", None, self._TENS)
        self.assertEqual((v.anchor, v.other, v.width_actual), (5710, 5720, 10.0))

    def test_only_two_x_pair_exists_is_chosen_and_flagged(self):
        v = snap_spread_to_listed(5700, 10, "credit", None, self._TWENTIES)
        self.assertEqual((v.anchor, v.other, v.width_actual), (5700, 5720, 20.0))
        self.assertTrue(v.widened); self.assertTrue(v.at_cap); self.assertFalse(v.narrower_than_nominal)

    def test_only_wider_than_cap_raises_structure_not_listed(self):
        with self.assertRaises(StructureNotListed):
            snap_spread_to_listed(5700, 10, "credit", None, self._25S)
        with self.assertRaises(StructureNotListed):        # CR-AP's 50-wide can no longer happen
            snap_spread_to_listed(5710, 10, "credit", None, self._FIFTIES)
        with self.assertRaises(StructureNotListed):        # 3× only
            snap_spread_to_listed(5700, 10, "debit", None, [5670, 5700, 5730])

    def test_structure_not_listed_is_a_strike_not_listed(self):
        with self.assertRaises(StrikeNotListed):
            snap_spread_to_listed(5700, 10, "credit", None, [])

    def test_never_wider_than_cap_even_when_nearer_anchor_exists(self):
        # 5700 is at the target but only pairs 5700/5750 (50) — the 5680/5690..5690/5700
        # pairs are the only ones within the cap; anchor moves rather than width breaching
        chain = [5680, 5690, 5700, 5750]
        v = snap_spread_to_listed(5700, 10, "credit", None, chain)
        self.assertLessEqual(v.width_actual, 20.0)
        self.assertEqual((v.anchor, v.other), (5690, 5700))

    def test_tie_break_a_width_closest_to_nominal(self):
        # target 5710 between 5700 and 5720 (both anchors 10 away); 5700 has only a 20-wide, 5720 a 10-wide
        chain = [5700, 5720, 5730]
        v = snap_spread_to_listed(5710, 10, "credit", None, chain, toward=5600)   # magnet below would favour 5700
        self.assertEqual((v.anchor, v.other, v.width_actual), (5720, 5730, 10.0))

    def test_tie_break_b_toward_magnet_direction(self):
        # target 5705; both 5700/5710 and 5710/5720 are 10-wide → magnet decides
        chain = [5700, 5710, 5720]
        below = snap_spread_to_listed(5705, 10, "credit", None, chain, toward=5600)
        above = snap_spread_to_listed(5705, 10, "credit", None, chain, toward=5800)
        self.assertEqual(below.anchor, 5700)
        self.assertEqual(above.anchor, 5710)
        self.assertEqual(snap_spread_to_listed(5705, 10, "credit", None, chain).anchor, 5700)   # no magnet → lower

    def test_debit_anchors_k_high_and_credit_anchors_k_low(self):
        d = snap_spread_to_listed(5710, 10, "debit", None, self._TENS)
        c = snap_spread_to_listed(5710, 10, "credit", None, self._TENS)
        self.assertEqual((d.k_low, d.k_high, d.anchor, d.other), (5700, 5710, 5710, 5700))
        self.assertEqual((c.k_low, c.k_high, c.anchor, c.other), (5710, 5720, 5710, 5720))
        self.assertEqual(d.side, "debit"); self.assertEqual(c.side, "credit")

    def test_explicit_direction_overrides_side_geometry(self):
        # a put credit spread: short put at the target, long put below → anchor is k_high
        v = snap_spread_to_listed(5710, 10, "credit", -1, self._TENS)
        self.assertEqual((v.k_low, v.k_high, v.anchor, v.other), (5700, 5710, 5710, 5700))

    def test_narrower_than_nominal_fallback_only_when_nothing_at_nominal(self):
        v = snap_spread_to_listed(5175, 10, "credit", None, [5175, 5180], toward=5100)
        self.assertEqual((v.anchor, v.other, v.width_actual), (5175, 5180, 5.0))
        self.assertTrue(v.narrower_than_nominal)
        # a 10-wide exists within the cap → it wins over the 5-wide even though the anchor moves
        v2 = snap_spread_to_listed(5175, 10, "credit", None, [5170, 5175, 5180, 5200], toward=5100)
        self.assertEqual((v2.anchor, v2.other, v2.width_actual), (5170, 5180, 10.0))
        self.assertFalse(v2.narrower_than_nominal)

    def test_cr_ap_wide_credit_trades_now_capped(self):
        # CR-AP snapped 2025-03-26 credit to 5850/5875 (25) on a 25-point grid; with a 20-wide
        # available it is chosen, otherwise the structure is unlistable
        with self.assertRaises(StructureNotListed):
            snap_spread_to_listed(5850, 10, "credit", None, [5800, 5825, 5850, 5875, 5900])
        v = snap_spread_to_listed(5850, 10, "credit", None, [5800, 5825, 5850, 5870, 5875, 5900])
        self.assertEqual((v.anchor, v.other, v.width_actual), (5850, 5870, 20.0))

    def test_anchor_shift_bounded_by_twice_nominal(self):
        # Step 0 amendment A1 — CR-AP's 2024-09-17: chain near the 5708 target is [5700] only;
        # the nearest pair within the cap sits at 5640/5650 (68 points away) → unlistable, not that trade
        chain = [5600, 5610, 5620, 5630, 5640, 5650, 5700, 5750, 5800]
        with self.assertRaises(StructureNotListed):
            snap_spread_to_listed(5708.23, 10, "credit", None, chain, toward=5633)
        with self.assertRaises(StructureNotListed):
            snap_spread_to_listed(5708.23, 10, "debit", None, chain, toward=5633)
        # exactly at the tolerance (20 points) is still allowed; beyond it is not
        v = snap_spread_to_listed(5660, 10, "credit", None, chain)          # anchor 5640 is 20 away; 5650 has no wing in cap
        self.assertEqual((v.anchor, v.other), (5640, 5650))
        with self.assertRaises(StructureNotListed):
            snap_spread_to_listed(5661, 10, "credit", None, chain)          # 5640 is now 21 away
        # the tolerance is a parameter
        v = snap_spread_to_listed(5708.23, 10, "credit", None, chain, max_anchor_shift=100)
        self.assertEqual((v.anchor, v.other), (5640, 5650))

    def test_cr_ap_2024_09_23_credit_moves_anchor_within_tolerance(self):
        # chain [5725, 5730, 5740, 5750, 5775], target 5754.32: 5750 only pairs to 5775 (25, over cap);
        # 5740/5750 (anchor 14 away, within 20) is the listed version of the trade
        v = snap_spread_to_listed(5754.32, 10, "credit", None, [5725, 5730, 5740, 5750, 5775], toward=5713)
        self.assertEqual((v.anchor, v.other, v.width_actual), (5740, 5750, 10.0))

    def test_invalid_side_or_width(self):
        with self.assertRaises(ValueError):
            snap_spread_to_listed(5700, 10, "iron", None, self._FULL)
        with self.assertRaises(ValueError):
            snap_spread_to_listed(5700, 0, "credit", None, self._FULL)


class TestSnapVerticalPair(unittest.TestCase):
    """DB-backed wrapper: prior-close chain → snap_spread_to_listed."""

    def test_uses_prior_close_chain_and_records_it(self):
        conn = _FakeConn(_PC, {(_PC, _EXP): _GRID_15D})
        # 7650 is listed but its only wing above is 7700 (50 wide); the 7610/7620 pair is
        # 45 points away → over the anchor tolerance → unlistable at this expiry
        with self.assertRaises(StructureNotListed):
            snap_vertical_pair(7655, 10, "credit", _EXP, _TD, conn, toward=7580)
        v = snap_vertical_pair(7615, 10, "credit", _EXP, _TD, conn, toward=7580)
        self.assertEqual((v.anchor, v.other, v.width_actual), (7610, 7620, 10.0))
        self.assertEqual(v.prior_close, _PC)

    def test_unlisted_expiry_raises_structure_not_listed(self):
        conn = _FakeConn(_PC, {})
        with self.assertRaises(StructureNotListed):
            snap_vertical_pair(7655, 10, "credit", _EXP, _TD, conn)

    def test_sqlalchemy_style_connection(self):
        conn = _SQLAlchemyLikeConn(_PC, {(_PC, _EXP): list(range(7600, 7705, 5))})
        v = snap_vertical_pair(7655, 10, "debit", _EXP, _TD, conn)
        self.assertEqual((v.anchor, v.other, v.width_actual), (7655, 7645, 10.0))
        self.assertTrue(getattr(conn, "_via_driver", False))


if __name__ == "__main__":
    unittest.main()
