"""CR-AO decisions 1–3: listed-strike snapping."""
import sys
from datetime import date
from pathlib import Path

_ROOT = str(Path(__file__).parent.parent.parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import unittest

from packages.shared.options_cache.strikes import (
    StrikeNotListed,
    listed_strikes,
    snap_to_candidates,
    snap_to_listed_strike,
    snap_vertical_legs,
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


if __name__ == "__main__":
    unittest.main()
