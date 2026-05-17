"""
Unit tests for condor leg derivation. No I/O.

Run with:
    python -m unittest packages.shared.options_cache.tests.test_condor
"""
from __future__ import annotations

import unittest
from datetime import date, datetime, timezone

from packages.shared.options_cache.condor import (
    Leg,
    condor_legs_for_row,
    condor_pricing_window_for_row,
    end_of_day_pt,
    map_spx_strike_to_spy,
    parse_utc_iso,
    utc_to_pt_naive,
)


# ────────────────────────────────────────────────────────────────────────
#  Pure helpers
# ────────────────────────────────────────────────────────────────────────

class TestMapSpxStrikeToSpy(unittest.TestCase):
    def test_atm(self):
        # SPX 4940, strike at 4940 → ratio 1.0 → SPY 494
        self.assertEqual(map_spx_strike_to_spy(4940.0, 4940.0), 494.0)

    def test_short_put_below_spot(self):
        # SPX spot 4940, strike 4935 → 0.99899 × 494 = 493.5 → rounds to 494
        # Note: banker's rounding in Python rounds 493.5 to 494
        result = map_spx_strike_to_spy(4935.0, 4940.0)
        self.assertIn(result, (493.0, 494.0))

    def test_short_call_above_spot(self):
        # SPX 4940, strike 4980 → 1.00810 × 494 = 498.0
        result = map_spx_strike_to_spy(4980.0, 4940.0)
        self.assertEqual(result, 498.0)

    def test_typical_condor_set(self):
        # Realistic SPX condor: spot 4940
        spot = 4940.0
        legs = {
            "short_put":  map_spx_strike_to_spy(4935.0, spot),
            "long_put":   map_spx_strike_to_spy(4925.0, spot),
            "short_call": map_spx_strike_to_spy(4980.0, spot),
            "long_call":  map_spx_strike_to_spy(4990.0, spot),
        }
        # Just verify all are reasonable SPY-range values
        for name, strike in legs.items():
            self.assertGreater(strike, 480, name)
            self.assertLess(strike, 510, name)

    def test_invalid_spot_raises(self):
        with self.assertRaises(ValueError):
            map_spx_strike_to_spy(4900.0, 0.0)
        with self.assertRaises(ValueError):
            map_spx_strike_to_spy(4900.0, -100.0)

    def test_invalid_strike_raises(self):
        with self.assertRaises(ValueError):
            map_spx_strike_to_spy(0.0, 4940.0)
        with self.assertRaises(ValueError):
            map_spx_strike_to_spy(-50.0, 4940.0)


class TestTimeHelpers(unittest.TestCase):
    def test_parse_utc_iso_with_z(self):
        from datetime import timedelta
        dt = parse_utc_iso("2024-02-02T17:14:00Z")
        # Don't compare tzinfo object identity (ZoneInfo vs timezone.utc); the
        # important property is that the offset is zero.
        self.assertEqual(dt.utcoffset(), timedelta(0))
        self.assertEqual(dt.year, 2024)
        self.assertEqual(dt.hour, 17)

    def test_parse_utc_iso_with_offset(self):
        dt = parse_utc_iso("2024-02-02T17:14:00+00:00")
        self.assertEqual(dt.hour, 17)

    def test_utc_to_pt_naive_winter(self):
        # Feb 2024 is no DST: 17:14 UTC = 09:14 PT
        dt_utc = parse_utc_iso("2024-02-02T17:14:00Z")
        dt_pt = utc_to_pt_naive(dt_utc)
        self.assertEqual(dt_pt, datetime(2024, 2, 2, 9, 14))
        self.assertIsNone(dt_pt.tzinfo)

    def test_utc_to_pt_naive_summer(self):
        # July 2024 is DST: 17:14 UTC = 10:14 PT
        dt_utc = parse_utc_iso("2024-07-02T17:14:00Z")
        dt_pt = utc_to_pt_naive(dt_utc)
        self.assertEqual(dt_pt, datetime(2024, 7, 2, 10, 14))

    def test_end_of_day_pt(self):
        eod = end_of_day_pt(date(2024, 2, 2))
        self.assertEqual(eod, datetime(2024, 2, 2, 13, 0))


# ────────────────────────────────────────────────────────────────────────
#  condor_legs_for_row
# ────────────────────────────────────────────────────────────────────────

def _make_row(**overrides) -> dict:
    """Build a minimal scan row."""
    row = {
        "trade_date": "2024-02-02",
        "target_ts_utc": "2024-02-02T17:14:00Z",
        "target_spx_price": 4940.0,
        "hypothetical_condor_120m": {
            "short_put_strike": 4935.0,
            "long_put_strike": 4925.0,
            "short_call_strike": 4980.0,
            "long_call_strike": 4990.0,
        },
        "hypothetical_condor_to_close": {
            "short_put_strike": 4935.0,
            "long_put_strike": 4925.0,
            "short_call_strike": 4980.0,
            "long_call_strike": 4990.0,
        },
    }
    row.update(overrides)
    return row


class TestCondorLegsForRow(unittest.TestCase):
    def test_basic_row_yields_4_legs(self):
        # Both horizons have identical strikes → 4 unique SPY contracts
        row = _make_row()
        legs = condor_legs_for_row(row)
        self.assertEqual(len(legs), 4)

        # OPRA format for SPY: SPY (3) + YYMMDD (6) + C/P (1) + strike (8)
        # So C/P character is at index 9
        puts = [l for l in legs if l.opra_symbol[9] == "P"]
        calls = [l for l in legs if l.opra_symbol[9] == "C"]
        self.assertEqual(len(puts), 2)
        self.assertEqual(len(calls), 2)

    def test_legs_have_correct_sides(self):
        row = _make_row()
        legs = condor_legs_for_row(row)
        sides = sorted(l.side for l in legs)
        # Iron condor: 2 short, 2 long
        self.assertEqual(sides, ["long", "long", "short", "short"])

    def test_legs_are_spy_opra(self):
        row = _make_row()
        legs = condor_legs_for_row(row)
        for leg in legs:
            self.assertTrue(leg.opra_symbol.startswith("SPY"))
            self.assertEqual(len(leg.opra_symbol), 18)  # SPY + 6 + C/P + 8

    def test_different_horizons_yield_more_legs(self):
        # Make horizons differ enough that strikes don't collide on rounding.
        # SPX→SPY divides by ~10, so a 30-pt SPX gap = 3 SPY strikes apart.
        row = _make_row(
            hypothetical_condor_120m={
                "short_put_strike": 4935.0,   # → SPY 494
                "long_put_strike": 4925.0,    # → SPY 492-493
                "short_call_strike": 4980.0,  # → SPY 498
                "long_call_strike": 4990.0,   # → SPY 499
            },
            hypothetical_condor_to_close={
                "short_put_strike": 4880.0,   # → SPY ~488 (different)
                "long_put_strike": 4870.0,    # → SPY ~487 (different)
                "short_call_strike": 5020.0,  # → SPY ~502 (different)
                "long_call_strike": 5030.0,   # → SPY ~503 (different)
            },
        )
        legs = condor_legs_for_row(row)
        # 4 from 120m + 4 from to_close, no overlap → 8 unique
        unique_opras = {l.opra_symbol for l in legs}
        self.assertEqual(len(unique_opras), 8)

    def test_overlapping_horizons_dedupe(self):
        # When both horizons produce identical SPY strikes, we get 4 legs
        # (not 8) and the role string mentions both horizons.
        row = _make_row()
        legs = condor_legs_for_row(row)
        self.assertEqual(len(legs), 4)
        # Each leg's role should mention both horizons, since both produced it
        for leg in legs:
            self.assertIn("120m", leg.role)
            self.assertIn("to_close", leg.role)

    def test_missing_spx_spot_returns_empty(self):
        row = _make_row(target_spx_price=None)
        self.assertEqual(condor_legs_for_row(row), [])

    def test_missing_strikes_returns_empty(self):
        row = _make_row(
            hypothetical_condor_120m=None,
            hypothetical_condor_to_close=None,
        )
        self.assertEqual(condor_legs_for_row(row), [])

    def test_missing_one_strike_skips_horizon(self):
        # One horizon missing a strike → that horizon is skipped, but the
        # other still produces legs
        row = _make_row(
            hypothetical_condor_120m={
                "short_put_strike": 4935.0,
                "long_put_strike": None,  # missing
                "short_call_strike": 4980.0,
                "long_call_strike": 4990.0,
            },
        )
        legs = condor_legs_for_row(row)
        # Only to_close horizon produced legs → 4 unique
        self.assertEqual(len(legs), 4)
        for leg in legs:
            self.assertIn("to_close", leg.role)

    def test_explicit_expiration_override(self):
        # If we pass expiration, it overrides trade_date
        row = _make_row()
        legs = condor_legs_for_row(row, expiration=date(2024, 2, 9))
        for leg in legs:
            # YYMMDD portion: chars 3-9
            self.assertEqual(leg.opra_symbol[3:9], "240209")


# ────────────────────────────────────────────────────────────────────────
#  condor_pricing_window_for_row
# ────────────────────────────────────────────────────────────────────────

class TestCondorPricingWindowForRow(unittest.TestCase):
    def test_typical_setup(self):
        row = _make_row()
        window = condor_pricing_window_for_row(row)
        self.assertIsNotNone(window)
        start_pt, end_pt = window
        # Setup at 17:14 UTC = 09:14 PT
        self.assertEqual(start_pt, datetime(2024, 2, 2, 9, 14))
        # End at 13:00 PT same day
        self.assertEqual(end_pt, datetime(2024, 2, 2, 13, 0))

    def test_falls_back_to_start_ts_utc(self):
        row = _make_row(target_ts_utc=None, start_ts_utc="2024-02-02T18:00:00Z")
        window = condor_pricing_window_for_row(row)
        self.assertIsNotNone(window)
        # 18:00 UTC = 10:00 PT
        self.assertEqual(window[0], datetime(2024, 2, 2, 10, 0))

    def test_setup_after_close_falls_back_to_single_minute(self):
        # 22:00 UTC = 14:00 PT, past 13:00 close
        row = _make_row(target_ts_utc="2024-02-02T22:00:00Z")
        window = condor_pricing_window_for_row(row)
        self.assertEqual(window[0], window[1])  # single minute

    def test_missing_timestamps_returns_none(self):
        row = _make_row(target_ts_utc=None, start_ts_utc=None)
        self.assertIsNone(condor_pricing_window_for_row(row))


if __name__ == "__main__":
    unittest.main()
