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
    condor_strikes_from_smile,
    end_of_day_pt,
    parse_utc_iso,
    utc_to_pt_naive,
)


# ────────────────────────────────────────────────────────────────────────
#  Pure helpers
# ────────────────────────────────────────────────────────────────────────

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
        # Both horizons have identical strikes → 4 unique SPX contracts
        row = _make_row()
        legs = condor_legs_for_row(row)
        self.assertEqual(len(legs), 4)

        # OPRA format for SPX: SPX (3) + YYMMDD (6) + C/P (1) + strike (8)
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

    def test_legs_are_spx_opra(self):
        row = _make_row()
        legs = condor_legs_for_row(row)
        for leg in legs:
            self.assertTrue(leg.opra_symbol.startswith("SPX"))
            self.assertEqual(len(leg.opra_symbol), 18)  # SPX + 6 + C/P + 8

    def test_different_horizons_yield_more_legs(self):
        # With native SPX, strike differences are preserved 1:1, so
        # distinct horizon strikes always yield distinct OPRAs.
        row = _make_row(
            hypothetical_condor_120m={
                "short_put_strike": 4935.0,
                "long_put_strike": 4925.0,
                "short_call_strike": 4980.0,
                "long_call_strike": 4990.0,
            },
            hypothetical_condor_to_close={
                "short_put_strike": 4880.0,
                "long_put_strike": 4870.0,
                "short_call_strike": 5020.0,
                "long_call_strike": 5030.0,
            },
        )
        legs = condor_legs_for_row(row)
        # 4 from 120m + 4 from to_close, no overlap → 8 unique
        unique_opras = {l.opra_symbol for l in legs}
        self.assertEqual(len(unique_opras), 8)

    def test_overlapping_horizons_dedupe(self):
        # When both horizons produce identical SPX strikes, we get 4 legs
        # (not 8) and the role string mentions both horizons.
        row = _make_row()
        legs = condor_legs_for_row(row)
        self.assertEqual(len(legs), 4)
        # Each leg's role should mention both horizons, since both produced it
        for leg in legs:
            self.assertIn("120m", leg.role)
            self.assertIn("to_close", leg.role)

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


# ────────────────────────────────────────────────────────────────────────
#  condor_strikes_from_smile
# ────────────────────────────────────────────────────────────────────────

class TestCondorStrikesFromSmile(unittest.TestCase):
    def test_basic_strike_geometry(self):
        # spx=5800, iv=15%, mins=120 → sigma ~= 5800 * 0.15 * sqrt(120/525600)
        # ~= 5800 * 0.15 * 0.01510 ≈ 13.14 pts
        out = condor_strikes_from_smile(5800.0, 15.0, 120)
        self.assertIsNotNone(out)
        # short_put floors (5800 - sigma) / 5
        self.assertLess(out["short_put"], 5800)
        # short_call ceils (5800 + sigma) / 5
        self.assertGreater(out["short_call"], 5800)
        # default wing=10 → long legs sit 10 beyond shorts
        self.assertAlmostEqual(out["long_put"], out["short_put"] - 10, places=6)
        self.assertAlmostEqual(out["long_call"], out["short_call"] + 10, places=6)
        # All strikes land on a multiple of 5
        for k in ("short_put", "long_put", "short_call", "long_call"):
            self.assertAlmostEqual(out[k] / 5.0, round(out[k] / 5.0), places=6)
        # sigma_pts roundtrips against the formula
        self.assertGreater(out["sigma_pts"], 12.0)
        self.assertLess(out["sigma_pts"], 14.0)

    def test_strike_rounding_floors_short_put_ceils_short_call(self):
        # Construct a case where sigma puts us between two strikes; verify
        # floor/ceil semantics — short_put rounded DOWN (further from spot),
        # short_call rounded UP (also further from spot).
        out = condor_strikes_from_smile(5800.0, 30.0, 60)
        # σ at 30% IV over 60min ≈ 5800 * 0.30 * sqrt(60/525600) ≈ 18.6 pts
        # → raw short_put ≈ 5781.4, raw short_call ≈ 5818.6
        # floor(5781.4 / 5) * 5 = 5780
        # ceil(5818.6 / 5) * 5 = 5820
        self.assertEqual(out["short_put"], 5780.0)
        self.assertEqual(out["short_call"], 5820.0)
        self.assertEqual(out["long_put"], 5770.0)
        self.assertEqual(out["long_call"], 5830.0)

    def test_custom_wing_and_increment(self):
        # 25-wide wings on 10-pt grid (wider underlying scenario).
        # The wing math rounds the requested width onto the grid, so the
        # actual wing distance can land at 20 or 30; the invariant we
        # care about is that both wings stay symmetric and on-grid.
        out = condor_strikes_from_smile(
            5800.0, 15.0, 120,
            wing_width_pts=25.0,
            strike_increment=10.0,
        )
        for k in ("short_put", "long_put", "short_call", "long_call"):
            self.assertEqual(out[k] % 10, 0, f"{k} not on 10-pt grid: {out[k]}")
        wp = out["short_put"] - out["long_put"]
        wc = out["long_call"] - out["short_call"]
        self.assertEqual(wp, wc, "wings asymmetric")
        self.assertIn(wp, (20.0, 30.0))

    def test_invalid_inputs_return_none(self):
        self.assertIsNone(condor_strikes_from_smile(None, 15.0, 120))
        self.assertIsNone(condor_strikes_from_smile(5800.0, None, 120))
        self.assertIsNone(condor_strikes_from_smile(5800.0, 15.0, None))
        self.assertIsNone(condor_strikes_from_smile(5800.0, 15.0, 0))
        self.assertIsNone(condor_strikes_from_smile(5800.0, 0, 120))
        self.assertIsNone(condor_strikes_from_smile(-1.0, 15.0, 120))
        self.assertIsNone(condor_strikes_from_smile("not-a-number", 15.0, 120))

    def test_matches_backend_compute_hypothetical_condor(self):
        # Regression: the helper is shared with backend
        # _compute_hypothetical_condor (BacktestsV2/service.py). Same
        # math, same inputs → same strikes. Pre-refactor backend output
        # for spx=4940, iv=20%, horizon=120m, wing=10, inc=5 was:
        #   short_put_strike=4925, long_put_strike=4915,
        #   short_call_strike=4955, long_call_strike=4965
        out = condor_strikes_from_smile(4940.0, 20.0, 120)
        self.assertEqual(out["short_put"], 4925.0)
        self.assertEqual(out["long_put"], 4915.0)
        self.assertEqual(out["short_call"], 4955.0)
        self.assertEqual(out["long_call"], 4965.0)


if __name__ == "__main__":
    unittest.main()
