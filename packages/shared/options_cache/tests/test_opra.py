"""
Unit tests for OPRA symbol formatting/parsing.

No DB required. Run with:
    python -m unittest packages.shared.options_cache.tests.test_opra
"""
from __future__ import annotations

import unittest
from datetime import date

from packages.shared.options_cache.opra import (
    OpraSymbol,
    format_opra,
    opra_to_orats_ticker,
    parse_opra,
)


class TestFormatOpra(unittest.TestCase):
    def test_spxw_put(self):
        # SPXW Jan 17 2026 5800 Put
        s = format_opra("SPXW", date(2026, 1, 17), "P", 5800.0)
        self.assertEqual(s, "SPXW260117P05800000")

    def test_spx_call(self):
        # SPX Mar 20 2026 5750 Call (monthly AM-settled)
        s = format_opra("SPX", date(2026, 3, 20), "C", 5750.0)
        self.assertEqual(s, "SPX260320C05750000")

    def test_equity_decimal_strike(self):
        # AAPL Jun 19 2026 210.5 Call
        s = format_opra("AAPL", date(2026, 6, 19), "C", 210.5)
        self.assertEqual(s, "AAPL260619C00210500")

    def test_root_normalized_to_upper(self):
        s = format_opra("spxw", date(2026, 1, 17), "P", 5800.0)
        self.assertEqual(s, "SPXW260117P05800000")

    def test_root_strips_whitespace(self):
        s = format_opra("  SPX  ", date(2026, 3, 20), "C", 5750.0)
        self.assertEqual(s, "SPX260320C05750000")

    def test_invalid_root_empty(self):
        with self.assertRaises(ValueError):
            format_opra("", date(2026, 1, 17), "P", 5800.0)

    def test_invalid_root_too_long(self):
        with self.assertRaises(ValueError):
            format_opra("TOOLONG", date(2026, 1, 17), "P", 5800.0)

    def test_invalid_root_non_alpha(self):
        with self.assertRaises(ValueError):
            format_opra("SPX1", date(2026, 1, 17), "P", 5800.0)

    def test_invalid_option_type(self):
        with self.assertRaises(ValueError):
            format_opra("SPX", date(2026, 1, 17), "X", 5800.0)  # type: ignore

    def test_invalid_negative_strike(self):
        with self.assertRaises(ValueError):
            format_opra("SPX", date(2026, 1, 17), "P", -1.0)

    def test_invalid_strike_too_large(self):
        with self.assertRaises(ValueError):
            format_opra("SPX", date(2026, 1, 17), "P", 100_000.0)

    def test_strike_rounding_handles_float_imprecision(self):
        # 210.5 is fine, but 0.1 + 0.2 = 0.30000...4 — make sure rounding
        # doesn't bite us
        s = format_opra("X", date(2026, 1, 17), "C", 0.3)
        self.assertEqual(s, "X260117C00000300")


class TestParseOpra(unittest.TestCase):
    def test_spxw_put(self):
        sym = parse_opra("SPXW260117P05800000")
        self.assertEqual(sym.root, "SPXW")
        self.assertEqual(sym.expir, date(2026, 1, 17))
        self.assertEqual(sym.option_type, "P")
        self.assertEqual(sym.strike, 5800.0)

    def test_equity_decimal(self):
        sym = parse_opra("AAPL260619C00210500")
        self.assertEqual(sym.root, "AAPL")
        self.assertEqual(sym.expir, date(2026, 6, 19))
        self.assertEqual(sym.option_type, "C")
        self.assertEqual(sym.strike, 210.5)

    def test_case_insensitive_input(self):
        sym = parse_opra("spxw260117p05800000")
        self.assertEqual(sym.root, "SPXW")

    def test_strips_whitespace(self):
        sym = parse_opra("  SPXW260117P05800000  ")
        self.assertEqual(sym.root, "SPXW")

    def test_year_pivot_pre_70(self):
        sym = parse_opra("SPX250117P05800000")
        self.assertEqual(sym.expir.year, 2025)

    def test_year_pivot_post_70(self):
        sym = parse_opra("SPX700117P05800000")
        self.assertEqual(sym.expir.year, 1970)

    def test_malformed_raises(self):
        with self.assertRaises(ValueError):
            parse_opra("not_a_symbol")

    def test_malformed_short_strike(self):
        with self.assertRaises(ValueError):
            parse_opra("SPX260117P0580000")  # 7 digit strike


class TestRoundTrip(unittest.TestCase):
    """Format -> parse should give back what we put in."""

    def test_round_trip_condor_legs(self):
        # Realistic SPX condor
        legs = [
            ("SPXW", date(2026, 1, 17), "P", 5750.0),
            ("SPXW", date(2026, 1, 17), "P", 5800.0),
            ("SPXW", date(2026, 1, 17), "C", 5900.0),
            ("SPXW", date(2026, 1, 17), "C", 5950.0),
        ]
        for root, expir, opt_type, strike in legs:
            sym = format_opra(root, expir, opt_type, strike)  # type: ignore
            parsed = parse_opra(sym)
            self.assertEqual(parsed.root, root)
            self.assertEqual(parsed.expir, expir)
            self.assertEqual(parsed.option_type, opt_type)
            self.assertEqual(parsed.strike, strike)


class TestOpraToOratsTicker(unittest.TestCase):
    """CR-004: strip the side character for ORATS' option-endpoint ticker param."""

    def test_spx_put(self):
        self.assertEqual(
            opra_to_orats_ticker("SPX240202P04935000"),
            "SPX24020204935000",
        )

    def test_spx_call(self):
        self.assertEqual(
            opra_to_orats_ticker("SPX240202C04935000"),
            "SPX24020204935000",
        )

    def test_both_sides_collide_by_design(self):
        # Put and call at the same strike+expir map to the same ORATS ticker.
        put_t = opra_to_orats_ticker("SPX240202P04935000")
        call_t = opra_to_orats_ticker("SPX240202C04935000")
        self.assertEqual(put_t, call_t)

    def test_3char_root_spy(self):
        self.assertEqual(
            opra_to_orats_ticker("SPY240202P00493500"),
            "SPY24020200493500",
        )

    def test_4char_root_spxw(self):
        self.assertEqual(
            opra_to_orats_ticker("SPXW260117P05800000"),
            "SPXW26011705800000",
        )

    def test_4char_root_aapl_decimal_strike(self):
        self.assertEqual(
            opra_to_orats_ticker("AAPL260619C00210500"),
            "AAPL26061900210500",
        )

    def test_malformed_input_raises(self):
        with self.assertRaises(ValueError):
            opra_to_orats_ticker("NOT_AN_OPRA")


if __name__ == "__main__":
    unittest.main()
