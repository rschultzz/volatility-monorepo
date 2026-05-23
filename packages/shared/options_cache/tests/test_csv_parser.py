"""
Unit tests for the ORATS CSV parser. No network/DB.

Run with:
    python -m unittest packages.shared.options_cache.tests.test_csv_parser
"""
from __future__ import annotations

import unittest
from datetime import date, datetime, timezone

from packages.shared.options_cache.csv_parser import parse_orats_csv
from packages.shared.options_cache.models import ChainFilter


_FIXTURE_HEADER = (
    "ticker,tradeDate,expirDate,dte,strike,stockPrice,"
    "callVolume,callOpenInterest,callBidSize,callAskSize,"
    "putVolume,putOpenInterest,putBidSize,putAskSize,"
    "callBidPrice,callValue,callAskPrice,"
    "putBidPrice,putValue,putAskPrice,"
    "callBidIv,callMidIv,callAskIv,"
    "smvVol,"
    "putBidIv,putMidIv,putAskIv,"
    "residualRate,delta,gamma,theta,vega,rho,phi,driftlessTheta,"
    "callSmvVol,putSmvVol,extSmvVol,extCallValue,extPutValue,"
    "spotPrice,quoteDate,updatedAt,snapShotEstTime,snapShotDate,"
    "expiryTod,tickerId,monthId"
)


def _make_row(**overrides) -> str:
    """Build one CSV data row with sensible defaults."""
    defaults = {
        "ticker": "SPY",
        "tradeDate": "2024-02-02",
        "expirDate": "2024-02-02",
        "dte": "0",
        "strike": "493.5",
        "stockPrice": "493.32",
        "callVolume": "100", "callOpenInterest": "500",
        "callBidSize": "5", "callAskSize": "5",
        "putVolume": "150", "putOpenInterest": "800",
        "putBidSize": "10", "putAskSize": "10",
        "callBidPrice": "0.85", "callValue": "0.90", "callAskPrice": "0.95",
        "putBidPrice": "1.00", "putValue": "1.05", "putAskPrice": "1.10",
        "callBidIv": "0.180", "callMidIv": "0.182", "callAskIv": "0.184",
        "smvVol": "0.181",
        "putBidIv": "0.179", "putMidIv": "0.181", "putAskIv": "0.183",
        "residualRate": "0.0",
        "delta": "0.50",
        "gamma": "0.012", "theta": "-2.5", "vega": "1.2",
        "rho": "0.05", "phi": "-0.04", "driftlessTheta": "-2.4",
        "callSmvVol": "0.182", "putSmvVol": "0.181", "extSmvVol": "0.181",
        "extCallValue": "0.91", "extPutValue": "1.06",
        "spotPrice": "493.32",
        "quoteDate": "2024-02-02T17:14:00Z",
        "updatedAt": "2024-02-02T17:14:01Z",
        "snapShotEstTime": "1214",
        "snapShotDate": "2024-02-02T17:14:00Z",
        "expiryTod": "pm", "tickerId": "1", "monthId": "100",
    }
    defaults.update(overrides)
    cols = _FIXTURE_HEADER.split(",")
    return ",".join(str(defaults[c]) for c in cols)


def _make_csv(*rows: str) -> str:
    return "\n".join([_FIXTURE_HEADER, *rows])


class TestParseOratsCsv(unittest.TestCase):
    def test_empty_csv_returns_empty_tuple(self):
        bars, recv, kept = parse_orats_csv("")
        self.assertEqual(bars, [])
        self.assertEqual(recv, 0)
        self.assertEqual(kept, 0)

    def test_header_only_returns_empty_tuple(self):
        bars, recv, kept = parse_orats_csv(_FIXTURE_HEADER)
        self.assertEqual(bars, [])
        self.assertEqual(recv, 0)
        self.assertEqual(kept, 0)

    def test_one_row_yields_two_bars(self):
        bars, recv, kept = parse_orats_csv(_make_csv(_make_row()))
        self.assertEqual(len(bars), 2)
        self.assertEqual(recv, 1)
        self.assertEqual(kept, 1)

        call, put = bars
        self.assertEqual(call.option_type, "C")
        self.assertEqual(put.option_type, "P")
        self.assertEqual(call.opra_symbol, "SPY240202C00493500")
        self.assertEqual(put.opra_symbol, "SPY240202P00493500")

        # Side-specific bid/ask
        self.assertEqual(call.bid_price, 0.85)
        self.assertEqual(call.ask_price, 0.95)
        self.assertEqual(put.bid_price, 1.00)
        self.assertEqual(put.ask_price, 1.10)

        # Greeks: call uses ORATS values directly; put delta = call - 1
        self.assertEqual(call.delta, 0.50)
        self.assertAlmostEqual(put.delta, -0.50, places=6)

    def test_timestamp_pacific_conversion(self):
        # snapShotDate 17:14 UTC = 09:14 PT (Feb, no DST)
        bars, _, _ = parse_orats_csv(_make_csv(_make_row()))
        call = bars[0]
        self.assertEqual(call.snapshot_pt, datetime(2024, 2, 2, 9, 14))
        self.assertEqual(
            call.snapshot_utc,
            datetime(2024, 2, 2, 17, 14, tzinfo=timezone.utc),
        )

    def test_html_response_raises(self):
        with self.assertRaises(ValueError):
            parse_orats_csv("<html><body>error</body></html>")

    def test_missing_required_column_raises(self):
        bad_header = ",".join(
            c for c in _FIXTURE_HEADER.split(",") if c != "callBidPrice"
        )
        with self.assertRaises(ValueError):
            parse_orats_csv(bad_header + "\n" + _make_row())

    def test_expected_ticker_match(self):
        bars, _, _ = parse_orats_csv(
            _make_csv(_make_row()), expected_ticker="SPY"
        )
        self.assertEqual(len(bars), 2)

    def test_expected_ticker_mismatch_raises(self):
        with self.assertRaises(ValueError):
            parse_orats_csv(_make_csv(_make_row()), expected_ticker="AAPL")

    def test_null_optional_greeks_become_none(self):
        bars, _, _ = parse_orats_csv(_make_csv(_make_row(rho="", phi="")))
        self.assertEqual(len(bars), 2)
        call, put = bars
        self.assertIsNone(call.rho)
        self.assertIsNone(call.phi)
        self.assertIsNone(put.rho)  # always None for puts

    def test_row_missing_required_value_skipped(self):
        good_row = _make_row()
        bad_row = _make_row(delta="")
        bars, recv, kept = parse_orats_csv(_make_csv(good_row, bad_row))
        # Two rows received, only one kept (yielded 2 bars)
        self.assertEqual(recv, 2)
        self.assertEqual(kept, 1)
        self.assertEqual(len(bars), 2)


class TestChainFilter(unittest.TestCase):
    """Tests for the chain_filter parameter."""

    def test_no_filter_keeps_everything(self):
        rows = [
            _make_row(strike="450.0", dte="0"),    # OTM put, 0DTE
            _make_row(strike="493.5", dte="0"),    # ATM, 0DTE
            _make_row(strike="540.0", dte="0"),    # OTM call, 0DTE
            _make_row(strike="493.5", dte="120"),  # far DTE
        ]
        bars, recv, kept = parse_orats_csv(_make_csv(*rows), chain_filter=None)
        self.assertEqual(recv, 4)
        self.assertEqual(kept, 4)
        self.assertEqual(len(bars), 8)

    def test_strike_filter_drops_out_of_range(self):
        # Spot is 493.32; ±10% = 444 to 542.65
        rows = [
            _make_row(strike="450.0", dte="0"),    # in range
            _make_row(strike="493.5", dte="0"),    # in range
            _make_row(strike="540.0", dte="0"),    # in range
            _make_row(strike="400.0", dte="0"),    # OUT (too low)
            _make_row(strike="600.0", dte="0"),    # OUT (too high)
        ]
        f = ChainFilter(min_strike_pct_of_spot=0.90, max_strike_pct_of_spot=1.10)
        bars, recv, kept = parse_orats_csv(_make_csv(*rows), chain_filter=f)
        self.assertEqual(recv, 5)
        self.assertEqual(kept, 3)
        self.assertEqual(len(bars), 6)

    def test_dte_filter_drops_out_of_range(self):
        rows = [
            _make_row(dte="0"),     # in range
            _make_row(dte="30"),    # in range
            _make_row(dte="60"),    # in range (boundary)
            _make_row(dte="61"),    # OUT
            _make_row(dte="120"),   # OUT
        ]
        f = ChainFilter(
            min_strike_pct_of_spot=0.50,
            max_strike_pct_of_spot=2.00,
            min_dte=0, max_dte=60,
        )
        bars, recv, kept = parse_orats_csv(_make_csv(*rows), chain_filter=f)
        self.assertEqual(recv, 5)
        self.assertEqual(kept, 3)

    def test_combined_filters(self):
        rows = [
            _make_row(strike="493.5", dte="0"),    # in
            _make_row(strike="493.5", dte="120"),  # OUT (dte)
            _make_row(strike="700.0", dte="0"),    # OUT (strike)
            _make_row(strike="493.5", dte="30"),   # in
        ]
        f = ChainFilter()  # default ±10%, 0-60 DTE
        bars, recv, kept = parse_orats_csv(_make_csv(*rows), chain_filter=f)
        self.assertEqual(recv, 4)
        self.assertEqual(kept, 2)

    def test_filter_with_missing_dte_drops_row(self):
        # Without DTE info we can't filter — drop conservatively
        rows = [_make_row(dte="")]
        f = ChainFilter()
        bars, recv, kept = parse_orats_csv(_make_csv(*rows), chain_filter=f)
        self.assertEqual(kept, 0)

    def test_filter_with_zero_spot_drops_row(self):
        rows = [_make_row(stockPrice="0")]
        f = ChainFilter()
        bars, recv, kept = parse_orats_csv(_make_csv(*rows), chain_filter=f)
        self.assertEqual(kept, 0)


class TestChainFilterDirect(unittest.TestCase):
    """Tests of ChainFilter.passes() in isolation (no CSV)."""

    def test_default_passes_atm(self):
        f = ChainFilter()
        self.assertTrue(f.passes(strike=493.5, dte=0, stock_price=493.32))

    def test_default_rejects_far_otm(self):
        f = ChainFilter()
        self.assertFalse(f.passes(strike=300.0, dte=0, stock_price=493.32))

    def test_default_rejects_long_dte(self):
        f = ChainFilter()
        self.assertFalse(f.passes(strike=493.5, dte=365, stock_price=493.32))

    def test_missing_dte_fails(self):
        f = ChainFilter()
        self.assertFalse(f.passes(strike=493.5, dte=None, stock_price=493.32))

    def test_missing_spot_fails(self):
        f = ChainFilter()
        self.assertFalse(f.passes(strike=493.5, dte=0, stock_price=None))


if __name__ == "__main__":
    unittest.main()
