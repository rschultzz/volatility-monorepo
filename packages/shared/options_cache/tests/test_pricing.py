"""
Unit tests for build_condor_pricing_payload.

fetch_option_bars and repo.get_bars_for_contract are patched at the
pricing-module import site — no DB, no network.

Run with:
    python -m unittest packages.shared.options_cache.tests.test_pricing
"""
from __future__ import annotations

import unittest
from datetime import date, datetime
from unittest.mock import MagicMock, patch

from packages.shared.options_cache import pricing
from packages.shared.options_cache.http_client import OratsPermanentError
from packages.shared.options_cache.models import (
    FetchOptionBarsSummary,
    OptionMinuteBar,
)


def _bar(opra, snapshot_pt, bid, ask):
    """Minimum-viable OptionMinuteBar for the price-projection path."""
    return OptionMinuteBar(
        opra_symbol=opra,
        ticker=opra.replace("SPX", "SPX"),
        expir_date="2026-05-19",
        expir_date_d=date(2026, 5, 19),
        strike=5800.0,
        option_type="P" if "P" in opra else "C",
        trade_date="2026-05-19",
        trade_date_d=date(2026, 5, 19),
        quote_date="2026-05-19",
        snapshot_pt=snapshot_pt,
        snapshot_utc=snapshot_pt,
        bid_price=bid,
        ask_price=ask,
        delta=0.0,
    )


def _summary():
    return FetchOptionBarsSummary(
        opras_processed=1, gaps_filled=0, bars_written=0, cache_hits=1,
    )


class TestBuildCondorPricingPayload(unittest.TestCase):
    # iv=30%, mins=60 → sigma ≈ 18.6pts → strikes 5780/5770/5820/5830
    # (chosen because the round numbers make the assertions easy to read)
    BASE_KW = dict(
        trade_date="2026-05-19",
        expiration_date="2026-05-19",
        spx=5800.0,
        iv_pct=30.0,
        minutes_to_expiry=60.0,
        entry_pt="07:32",
        eval_pt="08:42",
    )

    @patch("packages.shared.options_cache.pricing.fetch_option_bars")
    @patch("packages.shared.options_cache.pricing.repo.get_bars_for_contract")
    def test_cache_hit_path_full_payload(self, mock_get_bars, mock_fetch):
        # Every leg returns a bar at both minutes. fetch_option_bars
        # reports cache_hit; the values come from get_bars_for_contract.
        mock_fetch.return_value = _summary()

        def get_bars(opra, start, end):
            # Distinct mid at entry vs eval so net_credit and
            # net_cost_to_close differ — proves P&L wiring.
            if start == datetime(2026, 5, 19, 7, 32):
                # Entry quotes: collect a net credit
                table = {
                    "SP": (1.20, 1.30),  # short put
                    "LP": (0.80, 0.90),  # long put
                    "SC": (1.10, 1.20),  # short call
                    "LC": (0.70, 0.80),  # long call
                }
            else:
                # Eval quotes: prices have decayed → cost to close is less
                table = {
                    "SP": (0.60, 0.70),
                    "LP": (0.40, 0.50),
                    "SC": (0.55, 0.65),
                    "LC": (0.35, 0.45),
                }
            if opra.endswith("P05780000"):
                bid, ask = table["SP"]
            elif opra.endswith("P05770000"):
                bid, ask = table["LP"]
            elif opra.endswith("C05820000"):
                bid, ask = table["SC"]
            elif opra.endswith("C05830000"):
                bid, ask = table["LC"]
            else:
                return []
            return [_bar(opra, start, bid, ask)]

        mock_get_bars.side_effect = get_bars

        payload, status = pricing.build_condor_pricing_payload(**self.BASE_KW)
        self.assertEqual(status, 200)
        self.assertEqual(payload["warnings"], [])
        self.assertIn("sigma_pts", payload)
        # All four strikes round-tripped
        self.assertEqual(payload["strikes"]["short_put"], 5780.0)
        self.assertEqual(payload["strikes"]["short_call"], 5820.0)
        # Net credit = mid(SP) + mid(SC) - mid(LP) - mid(LC)
        # entry mids: 1.25 + 1.15 - 0.85 - 0.75 = 0.80
        self.assertAlmostEqual(payload["entry"]["net_credit"], 0.80, places=4)
        # eval mids: 0.65 + 0.60 - 0.45 - 0.40 = 0.40
        self.assertAlmostEqual(payload["eval"]["net_cost_to_close"], 0.40, places=4)
        # P&L: 0.80 - 0.40 = 0.40
        self.assertAlmostEqual(payload["pnl"]["gross"], 0.40, places=4)

    @patch("packages.shared.options_cache.pricing.fetch_option_bars")
    @patch("packages.shared.options_cache.pricing.repo.get_bars_for_contract")
    def test_orats_fallback_path_invokes_fetch(self, mock_get_bars, mock_fetch):
        # First get_bars_for_contract has no row → still returns []
        # after fetch_option_bars is called (simulates fetch failing to
        # find data). The test asserts fetch_option_bars was invoked
        # once per leg per minute (8 total: 4 legs × 2 minutes).
        mock_fetch.return_value = _summary()
        mock_get_bars.return_value = []

        payload, status = pricing.build_condor_pricing_payload(**self.BASE_KW)
        self.assertEqual(status, 200)
        # 4 legs × 2 timepoints = 8 fetch calls
        self.assertEqual(mock_fetch.call_count, 8)
        # Every leg should appear in warnings (8 entries)
        self.assertEqual(len(payload["warnings"]), 8)
        # P&L can't be computed without mids
        self.assertIsNone(payload["entry"]["net_credit"])
        self.assertIsNone(payload["eval"]["net_cost_to_close"])
        self.assertIsNone(payload["pnl"]["gross"])

    @patch("packages.shared.options_cache.pricing.fetch_option_bars")
    @patch("packages.shared.options_cache.pricing.repo.get_bars_for_contract")
    def test_partial_failure_one_leg_404(self, mock_get_bars, mock_fetch):
        # Three legs fetch cleanly; the fourth (long_call) raises 404.
        # Payload should still come back 200 with a warning, the bad
        # leg as None mids, and the net_credit suppressed.
        def fetch(opras, start, end, *, source=None):
            if opras[0].endswith("C05830000"):  # long_call OPRA
                raise OratsPermanentError("404 from ORATS")
            return _summary()

        mock_fetch.side_effect = fetch
        # Other legs price normally
        def get_bars(opra, start, end):
            if opra.endswith("C05830000"):
                return []
            return [_bar(opra, start, 1.0, 1.2)]
        mock_get_bars.side_effect = get_bars

        payload, status = pricing.build_condor_pricing_payload(**self.BASE_KW)
        self.assertEqual(status, 200)
        # At least one warning per missing minute (entry + eval) for the
        # one bad leg → 2 warnings minimum
        self.assertGreaterEqual(len(payload["warnings"]), 2)
        self.assertIsNone(payload["entry"]["legs"]["long_call"]["mid"])
        self.assertIsNone(payload["pnl"]["gross"])
        # The other legs still have valid mids
        self.assertIsNotNone(payload["entry"]["legs"]["short_put"]["mid"])

    def test_invalid_trade_date_returns_400(self):
        payload, status = pricing.build_condor_pricing_payload(
            **{**self.BASE_KW, "trade_date": "not-a-date"}
        )
        self.assertEqual(status, 400)
        self.assertIn("error", payload)

    def test_invalid_smile_inputs_returns_400(self):
        payload, status = pricing.build_condor_pricing_payload(
            **{**self.BASE_KW, "iv_pct": 0.0}
        )
        self.assertEqual(status, 400)

    def test_eval_pt_now_snaps_to_close_when_session_past(self):
        # Pin now to 14:00 PT on the same trade date (past 13:00 close).
        # Expect eval_pt snapped to 12:59 PT and is_live=False.
        with patch("packages.shared.options_cache.pricing.fetch_option_bars",
                   return_value=_summary()), \
             patch("packages.shared.options_cache.pricing.repo.get_bars_for_contract",
                   return_value=[]):
            payload, status = pricing.build_condor_pricing_payload(
                **{**self.BASE_KW, "eval_pt": "now"},
                now_pt=datetime(2026, 5, 19, 14, 0),
            )
        self.assertEqual(status, 200)
        self.assertEqual(payload["eval"]["snapshot_pt"], "12:59")
        self.assertFalse(payload["eval"]["is_live"])

    def test_eval_pt_now_is_live_during_session(self):
        # Pin now to 09:42 PT mid-session → snapped to 09:41 (prior
        # completed minute) and is_live=True.
        with patch("packages.shared.options_cache.pricing.fetch_option_bars",
                   return_value=_summary()), \
             patch("packages.shared.options_cache.pricing.repo.get_bars_for_contract",
                   return_value=[]):
            payload, status = pricing.build_condor_pricing_payload(
                **{**self.BASE_KW, "eval_pt": "now"},
                now_pt=datetime(2026, 5, 19, 9, 42, 30),
            )
        self.assertEqual(status, 200)
        self.assertEqual(payload["eval"]["snapshot_pt"], "09:41")
        self.assertTrue(payload["eval"]["is_live"])


# ── price_proposal_legs (CR-T Step 1) ────────────────────────────────────────

def _proposal_bar(opra: str, snapshot_pt: datetime, bid: float, ask: float) -> OptionMinuteBar:
    """Minimal OptionMinuteBar for proposal pricing tests."""
    exp_date = date(2023, 8, 11)
    return OptionMinuteBar(
        opra_symbol=opra,
        ticker="SPX",
        expir_date=exp_date.isoformat(),
        expir_date_d=exp_date,
        strike=float(opra[-8:]) / 1000.0,
        option_type="C" if "C" in opra else "P",
        trade_date="2023-07-28",
        trade_date_d=date(2023, 7, 28),
        quote_date="2023-07-28",
        snapshot_pt=snapshot_pt,
        snapshot_utc=snapshot_pt,
        bid_price=bid,
        ask_price=ask,
        delta=0.0,
    )


_ENTRY_PT = datetime(2023, 7, 28, 7, 0)   # 07:00 PT naive
_EXPIR_D  = date(2023, 8, 11)
_TRADE_D  = date(2023, 7, 28)

_PROPOSAL_LEGS = [
    # Long call at SPX 4580 (ES ~4589 after carry)
    {"flag": "c", "strike": 4589.0, "expiration": _EXPIR_D, "qty": 1, "side": "long"},
    # Short call at SPX 4625 (ES ~4634 after carry)
    {"flag": "c", "strike": 4634.0, "expiration": _EXPIR_D, "qty": 1, "side": "short"},
]


def _mock_bars_for(opra: str) -> list:
    bid_map = {
        "SPX230811C04580000": (5.20, 5.40),
        "SPX230811C04625000": (2.10, 2.20),
    }
    if opra in bid_map:
        bid, ask = bid_map[opra]
        return [_proposal_bar(opra, _ENTRY_PT, bid, ask)]
    return []


class TestPriceProposalLegs(unittest.TestCase):
    """price_proposal_legs — mock-based unit tests (no DB, no network)."""

    def _run(self, legs, bars_fn=None):
        if bars_fn is None:
            bars_fn = _mock_bars_for
        with patch("packages.shared.options_cache.pricing.fetch_option_bars",
                   return_value=_summary()), \
             patch("packages.shared.options_cache.pricing.repo.get_bars_for_contract",
                   side_effect=lambda opra, *_: bars_fn(opra)):
            return pricing.price_proposal_legs(
                legs,
                trade_date=_TRADE_D,
                entry_pt=_ENTRY_PT,
                r=0.053,
                q=0.015,
            )

    # ── cache-hit path ────────────────────────────────────────────────────

    def test_cache_hit_returns_mids(self):
        result = self._run(_PROPOSAL_LEGS)
        self.assertEqual(len(result["legs"]), 2)
        long_leg  = result["legs"][0]
        short_leg = result["legs"][1]
        self.assertAlmostEqual(long_leg["mid"],  5.30, places=2)
        self.assertAlmostEqual(short_leg["mid"], 2.15, places=2)

    def test_cache_hit_net_debit_sign_convention(self):
        """Long mid − short mid = net debit (positive for a debit spread)."""
        result = self._run(_PROPOSAL_LEGS)
        expected = 1 * 5.30 - 1 * 2.15
        self.assertAlmostEqual(result["net_debit"], expected, places=2)

    def test_cache_hit_no_warnings(self):
        result = self._run(_PROPOSAL_LEGS)
        self.assertEqual(result["warnings"], [])

    def test_spx_strike_conversion(self):
        """ES strikes are converted to SPX strikes via compute_spx_strike."""
        result = self._run(_PROPOSAL_LEGS)
        # With dte=14, r=0.053, q=0.015 the conversion shifts ~0.9 points
        # (well within the 5pt rounding). Check strikes are multiples of 5.
        for leg in result["legs"]:
            self.assertEqual(leg["spx_strike"] % 5, 0)

    def test_opra_format_is_spx_root(self):
        """OPRAs must use SPX root for all SPX expirations."""
        result = self._run(_PROPOSAL_LEGS)
        for leg in result["legs"]:
            self.assertTrue(leg["opra"].startswith("SPX"), f"OPRA {leg['opra']} must use SPX root")

    # ── fetch-on-miss path ────────────────────────────────────────────────

    def test_fetch_called_on_miss(self):
        """fetch_option_bars is called even when cache is initially empty."""
        with patch("packages.shared.options_cache.pricing.fetch_option_bars",
                   return_value=_summary()) as mock_fetch, \
             patch("packages.shared.options_cache.pricing.repo.get_bars_for_contract",
                   side_effect=lambda opra, *_: _mock_bars_for(opra)):
            result = pricing.price_proposal_legs(
                _PROPOSAL_LEGS, trade_date=_TRADE_D, entry_pt=_ENTRY_PT,
            )
        mock_fetch.assert_called_once()
        # Batched: one call for all unique OPRAs
        call_args = mock_fetch.call_args
        self.assertIsInstance(call_args[0][0], list)

    # ── partial-404 path ──────────────────────────────────────────────────

    def test_partial_miss_produces_none_net_debit_and_warning(self):
        """One leg missing mid → net_debit = None, warning present."""
        def bars_one_miss(opra):
            # Only the long leg has data; short leg returns nothing
            if "C04580000" in opra:
                return [_proposal_bar(opra, _ENTRY_PT, 5.20, 5.40)]
            return []

        result = self._run(_PROPOSAL_LEGS, bars_fn=bars_one_miss)
        self.assertIsNone(result["net_debit"])
        self.assertTrue(len(result["warnings"]) > 0, "Missing leg must produce a warning")

    def test_partial_miss_still_returns_available_mids(self):
        """Available legs still have their mids even when one is missing."""
        def bars_one_miss(opra):
            if "C04580000" in opra:
                return [_proposal_bar(opra, _ENTRY_PT, 5.20, 5.40)]
            return []

        result = self._run(_PROPOSAL_LEGS, bars_fn=bars_one_miss)
        long_leg  = result["legs"][0]
        short_leg = result["legs"][1]
        self.assertAlmostEqual(long_leg["mid"],  5.30, places=2)
        self.assertIsNone(short_leg["mid"])

    def test_permanent_error_during_fetch_produces_warning(self):
        with patch("packages.shared.options_cache.pricing.fetch_option_bars",
                   side_effect=OratsPermanentError("404 illiquid")), \
             patch("packages.shared.options_cache.pricing.repo.get_bars_for_contract",
                   return_value=[]):
            result = pricing.price_proposal_legs(
                _PROPOSAL_LEGS, trade_date=_TRADE_D, entry_pt=_ENTRY_PT,
            )
        self.assertTrue(any("permanent error" in w for w in result["warnings"]))
        self.assertIsNone(result["net_debit"])

    def test_empty_legs_returns_empty(self):
        result = pricing.price_proposal_legs([], trade_date=_TRADE_D, entry_pt=_ENTRY_PT)
        self.assertEqual(result["legs"], [])
        self.assertIsNone(result["net_debit"])


# ── build_real_strike_band (CR-T Step 2) ─────────────────────────────────────

class TestBuildRealStrikeBand(unittest.TestCase):
    """build_real_strike_band — mock-based unit tests (no DB, no network)."""

    _SPOT = 4582.0
    _IM   = 50.0
    _TD   = date(2023, 7, 28)
    _EXPD = date(2023, 8, 11)
    _EP   = datetime(2023, 7, 28, 7, 0)

    def _run(self, bars_fn=None):
        def default_bars(opra, *_):
            # Return a tight quote for every call OPRA in the band
            if "C" in opra:
                strike_int = int(opra[-8:])
                mid = max(0.1, (self._SPOT + 10 - strike_int / 1000.0) * 0.1)
                bid = round(mid * 0.98, 2)
                ask = round(mid * 1.02, 2)
                exp_d = self._EXPD
                return [OptionMinuteBar(
                    opra_symbol=opra, ticker="SPX",
                    expir_date=exp_d.isoformat(), expir_date_d=exp_d,
                    strike=strike_int / 1000.0, option_type="C",
                    trade_date=str(self._TD), trade_date_d=self._TD,
                    quote_date=str(self._TD), snapshot_pt=self._EP,
                    snapshot_utc=self._EP, bid_price=bid, ask_price=ask, delta=0.0,
                )]
            return []

        fn = bars_fn or default_bars
        with patch("packages.shared.options_cache.pricing.fetch_option_bars",
                   return_value=_summary()), \
             patch("packages.shared.options_cache.pricing.repo.get_bars_for_contract",
                   side_effect=fn):
            return pricing.build_real_strike_band(
                self._SPOT, self._IM,
                expiration_date=self._EXPD,
                entry_pt=self._EP,
            )

    def test_band_covers_spot_plus_minus_1p5_im(self):
        """Band must span at least spot ± 1.5×IM."""
        chain = self._run()
        self.assertTrue(len(chain) > 0, "Band must return at least one strike")
        lo = min(d["strike"] for d in chain)
        hi = max(d["strike"] for d in chain)
        self.assertLessEqual(lo, self._SPOT - 1.5 * self._IM + 5)
        self.assertGreaterEqual(hi, self._SPOT + 1.5 * self._IM - 5)

    def test_band_strikes_in_5pt_increments(self):
        chain = self._run()
        for d in chain:
            self.assertEqual(d["strike"] % 5, 0, f"Strike {d['strike']} not a 5pt multiple")

    def test_returns_call_prices_as_mids(self):
        chain = self._run()
        for d in chain:
            self.assertIn("strike", d)
            self.assertIn("call_price", d)
            self.assertGreater(d["call_price"], 0)

    def test_partial_miss_omits_strike(self):
        """Strikes with no bar data are omitted from the result."""
        def sparse(opra, *_):
            # Only return bars for 4 strikes to trigger sparse-path scenario
            if opra.endswith(("C04580000", "C04585000", "C04590000", "C04595000")):
                exp_d = self._EXPD
                s = float(opra[-8:]) / 1000.0
                return [OptionMinuteBar(
                    opra_symbol=opra, ticker="SPX",
                    expir_date=exp_d.isoformat(), expir_date_d=exp_d,
                    strike=s, option_type="C",
                    trade_date=str(self._TD), trade_date_d=self._TD,
                    quote_date=str(self._TD), snapshot_pt=self._EP,
                    snapshot_utc=self._EP, bid_price=5.0, ask_price=5.2, delta=0.0,
                )]
            return []

        chain = self._run(bars_fn=sparse)
        # Should only contain the 4 matched strikes, not the full band
        self.assertLessEqual(len(chain), 4)

    def test_total_miss_returns_empty_list(self):
        chain = self._run(bars_fn=lambda opra, *_: [])
        self.assertEqual(chain, [])


if __name__ == "__main__":
    unittest.main()
