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


if __name__ == "__main__":
    unittest.main()
