"""
Unit tests for CR-004: fetch_option_bars and _fetch_option_bars_from_orats.

No network, no DB. URL-construction tests mock get_csv at fetcher's import
boundary; cache-aware orchestration tests mock the private primitive and
the repository alias.

Run with:
    python -m unittest packages.shared.options_cache.tests.test_fetcher
"""
from __future__ import annotations

import unittest
from datetime import date, datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

from packages.shared.options_cache.fetcher import (
    _OPTION_PATH,
    _fetch_option_bars_from_orats,
    fetch_option_bars,
)
from packages.shared.options_cache.models import (
    FetchedWindow,
    OptionMinuteBar,
)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _make_bar(opra: str, when_pt: datetime, option_type: str = "P") -> OptionMinuteBar:
    """Minimal OptionMinuteBar for asserting fetcher behavior."""
    return OptionMinuteBar(
        opra_symbol=opra,
        ticker="SPX",
        expir_date="2024-02-02",
        expir_date_d=date(2024, 2, 2),
        strike=4935.0,
        option_type=option_type,
        trade_date="2024-02-02",
        trade_date_d=date(2024, 2, 2),
        quote_date="2024-02-02 09:30",
        snapshot_pt=when_pt,
        snapshot_utc=when_pt,
        bid_price=1.0,
        ask_price=1.5,
        delta=0.5 if option_type == "C" else -0.5,
    )


def _window(opra: str, start: datetime, end: datetime) -> FetchedWindow:
    return FetchedWindow(
        opra_symbol=opra,
        window_start_pt=start,
        window_end_pt=end,
        row_count=1,
        source="historical_backfill",
    )


# ─────────────────────────────────────────────────────────────────────
# _fetch_option_bars_from_orats — URL construction
# ─────────────────────────────────────────────────────────────────────

class TestFetchOptionBarsFromOratsUrl(unittest.TestCase):
    """Mock get_csv at fetcher's import boundary, assert call args."""

    @patch("packages.shared.options_cache.fetcher.parse_orats_csv")
    @patch("packages.shared.options_cache.fetcher.get_csv")
    def test_range_url_construction(self, mock_get_csv, mock_parse):
        mock_get_csv.return_value = "<csv-body>"
        mock_parse.return_value = ([], 0, 0)

        _fetch_option_bars_from_orats(
            "SPX240202P04935000",
            datetime(2024, 2, 2, 9, 30),
            datetime(2024, 2, 2, 10, 0),
        )

        mock_get_csv.assert_called_once()
        path, params = mock_get_csv.call_args.args
        self.assertEqual(path, _OPTION_PATH)
        # ORATS option endpoint requires the side-stripped form (no C|P).
        # Sending the full canonical OPRA returns 404.
        self.assertEqual(params["ticker"], "SPX24020204935000")
        # 09:30 PT == 12:30 ET; 10:00 PT == 13:00 ET. Comma-joined for range.
        self.assertEqual(params["tradeDate"], "202402021230,202402021300")

    @patch("packages.shared.options_cache.fetcher.parse_orats_csv")
    @patch("packages.shared.options_cache.fetcher.get_csv")
    def test_single_timestamp_url_construction(self, mock_get_csv, mock_parse):
        mock_get_csv.return_value = "<csv-body>"
        mock_parse.return_value = ([], 0, 0)

        ts = datetime(2024, 2, 2, 9, 30)
        _fetch_option_bars_from_orats("SPX240202P04935000", ts, ts)

        _, params = mock_get_csv.call_args.args
        self.assertEqual(params["tradeDate"], "202402021230")


# ─────────────────────────────────────────────────────────────────────
# _fetch_option_bars_from_orats — response handling
# ─────────────────────────────────────────────────────────────────────

class TestFetchOptionBarsFromOratsResponse(unittest.TestCase):
    @patch("packages.shared.options_cache.fetcher.parse_orats_csv")
    @patch("packages.shared.options_cache.fetcher.get_csv")
    def test_returns_all_bars_including_counterpart(self, mock_get_csv, mock_parse):
        mock_get_csv.return_value = "<csv-body>"
        when = datetime(2024, 2, 2, 9, 30)
        # csv_parser emits both call and put bars per row — mirror that.
        mock_parse.return_value = (
            [_make_bar("SPX240202C04935000", when, "C"),
             _make_bar("SPX240202P04935000", when, "P")],
            1, 1,
        )

        bars = _fetch_option_bars_from_orats("SPX240202P04935000", when, when)

        self.assertEqual(len(bars), 2)
        self.assertEqual({b.option_type for b in bars}, {"C", "P"})

    @patch("packages.shared.options_cache.fetcher.parse_orats_csv")
    @patch("packages.shared.options_cache.fetcher.get_csv")
    def test_empty_response_returns_empty_list(self, mock_get_csv, mock_parse):
        mock_get_csv.return_value = ""
        mock_parse.return_value = ([], 0, 0)

        result = _fetch_option_bars_from_orats(
            "SPX240202P04935000",
            datetime(2024, 2, 2, 9, 30),
            datetime(2024, 2, 2, 9, 30),
        )
        self.assertEqual(result, [])

    @patch("packages.shared.options_cache.fetcher.get_csv")
    def test_transient_http_error_bubbles(self, mock_get_csv):
        from packages.shared.options_cache.http_client import OratsTransientError
        mock_get_csv.side_effect = OratsTransientError("rate limited")

        with self.assertRaises(OratsTransientError):
            _fetch_option_bars_from_orats(
                "SPX240202P04935000",
                datetime(2024, 2, 2, 9, 30),
                datetime(2024, 2, 2, 9, 30),
            )

    @patch("packages.shared.options_cache.fetcher.get_csv")
    def test_permanent_http_error_bubbles(self, mock_get_csv):
        from packages.shared.options_cache.http_client import OratsPermanentError
        mock_get_csv.side_effect = OratsPermanentError("401 unauthorized")

        with self.assertRaises(OratsPermanentError):
            _fetch_option_bars_from_orats(
                "SPX240202P04935000",
                datetime(2024, 2, 2, 9, 30),
                datetime(2024, 2, 2, 9, 30),
            )


# ─────────────────────────────────────────────────────────────────────
# fetch_option_bars — cache-aware orchestration
# ─────────────────────────────────────────────────────────────────────

class TestFetchOptionBars(unittest.TestCase):
    def setUp(self):
        self.put_opra = "SPX240202P04935000"
        self.call_opra = "SPX240202C04935000"
        self.start_pt = datetime(2024, 2, 2, 9, 30)
        self.end_pt = datetime(2024, 2, 2, 10, 0)

    @patch("packages.shared.options_cache.fetcher.repo")
    @patch("packages.shared.options_cache.fetcher._fetch_option_bars_from_orats")
    def test_empty_cache_single_opra(self, mock_primitive, mock_repo):
        mock_repo.get_windows_for_contract.return_value = []
        mock_primitive.return_value = [
            _make_bar(self.call_opra, self.start_pt, "C"),
            _make_bar(self.put_opra, self.start_pt, "P"),
        ]
        mock_repo.insert_bars.return_value = 2

        result = fetch_option_bars(
            [self.put_opra], self.start_pt, self.end_pt,
        )

        mock_primitive.assert_called_once_with(
            self.put_opra, self.start_pt, self.end_pt,
        )
        recorded_opras = {
            call.args[0].opra_symbol
            for call in mock_repo.record_fetched_window.call_args_list
        }
        self.assertEqual(recorded_opras, {self.put_opra, self.call_opra})
        self.assertEqual(result.opras_processed, 1)
        self.assertEqual(result.gaps_filled, 1)
        self.assertEqual(result.bars_written, 2)
        self.assertEqual(result.cache_hits, 0)

    @patch("packages.shared.options_cache.fetcher.repo")
    @patch("packages.shared.options_cache.fetcher._fetch_option_bars_from_orats")
    def test_empty_cache_multi_opra(self, mock_primitive, mock_repo):
        mock_repo.get_windows_for_contract.return_value = []
        opra_A = "SPX240202P04935000"
        opra_B = "SPX240202P04940000"
        counterpart_A = "SPX240202C04935000"
        counterpart_B = "SPX240202C04940000"

        def primitive_side_effect(opra, *args, **kwargs):
            if opra == opra_A:
                return [_make_bar(counterpart_A, self.start_pt, "C"),
                        _make_bar(opra_A, self.start_pt, "P")]
            return [_make_bar(counterpart_B, self.start_pt, "C"),
                    _make_bar(opra_B, self.start_pt, "P")]

        mock_primitive.side_effect = primitive_side_effect
        mock_repo.insert_bars.return_value = 2

        result = fetch_option_bars(
            [opra_A, opra_B], self.start_pt, self.end_pt,
        )

        self.assertEqual(mock_primitive.call_count, 2)
        recorded_opras = {
            call.args[0].opra_symbol
            for call in mock_repo.record_fetched_window.call_args_list
        }
        self.assertEqual(
            recorded_opras,
            {opra_A, counterpart_A, opra_B, counterpart_B},
        )
        self.assertEqual(result.opras_processed, 2)
        self.assertEqual(result.gaps_filled, 2)
        self.assertEqual(result.bars_written, 4)
        self.assertEqual(result.cache_hits, 0)

    @patch("packages.shared.options_cache.fetcher.repo")
    @patch("packages.shared.options_cache.fetcher._fetch_option_bars_from_orats")
    def test_fully_cached_single_opra(self, mock_primitive, mock_repo):
        mock_repo.get_windows_for_contract.return_value = [
            _window(self.put_opra, self.start_pt, self.end_pt),
        ]

        result = fetch_option_bars(
            [self.put_opra], self.start_pt, self.end_pt,
        )

        mock_primitive.assert_not_called()
        mock_repo.insert_bars.assert_not_called()
        mock_repo.record_fetched_window.assert_not_called()
        self.assertEqual(result.cache_hits, 1)
        self.assertEqual(result.gaps_filled, 0)
        self.assertEqual(result.bars_written, 0)

    @patch("packages.shared.options_cache.fetcher.repo")
    @patch("packages.shared.options_cache.fetcher._fetch_option_bars_from_orats")
    def test_partial_overlap_at_start(self, mock_primitive, mock_repo):
        # Cache [09:35, 10:00], request [09:30, 10:00] → gap [09:30, 09:34]
        mock_repo.get_windows_for_contract.return_value = [
            _window(self.put_opra,
                    datetime(2024, 2, 2, 9, 35),
                    self.end_pt),
        ]
        mock_primitive.return_value = []
        mock_repo.insert_bars.return_value = 0

        fetch_option_bars([self.put_opra], self.start_pt, self.end_pt)

        mock_primitive.assert_called_once_with(
            self.put_opra,
            datetime(2024, 2, 2, 9, 30),
            datetime(2024, 2, 2, 9, 34),
        )

    @patch("packages.shared.options_cache.fetcher.repo")
    @patch("packages.shared.options_cache.fetcher._fetch_option_bars_from_orats")
    def test_partial_overlap_at_end(self, mock_primitive, mock_repo):
        # Cache [09:30, 09:45], request [09:30, 10:00] → gap [09:46, 10:00]
        mock_repo.get_windows_for_contract.return_value = [
            _window(self.put_opra,
                    self.start_pt,
                    datetime(2024, 2, 2, 9, 45)),
        ]
        mock_primitive.return_value = []
        mock_repo.insert_bars.return_value = 0

        fetch_option_bars([self.put_opra], self.start_pt, self.end_pt)

        mock_primitive.assert_called_once_with(
            self.put_opra,
            datetime(2024, 2, 2, 9, 46),
            datetime(2024, 2, 2, 10, 0),
        )

    @patch("packages.shared.options_cache.fetcher.repo")
    @patch("packages.shared.options_cache.fetcher._fetch_option_bars_from_orats")
    def test_gap_in_middle(self, mock_primitive, mock_repo):
        # Cache [09:30,09:40] ∪ [09:50,10:00], request [09:30,10:00]
        # → gap [09:41, 09:49]
        mock_repo.get_windows_for_contract.return_value = [
            _window(self.put_opra,
                    datetime(2024, 2, 2, 9, 30),
                    datetime(2024, 2, 2, 9, 40)),
            _window(self.put_opra,
                    datetime(2024, 2, 2, 9, 50),
                    datetime(2024, 2, 2, 10, 0)),
        ]
        mock_primitive.return_value = []
        mock_repo.insert_bars.return_value = 0

        fetch_option_bars([self.put_opra], self.start_pt, self.end_pt)

        mock_primitive.assert_called_once_with(
            self.put_opra,
            datetime(2024, 2, 2, 9, 41),
            datetime(2024, 2, 2, 9, 49),
        )

    @patch("packages.shared.options_cache.fetcher.repo")
    @patch("packages.shared.options_cache.fetcher._fetch_option_bars_from_orats")
    def test_adjacency_boundary_matches_find_gaps(self, mock_primitive, mock_repo):
        # [09:30,09:40] + [09:41,09:50] merge per find_gaps; gap [09:51, 10:00]
        # Mirrors test_windows.test_adjacent_windows_merge.
        mock_repo.get_windows_for_contract.return_value = [
            _window(self.put_opra,
                    datetime(2024, 2, 2, 9, 30),
                    datetime(2024, 2, 2, 9, 40)),
            _window(self.put_opra,
                    datetime(2024, 2, 2, 9, 41),
                    datetime(2024, 2, 2, 9, 50)),
        ]
        mock_primitive.return_value = []
        mock_repo.insert_bars.return_value = 0

        fetch_option_bars([self.put_opra], self.start_pt, self.end_pt)

        mock_primitive.assert_called_once_with(
            self.put_opra,
            datetime(2024, 2, 2, 9, 51),
            datetime(2024, 2, 2, 10, 0),
        )

    @patch("packages.shared.options_cache.fetcher.repo")
    @patch("packages.shared.options_cache.fetcher._fetch_option_bars_from_orats")
    def test_mixed_cache_states_across_opras(self, mock_primitive, mock_repo):
        opra_A = "SPX240202P04935000"
        opra_B = "SPX240202P04940000"

        def windows_side_effect(opra):
            if opra == opra_A:
                return [_window(opra_A, self.start_pt, self.end_pt)]
            return []
        mock_repo.get_windows_for_contract.side_effect = windows_side_effect

        mock_primitive.return_value = [
            _make_bar("SPX240202C04940000", self.start_pt, "C"),
            _make_bar(opra_B, self.start_pt, "P"),
        ]
        mock_repo.insert_bars.return_value = 2

        result = fetch_option_bars(
            [opra_A, opra_B], self.start_pt, self.end_pt,
        )

        mock_primitive.assert_called_once_with(opra_B, self.start_pt, self.end_pt)
        self.assertEqual(result.opras_processed, 2)
        self.assertEqual(result.cache_hits, 1)
        self.assertEqual(result.gaps_filled, 1)

    @patch("packages.shared.options_cache.fetcher.repo")
    @patch("packages.shared.options_cache.fetcher._fetch_option_bars_from_orats")
    def test_empty_response_records_only_requested_window(
        self, mock_primitive, mock_repo,
    ):
        mock_repo.get_windows_for_contract.return_value = []
        mock_primitive.return_value = []
        mock_repo.insert_bars.return_value = 0

        result = fetch_option_bars(
            [self.put_opra], self.start_pt, self.end_pt,
        )

        recorded = [
            call.args[0]
            for call in mock_repo.record_fetched_window.call_args_list
        ]
        self.assertEqual(len(recorded), 1)
        self.assertEqual(recorded[0].opra_symbol, self.put_opra)
        self.assertEqual(recorded[0].row_count, 0)
        self.assertEqual(recorded[0].window_start_pt, self.start_pt)
        self.assertEqual(recorded[0].window_end_pt, self.end_pt)
        self.assertEqual(result.bars_written, 0)

    @patch("packages.shared.options_cache.fetcher.repo")
    @patch("packages.shared.options_cache.fetcher._fetch_option_bars_from_orats")
    def test_counterpart_window_recorded(self, mock_primitive, mock_repo):
        # Request the call side; assert both call + put OPRAs get windows.
        mock_repo.get_windows_for_contract.return_value = []
        mock_primitive.return_value = [
            _make_bar(self.call_opra, self.start_pt, "C"),
            _make_bar(self.put_opra, self.start_pt, "P"),
        ]
        mock_repo.insert_bars.return_value = 2

        fetch_option_bars([self.call_opra], self.start_pt, self.end_pt)

        recorded_opras = {
            call.args[0].opra_symbol
            for call in mock_repo.record_fetched_window.call_args_list
        }
        self.assertEqual(recorded_opras, {self.call_opra, self.put_opra})

    @patch("packages.shared.options_cache.fetcher.repo")
    @patch("packages.shared.options_cache.fetcher._fetch_option_bars_from_orats")
    def test_post_write_state_correctness(self, mock_primitive, mock_repo):
        """One insert_bars call, two window records, all carrying gap bounds."""
        mock_repo.get_windows_for_contract.return_value = []
        mock_primitive.return_value = [
            _make_bar(self.call_opra, self.start_pt, "C"),
            _make_bar(self.put_opra, self.start_pt, "P"),
        ]
        mock_repo.insert_bars.return_value = 2

        fetch_option_bars([self.put_opra], self.start_pt, self.end_pt)

        mock_repo.insert_bars.assert_called_once()
        inserted = mock_repo.insert_bars.call_args.args[0]
        self.assertEqual(len(inserted), 2)

        windows = [
            call.args[0]
            for call in mock_repo.record_fetched_window.call_args_list
        ]
        self.assertEqual(len(windows), 2)
        for w in windows:
            self.assertEqual(w.window_start_pt, self.start_pt)
            self.assertEqual(w.window_end_pt, self.end_pt)
            self.assertEqual(w.row_count, 1)
            self.assertEqual(w.source, "historical_backfill")


# ─────────────────────────────────────────────────────────────────────
# Input validation
# ─────────────────────────────────────────────────────────────────────

class TestFetchOptionBarsValidation(unittest.TestCase):
    def test_tz_aware_start_rejected(self):
        with self.assertRaises(ValueError):
            fetch_option_bars(
                ["SPX240202P04935000"],
                datetime(2024, 2, 2, 9, 30, tzinfo=ZoneInfo("UTC")),
                datetime(2024, 2, 2, 10, 0),
            )

    def test_tz_aware_end_rejected(self):
        with self.assertRaises(ValueError):
            fetch_option_bars(
                ["SPX240202P04935000"],
                datetime(2024, 2, 2, 9, 30),
                datetime(2024, 2, 2, 10, 0, tzinfo=ZoneInfo("UTC")),
            )


if __name__ == "__main__":
    unittest.main()
