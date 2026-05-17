"""
Unit tests for the orchestrator. fetch_chain is mocked — no network/DB.

Run with:
    python -m unittest packages.shared.options_cache.tests.test_orchestrator
"""
from __future__ import annotations

import unittest
from datetime import datetime
from unittest.mock import patch

from packages.shared.options_cache.models import FetchSummary, TimeRange
from packages.shared.options_cache.orchestrator import (
    fetch_for_rows,
    get_strategy,
)


def _make_row(trade_date="2024-02-02", target_iso="2024-02-02T17:14:00Z", **extra):
    """Minimal scan row for orchestrator tests."""
    row = {
        "trade_date": trade_date,
        "target_ts_utc": target_iso,
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
    row.update(extra)
    return row


def _fake_summary(ticker, start, end):
    return FetchSummary(
        ticker=ticker,
        time_range=TimeRange(start_pt=start, end_pt=end),
        api_calls=1,
        rows_received=4000,
        rows_kept=1000,
        bars_inserted=2000,
        bars_total=2000,
        contracts_touched=1000,
    )


class TestFetchForRows(unittest.TestCase):
    @patch("packages.shared.options_cache.orchestrator.fetch_chain")
    def test_single_row_one_fetch(self, mock_fetch):
        mock_fetch.side_effect = lambda ticker, start_pt, end_pt, **kw: \
            _fake_summary(ticker, start_pt, end_pt)

        result = fetch_for_rows([_make_row()])

        self.assertEqual(result.rows_attempted, 1)
        self.assertEqual(result.rows_with_legs, 1)
        self.assertEqual(result.unique_windows_fetched, 1)
        self.assertEqual(mock_fetch.call_count, 1)

    @patch("packages.shared.options_cache.orchestrator.fetch_chain")
    def test_multiple_rows_same_day_one_fetch(self, mock_fetch):
        # Two condors on same trade date, both with same target time → one fetch
        mock_fetch.side_effect = lambda ticker, start_pt, end_pt, **kw: \
            _fake_summary(ticker, start_pt, end_pt)

        rows = [
            _make_row(target_iso="2024-02-02T17:14:00Z"),
            _make_row(target_iso="2024-02-02T17:14:00Z"),  # identical
        ]
        result = fetch_for_rows(rows)

        self.assertEqual(result.rows_attempted, 2)
        self.assertEqual(result.rows_with_legs, 2)
        # Both rows had same window → only 1 unique fetch
        self.assertEqual(result.unique_windows_fetched, 1)
        self.assertEqual(mock_fetch.call_count, 1)

    @patch("packages.shared.options_cache.orchestrator.fetch_chain")
    def test_multiple_rows_different_days_multiple_fetches(self, mock_fetch):
        mock_fetch.side_effect = lambda ticker, start_pt, end_pt, **kw: \
            _fake_summary(ticker, start_pt, end_pt)

        rows = [
            _make_row(trade_date="2024-02-02", target_iso="2024-02-02T17:14:00Z"),
            _make_row(trade_date="2024-02-13", target_iso="2024-02-13T15:23:00Z"),
        ]
        result = fetch_for_rows(rows)

        self.assertEqual(result.unique_windows_fetched, 2)
        self.assertEqual(mock_fetch.call_count, 2)

    @patch("packages.shared.options_cache.orchestrator.fetch_chain")
    def test_row_with_missing_data_skipped(self, mock_fetch):
        mock_fetch.side_effect = lambda ticker, start_pt, end_pt, **kw: \
            _fake_summary(ticker, start_pt, end_pt)

        # Use different timestamps so the rows have distinct row_keys
        rows = [
            _make_row(target_iso="2024-02-02T17:14:00Z"),  # valid
            _make_row(
                target_iso="2024-02-02T17:30:00Z",
                target_spx_price=None,                      # bad: no spot
            ),
        ]
        result = fetch_for_rows(rows)

        self.assertEqual(result.rows_attempted, 2)
        self.assertEqual(result.rows_with_legs, 1)
        # Only the bad row has an error
        failed = [r for r in result.rows if r.error]
        self.assertEqual(len(failed), 1)

    @patch("packages.shared.options_cache.orchestrator.fetch_chain")
    def test_fetch_chain_failure_marks_affected_rows(self, mock_fetch):
        from packages.shared.options_cache.http_client import OratsTransientError
        mock_fetch.side_effect = OratsTransientError("server down")

        # Use different timestamps to give rows distinct row_keys
        rows = [
            _make_row(target_iso="2024-02-02T17:14:00Z"),
            _make_row(target_iso="2024-02-02T17:30:00Z"),
        ]
        result = fetch_for_rows(rows)

        self.assertEqual(result.rows_with_legs, 2)
        self.assertEqual(result.unique_windows_fetched, 0)
        # Both rows should have an error recorded
        for r in result.rows:
            self.assertIsNotNone(r.error)
            self.assertIn("fetch_chain failed", r.error)

    def test_unknown_strategy_raises(self):
        with self.assertRaises(ValueError):
            fetch_for_rows([_make_row()], strategy="butterfly")


class TestGetStrategy(unittest.TestCase):
    def test_condor_registered(self):
        s = get_strategy("condor")
        self.assertEqual(s.name, "condor")

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            get_strategy("does_not_exist")


if __name__ == "__main__":
    unittest.main()
