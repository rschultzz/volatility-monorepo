"""
Unit tests for the orchestrator. fetch_option_bars is mocked — no network/DB.

Run with:
    python -m unittest packages.shared.options_cache.tests.test_orchestrator
"""
from __future__ import annotations

import unittest
from datetime import datetime
from unittest.mock import patch

from packages.shared.options_cache.condor import Leg
from packages.shared.options_cache.models import FetchOptionBarsSummary
from packages.shared.options_cache.orchestrator import (
    Strategy,
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


def _fake_option_summary(opras_processed=1, gaps_filled=1, bars_written=500, cache_hits=0):
    return FetchOptionBarsSummary(
        opras_processed=opras_processed,
        gaps_filled=gaps_filled,
        bars_written=bars_written,
        cache_hits=cache_hits,
    )


class TestFetchForRows(unittest.TestCase):
    @patch("packages.shared.options_cache.orchestrator.fetch_option_bars")
    def test_single_row_one_fetch_per_leg(self, mock_fetch):
        mock_fetch.side_effect = lambda **kw: _fake_option_summary()

        result = fetch_for_rows([_make_row()])

        self.assertEqual(result.rows_attempted, 1)
        self.assertEqual(result.rows_with_legs, 1)
        # One condor row → 4 unique OPRAs (4 legs) → 4 fetch calls
        self.assertEqual(result.unique_opras_fetched, 4)
        self.assertEqual(mock_fetch.call_count, 4)

    @patch("packages.shared.options_cache.orchestrator.fetch_option_bars")
    def test_multiple_rows_shared_legs_dedups(self, mock_fetch):
        # Two condors on same trade date + same target time → identical legs
        # and identical windows. Dedup at the OPRA level → 4 fetches total.
        mock_fetch.side_effect = lambda **kw: _fake_option_summary()

        rows = [
            _make_row(target_iso="2024-02-02T17:14:00Z"),
            _make_row(target_iso="2024-02-02T17:14:00Z"),  # identical
        ]
        result = fetch_for_rows(rows)

        self.assertEqual(result.rows_attempted, 2)
        self.assertEqual(result.rows_with_legs, 2)
        self.assertEqual(result.unique_opras_fetched, 4)
        self.assertEqual(mock_fetch.call_count, 4)

    @patch("packages.shared.options_cache.orchestrator.fetch_option_bars")
    def test_multiple_rows_different_days_multiple_fetches(self, mock_fetch):
        mock_fetch.side_effect = lambda **kw: _fake_option_summary()

        rows = [
            _make_row(trade_date="2024-02-02", target_iso="2024-02-02T17:14:00Z"),
            _make_row(trade_date="2024-02-13", target_iso="2024-02-13T15:23:00Z"),
        ]
        result = fetch_for_rows(rows)

        # Different trade dates → different expirations → 8 unique OPRAs
        self.assertEqual(result.unique_opras_fetched, 8)
        self.assertEqual(mock_fetch.call_count, 8)

    @patch("packages.shared.options_cache.orchestrator.fetch_option_bars")
    def test_row_with_missing_data_skipped(self, mock_fetch):
        mock_fetch.side_effect = lambda **kw: _fake_option_summary()

        # Use different timestamps so the rows have distinct row_keys
        rows = [
            _make_row(target_iso="2024-02-02T17:14:00Z"),  # valid
            _make_row(
                target_iso="2024-02-02T17:30:00Z",
                hypothetical_condor_120m=None,              # bad: no strikes
                hypothetical_condor_to_close=None,
            ),
        ]
        result = fetch_for_rows(rows)

        self.assertEqual(result.rows_attempted, 2)
        self.assertEqual(result.rows_with_legs, 1)
        failed = [r for r in result.rows if r.error]
        self.assertEqual(len(failed), 1)

    @patch("packages.shared.options_cache.orchestrator.fetch_option_bars")
    def test_fetch_option_bars_failure_marks_affected_rows(self, mock_fetch):
        from packages.shared.options_cache.http_client import OratsTransientError
        mock_fetch.side_effect = OratsTransientError("server down")

        # Same trade date + same strikes → both rows share all 4 OPRAs.
        # Different target_ts_utc gives distinct row_keys.
        rows = [
            _make_row(target_iso="2024-02-02T17:14:00Z"),
            _make_row(target_iso="2024-02-02T17:30:00Z"),
        ]
        result = fetch_for_rows(rows)

        self.assertEqual(result.rows_with_legs, 2)
        # All 4 fetch calls fail (one per shared OPRA)
        self.assertEqual(mock_fetch.call_count, 4)
        self.assertEqual(result.unique_opras_fetched, 4)
        # No summary recorded for failed fetches
        self.assertEqual(len(result.option_fetch_summaries), 0)
        for r in result.rows:
            self.assertIsNotNone(r.error)
            self.assertIn("fetch_option_bars failed", r.error)

    def test_unknown_strategy_raises(self):
        with self.assertRaises(ValueError):
            fetch_for_rows([_make_row()], strategy="butterfly")


# ────────────────────────────────────────────────────────────────────────
#  Tests for OPRA-level dedup semantics — use a custom Strategy so the
#  test can pin arbitrary legs and windows per row without going through
#  the condor strategy's date-derived window logic.
# ────────────────────────────────────────────────────────────────────────


def _legs_from_row(row):
    """Test strategy legs_fn: read legs verbatim from row['legs']."""
    return list(row["legs"])


def _window_from_row(row):
    """Test strategy window_fn: read window verbatim from row['window']."""
    return row["window"]


def _install_test_strategy():
    """Register a 'test' strategy in STRATEGY_DISPATCH and return a patcher."""
    test_strategy = Strategy(
        name="test", legs_fn=_legs_from_row, window_fn=_window_from_row,
    )
    return patch.dict(
        "packages.shared.options_cache.orchestrator.STRATEGY_DISPATCH",
        {"test": test_strategy},
        clear=False,
    )


def _leg(opra: str) -> Leg:
    return Leg(opra_symbol=opra, side="long", role="test", ratio=1)


class TestOpraLevelDedup(unittest.TestCase):
    """Verify the bounding-box union and shared-OPRA error attribution."""

    def setUp(self):
        # Custom row_key_fn so we can supply distinct keys for rows whose
        # default key (trade_date, target_ts_utc) would otherwise collide.
        self.row_key_fn = lambda r: r["row_key"]

    @patch("packages.shared.options_cache.orchestrator.fetch_option_bars")
    def test_overlapping_windows_unioned_to_bounding_box(self, mock_fetch):
        mock_fetch.side_effect = lambda **kw: _fake_option_summary()

        legs = [_leg("SPX240202C04980000"), _leg("SPX240202P04935000")]
        rows = [
            {
                "row_key": "r1",
                "legs": legs,
                "window": (datetime(2024, 2, 2, 9, 0), datetime(2024, 2, 2, 10, 0)),
            },
            {
                "row_key": "r2",
                "legs": legs,
                "window": (datetime(2024, 2, 2, 9, 30), datetime(2024, 2, 2, 11, 0)),
            },
        ]

        with _install_test_strategy():
            result = fetch_for_rows(
                rows, strategy="test", row_key_fn=self.row_key_fn,
            )

        self.assertEqual(result.rows_with_legs, 2)
        self.assertEqual(result.unique_opras_fetched, 2)
        self.assertEqual(mock_fetch.call_count, 2)
        # Each call covers the bounding-box union [09:00, 11:00].
        for call in mock_fetch.call_args_list:
            self.assertEqual(call.kwargs["start_pt"], datetime(2024, 2, 2, 9, 0))
            self.assertEqual(call.kwargs["end_pt"], datetime(2024, 2, 2, 11, 0))

    @patch("packages.shared.options_cache.orchestrator.fetch_option_bars")
    def test_disjoint_windows_unioned_to_bounding_box(self, mock_fetch):
        mock_fetch.side_effect = lambda **kw: _fake_option_summary()

        legs = [_leg("SPX240202C04980000"), _leg("SPX240202P04935000")]
        rows = [
            {
                "row_key": "r1",
                "legs": legs,
                "window": (datetime(2024, 2, 2, 9, 0), datetime(2024, 2, 2, 10, 0)),
            },
            {
                "row_key": "r2",
                "legs": legs,
                "window": (datetime(2024, 2, 2, 14, 0), datetime(2024, 2, 2, 15, 0)),
            },
        ]

        with _install_test_strategy():
            result = fetch_for_rows(
                rows, strategy="test", row_key_fn=self.row_key_fn,
            )

        self.assertEqual(result.unique_opras_fetched, 2)
        self.assertEqual(mock_fetch.call_count, 2)
        # The orchestrator over-fetches the in-between region intentionally;
        # the cache layer narrows on subsequent calls. Bounding box is [09:00, 15:00].
        for call in mock_fetch.call_args_list:
            self.assertEqual(call.kwargs["start_pt"], datetime(2024, 2, 2, 9, 0))
            self.assertEqual(call.kwargs["end_pt"], datetime(2024, 2, 2, 15, 0))

    @patch("packages.shared.options_cache.orchestrator.fetch_option_bars")
    def test_error_on_shared_opra_propagates_to_all_referring_rows(self, mock_fetch):
        shared = "SPX240202C04980000"
        r1_unique = ["SPX240202P04935000", "SPX240202P04925000", "SPX240202C04990000"]
        r2_unique = ["SPX240202P04920000", "SPX240202P04910000", "SPX240202C04995000"]

        def side_effect(*, opra_symbols, **kw):
            (opra,) = opra_symbols
            if opra == shared:
                raise RuntimeError("boom")
            return _fake_option_summary()

        mock_fetch.side_effect = side_effect

        window = (datetime(2024, 2, 2, 9, 0), datetime(2024, 2, 2, 13, 0))
        rows = [
            {
                "row_key": "r1",
                "legs": [_leg(shared)] + [_leg(o) for o in r1_unique],
                "window": window,
            },
            {
                "row_key": "r2",
                "legs": [_leg(shared)] + [_leg(o) for o in r2_unique],
                "window": window,
            },
            {
                "row_key": "r3",
                # No overlap with shared OPRA — should be unaffected
                "legs": [_leg("SPX240202C05000000")],
                "window": window,
            },
        ]

        with _install_test_strategy():
            result = fetch_for_rows(
                rows, strategy="test", row_key_fn=self.row_key_fn,
            )

        # 7 unique OPRAs across the 3 rows: shared, 3 from r1, 3 from r2, 1 from r3
        self.assertEqual(result.unique_opras_fetched, 8)

        by_key = {r.row_key: r for r in result.rows}
        self.assertIsNotNone(by_key["r1"].error)
        self.assertIn(shared, by_key["r1"].error)
        self.assertIn("fetch_option_bars failed", by_key["r1"].error)
        self.assertIsNotNone(by_key["r2"].error)
        self.assertIn(shared, by_key["r2"].error)
        self.assertIsNone(by_key["r3"].error)


class TestGetStrategy(unittest.TestCase):
    def test_condor_registered(self):
        s = get_strategy("condor")
        self.assertEqual(s.name, "condor")

    def test_unknown_raises(self):
        with self.assertRaises(ValueError):
            get_strategy("does_not_exist")


if __name__ == "__main__":
    unittest.main()
