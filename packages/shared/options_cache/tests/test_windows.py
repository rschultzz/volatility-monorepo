"""
Unit tests for gap detection.

No DB required. Run with:
    python -m unittest packages.shared.options_cache.tests.test_windows
"""
from __future__ import annotations

import unittest
from datetime import datetime

from packages.shared.options_cache.models import FetchedWindow, TimeRange
from packages.shared.options_cache.windows import find_gaps


def _w(start: str, end: str) -> FetchedWindow:
    """Helper: build a FetchedWindow from ISO strings."""
    return FetchedWindow(
        opra_symbol="TEST",
        window_start_pt=datetime.fromisoformat(start),
        window_end_pt=datetime.fromisoformat(end),
        row_count=1,
        source="manual",
    )


def _r(start: str, end: str) -> TimeRange:
    return TimeRange(
        start_pt=datetime.fromisoformat(start),
        end_pt=datetime.fromisoformat(end),
    )


class TestFindGaps(unittest.TestCase):
    def test_no_existing_returns_full_request(self):
        req = _r("2026-01-17 09:30", "2026-01-17 10:00")
        gaps = find_gaps(req, [])
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0].start_pt, req.start_pt)
        self.assertEqual(gaps[0].end_pt, req.end_pt)

    def test_full_coverage_returns_empty(self):
        req = _r("2026-01-17 09:30", "2026-01-17 10:00")
        existing = [_w("2026-01-17 09:30", "2026-01-17 10:00")]
        gaps = find_gaps(req, existing)
        self.assertEqual(gaps, [])

    def test_existing_outside_request_ignored(self):
        req = _r("2026-01-17 09:30", "2026-01-17 10:00")
        existing = [_w("2026-01-17 13:00", "2026-01-17 14:00")]
        gaps = find_gaps(req, existing)
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0].start_pt, req.start_pt)

    def test_partial_coverage_at_start(self):
        # Request 09:30-10:00, have 09:30-09:45
        req = _r("2026-01-17 09:30", "2026-01-17 10:00")
        existing = [_w("2026-01-17 09:30", "2026-01-17 09:45")]
        gaps = find_gaps(req, existing)
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0].start_pt, datetime.fromisoformat("2026-01-17 09:46"))
        self.assertEqual(gaps[0].end_pt, datetime.fromisoformat("2026-01-17 10:00"))

    def test_partial_coverage_at_end(self):
        # Request 09:30-10:00, have 09:45-10:00
        req = _r("2026-01-17 09:30", "2026-01-17 10:00")
        existing = [_w("2026-01-17 09:45", "2026-01-17 10:00")]
        gaps = find_gaps(req, existing)
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0].start_pt, datetime.fromisoformat("2026-01-17 09:30"))
        self.assertEqual(gaps[0].end_pt, datetime.fromisoformat("2026-01-17 09:44"))

    def test_hole_in_middle(self):
        # Request 09:30-10:00, have 09:30-09:40 and 09:50-10:00
        req = _r("2026-01-17 09:30", "2026-01-17 10:00")
        existing = [
            _w("2026-01-17 09:30", "2026-01-17 09:40"),
            _w("2026-01-17 09:50", "2026-01-17 10:00"),
        ]
        gaps = find_gaps(req, existing)
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0].start_pt, datetime.fromisoformat("2026-01-17 09:41"))
        self.assertEqual(gaps[0].end_pt, datetime.fromisoformat("2026-01-17 09:49"))

    def test_multiple_holes(self):
        # Request 09:30-10:00, have three small windows
        req = _r("2026-01-17 09:30", "2026-01-17 10:00")
        existing = [
            _w("2026-01-17 09:32", "2026-01-17 09:35"),
            _w("2026-01-17 09:40", "2026-01-17 09:45"),
            _w("2026-01-17 09:50", "2026-01-17 09:55"),
        ]
        gaps = find_gaps(req, existing)
        # Expected gaps: 09:30-09:31, 09:36-09:39, 09:46-09:49, 09:56-10:00
        self.assertEqual(len(gaps), 4)

    def test_adjacent_windows_merge(self):
        # 09:30-09:40 and 09:41-09:50 are adjacent (no minute between).
        # They should be treated as one window for gap purposes.
        req = _r("2026-01-17 09:30", "2026-01-17 10:00")
        existing = [
            _w("2026-01-17 09:30", "2026-01-17 09:40"),
            _w("2026-01-17 09:41", "2026-01-17 09:50"),
        ]
        gaps = find_gaps(req, existing)
        # Single gap from 09:51 to 10:00
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0].start_pt, datetime.fromisoformat("2026-01-17 09:51"))

    def test_overlapping_windows_handled(self):
        # 09:30-09:45 and 09:40-09:50 overlap; should merge to 09:30-09:50
        req = _r("2026-01-17 09:30", "2026-01-17 10:00")
        existing = [
            _w("2026-01-17 09:30", "2026-01-17 09:45"),
            _w("2026-01-17 09:40", "2026-01-17 09:50"),
        ]
        gaps = find_gaps(req, existing)
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0].start_pt, datetime.fromisoformat("2026-01-17 09:51"))

    def test_window_extending_outside_request_clipped(self):
        # Existing window extends way past the request — should be clipped
        req = _r("2026-01-17 09:30", "2026-01-17 10:00")
        existing = [_w("2026-01-17 08:00", "2026-01-17 12:00")]
        gaps = find_gaps(req, existing)
        self.assertEqual(gaps, [])


class TestTimeRangeValidation(unittest.TestCase):
    def test_inverted_range_raises(self):
        with self.assertRaises(ValueError):
            TimeRange(
                start_pt=datetime.fromisoformat("2026-01-17 10:00"),
                end_pt=datetime.fromisoformat("2026-01-17 09:30"),
            )

    def test_same_minute_range_ok(self):
        # Single-minute range should work
        r = TimeRange(
            start_pt=datetime.fromisoformat("2026-01-17 10:00"),
            end_pt=datetime.fromisoformat("2026-01-17 10:00"),
        )
        self.assertEqual(r.start_pt, r.end_pt)


if __name__ == "__main__":
    unittest.main()
