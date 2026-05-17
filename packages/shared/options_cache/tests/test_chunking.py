"""
Unit tests for date range chunking. No I/O.

Run with:
    python -m unittest packages.shared.options_cache.tests.test_chunking
"""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta

from packages.shared.options_cache.chunking import chunk_range, DEFAULT_CHUNK_DAYS
from packages.shared.options_cache.models import TimeRange


def _r(start: str, end: str) -> TimeRange:
    return TimeRange(
        start_pt=datetime.fromisoformat(start),
        end_pt=datetime.fromisoformat(end),
    )


class TestChunkRange(unittest.TestCase):
    def test_short_range_unchanged(self):
        # Single trading day, well under any chunk limit
        req = _r("2026-01-17 06:30", "2026-01-17 13:00")
        chunks = chunk_range(req)
        self.assertEqual(chunks, [req])

    def test_exactly_chunk_size(self):
        # Range exactly at the chunk boundary returns one chunk
        start = datetime(2026, 1, 1, 0, 0)
        end = start + timedelta(days=DEFAULT_CHUNK_DAYS - 1, hours=23, minutes=59)
        chunks = chunk_range(TimeRange(start_pt=start, end_pt=end))
        self.assertEqual(len(chunks), 1)

    def test_longer_than_chunk_splits(self):
        # 100 days at 40-day chunks → 3 chunks (40 + 40 + 20)
        start = datetime(2026, 1, 1, 0, 0)
        end = datetime(2026, 4, 11, 0, 0)  # 100 days later
        chunks = chunk_range(TimeRange(start_pt=start, end_pt=end))
        self.assertEqual(len(chunks), 3)

    def test_chunks_are_contiguous(self):
        # Each chunk's end_pt + 1 minute should equal the next chunk's start_pt
        start = datetime(2026, 1, 1, 0, 0)
        end = datetime(2026, 6, 1, 0, 0)
        chunks = chunk_range(TimeRange(start_pt=start, end_pt=end))
        for i in range(len(chunks) - 1):
            expected_next_start = chunks[i].end_pt + timedelta(minutes=1)
            self.assertEqual(
                expected_next_start, chunks[i + 1].start_pt,
                f"gap between chunk {i} and chunk {i+1}",
            )

    def test_chunks_cover_full_range_exactly(self):
        # First chunk start matches request start, last chunk end matches
        # request end
        start = datetime(2026, 1, 1, 0, 0)
        end = datetime(2026, 6, 1, 0, 0)
        chunks = chunk_range(TimeRange(start_pt=start, end_pt=end))
        self.assertEqual(chunks[0].start_pt, start)
        self.assertEqual(chunks[-1].end_pt, end)

    def test_each_chunk_under_limit(self):
        # No chunk should exceed the max size
        start = datetime(2026, 1, 1, 0, 0)
        end = datetime(2026, 12, 31, 0, 0)  # ~365 days
        chunks = chunk_range(TimeRange(start_pt=start, end_pt=end))
        for c in chunks:
            span_days = (c.end_pt - c.start_pt).total_seconds() / 86400
            self.assertLessEqual(span_days, DEFAULT_CHUNK_DAYS)

    def test_custom_chunk_size(self):
        # Smaller chunk size → more chunks
        req = _r("2026-01-01", "2026-02-15")  # ~45 days
        chunks_default = chunk_range(req)
        chunks_smaller = chunk_range(req, chunk_days=10)
        self.assertGreater(len(chunks_smaller), len(chunks_default))

    def test_invalid_chunk_size_raises(self):
        req = _r("2026-01-01", "2026-02-15")
        with self.assertRaises(ValueError):
            chunk_range(req, chunk_days=0)
        with self.assertRaises(ValueError):
            chunk_range(req, chunk_days=-5)


if __name__ == "__main__":
    unittest.main()
