"""
Date range chunking for ORATS API limits.

ORATS' historical 1-minute endpoints cap a single tradeDate range request at
40 trading days. For longer ranges we split into multiple chunks and stitch
the results back together.

Pure-function module, no I/O.

Trading-day awareness: the 40-day limit is in trading days, not calendar
days. A 60-calendar-day range is roughly 42 trading days (5/7 ratio plus
holidays). To stay safely under the limit, we use 40 calendar days as the
chunk size — this is conservative (always under the trading-day limit) and
avoids needing a market calendar for now. Cost: a slightly larger number
of chunks than strictly necessary.

If we ever need exact trading-day chunking (e.g., to minimize API calls
on a tight quota), we can add a market calendar dependency. For now,
calendar-day chunking is fine — extra chunks just hit cache on subsequent
runs anyway.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import List

from .models import TimeRange


# Conservative chunk size in calendar days. Stays under the 40-trading-day
# API limit even on stretches with no holidays.
DEFAULT_CHUNK_DAYS = 40


def chunk_range(
    requested: TimeRange,
    chunk_days: int = DEFAULT_CHUNK_DAYS,
) -> List[TimeRange]:
    """
    Split a time range into chunks of at most `chunk_days` calendar days each.

    The split is at midnight boundaries so chunks are aligned to whole days
    where possible. Each chunk is contiguous with the next (no gaps).

    Args:
        requested: The full time range to split.
        chunk_days: Maximum span of each chunk in calendar days.

    Returns:
        A list of TimeRange objects whose union exactly equals `requested`.
        For ranges shorter than chunk_days, returns [requested] unchanged.

    Raises:
        ValueError if chunk_days <= 0.
    """
    if chunk_days <= 0:
        raise ValueError(f"chunk_days must be > 0, got {chunk_days}")

    span_days = (requested.end_pt - requested.start_pt).days
    if span_days < chunk_days:
        return [requested]

    chunks: List[TimeRange] = []
    cursor = requested.start_pt

    while cursor <= requested.end_pt:
        # Each chunk spans up to chunk_days calendar days. End is one minute
        # before the next chunk start so chunks are non-overlapping.
        chunk_end = min(
            cursor + timedelta(days=chunk_days) - timedelta(minutes=1),
            requested.end_pt,
        )
        chunks.append(TimeRange(start_pt=cursor, end_pt=chunk_end))
        cursor = chunk_end + timedelta(minutes=1)

    return chunks
