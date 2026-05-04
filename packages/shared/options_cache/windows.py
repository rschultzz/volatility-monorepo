"""
Gap detection.

Given a request "I need pricing for contract X over [start, end]", figure
out which sub-ranges are already cached (via orats_options_fetched_windows)
and which need to be fetched.

The core algorithm is interval-difference: subtract a list of covered
ranges from a target range, returning the uncovered gaps.

Pure functions on time ranges — no DB access in this module. The
repository feeds in the existing windows; this module returns the gaps.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable

from .models import FetchedWindow, TimeRange


def find_gaps(
    requested: TimeRange,
    existing_windows: Iterable[FetchedWindow],
) -> list[TimeRange]:
    """
    Return the time ranges within `requested` not covered by `existing_windows`.

    Args:
        requested: The time range we want pricing for.
        existing_windows: All fetched windows for the relevant contract.
                          Order doesn't matter; we'll sort.

    Returns:
        A list of TimeRange objects representing the gaps that need fetching.
        Empty list means the request is fully cached. The full requested
        range (as a single-element list) means nothing is cached.
    """
    # Convert windows to TimeRanges and clip to the requested range.
    # Anything fully outside requested is irrelevant.
    covered = []
    for w in existing_windows:
        if w.window_end_pt < requested.start_pt or w.window_start_pt > requested.end_pt:
            continue  # no overlap
        covered.append(TimeRange(
            start_pt=max(w.window_start_pt, requested.start_pt),
            end_pt=min(w.window_end_pt, requested.end_pt),
        ))

    if not covered:
        return [requested]

    # Sort and merge overlapping/adjacent windows. Two windows are
    # "adjacent" if there's no minute between them (end + 1 minute >= next start).
    covered.sort(key=lambda r: r.start_pt)
    merged: list[TimeRange] = [covered[0]]
    for r in covered[1:]:
        last = merged[-1]
        if r.start_pt <= last.end_pt + timedelta(minutes=1):
            merged[-1] = TimeRange(
                start_pt=last.start_pt,
                end_pt=max(last.end_pt, r.end_pt),
            )
        else:
            merged.append(r)

    # Now compute gaps: walk through merged covered windows and emit the
    # spaces in between (and at the ends).
    gaps: list[TimeRange] = []
    cursor = requested.start_pt
    for w in merged:
        if w.start_pt > cursor:
            # Gap before this window
            gaps.append(TimeRange(
                start_pt=cursor,
                end_pt=w.start_pt - timedelta(minutes=1),
            ))
        cursor = max(cursor, w.end_pt + timedelta(minutes=1))

    if cursor <= requested.end_pt:
        gaps.append(TimeRange(start_pt=cursor, end_pt=requested.end_pt))

    return gaps


def coverage_summary(
    requested: TimeRange,
    existing_windows: Iterable[FetchedWindow],
) -> dict:
    """
    Convenience: high-level summary of coverage for a request.

    Returns a dict like:
        {
            "fully_cached": bool,
            "uncached": bool,
            "gap_count": int,
            "gaps": [TimeRange, ...],
        }
    """
    gaps = find_gaps(requested, existing_windows)
    return {
        "fully_cached": len(gaps) == 0,
        "uncached": len(gaps) == 1 and gaps[0].start_pt == requested.start_pt
                    and gaps[0].end_pt == requested.end_pt,
        "gap_count": len(gaps),
        "gaps": gaps,
    }
