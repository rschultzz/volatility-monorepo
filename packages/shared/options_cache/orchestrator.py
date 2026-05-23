"""
Multi-row pricing fetch orchestrator.

Given a list of scan rows representing trades to analyze, this module:
    1. Derives legs and pricing windows for each row (per strategy)
    2. Deduplicates fetches by OPRA symbol — many condors on the same
       trade date share OPRAs across rows
    3. For each unique OPRA, computes a bounding-box window across the
       rows that reference it and calls fetch_option_bars once
    4. Returns a summary mapping rows -> their fetched legs and any
       errors encountered

Phase 4a is synchronous: callers pass scan rows in, function returns
when done. Phase 4b will wrap this in a background job + API endpoint.
Phase 4c adds the UI button.

Strategy registry: today, only condor is wired in. When new strategies
land, they'll add entries to STRATEGY_DISPATCH below.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Iterable, Optional

from . import repository as repo
from .condor import (
    Leg,
    condor_legs_for_row,
    condor_pricing_window_for_row,
)
from .fetcher import fetch_option_bars
from .models import FetchOptionBarsSummary, FetchSource

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────
#  Strategy dispatch
# ────────────────────────────────────────────────────────────────────────

# Functions a strategy must implement.
LegFn = Callable[[dict], list[Leg]]
WindowFn = Callable[[dict], Optional[tuple[datetime, datetime]]]


@dataclass(frozen=True)
class Strategy:
    """A registered strategy."""
    name: str
    legs_fn: LegFn
    window_fn: WindowFn


STRATEGY_DISPATCH: dict[str, Strategy] = {
    "condor": Strategy(
        name="condor",
        legs_fn=condor_legs_for_row,
        window_fn=condor_pricing_window_for_row,
    ),
}


def get_strategy(name: str) -> Strategy:
    """Look up a registered strategy by name. Raises if unknown."""
    s = STRATEGY_DISPATCH.get(name)
    if s is None:
        raise ValueError(
            f"Unknown strategy {name!r}. "
            f"Registered: {sorted(STRATEGY_DISPATCH)}"
        )
    return s


# ────────────────────────────────────────────────────────────────────────
#  Result types
# ────────────────────────────────────────────────────────────────────────

@dataclass
class RowResult:
    """Per-row fetch result returned to the caller."""
    row_key: str            # caller-supplied identifier for the row
    legs: list[Leg] = field(default_factory=list)
    window: Optional[tuple[datetime, datetime]] = None
    error: Optional[str] = None  # set if leg/window derivation failed


@dataclass
class OrchestratorResult:
    """Aggregate result from fetch_for_rows."""
    rows: list[RowResult] = field(default_factory=list)
    option_fetch_summaries: list[FetchOptionBarsSummary] = field(default_factory=list)
    rows_attempted: int = 0
    rows_with_legs: int = 0
    unique_opras_fetched: int = 0
    cache_hits: int = 0
    total_api_calls: int = 0
    total_bars_inserted: int = 0


# ────────────────────────────────────────────────────────────────────────
#  Public entry point
# ────────────────────────────────────────────────────────────────────────

def fetch_for_rows(
    rows: Iterable[dict],
    *,
    strategy: str = "condor",
    source: FetchSource = "historical_backfill",
    row_key_fn: Callable[[dict], str] = None,
) -> OrchestratorResult:
    """
    Fetch pricing for the contracts referenced by a list of scan rows.

    Args:
        rows: Iterable of scan-row dicts. Each must contain the fields the
            chosen strategy needs (for condor: trade_date, target_ts_utc,
            target_spx_price, hypothetical_condor_*).
        strategy: Strategy name. Currently only 'condor'.
        source: Provenance tag for fetched_windows.
        row_key_fn: Function to derive a stable identifier per row, used
            for OrchestratorResult.rows. Defaults to a (trade_date,
            target_ts_utc) tuple-string.

    Returns:
        OrchestratorResult with per-row outcomes and aggregate stats.

    Note: this function is synchronous. For long-running fetches, wrap
    it in a background thread/job (Phase 4b).
    """
    strat = get_strategy(strategy)
    if row_key_fn is None:
        row_key_fn = _default_row_key

    rows_list = list(rows)
    result = OrchestratorResult(rows_attempted=len(rows_list))

    # Step 1: derive legs and windows per row; collect a (window list,
    # row_keys set) per unique OPRA across all rows.
    per_opra_windows: dict[str, list[tuple[datetime, datetime]]] = defaultdict(list)
    opra_to_row_keys: dict[str, set[str]] = defaultdict(set)

    for row in rows_list:
        row_key = row_key_fn(row)
        row_result = RowResult(row_key=row_key)

        try:
            legs = strat.legs_fn(row)
        except Exception as e:
            row_result.error = f"legs_fn failed: {e}"
            result.rows.append(row_result)
            continue

        if not legs:
            row_result.error = "no legs derivable from row"
            result.rows.append(row_result)
            continue

        try:
            window = strat.window_fn(row)
        except Exception as e:
            row_result.error = f"window_fn failed: {e}"
            result.rows.append(row_result)
            continue

        if window is None:
            row_result.error = "no pricing window derivable from row"
            result.rows.append(row_result)
            continue

        row_result.legs = legs
        row_result.window = window
        result.rows.append(row_result)
        result.rows_with_legs += 1

        for leg in legs:
            per_opra_windows[leg.opra_symbol].append(window)
            opra_to_row_keys[leg.opra_symbol].add(row_key)

    # Step 2: compute a bounding-box window per OPRA.
    # Disjoint sub-ranges inside the bounding box are discovered as
    # separate gaps by fetch_option_bars; over-fetch is bounded by the
    # cache layer. See spec for the design rationale.
    opra_bounding: dict[str, tuple[datetime, datetime]] = {}
    for opra, windows in per_opra_windows.items():
        start = min(w[0] for w in windows)
        end = max(w[1] for w in windows)
        opra_bounding[opra] = (start, end)

    logger.info(
        "fetch_for_rows: %d rows in, %d had legs, %d unique OPRAs",
        len(rows_list), result.rows_with_legs, len(opra_bounding),
    )

    # Step 3: fetch each unique OPRA once over its bounding window.
    for opra, (start_pt, end_pt) in sorted(opra_bounding.items()):
        logger.info(
            "fetch_for_rows: opra %s [%s, %s] covers %d row(s)",
            opra, start_pt, end_pt, len(opra_to_row_keys[opra]),
        )
        result.unique_opras_fetched += 1
        try:
            summary = fetch_option_bars(
                opra_symbols=[opra],
                start_pt=start_pt,
                end_pt=end_pt,
                source=source,
            )
        except Exception as e:
            logger.error(
                "fetch_for_rows: option fetch %s [%s, %s] failed: %s",
                opra, start_pt, end_pt, e,
            )
            for rk in opra_to_row_keys[opra]:
                rr = next((r for r in result.rows if r.row_key == rk), None)
                if rr and rr.error is None:
                    rr.error = f"fetch_option_bars failed for {opra}: {e}"
            continue

        result.option_fetch_summaries.append(summary)
        result.total_api_calls += summary.gaps_filled
        result.total_bars_inserted += summary.bars_written
        result.cache_hits += summary.cache_hits

    return result


# ────────────────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────────────────

def _default_row_key(row: dict) -> str:
    """Default row key: trade_date + target_ts_utc."""
    parts = [
        str(row.get("trade_date") or "?"),
        str(row.get("target_ts_utc") or row.get("start_ts_utc") or "?"),
    ]
    return ":".join(parts)


# ────────────────────────────────────────────────────────────────────────
#  Read-side helpers (for caller convenience)
# ────────────────────────────────────────────────────────────────────────

def get_legs_with_bars(
    row_result: RowResult,
) -> dict[str, list]:
    """
    For a completed RowResult, query the cache and return each leg's bars.

    Returns: {opra_symbol: [OptionMinuteBar, ...]} for every leg in the
    row, restricted to the row's pricing window.
    """
    if row_result.window is None:
        return {}

    start_pt, end_pt = row_result.window
    out = {}
    for leg in row_result.legs:
        bars = repo.get_bars_for_contract(leg.opra_symbol, start_pt, end_pt)
        out[leg.opra_symbol] = bars
    return out
