"""
Multi-row pricing fetch orchestrator.

Given a list of scan rows representing trades to analyze, this module:
    1. Derives legs and pricing windows for each row (per strategy)
    2. Deduplicates (ticker, time-window) tuples — many condors on the
       same trade date share the same chain fetch
    3. Calls fetch_chain once per unique (ticker, window)
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
from .fetcher import fetch_chain
from .models import ChainFilter, FetchSource, FetchSummary

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
    fetch_summaries: list[FetchSummary] = field(default_factory=list)
    rows_attempted: int = 0
    rows_with_legs: int = 0
    unique_windows_fetched: int = 0
    total_api_calls: int = 0
    total_bars_inserted: int = 0


# ────────────────────────────────────────────────────────────────────────
#  Public entry point
# ────────────────────────────────────────────────────────────────────────

def fetch_for_rows(
    rows: Iterable[dict],
    *,
    strategy: str = "condor",
    underlying: str = "SPY",
    chain_filter: Optional[ChainFilter] = None,
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
        underlying: Ticker to use for the OPRA symbols. Hardcoded to 'SPY'
            for now. SPX will be added when ORATS coverage allows.
        chain_filter: Filter passed to each fetch_chain call. None means
            use fetch_chain's default (±10% strike, 0-60 DTE).
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

    # Step 1: derive legs and windows per row, collect them.
    # A "fetch group" is one (ticker, window_start, window_end) tuple.
    # Multiple rows can share a group; we'll deduplicate before calling
    # fetch_chain.
    groups: dict[tuple[str, datetime, datetime], list[str]] = defaultdict(list)

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

        groups[(underlying, window[0], window[1])].append(row_key)

    # Step 2: fetch each unique (ticker, window) once.
    logger.info(
        "fetch_for_rows: %d rows in, %d had legs, %d unique fetch windows",
        len(rows_list), result.rows_with_legs, len(groups),
    )

    for (ticker, start_pt, end_pt), row_keys in sorted(groups.items()):
        logger.info(
            "fetch_for_rows: chain %s [%s, %s] covers %d row(s)",
            ticker, start_pt, end_pt, len(row_keys),
        )
        try:
            summary = fetch_chain(
                ticker=ticker,
                start_pt=start_pt,
                end_pt=end_pt,
                chain_filter=chain_filter,
                source=source,
            )
        except Exception as e:
            logger.error(
                "fetch_for_rows: chain fetch %s [%s, %s] failed: %s",
                ticker, start_pt, end_pt, e,
            )
            # Mark all rows in this group with the error
            for rk in row_keys:
                rr = next((r for r in result.rows if r.row_key == rk), None)
                if rr and rr.error is None:
                    rr.error = f"fetch_chain failed: {e}"
            continue

        result.fetch_summaries.append(summary)
        result.unique_windows_fetched += 1
        result.total_api_calls += summary.api_calls
        result.total_bars_inserted += summary.bars_inserted

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
