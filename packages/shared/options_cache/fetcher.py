"""
High-level fetch orchestrator (chain-first).

Primary entry point: fetch_chain(ticker, start, end, chain_filter).
- Hits ORATS' strikes/chain endpoint (one call returns all strikes at one minute).
- Applies a filter to keep only the useful neighborhood (default ±10% of
  spot, 0-60 DTE).
- Inserts both call and put bars for every kept row.
- Records per-contract fetched_windows so subsequent fetch_contract calls
  for any of those contracts hit cache.

Convenience wrapper: fetch_contract(opra, start, end).
- Parses the OPRA → ticker.
- Checks fetched_windows for the requested contract.
- If gaps exist, calls fetch_chain to fill them (which fills neighbors too).
- Returns just the bars matching the requested OPRA.

Cache-hit semantics:
- fetch_chain has NO gap detection: it always hits the API. Re-running the
  same chain fetch is correct (idempotent inserts) but costs API quota.
- fetch_contract HAS gap detection: re-running for the same contract
  produces zero API calls if the chain has been fetched previously.

The asymmetry exists because per-contract provenance is what the cache
already tracks; chain-level provenance would need additional schema work
that we've deferred to a future phase.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional, Sequence
from zoneinfo import ZoneInfo

from . import repository as repo
from . import windows as W
from .chunking import chunk_range
from .csv_parser import parse_orats_csv
from .http_client import get_csv
from .models import (
    DEFAULT_CHAIN_FILTER,
    ChainFilter,
    FetchedWindow,
    FetchOptionBarsSummary,
    FetchSource,
    FetchSummary,
    OptionMinuteBar,
    TimeRange,
)
from .opra import parse_opra

logger = logging.getLogger(__name__)


# ORATS uses Eastern Time for tradeDate parameters; we store naive PT.
_PT = ZoneInfo("America/Los_Angeles")
_ET = ZoneInfo("America/New_York")

# Strikes Chain History endpoint (Live Intraday subscription tier).
# This is the per-minute chain endpoint matching the user's subscription.
_CHAIN_PATH = "/datav2/hist/live/one-minute/strikes/chain"

# Strikes by OPRA History endpoint (same tier). Single-contract per call,
# but supports range-tradeDate — the architectural lever the pivot relies on.
_OPTION_PATH = "/datav2/hist/live/one-minute/strikes/option"


# ────────────────────────────────────────────────────────────────────────
#  fetch_option_bars (CR-004): per-OPRA, gap-aware, option-endpoint path
# ────────────────────────────────────────────────────────────────────────

def fetch_option_bars(
    opra_symbols: Sequence[str],
    start_pt: datetime,
    end_pt: datetime,
    *,
    source: FetchSource = "historical_backfill",
) -> FetchOptionBarsSummary:
    """
    Cache-aware fetch for one or more OPRA contracts via the option endpoint.

    For each OPRA, computes gaps against orats_options_fetched_windows,
    calls the private primitive one OPRA per gap, writes returned bars
    to orats_options_minute, and writes orats_options_fetched_windows rows
    for both the requested OPRA AND the counterpart OPRA returned in the
    same response (the chain-shape ORATS row emits both call and put bars).

    Idempotent: re-running over the same range produces zero HTTP calls.

    Args:
        opra_symbols: Full OPRA symbols (e.g. 'SPX240202P04935000').
        start_pt: Start of range, naive Pacific Time.
        end_pt: End of range, naive PT. Inclusive.
        source: Provenance tag for fetched_windows rows.

    Returns:
        FetchOptionBarsSummary with operation counters.

    Raises:
        OratsTransientError or OratsPermanentError on API failures.
        ValueError on tz-aware datetimes.
    """
    _validate_naive(start_pt, "start_pt")
    _validate_naive(end_pt, "end_pt")

    requested = TimeRange(start_pt=start_pt, end_pt=end_pt)

    gaps_filled = 0
    bars_written = 0
    cache_hits = 0

    for opra_symbol in opra_symbols:
        existing = repo.get_windows_for_contract(opra_symbol)
        gaps = W.find_gaps(requested, existing)

        if not gaps:
            cache_hits += 1
            logger.info(
                "fetch_option_bars %s: fully cached for [%s, %s], no API calls",
                opra_symbol, start_pt, end_pt,
            )
            continue

        logger.info(
            "fetch_option_bars %s: %d gap(s) over [%s, %s]",
            opra_symbol, len(gaps), start_pt, end_pt,
        )

        for gap in gaps:
            bars = _fetch_option_bars_from_orats(
                opra_symbol, gap.start_pt, gap.end_pt,
            )
            n_inserted = repo.insert_bars(bars)
            bars_written += n_inserted

            # Per-OPRA bar counts within this gap. Used as row_count on the
            # fetched_windows record so downstream observability sees how
            # dense each contract's coverage is.
            bars_per_opra: dict[str, int] = {}
            for b in bars:
                bars_per_opra[b.opra_symbol] = bars_per_opra.get(b.opra_symbol, 0) + 1

            # Record windows for every unique OPRA in the response (the
            # requested one + the counterpart at the same strike+expir).
            # On an empty response the requested OPRA still gets a row so
            # we don't perpetually refetch empty windows.
            opras_to_record = list(bars_per_opra.keys())
            if opra_symbol not in bars_per_opra:
                opras_to_record.append(opra_symbol)

            for sym in opras_to_record:
                repo.record_fetched_window(FetchedWindow(
                    opra_symbol=sym,
                    window_start_pt=gap.start_pt,
                    window_end_pt=gap.end_pt,
                    row_count=bars_per_opra.get(sym, 0),
                    source=source,
                ))

            gaps_filled += 1

    return FetchOptionBarsSummary(
        opras_processed=len(opra_symbols),
        gaps_filled=gaps_filled,
        bars_written=bars_written,
        cache_hits=cache_hits,
    )


def _fetch_option_bars_from_orats(
    opra_symbol: str,
    start_pt: datetime,
    end_pt: datetime,
) -> list[OptionMinuteBar]:
    """
    Pure HTTP. Hits the option endpoint with one OPRA + a tradeDate range
    (or single timestamp if start_pt == end_pt). PT→ET conversion via the
    shared _format_trade_date_param helper.

    Returns all bars from the parsed response. The CSV parser emits two
    bars per row (call OPRA + put OPRA at the same strike+expir), so a
    request for one side returns both sides' bars.

    No DB I/O. No cache awareness. Used only by fetch_option_bars.
    """
    _validate_naive(start_pt, "start_pt")
    _validate_naive(end_pt, "end_pt")

    chunk = TimeRange(start_pt=start_pt, end_pt=end_pt)
    params = {
        "ticker": opra_symbol,
        "tradeDate": _format_trade_date_param(chunk),
    }
    csv_text = get_csv(_OPTION_PATH, params)
    bars, _, _ = parse_orats_csv(csv_text)
    return bars


# ────────────────────────────────────────────────────────────────────────
#  Primary API: fetch_chain
# ────────────────────────────────────────────────────────────────────────

def fetch_chain(
    ticker: str,
    start_pt: datetime,
    end_pt: datetime,
    *,
    chain_filter: Optional[ChainFilter] = DEFAULT_CHAIN_FILTER,
    source: FetchSource = "historical_backfill",
) -> FetchSummary:
    """
    Fetch the options chain for `ticker` over [start_pt, end_pt] from ORATS.

    Inserts kept bars into the cache and records per-contract fetched_windows
    so future per-contract queries hit cache.

    Note: this function has NO gap detection. Each call hits the API for
    the full requested range, regardless of what's already cached. Inserts
    are idempotent (ON CONFLICT DO NOTHING) so re-runs are correct, just
    quota-wasteful. Use fetch_contract() for gap-aware single-contract
    fetches.

    Args:
        ticker: Underlying ticker (e.g., 'SPY', 'AAPL'). Case-insensitive.
        start_pt: Start of range, naive Pacific Time.
        end_pt: End of range, naive Pacific Time. Inclusive.
        chain_filter: Filter applied at parse time. None means keep
            everything. Default is ±10% strike-of-spot and 0-60 DTE.
        source: Provenance tag for fetched_windows.

    Returns:
        FetchSummary with counts of API calls, rows received/kept,
        bars inserted, and unique contracts touched.

    Raises:
        OratsTransientError or OratsPermanentError on API failures.
        ValueError if datetime arguments are tz-aware (must be naive PT).
    """
    _validate_naive(start_pt, "start_pt")
    _validate_naive(end_pt, "end_pt")
    ticker = ticker.strip().upper()

    requested = TimeRange(start_pt=start_pt, end_pt=end_pt)
    chunks = chunk_range(requested)

    logger.info(
        "fetch_chain %s: %d chunk(s) over [%s, %s], filter=%s",
        ticker, len(chunks), start_pt, end_pt, chain_filter,
    )

    summary = FetchSummary(
        ticker=ticker,
        time_range=requested,
        api_calls=0,
        rows_received=0,
        rows_kept=0,
        bars_inserted=0,
        bars_total=0,
        contracts_touched=0,
    )
    contracts_seen: set[str] = set()

    for chunk in chunks:
        chunk_summary = _fetch_chain_chunk(
            ticker=ticker,
            chunk=chunk,
            chain_filter=chain_filter,
            source=source,
            contracts_seen=contracts_seen,
        )
        summary.api_calls += chunk_summary.api_calls
        summary.rows_received += chunk_summary.rows_received
        summary.rows_kept += chunk_summary.rows_kept
        summary.bars_inserted += chunk_summary.bars_inserted
        summary.bars_total += chunk_summary.bars_total

    summary.contracts_touched = len(contracts_seen)
    return summary


# ────────────────────────────────────────────────────────────────────────
#  Convenience wrapper: fetch_contract
# ────────────────────────────────────────────────────────────────────────

def fetch_contract(
    opra_symbol: str,
    start_pt: datetime,
    end_pt: datetime,
    *,
    chain_filter: Optional[ChainFilter] = None,
    source: FetchSource = "historical_backfill",
) -> list[OptionMinuteBar]:
    """
    Ensure pricing data is cached for `opra_symbol` over [start_pt, end_pt],
    then return the bars for that contract.

    Internally calls fetch_chain to fill any cache gaps. Side effect: the
    cache also gets neighboring strikes/expirations from the chain (within
    chain_filter), which is generally desirable for cache value.

    Args:
        opra_symbol: Standard OPRA contract symbol (e.g. 'SPY240202P00493500').
        start_pt: Start of range, naive PT.
        end_pt: End of range, naive PT. Inclusive.
        chain_filter: Filter for the underlying chain fetch. None means
            keep everything (safest — guarantees the requested contract is
            cached even if it's far from spot). Default differs from
            fetch_chain's default; here we don't filter by default.
        source: Provenance tag.

    Returns:
        Cached bars for the requested OPRA in chronological order.

    Raises:
        OratsTransientError, OratsPermanentError on API failures.
        ValueError on tz-aware datetimes or unparseable OPRA symbols.
    """
    _validate_naive(start_pt, "start_pt")
    _validate_naive(end_pt, "end_pt")
    parsed = parse_opra(opra_symbol)

    requested = TimeRange(start_pt=start_pt, end_pt=end_pt)
    existing = repo.get_windows_for_contract(opra_symbol)
    gaps = W.find_gaps(requested, existing)

    if not gaps:
        logger.info(
            "fetch_contract %s: fully cached for [%s, %s], no API calls",
            opra_symbol, start_pt, end_pt,
        )
        return repo.get_bars_for_contract(opra_symbol, start_pt, end_pt)

    logger.info(
        "fetch_contract %s: %d gap(s), fetching chain for ticker=%s",
        opra_symbol, len(gaps), parsed.root,
    )

    for gap in gaps:
        fetch_chain(
            ticker=parsed.root,
            start_pt=gap.start_pt,
            end_pt=gap.end_pt,
            chain_filter=chain_filter,
            source=source,
        )

    return repo.get_bars_for_contract(opra_symbol, start_pt, end_pt)


def fetch_contracts(
    specs: Sequence[tuple[str, datetime, datetime]],
    *,
    chain_filter: Optional[ChainFilter] = None,
    source: FetchSource = "historical_backfill",
) -> dict[str, list[OptionMinuteBar]]:
    """
    Fetch multiple contracts sequentially. See fetch_contract for arg shape.

    Per-contract failures are logged and skipped — the returned dict only
    contains successful contracts.
    """
    results: dict[str, list[OptionMinuteBar]] = {}
    for opra_symbol, start_pt, end_pt in specs:
        try:
            bars = fetch_contract(
                opra_symbol, start_pt, end_pt,
                chain_filter=chain_filter,
                source=source,
            )
            results[opra_symbol] = bars
        except Exception as e:
            logger.error("fetch_contracts: %s failed: %s", opra_symbol, e)
    return results


# ────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ────────────────────────────────────────────────────────────────────────

def _fetch_chain_chunk(
    *,
    ticker: str,
    chunk: TimeRange,
    chain_filter: Optional[ChainFilter],
    source: FetchSource,
    contracts_seen: set[str],
) -> FetchSummary:
    """
    Fetch one chunk's worth of chain data, parse, store, record windows.

    Returns a partial FetchSummary for this chunk (caller aggregates).
    """
    trade_date_param = _format_trade_date_param(chunk)
    params = {
        "ticker": ticker,
        "tradeDate": trade_date_param,
    }

    csv_text = get_csv(_CHAIN_PATH, params)
    bars, rows_received, rows_kept = parse_orats_csv(
        csv_text,
        chain_filter=chain_filter,
    )

    if not bars:
        logger.info(
            "_fetch_chain_chunk %s [%s, %s]: %d rows received, 0 kept after filter",
            ticker, chunk.start_pt, chunk.end_pt, rows_received,
        )
        return FetchSummary(
            ticker=ticker,
            time_range=chunk,
            api_calls=1,
            rows_received=rows_received,
            rows_kept=0,
            bars_inserted=0,
            bars_total=0,
            contracts_touched=0,
        )

    n_inserted = repo.insert_bars(bars)

    # Per-contract fetched_windows: one entry per unique opra_symbol seen
    # in this chunk's bars. This is what makes future fetch_contract calls
    # for these contracts cache-hit.
    chunk_contracts = {b.opra_symbol for b in bars}
    contracts_seen.update(chunk_contracts)

    # Count bars per contract for the row_count field on each window.
    # Most contracts will have one bar per minute in the chunk, so
    # row_count ≈ minutes in the chunk for typical data.
    bars_per_contract: dict[str, int] = {}
    for b in bars:
        bars_per_contract[b.opra_symbol] = bars_per_contract.get(b.opra_symbol, 0) + 1

    for opra_symbol, count in bars_per_contract.items():
        repo.record_fetched_window(FetchedWindow(
            opra_symbol=opra_symbol,
            window_start_pt=chunk.start_pt,
            window_end_pt=chunk.end_pt,
            row_count=count,
            source=source,
        ))

    logger.info(
        "_fetch_chain_chunk %s [%s, %s]: %d rows received, %d kept, "
        "%d bars (%d inserted, rest dup), %d unique contracts",
        ticker, chunk.start_pt, chunk.end_pt,
        rows_received, rows_kept, len(bars), n_inserted, len(chunk_contracts),
    )

    return FetchSummary(
        ticker=ticker,
        time_range=chunk,
        api_calls=1,
        rows_received=rows_received,
        rows_kept=rows_kept,
        bars_inserted=n_inserted,
        bars_total=len(bars),
        contracts_touched=len(chunk_contracts),
    )


def _format_trade_date_param(chunk: TimeRange) -> str:
    """
    Format a TimeRange (naive PT) as ORATS' tradeDate parameter.

    ORATS expects Eastern Time in YYYYMMDDHHMM. Single-minute requests use
    one timestamp; ranges use comma-separated start,end.
    """
    start_et = _pt_to_et(chunk.start_pt)
    end_et = _pt_to_et(chunk.end_pt)

    if chunk.start_pt == chunk.end_pt:
        return start_et.strftime("%Y%m%d%H%M")

    return (
        f"{start_et.strftime('%Y%m%d%H%M')},"
        f"{end_et.strftime('%Y%m%d%H%M')}"
    )


def _pt_to_et(pt_naive: datetime) -> datetime:
    """Convert naive Pacific Time to naive Eastern Time."""
    pt_aware = pt_naive.replace(tzinfo=_PT)
    et_aware = pt_aware.astimezone(_ET)
    return et_aware.replace(tzinfo=None)


def _validate_naive(d: datetime, name: str) -> None:
    if d.tzinfo is not None:
        raise ValueError(
            f"{name} must be a naive datetime (interpreted as Pacific Time). "
            f"Got tz-aware: {d}"
        )
