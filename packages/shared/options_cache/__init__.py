"""
Options cache module.

On-demand caching of ORATS 1-minute options pricing data, keyed by OPRA
contract symbol. See SCHEMA_NOTES.md for design rationale.

Public API:

    Models (models.py):
        OptionMinuteBar, FetchedWindow, FetchJob, TimeRange,
        ChainFilter, DEFAULT_CHAIN_FILTER, FetchSummary

    OPRA helpers (opra.py):
        format_opra(root, expir, type, strike) -> str
        parse_opra(symbol) -> OpraSymbol

    Repository (repository.py):
        get_engine() / reset_engine()
        insert_bars(bars) -> int
        get_bars_for_contract(opra_symbol, start, end) -> list[OptionMinuteBar]
        count_bars_for_contract(opra_symbol, start, end) -> int
        record_fetched_window(window)
        get_windows_for_contract(opra_symbol) -> list[FetchedWindow]
        create_job(job), update_job_status, get_job

    Gap detection (windows.py):
        find_gaps, coverage_summary

    Fetching (fetcher.py):
        fetch_chain(ticker, start, end, chain_filter) -> FetchSummary
        fetch_contract(opra, start, end, chain_filter) -> list[OptionMinuteBar]
        fetch_contracts(specs, chain_filter) -> dict
        fetch_option_bars(opra_symbols, start_pt, end_pt) -> FetchOptionBarsSummary

    HTTP errors (http_client.py):
        OratsError, OratsTransientError, OratsPermanentError

Typical usage:

    from packages.shared.options_cache import fetch_chain, ChainFilter
    from datetime import datetime

    # Fetch the SPY chain for a 30-min window with default filter
    summary = fetch_chain(
        ticker="SPY",
        start_pt=datetime(2024, 2, 2, 9, 0),
        end_pt=datetime(2024, 2, 2, 9, 30),
    )
    print(f"inserted {summary.bars_inserted} bars")

    # Or get bars for a specific contract — internally fetches the chain
    # if needed, then returns just the matching bars
    from packages.shared.options_cache import fetch_contract
    bars = fetch_contract(
        "SPY240202P00493500",
        datetime(2024, 2, 2, 9, 0),
        datetime(2024, 2, 2, 9, 30),
    )
"""
from .condor import (
    Leg,
    condor_legs_for_row,
    condor_pricing_window_for_row,
)
from .fetcher import (
    fetch_chain,
    fetch_contract,
    fetch_contracts,
    fetch_option_bars,
)
from .http_client import OratsError, OratsPermanentError, OratsTransientError
from .models import (
    DEFAULT_CHAIN_FILTER,
    ChainFilter,
    FetchJob,
    FetchOptionBarsSummary,
    FetchSource,
    FetchSummary,
    FetchedWindow,
    JobKind,
    JobStatus,
    OptionMinuteBar,
    OptionType,
    TimeRange,
)
from .opra import OpraSymbol, format_opra, opra_to_orats_ticker, parse_opra
from .orchestrator import (
    OrchestratorResult,
    RowResult,
    Strategy,
    fetch_for_rows,
    get_strategy,
)

__all__ = [
    # models
    "OptionMinuteBar",
    "FetchedWindow",
    "FetchJob",
    "TimeRange",
    "OptionType",
    "FetchSource",
    "JobKind",
    "JobStatus",
    "ChainFilter",
    "DEFAULT_CHAIN_FILTER",
    "FetchSummary",
    "FetchOptionBarsSummary",
    # opra
    "OpraSymbol",
    "format_opra",
    "parse_opra",
    "opra_to_orats_ticker",
    # fetcher
    "fetch_chain",
    "fetch_contract",
    "fetch_contracts",
    "fetch_option_bars",
    # condor / strategies
    "Leg",
    "condor_legs_for_row",
    "condor_pricing_window_for_row",
    # orchestrator
    "fetch_for_rows",
    "get_strategy",
    "OrchestratorResult",
    "RowResult",
    "Strategy",
    # errors
    "OratsError",
    "OratsTransientError",
    "OratsPermanentError",
]
