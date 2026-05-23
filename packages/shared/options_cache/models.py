"""
Dataclasses mirroring the options cache tables.

These are read/write value objects used by the repository. They intentionally
match column names and types from the SQL schema 1:1 — when you see a field
here, it's the same field in the DB.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal, Optional

OptionType = Literal["C", "P"]
FetchSource = Literal["historical_backfill", "live_poll", "manual"]
JobKind = Literal["historical_backfill", "live_setup", "manual"]
JobStatus = Literal["pending", "running", "completed", "failed"]


@dataclass(frozen=True)
class ChainFilter:
    """
    Filter applied to chain-endpoint responses before insertion.

    The strikes/chain endpoint returns the full options chain (typically
    several thousand rows per minute). Most rows aren't useful for any
    given strategy — deep ITM/OTM strikes far from spot, LEAPS expirations
    hundreds of days out, etc. The filter trims to a useful neighborhood
    before bars hit the cache.

    Strike bounds are expressed as a fraction of the row's stockPrice so
    they auto-scale with the underlying. Default is ±10% of spot.

    DTE bounds are absolute. Default keeps the front 60 days.

    Defaults are tuned for short-dated SPY/SPX-style strategies. For
    longer-horizon work (calendars, diagonals on equities), broaden
    max_dte. For analyses that need the whole chain (e.g. surface
    visualization), pass None to skip filtering.
    """
    min_strike_pct_of_spot: float = 0.90
    max_strike_pct_of_spot: float = 1.10
    min_dte: int = 0
    max_dte: int = 60

    def passes(
        self,
        strike: float,
        dte: Optional[int],
        stock_price: Optional[float],
    ) -> bool:
        """
        Return True if a chain row's (strike, dte, spot) passes this filter.

        Rows with missing strike or stock_price fail the strike check
        (we don't have enough info to evaluate). Missing DTE fails the
        DTE check. Be conservative — when in doubt, drop the row.
        """
        if dte is None or not (self.min_dte <= dte <= self.max_dte):
            return False
        if stock_price is None or stock_price <= 0:
            return False
        ratio = strike / stock_price
        return self.min_strike_pct_of_spot <= ratio <= self.max_strike_pct_of_spot


# Default filter for typical 0DTE / short-dated workflows. Used by
# fetch_chain when no explicit filter is provided.
DEFAULT_CHAIN_FILTER = ChainFilter(
    min_strike_pct_of_spot=0.90,
    max_strike_pct_of_spot=1.10,
    min_dte=0,
    max_dte=60,
)


@dataclass
class OptionMinuteBar:
    """
    One row in orats_options_minute. One OPRA contract at one minute.

    Most fields can be populated from an ORATS API response after column
    renaming and side-normalization. The two _d / _utc fields are derived
    at insert time from the corresponding raw text/EST fields.
    """
    # Contract identity
    opra_symbol: str
    ticker: str
    expir_date: str            # raw text from ORATS, e.g. '2026-01-17'
    expir_date_d: date         # derived
    strike: float
    option_type: OptionType

    # Timestamps
    trade_date: str            # raw text from ORATS
    trade_date_d: date         # derived
    quote_date: str            # raw text from ORATS
    snapshot_pt: datetime      # naive PT (matches monies convention)
    snapshot_utc: datetime     # tz-aware UTC

    # Quote
    bid_price: float
    ask_price: float

    # Greeks (only delta is required)
    delta: float

    # Optional fields
    updated_at: Optional[str] = None
    snap_shot_est_time: Optional[int] = None
    snap_shot_date: Optional[str] = None

    stock_price: Optional[float] = None
    spot_price: Optional[float] = None
    dte: Optional[int] = None

    bid_size: Optional[int] = None
    ask_size: Optional[int] = None

    bid_iv: Optional[float] = None
    mid_iv: Optional[float] = None
    ask_iv: Optional[float] = None

    volume: Optional[int] = None
    open_interest: Optional[int] = None

    opt_value: Optional[float] = None
    smv_vol: Optional[float] = None
    ext_val: Optional[float] = None
    ext_smv_vol: Optional[float] = None
    residual_rate: Optional[float] = None

    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None
    phi: Optional[float] = None
    driftless_theta: Optional[float] = None

    expiry_tod: Optional[str] = None
    ticker_id: Optional[int] = None
    month_id: Optional[int] = None


@dataclass
class FetchedWindow:
    """One row in orats_options_fetched_windows."""
    opra_symbol: str
    window_start_pt: datetime
    window_end_pt: datetime
    row_count: int
    source: FetchSource
    fetched_at: Optional[datetime] = None  # set by DB default if None on insert


@dataclass
class FetchJob:
    """One row in orats_options_fetch_jobs."""
    kind: JobKind
    legs_requested: list[dict[str, Any]]
    setup_ref: Optional[str] = None
    status: JobStatus = "pending"
    legs_completed: int = 0
    error_message: Optional[str] = None

    # Set after insert (job_id) or by status transitions
    job_id: Optional[int] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


@dataclass
class TimeRange:
    """A simple [start, end] inclusive time range in naive PT."""
    start_pt: datetime
    end_pt: datetime

    def __post_init__(self) -> None:
        if self.end_pt < self.start_pt:
            raise ValueError(
                f"TimeRange end must be >= start; got "
                f"start={self.start_pt}, end={self.end_pt}"
            )

    def overlaps(self, other: "TimeRange") -> bool:
        """True if this range shares at least one minute with other."""
        return self.start_pt <= other.end_pt and other.start_pt <= self.end_pt

    def contains(self, other: "TimeRange") -> bool:
        """True if other is fully within this range."""
        return self.start_pt <= other.start_pt and other.end_pt <= self.end_pt


@dataclass
class FetchSummary:
    """
    Summary of a chain-fetch operation. Returned by fetch_chain so callers
    can see what happened without materializing every bar.
    """
    ticker: str
    time_range: TimeRange
    api_calls: int          # number of HTTP requests made
    rows_received: int      # raw chain rows from the API (pre-filter)
    rows_kept: int          # rows that passed the filter
    bars_inserted: int      # actually inserted (excludes ON CONFLICT skips)
    bars_total: int         # call+put bars built from kept rows (= 2 × rows_kept)
    contracts_touched: int  # unique opra_symbols seen


@dataclass
class FetchOptionBarsSummary:
    """
    Summary of a fetch_option_bars (option-endpoint) operation.

    Shaped for the per-OPRA, gap-aware fetch path. Distinct from FetchSummary
    (chain-shaped, single-ticker) — see CR-004 spec for the why.
    """
    opras_processed: int   # input list length
    gaps_filled: int       # count of gaps detected and fetched across all OPRAs
    bars_written: int      # inserted into orats_options_minute (excludes ON CONFLICT skips)
    cache_hits: int        # OPRAs that needed zero HTTP calls (fully cached)

