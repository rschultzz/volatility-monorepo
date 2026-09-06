"""Data models for the CR-AH backtest harness."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

from packages.shared.strategy_templates import Leg

# (strike: float, option_type: 'C'|'P') → mid price
QuoteMap = dict[tuple[float, str], float]

# Distance-band thresholds (in σ units, using drift_target_distance_sigma)
NEAR_SIGMA_MAX = 1.5
FAR_SIGMA_MIN = 2.0


def distance_band(sigma: float) -> str:
    """Map drift_target_distance_sigma to 'near' | 'mid' | 'far'."""
    if sigma < NEAR_SIGMA_MAX:
        return "near"
    if sigma < FAR_SIGMA_MIN:
        return "mid"
    return "far"


@dataclass
class TradeInput:
    signal_date: date
    partition: str           # 'train' | 'holdout'
    distance_band: str       # 'near' | 'mid' | 'far'
    drift_target_sigma: float
    structural_prob: float   # KNN or base-rate estimate for the target touch
    spread_width: float      # abs(long_strike - short_strike), e.g. 10.0
    legs: list[Leg]


@dataclass
class EntryMinuteScan:
    """One row per minute from the entry-day crawl.

    net_position_value is None when any leg is absent from the quote_map for
    that minute. Those minutes are still recorded so the full crawl is
    preserved (invariant #3).
    """
    snapshot_pt: datetime
    net_position_value: Optional[float]  # negative = net credit position
    net_credit: Optional[float]          # = -net_position_value; positive = credit received
    edge: Optional[float]                # structural_prob - (net_credit / spread_width)
    # CR-AN decision 2: False when the minute had a price that failed the
    # quote-sanity rule (out of [0, width] in the structure's direction) or a
    # leg was missing/invalid. Such minutes carry net_credit=None / edge=None.
    quote_valid: bool = True


@dataclass
class TradeResult:
    signal_date: date
    partition: str
    distance_band: str
    drift_target_sigma: float

    # Full per-minute crawl — ALL minutes, including those with None quotes
    entry_scan: list[EntryMinuteScan]

    # Fill (first minute where edge >= threshold)
    filled: bool
    fill_time: Optional[datetime]
    fill_net_credit: Optional[float]
    fill_edge: Optional[float]

    # Touch event (underlying only)
    touch_found: bool
    touch_time: Optional[datetime]       # ES minute when underlying reached drift_target
    touch_exit_minute: Optional[datetime]  # first option-quote minute at/after touch_time

    # P&L outcomes (filled trades only)
    touch_exit_pnl: Optional[float]      # P&L if closed at touch_exit_minute quotes
    close_pnl: Optional[float]           # P&L at expiry settlement
    close_zone: Optional[str]            # 'max_profit' | 'partial_loss' | 'max_loss'

    # Baseline: same outcomes but filled at first-quoted-minute regardless of edge
    baseline_net_credit: Optional[float]
    baseline_touch_exit_pnl: Optional[float]
    baseline_close_pnl: Optional[float]

    # Two-axis tagging (decision #12) — populated by Step 3 at write time, not by harness
    pattern_label: Optional[str] = None
    reversion_wilson_lo: Optional[float] = None
    continuation_wilson_lo: Optional[float] = None

    # CR-AN decision 7 observability + decision 6 exclusion
    n_minutes_total: int = 0
    n_minutes_valid: int = 0
    baseline_minute_offset: Optional[int] = None   # minutes from window open to first valid minute
    had_invalid_quote: bool = False
    excluded_reason: Optional[str] = None          # 'no_valid_entry_minute' when the window has no valid minute
