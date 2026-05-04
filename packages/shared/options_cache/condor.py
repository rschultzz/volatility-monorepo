"""
Strategy-specific leg derivation for the options cache.

Phase 4a starts with a single strategy: iron condor. Each strategy module
exposes two pure functions that the orchestrator calls:

    legs_for_row(row)      -> list[Leg]
        Given a scan row, return the OPRA contracts that make up the trade.

    pricing_window_for_row(row) -> (start_pt, end_pt)
        Given a scan row, return the [start, end] time range for which we
        want pricing data. Naive Pacific Time, matching cache convention.

When we add verticals, butterflies, etc., we'll write parallel modules
following the same shape. A registry will dispatch by strategy name.

For now, only condor is implemented. Imports stay loose so the registry
can expand later without code changes elsewhere.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from .opra import format_opra

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────────────────
#  Leg representation
# ────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Leg:
    """
    One leg of a multi-leg strategy.

    side:  'long' or 'short' (from the trader's perspective)
    role:  human-readable label like 'short_put', 'long_call', 'short_strike',
           'wing_long' — strategy-defined, used by P&L code in Phase 5.
    ratio: contract count (typically 1 for condors; could be 2/1 for
           ratio spreads later).
    """
    opra_symbol: str
    side: str         # 'long' | 'short'
    role: str
    ratio: int = 1


# ────────────────────────────────────────────────────────────────────────
#  Time utilities (PT-naive, matching the cache convention)
# ────────────────────────────────────────────────────────────────────────

_PT = ZoneInfo("America/Los_Angeles")
_UTC = ZoneInfo("UTC")

# Regular Trading Hours close in Pacific Time. SPX/SPXW PM-settled options
# expire at this time. SPY weeklies close here too.
RTH_CLOSE_PT = (13, 0)  # 13:00 PT = 16:00 ET


def parse_utc_iso(s: str) -> datetime:
    """Parse an ISO-8601 UTC string into a tz-aware datetime."""
    # Tolerate trailing 'Z' (Python 3.10 fromisoformat doesn't until 3.11)
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s).astimezone(_UTC)


def utc_to_pt_naive(dt_utc: datetime) -> datetime:
    """Convert tz-aware UTC datetime to naive Pacific Time."""
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=_UTC)
    return dt_utc.astimezone(_PT).replace(tzinfo=None)


def end_of_day_pt(trade_date: date) -> datetime:
    """Return naive PT timestamp for the close of the given trade date."""
    return datetime.combine(trade_date, datetime.min.time()).replace(
        hour=RTH_CLOSE_PT[0], minute=RTH_CLOSE_PT[1]
    )


# ────────────────────────────────────────────────────────────────────────
#  SPX → SPY strike mapping
# ────────────────────────────────────────────────────────────────────────

def map_spx_strike_to_spy(spx_strike: float, spx_spot: float) -> float:
    """
    Map an SPX strike to the equivalent SPY strike using spot-relative
    percentage scaling.

    Math: the SPX strike is at (spx_strike / spx_spot) of SPX spot.
    We want a SPY strike at the same fractional offset from SPY spot.
    Since SPY ≈ SPX/10 (within basis), we use SPX spot/10 as a stand-in
    for SPY spot — the rounding-to-nearest-dollar step absorbs the basis.

    Returns the SPY strike rounded to the nearest dollar.

    Examples (SPX spot 4940, SPX strike 4935):
        ratio = 4935 / 4940 = 0.998987...
        spy_spot ≈ 4940 / 10 = 494.0
        spy_strike = 0.998987 × 494.0 = 493.5
        rounded = 494
    """
    if spx_spot <= 0:
        raise ValueError(f"spx_spot must be positive, got {spx_spot}")
    if spx_strike <= 0:
        raise ValueError(f"spx_strike must be positive, got {spx_strike}")

    ratio = spx_strike / spx_spot
    spy_spot_proxy = spx_spot / 10.0
    spy_strike_unrounded = ratio * spy_spot_proxy
    return float(round(spy_strike_unrounded))


# ────────────────────────────────────────────────────────────────────────
#  Condor strategy
# ────────────────────────────────────────────────────────────────────────

# Roles used in returned Legs. P&L code in Phase 5 will reason about these.
ROLE_SHORT_PUT = "short_put"
ROLE_LONG_PUT = "long_put"
ROLE_SHORT_CALL = "short_call"
ROLE_LONG_CALL = "long_call"

# Horizon labels used to disambiguate when the same scan row has both
# 120m-horizon strikes and entry-to-close strikes.
HORIZON_120M = "120m"
HORIZON_TO_CLOSE = "to_close"


def _condor_strikes_for_horizon(row: dict, horizon: str) -> Optional[dict]:
    """
    Pull the 4 SPX strike values for one horizon from a scan row.

    Returns None if any strike is missing (skip the row in that case).
    The shape on the row is:
        row['hypothetical_condor_120m'] = {
            'short_put_strike': 4935.0, 'long_put_strike': 4925.0,
            'short_call_strike': 4980.0, 'long_call_strike': 4990.0,
            ...
        }
    """
    key = f"hypothetical_condor_{horizon}"
    block = row.get(key)
    if not block:
        return None

    strikes = {
        "short_put": block.get("short_put_strike"),
        "long_put": block.get("long_put_strike"),
        "short_call": block.get("short_call_strike"),
        "long_call": block.get("long_call_strike"),
    }
    if any(s is None for s in strikes.values()):
        return None
    return strikes


def condor_legs_for_row(
    row: dict,
    *,
    underlying: str = "SPY",
    expiration: Optional[date] = None,
) -> list[Leg]:
    """
    Derive the OPRA contracts for both horizons of a condor scan row.

    Reads SPX strikes from row['hypothetical_condor_120m'] and
    row['hypothetical_condor_to_close']. Maps each to the nearest SPY
    strike using spot-relative percentage from row['target_spx_price'].

    The returned list contains the union of both horizons — duplicates
    (when 120m and to_close strikes round to the same SPY strike) are
    removed but each unique leg gets a role tagged with the horizon(s)
    that produced it.

    Args:
        row: Scan row from saved-scan results.
        underlying: Ticker for the OPRA symbols. Currently always 'SPY'.
            When SPX intraday becomes available, this will be configurable.
        expiration: Override the expiration date. By default uses
            row['trade_date'] (correct for 0DTE setups, which is all we
            support today).

    Returns:
        List of Leg objects. Empty list if the row lacks the data needed
        to derive legs (missing strikes or spot).
    """
    spx_spot = row.get("target_spx_price")
    if spx_spot is None or spx_spot <= 0:
        logger.debug(
            "condor_legs_for_row: missing/invalid target_spx_price (%s) on row",
            spx_spot,
        )
        return []

    if expiration is None:
        trade_date_str = row.get("trade_date")
        if not trade_date_str:
            logger.debug("condor_legs_for_row: missing trade_date on row")
            return []
        try:
            expiration = date.fromisoformat(trade_date_str[:10])
        except ValueError:
            logger.debug(
                "condor_legs_for_row: unparseable trade_date %r",
                trade_date_str,
            )
            return []

    # Build a side+role map: for each horizon, get the 4 SPX strikes,
    # convert to SPY, and emit a Leg. We dedupe on (option_type, strike)
    # so if 120m and to_close round to the same SPY strike, we only
    # emit one Leg.
    seen: dict[tuple[str, float], Leg] = {}

    for horizon in (HORIZON_120M, HORIZON_TO_CLOSE):
        strikes = _condor_strikes_for_horizon(row, horizon)
        if strikes is None:
            continue

        for role, spx_strike in strikes.items():
            spy_strike = map_spx_strike_to_spy(spx_strike, spx_spot)
            option_type = "P" if "put" in role else "C"
            side = "short" if role.startswith("short") else "long"

            opra = format_opra(underlying, expiration, option_type, spy_strike)
            key = (option_type, spy_strike)

            if key in seen:
                # Same strike already added from the other horizon.
                # Annotate the role so P&L code can tell.
                existing = seen[key]
                # If this horizon's role differs from existing, append it.
                if horizon not in existing.role:
                    new_role = f"{existing.role}+{horizon}:{role}"
                    seen[key] = Leg(
                        opra_symbol=existing.opra_symbol,
                        side=existing.side,
                        role=new_role,
                        ratio=existing.ratio,
                    )
            else:
                seen[key] = Leg(
                    opra_symbol=opra,
                    side=side,
                    role=f"{horizon}:{role}",
                    ratio=1,
                )

    return list(seen.values())


def condor_pricing_window_for_row(row: dict) -> Optional[tuple[datetime, datetime]]:
    """
    Time range we want pricing for, in naive Pacific Time.

    Start: setup time (target_ts_utc, converted to PT).
    End: close of trading day same date (13:00 PT).

    Returns None if the row lacks the data needed.
    """
    target_ts_utc = row.get("target_ts_utc") or row.get("start_ts_utc")
    if not target_ts_utc:
        logger.debug("condor_pricing_window_for_row: missing target/start_ts_utc")
        return None

    try:
        if isinstance(target_ts_utc, str):
            target_dt = parse_utc_iso(target_ts_utc)
        elif isinstance(target_ts_utc, (int, float)):
            target_dt = datetime.fromtimestamp(target_ts_utc, tz=_UTC)
        else:
            target_dt = target_ts_utc  # assume datetime
            if target_dt.tzinfo is None:
                target_dt = target_dt.replace(tzinfo=_UTC)
    except (ValueError, TypeError) as e:
        logger.debug("condor_pricing_window_for_row: bad timestamp %r: %s",
                     target_ts_utc, e)
        return None

    start_pt = utc_to_pt_naive(target_dt)
    end_pt = end_of_day_pt(start_pt.date())

    # Edge case: setup happened past close (data anomaly or testing).
    # Fall back to a single-minute fetch at the setup time.
    if end_pt < start_pt:
        logger.warning(
            "condor_pricing_window_for_row: setup time %s is after RTH close; "
            "falling back to single-minute window",
            start_pt,
        )
        end_pt = start_pt

    return (start_pt, end_pt)
