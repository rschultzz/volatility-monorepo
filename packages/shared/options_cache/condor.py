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
import math
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from .opra import format_opra

logger = logging.getLogger(__name__)


# Calendar-time convention. Matches _implied_sigma_move in
# apps/web/modules/BacktestsV2/service.py and the inline formula in
# react_price_preview/src/App.jsx::bandsLevels.
_MINUTES_PER_CALENDAR_YEAR = 60.0 * 24.0 * 365.0


def condor_strikes_from_smile(
    spx: float,
    iv_pct: float,
    minutes_to_expiry: float,
    *,
    wing_width_pts: float = 10.0,
    strike_increment: float = 5.0,
) -> Optional[dict]:
    """
    Derive the four ±1σ iron-condor strikes from a smile timeslice.

    Returns a dict with sigma_pts plus the four SPX strikes. Returns None
    if inputs are invalid (non-positive, non-finite, or missing).

    Single source of truth for the strike math shared between:
      - backend _compute_hypothetical_condor (scan pipeline)
      - frontend bandsLevels useMemo (dashboard overlay, via the
        /api/condor-pricing endpoint that wraps this helper)

    wing_width_pts and strike_increment default to the values that have
    always been hardcoded on the frontend (10 and 5); the backend can
    override both via the scan-row pipeline's condor_wing_width_pts knob.
    """
    if spx is None or iv_pct is None or minutes_to_expiry is None:
        return None
    try:
        spx_f = float(spx)
        iv_f = float(iv_pct)
        mins_f = float(minutes_to_expiry)
    except (TypeError, ValueError):
        return None
    if spx_f <= 0 or iv_f <= 0 or mins_f <= 0:
        return None

    sigma_pts = spx_f * (iv_f / 100.0) * math.sqrt(mins_f / _MINUTES_PER_CALENDAR_YEAR)
    if not math.isfinite(sigma_pts) or sigma_pts <= 0:
        return None

    inc = float(strike_increment)
    wing = float(wing_width_pts)
    short_put = math.floor((spx_f - sigma_pts) / inc) * inc
    short_call = math.ceil((spx_f + sigma_pts) / inc) * inc
    long_put = round((short_put - wing) / inc) * inc
    long_call = round((short_call + wing) / inc) * inc

    return {
        "sigma_pts": sigma_pts,
        "short_put": short_put,
        "long_put": long_put,
        "short_call": short_call,
        "long_call": long_call,
    }


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
    underlying: str = "SPX",
    expiration: Optional[date] = None,
) -> list[Leg]:
    """
    Derive the OPRA contracts for both horizons of a condor scan row.

    Reads native SPX strikes from row['hypothetical_condor_120m'] and
    row['hypothetical_condor_to_close'] and encodes them directly into
    SPX OPRA symbols. ORATS' Live Intraday subscription covers SPX at
    the minute level, so no SPY proxy is needed.

    The returned list contains the union of both horizons — duplicates
    (when 120m and to_close share a strike) are removed but each unique
    leg gets a role tagged with the horizon(s) that produced it.

    Args:
        row: Scan row from saved-scan results.
        underlying: Root for the OPRA symbols. Defaults to 'SPX'.
        expiration: Override the expiration date. By default uses
            row['trade_date'] (correct for 0DTE setups, which is all we
            support today).

    Returns:
        List of Leg objects. Empty list if the row lacks the data needed
        to derive legs (missing strikes or trade_date).
    """
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

    # Build a side+role map: for each horizon, get the 4 SPX strikes and
    # emit a Leg. We dedupe on (option_type, strike) so if 120m and
    # to_close share a strike, we only emit one Leg.
    seen: dict[tuple[str, float], Leg] = {}

    for horizon in (HORIZON_120M, HORIZON_TO_CLOSE):
        strikes = _condor_strikes_for_horizon(row, horizon)
        if strikes is None:
            continue

        for role, spx_strike in strikes.items():
            option_type = "P" if "put" in role else "C"
            side = "short" if role.startswith("short") else "long"

            opra = format_opra(underlying, expiration, option_type, spx_strike)
            key = (option_type, spx_strike)

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
