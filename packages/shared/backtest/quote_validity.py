"""Quote-sanity rules for the edge harness (CR-AN, decision 2).

A minute is valid for a spread iff, for every leg, bid >= 0, ask > 0,
bid <= ask (both present), and the resulting spread value — the signed
position value the harness computes from mids (negative = net credit) — lies
in [0, width] in the structure's natural direction. Invalid minutes are
skipped, never clamped.

Two entry points:
  build_quote_map(quotes)        (strike, 'C'|'P', bid, ask) rows → QuoteMap of
                                  mids for the legs whose quote passes the
                                  leg rule; returns the count of rejected legs.
  spread_value_is_valid(v, w, legs)  the range rule on the signed value.
"""
from __future__ import annotations

from typing import Iterable, Optional

from packages.shared.backtest.models import QuoteMap
from packages.shared.strategy_templates import Leg

_EPS = 1e-9


def leg_quote_is_valid(bid, ask) -> bool:
    """bid >= 0, ask > 0, bid <= ask, both present and numeric."""
    if bid is None or ask is None:
        return False
    try:
        b = float(bid)
        a = float(ask)
    except (TypeError, ValueError):
        return False
    return b >= 0.0 and a > 0.0 and b <= a


def build_quote_map(quotes: Iterable[tuple]) -> tuple[QuoteMap, int]:
    """Build a QuoteMap of mids from (strike, option_type, bid, ask) rows.

    Legs failing leg_quote_is_valid are left out of the map (so
    net_price_from_real_quotes returns None for that minute) and counted.
    """
    qmap: QuoteMap = {}
    n_invalid = 0
    for strike, opt_type, bid, ask in quotes:
        if leg_quote_is_valid(bid, ask):
            qmap[(float(strike), opt_type)] = (float(bid) + float(ask)) / 2.0
        else:
            n_invalid += 1
    return qmap, n_invalid


def expected_position_sign(legs: list[Leg]) -> Optional[int]:
    """+1 when the structure is a net debit (long leg more in-the-money than
    the short leg), -1 when a net credit, None when not a two-leg vertical
    (the range rule then checks magnitude only)."""
    if len(legs) != 2:
        return None
    longs = [l for l in legs if l.side == "long"]
    shorts = [l for l in legs if l.side == "short"]
    if len(longs) != 1 or len(shorts) != 1 or longs[0].type != shorts[0].type:
        return None
    lg, sh = longs[0], shorts[0]
    if lg.type == "call":
        return 1 if lg.strike < sh.strike else -1
    return 1 if lg.strike > sh.strike else -1


def spread_value_is_valid(
    pos_val: Optional[float],
    width: float,
    legs: Optional[list[Leg]] = None,
    expected_sign: Optional[int] = None,
) -> bool:
    """Range rule: |value| <= width, and the value points in the structure's
    natural direction (a debit spread is never worth less than 0; a credit
    spread never more than 0). None is invalid."""
    if pos_val is None:
        return False
    if abs(pos_val) > width + _EPS:
        return False
    sign = expected_sign
    if sign is None and legs:
        sign = expected_position_sign(legs)
    if sign is None:
        return True
    return pos_val * sign >= -_EPS
