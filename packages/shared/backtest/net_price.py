"""Real-quote net-price helper for the CR-AH backtest harness."""
from __future__ import annotations

from typing import Optional

from packages.shared.strategy_templates import Leg
from packages.shared.backtest.models import QuoteMap

_OPT_TYPE: dict[str, str] = {"call": "C", "put": "P"}


def net_price_from_real_quotes(
    legs: list[Leg],
    quote_map: QuoteMap,
) -> Optional[float]:
    """Return the net position value at the given real mid-quotes.

    Sign convention — short leg = -1, long leg = +1:
      - Negative return value → net credit received (e.g. credit call spread).
      - Positive return value → net debit paid (e.g. debit call spread).

    Returns None if ANY leg's mid is absent from quote_map for this minute.
    The harness treats None as "no fill possible at this minute."

    quote_map keys: (strike: float, option_type: 'C'|'P') → mid price.
    The key uses the single-letter encoding ('C'/'P') as stored in
    orats_options_minute.option_type, not the verbose 'call'/'put'.
    """
    total = 0.0
    for leg in legs:
        opt_type = _OPT_TYPE[leg.type]
        mid = quote_map.get((leg.strike, opt_type))
        if mid is None:
            return None
        sign = -1.0 if leg.side == "short" else 1.0
        total += sign * leg.quantity * mid
    return total
