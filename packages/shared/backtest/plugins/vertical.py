"""VerticalPlugin — credit or debit call/put vertical spread."""
from __future__ import annotations

from typing import Optional

from packages.shared.strategy_templates import Leg
from packages.shared.backtest.models import QuoteMap
from packages.shared.backtest.net_price import net_price_from_real_quotes


class VerticalPlugin:
    """Plugin for two-leg vertical spreads (directional_spread_to_target and debit_spread_to_target).

    is_touch is an UNDERLYING event: ES price reaching or exceeding drift_target.
    Option quotes are only read AFTER a touch is found, not as part of touch detection.
    """

    def net_price(self, legs: list[Leg], quote_map: QuoteMap) -> Optional[float]:
        """Net position value at given quotes. Negative = credit spread (received cash)."""
        return net_price_from_real_quotes(legs, quote_map)

    def payoff(self, legs: list[Leg], underlying_price: float) -> float:
        """Intrinsic position value at expiry. Sign: short=-1, long=+1.

        For a credit call spread (short low-K, long high-K), returns:
          S ≤ short_K  → 0.0         (max profit zone: both OTM)
          short_K < S < long_K → -(S - short_K)  (partial loss)
          S ≥ long_K   → -(long_K - short_K)     (max loss = -width)

        The three-zone payoff is structurally correct for both credit and debit
        verticals because the sign convention is applied per-leg.
        """
        total = 0.0
        for leg in legs:
            if leg.type == "call":
                intrinsic = max(underlying_price - leg.strike, 0.0)
            else:
                intrinsic = max(leg.strike - underlying_price, 0.0)
            sign = -1.0 if leg.side == "short" else 1.0
            total += sign * leg.quantity * intrinsic
        return total

    def is_touch(self, drift_target: float, es_price: float) -> bool:
        """Touch = ES price reached or exceeded the drift_target level."""
        return es_price >= drift_target
