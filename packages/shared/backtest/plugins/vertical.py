"""VerticalPlugin — credit call vertical spread (directional_spread_to_target).

Structure: short target, long target+10.
Thesis: when the magnet is far, price is unlikely to reach it — profit when it fails.
Touch for credit = loss developing (breach diagnostic only, not a P&L).
"""
from __future__ import annotations

from typing import Optional

from packages.shared.strategy_templates import Leg
from packages.shared.backtest.models import QuoteMap
from packages.shared.backtest.net_price import net_price_from_real_quotes


class VerticalPlugin:
    """Credit call vertical: short lower strike, long higher strike.

    touch_exit_is_meaningful = False (decision #6): a touch means the short
    strike was reached — that is the loss developing for a credit fade. The
    harness records the touch event as a breach diagnostic but does NOT compute
    a touch-exit P&L. Only the close path (expiry settlement) is scored as P&L.

    is_touch is an UNDERLYING event: ES price reaching or exceeding drift_target.
    Option quotes are only read AFTER a touch is found, not as part of touch detection.
    """

    touch_exit_is_meaningful: bool = False

    def net_price(self, legs: list[Leg], quote_map: QuoteMap) -> Optional[float]:
        """Net position value at given quotes. Negative = credit spread (received cash)."""
        return net_price_from_real_quotes(legs, quote_map)

    def payoff(self, legs: list[Leg], underlying_price: float) -> float:
        """Intrinsic position value at expiry. Sign: short=-1, long=+1.

        For a credit call spread (short low-K, long high-K):
          S ≤ short_K  → 0.0         (max profit: both OTM)
          short_K < S < long_K → -(S - short_K)  (partial loss)
          S ≥ long_K   → -(long_K - short_K)     (max loss = -width)
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

    def close_zone(self, legs: list[Leg], settlement_price: float) -> str:
        """Credit-spread zone classification.

        max_profit  — settle ≤ lower strike (both OTM; fade paid in full)
        partial_loss — lower < settle < upper
        max_loss    — settle ≥ upper strike (both ITM; spread at full width)
        """
        strikes = sorted(set(l.strike for l in legs))
        if len(strikes) < 2:
            return "unknown"
        low, high = strikes[0], strikes[-1]
        if settlement_price <= low:
            return "max_profit"
        if settlement_price < high:
            return "partial_loss"
        return "max_loss"
