"""DebitVerticalPlugin — debit call vertical spread (debit_spread_to_target).

Structure: long target−10, short target.
Thesis: price is drawn UP to the magnet; ride to it, take profit on touch.
Touch for debit = WIN (the profitable exit). Both touch-exit and close P&L are scored.
"""
from __future__ import annotations

from typing import Optional

from packages.shared.strategy_templates import Leg
from packages.shared.backtest.models import QuoteMap
from packages.shared.backtest.net_price import net_price_from_real_quotes


class DebitVerticalPlugin:
    """Debit call vertical: long lower strike (target−10), short upper strike (target).

    touch_exit_is_meaningful = True (decision #6): a touch means the short
    strike (= drift_target) was reached — that is the WIN for the chase. The
    harness computes touch-exit P&L (close at touch-moment quotes) AND the
    expiry close P&L. Both outcomes are scored.

    Payoff orientation (MIRRORED from credit, decision #5):
      settle ≥ short (target reached) → max profit  = +(width − debit)
      settle ≤ long  (target−10)      → max loss    = −debit
      between                          → linear partial

    The per-leg sign convention (+1 for long, −1 for short) already produces
    this mirrored payoff correctly without any orientation override.

    is_touch is an UNDERLYING event: ES price reaching or exceeding drift_target.
    """

    touch_exit_is_meaningful: bool = True

    def net_price(self, legs: list[Leg], quote_map: QuoteMap) -> Optional[float]:
        """Net position value at given quotes. Positive = debit spread (paid cash)."""
        return net_price_from_real_quotes(legs, quote_map)

    def payoff(self, legs: list[Leg], underlying_price: float) -> float:
        """Intrinsic position value at expiry. Sign: short=-1, long=+1.

        For a debit call spread (long low-K, short high-K):
          S ≤ long_K   → 0.0              (max loss zone: both OTM; P&L = 0 − debit)
          long_K < S < short_K → +(S − long_K)  (partial profit)
          S ≥ short_K  → +(short_K − long_K)    (max profit = +width; P&L = width − debit)
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
        """Touch = ES price reached or exceeded the drift_target level (the WIN)."""
        return es_price >= drift_target

    def close_zone(self, legs: list[Leg], settlement_price: float) -> str:
        """Debit-spread zone classification (MIRRORED from credit).

        max_profit   — settle ≥ upper strike (target reached; chase paid in full)
        partial_loss — lower < settle < upper (partial profit for debit, not a loss)
        max_loss     — settle ≤ lower strike (both OTM; lost the debit)

        Note: 'partial_loss' is a carryover label from credit convention.
        For debit, the partial zone is actually a partial PROFIT. Step 4 analysis
        must interpret zones in light of the structure type.
        """
        strikes = sorted(set(l.strike for l in legs))
        if len(strikes) < 2:
            return "unknown"
        low, high = strikes[0], strikes[-1]
        if settlement_price <= low:
            return "max_loss"
        if settlement_price < high:
            return "partial_loss"
        return "max_profit"
