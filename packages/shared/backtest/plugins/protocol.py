"""BacktestPlugin protocol — any plugin must satisfy this interface."""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from packages.shared.strategy_templates import Leg
from packages.shared.backtest.models import QuoteMap


@runtime_checkable
class BacktestPlugin(Protocol):
    """Structure-specific logic injected into BacktestHarness.

    A plugin encapsulates:
      touch_exit_is_meaningful — whether a touch event has a profitable exit P&L
      net_price  — real-quote position valuation
      payoff     — expiry intrinsic settlement
      is_touch   — whether an underlying price constitutes a "touch" event
      close_zone — three-zone settlement label (zone semantics differ by structure)

    touch_exit_is_meaningful (decision #6):
      True  → DEBIT (chase): touch = WIN; harness computes touch-exit P&L.
      False → CREDIT (fade): touch = loss developing; harness records touch as
              breach diagnostic only; touch_exit_pnl set to None.
    """

    touch_exit_is_meaningful: bool

    def net_price(self, legs: list[Leg], quote_map: QuoteMap) -> Optional[float]:
        """Net position value at given quotes. None if any leg is missing."""
        ...

    def payoff(self, legs: list[Leg], underlying_price: float) -> float:
        """Intrinsic position value at expiry for the given underlying price."""
        ...

    def is_touch(self, drift_target: float, es_price: float) -> bool:
        """True if es_price constitutes a touch of drift_target for this structure."""
        ...

    def close_zone(self, legs: list[Leg], settlement_price: float) -> str:
        """Three-zone settlement classification.

        Returns one of: 'max_profit' | 'partial_loss' | 'max_loss'.
        Zone semantics differ by structure (decision #5):
          CREDIT: max_profit when settle ≤ lower strike (both OTM, fade paid)
          DEBIT:  max_profit when settle ≥ upper strike (target reached, chase paid)
        """
        ...
