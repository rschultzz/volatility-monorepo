"""BacktestPlugin protocol — any plugin must satisfy this interface."""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

from packages.shared.strategy_templates import Leg
from packages.shared.backtest.models import QuoteMap


@runtime_checkable
class BacktestPlugin(Protocol):
    """Structure-specific logic injected into BacktestHarness.

    A plugin encapsulates:
      net_price  — real-quote position valuation
      payoff     — expiry intrinsic settlement
      is_touch   — whether an underlying price constitutes a "touch" event
    """

    def net_price(self, legs: list[Leg], quote_map: QuoteMap) -> Optional[float]:
        """Net position value at given quotes. None if any leg is missing."""
        ...

    def payoff(self, legs: list[Leg], underlying_price: float) -> float:
        """Intrinsic position value at expiry for the given underlying price."""
        ...

    def is_touch(self, drift_target: float, es_price: float) -> bool:
        """True if es_price constitutes a touch of drift_target for this structure."""
        ...
