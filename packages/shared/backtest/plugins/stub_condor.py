"""StubCondorPlugin — satisfies BacktestPlugin protocol for seam testing.

An iron condor wins by price staying inside its range (NOT touching its
outer strikes). This stub implements only enough logic to prove the plugin
seam works end-to-end; it is not a production condor plugin.
"""
from __future__ import annotations

from typing import Optional

from packages.shared.strategy_templates import Leg
from packages.shared.backtest.models import QuoteMap
from packages.shared.backtest.net_price import net_price_from_real_quotes


class StubCondorPlugin:
    """Stub iron condor plugin. Passes BacktestPlugin protocol check."""

    touch_exit_is_meaningful: bool = True

    def net_price(self, legs: list[Leg], quote_map: QuoteMap) -> Optional[float]:
        return net_price_from_real_quotes(legs, quote_map)

    def payoff(self, legs: list[Leg], underlying_price: float) -> float:
        # Stub returns 0.0; real condor payoff is four-zone.
        return 0.0

    def is_touch(self, drift_target: float, es_price: float) -> bool:
        # Stub treats reaching drift_target as a touch (conservative).
        return es_price >= drift_target

    def close_zone(self, legs: list[Leg], settlement_price: float) -> str:
        # Stub always returns max_profit; real condor zone is four-zone.
        return "max_profit"
