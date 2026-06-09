"""
packages.shared.backtest — structure-agnostic backtest harness (CR-AH).

Public surface:
    BacktestHarness  — run_trade() entry point
    TradeInput       — per-date trade spec
    TradeResult      — full result including entry-day crawl
    EntryMinuteScan  — per-minute scan row (ALL minutes, not just fills)
    QuoteMap         — dict[(strike, 'C'|'P')] → mid price
    distance_band    — sigma → 'near' | 'mid' | 'far'

Plugins:
    VerticalPlugin   — credit/debit call or put vertical spread
    StubCondorPlugin — protocol stub for seam testing
"""
from packages.shared.backtest.models import (
    QuoteMap,
    TradeInput,
    EntryMinuteScan,
    TradeResult,
    distance_band,
    NEAR_SIGMA_MAX,
    FAR_SIGMA_MIN,
)
from packages.shared.backtest.harness import BacktestHarness
from packages.shared.backtest.plugins.vertical import VerticalPlugin
from packages.shared.backtest.plugins.stub_condor import StubCondorPlugin

__all__ = [
    "QuoteMap",
    "TradeInput",
    "EntryMinuteScan",
    "TradeResult",
    "distance_band",
    "NEAR_SIGMA_MAX",
    "FAR_SIGMA_MIN",
    "BacktestHarness",
    "VerticalPlugin",
    "StubCondorPlugin",
]
