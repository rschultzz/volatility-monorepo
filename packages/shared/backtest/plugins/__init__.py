"""Backtest plugins — one per structure type."""
from packages.shared.backtest.plugins.vertical import VerticalPlugin
from packages.shared.backtest.plugins.debit_vertical import DebitVerticalPlugin
from packages.shared.backtest.plugins.stub_condor import StubCondorPlugin

__all__ = ["VerticalPlugin", "DebitVerticalPlugin", "StubCondorPlugin"]
