"""BacktestHarness — structure-agnostic runner for per-date trade evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

from packages.shared.backtest.models import (
    EntryMinuteScan,
    QuoteMap,
    TradeInput,
    TradeResult,
)
from packages.shared.backtest.plugins.protocol import BacktestPlugin
from packages.shared.strategy_templates import Leg


@dataclass
class BacktestHarness:
    """Runs one trade through entry-crawl → touch-detect → close, returning a full TradeResult.

    Locked decisions encoded as constructor parameters (invariant #2):
      plugin          — structure-specific logic (net_price, payoff, is_touch)
      edge_threshold  — minimum edge required to fill on any given minute
      split_date      — dates <= split_date land in 'train'; later dates in 'holdout'
      fill_rule       — how to select the fill minute from the entry-day crawl
      baseline        — how to select the baseline fill minute (no edge gate)
    """

    plugin: BacktestPlugin
    edge_threshold: float   # e.g. 0.05
    split_date: date
    fill_rule: str = "first_above_threshold"
    baseline: str = "first_quoted_minute"

    def run_trade(
        self,
        trade_input: TradeInput,
        entry_day_scan: list[tuple[datetime, QuoteMap]],
        touch_datetime: Optional[datetime],
        touch_window_scan: list[tuple[datetime, QuoteMap]],
        settlement_price: Optional[float],
    ) -> TradeResult:
        """Evaluate one signal date end-to-end.

        Args:
            trade_input:       Per-date trade spec (legs, structural_prob, etc.)
            entry_day_scan:    [(snapshot_pt, quote_map), ...] sorted ascending.
                               Covers the full entry day open-to-close window.
            touch_datetime:    First ES-bar minute where underlying >= drift_target.
                               None if no touch occurred on the expiry day.
            touch_window_scan: Option quotes on/after the expiry day, used to find
                               the close price at/after touch_datetime.
                               Empty list if no touch.
            settlement_price:  ES settlement price at expiry (underlying only).
                               None if expiry data unavailable.

        Returns:
            TradeResult with ALL minutes in entry_scan (invariant #3), regardless
            of whether a fill occurred or quotes were present.
        """
        scan_rows, baseline_snap, fill_snap = self._run_entry_crawl(trade_input, entry_day_scan)

        filled = fill_snap is not None

        touch_exit_pnl, touch_exit_minute = self._touch_exit(
            filled, fill_snap, touch_datetime, touch_window_scan, trade_input
        )
        close_pnl, close_zone = self._close_outcome(
            filled, fill_snap, settlement_price, trade_input
        )
        baseline_touch_exit_pnl, baseline_close_pnl = self._baseline_outcomes(
            baseline_snap, touch_datetime, touch_window_scan, settlement_price, trade_input
        )

        return TradeResult(
            signal_date=trade_input.signal_date,
            partition=trade_input.partition,
            distance_band=trade_input.distance_band,
            drift_target_sigma=trade_input.drift_target_sigma,
            entry_scan=scan_rows,
            filled=filled,
            fill_time=fill_snap[0] if fill_snap else None,
            fill_net_credit=fill_snap[1] if fill_snap else None,
            fill_edge=fill_snap[2] if fill_snap else None,
            touch_found=touch_datetime is not None,
            touch_time=touch_datetime,
            touch_exit_minute=touch_exit_minute,
            touch_exit_pnl=touch_exit_pnl,
            close_pnl=close_pnl,
            close_zone=close_zone,
            baseline_net_credit=baseline_snap[1] if baseline_snap else None,
            baseline_touch_exit_pnl=baseline_touch_exit_pnl,
            baseline_close_pnl=baseline_close_pnl,
        )

    # ── private helpers ──────────────────────────────────────────────────────

    def _run_entry_crawl(
        self,
        trade_input: TradeInput,
        entry_day_scan: list[tuple[datetime, QuoteMap]],
    ) -> tuple[
        list[EntryMinuteScan],
        Optional[tuple[datetime, float]],       # baseline: (time, net_credit)
        Optional[tuple[datetime, float, float]], # fill:     (time, net_credit, edge)
    ]:
        scan_rows: list[EntryMinuteScan] = []
        baseline_snap: Optional[tuple[datetime, float]] = None
        fill_snap: Optional[tuple[datetime, float, float]] = None

        for snap_pt, qmap in entry_day_scan:
            pos_val = self.plugin.net_price(trade_input.legs, qmap)
            if pos_val is None:
                # Invariant #1: missing quote → skip this minute. Still persisted.
                scan_rows.append(EntryMinuteScan(
                    snapshot_pt=snap_pt,
                    net_position_value=None,
                    net_credit=None,
                    edge=None,
                ))
                continue

            net_credit = -pos_val
            edge = trade_input.structural_prob - (net_credit / trade_input.spread_width)
            scan_rows.append(EntryMinuteScan(
                snapshot_pt=snap_pt,
                net_position_value=pos_val,
                net_credit=net_credit,
                edge=edge,
            ))

            if baseline_snap is None and self.baseline == "first_quoted_minute":
                baseline_snap = (snap_pt, net_credit)

            if fill_snap is None and self.fill_rule == "first_above_threshold":
                if edge >= self.edge_threshold:
                    fill_snap = (snap_pt, net_credit, edge)

        return scan_rows, baseline_snap, fill_snap

    def _touch_exit(
        self,
        filled: bool,
        fill_snap: Optional[tuple[datetime, float, float]],
        touch_datetime: Optional[datetime],
        touch_window_scan: list[tuple[datetime, QuoteMap]],
        trade_input: TradeInput,
    ) -> tuple[Optional[float], Optional[datetime]]:
        if not filled or touch_datetime is None:
            return None, None

        fill_credit = fill_snap[1]  # type: ignore[index]
        for snap_pt, qmap in touch_window_scan:
            if snap_pt >= touch_datetime:
                pos_val = self.plugin.net_price(trade_input.legs, qmap)
                if pos_val is not None:
                    # P&L = entry_credit + position_value_at_close
                    # pos_val is negative for credit spread; adding to entry_credit
                    # gives the round-trip P&L correctly.
                    return fill_credit + pos_val, snap_pt
        return None, None

    def _close_outcome(
        self,
        filled: bool,
        fill_snap: Optional[tuple[datetime, float, float]],
        settlement_price: Optional[float],
        trade_input: TradeInput,
    ) -> tuple[Optional[float], Optional[str]]:
        if not filled or settlement_price is None:
            return None, None

        fill_credit = fill_snap[1]  # type: ignore[index]
        payoff_val = self.plugin.payoff(trade_input.legs, settlement_price)
        close_pnl = fill_credit + payoff_val
        zone = _close_zone(trade_input.legs, settlement_price)
        return close_pnl, zone

    def _baseline_outcomes(
        self,
        baseline_snap: Optional[tuple[datetime, float]],
        touch_datetime: Optional[datetime],
        touch_window_scan: list[tuple[datetime, QuoteMap]],
        settlement_price: Optional[float],
        trade_input: TradeInput,
    ) -> tuple[Optional[float], Optional[float]]:
        if baseline_snap is None:
            return None, None

        base_credit = baseline_snap[1]
        baseline_touch_exit_pnl: Optional[float] = None
        baseline_close_pnl: Optional[float] = None

        if touch_datetime is not None:
            for snap_pt, qmap in touch_window_scan:
                if snap_pt >= touch_datetime:
                    pos_val = self.plugin.net_price(trade_input.legs, qmap)
                    if pos_val is not None:
                        baseline_touch_exit_pnl = base_credit + pos_val
                        break

        if settlement_price is not None:
            payoff_val = self.plugin.payoff(trade_input.legs, settlement_price)
            baseline_close_pnl = base_credit + payoff_val

        return baseline_touch_exit_pnl, baseline_close_pnl


def _close_zone(legs: list[Leg], settlement_price: float) -> str:
    """Three-zone settlement classification for a credit vertical spread.

    Zones are defined by settlement_price vs the lower and upper strikes:
      max_profit  — S ≤ lower strike (both legs OTM for credit call spread)
      partial_loss — lower < S < upper
      max_loss    — S ≥ upper strike (both legs ITM for credit call spread)

    Note: zone names reflect credit-spread semantics. For a debit spread,
    'max_profit' and 'max_loss' are structurally reversed but the geometry
    (below/between/above) is correct in both cases.
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
