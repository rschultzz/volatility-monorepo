"""BacktestHarness — structure-agnostic runner for per-date trade evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

from packages.shared.backtest.models import (
    EntryMinuteScan,
    QuoteMap,
    TradeInput,
    TradeResult,
)
from packages.shared.backtest.plugins.protocol import BacktestPlugin
from packages.shared.backtest.quote_validity import spread_value_is_valid


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
        scan_rows, baseline_snap, fill_snap, obs = self._run_entry_crawl(trade_input, entry_day_scan)

        filled = fill_snap is not None
        # CR-AN decision 6: a window with no valid minute cannot be traded or
        # baselined; the trade is excluded with a reason (never silently).
        excluded_reason = "no_valid_entry_minute" if obs["n_minutes_valid"] == 0 else None

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
            n_minutes_total=obs["n_minutes_total"],
            n_minutes_valid=obs["n_minutes_valid"],
            baseline_minute_offset=obs["baseline_minute_offset"],
            had_invalid_quote=obs["had_invalid_quote"],
            excluded_reason=excluded_reason,
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
        dict,                                    # CR-AN decision 7 observability
    ]:
        scan_rows: list[EntryMinuteScan] = []
        baseline_snap: Optional[tuple[datetime, float]] = None
        fill_snap: Optional[tuple[datetime, float, float]] = None
        n_valid = 0
        window_open = entry_day_scan[0][0] if entry_day_scan else None
        baseline_offset: Optional[int] = None

        for snap_pt, qmap in entry_day_scan:
            pos_val = self.plugin.net_price(trade_input.legs, qmap)
            # CR-AN decision 2: a missing leg or an out-of-range spread value
            # makes the minute invalid → skipped, still persisted (invariant #3).
            if not spread_value_is_valid(pos_val, trade_input.spread_width, trade_input.legs):
                scan_rows.append(EntryMinuteScan(
                    snapshot_pt=snap_pt,
                    net_position_value=None,
                    net_credit=None,
                    edge=None,
                    quote_valid=False,
                ))
                continue
            n_valid += 1

            net_credit = -pos_val
            # abs(): market-implied P(win) = |net_credit| / width for both structures.
            # For credit, net_credit > 0; for debit, net_credit < 0. abs() normalises.
            edge = trade_input.structural_prob - (abs(net_credit) / trade_input.spread_width)
            scan_rows.append(EntryMinuteScan(
                snapshot_pt=snap_pt,
                net_position_value=pos_val,
                net_credit=net_credit,
                edge=edge,
            ))

            if baseline_snap is None and self.baseline == "first_quoted_minute":
                # CR-AN decision 3: first VALID minute (invalid ones were skipped above)
                baseline_snap = (snap_pt, net_credit)
                baseline_offset = int(round((snap_pt - window_open).total_seconds() / 60.0))

            if fill_snap is None and self.fill_rule == "first_above_threshold":
                if edge >= self.edge_threshold:
                    fill_snap = (snap_pt, net_credit, edge)

        obs = {
            "n_minutes_total": len(entry_day_scan),
            "n_minutes_valid": n_valid,
            "baseline_minute_offset": baseline_offset,
            "had_invalid_quote": n_valid < len(entry_day_scan),
        }
        return scan_rows, baseline_snap, fill_snap, obs

    def _touch_exit(
        self,
        filled: bool,
        fill_snap: Optional[tuple[datetime, float, float]],
        touch_datetime: Optional[datetime],
        touch_window_scan: list[tuple[datetime, QuoteMap]],
        trade_input: TradeInput,
    ) -> tuple[Optional[float], Optional[datetime]]:
        # Decision #6: credit plugin records touch as breach diagnostic only.
        if not self.plugin.touch_exit_is_meaningful:
            return None, None
        if not filled or touch_datetime is None:
            return None, None

        fill_credit = fill_snap[1]  # type: ignore[index]
        for snap_pt, qmap in touch_window_scan:
            if snap_pt >= touch_datetime:
                pos_val = self.plugin.net_price(trade_input.legs, qmap)
                # CR-AN decision 4: exit only on a valid minute
                if spread_value_is_valid(pos_val, trade_input.spread_width, trade_input.legs):
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
        zone = self.plugin.close_zone(trade_input.legs, settlement_price)
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

        # Decision #6: credit has no meaningful touch-exit P&L.
        if self.plugin.touch_exit_is_meaningful and touch_datetime is not None:
            for snap_pt, qmap in touch_window_scan:
                if snap_pt >= touch_datetime:
                    pos_val = self.plugin.net_price(trade_input.legs, qmap)
                    if spread_value_is_valid(pos_val, trade_input.spread_width, trade_input.legs):
                        baseline_touch_exit_pnl = base_credit + pos_val
                        break

        if settlement_price is not None:
            payoff_val = self.plugin.payoff(trade_input.legs, settlement_price)
            baseline_close_pnl = base_credit + payoff_val

        return baseline_touch_exit_pnl, baseline_close_pnl


