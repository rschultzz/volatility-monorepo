"""Unit tests for packages/shared/structural_distribution.py."""
from __future__ import annotations

import pytest

from packages.shared.structural_distribution import (
    compute_session_excursion_metrics,
    compute_terminal_prob_in_range,
    get_trade_thesis_range,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def _ohlc(lo, hi, op=None, cl=None):
    return {"session_low": lo, "session_high": hi, "session_open": op, "session_close": cl}


# ── compute_terminal_prob_in_range ──────────────────────────────────────────

class TestTerminalProbInRange:
    def test_all_in_range(self):
        closes = [100.0, 105.0, 110.0]
        r = compute_terminal_prob_in_range(closes, 95.0, 115.0)
        assert r is not None
        assert r["prob"] == pytest.approx(1.0)
        assert r["n"] == 3
        ci_lo, ci_hi = r["wilson_ci"]
        # Wilson CI at n=3 with all successes: lower ≈ 0.44, upper = 1.0
        assert 0.35 < ci_lo <= 1.0
        assert ci_hi == pytest.approx(1.0)

    def test_half_in_range(self):
        closes = [100.0, 200.0, 105.0, 300.0]
        r = compute_terminal_prob_in_range(closes, 90.0, 150.0)
        assert r is not None
        assert r["prob"] == pytest.approx(0.5)
        assert r["n"] == 4
        # Wilson CI for 2/4 should be wide
        ci_lo, ci_hi = r["wilson_ci"]
        assert ci_lo < 0.5
        assert ci_hi > 0.5

    def test_none_values_excluded_from_denominator(self):
        closes = [100.0, None, 105.0, None]
        r = compute_terminal_prob_in_range(closes, 90.0, 115.0)
        assert r is not None
        assert r["n"] == 2  # Nones excluded
        assert r["prob"] == pytest.approx(1.0)

    def test_all_none_returns_none(self):
        assert compute_terminal_prob_in_range([None, None, None], 90.0, 115.0) is None

    def test_empty_returns_none(self):
        assert compute_terminal_prob_in_range([], 90.0, 115.0) is None

    def test_both_bounds_none_returns_none(self):
        assert compute_terminal_prob_in_range([100.0, 105.0], None, None) is None

    def test_one_sided_upper(self):
        """lower=None → P(close <= upper)"""
        closes = [80.0, 90.0, 110.0, 120.0]
        r = compute_terminal_prob_in_range(closes, None, 100.0)
        assert r is not None
        assert r["prob"] == pytest.approx(0.5)   # 80 and 90 qualify

    def test_one_sided_lower(self):
        """upper=None → P(close >= lower)"""
        closes = [80.0, 90.0, 110.0, 120.0]
        r = compute_terminal_prob_in_range(closes, 100.0, None)
        assert r is not None
        assert r["prob"] == pytest.approx(0.5)   # 110 and 120 qualify

    def test_closed_interval_includes_boundary(self):
        """Boundary value is in range (closed interval)."""
        closes = [100.0, 200.0]
        r = compute_terminal_prob_in_range(closes, 100.0, 150.0)
        assert r is not None
        assert r["prob"] == pytest.approx(0.5)   # 100.0 exactly at lower boundary → in

    def test_zero_in_range(self):
        closes = [200.0, 300.0, 400.0]
        r = compute_terminal_prob_in_range(closes, 90.0, 150.0)
        assert r is not None
        assert r["prob"] == pytest.approx(0.0)

    def test_wilson_ci_nonnull_on_zero(self):
        """Wilson CI should not return None even when prob=0 (n>0)."""
        r = compute_terminal_prob_in_range([200.0, 300.0], 90.0, 150.0)
        assert r is not None
        ci_lo, ci_hi = r["wilson_ci"]
        assert ci_lo is not None
        assert ci_hi is not None


# ── compute_session_excursion_metrics ────────────────────────────────────────

class TestSessionExcursionMetrics:
    def test_all_intersect_and_stay_in(self):
        sessions = [_ohlc(102.0, 108.0), _ohlc(101.0, 107.0)]
        r = compute_session_excursion_metrics(sessions, 100.0, 110.0)
        assert r is not None
        assert r["intersected_prob"] == pytest.approx(1.0)
        assert r["stayed_in_prob"] == pytest.approx(1.0)
        assert r["n"] == 2

    def test_all_intersect_some_reverted(self):
        """Sessions touch range but go outside → intersected=1.0, stayed_in < 1.0."""
        sessions = [
            _ohlc(90.0, 115.0),   # touches range but goes out on both sides
            _ohlc(102.0, 108.0),  # stays in
        ]
        r = compute_session_excursion_metrics(sessions, 100.0, 110.0)
        assert r is not None
        assert r["intersected_prob"] == pytest.approx(1.0)
        assert r["stayed_in_prob"] == pytest.approx(0.5)

    def test_none_below_range(self):
        """Sessions staying entirely below range → both = 0."""
        sessions = [_ohlc(80.0, 90.0), _ohlc(70.0, 85.0)]
        r = compute_session_excursion_metrics(sessions, 100.0, 110.0)
        assert r is not None
        assert r["intersected_prob"] == pytest.approx(0.0)
        assert r["stayed_in_prob"] == pytest.approx(0.0)

    def test_null_ohlc_excluded_from_denominator(self):
        """Dicts with None session_high/low are excluded."""
        sessions = [
            {"session_low": None, "session_high": None},
            _ohlc(102.0, 108.0),
        ]
        r = compute_session_excursion_metrics(sessions, 100.0, 110.0)
        assert r is not None
        assert r["n"] == 1
        assert r["intersected_prob"] == pytest.approx(1.0)

    def test_all_null_returns_none(self):
        sessions = [{"session_low": None, "session_high": None}]
        assert compute_session_excursion_metrics(sessions, 100.0, 110.0) is None

    def test_both_bounds_none_returns_none(self):
        sessions = [_ohlc(100.0, 110.0)]
        assert compute_session_excursion_metrics(sessions, None, None) is None

    def test_one_sided_upper_only(self):
        """lower=None → intersected = session_low <= upper"""
        sessions = [
            _ohlc(80.0, 95.0),   # high < 100 (upper); intersected (low <= 100), stayed (high <= 100)
            _ohlc(80.0, 120.0),  # high > 100; intersected (low <= 100 from -inf), NOT stayed
        ]
        r = compute_session_excursion_metrics(sessions, None, 100.0)
        assert r is not None
        assert r["intersected_prob"] == pytest.approx(1.0)
        assert r["stayed_in_prob"] == pytest.approx(0.5)   # only first session stayed

    def test_one_sided_lower_only(self):
        """upper=None → intersected = session_high >= lower"""
        sessions = [
            _ohlc(105.0, 120.0),  # low >= 100; intersected and stayed
            _ohlc(80.0, 90.0),    # high < 100; neither intersected nor stayed
        ]
        r = compute_session_excursion_metrics(sessions, 100.0, None)
        assert r is not None
        assert r["intersected_prob"] == pytest.approx(0.5)
        assert r["stayed_in_prob"] == pytest.approx(0.5)


# ── get_trade_thesis_range ───────────────────────────────────────────────────

class TestGetTradeThesisRange:
    def test_magnet_above(self):
        rb = {"regime": "magnet-above", "drift_target": 4200.0}
        r = get_trade_thesis_range(rb, 4100.0)
        assert r["lower"] == pytest.approx(4100.0)
        assert r["upper"] == pytest.approx(4200.0)
        assert r["regime_kind"] == "magnet-above"

    def test_magnet_below(self):
        rb = {"regime": "magnet-below", "drift_target": 4000.0}
        r = get_trade_thesis_range(rb, 4100.0)
        assert r["lower"] == pytest.approx(4000.0)
        assert r["upper"] == pytest.approx(4100.0)
        assert r["regime_kind"] == "magnet-below"

    def test_magnetic_pin_from_tolerance_kwarg(self):
        rb = {"regime": "magnetic-pin", "drift_target": 4150.0}
        r = get_trade_thesis_range(rb, 4100.0, tolerance=50.0)
        assert r["lower"] == pytest.approx(4100.0)
        assert r["upper"] == pytest.approx(4200.0)
        assert r["regime_kind"] == "magnetic-pin"

    def test_magnetic_pin_from_regime_block(self):
        rb = {"regime": "magnetic-pin", "drift_target": 4150.0, "tolerance": 50.0}
        r = get_trade_thesis_range(rb, 4100.0)
        assert r["lower"] == pytest.approx(4100.0)
        assert r["upper"] == pytest.approx(4200.0)

    def test_bounded(self):
        rb = {
            "regime": "bounded",
            "containment_zone": {"lower_price": 4000.0, "upper_price": 4200.0},
        }
        r = get_trade_thesis_range(rb, 4100.0)
        assert r["lower"] == pytest.approx(4000.0)
        assert r["upper"] == pytest.approx(4200.0)
        assert r["regime_kind"] == "bounded"

    def test_amplification_returns_none_range(self):
        rb = {"regime": "amplification"}
        r = get_trade_thesis_range(rb, 4100.0)
        assert r["lower"] is None
        assert r["upper"] is None

    def test_untethered_returns_none_range(self):
        rb = {"regime": "untethered"}
        r = get_trade_thesis_range(rb, 4100.0)
        assert r["lower"] is None
        assert r["upper"] is None

    def test_broken_magnet_returns_none_range(self):
        rb = {"regime": "broken-magnet"}
        r = get_trade_thesis_range(rb, 4100.0)
        assert r["lower"] is None
        assert r["upper"] is None

    def test_magnet_above_missing_drift_target_raises(self):
        rb = {"regime": "magnet-above"}
        with pytest.raises(ValueError, match="drift_target"):
            get_trade_thesis_range(rb, 4100.0)

    def test_magnet_below_missing_drift_target_raises(self):
        rb = {"regime": "magnet-below"}
        with pytest.raises(ValueError, match="drift_target"):
            get_trade_thesis_range(rb, 4100.0)

    def test_magnetic_pin_missing_drift_target_raises(self):
        rb = {"regime": "magnetic-pin"}
        with pytest.raises(ValueError, match="drift_target"):
            get_trade_thesis_range(rb, 4100.0, tolerance=50.0)

    def test_magnetic_pin_missing_tolerance_raises(self):
        rb = {"regime": "magnetic-pin", "drift_target": 4150.0}
        with pytest.raises(ValueError, match="tolerance"):
            get_trade_thesis_range(rb, 4100.0)  # no tolerance kwarg, not in regime_block

    def test_bounded_missing_containment_zone_raises(self):
        rb = {"regime": "bounded"}
        with pytest.raises(ValueError, match="containment_zone"):
            get_trade_thesis_range(rb, 4100.0)

    def test_unknown_regime_returns_none_range(self):
        rb = {"regime": "future-unknown-regime"}
        r = get_trade_thesis_range(rb, 4100.0)
        assert r["lower"] is None
        assert r["upper"] is None
        assert r["regime_kind"] == "future-unknown-regime"
