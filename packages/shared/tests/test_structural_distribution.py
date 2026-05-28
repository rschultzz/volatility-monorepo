"""Unit tests for packages/shared/structural_distribution.py.

Normalization context:
  compute_terminal_prob_in_range and compute_session_excursion_metrics now
  project each analogue's close/high/low onto today's price scale before
  checking range membership.  The existing test cases use the trivial
  projection (anchor_spot == today_spot, anchor_im == today_im) so that
  projected_close = close — all existing expectations remain correct.

  New test class TestProjectionAcrossPriceEpochs verifies non-trivial
  projection with analogues from a different price epoch.
"""
from __future__ import annotations

import pytest

from packages.shared.structural_distribution import (
    compute_session_excursion_metrics,
    compute_terminal_prob_in_range,
    get_trade_thesis_range,
)


# ── helpers ──────────────────────────────────────────────────────────────────

# Trivial-projection defaults: anchor == today, so projected_close = close.
_TRIVIAL_SPOT = 100.0
_TRIVIAL_IM   = 25.0


def build_analogue(
    close,
    anchor_spot: float = _TRIVIAL_SPOT,
    anchor_im: float = _TRIVIAL_IM,
) -> dict:
    """Build a compute_terminal_prob_in_range analogue dict.

    Defaults produce the trivial projection (projected_close = close).
    """
    return {
        "close": close,
        "anchor_spot": anchor_spot,
        "anchor_implied_move": anchor_im,
    }


def build_excursion_analogue(
    lo,
    hi,
    anchor_spot: float = _TRIVIAL_SPOT,
    anchor_im: float = _TRIVIAL_IM,
) -> dict:
    """Build a compute_session_excursion_metrics analogue dict.

    Defaults produce the trivial projection (projected_high/low = original).
    """
    return {
        "session_low": lo,
        "session_high": hi,
        "anchor_spot": anchor_spot,
        "anchor_implied_move": anchor_im,
    }


# ── compute_terminal_prob_in_range ──────────────────────────────────────────

class TestTerminalProbInRange:
    """Trivial-projection tests: anchor_spot=today_spot, anchor_im=today_im."""

    S = _TRIVIAL_SPOT
    IM = _TRIVIAL_IM

    def test_all_in_range(self):
        analogues = [build_analogue(c) for c in [100.0, 105.0, 110.0]]
        r = compute_terminal_prob_in_range(analogues, self.S, self.IM, 95.0, 115.0)
        assert r is not None
        assert r["prob"] == pytest.approx(1.0)
        assert r["n"] == 3
        ci_lo, ci_hi = r["wilson_ci"]
        # Wilson CI at n=3 with all successes: lower ≈ 0.44, upper = 1.0
        assert 0.35 < ci_lo <= 1.0
        assert ci_hi == pytest.approx(1.0)

    def test_half_in_range(self):
        analogues = [build_analogue(c) for c in [100.0, 200.0, 105.0, 300.0]]
        r = compute_terminal_prob_in_range(analogues, self.S, self.IM, 90.0, 150.0)
        assert r is not None
        assert r["prob"] == pytest.approx(0.5)
        assert r["n"] == 4
        ci_lo, ci_hi = r["wilson_ci"]
        assert ci_lo < 0.5
        assert ci_hi > 0.5

    def test_none_close_excluded_from_denominator(self):
        analogues = [build_analogue(c) for c in [100.0, None, 105.0, None]]
        r = compute_terminal_prob_in_range(analogues, self.S, self.IM, 90.0, 115.0)
        assert r is not None
        assert r["n"] == 2  # Nones excluded
        assert r["prob"] == pytest.approx(1.0)

    def test_all_none_close_returns_none(self):
        analogues = [build_analogue(None), build_analogue(None), build_analogue(None)]
        assert compute_terminal_prob_in_range(analogues, self.S, self.IM, 90.0, 115.0) is None

    def test_empty_returns_none(self):
        assert compute_terminal_prob_in_range([], self.S, self.IM, 90.0, 115.0) is None

    def test_both_bounds_none_returns_none(self):
        analogues = [build_analogue(100.0), build_analogue(105.0)]
        assert compute_terminal_prob_in_range(analogues, self.S, self.IM, None, None) is None

    def test_one_sided_upper(self):
        """lower=None → P(projected_close <= upper)"""
        analogues = [build_analogue(c) for c in [80.0, 90.0, 110.0, 120.0]]
        r = compute_terminal_prob_in_range(analogues, self.S, self.IM, None, 100.0)
        assert r is not None
        assert r["prob"] == pytest.approx(0.5)   # 80 and 90 qualify

    def test_one_sided_lower(self):
        """upper=None → P(projected_close >= lower)"""
        analogues = [build_analogue(c) for c in [80.0, 90.0, 110.0, 120.0]]
        r = compute_terminal_prob_in_range(analogues, self.S, self.IM, 100.0, None)
        assert r is not None
        assert r["prob"] == pytest.approx(0.5)   # 110 and 120 qualify

    def test_closed_interval_includes_boundary(self):
        """Boundary value is in range (closed interval)."""
        analogues = [build_analogue(100.0), build_analogue(200.0)]
        r = compute_terminal_prob_in_range(analogues, self.S, self.IM, 100.0, 150.0)
        assert r is not None
        assert r["prob"] == pytest.approx(0.5)   # 100.0 exactly at lower boundary → in

    def test_zero_in_range(self):
        analogues = [build_analogue(c) for c in [200.0, 300.0, 400.0]]
        r = compute_terminal_prob_in_range(analogues, self.S, self.IM, 90.0, 150.0)
        assert r is not None
        assert r["prob"] == pytest.approx(0.0)

    def test_wilson_ci_nonnull_on_zero(self):
        """Wilson CI should not return None even when prob=0 (n>0)."""
        analogues = [build_analogue(200.0), build_analogue(300.0)]
        r = compute_terminal_prob_in_range(analogues, self.S, self.IM, 90.0, 150.0)
        assert r is not None
        ci_lo, ci_hi = r["wilson_ci"]
        assert ci_lo is not None
        assert ci_hi is not None


# ── compute_session_excursion_metrics ────────────────────────────────────────

class TestSessionExcursionMetrics:
    """Trivial-projection tests: anchor_spot=today_spot, anchor_im=today_im."""

    S = _TRIVIAL_SPOT
    IM = _TRIVIAL_IM

    def test_all_intersect_and_stay_in(self):
        analogues = [
            build_excursion_analogue(102.0, 108.0),
            build_excursion_analogue(101.0, 107.0),
        ]
        r = compute_session_excursion_metrics(analogues, self.S, self.IM, 100.0, 110.0)
        assert r is not None
        assert r["intersected_prob"] == pytest.approx(1.0)
        assert r["stayed_in_prob"] == pytest.approx(1.0)
        assert r["n"] == 2

    def test_all_intersect_some_reverted(self):
        """Sessions touch range but go outside → intersected=1.0, stayed_in < 1.0."""
        analogues = [
            build_excursion_analogue(90.0, 115.0),   # touches range but goes out
            build_excursion_analogue(102.0, 108.0),  # stays in
        ]
        r = compute_session_excursion_metrics(analogues, self.S, self.IM, 100.0, 110.0)
        assert r is not None
        assert r["intersected_prob"] == pytest.approx(1.0)
        assert r["stayed_in_prob"] == pytest.approx(0.5)

    def test_none_below_range(self):
        """Sessions staying entirely below range → both = 0."""
        analogues = [
            build_excursion_analogue(80.0, 90.0),
            build_excursion_analogue(70.0, 85.0),
        ]
        r = compute_session_excursion_metrics(analogues, self.S, self.IM, 100.0, 110.0)
        assert r is not None
        assert r["intersected_prob"] == pytest.approx(0.0)
        assert r["stayed_in_prob"] == pytest.approx(0.0)

    def test_null_ohlc_excluded_from_denominator(self):
        """Dicts with None session_high/low are excluded."""
        analogues = [
            {"session_low": None, "session_high": None,
             "anchor_spot": self.S, "anchor_implied_move": self.IM},
            build_excursion_analogue(102.0, 108.0),
        ]
        r = compute_session_excursion_metrics(analogues, self.S, self.IM, 100.0, 110.0)
        assert r is not None
        assert r["n"] == 1
        assert r["intersected_prob"] == pytest.approx(1.0)

    def test_all_null_returns_none(self):
        analogues = [
            {"session_low": None, "session_high": None,
             "anchor_spot": self.S, "anchor_implied_move": self.IM},
        ]
        assert compute_session_excursion_metrics(analogues, self.S, self.IM, 100.0, 110.0) is None

    def test_both_bounds_none_returns_none(self):
        analogues = [build_excursion_analogue(100.0, 110.0)]
        assert compute_session_excursion_metrics(analogues, self.S, self.IM, None, None) is None

    def test_one_sided_upper_only(self):
        """lower=None → open lower end; intersected always True when lower=None."""
        analogues = [
            build_excursion_analogue(80.0, 95.0),   # proj_high=95 ≤ 100; stayed
            build_excursion_analogue(80.0, 120.0),  # proj_high=120 > 100; NOT stayed
        ]
        r = compute_session_excursion_metrics(analogues, self.S, self.IM, None, 100.0)
        assert r is not None
        assert r["intersected_prob"] == pytest.approx(1.0)
        assert r["stayed_in_prob"] == pytest.approx(0.5)   # only first session stayed

    def test_one_sided_lower_only(self):
        """upper=None → intersected = proj_high >= lower."""
        analogues = [
            build_excursion_analogue(105.0, 120.0),  # proj_low=105 ≥ 100; intersected+stayed
            build_excursion_analogue(80.0, 90.0),    # proj_high=90 < 100; neither
        ]
        r = compute_session_excursion_metrics(analogues, self.S, self.IM, 100.0, None)
        assert r is not None
        assert r["intersected_prob"] == pytest.approx(0.5)
        assert r["stayed_in_prob"] == pytest.approx(0.5)


# ── TestProjectionAcrossPriceEpochs ─────────────────────────────────────────

class TestProjectionAcrossPriceEpochs:
    """Non-trivial projection: analogues from a different price epoch."""

    def test_analogue_from_different_epoch_projects_correctly(self):
        """Analogues from anchor_spot=5000, anchor_im=30.
        Closes: 5000 (no move), 5030 (+1 IM), 5060 (+2 IM).
        Today: spot=4200, im=20. Range: [4220, 4250].

        Projected closes:
          (5000-5000)/30 * 20 + 4200 = 4200  → NOT in [4220, 4250]
          (5030-5000)/30 * 20 + 4200 = 4220  → IN  [4220, 4250] (boundary)
          (5060-5000)/30 * 20 + 4200 = 4240  → IN  [4220, 4250]
        prob = 2/3 ≈ 0.667
        """
        analogues = [
            build_analogue(5000.0, anchor_spot=5000.0, anchor_im=30.0),
            build_analogue(5030.0, anchor_spot=5000.0, anchor_im=30.0),
            build_analogue(5060.0, anchor_spot=5000.0, anchor_im=30.0),
        ]
        r = compute_terminal_prob_in_range(analogues, 4200.0, 20.0, 4220.0, 4250.0)
        assert r is not None
        assert r["n"] == 3
        assert r["prob"] == pytest.approx(2 / 3, abs=1e-9)

    def test_analogue_with_zero_im_excluded(self):
        """Analogue with anchor_implied_move=0 is excluded from both n and k."""
        analogues = [
            build_analogue(5000.0, anchor_spot=5000.0, anchor_im=30.0),
            build_analogue(5030.0, anchor_spot=5000.0, anchor_im=30.0),
            build_analogue(5060.0, anchor_spot=5000.0, anchor_im=30.0),
            build_analogue(5040.0, anchor_spot=5000.0, anchor_im=0.0),   # zero IM → excluded
            build_analogue(5050.0, anchor_spot=5000.0, anchor_im=30.0),
        ]
        r = compute_terminal_prob_in_range(analogues, 4200.0, 20.0, 4220.0, 4250.0)
        assert r is not None
        assert r["n"] == 4  # 5 - 1 (zero IM excluded)

    def test_analogue_with_none_anchor_spot_excluded(self):
        """Analogue with anchor_spot=None is excluded from both n and k."""
        analogues = [
            build_analogue(5000.0, anchor_spot=5000.0, anchor_im=30.0),
            build_analogue(5030.0, anchor_spot=5000.0, anchor_im=30.0),
            {"close": 5040.0, "anchor_spot": None, "anchor_implied_move": 30.0},  # None anchor
            build_analogue(5060.0, anchor_spot=5000.0, anchor_im=30.0),
        ]
        r = compute_terminal_prob_in_range(analogues, 4200.0, 20.0, 4220.0, 4250.0)
        assert r is not None
        assert r["n"] == 3  # 4 - 1 (None anchor_spot excluded)

    def test_analogue_with_none_anchor_im_excluded(self):
        """Analogue with anchor_implied_move=None is excluded."""
        analogues = [
            build_analogue(5000.0, anchor_spot=5000.0, anchor_im=30.0),
            {"close": 5030.0, "anchor_spot": 5000.0, "anchor_implied_move": None},  # None IM
            build_analogue(5060.0, anchor_spot=5000.0, anchor_im=30.0),
        ]
        r = compute_terminal_prob_in_range(analogues, 4200.0, 20.0, 4220.0, 4250.0)
        assert r is not None
        assert r["n"] == 2  # None IM excluded

    def test_negative_projection_handled(self):
        """Analogues that moved DOWN produce projected close below today_spot.

        anchor_spot=5000, anchor_im=20, today_spot=4200, today_im=25.
          close=4990 → norm=-0.5 → proj=4200 + (-0.5)*25 = 4187.5 → NOT in [4150,4180]
          close=4980 → norm=-1.0 → proj=4200 + (-1.0)*25 = 4175.0 → IN  [4150,4180]
          close=4960 → norm=-2.0 → proj=4200 + (-2.0)*25 = 4150.0 → IN  [4150,4180] (boundary)
        prob = 2/3
        """
        analogues = [
            build_analogue(4990.0, anchor_spot=5000.0, anchor_im=20.0),
            build_analogue(4980.0, anchor_spot=5000.0, anchor_im=20.0),
            build_analogue(4960.0, anchor_spot=5000.0, anchor_im=20.0),
        ]
        r = compute_terminal_prob_in_range(analogues, 4200.0, 25.0, 4150.0, 4180.0)
        assert r is not None
        assert r["n"] == 3
        assert r["prob"] == pytest.approx(2 / 3, abs=1e-9)

    def test_projection_excursion_different_epoch(self):
        """Excursion metrics also project to today's scale.

        anchor_spot=5000, anchor_im=40, today_spot=4200, today_im=20.
        Session 1: low=4980, high=5040 → proj_low=(4980-5000)/40*20+4200=4190,
                                          proj_high=(5040-5000)/40*20+4200=4220
          → intersects [4200, 4220] (proj_low=4190 ≤ 4220 AND proj_high=4220 ≥ 4200) ✓
          → stayed_in: proj_low=4190 < 4200 → NOT stayed

        Session 2: low=5010, high=5020 → proj_low=4205, proj_high=4210
          → intersects [4200, 4220] ✓
          → stayed_in: 4205 ≥ 4200 AND 4210 ≤ 4220 → ✓
        """
        analogues = [
            build_excursion_analogue(4980.0, 5040.0, anchor_spot=5000.0, anchor_im=40.0),
            build_excursion_analogue(5010.0, 5020.0, anchor_spot=5000.0, anchor_im=40.0),
        ]
        r = compute_session_excursion_metrics(analogues, 4200.0, 20.0, 4200.0, 4220.0)
        assert r is not None
        assert r["n"] == 2
        assert r["intersected_prob"] == pytest.approx(1.0)
        assert r["stayed_in_prob"] == pytest.approx(0.5)


# ── get_trade_thesis_range ───────────────────────────────────────────────────

class TestGetTradeThesisRange:
    def test_magnet_above(self):
        # Step 3.6: one-sided upper-tail → P(close ≥ magnet)
        rb = {"regime": "magnet-above", "drift_target": 4200.0}
        r = get_trade_thesis_range(rb, 4100.0)
        assert r["lower"] == pytest.approx(4200.0)
        assert r["upper"] is None
        assert r["regime_kind"] == "magnet-above"

    def test_magnet_below(self):
        # Step 3.6: one-sided lower-tail → P(close ≤ magnet)
        rb = {"regime": "magnet-below", "drift_target": 4000.0}
        r = get_trade_thesis_range(rb, 4100.0)
        assert r["lower"] is None
        assert r["upper"] == pytest.approx(4000.0)
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
