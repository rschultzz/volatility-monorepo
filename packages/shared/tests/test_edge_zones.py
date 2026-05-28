"""Unit tests for packages/shared/edge_zones.py."""
from __future__ import annotations

import math
import warnings

import pytest

from packages.shared.edge_zones import (
    _auto_bin_size,
    _group_bins_into_zones,
    _price_axis_bins,
    classify_edge_ratio,
    compute_edge_zones,
)
from packages.shared.pricing.engine import black_scholes_price

# ── Shared helpers ────────────────────────────────────────────────────────────

def _bsm_chain(spot, strikes, sigma=0.20, T=5.0/252, r=0.05):
    return [{"strike": float(k), "call_price": black_scholes_price(spot, float(k), T, r, sigma, "c")}
            for k in strikes if k > 0]

def _analogues(close_vals, timeframe="t5", spot=4200.0, im=25.0):
    """Build analogue dicts for compute_edge_zones fixtures.

    Sets anchor_spot = spot and implied_move_1d = im so that the normalization
    projection is the identity (projected_close = close), preserving all
    existing test expectations.  For the integration-style tests, both values
    are set to today's reference so projection is trivially correct.
    """
    key = f"session_close_{timeframe}"
    return [
        {
            key:                 v,
            "session_open_t0":   spot,   # anchor == today → identity projection
            "implied_move_1d":   im,
            "reached_touch":     True,
        }
        for v in close_vals
    ]

def _regime_above(drift_target):
    return {"regime": "magnet-above", "drift_target": float(drift_target)}


# ── 1. classify_edge_ratio ───────────────────────────────────────────────────

class TestClassifyEdgeRatio:
    def test_strong_positive(self):
        assert classify_edge_ratio(2.01) == "strong-positive"
        assert classify_edge_ratio(100.0) == "strong-positive"

    def test_infinity_is_strong_positive(self):
        assert classify_edge_ratio(float("inf")) == "strong-positive"

    def test_exact_2_0_is_moderate_not_strong(self):
        # 2.0 ≤ 2.0 → NOT > 2.0 → moderate-positive
        assert classify_edge_ratio(2.0) == "moderate-positive"

    def test_moderate_positive(self):
        # (1.3, 2.0]
        assert classify_edge_ratio(1.31) == "moderate-positive"
        assert classify_edge_ratio(1.5)  == "moderate-positive"
        assert classify_edge_ratio(1.99) == "moderate-positive"

    def test_exact_1_3_is_neutral_not_moderate_positive(self):
        # 1.3 is included in neutral [0.7, 1.3]
        assert classify_edge_ratio(1.3) == "neutral"

    def test_neutral(self):
        # [0.7, 1.3]
        assert classify_edge_ratio(1.0)  == "neutral"
        assert classify_edge_ratio(0.7)  == "neutral"
        assert classify_edge_ratio(0.85) == "neutral"
        assert classify_edge_ratio(1.3)  == "neutral"

    def test_exact_0_7_is_neutral(self):
        assert classify_edge_ratio(0.7) == "neutral"

    def test_moderate_negative(self):
        # [0.5, 0.7)
        assert classify_edge_ratio(0.5)  == "moderate-negative"
        assert classify_edge_ratio(0.6)  == "moderate-negative"
        assert classify_edge_ratio(0.69) == "moderate-negative"

    def test_exact_0_5_is_moderate_not_strong(self):
        # 0.5 ≥ 0.5 → not strong-negative
        assert classify_edge_ratio(0.5) == "moderate-negative"

    def test_strong_negative(self):
        # < 0.5
        assert classify_edge_ratio(0.49) == "strong-negative"
        assert classify_edge_ratio(0.0)  == "strong-negative"

    def test_none_is_unknown(self):
        assert classify_edge_ratio(None) == "unknown"

    def test_nan_is_unknown(self):
        assert classify_edge_ratio(float("nan")) == "unknown"

    def test_negative_is_unknown_with_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = classify_edge_ratio(-0.5)
        assert result == "unknown"
        runtime_warns = [x for x in w if issubclass(x.category, RuntimeWarning)]
        assert len(runtime_warns) >= 1


# ── 2. _auto_bin_size ────────────────────────────────────────────────────────

class TestAutoBinSize:
    def test_chain_5pt_spacing_implied_move_80(self):
        # min(5, 0.25*80=20) = 5
        chain = _bsm_chain(4180, range(3980, 4381, 5))
        bs = _auto_bin_size(chain, 80.0)
        assert bs == pytest.approx(5.0)

    def test_chain_25pt_spacing_implied_move_20(self):
        # min(25, 0.25*20=5) = 5
        chain = _bsm_chain(4180, range(3980, 4381, 25))
        bs = _auto_bin_size(chain, 20.0)
        assert bs == pytest.approx(5.0)

    def test_chain_1pt_spacing_implied_move_40(self):
        # min(1, 0.25*40=10) = 1
        chain = _bsm_chain(4180, range(4100, 4261, 1))
        bs = _auto_bin_size(chain, 40.0)
        assert bs == pytest.approx(1.0)

    def test_floor_at_1_when_chain_very_fine(self):
        # min(0.5, ...) should floor at 1.0
        chain = [{"strike": 100.0 + i * 0.5, "call_price": 1.0} for i in range(10)]
        bs = _auto_bin_size(chain, 40.0)
        assert bs >= 1.0

    def test_no_chain_uses_implied_move_only(self):
        # Empty chain → strike_spacing = inf → min(inf, 0.25*40=10) = 10
        bs = _auto_bin_size([], 40.0)
        assert bs == pytest.approx(10.0)


# ── 3. _price_axis_bins ───────────────────────────────────────────────────────

class TestPriceAxisBins:
    def test_covers_expected_range(self):
        # spot=4180, implied_move=20, sigma=1.5, bin_size=5
        # raw: [4180-30, 4180+30] = [4150, 4210]
        # snapped: [4150, 4210] exactly
        bins = _price_axis_bins(4180.0, 20.0, 1.5, 5.0)
        assert bins[0][0] == pytest.approx(4150.0)
        assert bins[-1][1] == pytest.approx(4210.0)
        assert len(bins) == 12  # 60 pts / 5 per bin

    def test_no_overlapping_bins(self):
        bins = _price_axis_bins(4180.0, 20.0, 1.5, 5.0)
        for i in range(len(bins) - 1):
            assert bins[i][1] == pytest.approx(bins[i + 1][0])

    def test_bins_are_equal_width(self):
        bins = _price_axis_bins(4180.0, 20.0, 1.5, 5.0)
        for lo, hi in bins:
            assert (hi - lo) == pytest.approx(5.0)

    def test_monotonic_ascending(self):
        bins = _price_axis_bins(4180.0, 20.0, 1.5, 5.0)
        for i in range(len(bins) - 1):
            assert bins[i][0] < bins[i + 1][0]


# ── 4. _group_bins_into_zones ────────────────────────────────────────────────

class TestGroupBinsIntoZones:
    def _make_bin(self, lo, classification, ratio=1.0, n=20):
        return {
            "lower": float(lo), "upper": float(lo + 5),
            "structural_prob": 0.5, "structural_n": n, "structural_ci": (0.3, 0.7),
            "implied_prob": 0.5, "edge_ratio": ratio,
            "classification": classification,
        }

    def test_merges_adjacent_same_class(self):
        bins = [
            self._make_bin(4150, "strong-positive"),
            self._make_bin(4155, "strong-positive"),
            self._make_bin(4160, "strong-positive"),
        ]
        zones = _group_bins_into_zones(bins)
        assert len(zones) == 1
        assert zones[0]["classification"] == "strong-positive"
        assert zones[0]["lower"] == pytest.approx(4150.0)
        assert zones[0]["upper"] == pytest.approx(4165.0)
        assert zones[0]["n_bins"] == 3

    def test_does_not_merge_across_classification_boundary(self):
        bins = [
            self._make_bin(4150, "strong-positive"),
            self._make_bin(4155, "neutral"),
            self._make_bin(4160, "strong-negative"),
        ]
        zones = _group_bins_into_zones(bins)
        assert len(zones) == 3
        assert [z["classification"] for z in zones] == [
            "strong-positive", "neutral", "strong-negative"
        ]

    def test_mixed_sequence(self):
        bins = (
            [self._make_bin(4150 + i * 5, "strong-positive") for i in range(3)]
            + [self._make_bin(4165 + i * 5, "neutral") for i in range(2)]
            + [self._make_bin(4175 + i * 5, "strong-negative") for i in range(3)]
        )
        zones = _group_bins_into_zones(bins)
        assert len(zones) == 3
        assert zones[0]["n_bins"] == 3
        assert zones[1]["n_bins"] == 2
        assert zones[2]["n_bins"] == 3

    def test_empty_returns_empty(self):
        assert _group_bins_into_zones([]) == []

    def test_avg_edge_ratio(self):
        bins = [
            self._make_bin(4150, "strong-positive", ratio=4.0),
            self._make_bin(4155, "strong-positive", ratio=6.0),
        ]
        zones = _group_bins_into_zones(bins)
        assert zones[0]["avg_edge_ratio"] == pytest.approx(5.0)

    def test_min_structural_n(self):
        bins = [
            self._make_bin(4150, "neutral", n=20),
            self._make_bin(4155, "neutral", n=3),
            self._make_bin(4160, "neutral", n=15),
        ]
        zones = _group_bins_into_zones(bins)
        assert zones[0]["min_structural_n"] == 3


# ── 5. compute_edge_zones ─────────────────────────────────────────────────────

class TestComputeEdgeZones:
    SPOT = 4180.0
    IM   = 40.0  # implied_move 40 pts

    def _dense_chain(self, sigma=0.20, T=5.0/252):
        strikes = range(int(self.SPOT) - 200, int(self.SPOT) + 205, 5)
        return _bsm_chain(self.SPOT, strikes, sigma=sigma, T=T)

    # ── structure tests ──

    def test_non_range_regime_returns_empty(self):
        rb = {"regime": "amplification"}
        chain = self._dense_chain()
        analogues = _analogues([self.SPOT] * 20)
        result = compute_edge_zones(
            self.SPOT, self.IM, chain, analogues, "t5", rb
        )
        assert result == []

    def test_untethered_returns_empty(self):
        rb = {"regime": "untethered"}
        result = compute_edge_zones(
            self.SPOT, self.IM, self._dense_chain(), _analogues([self.SPOT]*20), "t5", rb
        )
        assert result == []

    def test_broken_magnet_returns_empty(self):
        rb = {"regime": "broken-magnet"}
        result = compute_edge_zones(
            self.SPOT, self.IM, self._dense_chain(), _analogues([self.SPOT]*20), "t5", rb
        )
        assert result == []

    def test_output_zone_keys(self):
        rb = _regime_above(self.SPOT + 50)
        result = compute_edge_zones(
            self.SPOT, self.IM, self._dense_chain(), _analogues([self.SPOT]*20), "t5", rb
        )
        assert len(result) > 0
        for zone in result:
            for key in ("lower", "upper", "classification", "n_bins",
                        "avg_edge_ratio", "min_structural_n", "representative"):
                assert key in zone, f"Zone missing key: {key}"

    def test_zones_are_sorted_ascending(self):
        rb = _regime_above(self.SPOT + 50)
        result = compute_edge_zones(
            self.SPOT, self.IM, self._dense_chain(), _analogues([self.SPOT]*20), "t5", rb
        )
        for i in range(len(result) - 1):
            assert result[i]["upper"] == pytest.approx(result[i + 1]["lower"])
            assert result[i]["lower"] < result[i + 1]["lower"]

    def test_zones_are_non_overlapping(self):
        rb = _regime_above(self.SPOT + 50)
        result = compute_edge_zones(
            self.SPOT, self.IM, self._dense_chain(), _analogues([self.SPOT]*20), "t5", rb
        )
        for i in range(len(result) - 1):
            assert result[i]["upper"] <= result[i + 1]["lower"] + 1e-9

    def test_price_range_coverage(self):
        """Zones should span approximately spot ± 1.5*implied_move."""
        rb = _regime_above(self.SPOT + 50)
        chain = self._dense_chain()
        result = compute_edge_zones(
            self.SPOT, self.IM, chain, _analogues([self.SPOT]*20), "t5", rb,
            price_range_sigma=1.5
        )
        assert len(result) > 0
        overall_lower = result[0]["lower"]
        overall_upper = result[-1]["upper"]
        # Axis should cover ≥ 1.0× implied move on each side
        assert overall_lower <= self.SPOT - 1.0 * self.IM
        assert overall_upper >= self.SPOT + 1.0 * self.IM

    def test_classifications_are_valid_strings(self):
        rb = _regime_above(self.SPOT + 50)
        valid = {"strong-positive","moderate-positive","neutral",
                 "moderate-negative","strong-negative","unknown"}
        result = compute_edge_zones(
            self.SPOT, self.IM, self._dense_chain(), _analogues([self.SPOT]*20), "t5", rb
        )
        for zone in result:
            assert zone["classification"] in valid

    # ── direction tests ──

    def test_concentrated_structural_gives_positive_classification_in_bin(self):
        """All analogues at SPOT → the bin containing SPOT has high structural_prob.
        With a high-sigma (spread) chain, implied_prob per bin is low → strong-positive."""
        rb = _regime_above(self.SPOT + 60)
        # Use bin_size=5 explicitly; sigma=0.40 spreads implied_pdf widely
        chain = self._dense_chain(sigma=0.40)
        # Pass spot=self.SPOT, im=self.IM so anchor == today → identity projection;
        # all closes land exactly at self.SPOT.
        analogues = _analogues([self.SPOT] * 20, spot=self.SPOT, im=self.IM)
        result = compute_edge_zones(
            self.SPOT, self.IM, chain, analogues, "t5", rb,
            bin_size=5.0,
        )
        # Find the zone whose range contains SPOT
        spot_zone = next(
            (z for z in result if z["lower"] <= self.SPOT < z["upper"]), None
        )
        assert spot_zone is not None
        # With all structural_prob in one bin and wide implied_pdf, expect positive
        assert spot_zone["classification"] in (
            "strong-positive", "moderate-positive"
        ), f"Expected positive at spot bin, got {spot_zone['classification']} (ratio={spot_zone.get('avg_edge_ratio')})"

    def test_structural_below_range_gives_negative_at_range(self):
        """All analogues far below the regime range → 0 structural_prob in range bins
        → edge_ratio = 0 → strong-negative in range bins."""
        # Drift target 60 pts above spot. Analogues all at spot-100 (well below range).
        rb = _regime_above(self.SPOT + 60)
        chain = self._dense_chain(sigma=0.15)
        below_vals = [self.SPOT - 100.0] * 20  # all below spot
        analogues = _analogues(below_vals)
        result = compute_edge_zones(
            self.SPOT, self.IM, chain, analogues, "t5", rb,
            bin_size=5.0,
        )
        # Find zones in the range [SPOT, SPOT+60]
        range_zones = [z for z in result if z["upper"] > self.SPOT and z["lower"] < self.SPOT + 60]
        assert len(range_zones) > 0
        # All range zones should have classification 'strong-negative' or 'unknown'
        for z in range_zones:
            assert z["classification"] in ("strong-negative", "unknown"), (
                f"Zone {z['lower']:.0f}-{z['upper']:.0f} has unexpected classification "
                f"'{z['classification']}' (ratio={z.get('avg_edge_ratio')})"
            )

    def test_null_close_vals_give_unknown(self):
        """All-NULL close values → structural_result = None → classification = 'unknown'."""
        rb = _regime_above(self.SPOT + 50)
        chain = self._dense_chain()
        analogues = _analogues([None] * 20)  # all NULL
        result = compute_edge_zones(
            self.SPOT, self.IM, chain, analogues, "t5", rb
        )
        assert len(result) > 0
        for zone in result:
            assert zone["classification"] == "unknown"

    def test_structural_n_zero_gives_unknown(self):
        """Analogues with all-NULL close values → structural_n=0 → unknown."""
        rb = _regime_above(self.SPOT + 50)
        chain = self._dense_chain()
        analogues = [{"session_close_t5": None}] * 10
        result = compute_edge_zones(
            self.SPOT, self.IM, chain, analogues, "t5", rb
        )
        for zone in result:
            assert zone["min_structural_n"] == 0
            assert zone["classification"] == "unknown"

    # ── bin-size auto-selection ──

    def test_bin_size_auto_from_chain_5pt_im_80(self):
        """5-pt chain, IM=80 → bin_size = min(5, 20) = 5; check via representative bin."""
        rb = _regime_above(self.SPOT + 50)
        chain = _bsm_chain(self.SPOT, range(int(self.SPOT)-200, int(self.SPOT)+205, 5))
        result = compute_edge_zones(self.SPOT, 80.0, chain, _analogues([self.SPOT]*20), "t5", rb)
        if result:
            rep = result[0]["representative"]
            bin_width = rep["upper"] - rep["lower"]
            assert bin_width == pytest.approx(5.0)

    def test_bin_size_auto_from_chain_25pt_im_20(self):
        """25-pt chain, IM=20 → bin_size = min(25, 5) = 5; check via representative bin."""
        rb = _regime_above(self.SPOT + 50)
        chain = _bsm_chain(self.SPOT, range(int(self.SPOT)-200, int(self.SPOT)+205, 25))
        result = compute_edge_zones(self.SPOT, 20.0, chain, _analogues([self.SPOT]*20), "t5", rb)
        if result:
            rep = result[0]["representative"]
            bin_width = rep["upper"] - rep["lower"]
            assert bin_width == pytest.approx(5.0)

    # ── timeframe keys ──

    def test_t1_uses_correct_key(self):
        rb = _regime_above(self.SPOT + 50)
        chain = self._dense_chain()
        # Must include session_open_t0 + implied_move_1d (new normalization fields).
        # anchor == today → identity projection; session_close_t5=None is ignored.
        analogues = [
            {
                "session_close_t1":  self.SPOT,
                "session_close_t5":  None,
                "session_open_t0":   self.SPOT,
                "implied_move_1d":   self.IM,
            }
        ] * 10
        result = compute_edge_zones(
            self.SPOT, self.IM, chain, analogues, "t1", rb
        )
        # t1 key present → at least one zone not 'unknown'
        assert any(z["classification"] != "unknown" for z in result)

    def test_t15_uses_correct_key(self):
        rb = _regime_above(self.SPOT + 50)
        chain = self._dense_chain()
        # Must include session_open_t0 + implied_move_1d (new normalization fields).
        analogues = [
            {
                "session_close_t15": self.SPOT,
                "session_close_t5":  None,
                "session_open_t0":   self.SPOT,
                "implied_move_1d":   self.IM,
            }
        ] * 10
        result = compute_edge_zones(
            self.SPOT, self.IM, chain, analogues, "t15", rb
        )
        assert any(z["classification"] != "unknown" for z in result)

    # ── magnetic-pin tolerance ──

    def test_magnetic_pin_with_tolerance(self):
        rb = {"regime": "magnetic-pin", "drift_target": self.SPOT}
        chain = self._dense_chain()
        result = compute_edge_zones(
            self.SPOT, self.IM, chain, _analogues([self.SPOT]*20), "t5", rb,
            tolerance=20.0
        )
        assert isinstance(result, list)

    def test_magnetic_pin_without_tolerance_returns_empty(self):
        rb = {"regime": "magnetic-pin", "drift_target": self.SPOT}
        chain = self._dense_chain()
        # No tolerance → get_trade_thesis_range raises → caught → returns []
        result = compute_edge_zones(
            self.SPOT, self.IM, chain, _analogues([self.SPOT]*20), "t5", rb
        )
        assert result == []

    # ── min_structural_n tracking ──

    def test_min_structural_n_tracked_per_zone(self):
        """Zones expose the smallest denominator in their bins."""
        rb = _regime_above(self.SPOT + 50)
        chain = self._dense_chain()
        analogues = _analogues([self.SPOT] * 20)
        result = compute_edge_zones(
            self.SPOT, self.IM, chain, analogues, "t5", rb
        )
        for zone in result:
            assert isinstance(zone["min_structural_n"], int)
            assert zone["min_structural_n"] >= 0

    # ── Step 3.6: one-sided magnet range semantics ──

    def test_structural_prob_above_magnet_when_closes_overshoot(self):
        """Step 3.6: magnet-above trade thesis is P(close ≥ magnet).

        Projected closes all landing above drift_target → bins above drift_target
        should have positive structural_prob in their representative.  This verifies
        that the one-sided range (lower=drift_target, upper=None) flows through
        get_trade_thesis_range correctly and that the bin-level structural_prob
        computation picks up analogues whose projected close exceeds the magnet.
        """
        drift_target = self.SPOT + 20.0  # magnet 20pts above spot (= 4200)
        rb = _regime_above(drift_target)
        chain = self._dense_chain()
        # All closes 30 pts above SPOT → all land above drift_target (+20) after
        # identity projection (anchor == today → no scaling).
        above_vals = [self.SPOT + 30.0] * 20
        analogues = _analogues(above_vals, spot=self.SPOT, im=self.IM)
        result = compute_edge_zones(
            self.SPOT, self.IM, chain, analogues, "t5", rb, bin_size=5.0
        )
        assert len(result) > 0
        # At least one zone above drift_target must show positive structural_prob
        above_magnet = [z for z in result if z["lower"] >= drift_target]
        assert len(above_magnet) > 0, "Expected zones above drift_target in price axis"
        any_positive_sp = any(
            z["representative"].get("structural_prob", 0) > 0
            for z in above_magnet
        )
        assert any_positive_sp, (
            f"Expected structural_prob > 0 in bins above drift_target={drift_target}; "
            f"got: {[(z['lower'], z['representative'].get('structural_prob')) for z in above_magnet]}"
        )
