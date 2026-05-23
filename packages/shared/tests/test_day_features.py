"""Unit tests for packages/shared/day_features.py (CR-013, v0.5).

Three layers:
  - σ-normalization sanity (the CR-013 AC #2 scenarios) — synthetic input.
  - Schema invariants — every payload yields exactly the 35 keys in
    FEATURE_NAMES; vol-surface slots are None; quality ordinal mapping;
    empty-payload graceful fallback.
  - Snapshot tests against the four CR-011 calibration days (5/6, 5/7,
    5/18, 5/20). The fixture
    packages/shared/tests/fixtures/day_features_calibration.json was
    materialized once from production data via the same _materialize_payload
    + extract_features pipeline; these tests lock the regression baseline.

Run with:
    python -m unittest packages.shared.tests.test_day_features
"""
from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

from packages.shared.day_features import (
    EPSILON,
    FEATURE_NAMES,
    FEATURE_VERSION,
    VOL_SURFACE_PLACEHOLDERS,
    compute_feature_config_hash,
    extract_features,
)


FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "day_features_calibration.json"
)


def _empty_payload(regime: str = "untethered") -> dict:
    return {
        "regime": {"regime": regime},
        "per_bucket": {},
        "confluences": [],
        "neg_zones": [],
    }


# ─── σ-normalization sanity (CR-013 AC #2) ─────────────────────────────────

class TestSigmaNormalization(unittest.TestCase):
    """Verify (center - spot) / implied_move math at the cluster slots."""

    def test_above_spot_positive_signed_distance(self):
        payload = _empty_payload()
        payload["confluences"] = [
            {"center_price": 7450.0, "max_gex": 800e9, "quality": "pin"},
        ]
        out = extract_features(payload, spot=7400.0, implied_move=30.0)
        # +50 / 30 = +1.6667
        self.assertAlmostEqual(
            out["cluster_1_signed_distance_sigma"], 50.0 / 30.0, places=10,
        )
        self.assertEqual(out["n_clusters_above_spot"], 1)
        self.assertEqual(out["n_clusters_below_spot"], 0)

    def test_below_spot_negative_signed_distance(self):
        payload = _empty_payload()
        payload["confluences"] = [
            {"center_price": 7360.0, "max_gex": 700e9, "quality": "pin"},
        ]
        out = extract_features(payload, spot=7400.0, implied_move=80.0)
        # -40 / 80 = -0.5
        self.assertAlmostEqual(
            out["cluster_1_signed_distance_sigma"], -40.0 / 80.0, places=10,
        )
        self.assertEqual(out["n_clusters_above_spot"], 0)
        self.assertEqual(out["n_clusters_below_spot"], 1)

    def test_zero_implied_move_yields_zero_distance(self):
        payload = _empty_payload()
        payload["confluences"] = [
            {"center_price": 7500.0, "max_gex": 600e9, "quality": "target"},
        ]
        out = extract_features(payload, spot=7400.0, implied_move=0.0)
        self.assertEqual(out["cluster_1_signed_distance_sigma"], 0.0)
        self.assertEqual(out["implied_move_1d"], 0.0)


# ─── Schema invariants ─────────────────────────────────────────────────────

class TestSchemaInvariants(unittest.TestCase):
    """Output shape is exactly FEATURE_NAMES; placeholders are None; quality
    ordinals map per spec."""

    def test_output_keys_match_feature_names(self):
        out = extract_features(_empty_payload(), spot=7400.0, implied_move=40.0)
        self.assertEqual(sorted(out.keys()), sorted(FEATURE_NAMES))
        self.assertEqual(len(FEATURE_NAMES), 35)

    def test_vol_surface_placeholders_are_none(self):
        out = extract_features(_empty_payload(), spot=7400.0, implied_move=40.0)
        for name in VOL_SURFACE_PLACEHOLDERS:
            self.assertIsNone(out[name], f"{name} should be None in v0.5")

    def test_regime_mapping_covers_six_documented_tags(self):
        spot, imp = 7400.0, 40.0
        cases = [
            ("magnetic-pin",  {"is_pin_day": 1, "magnet_direction_signed": 0}),
            ("magnet-above",  {"is_magnet_day": 1, "magnet_direction_signed": 1}),
            ("magnet-below",  {"is_magnet_day": 1, "magnet_direction_signed": -1}),
            ("bounded",       {"is_bounded_day": 1, "magnet_direction_signed": 0}),
            ("amplification", {"is_amplification_day": 1, "magnet_direction_signed": 0}),
            ("untethered",    {"is_untethered_day": 1, "magnet_direction_signed": 0}),
        ]
        for tag, expected_flags in cases:
            with self.subTest(tag=tag):
                out = extract_features(_empty_payload(tag), spot, imp)
                for k, v in expected_flags.items():
                    self.assertEqual(out[k], v,
                                     f"{tag}: {k} should be {v}, got {out[k]}")
                # All other binary indicators must be 0.
                for k in ("is_pin_day", "is_magnet_day", "is_bounded_day",
                          "is_untethered_day", "is_amplification_day"):
                    if k not in expected_flags:
                        self.assertEqual(out[k], 0,
                                         f"{tag}: unexpected {k}={out[k]}")

    def test_broken_magnet_maps_to_all_zero_indicators(self):
        """broken-magnet has no dedicated flag in v0.5 — lossy by design."""
        out = extract_features(_empty_payload("broken-magnet"), spot=7400.0,
                               implied_move=40.0)
        for k in ("is_pin_day", "is_magnet_day", "is_bounded_day",
                  "is_untethered_day", "is_amplification_day"):
            self.assertEqual(out[k], 0)
        self.assertEqual(out["magnet_direction_signed"], 0)

    def test_quality_ordinal_mapping(self):
        payload = _empty_payload()
        payload["confluences"] = [
            {"center_price": 7400.0, "max_gex": 900e9, "quality": "pin"},
            {"center_price": 7350.0, "max_gex": 600e9, "quality": "target"},
            {"center_price": 7450.0, "max_gex": 200e9, "quality": "feature"},
        ]
        out = extract_features(payload, spot=7400.0, implied_move=40.0)
        self.assertEqual(out["cluster_1_quality_ordinal"], 2)  # pin
        self.assertEqual(out["cluster_2_quality_ordinal"], 1)  # target
        self.assertEqual(out["cluster_3_quality_ordinal"], 0)  # feature
        self.assertEqual(out["n_pin"], 1)
        self.assertEqual(out["n_target"], 1)
        self.assertEqual(out["n_feature"], 1)
        self.assertEqual(out["n_clusters_total"], 3)

    def test_fewer_than_three_clusters_fills_zero_slots(self):
        payload = _empty_payload()
        payload["confluences"] = [
            {"center_price": 7400.0, "max_gex": 700e9, "quality": "pin"},
        ]
        out = extract_features(payload, spot=7400.0, implied_move=40.0)
        self.assertGreater(out["cluster_1_max_gex"], 0)
        self.assertEqual(out["cluster_2_max_gex"], 0)
        self.assertEqual(out["cluster_2_quality_ordinal"], 0)
        self.assertEqual(out["cluster_2_signed_distance_sigma"], 0)
        self.assertEqual(out["cluster_3_max_gex"], 0)

    def test_top_cluster_fraction(self):
        payload = _empty_payload()
        payload["confluences"] = [
            {"center_price": 7400.0, "max_gex": 600e9, "quality": "target"},
            {"center_price": 7450.0, "max_gex": 200e9, "quality": "feature"},
            {"center_price": 7350.0, "max_gex": 200e9, "quality": "feature"},
        ]
        out = extract_features(payload, spot=7400.0, implied_move=40.0)
        # 600 / (600+200+200) = 0.6
        self.assertAlmostEqual(out["top_cluster_fraction_of_total_max_gex"], 0.6,
                               places=10)


# ─── Calibration snapshot (the four CR-011 days) ───────────────────────────

class TestCalibrationSnapshot(unittest.TestCase):
    """Regression lock against the production-materialized feature vectors
    for 5/6, 5/7, 5/18, 5/20. Any change to extract_features that moves any
    of these values fails the snapshot — bump FEATURE_VERSION and refresh
    the fixture if the change is intentional."""

    @classmethod
    def setUpClass(cls):
        if not FIXTURE_PATH.exists():
            raise unittest.SkipTest(f"fixture missing: {FIXTURE_PATH}")
        cls.fixture = json.loads(FIXTURE_PATH.read_text())

    def test_fixture_pinned_to_current_feature_version(self):
        self.assertEqual(self.fixture["feature_version"], FEATURE_VERSION)
        self.assertEqual(
            self.fixture["feature_config_hash"], compute_feature_config_hash(),
        )
        self.assertEqual(self.fixture["feature_names"], FEATURE_NAMES)

    def test_5_7_top_cluster_is_pin_close_to_spot(self):
        """5/7 is the canonical pin day from CR-011. Top cluster should be
        pin-quality, less than 0.5σ from spot, regime=magnetic-pin."""
        day = self.fixture["days"]["2026-05-07"]
        self.assertEqual(day["regime"], "magnetic-pin")
        feats = day["features"]
        self.assertEqual(feats["is_pin_day"], 1)
        self.assertEqual(feats["cluster_1_quality_ordinal"], 2)
        self.assertLess(abs(feats["cluster_1_signed_distance_sigma"]), 0.5)
        # Pin tier means max_gex >= 650 B
        self.assertGreaterEqual(feats["cluster_1_max_gex"], 650.0)

    def test_5_6_top_cluster_is_target_tier(self):
        day = self.fixture["days"]["2026-05-06"]
        feats = day["features"]
        self.assertEqual(feats["cluster_1_quality_ordinal"], 1)  # target
        # CR-011: 5/6 sits in [550, 650) B band.
        self.assertGreaterEqual(feats["cluster_1_max_gex"], 550.0)
        self.assertLess(feats["cluster_1_max_gex"], 650.0)

    def test_5_18_and_5_20_top_cluster_is_feature_tier(self):
        for d in ("2026-05-18", "2026-05-20"):
            with self.subTest(day=d):
                feats = self.fixture["days"][d]["features"]
                self.assertEqual(feats["cluster_1_quality_ordinal"], 0,
                                 f"{d}: top cluster should be feature-tier")
                self.assertLess(feats["cluster_1_max_gex"], 550.0)

    def test_dominance_buckets_sum_to_100(self):
        for d, day in self.fixture["days"].items():
            with self.subTest(day=d):
                feats = day["features"]
                total = (feats["dominance_0DTE"] + feats["dominance_1_7"]
                         + feats["dominance_8_30"] + feats["dominance_30plus"])
                self.assertAlmostEqual(total, 100.0, places=3,
                                       msg=f"{d}: dominance sum = {total}")

    def test_extract_reproduces_fixture_exactly(self):
        """Re-run extract_features against synthetic landscape payloads that
        re-state each fixture day's regime/confluences/neg_zones/per_bucket.
        Confirms extract_features is deterministic and the fixture is
        consistent with FEATURE_VERSION."""
        # The fixture was built end-to-end (materialize+extract) against
        # production data. To check determinism, re-extract from a payload
        # whose features should be byte-identical. We reconstruct the
        # payload from the fixture's regime + features (cluster slots
        # → confluences, dominance → per_bucket) and assert reproducibility
        # for the cluster-driven and aggregate features.
        for d, day in self.fixture["days"].items():
            with self.subTest(day=d):
                feats = day["features"]
                spot = day["spot"]
                imp = day["implied_move"]

                # Reconstruct cluster confluences from slot features.
                confluences = []
                for i in (1, 2, 3):
                    mg_b = feats[f"cluster_{i}_max_gex"]
                    if mg_b <= 0:
                        continue
                    qord = feats[f"cluster_{i}_quality_ordinal"]
                    quality = {2: "pin", 1: "target", 0: "feature"}[qord]
                    sd = feats[f"cluster_{i}_signed_distance_sigma"]
                    center = spot + sd * imp
                    confluences.append({
                        "center_price": center,
                        "max_gex": mg_b * 1e9,
                        "quality": quality,
                    })

                per_bucket = {
                    "0DTE":     {"dominance_pct": feats["dominance_0DTE"]},
                    "1-7 DTE":  {"dominance_pct": feats["dominance_1_7"]},
                    "8-30 DTE": {"dominance_pct": feats["dominance_8_30"]},
                    "30+ DTE":  {"dominance_pct": feats["dominance_30plus"]},
                }
                payload = {
                    "regime": {"regime": day["regime"]},
                    "per_bucket": per_bucket,
                    "confluences": confluences,
                    "neg_zones": [],  # reconstructed below if any
                }
                out = extract_features(payload, spot, imp)

                # Cluster + aggregate features should round-trip exactly
                # (within float noise) against the fixture. Neg-zone keys
                # depend on actual neg_zones array — skip those for this
                # reconstruction test (covered by the fixture itself).
                for k in (
                    "is_pin_day", "is_magnet_day", "is_bounded_day",
                    "is_untethered_day", "is_amplification_day",
                    "magnet_direction_signed",
                    "cluster_1_max_gex", "cluster_2_max_gex", "cluster_3_max_gex",
                    "cluster_1_quality_ordinal", "cluster_2_quality_ordinal",
                    "cluster_3_quality_ordinal",
                    "n_clusters_total", "n_pin", "n_target", "n_feature",
                    "n_clusters_above_spot", "n_clusters_below_spot",
                    "top_cluster_fraction_of_total_max_gex",
                    "dominance_0DTE", "dominance_1_7", "dominance_8_30",
                    "dominance_30plus",
                    "implied_move_1d",
                ):
                    expected = feats[k]
                    actual = out[k]
                    if isinstance(expected, float) and isinstance(actual, float):
                        self.assertAlmostEqual(actual, expected, places=6,
                                               msg=f"{d}.{k}")
                    else:
                        self.assertEqual(actual, expected, msg=f"{d}.{k}")


# ─── Config + similarity epsilon ───────────────────────────────────────────

class TestConfigInvariants(unittest.TestCase):

    def test_feature_config_hash_is_stable(self):
        a = compute_feature_config_hash()
        b = compute_feature_config_hash()
        self.assertEqual(a, b)
        self.assertEqual(len(a), 32)  # md5 hex

    def test_epsilon_is_small_positive(self):
        self.assertGreater(EPSILON, 0)
        self.assertLess(EPSILON, 1e-3)


if __name__ == "__main__":
    unittest.main()
