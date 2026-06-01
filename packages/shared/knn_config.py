"""Versioned KNN ranking configuration (CR-029 / CR-030).

Parameters here are applied OUTSIDE the structural distance vector — after
similarity_distance() computes the σ-normalized Euclidean distance for each
candidate, and inside rank_analogues() before returning results.

  distance_ceiling  — hard upper bound on σ-normalized distance (effective
                      distance when recency is active). Candidates beyond this
                      threshold are excluded regardless of K.  Makes K adaptive.
                      math.inf = no ceiling (v0 / backward-compatible behavior).

  z_diff_cap        — per-feature cap on |z_q − z_c| before squaring (CR-030).
                      Prevents near-zero-mean ordinal/categorical features from
                      dominating via extreme z-scores (e.g. a rare value=2 on a
                      feature whose corpus mean≈0 would otherwise z-score to ~16).
                      None = no cap (backward-compatible).

  feature_weights   — dict[feature_name, float] applied as
                      `weight × capped_diff²` in the weighted distance summand
                      (CR-030).  Keys are FEATURE_NAMES; missing keys default to
                      1.0.  None = equal weighting (backward-compatible; uses the
                      original similarity_distance path, identical ordering).

  half_life_months  — recency decay half-life in months (CR-030, Option A).
                      After structural distance is computed, each candidate's
                      effective distance is scaled by 2^(years_ago / half_life_yr)
                      where half_life_yr = half_life_months / 12.  Applied BEFORE
                      the ceiling so the ceiling is on effective distance.
                      0 or None = no decay (backward-compatible).

Config changes are non-destructive and reproducible: the same anchor can be
re-ranked under different config versions for A/B comparison, exactly as the
feature_version pattern works for feature schemas.
"""
from __future__ import annotations

import math
from typing import Optional

# ── v2 first-pass expert-prior weights (CR-030 Step 0) ─────────────────────
# Rationale groups:
#   Regime flags (2.5): strongest structural signal — same regime = genuine match
#   magnet_direction_signed (2.0): directional nuance within regime
#   cluster_1_signed_distance_sigma (3.0): spot proximity to the top GEX wall
#   cluster_2/3_signed_distance_sigma (2.0/1.5): secondary / tertiary proximity
#   implied_move_1d (2.0): vol context ties structural setups to vol regime
#   cluster_1_quality_ordinal (1.5): quality of the TOP cluster matters
#   cluster_1_max_gex (1.0): absolute GEX strength (moderate; varies across eras)
#   cluster_2/3_max_gex (0.5/0.25): secondary/tertiary absolute GEX — less so
#   cluster_2/3_quality_ordinal (0.25): DOWN-WEIGHTED — near-zero-mean blowup
#     source AND correlated with n_pin; z_diff_cap also applies but weight kept
#     low to further limit influence
#   n_pin/n_target/n_feature (0.25): DOWN-WEIGHTED — correlated with quality
#     ordinals; count features are less informative than ordinal of top cluster
#   cluster aggregates (0.5): weakly informative topology counts
#   top_cluster_fraction (0.75): ratio captures relative dominance
#   bucket dominances (0.75 each): bucket exposure — informative but redundant
#   neg zone features (0.3–0.5): mostly structural trivia for matching purposes
_V2_FEATURE_WEIGHTS: dict[str, float] = {
    # Regime / structure
    "is_pin_day":                        2.5,
    "is_magnet_day":                     2.5,
    "is_bounded_day":                    2.5,
    "is_untethered_day":                 2.5,
    "is_amplification_day":              2.5,
    "magnet_direction_signed":           2.0,
    # Top 3 cluster slots
    "cluster_1_max_gex":                 1.0,
    "cluster_1_quality_ordinal":         1.5,
    "cluster_1_signed_distance_sigma":   3.0,
    "cluster_2_max_gex":                 0.5,
    "cluster_2_quality_ordinal":         0.25,   # down-weighted: blowup source
    "cluster_2_signed_distance_sigma":   2.0,
    "cluster_3_max_gex":                 0.25,
    "cluster_3_quality_ordinal":         0.25,   # down-weighted
    "cluster_3_signed_distance_sigma":   1.5,
    # Cluster aggregates
    "n_clusters_total":                  0.5,
    "n_pin":                             0.25,   # down-weighted: correlated
    "n_target":                          0.25,   # down-weighted: correlated
    "n_feature":                         0.25,   # down-weighted: correlated
    "n_clusters_above_spot":             0.5,
    "n_clusters_below_spot":             0.5,
    "top_cluster_fraction_of_total_max_gex": 0.75,
    # Per-bucket dominance
    "dominance_0DTE":                    0.75,
    "dominance_1_7":                     0.75,
    "dominance_8_30":                    0.75,
    "dominance_30plus":                  0.75,
    # Neg zones
    "n_neg_zones":                       0.3,
    "nearest_neg_signed_distance_sigma": 0.5,
    "total_neg_max_gex":                 0.3,
    # Vol regime
    "implied_move_1d":                   2.0,
    # Vol surface placeholders (NULL in v0.5 — weights present for schema
    # completeness but never evaluated since NULL values are skipped)
    "atm_iv_percentile":                 1.0,
    "skew_percentile":                   1.0,
    "smile_convexity":                   1.0,
    "term_structure_slope":              1.0,
    "vol_risk_premium":                  1.0,
}

KNN_CONFIGS: dict[str, dict] = {
    "v0": {
        "distance_ceiling": math.inf,
    },
    "v1": {
        "distance_ceiling": 4.0,
    },
    "v2": {
        "distance_ceiling": 5.0,      # re-picked post-reweighting; 1.6% zero rate
        "z_diff_cap":       3.0,       # per-feature |z_q−z_c| cap
        "feature_weights":  _V2_FEATURE_WEIGHTS,
        "half_life_months": 18.0,      # recency: 1-year-old day counts 2× harder
    },
}

CANONICAL_KNN_CONFIG_VERSION: str = "v2"


def get_knn_config(version: Optional[str] = None) -> dict:
    """Return the config dict for the given version (defaults to canonical).

    Raises ValueError for unknown versions so callers fail loudly rather than
    silently using the wrong config.
    """
    v = version or CANONICAL_KNN_CONFIG_VERSION
    cfg = KNN_CONFIGS.get(v)
    if cfg is None:
        raise ValueError(
            f"Unknown knn_config_version: {v!r}. "
            f"Valid versions: {list(KNN_CONFIGS)}"
        )
    return cfg
