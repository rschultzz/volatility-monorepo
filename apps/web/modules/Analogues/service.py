"""Analogues service — pure functions for the Day Analogue Comparison KNN
(CR-013, v0.5; CR-030 weighted breakdown).

Stateless: takes feature vectors + corpus stats, returns rankings. The
routes layer (routes.py) is responsible for all DB I/O.

feature_stats, similarity_distance, rank_analogues, and
weighted_similarity_distance were moved to packages/shared/knn.py (CR-C /
CR-030) so that packages/shared/probability.py can import them without
creating an apps → packages reverse dependency.
They are re-exported here unchanged so routes.py and other callers need
no modification.

Public entry points (re-exported from packages.shared.knn):
    feature_stats(feature_vectors)               → {feature: {mean, std}}
    similarity_distance(query, candidate, stats) → float
    weighted_similarity_distance(query, candidate, stats, *, feature_weights,
                                 z_diff_cap)     → float   (CR-030)
    rank_analogues(anchor_vec, candidates, k, *, exclude_date, stats, ...)
                                                 → list of (trade_date, distance)

Defined here:
    feature_distance_breakdown(anchor_vec, candidate_vec, stats, *, top_n,
                               feature_weights, z_diff_cap)
                                                 → list of dicts (CR-030: weighted)
"""
from __future__ import annotations

import math
from typing import Optional

from packages.shared.day_features import EPSILON, FEATURE_NAMES
from packages.shared.knn import (          # re-export; call sites unchanged
    feature_stats,
    rank_analogues,
    similarity_distance,
    weighted_similarity_distance,
)

__all__ = [
    "feature_stats",
    "similarity_distance",
    "weighted_similarity_distance",
    "rank_analogues",
    "feature_distance_breakdown",
]


def feature_distance_breakdown(
    anchor_vec: dict,
    candidate_vec: dict,
    stats: dict,
    *,
    top_n: int = 5,
    feature_weights: Optional[dict] = None,
    z_diff_cap: Optional[float] = None,
) -> list[dict]:
    """Return the top-N features by absolute σ-normalized contribution to distance.

    Each entry: { feature_name, anchor_value, analogue_value, sigma_delta,
                  contribution }

    sigma_delta  = effective (z_q − z_c) after z_diff_cap applied (CR-030).
    contribution = weight × sigma_delta²  (weight=1.0 when feature_weights=None).

    When feature_weights / z_diff_cap are provided (CR-030 v2 config), the
    breakdown reflects the *actual* weighted distance decomposition so the UI
    shows why this analogue was selected under the active config.
    """
    contributions = []
    for name in FEATURE_NAMES:
        q = anchor_vec.get(name)
        c = candidate_vec.get(name)
        if q is None or c is None:
            continue
        try:
            qf = float(q)
            cf = float(c)
        except (TypeError, ValueError):
            continue
        s = stats.get(name, {"mean": 0.0, "std": 0.0})
        std = max(s["std"], EPSILON)
        z_q = (qf - s["mean"]) / std
        z_c = (cf - s["mean"]) / std
        sigma_delta = z_q - z_c
        # Apply z_diff_cap if provided (CR-030)
        if z_diff_cap is not None and abs(sigma_delta) > z_diff_cap:
            sigma_delta = math.copysign(z_diff_cap, sigma_delta)
        # Apply feature weight (CR-030; defaults to 1.0 for backward compat)
        w = (feature_weights or {}).get(name, 1.0)
        contributions.append({
            "feature_name": name,
            "anchor_value": qf,
            "analogue_value": cf,
            "sigma_delta": round(sigma_delta, 4),
            "contribution": round(w * sigma_delta ** 2, 4),
        })
    contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return contributions[:top_n]
