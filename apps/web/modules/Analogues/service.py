"""Analogues service — pure functions for the Day Analogue Comparison KNN
(CR-013, v0.5).

Stateless: takes feature vectors + corpus stats, returns rankings. The
routes layer (routes.py) is responsible for all DB I/O.

feature_stats, similarity_distance, and rank_analogues were moved to
packages/shared/knn.py (CR-C) so that packages/shared/probability.py
can import them without creating an apps → packages reverse dependency.
They are re-exported here unchanged so routes.py and other callers need
no modification.

Public entry points (re-exported from packages.shared.knn):
    feature_stats(feature_vectors)               → {feature: {mean, std}}
    similarity_distance(query, candidate, stats) → float
    rank_analogues(anchor_vec, candidates, k, *, exclude_date, stats)
                                                 → list of (trade_date, distance)

Defined here:
    feature_distance_breakdown(anchor_vec, candidate_vec, stats, *, top_n)
                                                 → list of dicts
"""
from __future__ import annotations

import math
from typing import Optional

from packages.shared.day_features import EPSILON, FEATURE_NAMES
from packages.shared.knn import (          # re-export; call sites unchanged
    feature_stats,
    rank_analogues,
    similarity_distance,
)

__all__ = [
    "feature_stats",
    "similarity_distance",
    "rank_analogues",
    "feature_distance_breakdown",
]


def feature_distance_breakdown(
    anchor_vec: dict,
    candidate_vec: dict,
    stats: dict,
    *,
    top_n: int = 5,
) -> list[dict]:
    """Return the top-N features by absolute σ-normalized contribution to distance.

    Each entry: { feature_name, anchor_value, analogue_value, sigma_delta, contribution }
    where sigma_delta = (z_anchor - z_candidate) and contribution = sigma_delta^2.
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
        contributions.append({
            "feature_name": name,
            "anchor_value": qf,
            "analogue_value": cf,
            "sigma_delta": round(sigma_delta, 4),
            "contribution": round(sigma_delta ** 2, 4),
        })
    contributions.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return contributions[:top_n]
