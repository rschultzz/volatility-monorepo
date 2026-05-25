"""K-nearest-neighbour helpers for day-structure analogue ranking (CR-013, v0.5).

Moved from apps/web/modules/Analogues/service.py (CR-C) so that
packages/shared code (probability.py) can import without creating an
apps → packages reverse dependency.

Analogues/service.py re-exports these symbols for backward compatibility;
its call sites (routes.py) are unchanged.

Public entry points:
    feature_stats(feature_vectors)               → {feature: {mean, std}}
    similarity_distance(query, candidate, stats) → float
    rank_analogues(anchor_vec, candidates, k, *, exclude_date, stats)
                                                 → list of (trade_date, distance)
"""
from __future__ import annotations

import math
from typing import Iterable, Optional

from packages.shared.day_features import EPSILON, FEATURE_NAMES


def feature_stats(feature_vectors: Iterable[dict]) -> dict:
    """Compute corpus mean/std per feature across the candidate set.

    NULL-aware: only non-None values contribute to mean/std. Returns
    {feature: {"mean": float, "std": float}} for every FEATURE_NAMES key.
    Features with zero or one non-None observation get std=0 (the
    similarity function floors std at EPSILON, so this is safe).
    """
    stats: dict[str, dict[str, float]] = {}
    vectors = list(feature_vectors)
    for name in FEATURE_NAMES:
        values: list[float] = []
        for v in vectors:
            x = v.get(name)
            if x is None:
                continue
            try:
                values.append(float(x))
            except (TypeError, ValueError):
                continue
        if not values:
            stats[name] = {"mean": 0.0, "std": 0.0}
            continue
        mean = sum(values) / len(values)
        if len(values) <= 1:
            stats[name] = {"mean": mean, "std": 0.0}
            continue
        var = sum((v - mean) ** 2 for v in values) / len(values)
        stats[name] = {"mean": mean, "std": math.sqrt(var)}
    return stats


def similarity_distance(query_vec: dict, candidate_vec: dict,
                        feature_stats: dict) -> float:
    """Weighted Euclidean distance, NULL-aware. Smaller = more similar.

    Features whose value is None on either side are skipped. After
    summation, the distance is rescaled by
    sqrt(n_total_features / n_active_features) so distances are
    comparable across queries with different active-feature counts.
    Returns inf if no feature is active on both sides.
    """
    sum_sq = 0.0
    n_active = 0
    for name in FEATURE_NAMES:
        q = query_vec.get(name)
        c = candidate_vec.get(name)
        if q is None or c is None:
            continue
        try:
            qf = float(q)
            cf = float(c)
        except (TypeError, ValueError):
            continue
        stats = feature_stats.get(name, {"mean": 0.0, "std": 0.0})
        std = max(stats["std"], EPSILON)
        z_q = (qf - stats["mean"]) / std
        z_c = (cf - stats["mean"]) / std
        sum_sq += (z_q - z_c) ** 2
        n_active += 1
    if n_active == 0:
        return float("inf")
    rescale = math.sqrt(len(FEATURE_NAMES) / n_active)
    return math.sqrt(sum_sq) * rescale


def rank_analogues(
    anchor_vec: dict,
    candidates: Iterable[tuple],
    k: int,
    *,
    exclude_date: Optional[str] = None,
    stats: Optional[dict] = None,
) -> list[tuple]:
    """Rank candidate days by similarity to the anchor.

    candidates is an iterable of (trade_date_iso, feature_vector_dict).
    Returns a list of (trade_date_iso, distance) sorted closest-first,
    truncated to k. exclude_date (ISO string) is removed from the
    candidate list before ranking. stats may be precomputed; falls back
    to computing across candidates.
    """
    cand_list = [(d, v) for (d, v) in candidates if d != exclude_date]
    if not cand_list:
        return []
    if stats is None:
        stats = feature_stats(v for (_, v) in cand_list)
    scored = [
        (d, similarity_distance(anchor_vec, v, stats))
        for (d, v) in cand_list
    ]
    scored.sort(key=lambda x: x[1])
    return scored[:k]
