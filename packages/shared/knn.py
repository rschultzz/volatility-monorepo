"""K-nearest-neighbour helpers for day-structure analogue ranking (CR-013, v0.5;
CR-029 before_date + ceiling; CR-030 weights + z_diff_cap + recency).

Moved from apps/web/modules/Analogues/service.py (CR-C) so that
packages/shared code (probability.py) can import without creating an
apps → packages reverse dependency.

Analogues/service.py re-exports these symbols for backward compatibility;
its call sites (routes.py) are unchanged.

Public entry points:
    feature_stats(feature_vectors)               → {feature: {mean, std}}
    similarity_distance(query, candidate, stats) → float   (unweighted, unchanged)
    weighted_similarity_distance(query, candidate, stats, *, feature_weights,
                                 z_diff_cap)     → float   (CR-030)
    rank_analogues(anchor_vec, candidates, k, *, exclude_date, before_date,
                   stats, distance_ceiling,
                   feature_weights, z_diff_cap,  (CR-030)
                   half_life_months, anchor_date) → list of (trade_date, distance)
"""
from __future__ import annotations

import datetime as dt
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
    """Unweighted Euclidean distance, NULL-aware. Smaller = more similar.

    Features whose value is None on either side are skipped. After
    summation, the distance is rescaled by
    sqrt(n_total_features / n_active_features) so distances are
    comparable across queries with different active-feature counts.
    Returns inf if no feature is active on both sides.

    This function is UNCHANGED from CR-029 — it provides the exact
    baseline ordering. rank_analogues uses it when feature_weights=None.
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


def weighted_similarity_distance(
    query_vec: dict,
    candidate_vec: dict,
    feature_stats: dict,
    *,
    feature_weights: dict,
    z_diff_cap: Optional[float] = None,
) -> float:
    """Weighted, z_diff-capped Euclidean distance (CR-030).

    Summand: weight × min(|z_q − z_c|, z_diff_cap)²

    feature_weights  — dict[feature_name, float]. Missing keys default to 1.0.
    z_diff_cap       — per-feature cap on |z_q − z_c|. Prevents near-zero-mean
                       categorical features from dominating via extreme z-scores
                       (e.g. a rare value on a feature whose corpus mean≈0 would
                       otherwise produce |z_diff|≈16 and dominate the distance).
                       None = no cap.

    Rescaling: same sqrt(n_total / n_active) as similarity_distance so that
    distances across anchors with different active-feature counts are comparable.
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
        diff = z_q - z_c
        if z_diff_cap is not None and abs(diff) > z_diff_cap:
            diff = math.copysign(z_diff_cap, diff)
        w = feature_weights.get(name, 1.0)
        sum_sq += w * diff * diff
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
    before_date: Optional[str] = None,
    stats: Optional[dict] = None,
    distance_ceiling: float = math.inf,
    # CR-030 additions (all default to None → backward-compatible baseline)
    feature_weights: Optional[dict] = None,
    z_diff_cap: Optional[float] = None,
    half_life_months: Optional[float] = None,
    anchor_date: Optional[str] = None,
) -> list[tuple]:
    """Rank candidate days by similarity to the anchor.

    candidates is an iterable of (trade_date_iso, feature_vector_dict).
    Returns a list of (trade_date_iso, distance) sorted closest-first,
    truncated to k. Filters applied in order:

      1. exclude_date (ISO string) — removed before ranking (self-exclusion).
      2. before_date (ISO string) — excludes candidates with date >= before_date;
         used for as-of-history backtesting to prevent look-ahead bias (CR-029).
      3. feature_stats computed from the remaining pool when stats is None.
      4. Distance computation:
           feature_weights=None → similarity_distance (baseline, CR-029 ordering).
           feature_weights set  → weighted_similarity_distance with z_diff_cap.
      5. Recency scaling (CR-030): if half_life_months and anchor_date are set,
         effective_distance = raw_distance × 2^(years_ago / half_life_yr).
         Applied BEFORE the ceiling so the ceiling is on effective distance.
      6. Sort by (effective) distance.
      7. distance_ceiling — hard upper bound; candidates beyond are dropped,
         making K adaptive (CR-029, on effective distance when recency active).
      8. k-slice.

    Backward compatibility:
      feature_weights=None, z_diff_cap=None, half_life_months=None → identical
      to pre-CR-030 behavior; same ordering, same distances returned.
      distance_ceiling=math.inf reproduces pre-CR-029 behavior.
    """
    cand_list = [(d, v) for (d, v) in candidates if d != exclude_date]
    if before_date is not None:
        cand_list = [(d, v) for (d, v) in cand_list if d < before_date]
    if not cand_list:
        return []
    if stats is None:
        stats = feature_stats(v for (_, v) in cand_list)

    # ── Distance computation ────────────────────────────────────────────────
    if feature_weights is not None:
        scored = [
            (d, weighted_similarity_distance(anchor_vec, v, stats,
                                             feature_weights=feature_weights,
                                             z_diff_cap=z_diff_cap))
            for (d, v) in cand_list
        ]
    else:
        scored = [
            (d, similarity_distance(anchor_vec, v, stats))
            for (d, v) in cand_list
        ]

    # ── Recency scaling ─────────────────────────────────────────────────────
    if half_life_months and anchor_date:
        half_life_yr = half_life_months / 12.0
        anchor_dt = dt.date.fromisoformat(anchor_date)
        new_scored = []
        for (cand_date, raw_dist) in scored:
            try:
                cand_dt = dt.date.fromisoformat(cand_date)
            except ValueError:
                new_scored.append((cand_date, raw_dist))
                continue
            years_ago = (anchor_dt - cand_dt).days / 365.25
            if years_ago <= 0:
                new_scored.append((cand_date, raw_dist))
            else:
                effective = raw_dist * (2.0 ** (years_ago / half_life_yr))
                new_scored.append((cand_date, effective))
        scored = new_scored

    # ── Sort, ceiling, slice ────────────────────────────────────────────────
    scored.sort(key=lambda x: x[1])
    if math.isfinite(distance_ceiling):
        scored = [(d, dist) for (d, dist) in scored if dist <= distance_ceiling]
    return scored[:k]
