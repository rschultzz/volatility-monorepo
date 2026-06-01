"""Leave-one-out coherence check for KNN analogue configs (CR-030 Step 2).

Measures whether a KNN config's analogues actually share similar outcomes
with the anchor day — the quantitative acceptance signal for CR-X tuning.
No trade labels needed; uses EOD return and intraday range from bars.

Public entry point:
    run_coherence_check(conn, candidates, outcomes_by_date, config,
                        config_label, *, min_k, anchor_date_iso)
        → CoherenceResult

    compare_configs(conn, candidates, outcomes_by_date,
                    configs, *, min_k)
        → dict[label, CoherenceResult]

Coherence definition
────────────────────
For each anchor day where config returns K ≥ min_k analogues:
  - Get neighbour outcome distribution for each dimension
    (eod_return_pts, intraday_range_pts)
  - Resemblance = max(0, 1 − |anchor_outcome − mean(neighbours)| / std(neighbours))
    Bounded [0, 1]; 1.0 = anchor exactly matches mean; 0.0 = ≥2σ off.
  - Per-anchor score = mean(resemblance across dimensions with ≥ min_k outcomes)
  - Overall coherence = mean(per-anchor score) across all scored anchors

As-of-history safety: `before_date=anchor_date_iso` is passed to rank_analogues
for every anchor. The smoke-test helper `_coherence_with_lookahead` deliberately
omits it to verify the guard raises the score (demonstrating lookahead exists).
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Optional

from packages.shared.knn import feature_stats, rank_analogues


_OUTCOME_DIMS = ("eod_return_pts", "intraday_range_pts")

# Score = max(0, 1 - |z| / Z_SCALE). z=0 → 1.0; z=Z_SCALE → 0.0.
_Z_SCALE = 2.0


@dataclass
class CoherenceResult:
    label: str
    n_anchors_scored: int          # anchors where K≥min_k and outcome available
    n_anchors_total: int           # all corpus anchors attempted
    coherence: Optional[float]     # mean resemblance (None if no anchors scored)
    dim_scores: dict[str, float]   # per-dimension mean resemblance
    regime_scores: dict[str, float]  # per-regime mean resemblance
    k_stats: dict[str, float]      # min/median/max K returned per anchor

    def summary(self) -> str:
        if self.coherence is None:
            return f"{self.label}: no scored anchors"
        ks = self.k_stats
        return (
            f"{self.label}: coherence={self.coherence:.4f} "
            f"(n={self.n_anchors_scored}/{self.n_anchors_total}) "
            f"K=[{ks['min']:.0f}..{ks['median']:.0f}..{ks['max']:.0f}] "
            f"dims={self.dim_scores}"
        )


def _resemblance(anchor_val: float, neighbour_vals: list[float]) -> Optional[float]:
    """Resemblance of anchor_val against neighbour_vals distribution.

    Returns None if fewer than 2 neighbours have the outcome.
    Score: max(0, 1 − |z| / Z_SCALE) where z = |anchor − mean| / (std + ε).
    """
    valid = [v for v in neighbour_vals if v is not None and math.isfinite(v)]
    if len(valid) < 2:
        return None
    mean_n = statistics.mean(valid)
    std_n = statistics.stdev(valid) if len(valid) >= 2 else 0.0
    if std_n < 1e-9:
        # Degenerate: all neighbours identical — perfect if anchor also matches
        return 1.0 if abs(anchor_val - mean_n) < 1e-6 else 0.0
    z = abs(anchor_val - mean_n) / std_n
    return max(0.0, 1.0 - z / _Z_SCALE)


def run_coherence_check(
    candidates: list[tuple],      # [(date_iso, feature_vec), ...]
    outcomes_by_date: dict,       # {date_iso: {eod_return_pts, intraday_range_pts, ...}}
    regime_by_date: dict,         # {date_iso: regime_str}
    config: dict,                 # knn_config dict (v1, v2, ...)
    config_label: str,
    *,
    min_k: int = 5,
    lookahead: bool = False,      # if True, omit before_date (smoke-test only!)
    k: int = 20,
) -> CoherenceResult:
    """Run leave-one-out coherence check for a single config.

    Each anchor day uses only candidates strictly before it (before_date guard)
    unless lookahead=True (for the smoke test that demonstrates the guard matters).
    """
    global_stats = feature_stats(v for (_, v) in candidates)
    anchor_dates_with_outcomes = [
        d for (d, _) in candidates
        if d in outcomes_by_date and any(
            outcomes_by_date[d].get(dim) is not None
            for dim in _OUTCOME_DIMS
        )
    ]
    n_total = len(anchor_dates_with_outcomes)

    per_anchor_scores: list[float] = []
    per_dim_scores: dict[str, list[float]] = {dim: [] for dim in _OUTCOME_DIMS}
    per_regime_scores: dict[str, list[float]] = {}
    k_vals: list[int] = []

    # Pre-build vec lookup
    vec_by_date = {d: v for (d, v) in candidates}

    for anchor_date in anchor_dates_with_outcomes:
        anchor_vec = vec_by_date[anchor_date]
        anchor_outcomes = outcomes_by_date[anchor_date]

        ranked = rank_analogues(
            anchor_vec, candidates, k,
            exclude_date=anchor_date,
            before_date=None if lookahead else anchor_date,
            stats=global_stats,
            distance_ceiling=config.get("distance_ceiling", math.inf),
            feature_weights=config.get("feature_weights"),
            z_diff_cap=config.get("z_diff_cap"),
            half_life_months=config.get("half_life_months"),
            anchor_date=anchor_date,
        )
        if len(ranked) < min_k:
            continue
        k_vals.append(len(ranked))

        dim_scores_this_anchor = []
        for dim in _OUTCOME_DIMS:
            anchor_val = anchor_outcomes.get(dim)
            if anchor_val is None or not math.isfinite(anchor_val):
                continue
            neighbour_vals = [
                outcomes_by_date.get(d, {}).get(dim)
                for (d, _) in ranked
            ]
            r = _resemblance(anchor_val, neighbour_vals)
            if r is not None:
                per_dim_scores[dim].append(r)
                dim_scores_this_anchor.append(r)

        if not dim_scores_this_anchor:
            continue

        score = statistics.mean(dim_scores_this_anchor)
        per_anchor_scores.append(score)

        # Regime breakdown
        regime = regime_by_date.get(anchor_date, "unknown")
        per_regime_scores.setdefault(regime, []).append(score)

    overall = statistics.mean(per_anchor_scores) if per_anchor_scores else None
    dim_agg = {
        dim: round(statistics.mean(scores), 4) if scores else 0.0
        for dim, scores in per_dim_scores.items()
    }
    regime_agg = {
        r: round(statistics.mean(s), 4)
        for r, s in sorted(per_regime_scores.items())
    }
    k_stats: dict[str, float] = {}
    if k_vals:
        k_vals_sorted = sorted(k_vals)
        k_stats = {
            "min": float(min(k_vals)),
            "median": float(k_vals_sorted[len(k_vals_sorted) // 2]),
            "max": float(max(k_vals)),
        }

    return CoherenceResult(
        label=config_label,
        n_anchors_scored=len(per_anchor_scores),
        n_anchors_total=n_total,
        coherence=round(overall, 4) if overall is not None else None,
        dim_scores=dim_agg,
        regime_scores=regime_agg,
        k_stats=k_stats,
    )


def compare_configs(
    candidates: list[tuple],
    outcomes_by_date: dict,
    regime_by_date: dict,
    configs: list[tuple[str, dict]],  # [(label, config_dict), ...]
    *,
    min_k: int = 5,
    k: int = 20,
) -> dict[str, CoherenceResult]:
    """Run coherence check for multiple configs and return results keyed by label."""
    return {
        label: run_coherence_check(
            candidates, outcomes_by_date, regime_by_date, cfg, label,
            min_k=min_k, k=k,
        )
        for (label, cfg) in configs
    }
