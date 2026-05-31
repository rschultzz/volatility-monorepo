"""Versioned KNN ranking configuration (CR-029).

Parameters here are applied OUTSIDE the structural distance vector — after
similarity_distance() computes the σ-normalized Euclidean distance for each
candidate, and inside rank_analogues() before returning results.

  distance_ceiling  — hard upper bound on σ-normalized distance. Candidates
                      beyond this threshold are excluded regardless of K.
                      Makes K adaptive: the returned count reflects how common
                      the setup is. Rare setups get fewer analogues and wider
                      Wilson CIs — honest uncertainty rather than false precision.
                      math.inf = no ceiling (v0 / backward-compatible behavior).

Config changes are non-destructive and reproducible: the same anchor can be
re-ranked under different config versions for A/B comparison, exactly as the
feature_version pattern works for feature schemas.
"""
from __future__ import annotations

import math
from typing import Optional

KNN_CONFIGS: dict[str, dict] = {
    "v0": {"distance_ceiling": math.inf},   # legacy — no ceiling
    "v1": {"distance_ceiling": 4.0},        # CR-029: 4.0σ hard cutoff
}

CANONICAL_KNN_CONFIG_VERSION: str = "v1"


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
