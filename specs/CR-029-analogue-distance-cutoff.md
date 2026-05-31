# CR-029 — Analogue Distance Cutoff (CR-W)

**Branch:** `feat/CR-029-analogue-distance-cutoff`
**Phase:** 5 | **Size:** small | **Run mode:** interactive

## Goal

Install a hard distance ceiling on the KNN analogue ranking path:

1. Analogues beyond a threshold σ-distance are excluded — K becomes adaptive (count reflects how common the setup is; rare setups → wider CIs, not silent false precision).
2. Add `before_date` as-of-history guard to `rank_analogues` — callers running backtests pass the anchor date to exclude future candidates (required by CR-X coherence check).
3. Ceiling lives in a **versioned config** (`knn_config.py`), not hard-coded.
4. Applied outside the distance vector (after `similarity_distance`, inside `rank_analogues`).

No feature weights or recency decay — those are CR-X.

## Step 0 — Empirical findings

Scripts: `scripts/cr_w_step0_distance_distribution.py`, `scripts/cr_w_step0b_ceiling_analysis.py`

Corpus: 735 rows, K_MAX=20, feature_version=v0.5.0-rebuilt.

```
Nearest-neighbour distances: min=0.258 median=1.205 p90=2.979 p99=7.692 max=11.792
K=20th-neighbour distances:  min=1.200 median=3.750 p90=6.404 p95=7.688 max=23.705
All returned distances:       min=0.258 median=2.571 p90=5.537 p95=6.872 max=23.705

Ceiling sensitivity (% of anchors getting 0 analogues / median adaptive-K):
  2.0σ → 22.0% / K=4
  3.0σ →  9.7% / K=11
  3.5σ →  8.2% / K=16
  4.0σ →  7.3% / K=20   ← CHOSEN
  5.0σ →  4.4% / K=20
```

**Known problem case (2026-05-13, 0.54σ anchor):**
Full K=20 distances: `[1.29, 1.85, 2.32, 2.35, 3.92, 4.14, 4.18..4.98]`
Clear cluster gap between 2.35σ and 3.92σ.
At ceiling=4.0σ: 5 analogues retained (1.29, 1.85, 2.32, 2.35, 3.92); 15 outliers cut.

**Ceiling = 4.0σ (v1 config).** Rationale:
- 7.3% corpus impact (acceptable; these are genuinely rare setups)
- Median adaptive-K = 20 (non-regressive for common setups)
- Eliminates the 4.14–4.98σ outlier cluster in the known bad case
- Further refinement via CR-X feature weighting (esp. `cluster_1_signed_distance_sigma`)

## Step 1 — versioned config + rank_analogues update

**Commit:** `cr-w/step-1: knn_config + before_date + distance ceiling in rank_analogues`

### `packages/shared/knn_config.py` (new)

```python
"""Versioned KNN ranking configuration (CR-029).

Parameters applied OUTSIDE the structural distance vector:
  distance_ceiling  — hard σ-distance upper bound; analogues beyond excluded.
                      math.inf = no ceiling (v0 / backward-compat behavior).
"""
import math

KNN_CONFIGS: dict = {
    "v0": {"distance_ceiling": math.inf},
    "v1": {"distance_ceiling": 4.0},
}
CANONICAL_KNN_CONFIG_VERSION: str = "v1"

def get_knn_config(version=None) -> dict:
    v = version or CANONICAL_KNN_CONFIG_VERSION
    cfg = KNN_CONFIGS.get(v)
    if cfg is None:
        raise ValueError(f"Unknown knn_config_version: {v!r}")
    return cfg
```

### `packages/shared/knn.py` — `rank_analogues` signature change

Add parameters:
- `before_date: Optional[str] = None` — exclude candidates where `trade_date >= before_date`
- `distance_ceiling: float = math.inf` — hard σ cutoff after sorting

Application order:
1. Exclude `exclude_date`
2. Exclude candidates where `d >= before_date` (if set)
3. Compute `feature_stats` from the remaining pool (if not pre-supplied)
4. Score and sort
5. Filter to `dist <= distance_ceiling` (adaptive K)
6. Return `scored[:k]`

## Step 2 — wire through call sites

**Commit:** `cr-w/step-2: wire distance ceiling + before_date through call sites`

- `packages/shared/probability.py` — `_rank_analogues_with_outcomes` accepts `before_date=None`; passes it + ceiling from `get_knn_config()` to `rank_analogues`.
- `apps/web/modules/Analogues/routes.py` — import `get_knn_config`, `CANONICAL_KNN_CONFIG_VERSION`; accept optional `knn_config_version` query param; pass `distance_ceiling` to `rank_analogues` call.

## Step 3 — tests

**Commit:** `cr-w/step-3: tests for distance ceiling, before_date, adaptive K`

New: `packages/shared/tests/test_knn.py`

Smoke tests:
1. `before_date` excludes future candidates.
2. `before_date=None` → current behavior (no filter).
3. Distance ceiling excludes far candidates.
4. Adaptive K: fewer returned than K when ceiling is tight.
5. `distance_ceiling=inf` → backward compat, all K returned.
6. `before_date` applies before `feature_stats` computation when stats=None.
7. `exclude_date` + `before_date` both apply.

## Step 4 — empirical sanity pass (no commit)

For 2026-05-13: confirm analogue count drops from 20 → ≤5 via the probability module; all distances ≤ 4.0σ; Wilson CI visibly widens. Record findings in session note.

## Wrap criteria

- Steps 1–3 committed; Step 4 findings in vault session note.
- `test_knn.py` passes; full existing suite unaffected.
- `before_date` tested and ready for CR-X.
- Literal `4.0` nowhere outside `knn_config.py`.

## Out of scope

- Feature weights — CR-X
- Recency decay — CR-X
- Stats window localization — CR-X decision
- UI changes for 0-analogue state — deferred
