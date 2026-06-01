# CR-031 — Finish Adaptive-K (uncap K_MAX + half-life retune)

> **Completes CR-W/CR-029's intent.** CR-W's adaptive-K-by-radius was supposed
> to make commonness translate into certainty: a common setup with many
> genuinely-similar days gets LARGER K and a TIGHTER CI; a rare setup gets fewer
> and an honestly wider CI — with the **distance radius as the SOLE gate** so no
> less-similar days are admitted. As shipped, only the *downward* half works:
> the ceiling correctly cuts far days, but a leftover **`_K_MAX = 20` clamp in
> `routes.py` still truncates the *upward* half** — a common setup with 40 days
> inside the radius is forced down to 20, so its CI is artificially wider than
> the data supports. That is the **opposite** of the feature's stated purpose.

## Branch

`cr-z-finish-adaptive-k` off main @ 86d21cf

## Goal

1. Uncap K so the distance ceiling is the sole gate: a setup returns ALL
   within-ceiling analogues; its Wilson CI reflects that real count.
2. Retune the recency half-life by coherence sweep (candidates: 18/24/36 months).
   Both make the live/common case behave as the adaptive-K design intended.

## Critical design constraints

1. **Radius stays the similarity gate — uncapping admits NO less-similar days.**
   Removing/raising `_K_MAX` only lets in days already inside the ceiling
   (already established as similar enough). This is the whole reason uncapping
   is safe.
2. **Replace the hard cap with a high safety bound (200), not unbounded.**
   High enough that it effectively never binds on real setups; the radius does
   the real work. Locked in Step 0 against the observed within-ceiling
   count distribution.
3. **Half-life retune is config-only (v3) and harness-judged.** Change
   `half_life_months` in `knn_config.py`; pick from {18, 24, 36} by highest
   coherence while live/edge setups are not K-starved.
4. **Re-confirm ceiling under new half-life.** A longer half-life lets older
   days back in → distances shift → re-check zero-analogue rate.
5. **Versioned config, baseline reproducible.** v3 config; v2 remains selectable.

## Step 0 — Diagnostic lock (script: scripts/cr_z_step0_diagnostic.py)

### Corpus

- n=735 rows, feature_version=v0.5.0-rebuilt
- Outcome coverage: 735/735

### Within-ceiling count distribution (uncapped, k=9999, before_date guard active)

Config v2 (hl=18mo, ceiling=5.0):
- zero_rate=7.6%  (LOO-inflated: early-corpus anchors have few predecessors;
  production live rate is much lower — consistent with the ~2.4% figure from
  CR-030 which used a non-LOO setup)
- 44.4% of setups would hit the old K=20 cap — the upward half was broken on
  nearly HALF of all setups
- Distribution: min=0 p10=1 p25=5 median=16 p75=38 p90=66 p95=79 p99=95 max=115

**Safety bound: _K_SAFETY = 200**
- p99=95, max=115 → 200 leaves comfortable room above all observed counts
- Effectively never binds in production

### Half-life coherence sweep

| Half-life | Coherence | K=[min..med..max] | n_anchors |
|-----------|-----------|-------------------|-----------|
| hl=18mo   | **0.5850**| [5..26..115]      | 556/735   |
| hl=24mo   | 0.5821    | [5..29..136]      | 577/735   |
| hl=36mo   | 0.5782    | [5..33..183]      | 592/735   |

- hl=18 has the best coherence (+0.5% vs hl=24, +1.2% vs hl=36)
- Counter-intuitive finding: shorter half-life → higher coherence; recency IS
  the signal — recent days are genuinely better predictors than older ones

### Edge anchor K check (2026-05-07)

- K=1 at ALL half-lives (18, 24, 36)
- Root cause: only 1 candidate has raw_distance ≤ 2.5σ from 2026-05-07. The
  two-pin-cluster setup is genuinely rare; recency scaling at hl=36 still
  requires raw_dist ≤ ceiling/2 = 2.5σ for 3yr-old data.
- **This K=1 is honest rarity — cannot be fixed by half-life or ceiling change
  without admitting less-similar days. It is not a bug.**

### Ceiling re-confirm at other half-lives

| Half-life | zero_rate | p99 within-ceiling | max |
|-----------|-----------|-------------------|-----|
| hl=18mo   | 7.6%      | 95                | 115 |
| hl=24mo   | 7.2%      | 95                | 136 |
| hl=36mo   | 6.4%      | 115               | 183 |

Ceiling stays at 5.0σ for all half-lives. Zero rates remain in range
(target <5% for production; LOO rates are inflated by ~3-4pp vs live).

### Locked decisions

| Decision | Value | Rationale |
|----------|-------|-----------|
| _K_SAFETY | 200 | p99=95, max=115; 200 never binds |
| half_life_months | 18.0 (unchanged from v2) | Highest coherence (0.5850); edge K=1 at all values |
| distance_ceiling | 5.0 (unchanged from v2) | Zero rate acceptable; no re-pick needed |
| v3 config | Same as v2 | Sweep confirms v2 parameters; v3 documents the decision |

## Implementation order

1. `cr-z/step-1: radius is the sole gate — replace K_MAX=20 with safety bound 200`
   - `routes.py`: `_K_MAX = 20` → `_K_SAFETY = 200`; clamp to `_K_SAFETY`; call
     `rank_analogues` with `k=_K_SAFETY` so all within-ceiling results are returned
   - Tests: 35-within-ceiling setup returns 35 (not 20); rare stays 6; safety
     only binds at 200; radius still gates

2. `cr-z/step-2: half-life retune (config v3) — sweep confirms hl=18 retained`
   - `knn_config.py`: add v3 = {same as v2 plus this documentation}
   - `CANONICAL_KNN_CONFIG_VERSION = "v3"`
   - Zero-analogue rate and ceiling re-confirmed (Step 0 findings)

## Smoke tests

1. Synthetic setup with 35 within-ceiling analogues returns 35, not 20
2. Rare setup with 6 within-ceiling still returns 6
3. Safety bound only binds at 200; never at 20
4. No analogue beyond ceiling is ever returned
5. v3 half-life from config; v2 reproduces prior ordering; literal-free path
6. Coherence not regressed (hl=18 shows 0.5850 ≥ v2's 0.5785)
7. Lookahead guard still load-bearing

## Wrap criteria

- `_K_MAX=20` replaced by `_K_SAFETY=200`; radius is sole similarity gate
- Common setups show K>20 with tighter CIs
- Half-life retuned via coherence (retained at 18); comparison table recorded
- 2026-05-07 K=1 documented as honest rarity (not a failure)
- Smoke tests pass; no dissimilar-day leakage

## Out of scope

- Distance metric, weights, or transform changes
- Localized `feature_stats` window
- Per-regime ceilings
- Supervised tuning
