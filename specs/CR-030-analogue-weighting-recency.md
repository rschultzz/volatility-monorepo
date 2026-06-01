# CR-030 — Analogue Weighting and Recency (CR-X)

**Branch:** `cr-x-analogue-weighting-recency`
**Phase:** 5 | **Size:** medium | **Run mode:** interactive
**Depends on:** CR-029 (knn_config.py, 4.0σ ceiling, before_date guard — merged)

> [!info] What this CR delivers
> The second analogue-quality CR. CR-029 (CR-W) stopped absurdly-far days from counting; CR-X
> improves *which* close days count and *how much*. Three moves, each with a sensible default
> (NO blind hand-tuning), plus the tool that judges whether the defaults work:
> 1. **Feature weighting** — stop treating every feature as equally important. Up-weight what
>    matters (distance-from-magnet, regime); down-weight noise. First-guess set, not optimized.
> 2. **Recency decay** — gentle preference for recent days (~year-old day counts ~half), applied
>    outside the vector as a distance multiplier.
> 3. **Leave-one-out coherence check** — for each historical day (whose actual outcome is known),
>    do the config's analogues share that day's outcome? One score that says whether matching
>    finds genuinely similar days vs noise — tuning watches a number, not guesses.
>
> Philosophy: ship reasonable defaults, let the coherence score + a few real setups judge them,
> nudge once or twice. NO supervised optimization (no trade labels yet; would fit noise).

## Pre-Step-0 Evidence (2026-05-31 diagnostic, via CR-029 v0 ranking)

The 2026-05-07 zero-analogue pin day was diagnosed (script:
`scripts/cr_x_step0_may07_diagnostic.py`): nearest 20 days at **11–19σ** (honest zero, nothing
near the 4σ line), but distance is **artificially inflated by a near-zero-mean categorical**.

- `cluster_2_quality_ordinal` (value 2 on a rare two-pin-cluster day) z-scores to **~16** (corpus
  mean ~0), contributing **~60% of squared distance alone**; correlated `n_pin` adds ~10%.
- **Key insight: weighting alone won't fix this** — a 16σ axis dominates within any sane weight
  budget. CR-X must address *how rare categoricals contribute to distance* (z_diff cap), not just
  multiply weights. See Step 0 #1a.
- Full detail: open-questions/knn-feature-weighting-and-recency-decay.md "Live observation
  2026-05-31".

## Goal

1. **Feature weights** in the versioned config (extend CR-029's `knn_config.py`) — first-pass
   expert priors, applied as per-feature multipliers *outside* `similarity_distance`'s vector.
2. **Recency decay** — Option A (decay on neighbors only): scale each candidate's distance by
   an age factor; half-life framing; λ=0 reproduces current behavior.
3. **Leave-one-out coherence check** — offline harness: anchor each historical day (candidates
   restricted *before* it via CR-029's `before_date`), pick analogues, measure whether neighbor
   outcomes (EOD return / range from ES bars; NO trade label) resemble the anchor's actual
   outcome. One coherence scalar per config.
4. **Redundancy + rare-categorical handling** before trusting weights (named culprits from
   the diagnostic).

## Critical Design Constraints

1. **All tuning in the versioned config, applied OUTSIDE the vector.** Weights = per-feature
   multipliers in the z-score difference term (`weight × (z_q − z_c)²`); recency = post-distance
   age scaling. Extend `knn_config.py`; don't restructure. Do NOT add time as a feature.

2. **As-of-history correctness is NON-NEGOTIABLE for the coherence check.** Pass `rank_analogues`
   only candidates dated strictly before the anchor (use CR-029's `before_date`). Lookahead
   inflates the score and validates bad tuning. Smoke-test that a leaking variant fails.

3. **No supervised tuning. Expert priors only, judged by coherence.** `bt_signals` is
   label-empty. Priors: regime > proximity/cluster topology > bucket dominance > neg-zone trivia.
   ONLY acceptance signal is the coherence score.

4. **Redundancy AND rare-categorical blowup, before trusting weights:**
   - **(a) Correlation** — named: `cluster_2_quality_ordinal` + `n_pin`; also
     `is_magnet_day`/`magnet_direction_signed`, the `n_pin`/`n_target`/`n_feature` cluster.
   - **(b) z-score blowup on near-zero-mean categoricals** — a feature with corpus mean ~0 turns
     any nonzero value into a huge z-score (value 2 → z≈16) that dominates regardless of weight.
     **Weighting cannot fix this.** Fix: per-feature z_diff cap. Lock in Step 0 #1a.

5. **Recency gentle and reversible.** Half-life in months, config-driven; λ=0 reproduces current
   ordering. Do NOT localize `feature_stats` window here (Option B).

6. **Layer on CR-029; re-pick the ceiling.** Weighting + the categorical transform shift the
   distance distribution, so the 4.0σ ceiling almost certainly needs re-picking. Re-run the
   corpus count distribution post-reweighting; re-pick `distance_ceiling` in config.

## Step 0 — Diagnosis and design lock (no commits)

1. **Redundancy audit.** Correlation pass over `FEATURE_NAMES`; confirm the
   `cluster_2_quality_ordinal`+`n_pin` pair and other clusters; decide prune/merge/leave per
   cluster.
1a. **Rare-categorical transform (load-bearing 2026-05-07 fix).** Decide how near-zero-mean
    ordinal/categorical features contribute to distance. Decision: **per-feature z_diff cap** at
    `z_diff_cap = 3.0σ`. Applied as `weight × min(|z_q − z_c|, z_diff_cap)²` in the weighted
    distance summand. Config-driven. Baseline: `z_diff_cap=None` → no cap. Validate against
    2026-05-07 anchor: confirm `cluster_2_quality_ordinal` no longer dominates (~60% → <<60%).
2. **Expert-prior weights.** Up-weight `cluster_1_signed_distance_sigma` (spot-to-wall
   proximity) + regime flags + implied_move; down-weight quality ordinals for slots 2/3,
   `n_pin`/`n_target`/`n_feature`. First-pass, not optimized.
3. **Recency form + half-life.** Form: `effective_distance = distance × 2^(years_ago /
   half_life_years)`. Half-life: 18 months (1.5 years) first pass. λ=0 (half_life=∞) reproduces
   current ordering.
4. **Coherence design.** Outcome measures: EOD return (pts), intraday range (pts) from
   `ironbeam_es_1m_bars`. Resemblance score per anchor: z-score of anchor_outcome in analogue
   distribution, bounded [0,1]. Aggregated coherence = mean(resemblance) over all anchors with
   K≥5. `before_date=anchor_date_iso` for every anchor. Smoke test that omitting before_date
   produces a higher (spurious) score (demonstrating lookahead exists without the guard).
5. **Re-pick ceiling.** After #1a+#2, run corpus count distribution; pick ceiling such that
   zero-analogue rate <5% (down from 7.3%) and median K≥15. Confirm 2026-05-07 finds analogues.

**Step 0 gate: all five points locked before any implementation commit.**

## Step 1 — Config + weighted distance + recency + transform

**Commit:** `cr-x/step-1: weighted distance + recency + z_diff cap (config-driven, outside the vector)`

Files:
- **`packages/shared/knn_config.py`** — new v2 config: `feature_weights` dict, `z_diff_cap`,
  `half_life_months`, re-picked `distance_ceiling`. `CANONICAL_KNN_CONFIG_VERSION = "v2"`.
- **`packages/shared/knn.py`** — add `weighted_similarity_distance(query, candidate, stats, *,
  feature_weights, z_diff_cap)`. Update `rank_analogues` with new params: `feature_weights=None`,
  `z_diff_cap=None`, `half_life_months=None`, `anchor_date=None`. When feature_weights provided:
  use weighted distance; when half_life_months + anchor_date provided: apply recency scaling after
  distance computation.
- **`apps/web/modules/Analogues/routes.py`** — thread new knn_cfg keys through to `rank_analogues`.
- **`packages/shared/probability.py`** — thread new knn_cfg keys through.
- **`packages/shared/tests/test_knn.py`** — extend with smoke tests below.

Key invariants:
- `feature_weights=None` → uses existing `similarity_distance` (exact baseline, no reordering).
- `feature_weights={}` / all weights 1.0, `z_diff_cap=None` → identical to baseline (test this).
- Recency: `anchor_date` required when `half_life_months` set; callers pass it as ISO string.
- Recency applied after distance_ceiling filter? No — apply recency BEFORE ceiling so the ceiling
  is on effective (age-adjusted) distance, not raw structural distance.
- `feature_distance_breakdown` in service.py: update to use weights+cap when config provides them,
  so the UI breakdown reflects the weighted decomposition.

## Step 2 — Leave-one-out coherence harness (offline)

**Commit:** `cr-x/step-2: leave-one-out coherence check (as-of-history safe)`

New file: `scripts/cr_x_coherence_check.py`
Optional reusable fn: `packages/shared/knn_coherence.py`

Logic:
- Load all SPX corpus days + ES bar outcomes from `ironbeam_es_1m_bars`.
- For each anchor day: rank with `before_date=anchor_date_iso`, get top-K.
- Outcome resemblance: for EOD return and intraday range, compute z-score of anchor outcome in
  neighbour distribution → resemblance = max(0, 1 - |z| / 2).
- Overall coherence = mean(resemblance) over anchors with K≥5.
- Report baseline (v1) vs weighted (v2). Regime-stratified breakdown.
- Explicit smoke test: with `before_date=None`, a future date appears in neighbors (demonstrates
  lookahead), confirming that the guarded v2 run is meaningfully different.

## Step 3 — Apply, observe, nudge

**Commit:** `cr-x/step-3: first-pass config selected via coherence`

- Run harness: baseline (v1) vs Step-0 config (v2). Keep if coherence improves; else nudge
  weights/half-life once or twice, evidence-led. Lock v2 as canonical. Record numbers.
- Re-verify 2026-05-07 pin-day case and 2026-05-13 magnet-above case (the CR-029 problem case).
- Update `CANONICAL_KNN_CONFIG_VERSION = "v2"` if v2 beats v1.

## Step 4 — Empirical sanity (no code)

- Re-run 2026-05-07 under v2 (finds pin analogues?).
- Re-run 2026-05-13 magnet-above (CR-029 problem case still bounded).
- Confirm zero-analogue rate dropped from 7.3%.
- Spot-check 2–3 live setups; confirm analogue list looks right.
- Record numbers in session note status updates.

## Smoke Tests (package in test_knn.py)

1. **Baseline reproducibility** — `feature_weights=None` → ordering identical to CR-029 v1.
2. **Weight effect** — up-weighting `cluster_1_signed_distance_sigma` pushes days with different
   signed-distances down.
3. **Recency effect** — older candidate ranks below equally-similar newer one when half_life set.
4. **z_diff cap (rare-categorical regression)** — synthetic candidate where one feature has
   z_diff=8.0; without cap it dominates; with cap=3.0, contribution is bounded at 9×weight.
5. **2026-05-07 regression** — post-transform, `cluster_2_quality_ordinal` no longer contributes
   >50% of squared distance for this anchor; the day finds ≥1 analogue under the new ceiling.
6. **Config-driven** — all weights/cap/half_life/ceiling from config; no literals in the path.
7. **As-of-history (CRITICAL)** — leaking variant (no before_date) scores higher than guarded.
8. **Ceiling re-picked & sane** — new ceiling in config; zero-analogue rate < 7.3%.

## Out of Scope

- Supervised optimization (no labels).
- Localizing `feature_stats` to a trailing window (Option B).
- Changing `similarity_distance`'s core metric (the transform is a config-driven layer, not a
  core rewrite).
- Regime classification changes (`regime-classification-may-be-distance-blind`).
- Multi-DTE horizon build.

## Step 0 Findings (locked 2026-05-31)

Script: `scripts/cr_x_step0_analysis.py` + `scripts/cr_x_step0_may07_diagnostic.py`.
Corpus: 735 rows, `feature_version=v0.5.0-rebuilt`.

### Lock 1a — z_diff cap validates the 2026-05-07 fix ✓

`cluster_2_quality_ordinal` blowup: **60.0% → 17.5%** of squared distance for nearest
neighbor. Distance for 2026-05-07 → nearest: **11.47σ → 3.875σ** (3× compression). The
nearest 10 neighbors are all `magnetic-pin` (vs magnet-above top-ranked days under equal
weights). Transform confirmed working.

Under v2 (no ceiling), `cluster_2_quality_ordinal` still shows at 17.5% with z_diff_raw=16.4
but the cap+weight keeps its effective contribution to `0.25 × 9 = 2.25` vs the 67.2 it
contributed under equal-weight no-cap. Top contributor is now `cluster_1_signed_distance_sigma`
(21.7%) — the actual proximity-to-wall signal. ✓

### Lock 1 — Redundancy handled via weights ✓

Named down-weighting:
- `cluster_2_quality_ordinal`, `cluster_3_quality_ordinal`: 0.25 (was 1.0)
- `n_pin`, `n_target`, `n_feature`: 0.25 (was 1.0)
Full weight table in knn_config.py v2 (see Step 1 commit).

### Lock 2 — Expert-prior weights confirmed ✓

v2 weight table (in `PROPOSED_WEIGHTS` in the step0 script) produces sensible per-feature
breakdowns: `cluster_1_signed_distance_sigma` leads at w=3.0; regime flags at 2.5; vol at 2.0.

### Lock 3 — Recency: half_life_months=18.0 ✓

Formula: `effective_distance = distance × 2^(years_ago / 1.5)`.
Chosen by design (1-year-old day counts 2× harder; 3-year-old day (2023) counts 8× harder
for identical structural match). Config param: `half_life_months`. Zero/None → no decay.

### Lock 4 — Coherence design ✓

EOD return pts + intraday range pts from `ironbeam_es_1m_bars`.
Resemblance score per anchor: `max(0, 1 - |z_anchor_in_neighbour_dist| / 2)`.
Aggregated coherence = mean(score) over anchors with K≥5. `before_date=anchor_date` required.
Smoke test: omitting before_date raises score (lookahead demonstrated).

### Lock 5 — Ceiling: 5.0σ ✓

v2 corpus stats (no ceiling):
- NN distance: min=0.273σ  median=1.217σ  p90=3.008σ  p99=5.251σ  max=7.063σ
- K=20th neighbour: median=3.178σ  p90=5.487σ  max=8.764σ

Zero-analogue rates at candidate ceilings:
- 4.0σ → 4.5% zero (33 anchors), median K=20
- **5.0σ → 1.6% zero (12 anchors), median K=20  ← chosen**
- 6.0σ → 0.4% zero (3 anchors), median K=20

**Ceiling = 5.0σ** (zero-analogue rate 1.6%, down from 7.3% under v1; median K=20 unchanged).

### Spot-check: three anchors v1 vs v2

| Anchor | Regime | v1 K | v1 dists | v2 K | v2 dists | v2 regimes |
|--------|--------|------|----------|------|----------|------------|
| 2026-05-07 | magnetic-pin | 0 | — | 19 | 3.88–4.97σ | 19× pin |
| 2026-05-13 | magnet-above | 5 | 1.29–3.92σ | 20 | 1.32–3.85σ | 20× magnet-above |
| 2024-11-04 | amplification | 20 | 1.60–2.97σ | 20 | 1.93–2.85σ | 20× amplification |
| 2026-04-27 | untethered | 6 | 1.70–3.99σ | 10 | 1.69–4.99σ | 10× untethered |

Key: 2026-05-13 goes from K=5 to K=20 under v2 — the previously-outlier 4.14–4.98σ cluster
under equal-weighting moves inside 3.85σ after reweighting (those days' structural similarity
on regime/proximity features now dominates, bucket-dominance trivia no longer inflates). No
regression on typical cases.

**All five Step-0 locks confirmed. Proceeding to Step 1.**

## Step 2 Coherence Results (locked 2026-05-31)

Script: `scripts/cr_x_coherence_check.py`. Corpus: 735 rows with 735/735 outcome coverage.

| Config | Coherence | Anchors scored | Notes |
|--------|-----------|---------------|-------|
| v1 (ceiling-only baseline) | **0.5527** | 473 / 735 | |
| **v2 (weights+cap+recency)** | **0.5785** | 556 / 735 | **+4.7%** |

v2 per-regime: magnetic-pin 0.6004, magnet-above 0.5779, amplification 0.5713,
untethered 0.5764, bounded **0.5399** (was 0.2056 under v1 — biggest gain).

**Lookahead smoke test**: guarded (before_date=anchor) coherence=0.5785 vs
lookahead (before_date=None) coherence=0.6313. Guard is load-bearing (+7.3pp
inflation without it). ✓ as-of-history correctness confirmed.

**Verdict: v2 BEATS v1. Config selected as canonical.**

## Step 3 Sanity + Step 4 Empirical (2026-05-31)

Script: `scripts/cr_x_step3_sanity.py`.

### 2026-05-07 (magnetic-pin, two-pin-cluster — the regression test)

- v1: K=0 (the pre-fix state)
- v2: K=1, nearest=4.667σ → 2026-04-17 (magnetic-pin) ✓
- `cluster_2_quality_ordinal` contribution: **12.7%** (was ~60% under v0) ✓ blowup fixed
- Note: recency (18mo half-life) penalizes all 2023-2024 structurally-similar pin days
  by 3-4× effective distance from a May 2026 vantage → only 2026-04-17 (3 weeks ago)
  passes the 5σ effective ceiling. K=1 is thin but better than K=0. Flag as a
  "nudge" candidate if live observation shows this pattern persisting.

### 2026-05-13 (magnet-above, original CR-029 problem case — no regression)

- v1: K=5, dists=[1.29..3.92]σ, all magnet-above
- v2: K=15, dists=[3.36..4.99]σ, all magnet-above ✓ improved, no regression

### Corpus-wide K distribution under v2

- Zero-analogue: **18/735 = 2.4%** (down from 7.3% under v1) ✓ criterion met (<5%)
- Median K: 20 ✓
- K=20 (full): 568/735 = 77.3%
- K≥10: 629/735 = 85.6%

### Recency nuance (flagged for next session)

18-month half-life is aggressive against the current 3-year corpus. For live anchors
in 2026, data from 2023-2024 (2-3yr old) gets 2.5-4× effective-distance multiplier,
pushing many structurally-close analogues beyond the 5σ ceiling. The coherence check
validates recency helps overall (uses before_date so historical anchors see contemporary
corpus — modest multipliers). For live near-future use, consider nudging half_life_months
to 36 if K consistently <5 for new live setups. Deferred to post-deployment observation.

### Wrap check

- ✓ Steps 1–3 committed; Step 4 findings logged here
- ✓ Weighting + recency + z_diff cap live, config-driven, external to vector, clean baseline
- ✓ Coherence harness exists, as-of-history safe (smoke-tested), v2 beats v1 (+4.7%)
- ✓ v2 config locked as canonical (`CANONICAL_KNN_CONFIG_VERSION = "v2"`)
- ✓ 2026-05-07 regression fixed (K=0 → K=1, blowup 60%→12.7%)
- ✓ Corpus zero-analogue rate dropped (7.3% → 2.4%)
- ✓ Ceiling re-picked (4.0σ → 5.0σ, 1.6% zero without recency, 2.4% with recency)
- → knn-feature-weighting-and-recency-decay open-question needs status update
- → Supervised tuning deferred (no labels); Option B (localized stats window) deferred

## Status Updates

(filled during execution)
