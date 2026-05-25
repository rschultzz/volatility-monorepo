---
type: cr
cr_id: CR-C
title: Probability Output on Proposals
aliases: ["CR-C — Probability Output on Proposals", "CR-C"]
status: active
started: 2026-05-25
sequence_number: 23
run_mode: interactive
phase: 1
size: medium
estimated_days: 3-4
data_safety_class: read_only
dependencies: [CR-A, CR-B]
depended_on_by: [CR-G]
branch_name: cr-c-probability-output
tags: [dash, cr, proposals, knn, probability, backend, frontend]
---

# CR-C — Probability Output on Proposals

## Goal

Surface structural probability — derived from analogue outcome aggregation — alongside every proposed trade. The first end-to-end manifestation of the operating framework: "what does base rate say about this setup?"

## Context

[[CR-A — Landscape Backfill]] expanded the corpus. [[CR-B — Outcome Computation]] populated outcomes. CR-C joins them: given today's structural read, find the K nearest analogues, aggregate their outcomes, output a probability with confidence interval. See [[Operating Framework — Where When Buy or Sell]] for the role of structural_prob.

This isn't the edge ratio yet (that lands with market-implied prob in [[CR-G — Edge Visualization and P&L Engine]]). It's just the structural side of the comparison — what the corpus says historically happens in setups that look like today.

## Step 0 — Diagnosis (no commits)

1. **Locate analogue ranking module.** Where in the codebase does today's feature vector get ranked against the corpus? Probably `Analogues/service.py` or similar. Understand its current interface.

2. **Decide K.** Default: K=20 nearest neighbors. Tradeoff: smaller K = more locally specific but noisier; larger K = smoother but pulls in less-similar setups. K=20 is a starting point; revisit if smoke shows odd behavior.

3. **Confidence interval method.** Wilson 95% CI on a proportion (Wilson is appropriate for small-n proportions where normal-approximation fails). Implement once in `packages/shared/stats.py`.

4. **What probability to surface for each regime:**
   - magnet-above / magnet-below: `reached_touch` rate among K analogues (does the move happen?)
   - magnetic-pin: `reached_touch` rate (does price come within tolerance of pin?)
   - bounded / amplification / untethered: NULL in v1 (outcomes are NULL too); show "no base rate available"

5. **Provenance.** Each probability output needs to show:
   - K (number of analogues)
   - Hit rate (touch_rate)
   - 95% CI
   - Note about whether reached_close is meaningfully different (if so, may indicate "touches but doesn't hold")

## Step 1 — Backend: probability aggregation module

**Commit:** `cr-c/step-1: implement analogue outcome aggregation`

In `packages/shared/probability.py`:

```python
def compute_structural_probability(
    today_features: dict,
    conn,
    k: int = 20,
    feature_version: str = 'v0.5.0-rebuilt',
) -> dict:
    """
    Given today's features, find K nearest analogues from the corpus
    that have computed outcomes, aggregate, return probability output.

    Returns:
        {
            'regime_kind': str,
            'k': int,
            'k_with_outcomes': int,  # may be less if some have outcome_status='na_regime'
            'touch_rate': float,     # proportion that hit drift_target
            'close_rate': float,     # proportion that closed near drift_target
            'touch_ci_lower': float, # Wilson 95% CI
            'touch_ci_upper': float,
            'mean_days_to_reach': float | None,
            'mean_excursion_pct': float,  # mean max_excursion / expected_move
            'note': str,  # provenance / caveats
        }
    """
```

Joins logic:
- Rank corpus by KNN distance from `today_features`
- Pull top K analogues
- LEFT JOIN against `bt_daily_outcomes_active` on PK
- Filter to `outcome_status = 'computed'`
- Aggregate

Unit tests with synthetic feature vectors and outcomes.

**Deliverable:** module callable; well-tested.

## Step 2 — Backend: extend proposals response

**Commit:** `cr-c/step-2: extend /api/proposals response with structural_probability`

In `Proposals/service.py` (or equivalent):

- After current proposal generation, call `compute_structural_probability(today_features, conn)`
- Attach result to response under `structural_probability` key
- Each proposal also gets a `directional_thesis` field that ties to the probability (e.g., proposal targeting drift_target → structural_probability.touch_rate is the relevant base rate)

JSON shape:

```json
{
  "proposals": [...],
  "structural_probability": {
    "regime_kind": "magnet-above",
    "k": 20,
    "k_with_outcomes": 18,
    "touch_rate": 0.61,
    "close_rate": 0.42,
    "touch_ci_lower": 0.39,
    "touch_ci_upper": 0.80,
    "mean_days_to_reach": 2.3,
    "mean_excursion_pct": 0.85,
    "note": "Based on 18 analogues with computed outcomes (2 had pending_history)."
  }
}
```

**Deliverable:** API returns structural_probability alongside proposals.

**Verification:** hit `/api/proposals` on a known day, confirm structural_probability key present and sensible.

## Step 3 — Frontend: render probability with provenance

**Commit:** `cr-c/step-3: render structural probability on proposal cards`

In the proposal card UI (`SavedScans.jsx`, `ProposalCard.jsx`, or wherever proposals render):

- Add a section near the top of each card: "Structural probability"
- Show:
  - Headline: `61% chance of reaching target (CI: 39%-80%)`
  - Subtext: `Based on 18 historical analogues. Average days to reach: 2.3.`
- Visual: a horizontal bar or gauge showing touch_rate within CI bounds. CI bounds shown as light bracket.
- Color coding by edge strength (placeholder until CR-G):
  - 65-100%: green
  - 45-65%: yellow
  - <45%: red
- Tooltip on hover: "K=20 nearest analogues by KNN distance. Hit rate = fraction where price touched drift_target within horizon."

**Deliverable:** probability visible on every proposal card.

## Step 4 — Edge cases and provenance polish

**Commit:** `cr-c/step-4: handle low-confidence and non-directional regimes`

- If `k_with_outcomes < 10`: show probability but with a warning "Low confidence — only N analogues have outcomes."
- If `regime_kind` is bounded / amplification / untethered: show "No base rate available for this regime in v1" instead of probability.
- If CI width > 30 percentage points: surface the wide CI prominently (the probability point estimate may be noisy).

**Deliverable:** graceful handling of edge cases.

## Smoke tests

1. **Sanity check.** Pick a recent magnet-above day. Verify structural_probability surfaces. Verify touch_rate is between 0.3 and 0.8 (sane range).

2. **Non-directional regime.** Pick a bounded-regime day. Verify "No base rate available" displays correctly.

3. **Low-corpus regime.** If any regime has <10 instances in corpus, verify warning displays.

4. **CI calculation.** Pick a known proportion (e.g., 12/20 = 0.60) and verify Wilson CI matches hand-calculated value (~0.39 to 0.78).

5. **Cross-page consistency.** Same day's probability surfaces same value across all UI views.

## Wrap criteria

- All 4 steps committed
- Smoke tests pass
- [[Roadmap]] updated: CR-C marked complete; foundation laid for CR-G to add market-implied side

## Status updates

(filled during execution)

## Open questions

- Should K vary by regime (e.g., K=30 for more-common regimes, K=10 for rare ones)? **Default K=20; revisit if smoke shows precision issues for rare regimes.**
- Should mean_days_to_reach be surfaced prominently or as detail? **Detail for now; promote to headline if user feedback suggests it matters.**
