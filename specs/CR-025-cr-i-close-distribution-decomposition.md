---
type: cr
cr_id: CR-I
title: Close Distribution Decomposition
aliases: ["CR-I — Close Distribution Decomposition", "CR-I"]
status: active
started: 2026-05-26
sequence_number: 025
run_mode: interactive
phase: 2
size: medium-large
estimated_days: 4-5
data_safety_class: write_to_outcome_aggregate
dependencies: [CR-B, CR-C]
depended_on_by: [CR-F, CR-G]
branch_name: cr-i-close-distribution-decomposition
revised: 2026-05-25
tags: [dash, cr, outcomes, close-distribution, post-touch, proposals, base-rate, structural-probability, bucket-filter]
---

# CR-I — Close Distribution Decomposition

> [!info] Revised 2026-05-25 after design conversation
> This spec was substantially reframed during a chat-based design discussion before activation. Original framing: "decompose the existing `close_rate` into below/at/above fractions." Revised framing: "decompose **post-touch behavior** across **three fixed timeframes** (T+1, T+5, T+15), with same-bucket touchers as the denominator and a pooled fallback when same-bucket samples are too thin." The original framing inherited two latent problems that the revised design fixes:
>
> 1. **Mixed-timescale denominators.** The existing `reached_close` in `bt_daily_outcomes` measures terminal-horizon close at variable per-analogue horizons (1/5/20/60 sessions, set by `bucket_sessions(dominant_bucket)` in `packages/shared/buckets.py`). Aggregating across analogues with different bucket horizons conflates "where price was at session 5" with "where price was at session 60" — different physical states, treated as the same data point. The revised design uses fixed timeframes so all analogues are measured at the same temporal reference points.
>
> 2. **Conflated populations.** The existing `close_rate` denominator was "all analogues with computed outcomes," which conflates non-touchers (irrelevant to a touch-based trade decision) with touchers-that-left-the-magnet. The revised design conditions on touch — denominator becomes "analogues that touched the magnet" — so the decomposition directly answers the trade-relevant question: *given the magnet was reached, what happened after?*
>
> Additional design decisions baked in: a **bucket filter** layered on top of KNN-selected analogues (physical-regime separation, distinct from KNN's fuzzy similarity match), a **stacked horizontal bar UI** with inline Wilson CIs across three timeframes, and a **pattern label vocabulary** that summarizes the cross-timeframe shape into a single classifier (stepping-stone / touch-and-pin / touch-and-reject / overshoot-then-revert / mixed).

## Goal

Compute and surface the **post-touch close distribution** — for analogues that touched the magnet, where did price end up at T+1, T+5, and T+15 sessions after touch, below or at or above the magnet — filtered to same-bucket touchers so the underlying gamma physics is comparable across the sample. Render the distribution as a compact 3×3 visual on the proposals page, summarize the cross-timeframe shape as a pattern label, and feed the pattern signal into proposal direction selection (credit vs debit qualification) so direction is data-driven rather than heuristic.

After CR-I, the trader's decision flow becomes:

1. **Should I trade the move to the level?** — answered by existing touch rate.
2. **What does the post-touch behavior look like?** — answered by the new bucket-filtered, multi-timeframe close distribution.
3. **Which direction is supported (credit-fade vs debit-to-target)?** — answered by the distribution shape plus the pattern label.

CR-F later consumes the pattern label to pick the right **structure-within-direction** (capped vs uncapped vs hedged debit variants for the different post-touch patterns). CR-I produces the signal; CR-F refines structure choice from it.

## Context

**Surfaced 2026-05-25** during interactive review of the SPX 2026-05-13 magnet-above setup. The screenshot showed:

- Touch rate: 100% (17/17 with outcomes)
- Close rate at horizon: 11.8% (only 2/17 closed within tolerance of the 7446.9 magnet at their bucket horizon)
- Avg max excursion: 8.03× IM (~292 pt past starting spot)
- Avg days to reach: 1.3d

The system's "touches but doesn't hold" amber badge fired and the proposal generator emitted a bear-call credit spread (short 7450.6 / long 7460.6, 15 DTE). The pattern actually supported a *debit-to-target* trade, not a credit fade — analogues went 8× IM past the magnet on average, suggesting magnet-as-stepping-stone behavior, not magnet-as-resistance. The amber "touches but doesn't hold" badge correctly flagged that close ≠ touch, but it couldn't say *which way* the close-gap pointed. Without that direction signal, the credit-spread proposal couldn't be statistically validated.

The original CR-I spec attempted to address this by decomposing `close_rate` into below/at/above. During design review, two structural problems emerged: the existing `close_rate` measures at variable per-analogue horizons (mixed timescales), and the all-analogues denominator dilutes the post-touch signal with non-touchers. The revised design addresses both.

This CR connects to but is distinct from existing follow-ups:

- `close-rate-touch-rate-divergence-ux` — UX/threshold question for the existing amber badge. CR-I is upstream — it provides the data that makes the badge's question answerable. After CR-I lands, the badge layers on top of a richer signal rather than standing alone.
- `magnet-above-analogue-distance-range-mismatch` — KNN-distance concern surfaced alongside CR-I. Same conceptual neighborhood (bucket / distance / sample homogeneity within K=20). Investigation natural to revisit alongside CR-I's Step 0 empirical work.
- `structural-prob-distance-anomaly` — separate live-vs-stored feature_vector concern; orthogonal to CR-I but worth confirming before CR-G builds edge ratios on top of analogue distances.

Per Operating Framework — Where When Buy or Sell, the structural read needs to answer not just *where* (the magnet level) and *when* (the touch horizon) but also which *direction* of structure expresses the thesis. Post-touch close distribution is the missing input.

## Two-stage trade logic (operating principle)

CR-I operationalizes a two-stage decision pattern that's broader than this single CR but lands here concretely:

- **Stage 1 — Trade the move?** Touch rate answers whether the magnet is reliably reached by analogues. High touch rate (≥80%? — see Step 0 calibration) means the move-to-magnet itself is well-supported by historical evidence.
- **Stage 2 — Structure for what happens at the magnet?** The post-touch close distribution, bucket-filtered, answers what to expect after touch. The shape of the distribution across T+1/T+5/T+15 determines whether to structure for continuation (uncapped or extended debit), pinning (capped debit / iron fly), reversion (debit with credit-fade hedge), or overshoot-revert (calendar / multi-leg).

In options, stages 1 and 2 aren't sequential decisions — they're a single structure choice at entry that has different P/L characteristics at different prices and times. The structure encodes both stages simultaneously. The pattern label produced by CR-I is the input that lets CR-F choose the right structure encoding.

## Step 0 — Diagnosis (no commits)

Lock the following before implementation. Several questions are new vs the original spec; flagged as **(new)** where applicable.

1. **Inventory existing touch and close computation.** Find where `reached_touch`, `reached_close`, `days_to_reach`, and `final_close_distance_from_target` are computed. Confirmed locations (from prior reads):
   - Pure compute: `packages/shared/outcomes.py` `compute_outcome()` — operates on per-anchor inputs and ES bars; returns `bt_daily_outcomes` row content.
   - Persistence: `bt_daily_outcomes` table; runner `scripts/cr_b_backfill_outcomes.py`.
   - Live proposal-time aggregation: `packages/shared/probability.py` + `packages/shared/knn.py` (introduced in CR-C).
   - Saved-scan cache: `bt2_scan_cache` carries aggregate fields used by the saved scans UI.

   Confirm both paths (today-setup proposals and saved scans) have access to per-analogue ES bars *after* the touch session — needed to compute post-touch closes at T+1 / T+5 / T+15. The `compute_outcome()` runner currently receives the full forward-bars slice up to bucket horizon end; that slice contains the bars we need. The question is whether they're persisted anywhere or have to be re-fetched.

2. **`days_to_reach` semantics confirmation.** From `outcomes.py` docstring: 0-indexed; `days_to_reach=0` means the magnet was touched on `trade_date` itself (same session as the structural read). So the touch day's session-close is at index `days_to_reach` in the horizon DataFrame, and the close at T+N sessions after touch is at index `days_to_reach + N`. Confirm this indexing convention holds end-to-end.

3. **Tolerance definition for below/at/above.** Reuse the existing close-tolerance band — `0.25 × expected_move`, where `expected_move` is the anchor day's `implied_move_1d` from `feature_vector`. Whatever ±tolerance currently classifies "closed near magnet" in `reached_close` defines "at" in the new decomposition. Below: `close < drift_target - tolerance`. Above: `close > drift_target + tolerance`. The three categories partition the real line exactly; for each toucher at each timeframe, exactly one applies (modulo NULL when the bar is unavailable).

4. **Where the new data lives.** Two layers, decided in this step:
   - **Per-anchor post-touch positions.** For each anchor day where `reached_touch = TRUE`, three new columns on `bt_daily_outcomes`: `position_t1_post_touch`, `position_t5_post_touch`, `position_t15_post_touch`. Each is a small int: `-1` (below tolerance), `0` (within tolerance), `+1` (above tolerance), or `NULL` (if T+N is beyond available bars at the time of computation). This is the canonical per-anchor data — computed once at outcome time, reused for any future aggregation.
   - **Per-scan aggregate.** On `bt2_scan_cache`, nine fraction columns (`frac_below_t1`, `frac_at_t1`, `frac_above_t1`, ... for t5 and t15) plus their Wilson CI bounds and a single text `pattern_label`. Computed at scan time from the bucket-filtered same-bucket touchers among the K=20 KNN analogues. Whether to also persist `same_bucket_n` (denominator size) as a column for badge logic — yes, lean toward persisting it for fast UI rendering.
   - **Today-setup live path.** Mirrors saved-scan logic but computed live, not cached. Per CR-C's pattern.

   Confirm during Step 0 inventory: does `bt2_scan_cache` currently have room for ~13 new columns, or should the post-touch decomposition be a separate joined table (`bt2_scan_close_distribution`)? Lean: inline columns for now; refactor to separate table only if the row width becomes unwieldy.

5. **(new) Bucket-homogeneity empirical check.** Before Step 1 implementation, run a one-off query against the existing `bt2_scan_cache`: for several anchor days, examine the K=20 KNN-selected analogues and count how many share the anchor's `dominant_bucket_at_classification`. This determines:
   - Whether the bucket filter narrows aggressively (e.g., K=20 → typically N=14 same-bucket) or modestly (K=20 → typically N=18 same-bucket)
   - What the appropriate fallback threshold is (see #6)
   - Whether KNN feature weighting itself needs adjustment (a separate concern; if same-bucket fraction is routinely <50%, the `magnet-above-analogue-distance-range-mismatch` FU upgrades from "side concern" to "address before CR-I lands")

   Output: a short table (in this spec's Status updates after Step 0 commits) showing same-bucket fraction across, say, 10 hand-picked anchor days spanning all four dominant_bucket values.

6. **(new) Fallback threshold for thin same-bucket samples.** Calibration based on Step 0 sub-step 5's data:
   - **Strict bucket filter** when `same_bucket_n >= threshold` (use only same-bucket touchers as denominator)
   - **Pooled fallback** when `same_bucket_n < threshold` (use all touchers, badge the decomposition as "bucket-pooled fallback, low confidence")
   - **No decomposition** when `total_touchers < some lower threshold` (Wilson CIs too wide to be informative; render "insufficient post-touch sample" badge instead)

   Starting values to lock in spec: strict threshold = 10 same-bucket touchers; pooled-fallback minimum = 5 total touchers. Adjust based on Step 0 sub-step 5.

7. **(new) Pattern label vocabulary.** Lock the label set used in the synthesis line. Starting vocabulary:
   - `stepping-stone` — above dominates across all three timeframes (T+1 above-fraction > 0.50 AND T+5 above-fraction > 0.50 AND T+15 above-fraction > 0.50). Magnet acts as a waypoint; price keeps moving.
   - `touch-and-pin` — at-fraction is the largest segment in at least two of three timeframes. Magnet acts as gravity center.
   - `touch-and-reject` — below dominates across all three timeframes (for magnet-above; mirror for magnet-below). Magnet acts as resistance.
   - `overshoot-then-revert` — above (or below in magnet-below case) dominates at T+1 and T+5 but reverses at T+15. Magnet eventually held after initial overshoot.
   - `slow-revert` — pattern shifts gradually across timeframes from one extreme toward magnet (e.g., 80% above at T+1, 50% above at T+5, 25% above at T+15). Magnet pulling but slowly.
   - `mixed` — no clear pattern; no single timeframe has a dominant fraction >0.50, or the cross-timeframe shape is incoherent.

   Pattern classifier is deterministic given the 9 fractions. Spec the exact decision tree in Step 1's implementation; iterate on the boundary thresholds based on Step 0's empirical pull of existing anchor days.

8. **(new) Direction qualification thresholds.** Simplified vs original spec — direction logic in Step 4 is "credit vs debit qualification" only; finer structure choice is CR-F's job. Starting values:
   - **Credit-direction qualifying** (favor credit-fade structures): `pattern_label in {'touch-and-reject', 'slow-revert', 'overshoot-then-revert'}` AND the relevant Wilson lower-bound on the reversion direction is > 0.40
   - **Debit-direction qualifying** (favor debit-to-target structures): `pattern_label in {'stepping-stone', 'touch-and-pin'}` AND the relevant Wilson lower-bound is > 0.40
   - **Mixed / no preference**: anything else → emit both directions with low-confidence badge, or emit existing default behavior

   The asymmetry in original spec (credit needing stronger evidence than debit) is dropped — the pattern label already encodes asymmetric structure, and the Wilson lower-bound is a uniform 0.40 floor across qualifications. Re-introduce asymmetry only if empirical calibration shows it's needed.

9. **Wilson CI on each fraction.** Same treatment as CR-C's touch / close CIs. Each of the 9 fractions gets a 95% Wilson CI. Pattern label classifier uses point estimates for boundary checks; direction qualification uses the lower bound on the dominant fraction.

10. **What to do about analogues with missing post-touch closes.** If an analogue touched but the T+N bar is unavailable (e.g., the touch happened so late in the corpus that T+15 falls outside loaded bars), the `position_tN_post_touch` is NULL for that analogue. Exclude from the denominator for that timeframe only — denominators may differ across timeframes (denominator_t1 ≥ denominator_t5 ≥ denominator_t15). Document this in the schema comment.

11. **Pre-flight: CR-B outcome computation rerun?** The Step 0 inventory needs to confirm whether existing `bt_daily_outcomes` rows have enough information (touch flag, days_to_reach, raw bars accessible) to populate the new per-anchor position columns via a backfill, or whether the outcomes computation itself needs a rerun. Strong preference: backfill from existing rows + re-fetched bars — avoid rerunning CR-B's outcome computation entirely. If raw bars need re-fetching for the backfill, that's fine (ES 1m bars are cheap to query); rerunning outcome classification is not.

## Step 1 — Computation logic

**Commit:** `cr-i/step-1: post-touch position classifier + bucket-filtered aggregation + pattern labeling`

Two new pure-compute functions, both in `packages/shared/probability.py` (or a new `packages/shared/post_touch.py` if the file gets too large — decide during implementation):

### 1a. Per-analogue post-touch position classifier

```python
def classify_post_touch_positions(
    days_to_reach: int,
    horizon_bars: pd.DataFrame,
    drift_target: float,
    tolerance: float,
    timeframes_sessions: tuple[int, ...] = (1, 5, 15),
) -> dict[int, Optional[int]]:
    """
    For an analogue that touched the magnet, classify the close position at
    each post-touch timeframe.

    Returns {1: position_t1, 5: position_t5, 15: position_t15} where each
    position is -1 (below tolerance), 0 (within tolerance), +1 (above
    tolerance), or None (bar not available at that timeframe).

    days_to_reach is 0-indexed per outcomes.py convention; the touch session
    is at horizon_bars.iloc[days_to_reach]. T+N is at iloc[days_to_reach + N].
    """
```

Use this from the outcomes runner (or a Step 2 backfill script) to populate `bt_daily_outcomes.position_tN_post_touch` columns for every anchor day with `reached_touch = TRUE`.

### 1b. Bucket-filtered aggregation + Wilson CI + pattern label

```python
def aggregate_post_touch_distribution(
    analogues_with_outcomes: list[dict],
    anchor_bucket: str,
    fallback_threshold: int = 10,
    pooled_minimum: int = 5,
) -> dict:
    """
    Aggregate post-touch positions across K=20 analogues into the 9-cell
    matrix + Wilson CIs + pattern label.

    Steps:
      1. Filter to analogues where reached_touch=True
      2. Among those, count same-bucket vs different-bucket
      3. If same_bucket_n >= fallback_threshold: use same-bucket only (strict)
         elif total_touchers >= pooled_minimum: use all touchers (pooled fallback)
         else: return insufficient-sample marker
      4. For each timeframe (T+1, T+5, T+15):
           - Filter to analogues with non-NULL position at that timeframe
           - Count below / at / above
           - Compute fractions
           - Compute Wilson 95% CI for each fraction
      5. Apply pattern classifier (deterministic decision tree per Step 0 #7) to get label
      6. Return full structured dict including denominator(s), fractions, CIs,
         pattern label, and filter-mode flag (strict / pooled-fallback / insufficient).
    """
```

Output schema:

```python
{
    "filter_mode": "strict" | "pooled-fallback" | "insufficient",
    "denominator_t1": int,  # may differ across timeframes due to NULL handling
    "denominator_t5": int,
    "denominator_t15": int,
    "same_bucket_n": int,
    "total_touchers": int,
    "fractions": {
        "t1":  {"below": float, "at": float, "above": float},
        "t5":  {"below": float, "at": float, "above": float},
        "t15": {"below": float, "at": float, "above": float},
    },
    "wilson_cis": {
        "t1":  {"below": (lo, hi), "at": (lo, hi), "above": (lo, hi)},
        "t5":  {...},
        "t15": {...},
    },
    "pattern_label": "stepping-stone" | "touch-and-pin" | "touch-and-reject"
                   | "overshoot-then-revert" | "slow-revert" | "mixed",
}
```

Wire into the today-setup probability extraction path (`packages/shared/probability.py` orchestrator) so the response carries the full post-touch structure alongside the existing `touch_rate` and `close_rate`.

**Deliverable:** both functions imported, unit-tested, integrated into the live path. API response for `/api/setup/proposals` carries the post-touch distribution structure.

**Verification:** unit tests with synthetic analogue lists exercising:
- Pure stepping-stone (all touchers, all positions above at all three timeframes) → fractions all 0/0/1.0, pattern `stepping-stone`
- Pure touch-and-pin (all touchers, all positions at at all timeframes) → fractions 0/1.0/0, pattern `touch-and-pin`
- Touch-and-reject in magnet-above context → pattern `touch-and-reject`
- Overshoot-then-revert (T+1 above, T+5 above, T+15 below) → pattern `overshoot-then-revert`
- Mixed (1 below / 1 at / 1 above at all timeframes) → pattern `mixed`
- Strict filter triggers (12 same-bucket of 17 touchers → strict mode, denominator 12)
- Pooled fallback triggers (3 same-bucket of 8 touchers → pooled-fallback mode, denominator 8)
- Insufficient sample (4 touchers total → insufficient marker)
- NULL handling (some analogues have NULL at T+15 because bars not available → denominator_t15 < denominator_t1)

Hand-pull one historical anchor day with known touch behavior; verify the function output matches manual classification of its analogues' post-touch positions.

## Step 2 — Schema migration and backfill

**Commit:** `cr-i/step-2: schema columns + backfill per-anchor positions and per-scan aggregates`

### 2a. Schema changes

```sql
-- Per-anchor post-touch positions on bt_daily_outcomes
ALTER TABLE bt_daily_outcomes
  ADD COLUMN position_t1_post_touch  SMALLINT,
  ADD COLUMN position_t5_post_touch  SMALLINT,
  ADD COLUMN position_t15_post_touch SMALLINT;
-- Constraint: each is -1, 0, +1, or NULL. Enforce via CHECK if convenient.

-- Per-scan aggregate on bt2_scan_cache (9 fractions + 18 CI bounds + 1 label
-- + denominator/filter-mode metadata)
ALTER TABLE bt2_scan_cache
  ADD COLUMN frac_below_t1 REAL,
  ADD COLUMN frac_at_t1    REAL,
  ADD COLUMN frac_above_t1 REAL,
  ADD COLUMN ci_below_t1_lo REAL, ADD COLUMN ci_below_t1_hi REAL,
  ADD COLUMN ci_at_t1_lo    REAL, ADD COLUMN ci_at_t1_hi    REAL,
  ADD COLUMN ci_above_t1_lo REAL, ADD COLUMN ci_above_t1_hi REAL,
  -- ... repeated for t5 and t15
  ADD COLUMN post_touch_pattern_label VARCHAR(32),
  ADD COLUMN post_touch_filter_mode   VARCHAR(20),
  ADD COLUMN post_touch_same_bucket_n INTEGER,
  ADD COLUMN post_touch_total_touchers INTEGER;
```

Sum invariant check via app-layer validation rather than DB constraint — DB-level CHECK on `ABS(below + at + above - 1.0) < 0.01` is fragile across NULL semantics and rounding.

If column count on `bt2_scan_cache` becomes unwieldy after this addition (>50 columns?), consider splitting into a joined `bt2_scan_close_distribution` table keyed by the same scan id. Decide during Step 0 inventory.

### 2b. Backfill runner

`scripts/cr_i_backfill_post_touch_positions.py` — follows the CR-D backfill pattern with the corner-case bugs from `cr-d-backfill-corner-case-bugs` fixed (use `cursor.execute()` not `conn.execute()`; explicit `conn.commit()` after each batch).

Pre-flight per Data Safety Protocol:
- `SELECT current_user` must equal `dash_backfill_writer`
- Open a `bt_backfill_runs` row with `cr_id='CR-I'`, `status='running'`
- Use `BACKFILL_DATABASE_URL`, not `DATABASE_URL`

Pass 1: backfill `bt_daily_outcomes` position columns.
- Iterate over rows where `reached_touch = TRUE` AND `position_t1_post_touch IS NULL`
- For each: re-fetch the horizon ES bars (or load cached if available), use `classify_post_touch_positions()`, UPDATE the row with the three position values + `backfill_run_id`
- Idempotent: skips already-populated rows

Pass 2: backfill `bt2_scan_cache` aggregates.
- Iterate over rows with `frac_below_t1 IS NULL`
- For each: load the row's cached K=20 analogues, look up each analogue's `position_tN_post_touch` from `bt_daily_outcomes`, run `aggregate_post_touch_distribution()`, UPDATE the row with the 9 fractions + 18 CIs + pattern label + filter mode + denominators + `backfill_run_id`
- Idempotent

Smoke tests:
- Sum invariant: `SELECT COUNT(*) FROM bt2_scan_cache WHERE post_touch_filter_mode = 'strict' AND ABS(frac_below_t1 + frac_at_t1 + frac_above_t1 - 1.0) > 0.01` returns 0 (across all three timeframes)
- Pattern label coverage: `SELECT post_touch_pattern_label, COUNT(*) FROM bt2_scan_cache GROUP BY 1` — no NULL labels except where `post_touch_filter_mode = 'insufficient'`
- Position triangle: `SELECT COUNT(*) FROM bt_daily_outcomes WHERE reached_touch = TRUE AND position_t1_post_touch IS NULL` ≈ 0 (modulo end-of-corpus rows where T+1 bars aren't available yet — should be a small known set)
- Spot-check 5 hand-picked rows: manual recomputation matches stored values

Both passes complete in one runner execution. Estimated compute: ~30-60 min depending on bar re-fetch overhead. Eligible for unattended execution per Data Safety Protocol (`null_fill_update` class on existing rows; no existing data modified, only NULLs filled). Note: Step 2a (DDL) is `schema_change` class — must run interactively first, before the backfill runs unattended.

**Deliverable:** schema updated, both tables backfilled, smoke tests pass, `bt_backfill_runs` row marked `completed`.

## Step 3 — Frontend UI surfacing

**Commit:** `cr-i/step-3: PostTouchDistributionBlock — stacked-bar UI + pattern label + synthesis line`

Extend `react_today_setup/src/components/StructuralProbabilityBlock.jsx` (per CR-C) with a new section below the existing close_rate row, or split into a new sibling component `PostTouchDistributionBlock.jsx` if the existing block gets crowded. Decide during implementation based on visual density.

### Layout

```
Post-touch close distribution
(same-bucket touchers only, N=14)
                                                              <- bucket badge

T+1    ## 14% [4-32%]   . 7% [1-31%]    ########## 79% [54-92%]
T+5    ### 21% [7-42%]  .. 14% [4-32%]  ######## 64% [38-85%]   <- 15-DTE trade
T+15   ### 21% [7-42%]  ... 21% [7-42%] ####### 57% [31-80%]
       below             at               above

-> Pattern: stepping-stone. Direction signal: debit-to-target supported.
```

Elements:

- **Header line:** "Post-touch close distribution" title + denominator subline ("same-bucket touchers only, N=14" for strict mode; "bucket-pooled fallback, N=8" for pooled-fallback mode; "insufficient post-touch sample" for the no-data case).
- **Three stacked horizontal bars**, one per timeframe. Each is 100%-wide, split into below/at/above segments proportional to the fractions. Each segment carries its fraction percentage + Wilson CI bounds in parentheses, inline.
- **Color scheme:** below = subtle blue (#dataviz-cool token from existing palette), at = neutral gray, above = subtle orange/red (#dataviz-warm). Direction-neutral palette; the user maps it to trade implication mentally.
- **Trade-DTE-relevant row marker:** small arrow and label ("← 15-DTE trade") on the row whose timeframe most closely matches the proposed trade's DTE. Map: 3-DTE trade → T+1 row; 15-DTE trade → T+5 row (closest); 30-DTE+ → T+15 row.
- **Synthesis line:** plain English summary derived from the pattern label and the dominant direction. Pattern → message mapping:
   - `stepping-stone` → "Pattern: stepping-stone. Direction signal: debit-to-target supported."
   - `touch-and-pin` → "Pattern: touch-and-pin. Direction signal: pin structure (iron fly / iron condor) supported."
   - `touch-and-reject` → "Pattern: touch-and-reject. Direction signal: credit-fade supported."
   - `overshoot-then-revert` → "Pattern: overshoot-then-revert. Direction signal: complex — review timeframes manually."
   - `slow-revert` → "Pattern: slow-revert. Direction signal: gradual reversion; consider longer-dated structure."
   - `mixed` → "Pattern: mixed. Direction signal: insufficient consistency across timeframes."

### CSS / styling

Reuse `.sp-divergence-badge` pattern from CR-C for any new badges; extend with `.sp-direction-badge` (green for direction-qualifying clear signal, gray for mixed) and `.sp-bucket-fallback-badge` (amber for pooled-fallback mode). Use existing token-based colors throughout — no hard-coded hex.

### State handling

- `insufficient` filter mode: hide the entire stacked-bar matrix; show only the "insufficient post-touch sample (N=4 touchers)" message.
- `pooled-fallback` filter mode: render the matrix normally but with the amber `.sp-bucket-fallback-badge` next to the denominator subline ("bucket-pooled fallback, N=8 — low confidence").
- `strict` filter mode: standard render, no badge.

**Deliverable:** Base Rate block now shows the post-touch distribution matrix, pattern label, and synthesis line when sufficient touchers exist; falls back gracefully otherwise.

## Step 4 — Proposal direction logic

**Commit:** `cr-i/step-4: select proposal direction based on post-touch pattern label`

Update the proposal-emission logic (likely `apps/web/modules/TodaySetup/service.py` for the today-setup path; equivalent in saved-scan path) to consume the new `post_touch_pattern_label` and direction qualification logic from Step 0 #8.

Decision flow:

```
Given regime + post_touch result:

if post_touch.filter_mode == 'insufficient':
    # Not enough data to drive direction selection — fall back to legacy logic
    emit legacy default (current credit-default for magnet-above/below)
    badge proposal as "low-confidence — post-touch sample insufficient"

elif regime in (magnet-above, magnet-below):
    if credit_direction_qualifies(post_touch):
        emit credit-fade structure (current default behavior)
    elif debit_direction_qualifies(post_touch):
        emit debit-to-target structure
        # Note: CR-F will later select capped vs uncapped vs hedged
        # variant of the debit structure based on the pattern_label.
        # For CR-I, emit the existing single debit template.
    else:
        # Mixed — emit both with low-confidence badge
        emit both directions, both badged "mixed pattern — no clear direction"

elif regime in (magnetic-pin, bounded):
    # These regimes have their own templates that don't need direction selection
    # at this layer. CR-F handles structure-within-direction for pin/bounded later.
    pass  # existing behavior preserved
```

The `credit_direction_qualifies` and `debit_direction_qualifies` functions implement Step 0 #8's logic — pattern label membership AND Wilson lower-bound floor.

Add unit tests covering:
- `stepping-stone` pattern → debit emitted, no credit, badge "debit-to-target supported"
- `touch-and-reject` pattern → credit emitted, no debit, badge "credit-fade supported"
- `touch-and-pin` pattern → debit emitted (toward pin); CR-F will refine to iron-fly later
- `mixed` pattern → both directions emitted, both with mixed-pattern badge
- `insufficient` filter mode → legacy default + low-confidence badge
- Missing post-touch data on a pre-backfill row (NULL fractions) → fall back to legacy behavior

**Deliverable:** proposal generator picks direction based on the pattern label, with graceful fallback to legacy behavior on missing data.

## Smoke tests

1. **Sum invariant on a freshly computed row.** Pick an anchor day, hit `/api/setup/proposals`, verify all three timeframes' below+at+above sum to 1.0 ± 0.01.

2. **Strict vs pooled-fallback mode triggers.** Find one anchor where `same_bucket_n >= 10` (expect strict) and one where `same_bucket_n < 10` but `total_touchers >= 5` (expect pooled-fallback). Verify badge text and denominator subline match.

3. **Pattern classification — stepping-stone.** Find or hand-construct an anchor day where T+1/T+5/T+15 are all above-dominant. Confirm `pattern_label = 'stepping-stone'`, direction signal = debit-to-target, synthesis line correct.

4. **Pattern classification — touch-and-reject.** Magnet-above day where T+1/T+5/T+15 are all below-dominant. Pattern = `touch-and-reject`, direction = credit-fade.

5. **Pattern classification — overshoot-then-revert.** T+1 and T+5 above, T+15 below. Pattern = `overshoot-then-revert`, synthesis line flags as complex.

6. **Direction qualification — debit case (the 2026-05-13 anchor).** Run /today-setup against the SPX 2026-05-13 magnet-above setup. Expect: pattern = `stepping-stone` (8× IM excursions suggest above dominance), debit-direction qualifies, proposal output flips from credit-spread to debit-call-spread. This is the verifying case that motivated the CR.

7. **Backward compatibility.** Trigger the proposals path on a pre-backfill row (NULL post-touch fractions). Confirm proposal still emits with legacy logic, no crash, badge correctly indicates fallback.

8. **Insufficient-sample handling.** Find an anchor day with `total_touchers < 5` (e.g., low-frequency regime). Confirm the UI hides the matrix and shows the "insufficient sample" message; proposal still emits via legacy fallback.

9. **Wilson CI sensibility.** For a row with `denominator_t1 = 3` all-below, confirm `ci_below_t1` lower bound is well below 1.0 (small-sample uncertainty propagates). For `denominator = 50` with 40-below / 5-at / 5-above, confirm CIs are tight.

10. **Bucket-homogeneity sanity.** From the Step 0 sub-step 5 results, spot-check that the persisted `same_bucket_n` values match the empirical investigation.

## Wrap criteria

- All 4 steps committed on `cr-i-close-distribution-decomposition`
- Step 0 sub-step 5 (bucket-homogeneity check) findings appended to spec's Status updates section before Step 1 work begins
- Backfill complete: `bt_daily_outcomes` rows with `reached_touch = TRUE` carry the three post-touch position values; `bt2_scan_cache` rows with non-NULL close_rate carry the nine fractions + Wilson CIs + pattern label + denominator/filter metadata
- All smoke tests pass
- Frontend block renders the distribution matrix + pattern label + synthesis line on /today-setup
- The 2026-05-13 SPX magnet-above anchor verifies as expected — pattern label correctly identified, proposal direction flips from credit-spread to debit-to-target
- Roadmap updated: CR-I marked complete
- Retrospective note filed if the diagnosis-to-implementation arc surfaced anything worth durable preservation (the bucket-homogeneity empirical data probably warrants its own short note regardless)
- Related FUs surveyed: `magnet-above-analogue-distance-range-mismatch` — if Step 0 #5's bucket-homogeneity check exposed thin same-bucket fractions across many anchors, upgrade this FU to active priority (the KNN feature weighting may need adjustment before CR-G builds on top)

## Status updates

(Step 0 diagnosis findings will be appended here)

## Open questions

- **Should pattern labels be empirically calibrated against P&L outcomes before being used for direction selection?** The pattern→direction mapping in Step 4 is a hypothesis. After CR-G adds P&L reconstruction, we can validate: does `stepping-stone` actually correlate with debit-to-target profitability, etc. Calibration loop deferred to post-CR-G; CR-I's mapping is the first-pass hypothesis to be tested empirically once edge measurement exists.

- **Should the 0DTE bucket get its own special handling?** A 0DTE-dominant anchor has bucket horizon = 1 session in `outcomes.py`. By construction, T+5 and T+15 are beyond the bucket horizon for analogues classified as 0DTE-dominant — meaning their `position_t5_post_touch` and `position_t15_post_touch` will be NULL more often than for longer-bucket analogues. Effect: denominator shrinkage at longer timeframes is bucket-correlated. Either (a) accept this as a known property of the data, or (b) extend outcomes.py to compute T+15 close even for 0DTE-bucket anchors (just look further forward in bars regardless of bucket horizon). **Lean (b)** — bars are available; the only reason 0DTE outcomes were truncated to 1 session was to match strategy DTE targets, not because the data wasn't there. Step 0 inventory confirms whether (b) requires touching `outcomes.py` or whether the post-touch position classifier can look beyond the bucket horizon directly.

- **What happens if CR-F lands before CR-I?** Same answer as the original spec: probably nothing breaks. CR-F's IV-based preference is orthogonal to CR-I's pattern-based direction selection. If both land, CR-I picks direction (credit vs debit), CR-F picks variant (capped vs uncapped, debit-vs-credit IV preference). They compose. CR-I → CR-F → CR-G is the intended order.

- **Should the "touches but doesn't hold" amber badge from CR-C persist after CR-I lands?** Yes — the amber badge is a quick "scan for unusual touch/close divergence" signal. CR-I gives the deeper read on WHICH WAY the divergence points. Two stacked signals, complementary not redundant.

- **Bucket-homogeneity result feeds back to KNN tuning.** If Step 0 #5 shows K=20 is routinely <50% same-bucket, the KNN feature weighting needs revisiting — bucket should weigh more heavily in the distance metric. This is a finding for `magnet-above-analogue-distance-range-mismatch` and possibly a small follow-on CR. Worth noting that the bucket-filter in CR-I works as a robustness layer regardless: even if KNN is mis-weighted, the bucket filter restores physical-regime consistency at the decomposition layer.

- **The 0DTE-trade case — T+1 dominates.** For a 0DTE proposal (3 DTE target trade), the T+1 timeframe is essentially the trade horizon itself, and T+5/T+15 are post-expiration context. The UI marker correctly highlights T+1, but should the synthesis line emphasize T+1 even more in this case? **Lean: yes, but defer to post-implementation polish if it surfaces as a real issue.** Synthesis logic uses pattern label which weights across all three timeframes; for 0DTE this may produce a label that disagrees with what T+1 alone says. Worth re-checking after CR-I lands and 0DTE setups are observed.

## Related

- **Sessions:** `2026-05-25 - CR-022 — Outcome Computation` (origin of close_rate and the `bt_daily_outcomes` schema), `2026-05-25 - CR-023 — Probability Output on Proposals` (Base Rate block + KNN extraction + canonical promotion pattern)
- **Decisions:** `2026-05-25 - CR-023 Retrospective` (compute/persist separation, surface implementation-discovered refactors BEFORE), `2026-05-24 - Data Safety Protocol` (governs the Step 2 backfill)
- **Open questions:** `close-rate-touch-rate-divergence-ux` (related UX concern, distinct from this data decomposition — CR-I provides the data the badge's question needs), `magnet-above-analogue-distance-range-mismatch` (KNN-search-layer concern adjacent to CR-I's bucket-filter architecture; Step 0 #5 informs)
- **Foundational:** Operating Framework — Where When Buy or Sell (CR-I's direction logic operationalizes the buy/sell question for magnet regimes), Trade Construction Principles (CR-I's pattern label is the input to leg-by-leg structure choice that CR-F encodes)
- **Downstream:** CR-F — Debit Credit Variants (will consume the pattern label for structure-within-direction selection; spec amended 2026-05-25 to flag this scope expansion), CR-G — Edge Visualization and P&L Engine (post-touch fractions feed edge-ratio computation as one input)
- **Files (probable):** `packages/shared/probability.py`, `packages/shared/post_touch.py` (possible new file), `packages/shared/outcomes.py` (possible 0DTE-bucket extension per open question), `apps/web/modules/TodaySetup/service.py`, `react_today_setup/src/components/StructuralProbabilityBlock.jsx`, `react_today_setup/src/components/PostTouchDistributionBlock.jsx` (possible new), `react_today_setup/src/styles.css`, `scripts/cr_i_backfill_post_touch_positions.py` (new)
