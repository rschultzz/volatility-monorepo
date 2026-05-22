# CR-011 — Confluence Quality Recalibration

### Problem

`classify_confluence_quality` in `packages/shared/gex_landscape.py` tags each multi-bucket confluence cluster as one of `pin-grade`, `drift-grade`, or `waypoint`. The thresholds embedded in the function:

```python
if score_b_per_pt >= 30 and avg_fwhm < 40:
    return "pin-grade"
if score_b_per_pt < 5 or avg_fwhm > 80:
    return "waypoint"
return "drift-grade"
```

are documented as calibrated against 5/7 (`5/7 pin-grade : score ≈ 84 B/pt, fwhm ≈ 25pt`) and 5/20 (`5/20 waypoint : score ≈ 3 B/pt, fwhm ≈ 84pt`). The 5/7 anchor does not match what the live pipeline produces for that day's data. On 5/7 the confluences at 7353 and 7371 both come back tagged `waypoint` (rendered as "soft" in the React panel and as dotted lines in the matplotlib `_stacked.png` output), even though 5/7 is the canonical empirically-validated pin day in the project's reference set — the day the rest of the model documentation uses as the reference for what a pin looks like.

The mismatch traces to the FWHM gate. The dominant bucket on 5/7 is 8-30 DTE (42% dominance) and the cluster at 7353 is anchored on the 8-30 DTE peak. Per the Gaussian-smoothing physics in `compute_landscape` — `σ = spread_coef × sqrt(max(dte, 0.5))` — an 8-30 DTE peak has σ in the 22-44pt range and a FWHM around 50-100pt. That's not a calibration accident; it's the model design. The `< 40pt` FWHM gate is unreachable for any cluster anchored on a bucket with mean DTE ≥ ~5, so any pin day dominated by 1-7+ DTE positioning fails the `pin-grade` gate by construction.

Concretely:

- **5/7 (canonical pin day, 8-30 DTE dominates at 42%)** — confluences at 7353 ★★ and 7371 ★★ tagged `waypoint`. Should be `pin`.
- **5/22 (current session, 1-7 DTE dominates at 34%)** — confluences at 7517 ★★, 7549 ★★, 7449 ★★ tagged `waypoint`. Likely `target` or `feature` (lower-conviction structure than 5/7, broader-and-shallower aggregate); labeled-set entry pending observation of EOD price.

The functional consequence is that the quality tag — meant to be the user-facing summary of "is this a tradeable pin or just a feature of the field" — currently labels every empirically-real pin day "soft" whenever the dominant bucket is anything other than 0DTE. Downstream consumers that key off the quality tag (a future strategy-picker module under design, any quality-graded backtest filters, the line-style choice in the matplotlib and React renderings) inherit that error.

The deeper structural issue is that the model conflates *peak sharpness* with *pinning capability*. A 0DTE pin with FWHM=8pt and a 30+ DTE pin with FWHM=90pt can both produce real EOD stickiness, via different mechanisms (sharp gamma-induced mean reversion vs. broad layered hedging mass). The current quality formula penalizes the second case heavily and rewards only the first.

A second, separable problem is that the current label names — `pin-grade` / `drift-grade` / `waypoint` — are operationally ambiguous. The code's own docstring describes `drift-grade` as "transitional / waypoint level," using the third label's name in the second's definition. CR-011 renames them to `pin` / `target` / `feature` and embeds an operational test for distinguishing them (see *Operational definitions* below).

### Operational definitions

The labels classify what kind of interaction the marked confluence is predicted to have with the session's price action. The canonical test is:

> *"If this session has a directional move, where does price end up?"*

- If the answer is "right at this level, tight band" → **pin**
- If the answer is "near this level, give or take" → **target**
- If the answer is "somewhere else, or it depends on factors outside this level" → **feature**

Concrete operational definitions:

**pin** (was `pin-grade`)
: Price *locks in* at this level if it reaches it. Strong restoring force in both directions. EOD print is at the level, typically with a tight last-hour range around it (±5-10pt). Trade-wise, this is the level you center a butterfly on.

**target** (was `drift-grade`)
: This level is where price *ends up if there's any directional move* during the session. It's an attractor for drift, but not strong enough to lock price into a tight band. Price approaches it, may sit near it, may overshoot or undershoot, but the level is the destination of the directional move. Trade-wise, this is the level you target with a directional spread (call/put spread with the short strike at or near the level).

**feature** (was `waypoint`)
: This is a feature of the field — multi-bucket agreement is real — but it isn't a destination for the session. Either too far away, too weak, or just structurally not where the day's dynamics are headed. Price may pass through it, ignore it, or touch it briefly without engaging. Trade-wise, no level-specific trade based on this; the session's edge has to come from elsewhere (skew, IV richness, or sit out).

The labeling test is *observed price behavior*, not model output. A day's label answers "did the session engage with this level as a destination" — not "was this level structurally prominent in the GEX field." That distinction matters: a structurally prominent level on a day with no directional move is still a `feature` if price didn't engage with it.

**Worked examples** (from the calibration set):

- **5/7 = pin** — Spot 7400, marked level 7353. Price moved decisively from 7400 to 7353 and locked in tight. The level was both destination *and* trap.
- **5/6 = target** — Spot 7326, magnet above at ~7350-7360 (30pt away, within range). Price drifted upward across the session toward that level. The level was the destination of the directional move, though not a tight pin.
- **5/18 = feature** — Spot 7378, marked levels at 7510 and 7530 (132-152pt away). Even with a directional move, no realistic chance of reaching those levels. The day's actual price action was driven by factors other than the marked features.
- **5/20 = feature** — Spot 7392, marked levels at 7456 and 7505, sitting on a diffuse 30+ DTE bulge with 0DTE/1-7 DTE in amplification regime. Price topped at ~7460 (just above 7456) but didn't engage with it as a destination — the day's dynamics were driven by the amplification underbelly, not the marked levels.

### Display strings

The frontend (`GexLandscapePanel.jsx`, `PriceChart.jsx`) and matplotlib script (`explore_gex_landscape.py`) currently render the `quality` field via short display strings:

```javascript
const QUALITY_SHORT = { 'pin-grade': 'PIN', 'drift-grade': 'DRIFT', waypoint: 'soft' }
```

CR-011 changes the *internal* label names but keeps the *user-facing* display strings close to current to preserve UI continuity:

| Internal label | Display string |
|----------------|----------------|
| `pin`          | `PIN`          |
| `target`       | `TGT`          |
| `feature`      | `soft`         |

`soft` is retained for `feature` because it's already a familiar trading term and accurately conveys "weak level, don't trade it as a pin." `TGT` replaces `DRIFT` because the verb-framing of "DRIFT" never quite matched the noun-framing of the other two; "TGT" matches the 3-letter style of "PIN" and aligns with the trade implication (this is the directional spread target).

### Proposed Solution

Recalibrate the quality classifier to fire correctly across the range of pin types the landscape actually produces. Three implementation options, in increasing order of model change and increasing order of expected accuracy. The spec lists all three so the choice can be made during pre-implementation review with the calibration set in hand.

**Option A — Relax thresholds, keep the formula.** Lower the score floor and raise the FWHM ceiling on `pin`; lower the FWHM ceiling on `feature`. New candidate thresholds, to be confirmed by calibration:

```python
if score_b_per_pt >= 10 and avg_fwhm < 100:
    return "pin"
if score_b_per_pt < 3 or avg_fwhm > 130:
    return "feature"
return "target"
```

Rationale: the existing formula already encodes the score↔FWHM trade-off (broader peaks need higher max_gex to compensate). The gates just need to be set against the FWHM distribution the model actually produces, not the aspirational distribution the original comment described. Minimum invasive change; preserves the current API and the meaning of `score`.

Cost: still mis-classifies edge cases where mass concentration matters more than peak prominence (a broad-but-dense cluster from heavy 8-30 DTE on a low-vol day is structurally a pin, but a broad-and-shallow cluster of the same FWHM is not — Option A can't distinguish them).

**Option B — Make FWHM relative to the dominant bucket's DTE.** Compute an expected FWHM for each cluster based on the DTE of the bucket(s) contributing the largest peak, and gate on the ratio `actual_fwhm / expected_fwhm` rather than absolute FWHM. A cluster anchored on a 30 DTE peak has expected FWHM around 100pt, so an actual FWHM of 90pt is *tighter than expected* and earns a `pin` tag; the same 90pt FWHM on a 0DTE-anchored cluster would be *much broader than expected* and degrade the tag.

Implementation sketch:

- Augment `score_confluence` to compute `dominant_bucket_mean_dte` for the cluster (the mean DTE of the bucket contributing the highest peak). Bucket label → DTE midpoint mapping: `0DTE → 0.5`, `1-7 DTE → 4`, `8-30 DTE → 19`, `30+ DTE → 45`. Refine midpoints if calibration shows they're off.
- Compute `expected_fwhm = spread_coef × sqrt(dominant_bucket_mean_dte) × 2.355` (the Gaussian FWHM = 2.355σ).
- Add `fwhm_ratio = avg_fwhm / max(expected_fwhm, 5.0)` to the score record.
- New `classify_confluence_quality(score, fwhm_ratio)` gates on the ratio. Initial thresholds to be calibrated:

```python
if score_b_per_pt >= 10 and fwhm_ratio < 1.2:
    return "pin"
if score_b_per_pt < 3 or fwhm_ratio > 2.0:
    return "feature"
return "target"
```

Rationale: directly addresses the root cause — the model's own physics says different buckets produce different FWHM, so the quality threshold needs to know which bucket the cluster lives in.

Cost: adds a field to the data model (`fwhm_ratio` in the confluence record) and changes the second argument to `classify_confluence_quality`, a small breaking change to direct callers. The shared module is the only caller today.

**Option C — Replace FWHM with a mass-concentration metric.** Compute, for each cluster, the fraction of the day's total positive GEX that falls within a fixed window (e.g., ±15pt) around the cluster center. A pin can come from either tight FWHM (sharp peak) or high concentration (dense mass region) — concentration captures both.

Implementation sketch:

- New helper `compute_mass_concentration(landscape, center_price, window_pts)` returns `sum(gex_total in [center - window, center + window]) / sum(gex_total where positive)`.
- Add `mass_concentration` to the score record alongside (or replacing) `avg_fwhm`.
- New score formula: `score = n_buckets × max_gex × mass_concentration`.
- New quality gates calibrated against the labeled set.

Rationale: the most semantically correct metric for "is this a real pin." A high-concentration cluster is a pin regardless of whether the peak itself is sharp or broad.

Cost: largest model change. New data model field, new score formula, new thresholds, all needing calibration. Higher confidence in correctness but more surface area to verify.

**Recommendation.** Option B. It directly addresses the root cause (FWHM-vs-DTE mismatch), the cost is contained to one new field and a small change to one function signature, and the calibration is straightforward (the FWHM ratio against the model's own physics is well-defined). Option A leaves the root cause in place; Option C is over-investment for a metric that hasn't been validated to outperform the FWHM-ratio approach yet. Recommend shipping Option B and treating Option C as a future CR if the labeled-set calibration reveals cases B can't handle.

**Amendment 1 — pre-implementation review (metric pivot from Option B to peak strength).** Running `analyze_confluence` over the four confirmed days' stored landscapes (production DB, `orats_gex_landscape`) measured all three candidate metrics for each day's top cluster:

| Day  | Label   | score (B/pt) | max_gex ($B) | avg_fwhm | fwhm_ratio | mass_conc (±15pt) |
|------|---------|--------------|--------------|----------|------------|-------------------|
| 5/6  | target  | 16.2         | 612          | 75.5     | 2.00       | 0.122             |
| 5/7  | pin     | 13.1         | 712          | 109.0    | 2.89       | 0.111             |
| 5/18 | feature | 6.2          | 505          | 162.0    | 1.97       | 0.116             |
| 5/20 | feature | 3.0          | 125          | 84.0     | 2.23       | 0.121             |

Findings:

- **Option B's `fwhm_ratio` is anti-correlated with the labels.** The canonical pin (5/7) has the *highest* ratio (2.89); a feature day (5/18) has the *lowest* (1.97). Gating `pin` on a *low* fwhm_ratio is exactly backwards. Applying the spec's Option B thresholds yields only 2/4 correct (5/7 → feature, 5/18 → target). Root cause: 5/7's pin is a 1-7 DTE peak with FWHM 109 — roughly 3× the single-strike Gaussian ideal — because it is layered multi-strike mass. That breadth *is* the pinning capability (the Problem section says exactly this); normalizing it by DTE does not rescue it, because the breadth is not a DTE artifact.
- **Option C's `mass_concentration` is flat.** 0.111–0.122 across all four days — no discriminating power.
- **`max_gex` (peak GEX strength, $B) is monotonic in the correct order:** pin 712 > target 612 > feature 505 > feature 125. It is the only measured metric that separates all three tiers.

Decision: ship a **peak-strength classifier**, not Option B. This still honours the Problem section's thesis — "the model conflates peak sharpness with pinning capability" — but the fix is to *stop gating on sharpness at all*, not to normalize it. No FWHM term, absolute or relative, enters the quality gate.

Revised design:

```python
def classify_confluence_quality(max_gex: float) -> str:
    max_gex_b = max_gex / 1e9
    if max_gex_b >= 650:
        return "pin"
    if max_gex_b < 550:
        return "feature"
    return "target"
```

- `classify_confluence_quality` takes a single argument, `max_gex` (raw $, as already carried in the score record), not `(score, fwhm_ratio)`.
- **No new data-model field.** `max_gex` already exists in `score_confluence`'s output and in every `/api/gex-landscape` confluence item, so the classification is already inspectable from `quality` + `max_gex`. `fwhm_ratio` is not added — it proved non-predictive, and an unused field is dead weight.
- `score` keeps its current formula and meaning (`n_buckets × max_gex / max(avg_fwhm, 5)`); it still drives the `analyze_confluence` ranking. Only `classify_confluence_quality` stops consuming it.
- Thresholds 650 / 550 ($B): the 550 boundary is the midpoint of the highest feature (505) and the lone target (612); 650 sits 38 above the target and 55 below the lowest `pin` confluence (5/7's secondary cluster at 705 — the Problem section expects 5/7's 7371 confluence to tag `pin` too).

AC corrections:

- **AC #4** — `classify_confluence_quality` signature is `(max_gex)`. The empirical-anchor comment references the labeled set.
- **AC #5 / #6** — no new metric field. `score_confluence` and `analyze_confluence` outputs keep their existing field set unchanged; classification keys off the existing `max_gex` field.
- **Affected Files / Breaking-change advisory / AC #11** — the additive per-confluence field is dropped. The only response-shape change is the `quality` value domain.

The `target` tier rests on a single calibration point (5/6) — open questions #3 (whether `target` warrants its own tier) and #7 (re-assessment cadence) both stand. The "structural-bucket-dominated stretch day" pending coverage case is the known weak spot for a distance-blind, strength-only classifier; flagged for the post-merge re-assessment.

### Calibration set

The recalibration needs a labeled set of days, each tagged with the expected quality (`pin` / `target` / `feature`) based on **observed end-of-session price behavior, per the operational definitions above**. Labels must be defensible from session price charts — not from existing model output, which would be circular.

The GEX-landscape system has only been operational for a few days at the time of CR-011, so the initial set is small. The fixture format is designed for cheap incremental growth; new days are added as they're observed and labeled.

**Confirmed entries:**

- **5/6** — `target`. Spot opened 7326, magnet above at ~7350-7360 (~30pt away, within session range). Price drifted upward across the session toward the magnet. The marked level was the destination of the directional move, though not a tight pin.
- **5/7** — `pin`. Spot opened ~7400, fell to and locked in at 7353 (dominant 8-30 DTE peak). EOD range tight around the level. The canonical pin day.
- **5/18** — `feature`. Spot reference 7378, marked levels at 7510 and 7530 (132-152pt above, multi-sigma distance). Session range ~7370-7460; marked levels never approached. Regime tagged UNTETHERED — model itself flagging no directional pull. Coverage case: structural mass at distance.
- **5/20** — `feature`. Spot reference 7392, marked levels at 7456 and 7505 on a diffuse 30+ DTE bulge. 30+ DTE dominant at 45% but tagged untethered; 0DTE and 1-7 DTE in amplification regime. Session topped at ~7460 (just above 7456) but didn't engage with it as a destination — day's dynamics driven by the amplification underbelly, not the marked levels. Coverage case: diffuse mass + amplification underbelly.

**Pending entries (placeholders for coverage gaps):**

- **0DTE-dominated pin day** — a day where 0DTE owns 30%+ of GEX and the pin happens at or very close to spot. Sharp FWHM, small bucket DTE. The "easy" pin case the original thresholds were probably aimed at. Add when observed. Expected label: `pin`.
- **Structural-bucket-dominated stretch day** — 30+ DTE dominates and its peak is multi-sigma away from spot. Tests that the recalibration tags it as `feature` (magnet is structural, not today's range). Add when observed. Expected label: `feature`.
- **Catalyst day** — landscape said one thing, news (FOMC, CPI, etc.) overrode it. Tag `feature` regardless of structural read. Tests that the model can be "right about structure, wrong about outcome" without that distorting calibration. Add when observed. Expected label: `feature`.
- **Second target-grade day** — 5/6 is the only `target` example so far. One more would help anchor the threshold between `target` and `feature`. Add when observed. Expected label: `target`.

The fixture lives at `packages/shared/tests/fixtures/confluence_calibration.json`. Each entry contains:

```json
{
  "trade_date": "2026-05-07",
  "spot": 7400,
  "implied_move": 40,
  "dominant_bucket": "8-30 DTE",
  "top_cluster_center": 7353,
  "expected_quality": "pin",
  "rationale": "Price moved from ~7400 open to lock in at 7353 with tight last-hour range. Level was both destination and trap. EOD close at 7353."
}
```

The `rationale` field must describe **observed price behavior** (open, close, path, what the marked level did), not "the model said X" reasoning. Pending entries omit `expected_quality` and `rationale` but include a `coverage_case` field documenting what they're meant to test.

The calibration set is expected to grow over the weeks following CR-011's ship. A re-assessment of thresholds against the expanded set is planned for a few weeks post-merge — track in [[gex-landscape]] as a Phase 2 candidate.

### Affected Files

- `packages/shared/gex_landscape.py` — primary surface. `classify_confluence_quality` signature and thresholds change; `score_confluence` augmented to compute the additional metric (FWHM ratio for Option B). `analyze_confluence` output gains the new field. The `quality` field value domain changes from `pin-grade | drift-grade | waypoint` to `pin | target | feature`. The empirical-anchor comment in `classify_confluence_quality` is replaced with a reference to the labeled set, not single hardcoded days.
- `packages/shared/tests/test_gex_landscape.py` — new tests covering each labeled day's expected quality tag. Existing snapshot tests on `analyze_confluence` output need updates if the score field shape changes; all existing tests need their `quality` field values renamed to the new vocabulary.
- `packages/shared/tests/fixtures/confluence_calibration.json` (new) — labeled set of days with expected quality tags. Format described in the *Calibration set* section.
- `scripts/explore_gex_landscape.py` — `style_map`, line-width, and `quality_short` dicts updated to map the new label values to display styles (matching the *Display strings* table). The stdout printer's `[quality_tag]` formatting renders the new internal names.
- `react_price_preview/src/components/GexLandscapePanel.jsx` — `QUALITY_DASH` and `QUALITY_SHORT` maps updated to key off the new label values. Display strings `PIN | TGT | soft`. Exported `QUALITY_SHORT` is consumed by `PriceChart.jsx`, which gets the new strings automatically.
- `react_price_preview/src/components/PriceChart.jsx` — see Amendment 2: this file has its own quality→line-style map and hardcoded label literals, so it *does* need code changes.

**Amendment 2 — pre-implementation review (`PriceChart.jsx` needs code changes).** The drafted *Affected Files* entry assumed `PriceChart.jsx` only consumes the imported `QUALITY_SHORT` and needs no edits. Reading the file shows otherwise:

- `CONFLUENCE_LINE_STYLE` (a module-level map) is keyed on the old vocabulary — `'pin-grade'`, `'drift-grade'`, `waypoint` — and must be rekeyed to `pin` / `target` / `feature`.
- The default fallback `c.quality || 'waypoint'` must become `|| 'feature'`.
- The line-width literal `quality === 'pin-grade' ? 2 : 1` must compare against `'pin'`.

`PriceChart.jsx` is therefore an edited file in this CR. It still also gets the new `QUALITY_SHORT` strings for free via the import.

**Breaking change advisory.** The `quality` field's value domain changes. Any external or downstream consumer that hardcoded `pin-grade` / `drift-grade` / `waypoint` strings (none known today, but worth confirming) will break. The change is co-located: backend, frontend, and matplotlib script all update in the same CR. No backward-compat aliasing.

No endpoint contract changes structurally. The `/api/gex-landscape` response shape stays the same; only the *values* of the `quality` field on individual confluences change, plus the additive new field per confluence (`fwhm_ratio` for Option B, `mass_concentration` for Option C).

No DB schema changes. The recalibration is request-time logic; stored `peaks_by_bucket` already carries the per-peak FWHM that the new metric needs.

### Acceptance Criteria

**Calibration set passes:**

1. The labeled set at `packages/shared/tests/fixtures/confluence_calibration.json` exists and is checked in with the 4 confirmed entries (5/6, 5/7, 5/18, 5/20) plus pending placeholder entries for the 4 coverage gaps.
2. Running `analyze_confluence` over each confirmed day's stored landscape produces a top-cluster `quality` tag matching the entry's `expected_quality`:
   - 5/6's top cluster tags as `target`.
   - 5/7's top cluster (anchored at 7353) tags as `pin`.
   - 5/18's top cluster tags as `feature`.
   - 5/20's top cluster tags as `feature`.
3. A test exists for each confirmed labeled day in `test_gex_landscape.py` that loads the fixture, runs the classifier, and asserts the expected tag.

**Code surface:**

4. `classify_confluence_quality` signature reflects the chosen option (Option B: takes `score, fwhm_ratio`). The empirical-anchor comment block references the labeled set, not single hardcoded days.
5. `score_confluence` returns a record containing the new metric. The existing fields (`center_price`, `n_buckets`, `score`, `quality`, etc.) keep their current names and types.
6. `analyze_confluence` output's `confluences` list items contain the new metric. Field added in-place, no other field shape changes.
7. The `quality` field's value domain is `pin | target | feature` everywhere it appears (backend, tests, frontend, matplotlib script).
8. Display strings in `QUALITY_SHORT` (frontend) and `quality_short` (matplotlib) are `PIN | TGT | soft` respectively, mapped from the new internal labels.

**No regression on consumers:**

9. `vite build` from `react_price_preview/` passes. The React panel renders the new labels correctly: confluence lines for `pin` are solid, `target` are dashed, `feature` are dotted (matching the existing semantics, just renamed).
10. The matplotlib `_stacked.png` output renders the same — solid lines for `pin`, dashed for `target`, dotted for `feature`.

**Backward compatibility (only what's still backward-compatible):**

11. The `/api/gex-landscape` response shape is unchanged except for the `quality` field's value domain and the additive new field per confluence.

### Verification

**Automated:**

- Backend test suite expands by the calibration-set entries (4 new day-level tests, plus updates to existing snapshot tests for the renamed `quality` values). Target: from 176 to ~180-183 passing.
- New tests assert the quality tag for each confirmed labeled day's top cluster.
- Existing snapshot tests on `analyze_confluence` output get their `quality` field values updated to the new vocabulary. Where the snapshot also tests `score` or `fwhm`, those numeric values should remain unchanged (the metric formula doesn't change; only the threshold mapping does).

**Manual smoke against production DB:**

- Run `python scripts/explore_gex_landscape.py --date 2026-05-07 --spot 7400 --implied-move 40` and verify the stdout's CONFLUENCE section reports `[pin]` for the top cluster at 7353. Pre-CR-011, this reported `[waypoint]`.
- Same for the matplotlib `_stacked.png` output for 5/7: the 7353 confluence line renders as solid (pin style) with display string `PIN`.
- For 5/20: rerun and confirm the top cluster still tags `[feature]` with display string `soft`.
- For 5/6 and 5/18: rerun and confirm `[target]` and `[feature]` respectively.
- Spot-check the React panel: load 5/7 in `accuracy=high` mode and verify the confluence labels show "PIN" for the 7353 anchor.

**Calibration-set quality:**

- The labeled set should be reviewed before the implementation lands. Labels must be defensible from observed price behavior, not from existing model output. The fixture's `rationale` field documents the observed behavior — reviewer reads the rationale and the session chart side-by-side to confirm.

### Open design questions for pre-implementation review

1. **Which option to ship.** Recommend Option B; final choice made during pre-impl review with the calibration set in hand.
2. **Display string for `target`.** `TGT` (3-letter to match `PIN`) vs `TARGET` (full word). Spec ships with `TGT`; trivial to flip during impl if it looks bad on the chart.
3. **Whether to introduce a fourth quality tier.** Current proposal: no, keep three tiers. Flag if the labeled set turns up enough mid-range cases to warrant a fourth (e.g., between `target` and `feature`).
4. **Calibration fixture format.** JSON with per-day entries. Recommended.
5. **Whether Option B's `expected_fwhm` formula** should use `spread_coef × sqrt(dte) × 2.355` (Gaussian FWHM) or a calibrated proportionality constant. Recommend starting with the analytical formula and only tuning if calibration requires it.
6. **Whether the dominant-bucket DTE midpoints** (`0DTE → 0.5`, `1-7 → 4`, `8-30 → 19`, `30+ → 45`) are the right anchors. Recommend midpoints unless calibration shows a meaningful gap.
7. **Re-assessment cadence.** When to revisit the calibration thresholds against an expanded set. Recommend: a few weeks post-merge, after another ~10-15 sessions have been observed and labeled. Track as a Phase 2 candidate in `[[gex-landscape]]`.
