---
type: cr
cr_id: CR-G
title: Edge Visualization and P&L Engine
aliases: ["CR-G — Edge Visualization and P&L Engine", "CR-G"]
status: active
started: 2026-05-27
sequence_number: 026
run_mode: interactive
phase: 4
size: large
estimated_days: 5-7
data_safety_class: read_only
dependencies: [CR-C, CR-D, CR-I]
depended_on_by: [CR-H]
branch_name: feat/CR-026-cr-g-edge-visualization-pl-engine
tags: [dash, cr, frontend, p&l, edge-ratio, iv-overlay, py-vollib, forward-compat]
---

# CR-026 — Edge Visualization and P&L Engine

> [!info] Pre-activation amendments — 2026-05-27 design conversation
> Captured before CR-G activation during chat-based design review. Code applies these during the activation spec-freeze step (move/rename + apply amendments + commit). Original spec text below is preserved with strikethrough/annotation where modified, per the pattern used in CR-I's 2c revert. Amendments summary:
>
> 1. **Stale CR-letter references throughout the spec body.** Spec was drafted before the current CR-I (Close Distribution Decomposition) was inserted. References to "CR-I" in the body (e.g., "CR-I later passes other times to get pre-expiration P/L") refer to the *original* CR-I (pre-expiration P/L work), which is now CR-R. Replace "CR-I" with "CR-R" wherever the reference is to pre-expiration P/L work. Leave references to the new CR-I (Close Distribution) intact where they're being added by these amendments.
>
> 2. **Frontmatter dependencies.** Update `dependencies: [CR-C, CR-D]` to `dependencies: [CR-C, CR-D, CR-I]` per the Roadmap. The new CR-I (Close Distribution) is an upstream dependency — CR-G's structural probability work composes its pattern label and post-touch positions.
>
> 3. **New scope: session OHLC backfill on bt_daily_outcomes.** Adds a Step 0-A (executed before Step 1) that adds DDL + backfill for session-level open/high/low/close at each of T+1, T+5, T+15 (12 new columns total). This is the foundation for per-range probability distributions — close-only is too narrow because intraday excursion matters for any range-based trade thesis (iron condor breach, pin time-at-level, etc.). Bars are already available in `ironbeam_es_1m_bars`; no new ingestion. Mirrors CR-I's Step 2 backfill pattern with the known mitigations (UPDATE grants pre-check, view refresh, BACKFILL_DATABASE_URL).
>
> 4. **New shared module: `packages/shared/structural_distribution.py`.** Takes per-analogue OHLC + a trade-thesis range, returns analogue-derived probability for that range. Replaces the implicit "structural probability per range" assumption in Step 3 with an explicit, reusable module that any trade thesis can call. Per-regime trade-thesis-to-range mapping lives here as a small registry (e.g., bounded → stays-in-range; magnet-above → above-magnet; pin → within-tolerance).
>
>    Motivating example surfaced 2026-05-27: a Bounded regime currently surfaces "100% touch rate" against a single cluster level. For an iron condor on that regime, what actually matters is "probability of staying in range," not "probability of touching one of the bounds." The current single-level binary metrics structurally can't answer trade-thesis questions for range-based trades. The new module fixes this uniformly across regimes.
>
> 5. **Chart layer toggles.** Step 5's ProposalEdgeChart gains a chip row of independently toggleable layers (edge zones, IV curve, P/L curve, reference lines). Defaults visible: P/L curve, edge zones, current-spot reference. Defaults hidden: IV curve, breakeven/max-profit/max-loss markers.
>
>    **Also folded into CR-G:** layer toggles for the existing today-setup main candlestick chart (the GEX overlay lines visible in the screenshot from the 2026-05-27 review). Same `<LayerToggleChips>` component reused. Toggles for the GEX-level overlays on the main chart so the user can declutter when the new ProposalEdgeChart's information density would otherwise overwhelm the page. Originally considered as a separate tiny CR; folded in because the UX pattern, component code, and activation window are all shared.
>
> 6. **Confirm with the Step 0 inventory:** the `orats_monies_minute` table is *not* used by CR-G (it's reserved for CR-R's pre-expiration P/L backtesting). Just `ironbeam_es_1m_bars` for the structural side and the live option chain for the market-implied side. Document this in Step 0 so it's clear what data CR-G consumes vs what's deferred to CR-R.
>
> Wherever the original spec body says something the amendments above modify, treat the amendment as canonical and the original as strikethrough. Specific in-body amendments are tagged inline with "[AMENDED 2026-05-27 — see Pre-activation amendments]."

## Goal

Make the four-question framework visible in one chart: where (structural targets), when (horizon), structural probability (from analogues), market-implied probability (from current IV surface). Render a P/L profile with edge zones highlighted where structural prob exceeds market-implied prob. Lay the foundation for [[CR-H — Interactive Editing]] and future pre-expiration P/L work.

## Context

This is the visual integration point of the operating framework. By the time CR-G lands:
- [[CR-A — Landscape Backfill]] has expanded the corpus
- [[CR-B — Outcome Computation]] has populated historical outcomes
- [[CR-C — Probability Output on Proposals]] surfaces structural probabilities per proposal
- [[CR-D — Vol Surface Features]] populates the live IV state
- [[CR-E — Vol Regime Context Panel]] makes the buy/sell signal visible
- [[CR-F — Debit Credit Variants]] produces variant pairs with preferred flags

CR-G ties them together visually. The user opens a proposal and sees: P/L profile against price axis, IV curve overlaid (showing where the market thinks risk is), edge zones shaded (where the analogues say price is more likely to go than the market is pricing). One image, four questions answered.

This CR is large because it touches the chart layer, introduces a pricing library, and is the last visual integration step before interactive editing. It also bakes in forward-compatibility for pre-expiration P/L (CR-R) so that future work doesn't require rewriting the engine.

## Step 0 — Diagnosis and forward-compat decisions (no commits)

This CR's Step 0 is heavier than usual because architectural decisions made now constrain CR-H and CR-R.

> **[AMENDED 2026-05-27 — see Pre-activation amendments]** `orats_monies_minute` is explicitly NOT used by CR-G. CR-G data sources: `ironbeam_es_1m_bars` (for session OHLC backfill in Step 0-A and structural probability computation) and the live option chain (for market-implied probability). `orats_monies_minute` is reserved for CR-R's pre-expiration P/L backtesting.

### Forward-compat decisions for pre-expiration P/L

The P/L engine built here must support pre-expiration P/L computation in the future without rewriting. Lock these decisions:

1. **API signature includes `evaluation_time`.** The core function is:
   ```python
   def compute_position_pl(
       legs: list[Leg],
       price_grid: np.ndarray,
       evaluation_time: datetime,  # for MVP: always set to expiration of shortest leg
       market_state: MarketState,  # IV per leg, rate, etc.
   ) -> np.ndarray:
       """Returns P/L value for each price in price_grid at evaluation_time."""
   ```
   For MVP, callers pass `evaluation_time = expiration_of_shortest_leg`, giving expiration P/L. ~~CR-I later passes other times to get pre-expiration P/L.~~ [AMENDED 2026-05-27 — see Pre-activation amendments] **CR-R later passes other times to get pre-expiration P/L.** No callsite rewrites needed.

2. **Legs carry IV reference per leg, not a single position-level IV.** Each leg has its own `iv` attribute used by the pricing engine. Mixed-expiration positions get correct vol per leg.

3. **Pricing library: `py_vollib`.** Standard Black-Scholes implementation with greeks. Wired up in this CR, used for expiration P/L (which only needs intrinsic value but routing through the engine ensures the same code path works for pre-expiration). Greeks (delta, gamma, theta, vega, rho) are computed and returned even though only P/L renders in MVP.

4. **Chart layer accepts list of P/L curves, not a single one.** The render component takes `curves: [{label, color, data: [...]}, ...]`. For MVP, only one curve passes through (expiration P/L). CR-R will pass multiple (P/L at multiple times). No chart rewrite needed.

5. **Position state model includes `evaluation_time` field.** Even though MVP UI doesn't expose it as a control, the position state object carries the field. ~~CR-I and CR-H add the UI~~ [AMENDED 2026-05-27 — see Pre-activation amendments] **CR-R and CR-H add the UI**; the model already supports it.

### Other Step 0 decisions

6. **Edge zone definition.** An "edge zone" is a contiguous price range where:
   - Structural probability of touching that range (from analogues) exceeds
   - Market-implied probability of touching that range (from current IV surface)
   - by more than a threshold (default: 1.3x ratio, i.e., 30% relative edge)
   
   Zones are computed once per proposal load, cached.

7. **IV curve representation.** "IV by strike" curve is fetched from the live option chain — same data source the proposals already use to price the legs. Render as a secondary y-axis on the chart, dashed line, distinguishable from P/L curve.

8. **Color scheme for edge zones.**
   - Strong positive edge (ratio > 2.0): green-shaded band on price axis
   - Moderate positive edge (1.3-2.0): light-green band
   - Neutral (0.7-1.3): no shading
   - Moderate negative (0.5-0.7): light-red band (don't trade this range)
   - Strong negative (< 0.5): red band
   
   Bands run as faint vertical stripes behind the P/L curve, on the price axis.

9. **Market-implied probability per range.** Use the standard Breeden-Litzenberger interpretation: probability mass is the second derivative of the call price with respect to strike. For each contiguous price range, integrate the implied distribution to get market-implied probability. Helper function in `packages/shared/implied_distribution.py`.

10. **Confirm CR-C output schema.** Structural probability needs to be available per *range* not just per *point*. Confirm CR-C produces analogues' outcomes in a way that lets us bucket their realized prices into ranges and compute hit rates per range. If not, this CR adds a thin aggregation layer.

    > **[AMENDED 2026-05-27 — see Pre-activation amendments]** Per the amendments, CR-C and CR-I produce binary touch/close/position metrics, not per-range distributions. The new `packages/shared/structural_distribution.py` module (Step 3 amendment) is the explicit solution: takes per-analogue OHLC (from the Step 0-A backfill) + a trade-thesis range, returns analogue-derived probability for that range. Read `packages/shared/probability.py` and `packages/shared/knn.py` during Step 0 to confirm the current data shape empirically and document the inventory.

## Step 0-A — Session OHLC schema + backfill

> **[AMENDED 2026-05-27 — see Pre-activation amendments]** New prerequisite step, inserted before Step 1. Must complete (DDL + backfill + smoke check) before Step 1 begins.

**Commit:** `cr-g/step-0a: session OHLC DDL + backfill on bt_daily_outcomes`

### Schema additions to `bt_daily_outcomes`

12 new nullable columns (all REAL):

```sql
ALTER TABLE bt_daily_outcomes
  ADD COLUMN session_open_t1   REAL,
  ADD COLUMN session_high_t1   REAL,
  ADD COLUMN session_low_t1    REAL,
  ADD COLUMN session_close_t1  REAL,
  ADD COLUMN session_open_t5   REAL,
  ADD COLUMN session_high_t5   REAL,
  ADD COLUMN session_low_t5    REAL,
  ADD COLUMN session_close_t5  REAL,
  ADD COLUMN session_open_t15  REAL,
  ADD COLUMN session_high_t15  REAL,
  ADD COLUMN session_low_t15   REAL,
  ADD COLUMN session_close_t15 REAL;
```

Verify `backfill_run_id UUID` already exists on `bt_daily_outcomes` (it does from CR-I Step 2a — confirm before running DDL).

### Pre-flight checklist (per `ddl-propagation-to-dependent-objects` FU)

Run interactively via app role before the backfill:

1. Apply DDL above
2. `GRANT UPDATE (session_open_t1, session_high_t1, session_low_t1, session_close_t1, session_open_t5, session_high_t5, session_low_t5, session_close_t5, session_open_t15, session_high_t15, session_low_t15, session_close_t15) ON bt_daily_outcomes TO dash_backfill_writer;`
3. `CREATE OR REPLACE VIEW bt_daily_outcomes_active AS SELECT * FROM bt_daily_outcomes WHERE active = TRUE;`
4. Verify: `SELECT column_name FROM information_schema.columns WHERE table_name = 'bt_daily_outcomes' ORDER BY ordinal_position;` — confirm all 12 new columns present

### Backfill runner

`scripts/cr_g_backfill_session_ohlc.py` — follows the unattended backfill protocol from CLAUDE.md:

```python
from packages.shared.backfill_safety import (
    get_backfill_db_conn, assert_role_or_die, backfill_run, update_run_smoke,
)

conn = get_backfill_db_conn()   # uses BACKFILL_DATABASE_URL
assert_role_or_die(conn)        # hard-fail if role != dash_backfill_writer

with backfill_run(conn, "CR-G") as run_id:
    # ... iterate bt_daily_outcomes WHERE reached_touch = TRUE
    # ... for each anchor: fetch ironbeam_es_1m_bars at trade_date + days_to_reach + N sessions
    # ... FIRST(open), MAX(high), MIN(low), LAST(close) per session day
    # ... UPDATE bt_daily_outcomes SET session_{ohlc}_tN = ... WHERE id = ...
    update_run_smoke(conn, run_id, smoke_results, "self-assessment")
```

Data source: `ironbeam_es_1m_bars`. For each anchor row where `reached_touch = TRUE`:
- T+1 session OHLC: bars on the calendar date corresponding to `days_to_reach + 1` sessions after touch
- T+5: bars on `days_to_reach + 5` sessions after touch
- T+15: bars on `days_to_reach + 15` sessions after touch
- Aggregate: FIRST bar's open, MAX high, MIN low, LAST bar's close for that session day
- NULL when T+N exceeds available bars (same denominator-per-timeframe logic as CR-I position columns)

Pre-backfill: `SELECT current_user` must equal `dash_backfill_writer`. INSERT into `bt_backfill_runs` with `cr_id='CR-G'`, capture `run_id`.

Python: `python -u` launch; `cursor.execute()` not `conn.execute()`; explicit `conn.commit()` per batch.

### Smoke check (before proceeding to Step 1)

Hand-pick 3-5 anchor days from different buckets and manually verify OHLC values match bar data:
- Pull T+1 session bars from `ironbeam_es_1m_bars` for each hand-picked anchor
- Verify FIRST(open), MAX(high), MIN(low), LAST(close) match the stored columns
- Surface a table in Step 0-A's status update before marking Step 0-A complete

**Deliverable:** DDL applied, grants issued, view refreshed, backfill complete, smoke table surfaces 3-5 verified anchor days.

## Step 1 — Pricing engine

**Commit:** `cr-g/step-1: implement compute_position_pl with py_vollib`

Create `packages/shared/pricing/engine.py`.

- Install py_vollib dependency
- Implement `compute_position_pl(legs, price_grid, evaluation_time, market_state)`:
  - For each leg: compute leg value at each grid price at `evaluation_time` using py_vollib
  - Sum across legs to get position value
  - Subtract initial debit / add initial credit to get P/L
- Also expose `compute_position_greeks(legs, spot, evaluation_time, market_state)` returning dict of net greeks
- Unit tests with synthetic legs: long call at-the-money at expiration should produce intrinsic value; same call at T-1 day should produce intrinsic + small extrinsic

**Deliverable:** engine importable, tested for expiration and pre-expiration evaluation_times.

## Step 2 — Implied distribution helper

**Commit:** `cr-g/step-2: implement implied_distribution.py for market-implied probability`

Create `packages/shared/implied_distribution.py`.

- `compute_implied_pdf(option_chain, expiration) -> dict[strike, prob_density]` via Breeden-Litzenberger
- `compute_implied_prob_in_range(pdf, lower, upper) -> float` via integration
- Handle sparse strikes via interpolation (cubic spline on log-prices is standard)
- Unit tests with synthetic chains: lognormal-distributed BSM chain should recover near-lognormal PDF

**Deliverable:** function returns market-implied probability for any price range.

## Step 3 — Edge zone computation

**Commit:** `cr-g/step-3: implement edge zone classification + structural_distribution module`

> **[AMENDED 2026-05-27 — see Pre-activation amendments]** Step 3 now creates two modules: `packages/shared/structural_distribution.py` (new) and `packages/shared/edge_zones.py` (original). The structural distribution module is the bridge between per-analogue OHLC (from Step 0-A) and per-range probability.

### New: `packages/shared/structural_distribution.py`

Per-regime trade-thesis-to-range mapping registry:

```python
REGIME_RANGE_REGISTRY = {
    "magnet-above":   lambda ctx: (ctx.magnet_level, float("+inf")),
    "magnet-below":   lambda ctx: (float("-inf"), ctx.magnet_level),
    "magnetic-pin":   lambda ctx: (
                          ctx.pin_level - 0.25 * ctx.implied_move_1d,
                          ctx.pin_level + 0.25 * ctx.implied_move_1d,
                      ),
    "bounded":        lambda ctx: (ctx.lower_bound, ctx.upper_bound),
}
```

Tolerance convention for `magnetic-pin`: `0.25 × implied_move_1d` — matches CR-I's tolerance convention from `outcomes.py`.

For `bounded`: range is the inner range between the two cluster levels (the iron condor's "safe zone").

Function signature:

```python
def compute_structural_prob_in_range(
    analogue_ohlc_set: list[dict],   # [{open_t1, high_t1, low_t1, close_t1, ...}, ...]
    lower: float,
    upper: float,
    timeframe: int,                   # 1, 5, or 15
) -> tuple[float, tuple[float, float]]:
    """
    Returns (point_estimate, (wilson_lo, wilson_hi)) for the fraction of
    analogue sessions where the session range [low, high] intersects [lower, upper].
    """
```

Confirm Step 0 diagnostic surfaced whether all current regime types are covered by the four starting mappings. Flag any unmapped regime type as a stop condition requiring user input.

### Original: `packages/shared/edge_zones.py`

- Given: structural probabilities per range (from `structural_distribution.py`) + market-implied probabilities per range (from Step 2)
- Compute edge ratio per range = structural_prob / market_implied_prob
- Classify into bands per Step 0 decision (strong positive, moderate, neutral, moderate negative, strong negative)
- Return contiguous zones with their classification

**Deliverable:** `structural_distribution.py` importable + tested; `edge_zones.py` returns `{lower, upper, edge_ratio, classification}` list.

## Step 4 — Backend endpoint for proposal P/L data

**Commit:** `cr-g/step-4: /api/proposals/<id>/pl-data endpoint`

Endpoint returns everything the chart needs in one payload:

```json
{
  "price_grid": [...],
  "pl_curves": [{"label": "At expiration", "evaluation_time": "...", "data": [...]}],
  "iv_curve": [{"strike": ..., "iv": ...}, ...],
  "edge_zones": [{"lower": ..., "upper": ..., "edge_ratio": ..., "classification": ...}, ...],
  "structural_prob_by_range": [...],
  "market_implied_prob_by_range": [...],
  "current_spot": ...,
  "key_levels": {"breakeven": [...], "max_profit_price": ..., "max_loss_price": ...}
}
```

For MVP: `pl_curves` is a list of one element (expiration P/L). Future CR-R will add more curves; this schema doesn't change.

**Deliverable:** endpoint returns full chart data in one call.

## Step 5 — Frontend chart component

**Commit:** `cr-g/step-5: ProposalEdgeChart component + LayerToggleChips`

> **[AMENDED 2026-05-27 — see Pre-activation amendments]** Step 5 now includes a `<LayerToggleChips>` component with independently toggleable layers.

Create `frontend/src/components/ProposalEdgeChart.jsx`.

Uses the project's existing chart library (probably the same one as the landscape chart — verify and reuse).

Layers, back to front:
1. Edge zone vertical bands (colored stripes behind everything)
2. P/L curves (one in MVP, list to support multiple)
3. IV curve on secondary y-axis (dashed line)
4. Reference lines: current spot (vertical, solid), breakeven(s) (vertical, dotted), max profit/loss prices (markers)
5. X-axis: price; primary Y: P/L (dollars); secondary Y: IV (vol points)

### Layer toggle chips (`<LayerToggleChips>`)

Create `frontend/src/components/LayerToggleChips.jsx` — small reusable chip row of independently toggleable boolean layers.

ProposalEdgeChart defaults:
- **Visible by default:** P/L curve, edge zones, current-spot reference line
- **Hidden by default:** IV curve, breakeven/max-profit/max-loss markers

Chips live above the ProposalEdgeChart inline (not in a unified bar shared with the main candlestick chart — the two charts' toggles are logically independent).

Interactivity:
- Hover anywhere on the chart → tooltip shows: price, P/L, IV at that strike, structural prob to reach, market-implied prob to reach, edge ratio
- Click an edge zone band → small popover with "structural N=K analogues, M reached this range" and the prob comparison

**Deliverable:** chart renders cleanly with all layers; toggle chips control visibility; tooltip works.

## Step 6 — Wire into proposal detail view + main-chart GEX toggles

**Commit:** `cr-g/step-6: integrate ProposalEdgeChart into proposals UI + main-chart GEX layer toggles`

> **[AMENDED 2026-05-27 — see Pre-activation amendments]** Step 6 adds two things: (a) the original ProposalEdgeChart wire-up, and (b) GEX overlay layer toggles on the existing today-setup main candlestick chart using the same `<LayerToggleChips>` component.

### 6a — ProposalEdgeChart wire-up

- On expanding a proposal card, fetch from `/api/proposals/<id>/pl-data`
- Render `ProposalEdgeChart` inline
- Loading skeleton during fetch; clear error state on failure
- Legend explains what edge zones mean ("Green = structural prob exceeds market-implied by 30%+")

### 6b — Main-chart GEX layer toggles

- Add `<LayerToggleChips>` above the existing today-setup main candlestick chart
- Toggleable layers: GEX overlay lines (the strike-level GEX bands currently always visible)
- Defaults: all GEX layers visible (preserves current behavior — no regressions)
- User can toggle GEX off to declutter when ProposalEdgeChart is expanded below
- Mobile: chips collapse to a single "Show/Hide GEX" toggle to preserve screen space

**Deliverable:** every proposal card shows the full chart when expanded; main chart has GEX layer toggles.

## Step 7 — Greeks display (rendered but understated)

**Commit:** `cr-g/step-7: render net greeks below chart`

Below the chart, a compact row: delta, gamma, theta, vega.

For MVP, these are net greeks of the position at current spot at expiration evaluation_time (which means theta and vega will be ~0 at expiration — that's correct and intentional; ~~pre-expiration values will appear in CR-I~~ [AMENDED 2026-05-27 — see Pre-activation amendments] **pre-expiration values will appear in CR-R**).

Small text note: "Greeks shown at current spot, at expiration evaluation. ~~Mid-life greeks land in a future update.~~ [AMENDED 2026-05-27 — see Pre-activation amendments] **Mid-life greeks land in CR-R.**"

**Deliverable:** greeks visible, accurate, and clearly contextualized as expiration values.

## Smoke tests

1. **Pricing engine accuracy.** Long single call at expiration: P/L curve matches max(S - K, 0) - debit exactly.
2. **Pricing engine pre-expiration.** Same call evaluated one day before expiration with realistic IV: P/L curve has smooth curvature, max P/L slightly less than intrinsic at deep ITM.
3. **Edge zone integrity.** Construct a known scenario: 1.5x edge ratio in a price range, 0.5x outside. Verify zones come back correctly classified.
4. **Implied PDF sanity.** For a BSM-priced chain, recover near-lognormal PDF.
5. **End-to-end render.** Load a real proposal in the app. Verify P/L curve, IV overlay, edge zones, and tooltip all appear and interact.
6. **Greeks display.** Long-call proposal at expiration evaluation should show theta ≈ 0, vega ≈ 0. Same proposal hypothetically at T-1 (via direct engine call) should show theta < 0, vega > 0.
7. **No regressions.** Existing proposal card behavior unchanged when chart isn't expanded.
8. **Step 0-A OHLC smoke.** 3-5 anchor days manually verified: stored session OHLC matches bar aggregates from `ironbeam_es_1m_bars`.
9. **Layer toggle chips.** Toggle IV curve off → curve disappears. Toggle back on → reappears. Main-chart GEX toggle off → GEX bands disappear. All state is component-local (no global side effects).
10. **structural_distribution.py regime coverage.** All four regime types (magnet-above, magnet-below, magnetic-pin, bounded) return non-null range tuples for representative inputs.

## Wrap criteria

- All steps (0-A through 7) committed on the branch
- Step 0-A backfill smoke-checked and confirmed (3-5 anchor day OHLC verification table in status updates)
- All smoke tests pass
- Forward-compat decisions from Step 0 documented in `packages/shared/pricing/README.md`
- [[Roadmap]] updated: CR-G marked complete; CR-H moved to ready
- A short ADR-style decision note saved to `decisions/` titled "Forward-compat decisions for pre-expiration P/L" capturing the 5 architectural choices from Step 0

## Status updates

(filled during execution)

## Open questions

- Should the chart support mixed-expiration positions in MVP, ~~or defer until CR-I?~~ [AMENDED 2026-05-27 — see Pre-activation amendments] **or defer until CR-R?** **Support architecturally — the engine handles it, the schema carries per-leg IV. Don't expose mixed-exp positions through the proposal generator yet; that's a future CR. But the engine handles them if a future CR or CR-H produces one.**
- Breeden-Litzenberger requires close-to-continuous strikes. For sparse chains (less liquid expirations), the PDF estimate is noisy. Acceptable for MVP? **Yes — sparse expirations will produce wider edge bands or no classification, which is honest. Future improvement: smooth via parametric vol model.**
- Edge ratio threshold of 1.3x for "moderate positive" — too tight? Too loose? **Start at 1.3x; calibrate after seeing real proposals against real outcomes for a few weeks. This is a knob that should be tuned empirically.**

## Related

- **Sessions:** [[2026-05-26 - CR-025 — Close Distribution Decomposition]] (post-touch positions feed structural_distribution.py)
- **Decisions:** [[2026-05-26 - CR-025 Retrospective]] (mid-execution correction pattern; design predictions as priors), [[2026-05-24 - Data Safety Protocol]] (governs Step 0-A backfill)
- **Open questions:** [[ddl-propagation-to-dependent-objects]] (pre-flight checklist for Step 0-A DDL)
- **Downstream:** [[CR-H — Interactive Editing]] (depends on this CR's position state model + eval_time field)
