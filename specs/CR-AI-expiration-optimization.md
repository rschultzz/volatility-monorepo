# CR-AI — Expiration Optimization: Stage 2 Spec

> Spec frozen 2026-06-10. Stage 1 (days-to-touch diagnostic) and σ-definition confirmation
> are recorded in the CR vault note and inform every decision below.

## Purpose

CR-AH validated the magnet-above debit chase at a fixed 15-DTE expiry. Stage 2 determines
empirically whether a shorter DTE (or distance-matched DTE schedule) improves risk-adjusted
return, holding the sample fixed so no sample-composition confound is introduced.

## Settled scope (do not re-open)

- **Date set:** the existing CR-AH clean magnet-above sample (A-bucket debit-usable dates),
  unchanged. Holding it fixed is the whole confound-free design. Widening is out of scope.
- **Structure:** debit call spread (long target−10, short target). Exit: held-to-close.
  Only expiration varies.
- **Banding:** frozen 1-day-σ near/mid/far, computed on read.
  Formula: `sigma = (drift_target − spot) / compute_implied_move(spot, iv, dte=1.0)`
  where `iv` = 06:33 PDT open-straddle ATM IV. Bands: near σ<1.5, mid 1.5–2.0, far ≥2.0.
  **There is NO re-banding step.** A date's band is identical across all tested DTEs.
- **Inherited caveat:** thin holdout (~18 dates). Outcome is directional, not conclusive
  for the holdout partition. Report CIs everywhere; flag narrow wins as within-noise.

## Stage 1 findings (context for Stage 2)

- Overall days-to-touch: median=1, p90=9, max=19. 39% same-day.
- By band: near median=0 (p90=5), mid median=1 (p90=8), far median=3 (p90=11).
- Touch-by-DTE survival: near 91% by day 5, mid 83% by day 5, far 69% by day 5 / 88% by day 10.
- No-touch rate: near 7.6%, mid 20.7%, far 31.4%.
- DTE grid from Stage 1: {5, 7, 10, 15}. 3 excluded (far 51% survival). 21 optional.

## Step 0 — diagnosis gate (must be answered BEFORE backfilling)

### 0a. Confirm the CR-AH clean date set

Enumerate the A-bucket debit-usable dates from `orats_options_minute` — the same dates
used in CR-AH Step 4 analysis. Report n (expected ~110), train/holdout split at 2025-08-12,
and per-band breakdown.

### 0b. ORATS coverage probe for re-proposed strikes at each DTE

For each date × DTE {5, 7, 10, 15, 21}: compute the holiday-aware expiry date, derive the
debit spread strikes (long = round5(drift_target)−10, short = round5(drift_target)), then
count `orats_options_minute` rows present on entry day for both legs. Report the coverage
matrix as: for each (date_count, DTE) cell, how many dates have entry-day bars for both
legs. Holes must be identified BEFORE backfilling, not discovered during.

### 0c. Re-proposal logic confirmation

At each DTE, strike derivation: same as CR-AH debit logic — round drift_target to nearest 5pt,
long = that −10, short = that. The only change is the holiday-aware expiry date.
drift_target and band do NOT change with DTE. Confirm this is correct by inspecting one
sample date at two different DTEs.

### 0d. Contradiction stop

If Step 0 reveals that coverage at any DTE is materially below the fixed sample size, or
that the re-proposal logic requires a different strike set than expected, HALT and report
before proceeding.

## Implementation plan

### Step 1 — re-propose and backfill

For every (date × DTE) pair with confirmed coverage:
- Compute holiday-aware expiry date using `nth_business_day(trade_date, dte)`.
- OPRA symbols: long = `format_opra("SPX", expiry, "C", short_strike - 10)`,
  short = `format_opra("SPX", expiry, "C", short_strike)`.
- Fetch via `fetch_option_bars` using the backfill safety protocol:
  `get_backfill_db_conn() + assert_role_or_die() + backfill_run()`.
- Fetch window: full entry-day RTH (06:30–13:00 PT) + expiry settlement (12:50–13:00 PT).
- Touch window: same ES-based touch as CR-AH (DTE-independent: reuse existing
  `ironbeam_es_1m_bars` touch detection).

### Step 2 — P&L computation

For each (date × DTE) pair:
- Entry credit = mid of (short − long) at first valid entry-day minute.
- Settlement = mid of (short − long) at last available expiry minute.
- close_pnl = entry_credit − settlement_cost (positive = win).
- alive_at_touch: does the expiry date fall after the touch_session date?
  (touch_session from Stage-1 days_to_touch; alive = expiry >= touch_session for rth/gap touches)
- % return on premium = close_pnl / (10 − entry_credit) × 100
  (10pt spread width; denominator = max possible loss = spread_width − net_credit)

### Step 3 — Analysis

Three required deliverables:

**(i) (DTE × band) P&L grid:**
For each DTE × {near, mid, far}: n_filled, mean_pnl, win_rate, wilson_lo/hi,
mean_entry_credit, mean_%_return_on_premium. Plus "all bands" aggregate row.

**(ii) Keep-or-drop-far verdict:**
For the far band: report P&L at every DTE. Does far earn positive mean P&L at any DTE?
If far is negative or flat at every DTE, the verdict is DROP FAR from the strategy.

**(iii) Early-vs-late-touch conditioning cut:**
Band medians from Stage 1: near=0d, mid=1d, far=3d.
For each (band × DTE): split into early-touch (days_to_touch ≤ band_median) and
late-touch (days_to_touch > band_median, including no_touch as "never").
Report mean P&L for each slice. This diagnoses whether losses are from
never-touching (signal problem) vs touching-then-reversing (DTE/exit problem).

**(iv) Distance-matched schedule scoring:**
Schedule A: near=5, mid=7, far=10.
Schedule B: near=5, mid=10, far=15.
Compute mean P&L, win rate, % return for each schedule on the full fixed sample.
Frame as candidates among several; compare to every flat DTE, to fixed-15, and to
the live DTE_TARGET_BY_BUCKET rule (which maps by GEX bucket, not distance band).

## DTE_TARGET_BY_BUCKET baseline (live engine rule)

From `packages/shared/strategy_templates.py`. The live engine assigns DTE by GEX bucket:
- 0DTE bucket: DTE=1
- 1–7 DTE bucket: DTE=3
- 8–30 DTE bucket: DTE=15 (coincides with CR-AH fixed-15)
- 30+ DTE bucket: DTE=45
The bucket is determined by which GEX-bucket dominates for that date's landscape.
For comparison: look up the dominant_bucket_at_classification for each CR-AH date
and apply DTE_TARGET_BY_BUCKET to get the "live rule DTE" per date.

## Constraints / gates

- Safety protocol: `get_backfill_db_conn()` (BACKFILL_DATABASE_URL), `assert_role_or_die()`,
  `backfill_run()`, `update_run_smoke()`. No use of DATABASE_URL for writes.
- Contradiction stop: any Step 0 finding that contradicts scope → HALT before backfilling.
- No DB commits before `backfill_run()` context manager.
- No changes to the live engine in this CR.

## Step 0 findings (appended before backfill)

### 0a — CR-AH clean date set confirmed

n=110  (near=41, mid=36, far=33)  train=87 / holdout=23 (split 2025-08-12)

Reproduction logic: magnet-above rows from `bt_daily_features` (feature_version=v0.6.0-openiv)
→ stride-select 50/band → A-bucket filter (both DTE=15 legs have entry-day bars in
`orats_options_minute`). Exactly matches the CR-AH Step 4 analysis set.

### 0b — ORATS coverage probe for candidate DTEs

Probe method: for 3 spread-representative clean dates (2023-05-01, 2024-02-07, 2024-12-13),
live-fetch entry-day bars for both legs at each candidate DTE and count bars in
`orats_options_minute`.

| Date       | DTE=5 | DTE=7 | DTE=10 | DTE=21 |
|------------|-------|-------|--------|--------|
| 2023-05-01 | 390✓  | 390✓  | 390✓   | 390✓   |
| 2024-02-07 | 390✓  | 390✓  | 390✓   | 390✓   |
| 2024-12-13 | 390✓  | 390✓  | 390✓   | 404✗   |

DTE=21 failure: 2024-12-13 × DTE=21 → expiry 2025-01-16 → ORATS returns HTTP 404 for
SPX250116 calls. Affects any trade date where DTE=21 crosses into 2025 expiry territory.

DTE=15: 110/110 already cached from CR-AH Step 2 (no fetch needed).

**Decision: drop DTE=21 (was optional per spec; coverage gap confirmed for 2025 expiries).**
Final backfill grid: {5, 7, 10, 15}.

### 0c — Re-proposal logic confirmed

Strikes are DTE-invariant. For any clean date: `short_strike = round5(drift_target)`,
`long_strike = short_strike − 10`. Only `expiry_date = nth_business_day(trade_date, dte)`
changes. Example — 2023-05-01, short_strike=4205, long_strike=4195:
- DTE=5:  SPX230508C04205000 / SPX230508C04195000
- DTE=15: SPX230522C04205000 / SPX230522C04195000 (CR-AH baseline)

drift_target and band are DTE-agnostic (computed on read, not re-banded per DTE).

### 0d — Contradiction verdict

No contradiction-stop. Primary grid {5, 7, 10, 15} has full coverage.
DTE=21 dropped (optional, ORATS gap confirmed for 2025-spanning expiries).

Note: `bt_daily_outcomes.dominant_bucket_at_classification` stores buckets with " DTE"
suffix (e.g. "8-30 DTE") while `DTE_TARGET_BY_BUCKET` uses bare keys (e.g. "8-30").
The live-rule comparison in the analysis note strips the suffix. All 110 clean dates
map to "8-30 DTE" bucket → live DTE=15, matching the CR-AH baseline flat-DTE=15 row.

## Results (appended after backfill)

## Stage 2 Results

*n_clean_dates=110  DTEs=[5, 7, 10, 15]  filled=427  held-to-close*

### DTE × Band P&L (all partitions)

| DTE | Band | n | mean_pnl | win% | wilson_lo | %_return |
|-----|------|---|----------|------|-----------|----------|
| 5 | near | 40 | +0.38 | 52.5% | 0.37 | +17666.7% |
| 5 | mid | 36 | -1.45 | 22.2% | 0.12 | +8277.7% |
| 5 | far | 33 | -1.36 | 21.2% | 0.11 | +3421.4% |
| 5 | all | 109 | -0.75 | 33.0% | 0.25 | +10252.9% |
| 7 | near | 39 | -0.77 | 38.5% | 0.25 | +16404.4% |
| 7 | mid | 35 | -2.23 | 34.3% | 0.21 | +6398.8% |
| 7 | far | 31 | +1.47 | 41.9% | 0.26 | +3913.6% |
| 7 | all | 105 | -0.60 | 38.1% | 0.29 | +9381.4% |
| 10 | near | 40 | -3.30 | 35.0% | 0.22 | +6501.9% |
| 10 | mid | 33 | -2.18 | 27.3% | 0.15 | +11193.2% |
| 10 | far | 32 | +0.89 | 46.9% | 0.31 | +7601.0% |
| 10 | all | 105 | -1.67 | 36.2% | 0.28 | +8311.3% |
| 15 | near | 39 | +1.50 | 59.0% | 0.43 | +25148.9% |
| 15 | mid | 36 | -1.01 | 41.7% | 0.27 | +21515.6% |
| 15 | far | 33 | +2.24 | 36.4% | 0.22 | +28908.5% |
| 15 | all | 108 | +0.89 | 46.3% | 0.37 | +25086.6% |

<!-- Stage 2 P&L results will be appended here -->
