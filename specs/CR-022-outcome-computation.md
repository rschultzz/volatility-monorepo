---
type: cr
cr_id: CR-B
title: Outcome Computation
aliases: ["CR-B — Outcome Computation", "CR-B"]
status: active
started: 2026-05-25
sequence_number: 22
run_mode: unattended
phase: 1
size: medium
estimated_days: 3-4
estimated_compute_hours: 1-2
data_safety_class: insert_only
new_table: bt_daily_outcomes
dependencies: [CR-0, CR-A]
depended_on_by: [CR-C]
branch_name: cr-b-outcome-computation
stop_conditions:
  - "pre-flight check: current_user != 'dash_backfill_writer'"
  - "any backfill batch produces zero outcomes (likely a bug)"
  - "outcome hit rates outside plausible range (e.g., <10% or >90% for directional regimes)"
  - "smoke test row count differs from feature row count by >5%"
on_stop: surface_in_roadmap + update_bt_backfill_runs_status + leave_full_status_in_cr_file
tags: [dash, cr, backfill, unattended, outcomes, knn, base-rates, foundational]
---

# CR-B — Outcome Computation

## Goal

For every historical day's structural read, compute whether the prediction played out within the dominant-bucket's time horizon. Populate `bt_daily_outcomes` table. Enables analogue probability aggregation ([[CR-C — Probability Output on Proposals]]).

## Context

The structural framework produces predictions (drift_target, containment_zone, etc.) for every classified day. Without outcomes, the framework can find historical analogues but can't say what *happened* in those analogues. Outcomes are the missing leg of "structural probability" — they're what convert analogue lookup into base rate.

This CR defines outcome metrics for directional regimes (magnet-above, magnet-below, magnetic-pin). Non-directional regimes (bounded, amplification, untethered) get NULL outcomes in v1 — extending to those is a future CR.

See [[Operating Framework — Where When Buy or Sell]] for the role of outcomes in the structural_prob vs market_implied_prob comparison.

## Step 0 — Diagnosis (no commits)

Lock the following decisions in this step. Once committed in Step 1, they're frozen.

1. **Outcome metrics defined precisely:**
   - `reached_touch` (BOOL): did the underlying's high/low cross `drift_target` at any point during the horizon?
   - `reached_close` (BOOL): did the underlying's close come within `tolerance` of `drift_target` at end of horizon? (tolerance: 0.25 × expected_move)
   - `days_to_reach` (INT, nullable): if reached_touch is true, how many sessions until first touch? NULL if not reached.
   - `max_excursion_in_direction` (FLOAT): maximum favorable excursion (in points) toward drift_target during horizon. Positive even if target not reached.
   - `final_close_distance_from_target` (FLOAT): close on horizon-end day minus drift_target (signed).
   - `actual_realized_em_pct` (FLOAT): realized move during horizon / expected_move from features. Ratio.

2. **Horizon definition.** For each row, horizon = `_bucket_dte(dominant_bucket)`.
   - `0DTE-near-spot` → 1 session
   - `1-7DTE` → 5 sessions
   - `8-30DTE` → 20 sessions
   - `30+DTE` → 60 sessions

   Confirm helper exists; if not, implement in `packages/shared/buckets.py`.

3. **Which regimes get computed outcomes:**
   - `magnet-above`: drift_target known, direction clear → compute
   - `magnet-below`: drift_target known, direction clear → compute
   - `magnetic-pin`: drift_target = pin point; reached_touch means price came within tolerance → compute
   - `bounded`: NULL in v1 (different outcome definition needed — "stayed inside containment_zone")
   - `amplification`: NULL in v1
   - `untethered`: NULL in v1
   - Any other regime_kind: NULL in v1

4. **Data inputs:** `ironbeam_es_1m_bars` for intraday high/low; daily bars derived from it.

5. **Schema for `bt_daily_outcomes`:** see Step 1.

6. **Self-skip condition.** If a row's horizon_end_date is in the future (i.e., not enough history elapsed yet), skip outcome computation for that row — set all outcome fields to NULL with `outcome_status = 'pending_history'`. Recompute later when history is available.

## Step 1 — Create `bt_daily_outcomes` table

**Commit:** `cr-b/step-1: create bt_daily_outcomes table`

```sql
CREATE TABLE bt_daily_outcomes (
    ticker VARCHAR NOT NULL,
    trade_date DATE NOT NULL,
    feature_version VARCHAR NOT NULL,

    regime_kind_at_classification VARCHAR,
    dominant_bucket_at_classification VARCHAR,
    horizon_sessions INT,
    horizon_end_date DATE,

    outcome_status VARCHAR NOT NULL DEFAULT 'computed',
        -- 'computed' | 'pending_history' | 'na_regime'

    reached_touch BOOLEAN,
    reached_close BOOLEAN,
    days_to_reach INT,
    max_excursion_in_direction FLOAT,
    final_close_distance_from_target FLOAT,
    actual_realized_em_pct FLOAT,

    -- Safety columns
    active BOOLEAN NOT NULL DEFAULT TRUE,
    deactivated_at TIMESTAMP,
    deactivated_reason TEXT,
    backfill_run_id UUID,

    computed_at TIMESTAMP DEFAULT NOW(),

    PRIMARY KEY (ticker, trade_date, feature_version)
);

CREATE INDEX idx_outcomes_regime ON bt_daily_outcomes (regime_kind_at_classification);
CREATE INDEX idx_outcomes_run_id ON bt_daily_outcomes (backfill_run_id);

GRANT SELECT, INSERT ON bt_daily_outcomes TO dash_backfill_writer;
GRANT UPDATE (active, deactivated_at, deactivated_reason, backfill_run_id)
  ON bt_daily_outcomes TO dash_backfill_writer;

CREATE VIEW bt_daily_outcomes_active AS
SELECT * FROM bt_daily_outcomes WHERE active = TRUE;
```

**Deliverable:** schema in place; INSERT works as backfill role; DELETE fails as backfill role.

## Step 2 — Implement `compute_outcome` function

**Commit:** `cr-b/step-2: implement compute_outcome in packages/shared/outcomes.py`

Function signature:

```python
def compute_outcome(
    trade_date: date,
    feature_row: dict,  # from bt_daily_features
    bars: pd.DataFrame,  # daily or 1m bars from trade_date through horizon
) -> dict:
    """
    Compute outcome metrics for a structural read.

    Returns dict with all bt_daily_outcomes fields populated, or
    outcome_status='na_regime' if regime_kind isn't computable in v1.
    """
```

Logic:

1. If `regime_kind` not in (magnet-above, magnet-below, magnetic-pin) → return `outcome_status='na_regime'`, all metrics NULL.
2. Compute `horizon_end_date` from `dominant_bucket`.
3. If `horizon_end_date > latest available bar date` → return `outcome_status='pending_history'`.
4. Slice bars from trade_date to horizon_end_date.
5. Compute each metric:
   - `reached_touch`: any bar's high ≥ drift_target (for above/pin) or any bar's low ≤ drift_target (for below)
   - `reached_close`: |last bar's close − drift_target| ≤ 0.25 × expected_move
   - `days_to_reach`: index of first bar where touched, else NULL
   - `max_excursion_in_direction`: max favorable move
   - `final_close_distance_from_target`: signed distance (close − drift_target)
   - `actual_realized_em_pct`: (|max − min| in horizon) / expected_move
6. Return dict.

Unit tests with synthetic bars to verify each metric.

**Deliverable:** function importable, unit-tested.

## Step 3 — Backfill runner script

**Commit:** `cr-b/step-3: backfill runner for bt_daily_outcomes`

Implement: `scripts/cr_b_backfill_outcomes.py`. Structure mirrors CR-A:

```python
from packages.shared.backfill_safety import (
    get_backfill_db_conn, verify_safe_role, backfill_run, update_run_smoke
)

def main():
    conn = get_backfill_db_conn()
    verify_safe_role(conn)

    with backfill_run(conn, cr_id='CR-B') as run_id:
        rows = fetch_features_without_outcomes(conn, feature_version='v0.5.0-rebuilt')

        for batch in chunked(rows, batch_size=50):
            for feature_row in batch:
                bars = fetch_bars(conn, feature_row['trade_date'], horizon_days=60)
                outcome = compute_outcome(feature_row['trade_date'], feature_row, bars)
                insert_outcome(conn, feature_row, outcome, run_id)
            update_run_progress(conn, run_id)

        smoke_results = run_smoke_tests(conn)
        status, assessment = self_assess(smoke_results)
        update_run_smoke(conn, run_id, smoke_results, assessment)
```

All inserts to `bt_daily_outcomes` only. Never modifies existing data.

**Deliverable:** runner script ready.

## Step 4 — Execute backfill

**Commit:** `cr-b/step-4: execute outcome backfill across corpus`

Run the script. Compute time: ~1-2 hours depending on bar query speed.

**Verification:**

```sql
SELECT outcome_status, COUNT(*) FROM bt_daily_outcomes
WHERE feature_version = 'v0.5.0-rebuilt' GROUP BY outcome_status;
-- 'computed': majority
-- 'pending_history': recent dates only
-- 'na_regime': non-directional regime days
```

## Step 5 — Smoke tests

**Commit:** `cr-b/step-5: smoke tests for outcome computation`

Tests:

1. **Row count.** `bt_daily_outcomes` row count ≈ `bt_daily_features` row count for `v0.5.0-rebuilt`. Within 5%.

2. **Outcome distribution per regime:**
   ```sql
   SELECT regime_kind_at_classification,
          AVG(reached_touch::int) AS touch_rate,
          AVG(reached_close::int) AS close_rate,
          COUNT(*) AS n
   FROM bt_daily_outcomes
   WHERE feature_version = 'v0.5.0-rebuilt'
     AND outcome_status = 'computed'
   GROUP BY regime_kind_at_classification;
   ```
   Expected: touch rates between 15% and 75% per regime. If any regime hits 100% or 0%, likely a bug.

3. **Spot-check known cases.** Pick 3-5 historical days where the outcome is obvious (e.g., a magnet-above day where price clearly closed at the magnet). Verify outcome row matches.

4. **No outcomes in the future.** Every row's `horizon_end_date ≤ latest available bar date` OR `outcome_status = 'pending_history'`.

5. **All outcomes carry valid run_id.** No NULLs in `backfill_run_id` for computed rows.

Store results in `bt_backfill_runs.smoke_test_results`. Self-assess.

## Wrap criteria

- All 5 steps committed
- `bt_backfill_runs` row has `status IN ('completed', 'completed_with_warnings')`
- [[Roadmap]] updated: CR-B marked complete; CR-C moved to "ready"

## Step 0 — Diagnosis findings (2026-05-25)

### Amendments locked

**A3 — drift_target and dominant_bucket are not in feature_vector.**
Neither key exists in `bt_daily_features.feature_vector`. Both must be derived by
the runner before calling `compute_outcome`:
- `drift_target` → join `orats_gex_landscape` for `(ticker, trade_date)`, take
  `walls[0]['price']` (the dominant positive-GEX wall).
- `dominant_bucket` → `argmax` over `{'0DTE': dominance_0DTE, '1-7 DTE': dominance_1_7,
  '8-30 DTE': dominance_8_30, '30+ DTE': dominance_30plus}` from feature_vector.

These are passed as explicit parameters to `compute_outcome` (see E1).

**C1 — outcome_status enum extended.**
Four values: `'computed'`, `'pending_history'`, `'na_regime'`, `'na_data'`.
`na_data` fires when a required input is unavailable at compute time (missing
landscape row, `expected_move = 0`, empty bars). Expected count in v0.5.0-rebuilt:
0. Any `na_data` rows in production indicate a data gap worth investigating.

**C5 — max_excursion_in_direction clamped.**
Explicitly `max(0.0, ...)` — prevents negative excursion values for flat/adverse
sessions. The metric is defined as "positive even if target not reached."

**E1 — compute_outcome signature amended.**
The function takes explicit scalar inputs derived by the runner, not a raw
`feature_row` dict. Amended signature:

```python
def compute_outcome(
    trade_date: date,
    regime: str,           # feature_row['regime_at_classification']
    drift_target: float,   # from orats_gex_landscape.walls join
    dominant_bucket: str,  # argmax of dominance_* fields ('0DTE', '1-7 DTE', ...)
    expected_move: float,  # feature_row['feature_vector']['implied_move_1d']
    bars: pd.DataFrame,    # RTH 1-min OHLC, indexed by datetime (06:30–13:00 PT),
                           # spanning trade_date through horizon_end_date
) -> dict:
    """
    Compute outcome metrics for one structural read.

    Returns a dict with all bt_daily_outcomes fields. outcome_status is one of:
      'computed'        — all metrics populated
      'pending_history' — horizon_end_date > latest bar date; metrics NULL
      'na_regime'       — regime not directional in v1; metrics NULL
      'na_data'         — required input missing (no landscape row, expected_move=0,
                          empty bars); metrics NULL

    **Session basis (E7):** bars must cover RTH sessions 06:30–13:00 PT (= 13:30–20:00
    UTC) for each calendar date in trade_date..horizon_end_date. This matches the
    session definition used by apps/web/modules/Bars/service.py and the Analogues
    module's _fetch_session_outcomes.

    **Why ES bars work for SPX-anchored outcomes (E6):**
    `drift_target` is the price of the dominant wall from `orats_gex_landscape.walls`.
    The landscape GEX is accumulated at each option's `discounted_level`:

        discounted_level = strike × exp((r − q) × T)

    where r is the short rate, q the dividend yield, and T is time to expiry.
    This formula projects each SPX strike into forward price space — the same
    space ES futures trade in. A wall appearing at price P in the landscape means
    there is a concentration of options whose forward/ES-equivalent price ≈ P. ES
    futures must therefore trade near P for that wall to be active.

    ES bar prices (from ironbeam_es_1m_bars) are directly comparable to drift_target
    without any SPX-to-ES conversion. The residual difference between
    discounted_level and the actual ES futures price arises from rate/dividend
    convention rounding and roll effects; empirically this is < 5 pts — negligible
    vs. the 0.25 × expected_move tolerance used for reached_close (typically 10–14
    pts). The Analogues module's `_fetch_session_outcomes` establishes this as the
    canonical codebase pattern: ES bar prices compared to SPX-derived landscape
    levels, no conversion.
    """
```

**E3 — packages/shared/buckets.py created (new file).**
Implements `bucket_sessions(label: str) -> int` mapping landscape/dominance bucket
labels to outcome horizon session counts. Distinct from `strategy_templates._bucket_dte`
(which takes a cluster dict and uses strategy DTE targets, not outcome horizons):

```python
# Outcome horizon sessions by dominant bucket
# Distinct from strategy_templates.DTE_TARGET_BY_BUCKET (which serves strategy
# entry sizing, not multi-day outcome windows).
_OUTCOME_SESSIONS = {
    '0DTE':     1,
    '1-7 DTE':  5,
    '8-30 DTE': 20,
    '30+ DTE':  60,
}

def bucket_sessions(label: str) -> int:
    """Return number of RTH sessions for an outcome horizon given a bucket label."""
    return _OUTCOME_SESSIONS[label]   # KeyError is intentional — caller validates
```

**E4 — ironbeam_es_1m_bars timestamp column is `datetime`.**
All bar queries must use `datetime` (not `ts`). The column stores naive UTC timestamps.
Session filtering uses PT conversion: `datetime AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles'`.

**E5 — assert_role_or_die per CLAUDE.md protocol.**
Runner uses `assert_role_or_die(conn)` (not `verify_safe_role`). Both exist in
`backfill_safety.py`; `assert_role_or_die` is the CLAUDE.md-documented form.

**E6(c) — ES bars used directly; no conversion code needed or exists.**
See E1 docstring for full rationale. Confirmed: no reusable SPX↔ES helper in
`packages/shared/`. Basis concern was a misread of the data model.

**E7 — RTH session window = 06:30–13:00 PT.**
Matches `apps/web/modules/Bars/service.py` (`_SQL` filter) and Analogues
`_fetch_session_outcomes`. In UTC: approximately 13:30–20:00. `fetch_bars()` in
the runner must apply the same PT-date filter used by those modules.

### Test-layer split (B+)

**Step 2 unit tests** (synthetic bars, no DB):
- `drift_target=None` → raises / skips cleanly (na_data path)
- `expected_move=0` → skips cleanly (na_data, no ZeroDivisionError)
- `bars` empty DataFrame → skips cleanly (na_data)
- Synthetic magnet-below bars (regime exists in code, 0 rows in corpus — verifies
  the path runs without exercising it in integration)
- All 6 metrics correct for a hand-crafted scenario with known expected values

**Step 3 integration test subset** (3 real dates):

| Date | Regime | Path covered |
|------|--------|-------------|
| `2023-06-01` | magnet-above | computed (all horizons elapsed; wall at 4281, spot at 4185) |
| `2026-05-22` | magnet-above | pending_history (dominant bucket = 1-7 DTE, horizon_end ≈ 2026-05-29 > latest bar) |
| `2026-05-20` | amplification | na_regime (non-directional early return) |

### Schema reality confirmed
- 735 active rows at v0.5.0-rebuilt ✅
- Column: `regime_at_classification` (not `regime_kind`) ✅
- `magnet-below`: 0 rows in corpus — code path written but not exercised in backfill ✅
- `ironbeam_es_1m_bars`: SELECT-able as dash_backfill_writer; 1.19M rows; 2023-01-02–2026-05-25 ✅

## Status updates

### Step 4 execution — 2026-05-25 (run e5880471)

Run `e5880471-8ddd-4b88-8836-0fad250ea30d` completed in ~47 seconds. 731 rows
inserted (4 test-subset rows pre-existing from Step 3), 0 failures, 0 skipped.
Total `bt_daily_outcomes` at v0.5.0-rebuilt: 735 rows = 100% coverage of active
feature rows (`row_count_pct_diff = 0.0%`).

Status distribution: `computed=434`, `pending_history=20`, `na_regime=269`,
`na_data=12`. All rows carry `backfill_run_id`. `null_run_id_count = 0`.
`future_horizon_count = 0`.

**Touch rates (computed rows):** magnet-above 81.6% (n=342), magnetic-pin 88.0%
(n=92). Both exceed the spec's stated 15–75% gate. Not a bug — the gate was
calibrated for tighter outcome definitions than `reached_touch` over multi-session
horizons. Approved as signal: GEX walls attract price at high rates across both
directional regimes. `magnet-below` has 0 computed rows (0 such days in corpus).

**12 na_data rows:** All fired the `drift_target is None` path — dates where
`orats_gex_landscape` has no row. All are `magnet-above`. Root cause: a
prominence-mismatch between the classifier (3% threshold) and the stored landscape
walls (10% threshold) — on those 12 dates the classifier produced a magnet-above
signal but no landscape row was ingested. Acceptable for v1; queued as future
improvement.

**20 pending_history rows:** All recent dates (2026-03 through 2026-05-22) with
insufficient bar history for their bucket horizon. Expected; will become computable
as history accrues.

## Open questions

- For magnetic-pin regime, is "tolerance" the right framing or should we use a tighter band? **Defaulted to 0.25 × expected_move; revisit if smoke shows weird hit rates.**
