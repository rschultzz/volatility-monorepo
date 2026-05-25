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

## Status updates

(filled during execution)

## Open questions

- For magnetic-pin regime, is "tolerance" the right framing or should we use a tighter band? **Defaulted to 0.25 × expected_move; revisit if smoke shows weird hit rates.**
