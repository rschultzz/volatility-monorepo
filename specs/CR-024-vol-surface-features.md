---
type: cr
cr_id: CR-D
title: Vol Surface Features
aliases: ["CR-D — Vol Surface Features", "CR-D"]
status: active
started: 2026-05-25
sequence_number: 24
run_mode: unattended
phase: 2
size: medium-large
estimated_days: 4-5
estimated_compute_hours: 2-3
data_safety_class: null_fill_update
target_feature_version: v0.5.0-rebuilt
dependencies: [CR-0, CR-A]
depended_on_by: [CR-E, CR-F, CR-G]
branch_name: cr-d-vol-surface-features
stop_conditions:
  - "pre-flight check: current_user != 'dash_backfill_writer'"
  - "any UPDATE attempted on non-NULL vol surface columns"
  - "any update touches feature_version != 'v0.5.0-rebuilt'"
  - "percentile distributions outside [0, 100] range (computation bug)"
  - "smoke test shows >5% of rows with all vol features still NULL after backfill"
on_stop: surface_in_roadmap + update_bt_backfill_runs_status + leave_full_status_in_cr_file
tags: [dash, cr, backfill, unattended, vol-surface, iv, skew, term-structure]
last_commit_sha: ""
---

# CR-D — Vol Surface Features

## Goal

Populate the v0.5 vol surface NULL placeholders against historical IV / skew data. Enables the buy/sell decision in proposals ([[CR-F — Debit Credit Variants]]), the vol regime context panel ([[CR-E — Vol Regime Context Panel]]), and the edge visualization's IV overlay ([[CR-G — Edge Visualization and P&L Engine]]).

## Context

The v0.5 schema defined vol surface placeholders (`atm_iv_percentile`, `skew_percentile`, `term_structure_slope`, `smile_convexity`, `vol_risk_premium`) but never populated them. With 2.5+ years of `orats_monies_minute` data now in the corpus, we can compute these features against rolling history and fill the NULLs. Only fills NULL values — never modifies existing data.

See [[Operating Framework — Where When Buy or Sell]]: vol surface state is the BUY-or-SELL-PREMIUM axis. The structural read (GEX, regime) tells WHERE and WHEN; vol state tells which side of premium to be on.

## Step 0 — Diagnosis (no commits)

Lock these decisions:

1. **Historical window for percentiles.** Use 60 sessions. (Tradeoff: shorter = more responsive to regime change; longer = more stable distribution. 60 is a reasonable middle.)

2. **Each feature's precise definition:**

   - **`atm_iv_percentile`** — For each (trade_date, ATM strike, nearest-monthly-expiration), find the IV. Compare to the trailing 60-session distribution of ATM IV. Output: percentile 0-100.

   - **`skew_percentile`** — 25-delta put IV minus 25-delta call IV (same expiration as ATM IV anchor). Convert raw skew to percentile against trailing 60-session distribution.

   - **`term_structure_slope`** — ATM IV at 30 DTE minus ATM IV at 90 DTE. Raw spread (in IV points). Then percentile vs trailing 60 sessions.

   - **`smile_convexity`** — ((25P IV + 25C IV) / 2) − ATM IV. Same expiration. Percentile vs trailing 60.

   - **`vol_risk_premium`** — Trailing 20-session realized vol of underlying, minus current ATM IV. Raw difference in vol points. (Realized > implied = positive VRP = market under-pricing risk historically.)

3. **Data sources.** All from `orats_monies_minute`. Use end-of-day snapshots (last minute before session close) as the canonical daily value.

4. **NULL-fill discipline.** Updates use `WHERE column IS NULL` semantics via `COALESCE`. Enforced by code AND by the column-scoped GRANT in CR-0.

5. **Confirm helper columns exist.** All five columns must already be in `bt_daily_features` (they were in v0.5 schema as NULL placeholders). Confirm:
   ```sql
   \d bt_daily_features
   ```
   If any are missing, this CR needs to be prefixed with an `ALTER TABLE ADD COLUMN` — handle in Step 1.

## Step 1 — Implement vol feature functions

**Commit:** `cr-d/step-1: implement vol surface feature computations`

Create `packages/shared/vol_features.py` with:

```python
def compute_atm_iv_percentile(
    trade_date: date,
    iv_history: pd.DataFrame,  # 60+ sessions of (date, atm_iv)
) -> float:
    """Return percentile of trade_date's ATM IV against prior 60 sessions."""

def compute_skew_percentile(trade_date, skew_history) -> float: ...
def compute_term_structure_slope(trade_date, front_iv, back_iv, slope_history) -> tuple[float, float]:
    """Returns (raw spread, percentile)."""
def compute_smile_convexity(trade_date, convexity_history) -> float: ...
def compute_vol_risk_premium(trade_date, realized_vol_20d, current_atm_iv) -> float: ...
```

Plus a helper: `fetch_iv_history_for_date(conn, trade_date, lookback_sessions=60)`.

Unit tests with synthetic data: constant IV → percentile = 50; rising IV → percentile climbing.

**Deliverable:** functions implementable and unit-tested.

## Step 2 — Backfill runner script

**Commit:** `cr-d/step-2: backfill runner for vol surface features`

Implement: `scripts/cr_d_backfill_vol_features.py`.

```python
from packages.shared.backfill_safety import (
    get_backfill_db_conn, verify_safe_role, backfill_run, update_run_smoke
)

def main():
    conn = get_backfill_db_conn()
    verify_safe_role(conn)

    with backfill_run(conn, cr_id='CR-D') as run_id:
        rows = fetch_features_with_null_vol(conn, feature_version='v0.5.0-rebuilt')

        for batch in chunked(rows, batch_size=30):
            for feature_row in batch:
                iv_history = fetch_iv_history(conn, feature_row['trade_date'])

                if len(iv_history) < 60:
                    continue  # Not enough history; skip silently

                vol_features = {
                    'atm_iv_percentile': compute_atm_iv_percentile(...),
                    'skew_percentile': compute_skew_percentile(...),
                    'term_structure_slope': compute_term_structure_slope(...),
                    'smile_convexity': compute_smile_convexity(...),
                    'vol_risk_premium': compute_vol_risk_premium(...),
                }

                update_vol_features(conn, feature_row, vol_features, run_id)
            update_run_progress(conn, run_id)

        smoke_results = run_smoke_tests(conn)
        status, assessment = self_assess(smoke_results)
        update_run_smoke(conn, run_id, smoke_results, assessment)
```

`update_vol_features` enforces NULL-only writes via COALESCE:

```python
def update_vol_features(conn, feature_row, vol_features, run_id):
    sql = """
        UPDATE bt_daily_features
        SET atm_iv_percentile = COALESCE(atm_iv_percentile, %s),
            skew_percentile = COALESCE(skew_percentile, %s),
            term_structure_slope = COALESCE(term_structure_slope, %s),
            smile_convexity = COALESCE(smile_convexity, %s),
            vol_risk_premium = COALESCE(vol_risk_premium, %s),
            backfill_run_id = COALESCE(backfill_run_id, %s)
        WHERE ticker = %s AND trade_date = %s AND feature_version = %s
          AND (atm_iv_percentile IS NULL
               OR skew_percentile IS NULL
               OR term_structure_slope IS NULL
               OR smile_convexity IS NULL
               OR vol_risk_premium IS NULL)
    """
    # COALESCE preserves existing non-NULL values
    # WHERE ensures we only touch rows with at least one NULL
```

**Deliverable:** runner ready, NULL-fill semantics enforced.

## Step 3 — Execute backfill

**Commit:** `cr-d/step-3: execute vol surface feature backfill`

Run script. Compute time ~2-3 hours.

**Verification:**

```sql
SELECT
  COUNT(*) FILTER (WHERE atm_iv_percentile IS NULL) AS null_iv_pct,
  COUNT(*) FILTER (WHERE skew_percentile IS NULL) AS null_skew,
  COUNT(*) FILTER (WHERE term_structure_slope IS NULL) AS null_ts,
  COUNT(*) FILTER (WHERE smile_convexity IS NULL) AS null_conv,
  COUNT(*) FILTER (WHERE vol_risk_premium IS NULL) AS null_vrp,
  COUNT(*) AS total
FROM bt_daily_features
WHERE feature_version = 'v0.5.0-rebuilt';

-- Expected: NULL counts small (only first ~60 trade dates of corpus,
-- where insufficient history existed for percentile calculation)
```

## Step 4 — Smoke tests

**Commit:** `cr-d/step-4: smoke tests for vol surface features`

Tests:

1. **Percentile ranges.** All percentile columns should be in [0, 100]. No outliers.

2. **Distribution sanity.** Percentile columns should be approximately uniformly distributed across the corpus (since they're percentiles against rolling history). Histogram check.

3. **Known event verification.**
   - Pick 2-3 known high-IV events (Feb 2018, March 2020, Aug 2024, etc. if in corpus). `atm_iv_percentile` should be near 100.
   - Pick 2-3 known quiet periods. `atm_iv_percentile` near 0-30.

4. **Skew sign sanity.** `skew_percentile` should be positive (put skew > call skew) most of the time for SPX-like underlying. If majority is negative, computation bug.

5. **Original v0.5.0 untouched.**
   ```sql
   SELECT COUNT(*) FROM bt_daily_features
   WHERE feature_version = 'v0.5.0' AND backfill_run_id IS NOT NULL;
   -- Must be 0; CR-D should never have touched original version
   ```

6. **No structural feature changes.** Compare structural feature checksum pre- and post-CR-D for a sample of rows. Must be unchanged.

Store all results in `smoke_test_results`. Self-assess.

## Wrap criteria

- All 4 steps committed
- `bt_backfill_runs` row has `status IN ('completed', 'completed_with_warnings')`
- [[Roadmap]] updated: CR-D marked complete; CR-E, CR-F, CR-G moved to "ready"
- **Drafting trigger:** if CR-D completion is the halfway-point of current batch, draft next 6-8 CRs (see [[Roadmap]] drafting trigger section)

## Status updates

(filled during execution)

## Open questions

- 60-session lookback is a default; should it be longer for term_structure_slope and smile_convexity (slower-moving features)? **Default 60 for v1; revisit if smoke shows odd behavior.**
