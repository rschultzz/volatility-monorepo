# CR-021 — Landscape Backfill (CR-A)

> Spec freeze from vault session note `2026-05-24 - CR-021 — Landscape Backfill.md`
> Frozen: 2026-05-24

---
cr_id: CR-A
sequence_number: 021
branch: feat/CR-021-landscape-backfill
---

## Goal

Populate `bt_daily_features` with v0.5 vectors for the full historical corpus (2.5+ years), writing to a new feature_version `v0.5.0-rebuilt`. Original `v0.5.0` rows remain untouched.

Decouple app reads from `computed_at`-based dynamic version selection (Step 1) before any backfill writes happen, so the backfill does not silently auto-promote the new version mid-run.

## Context

The current 37-day corpus limits KNN statistical power. Ryan has 2.5+ years of historical inputs (price 1m bars, skew minute-by-minute, daily GEX) already in the DB. The landscape extraction code is feature-first and deterministic from these inputs. Backfill is mechanical: run the existing `compute_landscape` + `extract_features` over each historical trade date and insert rows.

This unlocks CR-B (outcome computation across the corpus), CR-C (probability output with statistical power), and CR-D (vol surface backfill).

**Run-mode note:** this CR is "mixed" — Step 1 is interactive (small app code change); Steps 2-3 are interactive (scripting); Step 4 is unattended (the multi-hour backfill execution); Step 5 is interactive (smoke review).

## Step 0 — Diagnosis (no commits)

This step is read-only; outputs decisions for the rest of the CR.

1. **Identify earliest backfillable trade date.** Query the inputs:
   - `SELECT MIN(trade_date) FROM ironbeam_es_1m_bars;`
   - `SELECT MIN(trade_date) FROM orats_oi_gamma;`
   - `SELECT MIN(trade_date) FROM orats_monies_minute;`
   - The earliest date where all three have data is the start of the backfillable window.

2. **Pick determinism check date.** Choose a date from the existing `v0.5.0` corpus (one of the current 37 days) where inputs still exist. Re-run `compute_landscape` + `extract_features` on that date's raw inputs. Compare the output to the stored `v0.5.0` row. Expected: bit-identical or near-identical (small float drift acceptable; large differences are a stop condition).

3. **Decide batch size.** Backfilling 625+ trade dates in one transaction is too long. Decide on batch size (recommend: 30 dates per batch, so ~20 batches total). Each batch is a separate transaction.

4. **Confirm idempotency.** The PK is `(ticker, trade_date, feature_version)`. Backfill uses `INSERT ON CONFLICT (ticker, trade_date, feature_version) DO NOTHING` so re-running the script is safe — already-inserted rows are skipped. Confirm this matches the existing table's actual constraints.

5. **Inventory `_latest_feature_version` call sites.** Grep the codebase. Count and list every callsite. Step 1 will replace each one. If the count is meaningfully larger than what was identified during CR-0's Step 5 diagnosis (which noted the function exists in `Analogues/routes.py:297`), surface for review before proceeding.

6. **Confirm `dash_backfill_writer` is fully set up.**
   ```sql
   SELECT current_user;  -- must be dash_backfill_writer
   ```
   If not, halt — CR-0 is incomplete.

7. **Verify `bt_backfill_runs` table exists.** From CR-0. If not, halt.

## Step 1 — Decouple app reads from `computed_at`-based version selection

**Commit:** `cr-a/step-1: replace _latest_feature_version with canonical-version constant`

**Why this is Step 1, not later:** per Data Safety Protocol, promotion of a new `feature_version` to canonical must be a manual deliberate step. The existing `_latest_feature_version` function (Analogues/routes.py:297) selects whichever version was most recently written via `ORDER BY computed_at DESC LIMIT 1`. The moment Step 4's backfill writes the first `v0.5.0-rebuilt` row, app reads would silently switch to it — auto-promotion mid-backfill, mixing versions in KNN candidate sets. Step 1 closes that hole before any writes happen.

Sub-steps:

1. **Create `packages/shared/canonical_version.py`:**
   ```python
   """Canonical feature_version for app reads.

   Promotion of a new feature_version to canonical is a manual
   deliberate step: edit this constant, commit, deploy. See
   Data Safety Protocol for the full promotion model.
   """

   CANONICAL_FEATURE_VERSION = "v0.5.0"
   ```

2. **Replace each callsite of `_latest_feature_version`.** Pattern:
   ```python
   # Before:
   version = _latest_feature_version(conn, ticker)

   # After:
   from packages.shared.canonical_version import CANONICAL_FEATURE_VERSION
   version = CANONICAL_FEATURE_VERSION
   ```

3. **Delete `_latest_feature_version` itself.** No callers remain; the function's whole purpose was the now-replaced lookup. Keeping a vestigial copy invites accidental future use.

4. **Verification:**
   - `grep -r "_latest_feature_version" .` returns zero matches in source code (only mentions in this spec or other vault notes are OK)
   - App endpoints that depended on it still return correct data: smoke-test the KNN candidate-load path, the day-browse path, and the audit-flag auto-regime path. Each must return non-empty results identical to pre-Step-1 behavior, since `CANONICAL_FEATURE_VERSION = 'v0.5.0'` matches what the dynamic lookup currently resolves to.
   - Confirm `CANONICAL_FEATURE_VERSION` matches the production version exactly. If unsure, query: `SELECT DISTINCT feature_version FROM bt_daily_features_active;` — should currently return only `'v0.5.0'`.

**Deliverable:** app reads use a constant for feature_version selection. Promotion of a new version is now a one-line edit to `canonical_version.py`.

**Stop conditions:**
- More callsites of `_latest_feature_version` found than Step 0's diagnosis identified — surface before replacing
- Any app endpoint returns different data than before with the constant in place — means the dynamic lookup was doing something the constant doesn't capture; surface for analysis

## Step 2 — Determinism check

**Commit:** `cr-a/step-2: determinism check harness for landscape extraction`

Implement: `scripts/cr_a_determinism_check.py`.

- Takes one `trade_date` argument
- Re-runs `compute_landscape` + `extract_features` for that date
- Compares output against the existing `bt_daily_features` row for that date (where `feature_version = 'v0.5.0'`)
- Reports diff field-by-field

Run on the determinism check date selected in Step 0.

**Deliverable:** confirms structural features are deterministic from inputs. If mismatch is more than float-precision drift (e.g., regime_kind differs, drift_target differs by >0.5pt), **STOP and surface**.

**Verification:** all structural fields match within tolerance.

## Step 3 — Backfill runner script

**Commit:** `cr-a/step-3: backfill runner script for landscape extraction`

Implement: `scripts/cr_a_backfill_landscape.py`.

Structure:

```python
from packages.shared.backfill_safety import (
    get_backfill_db_conn, verify_safe_role, backfill_run, update_run_smoke
)

def main():
    conn = get_backfill_db_conn()
    verify_safe_role(conn)

    with backfill_run(conn, cr_id='CR-A') as run_id:
        start_date = find_earliest_backfillable_date(conn)
        end_date = today()
        dates = enumerate_trading_dates(start_date, end_date)

        for batch in chunked(dates, batch_size=30):
            backfill_batch(conn, batch, run_id, feature_version='v0.5.0-rebuilt')
            update_run_progress(conn, run_id)

        smoke_results = run_smoke_tests(conn, feature_version='v0.5.0-rebuilt')
        status, assessment = self_assess(smoke_results)
        update_run_smoke(conn, run_id, smoke_results, assessment)
        # backfill_run context manager handles status update on exit
```

Each `backfill_batch` is its own transaction. INSERT ON CONFLICT DO NOTHING for idempotency. Every row tagged with `backfill_run_id = run_id` and `feature_version = 'v0.5.0-rebuilt'`.

**Deliverable:** script ready to run; covers full corpus in batches.

**Verification:** run on a tiny test subset (3 dates) first; confirm rows inserted with correct version and run_id.

## Step 4 — Execute backfill (unattended)

**Commit:** `cr-a/step-4: execute full historical landscape backfill`

Run the backfill script against the full date range.

Expected duration: 3-5 hours for 625+ dates depending on extraction speed.

Monitor: `bt_backfill_runs.rows_inserted` should grow steadily; no batch should take >30 minutes (stop condition).

**Deliverable:** `bt_daily_features` populated with `feature_version = 'v0.5.0-rebuilt'` rows for all backfillable dates.

**Verification:**

```sql
SELECT COUNT(*) FROM bt_daily_features WHERE feature_version = 'v0.5.0-rebuilt';
-- Should be approximately equal to count of trading days in backfill window

SELECT COUNT(*) FROM bt_daily_features WHERE feature_version = 'v0.5.0';
-- Should be UNCHANGED from pre-CR-A count (~37)
```

App behavior during and after Step 4: app reads continue to return `v0.5.0` data via the canonical constant — `v0.5.0-rebuilt` rows are silent until promotion. Verify by hitting the KNN endpoint mid-backfill (or post-backfill, before promotion) and confirming results match pre-backfill output.

## Step 5 — Smoke tests

**Commit:** `cr-a/step-5: smoke tests for landscape backfill`

Tests:

1. **Row count expected.** Count rows with new feature_version, compare to count of trading days in window. Expect within 5%.

2. **Original data untouched.** Count rows with `feature_version = 'v0.5.0'` — must equal pre-CR-A count.

3. **App-read isolation.** With `CANONICAL_FEATURE_VERSION` still pointing at `'v0.5.0'`, confirm the KNN candidate-load path returns only `v0.5.0` rows (the new `v0.5.0-rebuilt` rows must be invisible to app reads pre-promotion).

4. **Regime distribution sensible.**
   ```sql
   SELECT regime_kind, COUNT(*)
   FROM bt_daily_features
   WHERE feature_version = 'v0.5.0-rebuilt'
   GROUP BY regime_kind ORDER BY 2 DESC;
   ```
   Expect: all known regime_kinds present; no single regime >70% of rows (sanity check against extraction bug producing constant output).

5. **Spot-check known event days.**
   - Pick 2-3 FOMC days from the corpus. Verify regime_kind / drift_target look reasonable given known market behavior.
   - Pick 2-3 expected-quiet days (mid-summer). Verify lower expected_move.

6. **Bucket dominance distribution.**
   ```sql
   SELECT dominant_bucket, COUNT(*)
   FROM bt_daily_features
   WHERE feature_version = 'v0.5.0-rebuilt'
   GROUP BY dominant_bucket;
   ```
   Expect: `0DTE-near-spot` dominant on a meaningful fraction; `1-7DTE` and `8-30DTE` also represented.

Each test result stored in `bt_backfill_runs.smoke_test_results` JSONB.

**Self-assessment after smoke:**
- All pass: `status = 'completed'`
- 1-2 minor anomalies but within tolerance: `status = 'completed_with_warnings'`
- Anything outside tolerance: `status = 'suspect'` and HALT — do not proceed to dependent CRs

## Wrap criteria

- All 5 steps committed
- `bt_backfill_runs` row for this CR has `status IN ('completed', 'completed_with_warnings')`
- `CANONICAL_FEATURE_VERSION` still equals `'v0.5.0'` (promotion to `'v0.5.0-rebuilt'` is a future deliberate one-line commit, not part of this CR)
- Roadmap updated: CR-A marked complete; CR-B and CR-D moved to "ready"
- Status note in this file's `Status updates` section summarizing what landed

## Promotion (after CR-A completes — out of scope here)

When you've reviewed the smoke results and decided `v0.5.0-rebuilt` is canonical:

1. Edit `packages/shared/canonical_version.py`:
   ```python
   CANONICAL_FEATURE_VERSION = "v0.5.0-rebuilt"
   ```
2. Commit: `promote: v0.5.0-rebuilt → canonical`
3. Deploy. App reads now return the expanded corpus.
4. Original `v0.5.0` rows remain in DB indefinitely as rollback baseline. Revert the commit to roll back; no data restoration needed because original was never modified.

Promotion is its own deliberate decision moment, not a side effect of CR-A.

## Status updates

**Spec amendment 2026-05-24 (during CR-0 execution):** added new Step 1 (decouple app reads from `computed_at`-based version selection) per architectural concern surfaced during CR-0 Step 5 review. Existing Steps 1-4 renumbered to 2-5. `run_mode` changed from `unattended` to `mixed` to reflect that Step 1 is interactive code change while Step 4 is the unattended bulk operation. Stop-condition added for the canonical-constant being respected. Promotion section added at end.

**Activated 2026-05-24:** moved from `Dash/CRs/` to `Dash/sessions/`, assigned sequence number 021, branch `feat/CR-021-landscape-backfill`.

## Diagnosis findings (Step 0)

*To be appended after Step 0 queries run.*
