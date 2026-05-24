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

**Commit:** `cr-a/step-1: pin all dynamic version selection to canonical constant`

**Why this is Step 1, not later:** per Data Safety Protocol, promotion of a new `feature_version` to canonical must be a manual deliberate step. The existing `_latest_feature_version` function (`Analogues/routes.py:294`) selects whichever version was most recently written via `ORDER BY computed_at DESC LIMIT 1`. Step 0 diagnosis also found 3 additional inline `ORDER BY computed_at DESC LIMIT 1` patterns in `audit_overrides.py` and `AuditFlags/service.py` doing the same thing. The moment Step 4's backfill writes the first `v0.5.0-rebuilt` row (newer `computed_at`), all four sites would silently switch to it — auto-promotion mid-backfill, mixing versions in KNN candidate sets and corrupting auto-regime lookups. Step 1 closes all four holes before any writes happen.

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

4. **Pin inline `ORDER BY computed_at DESC LIMIT 1` patterns to canonical.** Three sites do dynamic-latest version selection inline without going through `_latest_feature_version`:

   - `packages/shared/audit_overrides.py` — `_AUTO_REGIME_SQL` (line ~34): used by `get_effective_regime()`
   - `packages/shared/audit_overrides.py` — `_AUTO_REGIME_BATCH_SQL` (line ~53): used by `get_effective_regimes()`
   - `apps/web/modules/AuditFlags/service.py` — `_AUTO_REGIME_SQL` (line ~35): used by `create_flag()`

   For each: add `AND feature_version = %s` to the WHERE clause (passing `CANONICAL_FEATURE_VERSION`), and remove the now-redundant `ORDER BY computed_at DESC LIMIT 1` — once version is pinned, the PK `(ticker, trade_date, feature_version)` guarantees at most one row, so the sort and limit are dead weight.

   Import pattern:
   ```python
   from packages.shared.canonical_version import CANONICAL_FEATURE_VERSION
   # then pass CANONICAL_FEATURE_VERSION as a query param alongside ticker / trade_date
   ```

5. **Verification:**
   - `grep -r "_latest_feature_version" .` returns zero matches in source code (only mentions in this spec or other vault notes are OK)
   - `grep -r "ORDER BY computed_at DESC" .` returns zero matches in source code
   - App endpoints that depended on all four sites still return correct data: smoke-test the KNN candidate-load path, the day-browse path, and the audit-flag auto-regime path. Each must return non-empty results identical to pre-Step-1 behavior, since `CANONICAL_FEATURE_VERSION = 'v0.5.0'` matches what the dynamic lookup currently resolves to.
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

4. **Regime distribution sensible.** (`regime_kind` is stored as `regime_at_classification` — a top-level column, not inside `feature_vector`.)
   ```sql
   SELECT regime_at_classification, COUNT(*)
   FROM bt_daily_features
   WHERE feature_version = 'v0.5.0-rebuilt'
   GROUP BY regime_at_classification ORDER BY 2 DESC;
   ```
   Expect: all known regime values present; no single value >70% of rows (sanity check against extraction bug producing constant output).

5. **Spot-check known event days.**
   - Pick 2-3 FOMC days from the corpus. Verify `regime_at_classification` and `feature_vector->>'implied_move_1d'` look reasonable given known market behavior.
   - Pick 2-3 expected-quiet days (mid-summer). Verify lower `implied_move_1d`.

6. **Bucket dominance distribution.** (`dominant_bucket` is not stored — derive inline from the four `dominance_*` JSONB keys.)
   ```sql
   WITH dom AS (
     SELECT
       CASE
         WHEN GREATEST(
                (feature_vector->>'dominance_0DTE')::float,
                (feature_vector->>'dominance_1_7')::float,
                (feature_vector->>'dominance_8_30')::float,
                (feature_vector->>'dominance_30plus')::float
              ) = (feature_vector->>'dominance_0DTE')::float  THEN '0DTE-near-spot'
         WHEN GREATEST(
                (feature_vector->>'dominance_0DTE')::float,
                (feature_vector->>'dominance_1_7')::float,
                (feature_vector->>'dominance_8_30')::float,
                (feature_vector->>'dominance_30plus')::float
              ) = (feature_vector->>'dominance_1_7')::float   THEN '1-7DTE'
         WHEN GREATEST(
                (feature_vector->>'dominance_0DTE')::float,
                (feature_vector->>'dominance_1_7')::float,
                (feature_vector->>'dominance_8_30')::float,
                (feature_vector->>'dominance_30plus')::float
              ) = (feature_vector->>'dominance_8_30')::float  THEN '8-30DTE'
         ELSE '30plus'
       END AS dominant_bucket
     FROM bt_daily_features
     WHERE feature_version = 'v0.5.0-rebuilt'
   )
   SELECT dominant_bucket, COUNT(*) FROM dom GROUP BY 1 ORDER BY 2 DESC;
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

Completed 2026-05-24.

**Sub-step 1 — Input coverage:**

| Table | Min date | Max date | Distinct dates |
|---|---|---|---|
| `ironbeam_es_1m_bars` | 2023-01-02 | 2026-05-24 | 1059 |
| `orats_oi_gamma` | 2023-01-02 | 2026-05-22 | 850 |
| `orats_monies_minute` | 2023-05-01 | 2026-05-22 | 769 |

Note: `ironbeam_es_1m_bars` uses `datetime` (timestamp), not `trade_date`; dates derived via `datetime::date`.

**Eligible backfill window: 2023-05-01 → 2026-05-22 — 738 dates.** Limiting factor: `orats_monies_minute` starts 2023-05-01.

**Sub-step 2 — Determinism check date:** 2026-05-01 (midpoint of current 37-day v0.5.0 corpus; range is 2026-04-01 → 2026-05-22, all SPX, all within overlap window).

**Sub-step 3 — Batch size:** 30 dates per batch (~25 batches for 738 dates).

**Sub-step 4 — Idempotency:** PK confirmed as `PRIMARY KEY (ticker, trade_date, feature_version)`. `INSERT ON CONFLICT DO NOTHING` will work as specified.

**Sub-step 5 — Callsite inventory (AMENDED — expanded scope):**
- `_latest_feature_version`: 1 definition (`Analogues/routes.py:294`) + 1 callsite (`Analogues/routes.py:372`)
- 3 additional inline `ORDER BY computed_at DESC LIMIT 1` patterns NOT through `_latest_feature_version`:
  - `packages/shared/audit_overrides.py:34` — `_AUTO_REGIME_SQL` (used by `get_effective_regime`)
  - `packages/shared/audit_overrides.py:53` — `_AUTO_REGIME_BATCH_SQL` (used by `get_effective_regimes`)
  - `apps/web/modules/AuditFlags/service.py:35` — `_AUTO_REGIME_SQL` (used by `create_flag`)
- All 4 sites fixed in Step 1 (see expanded sub-step 4 above).

**Sub-step 6 — dash_backfill_writer:** PASS — `current_user = dash_backfill_writer`.

**Sub-step 7 — bt_backfill_runs:** EXISTS.

**Schema findings (affect Step 5 smoke queries):**
- `regime_kind` column does NOT exist. Regime is stored in `regime_at_classification` (top-level column). Step 5 query corrected accordingly.
- `dominant_bucket` column does NOT exist. Dominance is stored in `feature_vector` JSONB as `dominance_0DTE`, `dominance_1_7`, `dominance_8_30`, `dominance_30plus`. Step 5 query now derives dominant bucket inline via CASE expression. No new key added to JSONB.
- Vol surface features (`atm_iv_percentile`, `skew_percentile`, `smile_convexity`, `term_structure_slope`, `vol_risk_premium`) are NULL in current v0.5.0 rows — these are CR-D's scope. CR-A backfill will produce the same NULLs for historical rows.

**FEATURE_VERSION vs CANONICAL_FEATURE_VERSION clarification:**
- `FEATURE_VERSION = "v0.5.0"` in `packages/shared/day_features.py` is the **write-side** constant — stamped on rows during live computation by the ingest job. Not changed by CR-A.
- `CANONICAL_FEATURE_VERSION = "v0.5.0"` in the new `packages/shared/canonical_version.py` is the **read-side** constant — controls what feature_version the app queries. Both are currently `v0.5.0`; they stay synchronized until promotion (which is a read-side constant edit only, never a write-side side effect).
