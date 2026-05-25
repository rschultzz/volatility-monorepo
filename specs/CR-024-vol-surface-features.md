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

1. **Historical window for percentiles.** Use 60 sessions.

2. **Each feature's precise definition (locked 2026-05-25 after Step 0 diagnosis):**

   - **`atm_iv_percentile`** — EOD `atmiv` from `orats_monies_minute` at the nearest-available expiration to 30 DTE (see note below). Percentile of trade_date's value against trailing 60-session distribution of ATM IV at that same DTE anchor. Output: 0–100.

   - **`skew_percentile`** — Raw skew = `vol75 − vol25` (25-delta put IV minus 25-delta call IV, same EOD snapshot and same expiration as ATM IV anchor). Percentile of raw skew against trailing 60-session distribution. Output: 0–100.

   - **`term_structure_slope`** — `atmiv_near − atmiv_far` where near = nearest-available expiration to 30 DTE, far = nearest-available expiration to 90 DTE (both at EOD snapshot). Raw spread stored directly (in IV points, positive = contango, negative = backwardation). Then percentile vs trailing 60-session distribution. Output: float (raw spread, stored as `term_structure_slope`).

   - **`smile_convexity`** — `(vol75 + vol25) / 2 − atmiv` at same EOD snapshot and same expiration as ATM IV anchor. Percentile vs trailing 60 sessions. Output: 0–100.

   - **`vol_risk_premium`** — Trailing 20-session realized vol of ES underlying (from `ironbeam_es_1m_bars` daily closes) minus EOD `atmiv` at nearest-to-30-DTE expiration. Raw difference in vol points (positive = realized > implied). Stored directly, not as a percentile. Output: float.

   **DTE anchoring note:** `orats_monies_minute` DTE values are not continuous (expirations skip non-listing dates). Use `ORDER BY ABS(dte - 30) LIMIT 1` and `ORDER BY ABS(dte - 90) LIMIT 1` within the EOD snapshot to find the nearest available expiration to 30 DTE and 90 DTE respectively.

3. **Data sources:**
   - `orats_monies_minute`: ATM IV, skew, convexity, term structure. EOD snapshot = `WHERE snapshot_pt = (SELECT MAX(snapshot_pt) FROM orats_monies_minute WHERE trade_date = %s AND ticker = %s)`. `trade_date` stored as TEXT — pass `.isoformat()`.
   - `ironbeam_es_1m_bars`: VRP realized vol. Daily close = last bar per date via `DATE(datetime)`. `datetime` is the column name (no `trade_date` column).
   - `vol25` = 25-delta call IV; `vol75` = 25-delta put IV (= 75-delta call). Confirmed from ORATS moneyness schema.

4. **NULL-fill discipline.** Updates use COALESCE semantics. Enforced by code AND by the column-scoped GRANT added in Step 1.

5. **`bt_daily_features` — columns DO NOT EXIST.** Step 0 diagnosis confirmed none of the 5 vol columns are in the table. The "v0.5 schema as NULL placeholders" assumption was incorrect. Step 1 (new) adds them via DDL migration before any feature functions or backfill runner land.

6. **`backfill_run_id` limitation.** All v0.5.0-rebuilt rows already have `backfill_run_id` set from CR-A. `COALESCE(backfill_run_id, %s)` in CR-D's UPDATE will not overwrite CR-A's attribution. CR-D tracking is via `bt_backfill_runs` aggregate only. Accepted limitation; a future architectural CR will add per-update provenance columns (queued alongside the `orats_gex_landscape` safety-columns gap from CR-A).

7. **Test subset (locked, per CR-021 Lesson 1 — design before writing the script):**

   | Date | Purpose |
   |---|---|
   | `2023-05-05` | Insufficient-history skip — ~4 trading days into corpus; IV history < 60 → all 4 percentile columns skip; VRP still computable |
   | `2023-09-01` | First full computation — ~87 trading days in; all 5 features should compute |
   | `2024-08-05` | Known high-IV event (Japan carry-trade VIX spike ~65); `atm_iv_percentile` expected near 100 |
   | `2025-01-10` | Normal mid-corpus date; confirmed schema-compatible via Step 0 queries |
   | Re-run any above | COALESCE idempotence — second run produces 0 row updates |

8. **Lesson 2 discipline (from CR-021).** `compute_*` functions return `None` or raise `ValueError` explicitly when inputs are insufficient. No default percentile fallbacks (no `except: return 50.0` patterns). Runner catches `None`/exception, logs a skip, continues.

## Step 1 — DDL migration: add vol surface columns + role grants

**Commit:** `cr-d/step-1: bt_daily_features vol surface columns + role grants`

File: `infra/sql/bt_daily_features_vol_surface_columns.sql`

```sql
-- Add vol surface feature columns to bt_daily_features.
-- All columns are nullable double precision; populated by CR-D backfill.
-- No feature_version bump — this is a schema extension, not a vector change.

ALTER TABLE bt_daily_features
  ADD COLUMN IF NOT EXISTS atm_iv_percentile    double precision,
  ADD COLUMN IF NOT EXISTS skew_percentile       double precision,
  ADD COLUMN IF NOT EXISTS term_structure_slope  double precision,
  ADD COLUMN IF NOT EXISTS smile_convexity       double precision,
  ADD COLUMN IF NOT EXISTS vol_risk_premium      double precision;

-- Column-scoped grant: dash_backfill_writer may read + write vol surface
-- columns only. Mirrors the pattern established in CR-0 for the safety columns.
GRANT SELECT, INSERT, UPDATE (
  atm_iv_percentile,
  skew_percentile,
  term_structure_slope,
  smile_convexity,
  vol_risk_premium
) ON bt_daily_features TO dash_backfill_writer;
```

Execute as superuser (`new_db_cred` / `rschultz`). GRANT must follow the ALTER in the same session (columns must exist before they can be granted).

**Verification (as `dash_backfill_writer`):**
- UPDATE on a vol column (`atm_iv_percentile`) succeeds
- UPDATE on a non-whitelisted column (`feature_vector`) still fails with permission denied
- `\d bt_daily_features` shows all 5 new columns
- `column_privileges` shows UPDATE granted on all 5 to `dash_backfill_writer`

**Note on daily cron impact:** existing `INSERT INTO bt_daily_features` statements do not name the new columns, so they will insert NULL for all 5 — correct behavior. No cron changes needed.

## Step 2 — Implement vol feature functions

**Commit:** `cr-d/step-2: implement vol surface feature computations`

Create `packages/shared/vol_features.py`. All functions are pure (DataFrames in, scalars or None out). No DB connections in this module.

```python
def compute_atm_iv_percentile(
    trade_date: date,
    iv_history: pd.DataFrame,  # columns: [trade_date, atm_iv], prior 60 sessions only
) -> float | None:
    """Percentile of trade_date's ATM IV against prior 60 sessions.
    Returns None if iv_history has fewer than 60 rows."""

def compute_skew_percentile(
    trade_date: date,
    skew_history: pd.DataFrame,  # columns: [trade_date, raw_skew]
) -> float | None: ...

def compute_term_structure_slope(
    trade_date: date,
    near_iv: float,   # ATM IV at nearest-to-30-DTE expiration, current day
    far_iv: float,    # ATM IV at nearest-to-90-DTE expiration, current day
    slope_history: pd.DataFrame,  # columns: [trade_date, slope]
) -> tuple[float, float] | None:
    """Returns (raw_spread, percentile). raw_spread = near_iv - far_iv."""

def compute_smile_convexity(
    trade_date: date,
    convexity_history: pd.DataFrame,  # columns: [trade_date, convexity]
) -> float | None: ...

def compute_vol_risk_premium(
    realized_vol_20d: float,  # annualized, from ironbeam_es_1m_bars
    current_atm_iv: float,
) -> float:
    """VRP = realized_vol_20d - current_atm_iv. No history needed; not a percentile."""
```

DB helpers (also in `vol_features.py`):

```python
def fetch_eod_vol_snapshot(conn, trade_date: date, ticker: str) -> dict | None:
    """Returns {atmiv, vol25, vol75, near_atm_iv, far_atm_iv} at EOD snapshot.
    near = nearest expiration to 30 DTE; far = nearest to 90 DTE.
    Returns None if no data for trade_date."""

def fetch_iv_history(conn, trade_date: date, ticker: str, lookback: int = 60) -> pd.DataFrame:
    """Returns up to `lookback` prior sessions of EOD vol data before trade_date."""

def fetch_es_daily_close(conn, trade_date: date) -> float | None:
    """Last ironbeam_es_1m_bars close for the date."""
```

Unit tests: constant IV → percentile = 50; rising IV sequence → percentile climbs toward 100; fewer than 60 rows → returns None; VRP sign check (realized > implied → positive). Target: 17–25 test cases.

**Deliverable:** functions implemented and unit-tested.

## Step 3 — Backfill runner script

**Commit:** `cr-d/step-3: backfill runner for vol surface features`

Implement: `scripts/cr_d_backfill_vol_features.py`.

```python
from packages.shared.backfill_safety import (
    get_backfill_db_conn, assert_role_or_die, backfill_run, update_run_smoke
)

def main():
    conn = get_backfill_db_conn()
    assert_role_or_die(conn)

    with backfill_run(conn, cr_id='CR-D') as run_id:
        rows = fetch_features_with_null_vol(conn, feature_version='v0.5.0-rebuilt')

        for batch in chunked(rows, batch_size=30):
            for feature_row in batch:
                snapshot = fetch_eod_vol_snapshot(conn, feature_row['trade_date'], feature_row['ticker'])
                iv_hist = fetch_iv_history(conn, feature_row['trade_date'], feature_row['ticker'])
                es_closes = fetch_es_closes_before(conn, feature_row['trade_date'], n=21)

                if snapshot is None or len(iv_hist) < 60:
                    log_skip(feature_row['trade_date'], 'insufficient_history')
                    continue

                realized_vol = compute_realized_vol_20d(es_closes)

                vol_feats = {
                    'atm_iv_percentile': compute_atm_iv_percentile(feature_row['trade_date'], iv_hist[['trade_date','atm_iv']]),
                    'skew_percentile': compute_skew_percentile(feature_row['trade_date'], iv_hist[['trade_date','raw_skew']]),
                    'term_structure_slope': compute_term_structure_slope(...)[0],  # raw spread only
                    'smile_convexity': compute_smile_convexity(feature_row['trade_date'], iv_hist[['trade_date','convexity']]),
                    'vol_risk_premium': compute_vol_risk_premium(realized_vol, snapshot['atmiv']),
                }

                if any(v is None for v in vol_feats.values()):
                    log_skip(feature_row['trade_date'], 'None_returned_from_compute')
                    continue

                update_vol_features(conn, feature_row, vol_feats, run_id)

            update_run_progress(conn, run_id)

        smoke_results = run_smoke_tests(conn)
        status, assessment = self_assess(smoke_results)
        update_run_smoke(conn, run_id, smoke_results, assessment)
```

`update_vol_features` COALESCE semantics (NULL-only writes, no overwrite of existing non-NULL values):

```python
sql = """
    UPDATE bt_daily_features
    SET atm_iv_percentile  = COALESCE(atm_iv_percentile, %s),
        skew_percentile    = COALESCE(skew_percentile, %s),
        term_structure_slope = COALESCE(term_structure_slope, %s),
        smile_convexity    = COALESCE(smile_convexity, %s),
        vol_risk_premium   = COALESCE(vol_risk_premium, %s)
    WHERE ticker = %s AND trade_date = %s AND feature_version = %s
      AND (atm_iv_percentile IS NULL
           OR skew_percentile IS NULL
           OR term_structure_slope IS NULL
           OR smile_convexity IS NULL
           OR vol_risk_premium IS NULL)
"""
```

Note: `backfill_run_id` is not touched — CR-A attribution preserved (accepted limitation per Step 0 diagnosis point 6).

**Deliverable:** runner ready, NULL-fill semantics enforced. Verify on test subset (Step 0 point 7) before full execution.

## Step 4 — Execute backfill

**Commit:** `cr-d/step-4: execute vol surface feature backfill`

Run script. Compute time ~2–3 hours.

**Pre-flight verification (test subset first):**
Run on the 5 locked test dates. Confirm:
- `2023-05-05` → skipped (insufficient history)
- `2023-09-01` → 5 features computed, row updated
- `2024-08-05` → computed; inspect `atm_iv_percentile` manually (expect high)
- `2025-01-10` → computed
- Re-run on `2023-09-01` → 0 rows updated (COALESCE idempotence)

Then run full corpus.

**Post-run verification:**

```sql
SELECT
  COUNT(*) FILTER (WHERE atm_iv_percentile IS NULL)    AS null_iv_pct,
  COUNT(*) FILTER (WHERE skew_percentile IS NULL)      AS null_skew,
  COUNT(*) FILTER (WHERE term_structure_slope IS NULL) AS null_ts,
  COUNT(*) FILTER (WHERE smile_convexity IS NULL)      AS null_conv,
  COUNT(*) FILTER (WHERE vol_risk_premium IS NULL)     AS null_vrp,
  COUNT(*) AS total
FROM bt_daily_features
WHERE feature_version = 'v0.5.0-rebuilt';
-- Expected: NULL counts ≤ ~60 (insufficient-history skips at corpus start)
```

## Step 5 — Smoke tests

**Commit:** `cr-d/step-5: smoke tests for vol surface features`

Tests:

1. **Percentile ranges.** All percentile columns in [0, 100]. No outliers.

2. **Distribution sanity.** Percentile columns approximately uniformly distributed across corpus. Histogram check.

3. **Known event verification.**
   - `2024-08-05` (VIX spike ~65): `atm_iv_percentile` expected near 100.
   - Pick 2–3 known quiet periods (mid-2024 low-vol dates): `atm_iv_percentile` near 0–30.

4. **Skew sign sanity.** Raw skew (`vol75 − vol25`) should be positive for >80% of corpus rows (SPX structural put-skew demand). If majority negative: computation bug. Note: `skew_percentile` is always 0–100 regardless of sign — check the raw skew via re-query, not the percentile column.

5. **Original v0.5.0 untouched.**
   ```sql
   SELECT COUNT(*) FROM bt_daily_features
   WHERE feature_version = 'v0.5.0'
     AND (atm_iv_percentile IS NOT NULL OR skew_percentile IS NOT NULL);
   -- Must be 0
   ```

6. **No structural feature changes.** Spot-check `feature_vector` JSONB on 3 rows pre/post CR-D. Must be unchanged.

Store all results in `bt_backfill_runs.smoke_test_results`. Self-assess.

## Wrap criteria

- All 5 steps committed
- `bt_backfill_runs` row has `status IN ('completed', 'completed_with_warnings')`
- [[Roadmap]] updated: CR-D marked complete; CR-E, CR-F, CR-G moved to "ready"
- **Drafting trigger:** if CR-D completion is the halfway-point of current batch, draft next 6-8 CRs (see [[Roadmap]] drafting trigger section)

## Status updates

**Spec amendment 2026-05-25 (Step 0 diagnosis):** Step 0 found the 5 vol columns do not exist in `bt_daily_features` (spec's assumption they were "v0.5 NULL placeholders" was incorrect). New Step 1 added for DDL migration (ALTER TABLE + column-scoped GRANT). Existing Steps 1–4 renumbered to 2–5. Additional amendments: term structure uses nearest-DTE expiration (ABS-distance LIMIT 1), not exact 30/90; VRP source explicitly `ironbeam_es_1m_bars` with `DATE(datetime)` grouping; smoke test #4 corrected to check raw skew sign (not percentile); `backfill_run_id` COALESCE preserves CR-A attribution (accepted limitation, future architectural CR). Test subset locked at 5 dates (2023-05-05, 2023-09-01, 2024-08-05, 2025-01-10, re-run idempotence). Lesson 2 discipline: `compute_*` returns None / raises on insufficient inputs; no default-percentile fallbacks.

## Open questions

- 60-session lookback is a default; should it be longer for `term_structure_slope` and `smile_convexity` (slower-moving features)? **Default 60 for v1; revisit if smoke shows odd behavior.**
- `term_structure_slope` stores raw spread (IV points), not percentile. Should it also store a percentile (like the other features)? Spec currently stores raw only. **Deferred; revisit in CR-E/F if the consuming feature needs it.**
