---
type: cr
cr_id: CR-AA
title: Pending Outcome Sweep — promote matured pending_history rows to computed
aliases: ["CR-AA — Pending Outcome Sweep", "CR-AA"]
status: in-progress
started: 2026-06-04
sequence_number: 36
run_mode: interactive
phase: 5
size: small
estimated_days: 1
data_safety_class: write_backfill
dependencies: [CR-022]
depended_on_by: []
branch_name: feat/CR-AA-pending-outcome-sweep
stop_conditions:
  - "pre-flight check: current_user != 'dash_backfill_writer'"
  - "Step-1 extraction changes CR-022 output in any way (byte-identical check fails)"
  - "sweep would touch computed, na_regime, or na_data rows"
tags: [dash, cr, bt_daily_outcomes, backfill, cron, sweep, outcomes]
---

# CR-AA — Pending Outcome Sweep

## Goal

A daily sweep that finds `pending_history` rows in `bt_daily_outcomes` whose
horizon window has closed (i.e. enough RTH sessions now exist in
`ironbeam_es_1m_bars`), re-runs `compute_outcome`, and UPDATEs each row to its
terminal status (`computed` or `na_data`). New separate script + its own Render
cron. The existing CR-022 backfill INSERT path (`ON CONFLICT DO NOTHING`) is
left untouched.

## Background

`compute_outcome` returns `pending_history` when fewer than
`bucket_sessions(dominant_bucket)` RTH sessions exist after `trade_date`. A
`30+ DTE` row stays pending for ~60 sessions (~3 calendar months). The function
is pure and correct: given enough bars it returns `computed`; given too few it
returns `pending_history`. CR-AA re-runs the existing function on rows whose
window has now closed.

The CR-022 backfill INSERT uses `ON CONFLICT DO NOTHING` — re-running it on a
date that already has a `pending_history` row is a permanent no-op. There is no
UPDATE path anywhere that promotes `pending_history → computed`.

## Critical design constraints (frozen)

1. **Separate script + separate cron.** `scripts/cr_aa_sweep_pending_outcomes.py`
   and its own Render Cron Job. Backfill INSERT path stays `ON CONFLICT DO
   NOTHING`; the sweep is a distinct `UPDATE`. A bug in sweep logic must not
   reach the insert path.

2. **Promote `pending_history` ONLY.** `WHERE outcome_status = 'pending_history'`.
   Never touch `computed`, `na_regime`, `na_data`. `na_data` is a permanent
   terminal state — retrying it would churn forever.

3. **Target only matured rows.** `horizon_end_date <= <latest fully-closed RTH
   session in ironbeam_es_1m_bars>`. Each row checked once, when its window
   closes — not re-swept daily.

4. **Shared per-date helper.** Extract the per-date computation from
   `cr_b_backfill_outcomes.py`'s main loop into ONE helper in
   `packages/shared/outcomes_runner.py`. Both the CR-022 backfill and the CR-AA
   sweep call it. CR-022 output must be byte-identical after extraction.

5. **`backfill_run_id` overwrite on promotion.** Overwrite `backfill_run_id`
   with the sweep's run id and set `computed_at = NOW()`.

6. **Window closed but bars unusable → promote to `na_data`.** If
   `compute_outcome` returns `na_data` on a matured row, promote
   `pending_history → na_data`. Leaving it pending would re-check forever.

## Step 0 — Diagnose (no commits)

1. Confirm `horizon_end_date` semantics (`horizon.index[-1]`).
2. Lock the SQL expression for latest fully-closed RTH session (reuse
   `future_horizon_count` smoke SQL in cr_b).
3. Identify exact lines to extract into shared helper; confirm extraction
   doesn't change CR-022 behavior.
4. Count current `pending_history` rows by bucket/horizon.

## Step 1 — Extract shared per-date helper

**Commit:** `cr-aa/step-1: extract shared per-date outcome computation from cr_b backfill`

- New `packages/shared/outcomes_runner.py` with `compute_outcome_for_date(...)`.
- Rewire `cr_b_backfill_outcomes.py` to call the helper.
- CR-022 output must be byte-identical (re-run on known date → identical rows).

## Step 2 — Sweep script

**Commit:** `cr-aa/step-2: pending_history → computed/na_data sweep script`

- New `scripts/cr_aa_sweep_pending_outcomes.py`.
- Safety scaffolding: `get_backfill_db_conn` + `assert_role_or_die` +
  `backfill_run`.
- Target query: pending rows where `horizon_end_date <= latest_closed_session`.
- Calls shared helper → `compute_outcome`. UPDATEs with new status +
  `backfill_run_id` + `computed_at`.
- `--dry-run`, `--limit`, `--ticker` (default SPX), optional `--from-date`.
- Smoke block: counts by status, promoted count, remaining matured pending
  (should be 0), null_run_id check.

## Step 3 — Render cron (Ryan)

- New Render Cron Job: `python scripts/cr_aa_sweep_pending_outcomes.py`
- Schedule: `5 11 * * 1-5` (11:05 AM UTC = 03:05 PT Mon–Fri).
- `BACKFILL_DATABASE_URL` env var set.
- Manual trigger verify: promotes expected pending count; exits 0.

## Smoke tests

1. A matured pending row with existing bars → `computed`.
2. A pending row whose `horizon_end_date` is in the future is NOT selected.
3. `computed`, `na_regime`, `na_data` rows are never touched.
4. A matured pending row with unusable bars → `na_data`.
5. Second immediate run promotes 0 (idempotent).
6. After Step-1 extraction, CR-022 re-run on a known date produces identical output.
7. Promoted rows carry the sweep run id and a fresh `computed_at`.

## Files

- **New:** `packages/shared/outcomes_runner.py`
- **Modified:** `scripts/cr_b_backfill_outcomes.py` (call shared helper)
- **New:** `scripts/cr_aa_sweep_pending_outcomes.py`
