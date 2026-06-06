# CR-037 (CR-AB) — Open-Straddle Implied Move

> Authority: vault note `2026-06-05 - CR-037 — Open-Straddle Implied Move.md`
> Branch: `feat/CR-AB-open-straddle-implied-move`

## Problem

`compute_and_upsert_daily_features` runs `_IMPLIED_MOVE_SQL` with `ORDER BY snapshot_pt DESC`
and `WHERE trade_date = store_trade_date` (the forward-stamped date). At 03:01 PDT that date
has no `orats_monies_minute` rows yet → `implied_move = 0` → `backfill_outcomes` skips the row
→ proposals endpoint errors "no session_open_t0". Recurs every trading day.

## Goal

1. EOD cron (03:01 PDT) writes the `bt_daily_features` row with non-IV features only
   (`implied_move_1d = NULL`, σ-features unpopulated). Landscape/GEX unchanged.
2. New post-open cron (13:35 UTC / 06:35 PDT) computes `implied_move_1d` from the pinned
   06:33 PDT open-straddle ATM IV, recomputes σ-features, and UPDATEs the row.
3. Historical corpus re-backfilled onto open-straddle basis under `v0.6.0-openiv`.
4. `backfill_outcomes` moved to 13:40 UTC (after post-open job). `sweep_pending_outcomes`
   stays at 11:05 UTC (reads only historical rows, not today's fresh IV).

## Locked decisions

| # | Decision | Value |
|---|----------|-------|
| 4 | Unpopulated state | `implied_move_1d = NULL` + σ-features NULL (NOT sentinel 0) |
| 5 | feature_version | `v0.6.0-openiv` — IV-basis change ONLY |
| 6 | Outcome ordering | Move `backfill_outcomes` to 13:40 UTC (option a) |

## Design constraints (frozen)

1. `orats-eod-gamma` schedule unchanged (11:00 UTC / 03:01 PDT). Only change: drop IV computation.
2. Pin the open straddle explicitly: first `orats_monies_minute` snapshot at/after 06:33 PDT
   with smallest `dte > 0`. NOT `ORDER BY snapshot_pt DESC`.
3. Separate new cron — do NOT fold into `backfill_outcomes` or `sweep_pending_outcomes`.
4. EOD leaves IV-dependent features as NULL (honest-missing, distinguishable from computed).
5. Re-backfill under `v0.6.0-openiv`; leave `v0.5.0-rebuilt` intact; flip canonical after verify.
6. `backfill_outcomes` moves to 13:40 UTC. `sweep_pending_outcomes` stays at 11:05 UTC.

## Step 0 — verification gates (no commits)

### Gate 0.1 — `_IMPLIED_MOVE_SQL` call path ✅

- **Only live producer** of `bt_daily_features.implied_move_1d`: `compute_and_upsert_daily_features`
  in `day_features.py`, called by `job_orats_eod.py` on `store_trade_date`.
- **σ-normalized features** (`cluster_1/2/3_signed_distance_sigma`, `nearest_neg_signed_distance_sigma`)
  are computed in `extract_features` by dividing by `implied_move`. The split MUST move BOTH
  the IV lookup AND these five expressions to the post-open job.
- **Secondary importers** (not the live path): `cr_a_backfill_landscape.py` and
  `cr_a_determinism_check.py` import `_IMPLIED_MOVE_SQL` directly. These are historical one-off
  scripts. To avoid breaking them, `_IMPLIED_MOVE_SQL` is retained as-is; the new pinned-floor
  query is a separate constant `_OPEN_STRADDLE_SQL`.
- `backfill_daily_features.py` calls `compute_and_upsert_daily_features` directly — this is
  also the live re-backfill path; it will call the refactored EOD-only function (no IV),
  which is correct for Step 4 (IV is written by a separate pass).

### Gate 0.2 — `snapshot_pt` timezone ✅

- Column type: `TIMESTAMP WITHOUT TIME ZONE`.
- Values are **PT wall-clock**: 06:30 = RTH open, 13:00 = RTH close. NOT UTC.
- `snapshot_pt::time >= '06:33'` is the correct PT-floor expression. No timezone cast needed.

### Gate 0.4 — Data-safety re-backfill pattern ✅ (with DDL prerequisite noted)

- `active BOOLEAN DEFAULT true` ✅; `backfill_run_id UUID nullable` ✅.
- `bt_daily_features_active` view: `SELECT ... FROM bt_daily_features WHERE active = true` ✅.
- PK: `(ticker, trade_date, feature_version)` — `v0.6.0-openiv` rows don't conflict with
  `v0.5.0-rebuilt` rows on INSERT. ✅
- **DDL prerequisite confirmed:** `dash_backfill_writer` currently has NO UPDATE on
  `bt_daily_features` (verified empirically during manual unblock — DATABASE_URL was required).
  The column-level GRANT in Step 3 (`infra/sql/bt_daily_features_backfill_writer_feature_update.sql`)
  must be applied before the post-open cron can run. This is a pre-existing plan item, not a
  new finding.

### Gate 0.6 — Open-straddle corpus coverage ✅ CLEAN

- Full corpus: **743 active dates** (2023-05-01 to 2026-06-05) at `v0.5.0-rebuilt`.
- Corpus dates missing a 06:33+ DTE>0 ATM IV snapshot: **0 of 743**.
- Oldest 5 dates (2023-05-01 through 2023-05-05) all confirmed present.
- No fallback logic needed for Step 4 re-backfill.

**All four gates clear — no contradictions. Proceeding to Step 1.**

## Step 1 — Split `day_features.py`

**Commit:** `cr-ab/step-1: separate implied-move computation from EOD feature write`

- Refactor `compute_and_upsert_daily_features` so implied-move lookup + σ-feature computation
  are separately callable. EOD path writes non-IV features with `implied_move_1d = NULL` and
  σ-features NULL.
- New callable `compute_and_upsert_open_implied_move(conn, ticker, trade_date, *, version)`:
  pins 06:33-or-later open-straddle ATM IV, computes `implied_move_1d`, recomputes σ-features,
  UPDATEs the existing row.
- `_IMPLIED_MOVE_SQL` floor changes from `ORDER BY snapshot_pt DESC` to
  `snapshot_pt::time >= '06:33' ORDER BY snapshot_pt ASC, dte ASC LIMIT 1`.

**Contradiction-stop:** Non-IV features must be byte-identical to current output for a known date.

## Step 2 — EOD cron wiring

**Commit:** `cr-ab/step-2: EOD job writes non-IV features only`

- Swap `compute_and_upsert_daily_features` call in `job_orats_eod.py` to non-IV variant.
- Log line updated so 03:01 run no longer prints `implied_move=0.00`.

## Step 3 — Post-open script + cron

**Commit:** `cr-ab/step-3: post-open implied-move job`

- New `scripts/cr_ab_open_implied_move.py`: resolves most recent trade_date needing implied move,
  calls `compute_and_upsert_open_implied_move`, uses `BACKFILL_DATABASE_URL` + `assert_role_or_die`
  + `backfill_run` scaffolding.
- Render Cron: branch `main`, Auto-Deploy ON, root dir blank,
  build `pip install -r apps/cron/requirements.txt`, `PYTHON_VERSION=3.13.4`,
  env `BACKFILL_DATABASE_URL`, command `python scripts/cr_ab_open_implied_move.py`,
  schedule `35 13 * * 1-5`.
- Move `backfill_outcomes` from `0 11 * * 1-5` → `40 13 * * 1-5` (manual Render step).
- **Prerequisite DDL** (surfaces as manual Render step — see gate 0.4):
  `GRANT UPDATE (feature_vector, feature_config_hash, regime_at_classification, computed_at)
  ON bt_daily_features TO dash_backfill_writer;`

## Step 4 — Corpus re-backfill

**Commit:** `cr-ab/step-4: re-backfill corpus onto open-straddle basis (v0.6.0-openiv)`

- INSERT with `v0.6.0-openiv` ON CONFLICT DO NOTHING for all corpus dates.
  Reads 06:33 open straddle for each historical date.
- `v0.5.0-rebuilt` untouched.
- Verify against manual 06:33-vs-12:57 comparison numbers from the ADR.

## Step 5 — Flip canonical + live-path review

**Commit:** `cr-ab/step-5: promote v0.6.0-openiv to canonical`

- Flip `CANONICAL_FEATURE_VERSION` in `canonical_version.py` to `v0.6.0-openiv`.
- Review `/api/analogues` live implied-move formula (`spot × landscapeIv × √(1/252)`) for
  consistency with open-straddle corpus. Document as in-scope fix or follow-on.

## Smoke tests

1. EOD row: `implied_move_1d = NULL`, σ-features NULL (not 0).
2. Post-open fill: `implied_move_1d` from 06:33 snapshot; σ-features computed; pinned (not fire-time).
3. Outcomes unblock: `backfill_outcomes` at 13:40 UTC sees non-NULL `implied_move`; proposals render.
4. Re-backfill: sampled `v0.6.0-openiv` rows carry open-basis IV; `v0.5.0-rebuilt` unchanged.
5. Non-IV features identical between versions for a known date.

## Render dashboard steps (manual)

- DDL: apply `infra/sql/bt_daily_features_backfill_writer_feature_update.sql` (new file, Step 1)
- New cron service: `35 13 * * 1-5`, command `python scripts/cr_ab_open_implied_move.py`
- Reschedule `backfill_outcomes`: `0 11 * * 1-5` → `40 13 * * 1-5`

## Out of scope

- Chart `expected_move_levels` / condor `sigma_pts` (separate pipelines).
- `next_business_day` holiday issue.
- Vol-column or per-bucket feature work — keep version diff clean.
