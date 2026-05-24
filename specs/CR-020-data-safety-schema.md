---
type: cr
cr_id: CR-0
title: Data Safety Schema
aliases: ["CR-0 — Data Safety Schema", "CR-0"]
status: active
run_mode: interactive
phase: 0
size: small
estimated_days: 1
data_safety_class: schema_migration
dependencies: []
depended_on_by: [CR-A, CR-B, CR-C, CR-D, CR-E, CR-F, CR-G, CR-H]
branch_name: cr-0-data-safety-schema
sequence_number: 020
started: 2026-05-24
completed:
last_commit_sha:
tags: [dash, cr, schema, data-safety, postgres, foundational]
---

# CR-0 — Data Safety Schema

## Goal

Establish the four-layer data safety architecture so all subsequent backfills and writes run within strong guarantees against accidental data loss.

## Context

All overnight and unattended runs land write operations against the live database. Without this foundation: any bug in Claude Code's execution could DELETE, TRUNCATE, or DROP. With this foundation: such operations are rejected at the database level, and even legitimate writes are tagged, soft-deletable, and reversible.

See [[2026-05-24 - Data Safety Protocol|Data Safety Protocol]] for the full model.

## Step 0 — Diagnosis (no commits)

- Verify current Postgres version on Render supports `gen_random_uuid()` (need `pgcrypto` extension or pg 13+).
- List all tables that will receive backfill writes: `bt_daily_features` exists; `bt_daily_outcomes` will be created in CR-B. Confirm no other tables need the columns added in this CR right now.
- Identify all current connection strings and which roles they use. Application reads currently use what user? Application writes use what user? (Need this to plan permission grants correctly.)
- Confirm the canonical feature_version constant location in code (probably `packages/shared/feature_version.py` or similar). This is where promotion will happen later.
- Decide where the `dash_backfill_writer` connection string lives. Probably new env var `BACKFILL_DATABASE_URL` separate from `DATABASE_URL`.

## Step 1 — Create `dash_backfill_writer` role

**Commit:** `cr-0/step-1: create dash_backfill_writer role with scoped permissions`

Run as superuser:

```sql
CREATE ROLE dash_backfill_writer LOGIN PASSWORD '<from env>';
GRANT CONNECT ON DATABASE dash TO dash_backfill_writer;
GRANT USAGE ON SCHEMA public TO dash_backfill_writer;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO dash_backfill_writer;
GRANT INSERT ON ALL TABLES IN SCHEMA public TO dash_backfill_writer;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO dash_backfill_writer;

ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT SELECT, INSERT ON TABLES TO dash_backfill_writer;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT USAGE ON SEQUENCES TO dash_backfill_writer;
```

**Deliverable:** role exists, can SELECT and INSERT, cannot DELETE/TRUNCATE/DROP.

**Verification (as `dash_backfill_writer`):**

```sql
SELECT count(*) FROM bt_daily_features;      -- should succeed
DELETE FROM bt_daily_features WHERE 1=0;      -- should FAIL: permission denied
TRUNCATE bt_daily_features;                   -- should FAIL
DROP TABLE bt_daily_features;                 -- should FAIL
```

## Step 2 — Add safety columns to existing data tables

**Commit:** `cr-0/step-2: add active/deactivated/run_id columns to bt_daily_features`

```sql
ALTER TABLE bt_daily_features
  ADD COLUMN active BOOLEAN NOT NULL DEFAULT TRUE,
  ADD COLUMN deactivated_at TIMESTAMP,
  ADD COLUMN deactivated_reason TEXT,
  ADD COLUMN backfill_run_id UUID;

CREATE INDEX idx_features_active ON bt_daily_features (active) WHERE active = FALSE;
CREATE INDEX idx_features_run_id ON bt_daily_features (backfill_run_id);

-- Grant UPDATE on safety columns only
GRANT UPDATE (active, deactivated_at, deactivated_reason, backfill_run_id)
  ON bt_daily_features TO dash_backfill_writer;

-- ALSO grant UPDATE on the vol surface NULL columns so CR-D can fill them
GRANT UPDATE (atm_iv_percentile, skew_percentile, term_structure_slope,
              smile_convexity, vol_risk_premium)
  ON bt_daily_features TO dash_backfill_writer;
```

Note: only specific columns get UPDATE permission. Structural feature columns (regime_kind, drift_target, expected_move, etc.) remain unwriteable post-INSERT by this role.

**Deliverable:** schema extended with safety columns; column-scoped UPDATEs granted.

**Verification:**

```sql
\d bt_daily_features  -- confirm new columns visible

-- As dash_backfill_writer:
UPDATE bt_daily_features SET active = FALSE WHERE id = X;          -- should work
UPDATE bt_daily_features SET regime_kind = 'changed' WHERE id = X; -- should FAIL
```

## Step 3 — Create `bt_backfill_runs` table

**Commit:** `cr-0/step-3: create bt_backfill_runs tracking table`

```sql
CREATE TABLE bt_backfill_runs (
    run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cr_id VARCHAR NOT NULL,
    started_at TIMESTAMP NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMP,
    status VARCHAR NOT NULL DEFAULT 'running',
    rows_inserted INT DEFAULT 0,
    rows_updated INT DEFAULT 0,
    smoke_test_results JSONB,
    self_assessment TEXT,
    notes TEXT
);

CREATE INDEX idx_runs_cr_started ON bt_backfill_runs (cr_id, started_at DESC);

GRANT SELECT, INSERT, UPDATE ON bt_backfill_runs TO dash_backfill_writer;
```

**Deliverable:** tracking table in place.

**Verification:**

```sql
-- As dash_backfill_writer:
INSERT INTO bt_backfill_runs (cr_id, status) VALUES ('TEST', 'running')
  RETURNING run_id;
UPDATE bt_backfill_runs SET status = 'completed' WHERE cr_id = 'TEST';
DELETE FROM bt_backfill_runs WHERE cr_id = 'TEST';  -- should FAIL

-- Cleanup as superuser:
DELETE FROM bt_backfill_runs WHERE cr_id = 'TEST';
```

## Step 4 — Create application read view

**Commit:** `cr-0/step-4: create bt_daily_features_active view`

```sql
CREATE VIEW bt_daily_features_active AS
SELECT * FROM bt_daily_features WHERE active = TRUE;

GRANT SELECT ON bt_daily_features_active TO dash_backfill_writer;
-- Also grant to whatever role the application uses for reads
```

**Deliverable:** application-facing view that excludes inactive rows.

## Step 5 — Update application reads to use view

**Commit:** `cr-0/step-5: route application reads through bt_daily_features_active`

Find all SELECT references to `bt_daily_features` in:
- `Analogues/service.py`
- `Proposals/service.py`
- Any KNN / ranking module
- Audit-related reads

Decide: rename references to use the view, OR add `WHERE active = TRUE` to each query directly. Recommend the view approach — single point of change, harder to forget.

**Deliverable:** application reads inactive rows nowhere.

**Verification:** for each updated query, run it and verify result counts match prior behavior (no inactive rows exist yet, so counts should be identical).

## Step 6 — Add `BACKFILL_DATABASE_URL` env var and safety helper

**Commit:** `cr-0/step-6: add BACKFILL_DATABASE_URL env var and backfill_safety module`

- Add `BACKFILL_DATABASE_URL` to env config / `.env.example`.
- Document in CLAUDE.md that unattended runs use this connection string.
- Create `packages/shared/backfill_safety.py` with:
  - `get_backfill_db_conn()` — connects using `BACKFILL_DATABASE_URL`
  - `verify_safe_role(conn)` — runs `SELECT current_user`, asserts result is `dash_backfill_writer`; raises if not
  - `backfill_run(conn, cr_id)` — context manager that creates a `bt_backfill_runs` row on enter, finalizes status on exit (with exception handling for `aborted` status)
  - `update_run_smoke(conn, run_id, smoke_results, self_assessment)` — writes smoke output to the run row

**Deliverable:** clean separation between application connection (default role) and backfill connection (locked-down role).

## Smoke test (run as `dash_backfill_writer`)

```sql
-- 1. Permission verification
SELECT current_user;                          -- must be 'dash_backfill_writer'

-- 2. INSERT works
INSERT INTO bt_backfill_runs (cr_id, status)
VALUES ('CR-0-SMOKE', 'running')
RETURNING run_id;

-- 3. DELETE blocked on data tables
DELETE FROM bt_daily_features LIMIT 1;        -- must FAIL: permission denied

-- 4. TRUNCATE blocked
TRUNCATE bt_daily_features;                   -- must FAIL

-- 5. DROP blocked
DROP TABLE bt_daily_features;                 -- must FAIL

-- 6. Soft delete works
INSERT INTO bt_daily_features (ticker, trade_date, feature_version, backfill_run_id)
VALUES ('TEST', '2026-01-01', 'TEST', '<run_id>');
UPDATE bt_daily_features SET active = FALSE
WHERE feature_version = 'TEST';               -- must work

-- 7. View excludes inactive
SELECT count(*) FROM bt_daily_features WHERE feature_version = 'TEST';        -- 1
SELECT count(*) FROM bt_daily_features_active WHERE feature_version = 'TEST'; -- 0

-- 8. Cleanup (as superuser only)
DELETE FROM bt_daily_features WHERE feature_version = 'TEST';
UPDATE bt_backfill_runs SET status = 'aborted' WHERE cr_id = 'CR-0-SMOKE';
```

All 7 "must" conditions pass → CR-0 wraps.

## Wrap criteria

- All 6 steps committed.
- Smoke test passes end-to-end.
- `BACKFILL_DATABASE_URL` is in env config and documented in CLAUDE.md.
- `packages/shared/backfill_safety.py` exists with role assertion + run context manager.
- Application reads route through view.
- [[Roadmap]] updated: CR-0 marked complete, CR-A status moved to "ready".

## Status updates

### 2026-05-24 — Activated

CR-0 activated as CR-020. Branch `cr-0-data-safety-schema` created off `origin/main` at `9298b75`.

### 2026-05-24 — Step 0 diagnosis

(filling in during execution)

## Open questions

(none at draft time)
