-- ============================================================
-- bt_backfill_runs — run-level tracking table (CR-020)
--
-- Every unattended backfill creates one row on entry and updates
-- it on exit. Provides an audit trail of what ran, when, and
-- whether it passed its own smoke test.
--
-- dash_backfill_writer has SELECT, INSERT, UPDATE — no DELETE.
-- Soft-cancellation is done via status='aborted'.
--
-- Idempotent: IF NOT EXISTS throughout.
-- ============================================================

BEGIN;

CREATE TABLE IF NOT EXISTS bt_backfill_runs (
    run_id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    cr_id               VARCHAR     NOT NULL,
    started_at          TIMESTAMP   NOT NULL DEFAULT NOW(),
    completed_at        TIMESTAMP,
    status              VARCHAR     NOT NULL DEFAULT 'running',
    rows_inserted       INT         DEFAULT 0,
    rows_updated        INT         DEFAULT 0,
    smoke_test_results  JSONB,
    self_assessment     TEXT,
    notes               TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_cr_started
  ON bt_backfill_runs (cr_id, started_at DESC);

GRANT SELECT, INSERT, UPDATE ON bt_backfill_runs TO dash_backfill_writer;

COMMIT;
