-- ============================================================
-- bt_daily_features — add safety columns (CR-020)
--
-- Four columns enabling the four-layer data safety architecture:
--   active           — soft-delete flag; FALSE = deactivated
--   deactivated_at   — when the row was soft-deleted
--   deactivated_reason — why (human note or CR ID)
--   backfill_run_id  — FK-by-convention to bt_backfill_runs.run_id
--
-- Idempotent: IF NOT EXISTS guards throughout.
-- Apply after dash_backfill_writer_role.sql (role must exist for GRANT).
-- ============================================================

BEGIN;

ALTER TABLE bt_daily_features
  ADD COLUMN IF NOT EXISTS active            BOOLEAN   NOT NULL DEFAULT TRUE,
  ADD COLUMN IF NOT EXISTS deactivated_at    TIMESTAMP,
  ADD COLUMN IF NOT EXISTS deactivated_reason TEXT,
  ADD COLUMN IF NOT EXISTS backfill_run_id   UUID;

CREATE INDEX IF NOT EXISTS idx_features_active
  ON bt_daily_features (active) WHERE active = FALSE;

CREATE INDEX IF NOT EXISTS idx_features_run_id
  ON bt_daily_features (backfill_run_id);

-- UPDATE permission scoped to safety columns only.
-- Structural feature columns (feature_vector, regime_at_classification, etc.)
-- remain unwriteable post-INSERT by this role.
-- Vol-surface column GRANTs deferred to CR-D (columns don't exist yet).
GRANT UPDATE (active, deactivated_at, deactivated_reason, backfill_run_id)
  ON bt_daily_features TO dash_backfill_writer;

COMMIT;
