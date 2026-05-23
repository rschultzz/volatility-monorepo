-- ============================================================
-- bt_daily_features — add regime_at_classification column (CR-016)
--
-- Step-0 discovery: bt_daily_features only stored feature_vector
-- (JSONB). The AuditFlags module needs to read the auto-classified
-- regime at flag-creation time, and get_effective_regime needs the
-- stored regime for its fallback.
--
-- Column is nullable for backwards-compatibility: rows written
-- before this migration have NULL; get_effective_regime re-
-- materializes from orats_gex_landscape in that case.
--
-- After applying this migration, run backfill_daily_features.py
-- (or wait for the next nightly cron) to populate the new column
-- on existing rows.
--
-- Idempotent: IF NOT EXISTS guard.
-- ============================================================

ALTER TABLE bt_daily_features
    ADD COLUMN IF NOT EXISTS regime_at_classification TEXT;
