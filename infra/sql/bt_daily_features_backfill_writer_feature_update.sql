-- ============================================================
-- bt_daily_features — extend dash_backfill_writer UPDATE grant (CR-037)
--
-- The CR-AB post-open implied-move cron (scripts/cr_ab_open_implied_move.py)
-- runs as dash_backfill_writer and must UPDATE feature_vector + computed_at
-- on the row previously written by the EOD cron once the 06:33 PDT
-- open-straddle snapshot is available.
--
-- Only the three data columns the post-open job legitimately writes are
-- granted — regime_at_classification is intentionally excluded because it
-- is set by the EOD cron from the landscape and must not be altered by the
-- post-open path.
--
-- Apply before activating the Step 3 cron on Render:
--   psql $DATABASE_URL \
--     -f infra/sql/bt_daily_features_backfill_writer_feature_update.sql
--
-- Idempotent: granting the same privilege twice is a no-op in PostgreSQL.
-- ============================================================

BEGIN;

GRANT UPDATE (feature_vector, feature_config_hash, computed_at)
  ON bt_daily_features TO dash_backfill_writer;

COMMIT;
