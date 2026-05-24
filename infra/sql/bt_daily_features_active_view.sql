-- ============================================================
-- bt_daily_features_active — application read view (CR-020)
--
-- Filters bt_daily_features to active=TRUE rows only.
-- All application reads should go through this view so that
-- soft-deleted (deactivated) rows are never surfaced.
--
-- rschultz (view owner) and new_db_cred (inherits from rschultz)
-- already have access — only dash_backfill_writer needs an explicit
-- grant.
-- ============================================================

CREATE OR REPLACE VIEW bt_daily_features_active AS
SELECT * FROM bt_daily_features WHERE active = TRUE;

GRANT SELECT ON bt_daily_features_active TO dash_backfill_writer;
