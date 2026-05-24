-- ============================================================
-- dash_backfill_writer — restricted backfill role (CR-020)
--
-- Has SELECT + INSERT on all tables; no DELETE / TRUNCATE / DROP.
-- Used by all unattended CR backfills via BACKFILL_DATABASE_URL.
--
-- Run with psql variable injection (never commit the password):
--   psql "$PSQL_URL" -v pw='<generated_password>' \
--     -f infra/sql/dash_backfill_writer_role.sql
-- ============================================================

BEGIN;

CREATE ROLE dash_backfill_writer LOGIN PASSWORD :'pw';
GRANT CONNECT ON DATABASE curve_trading TO dash_backfill_writer;
GRANT USAGE ON SCHEMA public TO dash_backfill_writer;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO dash_backfill_writer;
GRANT INSERT ON ALL TABLES IN SCHEMA public TO dash_backfill_writer;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO dash_backfill_writer;

ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT SELECT, INSERT ON TABLES TO dash_backfill_writer;
ALTER DEFAULT PRIVILEGES IN SCHEMA public
  GRANT USAGE ON SEQUENCES TO dash_backfill_writer;

COMMIT;
