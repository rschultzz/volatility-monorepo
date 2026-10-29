-- ============================================================
-- bt_derived_repair_archive — BLIND archive of derived rows before a repair UPDATE (CR-DO)
--
-- One row per derived row a repair run rewrites: the complete old row as JSON and the
-- columns written. Holds holdout / R-b values (B4 blind-maintenance clause, 2026-10-29):
-- never queried by Ryan or any Claude session.
--
-- Privileges (Ryan, chat 2026-10-29, D5): dash_backfill_writer INSERT only (no SELECT,
-- UPDATE, DELETE); no SELECT for any other role. The owner's own table privileges are
-- revoked too (it keeps ownership: ALTER / DROP / GRANT). The schema default ACL grants
-- dash_backfill_writer SELECT + INSERT on new tables, so that is revoked explicitly.
-- A restore is a deliberate act: the owner grants itself SELECT, recorded in a CR.
--
-- Run as the owner role (DATABASE_URL), not dash_backfill_writer.
-- ============================================================

BEGIN;

CREATE TABLE bt_derived_repair_archive (
    run_id           UUID        NOT NULL,
    cr_id            VARCHAR     NOT NULL,
    table_name       VARCHAR     NOT NULL,
    ticker           VARCHAR     NOT NULL,
    feature_version  VARCHAR     NOT NULL,
    trade_date       DATE        NOT NULL,
    regime           VARCHAR     NOT NULL CHECK (regime IN ('train', 'holdout', 'rb', 'fwd')),
    columns_written  TEXT[]      NOT NULL,
    old_row          JSONB       NOT NULL,
    archived_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (run_id, table_name, ticker, feature_version, trade_date)
);

COMMENT ON TABLE bt_derived_repair_archive IS
    'BLIND (CR-DO, B4): contains holdout / R-b values; never queried by Ryan or any Claude session. '
    'Insert-only for dash_backfill_writer; no role has SELECT.';

REVOKE ALL ON bt_derived_repair_archive FROM PUBLIC;
REVOKE ALL ON bt_derived_repair_archive FROM dash_backfill_writer;
GRANT INSERT ON bt_derived_repair_archive TO dash_backfill_writer;
REVOKE ALL ON bt_derived_repair_archive FROM CURRENT_USER;

COMMIT;
