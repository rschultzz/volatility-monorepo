-- CR-BZ DDL. Apply as the table owner (rschultz), not dash_backfill_writer.
-- NOT applied by the implementation commits; applied on Ryan's go, before the
-- "Ironbeam Cron" deploy (revision log) and before the repair run (repair log + grants).
-- New tables owned by rschultz get dash_backfill_writer INSERT/SELECT via the default ACL.

BEGIN;

-- A1.2: every snapshot bar that differs from the stored row (insert-only).
-- One row per distinct revision: re-seeing the same snapshot values is skipped by the PK.
CREATE TABLE IF NOT EXISTS ironbeam_es_1m_bars_revision_log (
    datetime        timestamp without time zone NOT NULL,  -- bar start, naive UTC (as stored)
    stored_open     double precision,
    stored_high     double precision,
    stored_low      double precision,
    stored_close    double precision,
    stored_volume   double precision,
    snap_open       double precision NOT NULL,
    snap_high       double precision NOT NULL,
    snap_low        double precision NOT NULL,
    snap_close      double precision NOT NULL,
    snap_volume     double precision NOT NULL DEFAULT -1,  -- -1 = snapshot carried no volume
    seen_at         timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (datetime, snap_open, snap_high, snap_low, snap_close, snap_volume)
);

COMMIT;
