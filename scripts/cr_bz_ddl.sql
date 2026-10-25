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

-- A1.3: old and new values of every repaired row, both tables, logged before the UPDATE.
CREATE TABLE IF NOT EXISTS ironbeam_es_1m_bars_repair_log (
    run_id        uuid NOT NULL,          -- bt_backfill_runs.run_id
    target_table  text NOT NULL CHECK (target_table IN ('ironbeam_es_1m_bars', 'es_minutes')),
    bar_ts        timestamptz NOT NULL,   -- bar start (ironbeam_es_1m_bars.datetime as UTC = es_minutes.ts_utc)
    old_open      double precision,
    old_high      double precision,
    old_low       double precision,
    old_close     double precision,
    old_volume    double precision,
    new_open      double precision NOT NULL,
    new_high      double precision NOT NULL,
    new_low       double precision NOT NULL,
    new_close     double precision NOT NULL,
    new_volume    double precision NOT NULL,  -- flow volume: a lower bound (CR-BZ Step 0 Q3)
    symbol        text NOT NULL,              -- contract of the flow minute
    logged_at     timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, target_table, bar_ts)
);

-- A1.3: the repair runs as dash_backfill_writer, which has only INSERT/SELECT on these
-- tables (measured 2026-10-25). Column-level UPDATE on OHLCV only; still no DELETE.
GRANT UPDATE (open, high, low, close, volume) ON ironbeam_es_1m_bars TO dash_backfill_writer;
GRANT UPDATE (open, high, low, close, volume) ON es_minutes TO dash_backfill_writer;

COMMIT;
