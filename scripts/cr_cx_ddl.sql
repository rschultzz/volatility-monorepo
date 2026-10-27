-- CR-CX DDL (Y8). Apply once with DATABASE_URL, as the table owner.
-- Forward log of the G1-down signal (B4: sessions >= 2026-10-27, written once, never changed).
-- Insert-only: the triggers below raise on any UPDATE or DELETE (as setup_reviews, CR-CC). TRUNCATE bypasses
-- row triggers; only the owner can truncate, and dash_backfill_writer has no TRUNCATE.
--
-- Grants: SELECT, INSERT on the three tables and USAGE on their sequences to dash_backfill_writer (the cron's
-- role). Nothing to PUBLIC.

BEGIN;

-- One row per session the script processed: logged (legs written) or skipped (reason), so a zero-event day
-- is distinguishable from a missed day.
CREATE TABLE IF NOT EXISTS forward_log_sessions (
    id                  bigserial PRIMARY KEY,
    definition_version  text NOT NULL,                      -- 'G1-down v1'
    session_date        date NOT NULL,
    status              text NOT NULL CHECK (status IN ('logged', 'skipped')),
    skip_reason         text,
    front_symbol        text,                               -- Databento raw_symbol of the session's front bars
    rth_bars            integer,                            -- of 390
    skew_snapshots      integer,                            -- 0DTE PM RTH snapshots kept by skew_minutes
    im_pts              double precision,                   -- implied_move_1d
    swing_pts           double precision,                   -- 12 x im / IM_train
    legs_total          integer,                            -- all swing legs in the CR-CL domain
    legs_primary        integer,                            -- e <= 11:00 PT
    legs_logged         integer,                            -- primary and ratio-defined (rows in forward_log_events)
    databento_rows      integer,                            -- es_flow_1s_databento rows inserted by this run
    databento_usd       double precision,
    code_sha            text NOT NULL,
    run_id              uuid,
    written_at          timestamptz NOT NULL DEFAULT now(),
    UNIQUE (definition_version, session_date)
);

-- Every primary, ratio-defined swing leg (RL2): G1-down is the signal; the rest carry the matched baseline.
CREATE TABLE IF NOT EXISTS forward_log_events (
    id                  bigserial PRIMARY KEY,
    signal              text NOT NULL,                      -- cell: 'G1-down', 'base-down', 'G1-up', ...
    grp                 text NOT NULL CHECK (grp IN ('G1', 'G2', 'base', 'neither')),
    dir                 smallint NOT NULL CHECK (dir IN (-1, 1)),
    definition_version  text NOT NULL,
    session_date        date NOT NULL,
    leg_start_ts_pt     timestamp without time zone NOT NULL,   -- naive PT, bar start of start_i
    leg_end_ts_pt       timestamp without time zone NOT NULL,   -- bar start of end_i (the extreme)
    confirm_ts_pt       timestamp without time zone NOT NULL,   -- bar start of confirm_i
    entry_price         double precision NOT NULL,          -- P0 = close of the confirmation bar
    start_px            double precision NOT NULL,
    end_px              double precision NOT NULL,
    swing_pts           double precision NOT NULL,
    size_pts            double precision NOT NULL,          -- |end_px - start_px|
    size_tercile        smallint NOT NULL,                  -- 0/1/2 at the frozen y26 cuts (Y5)
    cbucket             smallint NOT NULL,                  -- 30-min bucket of the confirmation bar from 06:30 PT
    edge                text NOT NULL CHECK (edge IN ('EDGE', 'INTERIOR')),
    ratio               double precision NOT NULL,
    actual_pp           double precision NOT NULL,
    expected_pp         double precision NOT NULL,
    band_pp             double precision NOT NULL,
    beyond_noise        boolean NOT NULL,
    front_symbol        text NOT NULL,
    code_sha            text NOT NULL,
    run_id              uuid,
    written_at          timestamptz NOT NULL DEFAULT now(),
    UNIQUE (definition_version, session_date, leg_end_ts_pt, dir)
);
CREATE INDEX IF NOT EXISTS forward_log_events_signal_idx ON forward_log_events (signal, session_date);

-- Outcomes for G1-down and base-down legs (h = +1), stop-free (CR-CL C9 / C10, RL4).
CREATE TABLE IF NOT EXISTS forward_log_outcomes (
    id                  bigserial PRIMARY KEY,
    event_id            bigint NOT NULL UNIQUE REFERENCES forward_log_events (id),
    race_10             text NOT NULL CHECK (race_10 IN ('win', 'loss', 'tie', 'unresolved', 'gap')),
    race_20             text NOT NULL CHECK (race_20 IN ('win', 'loss', 'tie', 'unresolved', 'gap')),
    move_30             double precision,                   -- h-signed pts at confirm + 30 bars; NULL past 12:59
    move_60             double precision,
    mfe_before_m10      double precision NOT NULL,          -- largest favourable excursion before -10 (or 13:00)
    code_sha            text NOT NULL,
    run_id              uuid,
    written_at          timestamptz NOT NULL DEFAULT now()
);

CREATE OR REPLACE FUNCTION forward_log_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION '% is insert-only (B4 forward log)', TG_TABLE_NAME;
END $$;

DROP TRIGGER IF EXISTS forward_log_sessions_guard ON forward_log_sessions;
CREATE TRIGGER forward_log_sessions_guard BEFORE UPDATE OR DELETE ON forward_log_sessions
    FOR EACH ROW EXECUTE FUNCTION forward_log_guard();
DROP TRIGGER IF EXISTS forward_log_events_guard ON forward_log_events;
CREATE TRIGGER forward_log_events_guard BEFORE UPDATE OR DELETE ON forward_log_events
    FOR EACH ROW EXECUTE FUNCTION forward_log_guard();
DROP TRIGGER IF EXISTS forward_log_outcomes_guard ON forward_log_outcomes;
CREATE TRIGGER forward_log_outcomes_guard BEFORE UPDATE OR DELETE ON forward_log_outcomes
    FOR EACH ROW EXECUTE FUNCTION forward_log_guard();

REVOKE ALL ON forward_log_sessions, forward_log_events, forward_log_outcomes FROM PUBLIC;
GRANT SELECT, INSERT ON forward_log_sessions, forward_log_events, forward_log_outcomes TO dash_backfill_writer;
GRANT USAGE ON SEQUENCE forward_log_sessions_id_seq, forward_log_events_id_seq, forward_log_outcomes_id_seq
    TO dash_backfill_writer;

COMMIT;
