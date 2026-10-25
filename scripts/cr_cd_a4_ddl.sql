-- CR-CD A4.1 — 1-minute flow aggregates. Applied once with DATABASE_URL (A4.2).
-- Grants: dash_backfill_writer SELECT, INSERT (backfill; no UPDATE/DELETE). Every other role gets
-- the same privileges it holds on the matching 1-second table (copied from the catalog below), so the
-- Ironbeam worker can upsert ironbeam_es_flow_1m with whatever role it already writes
-- ironbeam_es_flow_1s with. Never PUBLIC.
BEGIN;

CREATE TABLE es_flow_1m_databento (
    minute_utc      timestamptz NOT NULL,          -- minute start
    symbol          text        NOT NULL,          -- XCME:ES.H26
    raw_symbol      text,                          -- ESH6
    volume          bigint      NOT NULL,
    buy_vol         bigint      NOT NULL,
    sell_vol        bigint      NOT NULL,
    unknown_vol     bigint      NOT NULL,
    trade_count     bigint      NOT NULL,
    open            double precision,
    high            double precision,
    low             double precision,
    close           double precision,
    first_trade_ms  bigint,
    last_trade_ms   bigint,
    n_seconds       integer     NOT NULL,          -- 1-second rows in the minute
    source          text        NOT NULL DEFAULT 'databento',
    PRIMARY KEY (minute_utc, symbol)
);

CREATE TABLE ironbeam_es_flow_1m (
    minute_utc      timestamptz NOT NULL,
    symbol          text        NOT NULL,
    volume          bigint      NOT NULL,
    buy_vol         bigint      NOT NULL,
    sell_vol        bigint      NOT NULL,
    unknown_vol     bigint      NOT NULL,
    trade_count     bigint      NOT NULL,
    open            double precision,
    high            double precision,
    low             double precision,
    close           double precision,
    first_trade_ms  bigint,
    last_trade_ms   bigint,
    n_seconds       integer     NOT NULL,
    source          text        NOT NULL DEFAULT 'ironbeam',
    PRIMARY KEY (minute_utc, symbol)
);
CREATE INDEX idx_ironbeam_es_flow_1m_symbol_minute ON ironbeam_es_flow_1m (symbol, minute_utc);

REVOKE ALL ON es_flow_1m_databento, ironbeam_es_flow_1m FROM PUBLIC;
GRANT SELECT, INSERT ON es_flow_1m_databento, ironbeam_es_flow_1m TO dash_backfill_writer;

DO $$
DECLARE g record;
BEGIN
    FOR g IN
        SELECT DISTINCT grantee, privilege_type, table_name
        FROM information_schema.role_table_grants
        WHERE table_name IN ('es_flow_1s_databento', 'ironbeam_es_flow_1s')
          AND grantee NOT IN ('PUBLIC', 'dash_backfill_writer')
          AND grantee <> (SELECT tableowner FROM pg_tables WHERE tablename = table_name)
          AND privilege_type IN ('SELECT', 'INSERT', 'UPDATE')
    LOOP
        EXECUTE format('GRANT %s ON %I TO %I', g.privilege_type,
                       CASE g.table_name WHEN 'es_flow_1s_databento' THEN 'es_flow_1m_databento'
                                         ELSE 'ironbeam_es_flow_1m' END,
                       g.grantee);
    END LOOP;
END $$;

COMMIT;
