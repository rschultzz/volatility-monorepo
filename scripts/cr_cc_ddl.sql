-- CR-CC DDL (L2, A1.4, A1.5). Apply once with DATABASE_URL, as the table owner.
-- Insert-only tables for the setup review queue; the only permitted UPDATE is setting
-- setup_reviews.revealed_at once. The web role may own these tables, so grants cannot
-- enforce insert-only: the triggers below do. (TRUNCATE bypasses row triggers; only the
-- owner can truncate, and dash_backfill_writer has no TRUNCATE.)
--
-- Grants (A1.5): SELECT, INSERT on both tables and UPDATE (revealed_at) on setup_reviews
-- to the role that owns the existing app tables (owner of trade_log_trades, from pg_tables),
-- plus SELECT, INSERT on setup_candidates to dash_backfill_writer for the loader. Explicit
-- roles only; nothing is granted to PUBLIC.

BEGIN;

CREATE TABLE IF NOT EXISTS setup_candidates (
    id              text PRIMARY KEY,                       -- '<cr_id>:<set_name>:<n>'
    cr_id           text NOT NULL,
    set_name        text NOT NULL,
    session_date    date NOT NULL,
    cell            text,
    direction       text NOT NULL CHECK (direction IN ('long', 'short')),
    level_price     double precision,
    level_net_bn    double precision,
    level_gross_bn  double precision,
    entry_ts_pt     timestamp without time zone NOT NULL,   -- naive PT wall-clock; close of the signal bar (A1.2)
    entry_price     double precision,
    stop_price      double precision,
    extreme_price   double precision,
    test_ts_pt      timestamp without time zone,
    confirm_ts_pt   timestamp without time zone,
    retest_ts_pt    timestamp without time zone,
    meta            jsonb NOT NULL DEFAULT '{}'::jsonb,
    outcome         jsonb,                                  -- never sent to the frontend before a review exists (L3)
    loaded_at       timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS setup_candidates_cr_set_idx
    ON setup_candidates (cr_id, set_name, session_date, entry_ts_pt);

CREATE TABLE IF NOT EXISTS setup_reviews (
    id            bigserial PRIMARY KEY,
    candidate_id  text NOT NULL REFERENCES setup_candidates (id),
    label         text NOT NULL CHECK (label IN ('take', 'skip', 'unsure')),
    reasons       text[] NOT NULL DEFAULT '{}',
    comment       text,
    reviewed_at   timestamptz NOT NULL DEFAULT now(),
    revealed_at   timestamptz,
    post_reveal   boolean NOT NULL DEFAULT false            -- written after the candidate was revealed (L8)
);
CREATE INDEX IF NOT EXISTS setup_reviews_candidate_idx
    ON setup_reviews (candidate_id, reviewed_at);

CREATE OR REPLACE FUNCTION setup_review_guard() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION '% is insert-only', TG_TABLE_NAME;
    END IF;
    -- Nested IF: plpgsql does not short-circuit AND, and setup_candidates rows have no
    -- revealed_at field.
    IF TG_TABLE_NAME = 'setup_reviews' THEN
        IF OLD.revealed_at IS NULL AND NEW.revealed_at IS NOT NULL
           AND (NEW.id, NEW.candidate_id, NEW.label, NEW.reasons, NEW.comment,
                NEW.reviewed_at, NEW.post_reveal)
               IS NOT DISTINCT FROM
               (OLD.id, OLD.candidate_id, OLD.label, OLD.reasons, OLD.comment,
                OLD.reviewed_at, OLD.post_reveal) THEN
            RETURN NEW;
        END IF;
    END IF;
    RAISE EXCEPTION '% is insert-only (only setup_reviews.revealed_at may be set, once)', TG_TABLE_NAME;
END $$;

DROP TRIGGER IF EXISTS setup_candidates_guard ON setup_candidates;
CREATE TRIGGER setup_candidates_guard BEFORE UPDATE OR DELETE ON setup_candidates
    FOR EACH ROW EXECUTE FUNCTION setup_review_guard();
DROP TRIGGER IF EXISTS setup_reviews_guard ON setup_reviews;
CREATE TRIGGER setup_reviews_guard BEFORE UPDATE OR DELETE ON setup_reviews
    FOR EACH ROW EXECUTE FUNCTION setup_review_guard();

DO $$
DECLARE
    app_owner text;
BEGIN
    SELECT tableowner INTO app_owner
      FROM pg_tables WHERE schemaname = 'public' AND tablename = 'trade_log_trades';
    IF app_owner IS NULL THEN
        RAISE EXCEPTION 'trade_log_trades not found: cannot determine the app-table owner';
    END IF;
    RAISE NOTICE 'CR-CC grants to app-table owner %', app_owner;
    EXECUTE format('GRANT SELECT, INSERT ON setup_candidates TO %I', app_owner);
    EXECUTE format('GRANT SELECT, INSERT ON setup_reviews TO %I', app_owner);
    EXECUTE format('GRANT UPDATE (revealed_at) ON setup_reviews TO %I', app_owner);
    EXECUTE format('GRANT USAGE, SELECT ON SEQUENCE setup_reviews_id_seq TO %I', app_owner);
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'dash_backfill_writer') THEN
        GRANT SELECT, INSERT ON setup_candidates TO dash_backfill_writer;
    END IF;
END $$;

REVOKE ALL ON setup_candidates, setup_reviews FROM PUBLIC;

COMMIT;
