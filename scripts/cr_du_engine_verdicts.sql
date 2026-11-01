-- CR-DU item 5 (kickoff, Ryan 2026-10-31): the v1 engine's blind verdicts on review cards. Additive only.
-- Written by scripts/cr_du_set2_verdicts.py under the backfill protocol; read by no UI (counts only in reports).
BEGIN;
CREATE TABLE IF NOT EXISTS setup_engine_verdicts (
    candidate_id   text        NOT NULL REFERENCES setup_candidates (id),
    engine_version text        NOT NULL,
    verdict        text        NOT NULL CHECK (verdict IN ('SHORT', 'LONG', 'NO ACTION')),
    failed_rules   jsonb       NOT NULL DEFAULT '[]'::jsonb,
    evidence       jsonb,
    computed_at    timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (candidate_id, engine_version)
);
GRANT SELECT, INSERT ON setup_engine_verdicts TO dash_backfill_writer;
COMMIT;
