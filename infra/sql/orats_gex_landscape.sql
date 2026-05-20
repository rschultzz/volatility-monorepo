-- ============================================================
-- orats_gex_landscape — GEX landscape cache (CR-007)
--
-- Spot-agnostic nightly cache of the Gaussian-smoothed GEX field:
-- the landscape grid, the prominence walls, and the per-DTE-bucket
-- peaks. Written by compute_and_upsert_landscape (packages/shared/
-- gex_landscape.py), called from the EOD cron and the backfill script.
--
-- This repo has no migration tool; apply this file manually with psql
-- before the cron's first landscape run. Idempotent: IF NOT EXISTS
-- throughout, so re-running is safe.
-- ============================================================

BEGIN;

CREATE TABLE IF NOT EXISTS orats_gex_landscape (
    ticker          TEXT        NOT NULL,
    trade_date      DATE        NOT NULL,
    landscape       JSONB       NOT NULL,
    walls           JSONB       NOT NULL,
    peaks_by_bucket JSONB       NOT NULL,
    spread_coef     NUMERIC     NOT NULL,
    range_pts       NUMERIC     NOT NULL,
    step_pts        NUMERIC     NOT NULL,
    table_spot      NUMERIC,
    version         TEXT        NOT NULL,
    computed_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (ticker, trade_date)
);

CREATE INDEX IF NOT EXISTS orats_gex_landscape_date_idx
    ON orats_gex_landscape (trade_date DESC);

COMMIT;
