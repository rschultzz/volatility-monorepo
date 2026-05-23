-- ============================================================
-- bt_daily_features — per-day feature vector cache (CR-013)
--
-- Stores the v0.5 signal-layer feature vector for each (ticker,
-- trade_date), the input to the Day Analogue Comparison KNN endpoint.
-- Written by compute_and_upsert_daily_features (packages/shared/
-- day_features.py), called from the EOD cron after the landscape
-- upsert and by scripts/backfill_daily_features.py for historical
-- dates.
--
-- This repo has no migration tool; apply this file manually with psql
-- before the cron's first daily-features run. Idempotent: IF NOT EXISTS
-- throughout, so re-running is safe.
-- ============================================================

BEGIN;

CREATE TABLE IF NOT EXISTS bt_daily_features (
    trade_date          DATE        NOT NULL,
    ticker              TEXT        NOT NULL DEFAULT 'SPX',
    feature_vector      JSONB       NOT NULL,
    feature_version     TEXT        NOT NULL,
    feature_config_hash TEXT        NOT NULL,
    computed_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (ticker, trade_date, feature_version)
);

CREATE INDEX IF NOT EXISTS bt_daily_features_version_idx
    ON bt_daily_features (feature_version);

CREATE INDEX IF NOT EXISTS bt_daily_features_date_idx
    ON bt_daily_features (trade_date DESC);

COMMIT;
