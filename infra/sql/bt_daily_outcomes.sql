-- ============================================================
-- bt_daily_outcomes — per-day structural outcome cache (CR-B)
--
-- For every historical day's structural read, records whether the
-- prediction played out within the dominant-bucket's time horizon.
-- Written by scripts/cr_b_backfill_outcomes.py (unattended backfill)
-- and, post-CR-B, by the EOD cron after the daily-features upsert.
--
-- outcome_status values:
--   'computed'        — all metrics populated
--   'pending_history' — horizon_end_date > latest bar date; metrics NULL
--   'na_regime'       — non-directional regime (v1); metrics NULL
--   'na_data'         — required input missing at compute time; metrics NULL
--
-- App role (new_db_cred inherits from rschultz, table owner) — no
-- explicit grant needed.
-- dash_backfill_writer SELECT + INSERT covered by ALTER DEFAULT
-- PRIVILEGES in dash_backfill_writer_role.sql; stated explicitly for
-- clarity. UPDATE scoped to safety columns only.
--
-- This repo has no migration tool; apply this file manually before
-- the first backfill run. Idempotent re-runs safe if table absent.
-- ============================================================

CREATE TABLE bt_daily_outcomes (
    ticker                              VARCHAR     NOT NULL,
    trade_date                          DATE        NOT NULL,
    feature_version                     VARCHAR     NOT NULL,

    regime_kind_at_classification       VARCHAR,
    dominant_bucket_at_classification   VARCHAR,
    horizon_sessions                    INT,
    horizon_end_date                    DATE,

    -- outcome_status: 'computed' | 'pending_history' | 'na_regime' | 'na_data'
    outcome_status                      VARCHAR     NOT NULL DEFAULT 'computed',

    reached_touch                       BOOLEAN,
    reached_close                       BOOLEAN,
    days_to_reach                       INT,
    max_excursion_in_direction          FLOAT,
    final_close_distance_from_target    FLOAT,
    actual_realized_em_pct              FLOAT,

    -- Safety columns (mirrors bt_daily_features CR-020 pattern)
    active                              BOOLEAN     NOT NULL DEFAULT TRUE,
    deactivated_at                      TIMESTAMP,
    deactivated_reason                  TEXT,
    backfill_run_id                     UUID,

    computed_at                         TIMESTAMP   DEFAULT NOW(),

    PRIMARY KEY (ticker, trade_date, feature_version)
);

CREATE INDEX idx_outcomes_regime
    ON bt_daily_outcomes (regime_kind_at_classification);

CREATE INDEX idx_outcomes_run_id
    ON bt_daily_outcomes (backfill_run_id);

-- SELECT + INSERT: covered by ALTER DEFAULT PRIVILEGES, stated for clarity
GRANT SELECT, INSERT ON bt_daily_outcomes TO dash_backfill_writer;

-- UPDATE scoped to safety columns only
GRANT UPDATE (active, deactivated_at, deactivated_reason, backfill_run_id)
    ON bt_daily_outcomes TO dash_backfill_writer;

-- Active view: ALTER DEFAULT PRIVILEGES does not cover views;
-- explicit grant required (same pattern as bt_daily_features_active_view.sql)
CREATE VIEW bt_daily_outcomes_active AS
    SELECT * FROM bt_daily_outcomes WHERE active = TRUE;

GRANT SELECT ON bt_daily_outcomes_active TO dash_backfill_writer;
