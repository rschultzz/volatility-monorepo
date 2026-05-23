-- ============================================================
-- bt_audit_flags — per-day and per-pair audit flags (CR-016)
--
-- Captures user-flagged regime misclassifications and false-
-- analogue pairs for downstream classifier / KNN recalibration.
--
-- Two flag types:
--   regime_wrong          — one row per (ticker, trade_date)
--   not_a_true_analogue   — one row per (ticker, trade_date, analogue_date)
--
-- When promoted=TRUE on a regime_wrong flag, the corrected_regime
-- becomes the effective regime for display and template selection
-- (proposals, displayed labels). The stored feature_vector in
-- bt_daily_features is NOT recomputed — overrides are an escape
-- valve, not a batch recalibration mechanism.
--
-- Idempotent: IF NOT EXISTS throughout; safe to re-run.
-- ============================================================

BEGIN;

CREATE TABLE IF NOT EXISTS bt_audit_flags (
    flag_id          BIGSERIAL    PRIMARY KEY,
    flag_type        TEXT         NOT NULL
        CHECK (flag_type IN ('regime_wrong', 'not_a_true_analogue')),
    ticker           TEXT         NOT NULL,
    trade_date       DATE         NOT NULL,
    analogue_date    DATE,
    auto_regime      TEXT,
    corrected_regime TEXT,
    promoted         BOOLEAN      NOT NULL DEFAULT FALSE,
    note             TEXT,
    created_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    CONSTRAINT pair_flag_needs_analogue_date
        CHECK (flag_type <> 'not_a_true_analogue' OR analogue_date IS NOT NULL),
    CONSTRAINT regime_flag_needs_corrected_regime
        CHECK (flag_type <> 'regime_wrong' OR corrected_regime IS NOT NULL)
);

-- One regime_wrong flag per (ticker, trade_date).
CREATE UNIQUE INDEX IF NOT EXISTS bt_audit_flags_regime_unique
    ON bt_audit_flags (ticker, trade_date)
    WHERE flag_type = 'regime_wrong';

-- One pair flag per (ticker, anchor_date, analogue_date) — directional.
CREATE UNIQUE INDEX IF NOT EXISTS bt_audit_flags_pair_unique
    ON bt_audit_flags (ticker, trade_date, analogue_date)
    WHERE flag_type = 'not_a_true_analogue';

-- Lookup index: all flags for a given (ticker, trade_date).
CREATE INDEX IF NOT EXISTS bt_audit_flags_date_idx
    ON bt_audit_flags (ticker, trade_date);

-- Lookup index: pair flags where the flagged day is the analogue.
CREATE INDEX IF NOT EXISTS bt_audit_flags_analogue_date_idx
    ON bt_audit_flags (ticker, analogue_date)
    WHERE flag_type = 'not_a_true_analogue';

COMMIT;
