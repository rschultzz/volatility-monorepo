-- CR-025 (CR-I) Step 2a schema migration
-- Run interactively (schema_change class — NOT eligible for unattended execution).
-- Execute in a transaction; verify before COMMIT.
--
-- Pre-run verification:
--   SELECT current_user;
--   SELECT COUNT(*) FROM bt_daily_outcomes;   -- record baseline
--   SELECT COUNT(*) FROM bt2_scan_cache;       -- record baseline
--
-- Two tables modified:
--   1. bt_daily_outcomes: 3 SMALLINT columns for per-analogue post-touch positions
--   2. bt2_scan_cache:    13 count/metadata columns + backfill_run_id
--
-- Applied 2026-05-26.

BEGIN;

-- ── bt_daily_outcomes ─────────────────────────────────────────────────────────
-- For each anchor that reached_touch=TRUE, classify where the ES close was at
-- T+1, T+5, T+15 sessions after touch relative to the drift_target tolerance band:
--   -1 = below drift_target - tolerance
--    0 = within tolerance (|close - drift_target| <= tolerance)
--   +1 = above drift_target + tolerance
-- NULL = bar not available (analogue too recent for T+15 to exist in bars data).
-- Populated by CR-I Step 2b backfill (scripts/cr_i_backfill_post_touch_positions.py).
-- Denominators may differ across timeframes (denominator_t15 <= denominator_t5 <= denominator_t1).

ALTER TABLE bt_daily_outcomes
    ADD COLUMN position_t1_post_touch  SMALLINT,   -- -1 / 0 / +1 / NULL
    ADD COLUMN position_t5_post_touch  SMALLINT,
    ADD COLUMN position_t15_post_touch SMALLINT;


-- ── bt2_scan_cache ────────────────────────────────────────────────────────────
-- Counts-based design (revised from spec's 31 enumerated columns).
-- Fractions and Wilson CIs recomputed at read time from stored counts — closed-form,
-- microsecond cost, no information loss. Approved pre-DDL, 2026-05-26.
--
-- count_* columns store raw counts per position per timeframe.
-- Denominator for timeframe N = count_below_tN + count_at_tN + count_above_tN.
-- Sum <= post_touch_total_touchers (NULL-handling may reduce effective denominator).
-- Populated by CR-I Step 2c backfill (scripts/cr_i_backfill_scan_cache_aggregates.py).

ALTER TABLE bt2_scan_cache
    ADD COLUMN count_below_t1             INTEGER,
    ADD COLUMN count_at_t1                INTEGER,
    ADD COLUMN count_above_t1             INTEGER,
    ADD COLUMN count_below_t5             INTEGER,
    ADD COLUMN count_at_t5                INTEGER,
    ADD COLUMN count_above_t5             INTEGER,
    ADD COLUMN count_below_t15            INTEGER,
    ADD COLUMN count_at_t15               INTEGER,
    ADD COLUMN count_above_t15            INTEGER,
    ADD COLUMN post_touch_pattern_label   VARCHAR(32),   -- stepping-stone | touch-and-pin | ...
    ADD COLUMN post_touch_filter_mode     VARCHAR(32),   -- strict | pooled-fallback | insufficient | zero_dte_corpus_insufficient
    ADD COLUMN post_touch_same_bucket_n   INTEGER,       -- same-bucket touchers used as denominator (strict mode)
    ADD COLUMN post_touch_total_touchers  INTEGER,       -- all computed touchers in K=20
    ADD COLUMN backfill_run_id            UUID;          -- data safety protocol: links row to bt_backfill_runs

COMMIT;

-- ── Post-run verification ─────────────────────────────────────────────────────
-- Run AFTER commit to confirm schema:

-- bt_daily_outcomes — expect 3 rows, smallint, YES nullable:
-- SELECT column_name, data_type, is_nullable
-- FROM information_schema.columns
-- WHERE table_name = 'bt_daily_outcomes'
--   AND column_name LIKE 'position_t%_post_touch'
-- ORDER BY column_name;

-- bt2_scan_cache — expect 14 new rows:
-- SELECT column_name, data_type, is_nullable
-- FROM information_schema.columns
-- WHERE table_name = 'bt2_scan_cache'
--   AND (column_name LIKE 'count_%' OR column_name LIKE 'post_touch_%' OR column_name = 'backfill_run_id')
-- ORDER BY ordinal_position;

-- All new columns should be NULL (no data yet):
-- SELECT COUNT(*) FROM bt_daily_outcomes WHERE position_t1_post_touch IS NOT NULL;  -- expect 0
-- SELECT COUNT(*) FROM bt2_scan_cache WHERE count_below_t1 IS NOT NULL;             -- expect 0
