-- CR-025 (CR-I) Step 2c revert — drop bt2_scan_cache aggregate columns
-- Executed interactively 2026-05-26 (schema_change class — app role, not backfill role).
--
-- Context: the 14 columns added in step_2a for bt2_scan_cache were designed on the
-- assumption that each cache row is a per-anchor result (one trade_date, one ticker).
-- Actual schema: each row is a per-scan-run result containing hundreds of signal days
-- inside a JSONB `rows` blob.  A single pattern_label / 9 counts per scan row makes
-- no semantic sense — a 509-signal-day scan has no single pattern.
--
-- Resolution: drop the bt2_scan_cache additions; defer saved-scans UI integration
-- to a future CR.  Per-signal-day post-touch data already lives in bt_daily_outcomes
-- from Step 2b and is reachable via JOIN on render.
--
-- bt_daily_outcomes position_tN_post_touch columns from step_2a STAY — correct and
-- fully populated by Step 2b.
--
-- All 14 columns had count_below_t1 = 0 non-null rows (verified at 2a baseline);
-- no data was lost.  Applied 2026-05-26.

ALTER TABLE bt2_scan_cache DROP COLUMN count_below_t1;
ALTER TABLE bt2_scan_cache DROP COLUMN count_at_t1;
ALTER TABLE bt2_scan_cache DROP COLUMN count_above_t1;
ALTER TABLE bt2_scan_cache DROP COLUMN count_below_t5;
ALTER TABLE bt2_scan_cache DROP COLUMN count_at_t5;
ALTER TABLE bt2_scan_cache DROP COLUMN count_above_t5;
ALTER TABLE bt2_scan_cache DROP COLUMN count_below_t15;
ALTER TABLE bt2_scan_cache DROP COLUMN count_at_t15;
ALTER TABLE bt2_scan_cache DROP COLUMN count_above_t15;
ALTER TABLE bt2_scan_cache DROP COLUMN post_touch_pattern_label;
ALTER TABLE bt2_scan_cache DROP COLUMN post_touch_filter_mode;
ALTER TABLE bt2_scan_cache DROP COLUMN post_touch_same_bucket_n;
ALTER TABLE bt2_scan_cache DROP COLUMN post_touch_total_touchers;
ALTER TABLE bt2_scan_cache DROP COLUMN backfill_run_id;

-- Post-revert verification:
-- SELECT COUNT(*) FROM information_schema.columns WHERE table_name = 'bt2_scan_cache';
-- Expect: 14 (original pre-2a baseline).
-- SELECT column_name FROM information_schema.columns WHERE table_name = 'bt2_scan_cache'
--   ORDER BY ordinal_position;
-- Expect: scan_id, created_at, label, direction, start_date, end_date, params,
--         funnel, diagnostics, rows, row_count, notes, column_prefs, filter_presets
