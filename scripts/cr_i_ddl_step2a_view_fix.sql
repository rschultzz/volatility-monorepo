-- CR-025 (CR-I) Step 2a view fix — refresh bt_daily_outcomes_active to include position_tN columns
-- Executed interactively 2026-05-26 (schema_change class — app role, not backfill role).
--
-- Root cause: The Step 2a DDL (cr_i_ddl_step2a.sql) added position_t1/t5/t15_post_touch
-- columns to bt_daily_outcomes but did not refresh the view.  PostgreSQL views snapshot
-- their column list at creation time — a SELECT * inside a view definition expands to the
-- column list that existed when the view was created, NOT the current base-table schema.
-- New columns added to the base table afterward are invisible to the view until it is
-- recreated with CREATE OR REPLACE.
--
-- Discovery: compute_structural_probability() in packages/shared/probability.py queries
-- bt_daily_outcomes_active by name for position_t1_post_touch etc., producing
-- "column does not exist" until the view was refreshed.  The backfill itself
-- (cr_i_backfill_post_touch_positions.py) ran against the base table directly —
-- hence the 360-row UPDATE succeeded while the view-based reads failed.
--
-- Safe to re-run: CREATE OR REPLACE is idempotent.

CREATE OR REPLACE VIEW bt_daily_outcomes_active AS
SELECT ticker,
       trade_date,
       feature_version,
       regime_kind_at_classification,
       dominant_bucket_at_classification,
       horizon_sessions,
       horizon_end_date,
       outcome_status,
       reached_touch,
       reached_close,
       days_to_reach,
       max_excursion_in_direction,
       final_close_distance_from_target,
       actual_realized_em_pct,
       active,
       deactivated_at,
       deactivated_reason,
       backfill_run_id,
       computed_at,
       position_t1_post_touch,
       position_t5_post_touch,
       position_t15_post_touch
FROM bt_daily_outcomes
WHERE active = true;

-- Post-fix verification:
-- SELECT column_name FROM information_schema.columns
-- WHERE table_name = 'bt_daily_outcomes_active'
-- ORDER BY ordinal_position;
-- Expect: 22 columns (19 original + 3 new position_tN columns at the end).
--
-- SELECT COUNT(*) FROM bt_daily_outcomes_active
-- WHERE position_t1_post_touch IS NOT NULL;
-- Expect: 360 (the Step 2b backfill result).
