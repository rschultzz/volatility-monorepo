-- CR-AQ — session containment outcome columns on bt_daily_outcomes
-- Owner-role migration (schema_change class — NOT eligible for the backfill role).
-- Apply via scripts/cr_aq_run_migration.py under DATABASE_URL (table owner).
--
-- Adds 11 nullable columns (definition: specs/CR-AQ-session-containment-outcome.md):
--   session_high_t0 / session_low_t0 / session_close_t0   REAL   trade-date RTH high / low / close
--   wall_above_price / wall_below_price                    REAL   nearest landscape wall strictly above / below session_open_t0 (any sign)
--   contained_close / contained_range                      BOOL   close / whole range inside (wall_below, wall_above)
--   close_pos_in_band                                      REAL   (close − wall_below) / (wall_above − wall_below)
--   range_over_im / close_move_over_im                     REAL   (high − low) / implied_move_1d ; (close − open) / implied_move_1d
--   breach_side                                            VARCHAR(8)  'above' | 'below' | 'both' when contained_range = false
-- Column-level GRANT UPDATE to dash_backfill_writer (column grants do not auto-extend).
-- bt_daily_outcomes_active is recreated afterwards (views snapshot their column list).
--
-- Applied: 2026-09-06 (CR-AQ Step 1)

BEGIN;

ALTER TABLE bt_daily_outcomes
  ADD COLUMN session_high_t0    REAL,
  ADD COLUMN session_low_t0     REAL,
  ADD COLUMN session_close_t0   REAL,
  ADD COLUMN wall_above_price   REAL,
  ADD COLUMN wall_below_price   REAL,
  ADD COLUMN contained_close    BOOLEAN,
  ADD COLUMN contained_range    BOOLEAN,
  ADD COLUMN close_pos_in_band  REAL,
  ADD COLUMN range_over_im      REAL,
  ADD COLUMN close_move_over_im REAL,
  ADD COLUMN breach_side        VARCHAR(8);

GRANT UPDATE (
  session_high_t0, session_low_t0, session_close_t0,
  wall_above_price, wall_below_price,
  contained_close, contained_range, close_pos_in_band,
  range_over_im, close_move_over_im, breach_side
) ON bt_daily_outcomes TO dash_backfill_writer;

COMMIT;

-- ── View refresh (run after COMMIT; CREATE OR REPLACE may only append columns) ──
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
       position_t15_post_touch,
       session_open_t1,
       session_high_t1,
       session_low_t1,
       session_close_t1,
       session_open_t5,
       session_high_t5,
       session_low_t5,
       session_close_t5,
       session_open_t15,
       session_high_t15,
       session_low_t15,
       session_close_t15,
       session_open_t0,
       session_high_t0,
       session_low_t0,
       session_close_t0,
       wall_above_price,
       wall_below_price,
       contained_close,
       contained_range,
       close_pos_in_band,
       range_over_im,
       close_move_over_im,
       breach_side
FROM bt_daily_outcomes
WHERE active = true;

-- Post-run verification:
-- SELECT column_name FROM information_schema.columns WHERE table_name='bt_daily_outcomes' AND column_name IN ('contained_close','breach_side');
-- SELECT count(*) FROM information_schema.column_privileges WHERE table_name='bt_daily_outcomes' AND grantee='dash_backfill_writer' AND privilege_type='UPDATE' AND column_name IN (…11…);  -- 11
-- SELECT count(*) FROM bt_daily_outcomes WHERE session_close_t0 IS NOT NULL;  -- 0 before backfill
