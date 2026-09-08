-- CR-AR — width-normalised beat on bt_edge_backtest_results (decision 5)
-- Owner-role migration (schema_change class — NOT eligible for the backfill role).
-- Apply via scripts/cr_ar_run_migration.py under DATABASE_URL (table owner).
--
-- Adds 4 nullable columns (definition: specs/CR-AR-pair-snapping-width-cap.md):
--   close_pnl_per_point  FLOAT  Σ close_pnl_i / Σ width_actual_i over settled filled trades (points of P&L per point of width)
--   baseline_per_point   FLOAT  Σ baseline_close_pnl_i / Σ width_actual_i over baseline trades
--   beat_per_point       FLOAT  close_pnl_per_point − baseline_per_point (the width-weighted mean of beat_i / width_i;
--                               equals beat_baseline / 10 when every trade is 10-wide)
--   mean_width_actual    FLOAT  mean width_actual over the settled filled trades of the cell
-- dash_backfill_writer already holds table-level SELECT, INSERT (covers new columns; no column grants needed).
-- Rows persisted before CR-AR keep NULLs in the new columns.
--
-- Applied: see specs/CR-AR-pair-snapping-width-cap.md (Commit 5 log)

BEGIN;

ALTER TABLE bt_edge_backtest_results
  ADD COLUMN IF NOT EXISTS close_pnl_per_point FLOAT,
  ADD COLUMN IF NOT EXISTS baseline_per_point  FLOAT,
  ADD COLUMN IF NOT EXISTS beat_per_point      FLOAT,
  ADD COLUMN IF NOT EXISTS mean_width_actual   FLOAT;

COMMENT ON COLUMN bt_edge_backtest_results.close_pnl_per_point IS 'CR-AR: sum(close_pnl)/sum(width_actual) over settled filled trades';
COMMENT ON COLUMN bt_edge_backtest_results.baseline_per_point  IS 'CR-AR: sum(baseline_close_pnl)/sum(width_actual) over baseline trades';
COMMENT ON COLUMN bt_edge_backtest_results.beat_per_point      IS 'CR-AR: close_pnl_per_point - baseline_per_point (width-weighted beat per point of width)';
COMMENT ON COLUMN bt_edge_backtest_results.mean_width_actual   IS 'CR-AR: mean width_actual of the settled filled trades in the cell';

COMMIT;
