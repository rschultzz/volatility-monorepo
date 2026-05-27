-- CR-G Step 2.5a — Add session_open_t0 to bt_daily_outcomes
--
-- session_open_t0: the analogue's own RTH open on trade_date (T+0 session).
-- This is the reference price for normalising each analogue's close to a
-- dimensionless return: (close_tN - session_open_t0) / implied_move_1d.
-- Without this anchor, compute_terminal_prob_in_range compares absolute close
-- prices across price epochs, which is semantically broken.
--
-- Source: ironbeam_es_1m_bars FIRST bar open for trade_date RTH session.
-- Matches compute_outcome's horizon.iloc[0]["open"] reference.
--
-- Apply as app role (new_db_cred / rschultz), then run
-- scripts/cr_g_step_2_5a_backfill_session_open_t0.py as dash_backfill_writer.
--
-- Idempotent: ADD COLUMN IF NOT EXISTS is safe to re-run.
-- ============================================================

-- 1. Add column
ALTER TABLE bt_daily_outcomes
  ADD COLUMN IF NOT EXISTS session_open_t0 REAL;

-- 2. Grant UPDATE to backfill writer (column-scoped, per data safety protocol)
GRANT UPDATE (session_open_t0)
  ON bt_daily_outcomes TO dash_backfill_writer;

-- 3. Refresh active view (SELECT * snapshots columns at view creation time)
CREATE OR REPLACE VIEW bt_daily_outcomes_active AS
  SELECT * FROM bt_daily_outcomes WHERE active = TRUE;

-- 4. Verify
SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_name = 'bt_daily_outcomes'
  AND column_name = 'session_open_t0';
