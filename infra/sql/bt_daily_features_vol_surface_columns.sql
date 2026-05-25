-- ============================================================
-- bt_daily_features — add vol surface feature columns (CR-024 / CR-D)
--
-- Five nullable double precision columns for vol surface state:
--   atm_iv_percentile    — EOD ATM IV vs trailing 60-session distribution (0–100)
--   skew_percentile      — (25P IV − 25C IV) vs trailing 60-session distribution (0–100)
--   term_structure_slope — (near-30-DTE ATM IV) − (near-90-DTE ATM IV), raw IV points
--   smile_convexity      — ((25P + 25C) / 2 − ATM IV) vs trailing 60-session distribution (0–100)
--   vol_risk_premium     — 20-session realized vol − current ATM IV, raw vol points
--
-- All columns default NULL; populated by CR-D backfill script.
-- No feature_version bump — this is a schema extension, not a vector change.
-- Idempotent: IF NOT EXISTS guards throughout.
-- Apply after bt_daily_features_safety_columns.sql (dash_backfill_writer must exist).
-- ============================================================

BEGIN;

ALTER TABLE bt_daily_features
  ADD COLUMN IF NOT EXISTS atm_iv_percentile    DOUBLE PRECISION,
  ADD COLUMN IF NOT EXISTS skew_percentile       DOUBLE PRECISION,
  ADD COLUMN IF NOT EXISTS term_structure_slope  DOUBLE PRECISION,
  ADD COLUMN IF NOT EXISTS smile_convexity       DOUBLE PRECISION,
  ADD COLUMN IF NOT EXISTS vol_risk_premium      DOUBLE PRECISION;

-- Column-scoped UPDATE grant only. Table-level SELECT and INSERT were already
-- granted in CR-0 (dash_backfill_writer_role.sql); column-scoped versions
-- would be redundant. Structural columns (feature_vector, feature_config_hash,
-- etc.) remain unwriteable post-INSERT by this role.
GRANT UPDATE (
  atm_iv_percentile,
  skew_percentile,
  term_structure_slope,
  smile_convexity,
  vol_risk_premium
) ON bt_daily_features TO dash_backfill_writer;

COMMIT;
