-- ============================================================
-- bt_vol_state_daily + bt_range_labels_daily — CR-BL
--
-- Spec: specs/CR-BL-vol-state-range-labels.md (L1–L14, Amendment A1).
--
-- bt_vol_state_daily: one row per (ticker, trade_date, version). Every field is
-- observable before the 06:30 PT open of trade_date (L3): implied vol from the
-- prior session's latest orats_monies_minute snapshot <= 12:46 PT, realized vol
-- through the prior close (SPX cash, CR-BH hybrid series). Vol fields are in
-- annualized vol points (12.9 = 12.9 %). Percentiles 0–100 vs up to 250 prior
-- sessions, t excluded, minimum 120 (A1.4).
--
-- bt_range_labels_daily: one row per (ticker, trade_date, horizon_sessions,
-- version). SPX cash RTH (06:33–13:00 PT prints), horizon = sessions t..t+h-1.
-- implied_move_pts: h=1 the stored implied_move_1d (A1.3: NULL where none is
-- stored); h=5 SPX open × sqrt(total implied variance from the t−1 12:46 PT
-- snapshot to the horizon-end date) (A1.1).
--
-- Written INSERT-only by scripts/cr_bl_backfill.py under dash_backfill_writer.
-- No migration tool in this repo: apply manually with psql (owner role).
-- Idempotent: IF NOT EXISTS throughout.
-- ============================================================

BEGIN;

CREATE TABLE IF NOT EXISTS bt_vol_state_daily (
    ticker              TEXT        NOT NULL,
    trade_date          DATE        NOT NULL,
    version             TEXT        NOT NULL,
    iv_snapshot_date    DATE,
    iv_snapshot_pt      TIMESTAMP,              -- naive PT, as orats_monies_minute.snapshot_pt
    iv_7d               DOUBLE PRECISION,
    iv_30d              DOUBLE PRECISION,
    iv_7d_lo_expiry     DATE,
    iv_7d_hi_expiry     DATE,
    iv_30d_lo_expiry    DATE,
    iv_30d_hi_expiry    DATE,
    sessions_in_7d      SMALLINT,               -- NYSE sessions in the 7 calendar days after the snapshot date
    rv_last_close_date  DATE,
    rv_5_cc             DOUBLE PRECISION,
    rv_21_cc            DOUBLE PRECISION,
    rv_5_pk             DOUBLE PRECISION,
    rv_21_pk            DOUBLE PRECISION,
    vrp_7               DOUBLE PRECISION,       -- iv_7d − rv_5_cc
    vrp_30              DOUBLE PRECISION,       -- iv_30d − rv_21_cc
    iv_7d_pct           DOUBLE PRECISION,
    iv_30d_pct          DOUBLE PRECISION,
    rv_21_cc_pct        DOUBLE PRECISION,
    vrp_30_pct          DOUBLE PRECISION,
    vrp_7_pct           DOUBLE PRECISION,
    backfill_run_id     UUID,
    computed_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (ticker, trade_date, version)
);

CREATE TABLE IF NOT EXISTS bt_range_labels_daily (
    ticker                  TEXT        NOT NULL,
    trade_date              DATE        NOT NULL,
    horizon_sessions        SMALLINT    NOT NULL,
    version                 TEXT        NOT NULL,
    horizon_end_date        DATE        NOT NULL,
    spx_open                DOUBLE PRECISION NOT NULL,
    horizon_high            DOUBLE PRECISION NOT NULL,
    horizon_low             DOUBLE PRECISION NOT NULL,
    horizon_close           DOUBLE PRECISION NOT NULL,
    min_session_minutes     INTEGER     NOT NULL,   -- fewest SPX prints in any horizon session (outage flag)
    implied_move_basis      TEXT        NOT NULL,   -- 'implied_move_1d' | 'total_var_to_horizon_end'
    implied_vol_h           DOUBLE PRECISION,       -- h=5: interpolated vol points at implied_maturity_days
    implied_maturity_days   INTEGER,                -- h=5: calendar days, snapshot date → horizon_end_date
    implied_move_pts        DOUBLE PRECISION,
    range_over_implied      DOUBLE PRECISION,
    abs_close_over_implied  DOUBLE PRECISION,
    backfill_run_id         UUID,
    computed_at             TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (ticker, trade_date, horizon_sessions, version)
);

GRANT SELECT, INSERT ON bt_vol_state_daily, bt_range_labels_daily TO dash_backfill_writer;

COMMIT;
