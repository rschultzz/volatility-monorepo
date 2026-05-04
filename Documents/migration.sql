-- ============================================================
-- Trade Log migration
-- Run once against your PostgreSQL database.
-- Idempotent: uses IF NOT EXISTS throughout.
-- ============================================================

BEGIN;

-- ── Fills ────────────────────────────────────────────────────
-- Immutable raw fills ingested from TradingView CSV exports.
-- Deduplication key: order_id (TV's Order ID column).
-- Times stored as TIMESTAMPTZ (UTC internally); always display/
-- input as America/Los_Angeles (PT).

CREATE TABLE IF NOT EXISTS trade_log_fills (
    order_id          TEXT         PRIMARY KEY,
    symbol            TEXT         NOT NULL,
    side              TEXT         NOT NULL CHECK (side IN ('B', 'S')),
    order_type        TEXT,
    qty               INTEGER      NOT NULL DEFAULT 0,
    fill_qty          INTEGER      NOT NULL DEFAULT 0,
    fill_price        NUMERIC(12, 4) NOT NULL,
    commission        NUMERIC(10, 4) NOT NULL DEFAULT 0,
    duration          TEXT,
    placing_time_pt   TIMESTAMPTZ,
    status_time_pt    TIMESTAMPTZ,
    raw_csv_row       JSONB,
    imported_at       TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tlf_symbol       ON trade_log_fills (symbol);
CREATE INDEX IF NOT EXISTS idx_tlf_placing_time ON trade_log_fills (placing_time_pt);


-- ── Trades ───────────────────────────────────────────────────
-- Derived round-trip trades (FIFO-paired from fills).
-- Annotations and market context stored here.
-- (entry_order_id, exit_order_id) is the natural dedup key so
-- re-uploads don't duplicate already-paired trades.

CREATE TABLE IF NOT EXISTS trade_log_trades (
    id                          SERIAL        PRIMARY KEY,
    trade_date                  DATE          NOT NULL,
    symbol                      TEXT          NOT NULL,
    direction                   TEXT          NOT NULL CHECK (direction IN ('long', 'short')),
    qty                         INTEGER       NOT NULL DEFAULT 1,

    -- Fill references
    entry_order_id              TEXT          REFERENCES trade_log_fills(order_id) ON DELETE SET NULL,
    entry_ts_pt                 TIMESTAMPTZ,
    entry_price                 NUMERIC(12, 4),
    entry_order_type            TEXT,

    exit_order_id               TEXT          REFERENCES trade_log_fills(order_id) ON DELETE SET NULL,
    exit_ts_pt                  TIMESTAMPTZ,
    exit_price                  NUMERIC(12, 4),
    exit_order_type             TEXT,

    -- P&L (MES=$5, ES=$50, MNQ=$2, NQ=$20)
    realized_pts                NUMERIC(10, 4),
    realized_pnl_usd            NUMERIC(10, 2),
    fees_usd                    NUMERIC(10, 2) NOT NULL DEFAULT 0,
    net_pnl_usd                 NUMERIC(10, 2),

    -- Manual annotations (user-editable)
    setup_start_ts_pt           TIMESTAMPTZ,          -- initially = entry fill time
    setup_target_ts_pt          TIMESTAMPTZ,          -- initially = exit fill time
    setup_direction             TEXT,
    notes                       TEXT,

    -- Auto-computed market context (recomputed on demand)
    context_iv_atm_0dte_pct     NUMERIC(8, 4),
    context_target_spx_price    NUMERIC(10, 4),
    context_skew_delta_put_pct  NUMERIC(8, 4),
    context_skew_delta_call_pct NUMERIC(8, 4),
    context_skew_delta_atm_iv   NUMERIC(8, 4),
    context_minutes_to_close    NUMERIC(8, 2),
    context_computed_at         TIMESTAMPTZ,

    created_at                  TIMESTAMPTZ   NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ   NOT NULL DEFAULT NOW(),

    -- Prevent duplicate pairings on re-upload
    CONSTRAINT uq_trade_entry_exit UNIQUE (entry_order_id, exit_order_id)
);

CREATE INDEX IF NOT EXISTS idx_tlt_date      ON trade_log_trades (trade_date);
CREATE INDEX IF NOT EXISTS idx_tlt_direction ON trade_log_trades (direction);
CREATE INDEX IF NOT EXISTS idx_tlt_entry_ts  ON trade_log_trades (entry_ts_pt);

-- ── updated_at trigger ───────────────────────────────────────

CREATE OR REPLACE FUNCTION tl_set_updated_at()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS tg_tlt_updated_at ON trade_log_trades;
CREATE TRIGGER tg_tlt_updated_at
  BEFORE UPDATE ON trade_log_trades
  FOR EACH ROW EXECUTE FUNCTION tl_set_updated_at();

COMMIT;
