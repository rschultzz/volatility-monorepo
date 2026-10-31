-- CR-DR migration (approved by Ryan, chat 2026-10-31). Additive only.
BEGIN;
CREATE TABLE IF NOT EXISTS trade_log_orders (
    order_id        text PRIMARY KEY,
    symbol          text NOT NULL,
    side            text NOT NULL,
    order_type      text,
    qty             integer,
    limit_price     numeric,
    stop_price      numeric,
    status          text NOT NULL,
    placing_time_pt timestamptz,
    status_time_pt  timestamptz,
    duration        text,
    detected_tz     text,
    raw_csv_row     jsonb,
    imported_at     timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_tlo_symbol       ON trade_log_orders (symbol);
CREATE INDEX IF NOT EXISTS idx_tlo_placing_time ON trade_log_orders (placing_time_pt);

ALTER TABLE trade_log_fills  ADD COLUMN IF NOT EXISTS detected_tz text;

ALTER TABLE trade_log_trades
    ADD COLUMN IF NOT EXISTS setup_chips        text[],
    ADD COLUMN IF NOT EXISTS pre_trade_reason   text,
    ADD COLUMN IF NOT EXISTS mistake_chips      text[],
    ADD COLUMN IF NOT EXISTS checklist_followed boolean,
    ADD COLUMN IF NOT EXISTS lesson             text,
    ADD COLUMN IF NOT EXISTS grade              smallint,
    ADD COLUMN IF NOT EXISTS anchor_ts_pt       timestamptz;
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'trade_log_trades_grade_1_5') THEN
        ALTER TABLE trade_log_trades ADD CONSTRAINT trade_log_trades_grade_1_5 CHECK (grade BETWEEN 1 AND 5);
    END IF;
END $$;
COMMIT;
