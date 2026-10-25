-- CR-CA es_flow_1s_databento DDL (A1.5 / A2.3): run once with DATABASE_URL after Step 0 passed.
CREATE TABLE es_flow_1s_databento (
  ts_utc         timestamptz NOT NULL,
  symbol         text        NOT NULL,        -- Ironbeam format, e.g. XCME:ES.H26 (A1.1)
  raw_symbol     text        NOT NULL,        -- CME/Databento, e.g. ESH6 (A1.1)
  trade_date     date        NULL,            -- UTC date of ts_utc (A1.2)
  volume         integer     NOT NULL DEFAULT 0,
  notional       double precision NOT NULL DEFAULT 0,
  trade_count    integer     NOT NULL DEFAULT 0,
  buy_vol        integer     NOT NULL DEFAULT 0,
  sell_vol       integer     NOT NULL DEFAULT 0,
  unknown_vol    integer     NOT NULL DEFAULT 0,
  buy_count      integer     NOT NULL DEFAULT 0,
  sell_count     integer     NOT NULL DEFAULT 0,
  unknown_count  integer     NOT NULL DEFAULT 0,
  open           double precision NULL,
  high           double precision NULL,
  low            double precision NULL,
  close          double precision NULL,
  first_trade_ms bigint NULL,
  last_trade_ms  bigint NULL,
  source         text        NOT NULL DEFAULT 'databento',
  PRIMARY KEY (ts_utc, symbol)
);
CREATE INDEX idx_es_flow_1s_databento_trade_date ON es_flow_1s_databento (trade_date);
GRANT SELECT, INSERT ON es_flow_1s_databento TO dash_backfill_writer;
