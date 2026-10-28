-- CR-DD: exact aggressor volume at price, per minute. Owner applies once:
--   psql "$DATABASE_URL" -f scripts/cr_dd_ddl.sql
-- Insert-only by role: dash_backfill_writer gets SELECT, INSERT (no UPDATE / DELETE).
CREATE TABLE es_flow_price_1m (
  minute_utc  timestamptz      NOT NULL,  -- minute start, UTC
  symbol      text             NOT NULL,  -- Ironbeam format, e.g. XCME:ES.Z26
  price       double precision NOT NULL,  -- floor to the 0.25 grid
  buy_vol     integer          NOT NULL,  -- buy-aggressor contracts
  sell_vol    integer          NOT NULL,  -- sell-aggressor contracts
  trade_count integer          NOT NULL,  -- all prints, incl. unknown side (DD3)
  source      text             NOT NULL CHECK (source IN ('ironbeam', 'databento')),
  PRIMARY KEY (minute_utc, symbol, price, source)
);
GRANT SELECT, INSERT ON es_flow_price_1m TO dash_backfill_writer;
