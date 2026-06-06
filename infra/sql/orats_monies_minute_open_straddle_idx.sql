-- CR-037: covering index for the open-straddle implied-move query.
--
-- _OPEN_STRADDLE_SQL (packages/shared/day_features.py) selects the first
-- orats_monies_minute row at or after 06:33 PT for a given (trade_date, ticker):
--
--   WHERE trade_date  = <ISO-text>
--     AND ticker      = <ticker>
--     AND atmiv       IS NOT NULL
--     AND dte         > 0
--     AND snapshot_pt >= <floor_timestamp>    -- dt.datetime.combine(date, time(6,33,0))
--   ORDER BY snapshot_pt ASC, dte ASC
--   LIMIT 1
--
-- The existing idx_omm_trade_date_ticker_dte_snapshot has snapshot_pt DESC and
-- can't cover the ASC range scan from a floor timestamp.  This new index:
--   (trade_date, ticker, snapshot_pt ASC, dte ASC) partial WHERE dte>0 AND atmiv IS NOT NULL
-- lets PostgreSQL seek to (trade_date, ticker, snapshot_pt >= floor) and return
-- the first qualifying row without a sort — O(log N) instead of a full scan.
--
-- CONCURRENTLY is mandatory: orats_monies_minute has 16M+ rows.  A plain CREATE
-- INDEX takes an exclusive lock for several minutes, blocking the intraday ingest
-- crons.  CONCURRENTLY builds without an exclusive lock (two full-table scans).
-- CONCURRENTLY is ILLEGAL inside an explicit transaction — do NOT wrap in BEGIN/COMMIT.
--
-- Apply with:
--   psql $DATABASE_URL -f infra/sql/orats_monies_minute_open_straddle_idx.sql
--
-- Verify after applying:
--   \d orats_monies_minute   -- must NOT show "(INVALID)" next to the new index
--   SELECT indexname, idx_scan FROM pg_stat_user_indexes
--     WHERE tablename = 'orats_monies_minute';
--
-- If the index shows (INVALID) (connection dropped mid-build), drop and re-run:
--   DROP INDEX CONCURRENTLY IF EXISTS idx_omm_open_straddle;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_omm_open_straddle
    ON orats_monies_minute (trade_date, ticker, snapshot_pt ASC, dte ASC)
    WHERE dte > 0 AND atmiv IS NOT NULL;
