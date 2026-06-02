# CR-035 — `_fetch_carry_rates` Query Rewrite + `orats_monies_minute` Index

## Context

Driven by the Q2 diagnostic (session `2026-06-02 - CR-034`). The endpoint
`/api/setup/proposals` was measured at ~4.3s total wall-clock. The dominant
cost was a single LIMIT-1 SQL call in `_fetch_carry_rates`:

```sql
ORDER BY ABS(dte - %s) ASC, snapshot_pt DESC
LIMIT 1
```

`ABS(dte - target)` is a function expression — Postgres cannot use any index
for ordering. It materialised all ~21,098 rows for the queried (trade_date,
ticker) group, read 811 pages from disk (cold on historical dates), then ran a
full top-N heapsort. **EXPLAIN ANALYZE measured 3,501 ms** for a single call.

`orats_monies_minute` characteristics:
- 16,268,125 rows — all ticker='SPX'.
- Per trading day: 391 distinct snapshot timestamps × 54 distinct DTE values
  ≈ 21,098 rows per (trade_date, ticker) group.
- Existing indexes do not cover (trade_date, ticker) as a compound predicate;
  `ticker` is a post-index filter over the full day's rows.

## Scope

Two files:
1. `infra/sql/orats_monies_minute_carry_rate_idx.sql` — DDL migration (new
   index).
2. `apps/web/modules/TodaySetup/routes.py` — rewrite `_fetch_carry_rates`
   only. No other function, no other file.

**Visible-change discipline:** this CR has no frontend component. The
rendered output of the endpoint is identical; only latency changes.

## Step-0 Findings (2026-06-02)

Code verified against the spec's assumptions:

| Assumption | Actual code | Status |
|---|---|---|
| Signature `(conn, ticker, trade_date, dte_target=15)` | Lines 114–117 of routes.py | ✓ match |
| SELECT projects `risk_free_rate, yield_rate` (2 cols) | Line 126–130 | ✓ match |
| Unpack `row[0]` = rfr, `row[1]` = yield | Lines 137–138 | ✓ match |
| Fallback `(0.05, 0.0)` on no-row | Line 136 | ✓ match |
| Single call site, 2-tuple unpack | routes.py line 244 | ✓ match |
| No other callers | grep across monorepo | ✓ confirmed |

No contradictions. Proceed.

The rewrite adds `dte` as a 3rd column **inside** the UNION ALL sub-selects
(required for the outer `ORDER BY ABS(dte-target)` over ≤ 2 rows), but the
outer wrapper SELECT projects only `risk_free_rate, yield_rate`. The returned
`row` tuple stays length 2 — `row[0]`/`row[1]` unpack is unchanged.

## Index Migration (commit 2)

File: `infra/sql/orats_monies_minute_carry_rate_idx.sql`

```sql
-- CR-035: covering index for _fetch_carry_rates two-subselect rewrite.
--
-- CONCURRENTLY is intentional and mandatory:
--   - orats_monies_minute has 16.27M rows; a plain CREATE INDEX would take a
--     full table-lock for several minutes, blocking the intraday ORATS ingest
--     crons (orats_monies_today_ingest*.py) that write to this table during
--     market hours.
--   - CONCURRENTLY builds the index without an exclusive lock, at the cost of
--     two full-table scans instead of one.
--   - CONCURRENTLY is ILLEGAL inside an explicit transaction (BEGIN/COMMIT).
--     Apply this file with: psql <dsn> -f orats_monies_minute_carry_rate_idx.sql
--     Do NOT wrap it in BEGIN/COMMIT.
--
-- Index design rationale:
--   (trade_date, ticker)      — compound equality predicate used by both
--                               _fetch_carry_rates subqueries and _resolve_implied_move
--   dte                       — ORDER BY dte ASC (sub1) / dte DESC (sub2)
--                               allows each subquery to be an O(log N) index seek
--                               returning exactly 1 row instead of scanning 21K rows
--   snapshot_pt DESC          — secondary sort for sub1 (dte ASC, snapshot_pt DESC);
--                               returns the most-recent snapshot for the nearest DTE
--                               above target in a single index probe.

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_omm_trade_date_ticker_dte_snapshot
    ON orats_monies_minute (trade_date, ticker, dte, snapshot_pt DESC);
```

Apply procedure:
```
psql $DATABASE_URL -f infra/sql/orats_monies_minute_carry_rate_idx.sql
```
Verify after applying:
```
\d orats_monies_minute    -- must show the new index with no "(INVALID)" marker
SELECT indexname, idx_scan FROM pg_stat_user_indexes WHERE tablename = 'orats_monies_minute';
```
If the index shows `(INVALID)`, CONCURRENTLY failed mid-build (connection drop,
etc.) — drop it and re-run.

## Query Rewrite (commit 3)

Replace the current `_fetch_carry_rates` body with:

```python
def _fetch_carry_rates(
    conn, ticker: str, trade_date: dt.date, dte_target: int = 15
) -> tuple[float, float]:
    """Return (risk_free_rate, yield_rate) from orats_monies_minute.

    Selects the row closest to dte_target (calendar days), using the latest
    snapshot on trade_date.  Used to populate strike_spx on proposal legs.

    Rewritten in CR-035 to avoid ORDER BY ABS(dte-%s) heapsort (was 3,501ms
    for a 21K-row full sort).  Uses a UNION ALL of two bounded sub-selects —
    nearest DTE >= target and nearest DTE < target — each an O(log N) index
    seek on idx_omm_trade_date_ticker_dte_snapshot.  Outer ORDER BY ABS(dte-%s)
    runs over at most 2 rows.

    Tiebreak (equidistant DTE above and below target): immaterial for carry
    rates; risk_free_rate and yield_rate are stable across DTE values in normal
    markets.  The sub-select ordering (dte ASC / dte DESC) is deterministic but
    the tiebreaker is not preserved from the old single-sort query.

    Fallback: (0.05, 0.0) if no data is available for the date.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT risk_free_rate, yield_rate
            FROM (
                (SELECT risk_free_rate, yield_rate, dte
                 FROM orats_monies_minute
                 WHERE trade_date = %s AND ticker = %s AND dte >= %s
                 ORDER BY dte ASC, snapshot_pt DESC
                 LIMIT 1)
                UNION ALL
                (SELECT risk_free_rate, yield_rate, dte
                 FROM orats_monies_minute
                 WHERE trade_date = %s AND ticker = %s AND dte > 0 AND dte < %s
                 ORDER BY dte DESC
                 LIMIT 1)
            ) candidates
            ORDER BY ABS(dte - %s) ASC
            LIMIT 1
            """,
            (
                trade_date.isoformat(), ticker, dte_target,   # sub1
                trade_date.isoformat(), ticker, dte_target,   # sub2
                dte_target,                                    # outer ORDER BY
            ),
        )
        row = cur.fetchone()
    if not row:
        return 0.05, 0.0
    rfr     = float(row[0]) if row[0] is not None else 0.05
    yield_r = float(row[1]) if row[1] is not None else 0.0
    return rfr, yield_r
```

`row[0]` = `risk_free_rate`, `row[1]` = `yield_rate` — unpack unchanged.

## Acceptance Criteria

### AC #1 — Old-vs-new equality on ≥ 5 trade_dates (correctness gate, MUST pass)

Run both the old single-sort query and the new UNION ALL query for at least 5
sampled trade_dates spanning different market regimes, confirm `(rfr, yield_r)`
results match (within float rounding, ≤ 1e-10 tolerance).

Equidistant-DTE tiebreak divergence is explicitly acceptable and does not fail
this AC.

### AC #2 — EXPLAIN shows index seeks on new query

```
EXPLAIN (ANALYZE, BUFFERS)
SELECT ... FROM (...) candidates ORDER BY ABS(dte-15) LIMIT 1;
```
Expected: two `Index Scan` nodes on `idx_omm_trade_date_ticker_dte_snapshot`,
each returning ≤ 1 row. No `Sort` node over > 10 rows. Total execution
time < 5ms (Render-internal).

### AC #3 — `_resolve_implied_move` also benefits

The new index also covers `(trade_date, ticker)` more tightly for the
`_resolve_implied_move` query. Its EXPLAIN should now show a tighter scan
(fewer rows filtered, possibly switching to the new index). Verify it still
returns correct results and its execution time ≤ the pre-CR 4.7ms.

### AC #4 — Endpoint timing

Time `/api/setup/proposals?date=YYYY-MM-DD&ticker=SPX` via curl (3 calls,
take median). Expected improvement: ~4.3s → ≤ 1.0s wall-clock (accounting for
~400ms connect + ~315ms corpus transfer + ~37ms KNN + ~50ms other SQL).

### Operational notes

- The CONCURRENTLY build on a 16.27M-row table on Render's free/hobby tier
  will take several minutes. Monitor `pg_stat_progress_create_index` or check
  `pg_indexes` for `(INVALID)` afterward.
- No downtime required; ingest crons continue normally during the build.
- No rollback script needed: `DROP INDEX CONCURRENTLY` if the index causes
  unexpected write amplification (unlikely — it's a read-only optimisation).
