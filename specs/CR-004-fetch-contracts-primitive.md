CR-004 — Add fetch_contracts(opra_symbols, start, end) primitive to fetcher.py
Context
Second of three CRs implementing the [[2026-05-04 - Pivot from SPY+chain to SPX+option endpoint]]. CR-003 shipped condor.py emitting native SPX OPRAs; CR-005 will switch the orchestrator. This CR adds the fetch primitive both will use.
The primitive is also the foundation for [[live-condor-pricing-on-dash]], which assumes a cache-aware fetch_contracts as shared infrastructure.
Decision
Add two functions to packages/shared/options_cache/fetcher.py:
pythondef _fetch_contracts_from_orats(
    opra_symbols: list[str],
    start: datetime,  # ET, naive
    end: datetime,    # ET, naive
) -> list[ContractBar]:
    """Pure HTTP. Hits ORATS /strikes/option with range-tradeDate.
    No DB I/O. No cache awareness. Used only by fetch_contracts."""

def fetch_contracts(
    opra_symbols: list[str],
    start: datetime,
    end: datetime,
) -> FetchSummary:
    """Cache-aware fetch. Computes per-OPRA gaps against
    orats_options_fetched_windows, calls the private primitive
    only for gaps, writes results to orats_options_minute and
    new window rows to orats_options_fetched_windows.
    Idempotent: re-running over the same range is a no-op."""
Public re-export in packages/shared/options_cache/__init__.py: fetch_contracts only. _fetch_contracts_from_orats stays module-private.
Additive change. fetch_chain is untouched.
Affected files

packages/shared/options_cache/fetcher.py — new functions
packages/shared/options_cache/__init__.py — add fetch_contracts to re-exports
tests/test_fetcher.py — new unit tests
tests/test_fetcher_smoke.py (new file) — real-ORATS smoke test, marked @pytest.mark.smoke and skipped in default CI runs

Grep packages/shared/options_cache/__init__.py for fetch_contracts after changes — should appear exactly once in the re-export list. Also grep the whole repo for fetch_contracts to confirm no pre-existing call sites (it's a net-new symbol).
Behavior spec
_fetch_contracts_from_orats(opra_symbols, start, end)

URL construction per [[orats-endpoint-conventions]]: /datav2/hist/strikes/option, range-tradeDate parameter in ET format, tickers parameter as comma-joined OPRA symbols.
Returns parsed bar records. No DB I/O.
Errors bubble up; retry/backoff is the HTTP client's job (existing pattern, unchanged).

fetch_contracts(opra_symbols, start, end) — algorithm
For each OPRA in opra_symbols:

Query orats_options_fetched_windows WHERE opra_symbol = X → list of (window_start_pt, window_end_pt).
Compute gaps in [start, end] not covered by any existing window. Treat adjacency as covered (an existing window ending at exactly start means no gap at the boundary). Result is a list of (gap_start, gap_end) intervals, possibly empty.
For each gap, call _fetch_contracts_from_orats([X], gap_start, gap_end) (one OPRA per call — matches the ~1,440-call budget in the acceptance criteria).
Write returned bars to orats_options_minute.
Insert one row into orats_options_fetched_windows per gap: (opra_symbol=X, window_start_pt=gap_start, window_end_pt=gap_end).

Return a FetchSummary dict for observability: {opras_processed, gaps_filled, bars_written, cache_hits} or similar.
Empty-response handling (important)
If _fetch_contracts_from_orats returns zero bars for a gap (legitimate ORATS no-data case — illiquid OPRA, pre-listing window, etc.), still write the orats_options_fetched_windows row. The provenance semantics are "we asked ORATS over this window; here's what they had." Not writing the row would cause perpetual refetches of empty windows. This mirrors the reasoning in [[2026-05-04 - Options Cache Schema Design]] that explicitly rejects inferring coverage from the bars table.
Window-coalescing on write — explicitly out of scope
Multiple adjacent or overlapping orats_options_fetched_windows rows per OPRA are allowed and expected. Gap detection at read time unions them correctly. Coalescing into single contiguous windows is a separable optimization for later, if the table ever gets unwieldy. Do not implement it in this CR.
Tests required
Unit tests (tests/test_fetcher.py)
For _fetch_contracts_from_orats:

Single-OPRA URL construction (correct prefix, tradeDate ET format, tickers param)
Multi-OPRA URL construction (comma-joined tickers)
Response parsing into ContractBar shape
HTTP error propagation
Empty-response handling (returns empty list, no crash)

For fetch_contracts (mock the private primitive and DB):

Empty cache, single OPRA → one HTTP call over full range, both tables written
Empty cache, multi-OPRA → N HTTP calls (one per OPRA), bars and windows written for all
Fully cached, single OPRA → zero HTTP calls, zero DB writes, summary reflects cache hit
Partial overlap at start: cache [a, b], request [a-X, b] → one HTTP call over [a-X, a]
Partial overlap at end: cache [a, b], request [a, b+X] → one HTTP call over [b, b+X]
Gap in middle: cache [a, b] ∪ [c, d], request [a, d] → one HTTP call over [b, c]
Adjacency boundary: cache [a, b], request [b, c] → one HTTP call over [b, c] or zero calls if treating b as covered — pick one and assert. (Recommend: exact-boundary b is covered, so request [b, c] becomes a gap of [b, c] with b excluded — confirm with implementation.)
Mixed cache states across OPRAs: OPRA A fully cached, OPRA B uncached, request [a, b] → one HTTP call for B only
Empty ORATS response for a gap → orats_options_fetched_windows row still written, zero bars in orats_options_minute
Post-write state correctness: assert exact rows in both tables match expectations

Smoke test (tests/test_fetcher_smoke.py)

One real call to ORATS for one SPX OPRA over a one-day range
Assert non-empty response, basic shape (bid/ask present, greeks populated)
Marked @pytest.mark.smoke; skipped in default CI runs; runnable explicitly with pytest -m smoke

Existing tests
Read every tests/test_*.py in options_cache/ before implementing. CR-004 is additive, but the lesson from CR-003 is that integration/orchestrator tests can ripple unexpectedly. Specifically check:

tests/test_fetcher.py — existing fetch_chain tests must still pass unchanged
tests/test_orchestrator.py — should be unaffected (CR-005 wires it); confirm no incidental import or mock changes break it
tests/test_http_client.py — should be unaffected; confirm
tests/test_chunking.py, tests/test_csv_parser.py, tests/test_condor.py — should all pass unchanged

Acceptance criteria

All existing tests still pass (97 from post-CR-003).
New unit tests cover all gap-detection cases listed above.
Smoke test against real ORATS passes for one SPX OPRA over a one-day range.
fetch_chain behavior unchanged — grep call sites, confirm no swaps.
packages/shared/options_cache/__init__.py re-exports fetch_contracts (and not the private function).
Re-running fetch_contracts over an already-cached range produces zero HTTP calls (verified by mock assertions in unit tests).

Non-goals

Orchestrator wiring (CR-005).
Bounded concurrency in fetch_contracts (P1 tech debt; sequential is acceptable for now).
Window coalescing on write.
Any change to fetch_chain or its call sites.
End-to-end verification against scan_id=2 (CR-005).