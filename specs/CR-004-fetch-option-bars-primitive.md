CR-004 — Add fetch_option_bars(opra_symbols, start, end) primitive to fetcher.py
Context
Second of three CRs implementing the [[2026-05-04 - Pivot from SPY+chain to SPX+option endpoint]]. CR-003 shipped condor.py emitting native SPX OPRAs; CR-005 will switch the orchestrator. This CR adds the fetch primitive both will use.
The primitive is also the foundation for [[live-condor-pricing-on-dash]], which assumes a cache-aware fetch_option_bars as shared infrastructure.
Decision
Add two functions to packages/shared/options_cache/fetcher.py:
pythondef _fetch_option_bars_from_orats(
    opra_symbol: str,
    start_pt: datetime,  # naive PT (module convention)
    end_pt: datetime,    # naive PT
) -> list[OptionMinuteBar]:
    """Pure HTTP. Hits ORATS /datav2/hist/live/one-minute/strikes/option
    with the full OPRA symbol as the singular ticker param and a
    range-tradeDate (PT→ET conversion internal, reusing
    _format_trade_date_param). Returns ALL bars from the parsed response
    — including the counterpart OPRA at the same strike+expir, since the
    existing csv_parser emits 2 bars per CSV row. No DB I/O. No cache
    awareness. Used only by fetch_option_bars."""

def fetch_option_bars(
    opra_symbols: list[str],
    start_pt: datetime,  # naive PT
    end_pt: datetime,    # naive PT
) -> FetchOptionBarsSummary:
    """Cache-aware fetch. For each OPRA, computes gaps against
    orats_options_fetched_windows, calls the private primitive one OPRA
    at a time for each gap, writes returned bars to orats_options_minute,
    and writes orats_options_fetched_windows rows for both the requested
    OPRA AND the counterpart OPRA returned in the same response.
    Idempotent: re-running over the same range is a no-op."""
Public re-exports in packages/shared/options_cache/__init__.py: fetch_option_bars (function) and FetchOptionBarsSummary (dataclass). _fetch_option_bars_from_orats stays module-private.
Additive change. fetch_chain / fetch_contract / legacy fetch_contracts are untouched.
Affected files

packages/shared/options_cache/fetcher.py — new functions
packages/shared/options_cache/models.py — new FetchOptionBarsSummary dataclass
packages/shared/options_cache/opra.py — new opra_to_orats_ticker(opra: str) -> str helper (strips the side character for ORATS option-endpoint requests)
packages/shared/options_cache/__init__.py — add fetch_option_bars, FetchOptionBarsSummary, and opra_to_orats_ticker to re-exports
packages/shared/options_cache/tests/test_fetcher.py (new file) — new unit tests
packages/shared/options_cache/tests/test_fetcher_smoke.py (new file) — real-ORATS smoke test, marked @pytest.mark.smoke and skipped in default CI runs

Grep packages/shared/options_cache/__init__.py for fetch_option_bars after changes — should appear exactly once in the re-export list. Also grep the whole repo for fetch_option_bars to confirm no pre-existing call sites (it's a net-new symbol).
FetchOptionBarsSummary dataclass (in models.py)

opras_processed: int — count of OPRAs in the input list
gaps_filled: int — count of gaps detected and fetched across all OPRAs
bars_written: int — total bars inserted into orats_options_minute (excludes ON CONFLICT skips)
cache_hits: int — count of OPRAs that needed zero HTTP calls (fully cached)

Behavior spec
_fetch_option_bars_from_orats(opra_symbol, start_pt, end_pt)

URL construction per [[orats-endpoint-conventions]]:
  - path: /datav2/hist/live/one-minute/strikes/option (the Live Intraday tier's historical option path; NOT /datav2/hist/strikes/option)
  - ticker param (singular): the side-stripped OPRA form, i.e., ROOT + YYMMDD + 8-digit-strike with NO C|P character. Example: the OPRA `SPX240202P04935000` (18 chars, our internal canonical) becomes the ORATS ticker `SPX24020204935000` (17 chars). The option endpoint returns one row containing both call and put data for the requested strike+expir, so side isn't part of the query — sending the 18-char side-bearing form returns 404 (verified during CR-004 smoke testing). Conversion via a new opra_to_orats_ticker() helper in opra.py (see Affected files).
  - tradeDate param: ET format YYYYMMDDHHMM. Single-timestamp if start_pt == end_pt; otherwise comma-joined start,end. PT→ET conversion via the existing _format_trade_date_param helper.
Returns ALL bars from the parsed response (both call and put OPRAs at the strike+expir matching the request, as csv_parser emits 2 bars per row). No DB I/O.
Errors bubble up; retry/backoff is the HTTP client's job (existing pattern, unchanged).

fetch_option_bars(opra_symbols, start_pt, end_pt) — algorithm
Initialize counters: gaps_filled=0, bars_written=0, cache_hits=0.
For each OPRA X in opra_symbols:

Query orats_options_fetched_windows WHERE opra_symbol = X via repo.get_windows_for_contract(X).
Compute gaps in [start_pt, end_pt] not covered by any existing window via windows.find_gaps. Result is a list of TimeRange gaps, possibly empty.
If gaps is empty: increment cache_hits, continue to next OPRA.
For each gap, call _fetch_option_bars_from_orats(X, gap.start_pt, gap.end_pt) (one OPRA per call — matches the ~1,440-call budget in the acceptance criteria).
Write returned bars to orats_options_minute via repo.insert_bars (idempotent via ON CONFLICT). Add the returned insert count to bars_written.
Record orats_options_fetched_windows rows for every unique opra_symbol in the parsed response (in practice: the requested OPRA X plus the counterpart OPRA at the same strike+expir). One repo.record_fetched_window call per (opra_symbol, gap). row_count = bars-for-that-opra in the gap.
Increment gaps_filled by len(gaps).

Return FetchOptionBarsSummary(opras_processed=len(opra_symbols), gaps_filled=gaps_filled, bars_written=bars_written, cache_hits=cache_hits).
Why write windows for both sides
The ADR ([[2026-05-04 - Options Cache Schema Design]]) defines orats_options_fetched_windows as cache-state for gap detection, not request history. Recording only the requested OPRA would create a state where bars exist in orats_options_minute for an OPRA whose fetched_windows row says "not cached" — exactly the two-tables-disagreeing situation the ADR was designed to prevent. Request-history auditability belongs in orats_options_fetch_jobs, which the ADR sets up explicitly for that purpose. The aggressive recording here is also a direct benefit to [[live-condor-pricing-on-dash]], whose condors query both put and call sides.
Empty-response handling (important)
If _fetch_option_bars_from_orats returns zero bars for a gap (legitimate ORATS no-data case — illiquid OPRA, pre-listing window, etc.), still write the orats_options_fetched_windows row for the requested OPRA X. The provenance semantics are "we asked ORATS over this window; here's what they had." Not writing the row would cause perpetual refetches of empty windows. This mirrors the reasoning in [[2026-05-04 - Options Cache Schema Design]] that explicitly rejects inferring coverage from the bars table. (No counterpart-side rows are written in this case — there's no counterpart OPRA to identify when the response is empty.)
Window-coalescing on write — explicitly out of scope
Multiple adjacent or overlapping orats_options_fetched_windows rows per OPRA are allowed and expected. Gap detection at read time unions them correctly. Coalescing into single contiguous windows is a separable optimization for later, if the table ever gets unwieldy. Do not implement it in this CR.
Tests required
Unit tests (packages/shared/options_cache/tests/test_fetcher.py — new file)
For _fetch_option_bars_from_orats:

URL construction with comma-range tradeDate (start_pt != end_pt): correct path /datav2/hist/live/one-minute/strikes/option, correct ticker=<side-stripped OPRA form, e.g. SPX24020204935000>, correct ET-formatted comma-joined tradeDate (PT→ET converted)
URL construction with single-timestamp tradeDate (start_pt == end_pt): correct ET-formatted single timestamp
Response parsing returns full bars list (both call-side and put-side OPRA bars per CSV row, matching csv_parser behavior)
HTTP error propagation (OratsTransientError, OratsPermanentError both bubble)
Empty-response handling (returns empty list, no crash)

For opra_to_orats_ticker:

Put input → side-stripped form (e.g., SPX240202P04935000 → SPX24020204935000)
Call input → side-stripped form (e.g., SPX240202C04935000 → SPX24020204935000) — note that both sides map to the same ORATS ticker, by design
Multiple root lengths: SPX (3), SPXW (4), AAPL (4), SPY (3) — confirm string-slicing assumptions hold
Malformed OPRA input → ValueError (delegated to parse_opra)

For fetch_option_bars (mock the private primitive and DB):

Empty cache, single OPRA → one HTTP call over full range; bars written; fetched_windows rows written for BOTH the requested OPRA AND the counterpart OPRA seen in the response
Empty cache, multi-OPRA → N HTTP calls (one per OPRA); bars and windows written for all (including counterparts)
Fully cached, single OPRA → zero HTTP calls, zero DB writes, summary.cache_hits == 1
Partial overlap at start: cache [a, b], request [a-X, b] → one HTTP call over [a-X, a-1min] (per windows.find_gaps semantics)
Partial overlap at end: cache [a, b], request [a, b+X] → one HTTP call over [b+1min, b+X]
Gap in middle: cache [a, b] ∪ [c, d], request [a, d] → one HTTP call over [b+1min, c-1min]
Adjacency boundary: assert behavior matches windows.find_gaps semantics (existing test_windows.test_adjacent_windows_merge defines the convention — end+1min adjacency is treated as covered)
Mixed cache states across OPRAs: OPRA A fully cached, OPRA B uncached, request [a, b] → one HTTP call for B only; summary.cache_hits == 1, gaps_filled == 1
Empty ORATS response for a gap → fetched_windows row written for requested OPRA only (no counterpart known); zero bars in orats_options_minute
Counterpart-side window recording: requesting a call OPRA at strike K returns bars for both call and put OPRAs at strike K; assert orats_options_fetched_windows has rows for BOTH OPRAs after the call
Post-write state correctness: assert exact rows in both orats_options_minute and orats_options_fetched_windows match expectations

Smoke test (packages/shared/options_cache/tests/test_fetcher_smoke.py — new file)

One real call to ORATS for one SPX OPRA over a one-day range
Assert non-empty response, basic shape (bid/ask present, greeks populated)
Gated by @unittest.skipUnless(os.environ.get("RUN_ORATS_SMOKE") == "1", ...) + @unittest.skipUnless(os.environ.get("ORATS_API_KEY"), ...) so the test is skipped in every default test run. pytest.mark.smoke is set as a module-level pytestmark, guarded by `try: import pytest`, for forward compatibility if pytest infra ever lands. Run explicitly with: `RUN_ORATS_SMOKE=1 ORATS_API_KEY=... python -m unittest packages.shared.options_cache.tests.test_fetcher_smoke`. (Deviation from spec v2's "pytest -m smoke" mechanism: the repo has no pytest dep, no pytest.ini, no conftest.py — adding pytest infrastructure for one smoke test was out of scope.)

Existing tests
Read every packages/shared/options_cache/tests/test_*.py before implementing. CR-004 is additive, but the lesson from CR-003 is that integration/orchestrator tests can ripple unexpectedly. Specifically check:

packages/shared/options_cache/tests/test_orchestrator.py — should be unaffected (CR-005 wires it); confirm no incidental import or mock changes break it
packages/shared/options_cache/tests/test_http_client.py — should be unaffected; confirm
packages/shared/options_cache/tests/test_windows.py — gap-detection logic that fetch_option_bars consumes; read carefully, confirm no incidental breakage
packages/shared/options_cache/tests/test_chunking.py, packages/shared/options_cache/tests/test_csv_parser.py, packages/shared/options_cache/tests/test_condor.py — should all pass unchanged

No behavior change to fetch_chain / fetch_contract / legacy fetch_contracts; verified by grep + manual call-graph review including cli.py:91.
Acceptance criteria

All existing tests still pass (97 from post-CR-003; verify after venv activation).
New unit tests cover all cases listed above.
Smoke test against real ORATS passes for one SPX OPRA over a one-day range.
fetch_chain behavior unchanged — grep call sites, confirm no swaps.
packages/shared/options_cache/__init__.py re-exports fetch_option_bars and FetchOptionBarsSummary (and not the private function).
Re-running fetch_option_bars over an already-cached range produces zero HTTP calls (verified by mock assertions in unit tests).
After a fetch with non-empty response, orats_options_fetched_windows has rows for both the requested OPRA and the counterpart OPRA returned in the same response.

Non-goals

Orchestrator wiring (CR-005).
Bounded concurrency in fetch_option_bars (P1 tech debt; sequential is acceptable for now).
Window coalescing on write.
Any change to fetch_chain or its call sites.
Legacy chain-based fetch_chain / fetch_contract / fetch_contracts untouched. Retirement deferred to CR-005 or later.
End-to-end verification against scan_id=2 (CR-005).
