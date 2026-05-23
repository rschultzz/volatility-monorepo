CR-005 — Switch orchestrator to fetch_option_bars + end-to-end verification

Context
Third and final CR implementing [[2026-05-04 - Pivot from SPY+chain to SPX+option endpoint]]. CR-003 shipped condor.py emitting native SPX OPRAs ✅. CR-004 shipped the cache-aware fetch_option_bars primitive ✅. This CR rewires orchestrator.fetch_for_rows to use that primitive instead of fetch_chain, redesigns the dedup logic around OPRAs instead of underlyings, and verifies the full pipeline end-to-end against scan_id=2.
Phase 4a of the options cache is currently blocked end-to-end on this CR. Phases 4b (backend API endpoint), 4c (frontend "Fetch pricing" button), and 5 (P&L reconstruction) all gate on Phase 4a working.
Decision
Rewrite orchestrator.fetch_for_rows to:

Switch the fetch primitive from fetch_chain to fetch_option_bars.
Redesign the dedup key from (underlying_ticker, window) to (opra_symbol, window) — collecting tuples across all legs of all rows, then computing a per-OPRA bounding window (min-start, max-end) across the rows the OPRA appears in.
Drop the underlying and chain_filter parameters from the public signature — neither applies to the option pathway.
Update the OrchestratorResult aggregate fields to reflect option-endpoint semantics (rename unique_windows_fetched → unique_opras_fetched, retype fetch_summaries → option_fetch_summaries: list[FetchOptionBarsSummary], add cache_hits aggregate).
Verify end-to-end against scan_id=2 ('down-long', 360 rows). Manual run via the existing cli.py fetch-for-scan subcommand; PR description carries the verification checklist.

Behavior changes to fetch_for_rows callers:

The function now issues HTTP calls per unique OPRA, not per unique underlying-ticker. Call count scales with leg count, not row count.
OrchestratorResult.fetch_summaries field is renamed and retyped. Only known caller is cli.py:_cmd_fetch_for_scan; updated here.
underlying and chain_filter kwargs are removed. Only known caller is cli.py:_cmd_fetch_for_scan; the _build_filter call inside _cmd_fetch_for_scan is removed.

Legacy fetch_chain / fetch_contract / fetch_contracts are untouched. Retirement deferred to a future CR.
Affected files

packages/shared/options_cache/orchestrator.py — main rewrite. Replace fetch_chain import with fetch_option_bars. Replace dedup logic in fetch_for_rows. Update OrchestratorResult dataclass.
packages/shared/options_cache/cli.py — update _cmd_fetch_for_scan to drop chain_filter passing and to use the renamed unique_opras_fetched field. Remove _add_filter_args(p_scan) from the fetch-for-scan subparser (filter args no longer apply). The fetch-chain and fetch subcommands stay as-is (legacy is untouched per this CR's scope).
packages/shared/options_cache/tests/test_orchestrator.py — switch mocks from fetch_chain to fetch_option_bars. Update existing 6 tests for the new signature/dedup semantics. Add 3 new tests for OPRA-level dedup, bounding-window union, and error propagation across rows sharing an OPRA.
packages/shared/options_cache/__init__.py — no API surface change; fetch_for_rows, OrchestratorResult, RowResult, Strategy, get_strategy continue to be re-exported. Confirm by grep after edit.

Pre-implementation greps required (CR-004 lesson):

Grep the entire repo for fetch_for_rows to enumerate all call sites. Expect only cli.py:_cmd_fetch_for_scan and tests. If any other call site surfaces, stop and surface to chat before proceeding.
Grep the entire repo for unique_windows_fetched to confirm only CLI prints it. Field rename is a breaking surface; any unexpected reader needs flagging.
Grep __init__.py for the orchestrator's re-exports; confirm the public API surface is unchanged after edits.
Grep OrchestratorResult to find anyone touching .fetch_summaries or the aggregate counter fields.

OrchestratorResult dataclass — changes
Current fields (post-CR-004):
python@dataclass
class OrchestratorResult:
    rows: list[RowResult] = field(default_factory=list)
    fetch_summaries: list[FetchSummary] = field(default_factory=list)
    rows_attempted: int = 0
    rows_with_legs: int = 0
    unique_windows_fetched: int = 0
    total_api_calls: int = 0
    total_bars_inserted: int = 0
New fields:
python@dataclass
class OrchestratorResult:
    rows: list[RowResult] = field(default_factory=list)
    option_fetch_summaries: list[FetchOptionBarsSummary] = field(default_factory=list)
    rows_attempted: int = 0
    rows_with_legs: int = 0
    unique_opras_fetched: int = 0
    cache_hits: int = 0
    total_api_calls: int = 0
    total_bars_inserted: int = 0
Field semantics:

option_fetch_summaries: one entry per fetch_option_bars call made (one call per unique OPRA).
unique_opras_fetched: count of unique OPRAs the orchestrator called fetch_option_bars for. Includes OPRAs that hit cache entirely.
cache_hits: sum of s.cache_hits across option_fetch_summaries. Should equal "OPRAs that needed zero HTTP calls."
total_api_calls: sum of s.gaps_filled across option_fetch_summaries. One HTTP call per gap filled.
total_bars_inserted: sum of s.bars_written across option_fetch_summaries.

fetch_for_rows — new signature
pythondef fetch_for_rows(
    rows: Iterable[dict],
    *,
    strategy: str = "condor",
    source: FetchSource = "historical_backfill",
    row_key_fn: Callable[[dict], str] = None,
) -> OrchestratorResult:
Removed kwargs: underlying, chain_filter. Both no longer apply to the option pathway. cli.py:_cmd_fetch_for_scan is the only caller passing these; updated in this CR.
fetch_for_rows — new algorithm

Validate strategy + set row_key_fn. Unchanged from current.
Initialize state:

result = OrchestratorResult(rows_attempted=len(rows_list))
per_opra_windows: dict[str, list[tuple[datetime, datetime]]] = defaultdict(list) — per OPRA, list of windows from rows that reference it.
opra_to_row_keys: dict[str, set[str]] = defaultdict(set) — per OPRA, set of row_keys that reference it (used for error attribution).


Step 1: derive legs and windows per row (largely unchanged structure):

For each row: compute row_key, attempt strat.legs_fn(row), attempt strat.window_fn(row).
On any failure or empty result, append a RowResult with error set and continue. Match existing error messages: "legs_fn failed: …", "no legs derivable from row", "window_fn failed: …", "no pricing window derivable from row".
On success: populate RowResult.legs and RowResult.window; increment rows_with_legs; for each leg, do per_opra_windows[leg.opra_symbol].append(window) and opra_to_row_keys[leg.opra_symbol].add(row_key).


Step 2: compute per-OPRA bounding windows. For each (opra, windows) in per_opra_windows:

start = min(w[0] for w in windows)
end = max(w[1] for w in windows)
This is the bounding-box union per the design decision. Disjoint sub-ranges within [start, end] will be discovered as gaps inside fetch_option_bars' gap detection — over-fetching is bounded by the cache layer.


Step 3: call fetch_option_bars per OPRA. For each (opra, bounding) from step 2, sorted for determinism:

Log: "fetch_for_rows: opra %s [%s, %s] covers %d row(s)", mirroring current log format.
Increment result.unique_opras_fetched.
Call fetch_option_bars(opra_symbols=[opra], start_pt=bounding[0], end_pt=bounding[1], source=source).
On exception: for each row_key in opra_to_row_keys[opra], locate the matching RowResult and set error = f"fetch_option_bars failed for {opra}: {e}" if error is None (preserves first-error semantics; matches current code).
On success: append the summary to option_fetch_summaries; increment aggregates (total_api_calls += summary.gaps_filled, total_bars_inserted += summary.bars_written, cache_hits += summary.cache_hits).


Return.

Behavior notes
Why bounding-box per OPRA, not disjoint ranges. When two rows reference the same OPRA over non-contiguous windows (e.g., [09:00,10:00] and [15:00,16:00]), the orchestrator passes the bounding [09:00,16:00] to fetch_option_bars. The primitive's gap detection sees both pieces as gaps against the empty cache and issues two HTTP calls — same end state as a disjoint-ranges approach. On a partially-cached OPRA, gap detection narrows to only what's missing. The over-fetch case is "completely empty cache + disjoint windows" where we fetch the in-between region too; for typical condor scans (same trade day, hours apart), this is negligible. If pathological patterns surface in later phases, revisit with explicit disjoint ranges.
One OPRA per fetch_option_bars call. Even though the primitive's signature accepts a list, the orchestrator calls it with a single-element list per unique OPRA. This is simpler to test, simpler to attribute errors, and produces the same HTTP call count as any batching alternative. Batching multiple OPRAs into one call is a deferrable optimization.
Error attribution across shared OPRAs. A single failed fetch_option_bars call propagates the same error to every row referencing that OPRA. Rows with multiple failing legs only record the first error (matches current first-error-wins semantics). This matters because the new dedup pattern can cause one HTTP failure to affect many rows — important for downstream observability.
No retries at the orchestrator layer. fetch_option_bars raises on permanent errors; transient retries are the HTTP client's job (existing pattern, unchanged). Orchestrator catches and attributes; it doesn't retry.
CLI changes (cli.py:_cmd_fetch_for_scan)

Remove _build_filter(args) call; don't pass chain_filter to fetch_for_rows.
Remove _add_filter_args(p_scan) from the fetch-for-scan subparser definition. Filter args still apply to fetch-chain and fetch subcommands — those keep their _add_filter_args calls.
Update the print block to use new field names:

Unique OPRAs fetched: {result.unique_opras_fetched} (was: Unique windows fetched)
Cache hits: {result.cache_hits} (new line)
Total API calls, Total bars inserted — unchanged labels.


Do NOT remove or change the fetch-chain / fetch subcommands. Legacy retirement is out of scope for this CR.

Tests required
Unit tests (packages/shared/options_cache/tests/test_orchestrator.py — rewrite)
All existing 6 tests need their @patch("packages.shared.options_cache.orchestrator.fetch_chain") swapped to @patch("packages.shared.options_cache.orchestrator.fetch_option_bars"), and mock side-effects updated to return FetchOptionBarsSummary instead of FetchSummary.
Helper update: replace _fake_summary(ticker, start, end) with:
pythondef _fake_option_summary(opras_processed=1, gaps_filled=1, bars_written=500, cache_hits=0):
    return FetchOptionBarsSummary(
        opras_processed=opras_processed,
        gaps_filled=gaps_filled,
        bars_written=bars_written,
        cache_hits=cache_hits,
    )
Updated existing tests (semantic equivalents):

test_single_row_one_fetch → asserts mock_fetch.call_count == 4 (one per leg, four legs per row). unique_opras_fetched == 4.
test_multiple_rows_same_day_one_fetch → two identical rows share all 4 legs and the same window; still mock_fetch.call_count == 4, unique_opras_fetched == 4. (Rename: test_multiple_rows_shared_legs_dedups.)
test_multiple_rows_different_days_multiple_fetches → two rows with different trade dates → 8 unique OPRAs → mock_fetch.call_count == 8.
test_row_with_missing_data_skipped → unchanged in behavior; only mock target name changes.
test_fetch_chain_failure_marks_affected_rows → rename test_fetch_option_bars_failure_marks_affected_rows. Update error string assertion to "fetch_option_bars failed". The two rows share all 4 legs and the same window → 4 fetch calls, all fail (since side_effect raises), both rows get error.
test_unknown_strategy_raises → unchanged.

New tests (3):

test_overlapping_windows_unioned_to_bounding_box: Two rows with the same legs but partly overlapping windows (e.g., [09:00, 10:00] and [09:30, 11:00]). Assert fetch_option_bars is called 4 times (one per leg), each with start_pt=09:00, end_pt=11:00. Confirms the union math.
test_disjoint_windows_unioned_to_bounding_box: Two rows with the same legs but disjoint windows (e.g., [09:00, 10:00] and [14:00, 15:00]). Assert fetch_option_bars is called 4 times, each with start_pt=09:00, end_pt=15:00. Documents the over-fetch behavior is intentional.
test_error_on_shared_opra_propagates_to_all_referring_rows: Two rows sharing one specific OPRA (and three unique OPRAs each). Mock fetch_option_bars to fail only for the shared OPRA; succeed for the others. Both rows should get the error set. Rows with all-unique-legs unaffected.

Existing tests to verify still pass
Per CR-004's lesson on integration-test ripple:

tests/test_fetcher.py — fetcher unit tests; should be unaffected.
tests/test_fetcher_smoke.py — CR-004's smoke test; unchanged.
tests/test_condor.py — leg-derivation tests; should pass unchanged (we don't touch condor.py).
tests/test_windows.py, tests/test_chunking.py, tests/test_csv_parser.py, tests/test_http_client.py, tests/test_opra.py — module-internal; should pass unchanged.

Total expected: 124 (post-CR-004 baseline) – existing-6-replaced + existing-6-updated + 3-new = 127. Confirm exact count after implementation.
End-to-end verification (manual; documented in PR)
No new automated test for this. Run via CLI against a JSON dump of scan_id=2 rows. Verification checklist for the PR description:

Activate venv, ensure ORATS_API_KEY is set and DB is reachable.
Export scan_id=2's 360 rows to /tmp/scan_id_2_rows.json (manual step — Ryan does this from the DB).
Run: python -m packages.shared.options_cache.cli fetch-for-scan --rows-json /tmp/scan_id_2_rows.json --strategy condor -v
Expected output (within rough tolerances):

Rows attempted: 360
Rows with legs: 360 (or near, depending on data quality)
Unique OPRAs fetched: ≤1,440 (4 legs × 360 rows = 1,440 max if zero cross-row leg sharing; lower if any sharing)
Total API calls: ~1,440 on cold cache; near 0 on second run (idempotency check)
Total bars inserted: > 0 first run; 0 on second run
No ⚠ N row(s) had errors block, or only a small handful for known-bad scan rows


After the run, spot-check the DB: pick one row from the scan, look up its 4 condor leg OPRAs, query orats_options_minute for each — confirm bars exist over the expected pricing window.
Re-run the same command immediately. Confirm Total API calls: 0, Cache hits: <unique_opras_fetched>. Idempotency check.

Acceptance criteria

All existing tests still pass (124 from post-CR-004 baseline) after the test file is rewritten — i.e., the test file is replaced, not strictly "all 124 unchanged-and-passing." Expected new total: 127.
No reference to fetch_chain remains in orchestrator.py (grep verifies).
Legacy fetch_chain / fetch_contract / fetch_contracts still exported from __init__.py and importable; their tests still pass.
fetch_for_rows signature has no underlying or chain_filter kwarg.
OrchestratorResult has the new field names and types; old names (fetch_summaries, unique_windows_fetched) are gone (no aliases — clean break).
End-to-end run against scan_id=2 succeeds with the metrics above. PR description carries the manual verification log.
Idempotent: second run of the same scan produces 0 API calls.

Non-goals

Retirement of legacy fetch_chain / fetch_contract / fetch_contracts. Deferred to a future CR.
Bounded concurrency in fetch_for_rows. Sequential per-OPRA fetches are acceptable for Phase 4a. Revisit if 1,440 sequential calls becomes painful in practice (~tens of minutes is the rough budget).
Disjoint-range union per OPRA instead of bounding-box. Bounding-box is the explicit design choice; revisit later if pathological patterns surface.
New CLI subcommand for automated verification. Manual eyeball check in the PR description is sufficient for this CR.
Phase 4b (backend API endpoint), Phase 4c (frontend button), Phase 5 (P&L reconstruction). Unblocked by this CR but not part of it.