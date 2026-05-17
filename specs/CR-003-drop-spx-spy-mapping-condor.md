# CR-003: Drop SPX→SPY mapping in condor.py

## Status
Proposed — 2026-05-17

## Problem

`packages/shared/options_cache/condor.py` currently derives iron-condor legs by mapping SPX scan-row strikes to nearest SPY strikes via spot-relative percentage (`map_spx_strike_to_spy`). The mapping was a workaround for an assumed gap in ORATS coverage that turned out not to exist — ORATS' Live Intraday subscription does cover SPX intraday at the minute level, confirmed by ORATS support. With native SPX coverage available, the mapping introduces unnecessary basis-approximation error in backtest pricing and adds code complexity that no longer serves a purpose. See ADR `2026-05-04 - Pivot from SPY+chain to SPX+option endpoint` in the vault for the architectural decision.

This is the first of three CRs implementing the pivot:
- **CR-003 (this CR):** Drop the SPX→SPY mapping in `condor.py`.
- CR-004: Add `fetch_contracts` primitive to `fetcher.py` using `/strikes/option` with range `tradeDate`.
- CR-005: Switch orchestrator to `fetch_contracts` and verify end-to-end against `scan_id=2`.

CR-003 is isolated: no caller of `condor.py` is exercised end-to-end today (Phase 4a never completed end-to-end), so changing condor's output from SPY OPRAs to SPX OPRAs cannot regress anything currently in production. Unit tests must continue to pass.

## Proposed Solution

In `packages/shared/options_cache/condor.py`:

1. **Delete `map_spx_strike_to_spy`** function entirely. All call sites are inside this file.
2. **Simplify `condor_legs_for_row`**:
   - Change `underlying` parameter default from `"SPY"` to `"SPX"`. Keep the parameter; don't hardcode.
   - Remove the `target_spx_price` lookup and the spot-validity check at the top of the function. With no mapping, spot is not needed.
   - In the per-horizon loop, remove the `map_spx_strike_to_spy(spx_strike, spx_spot)` call. Pass the SPX strike directly to `format_opra`.
   - Update the dedup key from `(option_type, spy_strike)` to `(option_type, spx_strike)`. Semantically the same; just renamed for clarity.
   - Rewrite the docstring: drop the SPY mapping description; describe what the function actually does now (constructs SPX OPRAs from native scan-row strikes).
3. **Leave `_condor_strikes_for_horizon`, `condor_pricing_window_for_row`, `Leg`, time utilities, and role/horizon constants unchanged.**

In `packages/shared/options_cache/opra.py`:

4. **Docstring fix in the module header.** Replace the sentence about SPXW being for weeklies and SPX for monthlies with: "ORATS uses root `SPX` for all SPX expirations, including PM-settled weeklies (colloquially called SPXW). The `expiryTod` column on chain rows distinguishes AM vs PM settlement." No code change.

In `packages/shared/options_cache/tests/test_condor.py`:

5. **Delete `TestMapSpxStrikeToSpy` class** entirely (6 tests).
6. **Delete the `map_spx_strike_to_spy` import** at the top of the file.
7. **In `TestCondorLegsForRow`:**
   - `test_basic_row_yields_4_legs`: no logical change; verify the C/P character-index assertion still works (SPX is 3 chars same as SPY, so the index stays 9).
   - `test_legs_are_spy_opra`: rename to `test_legs_are_spx_opra`. Change `startswith("SPY")` to `startswith("SPX")`. Length assertion stays 18 (SPX + YYMMDD + C/P + 8-digit strike).
   - `test_different_horizons_yield_more_legs`: rewrite the inline comment explaining "SPX→SPY divides by ~10, so 30-pt SPX gap = 3 SPY strikes apart" — with native SPX, strike differences are preserved 1:1. The existing test data (`4935/4925/4980/4990` for 120m, `4880/4870/5020/5030` for to_close) already differs across horizons; the test logic still works.
   - `test_overlapping_horizons_dedupe`: still valid; dedup key on `(option_type, strike)` still functions correctly.
   - **Delete `test_missing_spx_spot_returns_empty`** — `target_spx_price` is no longer consulted, so missing it has no effect.
   - `test_explicit_expiration_override`: still works; the strike-portion check is unchanged.
8. **In `TestCondorPricingWindowForRow`:** no changes.

## Affected Files

- `packages/shared/options_cache/condor.py` — delete one function, simplify another, rewrite one docstring.
- `packages/shared/options_cache/opra.py` — module-header docstring fix only. No code.
- `packages/shared/options_cache/tests/test_condor.py` — delete one test class, delete one test, update two tests' assertions, remove one import.

No other files in the repo are touched. `fetcher.py`, `orchestrator.py`, `cli.py`, and all other tests are unchanged in this CR.

## Acceptance Criteria

Functional:
- `condor_legs_for_row(row)` (called with no explicit `underlying` arg) returns `Leg`s whose `opra_symbol` starts with `SPX` and encodes SPX strikes from `row['hypothetical_condor_120m']` / `hypothetical_condor_to_close` directly — no SPY rounding.
- For the reference row with SPX 4935 short put expiring 2024-02-02, the produced OPRA is exactly `SPX240202P04935000`.
- `target_spx_price` is no longer consulted; a row missing it still produces legs as long as strikes and `trade_date` are present.
- Pricing window logic is unchanged.

Structural:
- `map_spx_strike_to_spy` no longer exists in the codebase. `grep -rn map_spx_strike_to_spy packages/ apps/` returns zero hits.
- `condor_legs_for_row`'s `underlying` parameter defaults to `"SPX"` and remains overridable.

Test:
- `python -m unittest discover packages/shared/options_cache/tests` reports zero failures and zero errors.
- Expected pass count drops from 104 to roughly 97 (104 − 6 from `TestMapSpxStrikeToSpy` − 1 from `test_missing_spx_spot_returns_empty`). Claude Code should confirm the actual count.

## Verification

Automated (Claude Code runs these):

```
cd ~/code/volatility-monorepo
python -m unittest discover packages/shared/options_cache/tests -v
```

Expected: all tests pass; total ~97 (down from 104).

Spot-check the SPX OPRA construction:

```
python -c "
from packages.shared.options_cache.condor import condor_legs_for_row
row = {
    'trade_date': '2024-02-02',
    'target_ts_utc': '2024-02-02T17:14:00Z',
    'hypothetical_condor_120m': {
        'short_put_strike': 4935.0,
        'long_put_strike': 4925.0,
        'short_call_strike': 4980.0,
        'long_call_strike': 4990.0,
    },
}
for leg in condor_legs_for_row(row):
    print(leg.opra_symbol, leg.side, leg.role)
"
```

Expected: four legs, each OPRA starting with `SPX240202`, strike portions `04925000` / `04935000` / `04980000` / `04990000`.

Grep verification:

```
grep -rn "map_spx_strike_to_spy" packages/ apps/
```

Expected: zero matches.

Manual (Claude Code writes this checklist into the PR description):

- [ ] Test suite passes; pass count is ~97 (down from 104).
- [ ] Spot-check command produces SPX-prefixed OPRAs with correct strikes.
- [ ] `grep map_spx_strike_to_spy` over `packages/` and `apps/` returns zero hits.
- [ ] `condor.py` docstring on `condor_legs_for_row` no longer mentions SPY or mapping.
- [ ] `opra.py` module-header docstring reflects ORATS convention (root SPX for all SPX expirations; `expiryTod` distinguishes AM/PM).

## Out of Scope

- Adding `fetch_contracts` to `fetcher.py` — CR-004.
- Changing the orchestrator to use the new primitive — CR-005.
- Changing or removing `fetch_chain` — still useful for ad-hoc CLI; stays untouched.
- Cache-aware behavior on the future `fetch_contracts` primitive — design question for CR-004.
- Bounded concurrency — P1 tech debt, deferred until after CR-005.
- End-to-end verification — out of scope for CR-003 because Phase 4a was never end-to-end functional; CR-005 will restore and verify end-to-end working state.
- Any change to other strategy files (none exist yet).
- Adding new tests beyond preserving the existing set minus the deletions above. CR-004 will add new fetch-primitive tests; CR-005 will add end-to-end verification.

## Handoff Prompt for Claude Code

> Read `specs/CR-003-drop-spx-spy-mapping-condor.md` and implement it on a
> new branch `feat/CR-003-drop-spx-spy-mapping-condor` (branched from
> Main-Live). Follow the spec closely. The change deletes a function in
> `condor.py`, simplifies another, fixes one docstring in `opra.py`, and
> prunes tests accordingly — it should not require modifying any other
> file.
>
> Run the verification commands listed in the Verification section.
> Confirm the test pass count drops from 104 to roughly 97; if it drops
> by more or less than that, stop and explain before continuing.
>
> Commit in logical chunks with clear messages. Suggested chunking:
> (1) `condor.py` simplification, (2) `opra.py` docstring fix,
> (3) `test_condor.py` updates.
>
> When complete:
> 1. Push the branch to origin.
> 2. Create a PR against `Main-Live` using `gh pr create`. Title:
>    "CR-003: Drop SPX→SPY mapping in condor.py". Body should include a
>    summary, link to the spec, the commit list, and the manual
>    verification checklist above as Markdown checkboxes.
> 3. Print the PR URL.
>
> If you find something in the code that suggests the spec is wrong or
> ambiguous, stop and ask before changing scope.
