# CR-AL — Engine hypothesis re-test (pattern axis, train only, full vs walk-forward)

> Authority: vault session note `Dash/sessions/2026-09-05 - CR-AL — Engine Hypothesis Re-test.md`
> Branch: `feat/CR-AL-engine-hypothesis-retest` (off `origin/main` 66d1257, the CR-AK merge)
> Scope: `before_date` / `allow_lookahead` on `compute_structural_probability` + caller updates + tests; flags on `cr_ah_step4_analysis.py`; two read-mostly analysis runs. No data writes beyond `bt_backfill_runs`.
> Mode: unattended; halts at the first STOP gate that misses.

## Problem

CR-AH decision #13 — does the post-touch `pattern_label` split debit/credit profitability the way `post_touch_qualification.py` assumes — was recorded UNTESTABLE on 2026-06-09: 0/216 labels. The cause given was ~37% t15 coverage. `[Certain]` The real cause was 0% coverage: the canonical version had no post-touch columns at all until CR-AK (2026-09-05). `[[engine-qualification-untested-needs-corpus]]` is mis-diagnosed; its "primary path — expand coverage" was already satisfied by the old version and is now satisfied at canonical (387/387 touched rows labeled).

The live engine has been gating direction on this rule since CR-025 with no empirical support. CR-AL is the first test.

## What CR-AH Step 4 already does

`scripts/cr_ah_step4_analysis.py` (June run `6be26595`, 172 s): stratified 50/band stride over all magnet-above dates → A-bucket clean filter → per-date entry crawl / touch / settlement → threshold sweep on train → by-band, by-pattern, Summary D. The pattern axis is `print_by_pattern` + Summary D, both **train only**. Persists band cells to `bt_edge_backtest_results`.

Two things the script does that CR-AL must control:

1. **It reads the holdout.** By-band Phase 4 and Summaries A/B print holdout. Re-running as-is is a second holdout read. CR-AL is a train-only question.
2. **Selection depends on the universe.** `stride_select` picks 50/band from *all* magnet dates. The corpus now has 20 more magnet-above dates (all post-split). Re-running on the wider universe changes which train dates are selected. To reproduce June's exact train set: restrict the universe to `trade_date ≤ 2026-06-05` before striding.

## Lookahead finding (new, surfaced during spec)

`[Certain]` `compute_structural_probability` never passes `before_date`, so `structural_prob` and `pattern_label` for a historical anchor are computed against the whole corpus including future/holdout dates. See [[structural-prob-analogue-lookahead]]. CR-AL measures it rather than fixing it: two passes, `full` (as CR-AH ran) and `walk-forward` (`before_date = trade_date`), same dates, side by side.

## Locked decisions

| # | Decision | Value |
|---|---|---|
| 1 | Universe | `--universe-end 2026-06-05` on `load_signal_entries`. Gate: stratified train selection must equal June exactly — near 34 / mid 31 / far 40. |
| 2 | Partition | `--train-only`: drop holdout entries after selection, before Phase 2 collection. No holdout by-band, no Summary A/B. Holdout remains unread. |
| 3 | Persistence | `--no-persist`: skip Phase 7. `bt_edge_backtest_results` stays as CR-AH left it (CR-Y reads it). |
| 4 | Run record | `--cr-id CR-AL` on `backfill_run`. Default stays `CR-AH`. |
| 5 | Structural-prob mode | `--structural-prob-mode {full,walk-forward}`. `walk-forward` passes `before_date = trade_date.isoformat()`. `full` passes `allow_lookahead=True`. Per ADR [[2026-09-05 - Historical Structural-Probability Calls Must Declare a Cutoff]], `compute_structural_probability` **raises** when `exclude_date` is set without `before_date` unless `allow_lookahead=True`; live callers (no `exclude_date`) unchanged. All existing callers updated in the same commit. |
| 6 | Runs | Two runs, same flags except mode. `full` first (comparable to June), then `walk-forward`. |
| 7 | Extra output | `print_by_pattern` also prints the **labeled vs unlabeled** cell (mean close P&L, win%, Wilson) so the coverage effect is visible. Summary D adds a 1,000-resample bootstrap 95% CI on `mean(pattern_match) − mean(no_match)`. Seed 20260905. |
| 8 | Thresholds | Sweep runs as before (train only). Report the chosen threshold; June chose 0.00 for both structures. |
| 9 | Pre-registered read of Summary D | Per structure, at chosen threshold, train only: **supported** if match − no_match > +0.3 pts AND bootstrap CI excludes 0; **hurts** if < −0.3 AND CI excludes 0; otherwise **inconclusive**. Report both modes; a result that holds in `full` but not `walk-forward` is reported as lookahead-dependent, not as supported. |
| 10 | Not in scope | Wilson-floor sweep (open question item 3); walk-forward re-run of the CR-AH band/edge result; any change to `post_touch_qualification.py`; the holdout. |

## Expected values (gates)

| Gate | Expected | On miss |
|---|---|---|
| G0.1 — PR #43 merged; `main` HEAD contains CR-AK | yes | STOP |
| G0.2 — canonical touched rows with `position_t15_post_touch` non-null | 387 minus corpus-end NULLs (expect ≥ 370) | STOP if < 350 |
| G0.3 — label distribution across all 375 canonical magnet-above dates (Step 0 diagnostic, `full` mode): fraction with `pattern_label ≠ None` | ≥ 0.50 | note; if < 0.20 the run is still worth doing but flag as thin |
| G1 — stratified train selection | near 34 / mid 31 / far 40 | STOP |
| G2 — debit/credit train clean counts | within ±3 of June (June total 110 / 106 incl. holdout; train portion to be recorded from June's run log or spec) | note, continue |
| G3 — `full` mode: train dates with `pattern_label` | ≥ 30 per structure | if < 30, run still completes; Summary D is reported as underpowered |
| G4 — 420 shared tests + 105 backtest tests pass after the `before_date` / `allow_lookahead` change and caller updates | pass | STOP |
| G5 — caller inventory complete: every `compute_structural_probability` call site either untouched-live, patched, or listed as deferred | yes | STOP |


## Step 0 findings (read-only, 2026-09-05 ~06:30 UTC)

Interpreter: `apps/web/.venv/bin/python` (native arm64) for scripts and DB reads; Rosetta repo venv (`arch -x86_64 .venv/bin/python -m pytest`) for the test suites — it collects 420 (`packages/shared/tests`) + 105 (`packages/shared/backtest/tests`), matching G4's numbers. The apps venv has no pytest.

| Gate | Expected | Actual | Result |
|---|---|---|---|
| G0.1 PR #43 merged; `main` HEAD contains CR-AK | yes | `origin/main` = 66d1257 = merge of PR #43; branch cut from it | PASS |
| G0.2 canonical touched rows with `position_t15_post_touch` | ≥ 370 (STOP < 350) | **386** (387 touched − 1 corpus-end t15 NULL) | PASS |
| G0.3 labeled fraction across canonical magnet-above dates (full mode) | ≥ 0.50 | **0.934** (370 / 396) | PASS |

### G0.3 detail — `compute_structural_probability` in full mode (current code, `exclude_date` only), `regime_kind='magnet-above'`

Universe: `bt_daily_features` at canonical with `regime_at_classification = 'magnet-above'` = **396** dates (the vault note said 375; CR-AJ's June–September gap fill added 21 post-split dates — same count on the active view). Script: scratchpad `cr_al_step0.py`; per-date JSON `step0_labels_full.json`.

| Partition | n | stepping-stone | mixed | slow-revert | None | labeled | strict | pooled-fallback | insufficient | 0DTE-insufficient |
|---|---|---|---|---|---|---|---|---|---|---|
| train (≤ 2025-08-12) | 263 | 157 | 88 | 0 | 18 | 245 (0.932) | 236 | 9 | 16 | 2 |
| holdout | 133 | 60 | 64 | 1 | 8 | 125 (0.940) | 111 | 14 | 7 | 1 |
| **all** | **396** | 217 | 152 | 1 | 26 | 370 (0.934) | 347 | 23 | 23 | 3 |

Observations (diagnostic only — holdout P&L not read):

- Coverage is no longer the constraint. The label set is almost entirely `stepping-stone` (55%) and `mixed` (38%); the credit-side patterns (`touch-and-reject`, `slow-revert`, `overshoot-then-revert`) appear once in 396. Under `post_touch_qualification.py` that means the credit gate can essentially never open on magnet-above anchors, and Summary D's credit read will rest on 0–1 `pattern_match` rows — expect "too few pattern-labeled trades" for credit regardless of mode.
- `touch-and-pin` never appears; the debit side is carried entirely by `stepping-stone`.

### G2 reference — June train clean counts (derived)

The June run log is not in the vault; the CR-AH spec records the A-bucket clean sample as near/train 30, mid/train 27, far/train 26 = **83 train** (credit legs), and the Step 4 note gives debit 110 / credit 106 including holdout with debit holdout n = 18. Derived: **credit train ≈ 83, debit train ≈ 92**. G2 tolerance ±3 applies to these derived figures; G2 is note-and-continue.

### Caller inventory (pre-change grep, `compute_structural_probability(` call sites)

| File:line | Kind | Plan for commit 3 |
|---|---|---|
| `apps/web/modules/TodaySetup/routes.py:287` | `/api/setup/proposals` — passes `exclude_date=trade_date` for every requested date, today included | pass `before_date=trade_date.isoformat()` unconditionally: for today it excludes nothing that exists yet (live behaviour preserved); for a browsed past date it is the ADR's walk-forward cutoff |
| `scripts/cr_ah_step4_analysis.py:569` | CR-AH Step 4 | commit 4: `--structural-prob-mode` (walk-forward → `before_date`; full → `allow_lookahead=True`) |
| `scripts/cr_z_step4b_k_display.py:41` | CR-031 k-display verifier, `exclude_date=anchor_date` | trivial: add `before_date=anchor_date` |
| `scripts/cr_ah_posthoc_limit_entry.py:317` | **untracked** file (never committed), `exclude_date` only | not in the repo; listed under "Callers deferred to CR-AM" — will raise if run, by design |
| `scripts/cr_z_step4_sanity.py:3` | docstring mention only, no call | untouched |
| `packages/shared/tests/test_probability.py:6` | docstring mention only | untouched; new tests added here |
| `packages/shared/knn_coherence.py` | calls `rank_analogues` directly with its own `before_date` guard | not a caller; untouched |
