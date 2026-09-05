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

## Step 1 — full mode (as CR-AH ran in June), train only

Command: `PYTHONUNBUFFERED=1 apps/web/.venv/bin/python -u scripts/cr_ah_step4_analysis.py --universe-end 2026-06-05 --train-only --no-persist --cr-id CR-AL --structural-prob-mode full`. Log: `scripts/logs/cr_al_full_20260905_072039.log` (untracked). 342 s. Run row: `(UUID('2be34a8e-bf32-4272-b46d-b0a27aa4094b'), 'completed', datetime.datetime(2026, 9, 5, 14, 20, 40, 786839), datetime.datetime(2026, 9, 5, 14, 26, 21, 936478), "Step 4 complete [mode=full]: debit=87, credit=83, T_d=0.0, T_c=0.0, 342s; summary_d={'debit': 'INCONCLUSIVE', 'credit': 'untestable'}")`.

| Gate | Expected | Actual | Result |
|---|---|---|---|
| G1 stratified train selection | near 34 / mid 31 / far 40 | **near 34 / mid 31 / far 40** (holdout 16 / 19 / 10 selected then dropped) | PASS |
| G2 train clean counts vs June (±3) | credit ≈ 83, debit derived | **credit 83, debit 87** | PASS (note) |
| G3 train dates with `pattern_label` | ≥ 30 per structure | **debit 83 / 87, credit 79 / 83** | PASS |

G2 note: credit 83 equals June's recorded 83 train. For debit the vault only records total 110; with the same 23-date holdout as credit that is 87 train, matching exactly (the June note's "holdout n=18" is the near-band holdout cell, not the debit holdout total). The per-band tables below reproduce June's numbers to the cent (debit rth_touch 1.73 / 4.17, gap_touch 1.55 / 4.89; both thresholds 0.00), which confirms the universe pin reproduced June's train set.

Chosen thresholds: DEBIT 0.00, CREDIT 0.00 (same as June).

### Summary D (decision-9 read, full mode)

- **DEBIT:** pattern_match (stepping-stone) n=55, mean close P&L 2.96 vs no_match (mixed) n=25, 3.64 → diff **−0.69 pts**, bootstrap 95% CI **[−3.89, +2.19]** → **INCONCLUSIVE** (direction is "hurts", CI spans 0).
- **CREDIT:** pattern_match n=0, no_match n=79 → **untestable** — no credit-side pattern (touch-and-reject / slow-revert / overshoot-then-revert) occurs on any train date; the label set is stepping-stone + mixed only (Step 0 showed one slow-revert in 396 dates, and it is in the holdout).

Coverage effect (labeled vs unlabeled): debit labeled 80 → 3.17 pts vs unlabeled 4 → 0.64; credit labeled 28 → −4.36 vs unlabeled 2 → −5.43. Unlabeled cells are too small to read.

### Full output (Phases 1–6)

```

======================================================================
CR-AH Step 4 — Two-structure × two-axis analysis
  cr_id=CR-AL  structural-prob mode=full  train_only=True  no_persist=True  universe_end=2026-06-05  seed=20260905
======================================================================

Run ID: 2be34a8e-bf32-4272-b46d-b0a27aa4094b

----------------------------------------------------------------------
Phase 1: Loading signal dates and selecting clean subset...
  Universe pinned to trade_date <= 2026-06-05: 375/396 magnet-above dates kept.
  Loaded 374/375 signal entries (skipped 1).
  Stratified selection: 150 dates
  Selection by band/partition: {'far/holdout': 10, 'far/train': 40, 'mid/holdout': 19, 'mid/train': 31, 'near/holdout': 16, 'near/train': 34}
  --train-only: kept 105/150 train entries; holdout not read.

Filtering to A-bucket clean dates for each structure...
  Credit (target + target+10)...
    Credit clean: 83
  Debit (target + target-10)...
    Debit clean:  87

  Credit by band/partition: {'far/train': 25, 'mid/train': 27, 'near/train': 31}
  Debit  by band/partition: {'far/train': 28, 'mid/train': 27, 'near/train': 32}

----------------------------------------------------------------------
Phase 2: Collecting per-date trade data...
  Processing 87 debit dates...
  Processing 83 credit dates...

  Collected: debit=87, credit=83
  Settlement available: debit=84/87, credit=80/83
  Actionable touches: debit=62/87, credit=59/83

----------------------------------------------------------------------
Phase 3: Threshold sweep on TRAIN only...

  Chosen threshold — DEBIT: 0.00  CREDIT: 0.00

DEBIT — Threshold sweep (TRAIN only) [mode=full]:
       T  n_settled  fill_n  mean_pnl  win%   beat  chosen?
  ────────────────────────────────────────────────────────────
  0.00      84       87        3.05    64%    -0.28 ← CHOSEN
  0.05      84       87        2.91    64%    -0.42
  0.10      84       87        2.92    64%    -0.41
  0.15      84       87        2.85    64%    -0.48
  0.20      82       85        2.68    63%    -0.65

CREDIT — Threshold sweep (TRAIN only) [mode=full]:
       T  n_settled  fill_n  mean_pnl  win%   beat  chosen?
  ────────────────────────────────────────────────────────────
  0.00      30       32       -4.43    30%    -2.37 ← CHOSEN
  0.05      20       22       -4.60    30%    -2.55
  0.10      18       19       -5.28    22%    -3.23
  0.15      13       14       -5.75     8%    -3.70
  0.20       4        4       -7.11    25%    -5.06

----------------------------------------------------------------------
Phase 4: Full results (TRAIN only — holdout not read)

DEBIT — By distance band (train only, T=0.00) [mode=full]:
  band    part        n      pnl   win%  [lo–hi 95%]        base     beat
  ──────────────────────────────────────────────────────────────────────
  near    train      31     2.61    74%  [ 57%– 86%]     3.52    -0.91
  mid     train      27     2.81    63%  [ 44%– 78%]     2.61     0.20
  far     train      26     3.83    54%  [ 35%– 71%]     3.84    -0.02
  all     train      84     3.05    64%  [ 54%– 74%]     3.33    -0.28

CREDIT — By distance band (train only, T=0.00) [mode=full]:
  band    part        n      pnl   win%  [lo–hi 95%]        base     beat
  ──────────────────────────────────────────────────────────────────────
  near    train       7    -8.03     0%  [  0%– 35%]    -2.34    -5.69
  mid     train       7    -3.34    29%  [  8%– 64%]    -1.16    -2.18
  far     train      16    -3.33    44%  [ 23%– 67%]    -2.74    -0.59
  all     train      30    -4.43    30%  [ 17%– 48%]    -2.05    -2.37

DEBIT — By post-touch pattern (TRAIN only, T=0.00) [mode=full]:
  (n with pattern_label=83, n without=4)
  pattern                           n      pnl   win%  [lo–hi 95%]        base     beat
  ──────────────────────────────────────────────────────────────────────
  mixed                            25     3.64    68%  [ 48%– 83%]     2.62     1.02
  stepping-stone                   55     2.96    64%  [ 50%– 75%]     3.60    -0.64
  ──────────────────────────────────────────────────────────────────────
  (labeled)                        80     3.17    65%  [ 54%– 75%]     3.30    -0.12
  (unlabeled)                       4     0.64    50%  [ 15%– 85%]     4.01    -3.38

CREDIT — By post-touch pattern (TRAIN only, T=0.00) [mode=full]:
  (n with pattern_label=79, n without=4)
  pattern                           n      pnl   win%  [lo–hi 95%]        base     beat
  ──────────────────────────────────────────────────────────────────────
  mixed                             8    -6.98     0%  [  0%– 32%]    -4.02    -2.96
  stepping-stone                   20    -3.30    45%  [ 26%– 66%]    -0.97    -2.34
  ──────────────────────────────────────────────────────────────────────
  (labeled)                        28    -4.36    32%  [ 18%– 51%]    -1.85    -2.51
  (unlabeled)                       2    -5.43     0%  [  0%– 66%]    -5.89     0.46

DEBIT — Touch resolution breakdown (TRAIN only, T=0.00) [mode=full]:
  resolution                      n   touch_exit  close_pnl   base_close
  ─────────────────────────────────────────────────────────────────
  rth_touch                       37        1.73        4.17        4.30
  gap_touch                       25        1.55        4.89        4.48
  afterhours_touch_retraced        9           —        4.34        7.51
  no_touch                        16           —       -2.87       -2.87

DEBIT — Selection bias check (far band, decision #11) [mode=full]:
  far/all: n=40  clean: n=28 (mean σ=3.27)  dropped: n=12 (mean σ=3.04)
  ✓ No significant selection bias detected in far band.

CREDIT — Selection bias check (far band, decision #11) [mode=full]:
  far/all: n=40  clean: n=25 (mean σ=2.86)  dropped: n=15 (mean σ=3.77)
  ⚠ BIAS: clean far trades are significantly CLOSER than dropped far trades.
    The far-band result may be OPTIMISTICALLY SELECTED (closer to spot → easier trade).

======================================================================
SUMMARY READS A/B/C/D
======================================================================
[mode=full]

── Summary A: Debit near-band holdout ──
  holdout not read (--train-only).

── Summary B: Credit near-band holdout ──
  holdout not read (--train-only).

── Summary C: Structure crossover by distance (TRAIN, close P&L beat) ──
  near   debit_beat=  -0.91  credit_beat=  -5.69  → DEBIT leads by 4.78
  mid    debit_beat=   0.20  credit_beat=  -2.18  → DEBIT leads by 2.38
  far    debit_beat=  -0.02  credit_beat=  -0.59  → DEBIT leads by 0.58
  READ: Structure crossover = distance band where debit stops leading and credit starts.

── Summary D: Engine hypothesis (decision #13) ──
  DEBIT engine check:  pattern_match (n=55) pnl=2.96  vs no_match (n=25) pnl=3.64
  → Pattern filter HURTS debit (match performs WORSE). Engine rule may be wrong.
  DEBIT decision-9 read [mode=full]: match−no_match = -0.69 pts, bootstrap 95% CI [-3.89, +2.19] (1000 resamples, seed=20260905) → INCONCLUSIVE
  credit: too few pattern-labeled trades to score engine hypothesis (pattern_match n=0, no_match n=79).

----------------------------------------------------------------------
Phase 7: skipped (--no-persist) — bt_edge_backtest_results untouched.

======================================================================
Step 4 complete in 342s
======================================================================
```

## Step 2 — walk-forward mode (`before_date = trade_date`), train only

Command: same as Step 1 with `--structural-prob-mode walk-forward`. Log: `scripts/logs/cr_al_wf_20260905_072708.log` (untracked). 420 s. Run row: `(UUID('05ec3e81-5709-4baf-b6be-b30d05995943'), 'completed', datetime.datetime(2026, 9, 5, 14, 27, 9, 537272), datetime.datetime(2026, 9, 5, 14, 34, 8, 228883), "Step 4 complete [mode=walk-forward]: debit=87, credit=83, T_d=0.05, T_c=0.0, 419s; summary_d={'debit': 'INCONCLUSIVE', 'credit': 'untestable'}")`.

Selection and clean counts identical to Step 1 (G1 near 34 / mid 31 / far 40; credit 83 / debit 87) — the mode only changes the analogue pool behind `structural_prob` and `pattern_label`.

Chosen thresholds: **DEBIT 0.05** (was 0.00 in full mode and in June), CREDIT 0.00. The debit sweep is flat (beat −0.41 at 0.05 vs −0.43 at 0.00); the flip is noise-level but it is a visible consequence of the pool change.

### Summary D (decision-9 read, walk-forward)

- **DEBIT:** pattern_match (stepping-stone) n=33, 2.15 pts vs no_match (mixed) n=35, 3.04 → diff **−0.88 pts**, bootstrap 95% CI **[−3.56, +1.72]** → **INCONCLUSIVE** (direction "hurts", CI spans 0).
- **CREDIT:** pattern_match n=0 settled (one train date labeled `overshoot-then-revert`, no settlement price), no_match n=18 → **untestable**. The legacy Summary D line above the decision-9 read prints "Pattern filter adds value for credit" because the June logic compares a None mean (→ 0) against −2.41; that line is a pre-existing artifact of the n=0 case and is not a finding. The decision-9 read is authoritative.

### Side-by-side — full vs walk-forward (train only)

| | full | walk-forward |
|---|---|---|
| run_id | 2be34a8e | 05ec3e81 |
| debit train labeled / clean | 83 / 87 | 71 / 87 |
| credit train labeled / clean | 79 / 83 | 67 / 83 |
| debit threshold | 0.00 | 0.05 |
| debit Summary D: match n / no_match n | 55 / 25 | 33 / 35 |
| debit Summary D: mean match / no_match | 2.96 / 3.64 | 2.15 / 3.04 |
| debit Summary D: diff, CI, read | −0.69, [−3.89, +2.19], INCONCLUSIVE | −0.88, [−3.56, +1.72], INCONCLUSIVE |
| credit Summary D | match n=0 → untestable | match n=0 settled → untestable |
| Summary C far band | DEBIT leads by 0.58 | CREDIT leads by 0.87 |

Δ (wf − full) of the debit Summary D difference: −0.19 pts; both CIs span zero. **The engine-hypothesis read is the same in both modes: inconclusive for debit, untestable for credit.** Nothing here is lookahead-dependent in the decision-9 sense, because nothing was supported in either mode.

### How much the analogue pool moves the labels (scratchpad `cr_al_label_diff.py`, same 105 train-selected dates, `label_diff.json`)

- Label distribution, full: stepping-stone 64 / mixed 35 / None 6. Walk-forward: stepping-stone 43 / mixed 42 / None 19 / overshoot-then-revert 1.
- **41 of 105 train dates change `pattern_label` between modes.** Transitions: stepping-stone→mixed 21, stepping-stone→None 6, mixed→None 7, mixed→stepping-stone 6, mixed→overshoot-then-revert 1.
- By year: 2023 **21/33** changed, 2024 12/48, 2025 8/24. The 18 dates before 2023-09-01 are 15 mixed / 3 None under walk-forward — a thin pool defaults to "mixed".
- `touch_rate` (the structural-probability input to `edge`): wf − full mean −0.014, median +0.009, range −0.34 … +0.27; |Δ| > 0.05 on **59 of 105** dates.

The lookahead is real and large at the per-date level (39% of labels, 56% of touch rates move by more than 5 pp) even though the aggregate Summary D read does not change. Any per-date use of `pattern_label` or `structural_prob` from the June runs (CR-AH band cells, CR-AI DTE result) should be considered full-corpus numbers until CR-AM re-runs them walk-forward.

### Full output (Phases 3–6)

```
  Chosen threshold — DEBIT: 0.05  CREDIT: 0.00

DEBIT — Threshold sweep (TRAIN only) [mode=walk-forward]:
       T  n_settled  fill_n  mean_pnl  win%   beat  chosen?
  ────────────────────────────────────────────────────────────
  0.00      83       86        2.90    64%    -0.43
  0.05      83       86        2.92    64%    -0.41 ← CHOSEN
  0.10      81       84        2.86    63%    -0.47
  0.15      81       84        2.89    63%    -0.44
  0.20      78       81        2.63    62%    -0.70

CREDIT — Threshold sweep (TRAIN only) [mode=walk-forward]:
       T  n_settled  fill_n  mean_pnl  win%   beat  chosen?
  ────────────────────────────────────────────────────────────
  0.00      30       32       -3.82    37%    -1.77 ← CHOSEN
  0.05      22       23       -4.54    27%    -2.49
  0.10      15       15       -4.76    20%    -2.71
  0.15       9        9       -6.95    22%    -4.90
  0.20       5        5       -4.56    40%    -2.51

----------------------------------------------------------------------
Phase 4: Full results (TRAIN only — holdout not read)

DEBIT — By distance band (train only, T=0.05) [mode=walk-forward]:
  band    part        n      pnl   win%  [lo–hi 95%]        base     beat
  ──────────────────────────────────────────────────────────────────────
  near    train      30     2.61    73%  [ 56%– 86%]     3.52    -0.92
  mid     train      27     2.37    63%  [ 44%– 78%]     2.61    -0.24
  far     train      26     3.85    54%  [ 35%– 71%]     3.84     0.00
  all     train      83     2.92    64%  [ 53%– 73%]     3.33    -0.41

CREDIT — By distance band (train only, T=0.00) [mode=walk-forward]:
  band    part        n      pnl   win%  [lo–hi 95%]        base     beat
  ──────────────────────────────────────────────────────────────────────
  near    train      10    -7.97     0%  [  0%– 28%]    -2.34    -5.63
  mid     train       7    -1.54    43%  [ 16%– 75%]    -1.16    -0.38
  far     train      13    -1.87    62%  [ 36%– 82%]    -2.74     0.87
  all     train      30    -3.82    37%  [ 22%– 54%]    -2.05    -1.77

DEBIT — By post-touch pattern (TRAIN only, T=0.05) [mode=walk-forward]:
  (n with pattern_label=71, n without=16)
  pattern                           n      pnl   win%  [lo–hi 95%]        base     beat
  ──────────────────────────────────────────────────────────────────────
  mixed                            34     3.00    62%  [ 45%– 76%]     3.85    -0.85
  overshoot-then-revert             1     4.35   100%  [ 21%–100%]   -26.90    31.25
  stepping-stone                   33     2.15    58%  [ 41%– 73%]     3.76    -1.61
  ──────────────────────────────────────────────────────────────────────
  (labeled)                        68     2.61    60%  [ 48%– 71%]     3.35    -0.75
  (unlabeled)                      15     4.33    80%  [ 55%– 93%]     3.23     1.10

CREDIT — By post-touch pattern (TRAIN only, T=0.00) [mode=walk-forward]:
  (n with pattern_label=67, n without=16)
  pattern                           n      pnl   win%  [lo–hi 95%]        base     beat
  ──────────────────────────────────────────────────────────────────────
  mixed                             5    -2.68    40%  [ 12%– 77%]    -2.02    -0.66
  overshoot-then-revert             0        —      —  [   —–   —]        —        —
  stepping-stone                   13    -2.30    54%  [ 29%– 77%]    -0.57    -1.73
  ──────────────────────────────────────────────────────────────────────
  (labeled)                        18    -2.41    50%  [ 29%– 71%]    -1.33    -1.07
  (unlabeled)                      12    -5.95    17%  [  5%– 45%]    -4.93    -1.02

DEBIT — Touch resolution breakdown (TRAIN only, T=0.05) [mode=walk-forward]:
  resolution                      n   touch_exit  close_pnl   base_close
  ─────────────────────────────────────────────────────────────────
  rth_touch                       37        1.74        4.18        4.30
  gap_touch                       25        1.17        4.47        4.48
  afterhours_touch_retraced        9           —        4.34        7.51
  no_touch                        16           —       -2.87       -2.87

DEBIT — Selection bias check (far band, decision #11) [mode=walk-forward]:
  far/all: n=40  clean: n=28 (mean σ=3.27)  dropped: n=12 (mean σ=3.04)
  ✓ No significant selection bias detected in far band.

CREDIT — Selection bias check (far band, decision #11) [mode=walk-forward]:
  far/all: n=40  clean: n=25 (mean σ=2.86)  dropped: n=15 (mean σ=3.77)
  ⚠ BIAS: clean far trades are significantly CLOSER than dropped far trades.
    The far-band result may be OPTIMISTICALLY SELECTED (closer to spot → easier trade).

======================================================================
SUMMARY READS A/B/C/D
======================================================================
[mode=walk-forward]

── Summary A: Debit near-band holdout ──
  holdout not read (--train-only).

── Summary B: Credit near-band holdout ──
  holdout not read (--train-only).

── Summary C: Structure crossover by distance (TRAIN, close P&L beat) ──
  near   debit_beat=  -0.92  credit_beat=  -5.63  → DEBIT leads by 4.72
  mid    debit_beat=  -0.24  credit_beat=  -0.38  → DEBIT leads by 0.14
  far    debit_beat=   0.00  credit_beat=   0.87  → CREDIT leads by 0.87
  READ: Structure crossover = distance band where debit stops leading and credit starts.

── Summary D: Engine hypothesis (decision #13) ──
  DEBIT engine check:  pattern_match (n=33) pnl=2.15  vs no_match (n=35) pnl=3.04
  → Pattern filter HURTS debit (match performs WORSE). Engine rule may be wrong.
  DEBIT decision-9 read [mode=walk-forward]: match−no_match = -0.88 pts, bootstrap 95% CI [-3.56, +1.72] (1000 resamples, seed=20260905) → INCONCLUSIVE
  CREDIT engine check:  pattern_match (n=0) pnl=—  vs no_match (n=18) pnl=-2.41
  → Pattern filter adds value for credit (pattern_match outperforms).
  CREDIT decision-9 read [mode=walk-forward]: match−no_match = — pts, bootstrap 95% CI [—, —] (1000 resamples, seed=20260905) → untestable

----------------------------------------------------------------------
Phase 7: skipped (--no-persist) — bt_edge_backtest_results untouched.

======================================================================
Step 4 complete in 420s
======================================================================
```
