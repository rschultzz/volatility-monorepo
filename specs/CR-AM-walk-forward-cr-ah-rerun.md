# CR-AM — Walk-forward re-run of CR-AH Step 4; holdout split 2026-06-05; leg capture

> Authority: vault session note `Dash/sessions/2026-09-05 - CR-AM — Walk-Forward Re-run of CR-AH.md` and ADR `2026-09-05 - Holdout Split Moves to 2026-06-05`
> Branch: `feat/CR-AM-walk-forward-cr-ah-rerun` (off `origin/main` 51120c9, the CR-AL merge)
> Scope: `--split-date` flag + `cr_id` on the results INSERT; one walk-forward run persisted as `cr_id='CR-AM'`; holdout leg-quote capture with no read.
> Mode: unattended; halts at the first STOP gate that misses.

## Problem

CR-AH Step 4's verdicts — credit refuted [ROBUST], hold-to-close beats touch-exit [LIKELY], debit close-path edge [DIRECTIONAL] — rest on entries gated by `structural_prob` computed with analogue lookahead. CR-AL measured the leak on the train set: 59/105 dates move touch_rate by > 5 pp walk-forward, 41/105 change pattern label, aggregate reads unchanged. The band-level P&L verdicts have not been re-derived walk-forward. Until they are, "debit has edge" is a full-corpus number.

## Scope

**In:** CR-AH Step 4 re-run, walk-forward, on the ≤ 2026-06-05 universe with no holdout partition (everything is train). Persist as the new reference. Capture (not read) leg quotes for the post-2026-06-05 holdout dates.

**Out:** Any holdout P&L. CR-AI Stage 2 re-run (conditional — see Step 0). The direction gate (stays live per the CR-AL decision). Pattern mining.

## Locked decisions

| # | Decision | Value |
|---|---|---|
| 1 | Universe | `--universe-end 2026-06-05`, same stride as June → 150 selected, expect clean debit 110 / credit 106 (June's totals; identical selection because same universe). |
| 2 | Partition | `--split-date 2026-06-05` (new flag; default changes from hardcoded 2025-08-12 to this). With universe-end = split-date, `partition` is `train` for every row; holdout branches print "holdout: none (split = universe end)". `--train-only` implied. |
| 3 | Mode | `--structural-prob-mode walk-forward` (default after CR-AL). `full` is not run here; June's numbers are the full-corpus column. |
| 4 | Threshold sweep | On the whole 150-set (all train). Report the chosen threshold per structure; June chose 0.00 / 0.00. |
| 5 | Persistence | `--persist` (invert CR-AL's `--no-persist` default? No — keep flags additive: `--no-persist` absent means persist). `bt_edge_backtest_results` INSERT must write `cr_id = 'CR-AM'` (add `cr_id` to the INSERT using `--cr-id`). CR-AH rows remain; CR-Y / any reader should prefer the latest `cr_id` — note this in the spec for CR-Y. |
| 6 | Comparison | Claude Code composes, per structure × band, a three-column table: June train (full) / June holdout (full) / CR-AM all-train (walk-forward), for mean close P&L, win%, Wilson, beat-vs-baseline. Source for June columns: `specs/CR-AH-per-structure-edge-backtest.md` Step 4 appendix and `bt_edge_backtest_results` where `cr_id='CR-AH'`. |
| 7 | Pre-registered restatement of June verdicts | For each: **holds** if the walk-forward sign matches June and |beat| ≥ 50% of June's magnitude; **weakens** if sign matches but magnitude < 50%; **reverses** if sign flips. Applied to: (a) credit close-path beat, all bands; (b) debit hold-to-close vs touch-exit; (c) debit close-path beat by band; (d) mid-band negativity (cross-ref [[mid-band-drift-targets-lack-edge]]). |
| 8 | CR-AI | Step 0 greps `cr_ai_stage2_backfill.py` and its imports for `compute_structural_probability` / `structural_prob`. **Absent** → CR-AI is lookahead-free (open-price entry, no edge gate); record and do not re-run. **Present** → STOP, surface. |
| 9 | Holdout capture | For every canonical magnet-above date > 2026-06-05 with a computed or pending outcome: fetch entry-day RTH quotes and expiry-day settlement-window quotes for the debit legs (target−10 / target) and credit legs (target / target+10) at 15 DTE into `orats_options_minute` via the existing fetcher. **No P&L is computed or printed.** Log only: dates attempted, legs fetched, 404s. If `cr_ah_step2_stratified_backfill.py` cannot be restricted to an explicit date list without emitting analysis, write a minimal `scripts/cr_am_holdout_leg_capture.py` that reuses its fetch path. |
| 10 | Holdout read | Not in this CR. Requires a separate pre-registered spec with a minimum n. |

## Expected values (gates)

| Gate | Expected | On miss |
|---|---|---|
| G0.1 — `main` HEAD ≥ `51120c9` (CR-AL merged) | yes | STOP |
| G0.2 — CR-AI structural-prob grep | absent | STOP if present |
| G0.3 — canonical magnet-above dates > 2026-06-05 with outcome_status IN (computed, pending_history) | ~20 (18 computed + 6 pending at CR-AJ; some pending may have matured) | note |
| G1 — stratified selection | near 50 / mid 50 / far 50 = 150 (all train) | STOP |
| G2 — clean counts | debit 110 / credit 106 | STOP if either differs by > 3 |
| G3 — `compute_structural_probability` calls in walk-forward mode raise nothing; every call passed `before_date` | yes | STOP |
| G4 — per-band n matches June's train+holdout combined per band (June: near 34+16, mid 31+19, far 40+10 selected; clean subsets per spec) | ±0 | STOP |
| G5 — capture step: 0 fetch exceptions other than ORATS 404 | yes | note; 404s listed |
| G6 — no holdout P&L printed anywhere in the logs | grep the two logs for the holdout dates + "pnl" | STOP and redact before commit if found |


## Step 0 findings (read-only, 2026-09-05 ~08:10 UTC)

Interpreter: `apps/web/.venv/bin/python` for DB reads and runs; Rosetta repo venv for tests.

| Gate | Expected | Actual | Result |
|---|---|---|---|
| G0.1 `main` HEAD ≥ 51120c9 | yes | `origin/main` = 51120c9 (PR #44 merge); branch cut from it | PASS |
| G0.2 `compute_structural_probability` reachable from `cr_ai_stage2_backfill.py` | absent | **absent** (detail below) | PASS |
| G0.3 canonical magnet-above dates > 2026-06-05, computed or pending | ~20 | **21** (15 computed, 6 pending) | note |

### G0.2 detail

`grep -n structural scripts/cr_ai_stage2_backfill.py` → no hits. Its `packages/` imports: `backfill_safety`, `backtest.models`, `canonical_version`, `day_features`, `gex_landscape`, `options_cache.fetcher`, `options_cache.http_client`, `options_cache.opra`. Hits for "structural" in those: `backtest/models.py:33,49` (`structural_prob` / `edge` dataclass fields on `TradeResult`, no computation) and `gex_landscape.py` (comments about structural walls; line 213 mentions "probability" in a comment). None imports `packages.shared.probability`; `gex_landscape.py` imports only numpy / pandas / scipy. CR-AI Stage 2 is lookahead-free by construction (open-price entry, no edge gate) — **not re-run**, per ADR consequence 2.

### G0.3 detail — post-split magnet-above dates (the new holdout stream)

`bt_daily_outcomes` has `regime_kind_at_classification`; the list below joins `bt_daily_features.regime_at_classification = 'magnet-above'` at canonical.

computed (15): 2026-06-12, 06-16, 06-17, 06-18, 07-01, 07-02, 07-08, 07-13, 07-15, 07-16, 08-03, 08-07, 08-10, 08-26, 08-28.
pending_history (6): 2026-08-12, 08-13, 08-18, 08-27, 09-03, 09-04.

`orats_options_minute` coverage after 2026-06-05 today: 06-05 (4 symbols), 06-08 (8), 06-16 (14), 07-01 (8), 07-07 (26), 08-13 (8), 08-14 (8) — none of it is the 15-DTE leg set for these dates. Capture is required for the holdout to exist.

### June columns (CR-AH run `6be26595`, full-corpus, split 2025-08-12, T = 0.00 both)

Source: `bt_edge_backtest_results WHERE cr_id='CR-AH'` (16 rows; n = settled) and the CR-AH spec Step 4 appendix. Mean close P&L in points; beat = mean − baseline.

| structure | band | train n / pnl / win% / Wilson / base / beat | holdout n / pnl / win% / Wilson / base / beat |
|---|---|---|---|
| debit | near | 31 / +2.61 / 74% / [57–86] / +3.52 / −0.91 | 8 / +0.73 / 62% / [31–86] / +0.24 / +0.49 |
| debit | mid | 27 / +2.81 / 63% / [44–78] / +2.61 / +0.20 | 9 / +1.00 / 56% / [27–81] / −4.09 / +5.09 |
| debit | far | 26 / +3.83 / 54% / [35–71] / +3.84 / −0.02 | 5 / +1.42 / 40% / [12–77] / +1.42 / 0.00 |
| debit | all | 84 / +3.05 / 64% / [54–74] / +3.33 / −0.28 | 22 / +1.00 / 55% / [35–73] / −1.26 / +2.26 |
| credit | near | 7 / −8.03 / 0% / [0–35] / −2.34 / −5.69 | 2 / −3.25 / 50% / [9–91] / −2.51 / −0.74 |
| credit | mid | 6 / −3.01 / 33% / [10–70] / −1.16 / −1.85 | 3 / −2.07 / 67% / [21–94] / +0.01 / −2.07 |
| credit | far | 15 / −3.06 / 47% / [25–70] / −2.74 / −0.32 | 2 / −3.33 / 50% / [9–91] / −0.95 / −2.37 |
| credit | all | 28 / −4.29 / 32% / [18–51] / −2.05 / −2.24 | 7 / −2.76 / 57% / [25–84] / −1.13 / −1.64 |

June debit touch resolution (train): rth_touch n=37 touch-exit +1.73 / close +4.17; gap_touch n=25 +1.55 / +4.89; afterhours_retraced n=9 — / +4.34; no_touch n=16 — / −2.87. June verdicts to restate (decision 7): (a) credit close-path beat negative in all bands; (b) debit hold-to-close > touch-exit (+4.17–4.89 vs +1.55–1.73); (c) debit close-path beat by band (near −0.91 / mid +0.20 / far −0.02 train; holdout +0.49 / +5.09 / 0.00); (d) mid-band negativity — CR-AI Stage 2 found mid negative at every DTE; CR-AH's own mid/train beat was the one positive debit band (+0.20).

### Decision 9 plan — holdout leg capture

`scripts/cr_ah_step2_stratified_backfill.py` cannot be pointed at an explicit date list: `main()` re-derives the universe via `load_signal_dates_with_sigma` + stride, and `backfill_date` also runs ES touch detection, fetches a touch window and prints touch resolution (a partial read of the outcome path). Its fetch primitive is `fetch_option_bars(opras, start_pt, end_pt, source="historical_backfill", record_empty_windows=True)`, catching `OratsPermanentError` for 404s. Plan: a minimal `scripts/cr_am_holdout_leg_capture.py` that (1) routes `DATABASE_URL` to `BACKFILL_DATABASE_URL` before importing `options_cache` (same pattern as the CR-AH scripts), (2) for each G0.3 date takes the target from `orats_gex_landscape.walls` via `pick_drift_target`, expiry = `nth_business_day(+15)` (holiday-aware, imported from `cr_ah_step4_analysis`), legs = `round5(target)` −10 / +0 / +10 calls, (3) fetches the entry-day RTH window (06:30–13:00 PT) and the expiry settlement window (12:50–13:00 PT) via `fetch_option_bars` under `backfill_run(cr_id='CR-AM-capture')`, (4) logs date / target / legs / bars_written / cache_hits / 404 only. It also logs the CR-AH-style target (`_materialize_payload(...)['regime']['drift_target']`, which is what Step 4 uses for the same dates) and, where the two round to different strikes, fetches the union so the captured set is usable by either convention. No `net_price`, `payoff`, or P&L anywhere in the script.

## Step 1 — walk-forward run on the ≤ 2026-06-05 universe (all train), persisted as `cr_id='CR-AM'`

Command: `PYTHONUNBUFFERED=1 apps/web/.venv/bin/python -u scripts/cr_ah_step4_analysis.py --universe-end 2026-06-05 --split-date 2026-06-05 --cr-id CR-AM --structural-prob-mode walk-forward`. Log: `scripts/logs/cr_am_step4_wf_20260905_085254.log` (untracked). 339 s. Run row: `(UUID('c52daf00-ad7c-4c62-a1b1-5d8fcbada9c2'), 'completed', datetime.datetime(2026, 9, 5, 15, 52, 55, 700903), datetime.datetime(2026, 9, 5, 15, 58, 34, 246277), "Step 4 complete [mode=walk-forward]: debit=110, credit=106, T_d=0.0, T_c=0.0, 339s; summary_d={'debit': 'INCONCLUSIVE', 'credit': 'untestable'}")`. Persisted: 8 rows in `bt_edge_backtest_results` with `cr_id='CR-AM'` (train × {near, mid, far, all} × {debit, credit}); the 16 CR-AH rows untouched.

| Gate | Expected | Actual | Result |
|---|---|---|---|
| G1 stratified selection | near 50 / mid 50 / far 50, all train | **50 / 50 / 50**, `holdout: none (split = universe end)` | PASS |
| G2 clean counts | debit 110 / credit 106 (±3) | **debit 110 / credit 106** | PASS |
| G3 walk-forward calls raise nothing; every call passed `before_date` | yes | run completed; mode `walk-forward` passes `before_date=trade_date` on every call (no `allow_lookahead` path taken) | PASS |
| G4 per-band n = June train + holdout | selection 34+16 / 31+19 / 40+10 = 50/50/50; clean per spec | selection exact. Debit clean **41 / 36 / 33** = the CR-AI record exactly. Credit clean **40 / 36 / 30** = the split the same script produced on June's universe in CR-AL (train 31/27/25 + former holdout 9/9/5); the CR-AH spec's Step-2.5 prose table says 30/27/26 train — one near↔far label off in that hand-written table, not in the run (total 106 exact) | PASS (note) |
| G6 no holdout P&L in the run log | none | grep of all 21 post-split dates over the log: **0 hits** | PASS |

Chosen thresholds: DEBIT 0.00, CREDIT 0.00 (same as June). Debit sweep beat is now positive at T ≤ 0.15 (+0.32 at 0.00) where June's train was −0.28; the credit sweep is monotone negative as in June.

### Three-column comparison (decision 6) — mean close P&L, win%, Wilson, baseline, beat; T = 0.00

| structure | band | June train (full) | June holdout (full) | CR-AM all-train (walk-forward) |
|---|---|---|---|---|
| debit | near | n=31 / +2.61 / 74% [57–86] / base +3.52 / beat -0.91 | n=8 / +0.73 / 62% [31–86] / base +0.24 / beat +0.49 | n=39 / +2.01 / 69% [54–81] / base +2.12 / beat -0.11 |
| debit | mid | n=27 / +2.81 / 63% [44–78] / base +2.61 / beat +0.20 | n=9 / +1.00 / 56% [27–81] / base -4.09 / beat +5.09 | n=36 / +2.03 / 61% [45–75] / base +0.94 / beat +1.09 |
| debit | far | n=26 / +3.83 / 54% [35–71] / base +3.84 / beat -0.02 | n=5 / +1.42 / 40% [12–77] / base +1.42 / beat +0.00 | n=31 / +3.44 / 52% [35–68] / base +3.45 / beat -0.01 |
| debit | all | n=84 / +3.05 / 64% [54–74] / base +3.33 / beat -0.28 | n=22 / +1.00 / 55% [35–73] / base -1.26 / beat +2.26 | n=106 / +2.43 / 61% [52–70] / base +2.11 / beat +0.32 |
| credit | near | n=7 / -8.03 / 0% [0–35] / base -2.34 / beat -5.69 | n=2 / -3.25 / 50% [9–91] / base -2.51 / beat -0.74 | n=11 / -8.09 / 0% [0–26] / base -2.79 / beat -5.30 |
| credit | mid | n=6 / -3.01 / 33% [10–70] / base -1.16 / beat -1.85 | n=3 / -2.07 / 67% [21–94] / base +0.01 / beat -2.07 | n=11 / -1.25 / 55% [28–79] / base -0.87 / beat -0.38 |
| credit | far | n=15 / -3.06 / 47% [25–70] / base -2.74 / beat -0.32 | n=2 / -3.33 / 50% [9–91] / base -0.95 / beat -2.37 | n=15 / -2.01 / 60% [36–80] / base -2.42 / beat +0.41 |
| credit | all | n=28 / -4.29 / 32% [18–51] / base -2.05 / beat -2.24 | n=7 / -2.76 / 57% [25–84] / base -1.12 / beat -1.64 | n=37 / -3.59 / 41% [26–57] / base -2.01 / beat -1.58 |


June columns from `bt_edge_backtest_results` `cr_id='CR-AH'` (n = settled); CR-AM column from `cr_id='CR-AM'`. June's split was 2025-08-12; CR-AM has no holdout, so its column is the union of June's train and holdout dates re-run walk-forward.

### Data-quality finding surfaced by the restatement (new; affects June identically)

Per-trade range check at T = 0.00 (scratchpad `cr_am_range_diag.py`, `range_diag.json`): a 10-wide vertical bought/sold for X has close P&L in [−10, +10], touch-exit P&L in the same range, and a debit fill must have negative net credit. **26 / 110 debit and 20 / 106 credit trades violate at least one of these.** By flag — debit: baseline > width 22, fill sign 8, touch-exit > width 8, close > width 5; credit: baseline > width 20, fill sign 1, close > width 1. The dominant offender is the **baseline** (first-quoted-minute entry, decision #7): at 06:30 PT one leg's mid is often inverted or stale, giving spread values like +42.25 or −26.80. Since every "beat" is `mean − baseline`, the beat column in June and in CR-AM is contaminated in the same way. Gap-touch touch-exit (5 dates > width, e.g. 2026-04-07 at +78.10) is the second offender. The persisted CR-AM rows are as-run (no filter), per the spec; the sensitivity below excludes flagged trades and is a post-hoc data-validity view, **not** a pre-registered read.

| structure | band | as run: n / close / win% / base / beat | excluding out-of-range trades: n / close / win% / base / beat |
|---|---|---|---|
| debit | near | 39 / +2.01 / 69% / +2.12 / −0.11 | 30 / +2.54 / 73% / +2.58 / −0.04 |
| debit | mid | 36 / +2.03 / 61% / +0.94 / +1.09 | 25 / +1.68 / 60% / +0.57 / +1.11 |
| debit | far | 31 / +3.44 / 52% / +3.45 / −0.01 | 25 / +1.39 / 40% / +0.89 / +0.50 |
| debit | all | 106 / +2.43 / 61% / +2.11 / +0.32 | 80 / +1.91 / 59% / +1.44 / +0.48 |
| credit | near | 11 / −8.09 / 0% / −2.79 / −5.30 | 8 / −7.54 / 0% / −1.71 / −5.82 |
| credit | mid | 11 / −1.25 / 55% / −0.87 / −0.38 | 7 / −2.25 / 43% / −1.83 / −0.42 |
| credit | far | 15 / −2.01 / 60% / −2.42 / +0.41 | 13 / −1.02 / 69% / −0.37 / −0.65 |
| credit | all | 37 / −3.59 / 41% / −2.01 / −1.58 | 28 / −3.19 / 43% / −1.35 / −1.85 |

Debit touch-exit vs close, excluding flagged trades: rth_touch n=33 +1.17 vs +3.87; gap_touch n=19 +0.91 vs +3.31.

### Decision-7 restatement of the June verdicts (pre-registered rule: holds = same sign and |beat| ≥ 50% of June; weakens = same sign, < 50%; reverses = sign flip)

| June verdict | Cell | June | CR-AM walk-forward | Restatement | Reasoning |
|---|---|---|---|---|---|
| (a) credit refuted [ROBUST] | close-path beat, near | −5.69 | −5.30 | **holds** | Near-magnet fade loses 8 pts on 0 % wins in both runs. |
| (a) | mid | −1.85 | −0.38 | **weakens** | Same sign, 21 % of June's magnitude; n=11, mean P&L still −1.25. |
| (a) | far | −0.32 | +0.41 | **reverses** | Sign flip of 0.7 pts on n=15 with confirmed far-band selection bias; mean P&L still −2.01 and the beat is −0.65 once out-of-range trades are excluded — a baseline artifact, not a fade edge. |
| (a) | all | −2.24 | −1.58 | **holds** | Credit stays negative on P&L and on beat in every band's P&L; the refutation stands. |
| (b) hold-to-close > touch-exit [LIKELY] | rth_touch: close − touch-exit | +2.44 (4.17 − 1.73) | +2.45 (4.08 − 1.63) | **holds** | Identical gap; n 37 → 41. |
| (b) | gap_touch: close − touch-exit | +3.34 (4.89 − 1.55) | −1.07 (4.10 − 5.17) | **reverses** as run; **holds** (+2.40) excluding out-of-range trades | 5 of 29 gap-open touch-exit values exceed the spread width (up to +78.10); the reversal is quote junk at the next-open minute. Report as data-invalid, not as evidence. |
| (c) debit close-path edge [DIRECTIONAL] | all-train beat vs June train | −0.28 | +0.32 | **reverses** (to positive) | Walk-forward all-train beat turns slightly positive; June's train beat was slightly negative. |
| (c) | all vs June holdout (the basis of the verdict) | +2.26 | +0.32 | **weakens** | Same sign at 14 % of the June holdout magnitude; June's +2.26 rested on n=22 with a −1.26 baseline that this check shows to be contaminated. |
| (c) | near | −0.91 (train) / +0.49 (holdout) | −0.11 | **weakens** vs train; **reverses** vs holdout | Near beat is ~0 either way; the near-magnet chase does not beat its own first-minute baseline. |
| (c) | mid | +0.20 (train) / +5.09 (holdout) | +1.09 | **holds** vs train; **weakens** vs holdout | The only band where debit beats baseline by more than noise; +1.11 in the clean subset. |
| (c) | far | −0.02 (train) / 0.00 (holdout) | −0.01 | **holds** (≈ 0) | No far-band beat in any run; +0.50 in the clean subset on n=25. |
| (d) mid-band negativity ([[mid-band-drift-targets-lack-edge]], CR-AI) | debit mid close P&L / beat | CR-AH train +2.81 / +0.20; CR-AI DTE-15 mid −1.01 (open-price entry) | +2.03 / +1.09 (61 % win) | **holds** vs CR-AH; **not reproduced** vs CR-AI | Under CR-AH's edge-gated crawl entry, mid is positive and is the best-beating band walk-forward. CR-AI's negative mid is on a different entry convention (open price, no gate) and is lookahead-free by construction; the two are not the same measurement. |

Summary D on the full 150-set, walk-forward: debit match − no_match = −0.76 pts, bootstrap CI [−3.12, +1.43] → INCONCLUSIVE; credit UNTESTABLE (0 matches) — consistent with CR-AL. Far-band credit selection bias persists (clean σ 2.81 vs dropped 3.47); debit none.

**Net restatement.** Credit-refuted holds. Hold-to-close-beats-touch-exit holds on RTH touches and is not measurable on gap touches until the quote data is gated. Debit close-path edge weakens: the walk-forward, all-train beat is +0.32 pts (+0.48 excluding out-of-range trades), positive but small, and concentrated in the mid band; near and far do not beat baseline. No verdict reverses on clean data.

### Full output (Phases 1–7)

```

======================================================================
CR-AH Step 4 — Two-structure × two-axis analysis
  cr_id=CR-AM  structural-prob mode=walk-forward  train_only=False  no_persist=False  universe_end=2026-06-05  split_date=2026-06-05  seed=20260905
======================================================================

Run ID: c52daf00-ad7c-4c62-a1b1-5d8fcbada9c2

----------------------------------------------------------------------
Phase 1: Loading signal dates and selecting clean subset...
  Universe pinned to trade_date <= 2026-06-05: 375/396 magnet-above dates kept.
  Loaded 374/375 signal entries (skipped 1).
  Stratified selection: 150 dates
  Selection by band/partition: {'far/train': 50, 'mid/train': 50, 'near/train': 50}
  holdout: none (split = universe end)

Filtering to A-bucket clean dates for each structure...
  Credit (target + target+10)...
    Credit clean: 106
  Debit (target + target-10)...
    Debit clean:  110

  Credit by band/partition: {'far/train': 30, 'mid/train': 36, 'near/train': 40}
  Debit  by band/partition: {'far/train': 33, 'mid/train': 36, 'near/train': 41}

----------------------------------------------------------------------
Phase 2: Collecting per-date trade data...
  Processing 110 debit dates...
  Processing 106 credit dates...

  Collected: debit=110, credit=106
  Settlement available: debit=107/110, credit=103/106
  Actionable touches: debit=70/110, credit=67/106

----------------------------------------------------------------------
Phase 3: Threshold sweep on TRAIN only...

  Chosen threshold — DEBIT: 0.00  CREDIT: 0.00

DEBIT — Threshold sweep (TRAIN only) [mode=walk-forward]:
       T  n_settled  fill_n  mean_pnl  win%   beat  chosen?
  ────────────────────────────────────────────────────────────
  0.00     106      109        2.43    61%     0.32 ← CHOSEN
  0.05     106      109        2.36    60%     0.25
  0.10     104      107        2.23    59%     0.12
  0.15     104      107        2.25    59%     0.14
  0.20     101      104        1.95    56%    -0.15

CREDIT — Threshold sweep (TRAIN only) [mode=walk-forward]:
       T  n_settled  fill_n  mean_pnl  win%   beat  chosen?
  ────────────────────────────────────────────────────────────
  0.00      37       39       -3.59    41%    -1.58 ← CHOSEN
  0.05      28       29       -3.95    36%    -1.94
  0.10      20       20       -4.28    30%    -2.27
  0.15      10       10       -6.24    30%    -4.22
  0.20       5        5       -4.56    40%    -2.55

----------------------------------------------------------------------
Phase 4: Full results (all train; holdout: none (split = universe end))

DEBIT — By distance band (all splits, T=0.00) [mode=walk-forward]:
  band    part        n      pnl   win%  [lo–hi 95%]        base     beat
  ──────────────────────────────────────────────────────────────────────
  near    train      39     2.01    69%  [ 54%– 81%]     2.12    -0.11
  mid     train      36     2.03    61%  [ 45%– 75%]     0.94     1.09
  far     train      31     3.44    52%  [ 35%– 68%]     3.45    -0.01
  all     train     106     2.43    61%  [ 52%– 70%]     2.11     0.32
  holdout: none (split = universe end)

CREDIT — By distance band (all splits, T=0.00) [mode=walk-forward]:
  band    part        n      pnl   win%  [lo–hi 95%]        base     beat
  ──────────────────────────────────────────────────────────────────────
  near    train      11    -8.09     0%  [  0%– 26%]    -2.79    -5.30
  mid     train      11    -1.25    55%  [ 28%– 79%]    -0.87    -0.38
  far     train      15    -2.01    60%  [ 36%– 80%]    -2.42     0.41
  all     train      37    -3.59    41%  [ 26%– 57%]    -2.01    -1.58
  holdout: none (split = universe end)

DEBIT — By post-touch pattern (TRAIN only, T=0.00) [mode=walk-forward]:
  (n with pattern_label=93, n without=17)
  pattern                           n      pnl   win%  [lo–hi 95%]        base     beat
  ──────────────────────────────────────────────────────────────────────
  mixed                            43     2.57    60%  [ 46%– 74%]     3.05    -0.48
  overshoot-then-revert             1     4.35   100%  [ 21%–100%]   -26.90    31.25
  stepping-stone                   46     1.84    57%  [ 42%– 70%]     1.62     0.22
  ──────────────────────────────────────────────────────────────────────
  (labeled)                        90     2.22    59%  [ 49%– 68%]     1.99     0.23
  (unlabeled)                      16     3.65    75%  [ 51%– 90%]     2.74     0.91

CREDIT — By post-touch pattern (TRAIN only, T=0.00) [mode=walk-forward]:
  (n with pattern_label=89, n without=17)
  pattern                           n      pnl   win%  [lo–hi 95%]        base     beat
  ──────────────────────────────────────────────────────────────────────
  mixed                             9    -0.63    67%  [ 35%– 88%]    -0.49    -0.14
  overshoot-then-revert             0        —      —  [   —–   —]        —        —
  stepping-stone                   16    -3.48    44%  [ 23%– 67%]    -2.43    -1.06
  ──────────────────────────────────────────────────────────────────────
  (labeled)                        25    -2.46    52%  [ 33%– 70%]    -1.55    -0.91
  (unlabeled)                      12    -5.95    17%  [  5%– 45%]    -4.36    -1.59

DEBIT — Touch resolution breakdown (TRAIN only, T=0.00) [mode=walk-forward]:
  resolution                      n   touch_exit  close_pnl   base_close
  ─────────────────────────────────────────────────────────────────
  rth_touch                       41        1.63        4.08        4.05
  gap_touch                       29        5.17        4.10        2.66
  afterhours_touch_retraced       15           —        3.94        6.94
  no_touch                        25           —       -2.84       -4.43

DEBIT — Selection bias check (far band, decision #11) [mode=walk-forward]:
  far/all: n=50  clean: n=33 (mean σ=3.16)  dropped: n=17 (mean σ=2.90)
  ✓ No significant selection bias detected in far band.

CREDIT — Selection bias check (far band, decision #11) [mode=walk-forward]:
  far/all: n=50  clean: n=30 (mean σ=2.81)  dropped: n=20 (mean σ=3.47)
  ⚠ BIAS: clean far trades are significantly CLOSER than dropped far trades.
    The far-band result may be OPTIMISTICALLY SELECTED (closer to spot → easier trade).

======================================================================
SUMMARY READS A/B/C/D
======================================================================
[mode=walk-forward]

── Summary A: Debit near-band holdout ──
  holdout: none (split = universe end)

── Summary B: Credit near-band holdout ──
  holdout: none (split = universe end)

── Summary C: Structure crossover by distance (TRAIN, close P&L beat) ──
  near   debit_beat=  -0.11  credit_beat=  -5.30  → DEBIT leads by 5.19
  mid    debit_beat=   1.09  credit_beat=  -0.38  → DEBIT leads by 1.47
  far    debit_beat=  -0.01  credit_beat=   0.41  → CREDIT leads by 0.42
  READ: Structure crossover = distance band where debit stops leading and credit starts.

── Summary D: Engine hypothesis (decision #13) ──
  DEBIT engine check:  pattern_match (n=46) pnl=1.84  vs no_match (n=44) pnl=2.61
  → Pattern filter HURTS debit (match performs WORSE). Engine rule may be wrong.
  DEBIT decision-9 read [mode=walk-forward]: match−no_match = -0.76 pts, bootstrap 95% CI [-3.12, +1.43] (1000 resamples, seed=20260905) → INCONCLUSIVE
  credit: engine hypothesis UNTESTABLE (n_match=0, n_no_match=25)

----------------------------------------------------------------------
Phase 7: Persisting aggregate stats to bt_edge_backtest_results...
  ✓ bt_edge_backtest_results created/verified.
  debit: holdout: none (split = universe end)
  credit: holdout: none (split = universe end)
  ✓ Aggregate stats written.

======================================================================
Step 4 complete in 339s
======================================================================
```

## Step 2 — holdout leg capture (decision 9; no read)

Command: `PYTHONUNBUFFERED=1 apps/web/.venv/bin/python -u scripts/cr_am_holdout_leg_capture.py` (defaults: split 2026-06-05, cr_id CR-AM-capture). Log: `scripts/logs/cr_am_capture_20260905_091438.log` (untracked). 57 min (ORATS per-gap calls, 84 in total). Run row: `1ea36351-9b4d-4d6d-a0eb-7e7d36841558`, `completed`, smoke dict = the counters below.

| Gate | Expected | Actual | Result |
|---|---|---|---|
| G5 fetch exceptions other than ORATS 404 | 0 | **0** (8 × ORATS 404, all entry-day) | PASS (404s listed) |
| G6 no holdout P&L in the capture log | none | grep `pnl\|net_price\|payoff\|p&l` → 0 hits; the script has no P&L code path | PASS |

Counters: dates 21 (0 without target; wall and payload targets identical on all 21 → 63 legs, no union symbols) · entry-day windows fetched 13 · settlement windows fetched 15 · settlement windows deferred 6 (expiry after 2026-09-05: 08-18, 08-26, 08-27, 08-28, 09-03, 09-04) · bars_written 31,362 · cache_hits 0 · gaps_filled 84 · ORATS 404 8 · exceptions 0.

Entry-day 404s (15-DTE legs not served by ORATS for that session): **2026-06-16, 07-01, 07-13, 08-12, 08-13, 08-28, 09-03, 09-04.** The last two may be ingestion lag rather than absence (one and two days old at capture time); a 404 raises before any window row is written, so they remain fetchable. Settlement windows for the 06-16 / 07-01 / 07-13 / 08-12 / 08-13 expiries were captured even though their entry days were not.

Holdout stream state after capture: 21 dates; **10 fully captured** (entry + settlement), **3 entry-only pending expiry** (08-18, 08-26, 08-27), **2 nothing yet, expiry pending** (09-03, 09-04 — retry entry-day), **6 entry-day unavailable** (5 with settlement captured). Follow-up capture after 2026-09-28 fetches the deferred settlement windows; the script is idempotent (cache hits on already-fetched windows).

## What changed

- `scripts/cr_ah_step4_analysis.py`: `SPLIT_DATE` constant replaced by `--split-date` (default 2026-06-05 per the ADR; June's 2025-08-12 remains selectable); every holdout print path (selection, by-band, Summary A/B, Phase 7) prints `holdout: none (split = universe end)` when no holdout rows exist; the `bt_edge_backtest_results` INSERT writes `cr_id` from `--cr-id`.
- `scripts/cr_am_holdout_leg_capture.py` (new): captures entry-day RTH and expiry settlement-window quotes for the debit/credit legs of every post-split magnet-above date via `fetch_option_bars`, under `backfill_run(cr_id='CR-AM-capture')`. Logs dates / targets / legs / bars / 404s only; settlement windows with a future expiry are deferred (not fetched, not recorded as empty). No `net_price`, `payoff`, or P&L anywhere.
- Data: CR-AM run `c52daf00` persisted 8 all-train rows to `bt_edge_backtest_results` (`cr_id='CR-AM'`); CR-AH's 16 rows untouched. Capture run `1ea36351`: 13 entry-day + 15 settlement windows fetched, 6 settlement windows deferred (future expiry), 8 entry-day 404s, 0 exceptions, 31,362 bars written.
- Gates: G0.1–G0.3, G1–G4, G6 pass; G5 pass (8 ORATS 404s listed, 0 other exceptions). No halts. CR-AI Stage 2 not re-run (G0.2 absent).
- Result: **credit refuted holds; hold-to-close beats touch-exit holds on RTH touches (gap-touch cell not measurable — quote junk); debit close-path edge weakens** to +0.32 pts all-train walk-forward (+0.48 excluding out-of-range trades), concentrated in the mid band; no verdict reverses on clean data. Mid-band negativity (CR-AI) is not reproduced under CR-AH's edge-gated convention — different measurement, not a contradiction.

## Decisions

- **Applied decision 7 mechanically and reported the sign flips it produces, then explained them.** Two cells "reverse" as run (credit far beat, debit gap-touch touch-exit); both are driven by out-of-range quote values and neither survives the range filter. Recorded as reverses in the table per the pre-registered rule, with the reasoning column carrying the diagnosis, rather than silently applying a filter that was not pre-registered.
- **Persisted CR-AM rows as-run (no range filter),** because the spec says persist the re-run as the new reference and the filter is post-hoc. The sensitivity table lives in the spec, not in the table.
- **Capture fetched the union of wall-target and payload-target strikes** (identical on all 21 dates, so no extra symbols) and **deferred settlement windows whose expiry is after today** rather than recording them as empty, so they stay fetchable.
- **Did not gate the harness on quote sanity in this CR** (would change P&L on the persisted reference and June's comparison column). Filed as the first open question.
- **G4 read against the script's own June classification** (CR-AL run on the same universe) rather than the CR-AH spec's hand-written Step-2.5 table, which is off by one credit date near↔far; totals and debit split match exactly.

## Open questions

- **Quote-sanity gate for the edge harness.** 26/110 debit and 20/106 credit trades carry a spread value outside ±width at the baseline minute, the fill minute, or the touch minute (inverted or stale leg mid). Every "beat vs baseline" in June and CR-AM is contaminated by this. Add a validity gate in `compute_pnl` / `net_price_from_real_quotes` (reject spread values outside [0, width], and a debit fill with positive credit), then re-derive both columns. Until then, prefer mean close P&L over beat when reading the tables.
- **The debit edge is a mid-band effect** in walk-forward (+1.09 beat; near −0.11, far −0.01). This is the opposite of CR-AI's mid-negative finding under open-price entry — the entry rule, not the target distance, may be what separates them. See item 6 added to [[mid-band-drift-targets-lack-edge]].
- **Holdout stream is 21 dates and growing;** entry-day quotes are missing at ORATS for 8 of them (404 on the 15-DTE legs). A powered read needs a minimum n per band, pre-registered — [[debit-edge-needs-powered-holdout]]. Deferred settlement windows (6 dates with expiry after 2026-09-05) need a follow-up capture after 2026-09-28.
- **CR-Y reads `bt_edge_backtest_results`** — confirm it filters to the latest `cr_id` (CR-AM) and does not average across CR-AH and CR-AM rows.
- **`cr_ah_posthoc_limit_entry.py`** remains untracked and unpatched for the cutoff contract.
