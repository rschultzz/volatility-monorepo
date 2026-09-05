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
