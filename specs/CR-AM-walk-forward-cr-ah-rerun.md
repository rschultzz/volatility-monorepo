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

