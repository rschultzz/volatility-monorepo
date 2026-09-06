# CR-AP — Snapped clean-quote walk-forward reference (re-run of CR-AN with listed-strike snapping)

> Authority: vault session note `Dash/sessions/2026-09-06 - CR-AP — Snapped Reference Re-run.md`
> Branch: `feat/CR-AP-snapped-reference-rerun` (off `origin/main` 7016dd1; CR-AO merge 1111e8b is an ancestor)
> Scope: run-only. No code changes. One walk-forward run persisted as `cr_id='CR-AP'`.
> Mode: unattended; halts at the first STOP gate that misses.

## Problem

CR-AO Step 0: 9/110 debit and 12/106 credit trades in the CR-AN reference used a leg strike that was not in the prior-close chain on the signal date, and every one of those legs had entry-day quotes anyway — ORATS served the contract once it was listed, so the backtest priced a leg with data from a later listing. Credit crossed the 10% line. CR-AN's rows are not citable until the harness runs with `snap_to_listed_strike` (now on `main`).

## Locked decisions

| # | Decision | Value |
|---|---|---|
| 1 | Configuration | Identical to CR-AN: `--universe-end 2026-06-05 --split-date 2026-06-05 --structural-prob-mode walk-forward`, `--cr-id CR-AP`. Snapping is on by construction (CR-AO wired it into the leg builder). |
| 2 | Width | `width_actual` per trade as recorded by CR-AO. Band tables pool all widths; the spec appends a `width_actual ≠ 10` count per structure and a separate mean P&L for that subset so the pooling decision can be revisited. |
| 3 | Restatement | CR-AP vs CR-AN, same holds / weakens / reverses rule (sign + ≥ 50% magnitude). The 21 affected trades are listed with old leg / snapped leg / old P&L / new P&L. |
| 4 | Persistence | `cr_id='CR-AP'`. CR-AN / CR-AM / CR-AH rows kept. **CR-AP is the citable reference after this run.** |
| 5 | Not in scope | CR-AI re-run; any code. |

## Gates

| Gate | Expected | On miss |
|---|---|---|
| G0 — `main` contains the CR-AO merge | yes | STOP |
| G1 — selection 50/50/50; clean debit 110 / credit 106 | exact | STOP |
| G2 — trades with an unlisted leg after snapping | 0 | STOP |
| G3 — trades with `width_actual ≠ 10` | ≤ 21 | note |
| G4 — restatement vs CR-AN: the 189 unaffected trades identical to the cent | yes | STOP if any unaffected trade changed |
| G5 — no holdout P&L in the log | yes | STOP and redact |


## Step 0 findings (read-only, 2026-09-06)

| Gate | Expected | Actual | Result |
|---|---|---|---|
| G0 `main` contains the CR-AO merge | yes | `origin/main` = 7016dd1 (render.yaml autoDeploy commit) with 1111e8b (PR #47 merge) as an ancestor; branch cut from 7016dd1 | PASS |

`bt_edge_backtest_results` rows by `cr_id` before the run (expect CR-AH, CR-AM, CR-AN present):

```
('CR-AH', 16, datetime.date(2026, 6, 10))
('CR-AM', 8, datetime.date(2026, 9, 5))
('CR-AN', 8, datetime.date(2026, 9, 5))
```

Code state that the run inherits (nothing changed here): CR-AN's quote-sanity filter (`packages/shared/backtest/quote_validity.py`) and CR-AO's listed-strike snapping (`packages/shared/options_cache/strikes.py`, wired into `filter_clean_for_structure` / `build_legs` in `scripts/cr_ah_step4_analysis.py`; `width_actual` on `TradeData`). Interpreter `apps/web/.venv/bin/python`. CR-AN per-trade values for the restatement come from the CR-AN per-trade dump (`cr_an_trade_diag.json`, T = 0.00 / 0.05) and the unlisted-leg list from CR-AO's G0.4 (`cr_ao_g04.json`); a matching per-trade dump is taken for CR-AP with the snapped legs.

## Halt — G1 (clean counts) missed

**Halted 2026-09-06 ~08:12 PT at G1.** Selection is exact (50 / 50 / 50, all train) but the clean sample under snapping is **debit 101 / credit 94**, not 110 / 106. No CR-AP rows were persisted; nothing was cleaned up.

### What happened

1. First run (`d774351b`, log `cr_ap_step4_20260906_075950.log`) aborted in Phase 1 with `NameError: Counter` — CR-AO's wiring commit used `Counter` at module scope in `main()` without the module-level import (an indented local import matched the patch's guard) and `main()` was not exercised before merge. The `bt_backfill_runs` row is `aborted` with the traceback. Fixed by the one-line import in commit 65f0868 — a scope deviation from "run-only, no code", recorded here because the run cannot start without it.
2. Second run (`bb39a829`, log `scripts/logs/cr_ap_step4_20260906_080530.log`) reached Phase 2 and reported the clean counts above. Because `main()` persists unconditionally in Phase 7 and the vault names CR-AP "the citable reference after this run", the process was stopped at 60 / 195 trades before Phase 7 so a reduced-sample reference could not become the latest `cr_id`. `bt_edge_backtest_results` has **0** CR-AP rows. The run row `bb39a829` remains `status = 'running'` (a killed process cannot mark itself); left as is.

### Why the sample shrank — exactly the 21 CR-AO-affected trades

A read-only per-trade dump with the snapped legs (scratchpad `cr_ap_trade_diag.py`, `cr_ap_trade_diag.json`) has 195 trades; the 21 CR-AN trades absent from it are **exactly the 21 that CR-AO's G0.4 flagged** (9 debit, 12 credit), and 0 new trades appear. Snapping moves those legs from the unlisted 5-point strike to the listed 10-point strike (e.g. 4805 → 4800), and **those OPRAs were never fetched**: June's CR-AH backfill captured the `round5` legs only, so the snapped strike has no entry-day rows in `orats_options_minute` and `filter_clean_for_structure` ("both legs have entry-day bars") drops the date. `StrikeNotListed` did not fire (0 unlistable); every surviving trade has `width_actual = 10`.

| structure | date | band | CR-AN legs (short / other) | unlisted in CR-AN | snapped anchor |
|---|---|---|---|---|---|
| credit | 2023-12-22 | — | 4805 / 4815 | both | 4800 |
| credit | 2024-03-05 | — | 5175 / 5185 | other | 5175 |
| credit | 2024-07-11 | — | 5675 / 5685 | other | 5675 |
| credit | 2024-09-17 | — | 5710 / 5720 | both | 5700 |
| credit | 2024-09-23 | — | 5755 / 5765 | both | 5750 |
| credit | 2024-11-25 | — | 6070 / 6080 | other | 6070 |
| credit | 2024-12-13 | — | 6105 / 6115 | both | 6100 |
| credit | 2025-03-26 | — | 5850 / 5860 | other | 5850 |
| credit | 2025-05-16 | — | 6000 / 6010 | other | 6000 |
| credit | 2025-06-11 | — | 6075 / 6085 | other | 6075 |
| credit | 2025-06-24 | — | 6100 / 6110 | other | 6100 |
| credit | 2026-04-07 | — | 6700 / 6710 | other | 6700 |
| debit | 2023-11-10 | — | 4510 / 4500 | short | 4500 |
| debit | 2023-12-22 | — | 4805 / 4795 | both | 4800 |
| debit | 2024-07-11 | — | 5675 / 5665 | other | 5675 |
| debit | 2024-09-17 | — | 5710 / 5700 | short | 5700 |
| debit | 2024-09-23 | — | 5755 / 5745 | both | 5750 |
| debit | 2024-11-12 | — | 6055 / 6045 | both | 6050 |
| debit | 2024-12-13 | — | 6105 / 6095 | short | 6100 |
| debit | 2025-03-26 | — | 5850 / 5840 | other | 5850 |
| debit | 2025-06-11 | — | 6075 / 6065 | other | 6075 |

(Bands are in `cr_ao_g04.json`; omitted here to keep the table readable.) Note the "other unlisted" cases: the anchor stays (e.g. 5175) but the 10-point wing (5185) was not listed, so the snapped wing is 5190 or the next listed strike — again an OPRA that was never fetched.

### What the gate is telling us

CR-AO made the leg builder honest; the **data** for the honest legs does not exist yet on 21 signal dates. G1's "exact 110 / 106" assumed snapping would only relabel legs that already had quotes. It cannot: a strike that was not listed on the signal day never had a CR-AH backfill run against its listed neighbour.

### What unblocks CR-AP (not done here — data step, out of this CR's run-only scope)

Capture the snapped legs for the 21 dates (entry-day RTH window; the ES touch window for the 9 debit dates so `touch_exit` stays populated; settlement is the ES close and needs nothing) — the CR-AH Step 2 shape, restricted to an explicit date × OPRA list. `scripts/cr_am_holdout_leg_capture.py` already snaps and takes `--dates` but fetches only entry-day + settlement windows and is keyed to the post-split holdout query; either extend it with a `--pre-split` mode and a touch-window fetch, or write a small `cr_aq_snapped_leg_capture.py` from the same fetch path. Then re-run this CR's Step 1 unchanged; G1 should read 110 / 106 (the 21 dates regain both legs) with, possibly, a few `width_actual ≠ 10` where the listed wing is farther than 10.

### State left behind

- Branch `feat/CR-AP-snapped-reference-rerun`: spec freeze, Step 0, the Counter hotfix, this halt. Pushed.
- `bt_backfill_runs`: `d774351b` aborted (NameError), `bb39a829` running (killed). No CR-AP results rows. CR-AN remains the latest `cr_id` in `bt_edge_backtest_results` and remains non-citable per CR-AO's G0.4 until CR-AP completes.
- The Counter hotfix is needed on `main` regardless of CR-AP: without it every future Step 4 run aborts in Phase 1.

## Spec amendment (resume, 2026-09-06) — Step 1a and corrected gates

Authority: resume instruction 2026-09-06 after the G1 halt. Amended before any code.

### Step 1a — snapped-leg capture (new, precedes Step 1)

For the 21 (structure, date) trades in `## Halt` (14 distinct dates; 9 debit, 12 credit), fetch into `orats_options_minute` for both **snapped** legs (via `snap_vertical_legs`, the same call the harness makes): the entry-day RTH window (06:30–13:00 PT), the settlement window at expiry (12:50–13:00 PT), and — for the 9 debit dates — the touch-day window the harness reads for exit-on-touch: `get_touch_pos_val` queries `[touch_datetime_pt, touch_datetime_pt + 90 min]` where `detect_touch` gives the RTH touch minute (`rth_touch`) or the next 06:30 PT open (`gap_touch`); no window when there is no actionable touch. Script `scripts/cr_ap_capture_snapped_legs.py` (sibling of the CR-AM/CR-AO capture path: same fetcher, backfill role, `backfill_run` `cr_id='CR-AP-capture'`, logs dates / legs / 404s only, no P&L). Precondition for Step 1 (**G1b-pre**): every snapped leg for the 21 trades has entry-day rows in `orats_options_minute`, or is a listed ORATS 404.

### Corrected gates

| Gate | Expected | On miss |
|---|---|---|
| G1 (corrected) — selection 50/50/50; clean counts **after Step 1a** | debit 110 / credit 106; tolerance **−3 per structure** for ORATS 404s on listed 10-point strikes, each 404 listed by date / leg | beyond −3, or any shortfall not explained by a listed 404 → STOP |
| G1b (new) — trades with an unlisted leg after snapping | **0** | STOP |
| G2 — G5 | unchanged | unchanged |

Rationale: G1's original "exact" assumed snapping only relabels legs that already had quotes; the honest legs on the 21 dates had never been fetched. A listed 10-point strike can itself 404 at ORATS on a given day (CR-AO's re-capture saw none, but CR-AH's original D-bucket did), hence the small tolerance with every miss named.

### Run-record note

Run `bb39a829` (the stopped second run) is stuck at `status = 'running'`: `backfill_run` only marks `aborted` on an exception inside the context manager, and the backfill role has no path to close a row from outside (no UPDATE from a different session is part of the protocol; the script is the only writer). Appended to [[bt-backfill-runs-audit-record-fidelity]] as a third fidelity gap (stuck-running on external kill), alongside the dry-run and `rows_inserted` issues from CR-AJ.

## Step 1a — snapped-leg capture (result)

Command: `PYTHONUNBUFFERED=1 apps/web/.venv/bin/python -u scripts/cr_ap_capture_snapped_legs.py --dates <14 dates> --debit-dates <9> --credit-dates <12>`. Log: `scripts/logs/cr_ap_capture_20260906_081643.log` (untracked). Run row: `(UUID('8022295f-c7c5-453d-8b63-7b7dc3c95931'), 'completed', datetime.datetime(2026, 9, 6, 15, 16, 58, 506635), datetime.datetime(2026, 9, 6, 15, 59, 4, 876473))` (cr_id CR-AP-capture).

**Summary:** captured 14 entry-day + 13 settlement + 7 touch windows over 14 dates (0 unlistable; 2 debit dates without an actionable touch: 2024-12-13 afterhours-retraced, 2025-03-26 no-touch); 20,898 bars written; **1 ORATS 404, 0 exceptions**. The 404 is the **settlement** window of 2025-06-11 (expiry 2025-07-03, a half-day session that closed before 12:50 PT — the window does not exist); the harness settles on the ES close, so it does not affect the clean filter or P&L. Entry-day windows: 14 / 14.

Snapped legs (from the plan; width_actual in parentheses): 2023-11-10 debit 4490/4500 (10) · 2023-12-22 debit 4790/4800, credit 4800/4810 (10) · 2024-03-05 credit 5175/5180 (5) · 2024-07-11 debit 5670/5675, credit 5675/5680 (5) · 2024-09-17 debit 5675/5700 (25), credit 5700/5750 (50) · 2024-09-23 debit 5740/5750 (10), credit 5750/5775 (25) · 2024-11-12 debit 6050/6060 (10) · 2024-11-25 credit 6070/6075 (5) · 2024-12-13 debit 6090/6100, credit 6100/6110 (10) · 2025-03-26 debit 5825/5850, credit 5850/5875 (25) · 2025-05-16 credit 6000/6025 (25) · 2025-06-11 debit 6070/6075, credit 6075/6080 (5) · 2025-06-24 credit 6100/6125 (25) · 2026-04-07 credit 6700/6725 (25). Touch windows fetched for 7 debit dates (5 gap_touch at the next 06:30 PT open — cache hits where the touch day is the entry day — and 2 rth_touch).

### G1b-pre — every snapped leg has entry-day rows

| date | leg | entry-day rows | settlement rows |
|---|---|---|---|
| 2023-11-10 | SPX231204C04490000 | 390 | 11 |
| 2023-11-10 | SPX231204C04500000 | 390 | 11 |
| 2023-12-22 | SPX240117C04790000 | 390 | 11 |
| 2023-12-22 | SPX240117C04800000 | 390 | 11 |
| 2023-12-22 | SPX240117C04810000 | 390 | 11 |
| 2024-03-05 | SPX240326C05175000 | 390 | 11 |
| 2024-03-05 | SPX240326C05180000 | 390 | 11 |
| 2024-07-11 | SPX240801C05670000 | 381 | 11 |
| 2024-07-11 | SPX240801C05675000 | 381 | 11 |
| 2024-07-11 | SPX240801C05680000 | 381 | 11 |
| 2024-09-17 | SPX241008C05675000 | 390 | 11 |
| 2024-09-17 | SPX241008C05700000 | 390 | 11 |
| 2024-09-17 | SPX241008C05750000 | 390 | 11 |
| 2024-09-23 | SPX241014C05740000 | 390 | 11 |
| 2024-09-23 | SPX241014C05750000 | 390 | 11 |
| 2024-09-23 | SPX241014C05775000 | 390 | 11 |
| 2024-11-12 | SPX241204C06050000 | 390 | 11 |
| 2024-11-12 | SPX241204C06060000 | 390 | 11 |
| 2024-11-25 | SPX241217C06070000 | 390 | 11 |
| 2024-11-25 | SPX241217C06075000 | 390 | 11 |
| 2024-12-13 | SPX250107C06090000 | 390 | 11 |
| 2024-12-13 | SPX250107C06100000 | 390 | 11 |
| 2024-12-13 | SPX250107C06110000 | 390 | 11 |
| 2025-03-26 | SPX250416C05825000 | 390 | 11 |
| 2025-03-26 | SPX250416C05850000 | 390 | 11 |
| 2025-03-26 | SPX250416C05875000 | 390 | 11 |
| 2025-05-16 | SPX250609C06000000 | 390 | 11 |
| 2025-05-16 | SPX250609C06025000 | 390 | 11 |
| 2025-06-11 | SPX250703C06070000 | 390 | 0 |
| 2025-06-11 | SPX250703C06075000 | 390 | 0 |
| 2025-06-11 | SPX250703C06080000 | 390 | 0 |
| 2025-06-24 | SPX250716C06100000 | 390 | 11 |
| 2025-06-24 | SPX250716C06125000 | 390 | 11 |
| 2026-04-07 | SPX260428C06700000 | 390 | 11 |
| 2026-04-07 | SPX260428C06725000 | 390 | 11 |

legs checked: 35; legs with NO entry-day rows: 0

Result: **PASS** — all legs have entry-day rows; the only missing settlement rows are the 2025-07-03 half-day (listed 404).

## Step 1 — snapped clean-quote walk-forward run, persisted as `cr_id='CR-AP'`

Command: `PYTHONUNBUFFERED=1 apps/web/.venv/bin/python -u scripts/cr_ah_step4_analysis.py --universe-end 2026-06-05 --split-date 2026-06-05 --cr-id CR-AP --structural-prob-mode walk-forward` (identical to CR-AN; snapping on by construction). Log: `scripts/logs/cr_ap_step4_20260906_085953.log` (untracked). 321 s. Run row: `(UUID('ad221de1-28fb-4222-b7d4-7474d73a16e3'), 'completed', datetime.datetime(2026, 9, 6, 15, 59, 54, 871649), datetime.datetime(2026, 9, 6, 16, 5, 14, 923789), "Step 4 complete [mode=walk-forward]: debit=110, credit=106, T_d=0.05, T_c=0.0, 320s; summary_d={'debit': 'INCONCLUSIVE', 'credit': 'untestable'}")`. Persisted: 8 rows `cr_id='CR-AP'`. **CR-AP is now the citable reference** (decision 4); CR-AH / CR-AM / CR-AN rows kept.

| Gate | Expected | Actual | Result |
|---|---|---|---|
| G1 (corrected) selection; clean counts after Step 1a | 50/50/50; debit 110 / credit 106 (−3 tolerance for listed 404s) | **50 / 50 / 50; debit 110 / credit 106** — exact, tolerance unused (bands 41/36/33, 40/36/30 as CR-AN) | PASS |
| G1b trades with an unlisted leg after snapping | 0 | **0** (0 `StrikeNotListed`; every leg from the prior-close chain) | PASS |
| G2 post-filter out-of-range accepted values | 0 | **0** | PASS |
| G3 trades with `width_actual ≠ 10` | ≤ 21 | **14** — debit 4 (5-wide ×2, 25-wide ×2), credit 10 (5 ×4, 25 ×5, 50 ×1) | PASS (note) |
| G4 the 195 unaffected trades identical to CR-AN to the cent | yes | **yes** — 195 / 195 identical on fill, close and baseline at T = 0 and close at T = 0.05; 20 of the 21 affected trades changed (the 21st, 2025-06-11 credit, has no fill in either) | PASS |
| G5 no holdout P&L in the log | none | grep of the 21 post-split dates: **0** hits | PASS |

Chosen thresholds: DEBIT 0.05 (sweep flat: +0.09 / +0.10 / +0.08 / +0.08 / −0.05), CREDIT 0.00. Quote validity unchanged in character (debit p95 baseline offset 1 min, credit 1 min; 0 decision-6 exclusions).

### The 21 affected trades — CR-AN legs → snapped legs (P&L at T = 0.00; "—" = no fill or no settlement)

| structure | date | band | old short/other | snapped short/other (width) | old close / base | new close / base |
|---|---|---|---|---|---|---|
| credit | 2023-12-22 | mid | 4805/4815 [short,other unlisted] | 4800/4810 (10) | — / +4.35 | — / +4.30 |
| credit | 2024-03-05 | near | 5175/5185 [other unlisted] | 5175/5180 (5) | — / -6.45 | — / -3.25 |
| credit | 2024-07-11 | near | 5675/5685 [other unlisted] | 5675/5680 (5) | — / +4.25 | — / +2.15 |
| credit | 2024-09-17 | mid | 5710/5720 [short,other unlisted] | 5700/5750 (50) | — / -5.26 | — / -27.69 |
| credit | 2024-09-23 | near | 5755/5765 [short,other unlisted] | 5750/5775 (25) | -5.20 / -4.98 | -12.71 / -12.71 |
| credit | 2024-11-25 | far | 6070/6080 [other unlisted] | 6070/6075 (5) | — / +4.25 | — / +2.10 |
| credit | 2024-12-13 | near | 6105/6115 [short,other unlisted] | 6100/6110 (10) | — / +4.87 | — / +5.07 |
| credit | 2025-03-26 | mid | 5850/5860 [other unlisted] | 5850/5875 (25) | +2.90 / +2.90 | +7.65 / +8.95 |
| credit | 2025-05-16 | mid | 6000/6010 [other unlisted] | 6000/6025 (25) | -4.65 / -4.65 | +0.95 / +10.20 |
| credit | 2025-06-11 | near | 6075/6085 [other unlisted] | 6075/6080 (5) | — / — | — / — |
| credit | 2025-06-24 | mid | 6100/6110 [other unlisted] | 6100/6125 (25) | — / -4.45 | — / -13.70 |
| credit | 2026-04-07 | near | 6700/6710 [other unlisted] | 6700/6725 (25) | -9.30 / -5.20 | — / -16.15 |
| debit | 2023-11-10 | far | 4510/4500 [short unlisted] | 4500/4490 (10) | +8.50 / +8.50 | +8.20 / +8.20 |
| debit | 2023-12-22 | mid | 4805/4795 [short,other unlisted] | 4800/4790 (10) | -4.65 / -4.65 | -4.60 / -4.60 |
| debit | 2024-07-11 | near | 5675/5665 [other unlisted] | 5675/5670 (5) | -3.45 / -3.45 | -2.45 / -2.45 |
| debit | 2024-09-17 | mid | 5710/5700 [short unlisted] | 5700/5675 (25) | +4.96 / +4.96 | +11.19 / +11.19 |
| debit | 2024-09-23 | near | 5755/5745 [short,other unlisted] | 5750/5740 (10) | +5.00 / +4.72 | +5.00 / +4.57 |
| debit | 2024-11-12 | near | 6055/6045 [short,other unlisted] | 6060/6050 (10) | +5.50 / +5.50 | +5.65 / +5.65 |
| debit | 2024-12-13 | near | 6105/6095 [short unlisted] | 6100/6090 (10) | -5.27 / -5.27 | -5.47 / -5.47 |
| debit | 2025-03-26 | mid | 5850/5840 [other unlisted] | 5850/5825 (25) | -4.85 / -4.85 | -13.60 / -13.60 |
| debit | 2025-06-11 | near | 6075/6065 [other unlisted] | 6075/6070 (5) | — / — | — / — |


Reading the list: where the anchor was listed and only the 10-point wing was missing, the wing snapped to the nearest listed strike on its side — 5 away when the 5-point strike existed (2024-03-05, 07-11, 11-25, 2025-06-11) and 25 away when the grid was 25-point that day (2025-03-26, 05-16, 06-24, 2026-04-07); 2024-09-17 credit landed on a 50-wide because the chain for the 10-08 expiry had no strike between 5700 and 5750 on 09-16. Debit P&L moves by cents where the pair merely shifted 5 points (2023-11-10, 12-22, 2024-09-23, 11-12, 12-13) and by points where the width changed (2024-09-17 +4.96 → +11.19 on a 25-wide; 2025-03-26 −4.85 → −13.60).

### \`width_actual ≠ 10\` subset (decision 2)

| structure | n ≠ 10 | widths | settled in subset | mean close P&L (T = 0) | mean baseline |
|---|---|---|---|---|---|
| debit | 4 | 5 ×2, 25 ×2 | 3 | −1.62 | −1.62 |
| credit | 10 | 5 ×4, 25 ×5, 50 ×1 | 3 (at T = 0 only 3 of the 10 fill) | −1.37 | −5.57 |

The persisted band cells pool all widths (decision 2). The subsets are too small to read on their own, but the credit one matters for the restatement below because a 25- or 50-wide credit spread's baseline is several times a 10-wide's.

### Restatement — CR-AP vs CR-AN (decision 3; persisted cells at each run's chosen T)

| structure | band | CR-AN | CR-AP (snapped) | Δ beat | restatement |
|---|---|---|---|---|---|
| debit | near | n=39 / +2.12 / 69% [54–81] / base +2.05 / beat +0.07 | n=38 / +2.04 / 68% [53–81] / base +2.07 / beat -0.03 | -0.10 | **reverses** |
| debit | mid | n=36 / +1.17 / 56% [40–70] / base +0.87 / beat +0.30 | n=36 / +1.10 / 56% [40–70] / base +0.80 / beat +0.30 | -0.00 | **holds** |
| debit | far | n=31 / +2.00 / 48% [32–65] / base +1.97 / beat +0.03 | n=31 / +1.99 / 48% [32–65] / base +1.96 / beat +0.03 | -0.00 | **holds** |
| debit | all | n=106 / +1.76 / 58% [49–67] / base +1.63 / beat +0.13 | n=105 / +1.70 / 58% [49–67] / base +1.61 / beat +0.10 | -0.04 | **holds** |
| credit | near | n=4 / -6.45 / 0% [0–49] / base -2.28 / beat -4.17 | n=3 / -8.00 / 0% [0–56] / base -2.73 / beat -5.28 | -1.11 | **holds** |
| credit | mid | n=10 / -1.37 / 60% [31–83] / base -1.31 / beat -0.06 | n=10 / -0.34 / 70% [40–89] / base -1.61 / beat +1.27 | +1.34 | **reverses** |
| credit | far | n=13 / -1.20 / 69% [42–87] / base -0.76 / beat -0.43 | n=13 / -1.20 / 69% [42–87] / base -0.84 / beat -0.36 | +0.08 | **holds** |
| credit | all | n=27 / -2.04 / 56% [37–72] / base -1.53 / beat -0.51 | n=26 / -1.65 / 62% [43–78] / base -1.82 / beat +0.17 | +0.68 | **reverses** |


Per cell:

- **debit near — reverses** (+0.07 → −0.03): a 0.10-pt move across zero on n = 38; both values are noise around zero. Mean close P&L +2.12 → +2.04.
- **debit mid / far / all — hold** (Δ beat −0.00 / −0.00 / −0.04). All-train: +1.76 → +1.70 mean P&L, beat +0.13 → +0.10. The debit reference is essentially CR-AN with five trades re-priced on their listed neighbours.
- **credit near — holds** (−4.17 → −5.28; n = 3).
- **credit mid — reverses** (−0.06 → +1.27) and **credit all — reverses** (−0.51 → +0.17): this is the width-pooling artifact decision 2 anticipated, not a change in the fade's merit. The mid cell now contains 2025-03-26 (25-wide, baseline +8.95) and 2025-05-16 (25-wide, baseline +10.20 → close +0.95) and the credit pool includes 2024-09-17 (50-wide, baseline −27.69) and 2026-04-07 (25-wide, baseline −16.15). Beat = mean − baseline, so a handful of wide spreads with baselines an order of magnitude larger than a 10-wide's dominate the cell. **Mean close P&L stays negative in every credit band** (near −8.00, mid −0.34, far −1.20, all −1.65); the credit fade is not rehabilitated. Beat on a pooled-width cell is not a like-for-like number until widths are normalised (see Open questions).

Summary C now shows "CREDIT leads by 0.97" at mid — same artifact. Summary D: debit match − no_match = +0.12 pts, CI [−2.06, +2.29] → INCONCLUSIVE; credit UNTESTABLE — consistent with CR-AL / CR-AM / CR-AN. Touch resolution: rth_touch touch-exit +1.11 vs close +3.47; gap_touch +1.07 vs +3.25 — hold-to-close still beats touch-exit on both.

### Full output (Phases 1–7)

```

======================================================================
CR-AH Step 4 — Two-structure × two-axis analysis
  cr_id=CR-AP  structural-prob mode=walk-forward  train_only=False  no_persist=False  universe_end=2026-06-05  split_date=2026-06-05  seed=20260905
======================================================================

Run ID: ad221de1-28fb-4222-b7d4-7474d73a16e3

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
  Unlistable (StrikeNotListed, excluded before the clean filter): 0
  Snapped legs [credit]: {'n': 106, 'width_actual_dist': {'5.0': 4, '10.0': 96, '25.0': 5, '50.0': 1}, 'n_width_not_nominal': 10}
  Snapped legs [debit]: {'n': 110, 'width_actual_dist': {'5.0': 2, '10.0': 106, '25.0': 2}, 'n_width_not_nominal': 4}

----------------------------------------------------------------------
Phase 2: Collecting per-date trade data...
  Processing 110 debit dates...
  Processing 106 credit dates...

  Decision-6 exclusions (no valid entry minute): 0
  Quote validity [debit]: {'n': 110, 'had_invalid_quote': 49, 'valid_minute_fraction_median': 1.0, 'valid_minute_fraction_p05': 0.9974, 'baseline_minute_offset_median': 0, 'baseline_minute_offset_p95': 1, 'baseline_minute_offset_max': 9, 'baseline_offset_gt0': 32}
  Quote validity [credit]: {'n': 106, 'had_invalid_quote': 47, 'valid_minute_fraction_median': 1.0, 'valid_minute_fraction_p05': 0.9949, 'baseline_minute_offset_median': 0, 'baseline_minute_offset_p95': 1, 'baseline_minute_offset_max': 9, 'baseline_offset_gt0': 29}

  Collected: debit=110, credit=106
  Settlement available: debit=107/110, credit=103/106
  Actionable touches: debit=70/110, credit=67/106
  Post-filter out-of-range accepted values (G2, expect 0): 0

----------------------------------------------------------------------
Phase 3: Threshold sweep on TRAIN only...

  Chosen threshold — DEBIT: 0.05  CREDIT: 0.00

DEBIT — Threshold sweep (TRAIN only) [mode=walk-forward]:
       T  n_settled  fill_n  mean_pnl  win%   beat  chosen?
  ────────────────────────────────────────────────────────────
  0.00     106      109        1.70    58%     0.09
  0.05     105      108        1.70    58%     0.10 ← CHOSEN
  0.10     104      107        1.69    58%     0.08
  0.15     103      106        1.69    57%     0.08
  0.20     100      103        1.56    56%    -0.05

CREDIT — Threshold sweep (TRAIN only) [mode=walk-forward]:
       T  n_settled  fill_n  mean_pnl  win%   beat  chosen?
  ────────────────────────────────────────────────────────────
  0.00      26       27       -1.65    62%     0.17 ← CHOSEN
  0.05      19       19       -2.39    58%    -0.57
  0.10      12       12       -2.20    58%    -0.37
  0.15       6        6       -3.09    50%    -1.26
  0.20       3        3       -1.89    67%    -0.07

----------------------------------------------------------------------
Phase 4: Full results (all train; holdout: none (split = universe end))

DEBIT — By distance band (all splits, T=0.05) [mode=walk-forward]:
  band    part        n      pnl   win%  [lo–hi 95%]        base     beat
  ──────────────────────────────────────────────────────────────────────
  near    train      38     2.04    68%  [ 53%– 81%]     2.07    -0.03
  mid     train      36     1.10    56%  [ 40%– 70%]     0.80     0.30
  far     train      31     1.99    48%  [ 32%– 65%]     1.96     0.03
  all     train     105     1.70    58%  [ 49%– 67%]     1.61     0.10
  holdout: none (split = universe end)

CREDIT — By distance band (all splits, T=0.00) [mode=walk-forward]:
  band    part        n      pnl   win%  [lo–hi 95%]        base     beat
  ──────────────────────────────────────────────────────────────────────
  near    train       3    -8.00     0%  [  0%– 56%]    -2.73    -5.28
  mid     train      10    -0.34    70%  [ 40%– 89%]    -1.61     1.27
  far     train      13    -1.20    69%  [ 42%– 87%]    -0.84    -0.36
  all     train      26    -1.65    62%  [ 43%– 78%]    -1.82     0.17
  holdout: none (split = universe end)

DEBIT — By post-touch pattern (TRAIN only, T=0.05) [mode=walk-forward]:
  (n with pattern_label=93, n without=17)
  pattern                           n      pnl   win%  [lo–hi 95%]        base     beat
  ──────────────────────────────────────────────────────────────────────
  mixed                            43     1.31    56%  [ 41%– 70%]     1.05     0.26
  overshoot-then-revert             1     4.35   100%  [ 21%–100%]     4.35     0.00
  stepping-stone                   46     1.50    54%  [ 40%– 68%]     1.43     0.06
  ──────────────────────────────────────────────────────────────────────
  (labeled)                        90     1.44    56%  [ 45%– 65%]     1.28     0.16
  (unlabeled)                      15     3.30    73%  [ 48%– 89%]     3.35    -0.05

CREDIT — By post-touch pattern (TRAIN only, T=0.00) [mode=walk-forward]:
  (n with pattern_label=89, n without=17)
  pattern                           n      pnl   win%  [lo–hi 95%]        base     beat
  ──────────────────────────────────────────────────────────────────────
  mixed                             7     1.29    86%  [ 49%– 97%]    -1.01     2.30
  overshoot-then-revert             0        —      —  [   —–   —]        —        —
  stepping-stone                   10    -1.65    70%  [ 40%– 89%]    -1.99     0.34
  ──────────────────────────────────────────────────────────────────────
  (labeled)                        17    -0.44    76%  [ 53%– 90%]    -1.67     1.23
  (unlabeled)                       9    -3.95    33%  [ 12%– 65%]    -2.61    -1.33

DEBIT — Touch resolution breakdown (TRAIN only, T=0.05) [mode=walk-forward]:
  resolution                      n   touch_exit  close_pnl   base_close
  ─────────────────────────────────────────────────────────────────
  rth_touch                       41        1.11        3.47        3.44
  gap_touch                       29        1.07        3.25        3.20
  afterhours_touch_retraced       15           —        3.36        3.36
  no_touch                        25           —       -3.65       -4.08

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
  near   debit_beat=  -0.03  credit_beat=  -5.28  → DEBIT leads by 5.25
  mid    debit_beat=   0.30  credit_beat=   1.27  → CREDIT leads by 0.97
  far    debit_beat=   0.03  credit_beat=  -0.36  → DEBIT leads by 0.39
  READ: Structure crossover = distance band where debit stops leading and credit starts.

── Summary D: Engine hypothesis (decision #13) ──
  DEBIT engine check:  pattern_match (n=46) pnl=1.50  vs no_match (n=44) pnl=1.38
  → No clear separation by pattern for debit. Engine hypothesis inconclusive.
  DEBIT decision-9 read [mode=walk-forward]: match−no_match = +0.12 pts, bootstrap 95% CI [-2.06, +2.29] (1000 resamples, seed=20260905) → INCONCLUSIVE
  credit: engine hypothesis UNTESTABLE (n_match=0, n_no_match=17)

----------------------------------------------------------------------
Phase 7: Persisting aggregate stats to bt_edge_backtest_results...
  ✓ bt_edge_backtest_results created/verified.
  debit: holdout: none (split = universe end)
  credit: holdout: none (split = universe end)
  ✓ Aggregate stats written.

======================================================================
Step 4 complete in 321s
======================================================================
```

## What changed

- Halted at G1 on the first attempt (snapped legs never fetched), resumed with a spec amendment: **Step 1a** captures the snapped legs for the 21 affected trades; G1 corrected to debit 110 / credit 106 with a −3 tolerance for listed 404s; **G1b** added (0 unlisted legs).
- `scripts/cr_ap_capture_snapped_legs.py` (new): explicit date list, snapped legs via `snap_vertical_legs`, entry-day + settlement + the harness's 90-minute touch window for debit dates, backfill role, `cr_id='CR-AP-capture'`, no read. Run: 14 entry + 13 settlement + 7 touch windows, 20,898 bars, 0 exceptions, 1 404 (2025-06-11 settlement window on the 2025-07-03 half-day; ES-settled, no effect).
- `scripts/cr_ah_step4_analysis.py`: one-line `Counter` import (CR-AO's Phase 1 report aborted every run without it) — the only code change to the analysis path; a scope deviation from "run-only", recorded.
- Run `ad221de1` persisted as `cr_id='CR-AP'` — **the citable walk-forward, clean-quote, listed-strike reference.** All gates pass; G1 exact. Two earlier CR-AP run rows remain: `d774351b` aborted (NameError), `bb39a829` stuck `running` (killed before persistence; see [[bt-backfill-runs-audit-record-fidelity]]).
- Result: debit reference unchanged in substance (all-train mean +1.70, beat +0.10; mid +0.30 is still the only band with a beat); credit mean P&L negative in every band; the credit beat cells are not readable while widths are pooled.

## Decisions

- **Resumed rather than re-planned** when the halt cause was a missing data step: the spec amendment landed before any code, the capture is a sibling of the existing path, and Step 1 ran unchanged.
- **Touch window taken from the harness, not guessed**: `get_touch_pos_val` reads `[touch_datetime_pt, +90 min]`; `detect_touch` supplies the RTH touch minute or the next 06:30 PT open. The capture calls the same functions.
- **The `Counter` hotfix stays in this branch** (needed for any Step 4 run); flagged as the one code change.
- **Reported the credit reversals as a width-pooling artifact** with the per-trade evidence rather than as a finding; per decision 2 the persisted cells pool widths, and the spec carries the subset counts so the pooling decision can be revisited.
- **Left the stuck `running` row alone** and documented the gap; no out-of-band UPDATE from the backfill role.

## Open questions

- **Width normalisation for beat.** With `width_actual ∈ {5, 10, 25, 50}` in one cell, `mean − baseline` mixes scales; the credit mid/all sign flips come from 4 wide spreads. Either report beat per unit of width (or per max-loss), or keep `width_actual = 10` as the reference cell and the rest as a labelled subset. Decide before CR-Y shows CR-AP cells.
- **Snapping produces 5-wide spreads** when the 5-point wing exists but the 10-point one does not (the tie rule picks the strike nearer the anchor). A 5-wide is a different trade; consider preferring the wider side when the two candidates are equidistant from the nominal wing, or requiring `width_actual ≥ width_nominal`.
- **Half-day settlement windows**: the 12:50–13:00 PT settlement window does not exist on early-close sessions (2025-07-03); harmless here (ES-settled) but any option-quote settlement path would 404. Cross-ref the holiday open question.
- CR-AI re-run with clean quotes + snapping (carried from CR-AO); `TodaySetup` display `strike_spx` (carried); the attended web deploy (carried).
