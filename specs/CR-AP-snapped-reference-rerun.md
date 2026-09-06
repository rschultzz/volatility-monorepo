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
