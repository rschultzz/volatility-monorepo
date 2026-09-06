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
