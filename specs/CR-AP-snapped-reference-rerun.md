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

