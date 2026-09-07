# CR-AR — Pair snapping with width cap; width-normalised beat; reference cr_id=CR-AR

> Authority: vault session note `Dash/sessions/2026-09-06 - CR-AR — Pair-Snapping with Width Cap.md`
> Branch: `feat/CR-AR-pair-snapping-width-cap` (off `origin/main` a076d49, the CR-AQ merge; CR-AP merge 122a763 is an ancestor)
> Scope: `snap_spread_to_listed` + `StructureNotListed` replacing per-leg snapping for verticals; hard width cap at 2× nominal; harness `unlistable` status; Proposals `listed:false` payload + card; `beat_per_point` / `close_pnl_per_point`; re-run of CR-AP's configuration as `cr_id='CR-AR'`. **Touches the live Proposals path. No deploy.**
> Mode: unattended through PR; halts at the first STOP gate that misses.

## Problem

`[Certain]` CR-AO snaps each leg independently to the nearest listed strike. Where the chain is sparse (non-monthly expiries at ~15 DTE: 20% five-point completeness, and some regions only on a 25- or 50-point grid), a 10-point intent becomes a 25- or 50-point structure. CR-AP found four such credit spreads; their baselines run to −27.69 and, pooled with 10-wides in a points-denominated beat column, flipped the credit mid/all signs. The live leg builder has the same behaviour. A 50-point spread is five times the capital at risk of the trade Dash is proposing; it is not the same trade.

`[Certain]` The `beat` column is in points regardless of width, so any non-10 width distorts pooled cells.

## Locked decisions

| # | Decision | Value |
|---|---|---|
| 1 | Pair snapping | `snap_spread_to_listed(target, width_nominal, side, direction, chain)` replaces per-leg snapping for verticals. Candidate pairs = all `(k_low, k_high)` from the prior-close chain for the expiry with `k_high − k_low ∈ [width_nominal, 2 × width_nominal]`. Choose the pair minimising `|anchor − intended_anchor|` where the anchor is the leg nearest the target (debit: `k_high` → target; credit: `k_low` → target), tie-break: (a) width closest to nominal, (b) toward the magnet direction. If no candidate pair → `StructureNotListed`. |
| 2 | Width cap | Hard: `width_actual ≤ 2 × width_nominal`. Never wider. Narrower than nominal is allowed only if nothing at ≥ nominal exists within the cap, and is recorded. |
| 3 | Harness | `StructureNotListed` → trade status `unlistable`, excluded from the clean sample with reason, counted in the smoke dict. |
| 4 | Live | Proposals returns `listed: false, reason: 'no_listed_structure'` for that structure; the card shows "no listed structure at this expiry" instead of legs. Never a widened spread without `width_actual` shown prominently; at 2× it shows the width in the card title. |
| 5 | Beat normalisation | `bt_edge_backtest_results` gains `beat_per_point = beat / width_actual` (persisted per cell as the width-weighted mean); the printed by-band table shows both `beat` and `beat_per_point`, and cells pool by `beat_per_point`. `close_pnl` stays in points but the table also shows `close_pnl_per_point`. |
| 6 | Reference | Re-run CR-AP's configuration (run-only, ~5 min) after the fix, `cr_id='CR-AR'`. Expect the four wide credit trades to become ≤ 20-wide or `unlistable`; expect debit unchanged except any of its 5 re-priced trades that were > 20-wide (CR-AP spec lists them). CR-AR becomes the citable reference. |
| 7 | Not in scope | Iron condors (four legs) — the pair function is written so a condor is two pair-snaps, but no condor path exists yet; the CR-AI SQL-mid path (already patched for quotes; snapping there follows the shared function automatically if it calls it — Step 0 confirms). |

## Gates

| Gate | Expected | On miss |
|---|---|---|
| G0 — `main` contains the CR-AP merge; current callers of `snap_to_listed_strike` listed | yes | STOP |
| G1 — tests: nominal pair found; only 2× pair exists → chosen and flagged; only 3× exists → `StructureNotListed`; tie-breaks; debit vs credit anchor; narrower-than-nominal fallback | pass; suites pass | STOP |
| G2 — Proposals unit test: unlistable structure → `listed:false` payload, no legs | pass | STOP |
| G3 — re-run: selection 50/50/50; clean debit 110 ± 3 / credit 106 ± 3 (unlistable listed by date) | yes | STOP beyond ±3 |
| G4 — `max(width_actual) ≤ 20` across all trades | yes | STOP |
| G5 — unaffected trades identical to CR-AP to the cent | yes | STOP |
| G6 — no holdout P&L in logs | yes | STOP and redact |

## Kickoff prompt

```
CR-AR — pair snapping with width cap. Unattended through PR. Authority: vault
note "Dash/sessions/2026-09-06 - CR-AR — Pair-Snapping with Width Cap.md".
Read it. Halt only at STOP gates; on halt write "## Halt", commit, push, end.
DO NOT deploy anything.

Interpreter: apps/web/.venv/bin/python for runs; Rosetta repo venv for tests.
Branch: git fetch; git checkout -b feat/CR-AR-pair-snapping-width-cap origin/main.

Commit 1 — spec freeze from the note.
Step 0 — G0: list every caller of snap_to_listed_strike (harness leg builder,
  capture scripts, CR-AI, Proposals). From the CR-AP spec, list the 14
  non-10-width trades with their widths. Commit 2.
Commit 3 — snap_spread_to_listed + StructureNotListed + tests (decisions 1–2).
Commit 4 — wire: harness (decision 3, width_actual/width_nominal retained),
  capture scripts, CR-AI if it snaps, Proposals (decision 4) + test (G2).
  Per-leg snap_to_listed_strike stays for single-leg callers only; verticals
  must go through the pair function (grep to confirm no vertical builder
  still calls the per-leg one).
Commit 5 — beat_per_point / close_pnl_per_point (decision 5): migration or
  column add on bt_edge_backtest_results under the owner URL (log it), the
  INSERT, and the printed tables. Run both suites (G1).
Step 1 — re-run:
  PYTHONUNBUFFERED=1 <python> -u scripts/cr_ah_step4_analysis.py \
    --universe-end 2026-06-05 --split-date 2026-06-05 --cr-id CR-AR \
    --structural-prob-mode walk-forward 2>&1 \
    | tee scripts/logs/cr_ar_step4_$(date +%Y%m%d_%H%M%S).log
  G3–G6. Commit 6: run_id; by-band tables with beat and beat_per_point;
  width_actual distribution; unlistable list; restatement vs CR-AP (holds /
  weakens / reverses on beat_per_point) with the four credit trades' before/after.
Wrap — Commit 7: spec What changed / Decisions / Open questions. Vault: run
  log; mid-band row (beat_per_point); Sessions MOC. Push, open PR
  "CR-AR — Pair snapping with width cap; width-normalised beat; reference
  cr_id=CR-AR". DO NOT MERGE. Print PR URL, run id, restatement, max width.
```

## Execution note (worktree)

Run from the git worktree `.claude/worktrees/cr-ar` (branch `feat/CR-AR-pair-snapping-width-cap`) so the CR-AS capture session on the main checkout is undisturbed; the branch step of the kickoff prompt is satisfied by `git worktree add … -b feat/CR-AR-pair-snapping-width-cap origin/main`. Interpreters are the main checkout's: `/Users/ryan/code/volatility-monorepo/apps/web/.venv/bin/python` (arm64, runs) and `arch -x86_64 /Users/ryan/code/volatility-monorepo/.venv/bin/python -m pytest` (Rosetta, suites); `.env` is an untracked symlink to the main checkout's file.
