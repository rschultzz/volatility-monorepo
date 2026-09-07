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

## Step 0 findings (read-only, 2026-09-07)

Interpreter: `/Users/ryan/code/volatility-monorepo/apps/web/.venv/bin/python` (arm64) for the DB reads; Rosetta repo venv for the suites. Worktree `.claude/worktrees/cr-ar`.

| Gate | Expected | Actual | Result |
|---|---|---|---|
| G0 — `main` contains the CR-AP merge | yes | `origin/main` = a076d49 (PR #49 merge); `git branch -r --contains 122a76356ff3bbb4076e1e22d4b3efd7d837e6e0` → `origin/main` ✓; branch cut from a076d49 | PASS |
| G0 — callers of `snap_to_listed_strike` listed | yes | **no production caller** — only `test_strikes.py`. Every vertical builder calls `snap_vertical_legs` (per-leg anchor-then-wing); the live path calls `snap_to_candidates` directly. Full inventory below. | PASS |

### Callers of the snapping family on `origin/main`

| Caller | Call | CR-AR action |
|---|---|---|
| `scripts/cr_ah_step4_analysis.py:389` `filter_clean_for_structure` (harness leg builder) | `snap_vertical_legs(target, ±10, …, toward=spot)`; `except StrikeNotListed` → `UNLISTABLE` | → `snap_vertical_pair(target, 10, structure, …)`; `StructureNotListed` → status `unlistable`, reason, smoke counts (decision 3) |
| `scripts/cr_ap_capture_snapped_legs.py:118,122` (capture) | `snap_vertical_legs` debit / credit | → `snap_vertical_pair` (the Step 1a capture uses this) |
| `scripts/cr_am_holdout_leg_capture.py:167-168` (holdout capture) | `snap_vertical_legs` debit / credit | → `snap_vertical_pair` |
| `scripts/cr_ai_stage2_backfill.py:251` `load_clean_dates` (CR-AI Stage 2) | `snap_vertical_legs(target, −10, …)` — **it does snap** (decision 7 question answered) | → `snap_vertical_pair(target, 10, "debit", …)`; not re-run |
| `packages/shared/options_cache/pricing.py:460-468` `_snap_leg_strikes` (live Proposals leg builder) | vertical branch: `snap_to_candidates` anchor, then `snap_to_candidates` wing on its side (per-leg, unbounded width) | → `snap_spread_to_listed` for the vertical branch; `listed:false` / `no_listed_structure` (decision 4); per-leg `snap_to_candidates` stays for non-vertical structures |
| `packages/shared/options_cache/strikes.py` `snap_to_listed_strike` | — | stays for single-leg callers (none today) |
| `scripts/cr_ah_step2_stratified_backfill.py` `round_to_5pt` | historical June backfill, no snapping | untouched |

`snap_vertical_legs` / `SnappedVertical` are removed in Commit 4 so no vertical builder can fall back to per-leg snapping (grep-verified).

### The 14 `width_actual ≠ 10` trades in CR-AP (from `specs/CR-AP-snapped-reference-rerun.md`)

| structure | date | band | CR-AP legs short/other | width |
|---|---|---|---|---|
| credit | 2024-03-05 | near | 5175/5180 | 5 |
| credit | 2024-07-11 | near | 5675/5680 | 5 |
| credit | 2024-09-17 | mid | 5700/5750 | **50** |
| credit | 2024-09-23 | near | 5750/5775 | 25 |
| credit | 2024-11-25 | far | 6070/6075 | 5 |
| credit | 2025-03-26 | mid | 5850/5875 | 25 |
| credit | 2025-05-16 | mid | 6000/6025 | 25 |
| credit | 2025-06-11 | near | 6075/6080 | 5 |
| credit | 2025-06-24 | mid | 6100/6125 | 25 |
| credit | 2026-04-07 | near | 6700/6725 | 25 |
| debit | 2024-07-11 | near | 5675/5670 | 5 |
| debit | 2024-09-17 | mid | 5700/5675 | 25 |
| debit | 2025-03-26 | mid | 5850/5825 | 25 |
| debit | 2025-06-11 | near | 6075/6070 | 5 |

Debit 4 (5 ×2, 25 ×2), credit 10 (5 ×4, 25 ×5, 50 ×1). The four wide credit trades of the Problem statement are 2024-09-17 (50), 2024-09-23, 2025-03-26, 2025-05-16, 2025-06-24 and 2026-04-07 (25) — six 25/50-wides; the "four" in the note are the ones whose baselines dominate the mid/all cells (09-17, 03-26, 05-16, 04-07).

### What decision 1 does to them, read from the prior-close chain (scratchpad `cr_ar_step0.py`, prototype of the rule)

Chain = strikes within ±30 of the target listed for the 15-bday expiry at the prior close (`orats_oi_gamma`); "rows" = entry-day minute rows in `orats_options_minute` for the predicted legs.

| structure | date | target | chain (±30) | CR-AP pair (w) | decision-1 pair, unbounded anchor (w) | anchor shift | rows short/other |
|---|---|---|---|---|---|---|---|
| credit | 2024-03-05 | 5173.25 | 5145 … 5175, 5180, 5190, 5200 | 5175/5180 (5) | 5175/5190 (15) | 1.8 | 390 / **0** |
| credit | 2024-07-11 | 5676.07 | 5650, 5660, 5670, 5675, 5680, 5700 | 5675/5680 (5) | 5680/5700 (20) | 3.9 | 381 / **0** |
| credit | 2024-09-17 | 5708.23 | **5700 only** | 5700/5750 (50) | 5640/5650 (10) | **68.2** | 0 / 0 |
| credit | 2024-09-23 | 5754.32 | 5725, 5730, 5740, 5750, 5775 | 5750/5775 (25) | 5740/5750 (10) | 14.3 | 390 / 390 |
| credit | 2024-11-25 | 6067.52 | 6040, 6050, 6060, 6070, 6075 | 6070/6075 (5) | 6060/6070 (10) | 7.5 | 390 / 390 |
| credit | 2025-03-26 | 5848.20 | 5820, 5825, 5850, 5875 | 5850/5875 (25) | 5810/5820 (10) | **38.2** | 0 / 0 |
| credit | 2025-05-16 | 6001.45 | 5975, 5980, 5990, 6000, 6025 | 6000/6025 (25) | 5990/6000 (10) | 11.5 | 390 / 390 |
| credit | 2025-06-11 | 6076.27 | 6050 … 6075, 6080, 6090, 6100 | 6075/6080 (5) | 6075/6090 (15) | 1.3 | 390 / **0** |
| credit | 2025-06-24 | 6102.35 | 6075, 6080, 6090, 6100, 6125 | 6100/6125 (25) | 6090/6100 (10) | 12.4 | 390 / 390 |
| credit | 2026-04-07 | 6698.30 | 6670 … 6690, 6700, 6725 | 6700/6725 (25) | 6690/6700 (10) | 8.3 | 390 / 390 |
| debit | 2024-07-11 | 5676.07 | 5650 … 5680, 5700 | 5675/5670 (5) | 5675/5660 (15) | 1.1 | 381 / **0** |
| debit | 2024-09-17 | 5708.23 | **5700 only** | 5700/5675 (25) | 5650/5640 (10) | **58.2** | 0 / 0 |
| debit | 2025-03-26 | 5848.20 | 5820, 5825, 5850, 5875 | 5850/5825 (25) | 5825/5810 (15) | **23.2** | 390 / **0** |
| debit | 2025-06-11 | 6076.27 | 6050 … 6100 | 6075/6070 (5) | 6075/6060 (15) | 1.3 | 390 / **0** |

Two things the note did not anticipate:

1. **Decision 1 has no bound on the anchor.** "Candidate pairs = all (k_low, k_high) from the chain" lets the rule walk away from the target until it finds a pair inside the width cap: 2024-09-17 (chain near 5708 is `[5700]` alone) becomes a 5640/5650 credit spread 68 points below the wall; 2025-03-26 becomes 5810/5820, 38 points away. Those are not the trade either — the width cap exists because a 50-wide "is not the same trade", and a structure two implied moves from its target fails the same test.
2. **Nine of the re-snapped legs were never captured** (CR-AP's Step 1a fetched only the per-leg-snapped strikes), so without a capture step the clean filter drops those trades and G3 misses (credit would fall to ≤ 101).

## Spec amendments (Step 0, before any implementation code)

**A1 — anchor tolerance.** `snap_spread_to_listed` only considers pairs whose anchor lies within `2 × width_nominal` of the intended anchor (`max_anchor_shift`, default 2 × nominal, a parameter). Beyond that → `StructureNotListed`. Rationale: the same "same trade" argument as the width cap, expressed in the same unit; 20 points covers any 25-point grid (max shift 12.5) and every anchor move CR-AP made (≤ 10), while refusing the 38- and 68-point relocations above. Under A1 the 14 trades resolve as:

| structure | date | pair (w) | note |
|---|---|---|---|
| credit | 2024-03-05 | 5175/5190 (15) | capture 5190 |
| credit | 2024-07-11 | 5680/5700 (20) | at the cap; capture 5700 |
| credit | 2024-09-17 | **unlistable** | chain `[5700]` near the target; 5640 is 68 away |
| credit | 2024-09-23 | 5740/5750 (10) | anchor moved 14 |
| credit | 2024-11-25 | 6060/6070 (10) | |
| credit | 2025-03-26 | **unlistable** | 5850's only wing is 5875 (25); 5810 is 38 away |
| credit | 2025-05-16 | 5990/6000 (10) | |
| credit | 2025-06-11 | 6075/6090 (15) | capture 6090 |
| credit | 2025-06-24 | 6090/6100 (10) | anchor moved 12 |
| credit | 2026-04-07 | 6690/6700 (10) | |
| debit | 2024-07-11 | 5675/5660 (15) | capture 5660 |
| debit | 2024-09-17 | **unlistable** | as credit |
| debit | 2025-03-26 | **unlistable** | 5825 is 23 away |
| debit | 2025-06-11 | 6075/6060 (15) | capture 6060 |

Expected clean sample: **debit 108 / credit 104** (2 unlistable each) — inside G3's ± 3. Expected `max(width_actual)` = 20 (G4). The other 202 trades are 10-wide with the anchor at the nearest listed strike under both rules, so they must be identical to CR-AP (G5).

**A2 — Step 1a, snapped-leg capture.** Before Step 1, run `scripts/cr_ap_capture_snapped_legs.py` (now on the pair function) for the three dates whose re-snapped wing has no entry-day rows — `--dates 2024-03-05,2024-07-11,2025-06-11 --debit-dates 2024-07-11,2025-06-11` — with `--cr-id CR-AR-capture` (backfill role, `bt_backfill_runs` row, entry-day + settlement + harness touch window for debit dates, no read). Legs: 5190 C 2024-03-26; 5660 C / 5700 C 2024-08-01; 6060 C / 6090 C 2025-07-03 (the 07-03 half-day settlement window will 404 as in CR-AP; ES-settled, harmless). **G1a-pre**: every predicted leg for the 12 listed trades has entry-day rows, or is a listed ORATS 404.

**G3 restated** (no change to the tolerance): selection 50/50/50; clean debit 110 ± 3 / credit 106 ± 3 with the unlistable trades listed by date; expected 108 / 104.

**Decision 5 reading recorded here so the run is reviewable:** the per-cell `beat_per_point` is the *width-weighted* mean of per-trade `beat_i / width_i`, i.e. `Σ(close_pnl_i) / Σ(width_i) − Σ(baseline_i) / Σ(width_i)` (each sum over the trades that carry that value), which equals `beat / 10` exactly when every width is 10. The threshold sweep (Phase 3) keeps choosing on points-`beat` so CR-AP's configuration is re-run unchanged; both columns print, and the cells persist both.
