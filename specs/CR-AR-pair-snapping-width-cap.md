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

## Commit 5 — `beat_per_point` / `close_pnl_per_point` (decision 5)

Migration `infra/sql/bt_edge_backtest_results_per_point.sql` applied 2026-09-07 under the owner URL via `scripts/cr_ar_run_migration.py` (schema_change class; `DATABASE_URL`, role `rschultz` = table owner). Log:

```
Connected as rschultz; bt_edge_backtest_results owner = rschultz
new columns already present before migration: none
Applying infra/sql/bt_edge_backtest_results_per_point.sql (1933 bytes)
ALTER TABLE bt_edge_backtest_results
  ADD COLUMN IF NOT EXISTS close_pnl_per_point FLOAT,
  ADD COLUMN IF NOT EXISTS baseline_per_point  FLOAT,
  ADD COLUMN IF NOT EXISTS beat_per_point      FLOAT,
  ADD COLUMN IF NOT EXISTS mean_width_actual   FLOAT;
applied ✓
\d bt_edge_backtest_results — 24 columns (… beat_baseline, created_at, close_pnl_per_point, baseline_per_point, beat_per_point, mean_width_actual)
dash_backfill_writer table privileges: ['INSERT', 'SELECT']
rows by cr_id: [('CR-AH', 16), ('CR-AM', 8), ('CR-AN', 8), ('CR-AP', 8)]
```

Harness: `CellStats` carries `Σ width_actual` over the settled-filled and the baseline trades; `fmt_stats` adds `mean_pnl_per_point = Σ close_pnl / Σ width`, `baseline_per_point = Σ baseline / Σ width`, `beat_per_point` (their difference — the width-weighted mean of `beat_i / width_i`, exactly `beat / 10` when every width is 10) and `mean_width`. The sweep table prints `beat/pt`; the by-band table prints `pnl/pt`, `beat/pt` and mean `width`; the pattern tables print `beat/pt`; Summary C shows both. `persist_cell_stats` writes the four new columns; `ensure_catalog_table` also runs the idempotent `ADD COLUMN IF NOT EXISTS` so an un-migrated DB never fails the INSERT. Threshold choice unchanged (points-`beat`, decision 5 reading in Step 0). Pre-CR-AR rows keep NULLs. Unit check: uniform 10-wides → `beat_per_point == beat / 10`; a 10-wide (+1) with a 20-wide (−2) → `beat = −0.5`, `beat_per_point = −1/30`.

## Step 1a — snapped-leg capture (amendment A2), `cr_id='CR-AR-capture'`

Command: `PYTHONUNBUFFERED=1 apps/web/.venv/bin/python -u scripts/cr_ap_capture_snapped_legs.py --cr-id CR-AR-capture --dates 2024-03-05,2024-07-11,2025-06-11 --debit-dates 2024-07-11,2025-06-11` (dry-run plan matched A2 exactly). Log: `scripts/logs/cr_ar_capture_20260907_074827.log` (untracked). Run row `65c24600-d670-47f4-a51e-793639f7b3b1`, `completed`, 14:49 → 15:28 UTC: "captured 3 entry + 2 settlement + 2 touch windows over 3 dates (0 unlistable); bars_written=3986 404s=1 exceptions=0; no P&L computed".

| date | pair(s) | entry-day | settlement | touch window |
|---|---|---|---|---|
| 2024-03-05 | credit 5175/5190 (15) | 782 bars | 22 bars | — |
| 2024-07-11 | debit 5660/5675 (15), credit 5680/5700 (20) | 1528 bars | 44 bars | gap_touch 07-11 06:30 +90 m: 0 written, 2 cache hits |
| 2025-06-11 | debit 6060/6075 (15), credit 6075/6090 (15) | 1564 bars | **404** (2025-07-03 half-day, as in CR-AP; ES-settled) | rth_touch 06-23 12:53 +90 m: 46 bars |

**G1a-pre: PASS** — entry-day rows for the five new legs: 5190 C 03-26 → 390; 5660 C / 5700 C 08-01 → 381 / 381; 6060 C / 6090 C 07-03 → 390 / 390.

## Step 1 — pair-snapped, width-capped walk-forward run, persisted as `cr_id='CR-AR'`

Command: `PYTHONUNBUFFERED=1 apps/web/.venv/bin/python -u scripts/cr_ah_step4_analysis.py --universe-end 2026-06-05 --split-date 2026-06-05 --cr-id CR-AR --structural-prob-mode walk-forward` (CR-AP's configuration; snapping through the pair function by construction). Log: `scripts/logs/cr_ar_step4_20260907_082934.log` (untracked). 1884 s (the DB was shared with the CR-AS capture session and the G5 dump; CR-AP took 321 s). Run row `b07972a5-942b-4551-b17d-fbd0fd76a5d3`, `completed`, 15:29 → 16:00 UTC: "Step 4 complete [mode=walk-forward]: debit=108, credit=106, T_d=0.05, T_c=0.0, 1883s; summary_d={'debit': 'INCONCLUSIVE', 'credit': 'untestable'}". Persisted: 8 rows `cr_id='CR-AR'` with the four per-point columns. **CR-AR is now the citable reference** (decision 6); CR-AH / CR-AM / CR-AN / CR-AP rows kept.

| Gate | Expected | Actual | Result |
|---|---|---|---|
| G3 selection; clean counts; unlistable by date | 50/50/50; debit 110 ± 3 / credit 106 ± 3 | **50 / 50 / 50; debit 108 / credit 106** (bands debit 41/34/33, credit 40/34/32); 14 unlistable entries listed below | PASS |
| G4 `max(width_actual)` | ≤ 20 | **20** (credit 2024-07-11 5680/5700); distribution credit {10: 103, 15: 2, 20: 1}, debit {10: 106, 15: 2}; 0 narrower than nominal | PASS |
| G5 unaffected trades identical to CR-AP to the cent | yes | **202 / 202** identical on fill, close, baseline, touch-exit at T = 0 and T = 0.05 (read-only per-trade dumps of both code states, scratchpad `cr_ar_trade_diag.py`) | PASS |
| G6 no holdout P&L in the log | none | grep of post-2026-06-05 dates → **0** (split = universe end; no holdout exists) | PASS |
| CR-AN G2 post-filter out-of-range values | 0 | 0 | PASS |

Chosen thresholds: DEBIT 0.05 (sweep +0.09 / +0.10 / +0.08 / +0.06 / −0.07), CREDIT 0.00 — same as CR-AP.

### Unlistable (status `unlistable`, 14 entries of the 150 selected dates)

| date | band | structures | why |
|---|---|---|---|
| 2024-08-14 | mid | credit, debit | no pair within [10, 20] with its anchor within 20 of 5562.1 |
| 2024-09-17 | mid | credit, debit | chain near 5708 is `[5700]` alone (CR-AP: 5700/5750 credit, 5700/5675 debit) |
| 2024-10-31 | far | credit, debit | target 6010.9 |
| 2025-03-26 | mid | credit, debit | 5850's only wing is 5875 (CR-AP: 5850/5875, 5850/5825) |
| 2025-05-09 | mid | credit, debit | target 5803.2 |
| 2025-05-12 | far | credit, debit | target 5804.7 |
| 2025-06-04 | mid | credit | target 6052.8 |
| 2025-11-14 | far | credit | target 6992.3 |

Four of these (2024-09-17 and 2025-03-26, both structures) were in the CR-AP clean sample; the other ten were never clean (their per-leg-snapped wings had no quotes), so the clean counts move by −4 (removed) +2 (entered) −0 = credit 106, debit 108.

### Sample composition vs CR-AP

- **Removed (unlistable):** credit 2024-09-17 (was 5700/5750, 50-wide, baseline −27.69), credit 2025-03-26 (5850/5875, 25-wide, close +7.65 / baseline +8.95), debit 2024-09-17 (5700/5675, +11.19), debit 2025-03-26 (5850/5825, −13.60).
- **Entered (credit only):** 2023-09-11 far → 4640/4650 (target 4649.2; 4650 is listed but its nearest listed wing above is beyond the cap, so the anchor moved 9 points to the 10-wide whose legs June's backfill had captured; close +2.55) and 2023-11-10 far → 4490/4500 (target 4508.9; 4510 unlisted, 4500's wing beyond the cap, anchor moved 19; close −8.20). Both anchors within the A1 tolerance; both pairs are the mirror of the debit legs already in the sample.
- **Re-snapped (10):** the table below.

re-snapped trades: 10
| structure | date | band | CR-AP legs (w) | CR-AR legs (w) | CR-AP close / base (T=0) | CR-AR close / base (T=0) | CR-AP close / base (chosen T) | CR-AR close / base (chosen T) |
|---|---|---|---|---|---|---|---|---|
| credit | 2024-03-05 | near | 5175/5180 (5) | 5175/5190 (15) | — / -3.25 | — / -9.70 | — / -3.25 | — / -9.70 |
| credit | 2024-07-11 | near | 5675/5680 (5) | 5680/5700 (20) | — / +2.15 | — / +8.40 | — / +2.15 | — / +8.40 |
| credit | 2024-09-23 | near | 5750/5775 (25) | 5740/5750 (10) | -12.71 / -12.71 | -5.00 / -4.57 | -12.71 / -12.71 | -5.00 / -4.57 |
| credit | 2024-11-25 | far | 6070/6075 (5) | 6060/6070 (10) | — / +2.10 | — / +4.48 | — / +2.10 | — / +4.48 |
| credit | 2025-05-16 | mid | 6000/6025 (25) | 5990/6000 (10) | +0.95 / +10.20 | -5.75 / -5.75 | +0.95 / +10.20 | -5.75 / -5.75 |
| credit | 2025-06-11 | near | 6075/6080 (5) | 6075/6090 (15) | — / — | — / — | — / — | — / — |
| credit | 2025-06-24 | mid | 6100/6125 (25) | 6090/6100 (10) | — / -13.70 | — / -4.35 | — / -13.70 | — / -4.35 |
| credit | 2026-04-07 | near | 6700/6725 (25) | 6690/6700 (10) | — / -16.15 | — / -2.15 | — / -16.15 | — / -2.15 |
| debit | 2024-07-11 | near | 5675/5670 (5) | 5675/5660 (15) | -2.45 / -2.45 | -7.45 / -7.45 | -2.45 / -2.45 | -7.45 / -7.45 |
| debit | 2025-06-11 | near | 6075/6070 (5) | 6075/6060 (15) | — / — | — / — | — / — | — / — |
| credit | 2024-09-17 | mid | 5700/5750 (50) | unlistable | — / -27.69 | — | — / -27.69 | — |
| credit | 2025-03-26 | mid | 5850/5875 (25) | unlistable | +7.65 / +8.95 | — | +7.65 / +8.95 | — |
| debit | 2024-09-17 | mid | 5700/5675 (25) | unlistable | +11.19 / +11.19 | — | +11.19 / +11.19 | — |
| debit | 2025-03-26 | mid | 5850/5825 (25) | unlistable | -13.60 / -13.60 | — | -13.60 / -13.60 | — |


The four credit trades of the Problem statement: 2024-09-17 (50-wide, baseline −27.69) → **unlistable**; 2025-03-26 (25-wide, +7.65 / +8.95) → **unlistable**; 2025-05-16 (25-wide, +0.95 / +10.20) → 5990/6000 10-wide, −5.75 / −5.75; 2026-04-07 (25-wide, — / −16.15) → 6690/6700 10-wide, — / −2.15. The 25-wide baselines that dominated the CR-AP credit mid/all cells are gone.

### By-band tables (all train; `beat` in points, `beat/pt` width-weighted per point of width, mean `width`)

```
DEBIT — By distance band (all splits, T=0.05) [mode=walk-forward]:
  band    part        n      pnl   win%  [lo–hi 95%]        base     beat   pnl/pt  beat/pt  width
  ────────────────────────────────────────────────────────────────────────────────────────────────
  near    train      38     1.91    68%  [ 53%– 81%]     1.95    -0.03    0.189   -0.003   10.1
  mid     train      34     1.23    56%  [ 39%– 71%]     0.91     0.32    0.123    0.032   10.0
  far     train      31     1.99    48%  [ 32%– 65%]     1.96     0.03    0.199    0.003   10.0
  all     train     103     1.71    58%  [ 49%– 67%]     1.62     0.10    0.170    0.010   10.0
  holdout: none (split = universe end)
CREDIT — By distance band (all splits, T=0.00) [mode=walk-forward]:
  band    part        n      pnl   win%  [lo–hi 95%]        base     beat   pnl/pt  beat/pt  width
  ────────────────────────────────────────────────────────────────────────────────────────────────
  near    train       3    -5.43     0%  [  0%– 56%]    -2.17    -3.27   -0.543   -0.335   10.0
  mid     train       9    -1.97    56%  [ 27%– 81%]    -1.35    -0.62   -0.197   -0.062   10.0
  far     train      15    -1.42    67%  [ 42%– 85%]    -0.89    -0.52   -0.142   -0.052   10.0
  all     train      27    -2.05    56%  [ 37%– 72%]    -1.52    -0.52   -0.205   -0.054   10.0
  holdout: none (split = universe end)
```

Every persisted cell has mean `width_actual` 10.0 (debit near 10.1), so `beat/pt` ≈ `beat / 10` throughout; the per-point column changes no sign in CR-AR — its job was to stop CR-AP's 25/50-wides from dominating, and the cap has removed them.

### Restatement CR-AR vs CR-AP (each at its chosen T; beat in points and per point of width)
| structure | band | CR-AP n / pnl / win / base / beat / beat_pp | CR-AR n / pnl / win / base / beat / beat_pp | Δ beat | Δ beat_pp | restatement (beat) | restatement (beat_pp) |
|---|---|---|---|---|---|---|---|
| debit | near | 38 / +2.04 / 68% / +2.07 / -0.03 / -0.003 | 38 / +1.91 / 68% / +1.95 / -0.03 / -0.003 | -0.01 | -0.001 | **holds** | **holds** |
| debit | mid | 36 / +1.10 / 56% / +0.80 / +0.30 / +0.028 | 34 / +1.23 / 56% / +0.91 / +0.32 / +0.032 | +0.02 | +0.004 | **holds** | **holds** |
| debit | far | 31 / +1.99 / 48% / +1.96 / +0.03 / +0.003 | 31 / +1.99 / 48% / +1.96 / +0.03 / +0.003 | +0.00 | +0.000 | **holds** | **holds** |
| debit | all | 105 / +1.70 / 58% / +1.61 / +0.10 / +0.009 | 103 / +1.71 / 58% / +1.62 / +0.10 / +0.010 | +0.00 | +0.000 | **holds** | **holds** |
| credit | near | 3 / -8.00 / 0% / -2.73 / -5.28 / -0.274 | 3 / -5.43 / 0% / -2.17 / -3.27 / -0.335 | +2.01 | -0.061 | **holds** | **holds** |
| credit | mid | 10 / -0.34 / 70% / -1.61 / +1.27 / +0.104 | 9 / -1.97 / 56% / -1.35 / -0.62 / -0.062 | -1.89 | -0.166 | **reverses** | **reverses** |
| credit | far | 13 / -1.20 / 69% / -0.84 / -0.36 / -0.034 | 15 / -1.42 / 67% / -0.89 / -0.52 / -0.052 | -0.16 | -0.018 | **holds** | **holds** |
| credit | all | 26 / -1.65 / 62% / -1.82 / +0.17 / +0.025 | 27 / -2.05 / 56% / -1.52 / -0.52 / -0.054 | -0.69 | -0.080 | **reverses** | **reverses** |

Rule: holds = same sign and ≥ 50 % of the CR-AP magnitude; weakens = same sign, < 50 %; reverses = sign flip. Read:

- **Debit — holds in every cell, on `beat` and on `beat_per_point`.** All-train n 105 → 103, mean +1.70 → +1.71, beat +0.10 → +0.10 (+0.010 / pt); mid +0.30 → +0.32 (+0.032 / pt) is still the only band with a beat; near −0.03, far +0.03 are noise around zero. The two removed debit trades (+11.19 and −13.60 on 25-wides) cancel almost exactly.
- **Credit mid — reverses (+1.27 → −0.62; +0.104 → −0.062 / pt) and credit all — reverses (+0.17 → −0.52; +0.025 → −0.054 / pt).** This is the CR-AP width-pooling artifact being undone, not a new finding: the two 25-wide mid trades with baselines +8.95 / +10.20 are now unlistable / a 10-wide at −5.75, and the 50-wide with baseline −27.69 is unlistable. Credit is now negative on mean P&L **and** on beat in every band (near −3.27, mid −0.62, far −0.52, all −0.52); Summary C reads DEBIT leads in all three bands. The credit fade stays refuted, now on like-for-like widths.
- **Credit near / far — hold** (−5.28 → −3.27; −0.36 → −0.52).

Summary D: debit match − no_match = −0.56 pts, CI [−2.57, +1.58] → INCONCLUSIVE (pattern-match now slightly worse, CI spans zero — consistent with CR-AL … CR-AP); credit UNTESTABLE. Touch resolution: rth_touch touch-exit +1.09 vs close +3.47; gap_touch +1.01 vs +2.74 — hold-to-close still beats touch-exit. Selection bias: neither structure flagged (credit far clean σ 3.09 vs dropped 3.05 — CR-AP's ⚠ on credit far is gone with the sample change).

### Full output (Phases 1–7)

```

======================================================================
CR-AH Step 4 — Two-structure × two-axis analysis
  cr_id=CR-AR  structural-prob mode=walk-forward  train_only=False  no_persist=False  universe_end=2026-06-05  split_date=2026-06-05  seed=20260905
======================================================================

Run ID: b07972a5-942b-4551-b17d-fbd0fd76a5d3

----------------------------------------------------------------------
Phase 1: Loading signal dates and selecting clean subset...
  Universe pinned to trade_date <= 2026-06-05: 375/397 magnet-above dates kept.
  Loaded 374/375 signal entries (skipped 1).
  Stratified selection: 150 dates
  Selection by band/partition: {'far/train': 50, 'mid/train': 50, 'near/train': 50}
  holdout: none (split = universe end)

Filtering to A-bucket clean dates for each structure...
  Credit (target + target+10)...
    Credit clean: 106
  Debit (target + target-10)...
    Debit clean:  108

  Credit by band/partition: {'far/train': 32, 'mid/train': 34, 'near/train': 40}
  Debit  by band/partition: {'far/train': 33, 'mid/train': 34, 'near/train': 41}
  Unlistable (StructureNotListed, status='unlistable', excluded before the clean filter): 14
    credit 2024-08-14 mid: no listed pair within [10, 20] points with its anchor within 20 of target 5562.1 (side credit; 100 strikes listed)
    debit 2024-08-14 mid: no listed pair within [10, 20] points with its anchor within 20 of target 5562.1 (side debit; 100 strikes listed)
    credit 2024-09-17 mid: no listed pair within [10, 20] points with its anchor within 20 of target 5708.225 (side credit; 110 strikes listed)
    debit 2024-09-17 mid: no listed pair within [10, 20] points with its anchor within 20 of target 5708.225 (side debit; 110 strikes listed)
    credit 2024-10-31 far: no listed pair within [10, 20] points with its anchor within 20 of target 6010.875 (side credit; 110 strikes listed)
    debit 2024-10-31 far: no listed pair within [10, 20] points with its anchor within 20 of target 6010.875 (side debit; 110 strikes listed)
    credit 2025-03-26 mid: no listed pair within [10, 20] points with its anchor within 20 of target 5848.2 (side credit; 99 strikes listed)
    debit 2025-03-26 mid: no listed pair within [10, 20] points with its anchor within 20 of target 5848.2 (side debit; 99 strikes listed)
    credit 2025-05-09 mid: no listed pair within [10, 20] points with its anchor within 20 of target 5803.225 (side credit; 83 strikes listed)
    debit 2025-05-09 mid: no listed pair within [10, 20] points with its anchor within 20 of target 5803.225 (side debit; 83 strikes listed)
    credit 2025-05-12 far: no listed pair within [10, 20] points with its anchor within 20 of target 5804.675 (side credit; 80 strikes listed)
    debit 2025-05-12 far: no listed pair within [10, 20] points with its anchor within 20 of target 5804.675 (side debit; 80 strikes listed)
    credit 2025-06-04 mid: no listed pair within [10, 20] points with its anchor within 20 of target 6052.75 (side credit; 106 strikes listed)
    credit 2025-11-14 far: no listed pair within [10, 20] points with its anchor within 20 of target 6992.275 (side credit; 121 strikes listed)
  Snapped pairs [credit]: {'n': 106, 'width_actual_dist': {'10.0': 103, '15.0': 2, '20.0': 1}, 'n_width_not_nominal': 3, 'n_width_narrower': 0, 'width_actual_max': 20.0, 'n_unlistable': 8}
  Snapped pairs [debit]: {'n': 108, 'width_actual_dist': {'10.0': 106, '15.0': 2}, 'n_width_not_nominal': 2, 'n_width_narrower': 0, 'width_actual_max': 15.0, 'n_unlistable': 6}
  max(width_actual) across structures (G4, expect ≤ 20): 20

----------------------------------------------------------------------
Phase 2: Collecting per-date trade data...
  Processing 108 debit dates...
  [20/214] 841s elapsed
  [40/214] 1013s elapsed
  [60/214] 1161s elapsed
  [80/214] 1254s elapsed
  [100/214] 1335s elapsed
  Processing 106 credit dates...
  [120/214] 1416s elapsed
  [140/214] 1518s elapsed
  [160/214] 1616s elapsed
  [180/214] 1702s elapsed
  [200/214] 1802s elapsed

  Decision-6 exclusions (no valid entry minute): 0
  Quote validity [debit]: {'n': 108, 'had_invalid_quote': 48, 'valid_minute_fraction_median': 1.0, 'valid_minute_fraction_p05': 0.9974, 'baseline_minute_offset_median': 0, 'baseline_minute_offset_p95': 1, 'baseline_minute_offset_max': 9, 'baseline_offset_gt0': 32}
  Quote validity [credit]: {'n': 106, 'had_invalid_quote': 49, 'valid_minute_fraction_median': 1.0, 'valid_minute_fraction_p05': 0.9949, 'baseline_minute_offset_median': 0, 'baseline_minute_offset_p95': 1, 'baseline_minute_offset_max': 9, 'baseline_offset_gt0': 31}

  Collected: debit=108, credit=106
  Settlement available: debit=105/108, credit=103/106
  Actionable touches: debit=69/108, credit=67/106
  Post-filter out-of-range accepted values (G2, expect 0): 0

----------------------------------------------------------------------
Phase 3: Threshold sweep on TRAIN only...

  Chosen threshold — DEBIT: 0.05  CREDIT: 0.00

DEBIT — Threshold sweep (TRAIN only) [mode=walk-forward]:
       T  n_settled  fill_n  mean_pnl  win%   beat  beat/pt  chosen?
  ──────────────────────────────────────────────────────────────────────
  0.00     104      107        1.70    59%     0.09    0.009
  0.05     103      106        1.71    58%     0.10    0.010 ← CHOSEN
  0.10     102      105        1.70    58%     0.08    0.008
  0.15     101      104        1.68    57%     0.06    0.006
  0.20      98      101        1.55    56%    -0.07   -0.007

CREDIT — Threshold sweep (TRAIN only) [mode=walk-forward]:
       T  n_settled  fill_n  mean_pnl  win%   beat  beat/pt  chosen?
  ──────────────────────────────────────────────────────────────────────
  0.00      27       28       -2.05    56%    -0.52   -0.054 ← CHOSEN
  0.05      20       20       -2.18    55%    -0.65   -0.067
  0.10      14       14       -2.92    50%    -1.40   -0.142
  0.15       9        9       -2.46    56%    -0.93   -0.096
  0.20       4        4       -1.19    75%     0.33    0.031

----------------------------------------------------------------------
Phase 4: Full results (all train; holdout: none (split = universe end))

DEBIT — By distance band (all splits, T=0.05) [mode=walk-forward]:
  band    part        n      pnl   win%  [lo–hi 95%]        base     beat   pnl/pt  beat/pt  width
  ────────────────────────────────────────────────────────────────────────────────────────────────
  near    train      38     1.91    68%  [ 53%– 81%]     1.95    -0.03    0.189   -0.003   10.1
  mid     train      34     1.23    56%  [ 39%– 71%]     0.91     0.32    0.123    0.032   10.0
  far     train      31     1.99    48%  [ 32%– 65%]     1.96     0.03    0.199    0.003   10.0
  all     train     103     1.71    58%  [ 49%– 67%]     1.62     0.10    0.170    0.010   10.0
  holdout: none (split = universe end)

CREDIT — By distance band (all splits, T=0.00) [mode=walk-forward]:
  band    part        n      pnl   win%  [lo–hi 95%]        base     beat   pnl/pt  beat/pt  width
  ────────────────────────────────────────────────────────────────────────────────────────────────
  near    train       3    -5.43     0%  [  0%– 56%]    -2.17    -3.27   -0.543   -0.335   10.0
  mid     train       9    -1.97    56%  [ 27%– 81%]    -1.35    -0.62   -0.197   -0.062   10.0
  far     train      15    -1.42    67%  [ 42%– 85%]    -0.89    -0.52   -0.142   -0.052   10.0
  all     train      27    -2.05    56%  [ 37%– 72%]    -1.52    -0.52   -0.205   -0.054   10.0
  holdout: none (split = universe end)

DEBIT — By post-touch pattern (TRAIN only, T=0.05) [mode=walk-forward]:
  (n with pattern_label=91, n without=17)
  pattern                           n      pnl   win%  [lo–hi 95%]        base     beat  beat/pt
  ────────────────────────────────────────────────────────────────────────────────
  mixed                            42     1.67    57%  [ 42%– 71%]     1.40     0.27    0.027
  overshoot-then-revert             1     4.35   100%  [ 21%–100%]     4.35     0.00    0.000
  stepping-stone                   45     1.17    53%  [ 39%– 67%]     1.10     0.07    0.006
  ────────────────────────────────────────────────────────────────────────────────
  (labeled)                        88     1.44    56%  [ 45%– 66%]     1.28     0.16    0.016
  (unlabeled)                      15     3.30    73%  [ 48%– 89%]     3.35    -0.05   -0.005

CREDIT — By post-touch pattern (TRAIN only, T=0.00) [mode=walk-forward]:
  (n with pattern_label=89, n without=17)
  pattern                           n      pnl   win%  [lo–hi 95%]        base     beat  beat/pt
  ────────────────────────────────────────────────────────────────────────────────
  mixed                             8    -0.53    75%  [ 41%– 93%]    -1.34     0.81    0.081
  overshoot-then-revert             0        —      —  [   —–   —]        —        —        —
  stepping-stone                   10    -1.65    70%  [ 40%– 89%]    -1.09    -0.56   -0.059
  ────────────────────────────────────────────────────────────────────────────────
  (labeled)                        18    -1.15    72%  [ 49%– 88%]    -1.25     0.10    0.007
  (unlabeled)                       9    -3.84    22%  [  6%– 55%]    -2.93    -0.90   -0.090

DEBIT — Touch resolution breakdown (TRAIN only, T=0.05) [mode=walk-forward]:
  resolution                      n   touch_exit  close_pnl   base_close
  ─────────────────────────────────────────────────────────────────
  rth_touch                       41        1.09        3.47        3.44
  gap_touch                       28        1.01        2.74        2.72
  afterhours_touch_retraced       15           —        3.36        3.36
  no_touch                        24           —       -3.23       -3.69

DEBIT — Selection bias check (far band, decision #11) [mode=walk-forward]:
  far/all: n=50  clean: n=33 (mean σ=3.16)  dropped: n=17 (mean σ=2.90)
  ✓ No significant selection bias detected in far band.

CREDIT — Selection bias check (far band, decision #11) [mode=walk-forward]:
  far/all: n=50  clean: n=32 (mean σ=3.09)  dropped: n=18 (mean σ=3.05)
  ✓ No significant selection bias detected in far band.

======================================================================
SUMMARY READS A/B/C/D
======================================================================
[mode=walk-forward]

── Summary A: Debit near-band holdout ──
  holdout: none (split = universe end)

── Summary B: Credit near-band holdout ──
  holdout: none (split = universe end)

── Summary C: Structure crossover by distance (TRAIN, close P&L beat) ──
  near   debit_beat=  -0.03  credit_beat=  -3.27  (per point: -0.003 / -0.335)  → DEBIT leads by 3.23
  mid    debit_beat=   0.32  credit_beat=  -0.62  (per point: 0.032 / -0.062)  → DEBIT leads by 0.94
  far    debit_beat=   0.03  credit_beat=  -0.52  (per point: 0.003 / -0.052)  → DEBIT leads by 0.56
  READ: Structure crossover = distance band where debit stops leading and credit starts.

── Summary D: Engine hypothesis (decision #13) ──
  DEBIT engine check:  pattern_match (n=45) pnl=1.17  vs no_match (n=43) pnl=1.73
  → Pattern filter HURTS debit (match performs WORSE). Engine rule may be wrong.
  DEBIT decision-9 read [mode=walk-forward]: match−no_match = -0.56 pts, bootstrap 95% CI [-2.57, +1.58] (1000 resamples, seed=20260905) → INCONCLUSIVE
  credit: engine hypothesis UNTESTABLE (n_match=0, n_no_match=18)

----------------------------------------------------------------------
Phase 7: Persisting aggregate stats to bt_edge_backtest_results...
  ✓ bt_edge_backtest_results created/verified.
  debit: holdout: none (split = universe end)
  credit: holdout: none (split = universe end)
  ✓ Aggregate stats written.

======================================================================
Step 4 complete in 1884s
======================================================================
```
