# CR-DL — Recompute derived tables for the March 2026 patched window

Branch `feat/CR-DL-march-2026-derived-recompute` (worktree `../volatility-monorepo-cr-dl`) off `origin/main` `ce13906` (CR-DJ merge).
Kickoff: chat 2026-10-29 (Ryan). Data CR (`data_safety_class: write_backfill`).
Source: vault open-question `march-2026-roll-patch-derived-recompute`.

## Problem (from the open-question)

CR-DJ (run `2adcc833-5f7f-4406-b645-0ce1b7d61bbc`) replaced the bars of 2026-03-15 22:00 → 03-20 20:59 UTC (dying ES H26, ~50 pts below the front month) with Databento ES M26 in `ironbeam_es_1m_bars` and `es_minutes`. Rows computed from the old H26 bars before the patch still hold H26-derived values. CR-DJ Step 0d inventory:

| Candidate | Status |
| --- | --- |
| `bt_daily_outcomes` | **84 active rows** (21 trade dates 2025-12-19 → 2026-03-20 × 4 feature versions) whose horizon includes a window session — recompute in place, CR-BG pattern (`scripts/cr_bg_recompute_missed_roll.py`) |
| `bt_daily_features.vol_risk_premium` | 23 non-null rows 2026-03-16 → 04-17 (ES daily closes, written by `cr_d_backfill_vol_features.py`) — recompute the affected ones |
| Views (`es_minutes_with_features*`), live chart / REVIEW readers | self-heal — no action |
| `bt_signals`, `es_minute_features` | 0 rows — no action |
| `ironbeam_es_flow_1s` H26 rows | symbol-tagged; stay (CR-BF / CR-BG precedent) |

## Kickoff (frozen)

Branch first (after merging CR-DJ). Spec from the vault open-question. This is a data CR: standing-rulings conformance table required.

**STEP 0 (commit before any write; STOP and report):**
- a. Exactly which `bt_daily_outcomes` rows (84) and `vol_risk_premium` rows are derived from 2026-03-16 → 03-20 bars, how each was computed (script / CR), and whether the same code can recompute them unchanged.
- b. List the past research CRs whose inputs included those sessions (from specs / logs / run rows). For each: how many sessions in its sample, and whether 03-16 → 03-19 could matter to its verdict (e.g. a sample under 50 sessions, or a result near its bar). Report only; don't re-run any research.
- c. B4: these sessions are train (≤ 2026-06-05), so reading outcomes there is allowed. Confirm nothing in the holdout is touched.
- d. Before/after for each recomputed row (dry run), and the write plan under the unattended backfill protocol. If any table needs grants the writer lacks, say so; don't apply grants without Ryan's approval.

Don't recompute until Ryan approves Step 0.

## Standing rulings conformance

| Ruling | Status | Note |
|---|---|---|
| A1–A22 (the setup) | n/a | Data repair of stored derived rows; no setup, level, entry, exit or event filter is defined or tested. |
| B1, B2, B3 | n/a | No verdict, parameter sweep or setup optimisation. Step 0b lists past CRs and their sample sizes; no research is re-run. |
| B4 (data split) | honoured (to be confirmed in Step 0c) | The window sessions 2026-03-16 → 03-20 are train. Outcome rows whose horizon runs past 2026-06-05 would read holdout bars on recompute — Step 0c lists them and whether any exist. |
| B5 (Databento is the reference) | honoured | The recompute reads the CR-DJ-patched bars (Databento ES M26 in the window). |
| B6, B7, B8 | n/a | No exits, placebos or labels. |
| C1, C2 | n/a | No UI or display change. |
| Revoked R1–R7 | none used | R6 (`discounted_level`) is the stored `bt_daily_outcomes` target frame; the recompute reproduces the existing computation unchanged and does not introduce R6 into any new test. |
