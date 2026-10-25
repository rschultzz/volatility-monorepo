---
type: cr
cr_id: CR-CE
title: First-Touch Fade, Stop-Free Direction — does the CR-BW level-touch fade call the direction right, ignoring the stop? (pre-registered, test-only; two-stage)
status: created
run_mode: interactive
size: small
estimated_days: 1
data_safety_class: read_only
dependencies: [CR-CA, CR-BT]
soft_dependencies: [CR-BW, CR-CB]
depended_on_by: []
branch_name: feat/CR-CE-first-touch-fade-direction
last_reviewed: 2026-10-25
related_decisions: []
related_open_questions: []
related_sessions: []
tags: [dash, cr, created, levels, order-flow, direction, stop-free, pre-registered, test-only]
---

# CR-CE — First-Touch Fade, Stop-Free Direction

## Why

**Ryan, 2026-10-25, after reviewing the 5 CR-CC fixture cards.** They are CR-BW level touches (top-6 Corrected levels ≥ 40 $bn gross, flow into the level, fade with a small fixed stop), one session per month, Jan–May 2026. On most of them **the direction was right and the stop was what failed**. Something screened those, and it was accurate: don't lose sight of it.

**CR-BW never tested this.** It scored only an 8-tick race with a stop (T1 +0.6 [−8.4, +9.9]) and ran on Ironbeam flow, which CR-CA measured at about 0.7× true buy − sell. So whether the CR-BW fade calls direction is **untested**.

## Locked decisions (Ryan to confirm before spec freeze)

| # | Decision | Value |
|---|---|---|
| L1 | Scope | Test only. Read-only, no writes. |
| L2 | Events | CR-BW A1.1 levels (top 6 Corrected, ≥ 40 $bn gross, ≤ 30 DTE, within ±1·IM of the open) and A2 touch rules (re-arm, first touch with D > 0 = flow into the level), **recomputed on `es_flow_1s_databento`** with Databento-scale baselines. Fade direction = against the approach. |
| L3 | Stage 1 | Train 2026-01-06 → 2026-06-05, the CR-BW universe and exclusions. |
| L4 | Stage 2 | R-a = 2024-04-03 → 2024-12-31 (CR-CB A10.2 split), frozen as in Stage 1. R-b and the holdout are not read. |
| L5 | **Primary measure: stop-free move** | The signed move in the fade direction from the touch price to **+30 min**, **+60 min** and 13:00 PT, in points. Primary horizon **+60 min**. |
| L6 | Comparisons | **(i) placebo levels:** CR-BW A1.2 distance-matched pseudo-levels, same touch and flow rules. **(ii) random entries:** 5 random RTH seconds in the same session within ±60 min, same direction. **(iii) zero:** the move itself. |
| L7 | Cells | {put-heavy, call-heavy} × {approached from below, from above}, plus pooled. |
| L8 | Readouts | **(a)** the A_rel absorption tercile split (does absorption sort the stop-free move?);<br>**(b)** the MAE before +10 pts (how much room the right calls needed: the stop question);<br>**(c)** the share positive at each horizon;<br>**(d)** ≥ 100 $bn levels only;<br>**(e)** the CR-CB A9.5′ re-arm, a second touch after a stop-sized adverse excursion. |
| L9 | Pass (Stage 2, per cell, Holm across cells) | Mean stop-free move at +60 min > placebo **and** > random entries, CI lower bound > 0 on both. Three-verdict rule: **refuted** if the CI upper bound is < 0 vs placebo. |
| L10 | Bootstrap | Session-cluster, 10 000 resamples, seed 20260920. |
| L11 | Registry | A pass enters `research/setup-registry.md` as "First-touch fade (CR-BW events)", and goes to the optimize step (stops, re-arm, sizing) under CR-CB A10.3. |

## Step 0 (Stage 1)

Event counts per cell on Databento vs the CR-BW Ironbeam counts; the A/A on placebo stop-free moves (CR-BW A1.3 design, 200 splits); and the 5 CR-CC fixture events identified in the new event set (yes/no, same timestamps). No outcomes.

## Kickoff prompt

```
Start CR-CE in its own worktree: git fetch && git worktree add <scratch>/crce-wt -b feat/CR-CE-first-touch-fade-direction origin/main (confirm CR-CE unused). Other sessions run CR-CB and CR-CD in their own worktrees — do not touch them.
Read the vault CR "/Users/ryan/My Drive/Obsidian Vault/Anonymous/Dash/CRs/CR-CE — First-Touch Fade, Stop-Free Direction.md", specs/CR-BW-aggressor-absorption-at-levels.md, specs/CR-CA-databento-es-flow-backfill.md, components/field-conventions.md.
Commit 1: spec freeze, verbatim. Commit 2: Step 0 (counts, A/A, fixture match). No outcomes. Read-only DB. Stop and report.
```

---

## Spec freeze (2026-10-25)

Body above is the vault note copied verbatim, frontmatter included (vault `last_reviewed` 2026-10-25). Branch `feat/CR-CE-first-touch-fade-direction` off `origin/main` `578a8d5` (PR #80 merge), in its own worktree. Locked decisions L1–L11 taken as confirmed by the kickoff prompt (Ryan, 2026-10-25), which orders this freeze. Read-only against the database; no table writes.
