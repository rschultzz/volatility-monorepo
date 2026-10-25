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

---

## Step 0 findings (Stage 1, 2026-10-25)

Script `scripts/cr_ce_step0.py` (log `scripts/logs/cr_ce/step0.log`, JSON `scripts/logs/cr_ce/step0.json`, git-ignored). One run, exit 0. Read-only: `get_backfill_db_conn()` (role asserted) + `default_transaction_read_only = on` (asserted), `statement_timeout = 600s`; SELECTs only. Databento reads bounded to `ts_utc` ∈ [2026-01-02 05:30 PT, 2026-06-05 21:00 UTC): **no 2025 row (R-b) and no holdout row was read**. Event construction is `cr_bw_absorption.py`'s code with the flow table swapped; `place_levels` / `bucket_levels` imported unchanged. Every operational choice the CR leaves open is fixed in the script docstring before the run.

**No outcome was computed at real levels.** Stop-free moves were computed only for placebo events inside the A/A splits; only the split statistic's distribution is reported. The CR-CC fixture's outcome fields were not read.

**Verdict: Step 0 passes its gate (A/A).** Counts and fixture match are reported. Four reading questions for Ryan are listed at the end; Stage 1 is not started.

### Operational choices (not stated in the CR)

- **Front symbol:** one symbol per kept session (asserted). Two symbols are stored only on 2026-03-09 → 03-19, inside the excluded March roll week. H26 to 03-06, M26 from 03-23 on kept sessions.
- **RTH open (A2.1):** first Databento row ≥ 06:30 PT. Equal to CR-BW's Ironbeam open on 83 of 85 common sessions; 2026-03-02 differs by 0.25; **2026-03-04 differs by 38.0 pts** (Ironbeam's feed starts 07:57:31 PT that day, so CR-BW used 6876.75; Databento has the 06:30 print, 6838.75). The ±1·IM window and the placebo bins move with it on that session.
- **Databento-scale baseline B_d (L2, L7, A1.5):** median A over the prior up-to-20 baseline sessions, ≥ 5 required. Baseline sessions are 2026 only: R-b (2025) may not be read, so there is no December warm-up. 2026-01-02 is admitted to the baseline without the L14b check (its A is 2025-12-31).
- **Event (L2):** an armed touch (CR-BW L5 re-arm) with D > 0, after the A1.4 pre-window gap rule. Every armed touch counts; the first armed touch of a level in a session is flagged and counted separately (`first`). See question 2.
- **Cells (L7):** put-heavy = level net < 0, call-heavy = net > 0. From below = side −1 (fade short), from above = side +1 (fade long). Placebo events take the parent level's net sign.
- **Horizon availability:** an event has horizon H only if t + H ≤ 13:00 PT. See question 3.
- **A/A (A1.3 design, stop-free move):** placebo events (D > 0) with a measured move. Move at H = s · (P(t + H) − L), with P = close of the last Databento row ≤ t + H, missing if that row is > 15 s old (none were). The split unit is the pseudo-level (d, L). 200 splits use `default_rng(20261025)`. Statistic = event-weighted mean move, half 1 − half 2, in points. CI: session-cluster bootstrap, 2,000 resamples per split, one `default_rng(20260920)` stream, halves resampled jointly by session. Gate thresholds as written in A1.3 (≤ 10 % of splits exclude 0; |median| ≤ 3, read here in points).

### Sessions

| rule | removed | remaining |
|---|---|---|
| NYSE sessions 2026-01-06 → 2026-06-05 | | 105 |
| L14a roll week (03-09 → 03-20) | 10 | 95 |
| L14b no ES 15:45 ET bar on A (02-18, 02-19) | 2 | 93 |
| L14c roll mismatch | 0 | 93 |
| no Databento flow / no RTH open | 0 | 93 |
| no canonical `bt_daily_features` row (01-20, 02-17, 04-06, 05-26) | 4 | 89 |
| IM null / ≤ 0, no `orats_oi_gamma` rows | 0 | 89 |
| < 5 prior 2026 Databento baseline sessions (01-06, 01-07, 01-08) | 3 | **86** |

- **Usable: 86 sessions, 2026-01-09 → 2026-06-05** (CR-BW: 88, 01-06 → 06-05). Versus CR-BW: −3 (01-06/07/08, baseline warm-up without 2025), +1 (2026-03-03, which has no Ironbeam flow but full Databento flow). See question 1.
- **Databento RTH flow gaps:** max gap per session p5/p50/p95 3 / 6 / 10 s, max 14 s. No session has a gap > 15 s. The A1.4 rule drops 0 events (real and placebo).
- **B_d (Databento scale):** p5 / p25 / median / p75 / p95 = 2.65 / 3.10 / 3.62 / 4.19 / 4.85 pts per 1,000 contracts; range 2.45–4.96. CR-BW Ironbeam median was 4.14: Databento's larger D lowers A. 14 sessions rest on fewer than 20 prior sessions (min 5).

### Levels

- Real: **510** (CR-BW 522 on 88 sessions); 6 per session on every usable session; ≥ 100 $bn 389; net − / + 168 / 342.
- Placebo (A1.2): 433 pseudo-levels.
- Every level that carries a CR-BW Ironbeam touch on a common session is also in the Databento level set (0 missing).

### Event counts per cell (L7; no outcomes)

A_rel tercile cut points (real, Databento): 1.521 / 2.929 (CR-BW Ironbeam 1.655 / 3.187).

| cell | real events (D > 0) | first touch only | sessions | ≥ 100 $bn | A_rel terciles (low / mid / high) | with +30 min | with +60 min | **CR-BW Ironbeam (D > 0)** |
|---|---|---|---|---|---|---|---|---|
| put-heavy, from below | 138 | 36 | 34 | 98 | 42 / 45 / 51 | 131 | 124 | 126 |
| put-heavy, from above | 136 | 33 | 33 | 100 | 31 / 47 / 58 | 129 | 123 | 131 |
| call-heavy, from below | 285 | 90 | 62 | 221 | 107 / 88 / 90 | 267 | 251 | 297 |
| call-heavy, from above | 244 | 59 | 55 | 187 | 88 / 87 / 69 | 228 | 219 | 247 |
| **pooled** | **803** | **218** | 84 | 606 | 268 / 267 / 268 | 755 | 717 | **801** |

- Real touches 1,022 (all kept; side undetermined 0); D ≤ 0: 219.
- Placebo events (D > 0): 602 — per cell 162 / 148 / 161 / 131 (same order as above), first-touch-only 176; with +60 min 534.
- **Databento vs CR-BW Ironbeam events, 84 common sessions:** of 995 Ironbeam kept touches, 980 have a Databento touch at the same level in the **same second** (985 within 2 s, 992 within 60 s). D > 0 agrees on 882 of 980. 35 flip from Ironbeam D > 0 to Databento D ≤ 0, and 63 the other way. On events where both are positive, median D(Databento) / D(Ironbeam) = 1.35, consistent with CR-CA's delta ≈ 1.4×. Per cell, Databento has more put-heavy and fewer call-heavy events, mostly from the flips and the session swap.

### CR-CC fixture match (5 events)

Matched on session, level and touch second. Only identity fields of `research/gallery/cc-fixture.json` were read.

| id | session | level | fade | same second on Databento | first touch of level | D Ironbeam / Databento | in CR-CE event set |
|---|---|---|---|---|---|---|---|
| 1 | 2026-01-06 | 6964 | long (from above) | yes | no (3 touches that day) | **−232 / −21** | **no**: D ≤ 0 on both feeds; also outside the universe (baseline warm-up) |
| 2 | 2026-02-02 | 6991 | long | yes | no (2) | 954 / 1,015 | **yes** |
| 3 | 2026-03-02 | 6845 | long | yes | no (5) | 27 / 554 | **yes** |
| 4 | 2026-04-01 | 6641 | short (from below) | yes | **yes** | 1,136 / 1,829 | **yes** |
| 5 | 2026-05-01 | 7277 | long | yes | no (6) | 1,835 / 1,973 | **yes** |

- All 5 touches are found on Databento at the same timestamp, same level, same approach side.
- **4 of 5 are in the event set.** Fixture 1 was never a D > 0 event: CR-CC's fixture builder did not filter on D, so it is a flow-against-the-approach touch (CR-BW S6). It would be excluded by L2 on any session universe.
- **Only fixture 4 is the first touch of its level that session.** 2, 3 and 5 are re-armed touches (CR-BW L5). This bears on question 2.

### A/A on placebo stop-free moves (A1.3 design) — PASS

Placebo moves measured: +60 min 534 (68 lack the horizon before 13:00 PT), +30 min 564 (38 lack it), 13:00 PT 602.

| | events / pseudo-levels / sessions | median stat | sd | p5 / p95 | median CI half-width | sd ÷ (hw / 1.96) | **splits excluding 0** | gate |
|---|---|---|---|---|---|---|---|---|
| **pooled, +60 min (gate)** | 534 / 184 / 71 | **+0.09** | 1.94 | −3.38 / +3.17 | 3.68 | 1.03 | **6.5 % (13)** | **PASS** |
| pooled, +30 min | 564 / 187 / 71 | +0.04 | 1.37 | −2.13 / +2.31 | 2.51 | 1.06 | 6.0 % | pass |
| pooled, 13:00 PT | 602 / 191 / 72 | +0.02 | 2.89 | −4.99 / +5.00 | 5.23 | 1.08 | 8.5 % | pass |
| put-heavy, from below, +60 | 154 / 75 / 31 | −0.99 | 4.73 | −7.70 / +8.10 | 8.83 | 1.05 | 6.0 % | pass |
| put-heavy, from above, +60 | 125 / 68 / 31 | −0.90 | 5.29 | −9.39 / +8.43 | 10.60 | 0.98 | 5.5 % | pass |
| call-heavy, from below, +60 | 135 / 73 / 45 | +0.02 | 3.14 | −4.80 / +5.28 | 6.23 | 0.99 | 5.5 % | pass |
| call-heavy, from above, +60 | 120 / 57 / 34 | −0.12 | 3.45 | −5.70 / +5.19 | 7.17 | 0.94 | 6.0 % | pass |

The session-cluster bootstrap is calibrated on the stop-free move: the split spread matches the CI width (ratios 0.94–1.08), and CIs exclude 0 at 5.5–8.5 % of splits against a nominal 5 %.

**Power, for the record (not a redefinition).** Each A/A half holds about half the placebo arm. Per cell at +60 min, the half-arm CI half-width is 6–11 pts. The L9 comparisons (real vs placebo, real vs random) have about 120–250 real events per cell against 120–155 placebo, so the per-cell CI half-width vs placebo will be roughly 4–8 pts before Holm. Under the first-touch-only reading (question 2), cells hold 33–90 real events, and half-widths would be about twice that.

### Questions for Ryan (before Stage 1)

1. **Baseline warm-up without 2025.** The strict L4 reading (R-b not read) costs 01-06/07/08, including the fixture-1 session. Options: (a) accept 86 sessions; (b) allow the Dec 2025 Databento rows for the B_d baseline only (a scale input, not outcomes), which is a read of R-b.
2. **What "first touch" means (L2).** As run: every armed touch (CR-BW L5 re-arm), 803 real events. The alternative is the first armed touch of each level per session only: 218 events. Fixtures 2, 3 and 5 are re-armed touches, so the cards Ryan reviewed are mostly *not* first touches. L8(e) (the A9.5′ re-arm after a stop-sized excursion) reads as a separate readout either way.
3. **Events whose horizon passes 13:00 PT** (real: +30 min 48, +60 min 86 of 803). Drop for that horizon, as the A/A did, or truncate at 13:00?
4. **A/A gate units.** A1.3's "median within ±3" was written for win-rate points. Here it is read as ES points. It passes either way (pooled median +0.09), but per-cell half-widths of 6–11 pts make ±3 a loose test there.

---

## Amendment A1 (Ryan, 2026-10-25, after Step 0; no outcome computed at real levels)

| # | Clause |
|---|---|
| A1.1 | **Baseline:** accept the 86 usable sessions (2026-01-09 → 2026-06-05). No 2025 data is read for the B_d baseline. |
| A1.2 | **L2 "first touch"** = every re-armed touch (CR-BW L5) with D > 0: the **803-event primary set**. This is what screened the reviewed cards. First touch of a level per session (218 events) is a **readout**. |
| A1.3 | **Horizons:** at +30 and +60 min, drop events that cannot reach the horizon before 13:00 PT, as the Step 0 A/A did. The version truncated at 13:00 PT is a **readout**. |
| A1.4 | **A/A threshold** ("median within ±3") is read in ES points. Accepted; the Step 0 A/A pass stands. |
| A1.5 | **Run:** Stage 1 runs once, with the run script committed before it runs. Primary: +60 min stop-free move vs placebo and vs random entries, per cell and pooled. Also all L8 readouts, the first-touch-only readout and the truncated readout. All statistics are saved before printing. The result is entered in `research/setup-registry.md` (Stage 1 = screen; three-verdict wording). Do not push. |

---

## Stage 1 results (single run, 2026-10-25)

Run of `scripts/cr_ce_stage1.py` at commit `ea3935f` (sha256 `495cdee7…906564`), one run, exit 0. Read-only connection (`default_transaction_read_only = on`, `statement_timeout = 600s`). Train data only: 86 sessions, 2026-01-09 → 2026-06-05, Databento flow; no 2025 row and no holdout row read. Every statistic was written to `scripts/logs/cr_ce/stage1.json` before anything was printed; the printout is in `scripts/logs/cr_ce/stage1.log` (both git-ignored). Universe asserted equal to Step 0 (86 sessions, 803 events, tercile cuts 1.521 / 2.929). No price lookup was missing (all 15 s checks passed).

Bootstrap: session-cluster, 10,000 resamples, seed 20260920, one weight matrix for every statistic. All moves are ES points in the fade direction from the level (stop-free).

### Primary — +60 min stop-free move (A1.2 events; A1.3 drop of events past 13:00)

| cell | n real / placebo / random | real mean [CI] | placebo mean | random mean | **vs placebo** | **vs random** | Stage 1 screen |
|---|---|---|---|---|---|---|---|
| put-heavy, from below (fade short) | 124 / 154 / 138 | −4.14 [−9.63, +1.25] | −8.03 | −6.18 | +3.89 [−1.82, +9.25] | +1.54 [−1.13, +4.24] | **INCONCLUSIVE** |
| put-heavy, from above (fade long) | 123 / 125 / 136 | +3.07 [−1.35, +7.56] | +3.18 | −2.23 | −0.11 [−5.15, +5.45] | +5.08 [+2.28, +7.96] | **INCONCLUSIVE** |
| call-heavy, from below (fade short) | 251 / 135 / 285 | +0.58 [−2.80, +4.32] | −0.59 | −3.12 | +1.17 [−3.13, +5.79] | +2.59 [+0.69, +4.50] | **INCONCLUSIVE** |
| call-heavy, from above (fade long) | 219 / 120 / 244 | −0.13 [−3.28, +2.79] | +1.34 | −3.97 | −1.48 [−6.15, +3.30] | +3.63 [+1.76, +5.28] | **INCONCLUSIVE** |
| **pooled** | 717 / 534 / 803 | **−0.03 [−1.46, +1.32]** | −1.42 | −3.75 | **+1.39 [−0.62, +3.43]** | **+3.15 [+2.14, +4.17]** | **INCONCLUSIVE** |

(The random column's n is events with ≥ 1 valid draw; the paired vs-random difference uses events with both.)

- **No cell is shortlisted.** None beats placebo with CI lower bound > 0. None is refuted: every vs-placebo CI upper bound is > 0.
- **Holm (L9, readout at Stage 1):** adjusted one-sided p vs placebo 0.37 / 1.00 / 0.93 / 1.00; vs random 0.12 / 0.001 / 0.007 / 0.001. No cell passes after Holm.
- **Reading the "vs random" wins.** Three cells and the pooled set beat random entries, but the real move itself is about zero (pooled −0.03). The random baseline is negative (pooled −3.75): over ±60 min around a touch, price keeps going in the approach direction, so a random entry in the fade direction loses. The placebo touches' mean (−1.42) also sits above that baseline. This is indicative only: random entries were drawn around real touches, not placebo ones. So "beats random" here measures that **touch moments are better fade entries than random moments, at real and pseudo-levels alike**. It does not show that the level matters. Only the vs-placebo comparison isolates the level, and it is +1.39 [−0.62, +3.43] pooled.

### Verdict (three-verdict rule, Stage 1 screen)

**INCONCLUSIVE in all four cells and pooled. Not shortlisted.** The CR-BW fade direction, measured stop-free at +60 min, is not distinguishable from the same rule at distance-matched pseudo-levels. Per CR-CB A10.3 / A10.5, Stage 2 (R-a) does not run for this setup, and the result is logged in `research/setup-registry.md`.

### Readouts (not gated; no conclusion drawn from any one)

**Other horizons** (pooled; real / vs placebo / vs random):
- +30 min (n 755): +0.47 / +0.57 [−1.15, +2.24] / +2.32 [+1.12, +3.48].
- 13:00 PT (n 803): −1.88 / −0.54 [−3.81, +2.79] / +2.98 [+2.16, +3.82].
- No cell beats placebo with CI lower > 0 at either horizon.

**Truncated at 13:00 (A1.3 readout)** (pooled, n 803):
- +60 min: real −0.17, vs placebo +1.29 [−0.50, +3.09], vs random +3.17 [+2.20, +4.12].
- +30 min: real +0.30, vs placebo +0.45 [−1.14, +2.00].
- Same picture as the primary.

**First touch of the level per session only (A1.2 readout)** (+60 min):

| cell | n | real | vs placebo | vs random | screen |
|---|---|---|---|---|---|
| put-heavy, from below | 36 | −13.12 [−20.22, −6.11] | −4.59 [−10.85, +0.72] | +1.82 [−3.39, +6.96] | inconclusive |
| put-heavy, from above | 33 | +2.40 | +1.87 [−7.64, +11.17] | +9.44 [+2.86, +15.92] | inconclusive |
| call-heavy, from below | 89 | −0.70 | +0.47 [−6.22, +6.39] | +3.79 [+1.24, +6.25] | inconclusive |
| call-heavy, from above | 55 | −1.86 | +0.79 [−12.55, +13.61] | +6.35 [+1.05, +11.18] | inconclusive |
| pooled | 213 | −2.62 [−5.78, +0.43] | +0.42 [−3.72, +4.41] | +5.00 [+2.84, +7.11] | inconclusive |

The put-heavy-from-below first touch loses on its own: −13.1 pts at +60 min, CI excluding 0 vs zero. Price continues up through put-heavy levels on the first approach from below. That is one readout among many, on n 36.

**L8 (a) A_rel absorption terciles** (+60 min, low − high):
- Pooled: real −1.51 [−5.58, +3.00]; placebo −0.91; difference-in-differences −0.60 [−5.37, +4.60].
- Per cell, real low − high: +6.87 / −5.80 / −3.30 / −2.85 (put-below / put-above / call-below / call-above); every CI covers 0.
- **Absorption does not sort the stop-free move.** This agrees with CR-BW T1.

**L8 (b) MAE before +10 pts (the stop question)**, pooled:

| | n | reach +10 by 13:00 | MAE p25 / p50 / p75 / p90 (reached) | MAE ≤ 2 / 4 / 6 / 8 / 10 | median minutes to +10 |
|---|---|---|---|---|---|
| real | 803 | 67.1 % | 1.75 / 5.0 / 10.5 / 19.0 | 28.8 / 44.2 / 56.8 / 66.4 / 73.7 % | 19.6 |
| placebo | 602 | 69.8 % | 2.0 / 5.5 / 13.25 / 21.8 | 25.5 / 41.7 / 54.0 / 61.7 / 68.1 % | 14.1 |

- About two-thirds of touches eventually reach +10 in the fade direction, but only 29 % do so without first going more than 2 pts against (CR-BW's 8-tick stop). Half needed ≥ 5 pts of room.
- **Placebo pseudo-levels show the same profile**: "the direction was right and the stop failed" is about as common away from levels as at them.
- By cell, real reach shares are 69 / 72 / 64 / 68 % and median MAEs 7.0 / 4.4 / 4.5 / 5.0 pts.
- Caveat: reaching +10 at some point before 13:00 does not by itself measure direction. It is a two-sided path property over hours.

**L8 (c) share of moves > 0** (pooled, real / placebo / random):
- +30 min: 52.6 / 49.1 / 43.6 %.
- +60 min: 49.7 / 45.9 / 40.4 %; real − placebo +3.8 pts [−1.6, +9.2].
- 13:00 PT: 48.2 / 47.8 / 42.9 %.

**L8 (d) levels ≥ 100 $bn** (+60 min; n 534 real):
- Pooled real +0.20; vs placebo +1.59 [−0.50, +3.81]; vs random +3.12 [+1.92, +4.34].
- Put-heavy from below: vs placebo +6.18 [−0.60, +13.06].
- All cells inconclusive.

**L8 (e) A9.5′ re-arm: attempt 2 after a 2-pt adverse excursion, back at the level** (575 real / 465 placebo):
- Pooled +60 min: real −0.58; vs placebo +0.74 [−1.21, +2.74]; vs random +2.83 [+1.96, +3.73].
- No cell beats placebo with CI lower > 0.

### Notes

- The fixture cards (4 of 5 in the event set) are a subset of this population. The screen says that their "right direction, failed stop" pattern is the base rate of fading any touched price after a ≥ 0.10·IM approach. It does not show a level-specific edge.
- The holdout and R (R-a, R-b) remain unread.
