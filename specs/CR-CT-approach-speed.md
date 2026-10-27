# CR-CT — Speed into the level: fast approaches break, slow drifts fail

Spec freeze. Branch `feat/CR-CT-approach-speed` off `origin/main` `57adce0` (PR #99 merge, CR-CQ), in its own worktree.
Read-only against the database; no table writes, no cron, no frontend. Train only (2026-01-06 → 2026-06-05).

## Spec body (kickoff, verbatim, Ryan in chat 2026-10-26)

> CR-CT — Speed into the level: fast approaches break, slow drifts fail (read-only, pre-registered,
> train first).
>
> Branch first: git checkout -b feat/CR-CT-approach-speed off origin/main. Own worktree.
> Read CLAUDE.md, field-conventions.md, standing-rulings.md, Dash/research/no-bs-day-trading-course-notes.md
> (ideas 1 and 5), and the CR-CF and CR-CL specs (reuse their level and race code). Freeze
> specs/CR-CT-approach-speed.md with the conformance table. A3 deviation PRE-APPROVED by Ryan: entry
> at the touch is the setup here ("as she goes", the R2 level-entry variant, labelled as such);
> record it. Stop on any other "deviates". List choices the brief doesn't fix (C1, C2, …) first.
>
> Idea (course: "the quicker the market gets to a level, the more likely it is to go through";
> momentum is the default — consistent with CR-CE and CR-CP): at a level everyone knows, a FAST
> approach on heavy volume breaks through; a SLOW drift on light volume fails.
> Levels (primary): CR-CF F1 prior RTH high/low/close and F2 overnight high/low, plus the running RTH
> high/low once it is ≥ 30 min old. First touch per level per session (A21); second touch = readout.
> Approach (last 10 min before the touch bar, causal):
>   speed = points moved toward the level ÷ 10 min ÷ session implied move (so days compare);
>   relative volume = approach volume ÷ the trailing 20-session median volume for the same clock
>   window (no future sessions).
>   FAST = speed in the top tercile AND relative volume ≥ 1; SLOW = speed in the bottom tercile AND
>   relative volume < 1. Tercile cuts fixed in Step 0 from train touches, before any outcome.
> Entry: close of the touch bar. Outcome (A15, A11): stop-free race from entry, first to ±10 pts
> IN THE BREAK DIRECTION (through the level) by 13:00 PT; ties / unresolved excluded and counted.
> Primary: break win rate FAST minus SLOW (session-cluster bootstrap, 10 000 draws, seed 20260920).
> PASS if CI lower bound > 0; REFUTED if CI upper bound < 0; else INCONCLUSIVE; ≥ 40 resolved in EACH
> cell, else underpowered. One hypothesis.
> Readouts (not gated): FAST vs 50 % (break) and SLOW vs 50 % (fade); ±20 race; move at +30 / +60;
> max favourable before −10; matched random baseline (B7 EDGE key); speed alone and volume alone;
> approach window 5 / 20 min; "before she goes": ≥ 2 pullbacks from the level in the prior 60 min with
> each shallower than the last → break rate; by level family; time of day (first hour vs later);
> second touch; strong GEX line within 3 pts (A12); put vs call (A8); vol-scaled swing leg (A7) as the
> approach instead of the fixed 10 min.
> Data: train 2026-01-06 → 2026-06-05, Databento 1-min bars and flow. R-a, R-b, holdout NOT read.
> Step 0 (no outcomes): touches per family, tercile cuts, FAST / SLOW / neither counts, A/A on the
> random baseline. If FAST and SLOW each ≥ 40 on ≥ 30 sessions and |A/A mean| ≤ 1 pp, commit the run
> script and run once without stopping; else stop and report.
> Results + plain-English summary to the vault CR note, setup-registry, session note; push, PR, don't
> merge; copy logs before removing the worktree. Plain-English summary for Ryan: does a fast, heavy
> push into a level break it more often than a slow, light drift?

## Approvals and amendments before freeze (Ryan in chat, 2026-10-26)

- **Entry (E-R2).** The kickoff text "Entry: close of the touch bar" is revoked ruling **R1**; the pre-approval names
  the **R2 level-entry variant**. Asked in chat; Ryan chose **"Level price, R2"**: entry **at the level price at the
  touch second**, raced on Databento 1-s bars from the next second. The touch-bar close is **not used anywhere** in
  this CR (no R1 readout). This replaces the kickoff's "Entry" line.
- **A3 deviation — pre-approved** in the kickoff: entering at the touch is the setup tested here ("as she goes", R2,
  labelled). Not a model of the range-then-edge trade (A3 / A20).
- **A7 deviation — approved** in chat ("Approve as written"): the primary approach is the fixed last-10-min window
  (the course's speed-into-the-level idea); the A7 vol-scaled swing leg is a readout.

## Choices the brief doesn't fix (Code, 2026-10-26, before any data read)

- **C1 — data.** Bars: `es_flow_1m_databento` (outright contracts only; present 06:15 → 12:59 PT), price from the
  session's most-RTH-volume symbol (CR-CP C3). **Volume = total ES volume across all outright symbols in the minute**
  (roll-robust; the front symbol's share drops in roll weeks). Touch second and every outcome: `es_flow_1s_databento`,
  front symbol, 06:30 → 13:00 PT. F2 from `ironbeam_es_1m_bars` (Databento has no overnight). Every read asserted
  ≥ 2026-01-01 00:00 UTC and < 2026-06-05 21:00 UTC. Sessions 2026-01-02 and 01-05 are read only as priors (F1,
  trailing volume).
- **C2 — universe.** Train sessions 2026-01-06 → 06-05 through CR-CP's funnel (no Databento RTH flow, starts after
  06:31, early close, missing-minute run > 5, no canonical IM). Roll weeks kept; levels are guarded per C3.
  IM = canonical `implied_move_1d` (SPX, frame A — used only as a scale).
- **C3 — levels.** **F1** = prior session's RTH (06:30 → 12:59 PT) high, low and last close from the 1-min table,
  only when the prior session's front symbol equals today's. **F2** = Ironbeam high / low 15:00 PT prior → 06:30 PT,
  kept only if the Ironbeam 06:29 PT close is within 2 pts of the Databento 06:29 close (CR-CF). F1 / F2 kept within
  ±1·IM of the RTH open (CR-CF L4); within a family, levels < 0.5 pt apart merged.
  **RUN (running RTH high / low)** = the running high value H* (max high of RTH bars so far); it becomes a level once
  the last bar that printed ≥ H* is ≥ 30 bars old (so the earliest is 07:00 PT), and stays one until exceeded. Mirror
  for the low. Each distinct qualifying value is its own level.
- **C4 — arming and touch (1-min).** Arm = max(0.10·IM, 4) pts (CR-CB A17 / CR-CF L5). A level is armed on side s
  once a 1-min close is ≥ arm beyond it on that side, at or after its known bar (F1 / F2: 06:30; RUN: when it
  qualifies). **Touch** = the first later bar whose high ≥ L (armed below) or low ≤ L (armed above). **Break
  direction** b = up if armed below, down if armed above. **First touch** = the first touch of the level that session
  (primary); **second touch** = the next touch after re-arming on either side (readout). A bar that is the arming bar
  cannot also be the touch bar.
- **C5 — pooled event set.** The primary pools F1, F2, RUN first touches. Touches of different levels on the same bar,
  same b, within 1 pt are one event (kept: F1 > F2 > RUN; all kept in family readouts).
- **C6 — touch second and entry (R2).** Within the touch bar, the first 1-s bar of the front symbol with high ≥ L
  (b up) / low ≤ L (b down); **entry = L**. If no 1-s bar in that minute reaches L, the touch is dropped and counted.
- **C7 — approach window (primary).** Bars j−10 … j−1 (j = touch bar). speed = b·(C[j−1] − C[j−11]) ÷ 10 ÷ IM_d
  (fraction of IM per minute; can be negative). Approach volume = Σ total volume over j−10 … j−1. Windows may start
  before 06:30 (data from 06:15); a window that needs bars before 06:15 has no speed (counted).
- **C8 — relative volume.** Denominator = the median, over the up-to-20 most recent prior trading sessions with
  Databento minutes (from 2026-01-02; any universe status except early close), of total volume in the same clock
  window. **≥ 10 prior sessions required**, else relvol is undefined (touch counted, in neither FAST nor SLOW): with
  2025 unread, touches before 2026-01-16 have no relvol.
- **C9 — terciles.** Speed cut points p33.3 / p66.7 over all pooled first touches with defined speed and relvol on
  train, computed and written to `step0.json` before any outcome. FAST = speed ≥ p66.7 and relvol ≥ 1; SLOW = speed
  ≤ p33.3 and relvol < 1; else NEITHER. The run script reads the cuts from `step0.json`.
- **C10 — race.** On 1-s bars from the second after the touch second to 12:59:59 PT: win = b·(high/low − L) ≥ +10
  first; loss = ≤ −10 first; tie = both in one 1-s bar; unresolved = neither by 13:00. Seconds with no trade are not
  gaps. ±20 readout the same.
- **C11 — statistic.** Break win rate = wins ÷ (wins + losses), ratio of session-summed counts; FAST − SLOW;
  session-cluster bootstrap (CR-CL `boot_weights`, 10 000 draws, seed 20260920), percentile 95 % CI; B1 verdict;
  ≥ 40 resolved in each of FAST and SLOW else UNDERPOWERED. One hypothesis, no Holm.
- **C12 — random baseline (B7).** Every 1-min close m (06:30 … 12:58), both directions, raced ±10 on 1-min bars
  (CR-CL `race`). Key (m, b, EDGE), EDGE = entry within 2 pts of the running RTH high **or** low through bar m (a
  location key, either side). A touch's key = (j, b, EDGE of L through bar j). Matched difference via CR-CL
  `strata_matrices` / `matched_diff`. A/A as CR-CP Step 0 §7 (200 session half-splits, seed 20261025) with keys =
  FAST ∪ SLOW touches.
- **C13 — readout definitions.** Move at +30 / +60 = b·(last 1-s close at or before touch second + 30 / 60 min − L)
  (none past 13:00). MFE = largest b-favourable excursion before the first −10 (1-s). Speed alone = top vs bottom
  tercile, any relvol; volume alone = relvol ≥ 1 vs < 1. Windows 5 / 20 min: same formulas, terciles recomputed in
  Step 0. Time of day: touch before 07:30 PT vs later. Second touch: FAST − SLOW using the first-touch cuts. Strong
  GEX: CR-CM strong lines (`cr_cp_lib.load_levels`, net beyond the day's p95 |net|) within 3 pts of L; A8 = the
  nearest such line's net sign (put < 0, call > 0). **A7 leg:** swing legs on bars 0 … j−1 at swing_d = 12 × IM_d ÷
  65.714 (CR-CO / CR-CP A0); the approach starts at the most b-adverse extreme since the last confirmed b-direction
  swing extreme (or bar 0); speed = b·(C[j−1] − start px) ÷ minutes ÷ IM; relvol over the same clock span; its own
  terciles (Step 0).
- **C14 — "before she goes".** In bars max(0, j−60) … j−1, swing legs at 3 pts; pullbacks = the −b legs; depth =
  b·(L − leg end px). Flag = ≥ 2 pullbacks with depths strictly decreasing. Break rate flagged vs not.
- **C15 — run once.** `results.json` must not exist; the run script asserts the Step 0 gate and the stored cuts.

## Standing rulings conformance (Dash/research/standing-rulings.md, status current, read 2026-10-26)

| # | Ruling (short) | Mark | How |
|---|---|---|---|
| A1 | levels slow price, not a trade by themselves | honoured | the trade is conditioned on the approach (speed + volume), not on the level alone |
| A2 | a range ≥ minutes, relative | n/a | no range in this setup |
| A3 | no automatic entry at the first touch | **deviates — pre-approved** (kickoff) | entry at the level at the touch second = R2 variant, labelled; not a model of the range trade |
| A4 | stops beyond the zone | n/a | stop-free direction test (A15) |
| A5 | flow decides fade vs break; never "always" either | honoured | speed + volume pick break vs fade per touch; both outcomes scored |
| A6 | skew divergence | n/a (parked) | |
| A7 | approach = vol-scaled swing leg; fixed readout | **deviates — approved** in chat | fixed 10-min window primary; vol-scaled leg readout (C13) |
| A8 | put / call levels separately | honoured (readout) | primary families are not option levels; A8 split on the strong-line readout |
| A9 | two close levels → wait for the second | n/a | not a readout of this single-hypothesis CR |
| A10 | earlier same-day S/R matters | honoured | RUN (running RTH high / low ≥ 30 min old) in the primary |
| A11 | ±10 primary, ±20 readout, favourable-excursion distribution | honoured | C10, C13 |
| A12 | levels = chart's strongest colours | honoured (readout) | strong-line readout; primary families are CR-CF's, set by the kickoff |
| A13 | sizing | n/a | |
| A14 | skew vs expected | n/a (parked) | |
| A15 | direction first, stop-free | honoured | ±10 race |
| A16 | skew trend | n/a (parked) | |
| A17 | flow first, skew optional | honoured | volume / speed only; no skew |
| A18 | range position split | n/a | no range |
| A19 | in-range confirmation | n/a | no range |
| A20 | trade the range edges | n/a | no range (the A3 deviation) |
| A21 | second touch a different setup; separate cells | honoured | first touch primary; second touch a separate readout cell |
| B1 | three verdicts | honoured | |
| B2 | many parameters disclosed; plateau | honoured | all readouts reported, windows 5 / 10 / 20 |
| B3 | one setup at a time | honoured | single hypothesis |
| B4 | train only; R-a, R-b, holdout unread | honoured | C1 guards (< 2026-06-05 21:00 UTC, ≥ 2026-01-01) |
| B5 | Databento reference | honoured | C1 (Ironbeam only for F2's overnight extremes, checked against Databento) |
| B6 | mechanical exits labelled | honoured | ±10 race is a direction test, not Ryan's exit |
| B7 | placebos matched EDGE / INTERIOR | honoured | C12 |
| B8 | Code labels ≠ Ryan's | n/a | |
| C1, C2 | UI | n/a | |
| Revoked | R1 touch-bar close | **not used** (E-R2) | R2 level entry, labelled; R3–R7 not used |

Two "deviates" (A3, A7), both approved by Ryan; nothing else deviates. Proceeding to Step 0.

## Step 0 findings (2026-10-26; `scripts/cr_ct_step0.py` + `scripts/cr_ct_lib.py` → `scripts/logs/cr_ct/step0.json`; no touch outcome computed)

Read-only role asserted, `default_transaction_read_only = on`; every read inside [2026-01-01, 2026-06-05 21:00 UTC).
The only races run are the random-baseline entries for the A/A (C12). Two runs: run 1 crashed in §5 on a
`dict(...)` line (removed); run 2 is reported, and its §1–§4 output is identical to run 1's.

**Universe.** 105 train sessions → 4 without canonical IM (as CR-CF) → **101**; no early close, no missing-flow
session, none starting late. IM median 67.4 pts. 1-min ext data 06:15 → 12:59 PT on every session; every touch had a
1-s print at the level in its minute (0 dropped, C6).

**Levels per session** (median [IQR]): F1 3 [2, 3] (F1 missing 1 session: prior session other front symbol); F2 2
[2, 2] (missing 6: 06:29 Ironbeam vs Databento check / no bar); RUN 4 [3, 5].

**Touches** (n / sessions; pooled duplicates C5):

| | F1 | F2 | RUN |
|---|---|---|---|
| first touch | 141 / 86 (1 dup) | 95 / 85 (8 dup) | 198 / 91 (3 dup) |
| second touch | 101 / 66 | 66 / 63 (4 dup) | 128 / 78 (3 dup) |

Primary event set (first touches, deduped): **422 on 101 sessions** — RUN 195, F1 140, F2 87; break up 233 / down 189;
173 in the first hour; EDGE 257 / INTERIOR 165. Relvol undefined on 37 (touches before 2026-01-16, < 10 priors, C8).

**Tercile cuts (C9, fixed here; speed = fraction of IM per minute):**

| window | p33.3 | p66.7 | n |
|---|---|---|---|
| **10 min (primary)** | **0.00725** | **0.01400** | 385 |
| 5 min | 0.01016 | 0.02102 | 385 |
| 20 min | 0.00404 | 0.00960 | 368 |
| A7 leg | 0.00929 | 0.02105 | 384 |

At the median IM (67 pts) the 10-min cuts are ≈ 4.9 and 9.4 pts in 10 min. Relvol (10 min) quantiles p10 / p50 / p90
0.68 / 0.99 / 1.67.

**Classes (first touches):** 10 min **FAST 85 (44 sessions), SLOW 75 (46)**, NEITHER 225, UNDEFINED 37. 5 min 90 /
76; 20 min 74 / 85; A7 leg 78 / 94. Speed alone top / bottom 140 / 139; volume alone heavy / light 189 / 196. By
family FAST / SLOW: F1 32 / 16, F2 19 / 20, RUN 34 / 39 (each family cell < 40: readout only). Second touch FAST 39 /
SLOW 63. Before she goes flagged on 33 of 422. Strong line within 3 pts: 22 (call 20, put 2), none 350, no levels 50 —
the A12 / A8 readouts will be too thin to read.

**A/A (C12, 200 splits, keys = the 160 FAST ∪ SLOW touches):** mean **+0.24 pp**, median +0.11, SD 4.0, p5 / p95
−6.0 / +7.4; 0.9 % of keys unmatched. Baseline 69,948 resolved entries, win rate 0.500 (symmetric by construction),
EDGE share 0.14 (touches 0.61 — hence the EDGE key).

**Gate:** FAST 85 ≥ 40 on 44 ≥ 30 sessions; SLOW 75 ≥ 40 on 46 ≥ 30; |A/A| 0.24 ≤ 1 pp → **passed**. Committing the
run script, then running once without stopping.

## Result (single run, 2026-10-26) — `scripts/cr_ct_run.py` @ `e3ffe38` → `scripts/logs/cr_ct/results.json`

One run, exit 0, no re-run. Event set and tercile cuts asserted equal to Step 0 (422 first touches; cuts unchanged).
Entry = the level price at the touch second (R2 level-entry variant, labelled); ±10 stop-free race on 1-s bars to
13:00 PT — a direction test, not Ryan's exit (B6).

### Primary — break win rate, FAST − SLOW (first touches, ±10)

| cell | n | resolved (sessions) | break win rate |
|---|---|---|---|
| FAST (fast + heavy) | 85 | 85 (44) | **56.5 %** (48 / 37) |
| SLOW (slow + light) | 75 | 70 (45); 5 unresolved | **54.3 %** (38 / 32) |
| **FAST − SLOW** | | | **+2.2 pp [−15.2, +19.5] → INCONCLUSIVE** |

### Readouts (no verdicts, unadjusted; ~20 readouts, expect about one CI to exclude 0 by chance)

- FAST vs 50 % (break) +6.5 [−6.5, +18.4]; SLOW vs 50 % (**fade**) −4.3 [−15.7, +7.5] — slow drifts did **not** fail
  more often than not. All 422 first touches broke 54.1 % (405 resolved), +4.1 [−1.2, +9.4]; NEITHER 54.0 %.
- ±20 race FAST − SLOW −0.5 [−22.5, +22.3] (52.6 % n 78 vs 53.1 % n 49). Move at +30 min +0.05 pts [−5.7, +6.1];
  +60 min −1.0 [−8.0, +7.0]. MFE before −10: median 11.75 vs 10.25 pts, mean +5.8 [−2.5, +14.5]; reached +10 56 % vs 51 %.
- Matched random baseline (B7 key m, b, EDGE): FAST +5.5 [−7.7, +18.0]; SLOW +3.6 [−7.6, +13.8]; all touches +3.6
  [−1.6, +8.8] (coverage 1.00).
- Speed alone (top vs bottom tercile) +5.3 [−7.8, +18.3] (57.6 % n 139 vs 52.3 % n 132). Volume alone (heavy vs
  light) −5.4 [−17.1, +6.3] — light-volume approaches broke slightly *more* (57.3 % vs 51.9 %).
- Window 5 min −4.9 [−22.7, +12.5]; 20 min +3.1 [−14.6, +21.3]; A7 vol-scaled leg −0.1 [−18.1, +17.3]. No plateau:
  the sign flips with the window.
- Before she goes (33 flagged): 54.8 % vs 54.0 %, +0.8 [−21.1, +20.0].
- Family FAST − SLOW: F1 −9.4 [−41.4, +24.0] (32 / 16); F2 −21.1 [−53.3, +10.1] (19 / 19); RUN +18.8 [−6.1, +43.1]
  (34 / 35). Every family cell < 40. Break rate by kind 50.0 % (running low) → 60.5 % (prior high), every CI covers 50.
- Break up / down: −1.1 / +7.3. First hour +1.1 [−25.6, +26.7]; later +2.6 [−20.5, +25.1].
- EDGE (level at the session extreme) −14.8 [−37.7, +8.4]; INTERIOR +27.1 [+2.2, +53.3] (35 / 26).
- **Second touch** (FAST / SLOW on the first-touch cuts): FAST broke 35.9 % (n 39), SLOW 63.8 % (n 58): **−27.9
  [−46.9, −7.9]** — the reverse of the hypothesis. All second touches 52.7 % (275).
- Strong line within 3 pts: 22 touches (20 call, 2 put), none FAST or SLOW; broke 38.1 % vs 55.7 % without, −17.6
  [−40.9, +6.1]. A8 not readable (2 put touches).

### Reading

- **No evidence that approach speed and volume decide break vs fail at these levels.** A fast, heavy push broke 56.5 %;
  a slow, light drift broke 54.3 %. The difference is 2 pp with a CI of ±17 pp: compatible with nothing, and with a
  moderate effect either way. Not shortlisted.
- Momentum "through the level" is mildly the default (all first touches 54 %, +3.6 vs matched random), but the CI
  covers 0 and the matched random baseline says most of it is where price is, not the level.
- The pieces disagree: speed alone leans +5, volume alone leans −5 (light volume broke more), and the window sign flips
  (5 min −4.9, 10 min +2.2, 20 min +3.1). That is noise, not a plateau (B2).
- Two readouts exclude 0 — second touch (−27.9: fast second approaches **failed**) and INTERIOR (+27.1) — out of ~20
  unadjusted readouts on cells of 26–63. Leads at most. The second-touch reversal is the one worth a look: it fits A21
  (a second touch is a different setup) and Ryan's "something about the respective approaching legs". It would need
  its own pre-registered CR (train readout → R-a).
