# CR-CW — Momentum from the course: clean-path approach (H1) and the snap-point hold (H2)

Spec freeze. Branch `feat/CR-CW-momentum-path-snap` off `origin/main` `625d2ec` (PR #103 merge, CR-CV), in its own
worktree. Read-only against the database; no table writes, no cron, no frontend. Train only (2026-01-06 → 2026-06-05).

## Spec body (kickoff, verbatim, Ryan in chat 2026-10-27)

> CR-CW — Momentum ideas from the course: clean-path approach, and the snap-point hold (read-only,
> pre-registered, train only).
>
> Branch first: git checkout -b feat/CR-CW-momentum-path-snap off origin/main (after #103). Own worktree.
> Read CLAUDE.md, field-conventions.md, standing-rulings.md, Dash/research/no-bs-day-trading-course-notes.md
> (video section, items 1–2), and the CR-CT spec (reuse its level, touch, entry and race code). Freeze
> specs/CR-CW-momentum-path-snap.md with the conformance table.
> Pre-approved, record don't stop: A3/R2 level-price entry at the touch second for H1 (as CR-CT); A7
> fixed 10-min approach window primary (as CR-CT), vol-scaled swing readout; B3 two independent
> hypotheses in one CR, each with its own verdict, nothing tuned across them. Stop on any other
> "deviates". List choices the brief doesn't fix (C1, …) BEFORE loading data; pick the most
> conservative where unsure.
>
> Data: train 2026-01-06 → 2026-06-05, Databento 1-SECOND bars/flow (es_flow_1s_databento) and 1-min
> bars. Volume at price: each second's volume assigned to its close rounded to 0.25 pt (record as C#;
> report the share of seconds whose high − low > 0.25 as a check). R-a, R-b, holdout NOT read.
>
> H1 — Clean-path approach breaks (course: "moves easily with very little back-ticking = real
> momentum"; CR-CT measured speed, not path).
>   Population: CR-CT's first touches exactly (F1, F2, running RTH extreme ≥ 30 min old; 422 on train).
>   Path efficiency over the 10-min approach = |net move toward the level| ÷ Σ |1-s close changes|.
>   CLEAN = top tercile, CHOPPY = bottom tercile (cuts from train touches in Step 0, before outcomes).
>   Entry and outcome exactly as CR-CT: level price at the touch second, 1-s race ±10 in the break
>   direction by 13:00 PT.
>   Primary: break win rate CLEAN − CHOPPY.
>   Readouts: efficiency × FAST/SLOW; retrace count (number of ≥ 1-pt counter-moves in the approach);
>   5/20-min windows; by level family; vs matched random baseline (B7).
>
> H2 — Snap-point hold: after a light-volume break, a pullback that stops short of the snap point
> continues (course: "if it fills back in the snapping point you've lost your momentum").
>   Break: from a CR-CT first touch, price trades ≥ 6 pts beyond the level before 12:30 PT.
>   Snap band: within the break leg (level → first ≥ 6-pt extreme), the 1-pt price band with the LOWEST
>   volume per point (from the per-price volume above).
>   Pullback: after the break extreme, the first counter-move of ≥ 3 pts. Classify at the pullback's
>   turn: HOLD = the pullback's extreme stays outside the snap band; FILL = it trades into or through it.
>   Turn bar (both groups, same rule): the first 1-min bar after the pullback extreme that closes back
>   in the break direction beyond the prior bar's close; entry = that bar's close. No turn before
>   12:45 PT → no trade (counted).
>   Outcome: 1-s race ±10 in the break (continuation) direction by 13:00 PT.
>   Primary: continuation win rate HOLD − FILL.
>   Readouts: each group vs 50 %; snap band from the bottom-quartile volume price instead of the
>   minimum; break ≥ 4 / 8 pts; pullback ≥ 2 / 4 pts; light vs heavy break (snap-band volume vs the
>   session's median volume per point).
>
> Both: session-cluster bootstrap, 10 000 draws, seed 20260920, 95 % CI; Holm across H1 and H2. PASS if
> the Holm-adjusted CI lower bound > 0; REFUTED if CI upper bound < 0; else INCONCLUSIVE; ≥ 40
> resolved per compared group, else underpowered. Readouts ±20 race and max favourable before −10.
> Step 0 (no outcomes): counts per group for H1 and H2, tercile cuts, snap-band check, A/A on the
> random baseline. For each hypothesis that clears ≥ 40 per group on ≥ 30 sessions with |A/A mean|
> ≤ 1 pp, commit the run script and run once without stopping; report any underpowered one as such.
> Results + plain-English summary to the vault CR note, setup-registry, session note; push, PR, don't
> merge; copy logs before removing the worktree. Summary for Ryan: (1) do clean, low-retrace pushes
> into a level break more often than choppy ones? (2) after a light-volume break, does a pullback that
> holds above the snap point keep going more than one that fills it in?

## Pre-approved deviations (kickoff; recorded, not stopped for)

- **A3 / R2 (H1).** Entry at the level price at the touch second, as CR-CT (the R2 level-entry variant, labelled).
  Not a model of the range-then-edge trade (A3 / A20). R1 (touch-bar close) is not used anywhere.
- **A7.** Fixed 10-min approach window primary for H1 (as CR-CT); the vol-scaled swing leg is a readout.
- **B3.** Two independent hypotheses in one CR, each with its own verdict; nothing tuned across them; Holm across the
  two.

## Choices the brief doesn't fix (Code, 2026-10-27, before any data read)

- **C1 — data and guards.** Everything CR-CT C1–C6 uses is reused by import (`scripts/cr_ct_lib.py`: universe
  funnel, ext 1-min arrays, levels F1 / F2 / RUN, arming, touches, pooled dedupe, touch second, `race1s`, `mfe1s`,
  `baseline_table`). New read: `es_flow_1s_databento`, the session's front symbol (CR-CT's), **06:15 → 13:00 PT**
  (the table's window), columns o / h / l / c / `volume`. Every read asserted inside [2026-01-01 00:00 UTC,
  2026-06-05 21:00 UTC). The 1-s seconds from 06:30 must equal CR-CT's cached 1-s h / l / c (checked in Step 0).
- **C2 — population.** H1 and H2 both start from CR-CT's primary set exactly: first touches, deduped (C5), with a 1-s
  touch second — asserted = 422. Break direction b, level L, touch second as CR-CT.
- **C3 — volume at price.** Each 1-s row's `volume` is assigned to its close rounded to the nearest 0.25 (closes
  are already on the tick grid; rounding is a guard). Check (Step 0): share of RTH front-symbol seconds with
  high − low > 0.25, and the volume share in those seconds.
- **C4 — path efficiency (H1).** Window = ext bars J−10 … J−1 (J = touch bar + 15), i.e. 1-s rows with time in
  [start of bar J−10, start of bar J). Anchor = the last 1-s close before the window (= CR-CT's C[J−11]).
  net = b · (last close in the window − anchor); path = Σ |Δ close| over the anchor + window closes.
  **efficiency = net ÷ path, signed** (in [−1, 1]; a clean move *away* from the level scores negative, not clean —
  the brief's "|net move toward the level|" read as the toward-component). Undefined (counted) if the window needs
  bars before 06:15 or path = 0. 5- and 20-min windows the same.
- **C5 — H1 classes.** Tercile cuts p33.3 / p66.7 of efficiency over the 422 with defined efficiency, fixed in
  Step 0 (`step0.json`) before any outcome. CLEAN ≥ p66.7, CHOPPY ≤ p33.3. Windows 5 / 20 get their own cuts.
  No volume condition (the brief has none; FAST / SLOW is a readout).
- **C6 — retrace count (H1 readout).** Walk the anchor + window 1-s closes: track the running b-favourable extreme;
  when b · (extreme − close) ≥ 1.0 count one retrace and restart the extreme at that close. Readout: break rate by
  count 0–1 / 2–4 / ≥ 5 (bands fixed here) and low (≤ train median count) vs high.
- **C7 — H1 readouts.** Efficiency × FAST / SLOW = CLEAN / CHOPPY within CR-CT's cls10 FAST and SLOW (CR-CT cuts
  from its `step0.json`, recomputed and asserted equal). A7 vol-scaled leg (CR-CT C13 leg start): efficiency over
  the 1-s closes from the leg start minute to J−1, own terciles. By family F1 / F2 / RUN. Matched random baseline =
  CR-CT C12 (key m, b, EDGE).
- **C8 — break (H2).** From the touch second k: the first 1-s row x ≥ k with b · (extreme − L) ≥ 6
  (extreme = high for b up, low for b down) and time < 12:30 PT. No condition on what price does in between (the
  brief has none) — the leg is touch second → break second. Not reached by 12:30 → no break (counted).
- **C9 — snap band (H2).** Break leg = 1-s rows k … x inclusive. E = the break-second extreme. Bands = 1-pt bins on
  the tick grid anchored at L: band i holds ticks L + b·(i + {0, .25, .5, .75}), i = 0 … ⌊b·(E − L)⌋ − 1 (bins wholly
  between L and E). Band volume = leg volume at those ticks (C3). Snap band = the band with the lowest volume;
  **ties → the band farthest from L** (the stricter HOLD). Its far edge = its tick farthest from L.
- **C10 — pullback (H2).** From the break second on, track the running b-extreme P (starts at E). The pullback is
  confirmed at the first 1-s row r where b · (P − counter-extreme of that row) ≥ 3.0; the break extreme = P at that
  moment. Rows before 12:45 PT only; none → no trade (counted).
- **C11 — turn bar, entry, class (H2).** On front 1-min RTH bars (CR-CT `BARS`): turn bar t = the first bar with
  index ≥ the bar containing r whose close is beyond the prior bar's close in direction b (b · (c[t] − c[t−1]) > 0);
  t must close by 12:45 PT (t ≤ 134, the 12:44 bar), else no trade (counted). Entry = c[t]. **Pullback extreme** =
  the most b-adverse 1-s price from the break-extreme second through the end of bar t (all known at the entry).
  **HOLD** = b · (pullback extreme − snap far edge) > 0 (never traded into the band); **FILL** = otherwise (into or
  through). Classified at t, before the race.
- **C12 — H2 outcome.** 1-s race ±10 in direction b from entry c[t], over 1-s rows from the start of bar t+1 to
  12:59:59 PT (CR-CT `race1s` semantics: win / loss / tie / unresolved). One trade per touch (the first pullback);
  touches sharing a turn bar and b on a session are one trade (kept: CR-CT family priority).
- **C13 — H2 readouts.** (a) HOLD and FILL vs 50 %. (b) **Bottom-quartile snap:** snap ticks = leg ticks (L … E,
  inclusive, 0.25 grid, zero-volume ticks included) with volume ≤ the leg's p25 tick volume; HOLD iff the pullback
  extreme stays beyond the b-most snap tick. (c) Break 4 / 8 pts, pullback 2 / 4 pts: same rules, each change alone.
  (d) Light / heavy: session volume profile 06:30 → break second on the same L-anchored 1-pt bins; light = the snap
  band's cumulative session volume < the median of the session's non-empty bins; HOLD − FILL within light and heavy.
  (e) vs matched random baseline (key t, b, EDGE of the entry at t).
- **C14 — statistics.** Win rate = wins ÷ (wins + losses), ratio of session-summed counts; ties and unresolved
  excluded and counted. Session-cluster bootstrap (`cr_cv_lib.boot_weights`, 10 000 draws, seed 20260920). Each
  primary: two-sided bootstrap p = 2 · min(P(Δ ≤ 0), P(Δ ≥ 0)); **Holm, m = 2 fixed** (also if one hypothesis is
  underpowered and not run); Holm CI at level 1 − α/(m − rank + 1). **PASS** = Holm p < 0.05 and Holm CI lower > 0;
  **REFUTED** = Holm CI upper < 0 (the stricter of the two CIs); else INCONCLUSIVE; < 40 resolved in either group →
  UNDERPOWERED. Readouts ±20 race and MFE before −10 (CR-CT `mfe1s`) for both; readouts unadjusted, no verdicts.
- **C15 — Step 0 gate.** Per hypothesis: each compared group ≥ 40 events on ≥ 30 sessions (event counts; resolution
  is an outcome and is not computed), and |A/A mean| ≤ 1 pp (CR-CT C12 A/A: 200 session half-splits, seed 20261025,
  keys = that hypothesis's two groups). The run script runs only the cleared hypotheses; an uncleared one is reported
  UNDERPOWERED (or A/A-failed) and not run.
- **C16 — run once.** `results.json` must not exist; the run script asserts the Step 0 gate, the stored cuts and the
  event sets.

## Standing rulings conformance (Dash/research/standing-rulings.md, status current, read 2026-10-27)

| # | Ruling (short) | Mark | How |
|---|---|---|---|
| A1 | levels slow price, not a trade by themselves | honoured | H1 conditions on the approach path; H2 on the break and pullback |
| A2 | a range ≥ minutes, relative | n/a | no range |
| A3 | no automatic entry at the first touch | **deviates — pre-approved** (kickoff) | H1 entry at the level at the touch second (R2 variant, labelled). H2 enters after the break and pullback, not at the touch — honoured there |
| A4 | stops beyond the zone | n/a | stop-free direction test (A15) |
| A5 | flow decides fade vs break; never "always" either | honoured | path / volume-at-price pick the group; both groups scored |
| A6 | skew divergence | n/a (parked) | |
| A7 | approach = vol-scaled swing leg; fixed readout | **deviates — pre-approved** (kickoff) | H1 fixed 10-min window primary, vol-scaled leg readout (C7). H2 has no approach window |
| A8 | put / call levels separately | n/a | levels are CR-CF families, not option levels (CR-CT's strong-line readout had 2 put touches) |
| A9 | two close levels → wait for the second | n/a | |
| A10 | earlier same-day S/R matters | honoured | RUN family in the population |
| A11 | ±10 primary, ±20 readout, favourable excursion | honoured | C12, C14 |
| A12 | levels = chart's strongest colours | n/a | population fixed by the kickoff as CR-CT's first touches |
| A13 | sizing | n/a | |
| A14 | skew vs expected | n/a (parked) | |
| A15 | direction first, stop-free | honoured | ±10 races |
| A16 | skew trend | n/a (parked) | |
| A17 | flow first, skew optional | honoured | price path and volume at price only |
| A18 | range position split | n/a | no range |
| A19 | in-range confirmation | n/a | no range |
| A20 | trade the range edges | n/a | no range |
| A21 | second touch a separate setup | honoured | first touches only |
| B1 | three verdicts | honoured | C14 |
| B2 | many parameters disclosed; plateau | honoured | windows 5 / 10 / 20; break 4 / 6 / 8; pullback 2 / 3 / 4; all reported |
| B3 | one setup at a time | **deviates — pre-approved** (kickoff) | two independent hypotheses, own verdicts, Holm across, nothing tuned across them |
| B4 | train only; R-a, R-b, holdout unread | honoured | C1 guards |
| B5 | Databento reference | honoured | 1-s and 1-min Databento; Ironbeam only for F2's overnight (CR-CT, checked) |
| B6 | mechanical exits labelled | honoured | ±10 race is a direction test, not Ryan's exit |
| B7 | placebos matched EDGE / INTERIOR | honoured | C7, C13(e) |
| B8 | Code labels ≠ Ryan's | n/a | |
| C1, C2 | UI | n/a | |
| Revoked | R1 touch-bar close | **not used** | R2 level entry (H1, labelled); H2 enters at a turn-bar close after a pullback, not the touch bar. R3–R7 not used |

Three "deviates" (A3, A7, B3), all pre-approved in the kickoff; nothing else deviates. Proceeding to Step 0.

## Step 0 amendments (2026-10-27, after Step 0 run 1, before any outcome)

Step 0 run 1 (`logs/cr_cw/step0_run1*.{json,log}`, kept) exposed two code bugs and two confounds. No outcome had been
computed. Fixed / added, then Step 0 re-run (run 2, reported below):

- **C11 bug — turn-bar cut-off.** `T_MAX` was 134 (the 08:44 bar); the 12:44 PT bar is RTH index **374**. Run 1 sent
  every pullback after 08:44 to "no turn" (132 of 333 breaks). Fixed to 374, as C11 states.
- **C6 bug — retrace count.** Run 1 restarted the running extreme after each counted retrace, so one continuing
  counter-move was counted once per point. Now a 1-pt zigzag: each −b leg ≥ 1 pt counts once (median 30 → 18 per
  10 min). Bands 0–1 / 2–4 / ≥ 5 stay as frozen (they turn out degenerate: 399 of 422 are ≥ 5; the median split is
  the usable readout).
- **C7 added readout — efficiency within speed.** Efficiency correlates **0.84** with CR-CT's speed (net ÷ path vs
  net ÷ 10·IM; 10-min path length varies less than net). CLEAN − CHOPPY is therefore largely CR-CT's speed contrast
  again. Added (readout, no verdict): CLEAN − CHOPPY within CR-CT's middle speed tercile (CLEAN 29 / CHOPPY 21 — the
  only tercile with both; top 108 / 4, bottom 4 / 116).
- **C13(f) added readout — snap band without the final band.** The lowest-volume band is the last full band before
  the 6-pt mark (band 5) on **190 / 333** breaks (uniform would be ~55): the leg is cut at the first ≥ 6 second, so
  price spends least time in the band it crosses last. With snap = band 5, HOLD needs the pullback low above
  L + 5.75 after a ≥ 3-pt pullback, i.e. the break must have run to ≈ 8.75+ — so HOLD partly means "the break ran
  further". Added readout (no verdict): snap chosen from bands 0 … nb − 2 (final band excluded).

Primary definitions (H1 and H2) are unchanged.

## Step 0 findings (2026-10-27, run 2; `scripts/cr_cw_step0.py` + `scripts/cr_cw_lib.py` → `scripts/logs/cr_cw/step0.json`; no touch or trade outcome computed)

Read-only, every read inside [2026-01-01, 2026-06-05 21:00 UTC). The only races run are the random-baseline entries
for the A/A (CR-CT C12).

**Data checks.** 101 sessions, CR-CT's 422 first touches reproduced exactly. 1-s front rows 06:30 → 13:00 equal
CR-CT's cache on every h / l / c (2,224,137 rows, 0 mismatches); every close on the 0.25 grid. Largest gap between
trading seconds in RTH: median 6 s, max 14 s per session (no feed holes). Path-efficiency net = CR-CT's speed
numerator on all 422 (max diff 7e-15). **Volume-at-close check (C3): 31.5 % of RTH seconds have high − low > 0.25,
carrying 61.4 % of the volume** — most volume trades in multi-tick seconds, so volume at price is an approximation
at the 1–2-tick scale (the 1-pt bands are 4 ticks).

**H1.** Efficiency (10 min) quantiles p5 / p25 / p50 / p75 / p95 −0.008 / 0.030 / 0.052 / 0.078 / 0.124 (a 10-min
approach nets ~5 % of the distance it travels). Cuts fixed here:

| window | p33.3 | p66.7 | n |
|---|---|---|---|
| **10 min (primary)** | **0.03879** | **0.06722** | 422 |
| 5 min | 0.05454 | 0.10122 | 422 |
| 20 min | 0.02358 | 0.04610 | 403 |
| A7 leg | 0.03870 | 0.07086 | 420 |

**CLEAN 141 (73 sessions), CHOPPY 141 (77)**, MIDDLE 140. Against CR-CT: FAST = CLEAN 74 / CHOPPY 3; SLOW = CLEAN 3 /
CHOPPY 59; NEITHER 53 / 64 — the efficiency × FAST/SLOW readout is degenerate. Efficiency vs speed r = 0.84;
efficiency vs retrace count Spearman −0.03 (retrace count measures something else: it is mostly how much the
approach moved). Retraces per 10 min p10 / p50 / p90 6 / 18 / 35. By family CLEAN / CHOPPY: F1 45 / 46, F2 23 / 37,
RUN 73 / 58.

**H2 funnel (primary, break 6 / pullback 3).** 422 touches → 89 no break by 12:30 → 333 breaks → 333 pullbacks and
turns by 12:45 → 36 duplicate (same session, turn bar and direction) → **297 trades: HOLD 78 (60 sessions),
FILL 219 (95)**. Turn bar median 1 bar after the pullback bar (p90 3). By family F1 114, F2 68, RUN 115; up 165 /
down 132; EDGE 133 / INTERIOR 164. Snap band: median 6 bands; 71 of 333 snap bands have zero volume; snap volume ÷
median band volume p50 0.22.

| variant | HOLD | FILL |
|---|---|---|
| primary (6 / 3) | 78 (60 s) | 219 (95 s) |
| break 4 / 8 | 77 / 69 | 235 / 204 |
| pullback 2 / 4 | 82 / 63 | 224 / 222 |
| bottom-quartile ticks | 39 (39 s) | 258 |
| final band excluded (C13 f) | 118 (73 s) | 179 |
| light / heavy break | 289 / **8** | — |

The light / heavy split is degenerate (the leg's minimum band is almost always below the session median): 8 heavy
breaks, not readable.

**A/A (200 half-splits):** H1 keys 282, mean **+0.23 pp** (SD 3.4, p5 / p95 −5.4 / +5.4); H2 keys 297, mean
**−0.03 pp** (SD 3.0). Baseline win rate 0.500.

**Gate (C15):** H1 CLEAN 141 / 73 s, CHOPPY 141 / 77 s, |A/A| 0.23 → **passed**. H2 HOLD 78 / 60 s, FILL 219 / 95 s,
|A/A| 0.03 → **passed**. Committing the run script (both hypotheses, Holm m = 2), then running once without stopping.

## Result (single run, 2026-10-27) — `scripts/cr_cw_run.py` @ `f662b2d` → `scripts/logs/cr_cw/results.json`

One run, exit 0, no re-run. Event sets, cuts and HOLD / FILL counts asserted equal to Step 0. Races are stop-free ±10
direction tests, not Ryan's exit (B6). H1 entry = the level price at the touch second (R2 level-entry variant,
labelled); H2 entry = the turn-bar close.

### Primaries (Holm m = 2)

| | group A | group B | A − B (95 % CI) | Holm p | Holm CI | verdict |
|---|---|---|---|---|---|---|
| **H1** break, CLEAN − CHOPPY | 54.5 % (73 / 61; n 134, 71 s) | 55.9 % (76 / 60; n 136, 76 s) | **−1.4 pp [−13.7, +11.4]** | 1.00 (raw 0.83) | [−15.5, +13.3] (97.5 %) | **INCONCLUSIVE** |
| **H2** continuation, HOLD − FILL | 48.7 % (37 / 39; n 76, 58 s) | 49.3 % (106 / 109; n 215, 95 s) | **−0.6 pp [−13.5, +12.4]** | 1.00 (raw 0.91) | [−13.5, +12.4] (95 %) | **INCONCLUSIVE** |

### H1 readouts (unadjusted, no verdicts)

- CLEAN vs 50 % +4.5 [−5.2, +13.9]; CHOPPY +5.9 [−2.1, +13.6]; MIDDLE 51.9 %; all 422 first touches 54.1 % (as CR-CT).
- ±20: −4.1 [−19.0, +10.9] (48.6 % vs 52.8 %). MFE before −10: median 10.0 vs 12.25 pts, mean diff −0.3 [−6.6, +6.1];
  reached +10 52 % vs 54 %.
- vs matched random (B7): CLEAN +3.2 [−6.3, +12.8], CHOPPY +5.6 [−2.6, +13.0].
- **Efficiency at equal speed** (CR-CT middle speed tercile): CLEAN 50.0 % (n 24) vs CHOPPY 60.0 % (n 20), −10.0
  [−43.4, +21.6]. Within CR-CT NEITHER: −1.0 [−20.8, +19.3] (49 / 61). Within FAST / SLOW and the outer speed
  terciles one side has 2–4 trades: not readable.
- Retrace count low (≤ 18) vs high: 56.9 % vs 51.2 %, +5.6 [−5.1, +16.6]. Bands: 2–4 retraces 80 % (n 15), ≥ 5 53.1 %.
- Windows: 5 min −2.6 [−15.2, +9.7]; 20 min +10.1 [−1.7, +21.7]; A7 leg +0.6 [−12.6, +14.0]. No plateau (sign flips).
- Family: F1 +2.7, F2 −10.5, RUN +0.4, all CIs ±20 pp.

### H2 readouts (unadjusted, no verdicts)

- All 297 trades 49.1 % (n 291), vs matched random −0.9 [−6.9, +5.5]: buying the first turn after a 3-pt pullback
  from a ≥ 6-pt break is a coin flip. HOLD vs 50 % −1.3 [−12.7, +9.7]; FILL −0.7 [−8.3, +6.9].
- ±20: −2.7 [−17.9, +11.9]. MFE median 7.9 vs 9.5 pts, mean diff −0.2 [−5.6, +5.4].
- Snap variants: bottom-quartile ticks +8.7 [−8.5, +25.3] (HOLD 56.8 % n 37 vs 48.0 %); final band excluded −0.7
  [−12.7, +10.7] (n 115 vs 176); snap = final band −4.7, snap earlier −4.0.
- Break 4 / 8: −3.1 / +4.8. Pullback 2 / 4: −1.1 / −2.9. No plateau; every CI covers 0.
- Light / heavy: heavy n 8, not readable. Family: F1 −1.2, F2 +6.9, RUN −3.1.

### Reading

- **H1: no.** A clean, low-back-ticking push into a level broke 54.5 %; a choppy one 55.9 %. Path efficiency is mostly
  CR-CT's speed again (r 0.84), and at equal speed the small sample leans the wrong way. The course's "real momentum"
  path test does not separate breaks from failures at ±10 on these levels.
- **H2: no.** After a ≥ 6-pt break and a ≥ 3-pt pullback, continuation is 49 % whether the pullback held above the
  snap band or filled it. Caveats: the lowest-volume band is mostly the last band the leg crossed (construction), and
  volume at the close is a 1–2-tick approximation (61 % of volume trades in multi-tick seconds). The variant that
  avoids the final-band artefact also reads −0.7.
- Nothing shortlisted. The one readout with a CI excluding 0 (2–4 retraces, 80 % n 15) is one of ~40 unadjusted
  readouts on a cell of 15.
