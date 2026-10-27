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
