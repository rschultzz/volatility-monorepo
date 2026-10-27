# CR-CU — Replicate CR-CT's second-touch reversal on 2023-05-01 → 2024-04-02

Spec freeze. Branch `feat/CR-CU-second-touch-2023` off `origin/main` `cd2fe36` (PR #101 merge, CR-CT), in its own
worktree (`.claude/worktrees/cr-cu`). Read-only against the database; no table writes, no cron, no deploy, no display
change. PR, don't merge.

## Spec body (kickoff, verbatim, Ryan in chat 2026-10-26)

> CR-CU — Replicate CR-CT's second-touch reversal on 2023-05-01 → 2024-04-02 (read-only,
> pre-registered, train-era data not yet read for this question).
>
> Branch first: git checkout -b feat/CR-CU-second-touch-2023 off origin/main (after #101). Own worktree.
> Read CLAUDE.md, field-conventions.md (2023 quote lag, ES.c.0 roll-week volume, third-Friday rows),
> standing-rulings.md, and the CR-CT and CR-CO specs. Freeze the spec with the conformance table.
> Pre-approved by Ryan, record don't stop: A3 / R2 level-price entry at the touch second (as CR-CT);
> A7 fixed 10-min approach window primary (as approved in CR-CT). Stop on any other "deviates".
>
> Hypothesis (CR-CT readout, train 2026: second touch, FAST break rate 36 % (n 39) vs SLOW 64 % (n 58),
> −27.9 pp [−46.9, −7.9]; one of ~20 readouts, so a lead): on the SECOND touch of a level, a FAST, heavy
> approach breaks LESS often than a SLOW, light one.
> Definitions EXACTLY as CR-CT ran them: level families (F1 prior RTH H/L/C, F2 overnight H/L, running
> RTH high/low ≥ 30 min old), second-touch definition, approach window, speed (÷ session implied move),
> relative volume (trailing 20-session median, same clock window), FAST / SLOW using CR-CT's FROZEN
> train tercile cuts (do not re-fit), level-price entry at the touch second, 1-s race ±10 in the break
> direction by 13:00 PT. Reuse CR-CT's code; switch the date range only. If 1-s data does not exist
> for 2023, use 1-min Databento bars for the race and report the change as C1 BEFORE loading outcomes.
> Primary: break win rate SLOW minus FAST on second touches, session-cluster bootstrap, 10 000 draws,
> seed 20260920. PASS if CI lower bound > 0; REFUTED if CI upper bound < 0; else INCONCLUSIVE; ≥ 40
> resolved in EACH cell, else underpowered. One hypothesis.
> Data: 2023-05-01 → 2024-04-02, Databento ES bars; apply CR-CO's exclusions (2023-09, roll-week
> Wed–Fri, early closes). R-a (from 2024-04-03), R-b, 2026 train and holdout NOT read.
> Readouts: first touch FAST vs SLOW (CR-CT primary, replicated); each cell vs 50 %; ±20 race; by level
> family; 5 / 20-min window; matched random baseline (B7 EDGE key).
> Step 0 (no outcomes): sessions after exclusions, second touches per cell with the frozen cuts, A/A.
> If each cell ≥ 40 on ≥ 30 sessions and |A/A mean| ≤ 1 pp, commit the run script and run once without
> stopping; else stop and report.
> Results + plain-English summary to the vault CR note, setup-registry, session note; push, PR, don't
> merge; copy logs before removing the worktree. Summary for Ryan: on 2023 data, does a fast push into
> a level on its second touch fail more often than a slow one?

## C1 — 1-s data does not exist for 2023 (reported before any outcome is loaded)

`es_flow_1s_databento` and `es_flow_1m_databento` start 2024-04-03 (field-conventions; CR-CO O4). In 2023-05-01 →
2024-04-02 the only Databento ES series is **`ironbeam_es_1m_bars` with `source = 'databento'`**: 1-min bars fetched as
`ES.c.0` (calendar-roll continuous, no contract column). So, as the kickoff directs:

- **Touch "second" becomes the touch bar.** The touch is CR-CT's C4 touch bar j. CR-CT's C6 drop (no 1-s print at the
  level in the minute) cannot happen: bar j reaches L by definition. **Entry = L** (R2 level-entry variant, labelled),
  as CR-CT.
- **The race runs on 1-min bars from bar j + 1** to the 12:59 PT bar: win = b·(high/low − L) ≥ +10 first; loss =
  ≤ −10 first; tie = both in one bar; unresolved = neither. The rest of bar j after the touch is **not credited either
  way**, because the order of prices inside a minute is unknown (in CR-CT the race started the second after the touch,
  so it saw the rest of minute j). The run reports how many touches already reached +10 or −10 inside bar j.
- **Volume** is the `ES.c.0` bar volume (one series). CR-CT summed all outright symbols in the minute (roll-robust);
  outside roll weeks `ES.c.0` is the front contract and carries nearly all volume.
- The random baseline (CR-CT C12) already raced on 1-min bars from m + 1; it is unchanged, so the touch race and the
  baseline race now use the same bar resolution.

## Choices the brief doesn't fix (Code, 2026-10-26, before any 2023 price was read)

Everything not listed is CR-CT as shipped (`specs/CR-CT-approach-speed.md` C1–C15, E-R2), reused through
`scripts/cr_ct_lib.py` functions (`merge`, `run_levels`, `touches`, `approach`, `prior_median`, `edge_at`, `classify`,
`baseline_table`) and `scripts/cr_cl_lib.py` (`boot_weights`, `strata_matrices`, `matched_diff`). The new module
`scripts/cr_cu_lib.py` replaces only the data loaders, the universe, the F1/F2 contract guard, and the 1-s outcome
functions (with 1-min ones).

- **U1 — data window and guard.** Every read inside [2023-05-01 00:00 PT, 2024-04-02 13:00 PT] (asserted), so no R-a
  row (from 2024-04-03), 2025, 2026 train or holdout row is read. Bars: `ironbeam_es_1m_bars`, 06:15 → 12:59 PT starts
  (CR-CT's ext window), plus 15:00 → 06:30 PT for F2. Step 0 asserts every row in the window is `source = 'databento'`
  (CR-CO found no other source). IM: canonical `bt_daily_features.implied_move_1d` (SPX, frame A; a scale only).
  Nothing is read before 2023-05-01, so session 2023-05-01 has no F1 / F2 (no prior), as CR-CT's first priors.
- **U2 — universe.** NYSE sessions 2023-05-01 → 2024-04-02 through CR-CT's funnel (CR-CP C4 / C18: no bars, starts
  after 06:31, early close = last RTH bar before 12:55 PT, missing-minute run > 5, no canonical IM), then **CR-CO's
  exclusions verbatim** (`scripts/logs/cr_co/exclusions.json`, embedded as a constant): quote-time month 2023-09 (20
  sessions) and the roll-contaminated sessions (expiry-week Wed–Fri plus 2023-09-12). Early closes (2023-07-03,
  2023-11-24) fall to the funnel. Expiry-week Mondays and Tuesdays stay in (CR-CO kept them).
- **U3 — F1 / F2 contract guard.** CR-CT required the prior session's front symbol to equal today's. `ES.c.0` has no
  symbol, so the equivalent: F1 and F2 are **missing** when the prior session is a quarterly expiry Friday
  (2023-06-16, 09-15, 12-15; 2024-03-15) or one lies between the prior session and today. The 06:29 Ironbeam-vs-Databento
  check (CR-CF) is kept as written; here both sides are the same series, so it only checks that the 06:29 bar exists.
- **U4 — relvol priors.** CR-CT C8 unchanged (up to 20 most recent prior sessions with bars, last bar ≥ 12:55 PT, ≥ 10
  required), from 2023-05-01 only (so touches in roughly the first two weeks have no relvol, as CR-CT's first two
  weeks). **Roll-contaminated sessions are not priors** (their `ES.c.0` volume is the dying contract at 0.12–0.35×).
  2023-09 sessions **are** priors: they were excluded for ORATS quote timing, which ES volume doesn't depend on.
- **U5 — frozen cuts.** CR-CT's Step 0 cuts, embedded and asserted equal to `scripts/logs/cr_ct/step0.json` where
  that file exists: 10 min **[0.007250746319479294, 0.014000729018632885]** (primary); 5 min [0.010157656953575242,
  0.021024723131091827]; 20 min [0.004038630778980846, 0.009603635524005163]. No re-fit. FAST = speed ≥ upper cut and
  relvol ≥ 1; SLOW = speed ≤ lower cut and relvol < 1 (CR-CT C9).
- **U6 — event sets.** Touches, arming, break direction and the second touch exactly as CR-CT C4 (the next touch after
  re-arming on either side); pooled duplicates as C5 per touch number. **Primary set = deduped second touches**;
  readout set = deduped first touches.
- **U7 — primary statistic.** Break win rate = wins ÷ (wins + losses), ratio of session-summed counts; **SLOW − FAST**;
  `boot_weights` 10 000 draws, seed 20260920, percentile 95 % CI; B1 verdict; ≥ 40 resolved in each of FAST and SLOW,
  else UNDERPOWERED. One hypothesis, no Holm.
- **U8 — readouts (no verdicts, unadjusted).** First touches FAST − SLOW (CR-CT's primary sign); second-touch FAST vs
  50 % (break) and SLOW vs 50 % (break); all second touches vs 50 %; ±20 race SLOW − FAST; second touches by level
  family; windows 5 / 20 min on their frozen cuts; matched random baseline (CR-CT C12 key (j, b, EDGE)) for second-touch
  FAST, SLOW and all; MFE before −10 (A11; 1-min, bar j not credited); count of touches with a ±10 excursion inside
  bar j; sensitivity: primary without expiry-week Mondays and Tuesdays (dying-contract volume, U2). CR-CT's A7-leg,
  before-she-goes, strong-line and A8 readouts are **not run** (not in the kickoff; the strong-line placement is undefined
  before 2024-04-03, CR-CO O10).
- **U9 — A/A.** CR-CT C12 / CR-CP §7 unchanged (200 session half-splits, seed 20261025) with keys = the FAST ∪ SLOW
  **second** touches.
- **U10 — run once.** Step 0 = `scripts/cr_cu_step0.py` (no touch outcome; the only races are baseline entries for the
  A/A). Gate: FAST and SLOW second touches each ≥ 40 on ≥ 30 sessions and |A/A mean| ≤ 1 pp. If it passes, commit
  `scripts/cr_cu_run.py`, then run it once: it asserts the gate, the event set and the cuts, and refuses if
  `results.json` exists. A crash before any outcome is printed or saved allows a fix-and-rerun, logged. Connection:
  `get_backfill_db_conn()`, read-only transaction asserted, SELECTs only. Outputs `scripts/logs/cr_cu/` (git-ignored),
  copied to the main checkout before the worktree is removed. Interpreter `apps/web/.venv/bin/python`.

## Standing rulings conformance (Dash/research/standing-rulings.md, status current, read 2026-10-26)

| # | Ruling (short) | Mark | How |
|---|---|---|---|
| A1 | levels slow price, not a trade by themselves | honoured | the call is conditioned on the approach (speed + volume), not on the level alone |
| A2 | a range ≥ minutes, relative | n/a | no range (second touch, A21) |
| A3 | no automatic entry at the first touch | **deviates — pre-approved** (kickoff) | entry at the level price at the touch (R2 level-entry variant, labelled), as CR-CT; the primary is the second touch, which A21 names as a possible entry |
| A4 | stops beyond the zone | n/a | stop-free direction test (A15) |
| A5 | flow decides fade vs break; never "always" either | honoured | speed + volume pick the side per touch; both outcomes scored |
| A6 | skew divergence | n/a (parked) | |
| A7 | approach = vol-scaled swing leg; fixed readout | **deviates — pre-approved** (kickoff; as approved in CR-CT) | fixed 10-min window primary; the A7 leg readout is not run (U8) |
| A8 | put / call levels separately | n/a | no option levels; strong-line placement undefined before 2024-04-03 (CR-CO O10) |
| A9 | two close levels → wait for the second | n/a | |
| A10 | earlier same-day S/R matters | honoured | RUN (running RTH high / low ≥ 30 min old) in the event set |
| A11 | ±10 primary, ±20 readout, favourable-excursion distribution | honoured | U7, U8 (±20, MFE) |
| A12 | levels = chart's strongest colours | n/a | primary families are CR-CF's, set by the kickoff; strong lines undefined in this era |
| A13 | sizing | n/a | |
| A14 | skew vs expected | n/a (parked) | |
| A15 | direction first, stop-free | honoured | ±10 race |
| A16 | skew trend | n/a (parked) | |
| A17 | flow first, skew optional | honoured | volume / speed only; no skew |
| A18 | range position split | n/a | no range |
| A19 | in-range confirmation | n/a | no range |
| A20 | trade the range edges | n/a | no range (the A3 deviation) |
| A21 | second touch a different setup; separate cells | honoured | second touch is the primary cell; first touches a separate readout; not pooled, not re-armed past the second |
| B1 | three verdicts | honoured | U7 |
| B2 | many parameters disclosed; plateau | honoured | all readouts reported; frozen cuts, nothing chosen |
| B3 | one setup at a time | honoured | single hypothesis |
| B4 | data split | honoured | 2023-05-01 → 2024-04-02 is train-era; R-a, R-b, 2026 train, holdout not read (U1 guard). CR-CT's frozen cuts and CR-CO's exclusion list are read from logs, not data |
| B5 | Databento reference | honoured, with a note | the only Databento ES series in 2023 is the `ES.c.0` 1-min bars (C1, CR-CO O4); roll weeks handled by CR-CO's exclusions |
| B6 | mechanical exits labelled | honoured | ±10 race is a direction test, not Ryan's exit |
| B7 | placebos matched EDGE / INTERIOR | honoured | U8 baseline key (j, b, EDGE) |
| B8 | Code labels ≠ Ryan's | n/a | |
| C1, C2 | UI | n/a | |
| Revoked | R1 touch-bar close | **not used** | R2 level entry, labelled; R3–R7 not used |

Two "deviates" (A3, A7), both pre-approved by Ryan in the kickoff; nothing else deviates. Proceeding to Step 0.

## Step 0 findings (2026-10-26; `scripts/cr_cu_step0.py` + `scripts/cr_cu_lib.py` → `scripts/logs/cr_cu/step0.json`; no touch outcome computed)

One launch, exit 0. Read-only role and transaction asserted; every read inside [2023-05-01 00:00 PT, 2024-04-02 13:00
PT]. Embedded constants checked against the main-checkout logs: CR-CT cuts (`cr_ct/step0.json`) and CR-CO exclusions
(`cr_co/exclusions.json`) equal. Every bar read is `source = 'databento'`. The only races run are the random-baseline
entries for the A/A.

**C1 confirmed:** no 1-s (or 1-min Databento flow) table covers 2023, so the touch is the 1-min touch bar and the race
runs on `ES.c.0` 1-min bars from bar j + 1 (as frozen above). Reported here, before any outcome was loaded.

**Universe (U2):** 234 NYSE sessions → no Databento RTH bars 3 (expiry Fridays 2023-09-15, 12-15, 2024-03-15) → early
close 3 (2023-06-16 expiry Friday with 1 bar, 07-03, 11-24) → no canonical IM 10 → CR-CO roll-contaminated 10 → CR-CO
2023-09 15 (5 more of its 20 already out) → **191 sessions**. IM p10 / p50 / p90 21.0 / 34.6 / 54.5 pts (2026 train
median 67). 216 relvol priors.

**Levels per session** (median [IQR]): F1 3 [2, 3], F2 2 [2, 2], RUN 4 [3, 5]. F1 and F2 missing on 3 sessions each:
2023-05-01 (no prior read) and the two Mondays after an expiry Friday in the window (U3).

**Touches** (n / sessions, duplicates C5):

| | F1 | F2 | RUN |
|---|---|---|---|
| first touch | 253 / 147 (1 dup) | 215 / 177 (15 dup) | 430 / 182 (31 dup) |
| second touch | 193 / 125 | 168 / 153 (4 dup) | 261 / 150 (15 dup) |

Primary set (deduped second touches): **603 on 179 sessions**, RUN 246 / F1 193 / F2 164; break up 293 / down 310;
EDGE 153 / INTERIOR 450; 14 on expiry-week Mon / Tue. Relvol undefined on 26 (< 10 priors, the first weeks of May).

**Classes on CR-CT's frozen cuts (no re-fit):**

| window | second touch FAST | SLOW | NEITHER | undefined | first touch FAST | SLOW |
|---|---|---|---|---|---|---|
| **10 min (primary)** | **73 (57 s)** | **164 (95 s)** | 340 | 26 | 188 (105 s) | 199 (117 s) |
| 5 min | 94 (70) | 135 (90) | 348 | 26 | 204 (120) | 170 (101) |
| 20 min | 69 (51) | 157 (89) | 351 | 26 | 165 (91) | 171 (98) |

Second touches by family, FAST / SLOW: F1 26 / 46, F2 16 / 48, RUN 31 / 70 (FAST cells < 40, so readout only). Without
expiry-week Mon / Tue: FAST 73, SLOW 158. **Era note:** on the frozen cuts, 57 % of 2023 second touches are at or below
the lower speed cut and 19 % at or above the upper (first touches 37 % / 36 %). So second-touch approaches in 2023 were
slower per unit of IM than the train first touches the cuts came from. Relvol p10 / p50 / p90 0.62 / 1.02 / 2.11.

**A/A (U9, 200 splits, keys = 237 second-touch FAST ∪ SLOW):** mean **+0.26 pp**, median +0.10, SD 3.2, p5 / p95
−4.8 / +5.3; 3.0 % of keys unmatched. Baseline 116,362 resolved entries (win rate 0.500 by construction), EDGE share
0.24.

**Gate:** FAST 73 ≥ 40 on 57 ≥ 30 sessions; SLOW 164 ≥ 40 on 95 ≥ 30; |A/A| 0.26 ≤ 1 pp → **passed**. Committing the
run script, then running once without stopping.

## Result (single run, 2026-10-26) — `scripts/cr_cu_run.py` @ `bdccf9f` → `scripts/logs/cr_cu/results.json`

One run, exit 0, no re-run. Gate, session count (191), event sets (603 second / 851 first touches), class counts and the
frozen cuts asserted equal to Step 0. Entry = the level price at the touch bar (R2 level-entry variant, labelled); ±10
stop-free race on `ES.c.0` 1-min bars from the next bar to 13:00 PT (C1). This is a direction test, not Ryan's exit (B6).

### Primary — break win rate, SLOW − FAST, second touches (±10)

| cell | n | resolved (sessions) | break win rate |
|---|---|---|---|
| SLOW (slow + light) | 164 | 140 (89); 24 unresolved | **54.3 %** (76 / 64) |
| FAST (fast + heavy) | 73 | 63 (51); 10 unresolved | **54.0 %** (34 / 29) |
| **SLOW − FAST** | | | **+0.3 pp [−14.8, +15.8] → INCONCLUSIVE** |

Both cells ≥ 40 resolved, so the verdict is not underpowered. Train's readout was SLOW − FAST **+27.9** [+7.9, +46.9].
That value lies **above this CI's upper bound**, so an effect as large as train's is not supported by 2023. The CI
still allows a moderate effect either way.

### Readouts (no verdicts, unadjusted)

- Second-touch cells vs 50 % (break): FAST +4.0 [−10.0, +17.7]; SLOW +4.3 [−4.2, +13.0]. All second touches broke
  50.5 % (509 resolved), +0.5 [−5.1, +6.0]; NEITHER 48.2 %.
- **First touches (CR-CT's primary, replicated): FAST − SLOW +4.3 [−7.5, +16.0]** (FAST 47.3 % n 169, SLOW 43.0 %
  n 158). All first touches broke 49.0 % (729), −1.0 [−5.7, +3.6]; in 2026 train it was 54.1 %.
- ±20 race SLOW − FAST +0.6 [−22.1, +24.9] (49.3 % n 67 vs 48.6 % n 37; most ±20 races unresolved).
- MFE before −10 (1-min): SLOW mean 12.4 vs FAST 13.5 pts, −1.1 [−5.6, +3.0]; medians 8.9 / 9.3; reached +10 46 % /
  47 %.
- Family SLOW − FAST: F1 +11.4 [−15.7, +37.5] (44 / 24); F2 +1.0 [−29.1, +33.8] (43 / 14); RUN −10.7 [−34.7, +13.0]
  (53 / 25). The FAST cells are all < 40.
- Windows (frozen cuts): 5 min +6.4 [−8.3, +21.5] (118 / 81); 20 min +4.4 [−11.7, +20.4] (136 / 58). Same sign as train,
  small, CIs cover 0.
- Without expiry-week Mon / Tue: −0.6 [−15.8, +14.9] (135 / 63).
- Break down / up: −3.5 [−24.7, +18.9] / +5.6 [−16.9, +28.3]. EDGE −3.3 [−38.8, +29.8]; INTERIOR +3.2 [−14.3, +21.2].
- Matched random baseline (key j, b, EDGE; coverage 1.00): second-touch FAST +4.2 [−10.0, +18.2], SLOW +3.6 [−4.6,
  +12.1], all +0.7 [−4.9, +6.2].
- C1 disclosure: bar j reached −10 from L inside the touch minute on 5 of 603 second touches (SLOW 1, NEITHER 4) and
  +10 on none. The uncredited part of bar j barely matters.

### Reading

- **The second-touch reversal did not replicate in 2023.** Slow, light second approaches broke 54.3 % and fast, heavy
  ones 54.0 %, essentially the same. Train's 28-pp gap is outside this CI, so train's readout was most likely chance
  (one of ~20 readouts) or specific to 2026. Not shortlisted.
- On 2023 data, neither first nor second touches show a momentum default (49–50 % break); 2026 train first touches
  broke 54 %.
- Caveats: the race is on 1-min bars (C1), not 1-s, and in 2023 the frozen cuts put 57 % of second touches in the slow
  tail and 19 % in the fast tail, so the FAST cell is the thin one (63 resolved).
