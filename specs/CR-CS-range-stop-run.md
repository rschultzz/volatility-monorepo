# CR-CS — Stop run at a range edge: fade the failed sweep

Read-only, pre-registered, train first, single hypothesis. Branch `feat/CR-CS-range-stop-run` off `origin/main` `57adce0`,
own worktree (`.claude/worktrees/cr-cs`). Kickoff: Ryan, 2026-10-26 (chat). Frozen 2026-10-26 **before any CR-CS data was
read**.

---

## Spec body (kickoff, verbatim)

> CR-CS — Stop run at a range edge: fade the failed sweep (read-only, pre-registered, train first).
>
> Branch first: git checkout -b feat/CR-CS-range-stop-run off origin/main. Own worktree.
> Read CLAUDE.md, field-conventions.md, standing-rulings.md, and the CR-CP and CR-CR specs (reuse
> their range code). Freeze specs/CR-CS-range-stop-run.md with the conformance table; stop on any
> "deviates". List choices the brief doesn't fix (C1, C2, …) before loading data.
>
> Idea (Ryan, from the Jigsaw "volume stop hit and run" / failed-breakout concept, and A8/A18
> overshoot): after a clear range, price sweeps one edge fast on a volume burst (stops trigger),
> then closes back inside the box. Fade it. CR-CP's void rule discarded exactly these events.
>
> Definitions (fixed before any outcome):
> - Range: EXACTLY CR-CP after A0/A1 (vol-scaled swing approach with start confirmed, closes box
>   ≤ 12 pts, ≥ 15 min, box edges = highest/lowest close). Box frozen at qualification q.
> - Sweep (after q, before 12:30 PT): the first bar whose HIGH exceeds the box high by ≥ 2 pts
>   (mirror for the low), with sweep volume ≥ 2× the median 1-min volume of the range bars.
>   Sweep window = first breaking bar → bar of the sweep extreme, max 5 min. First sweep per edge
>   per range only (not re-armed).
> - Failure: within 5 min of the sweep's first bar, a bar CLOSES back inside the box. No re-entry
>   within 5 min → breakout: no trade (counted; readout: trade the breakout direction from the
>   5-min close).
> - Entry: close of the re-entry bar, in the fade direction (short after an up-sweep, long after a
>   down-sweep). Stop reference (readouts only): sweep extreme + 1 pt.
> - Outcome (A15, A11): stop-free race from entry, first to ±10 pts by 13:00 PT; ties / unresolved
>   excluded and counted.
> Primary (paired, same entries): fade win rate minus the continuation side (same entry price and
> time, other direction). Session-cluster bootstrap, 10 000 draws, seed 20260920. PASS if CI lower
> bound > 0; REFUTED if CI upper bound < 0; else INCONCLUSIVE; ≥ 40 resolved fades, else
> underpowered. One hypothesis.
> Readouts (not gated): ±20 race; move at +30 / +60 min; max favourable move before −10; share
> stopped at sweep extreme + 1 before +10; win vs a matched random baseline (B7 EDGE key);
> sweep size ≥ 1 / 4 pts; volume ratio 1.5 / 3; re-entry within 3 / 10 min; sweep delta share
> (aggressive flow in the sweep: top vs bottom half); up vs down sweeps; edge at a strong line (A12),
> sweep beyond a strong line (A18), put vs call (A8); hour of day; breakout-direction trades.
> Data: train 2026-01-06 → 2026-06-05, Databento 1-min bars/flow (es_flow_1m_databento). R-a, R-b,
> holdout NOT read.
> Step 0 (no outcomes): ranges, sweeps per edge, failures vs breakouts, fills, counts per readout
> cell, A/A on the random baseline. If ≥ 40 fades on ≥ 30 sessions and |A/A mean| ≤ 1 pp, commit
> the run script and run once without stopping; else stop and report.
> Results + plain-English summary to the vault CR note, setup-registry, session note; push, PR,
> don't merge; copy logs before removing the worktree. Plain-English summary for Ryan: when a
> range edge gets swept and price comes straight back in, does fading it work?

Everything not named here is CR-CP's definition as merged (`specs/CR-CP-range-edge-flow.md`: D0, D1, C1–C6, C13, C14,
C17, C18, Approvals, A0, A1 incl. A1.a), unchanged. CR-CP's D2–D4 (flow call, decision bar, edge fill) are **not** used.

## Standing rulings conformance (Dash/research/standing-rulings.md, status current, read 2026-10-26)

| # | Ruling (short) | Mark | How |
|---|---|---|---|
| A1 | levels slow price, not a trade; range in a zone around a level | honoured | level-agnostic primary; strong-line position is a readout |
| A2 | 2–3 min is not a range; up to ~30 min | honoured | CR-CP's range (≥ 15 min hold) unchanged |
| A3 | don't auto-enter at the first touch; entry after the range forms | honoured | entry only after the range qualifies, a sweep and a close back inside |
| A4 | stops beyond the zone | honoured (readout) | primary is stop-free (A15); the stop readout sits beyond the sweep extreme, i.e. beyond the zone |
| A5 | flow and skew decide fade vs break at the range edges; never "always fade" | honoured — **interpretation flagged** | fade vs break is not fixed: it is decided at the edge by a volume burst (order flow) plus whether price is accepted outside (breakout → no fade; breakout-direction trade as a readout). Delta in the sweep is a readout, not the trigger: the kickoff fixes volume as the trigger |
| A6 | skew divergence direction | n/a (parked) | no skew |
| A7 | approach = vol-scaled swing, 12 × IM ÷ 65.71 | honoured | CR-CP D0 / A0 unchanged |
| A8 | put levels overshoot; analyse put and call separately | honoured (readout) | put vs call split of the strong line at / pierced by the sweep (C13) |
| A9 | two close levels → wait for the second | n/a | not in the kickoff |
| A10 | earlier same-day S/R | n/a | not in the kickoff |
| A11 | target open; ±10 primary, ±20 readout, MFE | honoured | |
| A12 | levels = chart's yellow / darkest (net ≥ day p95 \|net\|) | honoured (readout) | CR-CM strong set via CR-CP `load_levels` |
| A13 | sizing | n/a | |
| A14 | skew vs expected | n/a (parked) | |
| A15 | direction first, stop-free | honoured | primary race is stop-free |
| A16 | skew trend | n/a (parked) | |
| A17 | two stages; flow in the range decides; skew optional, flow alone first | honoured — **interpretation flagged** | flow alone, no skew. The trigger is the sweep's volume relative to the range's own volume (the range bars set the baseline) plus the failure; in-range delta (CR-CP's D2) is not the trigger here, by the kickoff's definition. Recorded so Ryan can disagree; not marked "deviates" because the kickoff (chat-checked against this file) fixes the trigger |
| A18 | range before or beyond the level; overshoot | honoured (readout) | "sweep beyond a strong line" split |
| A19 | in-range confirmation (effort, no progress) | n/a | not the trigger in this setup |
| A20 | trade the range edges | honoured | entry at the close back inside, at the swept edge |
| A21 | second touch is a different setup | n/a | |
| B1 | three verdicts | honoured | |
| B2 | many parameters fine if disclosed; plateau not peak | honoured | grid reported in full; primary not re-selected |
| B3 | one setup at a time | honoured | |
| B4 | train ≤ 2026-06-05; R-a / R-b / holdout not read | honoured | CR-CP's train guards (C17): every read ≥ 2026-01-01 and < 2026-06-05 21:00 UTC |
| B5 | Databento is the reference | honoured | `es_flow_1m_databento` bars and flow |
| B6 | non-Ryan exits are a mechanical cousin, label them | honoured | ±10 race = direction test; "stop at sweep extreme + 1" = labelled mechanical readout |
| B7 | placebos matched EDGE vs INTERIOR | honoured | CR-CP C13 baseline, key (minute, side, EDGE) |
| B8 | Code labels ≠ Ryan's labels | n/a | |
| C1, C2 | UI | n/a | |

Revoked / superseded: R1 (touch-bar-close entry) not used — the entry is the close of the bar that re-enters the box after
a sweep, after the range has qualified; R2 n/a; R3 (CR-BW exit) not used; R5 not used; R6 not used (Corrected placement
via CR-CM); R7 not used.

**No "deviates".** A5 and A17 carry an interpretation flag (the kickoff's trigger is a volume burst + failure, not
in-range delta); listed for Ryan, not a stop.

## Choices the kickoff doesn't fix (Code, 2026-10-26, before any data)

- **C1 — range and universe.** `cr_cp_lib` with `CR_CP_PERIOD=train` (asserted), `PRIMARY` (close box, close2 restart,
  vol swing), `build` (101-session universe of CR-CP Step 0, IM_train 65.71 asserted), `approach_legs`, `find_range`:
  one range per approach leg, box [lo, hi] = lowest / highest close over r0..q, frozen at q. CR-CP's caches
  (`m1_train.parquet`, `levels_train.pkl`, written by the same code from the same guarded queries) are copied from the
  main checkout's `scripts/logs/cr_cp/` and checked (rows, date span) instead of re-querying. CR-CS outputs go to
  `scripts/logs/cr_cs/`.
- **C2 — armed span.** A range is armed from bar q + 1. A break bar b must start before 12:30 PT (index < 360). The
  range is disarmed by a breakout on either edge (C5) or after both edges have broken. After a failure on one edge the
  other edge stays armed. A bar that breaks both edges at once: counted, no trade, range ends.
- **C3 — break (the sweep's first bar).** Up: the first bar b > q with H[b] ≥ hi + 2 (box hi = highest close). Down
  mirror with L[b] ≤ lo − 2. Each edge's **first** break is its only candidate (not re-armed), whether or not it passes
  the volume test.
- **C4 — volume burst and sweep window.** Re-entry bar r (C5) or, with none, b + 4. Sweep extreme x = the bar of the
  highest high (up) / lowest low (down) over [b, min(r, b + 4)], first occurrence; window = b..x (≤ 5 bars). Burst if
  the **mean 1-min volume over the window** ≥ 2 × the median 1-min volume of the range bars r0..q. (Mean, not sum: a
  sum over up to 5 bars against a 1-min median would make the 2× bar nearly free.) A break without a burst: counted
  ("no burst"), no trade; its failure / breakout is still recorded and a breakout still disarms the range.
- **C5 — failure vs breakout.** Failure = the first r ∈ [b, b + 4] with c[r] ≤ hi (up sweep) / c[r] ≥ lo (down), i.e.
  the bar closes no later than 5 min after b's start, back at or inside the swept edge (a close beyond the other edge
  also counts). None → breakout.
- **C6 — entry and sides.** Fade: h = −1 after an up sweep, +1 after a down sweep, at c[r]. Continuation: −h at the same
  price and bar. **Because both start at the same price and the ±10 race is symmetric, fade-win ⇔ continuation-loss
  and the resolved sets are identical: the primary equals 2 × (fade win rate − 50 %)**; it is reported as the kickoff
  states, with fade vs 50 % alongside.
- **C7 — race.** CR-CL `race` from bar r (bars r + 1 … 12:59 PT, ±10). The entry bar's own range is not used (entry at its
  close). Ties and unresolved excluded and counted. (C18 flat fill: no gaps.)
- **C8 — overlapping ranges.** Two ranges of one session can be armed together. A trade is keyed (session, r, h); a key
  already taken by an earlier-qualifying range is dropped (counted).
- **C9 — statistic.** CR-CP run's `compare` (fade = "chosen", continuation = "opposite"), session-cluster bootstrap
  (`CL.boot_weights`, 10 000, seed 20260920), percentile 95 % CI; B1 verdict on the chosen − opposite CI; ≥ 40 resolved
  fades else UNDERPOWERED.
- **C10 — random baseline (B7) and A/A.** CR-CP C13: `baseline_table` (every session-minute, both sides, entry at the
  close, EDGE through that minute); trade key (r, h, EDGE tag at r through r). A/A as CR-CP Step 0 (200 random session
  halves, seed 20261025) on the fade keys.
- **C11 — readout grid** (one swap each, the rest primary): sweep size 1 / 4 pts (C3 threshold); volume ratio 1.5 / 3
  (C4); re-entry within 3 / 10 min (C5 window [b, b + 2] / [b, b + 9]; the sweep window stays ≤ 5 bars).
- **C12 — other readouts.** ±20 race; move at +30 / +60 (CR-CL `move_at`); max favourable before −10 (`mfe_before`);
  **stop readout (mechanical, B6):** from entry, stop = sweep extreme over [b, r] + 1 pt beyond it vs +10, first
  touched; same bar → "tie"; neither → "neither"; share stopped first reported. Sweep delta share = Σ(buy − sell) ÷
  Σ volume over the sweep window, × sweep direction (+ = aggressive in the sweep's direction); split at the median of
  all primary fades (a feature median, no outcome), top vs bottom half. Up vs down sweeps. Entry hour (PT).
  Breakout-direction trades: burst breaks with no re-entry, entry c[b + 4] in the sweep direction, ±10 race from
  b + 4, reported vs 50 %.
- **C13 — levels.** CR-CP `load_levels` (CR-CM strong set, |net| ≥ day p95, within ±1·IM of the open). "Edge at a strong
  line" = a strong line within 2 pts of the swept edge (hi for up). "Sweep beyond a strong line" (A18) = a strong
  line strictly beyond the edge and reached by the sweep extreme (hi < L ≤ extreme; mirror). Tag order: beyond, else
  at edge, else "no strong line"; sessions outside CR-CM's universe "no levels". Put / call = sign of that line's net.
  Cells < 40 resolved fades are descriptive; put / call "not readable on train" if < 40.
- **C14 — Step 0 (no outcomes).** Cache checks; universe and swing; ranges; breaks per edge (up / down), burst vs no burst,
  failures vs breakouts, both-edges bars, overlap drops; fades (and sessions) for the primary and each C11 cell; split
  counts (C12/C13); A/A (C10). Gate: ≥ 40 fades on ≥ 30 sessions and |A/A mean| ≤ 1 pp → commit the run script, run once.
  Otherwise stop and report.
- **C15 — run script.** `scripts/cr_cs_run.py` asserts the primary schedule equals Step 0's (ranges, fades) and refuses
  to run if `results.json` exists.

## Step 0 findings (2026-10-26; `scripts/cr_cs_step0.py` @ `c2357a3` → `scripts/logs/cr_cs/step0.json`; no fade or continuation outcome computed)

- **Caches / universe (C1).** CR-CP's `m1_train.parquet` copied from the main checkout: 44,460 rows, 2026-01-06 14:30 →
  06-05 19:59 UTC (inside the C17 guard; equals CR-CP Step 0's 44,460). 101 sessions, IM_train 65.71 (asserted).
- **Ranges.** 470 on 100 sessions (asserted = CR-CP Step 0 v2 primary).
- **Breaks (primary).** 560 edge breaks after q on 423 ranges (up 283, down 277); no bar broke both edges. Break size
  median 2.75 pts (p95 5.5). Break bar − q: median 8 bars (p95 56).
  Sweep volume ÷ range median 1-min volume over all breaks: median **1.28** (p75 1.77, p95 3.6) → **103 bursts (18 %)**,
  457 without a burst.

  | | failure (close back in ≤ 5 min) | breakout |
  |---|---|---|
  | burst (≥ 2×) | **59** | 44 |
  | no burst | 204 | 253 |

  One fade dropped as an overlap duplicate (C8). **Fades: 58 on 41 sessions** (up sweeps → short 26, down sweeps →
  long 32). Sweep window 1 bar on 45, 2 on 11, 3 on 2; re-entry lag 0–4 bars (0: 17, 1: 20, 2: 9, 3: 9, 4: 3). Sweep
  extreme beyond the edge median 3.5 pts (p95 9.5); stop distance (extreme + 1 − entry) median **6.1 pts** (p25 4.3,
  p75 8.3, p95 17). Sweep delta share (signed to the sweep) median +8.1 % (p5 −0.5 %, p95 +26.7 %). Breakout-direction
  readout trades: 44.
- **Readout grid (fades / sessions):** size 1 → 54 / 40; size 4 → 34 / 27; volume 1.5× → 91 / 54; volume 3× → 19 / 13;
  re-entry 3 min → 42 / 32; re-entry 10 min → 79 / 53. Size 4 and volume 3× are under 40 before outcomes.
- **Splits (fades).** Up / down sweep 26 / 32; delta top / bottom half 29 / 29; hour 07 4, 08 7, 09 9, 10 14, 11 15,
  12 9; **levels: sweep beyond a strong line 2, edge at a strong line 2, no strong line 51, no levels 3 → the A12 / A18 /
  A8 level splits are not readable on train** (all call lines; put 0). EDGE / INTERIOR: 0 / 58.
- **A/A (C10).** 200 random halves on the 58 fade keys: mean **+0.39 pp**, median +0.17, SD 6.9, p5 −11.3, p95 +11.7;
  no key unmatched.
- **Gate (C14): passed** — 58 ≥ 40 fades, 41 ≥ 30 sessions, |A/A| 0.39 ≤ 1 pp. The run script is committed and run once.
  Note: 58 fades leaves little room; if > 18 are ties / unresolved the primary is UNDERPOWERED by the ≥ 40 rule.

## Result (single run, 2026-10-26) — `scripts/cr_cs_run.py` @ `bca648a` → `scripts/logs/cr_cs/results.json`

One launch, exit 0 (2.8 s on cached loads). Schedule asserts held (470 ranges, 58 fades on 41 sessions, 1 overlap drop).
Train only; R-a, R-b and the holdout were not read. 101 sessions; 10,000 session-cluster draws, seed 20260920.

### Primary — fade minus continuation, ±10 race (stop-free; a direction test, not Ryan's exit, B6)

| | resolved | win rate |
|---|---|---|
| fade | 51 (37 sessions; 7 unresolved, 0 ties) | **39.2 %** (−10.8 pp vs 50 % [−24.2, +2.2]) |
| continuation (same price and bar) | 51 | 60.8 % |
| **fade − continuation** | | **−21.6 pp [−48.4, +4.3] → INCONCLUSIVE** |

As C6 notes, the two sides share entries, so the difference is 2 × (fade − 50 %).

### Readouts (no verdicts, unadjusted)

- vs B7 matched random baseline (minute, side, EDGE; coverage 100 %): fade −9.9 [−23.0, +3.1]; continuation +9.4
  [−3.7, +22.4]. All fades are INTERIOR.
- ±20 race: fade 50 % (n 32), diff 0.0 [−38.5, +33.3].
- Move at +30 min (fade side): −1.3 pts [−3.9, +1.1]; at +60: 0.0 [−5.1, +4.6] (n 49).
- Max favourable before −10: fade median 7.1 pts (≥ 10: 34 %, ≥ 20: 17 %); continuation 11.6 (53 %, 24 %).
- **Stop at the sweep extreme + 1 vs +10 (mechanical, B6):** stopped first 37, target first 18, neither 3 → stopped
  first on 64 %; target share of resolved 33 % [20, 45]. Median stop distance 6.1 pts (Step 0).
- Breakout-direction trades (burst breakouts, entry at the 5-min close, sweep direction): 64.1 % (n 39, descriptive),
  +14.1 pp vs 50 % [−4.8, +32.1].
- Grid (fade − continuation, pp; resolved fades): size 1 −25.0 [−55.6, +5.3] (48); size 4 −18.8 [−53.3, +21.2] (32);
  volume 1.5× −14.3 [−35.2, +5.9] (84); volume 3× −11.1 [−60.0, +45.5] (18); re-entry 3 min −21.1 [−52.4, +9.1] (38);
  re-entry 10 min −21.1 [−43.3, +2.3] (71). **All six point estimates negative; no CI excludes 0.**
- Splits (all < 40 resolved, descriptive): up sweep → short 43.5 % (23), down sweep → long 35.7 % (28); sweep delta top
  half 41.7 % (24), bottom half 37.0 % (27); hour cells 4–14. Levels: no strong line 45 resolved, −15.6 [−45.0, +11.8];
  at / beyond a strong line 3 in all → **A12 / A18 / A8 not readable on train**.

### Reading

On train, when a range edge is swept on a volume burst and price closes straight back inside, fading it did **not** work:
the fade won 39 % of ±10 races and the continuation 61 %. The CI (−48 to +4) is wide, so the verdict is INCONCLUSIVE,
leaning against the fade. Every grid variant leans the same way; burst breakouts that held outside kept going (64 %,
descriptive). Not shortlisted; R-a not read.
