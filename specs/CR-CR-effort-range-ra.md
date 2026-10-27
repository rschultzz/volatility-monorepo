# CR-CR — Confirm CR-CP's effort rule on R-a

Read-only, pre-registered, single hypothesis. Branch `feat/CR-CR-effort-range-ra` off `origin/main` `630ac16` (#97
merged), own worktree. Kickoff: Ryan, 2026-10-26 (chat). Frozen 2026-10-26 **before any R-a data was read for this
question**.

---

## Spec body (kickoff, verbatim)

> CR-CR — Confirm CR-CP's effort rule on R-a (read-only, pre-registered, single hypothesis).
>
> Branch first: git checkout -b feat/CR-CR-effort-range-ra off origin/main (after #97 merges).
> Own worktree. Read CLAUDE.md, field-conventions.md (incl. the 2024 ORATS quote lag, ES.c.0 roll
> weeks, es_flow_1m_databento = summed 1-s), standing-rulings.md, and the CR-CP spec
> (specs/CR-CP-range-edge-flow.md, C1–C18, A0, A1). Freeze with the conformance table; stop on any
> "deviates".
>
> Hypothesis (the CR-CP "contracts/pt pushes" readout, promoted to the single primary; train gave
> +14.6 pp [−0.8, +30.0] on 90 trades, one of 16 readouts, so a lead, not evidence):
> everything EXACTLY as CR-CP ran it after A0/A1 — approach (vol-scaled swing, 65.71 denominator,
> start confirmed, C1/C5), range (C6 with A1: closes box ≤ 12 pts, ≥ 15 min, close-based void and
> restart), fill (C9), race ±10 (C10), statistic (C11) — EXCEPT the call rule (D2) is replaced by
> C12's effort rule: over r0..t, cpp_up = Σ volume ÷ Σ |c − o| over up bars, cpp_down the same over
> down bars; SHORT if cpp_up ÷ cpp_down ≥ 1.5 and progress ≤ 2 pts (C7); LONG mirror (≤ 1 ÷ 1.5,
> declined ≤ 2); else no trade; decision bar per C8. Reuse CR-CP's code with only the call rule and
> date range switched; no other change, no re-tuning.
> Data: R-a only (2024-04-03 → 2024-12-31), Databento flow + bars; exclude roll-week sessions as
> CR-CO did if the ES.c.0 contract splices intraday (report counts). Train, R-b, holdout NOT read.
> Primary: chosen − opposite edge race win rate (C11), session-cluster bootstrap, 10 000 draws, seed
> 20260920. PASS if CI lower bound > 0; REFUTED if CI upper bound < 0; else INCONCLUSIVE; ≥ 40
> resolved chosen-side trades, else underpowered. One hypothesis, no Holm.
> Readouts (not gated): each side vs 50 %; true pairs; C13 random baseline; ratio 1.3 / 2.0;
> swing fixed 12; level split (C14); call side; time of day; G1-down skew flag on the approach.
> Step 0 (no outcomes): universe, ranges, calls (short/long/no trade), fills, A/A (C13). If ≥ 40
> chosen-side fills on ≥ 30 sessions and |A/A mean| ≤ 1 pp, commit the run script and run the
> primary once without stopping; otherwise stop and report.
> Results + plain-English summary to the CR note, setup-registry, session note; push, PR, don't
> merge; copy logs before removing the worktree. Plain-English summary for Ryan: in 2024, did
> "heavy effort with no progress" inside a range pick the right edge?

Everything not named here is CR-CP's definition as merged (`specs/CR-CP-range-edge-flow.md`: D0–D5, C1–C18,
Approvals, A0, A1 incl. A1.a–A1.e), unchanged.

## Choices the kickoff doesn't fix (Code, 2026-10-26, before any R-a data)

- **E1 — period switch.** `scripts/cr_cp_lib.py` gains one switch, `CR_CP_PERIOD` (default `train`, as merged; `ra` for
  this CR), the pattern CR-CN / CR-CO used for `cr_cl_lib` (`CR_CL_PERIOD`). Under `ra`: sessions 2024-04-03 →
  2024-12-31; **every DB read asserted ≥ 2024-04-01 00:00 UTC and < 2025-01-01 00:00 UTC** (C17 with R-a bounds);
  logs and caches in `scripts/logs/cr_cr/`. Under the default nothing changes: a parity check (subprocess, no data,
  as CR-CO O2) asserts the default resolves train dates, train guards and `logs/cr_cp`.
- **E2 — swing denominator.** IM_train = **65.71410292087306** (CR-CO A1.1 / CR-CP A0, from
  `scripts/logs/cr_co/results.json` `im_train` = CR-CP `step0.json` `median_im`) is a constant under `ra`, so no
  train row is read. swing_d = 12 × IM_d ÷ IM_train, IM_d = canonical `implied_move_1d` on the R-a session.
- **E3 — roll weeks.** `es_flow_1m_databento` holds **outright contracts only, never `ES.c.0`** (field-conventions);
  C3 builds each session from its one most-RTH-volume symbol, so no session series splices contracts intraday and the
  kickoff's exclusion condition does not arise. Roll-window sessions (Monday of the week before expiry week → expiry
  Friday, `cr_cm_lib.roll_week`) are **kept, as in CR-CP**, and Step 0 reports their count, the chosen symbol and its
  share of RTH volume. If Step 0 finds any session whose rows are `ES.c.0` or that has a missing-minute run on the
  chosen symbol (C18), it is excluded by the existing funnel and counted.
- **E4 — C3 on R-a.** The `es_flow_1m_databento` = summed 1-s identity was measured on train only. Step 0 re-measures
  it on R-a RTH (same query) before use; field-conventions row extended.
- **E5 — primary object.** `PRIMARY_CR = replace(cr_cp_lib.PRIMARY, rule="cpp")` — CR-CP's `cpp_pushes` variant
  exactly (ratio 1.5, progress 2, all A1 settings). The C12 code path (`side_call`, `rule == "cpp"`) is not edited.
- **E6 — readouts.** As the kickoff: each side vs 50 %; true pairs; C13 random baseline (chosen and opposite);
  `ratio 1.3`, `ratio 2.0` (cpp_ratio, one swap each); `swing fixed 12`; C14 level split (strength, position vs the
  nearest strong line, put / call); call side; time of day (decision hour, PT); G1-down flag on the approach (CR-CP
  `tag_skew` as merged: beyond noise, ratio < 0, down approach, leg end ≤ 11:00 PT, on the approach leg). **Added to
  honour A11** (not named in the kickoff): ±20 race and the max-favourable-before-−10 distribution. No other CR-CP
  readout is run. Cells under 40 resolved chosen trades are labelled descriptive; put / call "not readable" if < 40.
- **E7 — skew loader on R-a.** `tag_skew` imports `cr_cl_lib` with `CR_CL_PERIOD=ra` (CR-CN's R-a loader: 0DTE PM skew
  paired with ES at the snapshot's **`quote_date`**, the 2024 ~129 s lag handled as CR-CJ A1). Shipped CR-CK band
  unchanged (A14 parked; readout only).
- **E8 — levels on R-a.** `cr_cm_lib.load("ra")` (its own R-a guard: ≥ 2024-03-01, < 2025-01-01; Corrected placement is
  defined from 2024-04-03, field-conventions). Sessions outside CR-CM's R-a universe are "no levels".
- **E9 — Step 0 gate** (kickoff): ≥ 40 chosen-side fills on ≥ 30 sessions and |A/A mean| ≤ 1 pp → commit the run script,
  run once, no stop. Otherwise stop and report. A/A as CR-CP C13 (200 splits, seed 20261025) on the primary's
  chosen-fill keys.
- **E10 — run script.** `scripts/cr_cr_run.py`: CR-CP's run helpers (`outcomes`, `compare`, `verdict`, …) copied
  unchanged; asserts the schedule equals Step 0's (ranges, chosen fills) and refuses to run if `results.json` exists.

## Standing rulings conformance (Dash/research/standing-rulings.md, status current, read 2026-10-26)

| # | Ruling (short) | Mark | How |
|---|---|---|---|
| A1 | levels slow price, not a trade | honoured | level-agnostic primary (CR-CP); level position a readout |
| A2 | pause ≥ a few min, relative, up to ~30 min | honoured | CR-CP A1 range: ≥ 15 min, close box ≤ 12 |
| A3 | no auto entry at first touch; entry after the range forms | honoured | C9 entry after the range qualifies and the effort rule calls |
| A4 | stops beyond the zone | n/a | stop-free direction test (A15) |
| A5 | flow decides fade vs break; never "always fade" | honoured | effort ratio picks the side per range; both sides possible |
| A6 | skew divergence direction | n/a (parked) | G1-down flag readout only |
| A7 | approach swing vol-scaled, 2026-train median IM; fixed 12 readout | honoured | E2: 65.71 denominator; fixed 12 readout |
| A8 | put / call levels separately | honoured (readout) | C14 put / call split, never pooled into the primary |
| A9 | two close levels → wait for the second | n/a | level-agnostic; the A9 tag is not a readout of this single-hypothesis CR |
| A10 | earlier same-day S/R | n/a | not a readout here |
| A11 | ±10 primary, ±20 readout, favourable-excursion distribution | honoured | ±10 primary; ±20 and MFE readouts added (E6) |
| A12 | levels = chart's yellow / darkest | honoured (readout) | CR-CM `chart_levels`, strong = p95 |
| A13 | sizing | n/a | |
| A14 | skew vs expected, beyond noise | n/a (parked) | shipped band, readout only |
| A15 | direction first, stop-free | honoured | ±10 race |
| A16 | skew trend over the leg | n/a (parked) | CR-CK leg read, readout only |
| A17 | flow in the range decides; flow alone first | honoured | primary is flow alone (effort); skew a readout |
| A18 | range short of / beyond the level, split | honoured (readout) | C14 position split |
| A19 | in-range confirmation, both sides; heavy contracts/pt on the pushes | honoured | this primary is A19's contracts/pt component, short and long mirror |
| A20 | trade the range edges | honoured | C9 edge fills |
| A21 | second touch a different setup | n/a | |
| B1 | three verdicts | honoured | |
| B2 | many parameters fine if disclosed; plateau not peak | honoured, with a note | the primary is the best of 16 train readouts (a peak); it is **tested on fresh data**, not adopted from train, and the train origin is disclosed. Ratio 1.3 / 2.0 readouts show whether it is a plateau |
| B3 | one setup at a time | honoured | single hypothesis |
| B4 | R-a for confirmation; R-b, holdout never read; train not re-read | honoured | E1 guards (< 2025-01-01); E2 constant, no train read |
| B5 | Databento reference for flow and bars | honoured | `es_flow_1m_databento` (outright contracts); E4 measurement |
| B6 | mechanical exits labelled | honoured | the ±10 race is a direction test, not Ryan's exit |
| B7 | placebos matched on EDGE vs INTERIOR | honoured | C13 random baseline, EDGE-matched |
| B8 | Code labels ≠ Ryan's | n/a | |
| C1, C2 | UI | n/a | |

**No "deviates".** Proceeding to Step 0 per the kickoff.

## Step 0 findings (2026-10-26; `scripts/cr_cr_step0.py` @ `5a63f6c` → `scripts/logs/cr_cr/step0.json`; no range outcome computed)

- **Parity (E1):** under the default, `cr_cp_lib` resolves `train`, 2026-01-06 → 06-05, guard [2026-01-01, 2026-06-05
  21:00) UTC, `logs/cr_cp`. Unchanged.
- **C3 on R-a (E4):** `es_flow_1m_databento` = the `es_flow_1s_databento` minute aggregate **exactly**: 83,345 / 83,345
  RTH symbol-minutes equal on o, h, l, c, volume, buy, sell, n_seconds; none in one table only; a 1-min shift matches
  8–11 % of prices (rejected). Symbols: ESM4, ESU4, ESZ4, ESH5 (outrights only, **no `ES.c.0`**).
- **Universe:** 189 trading days → early close 3 (2024-07-03, 11-29, 12-24) → no canonical IM 6 → **180 usable**. No
  missing-minute run > 5; C18 fill never needed.
- **Roll weeks (E3):** 28 roll-window sessions kept; 25 have both contracts in RTH; the session uses its
  most-volume contract, whose share of RTH volume is ≥ 0.60 on every one (lowest 2024-09-16 0.60, 2024-12-16 0.64).
  No intraday splice, so nothing excluded.
- **Swing (E2):** IM_train 65.714 (constant). R-a IM median **41.7 pts** (p5 20.7, p95 87.4), so swing_d median **7.6
  pts** (p5 3.8, p25 6.0, p75 10.8, p95 16.0); 44 sessions under 6 pts. 2024 moved less than 2026 (train swing_d
  median 12.3); per A7 the swing scales down with it. Legs per session median 15.

| primary (effort rule) | count |
|---|---|
| ranges | **990** on 180 sessions (median 5 per session); close-box width median 4.0 pts (p5 2.0, p95 9.25) |
| void before decision | 620 (63 %); no call by r0 + 60: 51 |
| calls | **319**: short 150, long 169 (up → short 79, up → long 88, down → short 71, down → long 81) |
| decision bar − qualifying bar | median 0 (p75 2, p95 10) |
| cpp_up ÷ cpp_down at the call | p25 0.61, p50 0.66, p75 1.60 (thresholds 1 ÷ 1.5 and 1.5) |
| **chosen-side fills** | **218 (68 %) on 125 sessions**; box broke 97, 13:00 4 |
| opposite-side fills | 281 (88 %); both filled 184 |
| chosen fills EDGE / INTERIOR | 22 / 196 |

Readouts (ranges / calls S-L / chosen fills / sessions): ratio 1.3 990 / 290-295 / 422 / 167; **ratio 2.0 990 / 24-33 /
39 / 35 (under 40: descriptive)**; swing fixed 12 736 / 100-119 / 140 / 94.

Splits (chosen fills): levels (CR-CM R-a universe 147 sessions) strong within 10 pts 59, weak 115, none 8, no levels 36;
position short of 24 / straddling 11 / beyond 24; **put 15 / call 44** (put not readable); call side long 113 / short
105; hour 06: 6, 07: 26, 08: 30, 09: 34, 10: 38, 11: 45, 12: 39; **G1-down flag 3** (skew beyond noise 10): descriptive.

**A/A (C13):** 200 splits, mean **+0.30 pp** (median +0.36, SD 3.4, p5 −5.2, p95 +5.7); 0.007 % keys unmatched;
119,040 baseline entries, EDGE share 10 %.

**E9 gate: passed** (218 ≥ 40 fills, 125 ≥ 30 sessions, |A/A| 0.30 ≤ 1 pp). Per the kickoff the run script is committed
and the primary is run once without stopping.

## Result (single run, 2026-10-26) — `scripts/cr_cr_run.py` @ `21dd906` → `scripts/logs/cr_cr/results.json`

One launch, exit 0 (6.8 s on Step 0's caches). The schedule asserts held (990 ranges, 218 chosen fills, as Step 0). R-a
only: 180 sessions, 10,000 session-cluster draws, seed 20260920. Train, R-b and the holdout were not read.

### Primary — effort-chosen edge minus the opposite edge, ±10 race (stop-free; a direction test, not Ryan's exit, B6)

| | resolved | win rate | vs 50 % |
|---|---|---|---|
| chosen side | 173 (111 sessions; 44 unresolved, 1 tie) | **52.0 %** | +2.0 [−5.6, +9.8] |
| opposite side | 231 (49 unresolved, 1 tie) | **50.2 %** | +0.2 [−6.3, +6.8] |
| **chosen − opposite** | | **+1.8 pp [−10.6, +14.6]** | **INCONCLUSIVE** |

Train gave +14.6 [−0.8, +30.0]; R-a's point estimate is an eighth of that and its CI is centred near 0.

### Readouts (no verdicts, unadjusted)

- True pairs (both sides filled and resolved, n 143): −4.2 [−19.6, +11.1].
- vs the C13 matched random baseline (coverage 100 %): chosen +2.4 [−5.4, +10.1]; opposite +0.1 [−6.4, +6.6].
- ±20 race (A11): chosen 55.2 % (n 87) vs opposite 42.7 %, +12.4 [−6.6, +31.4].
- Max favourable before −10 (A11): chosen median 7.75 pts (≥ 10: 41 %, ≥ 20: 19 %); opposite 6.5 (41 %, 17 %).
- Grid (chosen − opposite, pp): ratio 1.3 +4.4 [−4.6, +13.9] (332); ratio 2.0 +4.5 [−29.5, +37.6] (31, descriptive);
  swing fixed 12 +7.0 [−5.7, +19.7] (123). All positive and all cover 0: a flat, small lean, no plateau above 0.
- Splits:
  - call side: short +10.1 [−9.5, +28.9] (85); long −6.3 [−25.3, +13.3] (88).
  - levels: strong line within 10 pts +12.5 [−13.6, +39.0] (47); weak −0.4 [−16.4, +15.3] (95); none / no levels
    descriptive. Position vs the strong line: every cell < 40 (descriptive). **Put / call not readable on R-a** (call
    33, put 14).
  - decision hour: every hour < 40 (descriptive).
  - G1-down flag on the approach: 3 trades (descriptive); the rest +1.8 (170).

### Reading

On R-a, ranges where the up pushes (or down pushes) took far more contracts per point without price getting
anywhere did **not** pick the better edge in a way the data can tell apart from chance: chosen edge 52 %, opposite 50 %,
+1.8 pp with a CI from −11 to +15. INCONCLUSIVE. The train lead (+14.6) does not repeat at its size; every readout
leans slightly positive but none excludes 0. Not shortlisted.
