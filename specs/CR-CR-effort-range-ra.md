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
