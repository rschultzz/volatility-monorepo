# CR-CV — Do today's setups work at scalping scale, after costs?

Read-only, pre-registered, train only, **unattended overnight run**. Branch `feat/CR-CV-scalping-scale` off
`origin/main` `d6ae7b5` (PR #102 merge, CR-CU; rebased from `cd2fe36` before any data — CR-CU added only its own files), own worktree (`.claude/worktrees/cr-cv`). No table writes, no cron, no
deploy, no display change. PR, don't merge. Kickoff: Ryan, 2026-10-26 (chat). Frozen 2026-10-26 **before any CR-CV
data was read** (the only reads so far: the specs, the libraries and the stored event files of CR-CL / CP / CS / CT).

---

## Spec body (kickoff, verbatim)

> CR-CV — Do today's setups work at scalping scale, after costs? (read-only, pre-registered, train
> only, UNATTENDED overnight run.)
>
> Run start to finish without asking. Ryan is asleep. Every choice below is pre-approved; where the
> brief is silent, choose the most conservative option, record it as C# in the spec, and continue.
> STOP only if: a standing-rulings "deviates" not pre-approved here appears; any read would touch
> R-a (2024), R-b (2025), or the holdout (> 2026-06-05); a data check fails; or tests fail. If you
> stop, write the reason to the vault CR note and session note and end cleanly.
>
> Branch first: git checkout -b feat/CR-CV-scalping-scale off origin/main. Own worktree.
> Read CLAUDE.md, field-conventions.md, standing-rulings.md, and the specs for CR-CL, CR-CP, CR-CS,
> CR-CT. Freeze specs/CR-CV-scalping-scale.md with the conformance table.
> Pre-approved deviations (record, don't stop): A11 (small races are the question here; ±10 kept as
> reference); A3/R2 (entries exactly as each original CR used them); B6 (mechanical, not Ryan's trade).
> Read-only DB connection, default_transaction_read_only = on. No DB writes.
>
> Question: none of today's setups showed an edge at ±10 pts. Do any work at ±2 or ±4 pts after
> realistic costs?
> Setups — re-use each CR's exact event set, definitions and code on 2026 train (2026-01-06 →
> 2026-06-05); change only the race size and fill model:
>   S1 CR-CL G1-down (long after the signal leg) — and its opposite side;
>   S2 CR-CP flow-chosen range edge — and the opposite edge;
>   S3 CR-CS failed-sweep fade — and continuation;
>   S4 CR-CT first touch: FAST → break, SLOW → fade (pooled as one "hypothesis side") — and its opposite.
> Races: ±2 and ±4 pts (primary), ±10 (reference), resolved on Databento 1-SECOND bars from the
> second after entry; both levels inside the same second = tie (excluded, counted); unresolved by
> 13:00 PT excluded, counted.
> Fill model (primary, conservative): limit entries (S2's edge limit, S4's level price) fill only if
> price trades at least 1 tick THROUGH the limit within the original fill window; market-style
> entries (bar-close entries in S1/S3) fill at the next second's open + 0.25 pt adverse. Readout: the
> original touch-fill model.
> Primary statistic per cell (4 setups × 2 sides × 2 sizes = 16 cells): after-cost expectancy per
> trade = (2·p − 1)·T − 0.50 pts (round-trip cost), session-cluster bootstrap 10 000 draws, seed
> 20260920, 95 % CI. Holm across the 16 cells. PASS if Holm-adjusted lower bound > 0; REFUTED if the
> CI upper bound < 0; else INCONCLUSIVE. ≥ 40 resolved per cell, else underpowered.
> Readouts: cost 0.25 pt (limit-in, market-out) and 0.75 pt; win rate vs break-even (56 % at ±4,
> 62.5 % at ±2 for 0.5 pt); touch-fill model; ±10 reference reproducing each CR's original win rates
> (check: must match within 1 pp, else STOP — the re-use is wrong); share of limit entries that fill
> under trade-through.
> Steps: Step 0 (counts, fill rates, ±10 reproduction check, no small-race outcomes) → if each primary
> cell has ≥ 40 fills or is reported underpowered, commit the run script → run once → results.
> Deliver: results + plain-English summary to the vault CR note ("CRs/CR-CV — Scalping Scale.md"),
> setup-registry row, session note; push, open a PR, don't merge. Copy logs to the main checkout's
> scripts/logs/cr_cv/ and verify; keep the worktree. Morning summary for Ryan, first line: "Does
> anything work at ±2 or ±4 after costs: yes / no", then the 16-cell table.

---

## Choices the kickoff doesn't fix (Code, 2026-10-26, before any CR-CV data was read)

Where the kickoff is silent the most conservative option is taken (kickoff rule).

- **C1 — event sets, re-used through each CR's own code.** `scripts/cr_cv_lib.py` imports `cr_cl_lib`, `cr_cp_lib`,
  `cr_cs_lib`, `cr_ct_lib` unchanged and rebuilds each event set on that CR's own universe and caches (the caches are
  copied from the main checkout's `scripts/logs/cr_cj|cr_cp|cr_ct/`, as CR-CS C1 did; same guarded queries, checked by
  hash). Step 0 asserts the rebuilt set equals the stored event file of the original run (copied to
  `scripts/logs/cr_cv/ref/`) on every key column.
  - **S1** = CR-CL A2.1 primary cell: swing-12 primary legs (end ≤ 11:00 PT), G1, leg direction down; hypothesis
    side long (h = +1), opposite short. **All 49 legs** are the event set. CR-CL's 66.0 % is over the 47 legs its
    *matched baseline* covered; coverage is a property of that estimator, not of the setup, and CR-CV has no matched
    baseline. Both 32 / 49 and the covered 31 / 47 are reproduced (C8). CR-CL's universe is CR-CJ `y26`
    (2026-01-02 → 06-05); re-using the exact event set keeps any G1-down leg on 01-02 / 01-05 (reported; both days
    are 2026 train, not R-a / R-b / holdout).
  - **S2** = CR-CP A1 primary (`cr_cp_lib.schedule(D, PRIMARY)`): the 235 ranges with a D2 call. Hypothesis side =
    the chosen edge, opposite = the other edge after the same decision bar (CR-CP C11).
  - **S3** = CR-CS primary (`cr_cs_lib.events` + `trades`): the 58 fades. Hypothesis = fade (h), opposite =
    continuation (−h), same entry (CR-CS C6).
  - **S4** = CR-CT primary first touches (`cr_ct_lib.events`, touch 1, not duplicate, with a 1-s touch second),
    classes from the frozen 10-min cuts in CR-CT's `step0.json` (C9, asserted): FAST 85, SLOW 75. Hypothesis side =
    break (h = b) for FAST, fade (h = −b) for SLOW, pooled; opposite = fade for FAST, break for SLOW, pooled. NEITHER /
    UNDEFINED touches are not used.
- **C2 — 1-second data.** `es_flow_1s_databento` (Databento, B5) `ts_utc, open, high, low, close`, for each setup's
  front symbol per session (the symbol each CR's own minute bars use), 06:30:00 → 12:59:59 PT. Every read asserted
  ≥ 2026-01-01 00:00 UTC and < 2026-06-05 21:00 UTC; connection via `cr_cp_lib.conn()` (`dash_backfill_writer`
  asserted, `SET default_transaction_read_only = on`, asserted `on`), and the CR-CJ loader's connection is replaced
  by the same read-only connection before any import can query. Seconds without a trade are absent and are not gaps.
  **Measured in Step 0 before use** (`open` is not in field-conventions for this table): (a) o, c ∈ [l, h] on every
  row; (b) the 1-s minute aggregate (first open, max high, min low, last close) equals each setup's own minute bars on
  every RTH minute of every session it uses. Any failure is a data-check STOP.
- **C3 — primary fill model (conservative), per entry type.**
  - **Market-style** (S1 at the close of the confirmation bar `ci`; S3 at the close of the re-entry bar `r`): the
    entry second is the first 1-s row at or after the start of the next minute (the next trade after the bar close);
    fill = that second's **open + 0.25 pt adverse** (long +0.25, short −0.25). No trade before 13:00 → no fill
    (counted).
  - **S2 edge limit** (long at box lo + 1, short at box hi − 1; CR-CP D4 / C9 / A1.b): CR-CP's `fill` loop unchanged
    — minutes t + 1 … 12:59, the void (a close > 2 pts beyond the other edge) checked first in each minute exactly as
    `fill` does — except that a minute fills only if one of its seconds trades **≥ 1 tick through** the limit
    (long: 1-s low ≤ limit − 0.25; short: 1-s high ≥ limit + 0.25). Fill at the limit, at the first such second. A
    minute that touches but does not trade through does not end the search (the original window continues).
  - **S4 level** (CR-CT C6 window = the touch minute j): the first second of minute j that trades **≥ 1 tick
    through L in the break direction** (b up: 1-s high ≥ L + 0.25; b down: low ≤ L − 0.25). None in that minute →
    no fill (counted). The **fade side** (h = −b) is a resting limit at L: fill L. The **break side** (h = b) buys
    above (sells below) the market, which cannot rest as a limit; it is modelled as a **stop filled one tick
    through: L + 0.25·b**. The literal "fill at the limit" would hand the break side a free tick; the stop fill is
    the conservative reading.
  - Both sides of a setup are filled by their own model (S2 opposite edge: its own trade-through search; S1 / S3
    opposite: the same market fill with its own adverse tick).
- **C4 — race (every model and size).** On the session's front-symbol 1-s bars from the entry second + 1 to
  12:59:59 PT: win = h·(1-s high / low − fill) ≥ +T first; loss = ≤ −T first; **both in one second = tie** (excluded,
  counted); neither = unresolved (excluded, counted). T ∈ {2, 4} primary, {10} reference.
  **Entry second (primary model, conservative):** if the entry second's own adverse extreme beyond the fill is ≥ T
  the trade is a loss; its favourable side is not credited (CR-CP C10's fill-bar rule, applied at 1 s). The kickoff's
  "from the second after entry" is kept for wins.
- **C5 — touch-fill model (readout) = each CR's original entry, on 1-s bars.** S1 / S3: fill at the bar close, race
  from the first second of the next minute (the entry is at the bar's end; no entry-second rule). S2: fill at the
  limit at the first second of the original fill minute that touches it (1-s low ≤ limit / high ≥ limit), CR-CP's
  fill-bar adverse rule applied to that second. S4: exactly CR-CT (L at the touch second `k1s`, race from `k1s + 1`, no
  entry-second rule).
- **C6 — cells and statistic.** 16 cells = setup (S1–S4) × side (hypothesis / opposite) × T (2, 4). Per cell
  p = W ÷ (W + L) (session-summed counts); **E = (2p − 1)·T − c**, c = 0.50 pt primary. Session-cluster bootstrap,
  10 000 draws, seed 20260920: one weight matrix per setup over **that CR's own session universe** (S1: CR-CL's
  `y26` sessions; S2 / S3 / S4: the 101-session CR-CP universe), drawn once and shared by every cell, readout and
  cost of that setup. Percentile 95 % CI.
- **C7 — Holm and verdicts.** Two-sided bootstrap p = min(1, 2·min(share E* ≤ 0, share E* ≥ 0)). Holm over **all 16
  cells** (underpowered cells stay in the family: m = 16 always, the conservative count): ranked by p, adjusted
  p_(i) = max_{k ≤ i} min(1, (17 − k)·p_(k)); the cell at rank i gets the percentile CI at level 1 − 0.05 ÷ (17 − i).
  **UNDERPOWERED** if resolved < 40. Else **PASS** if Holm-adjusted p < 0.05 **and** the Holm-level lower bound > 0;
  **REFUTED** if the **95 %** CI upper bound < 0 (the kickoff's "the CI"; the Holm-level upper bound is reported
  alongside); else **INCONCLUSIVE**. "Does anything work: yes" only if at least one cell is PASS.
- **C8 — ±10 reproduction check (Step 0; STOP on failure).** (a) Each CR's own race function on its own minute bars
  (`cr_cl_lib.race`, `cr_cp_lib.race_from`, `cr_ct_lib.race1s`) reproduces the stored ±10 outcome of every event of
  the original run, both sides (S1's opposite side is h = −1 on the same legs). (b) The 1-s touch-fill model (C5) at
  ±10 gives each original reported win rate within 1 pp: S1 long 66.0 % (covered; raw 65.3 %), S2 chosen 46.4 % /
  opposite 55.2 %, S3 fade 39.2 % / continuation 60.8 %, S4 FAST break 56.5 % / SLOW break 54.3 % (and the pooled
  hypothesis / opposite rates derived from them). Any miss > 1 pp → STOP ("the re-use is wrong").
- **C9 — Step 0 content (no ±2 / ±4 outcome, no primary-fill outcome of any size).** Event counts and sessions per
  setup; 1-s data checks (C2); fills per cell under the primary model (market: next-trade found; limit: trade-through
  share vs touch share, no-fill reasons); the C8 reproduction. Gate: every primary cell has ≥ 40 fills, or is listed
  as UNDERPOWERED-by-fills (it then runs and is reported underpowered, per the kickoff); data checks and C8 pass.
- **C10 — readouts (no verdicts, unadjusted 95 % CIs).** Costs 0.25 and 0.75 pt (E and its CI); win rate vs
  break-even p_BE = (1 + c ÷ T) ÷ 2 (56.25 % at ±4, 62.5 % at ±2 for c = 0.5); touch-fill model at ±2 / ±4 (E at
  0.5 pt); ±10 under both fill models; trade-through fill share of every limit cell; tie and unresolved counts per
  cell; S4 split FAST-break / SLOW-fade within the hypothesis side (descriptive).
- **C11 — one run.** `scripts/cr_cv_run.py` is committed before it runs, asserts Step 0's event counts and fill
  counts, and refuses to run if `scripts/logs/cr_cv/results.json` exists. A crash before any outcome is printed or
  saved allows a fix-and-rerun (CR-BY precedent), logged.
- **C12 — inherited deviations.** S1 inherits CR-CL's approved A17 deviation (skew alone as the signal) and S4
  inherits CR-CT's approved A7 deviation (fixed 10-min approach window). The kickoff fixes "each CR's exact event set,
  definitions and code", so they are carried on the terms Ryan approved them: S1 remains **a measurement only —
  nothing is shortlisted from S1** (CR-CL Approvals 1); S4's fixed window as in CR-CT. Recorded, not a new deviation.
- **C13 — tests.** `scripts/test_cr_cv_lib.py` (pytest, synthetic 1-s arrays, no DB): race win / loss / tie /
  unresolved, entry-second rule, market fill slippage sign, trade-through vs touch fills, stop-style break fill, E and
  break-even. Must pass before Step 0 runs; a failure is a STOP.
- **C14 — outputs.** `scripts/logs/cr_cv/` (git-ignored): `step0.json`, `step0_stdout.log`, `results.json`,
  `run_stdout.log`, `trades.parquet`, `s1_bars.parquet` (1-s cache), `ref/`. Copied to the main checkout's
  `scripts/logs/cr_cv/` and verified by hash at the end; the worktree is kept.

## Standing rulings conformance (Dash/research/standing-rulings.md, status current, read 2026-10-26)

| # | Ruling (short) | Mark | How |
|---|---|---|---|
| A1 | levels slow price, not a trade by themselves | honoured (inherited) | S2 / S3 level-agnostic; S4 conditions the level on the approach (CR-CT); S1 no levels |
| A2 | a range ≥ minutes, relative | honoured (inherited) | S2 / S3 use CR-CP's ≥ 15-min range unchanged; S1 / S4 no range |
| A3 | no automatic entry at the first touch | **deviates — pre-approved (kickoff: "A3/R2, entries exactly as each original CR used them")** | S4 enters at the level at the touch (R2 variant, labelled, as CR-CT). S1 (leg confirmation close), S2 (edge after the range and the flow call), S3 (close back inside after a sweep) are not first-touch entries |
| A4 | stops beyond the zone | n/a — labelled per B6 (pre-approved) | a ±2 / ±4 race is a bracket with a 2 / 4-pt stop: a mechanical cousin, not Ryan's stop |
| A5 | flow / skew decide fade vs break | honoured (inherited) | each setup's own rule picks the side; both sides are scored |
| A6 | skew divergence direction | honoured (inherited, S1) | S1 is CR-CL's G1-down (A6 in ratio units); parked block used only as CR-CL defined it |
| A7 | approach = vol-scaled swing | honoured (S2 / S3); **inherited approved deviation (S4, CR-CT)** | C12 |
| A8 | put / call separately | n/a | no level split is read (thin on train in every source CR) |
| A9 | two close levels | n/a | |
| A10 | earlier same-day S/R | honoured (inherited, S4) | RUN levels in CR-CT's primary |
| A11 | ±10 primary, ±20 readout | **deviates — pre-approved (kickoff)** | ±2 / ±4 primary; ±10 reference |
| A12 | levels = chart's strongest colours | n/a | no level readout here |
| A13 | sizing | n/a | |
| A14 | skew vs expected, stickiness ratio | honoured (inherited, S1) | CR-CK / CR-CL read unchanged |
| A15 | direction first, stop-free | honoured | symmetric races; no exits tuned (the race size is the question) |
| A16 | skew as a trend | honoured (inherited, S1) | |
| A17 | flow first, skew optional | **inherited approved deviation (S1, CR-CL: measurement only, nothing shortlisted)**; honoured S2–S4 | C12 |
| A18 | range position split | n/a | not read |
| A19 | in-range confirmation | honoured (inherited, S2) | CR-CP D2 |
| A20 | trade the range edges | honoured (inherited, S2 / S3) | |
| A21 | second touch separate | honoured | S4 first touches only |
| B1 | three verdicts | honoured | C7 |
| B2 | many parameters disclosed | honoured | 16 cells Holm-adjusted; all readouts reported |
| B3 | one setup at a time | honoured | no setup is optimised; four fixed setups re-scored side by side |
| B4 | train only | honoured | every read guarded to [2026-01-01, 2026-06-05 21:00 UTC); R-a, R-b, holdout not read |
| B5 | Databento reference | honoured | 1-s Databento bars; minute bars as each CR used (Databento) |
| B6 | mechanical exits labelled | **pre-approved (kickoff)** | the ±2 / ±4 / ±10 races are mechanical brackets, not Ryan's trade; labelled |
| B7 | placebos EDGE-matched | n/a | no placebo; the comparison is each setup's opposite side |
| B8 | Code labels ≠ Ryan's | n/a | |
| C1, C2 | UI | n/a | |

Revoked / superseded: **R1 not used** (no touch-bar-close entry: S1 and S3 enter at the close of a confirmation /
re-entry bar after the setup formed, not of a level-touch bar); R2 used as a labelled variant (S4, pre-approved);
R3–R7 not used.

**Deviations:** A3, A11, B6 pre-approved in the kickoff; A7 (S4) and A17 (S1) inherited from CR-CT / CR-CL on the
terms Ryan approved them (C12). No other "deviates".

---

## Step 0 findings (2026-10-26; `scripts/cr_cv_step0.py` @ `d70673b` + the comparison fix below → `scripts/logs/cr_cv/step0.json`; no ±2 / ±4 outcome and no primary-fill outcome computed)

Read-only connection asserted (`default_transaction_read_only` = on, role `dash_backfill_writer`). Every 1-s read was
inside [2026-01-01, 2026-06-05 21:00 UTC). R-a, R-b and the holdout were not read. Unit tests (C13): 8 / 8 pass
(`apps/web/.venv/bin/python scripts/test_cr_cv_lib.py`; that interpreter has no pytest, so the file carries a plain
runner).

**Run note.** Run 1 failed one check, "S4 event set = CR-CT stored first touches". Cause: the stored parquet holds the
session date as a string, the rebuilt set as a `date`. All seven key columns (d, L, j, b, k1s, cls10, fam) were equal
as values. The comparison now casts `d` to str. Run 2 (reported) passed every check, and its other numbers equal run 1's.
Run 1 log kept as `step0_run1_stdout.log`.

### Event sets (C1): each CR's own code, equal to its stored run

| Setup | Events | Sessions | Rebuilt = stored | Own race reproduces stored ±10 outcome |
|---|---|---|---|---|
| S1 CR-CL G1-down | 49 (47 covered) | 36 of 107 | yes (8 key columns) | yes: 32 W / 17 L |
| S2 CR-CP calls | 235 (short 108, long 127) | 88 of 101 | yes (decision bar, side, box, both fills) | yes, chosen and opposite |
| S3 CR-CS fades | 58 | 41 of 101 | yes | yes, fade and continuation |
| S4 CR-CT FAST / SLOW | 160 (FAST 85, SLOW 75) | 72 of 101 | yes | yes: 86 break W / 69 L / 5 unresolved |

One S1 leg falls on 2026-01-02 / 01-05: CR-CL's own `y26` sessions, 2026 train.

### 1-s data (C2), measured before use

- 2,291,024 rows on 104 (session, symbol) pairs. o and c lie within [l, h] on every row.
- **The 1-s minute aggregate equals each setup's own minute bars on all 53,430 minutes** (S1's CR-CJ bars on its 36
  sessions; CR-CP's bars on all 101 × 390), on o / h / l / c. No minute exists in one source only.
- h / l / c equal CR-CT's own 1-s cache on all 101 sessions.

### Fills (C3, C9)

| Cell | Primary fills | Touch-fill fills | Trade-through share of touch fills | No primary fill |
|---|---|---|---|---|
| S1 long / short | 49 / 49 | 49 / 49 | (market) | — |
| S2 chosen | 175 (76 sessions) | 178 | **98.3 %** | box broke 58, 13:00 2 |
| S2 opposite | 170 (81 sessions) | 171 | **99.4 %** | box broke 64, 13:00 1 |
| S3 fade / continuation | 58 / 58 | 58 / 58 | (market) | — |
| S4 hypothesis / opposite | 138 / 138 (68 sessions) | 160 / 160 | **86.3 %** | no trade-through in the touch minute 22 |

- S2's edge limit sits 1 pt inside the box edge. Price nearly always trades through it once it touches, and when the
  touch minute does not trade through, a later minute does (the fill search continues).
- S4's level is the touch extreme itself. On 22 of 160 touches price reached the level but never traded a tick
  beyond it in that minute: those are the touches where a resting order at L may not fill.
- The touch-fill model finds CR-CP's own fill minute on every S2 trade, and CR-CT's own touch second on every S4 touch.

### ±10 reproduction (C8b): 1-s touch-fill model vs each CR's reported win rate

| Side | 1-s touch-fill | Original | Δ |
|---|---|---|---|
| S1 long, covered (31 / 47) | 65.96 % | 66.0 % | < 0.1 pp |
| S1 long, all 49 (32 / 49) | 65.31 % | 65.31 % (CR-CL race) | 0 |
| S1 short, all 49 | 34.69 % | 34.69 % | 0 |
| S2 chosen (70 / 151) | 46.36 % | 46.4 % | < 0.1 |
| S2 opposite (80 / 145) | 55.17 % | 55.2 % | < 0.1 |
| S3 fade (20 / 51) | 39.22 % | 39.2 % | < 0.1 |
| S3 continuation | 60.78 % | 60.8 % | < 0.1 |
| S4 FAST break (48 / 85) | 56.47 % | 56.5 % | < 0.1 |
| S4 SLOW break (38 / 70) | 54.29 % | 54.3 % | < 0.1 |
| S4 hypothesis pooled (80 / 155) | 51.61 % | 51.61 % | 0 |
| S4 opposite pooled | 48.39 % | 48.39 % | 0 |

Every side reproduces within 0.05 pp. The 1-s race also reproduces the minute-bar races outcome for outcome: the
originals had no same-minute ties, so the finer clock changes no ±10 result.

### Gate (C9): passed

- Every primary cell has ≥ 40 fills (minimum 49: S1).
- All 28 data and reproduction checks pass.
- S1 (49 legs) and S3 (58 fades) are close to the 40-resolved floor. At ±2 / ±4 almost every trade resolves before
  13:00, so this should not bind; the run reports it either way.
- Next: commit `scripts/cr_cv_run.py`, then run it once.

---

## Result (single run, 2026-10-26) — `scripts/cr_cv_run.py` @ `bb4d25e` → `scripts/logs/cr_cv/results.json`

One launch, exit 0, no re-run (50 s). Schedule asserted equal to Step 0 (2,008 rows; fills per cell unchanged). Train
only; R-a, R-b and the holdout were not read. 10,000 session-cluster draws, seed 20260920, one matrix per setup universe.
Every race is a **mechanical bracket (B6), not Ryan's trade**; S1 is a measurement only (C12). No same-second ties
occurred in any cell; unresolved ≤ 2 per primary cell.

### Primary — 16 cells, after-cost expectancy per trade E = (2p − 1)·T − 0.50 pt (primary fill model)

Break-even win rate: 62.5 % at ±2, 56.25 % at ±4.

| Setup | Side | T | n (sessions) | Win rate [95 %] | E pts [95 % CI] | Holm level: CI | Holm p | Verdict |
|---|---|---|---|---|---|---|---|---|
| S1 G1-down | long (hyp) | 2 | 49 (36) | 51.0 % [37.5, 65.3] | −0.46 [−1.00, +0.11] | 98.75 %: [−1.14, +0.26] | 0.49 | INCONCLUSIVE |
| S1 | long (hyp) | 4 | 49 (36) | 55.1 % [41.5, 69.6] | −0.09 [−1.18, +1.07] | 95.0 %: [−1.18, +1.07] | 0.91 | INCONCLUSIVE |
| S1 | short (opp) | 2 | 49 (36) | 34.7 % [22.2, 46.5] | −1.11 [−1.61, −0.64] | 99.69 %: [−1.85, −0.41] | < 0.001 | REFUTED |
| S1 | short (opp) | 4 | 49 (36) | 36.7 % [24.4, 49.0] | −1.56 [−2.54, −0.58] | 99.38 %: [−2.87, −0.14] | 0.019 | REFUTED |
| S2 range edge | chosen (hyp) | 2 | 175 (76) | 37.1 % [30.1, 44.3] | −1.01 [−1.30, −0.73] | 99.67 %: [−1.42, −0.57] | < 0.001 | REFUTED |
| S2 | chosen (hyp) | 4 | 173 (76) | 39.3 % [30.6, 48.3] | −1.36 [−2.05, −0.64] | 99.50 %: [−2.34, −0.31] | 0.002 | REFUTED |
| S2 | opposite | 2 | 170 (81) | 45.9 % [38.7, 53.1] | −0.66 [−0.95, −0.37] | 99.64 %: [−1.11, −0.25] | < 0.001 | REFUTED |
| S2 | opposite | 4 | 168 (80) | 48.8 % [40.8, 56.9] | −0.60 [−1.24, +0.05] | 99.00 %: [−1.45, +0.26] | 0.41 | INCONCLUSIVE |
| S3 failed sweep | fade (hyp) | 2 | 58 (41) | 50.0 % [37.1, 61.5] | −0.50 [−1.02, −0.04] | 99.29 %: [−1.17, +0.13] | 0.26 | REFUTED * |
| S3 | fade (hyp) | 4 | 58 (41) | 48.3 % [35.4, 60.0] | −0.64 [−1.67, +0.30] | 98.33 %: [−1.93, +0.51] | 0.58 | INCONCLUSIVE |
| S3 | continuation | 2 | 58 (41) | 36.2 % [25.0, 48.3] | −1.05 [−1.50, −0.57] | 99.44 %: [−1.70, −0.35] | 0.004 | REFUTED |
| S3 | continuation | 4 | 58 (41) | 44.8 % [33.3, 57.1] | −0.91 [−1.83, +0.07] | 99.17 %: [−2.16, +0.48] | 0.41 | INCONCLUSIVE |
| S4 first touch | FAST break + SLOW fade (hyp) | 2 | 138 (68) | 42.0 % [33.6, 50.3] | −0.82 [−1.16, −0.49] | 99.62 %: [−1.29, −0.34] | < 0.001 | REFUTED |
| S4 | hyp | 4 | 137 (68) | 53.3 % [45.7, 61.0] | −0.24 [−0.84, +0.38] | 97.50 %: [−0.94, +0.47] | 0.91 | INCONCLUSIVE |
| S4 | opposite | 2 | 138 (68) | 41.3 % [34.0, 49.2] | −0.85 [−1.14, −0.53] | 99.58 %: [−1.26, −0.38] | < 0.001 | REFUTED |
| S4 | opposite | 4 | 138 (68) | 41.3 % [33.9, 48.9] | −1.20 [−1.79, −0.59] | 99.55 %: [−2.08, −0.31] | < 0.001 | REFUTED |

**No cell passes. 9 REFUTED, 7 INCONCLUSIVE, 0 underpowered.** Every point estimate is below 0. The best are S1 long
±4 (−0.09) and S4 hypothesis ±4 (−0.24), both with CIs centred near 0.

\* S3 fade ±2 is REFUTED on the 95 % CI (upper −0.04), the rule C7 fixed from the kickoff's "the CI". Its Holm-level
upper bound is +0.13, so under a Holm-adjusted refutation it would be INCONCLUSIVE. It is the only cell where the two
readings differ.

### Readouts (no verdicts, unadjusted)

**Cost 0.25 / 0.75 pt** (primary fills; E [95 % CI]). Positive point estimates appear only at 0.25 pt, and only in
three cells:

| Cell | E at 0.25 | E at 0.75 |
|---|---|---|
| S1 long ±4 | +0.16 [−0.93, +1.32] | −0.34 [−1.43, +0.82] |
| S4 hyp ±4 | +0.01 [−0.59, +0.63] | −0.49 [−1.09, +0.13] |
| S3 fade ±2 | −0.25 [−0.77, +0.21] | −0.75 [−1.27, −0.29] |
| every other cell | −0.21 to −1.31 | −0.71 to −1.81 |

**Touch-fill model** (each CR's original entry, no slippage; E at 0.50):

| Cell | Touch-fill win rate / E | Primary win rate / E |
|---|---|---|
| S1 long ±2 / ±4 | 57.1 % / −0.21 [−0.70, +0.29]; 59.2 % / +0.23 [−0.78, +1.32] | 51.0 % / −0.46; 55.1 % / −0.09 |
| S2 chosen ±2 / ±4 | 41.0 % / −0.86; 41.5 % / −1.18 | 37.1 % / −1.01; 39.3 % / −1.36 |
| S2 opposite ±2 / ±4 | 48.5 % / −0.56; 49.1 % / −0.57 | 45.9 % / −0.66; 48.8 % / −0.60 |
| S3 fade ±2 / ±4 | 60.3 % / −0.09 [−0.59, +0.37]; 50.0 % / −0.50 | 50.0 % / −0.50; 48.3 % / −0.64 |
| S3 continuation ±2 / ±4 | 39.7 % / −0.91; 50.0 % / −0.50 | 36.2 % / −1.05; 44.8 % / −0.91 |
| S4 hyp ±2 / ±4 | 48.8 % / −0.55; 50.9 % / −0.42 | 42.0 % / −0.82; 53.3 % / −0.24 |
| S4 opposite ±2 / ±4 | 51.2 % / −0.45; 49.1 % / −0.58 | 41.3 % / −0.85; 41.3 % / −1.20 |

- Even with free fills, only S1 long ±4 has a positive E (+0.23), and its CI spans −0.8 to +1.3.
- The quarter-point fill cost decides a lot at ±2. For the bar-close entries (S1, S3), win rates fall 6–10 pp
  (S3 fade 60.3 → 50.0 %). A 0.25-pt adverse fill turns a ±2 race into +2.25 / −1.75.

**±10 reference** (win rate; E at 0.50, [95 % CI]):

| Side | Primary fill | Touch fill (= original) |
|---|---|---|
| S1 long | 63.3 % (n 49), E +2.15 [−0.50, +4.82] | 65.3 %, E +2.56 [+0.06, +5.06] |
| S2 chosen / opposite | 44.6 % (148) / 54.2 % (144) | 46.4 % / 55.2 % |
| S3 fade / continuation | 38.5 % (52) / 60.8 % (51) | 39.2 % / 60.8 % |
| S4 hyp / opposite | 51.5 % (136) / 46.3 % (136) | 51.6 % / 48.4 % |

The ±10 touch-fill rates are the Step 0 reproduction. The primary fill costs each ±10 side 0–2 pp.

**Trade-through fill share** (Step 0): S2 chosen 98.3 %, S2 opposite 99.4 %, S4 86.3 % (22 of 160 touches never
traded a tick beyond the level in the touch minute).

**S4 split, hypothesis side** (descriptive): FAST break ±2 40.0 % (n 75), ±4 **58.7 %** (75), E +0.19 [−0.56, +0.95];
SLOW fade ±2 44.4 % (63), ±4 46.8 % (62), E −0.76. The ±4 hypothesis number is FAST breaks; SLOW fades lose.

### Reading

- **Nothing works at ±2 or ±4 after costs.** No cell passes. Nine are refuted (credibly negative), and the other seven
  have point estimates below 0.
- **±2 is worse than ±4 everywhere.** At ±2 the break-even is 62.5 % and no setup gets near it. The best is S3 fade at
  50 % (60 % with free fills). At ±4 three cells are near break-even (S1 long 55 %, S4 hyp 53 %, S2 opposite 49 %), and
  none clears it.
- **Limit entries at range edges get picked off.** S2's chosen-edge limit fills win 37 % at ±2 (41 % with touch fills).
  When price trades through a limit at the edge, it tends to keep going: the range breaks more often than it holds.
  This is the CR-CP result (the flow-chosen edge loses) at a smaller scale.
- **S1 (G1-down long) is the only setup whose ±10 edge carries down, and it does not survive costs.** About 55 % at ±4
  against a 56 % break-even; E −0.09 [−1.18, +1.07]. Its ±10 E is +2.2 pts, but that was already inconclusive in
  CR-CL, and S1 is a measurement only.
- Not shortlisted. Nothing goes to R-a.
