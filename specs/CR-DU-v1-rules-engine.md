# CR-DU — v1 rules engine: panel verdict, train backtest, contradiction bars, v1-trades overlay

Branch `feat/CR-DU-v1-rules-engine` off `origin/main` `96345a7`, worktree `../volatility-monorepo-cr-du`. Research CR
(standing-rulings conformance table below). Session note: vault `sessions/2026-10-31 - CR-DU — v1 Rules Engine.md`.

## Kickoff (Ryan, chat 2026-10-31, verbatim)

```
CR-DU — v1 rules engine: panel verdict, train backtest, contradiction bars, v1-trades overlay

Kickoff
1. git fetch && git checkout -b feat/CR-DU-v1-rules-engine origin/main (tip 96345a7 or later), in its own worktree. This is the first action, before any reads. If CR-DV (#128) merges during this CR, rebase onto it.
2. Server discipline: one DB-backed local server at a time across all sessions. CR-DV may still be re-timing. Do Step 0 without a server (code reading and read-only DB only). Start a server only when :8068 is free and no other CR session is mid-smoke. Stop it when done and say "released :8068" in your report.
3. Read:
   - CLAUDE.md
   - Dash/components/field-conventions.md
   - Dash/research/standing-rulings.md
   - Dash/research/discretionary-rules-v1.md (THE rule set; rule numbers below are v1's)
   - Dash/research/order-flow-reading-notes.md
   - Dash/research/CR-CZ learning set — results 2026-10-31.md
   - the CR-DQ, CR-DS, CR-DR specs (panel, verdict, shared boxPanelOf path)
4. Session note + spec freeze at specs/CR-DU-v1-rules-engine.md with the standing-rulings conformance table (research CR). Any "deviates" is a STOP.

Goal
Make discretionary-rules-v1 one mechanical engine, used by the live panel, a train-only backtest and a per-day "v1 trades" overlay, so all three give the same answer. Test whether contradiction bars add value as in-box evidence. Store the engine's verdicts on set-2 cards blind (Ryan labels set 2 in parallel).

Data rules (hard)
- Backtest and overlay statistics use train sessions only (≤ 2026-06-05). Never read holdout (2026-06-06 → 10-26) outcomes. Never query bt_derived_repair_archive.
- B5: never pool feeds. The backtest uses Databento (es_flow_1s_databento / es_flow_1m_databento; exact cells from es_flow_price_1m source 'databento'). The live panel uses Ironbeam. Flow thresholds are per-feed percentiles, never raw counts carried across feeds.
- No lookahead: a decision at minute t uses only data stamped before t+1 min. Percentile baselines come from prior sessions only.
- v1 is fixed. No tuning to results. Only the pre-declared sensitivity rows below.
- Label every backtest output "train, in-sample". The holdout run is a later CR with a pre-registration file.

Step 0, diagnosis gate (all of these in the Step-0 commit; STOP after)
Q1. Map the current code: where box, Δ share, c/pt, absorption, conflict and the verdict live (file:function), including CR-DR's shared boxPanelOf. Can they run as pure functions over (session data up to t, settings)?
Q2. For each v1 rule 1–20, give one of:
   (a) already mechanical: cite the code;
   (b) mechanical with a definition you propose;
   (c) not mechanizable.
   Expected (b) items:
   - rule 4 "strong" GEX level (the A12 yellow/darkest definition as coded) and "break a major level before trading through it";
   - rule 5 "absorption at the same level that held";
   - rule 7 "highs not extending";
   - rule 12 "delta-supported confirmation";
   - rule 13 fill price (CR-CP D4 edge ± 1 tick vs the exact edge, propose one);
   - rule 15 "beyond the zone" (A4 as coded).
   Every (b) definition is a STOP for Ryan's approval in chat. If rule 12 is (c), say so; the engine runs without it.
Q3. FOMC decision days (rule 1): there's no events table. Propose a hard-coded list of 2026 decision dates with a source URL.
Q4. Fill and exit realism:
   - a limit fills only if price trades through it by ≥ 1 tick;
   - same-minute stop and target: resolve from 1-s flow, otherwise count the stop;
   - flatten at 12:59 PT;
   - report gross, and net of 0.25 pt per round trip.
   Flag anything the data can't support.
Q5. Contradiction-bar percentiles: per feed and per 30-min RTH bucket, p10/p50/p80/p90 of minute volume, signed delta and range, built from train. Show Databento, Ironbeam where it has train-period data, and the ratio between them.
Q6. Losses per day and re-entry (rules 19–20): simulated in the backtest. Live: can CR-DR's tables count today's losses intraday (only if trades are logged during the session)? If not, live shows the 20-min re-entry timer only, and the loss count is out of scope.
Q7. Live freshness: for each engine input (box, Δ share, c/pt, absorption incl. CR-DV provisional marks, contradiction bars, GEX levels), name the live source and confirm it's readable within about 15 s of minute close (CR-DV measured the absorption path). List any input that isn't.

Implementation (one commit per item)
1. Rule engine.
   - A shared pure module: evaluate(session data up to minute t, settings) → {verdict SHORT / LONG / NO ACTION, failed_rules[], evidence{}}.
   - Built on boxPanelOf, no parallel copy of the box/verdict code.
   - Defaults: box min 8, entry cutoff 12:15 PT, target +10, max 3 losses a day, 20-min re-entry wait plus a new 8-min box on the same side.
   - Unit tests on fixtures for every rule and every failure reason.
2. Panel.
   - The CR-DS box table shows the engine's verdict and failed-rule reasons using the existing row styles.
   - The ⚙ box-minimum default becomes 8.
   - Rename the Δ share row's "buyers leaning" / "sellers leaning" to "net buying" / "net selling" (approved here).
   - Optional alert, off by default: a sound plus a browser notification when the verdict changes to SHORT or LONG.
   - No other layout or colour changes (visible-change discipline).
3. Train backtest: scripts/cr_du_backtest_v1.py, read-only.
   - Every trade: date, side, box, entry / stop / target / exit, pts, reasons.
   - Totals: trades, trades/day, win %, net pts gross and net, avg win/loss, max drawdown, longest losing streak.
   - Splits by month, side and hour; no-trade reason counts.
   - Sensitivity, pre-declared and report-only: box min 10; no 12:15 cutoff; no loss cap.
   - Overlap with CR-CZ cards: how many of Ryan's takes and skips the engine also takes. Counts only, with no new outcome reveals.
4. Contradiction bars.
   - Primary, pre-declared definition, within the qualifying box on the edge being faded:
     - bullish: signed delta ≤ bucket p10, lower wick ≥ 50% of the range, volume ≥ bucket p80, range ≥ 2 pts;
     - bearish: the mirror.
   - Secondary, report-only: p20 delta, 40% wick, no volume floor.
   - Compare v1 trades with ≥ 1 such bar against those with none: n, win %, net pts, bootstrap 90% CI on the win-rate difference. State the small n.
   - Optional chart marker, off by default, in the existing marker style.
4b. "v1 trades" overlay.
   - A toggle in the Flow v3 Display group beside "My trades", off by default.
   - For the session shown, run the engine minute by minute with no lookahead. Draw each trade: entry marker, stop / +10 target lines while open, exit marker, pts. Use CR-DR's marker style in a distinct colour.
   - On hover, show the verdict and failed rules for that minute.
   - Blocked dates (toggle disabled, with the reason shown):
     - holdout 2026-06-06 → 10-26: "holdout date: not shown";
     - any session_date of a set-2 card while set 2 is unfinished (any set-2 card without a review in setup_reviews): "set 2 in progress".
   - Unit test the blocking.
5. Set-2 blind verdicts.
   - Run the engine at each set-2 card's decision minute.
   - Store the results in a new table setup_engine_verdicts (candidate_id, engine_version, verdict, failed_rules, evidence jsonb, computed_at), written via the backfill_safety protocol.
   - Not shown in any UI. Report counts only (SHORT / LONG / NO ACTION), never per card.
   - If set-2 cards aren't loaded yet, build the writer and run it on CR-CZ cards as a dry run (counts only). Say so.
6. Smoke + wrap.
   - Browser check on 2026-10-30, #5's box (anchor 07:40): the verdict at box min 8 vs CR-DQ's "box still forming" at 10.
   - The v1 overlay on a train day and on 10-30.
   - The blocked-date message on a holdout date.
   - Existing specs reported as a diff against origin/main.
   - Session note with the results tables, rulings re-check, copy the logs, remove stale .claude/launch.json entries, PR. Report the PR number and head SHA, and "released :8068".

Report at the Step-0 STOP with the Q2 proposed definitions, the FOMC list and the Q5 tables.
```

## Standing rulings conformance (`Dash/research/standing-rulings.md`, status current, read 2026-10-31)

| # | Mark | How |
|---|---|---|
| A1 | honoured | GEX levels and earlier absorption are obstacles / location (rules 4–5), never a trade by themselves |
| A2 | honoured | box ≥ 8 min by closes (rule 6), no upper duration limit |
| A3 | honoured | entry only after the box has formed and the verdict names a side; never at the first touch |
| A4 | honoured, definition pending | stop beyond the zone (rule 15); "beyond the zone" is not coded anywhere — Step 0 Q2 proposes a definition (STOP) |
| A5 | honoured | flow decides; v1 trades the fade only (the break side, H-CZ1, is a set-2 candidate, not a v1 rule) |
| A6 | n/a | skew not used (parked) |
| A7 | honoured, definition pending | rule 6 "after a larger move" = the vol-scaled swing; proposed in Q2 (STOP) |
| A8 | honoured | rule-4 blocks reported split by call (yellow) / put (darkest) level |
| A9 | n/a | not a v1 rule |
| A10 | honoured | earlier same-day absorption is location (rules 4–5) |
| A11 | honoured | +10 target fixed by v1 rule 16 (Ryan 2026-10-31); MFE per trade reported as a readout; no target tuning |
| A12 | honoured | strong level = \|net\| ≥ the chart colour span (p95 of \|net\| over the drawn levels, `_compute_color_span`), corrected placement |
| A13 | n/a | one contract; sizing deferred |
| A14 | n/a | skew not used |
| A15 | honoured | bracket fixed by v1, nothing tuned; a stop-free ±10 race of the same entries is proposed as a readout |
| A16 | n/a | skew not used |
| A17 | honoured | order flow alone |
| A18 | n/a | the engine's boxes are not defined relative to a level |
| A19 | honoured | rules 7–8 (Δ share lean + effort / no advance) |
| A20 | honoured | limit at the box edge (rule 13; exact price proposed in Q2, STOP) |
| A21 | n/a | second-touch setup not in v1 |
| A22 | honoured | FOMC decision days: no trades; counted as their own cell in the report |
| B1 | honoured | contradiction-bar comparison reported as pass / inconclusive / refuted on the CI, small n stated |
| B2 | honoured | only the three pre-declared sensitivity rows and the pre-declared secondary contradiction definition; all disclosed |
| B3 | honoured | v1 fixed; contradiction bars evaluated as one add-on, nothing tuned |
| B4 | honoured | train ≤ 2026-06-05 only; holdout blocked in the overlay and never read; `bt_derived_repair_archive` never queried. Set-2 caveat raised in Step 0 |
| B5 | honoured | backtest on Databento only (bars, flow, cells); live on Ironbeam; per-feed percentiles |
| B6 | honoured | the bracket is v1's (A4 stop, +10); labelled "v1 rules, train, in-sample" |
| B7 | n/a | no placebo |
| B8 | n/a | engine verdicts go to their own table, never `setup_reviews` |
| C1 | **deviates — STOP** | item 4b asks for "a distinct colour" for the v1-trade markers; CR-DR uses blue (long) / slate (short), so any distinct colour is outside the palette. Needs Ryan's colour choice and approval |
| C2 | deviates — pre-approved in the kickoff | panel verdict replaced by the engine's, ⚙ box-minimum default 10 → 8, Δ share wording "net buying" / "net selling"; new features (alert, contradiction marker, overlay) off by default. Recorded, not stopped for (CR-DT precedent) |

Revoked / superseded: R1 (touch-bar close entry), R2 (automatic level entry), R3 (CR-BW exit), R5, R6, R7 — none used.
R4 n/a.

## Step 0 — diagnosis (read-only; no server started; code at `96345a7`)

Scripts `scripts/cr_du_step0/`, logs `scripts/logs/cr-du/` (git-ignored). DB reads: `setup_candidates` / `setup_reviews`
counts (no `outcome` column read), and train-only minute flow for Q5. No outcome, holdout row or
`bt_derived_repair_archive` read.

### Q1 — code map

| Piece | Where | Pure? |
|---|---|---|
| Box chain (edges, growth, pending break, probes, freeze) | `react_price_preview/src/flowV3.js:edgeReadout` (+ `boxStats`, `probeStats`, `drawnBoxes`) | yes: (cells payload, flat bars, anchor, shift, holdMin 15, end, breakHoldMin 5) |
| Box Δ / volume / dPrice / minutes | `flowV3.js:anchorReadout` via `segmentReadouts`, rows = `flowShared.alignV2ToCandles(v2 bars, candles)`, closes = `rowCloses(rows, candles)` | yes |
| Δ share | `flowV3.js:readoutShare` (Σ delta ÷ Σ vol) | yes |
| c/pt + percentiles | `flowV3.js:boxPercentiles` on `src/data/boxBaseline.json` (`scripts/cr_dq_baseline.py`) | yes, but see caveat 3 |
| Absorption levels | server `packages/shared/flow_heat.py:absorption_marks` (called by `apps/web/modules/Ironbeam/flow_v3.py:_build`); client `flowV3.js:absorptionLevels` (top 8, `asOf`) and `boxLevels` | yes, but see caveat 4 |
| Conflict / two-sided / reads / verdict | `flowV3.js:boxVerdict` (order A22 → forming → A24 → two-sided → A19/A23 → watch break → nothing notable); thresholds `BOX_RULES_DEFAULTS` | yes |
| Evidence table | `flowV3.js:boxEvidenceRows` → `components/BoxPanel.jsx` | yes |
| Shared path (CR-DR) | `flowV3.js:edgeBoxEntries` → `boxPanelOf` → `currentBoxVerdict`; used by `PriceChart.jsx:2821` and `tradeContext.js:entryContext` (until-replay) | yes |
| Minute replay | `flowV3.js:readHistory` (CR-DS item 4) | yes |
| FOMC list | `flowV3.js:FOMC_2026_DECISION_DAYS` (Python copy `scripts/cr_dq_step0.py:FOMC_2026`) | constant |

So the engine can be `evaluate(sessionData ≤ t, settings)` over these functions, in `flowV3.js` (or a new
`v1Engine.js` that imports them; no copy). The Python backtest drives it through `node` (v22 available) on JSON
session inputs it builds read-only, so the box / verdict code exists once. Caveats that the engine must fix to be
no-lookahead and single-feed:

1. **Anchor.** `edgeReadout` needs an anchor (the user's click). Backtest and overlay need an automatic one → Q2 rule 6.
2. **Mixed feeds on train dates (B5).** On a train date the box edges / lastClose come from the cells payload's
   flow bars (Databento), but dPrice comes from the chart candles (`/api/ironbeam/bars` → `ironbeam_es_1m_bars`,
   Ironbeam in 2026). The backtest and overlay will pass Databento minute closes as the candles. Live is Ironbeam
   for both, unchanged.
3. **Baseline lookahead.** `boxBaseline.json` is one train-wide baseline (2026-01-02 → 06-05, 100 / 99 sessions);
   on a train date it contains that day and later days. Backtest and overlay need a per-session baseline from
   prior train sessions only (same script, expanding window), with a warm-up of 20 prior sessions → first scored
   session ≈ 2026-02-03 (R-a 2024 / R-b 2025 can't be used as warm-up, B4). Live keeps the fixed train baseline
   (all prior to the live date).
4. **Absorption lookahead.** `absorptionLevels(asOf)` drops marks confirmed or broken after `asOf`, but a mark's
   `merged_vol` includes later merges, so contracts and the top-8 rank can use the future (noted in CR-DS). The
   backtest / overlay will compute marks from cells cut at each minute (`flow_heat.absorption_marks` with
   end = t, the CR-DV closed-minute rule) and hand the engine a per-minute marks list.
5. **GEX band lookahead.** The chart's drawn levels use the band of the bars loaded (whole day on a past date).
   Rule 4 needs the band of bars ≤ t (CR-DT T5 did this per card) → computed server-side, recomputed when the
   running high / low moves.

### Q2 — rules 1–20

| v1 rule | Class | Code / proposed definition |
|---|---|---|
| 1 FOMC days | (a) | `boxVerdict` `fomc` → NO TRADES; list Q3 |
| 2 RTH only | (a) | engine runs on the RTH segment only (06:30–13:00 PT) |
| 3 no entries after 12:15 PT | (a) new constant | no fill at or after 12:15:00 PT; a resting limit is cancelled then; open trades run to the flatten |
| **4 nothing strong in the path** | **(b) STOP** | at the decision minute, for entry E and target E ∓ 10: (i) **strong GEX level** = a corrected drawn level (`_select_levels`, gross ≥ 40 $bn, band from RTH bars ≤ t ± 150) with \|net\| ≥ the colour span (p95 of \|net\| over those drawn levels, `_compute_color_span`, = yellow / darkest); blocked if one lies strictly between E and the target. (ii) **earlier absorption zone** = an absorption level (the chart's top 8, as of t) that started **before the box's first bar**, is unbroken at t, and opposes the trade (sellers absorbed for a short, buyers absorbed for a long) strictly between E and the target. "Wait for a major level to break before trading through it" needs nothing extra: once a level has broken it is behind the price, so a later box can trade. No corrected levels for the day (e.g. 2026-02-19 `no_es_snapshot`) → no trade, reason "no GEX levels" (counted) |
| **5 earlier absorption that held** | **(b) STOP** | an earlier (started before the box's first bar), unbroken absorption level of the trade's side (buyers absorbed for a short) within 2 pts (`absorbPts`) of the faded edge's wick extreme = evidence "location support", never a trigger. **Change vs today's panel:** `boxVerdict` counts any absorption at the top that started ≤ the box end as a trigger (`absTop`); the engine counts only marks that start inside the box as triggers |
| 6 the box | (a) + **(b) STOP** | ≥ 8 min (`minAgeMin` 10 → 8) and ≥ 2 pts by closes: (a) `boxVerdict` `formed`. **Anchor (b):** the 06:30 PT RTH open (`sessionOpenAnchor`), boxes chained by the CR-DI / CR-DN break rule at their display defaults (edge hold 15 min, break hold 5 min), which become fixed engine settings. The live panel evaluates the chart's anchor, so it equals the overlay when the anchor is the open. **"After a larger move" (b):** at the box's first bar, the RTH bars of the previous 60 min contain a high or low at least one A7 swing from that bar's close (swing = 12 × session implied move ÷ 2026-train median implied move, CR-CP `swing_of`; 06:30 box: only RTH bars since 06:30 count) |
| **7 short at the top** | **(b) STOP** | (a) as coded: price ≤ 2 pts below the top and (Δ share lean ≥ p80 buying and / or buyers absorbed at the top inside the box) and no advance. **"Highs not extending" (proposed):** no 1-min high in the last 3 closed bars above the box's wick high before them. Stall = c/pt ≥ p80 **or** highs not extending. Alternative: keep today's "progress ≤ 2 pts" (CR-DS amendment) as the stall leg. Ryan picks one |
| 8 long at the bottom | (b) mirror of 7 | lows not extending (no new wick low in the last 3 bars) |
| 9 conflict | (a) | `boxVerdict` CONFLICT (A24) and two-sided absorption → NO ACTION |
| 10 balanced delta | (a) | a lean needs \|Δ share\| ≥ p80; a move on unremarkable share is never a side |
| 11 one counter-move minute | (a) | every read is over the whole box (≥ 8 min of readout); no single-minute trigger |
| **12 response vs push** | **(a) constraint + (b) STOP** | the constraint ("price alone is not confirmation") holds: the engine never confirms on price. Proposed pattern, **evidence only, not a gate**: for a long, a new box low L1, then within 10 min a low L2 ≥ L1 + 1 tick, with sell volume over the L2 push (bars from the last high before L2 to L2) < sell volume over the L1 push; mirror for shorts. Recorded in `evidence.rule12`; recommend the engine does not require it in v1 |
| **13 limit at the edge** | **(b) STOP** | note: CR-CP D4 is the edge ∓ **1 pt** (not 1 tick). Proposed: limit at the **exact close-based edge** (short at the top, long at the bottom), placed at the close of the minute the verdict turns SHORT / LONG, live from the next minute; filled when a print trades ≥ 1 tick through it (Q4); cancelled at the first minute close whose verdict for that box is not the same side, at a box break, or at 12:15 PT. Alternative: D4 (edge ∓ 1 pt) |
| 14 bracket | (a) | simulated bracket on every fill; assert stop on the far side of the entry |
| **15 stop beyond the zone** | **(b) STOP** | A4 is not coded anywhere (CR-CP / CR-CZ were stop-free or fixed 10). Proposed: stop = the box's wick extreme on the faded side at the decision minute **+ 2 pts** (the box's break threshold `EDGE_BREAK_PTS`) for a short, mirror for a long. Alternative: wick extreme + 1 tick. Stop distances reported as a distribution |
| 16 target +10 | (a) | constant |
| 17 no stop moves | (a) | bracket fixed until stop / target / flatten |
| 18 no wider stop | (a) | stop is the rule-15 definition only |
| 19 ≤ 3 losses a day | (a) backtest | count of stopped trades per session; live Q6 |
| **20 re-entry** | **(b) STOP** | after a stop: no entry on either side for 20 min from the stop fill; a same-side re-entry needs a box that started after the stopped trade's entry and is ≥ 8 min old. Plus **one trade per box** (any outcome). Alternative: the 20-min wait applies to the same side only |

### Q3 — FOMC decision days (2026)

`flowV3.js:FOMC_2026_DECISION_DAYS` already holds them (CR-DQ): **2026-01-28, 03-18, 04-29, 06-17, 07-29, 09-16,
10-28, 12-09** (the second day of each scheduled meeting), source
https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm (checked 2026-10-31 by CR-DQ). Proposed: the engine
reuses this constant (no second list). Train days affected: 01-28, 03-18, 04-29.

### Q4 — fill and exit realism

- **Trade-through ≥ 1 tick:** supported. Short limit at P fills on the first minute after the order with high ≥
  P + 0.25 (Databento 1-min), at P; the fill second = the first `es_flow_1s_databento` second with high ≥ P + 0.25.
- **Same-minute stop / target:** supported from the 1-s table (per-second OHLC, 06:15–13:00 PT): walk seconds after
  the fill second; a second whose range covers both counts the stop (needs ≥ ~12 pts in one second, practically
  never). The fill second's own adverse range beyond P counts toward the stop (as CR-CP C10's fill bar).
- **Flatten 12:59 PT:** exit at the open of the 12:59 bar (first 1-s print ≥ 12:59:00). Supported.
- **Gross and net of 0.25 pt / round trip:** reported both ways.
- **Not supported:** queue position / partial fills (trade-through is the proxy); stop slippage — no book, so a stop
  fills at the stop price, or at the second's open if a second opens beyond it; MES vs ES (± 1 tick, CR-DR S0.7) is
  not modelled.

### Q5 — contradiction-bar percentiles (train 2026-01-02 → 06-05, FOMC days and the March expiry week out; front symbol by RTH volume)

Databento 100 sessions, Ironbeam 99. Full tables in `scripts/logs/cr-du/q5_contradiction_pcts.txt`.

Minute volume (contracts), p10 / p50 / p80 / p90:

| Bucket PT | Databento | Ironbeam | IB / DB |
|---|---|---|---|
| 06:30–07:00 | 3008 / 4944 / 6955 / 8646 | 2829 / 4568 / 6238 / 7632 | 0.94 / 0.92 / 0.90 / 0.88 |
| 07:00–07:30 | 2420 / 3958 / 5718 / 6878 | 2287 / 3648 / 5141 / 6184 | 0.94 / 0.92 / 0.90 / 0.90 |
| 07:30–08:00 | 1923 / 3282 / 4788 / 5975 | 1797 / 3031 / 4387 / 5393 | 0.93 / 0.92 / 0.92 / 0.90 |
| 08:00–08:30 | 1615 / 2798 / 4087 / 5095 | 1517 / 2588 / 3712 / 4526 | 0.94 / 0.93 / 0.91 / 0.89 |
| 08:30–09:00 | 1359 / 2396 / 3546 / 4567 | 1295 / 2232 / 3285 / 4144 | 0.95 / 0.93 / 0.93 / 0.91 |
| 09:00–09:30 | 1056 / 1935 / 3095 / 4016 | 994 / 1805 / 2854 / 3655 | 0.94 / 0.93 / 0.92 / 0.91 |
| 09:30–10:00 | 958 / 1819 / 2966 / 3759 | 906 / 1687 / 2718 / 3434 | 0.95 / 0.93 / 0.92 / 0.91 |
| 10:00–10:30 | 932 / 1765 / 2749 / 3513 | 888 / 1654 / 2520 / 3184 | 0.95 / 0.94 / 0.92 / 0.91 |
| 10:30–11:00 | 815 / 1596 / 2657 / 3535 | 775 / 1484 / 2428 / 3186 | 0.95 / 0.93 / 0.91 / 0.90 |
| 11:00–11:30 | 887 / 1622 / 2542 / 3357 | 837 / 1506 / 2324 / 2998 | 0.94 / 0.93 / 0.91 / 0.89 |
| 11:30–12:00 | 856 / 1540 / 2492 / 3343 | 808 / 1439 / 2291 / 3001 | 0.94 / 0.93 / 0.92 / 0.90 |
| 12:00–12:30 | 1003 / 1779 / 2748 / 3584 | 950 / 1665 / 2505 / 3234 | 0.95 / 0.94 / 0.91 / 0.90 |
| 12:30–13:00 | 1414 / 2859 / 5426 / 7657 | 1322 / 2650 / 4942 / 7044 | 0.94 / 0.93 / 0.91 / 0.92 |

Minute signed delta (buy − sell), p10 / p50 / p80 / p90 (p50 ratios are noise around 0):

| Bucket PT | Databento | Ironbeam | IB / DB (p10, p80, p90) |
|---|---|---|---|
| 06:30–07:00 | −503 / 32 / 363 / 593 | −350 / 29 / 299 / 477 | 0.70 / 0.82 / 0.80 |
| 07:00–07:30 | −463 / 18 / 309 / 512 | −329 / 21 / 236 / 369 | 0.71 / 0.76 / 0.72 |
| 07:30–08:00 | −413 / 2 / 275 / 451 | −296 / 4 / 212 / 345 | 0.72 / 0.77 / 0.76 |
| 08:00–08:30 | −367 / −4 / 235 / 405 | −273 / 5 / 188 / 300 | 0.74 / 0.80 / 0.74 |
| 08:30–09:00 | −334 / −9 / 194 / 337 | −249 / −8 / 150 / 258 | 0.74 / 0.77 / 0.77 |
| 09:00–09:30 | −313 / 0 / 176 / 307 | −237 / 4 / 145 / 243 | 0.76 / 0.82 / 0.79 |
| 09:30–10:00 | −292 / −10 / 173 / 309 | −226 / −7 / 140 / 234 | 0.77 / 0.81 / 0.76 |
| 10:00–10:30 | −295 / −4 / 164 / 284 | −213 / −4 / 124 / 216 | 0.72 / 0.76 / 0.76 |
| 10:30–11:00 | −280 / −7 / 163 / 297 | −215 / −3 / 129 / 218 | 0.77 / 0.79 / 0.73 |
| 11:00–11:30 | −269 / −1 / 159 / 260 | −211 / 0 / 124 / 207 | 0.78 / 0.78 / 0.80 |
| 11:30–12:00 | −280 / −9 / 145 / 264 | −219 / −8 / 111 / 211 | 0.78 / 0.77 / 0.80 |
| 12:00–12:30 | −292 / −15 / 162 / 296 | −226 / −9 / 133 / 228 | 0.77 / 0.82 / 0.77 |
| 12:30–13:00 | −505 / 8 / 294 / 506 | −379 / 12 / 259 / 430 | 0.75 / 0.88 / 0.85 |

Minute range (pts), p10 / p50 / p80 / p90 — Databento and Ironbeam equal to the tick in every bucket except the
p80 / p90 of 06:30–07:30 and 12:00–13:00 (Ironbeam 0.25 lower):

| Bucket PT | Databento | Bucket PT | Databento |
|---|---|---|---|
| 06:30–07:00 | 2.50 / 4.75 / 7.25 / 9.00 | 09:30–10:00 | 1.25 / 2.50 / 4.00 / 5.00 |
| 07:00–07:30 | 2.25 / 4.00 / 6.50 / 8.00 | 10:00–10:30 | 1.25 / 2.50 / 4.00 / 5.25 |
| 07:30–08:00 | 1.75 / 3.50 / 5.50 / 6.75 | 10:30–11:00 | 1.00 / 2.25 / 3.75 / 4.75 |
| 08:00–08:30 | 1.50 / 3.00 / 4.75 / 5.75 | 11:00–11:30 | 1.00 / 2.25 / 3.50 / 4.50 |
| 08:30–09:00 | 1.50 / 2.75 / 4.50 / 5.50 | 11:30–12:00 | 1.00 / 2.00 / 3.50 / 4.25 |
| 09:00–09:30 | 1.25 / 2.50 / 4.00 / 5.25 | 12:00–12:30 | 1.00 / 2.00 / 3.50 / 4.50 |
| | | 12:30–13:00 | 1.25 / 2.50 / 4.25 / 5.50 |

Paired minutes (38,064): Ironbeam / Databento volume median 0.943 (p5 0.855, p95 0.981); delta slope 0.68, corr
0.91; range difference median 0 — consistent with CR-CA (0.943, 0.71). So a raw delta threshold set on one feed
would be ~25–30 % off on the other; percentiles per feed as the kickoff says. Reading: "range ≥ 2 pts" is below the
median minute before 08:30 PT and near it after, so the volume and delta legs do the filtering.
Proposed detail for item 4: "on the edge being faded" = a bar inside the box span, before the decision minute,
whose low (bullish, for a long) / high (bearish, for a short) is within 2 pts of the box's wick bottom / top;
percentiles per feed from **prior** train sessions only in the backtest (the table above is whole-train, descriptive).

### Q6 — losses per day and re-entry

Backtest / overlay: simulated (rule 19 count of stops, rule 20 timer). Live: CR-DR's tables are filled only by
`POST /api/trade-log/upload` of a TradingView export (`apps/web/modules/TradeLog/routes.py`); there is no
intraday broker feed. So Ryan's own losses can't be counted live unless he uploads mid-session, and **the 20-min
timer can't start either** (it needs a stop time). Proposed (STOP): the live panel shows the engine's own
simulated trades for today (the overlay engine run on the live session): its losses today and its re-entry timer,
labelled "engine's trades, not yours". Alternative: rule 19 / 20 out of scope live.

### Q7 — live freshness (Ironbeam)

| Input | Live source | Readable after minute close |
|---|---|---|
| Box (edges, lastClose) | `/api/ironbeam/flow/cells` live RTH segment bars (`ironbeam_es_flow_1s` minutes), polled every 15 s (`FLOW_V2_LIVE_POLL_MS`) | ≤ ~15 s + endpoint (warm p95 ~0.7 s, CR-DV) |
| Δ share, c/pt (delta, vol) | `/api/ironbeam/flow?mode=v2` rows, last 5 min from the 1-s table (`FRESH_1S`), 15 s poll | ≤ ~15 s |
| dPrice (candle closes) | `/api/ironbeam/bars` (`ironbeam_es_1m_bars`, insert-once cron), full poll 30 s, live overlay 1 s | **not confirmed ≤ 15 s**: closed-minute candles can wait for the 30-s poll unless the 1-s overlay finalises them. Proposed: the engine takes dPrice from the cells payload's flow bars (same feed as the box), which also removes caveat 2 |
| Absorption | cells payload marks (`flow_heat.absorption_marks`) | on `origin/main` marks are computed to 13:00 and **repaint** (CR-DV Step 0); with CR-DV #128 (not merged): provisional ~15 s, confirmed ~195 s (3-min hold, by definition). The engine reads confirmed marks only. Depends on #128 for no-repaint equality with the backtest |
| Contradiction bars | v2 rows (delta, vol) + cells bars (h / l / o / c) | ≤ ~15 s |
| GEX levels | corrected levels from the prior session's 15:45 ET snapshot (`_corrected_gex_tables`) | fixed before the open; the band moves with price (server) |

### Other findings for Ryan (STOP)

- **Set 2 isn't loaded.** `setup_candidates` sets: `cc:fixture` 5, `cr-cb:round-1` 63, `cr-cb:round-2` 40,
  `cr-cz:cp-ranges` 120 (120 reviewed). So item 5 runs as a dry run on CR-CZ cards, and item 4b's "set 2 in
  progress" block needs the set-2 `set_name` (proposed: a constant set when the set-2 loader CR names it; empty
  list until then).
- **Set-2 contamination through the backtest.** Set-2's ~115 calls sit on train sessions the backtest scores. A
  per-trade list with dates, times and P&L would reveal outcomes near set-2 calls. Proposed: per-trade rows (and
  per-month splits) exclude sessions with a set-2 call until set 2 is reviewed; totals reported with and without
  them. Needs Ryan's call.
- **C1 (overlay colour)** — see the conformance table.
- **Engine vs today's panel verdict.** Besides box min 8: rule 5 (earlier absorption no longer a trigger), rule 7
  stall leg (if "highs not extending" is chosen), WATCH BREAK becomes NO ACTION, rule 4 adds blocks.

## Amendment 1 — Ryan's rulings at the Step-0 STOP (chat 2026-10-31)

| Item | Ruling |
|---|---|
| Q1 fixes 1–5 | approved as proposed (06:30 anchor, flow-bar closes, prior-sessions baseline with 20-session warm-up, marks from cells cut at t, GEX band ≤ t) |
| Rules 4, 5, 7 / 8, 12, 20 | approved as written; rule 7 / 8 stall = c/pt ≥ p80 **or** no new wick high (low) in the last 3 closed bars |
| Rule 6 | approved; confirm the 15-min edge hold doesn't stop a box qualifying at 8 min (below) |
| Rule 13 | primary = limit at the exact edge by closes, placed when the verdict turns, cancelled on lapse / break / 12:15. **Sensitivity row added:** CR-CP D4 (1 pt inside the edge) |
| Rule 15 | primary = box wick extreme + 2 pts. **Sensitivity row added:** fixed 10-pt stop from entry |
| Q3, Q4, Q5 | approved |
| Q6 | live panel shows the engine's own simulated trades for today, labelled "engine's trades, not yours" |
| Q7 | closes from the flow bars |
| C1 | overlay colour = **teal**; approved C1 deviation, logged in the rulings change log |
| Set-2 contamination | approved: per-trade rows and monthly splits withheld for set-2 sessions until set 2 is reviewed; totals with and without them |
| **New item 0** (before item 1) | load set 2 as `set-2:cp-calls`: the CR-CP calls not already in `cr-cz:cp-ranges` (train, ≤ 06-05), built the same way as CR-CZ (same meta and outcome fields; outcomes stored, not shown). Report the count. The "set 2 in progress" block and item 5 use this set name; item 5 runs on set 2 for real, counts only |

Pre-declared sensitivity rows (report-only) are now: box min 10; no 12:15 cutoff; no loss cap; D4 entry; fixed
10-pt stop.

**Rule 6 check (edge hold vs box min).** No conflict. "Box formed" is `boxVerdict`'s `formed` = the box readout's
minutes (first bar → last bar of the current box) ≥ `minAgeMin` and height by closes ≥ 2 pts; it never reads the edge
hold. The 15-min hold (`edgeReadout` `holdMin`) only decides whether a close > 2 pts beyond an edge *grows* the box
(edge set < 15 min ago) or starts a pending break (edge held ≥ 15 min). So a box qualifies at 8 min with whatever edges
it has grown to; an engine unit test asserts it (a box first readable as formed at minute 8).

Rulings conformance re-check: C1 changes from "deviates — STOP" to **deviates — approved by Ryan (teal)**. Item 0
writes `setup_candidates` only, insert-only, under the backfill protocol; B4 honoured (train ≤ 06-05, outcomes stored,
never shown or read), B8 honoured (no reviews written). Every other row unchanged.

## Implementation and smoke (2026-10-31)

| Commit | Item |
|---|---|
| `0f67e93` | 0 — `set-2:cp-calls` loaded: 115 cards (235 CR-CP calls − CR-CZ's 120), short 48 / long 67, 72 sessions 01-07 → 06-05, run `fcd42d44`; outcomes stored, not shown |
| `4ff652a` | 1 — `react_price_preview/src/v1Engine.js` (`evaluateAt`, `simulateSession`), boxVerdict `v1` option, cells bars o / buy / sell / vol, marks `merges`, prior-sessions baselines, `modules/Ironbeam/v1_engine.py` inputs |
| `2f2d629` | 2 — panel verdict from the engine + failed rules, engine's trades line, ⚙ box min 8 (+ one-time migration of a stored 10), "net buying / net selling", optional alert; inputs route |
| `6c6abf7` | 3 — `scripts/cr_du_backtest_v1.py` |
| `6696e3b` | 4 — contradiction-bar comparison and ⚙ markers (the 4b toggle markup also landed here) |
| `0ce58bd` | 4b — v1 trades overlay (teal), hover, blocked dates; Python blocking test |
| `2663912` | 5 — `setup_engine_verdicts` + `scripts/cr_du_set2_verdicts.py`; run `709a9b94` |
| `d9701f7` | smoke fixes + `tests/e2e/cr-du.spec.js` |

Design notes (no rule changed): the panel runs the engine at the chart's anchor (clamped to 06:30, rule 2) with the
engine's own trades since 06:30 as state; overlay, backtest and item 5 use the 06:30 anchor. A placed order keeps its
price and stop; it is cancelled at the first close whose verdict is not the same side for that box. Without 1-s data
(live panel inputs carry them; a missing second) a same-minute stop + target counts the stop, and a target in the fill
minute is not counted. On blocked dates the panel keeps the CR-DS verdict and says "v1 engine off: …".

Results (train, in-sample): see the session note (vault `sessions/2026-10-31 - CR-DU — v1 Rules Engine.md`) and
`scripts/logs/cr-du/backtest_report.md`. Primary: 90 trades on 86 sessions, 27.8 % wins, −69.5 pts gross / −92.0 after
0.25 per round trip; fixed 10-pt stop: 80 trades, 50.0 %, −23.25 after cost. Contradiction bars: inconclusive (n = 6
with a primary bar). Set 2: SHORT 0 · LONG 0 · NO ACTION 115 (27 warm-up). Smoke 4 / 4.

## Amendment 2 — Ryan, chat 2026-10-31 (after the first backtest)

- **Rule 15 primary stop = a fixed 10 pts from entry.** The box wick extreme + 2 pts (Amendment 1's primary) becomes a
  pre-declared sensitivity row. Pre-declared rows are now: box min 10; no 12:15 cutoff; no loss cap; D4 entry; wick + 2 stop.
- The backtest totals are rerun only after diagnosis D1 is reported (below). The code default changes with that rerun
  (not during the running e2e comparison, whose branch frontend serves `src/`).
- **Diagnosis D1** (read-only, no outcomes, no tuning): for each CR-CZ card by Ryan's first pre-reveal label, (a) the
  engine's box at the card's decision minute vs the card's box (overlap in pts and minutes); (b) if they overlap, the
  first and every failed v1 rule; (c) the same with the engine anchored at the card's range start. Counts by first
  failed rule for (b) and (c).

Rulings re-check: A4 still honoured (a fixed 10-pt stop is beyond the box on every trade where the box is < 10 pts from
entry; reported); A11 unchanged (+10). No other row changes.

### D1 result (read-only; `scripts/cr_du_d1/`, logs `scripts/logs/cr-du/d1/`; no outcomes)

Labels (first pre-reveal): take 44, skip 75 (Ryan's "29 takes" = the takes whose limit filled; telling them apart needs
the stored fill status, an outcome field, so all 44 are reported). Warm-up sessions (before 02-03, 13 takes / 17 skips):
box geometry only. Rules on the card's side are listed even without a read (`forceSide`, a diagnostic option in a
scratch copy of `v1Engine.js`; to be committed to `src/` after the e2e comparison stops serving it).

- (a) The engine has a formed box overlapping the card's on 42 / 44 takes and 73 / 75 skips; its box usually starts
  earlier (median 21 min before the card's range start for takes, 25 for skips) and is longer (median 35 vs 14 min) and
  taller (median +4 pts).
- (b) Own box, overlapping, rules evaluated (31 takes / 57 skips): first failed = no read 23 / 45, conflict 2 / 3,
  FOMC 2 / 0, one per box 2 / 1, other 2 / 8. "No read" is mostly "nothing notable" (16 / 39): Δ share below p80.
- (c) Anchored at the card's range start (31 / 58): first failed = no read 22 / 48, box forming 1 / 5, gates 4 / 4;
  verdict SHORT / LONG on 4 takes and 1 skip. Engine box = card box (± 0.25) on 7 / 31 and 11 / 58.
- Binding leg: Δ share ≥ p80 with the card's side on 10 / 31 takes (own box) and 13 / 31 (card anchor); the stall leg
  passes on 30 / 31 and 26 / 31.

### D2 item 5 — absorption at the faded edge (read-only; `scripts/cr_du_d1/d2_absorption.py`; no outcomes)

Card's side only (short: buyers absorbed; long: sellers absorbed); all exact marks of the session (not only the top 8);
within 2 pts of the card box's faded edge (by closes); earlier than and confirmed by the decision bar's close;
**unbroken by closes** = no 1-min close more than 1 tick beyond the mark's price between the mark and the decision bar
(definition fixed here; D2 items 1–4 were not received in this session).

| | Takes (44) | Skips (75) |
|---|---|---|
| ≥ 1 such mark | 5 (11 %) | 3 (4 %) |
| nearest mark: distance to the edge, median | 0.5 pts | 0.75 pts |
| nearest mark: minutes before the decision, median (p25–p75) | 12 (6–29) | 8 (5–16) |
| marks inside the card's box window / before it | 3 / 2 | 4 / 0 |
| failed the lean (D1 (b)) / of those with a mark | 21 / 2 (1 inside the window) | 49 / 1 (1 inside) |

Attrition (all 119 cards): marks confirmed before the decision median 3 per card; 190 on the card's side, 39 within 2 pts
of the edge (57 within 4), 9 of those unbroken by closes. The close test removes 30 of 39: price usually closes beyond an
edge-mark later inside the box (a close up to 2 pts beyond an edge is a box "test", not a break).

### D2 items 1–5 (read-only; `scripts/cr_du_d1/d2.mjs`, `d2.py`; evaluated sessions 31 takes / 58 skips; no outcomes)

Item 1, code: absorption alone is a trigger — `flowV3.js:boxVerdict` `shortWhy = lean > 0 || absTop.length > 0` (l. 1285; long mirror l. 1287), fired by `if (nearTop && wouldShort)` (l. 1325) with the stall leg (l. 1289); `absTop` = buyers absorbed within 2 pts of the box high (l. 1282, `boxLevels`). The engine passes only levels that started inside its box (`v1Engine.js` l. 204, rule 5), from the chart top 8 (`absorptionLevels`, `LEVELS_MAX`). So absorption alone fires only if it is in the top 8, starts inside the engine box (which usually starts ~20 min before the card range), sits at the faded edge, and price is within 2 pts of the edge, with no conflict / two-sided.

Caveat: CR-CP called each card side from the box delta share, so 2(b) and 3 point to the card side by construction; their magnitude is the signal.

## 1. Absorption alone (lean failed, engine's own box)

| Label | Cards | Lean failed | …with card-side absorption the verdict saw (in the engine box) | …at the faded edge | First failed rule on those | …with a card-side mark at the edge in the card's window (any of the session's marks) | …of those, stall leg passes |
|---|---|---|---|---|---|---|---|
| take | 31 | 21 | 1 | 1 | nothing notable: long read, but price not at the bottom (1) | 1 | 1 |
| skip | 58 | 49 | 4 | 3 | two-sided absorption (2); conflicting evidence: buyers leaning vs sellers absorbed at the bottom (1) | 3 | 3 |

## 2. |Δ share| percentile at the decision minute (p25 / p50 / p75)

| Label | Measured over | Percentile | Signed to the card side | ≥ p80 on the card side |
|---|---|---|---|---|
| take | (a) engine's box | 37.0 / 70.7 / 90.7 (n 31) | -15.4 / 57.9 / 84.3 (n 31) | 10 / 31 |
| take | (b) card's box | 58.1 / 72.4 / 89.4 (n 31) | 58.1 / 72.4 / 89.4 (n 31) | 13 / 31 |
| take | (c) last 10 closed minutes | 29.9 / 54.7 / 77.5 (n 31) | 29.9 / 54.7 / 77.5 (n 31) | 5 / 31 |
| skip | (a) engine's box | 30.7 / 49.3 / 73.8 (n 58) | -20.3 / 33.7 / 57.6 (n 58) | 9 / 58 |
| skip | (b) card's box | 47.1 / 57.0 / 73.2 (n 58) | 45.3 / 57.0 / 73.2 (n 58) | 7 / 58 |
| skip | (c) last 10 closed minutes | 19.8 / 46.5 / 71.1 (n 58) | 13.0 / 46.5 / 71.1 (n 58) | 10 / 58 |

## 3. CVD inside the card's box (OLS of cumulative delta on minutes; t signed to the card side)

| Label | Cards | Slope on the card side (t > 0) | t p25 / p50 / p75 | t ≥ 2 | t ≤ −2 |
|---|---|---|---|---|---|
| take | 31 | 30 | 3.93 / 5.70 / 8.28 (n 31) | 29 | 0 |
| skip | 58 | 54 | 2.86 / 4.82 / 6.99 (n 58) | 47 | 2 |

## 4. Card-side absorption inside the card's box window (mark minute ≥ range start, confirmed by the decision)

| Label | Cards | Any price | Within the box high–low ± 2 | At the faded edge (≤ 2 pts) | Among the chart's top 8, unbroken |
|---|---|---|---|---|---|
| take | 31 | 5 | 5 | 5 | 3 |
| skip | 58 | 4 | 4 | 4 | 1 |

## 5. Absorption at the faded edge (≤ 2 pts, any earlier mark), unbroken by closes: two thresholds

| Label | Unbroken = | Cards with ≥ 1 | Nearest distance (median) | Minutes before (median) | Marks in window / before |
|---|---|---|---|---|---|
| take | 1 tick (D2.5 first run) | 4 / 31 | 0.38 | 9 | 3 / 1 |
| take | 2 pts (box break threshold) | 6 / 31 | 0.38 | 6 | 6 / 2 |
| skip | 1 tick (D2.5 first run) | 2 / 58 | 0.62 | 10 | 3 / 0 |
| skip | 2 pts (box break threshold) | 6 / 58 | 0.75 | 15 | 5 / 2 |

## Amendment 3 — Ryan, chat 2026-10-31

- **Rules 7 / 8 lean cut = Δ share ≥ p70 on the card's side** (was p80), measured over the box as now. p80 becomes a
  sensitivity row. Basis: D2's takes-vs-skips Δ share distributions (takes median ≈ p72 over the engine's and the
  card's box, skips ≈ p49 / p57), not P&L. Only the lean leg changes: the effort leg (c/pt) and the panel's percentile
  tiers stay at p80.
- Backtest rerun once: **primary = p70 lean + fixed 10-pt stop** (Amendment 2). Sensitivity rows: p80 lean; wick + 2
  stop; D4 fill; box min 10. (The earlier no-cutoff / no-loss-cap rows are not rerun.) Set-2 withholding unchanged.
- Engine version v1.0 → **v1.1** (lean p70, fixed-10 stop). `setup_engine_verdicts` holds v1.0 rows for set 2; a v1.1 run
  is not part of this amendment.

Rulings re-check: B2 — a threshold chosen from the learning set's label distributions, disclosed here, not from
outcomes; the train backtest stays "in-sample" and the holdout run will be pre-registered. No other row changes.

### Rerun after Amendments 2 / 3 (engine v1.1; train, in-sample; set-2 per-trade rows and monthly splits withheld)

## Totals (train, in-sample)

| Variant | Sessions | Trades | Trades / day | Win % | Net pts gross | Net pts after 0.25 / RT | Avg win | Avg loss | Max DD | Longest losing streak | Exits |
|---|---|---|---|---|---|---|---|---|---|---|---|
| primary | all sessions | 93 | 1.08 | 51.6 | 33.25 | 10.0 | 9.66 | -9.56 | 79.0 | 7 | flatten 6, stop 42, target 45 |
| primary | without set-2 sessions | 30 | 1.0 | 40.0 | -57.0 | -64.5 | 9.38 | -9.42 | 124.25 | 7 | flatten 3, stop 16, target 11 |
| p80 lean | all sessions | 80 | 0.93 | 50.0 | -3.25 | -23.25 | 9.59 | -9.67 | 73.25 | 5 | flatten 5, stop 38, target 37 |
| p80 lean | without set-2 sessions | 24 | 0.8 | 29.2 | -103.5 | -109.5 | 8.93 | -9.76 | 141.0 | 8 | flatten 2, stop 16, target 6 |
| wick + 2 stop | all sessions | 104 | 1.21 | 25.0 | -121.25 | -147.25 | 9.86 | -4.84 | 176.5 | 9 | flatten 1, stop 78, target 25 |
| wick + 2 stop | without set-2 sessions | 30 | 1.0 | 20.0 | -52.25 | -59.75 | 10.0 | -4.68 | 90.75 | 12 | stop 24, target 6 |
| D4 fill (1 pt inside) | all sessions | 105 | 1.22 | 49.5 | -4.25 | -30.5 | 9.62 | -9.52 | 89.75 | 4 | flatten 7, stop 49, target 49 |
| D4 fill (1 pt inside) | without set-2 sessions | 33 | 1.1 | 39.4 | -70.0 | -78.25 | 9.35 | -9.57 | 128.25 | 8 | flatten 3, stop 18, target 12 |
| box min 10 | all sessions | 89 | 1.03 | 49.4 | -6.75 | -29.0 | 9.62 | -9.56 | 101.5 | 7 | flatten 6, stop 42, target 41 |
| box min 10 | without set-2 sessions | 28 | 0.93 | 35.7 | -77.0 | -84.0 | 9.25 | -9.42 | 134.0 | 8 | flatten 3, stop 16, target 9 |

### By side (primary, non-set-2 sessions, train, in-sample)

| Side | Trades | Win % | Net gross | Net after cost |
|---|---|---|---|---|
| long | 9 | 66.7 | 30.0 | 27.75 |
| short | 21 | 28.6 | -87.0 | -92.25 |

### By hour (PT, entry) (primary, non-set-2 sessions, train, in-sample)

| Hour (pt, entry) | Trades | Win % | Net gross | Net after cost |
|---|---|---|---|---|
| 07:00 | 3 | 0.0 | -30.0 | -30.75 |
| 08:00 | 6 | 50.0 | 0.0 | -1.5 |
| 09:00 | 7 | 71.4 | 30.0 | 28.25 |
| 10:00 | 8 | 37.5 | -20.0 | -22.0 |
| 11:00 | 5 | 20.0 | -27.0 | -28.25 |
| 12:00 | 1 | 0.0 | -10.0 | -10.25 |

### By month (primary, non-set-2 sessions, train, in-sample)

| Month | Trades | Win % | Net gross | Net after cost |
|---|---|---|---|---|
| 2026-02 | 12 | 41.7 | -13.5 | -16.5 |
| 2026-03 | 7 | 28.6 | -30.0 | -31.75 |
| 2026-04 | 3 | 0.0 | -26.0 | -26.75 |
| 2026-05 | 8 | 62.5 | 12.5 | 10.5 |

### Readouts (primary, non-set-2 sessions, train, in-sample)

- Stop distance (rule 15) pts: p10 10.00 · p50 10.00 · p90 10.00 · max 10.00
- MFE (from fill to exit, minute bars) pts: p50 7.38 · p90 10.83

- Stop-free ±10 race of the same entries (A15 readout): {'win': 11, 'loss': 16, 'unresolved': 3}

## Contradiction bars in the box at the faded edge (item 4; primary engine trades, train, in-sample)

| Sessions | Definition | n with | Win % with | Net with (after cost) | n none | Win % none | Net none (after cost) | Win-rate diff, pts (90 % bootstrap CI) | B1 |
|---|---|---|---|---|---|---|---|---|---|
| all sessions | primary (p10 / p90 delta, wick ≥ 50 %, vol ≥ p80, range ≥ 2) | 8 | 50.0 | -2.0 | 85 | 51.8 | 12.0 | -1.8 (-32.2, +27.9) | inconclusive |
| all sessions | secondary (p20 / p80 delta, wick ≥ 40 %, no volume floor) | 40 | 47.5 | -28.75 | 53 | 54.7 | 38.75 | -7.2 (-24.2, +10.3) | inconclusive |
| without set-2 sessions | primary (p10 / p90 delta, wick ≥ 50 %, vol ≥ p80, range ≥ 2) | 3 | 33.3 | -10.75 | 27 | 40.7 | -53.75 | -7.4 (-51.9, +40.7) | inconclusive |
| without set-2 sessions | secondary (p20 / p80 delta, wick ≥ 40 %, no volume floor) | 10 | 30.0 | -36.0 | 20 | 45.0 | -28.5 | -15.0 (-45.0, +15.0) | inconclusive |

Small n: a difference this size can't be separated from chance here; the secondary definition is report-only. Bars are counted from the box start through the decision minute, on the edge being faded (bar low / high within 2 pts of the box's wick low / high); percentiles per 30-min RTH bucket from prior sessions (Databento).

## Overlap with CR-CZ cards (verdict at the card's decision minute; counts only, no outcomes)

| Ryan's label | Cards | Engine takes (same side) | Engine: other side | Engine: NO ACTION | No bar | Warm-up session (not evaluated) |
|---|---|---|---|---|---|---|
| skip | 75 | 2 | 0 | 56 | 0 | 17 |
| take | 44 | 0 | 0 | 31 | 0 | 13 |

First failed rule on those NO ACTION cards: no_read 66, conflict 6, one_per_box 5, fomc 3, box_forming 2, cutoff 2, absorption_in_path 1, two_sided 1, no_larger_move 1

D1 (b) recomputed at p70 (engine's own box, overlapping, rules evaluated): takes 31 — fires 0; first failed no read 22
(nothing notable 12, read but price not at the edge 10), one per box 4, FOMC 2, conflict 2, box forming 1. Skips 57 —
fires 2 (both the card's side); no read 44 (nothing notable 34, not at the edge 10), conflict 4, cutoff 2, other 5.
(c) anchored at the card's range start: takes fire 4 (3 on the card's side), skips 3 (2). At p80 (v1.0) "nothing
notable" was 16 takes / 39 skips; at p70 the lean passes on more cards but the read then fails on location — price is
not within 2 pts of the faded edge at the card's decision minute (CR-CP called the side at the qualifying bar and
rested a limit at the edge; the engine reads at that minute).

## Amendment 4 — Ryan, chat 2026-10-31 (last before freeze)

- **Location moves from the verdict to the order.** A valid read = (lean ≥ p70 on the side **or** in-box absorption on
  the side at the faded edge) + the stall leg + every gate (rules 1–6, 9, 19, 20, rule 4's path). It places a resting
  limit at the faded edge by closes (box top for a short, bottom for a long), re-pegged to the current edge while the read
  holds; cancelled when the read lapses, the box breaks, or at 12:15 PT. A fill happens only at the edge, on a 1-tick
  trade-through. Stop = fixed 10 pts, target +10, from the fill price.
- Panel: "READ — limit at <price>" while a read is valid and price isn't within 2 pts of the edge; "SHORT at top" /
  "LONG at bottom" when it is. The Location row becomes informational (no longer red).
- **Engine tagged v1.2 and frozen**: no further changes on train. Next: backtest once (primary + p80 lean, wick + 2 stop,
  D4 fill, box min 10), D1 (b) at v1.2, item 5 re-run (v1.2 rows), CR-DU smoke on v1.2, PR (not merged).

Rulings re-check: A3 / A20 honoured (the entry is a limit at the edge after the read, never a touch-bar close). No other
row changes.

### Results at v1.2 (frozen; tag `cr-du-engine-v1.2`; train, in-sample; set-2 per-trade rows and monthly splits withheld)

| Variant | Sessions | Trades | / day | Win % | Net gross | Net after 0.25 / RT | Avg win | Avg loss | Max DD | Losing streak |
|---|---|---|---|---|---|---|---|---|---|---|
| **primary** | all | 102 | 1.19 | 50.0 | +3.25 | −22.25 | 9.68 | −9.61 | 83.5 | 5 |
| primary | without set-2 | 34 | 1.13 | 41.2 | −57.0 | −65.5 | 9.46 | −9.47 | 125.25 | 5 |
| p80 lean | all | 90 | 1.05 | 48.9 | −23.25 | −45.75 | 9.62 | −9.71 | 85.0 | 6 |
| p80 lean | without set-2 | 28 | 0.93 | 32.1 | −103.5 | −110.5 | 9.17 | −9.79 | 142.0 | 9 |
| wick + 2 stop | all | 113 | 1.31 | 23.9 | −112.5 | −140.75 | 9.86 | −4.40 | 170.0 | 15 |
| wick + 2 stop | without set-2 | 34 | 1.13 | 20.6 | −51.5 | −60.0 | 10.0 | −4.50 | 91.0 | 13 |
| D4 fill (1 pt inside) | all | 112 | 1.30 | 51.8 | +45.75 | +17.75 | 9.66 | −9.53 | 72.0 | 4 |
| D4 fill (1 pt inside) | without set-2 | 36 | 1.20 | 44.4 | −40.0 | −49.0 | 9.47 | −9.57 | 99.0 | 5 |
| box min 10 | all | 98 | 1.14 | 48.0 | −36.75 | −61.25 | 9.65 | −9.61 | 112.75 | 6 |
| box min 10 | without set-2 | 32 | 1.07 | 37.5 | −77.0 | −85.0 | 9.38 | −9.47 | 135.0 | 5 |

Primary exits (all): target 48, stop 48, flatten 6. Non-set-2 splits: long 9 (66.7 %, +27.75 net), short 25 (32.0 %,
−93.25); by hour 08:00 9 (56 %, +7.75), 09:00 7 (71 %, +28.25), 10:00 9 (33 %, −32.25), 07:00 / 11:00 / 12:00 negative;
by month Feb 12 (42 %), Mar 10 (40 %), Apr 3 (0 %), May 9 (56 %). MFE p50 7.4; stop-free ±10 race of the same entries
13 / 18 / 3. Contradiction bars: all four rows inconclusive (9 trades with a primary bar). CR-CZ overlap at the decision
minute: takes 1 same side / 30 NO ACTION (13 warm-up); skips 3 same side, 2 other side, 53 NO ACTION (17 warm-up).

D1 (b) at v1.2 (engine's own box): takes 31 — fires 1 (card side); first failed: no read 12 (all "nothing notable"),
one per box 5, no larger move 5, cutoff 2, FOMC 2, conflict 2, re-entry wait 1, box forming 1. Skips 57 — fires 5 (3
card side); no read 34, cutoff 4, no larger move 4, conflict 4, one per box 3, other 3. Anchored at the card's range
start (c): takes fire 8 (7 card side), skips 5 (3).

Item 5 at v1.2 (`setup_engine_verdicts`, run `a5d15389`, counts only): set 2 — SHORT 1 · LONG 6 · NO ACTION 108 (27
warm-up, not evaluated). The v1.0 rows stay alongside (engine_version is part of the key).

## Smoke + wrap (v1.2)

- **CR-DU smoke (`tests/e2e/cr-du.spec.js`) on v1.2:** 4 / 4 passed (10-30 box min 8 vs 10; the overlay on 2026-02-03 equals the backtest; the 10-30 overlay; holdout and set-2 dates blocked). Unit 142 / 142.
- **Existing specs, branch vs origin/main** (same backend :8068, branch frontend :5197, origin/main frontend :5198 from a worktree): origin/main 87 passed · 16 failed · 4 timed out · 19 skipped. Every failure on the branch except the four below also fails on origin/main, or doesn't run here.
- **The four branch-only failures are intended CR-DU changes.** The assertions were updated, and all four pass on v1.2:
  - `cr-dp` items 1–3: the Display group now ends "· v1 trades off" (item 4b toggle).
  - `cr-dq` item 7 (10-28 FOMC): still NO TRADES. The pill title is the v1 engine's, not "Rules: A22 … Draft rules, untested".
  - `cr-ds` smoke 1: passes unchanged on v1.2 (its failure was at v1.0 / v1.1).
  - `cr-ds` smoke 2: still SHORT at 07:49 when anchored at 06:30. The absorption row is now neutral, not green. Rule 5: the 07:05 buyers-absorbed mark at 7660.25 starts before the 07:38 box, so it counts as location support, not a trigger.
- **Timeout, not a CR-DU change:** `cr-dq` item 7 verdict scan timed out once waiting for the chart to load. It passed on rerun.

## Merge of origin/main (CR-DV #128, CR-DW #130) — Ryan, chat, 2026-11-01

Merged, not rebased; tag `cr-du-engine-v1.2` stays on its commit.

- **Kept from CR-DV:** closed minutes only, provisional hollow marks, feed-stall flag, stored + trades splice.
- **Kept from CR-DU:** per-minute cut of marks (`merges`). A merge is recorded only once it confirms.
- **Engine input = confirmed marks only, as of minute t.** `prepareSession` drops provisional marks and `absorptionLevels` skips them; `chartMarks` (display only) is the only other reader and the engine doesn't call it. Unit test: verdict, per-minute evaluation and simulation are identical with and without provisional marks, even with `confirm_time` ≤ t.
- **Visible change (approved):** "engine v1.2 — untested" beside the verdict pill, in the panel's muted secondary text.

**Engine outputs unchanged.** All 86 train sessions were rebuilt with the merged code: inputs are identical except the new `provisional: false` on each mark, `backtest_results.json` is identical, and the report differs only in its build timestamp.

**Reruns on the merged code:**
- Unit 147 / 147 (JS). Python: `test_flow_heat` + `test_flow_cells` 48 / 48 before the added `merges` assertion, then `test_flow_heat` 24 / 24 with it.
- CR-DU smoke 4 / 4, including overlay = backtest on 2026-02-03.
- CR-DV 10-30 replay (15 s, 1,580 polls, `scripts/logs/cr-dv/replay_after_crdu_merge_15s.txt`): same as CR-DV's final replay except runtime. 0 confirmed marks that aren't final, 0 confirmed before their hold ended, minute close → provisional 15 s, → confirmed 195 s, 0 polls missing a closed minute.
- CR-DV simulated-now API, 8 of 8: 7633.50 absent → hollow → hollow → gone (never solid); 7631.75 hollow → solid; 7628.25 hollow → gone.
- Live provisional marks vs the engine: at 15:40:15, 15:40:45, 15:44:15 and 15:47:15 the panel verdict, every minute's verdict and the trades are identical with those marks stripped. Browser at 15:44:15, anchor session open: "NO ACTION", rule 7, absorption row "none at the box" with the hollow 7631.75 at the box bottom; no console errors.
- Not rerun: the full existing-spec comparison against origin/main.

## Open: "Box formed" height (Ryan, chat, 2026-11-01) — not fixed, waiting on Ryan

The row shows `box.top − box.bottom` (edge height; an edge moves only on a close > 2 pts beyond it), so it can read 0.00 pts while closes span more. Rule 6's `formed` in engine v1.2 gates on the same value, so max close − min close is an engine change. Train counts only (86 sessions, no outcomes): the heights differ on 26,058 of 32,852 minutes; on boxes old enough to count, the height leg would go from failing to passing on 333 minutes in 49 sessions, never the other way.

## Amendment 5 (Ryan, chat, 2026-11-01): engine v1.3, a bug fix, not tuning

Implementation brought into line with A1.1 / v1 rule 6. Found from Ryan's screenshot, 2026-04-15, box 09:18–09:29: "Box formed 11 min · 0.00 pts" while closes moved. No P&L was read to make this change.

1. **While a box is forming** (before it qualifies at box min and 2 pts), its edges = the running max / min 1-min close from the box start (CR-CP A1.1).
2. **`EDGE_BREAK_PTS` (2 pts) applies only after the box qualifies.** Edges freeze at qualification; a close ≤ 2 pts beyond an edge is a test, > 2 pts is a break.
3. **Height** = top − bottom of those edges = max close − min close while forming.
4. Panel and engine share the code (`edgeReadout`). Tag `cr-du-engine-v1.3`.

**Readings fixed for implementation (reported to Ryan):**
- **15-min edge hold (`holdMin`, CR-DI):** read literally, it no longer gates anything. Frozen edges can't grow, and "> 2 pts is a break" leaves no state in which a qualified box ignores a > 2-pt close. The ⚙ control is kept but has no effect on the box chain.
- **5-min break hold (`breakHoldMin`, CR-DN):** unchanged. A > 2-pt close starts a pending break; a close back strictly inside within 5 bars is a probe and the box stands.
- **A forming box cannot break.** A box under 2 pts tall stays forming however far a close moves, and that close becomes its edge.
- **Panel ⚙ rules:** the panel qualifies boxes at its own box min and height, as the engine does with its settings (the "box min 10" row qualifies at 10).

**Real bars, 2026-04-15, anchor 09:18, set end 09:29:** closes 7037.75, 7037.75, 7039.50, 7039.50, 7039.25, 7039.75, 7039.75, 7037.25, 7038.00, 7038.50, 7037.50, 7035.75. True max − min = **4.00** (the 3.00 quoted in chat came from a synthetic reproduction). By this amendment the box qualifies by 09:26 with edges 7039.75 / 7037.25, height **2.50**; the 09:29 close is a 1.50-pt test below the bottom, so it doesn't widen the box. **The row shows 2.50, not 4.00**: flagged to Ryan. Display only: once qualified, a box stays formed either way.

Rulings re-check: A2 honoured; CR-DI's C2 box-restart approval changes as above (recorded in the change log); every other row unchanged. Nothing in "Revoked / superseded" is used.

## v1.3 results (train, in-sample)

Rerun in Ryan's order: D1, backtest once, set-2 verdicts, then tests.

**(a) D1, copying score (no P&L): cards where the engine fires**

| | v1.2 takes | v1.2 skips | v1.3 takes | v1.3 skips |
|---|---|---|---|---|
| (b) engine's own box (06:30 anchor) | 1 / 31 | 5 / 57 | 1 / 31 | 7 / 58 |
| (c) anchored at the card's range start | 8 / 31 | 5 / 58 | 9 / 31 | 5 / 58 |

(c) box vs card: engine box = card box (±0.25 pts both edges) on 9 / 31 takes.

**(b) Backtest (primary + the same rows). Set-2 per-trade rows and monthly splits withheld.**

| Variant | All: trades | Win % | Net after cost | Without set-2: trades | Win % | Net after cost |
|---|---|---|---|---|---|---|
| **Primary** | 190 | 53.2 | **+57.0** | 57 | 54.4 | **+39.5** |
| p80 lean | 149 | 55.0 | +94.25 | 39 | 51.3 | −11.0 |
| Wick + 2 stop | 194 | 35.1 | −49.75 | 58 | 41.4 | +25.25 |
| D4 fill (1 pt inside) | 195 | 51.8 | +24.0 | 58 | 51.7 | +11.0 |
| Box min 10 | 157 | 50.3 | −40.5 | 47 | 44.7 | −57.5 |

v1.2 primary: 102 trades, 50.0 %, −22.25 (without set-2: 34, 41.2 %, −65.5). Primary max drawdown 146 pts. Without set-2 by month: February +93.25, March −26.0, April −21.5, May −6.25. Side: long 23, +0.5; short 34, +39.0. Contradiction bars still inconclusive.

Not a tested edge: the sign changes with box min (+57.0 at 8, −40.5 at 10), one month carries the non-set-2 total, and this is in-sample. The change wasn't chosen on P&L; it came from Ryan's screenshot.

**(c) Set 2 at v1.3 (run `0d6cd6f5`):** SHORT 3 · LONG 2 · NO ACTION 110 (27 warm-up, not evaluated). The v1.0 and v1.2 rows remain beside them.

**(d) Tests:** unit 151 / 151; `test_flow_heat` OK; CR-DU smoke 5 / 5, including overlay = backtest on 2026-02-03 and 2026-04-15 reading "11 min · 2.50 pts (min 8 · 2)". The full existing-spec comparison and the other panel specs (cr-dq, cr-ds, cr-dr, cr-dp) were not rerun; their boxes can move under v1.3.

**Open:**
- 2026-04-15 shows 2.50 (frozen edges) while max − min close over the box is 4.00. Ryan asked to confirm v1.3 shows the true max − min; by Amendment 5's freeze it doesn't. Display only.
- The ⚙ 15-min edge hold control no longer has any effect on the box chain.
- Display bug, not fixed: in the engine panel the absorption row prints a raw epoch ("buyers absorbed 7040.00 (1776270180)") because `evaluateAt` calls `boxVerdict` without `hhmm`.
