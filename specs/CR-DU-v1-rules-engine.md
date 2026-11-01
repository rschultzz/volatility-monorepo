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
