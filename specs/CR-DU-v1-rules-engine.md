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
