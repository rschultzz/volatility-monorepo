# CR-DY — selectable boxes: drawn box + engine trade boxes, one stats panel

Branch `feat/CR-DY-selectable-boxes` off `origin/main` `dcbcad2` (CR-DU #131 merged), worktree `../volatility-monorepo-cr-dy`.
UI CR. Session note: vault `sessions/2026-11-01 - CR-DY — Selectable Boxes.md`.

## Kickoff (Ryan, chat 2026-11-01, verbatim)

```
1. git fetch && git checkout -b feat/CR-DY-selectable-boxes origin/main (tip after CR-DX merge), worktree ../volatility-monorepo-cr-dy. First action.
2. Read CLAUDE.md, Dash/components/field-conventions.md (incl. the CR-DU one-minute alignment rows), Dash/research/standing-rulings.md (C1, C2), specs/CR-DU-v1-rules-engine.md (Amendments 1–7, follow-on items 2–7).
3. UI CR. Engine logic frozen: v1Engine.js outputs must stay identical to tag cr-du-engine-v1.4 (rerun the 86-session / 33,540-minute check). Panel stat computations unchanged; only which box they're fed changes.
4. Spec freeze specs/CR-DY-selectable-boxes.md, session note in the vault. Specs by file path only.

Problem (Ryan, chat 2026-11-01, on 10-30)
- The panel's engine line reads the engine's chained box at the panel end (last bar / set end), e.g. "box 12:59–12:59 · rule 6", not the box that produced each engine trade. Confusing.
- With only an anchor set, the manual box runs anchor → session end (e.g. "310 min").
- Boxes aren't drawn as selectable objects; stats exist only for the one manual box.

Goal
1. Drawn box: arm "draw box", click start bar, click end bar (chart only). Top/bottom = high/low of the bars in [start, end], same definition the engine uses. Drawn as a rectangle on the price chart. Replace anchor/set-end as the primary manual input (keep the existing quick picks — session open, last swing — as ways to set the start).
2. Engine trade boxes: with "v1 trades on", each engine trade draws the engine's box as it stood at that trade's decision minute (frozen at that minute; start, end, top, bottom from the engine's own state, not recomputed). Box ↔ trade mark visibly linked (same colour family / hover shows both).
3. One stats panel: clicking any box (drawn or engine) selects it and loads the existing Box panel rows (formed, location, Δ share, effort, progress, absorption, edge counts, verdict) for that box over its own window. For an engine box, the panel also shows the trade (side, entry, stop, TP, outcome) and the engine's verdict at that minute.
4. Highlighting: all boxes light fill; selected box darker fill + solid border; hover slightly darker. Only one selected at a time. Escape / click empty chart clears selection. Panel title says which: "Your box 07:49–08:12" / "Engine box 07:34–07:49 · trade 2 (short)".
5. Remove the "engine line at panel end" readout from the panel (or relabel it explicitly as "Engine at <end minute>"); Step 0 recommends which.
6. Folded in: (a) clicks over the panel while armed are not swallowed — they're ignored without consuming the pick, or the panel is click-through while armed; (b) fix the stale "Restart on a break" help text (AggressorFlowV3Panel.jsx:163) to describe v1.4 / manual-box behaviour.
7. setup_reviews: the anchor_ts / box fields (7b) record the drawn box's start, end, top, bottom on review save. Additive migration if end/top/bottom aren't already stored.

Step 0 (read-only, commit; STOP only if an answer changes the plan)
Q1. Does v1Engine expose, per engine trade, the box (start, end, top, bottom) at the decision minute? If not, the smallest frozen-safe way to surface it (a returned field, no logic change).
Q2. Can the existing panel row functions take an arbitrary (start, end) window without anchor-chaining? List the call sites.
Q3. Interaction with set 2 / review cards (card replay cuts bars at entry), CR-DP y-zoom memory, CR-DX default fit, CR-CD review fit, CR-DV live absorption marks (live session: a drawn box whose end is the current minute keeps updating until you click an end bar — confirm or propose).
Q4. Recommendation for goal 5.

Visible change (C2, pre-approved by Ryan, chat 2026-11-01): new drawn/selectable box rectangles, selection highlight, engine trade boxes, panel title changes, removal/relabel of the panel-end engine line, help text. No change to engine logic, verdicts or trades.

Smoke + wrap
- 10-30: draw 07:34–07:49 → rectangle, panel rows for that window; v1 trades on → one box per engine trade at its decision minute; click each → panel switches, darker highlight moves; engine output identical to v1.4 tag.
- 2026-01-26 (set-2 session, engine blocked): drawn box works, engine boxes absent with the blocked reason shown once.
- Armed click over the panel doesn't swallow the pick.
- Review save writes the drawn box fields (rolled back).
- Live check deferred to first RTH session after deploy.
- Existing e2e specs as a diff vs origin/main by file path; screenshots in specs/assets/cr-dy/.
- Session note, PR, head SHA, "released :8068".
```

## Standing rulings conformance (`Dash/research/standing-rulings.md`, read 2026-11-01)

UI CR: the kickoff names C1 and C2; A and B rulings have no new use here.

| # | Mark | How |
|---|---|---|
| C1 | honoured | no new hue: drawn box keeps the existing magenta, engine boxes the teal approved for the v1 overlay (CR-DU, 2026-10-31); the panel's muted status colours unchanged |
| C2 | deviates — pre-approved in the kickoff | rectangles, selection highlight, engine trade boxes, panel title, engine line relabel, help text. Recorded in the rulings change log, not stopped for |
| B4 | honoured | engine boxes appear only where the overlay is allowed; holdout and set-2 dates stay blocked, so no trade or outcome is shown there |
| others | n/a | no test, threshold or outcome in this CR |

Revoked / superseded: none used.

## Step 0 — diagnosis (read-only; no server; code at `dcbcad2`)

Script `scripts/cr_dy/engine_parity.mjs` (counts and hashes only; session files `scripts/logs/cr-du/sessions/`, machine-local).

**Base.** The kickoff says "tip after CR-DX merge". CR-DX has no trace: no branch, PR, commit, spec or vault note, and no other session mentions it. The tip is `dcbcad2` (CR-DU #131). Branched there; re-check Q3 against CR-DX if it lands first.

### Q1 — engine box per trade: yes, no engine change

`simulateSession` already carries it. `order = { ...ev.order, decisionT: t, box: ev.box, … }` (`v1Engine.js:368`), kept through the re-peg (`:365`) and into the position and trade (`:348`, `:314`). So every trade and `sim.open` has `decisionT` and `box { from, top, bottom, high, low, minutes }`. End = `decisionT`. `v1Engine.js` is not edited.

Measured on the 86 train sessions, 33,540 minutes, 142 trades (counts only, set-2 withholding kept):

| Check | Result |
|---|---|
| Tag `cr-du-engine-v1.4` vs this checkout, hash of trades + minutes + state | equal (`1d4707278f8edea0`) |
| Trades with `decisionT` and a full box | 142 / 142 |
| Manual window (`box.from` → `decisionT`, engine box min / height) gives the engine's from, to, top, bottom, no break | 142 / 142 |
| `panelEngine` at `decisionT` = the trade's side, same box | 142 / 142 |
| Entry = the faded edge of the decision box | 142 / 142 |

(`1d47…` is this script's hash, not comparable with CR-DU's `6fcd03aa…`, which hashed other fields. The comparison that matters is the tag's source vs this checkout, run in one pass.)

### Q2 — panel rows on an arbitrary window: yes, no chaining

`edgeReadout(…, anchor, shift, end, hold, { manual: true })` already is that window (CR-DU amendment 7). Call sites that feed the rows, all taking (start, end):

| Function | Call site |
|---|---|
| `edgeReadout` manual | `App.jsx:1784` (`flowV3Edges`); `PriceChart.jsx:2814` through `readHistory` (`flowV3.js:1517`); `tradeContext.js` (`entryContext`) |
| `boxSegments` → `segmentReadouts` (Δ, volume, dPrice, c/pt) | `App.jsx:1792`–`1796` |
| `edgeBoxEntries`, `boxPanelOf`, `currentBoxVerdict` | `PriceChart.jsx:2787`, `:2822`, `:2841` |
| `panelEngine` (`endShifted`) | `PriceChart.jsx:2852` |

Because App's `flowV3Edges` is the one input, the plan is to feed it the selected box's window. Rows, read history, delta line and engine line then all follow with no new stat code, and rows and delta line can't read different windows (CR-DU bug 2).

### Q3 — interactions

| Item | Answer |
|---|---|
| Set 2 / review cards | Bars are cut at `entry_ts_pt` (`until`); an end pick is already limited to the last bar (`endFromPick`). Drawn box works there. The engine is blocked on set-2 dates, so no engine boxes. A review save records the drawn box only, never an engine box |
| CR-DP y-zoom memory | The new primitive has no `autoscaleInfo` and selection calls no scale API, so zoom memory and auto-fit are untouched |
| CR-DX default fit | Not in the repo (above); nothing to check |
| CR-CD review fit | Selection and drawing call no `setVisibleRange` / `fitContent`, so the fit is untouched |
| CR-DV live marks | Rows read the same levels as today. The engine reads confirmed marks only (`prepareSession`), so provisional marks can't move an engine box. **Live: confirmed as proposed.** A drawn box with no end runs to the last closed minute and updates on each poll until an end bar is clicked |

Proposal for a past session, from the "310 min" complaint: between the start click and the end click nothing is computed (no rectangle, no panel). Esc there cancels the draw. Live, Esc there keeps the running box.

### Q4 — goal 5: relabel, don't remove

Recommend **"Engine at 12:59: NO ACTION — rule 6: box still forming"**. The engine's box moves to the hover title, so "box 12:59–12:59" is gone from the line. Removing it would lose the live engine read (the panel is where CR-DU put it) and the "Engine's trades, not yours" state. For an engine box the same line reads the decision minute ("Engine at 07:49: SHORT").

### Reading to confirm: what "top / bottom" means (goal 1)

The kickoff's two clauses disagree. "High / low of the bars" is wicks; "same definition the engine uses" is closes, frozen once the box qualifies. On 2026-10-30 (`es_minutes`):

| Window | Wick high / low | Max / min close | Engine rule (frozen at) |
|---|---|---|---|
| 07:34–07:49 | 7659.75 / 7636.50 | 7657.00 / 7637.25 | 7654.25 / 7637.25 (07:42) |
| 07:49–08:12 | 7657.50 / 7641.75 | 7656.50 / 7642.50 | 7656.50 / 7648.50 (07:57) |

On the 142 engine trades the wick range is taller than the edges every time (median +4.5 pts, p90 +9.75).

**Reading used: the engine rule**, through `edgeReadout` manual. It keeps "panel stat computations unchanged" and "same definition", and the same window then gives the same edges on a drawn and an engine box (142 / 142 above). Consequence, visible: closes can pass the rectangle after it freezes (07:34–07:49: rectangle top 7654.25, a later close 7657.00). A held break is flagged; that one was not held inside the window. Wicks stay available as `high` / `low`. The rectangle and the saved fields read one function (`drawnBoxOf`), so changing this is one line. Not a STOP: the plan doesn't change, and it is for Ryan to confirm on the screenshots.

### Design fixed by Step 0

- **Selection** lives in App: `null`, `{ kind: 'drawn' }` or `{ kind: 'engine', … }`. An engine selection replaces the window fed to `flowV3Edges`. A drawn box is selected when completed. Empty selection = no panel, and the × does the same.
- **One drawn box at a time.** A new draw replaces it (the kickoff has one "your box").
- **Controls.** New first chip "draw box". The two old chips stay, secondary: "start only" and "set end" (existing behaviour and test ids). Session open / last swing set the start of a draw when one is armed, and are unchanged otherwise.
- **Esc:** armed → cancel the pick; else selected → clear selection (the box stays); else → remove the drawn box (today's Esc).
- **Empty chart click** with a selection clears it and does nothing else. Box hit-test comes after the armed pick, the trade markers and the GEX lines (a GEX line has an 8-px hit band and a box is large, so a level stays clickable inside a box).
- **Blocked reason once:** the v1 chip keeps it; the panel's "v1 engine off: …" line goes.
- **Engine box rows** use the engine's box min / height, so their edges are the engine's; drawn boxes keep ⚙.
- **Engine box verdict pill** = the panel's `boxVerdict` over the window, as a drawn box; the engine's own verdict is the "Engine at …" line. They can differ (p80 vs p70 lean, location), as they do today on a manual box.
- **`setup_reviews`** gains nullable `box_end_ts`, `box_top`, `box_bottom` (`anchor_ts` is already the start). The migration runs before any deploy that writes them.

## Smoke + wrap (2026-11-01)

| Commit | Step |
|---|---|
| `7d95ec4` | spec freeze |
| `ff92f94` | Step 0 |
| step 1 | helpers (`src/selectableBoxes.js`), rectangle primitive |
| `5e75b80` | step 2: draw box, engine boxes, selection, panel, relabel, click-through, help text, review fields, migration |
| `163d0bc` | smoke `tests/e2e/cr-dy.spec.js` |
| `8b62f5d` | Esc fix, moved assertions |

**Engine.** `v1Engine.js` is not edited (no diff vs `origin/main`). 86 sessions / 33,540 minutes: hash of trades + minutes + state equal to tag `cr-du-engine-v1.4`; 142 / 142 trades on box edges and verdict through the panel path (`scripts/logs/cr-dy/engine_parity.txt`). `engineBoxLine` in `v1Engine.js` now has no caller; left there to keep the file untouched.

**Smoke `cr-dy.spec.js`, 7 / 7.** 10-30 draw 07:34–07:49 gives rectangle 7654.25 / 7637.25, title "Your box 07:34–07:49", all rows. 10-30 with v1 trades: 5 trades, 5 boxes at their decision minutes (08:03–08:16, 08:27–08:35, 09:11–09:21, 09:50–09:58, 11:01–11:10); each click moves the panel and the highlight; the delta line follows. 01-26: drawn box works, no engine boxes, "set 2 in progress" once on the page. Armed click over the panel lands on the bar. Review POST carries the four fields. Unit 167 / 167; review routes 24 / 24 on temp copies; a save on the real `setup_reviews` in a transaction wrote all four, rolled back (154 rows before and after). Live check deferred to the first RTH session after deploy.

**Migration.** `scripts/cr_dy_setup_reviews_box.py`, run `9188623c`: `box_end_ts`, `box_top`, `box_bottom` nullable, 154 rows before and after, all NULL. Already applied, so the deploy has nothing to wait for.

**Existing specs vs baseline.** Baseline: CR-DU's `d464ec4` run (`scripts/logs/cr-du/specs/full_d464ec4.*`); `react_price_preview`, `apps` and `packages` are identical between `d464ec4` and `dcbcad2`.

| | Passed | Failed / timed out | Skipped |
|---|---|---|---|
| baseline | 99 | 20 | 19 |
| branch | 106 | 19 | 20 |

**Failing only on the branch: none.** Passes: +7 `cr-dy`. `cr-dg` live moved from timed out to skipped (it needs a live session; not this CR). The 19 failing are the same titles; messages differ on 3 only (`cr-dc` live and `cr-df` live, weekend, timeout vs assertion; `cr-dm` the worktree path in a missing-log error). This is one full run plus reruns of the changed files after the fixes (`cr-dc`, `cr-dp`, `cr-dq`, `cr-du`, `cr-dy`), not one clean full run after the last commit. Assertions changed, all for an intended change: `cr-dp` items 1–3 (chips), `cr-dq` shade helper and item 4 (primitive name), `cr-dq` FOMC and `cr-du` `ENGINE_LINE` ("Engine at"), `cr-du` 6 (reason on the chip), `cr-du` 9 (waits for the chart after the card's remount).

**Bugs the suite and smoke caught.** `_review_num` capped values at 1000, which would have refused every box price. With Edges off, the first Esc cleared an invisible selection.

**For Ryan.**
1. **Top / bottom** (reading above): engine rule used, not wicks. Screenshot `specs/assets/cr-dy/1-draw-0734-0749.png`.
2. **Pill vs engine line on an engine box:** all 5 on 10-30 read pill NO ACTION with "Engine at …: SHORT / LONG". The pill is `boxVerdict` on panel rules (lean p80), the engine leans at p70 and only counts in-box absorption. Kickoff followed; the alternative is the engine's verdict in the pill on an engine box.
3. **CR-DX** doesn't exist in the repo; its fit is unchecked.
4. **Header width:** with an end set at 1440 px the Flow header scrolls and clips "v1 trades" (it already scrolled; the new chip brings it on). Screenshot 1.
5. Two clicks inside lightweight-charts' 500 ms double-click window lose the second; the pick stays armed, so click again.
