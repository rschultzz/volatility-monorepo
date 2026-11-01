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
