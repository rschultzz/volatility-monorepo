# CR-DP — Chart cleanup, box stats popup, delta-share box line, adjustable flow pane and y-axis zoom

Branch `feat/CR-DP-chart-cleanup`, worktree `../volatility-monorepo-cr-dp`, off `origin/main` `6860fc5`
(CR-DN merge, PR #122). UI CR, no research claims. Spec: Ryan, chat 2026-10-29 (revised kickoff, superseding the
first CR-DP kickoff the same day; frozen below, lightly formatted).

## Spec (Ryan, chat 2026-10-29)

1. **Flow: v3 only.** Remove the Flow mode toggle (Current / v2 / v3) and the Current and v2 panes from the UI; Flow
   always renders v3.
   - Keep every backend endpoint (v3 loads the v2 payload).
   - Remove frontend code used only by the removed panes; keep anything v3 imports.
   - Saved per-browser settings that point at Current / v2 open as v3.
2. **Data-source pills:** remove the "ETH exact / approximate" and "RTH exact" pills. When either segment is on
   approximate (1-s) data, show one small header note, "approx data — absorption & edges off". Nothing on exact
   sessions.
3. **Header layout**, left to right, in groups with a small gap between groups:
   - Anchor: 📍 set anchor · set end · clear end · session open · last swing · clear
   - Edges: Edges · Edge effort
   - Display: Absorption levels · Profile · Effort
   - ⚙ and ? at the far right.

   Same buttons and behaviour; only order and grouping change. The anchored readout text (box from…, Top…, Bottom…)
   leaves the header (see 4).
4. **Box stats popup.** With Edges on and an anchor set, a small floating panel on the price chart near the current
   box (default: just right of the box's top-right, kept inside the plot).
   - Contents: box from hh:mm (→ end, if set) · Δ · Δ share · pts · minutes · contracts/pt; Top price · buyers /
     sellers · held · moved / probe / testing-break state; Bottom price · sellers / buyers · held · state. One source
     with today's header readout and per-box label.
   - Draggable by its body. Position kept while the anchor stays; resets to default on a new anchor.
   - Goes away on clear (or Esc clear) and when Edges is turned off. A small × hides it until the next anchor.
   - Updates live each refresh. Frozen boxes keep their hover stats (no popup).
   - Dark tag background, the chart's label style, compact.

   The in-pane "box from…" line at the Flow v3 pane's top-left is removed.
5. **Remove the Signals and Analogues buttons** from the chart (and any overlay they open). Keep REVIEW and LANDSCAPE.
6. **Anchored line as delta share, fixed scale.**
   - a. Whenever an anchor is set, the anchored line in the Flow v3 pane plots cumulative delta share: Σ delta ÷ Σ
     volume since the segment start (the same volume field as the CR-DM volume bars). Segments are per box with Edges
     on (restarting at each box, as CR-DN); with Edges off, one segment anchor → now / end.
   - b. Fixed scale ±10 %, with faint guides at 0, ±2 % and ±5 % (existing slate, C1). Values beyond ±10 % clip with a
     small cap mark; hover shows the true value.
   - c. Hover: time, delta share so far, Σ delta, Σ volume.
   - d. No anchor: raw session CVD, as today. ⚙ "Anchored line: delta share / raw CVD", default delta share; raw keeps
     today's behaviour.
7. **Adjustable flow pane slots.**
   - a. Draggable dividers between the Flow v3 pane's slots (top line area · volume · delta bars, plus rolling delta
     when on) set how much of the pane each slot takes. Minimum slot height 24 px. The pane's total height is
     unchanged.
   - b. Remembered per browser. "Reset layout" in ⚙ restores defaults.
   - c. Optional, report feasibility in Step 0: a draggable divider between the price chart and the Flow v3 pane, so
     Ryan can give the price chart more or less room. If it fits the existing layout cleanly, include it (remembered,
     with reset). If it needs an App.jsx layout rework, stop and report.
8. **Y-axis zoom.**
   - a. Price chart: drag on the right price axis stretches / compresses the vertical scale; double-click the axis
     returns to auto-fit. The setting survives refreshes until reset or a date change.
   - b. Flow v3 pane: the same on its axis for the slot under the pointer, if the chart library supports per-slot
     scaling; otherwise for the whole pane. Report which in Step 0. Double-click resets.
   - c. Zoom never changes the data, only the view. The live refresh must not undo a manual zoom.

## STOP check (Step 0, before removing anything in items 1 and 5)

Find what the Signals button / SignalPanel does when shown or hidden. If removing it stops any write to `bt_signals`
(or any other table) or any scheduled / background behaviour, STOP and report what writes when; Ryan decides. Same
check for Analogues and the Current / v2 panes.

## Standing rulings conformance (UI CR)

Read `research/standing-rulings.md` (2026-10-29 copy, last ruling A22, last change-log line CR-DN amendment 3) before
the freeze.

| Ruling | Status | Note |
|---|---|---|
| A1–A22 | n/a | no range, level, setup, entry, exit or outcome logic changes; the popup shows the existing box stats |
| A23–A28 (Ryan's drafts) | n/a — not rulings | |
| B1–B7 | n/a | no hypothesis tested, no data split read; delta share is a display aid with no research meaning |
| B8 — Code smoke labels are never Ryan's | honoured | no review labels written; smoke runs against a read-only backend |
| C1 — no green / red; Dash palette | honoured | no new colours: popup uses the chart's tag background and label style; guides in the existing slate; the approx note in existing header text colours |
| C2 — new display features off by default, existing views unchanged | **deviates — APPROVED by Ryan in chat 2026-10-29** | visible changes limited to items 1–8 (removes controls and views; changes the anchored line's default to delta share). Change-log line added to `standing-rulings.md` |

Nothing in "Revoked / superseded" (R1–R7) is used. Delta share, slot sizes and zoom are display aids with no research
meaning.

## Step 0 (commit before implementation)

- The stop check above.
- Every component, state and storage key tied to Current / v2 / Signals / Analogues, and what imports them.
- Where today's header readout and per-box label are computed (items 4 and 6 must reuse them).
- How the chart's tag overlays are positioned (for the popup).
- The chart library's support for axis-drag scaling and per-scale margins (items 7 and 8).
- Feasibility of 7c.

## Commits

spec freeze → Step 0 → one commit per item → smoke + wrap.

## Smoke (read-only backend)

- Header groups in order; no mode toggle, no pills; an approximate session shows the one note.
- 2026-10-29, anchor 06:49, Edges on: popup near the box with the same numbers as origin/main's header readout plus
  Δ share; drags and stays; clear removes it; a new anchor resets its position.
- Box 10:15–10:34 line sits near +1 % (hand-check Σ delta ÷ Σ volume from the minute data); no anchor shows raw CVD
  as origin/main.
- Slot dividers resize slots and persist across reload; Reset layout restores.
- Axis drag zooms; double-click resets; a live refresh doesn't undo the zoom.
- Signals / Analogues gone; REVIEW and LANDSCAPE still work.
- Existing specs green; only assertions testing removed controls updated or retired (list them).

## Wrap

Session note, PR (don't merge), logs copied. Don't deploy. Only one DB-backed local server at a time; never stop
another session's server without asking Ryan. Never print connection strings. Don't touch CR-DO's worktree.
