---
type: cr
cr_id: CR-DG
title: Absorption colour by side (sellers absorbed = orange) + live held minutes + cursor time snaps to the bar
status: in-progress
run_mode: interactive
data_safety_class: read_only
dependencies: [CR-DF]
branch_name: feat/CR-DG-absorption-side-colours
tags: [dash, cr, order-flow, absorption, price-chart, frontend, display-only]
---

# CR-DG — absorption colour by side + two small display fixes

Display only, read-only DB. Branch `feat/CR-DG-absorption-side-colours` off `origin/main` `0b4eb05` (CR-DF and
CR-DE merged), own worktree (`../volatility-monorepo-cr-dg`). Kickoff: Ryan, 2026-10-28 (chat). Frozen 2026-10-28
before any code.

---

## Kickoff (verbatim)

```
CR-DG — absorption colour by side + two small display fixes

Branch first: git fetch origin && git worktree add ../volatility-monorepo-cr-dg -b feat/CR-DG-absorption-side-colours origin/main. Work only in that worktree. Run git branch -v before every commit.

Read CLAUDE.md, components/field-conventions.md and research/standing-rulings.md (vault) first. This is a UI CR, not research: no conformance table needed beyond the C1 / C2 entries below.

SPEC
1. Absorption levels coloured by side (Flow v3, react_price_preview):
   - "buyers absorbed" (ceiling) keeps magenta #e879f9.
   - "sellers absorbed" (floor) becomes orange #fb923c: line, label and the legend swatch.
   - Fading, thickness, label placement and hover are unchanged; only the colour depends on side.
   - The edge readout's dashed lines stay magenta (out of scope).
   - Update the ⚙/legend text in AggressorFlowV3Panel.jsx to name both colours.
   Colour constants now live in flowV3Primitives.js (MAGENTA_RGB), AggressorFlowV3Panel.jsx (MAGENTA_RGBA) and PriceChart.jsx (level: '#e879f9'). Add one constant for the sellers colour; don't scatter new literals.
2. "Minutes held" for an unbroken level counts to now: label and hover show minutes from start to the payload's last bar (live: now; replay: the until time), not a frozen value. Broken levels keep start to break.
3. Cursor tooltip time snaps to the bar under the cursor (the same bar the axis label shows) instead of the interpolated position. Price text unchanged.

Standing rulings:
- C1 deviation APPROVED by Ryan in chat 2026-10-28: orange #fb923c added for "sellers absorbed"; still no red or green. Add a change-log line to standing-rulings.md saying exactly that.
- C2: honoured. No new feature and nothing on by default changes; the existing absorption display only changes colour by side.
- Visible changes are limited to items 1–3.

STEP 0 (commit before any implementation):
- Find the exact hex of every GEX line colour (yellow / green / blue / indigo) and the candle colours. Report the hue distance of #fb923c from each. If it's within 25° of the yellow GEX colour, STOP and report with two alternatives. Don't pick one.
- Confirm where "held" is computed for labels and hover, and why it's frozen for unbroken levels.

COMMITS: spec freeze → Step 0 → one commit per item (1, 2, 3) → smoke + wrap.

SMOKE:
- 2026-10-27 ETH in review: the 12:30–12:50 PT buyers-absorbed stack is magenta, any sellers-absorbed level is orange, and the top levels' held minutes run to the end of the payload.
- Live session: an unbroken level's minutes increase between two refreshes.
- Cursor on the 07:22 PT candle of 2026-10-27 reads 07:22.
- Existing cr-df / cr-dc Playwright specs pass; update only the colour assertions.
- Only one DB-backed local server at a time (capped pool). Never print connection strings.

Then open a PR. Don't merge; Ryan merges. Before removing the worktree, copy scripts/logs/cr-dg/ into the main checkout and verify the copies.
```

## Standing rulings (C only — UI CR)

| Ruling | Status | Note |
|---|---|---|
| C1 | **deviates — approved** | Ryan, chat 2026-10-28: orange `#fb923c` for "sellers absorbed"; still no red / green. Logged in the rulings change log. |
| C2 | honoured | No new feature, nothing on by default changes; the existing absorption display only changes colour by side. |

Visible changes: items 1–3 only.

## Implementation order

1. Spec freeze (this commit).
2. Step 0 findings (appended below).
3. Item 1 — sellers-absorbed colour.
4. Item 2 — held minutes to the payload's last bar.
5. Item 3 — cursor tooltip time snaps to the bar.
6. Smoke + wrap.
