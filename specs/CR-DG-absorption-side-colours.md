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

Visible changes: items 1–3, plus item 4 (amendment below).

## Implementation order

1. Spec freeze (this commit).
2. Step 0 findings (appended below).
3. Item 1 — sellers-absorbed colour.
4. Item 2 — held minutes to the payload's last bar.
5. Item 3 — cursor tooltip time snaps to the bar.
6. Smoke + wrap.

---

## Step 0 — findings (2026-10-28, read-only)

Scripts: `scripts/cr_dg_step0/dump_cells.py` (one read-only connection on the backfill role, no server — a CR-DE
smoke server was already running, so no second server was started) and `scripts/cr_dg_step0/levels_held.mjs`
(runs the shipped `flowV3.absorptionLevels` on the dumps). Logs: `scripts/logs/cr-dg/step0/`.

### S0.1 — GEX line and candle colours; hue distance of `#fb923c` (hue 27.0°)

The React price chart draws each GEX segment with `seg.color` from the Ironbeam backend
(`callbacks.py:1029` `_color_for_net_gex` → `GEX_HEATMAP_COLORSCALE`, `callbacks.py:164`). Net GEX is clipped to
±the day's 95th percentile of |net| (`GEX_COLOR_PERCENTILE` 95), so the strongest call levels are exactly the
`1.00` stop and the strongest put levels exactly the `0.00` stop (A12). Plotly interpolates between stops in RGB.

| Role | Hex | Hue | Distance from `#fb923c` |
|---|---|---|---|
| **yellow (call, 1.00)** | `#fef08a` | 52.8° | **25.7°** |
| lime (0.85) | `#a3e635` | 82.7° | 55.7° |
| green (0.70) | `#4ade80` | 141.9° | 114.9° |
| pale green (0.55) | `#bbf7d0` | 141.0° | 114.0° |
| neutral (0.50) | `#4b5563` | 215.0° | 172.0° |
| pale blue (0.45) | `#bae6fd` | 200.6° | 173.6° |
| sky (0.30) | `#38bdf8` | 198.4° | 171.4° |
| blue (0.15) | `#1d4ed8` | 224.3° | 162.7° |
| **indigo (put, 0.00)** | `#312e81` | 242.2° | 144.8° |
| up candle | `#60a5fa` | 213.1° | 173.9° |
| down candle | `#e5e7eb` | 220.0° | 167.0° |
| (ref) buyers absorbed, magenta | `#e879f9` | 292.0° | 95.0° |

Interpolated colours between 0.85 and 1.00 run 82.7° → 52.8°, so no GEX colour is nearer than the yellow end.
**Gate: 25.7° > 25° → not a STOP**, by 0.7°. Yellow is also much paler (L 0.77) than the orange (L 0.62).

Not GEX, disclosed because they can share the same chart: the condor σ-band lines `#ff6a00` (`PriceChart.jsx:3000`,
hue 24.9°, **2.1° away**; drawn only once a condor anchor is picked), the short-strike lines `#fcd34d` (18.9°)
and the review amber zone `#fbbf24` (16.2°). These are outside the gate as written.

### S0.2 — where "held" is computed, and what is actually frozen

- **Level label and level hover:** client-side, `flowV3.js:337` `absorptionLevels`:
  `end = broken ? break_time : payloadNow(payload)`, where `payloadNow` is the end of the payload's last 1-minute
  bar (capped at the segment end). Live: last completed minute; replay: `until`. Verified on 2026-10-27 (FULL):
  - replay until 12:55 PT: unbroken levels 12:19 / 12:48 / 12:49 end 12:55 (36 / 7 / 6 min);
  - complete session: unbroken levels end 13:00 (12 / 11 / 5 min).
  So the level label already runs to the payload's last bar and is recomputed on every refresh. **It is not frozen.**
- **What is frozen — (a) the Absorbed (marks) hover** (`PriceChart.jsx:3518`) prints the server's `held_min`
  (`packages/shared/flow_heat.py:150`): `stop = break_time or end_epoch`, and `end_epoch` is the RTH segment end,
  which on a live session is **13:00 PT** (`flow_v3._build`: `end = rth_b` with no `until`). So live it shows
  start → 13:00, the same value on every refresh (the CR-DF open question). Replay is correct (end = `until`).
- **(b) the level hover is stale while the pointer is still:** `setV3LevelHover` stores the level object captured at
  the last crosshair move; a payload refresh rebuilds `v3Levels` but the open hover keeps the old object
  (`PriceChart.jsx:2606–2610`) until the pointer moves.

→ Item 2 as written ("label and hover … not a frozen value") targets the level label, which is already right. The
frozen numbers are (a) and (b). Scope question put to Ryan before any item-2 code (spec amendment rule).

### S0.3 — item 3 (for the record)

`updateFloatingTooltip` (`PriceChart.jsx:1770`) formats `interpolateShiftedEpoch(clampedLogical, …)` — the
fractional logical index under the pointer — so the time drifts between bars. `param.time` (the bar the crosshair
snaps to, the one the axis label shows) is available in the same handler.

### Step 0 rulings (Ryan, chat, 2026-10-28) — spec amendment

- **Item 2 → both, client-side.** (a) The Absorbed (marks) hover shows held computed on the client from the mark's
  start to its break, or else to the payload's last bar (`payloadNow`), not the server's `held_min`. (b) The level
  hover re-reads its level from each new payload, so an open hover updates on refresh. The server `held_min` and
  the CR-DC golden are unchanged. The level label is already right and is unchanged.
- **Orange `#fb923c` confirmed** after the 25.7° / σ-band (2.1°) disclosure.

---

## Spec amendment — item 4: Effort toggle in Flow v3 (Ryan, chat, 2026-10-28; before more implementation)

Ryan: "On V3, I can no longer turn the effort on and off." Today the effort labels are v2-only: `App.jsx`
`flowEffortLegs` returns null when `flowV3` is on (since CR-DC step 3), and the Effort chip exists only in the v2
header (`flow-v2-effort-toggle`).

4. Add an Effort chip to the Flow v3 header (`flow-v3-effort-toggle`). It uses the **same state** as v2
   (`flowEffort` / localStorage `ib-react-flow-v2-effort`), so one switch covers both modes. When it is on in v3, the
   price chart shows the same effort labels as v2, from the v2 legs v3 already fetches (`effort=1`), with the same
   ⚙ swing size and `effortRatio` threshold. Off by default (C2) unless the viewer already turned it on in v2.
   Nothing else in v3 changes. Leg hover detail is required only where v2 has it on the price chart; v2's leg hover
   (`flow-v2-effort-hover`) is in the v2 flow pane, not on the price chart, so no v3 hover is added.

Rulings: **C2 honoured** (the new chip is off by default; existing views are unchanged with it off). Visible change in
scope, approved by Ryan's request above. Visible changes are now limited to items 1–4.

Commits: one commit for item 4, after this amendment.

Smoke (item 4): 2026-10-27 in v3 — Effort off → no labels (price chart identical to before); on → the same labels as
v2 for the same settings; the switch carries between v2 and v3. Existing specs stay green.

---

## Smoke + wrap (2026-10-28)

Servers: `scripts/cr_dg_smoke_backend.py` (8064, read-only, pool 3 + 2 — the only DB-backed local server; CR-DE's idle
smoke server was stopped first with Ryan's OK), this branch on 5181, origin/main on 5180. Logs: `scripts/logs/cr-dg/`.

| Check | Result |
|---|---|
| Unit (`npm run test:unit`) | 30/30 |
| `cr-dg.spec.js` | 5 passed, 1 skipped (live, deferred — below) |
| `cr-df.spec.js` | 6/6 |
| `cr-dc.spec.js` | 6/6 (the superseded "unchanged" width assertion updated — below) |
| Build | clean |

- **Item 1, 2026-10-27 FULL:** every drawn level samples its side's colour on the canvas — 7 buyers-absorbed magenta
  (including the 12:48 / 12:49 / 12:55 stack), sellers absorbed 7762.25 (from 12:19) orange. Asset
  `specs/assets/cr-dg/levels-2026-10-27.png`.
- **Item 2:**
  - Unbroken levels' held runs to the payload end: 13:00 for the complete session, 12:55 with `until` 12:55.
  - The level and mark hovers on the 12:55 level both read "held 5 min · not broken".
  - Unit test: a refresh from T to T + 2 min takes held n → n + 2 in the label and the marks hover, and an open hover updates with the pointer still.
- **Item 2, live: deferred to Ryan's first live use** (Ryan's ruling). From 06:11 to 06:56 PT one mark formed (buyers 7773, 06:35), and it broke before it confirmed at 06:39. What to look for: hold the pointer on an unbroken level; its hover "held N min" and its label rise as each new 1-minute bar lands; the same on an Absorbed (marks) triangle.
- **Item 3:** the cursor on the 07:22 candle reads `10/27 07:22:00 PT` at three spots across the bar: 1-min, 1-min with a day either side, and 5-min (07:20). **origin/main reads 07:23** with a day either side (bug reproduced).
- **Item 4, 2026-10-27 in v3:**
  - Effort off: the price chart is pixel-identical to origin/main's v3.
  - Effort on: the price chart is pixel-identical to v2 with Effort on, and differs from Effort off.
  - The switch carries v2 → v3 (on) and v3 → v2 (off); `ib-react-flow-v2-effort` is "1" then "0".
  - Assets `effort-v3-{on,off}-2026-10-27.png`.
- **cr-dc "unchanged" (test only, no visible change):** DC1's `toggleW > main` assumed main had no "v3" button. Since CR-DC merged, it has, so the test failed on every branch. It now asserts equal widths and the v3 button visible on both builds. Every other check is unchanged.
- **cr-df / cr-dc assets:** the runs rewrite their tracked screenshots. The rewritten copies are kept in the logs, and the committed ones were restored.
