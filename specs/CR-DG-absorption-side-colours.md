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
