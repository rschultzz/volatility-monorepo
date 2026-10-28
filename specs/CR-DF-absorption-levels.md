---
type: cr
cr_id: CR-DF
title: Flow v3 — absorption levels and an edge readout (display only; replaces the heatmap as the main view)
status: in-progress
run_mode: interactive
data_safety_class: read_only
dependencies: [CR-DC, CR-DD]
branch_name: feat/CR-DF-absorption-levels
tags: [dash, cr, order-flow, aggressor, price-chart, frontend, absorption, display-only]
---

# CR-DF — Flow v3: absorption levels and an edge readout

Display only, read-only DB. Branch `feat/CR-DF-absorption-levels` off `origin/main` `4db4173` (CR-DC merged), own
worktree (`.claude/worktrees/cr-df`). Kickoff: Ryan, 2026-10-27 (chat). Rulings DF1–DF4: Ryan, 2026-10-27 (chat),
before freeze. Frozen 2026-10-27 before any code.

---

## Kickoff (verbatim)

```
CR-DF — Flow v3: absorption LEVELS and an edge readout (display; replaces the heatmap as the main view).

Branch first: git checkout -b feat/CR-DF-absorption-levels off origin/main. Own worktree.
Read CLAUDE.md, standing-rulings.md (C1, C2), the CR-DC spec (cells endpoint, exact-data rule, absorption
rule, performance rules). Ryan, 2026-10-27: the heatmap is hard to see and looks like the bars; he wants
to see levels that have been absorbing recently, and whether we are absorbing at the range edges.
C1 deviation APPROVED by Ryan: absorption uses a new distinct colour (magenta, e.g. #e879f9) so it can't be
confused with candles (blue/grey) or GEX lines (yellow/green/blue/indigo); record it in the rulings change
log. Still no red/green.
Build (exact data only — es_flow_price_1m / ironbeam_es_trades; nothing on 1-s approximate data):
1. "Absorption levels" toggle in the v3 header (off by default): a horizontal magenta line at each price
   that meets CR-DC's absorption rule, starting at the minute the volume built up and ending when price
   trades through it (or continuing to now if it hasn't). Thickness ∝ contracts absorbed (log scale,
   capped); opacity fades with age (full at 0 min → 25 % at 60 min). Small label at the right end:
   "sellers absorbed 3.2k · 4 min" / "buyers absorbed …". Up to the 8 most recent/largest at once.
   Hover: start time, contracts, opposite-side contracts, held minutes, broke at.
2. Edge readout in the v3 header, when an anchor is set: box = highest / lowest 1-min CLOSE since the
   anchor; for each edge, aggressor contracts traded within 1 pt of it since the anchor (buyers at the top,
   sellers at the bottom), whether price closed beyond it by > 2 pts (held / broke), and minutes since the
   edge was first touched. Format: "Top 7724.75 · 1.8k buyers · held 9 m | Bottom 7715.25 · 4.1k sellers ·
   held 6 m". Thin dashed magenta lines mark the two edges when the readout is on.
3. Heatmap: keep as is but rename "Heatmap (cells)" and move it under ⚙; still off by default.
Performance as CR-DC (one canvas primitive per layer, visible range only, memoised). '?' guide: "visual aid
— untested". REVIEW replay never past `until`.
Tests: unit tests for level start/end, fading, edge stats; Playwright on 2026-05-01 and the live date with
levels and the readout on; Current and v2 pixel-identical; the headed-Chrome 10-min drag/zoom test with all
v3 layers on — no crash, no heap growth. PR, don't merge; Ryan merges, chat deploys.
```

---

## Rulings before freeze (Ryan, 2026-10-27, chat)

The brief left four choices open; Claude Code listed them with recommendations and Ryan ruled before freeze:
"DF1–DF4 approved as recommended."

| # | Choice | Ruling |
|---|---|---|
| DF1 | CR-DC's **Absorbed** toggle (slate triangles + slate line) | **Absorption levels replace it as the main view** — same rule (DC9), same ⚙ K / M. The slate marks move under ⚙ as "Absorbed (marks)", off by default, like the heatmap. |
| DF2 | Level start, end and age | The line **starts at the cell's minute** (`time`, the minute the volume built up) but **appears only once the hold window has ended** (`confirm_time` ≤ now / `until`) — causal; replay never shows a level early. It **ends at the break bar** (DC9 break: first bar with low < P − 1 tick / high > P + 1 tick) or runs to now / `until`. **Age** = now / `until` − start minute; broken levels fade too. |
| DF3 | "Up to the 8 most recent/largest" | The **top 8 by contracts × fade weight** (fade weight = the opacity of DF2's age), so old levels drop out by themselves and big recent ones win. |
| DF4 | The brief's edge rule contradicts itself (a highest close can never be closed beyond) | **The edge moves only with the 2-pt buffer:** a close ≤ 2 pts beyond the edge is a test (edge stays, still "held"); a close **> 2 pts** beyond shows **"broke hh:mm"** and the box moves to that close. **"held N m"** counts from the first minute whose high (top) / low (bottom) came within 1 pt of the edge. **Edge contracts from exact cells only** ("exact data only" otherwise). Ryan: this matches **CR-CP's A1 close-based box and void rule** (CR-CP A1.1: box edges on 1-min closes, wicks neither define nor void it, void only on a close > 2 pts beyond an edge; A1.2 / A1.a: the extreme moves only when a close is > 2 pts beyond the running extreme). |

Kickoff-approved deviations: **C1** — magenta `#e879f9` for absorption levels and edge lines (Ryan, kickoff).
**C2** — item 3 moves "Heatmap" from the v3 header into ⚙ (and DF1 moves "Absorbed" there), which changes the v3
header CR-DC built; both stay off by default, and Current / v2 stay pixel-identical. Recorded in the rulings change
log, not stopped for.

---

## Standing rulings conformance (`Dash/research/standing-rulings.md`, status current, read 2026-10-27)

CR-DF is a display CR, not a research CR; the table is filled anyway because the kickoff names C1 and C2.
Nothing in "Revoked / superseded" (R1–R7) is used.

| ruling | status | how |
|---|---|---|
| A1–A4, A6–A16, A18–A21 (setup) | n/a | no setup is defined, tested or shortlisted. The edge readout's box follows CR-CP A1's close rule (DF4) as a display, not a range qualification (no 15-min hold, no 12-pt width) |
| A5, A17, A19, A20 (flow at the range edges) | n/a | the readout shows aggressor contracts at the edges; it makes no fade / break call. Every "?" guide says "visual aid — untested" |
| B1–B3, B6–B8 (tests) | n/a | no test of a setup; no verdict |
| B4 (data split) | **honoured** | sources exactly as CR-DC (Databento served only for sessions ≤ 2026-06-05). Step 0 reads are mechanical (counts of levels, contract sizes for the line-width scale, timing) on train sessions and the live / forward session — no direction, outcome or feature statistic |
| B5 (Databento is the reference) | honoured | display only; badges name the source, as CR-DC |
| C1 (palette) | **deviates — approved by Ryan in the kickoff** | magenta `#e879f9` for absorption levels, their labels and the dashed edge lines, so they can't be confused with candles (blue / grey) or GEX lines (yellow / green / blue / indigo). No red / green. Logged in the rulings change log |
| C2 (off by default; existing views unchanged) | honoured for Current / v2; **v3 header change pre-approved in the kickoff** | everything new is off by default; Heatmap and Absorbed (marks) move into ⚙ (kickoff item 3, DF1), still off by default; Flow Current and v2 pixel-identical to `origin/main` (Playwright) |

---

## Design (as ruled)

**Data — exact only.** Levels come from CR-DC's marks unchanged (`flow_heat.absorption_marks`, DC9 rule, ⚙ K / M,
RTH only, exact tiers only — A1.2). The edge readout's contracts come from the payload's exact cells; if any segment
the anchor window touches is approximate, the readout prints "exact data only" in place of the contracts (the box,
held / broke and minutes still show, because they need only prices). The box uses the segment's **1-minute flow
closes** — the same series the marks' hold / break use (CR-DC A1 readings) and the same feed as the cells — which the
cells payload gains as `bars: {t, h, l, c}` per segment (the only backend change; cut at now / `until` as the cells
are). On 5-minute charts the readout still uses 1-minute closes (the brief says 1-min CLOSE).

**Levels (item 1).** For each mark with `confirm_time` ≤ now / `until`:
- start = `time`; end = `break_time` or now / `until` (DF2); contracts = `aggressor + merged_vol`;
- age (min) = (now / `until` − start) / 60; opacity = 1 − 0.75 × min(age, 60) / 60 (100 % → 25 % at 60 min,
  25 % after);
- thickness = log-scaled contracts, capped (scale fixed in Step 0 from the measured contract distribution);
- shown: the top 8 by contracts × opacity (DF3);
- label at the right end: "sellers absorbed 3.2k · 4 min" / "buyers absorbed …" (minutes = held minutes, DC9);
- hover: start time, contracts, opposite-side contracts, held minutes, "broke at hh:mm" / "not broken".
One canvas primitive draws all levels and labels; visible range only; inputs memoised.

**Edge readout (item 2)**, with an anchor set, from the anchor bar's minute to the last minute with data:
- top = bottom = the anchor minute's close; each later close c: c > top + 2 → **break**, top = c (record the
  break time); c ≤ top + 2 → test, top unchanged. Mirror for the bottom (c < bottom − 2);
- contracts = Σ aggressor contracts of cells since the anchor with |price − edge| ≤ 1 pt (buyers at the top,
  sellers at the bottom), against the current edge;
- first touch = the first minute since the anchor with high ≥ top − 1 pt (bottom: low ≤ bottom + 1 pt); "held N m"
  = last minute − first touch;
- status: "held N m"; after a break since the anchor the edge text adds "broke hh:mm" (the latest break);
- dashed magenta lines (thin) at the two edges from the anchor to now / `until`, drawn by one primitive.

**Header and ⚙ (item 3, DF1).** v3 header: "Absorption levels" toggle (off by default), "Edge readout" toggle (off by
default; shown with an anchor), Profile (unchanged). ⚙ gains "Heatmap (cells)" and "Absorbed (marks)" switches, both
off by default; the existing sell / buy layer switches and K / M stay. Stored layer keys: an old stored `heat` /
`marks` value carries over (a viewer who had them on keeps them on, now under ⚙).

**REVIEW replay** never past `until`: the payload is already cut at `until` (CR-DC); levels and edges read only it.

---

## Implementation order (one commit each)

1. Backend: `bars` per segment in `GET /api/ironbeam/flow/cells`; test (cut at `until`, equal to the minutes used for
   marks). No other route touched.
2. `flowV3.js`: `absorptionLevels` (start / end / fade / top 8 / labels) and `edgeReadout` (DF4 box, contracts, first
   touch, breaks, exact-only); unit tests.
3. Frontend: levels and edge primitives (magenta, one each), header toggles, ⚙ Heatmap (cells) / Absorbed (marks),
   hover, "?" guides; layer-key carry-over.
4. Smoke: Playwright (Current / v2 pixel identity vs `origin/main`; 2026-05-01 and live with levels and readout on;
   defaults off; replay `until`); headed Chrome 10-min drag / zoom with all v3 layers on, heap samples; PR.
