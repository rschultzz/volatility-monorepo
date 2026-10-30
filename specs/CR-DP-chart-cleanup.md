# CR-DP — Chart cleanup, box stats popup, delta-share box line, adjustable flow pane and y-axis zoom

Branch `feat/CR-DP-chart-cleanup`, worktree `../volatility-monorepo-cr-dp`, off `origin/main` `6860fc5`, rebased onto `ce854ce`
(CR-DN merge, PR #122; CR-DO merge PR #123, no frontend overlap). UI CR, no research claims. Spec: Ryan, chat 2026-10-29 (revised kickoff, superseding the
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

## Step 0 — findings (2026-10-29, on `6860fc5`)

### S0.1 Stop check — Signals writes `bt_signals` (reported; Ryan decided)

- `SignalPanel` (`PriceChart.jsx:4045`) is **mounted whenever the chart is not in a review replay, whether its SIGNALS
  pill is open or collapsed** (the pill only toggles `collapsed`).
- Self-managed: when `tradeDate === new Date().toISOString().slice(0, 10)` (UTC date) it POSTs
  `/api/backtests-v2/signals/scan` on mount / date change and **every 60 s** while the tab is open
  (`SignalPanel.jsx:243–262`). `scan_signals_api` (`apps/web/modules/BacktestsV2/routes.py:1072`) runs the default
  strategy's scan and **upserts `bt_signals`** (UPDATE `:1193` / INSERT `:1216`, labels preserved).
- Its label buttons POST `/api/backtests-v2/signals/label` → UPDATE `bt_signals` (`routes.py:1299`), only on a click.
- No other caller of `/signals/scan` in the repo (no cron, no Dash page): the React price chart open on today was the
  only thing filling `bt_signals`. The Analogues backend (`apps/web/modules/Analogues/routes.py`) reads its labels.
- **Ryan's decision (chat 2026-10-29):** remove the Signals panel, its 60 s scan and the button. `bt_signals` stops
  filling from 2026-10-29. Keep the table and both endpoints (history stays readable). Recorded in CLAUDE.md. Remove
  Analogues as specced.
- Analogues: GET `/api/analogues` only while its panel is open; the module has no DB writes. No side effects.
- Flow Current: GET `/api/ironbeam/flow` (`apps/web/modules/Ironbeam/callbacks.py:3134`) only in Current mode (plus a
  live poll at `App.jsx:1864`, Current only); the handler writes nothing. Flow v2: the same v2 payload v3 loads. Side
  effects: localStorage only.

### S0.2 Inventory — what goes, what stays

| Thing | Where | Imported by | Fate |
|---|---|---|---|
| `AggressorFlowPanel.jsx` (Flow Current pane) | default export + `FlowModeToggle` | App; helpers by V2 / V3 panels | pane + toggle removed; helpers V3 imports (`utcEpochShowingZoneTime`, `formatShiftedHHMM`, `normalizeLogicalRange`, `rangesClose`, `normalizeShiftedCandleTimes`, `inferStepSeconds`, `computeSessionBands`, `coordinateForLinkedCrosshair`) move to `src/flowShared.js` |
| `AggressorFlowV2Panel.jsx` (Flow v2 pane) | default export | App; `alignV2ToCandles` by V3 | pane removed; `alignV2ToCandles`, `DEFAULT_MARKER_SETTINGS`, `effortLabel`, `effortIsHot`, `effortChartMarkers` (App: effort labels on the price chart) move to `flowShared.js` |
| App Current-only state | `flowPoints` / `flowLoading` / `flowError`, the two `/api/ironbeam/flow` effects (`:1512`, `:1864`), `flowEmaMinutes` + `ib-react-flow-ema-minutes`, `flowHistAlpha`, `flowPointsSignature` | Current pane | removed |
| App mode state | `flowMode` (`ib-react-flow-mode`), `flowV3` (`ib-react-flow-mode-v3`), `handleFlowModeChange`, `flowPaneMode`, `flowV2Data` | toggle / panes | removed; Flow is always v3. Both keys are no longer read or written (a stored `current` / `v2` simply opens v3); the stale keys are removed once on load |
| `flowMarkerSettings` (`ib-react-flow-v2-markers`) | swing size, push / div / fade / gap, effortRatio | **v3**: v2 payload request (`m_swing` → legs → "last swing", effort labels), effort label ratio | **kept** (same key). Its only editor was the Flow v2 ⚙; after this CR it is read-only (stored value or defaults). Open question for Ryan, not added (no spec language) |
| `flowEffort` (`ib-react-flow-v2-effort`) | Effort switch | **v3** header "Effort" | kept (same key) |
| `SignalPanel.jsx` | floating panel, 60 s scan | PriceChart | removed (S0.1 decision); `signalsCollapsed` state and SIGNALS pill removed; stale keys `ib-signal-panel-pos` / `-size` no longer read |
| `AnaloguesPanel.jsx`, `AnalogueDetailModal.jsx` | right-docked panel + modal | PriceChart (render), App (`PANEL_WIDTH`) | removed; App `analogues*` state, `/api/analogues` fetch, `reviewRightOffset`'s analogues term and the PriceChart props / pill removed |
| `ChartToggleBar` | pills | PriceChart | kept (SMILE, GEX, LANDSCAPE, REVIEW); its header comment updated |

Backend endpoints (`/api/ironbeam/flow` all modes, `/signals/scan`, `/signals/label`, `/api/analogues`) unchanged.

### S0.3 Header readout and per-box label (items 4, 6 reuse them)

- Edges: `flowV3.edgeReadout` (App `:2041`, memo `flowV3Edges`) → the header's `edgeText()` line
  (`flow-v3-edge-readout`) and the price chart's boxes (`PriceChart` `v3Edges`, `:2702`).
- Per-box stats: App `flowV3Segments` (`:2047`) = `boxSegments` + `segmentReadouts` (= `anchorReadout` per span) over
  the same aligned v2 rows the pane draws. The pane's in-pane line (`flow-v3-readout`) uses
  `segments.readouts[last]` (or `anchorReadout` without Edges); the price chart's box label is
  `boxLabelText(readout)` and its hover (`flow-v3-box-tooltip`) adds top / bottom contracts, held, probes, pending.
- `anchorReadout` already returns `vol` (Σ volume, same `vol` field as the CR-DM volume bars), so **Δ share =
  `delta / vol`** comes from the same object; the popup is built from `v3Edges`' current (non-frozen) box: its
  `readout`, `top` / `bottom` (`price`, `contracts`, `opposite`, `heldMin`, `brokeAt`), `pending`, `probes`, `ended`.
- Item 6 segments: `segmentCvd(rows, spans)` (Edges on) / `anchoredCvd(rows, anchor, end)` (Edges off). A parallel
  `segmentShare` / `anchoredShare` runs over the same spans and rows (Σ delta and Σ vol from the bar after the segment
  start, as `anchorReadout`), so the line's last value equals the popup's Δ share.

### S0.4 Overlay positioning (for the popup)

Price-chart overlays are either canvas primitives (boxes, labels, levels; `v3PrimRef`) or React divs placed from the
chart's own transforms (`series.priceToCoordinate`, `timeScale().timeToCoordinate`) plus
`hostRef.offsetLeft / offsetTop` (box / level / probe tooltips, `:3763`), re-read on an animation frame (review shapes,
`:2830`). The popup is a React div in the same container, default position from the current box's `to` / `top`
coordinates, clamped to the plot (host width minus the price axis, pane height minus the time axis).

### S0.5 Chart library (lightweight-charts 5.1.0 installed, `^5.0.8`)

- `IPriceScaleApi.setVisibleRange` / `getVisibleRange` / `setAutoScale` exist for **any** price scale id, overlay
  scales included; `handleScale.axisPressedMouseMove` and `axisDoubleClickReset` act only on the visible axis scale.
- **8a:** the price chart already has `axisPressedMouseMove: true` (drag on the axis scales the price, library) and a
  custom wheel zoom; the library's default `axisDoubleClickReset` re-fits. New: remember the manual range (per trade
  date, localStorage) and restore it after a reload; a double-click or a date change clears it. Live refreshes call
  `setData` only (no `setAutoScale(true)` outside review fit and IV), so a manual zoom already survives them; verified
  in the smoke.
- **8b: per slot.** The Flow v3 pane's slots are separate price scales in pane 0 (`cvd`, `right` = volume / rolling,
  `vol`, `delta`) placed by `scaleMargins`. A custom drag on the pane's right axis finds the slot under the pointer
  from the margins and calls `setVisibleRange` on that slot's scale (stretch / compress about the slot's centre);
  double-click returns that slot to auto-fit. A stretched slot's series can draw past its slot's edge (the library has
  no per-scale clipping).
- **7a:** slot heights are those margins, so dividers are DOM handles over the pane at the slot boundaries that rewrite
  `paneSlotMargins` from stored fractions (min 24 px each; the pane's height unchanged). The Edge effort strip is its
  own pane (fixed 64 px), not a slot.
- **Item 6:** fixed ±10 % via the line series' `autoscaleInfoProvider`; guides as price lines on that series (slate, no
  axis label); values beyond ±10 % drawn at ±10 % with a cap mark (a small primitive like CR-DN's `VolumeCapPrimitive`).

### S0.6 Feasibility of 7c — **already in place**

A draggable splitter between the price chart and the Flow pane exists (`App.jsx:2629`, `.panel-splitter`,
`beginResize`): it sets `flowPanelHeight` (clamped `FLOW_MIN_HEIGHT`–`FLOW_MAX_HEIGHT`), remembered in
`ib-react-flow-panel-height`; the price chart takes the rest (flex). No App layout rework: 7c = keep it and have ⚙
"Reset layout" also restore the default height (220).

### S0.7 Tests that assert removed controls (to retire / update)

`cr-cd.spec.js` (Current / v2 toggle), `cr-ch.spec.js` (v2 pane fiber), `cr-cd-a5.spec.js` (v2 pane / storage),
`cr-dc.spec.js` (mode toggle, `ib-react-flow-mode-v3`), `cr-cc.spec.js` (SIGNALS pill hidden in replay / restored on
reveal), plus any v3 assertions on `flow-v3-source`, `flow-v3-cells-badge`, `flow-v3-readout`,
`flow-v3-edge-readout`. Final list with what changed per assertion goes in the smoke commit.

## Implementation notes (deltas found while building; no spec change)

- **Item 4:** the popup's plot clamp uses the time scale's width — the pane element is the chart's table row, price axis
  included. With Edges off the in-pane "since hh:mm" readout stays (now with Δ share); only the Edges-on "box from…" line
  is gone. The header's Edges hints ("edges: set an anchor / loading…") left with the readout.
- **Item 6:** a box's first one or two minutes are single-minute shares, often beyond ±10 % (drawn clipped with a cap
  mark). The chart draws through a box's empty start bar, so the joining segment is transparent (`gapAt`, as CR-DN's CVD).
  The smoke's "box 10:15–10:34" is anchor 10:15 + set end 10:34 (anchored at 06:49 no box starts at 10:15; boxes run
  10:12–10:40 and 10:41–12:25).
- **Item 7:** a divider's handle stays inside the gap between its slots (4–8 px) so a slot's hover keeps its tooltip.
- **Item 8:** price-chart axis drag / double-click are the library's (unchanged); new is the per-date memory. Flow v3 zoom
  is a factor on each slot's auto range (autoscale stays on), so live refreshes keep it; a stretched slot can draw past its
  slot's edge (no per-scale clipping in the library).

## Smoke (2026-10-29, read-only backend `scripts/cr_dp_smoke_backend.py` :8067; branch :5191, origin/main `ce854ce` :5192)

`tests/e2e/cr-dp.spec.js` **6 / 6** (logs `scripts/logs/cr-dp/`):

| Smoke item | Result |
|---|---|
| Header groups in order; no toggle, no pills; approx note | 10-29: `Flow v3 · Ironbeam` / `📍 set anchor · set end · session open · last swing · clear` / `Edges · Edge effort` / `Absorption levels · Profile · Effort` / `⚙ · ?` (right edge); a stored `ib-react-flow-mode=v2` opens v3 and the retired keys are removed. 10-20 (approx): one note "approx data — absorption & edges off", no pills, no "Absorbed: exact data only" |
| 06:49 anchor, Edges on: popup = origin/main's numbers + Δ share | replay to 11:00 (current box 10:41): branch label = origin/main label `Δ -1.3k · -5.8 pts · 18 m · 9.5k/pt`; popup head `box from 10:41 · Δ -1.3k · share -2.39 % · -5.8 pts · 18 m · 9.5k/pt`; Top 7738.25 · buyers 4.9k · held 18 m and Bottom 7728.75 · sellers 4.4k · held 12 m · moved 10:53 = origin/main's header readout. Default place right of the box top-right (clamped in the plot), < 8 % of the plot |
| drags and stays; clear removes; new anchor resets; × / Esc / Edges off | drag −220 / +112 px held across Profile on/off and a pan / zoom; clear → gone; new anchor → default place; × hides until the next anchor; Esc and Edges off remove it |
| Box 10:15–10:34 ≈ +1 % (hand Σ delta ÷ Σ volume) | **+1.39 %** = 591 / 42,591 from the v2 minute bars = line value at 10:34 = popup share; nothing drawn outside the window; every 06:49 box's last value = its hand sums; scale ±10 % |
| No anchor = raw CVD as origin/main; ⚙ raw CVD | line data identical to origin/main (no anchor; and anchored 06:49 with ⚙ raw CVD) |
| Slot dividers; persist; Reset layout | +12 px moves one boundary only (middle slot floors at 24 px), pane height unchanged; kept after reload; the existing price / Flow splitter (7c) +60 px remembered; Reset layout restores slots and 220 px |
| Axis drag zooms; double-click resets; live refresh keeps it | 2026-10-30 (live): price range held while a new bar arrived (66 → 67 bars) and after a reload; double-click → autoscale, key cleared. Flow v3: drag over the line slot stretched it ×1.65 (volume / delta unchanged), delta slot compressed; both kept after reload; double-click reset only the line slot |
| Signals / Analogues gone; REVIEW and LANDSCAPE work | pills `SMILE / GEX / LANDSCAPE / REVIEW`; `/api/gex-landscape` 200; REVIEW panel opens; no `/signals/*` or `/api/analogues` request in 65 s |

Unit **74 / 74** (`flowV3Share` 6, `flowV3Slots` 4 new). Build clean.

### Existing specs (all run against the branch and origin/main)

First run 54 passed / 38 failed / 4 skipped (`existing-specs.log`); after the updates below, the 11 changed files
re-ran 40 passed / 17 skipped / 3 failed (`rerun-specs.log`), then the cr-db pixel test passed with its clip moved
below the pill shadow. Remaining failures, all also failing on origin/main or not runnable here:
`cr-db` "keys T / S / U" (fails on main, as noted in CR-DN), `cr-dn` item 6 "57k closing minute" (asserts origin/main
does not cap — stale since CR-DN merged; fails with both sides = main), `cr-dh` Dashboard ×4 (need the Dash app; not
served by the smoke backend).

**Retired (`test.skip`, reason in the file)** — they test removed controls:
- Flow Current / v2 / mode toggle (item 1): `cr-cd` ×3 (toggle; markers; screenshots + marker counts), `cr-ch` ×1,
  `cr-cd-a5` ×2 (Current / v2 unchanged when off; v2 Effort toggle), `cr-dc` ×2 (Current / v2 pixel-identical; v3
  defaults "back to Current"), `cr-df` ×2 (Current / v2 unchanged; defaults via the toggle), `cr-dg` ×1 (one Effort
  switch for v2 and v3), `cr-dm` ×1 (Flow v2 draws no live label).
- `cr-di` "chip off: pixel-identical to origin/main" — with an anchor + Edges the popup (4), the share line (6) and the
  pill row (5) differ by design.

**Updated:**
- `cr-cc`: SIGNALS is not restored on reveal (count 0).
- `cr-dc`: exact / approximate pills → the approx note (approximate) or none (exact) ×3; no "Absorbed: exact data only"
  on approximate data; the narrow-header mode-toggle width check removed.
- `cr-df`: anchor read from `data-anchor` (no Edges-on readout); "edges: set an anchor" → no popup; the header edge
  readout ×3 → the popup's Top / Bottom lines (`boxPopupLines` of the same `edgeReadout`).
- `cr-di`: header edge readout → popup Top / Bottom (×3 cases); the approximate log reads the popup.
- `cr-dn`: header edge readout ×4 → popup Top / Bottom; in-pane "box from…" ×3 → popup head; these tests check the
  anchored CVD, so they run with ⚙ raw CVD (`anchoredLine: 'raw'`).
- `cr-db` pixel: compared below the pill row and its shadow (top 96 px).

### Open questions (for Ryan)

1. **Swing size has no editor now.** The v2 marker settings (`ib-react-flow-v2-markers`: swing size for "last swing" /
   the legs, effort ratio) were only editable in Flow v2's ⚙. They keep their stored value (or defaults: swing 8,
   ratio 1.5). Add "Swing size" to Flow v3 ⚙?
2. **Mixed-sources note.** The Edges-on "box from…" line carried " · mixed sources (Ironbeam + Databento)" when a box
   spans ETH and RTH from different sources; the popup doesn't show it. Add it?
3. `bt_signals` stops filling from 2026-10-29 (Ryan's decision): the repo has no other caller of `/signals/scan` (no cron, no Dash page).
