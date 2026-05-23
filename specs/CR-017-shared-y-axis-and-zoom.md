# CR-017 — Shared Y Axis and Zoom

## Problem

CR-016 wired up shared Y-axis behavior on `/today-setup` but the chart and landscape don't actually display the same range. Observed on 2026-05-22:

- Anchor chart Y: ~7440 → ~7580 (~140pt range)
- Anchor landscape Y: ~7472 → ~7526 (~54pt range)

The wiring is in place — `MiniPriceChart` emits a range via `onPriceRangeChange`, `DayView` passes it to `GexLandscape` as `visiblePriceRange` — but something between the read and the consumption produces a mismatch.

Separately, neither component currently supports user-controlled Y-axis zoom. The chart's wheel + drag handlers are disabled (`handleScroll: false, handleScale: false` on `createChart`), so the user can't adjust the visible price window themselves.

A third issue: anchor and selected DayViews render side-by-side at the top level. The intent is for them to stack vertically so each row has the full page width.

## Goal

Three changes:

1. **Vertical-stack layout for anchor vs selected.** Anchor DayView on top, selected directly beneath. Each DayView's internal layout (chart + landscape side-by-side, 480px tall) is unchanged. Page max-width 1900px.
2. **Genuine shared Y axis.** Chart Y range and landscape Y range display identical bounds.
3. **Interactive Y-axis zoom on the shared axis.** Wheel-scroll on either component zooms the shared range. Both components stay in lock-step.

## Changes

Pure frontend. No backend or DB changes.

**Layout (App.jsx):** Change dual-view container from `display: flex` (row) to `flexDirection: column`. Change DayView wrappers from `flex: '1 1 380px'` to `width: '100%'`.

**Axis fix (MiniPriceChart + GexLandscape):**
- `MiniPriceChart`: after computing union range, call `chart.priceScale('right').setVisibleRange({ from: priceBot, to: priceTop })` so the chart displays exactly the computed range.
- `GexLandscape`: fix synced `yOf` to use landscape's own SVG coordinates: `PAD.top + ((pHi - price) / (pHi - pLo)) * plotH`. Remove `paneHeight` from synced check.

**Zoom:**
- `MiniPriceChart`: wheel handler on container div computes zoom → `setVisibleRange` → emits via `onPriceRangeChange`.
- `GexLandscape`: wheel handler → calls new `onRangeChange` prop.
- `DayView`: new `externalRange` → MiniPriceChart prop so landscape zoom drives chart. Loop-breaking ref prevents feedback.
- `DayView`: reset `priceRange` to null on `date` prop change.

## Acceptance criteria

1. Anchor DayView at top, selected directly beneath. Full-width within 1900px max. Internal layout unchanged.
2. Chart Y top/bottom labels match landscape Y top/bottom labels within ±0.5pt on 5/22 and selected day.
3. Cluster lines align horizontally across chart and landscape.
4. Mouse-wheel on chart body zooms Y axis; landscape follows in real time.
5. Mouse-wheel on landscape zooms Y axis; chart follows.
6. Clicking a new analogue resets shared range to the new day's natural extent.
7. Anchor and selected views are independent — zooming one doesn't affect the other.
8. `vite build` clean for `react_today_setup/` and `react_price_preview/`.

## Verification plan

1. Reload `/today-setup?date=2026-05-22`. Confirm anchor DayView at top. Click 2026-05-21. Confirm selected DayView renders directly beneath anchor. Both full-width within 1900px.
2. Confirm chart and landscape Y labels match within each DayView on 5/22 and 5/21.
3. Wheel-scroll over chart body. Both Y axes zoom in lock-step.
4. Drag on chart price-axis column. Same lock-step behavior.
5. Wheel-scroll over landscape. Chart Y zooms with it (or document deferral).
6. Click 2026-05-19 analogue. Selected range resets to natural extent.
7. Zoom selected; anchor range untouched.

## Out of scope

- Landscape title regime label inconsistency
- X-axis sharing
- Cross-day Y-axis sharing (anchor/selected ranges remain independent)
- Persisting zoom state across navigations

---

## Step-0 Diagnosis Findings

*Appended before any implementation code per the diagnosis gate.*

### (a) lightweight-charts version

v5.2.0 installed (`react_today_setup/node_modules/lightweight-charts/package.json`).

`PriceScaleApi` in v5.2.0 has:
- `setVisibleRange({ from, to })` — sets explicit range, calls `setAutoScale(false)` internally
- `getVisibleRange()` → `{ from, to }` (null if no data)
- `setAutoScale(on)`

No direct "price scale changed" subscription event in the public API. Zoom detection must be done via wheel/mouse handlers.

### (b) What `onPriceRangeChange` emits

`MiniPriceChart` does NOT call `getVisibleRange()`. It computes `{ priceBot, priceTop }` from the union of bar highs/lows + cluster prices + 4% padding, then emits that directly (MiniPriceChart.jsx:160). The chart's actual displayed range is wider because lightweight-charts auto-scale adds default `scaleMargins` (typically 10% top + 10% bottom) on top of the range anchor data points.

### (c) GexLandscape `visiblePriceRange` consumption

Lines 151–166. `synced` check:
```js
const synced =
  visiblePriceRange &&
  Number.isFinite(visiblePriceRange.priceTop) &&
  Number.isFinite(visiblePriceRange.priceBot) &&
  Number.isFinite(visiblePriceRange.paneHeight) &&   // DayView passes ROW_HEIGHT=480 → true
  visiblePriceRange.priceTop !== visiblePriceRange.priceBot
```

`synced` IS true (DayView supplies `paneHeight: ROW_HEIGHT`). The bug is in the `yOf` formula:
```js
yOf = (price) =>
  ((priceTop - price) / (priceTop - priceBot)) * paneHeight - offset
```
`paneHeight=480`, `offset = size.offsetTop ≈ 36` (header height). This maps prices to chart-absolute pixel coordinates, so:
- `yOf(priceTop) = 0 - 36 = -36` → above the SVG viewport (clipped)
- `yOf(priceBot) = 480 - 36 = 444` → at SVG bottom

The effective visible landscape range is only the subset of `priceBot..priceTop` with y ∈ [0, size.height], which is narrower than the full range. This explains the 54pt observed vs 140pt chart range.

### (d) Root cause — hypothesis (c): combination of both sides

1. **Chart side**: emitted `priceTop/priceBot` doesn't match the chart's actual displayed range. Lightweight-charts adds scale margin padding (10% top + 10% bottom) on top of the range anchor data, so the chart shows a 20–25% wider range than emitted.

2. **Landscape side**: synced `yOf` uses chart-absolute pixel coordinates, clipping `priceTop` above the SVG viewport. Only the lower portion of the price range is visible in the landscape.

**Fix plan:**
1. `MiniPriceChart`: call `chart.priceScale('right').setVisibleRange({ from: priceBot, to: priceTop })` after computing the union range. This pins the chart to exactly the emitted range.
2. `GexLandscape`: replace synced `yOf` with landscape-native coordinates: `PAD.top + ((pHi - price) / (pHi - pLo)) * plotH`. Remove `paneHeight` from the synced check (no longer needed).

### (e) App.jsx current layout rule for anchor-vs-selected

`react_today_setup/src/App.jsx` line 426:
```jsx
<div style={{ flex: 1, minWidth: 0, display: 'flex', gap: 16, flexWrap: 'wrap', alignItems: 'flex-start' }}>
```
No explicit `flex-direction` → defaults to `row`. Anchor wrapper (line 430): `flex: '1 1 380px', minWidth: 320`. Selected wrapper (line 450): `flex: '1 1 380px', minWidth: 320`. Both sit side-by-side in a flex row with `flexWrap: 'wrap'`.

**Layout fix**: change to `flexDirection: 'column'`, remove `flexWrap`, change DayView wrappers to `width: '100%'`.
