# CR-018 — Chart-Driven Shared Y Axis (today-setup)

## Problem

CR-017's axis-sharing approach is structurally different from the proven pattern on the main `/` page.

**CR-017 (today-setup) architecture — data-driven, both sides pinned:**

- `MiniPriceChart` computes `{ priceBot, priceTop }` from the union of bar highs/lows + cluster centers + 4% padding.
- It pins the chart to that range via `chart.priceScale('right').setVisibleRange({ from: priceBot, to: priceTop })`.
- CR-017 hotfix 1 added `scaleMargins: { top: 0, bottom: 0 }` to force the chart's displayed range to equal the data range.
- It emits the same range via `onPriceRangeChange` to `DayView`, which forwards it to `GexLandscape` as `visiblePriceRange`.
- `GexLandscape` renders its curves using its own SVG coordinate space (`PAD.top + ((pHi - price) / (pHi - pLo)) * plotH`), independent of the chart's pixel layout.
- Two independent coordinate systems displaying the same numbers. Cluster lines on the chart and labels on the landscape do NOT visually align even when ranges match.
- Custom wheel handlers on both components round-trip through React state to keep ranges in sync. Chart's native zoom is disabled.

**Main page (`react_price_preview/PriceChart.jsx`) architecture — chart-driven, landscape conforms:**

- Chart runs with default `scaleMargins` and `autoScale`. The user sees the chart's native axis labels.
- Every animation frame, `PriceChart.jsx` lines ~1916–1948 polls `series.coordinateToPrice(0)` and `series.coordinateToPrice(paneHeight)` to read the chart's *actual displayed* top and bottom prices. Publishes `visiblePriceRange = { priceTop, priceBot, paneHeight }`.
- `GexLandscape` consumes `visiblePriceRange` and renders using chart-pixel coordinates.
- Chart's native zoom handles wheel + drag on the price axis. The rAF poll picks up the new range every frame; landscape follows automatically.

The main-page pattern is the right one. This CR brings today-setup to that pattern.

## Goal

Three deliverables:

1. **Chart-driven coordinate system.** `MiniPriceChart` owns the price scale. A per-frame `coordinateToPrice` poll reads the true displayed range and publishes `{ priceTop, priceBot, paneHeight }` to `DayView`.

2. **Landscape conforms via chart-pixel coordinates.** `GexLandscape`'s synced `yOf` uses `paneHeight` and maps prices directly to chart-pixel space — no independent SVG plot area.

3. **Bidirectional native zoom.** Mouse-wheel on the chart zooms natively (rAF poll picks it up, landscape follows). Mouse-wheel on the landscape calls the chart's `setVisibleRange` directly via a forwarded chart ref.

## Changes

Pure frontend. No backend or DB changes.

**`packages/web-shared/src/MiniPriceChart.jsx`:**
- Remove `chart.priceScale('right').setVisibleRange(...)` pin (CR-017).
- Remove `scaleMargins: { top: 0, bottom: 0 }` from `rightPriceScale` (CR-017 hotfix 1) — revert to default.
- Re-enable native zoom: `handleScroll: { vertTouchDrag: true }`, `handleScale: { axisPressedMouseMove: { price: true, time: false }, mouseWheel: true, pinch: true }`.
- Remove the custom outer-div wheel handler.
- Remove the `externalRange` prop and its useEffect handler.
- Add a rAF polling loop: `coordinateToPrice(0)` → `priceTop`, `coordinateToPrice(paneHeight)` → `priceBot`, emit `{ priceTop, priceBot, paneHeight }` via `onPriceRangeChange`.
- Forward a chart ref via `useImperativeHandle` so `DayView`/`GexLandscape` can call `chart.priceScale('right').setVisibleRange(...)`.
- Keep the `rangeAnchor` LineSeries with hidden points at `priceBot`/`priceTop` for auto-scale coverage.

**`packages/web-shared/src/GexLandscape.jsx`:**
- Restore chart-coordinate `yOf`: `(price) => ((priceTop - price) / (priceTop - priceBot)) * paneHeight` (no `PAD.top` offset).
- Restore `paneHeight` requirement in `synced` check.
- Wheel handler: call `chart.priceScale('right').setVisibleRange(...)` directly on the forwarded chart ref instead of calling `onRangeChange`.

**`react_today_setup/src/components/DayView.jsx`:**
- Restore `paneHeight` in `visiblePriceRange` forwarded to `GexLandscape`.
- Receive chart ref from `MiniPriceChart` and forward to `GexLandscape`.

## Acceptance criteria

1. Visual alignment — cluster at P appears at same y-pixel on chart and landscape. Verified on 2026-05-22 anchor.
2. Native chart Y-axis labels shown (auto-generated).
3. Native chart zoom — wheel/drag zooms price axis; landscape follows in real time.
4. Landscape-initiated zoom — wheel over landscape calls chart's `setVisibleRange` directly.
5. Date change resets to new day's natural range.
6. Anchor and selected views are independent.
7. `vite build` clean for both apps.
8. Main-page parity check.

## Verification plan

1. Reload `/today-setup?date=2026-05-22`. Confirm native Y-axis labels and cluster alignment.
2. Wheel-scroll over chart body. Landscape follows; time axis doesn't shift.
3. Drag on price-axis column. Lock-step.
4. Wheel-scroll over landscape. Chart Y zooms; landscape follows.
5. Click 2026-05-21 analogue. Selected resets; anchor untouched.
6. Compare with main page (`/`).
7. `vite build` both apps.

## Out of scope

- Landscape title regime label inconsistency
- X-axis sharing
- Cross-day Y-axis sharing
- Persisting zoom state across navigations

---

## Step-0 Diagnosis Findings

**Branch:** `feat/CR-018-chart-driven-shared-y-axis` off `292480a` (Main-Live tip after CR-017 merge).

### (a) visiblePriceRange emission path in PriceChart.jsx

Lines 1915–1949 — rAF polling useEffect, triggered when `landscapeOpen` becomes true:

```js
const paneHeight = getPlotHeight(stage)          // line 1926
const priceTop = series.coordinateToPrice(0)      // line 1927
const priceBot = series.coordinateToPrice(paneHeight) // line 1928
...
last = { priceTop, priceBot, paneHeight }
setVisiblePriceRange(last)                        // line 1941
```

`visiblePriceRange` React state flows directly to `<GexLandscape visiblePriceRange={visiblePriceRange} />` at line 2915.

### (b) paneHeight derivation

`getPlotHeight(container)` at lines 286–288:
```js
return Math.max(80, container.clientHeight - TIME_AXIS_HEIGHT)
```
- `TIME_AXIS_HEIGHT = 24` (line 28 of PriceChart.jsx)
- `container` = `stageRef.current` = the `.chart-stage` DOM element (a `position: relative` div)

In MiniPriceChart (today-setup), the equivalent is `containerRef.current.clientHeight - 24`, where `containerRef` is the chart's mount div (full height of the chart wrapper).

### (c) handleScale v5.2 API confirmation

Installed: `lightweight-charts@5.2.0` (both react_today_setup and react_price_preview).

From `react_today_setup/node_modules/lightweight-charts/dist/typings.d.ts`:
- `HandleScaleOptions.mouseWheel: boolean` (line 1270)
- `HandleScaleOptions.pinch: boolean` (line 1276)
- `HandleScaleOptions.axisPressedMouseMove: AxisPressedMouseMoveOptions | boolean` (line 1280)
- `AxisPressedMouseMoveOptions = { time: boolean; price: boolean }` (lines 575–588)

Correct v5.2 usage:
```js
handleScale: { axisPressedMouseMove: { price: true, time: false }, mouseWheel: true, pinch: true }
handleScroll: { vertTouchDrag: true }
```

### (d) Vertical offset between chart plot-area top and landscape plot-area top in DayView

Chart (MiniPriceChart): lightweight-charts coordinate y=0 maps to the TOP of the chart container element (no internal top padding offset). Chart container top = DayView flex row top.

Landscape (GexLandscape): the component renders `position: absolute, top: 0` in its container (same top as chart). But the SVG body (where curves render) begins BELOW the landscape header. Header styles: `padding: '8px 10px'`, `borderBottom: '1px solid'`, flex content ~20–24px tall → header height ≈ 37–40px.

**Estimated offset: ~37–40px** (landscape body is that many pixels lower than chart's y=0).

With CR-018's new `yOf = ((priceTop - price) / span) * paneHeight` (no PAD.top), yOf(priceTop) = 0 = SVG body y=0 = ~37–40px from flex row top. Chart y=0 = 0px from flex row top. Net misalignment = ~37–40px.

This is the SAME offset that exists on the main page (GexLandscape is an absolute overlay there too; its header pushes the body down). Step 5 will measure the actual header height and fix with CSS on the chart wrapper in DayView (paddingTop = landscapeHeaderHeight on chart wrapper, with a matching height reduction).

### (e) MiniPriceChart removal line numbers

- **scaleMargins override** (`{ top: 0, bottom: 0 }`): line 65 (inside `rightPriceScale` options)
- **`setVisibleRange` pin**: line 178 (`chartRef.current?.priceScale('right').setVisibleRange(...)`)
- **`externalRange` prop**: line 23 (prop declaration), lines 187–199 (useEffect applying it)
- **custom wheel handler** (outerRef + event listener): lines 204–229

All four removals confirmed. No gate blockers.
