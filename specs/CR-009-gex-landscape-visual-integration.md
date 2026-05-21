### Problem

After CR-008 the GEX landscape panel is on the chart, classifications update against live spot, and the LIVE/OPEN toggle works — but the panel reads as a *separate* visualization rather than an integrated extension of the price chart, and the chart's price ruler isn't visible at all. Five gaps:

0. **The chart's right-edge price scale is hidden.** The `GexLandscapePanel` is positioned flush against the right edge of the lightweight-charts canvas and occludes the right price-scale ruler. Numeric price labels are not visible on the chart. This is a CR-008 layout regression — and a hard precondition for the rest of CR-009: items 2 and 4 below render their right-edge labels *on* the price scale via `createPriceLine`, so without item 0 they have no visible label surface.

1. **Y-axis is not synced.** The panel renders its full stored range (~±300pt around `table_spot`); the chart shows only the price window relevant to recent candles. A confluence horizontal line drawn at price X on the panel does *not* appear at the same vertical pixel position as price X on the chart. The user cannot visually trace a confluence from the right panel to the corresponding price level on the candles. This is the headline gap from CR-008's in-browser smoke. See [[gex-landscape-yaxis-sync]] for the failure description.

2. **Confluence lines stop at the panel boundary.** Each confluence is a structural price level the chart user benefits from seeing across the candles, not only in the right panel. Currently they render only inside `GexLandscapePanel`.

3. **Negative-zone walls are visible only as cyan tick marks inside the panel.** The proximate-negative-zone prices (`neg_zones` in the endpoint response) are levels where the user wants to see candle behavior — they should render as thin bands behind the candles on the price chart.

4. **Intraday subtarget has no chart-side rendering.** When `regime` carries an `intraday_subtarget` (a confluence or wall the structural model expects price to drift toward), the price chart has no visual indication. The user has to read the regime chip in the panel header.

### Proposed Solution

**Item 0 — Restore the chart's right-edge price scale.**

The lightweight-charts right `priceScale` is occluded by the `GexLandscapePanel` in the CR-008 implementation — either because `priceScale.visible` was disabled, or because the panel's container is positioned over the chart's full bounds including the scale area. Fix the layout so the price scale renders between the candle area and the panel. The scale becomes the natural visual divider between the two views, and is the surface on which items 2 and 4 render their right-edge labels.

Mechanism is layout-only. No coordinate-sync work in this item — that's item 1. Concretely:

- Confirm `chart.priceScale('right').applyOptions({ visible: true })` (or equivalent on the pinned version) is in effect.
- Adjust the container layout (likely `App.jsx` or a chart-container component) so the panel is positioned to the right of the chart's full extent — chart canvas + its right price scale + a small gap — rather than overlapping the scale.

**Item 1 — Y-axis sync (rendering invariant: panel follows chart's visible range).**

The price chart owns the lightweight-charts instance and is the source of truth for the currently-visible Y-range. `PriceChart.jsx` subscribes to its own right `priceScale` for visible-range changes, lifts the `[min, max]` tuple into `App.jsx` state, and passes it as a `visiblePriceRange` prop to `GexLandscapePanel`. The panel uses the prop as its Y-bounds — replacing the current behavior of using the stored `landscape` array's full price span.

When `visiblePriceRange` is undefined (first render before the subscription has fired), the panel falls back to its current behavior: render against the full stored landscape range. Once the subscription fires, the panel re-renders against the chart's range.

**Rendering invariant to assert in AC:** a horizontal line at price X is drawn at the same vertical pixel position on both the price chart and the panel, when X is within the chart's visible Y-range. Outside that range, the panel clips (structural features above/below the visible window aren't shown; the user can pan/zoom the chart to bring them into view).

**Amendment — pre-implementation review (Item 1 mechanism):** Verified against lightweight-charts 5.1.0 (`react_price_preview/node_modules/lightweight-charts/dist/typings.d.ts`). Three reconciliations:

- **No price-scale visible-range subscription exists.** `IPriceScaleApi` exposes `getVisibleRange()` / `setVisibleRange()` (synchronous) but no `subscribeVisibleRangeChange`. The only range-change events — `subscribeVisibleLogicalRangeChange` / `subscribeVisibleTimeRangeChange` — live on the *time* scale and fire on horizontal interaction only; they do not fire on the vertical price-scale pan/zoom this chart implements via custom handlers (`panPriceByPixels` / `zoomPriceAtY` in `PriceChart.jsx`). Replace "`PriceChart.jsx` subscribes to its own right `priceScale` for visible-range changes" with: **`PriceChart` runs a `requestAnimationFrame` polling loop** (active only while the LANDSCAPE pill is on) that samples the candlestick series' `coordinateToPrice()` each frame and publishes when the value changes.
- **`visiblePriceRange` is an object, not a `[min, max]` tuple.** To make AC #2 / #5 pixel-alignment achievable (handoff checklist point 3), the panel must reproduce the chart pane's *exact* affine price→pixel transform, not merely clamp to a price span. `PriceChart` publishes `{ priceTop, priceBot, paneHeight }` — the prices at pixel `y = 0` and `y = paneHeight` of the price pane, plus the pane height. The panel maps `yOf(price) = ((priceTop - price) / (priceTop - priceBot)) * paneHeight`. Because the default lightweight-charts price scale is linear, two sampled points reproduce the chart's transform exactly; the panel SVG and the chart pane share the `.chart-stage` `top: 0` origin, so the pixel-Y matches. The panel's existing `PAD.top` / `PAD.bottom` Y-insets are bypassed in synced mode (they remain for the undefined-prop fallback).
- **State stays in `PriceChart`, no `App.jsx` round-trip.** `PriceChart` already renders `<GexLandscapePanel>` directly — it is the common parent of both the chart and the panel. `visiblePriceRange` is therefore `PriceChart`-local `useState`, passed straight to `<GexLandscapePanel visiblePriceRange={...}>`. The spec's original "lift into `App.jsx` state" predates confirming the component tree; no `App.jsx` change is needed for Item 1.

**Item 2 — Confluence lines across the price chart.**

For each confluence in `gexLandscapeData.confluences`, draw a horizontal line on the chart's main candlestick series using lightweight-charts' `ISeriesApi.createPriceLine`. Styling mirrors the panel:

- Quality → line style: `pin` solid, `drift` dashed, `soft` dotted
- `n_buckets` → color: 2 = yellow, 3 = orange, 4 = green
- `title` includes star count + quality tag (e.g. `★ × 2 PIN`) so the chart's right-edge price ruler shows the label

Lifecycle: hold the created `IPriceLine` refs in a `useRef` array; on landscape data refresh (LIVE-mode debounced spot delta, date change, or panel toggle off→on), remove the existing lines via `series.removePriceLine(ref)` and recreate from the new payload.

**Item 3 — Negative-zone bands behind candles.**

For each `neg_zones[i]`, render a thin horizontal band at `neg_zones[i].price` on the price chart, behind the candles. Semi-transparent cyan (matching the panel's neg-wall tick mark color), band height ±2pt around the price.

lightweight-charts does not natively support filled price bands, so this requires a small SVG overlay layer positioned absolutely over the chart container, with band Y-positions derived from `chart.priceScale('right').priceToCoordinate(price)`. The overlay sits between the chart canvas and any pointer-event capture, so it doesn't interfere with crosshair / tooltip / drag / scroll-wheel zoom.

Re-position on chart visible-range change (same subscription used by item 1).

**Item 4 — Intraday subtarget annotation.**

When `gexLandscapeData.intraday_subtarget` is non-null, render an annotated marker on the price chart at `intraday_subtarget.price`. Implementation via `createPriceLine` with:

- Distinctive color — pale green for above-spot drift target, pale red for below-spot (derive direction from current spot vs subtarget price)
- Dotted line style, slightly thicker than soft-confluence lines
- `title` = `→ {price} {type}` (e.g. `→ 7452 confluence (★ × 2)`)

When `intraday_subtarget` is null or absent, no annotation renders. Same lifecycle as confluence lines (clean up and recreate on data refresh).

### Affected Files

- `react_price_preview/src/components/PriceChart.jsx` — restore the right price scale by shrinking the `.chart-host` width when the panel is open, so the scale renders left of the panel (item 0); run the `requestAnimationFrame` poll that publishes `visiblePriceRange` and hold it as local state (item 1); host `createPriceLine` calls for confluence lines (item 2) + subtarget annotation (item 4); mount the SVG overlay for neg-zone bands (item 3); pass `visiblePriceRange` to `<GexLandscapePanel>`.
- `react_price_preview/src/components/GexLandscapePanel.jsx` — accept `visiblePriceRange` prop; consume for Y-bounds with fallback to full landscape range when prop is undefined.
- `react_price_preview/src/App.jsx` — **no change required.** `landscapeData` is already passed to `PriceChart`; the Item 1 amendment keeps `visiblePriceRange` state inside `PriceChart` (the common parent of the chart and the panel), so no App-level piping is needed. The item-0 layout fix lives in `PriceChart.jsx`, which owns the `.chart-host` container and renders the panel.

A new `react_price_preview/src/components/LandscapeChartOverlay.jsx` factoring out the SVG overlay layer (and optionally hosting `createPriceLine` orchestration) is acceptable if the pre-implementation review prefers it for separation of concerns. Not required.

No backend changes expected. No new Python files. No new endpoint params. No new npm packages expected.

### Acceptance Criteria

**Layout precondition (item 0):**

1. The lightweight-charts right `priceScale` is visible and renders numeric price labels between the chart canvas and the `GexLandscapePanel`.
2. Price labels on the scale update on chart pan/zoom and accurately reflect the visible Y-range.
3. The `GexLandscapePanel` is positioned to the right of the price scale; no DOM overlap between the panel and the chart's scale; no visual collision.

**Y-axis sync (item 1):**

4. When the LANDSCAPE pill is on, panning or zooming the price chart causes the panel's curves, spot line, confluence lines, and negative-wall ticks to re-render against the chart's new visible Y-range.
5. The spot dashed line sits at the same vertical pixel position on both the price chart and the panel. The same invariant holds for any confluence price drawn on both sides, when the price is within the chart's visible Y-range.
6. Structural features (peaks, confluences, neg walls) outside the chart's visible Y-range are clipped on the panel without visual artifacts at the panel edges.
7. Before the chart's visible-range polling loop has published its first value (initial mount, pre-interaction), the panel renders against the full stored landscape range as a sensible default. Once the loop publishes, the panel re-renders against the chart's range.

**Confluence lines extended (item 2):**

8. Each entry in `confluences` renders as a horizontal price line on the price chart, with line style by quality (pin solid / drift dashed / soft dotted) and color by `n_buckets` (2 = yellow / 3 = orange / 4 = green).
9. Confluence labels (`★ × N <quality>`) appear on the chart's right-edge price ruler (the same scale restored in item 0).
10. Lines are removed and recreated on landscape data refresh (LIVE-mode spot delta, date change, panel toggle off→on).
11. When the LANDSCAPE pill is off, no confluence price lines are present on the chart.

**Neg-zone bands (item 3):**

12. Each entry in `neg_zones` renders as a thin semi-transparent cyan horizontal band on the price chart at the entry's `price`, behind the candles.
13. Bands re-position on chart pan/zoom such that the same `price` always maps to the same pixel-Y on the chart.
14. Bands are removed when the panel is toggled off or when a new landscape payload arrives without that entry.
15. Bands don't interfere with chart pointer events — crosshair, tooltip, drag, and scroll-wheel zoom all work normally over band areas.

**Intraday subtarget annotation (item 4):**

16. When `intraday_subtarget` is non-null in the payload, an annotated price line renders on the chart at `intraday_subtarget.price`, with the `title` including the subtarget type.
17. When `intraday_subtarget` is null or absent (e.g., neutral regime), no annotation renders.
18. The annotation color reflects drift direction (above-spot vs below-spot subtarget) and is visually distinct from confluence lines.

**Visual integration (cross-cutting):**

19. All chart annotations (confluence lines, neg bands, subtarget) appear when the LANDSCAPE pill is on and disappear cleanly when it's off — no orphan refs, no lingering DOM, no console warnings.

### Verification

**Automated:**

- `vite build` passes from `react_price_preview/`.
- Any existing JS test suite passes.
- Backend test suite remains at 165/165 passing (no backend changes expected — if any backend change is required, the spec is wrong and should be amended).

**In-browser smoke (against `date=2026-05-20`, the canonical Phase 0 reference day):**

- Pre-merge run from the worktree (Flask serving the worktree's React bundle + worktree's Python — same pattern as CR-008 in-browser smoke; remember `npm install` in the worktree first).
- **First — confirm the chart's right-edge price scale is visible**, with readable numeric labels between the candle area and the panel. If this fails, item 0's layout fix didn't land and the rest of the smoke can't proceed meaningfully (items 2 and 4 render their labels on this surface).
- LANDSCAPE pill on → panel opens with Y-range matching the chart's current visible price window. Spot dashed line at the same pixel-Y on both sides.
- Pan the chart up and down — panel re-renders at each frame; spot line stays aligned across panels.
- Zoom in on a narrow Y-range — panel re-renders to the narrow range; off-range structural features clip cleanly.
- Confluence lines render across the chart with correct line style + color + label. Verify the 5/20 confluences (7505 and 7456 from CR-008's smoke) appear with the right quality tags.
- Neg-zone bands visible behind candles at the proximate-neg price (5/20 had a neg zone at 7343 — should render as a cyan band at that level).
- Subtarget annotation visible — 5/20's regime is `MAGNET ABOVE → 7621` post-CR-008, so the chart should show a `→ 7621` annotation.
- Toggle LANDSCAPE off → all chart annotations disappear; no console errors.
- Toggle LIVE / OPEN — the spot-mode toggle still works; LIVE-mode debounced refresh continues to update everything (panel curves, confluence lines, neg bands, subtarget).

### Out of Scope

Explicitly deferred to Phase 2:

- Yesterday's landscape as a ghost outline (time-evolution viz)
- Time-evolving landscape animation through the trading day
- Heatmap halo behind candles (vs the thin neg-band approach chosen here)
- Containment zone bands spanning both panels
- Live intraday landscape recomputation from `orats_oi_gamma` at request time (Phase 1 is still EOD-only for the landscape field itself; only spot-dependent classifications update intraday)
- `accuracy=high` endpoint query param for grid-recentering on analytical spot — would enable bucket-level 30+ DTE peak detection through the endpoint; documented limitation from CR-008

### Handoff prompt (implementation phase)

> **You are implementing CR-009 — GEX Landscape Visual Integration.** The spec is in `specs/CR-009-gex-landscape-visual-integration.md`. Branch `feat/CR-009-gex-landscape-visual-integration` off `Main-Live`.
>
> **Step 0 — Read these in full before writing any code:**
> - `specs/CR-009-gex-landscape-visual-integration.md` (this spec)
> - `react_price_preview/src/components/PriceChart.jsx`
> - `react_price_preview/src/components/GexLandscapePanel.jsx`
> - `react_price_preview/src/App.jsx`
> - `specs/CR-008-gex-landscape-delivery-layer.md` (predecessor — context on the GEX landscape data shape and the existing panel/chart integration surface)
>
> **Step 1 — Pre-implementation review. Run the contradiction-stop checklist before writing any code.** For every contradiction found, amend the spec in its own commit (`CR-009: spec amendment — <what was reconciled>`) before implementing. Do NOT silently work around a spec mismatch.
>
> 1. **lightweight-charts API surface — confirm against the pinned version in `react_price_preview/package.json`.** Inspect `node_modules/lightweight-charts/dist/typings.d.ts` directly. Specifically:
>    - Does the pinned version expose a subscription mechanism for right-`priceScale` visible-range changes? If `IPriceScaleApi.subscribeVisibleRangeChange` (or equivalent) does not exist, identify the correct mechanism — could be `IChartApi.subscribeCrosshairMove` plus polling, a different event hook, or `ResizeObserver` plus manual range polling. Amend the spec's Item 1 implementation note if the actual API requires a different approach than "subscribe and lift".
>    - Does `ISeriesApi.createPriceLine` support the four line styles needed: solid, dashed, dotted (for soft confluences), and a distinct dotted-thicker style for the subtarget? If dotted isn't directly supported on this version, identify the closest available style and amend the spec.
>    - For the neg-zone band overlay, confirm that an absolutely-positioned SVG sibling of the lightweight-charts container does not interfere with pointer-event handling. If it does, identify the correct overlay strategy (e.g., `customSeriesView` in v4+, or canvas overlay into the same parent) and amend the spec.
>
> 2. **package.json + node_modules — verify no new dependencies are needed.** This CR is frontend-only and shouldn't require new packages. If pre-impl review surfaces a need (e.g., a coordinate-sync helper), amend the spec and `package.json` in separate commits.
>
> 3. **AC empirical viability — confirm AC #2 (pixel alignment of horizontal lines at price X between chart and panel) is achievable with the chosen API surface.** If lightweight-charts' price-to-coordinate function and the panel's SVG/canvas coordinate model can't both be driven from the same `[min, max]` range to produce identical pixel-Y for a given price, identify what's needed — e.g., the panel may need to consume `priceScale.priceToCoordinate` directly rather than doing its own remapping. Amend the spec if so.
>
> 4. **Re-verify the structural facts CR-008 caught the hard way:**
>    - `scipy` is now in `apps/cron/requirements.txt` and `apps/web/requirements.txt`. No new Python imports expected in this CR, but if any get added in a one-off helper, re-check requirements.
>    - The Y-axis sync invariant chosen is **Option 1** (panel follows chart). Options 2 and 3 from [[gex-landscape-yaxis-sync]] have been ruled out. Do not re-litigate.
>
> **Step 2 — Implementation.** Once the spec is reconciled with reality, implement per the Proposed Solution section. Suggested commit slicing — each commit a working state:
>
> 1. Layout precondition: restore the chart's right-edge price scale (item 0). Pure layout/visibility fix; no coordinate work.
> 2. Y-axis sync subscription + state lift + panel prop consumption (item 1).
> 3. Confluence lines on chart via `createPriceLine` (item 2).
> 4. Neg-zone bands via SVG overlay (item 3).
> 5. Intraday subtarget annotation (item 4).
>
> If a piece can't be cleanly isolated, bundle into the prior commit with a note in the message.
>
> **Step 3 — Verification.** Run `vite build`, the JS test suite if present, and confirm the backend test suite is unaffected (it should be — no backend changes). Then write the manual verification checklist into the PR description, walking through the Acceptance Criteria one by one. In-browser smoke requires running Flask from the worktree directory to serve the worktree's React bundle (lesson from CR-008). Remember `npm install` is per-worktree.
>
> **Step 4 — PR.** Open against `Main-Live`. Title: `CR-009 — GEX Landscape Visual Integration`. Body: brief summary + the manual AC checklist + links to the spec file and the CR-008 predecessor. Use `Create a merge commit` (not Squash) to preserve the spec → amendments → implementation commit narrative.
