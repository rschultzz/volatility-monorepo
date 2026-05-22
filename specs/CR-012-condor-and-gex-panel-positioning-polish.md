# CR-012 — Condor and GEX Panel Positioning Polish

### Problem

Two chart-overlay panels carry finish-grade default-positioning defects, each tracked
in `open-questions/` since an earlier CR:

- The `CondorPricingPanel` (shipped in CR-006) defaults to `top: 8px; right: 8px` inside
  the chart, which places it over the price axis and occludes the y-axis price labels.
  It is not draggable, so the user cannot move it off the labels. Tracked in
  `condor-pricing-panel-ui`.
- The GEX legend popup in `PriceChart.jsx` defaults to a right-edge anchor that, when the
  LANDSCAPE pill is also on, lands inside the docked landscape-panel column. Tracked in
  `gex-legend-popup-default-position`.

Both are the same problem class — a chart-overlay panel's default anchor against a chart
container that can shrink ~300px when the LANDSCAPE pill is on (CR-009). They are bundled
into one CR per CR-010's wrap-up note. The condor panel additionally needs drag-with-
persistence added; the GEX legend popup is already draggable and only its default opening
position changes.

This is a frontend-only CR. No backend, endpoint, or DB changes.

### Goal

Three things, in priority order:

1. **Condor panel default anchor at top-right, inboard of the price axis** so the y-axis
   labels remain visible whenever the condor pricing panel is open.
2. **Condor panel draggable, with position persisted across sessions**, and a reset
   affordance for returning to default. A user-dragged position preserves its horizontal
   offset from the price axis whether the LANDSCAPE pill is open or closed (the same
   offset from the price axis on both layouts).
3. **GEX legend popup default anchor at top-left, below the row of pills/buttons.** The
   popup remains draggable as today; only the default opening position changes.

Constraint: with the LANDSCAPE pill off, no observable change from pre-CR-012 behavior for
either panel except the condor default position itself. Match CR-009's panel-open-specific
scoping discipline.

### Item 1 — Condor panel default position

- New default anchor: top-right of the chart container, with a horizontal offset large
  enough to clear the price-axis label column. Vertical offset: a small inset below the
  top edge of the chart.
- Implementation note: position the panel via `right: Xpx; top: Ypx` (right-anchored)
  inside the chart container rather than `left` / `top`. The price axis sits at the chart
  container's right edge; when CR-009's `.chart-host` shrinks ~300px to make room for the
  LANDSCAPE column, a right-anchored child element automatically follows the new right
  edge with no extra logic. This is what makes Item 2's "preserve relationship to price
  axis across LANDSCAPE toggle" requirement satisfiable with the same code path the
  default uses.

### Item 2 — Condor panel drag-with-persistence

- Click-and-drag enabled. A small drag handle in the panel header (rather than
  drag-anywhere) is preferred, but the implementation can pick whichever feels right.
- Persist position to `localStorage` under a new key. Mirror the `FLOW_HEIGHT_STORAGE_KEY`
  / `FLOW_EMA_MINUTES_STORAGE_KEY` patterns already in `App.jsx`. Suggested key:
  `CONDOR_PANEL_POSITION_STORAGE_KEY`.
- Persisted payload: `{ rightOffset: number, topOffset: number }` (right-anchored offsets,
  not `{ x, y }` absolute coords). This is what gives "preserve relationship to price axis
  across LANDSCAPE toggle" for free — the offsets are measured from the chart container's
  right/top edges, which already shift when LANDSCAPE opens.
- Drag mechanics: reuse the `dragStateRef` + window-level mousemove/mouseup pattern
  already present in `App.jsx` for the flow-panel resize. Clamp dragging to chart
  container bounds (don't allow the panel to be dragged offscreen — offscreen-then-reset
  is annoying).

### Item 3 — Reset affordance on condor panel

- Small ↺ icon in the condor panel header.
- Click clears the persisted position from `localStorage` and returns the panel to the
  default anchor from Item 1.
- No confirmation prompt — it's reversible (user just re-drags).

### Item 4 — GEX legend popup default position

- New default anchor: top-left of the chart container, below the row of pills/buttons at
  the top of the chart.
- The implementation will need to identify the height of the button row in
  `PriceChart.jsx` (the relevant default-anchor logic lives around line 3411 per CR-009's
  investigation in `gex-legend-popup-default-position`) and pick a top offset that clears
  it with a small inset.
- No drag/persistence changes — the popup is already draggable; only the default opening
  position is changing.

### Acceptance Criteria

1. With the chart open and no persisted condor position in `localStorage`, opening the
   condor pricing panel anchors at top-right with the full price-axis label column visible
   to its right.
2. Dragging the condor panel by its handle moves it; releasing leaves it at the dropped
   position.
3. The condor panel's position cannot be dragged outside the chart container's bounds.
4. After dragging, the condor panel's position persists across page reloads via
   `localStorage`.
5. The condor panel's reset (↺) icon clears the persisted position and returns the panel
   to its default anchor.
6. With the condor panel at any position (default or user-dragged), toggling the LANDSCAPE
   pill on/off preserves the panel's horizontal offset from the price axis. (Verification:
   drag the condor panel to ~200px from the price axis; toggle LANDSCAPE; the panel should
   still sit ~200px from the price axis in the new layout.)
7. With LANDSCAPE pill off and no persisted condor position, the condor panel default
   position is identical to AC #1 (chart container's right edge is the price axis itself
   in this layout).
8. Opening the GEX legend popup with no persisted position anchors at top-left, below the
   row of pills/buttons at the top of the chart, with the button row remaining fully
   visible above it.
9. The GEX legend popup remains draggable as today (no regression).
10. No regression to existing flow-panel resize/drag interactions or to the LANDSCAPE
    pill's open/close behavior.

### Affected Files

- `react_price_preview/src/components/CondorPricingPanel.jsx` — drag handlers, reset
  button, header drag handle, position-prop wiring.
- `react_price_preview/src/App.jsx` — `localStorage` state for the condor position +
  default-position constant + condor-position passing into `CondorPricingPanel`.
- `react_price_preview/src/components/PriceChart.jsx` — GEX legend popup default-position
  constant (around line 3411 per CR-009 investigation); possibly a chart-container ref
  export if needed for boundary clamping.

### Decisions (settled before drafting)

- **Right-anchored positioning for the condor panel.** Storing position as
  `{rightOffset, topOffset}` rather than `{x, y}` gives "preserve relationship to price
  axis across LANDSCAPE toggle" for free, since the chart container's right edge is the
  price axis and `.chart-host` shrinks symmetrically when LANDSCAPE opens. Alternative
  considered: absolute `{x, y}` coords with a LANDSCAPE-toggle event listener that
  translates the saved position; rejected as more code and more event-listener fragility
  for the same outcome.
- **GEX legend popup gets default-position change only, not drag/persistence rework.**
  It's already draggable. Persistence for it is an optional follow-up; deferred for now.
- **Reset affordance is in scope for the condor only.** Matches what the open question
  proposed. The GEX legend popup doesn't get a reset button this CR — it's already
  draggable and its new default is presumably good.
- **Boundary clamping is in (drag stays inside chart container bounds).** Offscreen-then-
  reset is more annoying than not being able to drag off the chart.

### Open Questions

- **Button-row height for Item 4.** The implementation needs the height of the row of
  pills/buttons at the top of the chart — either hard-code an offset or measure the button
  row's `getBoundingClientRect()` at mount time. Decided during pre-implementation review;
  see Amendments.
- **Drag from header only vs drag-anywhere.** Header-only is cleaner (no accidental drags
  when clicking panel content); drag-anywhere is more forgiving on small panels. Decided
  during implementation against the existing draggable-panel affordance.
- **Migration of persisted offsets across breaking layout changes.** If a future CR
  materially changes the chart container's dimensions, a persisted `{rightOffset,
  topOffset}` could land outside the new bounds. Deferred — the reset button is the
  user-facing escape hatch, and clamping the persisted position to current bounds on load
  is a cheap additional safeguard worth adding.

### Pre-implementation review amendments

Reading `CondorPricingPanel.jsx`, `PriceChart.jsx`, `App.jsx`, and `ChartToggleBar.jsx`
against the drafted spec surfaced four reconciliations. Each is recorded here per the
CR-008/009/010/011 amendment-thread discipline; the original draft text above is left
intact so the reconciliation is visible.

**Amendment 1 — Condor panel positioning context: `.chart-stage` vs `.chart-host`.**
Item 1 claims a right-anchored child "automatically follows the new right edge with no
extra logic" when `.chart-host` shrinks. That is wrong about the DOM. `<CondorPricingPanel>`
renders as a child of `.chart-stage` (`PriceChart.jsx:2722`), which spans the full chart
width and does **not** shrink when the LANDSCAPE pill is on. Only `.chart-host` — the
lightweight-charts mount node (`hostRef`) — shrinks: its width is
`calc(100% - LANDSCAPE_PANEL_WIDTH)` when LANDSCAPE is open (`PriceChart.jsx:4074`). The
docked `GexLandscapePanel` occupies the rightmost `LANDSCAPE_PANEL_WIDTH` (300px) of
`.chart-stage`. The price-axis label column sits at `.chart-host`'s right edge — confirmed
no CSS scrollbar or other element between the candle area and the price axis.

Reparenting the condor panel into `.chart-host` is not viable — lightweight-charts owns
that node's DOM.

Reconciliation: the panel stays a child of `.chart-stage`. The persisted `rightOffset` is
measured from the price axis (`.chart-host`'s right edge). The rendered CSS `right` value
(relative to `.chart-stage`) is computed at render time:

```
right = rightOffset + (landscapeOpen ? LANDSCAPE_PANEL_WIDTH : 0)
```

The `+ LANDSCAPE_PANEL_WIDTH` term is the "extra logic" Item 1 said was unnecessary. It is
one conditional term in the style object. AC #6 and AC #7 still hold: with LANDSCAPE off
the term is 0 and `right` is the price-axis offset directly; with LANDSCAPE on the term
shifts the panel left by exactly the landscape-column width, so its distance from the
price axis is unchanged. `LANDSCAPE_PANEL_WIDTH` is the existing `PANEL_WIDTH` (300)
re-exported from `GexLandscapePanel.jsx` and already imported into `PriceChart.jsx`.

**Amendment 2 — Condor panel position state lives in `PriceChart.jsx`, not `App.jsx`.**
The drafted Affected Files entry places the condor position state in `App.jsx`.
`<CondorPricingPanel>` is rendered in `PriceChart.jsx:2722` — `App.jsx` only passes the
`condorPricing` payload down to `<PriceChart>`, which renders the panel. The position
state, `CONDOR_PANEL_POSITION_STORAGE_KEY`, the default-position constant, the drag
handler, and the reset all belong in `PriceChart.jsx`, mirroring the GEX legend panel's
existing `gexPanelPos` state and `ib-react-gex-panel-pos` persistence already in that
file. `App.jsx` requires no change. This mirrors CR-009's amendment ("state stays in
`PriceChart`, no `App.jsx` round-trip"). `CondorPricingPanel.jsx` stays presentational —
it receives the computed position style plus `onHandleMouseDown` / `onResetPosition`
callbacks.

Affected Files corrected: `App.jsx` is **removed** from the list. The two edited files are
`CondorPricingPanel.jsx` and `PriceChart.jsx`.

**Amendment 3 — Pattern to mirror is `gexPanelDragRef`, not `dragStateRef`.**
Item 2 says to mirror `App.jsx`'s `dragStateRef` flow-panel resize pattern. That pattern
is a 1-D *resize* (height delta only, persistent window listeners in a `useEffect`).
`PriceChart.jsx` already contains a complete 2-D draggable-overlay-panel implementation —
the GEX legend panel's `gexPanelDragRef` (`PriceChart.jsx:3447-3493`): `onMouseDown`
captures the start cursor and start panel position, adds window `mousemove`/`mouseup`
listeners for the duration of the drag, computes clamped offsets in `mousemove`, and tears
the listeners down in `mouseup`. That is the closer pattern — same file, same interaction
class — and the condor drag handler mirrors it. The window-level `mousemove`/`mouseup`
mechanism the spec named is identical; the GEX legend version is the 2-D specialization
with clamping already worked out. Clamping is done at drag time against `.chart-host`'s
bounds captured at `mousedown` (satisfies AC #3); on-load re-clamping of stale persisted
offsets stays deferred per the third Open Question (reset button is the escape hatch).

`CONDOR_PANEL_POSITION_STORAGE_KEY` value: `'ib-react-condor-panel-pos'`, matching the
`ib-react-gex-panel-pos` / `ib-react-flow-panel-height` naming family.

**Amendment 4 — GEX legend popup is the GEX legend *panel*; button-row offset is hard-coded.**
The "GEX legend popup" is the GEX legend panel block gated by `gexPanelOpen`
(`PriceChart.jsx:3422`). Its default-anchor logic is at `PriceChart.jsx:3430` —
`{ top: '8px', right: PRICE_AXIS_HIT_WIDTH + 8, bottom: TIME_AXIS_HEIGHT + 8 }`, a
right-anchored default against `.chart-stage`. Because `.chart-stage` does not shrink
(Amendment 1), that right anchor lands inside the landscape column when LANDSCAPE is on —
exactly the defect Item 4 fixes. Item 4 changes only the `: { ... }` default branch of the
`gexPanelPos ? ... : ...` ternary; the dragged-position branch and the persistence
(`ib-react-gex-panel-pos`) are untouched, so AC #9 (still draggable) holds for free.

New default: left-anchored — `{ top: GEX_LEGEND_DEFAULT_TOP, left: GEX_LEGEND_DEFAULT_LEFT }`.
Being left-anchored, it does not overlap the landscape column regardless of LANDSCAPE
state, so no LANDSCAPE-conditional term is needed for it.

Button-row height is **hard-coded**, not measured. The top-of-chart controls are
fixed-size literals already in the codebase: the settings gear button is `top: 8px`,
`height: 44px` (`PriceChart.jsx`), and `ChartToggleBar` pills are `top: 8`, `height: 32px`
(`ChartToggleBar.jsx`). The button row's bottom edge is therefore a compile-time constant
(`8 + 44 = 52`, the gear being the taller element). Measuring `getBoundingClientRect()` at
mount would add a ref + layout effect + resize re-measure for a value that never changes.
`GEX_LEGEND_DEFAULT_TOP = 56` (gear bottom + 4px inset) clears the row;
`GEX_LEGEND_DEFAULT_LEFT = 12` aligns with the settings gear's left inset. This matches
how `PRICE_AXIS_HIT_WIDTH` / `TIME_AXIS_HEIGHT` are already hard-coded module constants.

**Drag affordance decision (Open Question resolved): header-only.** The condor panel's
`condor (1σ)` title row becomes the drag handle, matching the GEX legend panel (drags from
its header). The panel root currently sets `pointerEvents: 'none'` so chart interactions
pass through; the header handle and the reset button set `pointerEvents: 'auto'` to
receive their own events, while the rest of the panel body keeps `pointerEvents: 'none'`.

**AC corrections:**

- **AC #8** — "below the row of pills/buttons" is satisfied by `GEX_LEGEND_DEFAULT_TOP = 56`,
  clearing the 44px settings gear (the tallest top-left control) plus a 4px inset.
- **Affected Files** — `App.jsx` is removed (Amendment 2). The edited files are
  `CondorPricingPanel.jsx` and `PriceChart.jsx`.

### Verification

**Automated:**

- `vite build` from `react_price_preview/` passes.

**In-browser smoke (manual):**

- Walk through each of AC #1 through #10.
- Drag the condor panel with LANDSCAPE off, toggle LANDSCAPE on, confirm the panel's
  offset from the price axis is preserved.
- Click the reset button; refresh the page; confirm the persisted position is cleared.
- Toggle the GEX legend popup and verify it opens at top-left below the buttons.
- Confirm no regression to existing flow-panel resize/drag and LANDSCAPE pill behavior.

**Tests:** Unit tests for the drag math are skipped — DOM event handling is awkward to
mock and the math is trivial. The open-question note explicitly flagged manual e2e as the
verification path.

### Out of Scope

- GEX legend popup drag/persistence rework — it is already draggable, and position
  persistence for it already exists. Only the default opening position changes.
- A reset button for the GEX legend popup.
- Migration/reset of the condor panel's persisted offsets across future breaking layout
  changes (see Open Questions).
