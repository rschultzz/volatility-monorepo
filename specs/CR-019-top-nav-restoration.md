# CR-019 — Top Nav Restoration

## Problem

The Surface Dynamics top nav used to be three evenly-spaced pills produced
by `dcc.Tabs` taking 100% width of its block-level wrap. Adding Today's
Setup required seating a fourth element next to `dcc.Tabs`, so the wrap
was changed to `display: flex`. In flex layout `dcc.Tabs` no longer
expands to fill the row — it shrinks to the content width of its three
internal tabs. The fourth link sits to the right of that cluster. Net
result: nav is left-clustered and visually broken.

Separately, the Today's Setup React page only exposes a single
"← Dashboard" link in its header (`react_today_setup/src/App.jsx` near
the end of the header `<div>`). To navigate to Price Chart or Backtests,
the user has to land on the dashboard first and click the right tab.
There's no equivalent of the Surface Dynamics top bar on this page.

## Goal

Three deliverables:

1. **Restore four evenly-spaced pills on Surface Dynamics.** Today's
   Setup becomes a real fourth `dcc.Tab` with identical `TAB_STYLE` /
   `TAB_SELECTED_STYLE` to the other three. The wrap reverts to
   `TABS_WRAP_STYLE` (block-level, no flex override). `dcc.Tabs` regains
   100% width and distributes the four tabs across the row natively.
   Button label reads "Price Chart" (visible label — never in question).

2. **Today's Setup tab triggers navigation to `/today-setup`.** A
   `dcc.Location` + redirect callback watches `MAIN_TABS_ID, value` and,
   when it becomes `TAB_TODAY_SETUP`, sets `page-url.pathname` to
   `/today-setup`. For all other tab values the callback returns
   `no_update`, leaving the normal `_switch_main_tab` style toggle in
   charge of in-page tab switching.

3. **Full top bar replicated on Today's Setup React page.** The same
   two-row top bar that's on `/` appears at the top of `/today-setup`:

   - **Row 1:** "Surface Dynamics" centered title + "Home" link at
     top-right, matching `apps/web/app.py`'s existing top bar (the
     `html.Div` containing the centered "Surface Dynamics" text and the
     `html.A("Home", ...)` link).
   - **Row 2:** Four-pill nav row — Dashboard, Price Chart, Backtests,
     Today's Setup. Today's Setup is rendered in the selected state.
     Dashboard / Price Chart / Backtests link to `/`,
     `/?tab=price-chart`, `/?tab=backtests` respectively.

   The existing `← Dashboard` link at the far right of the today-setup
   page's own `.header` is removed (subsumed by the new nav strip). The
   page's `<h1>Day Setup</h1>` stays as a sub-header below the new top
   bar.

   To make `?tab=...` actually land on the requested tab when the user
   clicks Price Chart or Backtests from `/today-setup`, Dash gets a
   small URL→tab callback that reads `page-url.search` on load and sets
   `MAIN_TABS_ID, value` accordingly.

## Changes

**`apps/web/app.py`:**

- Add constant `TAB_TODAY_SETUP = "tab-today-setup"` next to the other
  `TAB_*` constants (~line 78).
- Add a fourth `dcc.Tab(label="Today's Setup", value=TAB_TODAY_SETUP,
  style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE)` to the
  `dcc.Tabs` children list (~line 560).
- Remove the sibling `html.A("Today's Setup", href="/today-setup", ...)`
  (~lines 511–526).
- Revert the wrap `style` to `TABS_WRAP_STYLE` only — drop the
  `"display": "flex", "alignItems": "stretch"` overrides (~line 528).
- Add `dcc.Location(id="page-url", refresh=True)` to the layout
  top-level (alongside the existing `dcc.Store` / `dcc.Interval`
  components ~line 456).
  Note: `refresh=True` (not `False`) is required because `/today-setup`
  is a separate Vite app served by Flask, not a Dash route. With
  `refresh=False`, `dcc.Location` updates the URL via pushState only —
  no page reload — so the browser stays on the Dash shell. `refresh=True`
  triggers a real navigation to `/today-setup`.
- Add a callback:
  ```python
  @app.callback(
      Output("page-url", "pathname"),
      Input(MAIN_TABS_ID, "value"),
      prevent_initial_call=True,
  )
  def _redirect_to_today_setup(tab_value):
      if tab_value == TAB_TODAY_SETUP:
          return "/today-setup"
      return no_update
  ```
- Add a URL→tab callback so query-param navigation from the React app
  lands on the right tab:
  ```python
  @app.callback(
      Output(MAIN_TABS_ID, "value", allow_duplicate=True),
      Input("page-url", "search"),
      prevent_initial_call="initial_duplicate",
  )
  def _tab_from_url(search):
      if not search:
          return no_update
      params = dict(
          p.split("=", 1)
          for p in search.lstrip("?").split("&")
          if "=" in p
      )
      mapping = {
          "dashboard": TAB_DASHBOARD,
          "price-chart": TAB_PRICE_CHART,
          "backtests": TAB_BACKTESTS,
      }
      return mapping.get(params.get("tab", ""), no_update)
  ```
  Note: `allow_duplicate=True` is required because
  `apply_backtests_selection` is already a writer to
  `MAIN_TABS_ID, value`.

**`react_today_setup/src/App.jsx`:**

- Add a top-bar header above the existing `.header` div (~line 311),
  with two rows:

  - **Row 1:** "Surface Dynamics" centered title + "Home" link at the
    right. Mirror the styling from `apps/web/app.py`'s existing top
    bar (centered title in `#e5e7eb` at 20px/600 weight; "Home" link
    in `#93c5fd` with `1px solid #1f2937` border, padding `4px 10px`,
    radius `6px`, absolutely positioned at the right edge). Use the
    same dark border-bottom separator.

  - **Row 2:** Four-pill nav row. Dark wrap (`#0b1220` bg,
    `1px solid #1f2937`, radius `14px`, padding `6px`) — mirror
    `TABS_WRAP_STYLE`. Four `<a>` pills inside, each styled to match
    `TAB_STYLE` and `TAB_SELECTED_STYLE`:
      - `padding: 10px 16px`, radius `12px`, `fontSize: 13`,
        `fontWeight: 700`
      - Unselected: text `#93c5fd`, transparent background, no border
      - Selected (Today's Setup): background `#111827`, border
        `1px solid #60a5fa`, text `#bfdbfe`, `fontWeight: 800`
      - Use `display: flex` on the wrap with `flex: 1` on each pill
        so the four span the full width evenly (matches Dash's native
        even distribution).

  - Pills:
      - Dashboard → `<a href="/">`
      - Price Chart → `<a href="/?tab=price-chart">`
      - Backtests → `<a href="/?tab=backtests">`
      - Today's Setup → render with selected styling; `href="/today-setup"`

- Remove the existing
  `<a href="/" style={{ marginLeft: 'auto', ... }}>← Dashboard</a>`
  from inside `.header` — the new top bar replaces it.

- Leave the `<h1>Day Setup</h1>` and the rest of `.header` intact —
  it now reads as a sub-header below the new top bar.

**`react_today_setup/src/styles.css`:**

- Add `.top-bar` + `.top-bar-title` + `.top-bar-home` classes for Row 1.
- Add `.top-nav` + `.top-nav-pill` + `.top-nav-pill.selected` classes
  for Row 2.
- Use Flexbox with `flex: 1` on each pill so the four span full width
  identically to the Dash-side native distribution.

## Acceptance criteria

1. **Four evenly-spaced pills on Surface Dynamics.** Open `/`. The top
   nav row shows Dashboard, Price Chart, Backtests, Today's Setup as
   four identically-styled pills, evenly distributed across the full
   width of the wrap. Visually matches the pre-Today's-Setup layout.

2. **Tab switching still works in-page.** Clicking Dashboard / Price
   Chart / Backtests toggles the right container visible via the
   existing `_switch_main_tab` callback. No page reload.

3. **Today's Setup tab navigates.** Clicking Today's Setup pill on
   Surface Dynamics triggers a full-page navigation to `/today-setup`.

4. **Full top bar on Today's Setup.** Open `/today-setup`. The page
   shows the "Surface Dynamics" centered title, "Home" link at
   top-right, and the four-pill nav strip below — visually identical
   to `/`. Today's Setup is styled as the selected pill.

5. **Deep-link navigation works.** From `/today-setup`, clicking Price
   Chart navigates to `/?tab=price-chart` and the Price Chart tab is
   pre-selected on load. Same for Backtests via `/?tab=backtests`.
   Dashboard via `/` lands on the default Dashboard tab.

6. **No regression in `apply_backtests_selection`.** Clicking a result
   in Backtests still jumps to Price Chart with the right time slices
   loaded.

7. **`← Dashboard` link removed from today-setup `.header`.** The
   far-right link inside `.header` is gone.

8. **`<h1>Day Setup</h1>` and the rest of `.header` still render.**
   Page sub-header below the new top bar is intact.

9. **`vite build` clean** for `react_today_setup/`.

10. **Switching dates clears any stale error from the previous load.**
    Verify: load `/today-setup` with date defaulted to a no-landscape day
    (today), confirm red error renders, pick an analogue-data date (e.g.
    5/21), confirm error disappears and the chart fills the freed vertical
    space.

## Amendment — stale-error race condition (discovered during smoke)

**Root cause:** In `loadAnchor`, `setError(null)` is called at the top of
the function, but the success branch (`propR.value?.ok` true) did not
re-clear it. If a prior in-flight request (for the old date) resolved
*after* the new request started and cleared the error at the top, the old
request's `setError(msg)` would overwrite the cleared state. The new
request's success path had no subsequent `setError(null)` to fix it.

**Fix (`react_today_setup/src/App.jsx`):**
- Added `anchorAbortRef = useRef(null)` to hold the current
  `AbortController`.
- At the start of `loadAnchor`: abort the previous controller, create a
  new one, store it, extract `signal`.
- Pass `signal` to all four `fetch*` helpers (`fetchProposals`,
  `fetchAnalogues`, `fetchLandscape`, `fetchFlags`) as a new optional
  third parameter.
- After `Promise.allSettled`, guard with `if (signal.aborted) return` to
  discard results from superseded requests.
- In the success branch: add `setError(null)` to clear any stale error
  that slipped through before the guard fires.
- In `finally`: guard `setLoading(false)` with `if (!signal.aborted)`.

**AbortController pattern** borrowed from
`react_price_preview/src/App.jsx`'s analogues fetch effect
(`controller.abort()` + `disposed` flag pattern).

## Step-0 Diagnosis Findings

**(a) `dcc.Tabs` + `html.A` + flex override location in `apps/web/app.py`:**
- `dcc.Tabs` component: lines 500–510 (id=`MAIN_TABS_ID`, 3 children tabs)
- `html.A("Today's Setup", href="/today-setup", ...)`: lines 511–526 (sibling to `dcc.Tabs`)
- Wrap `style={**TABS_WRAP_STYLE, "display": "flex", "alignItems": "stretch"}`: line 528

**(b) `dcc.Location` presence:**
- **Absent** from the layout. This CR adds `dcc.Location(id="page-url", refresh=False)` at the top-level store block (~line 456, alongside `dcc.Store` and `dcc.Interval` components).

**(c) Writers to `MAIN_TABS_ID, value`:**
- Only writer: `apply_backtests_selection` callback (Output at line 595).
- `_switch_main_tab` is a *reader* (Input at line 575), not a writer.
- New `_tab_from_url` callback therefore requires `allow_duplicate=True` + `prevent_initial_call="initial_duplicate"`.

**(d) `.header` block and `← Dashboard` link in `react_today_setup/src/App.jsx`:**
- `.header` div starts at line 311.
- `<a href="/" style={{ marginLeft: 'auto', color: '#60a5fa', fontSize: 12, textDecoration: 'none' }}>← Dashboard</a>` at lines 392–394. This is the element to remove.

**(e) Top-bar block in `apps/web/app.py` (lines 461–495):**
- Outer wrap: `position: relative`, `padding: 8px 16px`, `borderBottom: 1px solid #1f2937`, `marginBottom: 8px`
- Title div: "Surface Dynamics", `fontWeight: 600`, `fontSize: 20px`, `color: #e5e7eb`, `textAlign: center`, `width: 100%`
- Home link: `href="https://blog.surfacedynamics.io"`, `color: #93c5fd`, `textDecoration: none`, `fontWeight: 500`, `padding: 4px 10px`, `borderRadius: 6px`, `border: 1px solid #1f2937`, `position: absolute`, `right: 16px`, `top: 50%`, `transform: translateY(-50%)`

## Verification plan

1. `python apps/web/app.py` → open `/`. Confirm AC #1 visually. Click each of the three in-page tabs — AC #2.
2. Click the Today's Setup pill — confirm full navigation to `/today-setup` (AC #3).
3. On `/today-setup`, confirm title row + four-pill nav + sub-header intact (AC #4, #7, #8).
4. Click Price Chart in the today-setup nav — confirm URL becomes `/?tab=price-chart` and tab pre-selected. Repeat Backtests and Dashboard (AC #5).
5. Open Backtests tab on `/`, double-click a result row — confirm jump to Price Chart still works (AC #6).
6. `cd react_today_setup && npm run build` — confirm clean build (AC #9).

## Out of scope

- Restyling the Today's Setup page beyond the new top bar.
- Extracting a shared React nav component.
- Hash-based or React Router navigation.
- CSS-level cleanup of `TABS_WRAP_STYLE` / `TAB_STYLE` / `TAB_SELECTED_STYLE`.

## Data integrity dependencies

None. Pure frontend / routing change.
