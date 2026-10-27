# CR-CQ — GEX Min |GEX| filter defaults to 40 $bn and never silently resets to 0

Spec freeze. Branch `feat/CR-CQ-gex-filter-default-40` off `origin/main` `4075d27` (PR #96 merge, CR-CO), in its own worktree (`../volatility-monorepo-CR-CQ`). Display only: no research data, no DB writes. PR, don't merge; Ryan merges and chat deploys Dash.

Source: the kickoff prompt (Ryan, 2026-10-26), transcribed as K1–K7.

## Why

Ryan: "many times when I refresh the app the filter goes back to 0 and it's too much on the screen."

## Kickoff (Ryan, 2026-10-26)

| # | Decision | Value |
|---|---|---|
| K1 | Scope | Visible change in scope: the default value of the price chart's Min \|GEX\| filter. Nothing else on screen changes (CLAUDE.md visible-change discipline; standing-rulings C2 n/a — an explicit default change Ryan asked for). |
| K2 | Step 0 | Diagnose before fixing: every source of the filter value and their precedence — `react_price_preview/src/App.jsx` (`coerceGexMinAbsB` fallback 10, localStorage `ib-react-gex-min-abs-b`, the `gex_min_abs_b` URL param) and the Dash side (`apps/web/modules/Ironbeam/callbacks.py` ~4730–4761: `indicator_state` `cfg.gex_overlay.min_abs_b` passed into the iframe src, and wherever that store is initialised/persisted). Reproduce the reset to 0. Write the cause into the spec. |
| K3 | Fix 1 | Default 40 everywhere a default exists (React fallback, Dash `gex_overlay` `min_abs_b` default). |
| K4 | Fix 2 | An empty or invalid input never stores 0 — keep the previous value until a valid number is entered. 0 stays allowed only when typed explicitly. |
| K5 | Fix 3 | A refresh shows the last value Ryan set; with nothing stored, 40. |
| K6 | Tests | Unit tests for the coercion/precedence; Playwright: fresh profile shows 40; set 60 → refresh → 60; clear the box → refresh → previous value, not 0; existing suites unchanged apart from the default. |
| K7 | Delivery | PR, don't merge; then Ryan merges and chat deploys Dash. |

---

## Step 0 — diagnosis (2026-10-26, before any fix)

### Every source of the chart's Min |GEX| value, in precedence order (origin/main `4075d27`)

Held in React state `gexMinAbsB` (`react_price_preview/src/App.jsx`), initialised once per page load:

1. **localStorage `ib-react-gex-min-abs-b`** — `coerceGexMinAbsB(localStorage.getItem(key), fallback)`. Wins whenever it yields a finite number.
2. **URL param `gex_min_abs_b`** — `parseFloatOrNull(...)`, used only as the `fallback` of (1).
3. **Hard default** — `coerceGexMinAbsB(initialGexMinAbsB, 10)`, the fallback of (2).

After load, the value changes only through the price-chart settings dialog (`PriceChart.jsx` `applySettings` → `onApplyGexMinAbsB` → `handleGexMinAbsBChange`, a range slider 0–200, so it cannot be emptied). Every change is written straight back to the same localStorage key by a `useEffect`. The value is sent as `gex_min_abs_b` on every `/api/ironbeam/bars` request; the server filters levels with it.

Dash side (`apps/web/modules/Ironbeam`):

- `ib-indicator-state` (`components.py`) is a `dcc.Store(storage_type="local")` initialised to `{"enabled": [...], "cfg": {}}` — no `min_abs_b`.
- The only writer of `cfg.gex_overlay.min_abs_b` is `ib_persist_gex_settings` (`callbacks.py` ~4380, `Input("ib-gex-min-abs-b")`). That input is no longer rendered — the Dash indicator-settings panel for `gex_overlay` now says "settings now live in the price chart gear" — so the callback never fires and the store normally has no `min_abs_b`.
- `ib_update_react_preview_src` (~4730–4761) appends `gex_min_abs_b=` to the iframe src **only** when the store has a value. Normally it doesn't, so source (2) is absent and the React hard default decides.
- `GexOverlay.default_config()` (`indicators/gex_overlay.py`) returns `GEX_ABS_THRESHOLD / 1e9` = 10 (env unset locally). It feeds the Dash-side (Plotly) overlay threshold, not the iframe. `GexOverlay.compute` falls back with `cfg.get("min_abs_b") or 10`, so an explicit 0 there becomes 10 (the opposite bug).
- `apps/web/modules/gamma/callbacks_replacement_autoresolve_symbols.py` duplicates this code but is imported nowhere — dead, untouched.
- The GEX side panel's own Min |GEX| (`ib-react-gex-panel-min-abs-b`, default 10) is panel-local, already guards `null`/`''`, and is out of scope.

### Cause of the reset to 0

`coerceGexMinAbsB` uses `Number(value)` and only rejects non-finite results. But `Number(null) === 0` and `Number('') === 0` (also `Number('  ')`). So:

- **No stored key** (`getItem` → `null`) → `0`, not the fallback. Any load with empty storage for the iframe's origin — a new browser/profile/device, cleared site data, a different host (localhost dev vs Render), storage evicted — yields 0.
- **No URL param** (the normal case, see above) → `coerceGexMinAbsB(null, 10)` → `0`, so even the "default 10" is in fact 0; if localStorage throws, the result is 0 too.
- **Invalid stored value** (`'abc'`) → falls back to the hard default → 0.
- Then the persist `useEffect` writes `"0"` into localStorage, so the 0 **sticks** on every later refresh until Ryan moves the slider again. That is the "goes back to 0".

Reproduced on unchanged code (this worktree before any fix, Playwright against `scripts/cr_ck_smoke_backend.py`, 2026-10-26): a fresh browser context requests `/api/ironbeam/bars?...&gex_min_abs_b=0` and stores `ib-react-gex-min-abs-b = "0"`; a stored `'abc'` also gives 0; the in-app browser pane, a fresh origin, stored `"0"` on its first load. The Dash store cannot produce the 0 (it is not written), and the slider cannot produce an empty value.

### Implementation decisions (fixed before code)

| # | Decision |
|---|---|
| D1 | One pure module `react_price_preview/src/gexMinAbsB.js`: `GEX_MIN_ABS_B_DEFAULT = 40`, `parseGexMinAbsB` (null / undefined / blank / non-finite → `null`; else rounded and clamped 0–200), `coerceGexMinAbsB(value, fallback = 40)`, `resolveInitialGexMinAbsB({ stored, legacyStored, urlParam })`. App.jsx and PriceChart.jsx import it (PriceChart's private copy with fallback 10 goes; the panel-local filter keeps passing its own fallback 10). PriceChart's `gexMinAbsB` prop default 50 → 40. |
| D2 | Precedence on load: stored value → URL param → 40. Unchanged order, so a refresh shows the last value Ryan set (K5). |
| D3 | Values already stored are untrustworthy at 0: the old code wrote `"0"` into empty storage on every such load, so a legacy `0` is indistinguishable from a choice. New writes go to `ib-react-gex-min-abs-b-v2`. On load the v2 key is read first (0 there is an explicit choice and is kept); if absent or invalid, the legacy key is accepted only when > 0; otherwise URL param, otherwise 40. One consequence: if Ryan had deliberately set 0 before this CR, he sees 40 once and sets 0 again. |
| D4 | An empty / invalid value never replaces the current one: `handleGexMinAbsBChange` keeps `prev` (K4); `applySettings` already refuses non-numbers. 0 is accepted when chosen explicitly (slider at 0 → Apply). |
| D5 | Dash: `GexOverlay.default_config()` returns 40 (constant `GEX_OVERLAY_DEFAULT_MIN_ABS_B`), no longer the compute env `GEX_ABS_THRESHOLD`; `GexOverlay.compute` falls back to it only for `None` / blank, so an explicit 0 is honoured. The iframe src builder is unchanged (it passes a value only when the store has one). Server-side `GEX_ABS_THRESHOLD_DEFAULT` (route fallback when no param is sent) is not a display default and stays — the React chart always sends the param. |
| D6 | "Clear the box" (K6): the chart control is a range slider and cannot be emptied, so the Playwright test covers the empty paths that exist — a blank / junk stored value, an empty `gex_min_abs_b=` URL param, a missing key — plus the legacy-0 migration and an explicit 0. Unit tests cover `''` through the change handler's coercion. |
