# CR-032 — Today's Setup Date Persistence and "Open in Price Chart" Jump

## Scope

Frontend-only, two files in `react_today_setup/`:
- `src/App.jsx`
- `src/components/DayView.jsx`

Explicitly **out of scope / must remain untouched:** `apps/web/app.py`, the selection poller,
the `TRADE_DATE_PICK` writers, all Dash wiring, `react_price_preview`, backend routes.
No layout/color/spacing/sizing changes anywhere except the single new button.

---

## Spec

### Part 1 — Date init / persistence (`App.jsx`)

**1a. Local-time trading-day helper.** Replace the UTC `toISOString()` computation in
`mostRecentTradingDay()` with a local-time one so "today" matches the user's wall clock
(Pacific). Compute year/month/day from local getters and format `YYYY-MM-DD` manually;
keep the existing weekend guard (Sat → −1 day, Sun → −2 days).

> Note: CLAUDE.md UTC convention governs market data. This is a UI "what calendar day is it
> for the user" calculation; local time is correct here.

**1b. Reload detection.** Add a helper `isPageReload()` returning
`performance.getEntriesByType('navigation')[0]?.type === 'reload'`.
Guard for environments where the entry is missing (return `false`).

**1c. sessionStorage key.** Use a single key: `TODAY_SETUP_DATE_KEY = 'today-setup-date'`.

**1d. Date resolution on load** (replace the `date:` line in `parseQS()` or factor into an
init function). Priority:
1. If `?date=` URL param present and valid → use it (URL wins; preserves bookmarks/shared links).
2. Else if `isPageReload()` → `mostRecentTradingDay()`, and clear the stored key
   (`sessionStorage.removeItem`).
3. Else → stored value from `sessionStorage` if present and valid, else `mostRecentTradingDay()`.

Validity check: matches `^\d{4}-\d{2}-\d{2}$`.

**1e. Persist on change.** Add a `useEffect` keyed on `date` that writes the current `date`
to `sessionStorage` (wrap in try/catch). This runs for every picker change and every
programmatic set.

All `sessionStorage` and `performance` access wrapped defensively so a storage-disabled
environment degrades to "always current trading day," never throws.

### Part 2 — "Open in Price Chart" button

**2a. Handler (in `App.jsx`, passed to `DayView` as prop `onOpenInPriceChart`).**
Signature: `onOpenInPriceChart(targetDate)`. Behavior:
1. `try { await fetch(\`${API_BASE}/api/backtests-v2/select-trade\`, { method: 'POST',
   headers: {'Content-Type':'application/json'}, body: JSON.stringify({ trade_date: targetDate }) }) }
   catch { /* non-fatal; still navigate */ }`
   - Await so the server stores the selection before the page unloads.
2. Build navigation target: base `'/?tab=price-chart'`; if current
   `window.location.search` contains `api_base`, append `&api_base=<value>`.
3. `window.location.href = target` (same-tab navigation).

**2b. Wire into both `DayView` instances.**
- Anchor `DayView`: `onOpenInPriceChart={() => date && onOpenInPriceChart(date)}`
- Selected `DayView`: `onOpenInPriceChart={() => selectedDate && onOpenInPriceChart(selectedDate)}`

**2c. Button in `DayView.jsx` header row.** Add after the date / regime pill, gated on `date`
being truthy. Label: **"Open in Price Chart"**. Style: `smallBtn('#1e3a5f', '#60a5fa')`.
`type="button"`. `onClick={onOpenInPriceChart}`. Accept `onOpenInPriceChart` as a new optional
prop; if absent, don't render the button.

### Behaviors noted (decisions, not bugs)

- The sent `trade_date` persists in `_SELECTION_STATE` server-side until overwritten by the
  next selection. Benign — the main app "remembers" the last date sent.
- Returning to `/today-setup` after the jump is a fresh navigation (not a reload), so Part 1
  restores the `sessionStorage` date. The two parts compose correctly.

### Acceptance criteria

1. Fresh load of `/today-setup` (no `?date=`) on a weekday shows today's date; on a weekend
   shows the prior Friday — computed in local (Pacific) time.
2. `/today-setup?date=2026-05-14` shows 2026-05-14; a refresh of that URL keeps 2026-05-14.
3. With no `?date=`: pick a non-default date, click the Price Chart top-nav pill, then click
   the Today's Setup pill → the picked date is still shown.
4. With no `?date=`: pick a non-default date, hard-refresh (Cmd-R) → date resets to current
   trading day.
5. "Open in Price Chart" button is present in both Anchor panel header and Selected panel
   header (the latter only after an analogue is selected).
6. Clicking the button on the Anchor panel navigates to Price Chart with the anchor date
   loaded; the expiration picker mirrors it automatically.
7. Clicking the button on the Selected panel does the same with the analogue's date.
8. In dev (`localhost:5173?...&api_base=http://127.0.0.1:<port>`), the button carries
   `api_base` through to `/?tab=price-chart` so the Price Chart app finds the backend.
9. `npm run build` (in `react_today_setup`) is clean. No backend files changed.
   `git diff --stat` shows only `App.jsx`, `DayView.jsx`, and this spec file.

---

## Step 0 Diagnosis — 2026-06-01

### Gate 1: `select_trade_api` + `apply_backtests_selection` with only `trade_date`

`select_trade_api` (BacktestsV2/routes.py:1041):
- All `_ts_pt` fields use `str(payload.get("...") or "").strip()` — they become empty
  strings when absent from the request body.
- Only `trade_date` is validated/required.

`apply_backtests_selection` (app.py:623):
- Passes each `_ts_pt` value through `_parse_hhmm()`.
- `_parse_hhmm("")` returns `None` (line 140: `if not s: return None`).
- The times accumulation loop (lines 639–643) only appends non-None values.
- Result: `out_times = []` when all `_ts_pt` are absent — safe, no error.

**✅ Confirmed: posting only `{trade_date}` is safe end-to-end.**

### Gate 2: `react_price_preview` reads `api_base` from URL

`react_price_preview/src/App.jsx` line 68: `function inferApiBase()` reads
`params.get('api_base')` (line 71). Called at line 461: `const apiBase = useMemo(() => inferApiBase(), [])`.

**✅ Confirmed: carrying `&api_base=` through `/?tab=price-chart` reaches the Price Chart app.**

### Gate 3: `DayView` props and `smallBtn` helper

`DayView.jsx` props: `label`, `date`, `ticker`, `apiBase`, `landscapeData`, `regime`,
`autoRegime`, `flag`, `allowPairFlag`, `onRegimeFlag`, `onPromote`, `onDemote`,
`onDeleteFlag`, `onPairFlag` — all match spec assumptions.

`smallBtn(bg, color)` helper defined at the bottom of `DayView.jsx` — ✅ exists.

**✅ Confirmed: all prop names correct; `smallBtn` available; no contradictions.**
