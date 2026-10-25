# CR-BV — Landscape and Chart in One Space

Spec freeze. Branch `feat/CR-BV-landscape-chart-one-space` off `origin/main` `0e943d8` (the CR-BT merge, PR #71). Read-only against the database; no table writes, no cron changes. Current-mode `/api/gex-landscape` and `/api/ironbeam/bars` responses must stay byte-identical.

Source: vault `Dash/CRs/CR-BV — Landscape and Chart in One Space.md` (last_reviewed 2026-10-25). Body below is transcribed verbatim from that note (frontmatter omitted); the locked decisions L1–L10 were confirmed by the kickoff prompt (Ryan, 2026-10-25).

---


# CR-BV — Landscape and Chart in One Space

## Why

Ryan, 2026-10-25: the landscape panel and the price chart have never lined up, despite CR-017/CR-018. He wants them verified to be in the same space (ES), sharing one axis. He also wants the horizontal half of the chart's cursor crosshair (the dashed line that follows the mouse) to extend across the landscape, and an x-axis on the landscape showing $gamma.

Chat's code read on 2026-10-25 (origin/main `0e943d8`, no DB or UI run) found two independent causes.

1. **Pixel offset.**
   - `packages/web-shared/src/GexLandscape.jsx` (CR-018) maps prices with the chart's own transform: `yOf(price) = ((priceTop − price) / span) · paneHeight`, from `visiblePriceRange` published by `PriceChart.jsx`. Its comment says "the landscape SVG body's y=0 corresponds to the chart's y=0 (plot-area top)".
   - **That is false.** The SVG sits in `bodyRef`, *below* the panel's header row (regime badge, LIVE/OPEN switch). The component measures that offset (`size.offsetTop = el.offsetTop`) and never uses it.
   - **Effect:** every landscape price is drawn `offsetTop` px too low. On Ryan's 2026-10-06 screenshot the 7861 tick sits about 33 px below the chart's 7861, about 17–20 pts at that zoom, roughly the header height.
2. **Price frame.**
   - The landscape's price axis is `discounted_level` (frame B) on both paths: the stored `orats_gex_landscape` and the request-time path in `gex_landscape_api.py`.
   - Its `spot` is ES: the live ES price, or the ES RTH open in OPEN mode.
   - The Current chart lines are frame B too. The Corrected lines (CR-BT) are ES-placed.
   - **So in Corrected mode the two panels plot different spaces,** and in either mode the landscape's spot (ES) and curves (B) are in different frames.
3. **Stale spot label.** The amber "spot" line is the ES price at the last landscape fetch. LIVE mode re-fetches only after a 5-pt move (`GEX_LANDSCAPE_SPOT_DELTA_PTS`), and OPEN mode freezes it at the open. It can sit 20+ pts from the chart's last price (2026-10-06: 7851 vs 7873).

## Locked decisions (Ryan to confirm before spec freeze)

| # | Decision | Value |
|---|---|---|
| L1 | Scope | Frontend alignment of the landscape panel with the price chart; the landscape's price frame follows the CR-BT `levels` toggle; the chart crosshair's horizontal line extends across the panel; a $gamma x-axis on the panel. No table writes, no cron, no change to `orats_gex_landscape`, `compute_landscape` math (kernel, `spread_coef`), regime / confluence classifiers, or the Current-mode response. |
| L2 | One frame | **Current mode:** landscape request and response unchanged (byte-identical; test enforces). **Corrected mode:** the landscape is computed at request time from `orats_oi_gamma` rows, with each row's kernel centre = CR-BT `place_levels` corrected L (P_target = ES 15:45 ET bar close on A, same function, same rows and filters as the chart's corrected lines). Everything else in the computation is unchanged. CR-BT A1.1/A1.2 apply: before 2024-04-03, or with no ES snapshot, the panel shows a note and no curve. The classifier chain runs on this ES-frame landscape with the ES spot, so both are in one frame. |
| L3 | Request plumbing | `/api/gex-landscape` gains `levels=corrected` (absent = current), sent by `App.jsx` from the same state the bars requests use. |
| L4 | Shared axis | The panel draws with the chart's transform corrected for the measured vertical offset: `y_panel(price) = y_chart(price) − (panelBodyTop − chartPaneTop)`, both measured from the DOM (`getBoundingClientRect`) on resize and in the existing rAF poll. **Acceptance:** Playwright at two viewport sizes, three prices (top, middle, bottom of the visible range): |y_panel − y_chart| ≤ 1 px. |
| L5 | Crosshair across both | **Chart → panel:** when the cursor is over the price chart, the crosshair's horizontal line continues across the panel at the same price (from `subscribeCrosshairMove`), in the same color and dash, plus a small readout at the panel edge: net GEX at that price in $bn (total curve, linearly interpolated). **Panel → chart:** when the cursor is over the panel, the chart shows its horizontal crosshair at the cursor's price (lightweight-charts v5 `setCrosshairPosition` / `clearCrosshairPosition`), and the panel draws the same line and readout. Hidden when the cursor leaves both. The vertical half of the crosshair is unchanged. The existing wheel-zoom on the panel is unchanged. |
| L6 | Amber line | Kept, relabelled **"regime spot N"**: it is the price the classifier used. Shown only when it differs from the last bar's close by > 1 pt. |
| L7 | Panel ticks | Hidden when synced (`showPriceAxis` false, as on Today Setup); the chart's right axis is the single axis. Unsynced fallback unchanged. |
| L8 | Today Setup | `GexLandscape` is shared. If Step 0 shows Today Setup has the same offset, the shared fix applies there and its acceptance check runs there too. Otherwise Today Setup is untouched. Its frame (L2) is unchanged: Today Setup has no levels toggle. |
| L9 | Visible-change discipline | Only L5–L7, L10 and the axis correction are visible changes. No other layout, color or spacing edits. |
| L10 | $gamma x-axis | Tick marks and labels along the bottom of the panel in **$bn of net GEX** (the existing `xOf` scale, stored gex ÷ 1e9, the same units as the chart's GEX popup). 3–5 rounded ticks with 0 always labelled; the existing zero line and the "GEX ($B) — by DTE bucket" caption are kept. The axis rescales with the data, as `xOf` does now. |

## Step 0 — diagnosis gate (answer all before implementation; contradiction-stop)

1. **Offset, measured.** With the panel open on 2026-10-06 and 2026-10-23, report:
   - `chartPaneTop`, `panelBodyTop`, `bodyRef.offsetTop` and `paneHeight` in px;
   - for three prices, `y_chart` (`series.priceToCoordinate`) vs the panel's current `yOf`.
   Confirm or refute that the error equals the body offset. If the error is not a constant offset (for example a scale difference), stop and report.
2. **Frame inventory.**
   - For each of chart lines (Current, Corrected), landscape curves, landscape `spot`, amber line, neg-zone bands (`LandscapeChartOverlay`) and confluence price lines on the chart, state the price frame (ES / B / SPX cash) and the code site.
   - Name which `/api/gex-landscape` path the price chart hits (stored vs request-time `accuracy`).
3. **Corrected landscape feasibility.** Confirm the request-time path can take a per-row level column without changing `compute_landscape`'s signature for other callers, and describe the smallest change. Report the response time on 2026-10-23 in Corrected mode.
4. **Worked day 2026-10-23.**
   - Landscape top-3 positive/negative peaks under Current vs Corrected.
   - Check the Corrected peaks against the Corrected chart lines (expected near 7681 / 7781 / 7687).
5. **Today Setup.** Does `react_today_setup` show the same offset? Measure it as in Q1.
6. **Crosshair hooks.** Confirm `handleCrosshairMove` in `PriceChart.jsx` (around line 1863) exposes the crosshair price, and that `setCrosshairPosition` works with the installed lightweight-charts version (`^5.0.8`) and the candlestick series. Report any conflict with existing crosshair handlers (tooltips, popups).
7. **Overlays.** `LandscapeChartOverlay` (neg-zone bands) and the confluence price lines on the chart. Are they in frame B? They must follow L2 in Corrected mode. List any other landscape-derived drawing on the chart.

## Implementation order (one commit each)

1. Backend: `levels=corrected` on `/api/gex-landscape` (L2/L3) and the Current-mode byte-identity test.
2. Shared axis (L4) in `GexLandscape.jsx` and the `PriceChart.jsx` publisher (add the pane's DOM top); ticks hidden when synced (L7).
3. Crosshair across both (L5), amber relabel (L6), $gamma x-axis (L10).
4. `App.jsx`: pass the levels state to the landscape fetch (L3); overlays follow (Q7).
5. Smoke:
   - the Playwright pixel test (L4), at two sizes, on the price chart and, if L8 applies, on Today Setup;
   - a Playwright crosshair test (L5): hover the chart at a price and check the panel line is within 1 px of the chart crosshair; hover the panel and check the chart crosshair appears at that price;
   - screenshots for 2026-10-06 and 2026-10-23, Current and Corrected.

## Not in scope

- Kernel width or the CR-BK repriced profile on the panel (separate toggle CR).
- Any change to regimes, cards, stored landscapes or crons.

## Kickoff prompt for Claude Code

```
Start CR-BV. First action: git fetch && git checkout -b feat/CR-BV-landscape-chart-one-space origin/main (confirm CR-BV is unused; rename if not). Do not touch the CR-BU branch.
Read the vault CR "/Users/ryan/My Drive/Obsidian Vault/Anonymous/Dash/CRs/CR-BV — Landscape and Chart in One Space.md", components/field-conventions.md, specs/CR-BT-corrected-level-placement.md, and the open questions it links.
Commit 1: spec freeze to specs/CR-BV-landscape-chart-one-space.md, locked decisions verbatim.
Commit 2: Step 0 findings appended. Answer all seven gate questions with measurements (DOM pixel values from a local run, not inferred from code). Stop and report before implementation; stop immediately if Q1 shows the error is not a constant vertical offset or anything contradicts the spec.
Read-only DB. No table writes, no cron changes. Current-mode /api/gex-landscape and /api/ironbeam/bars responses must stay byte-identical. Do not push.
```

---

## Step 0 findings (2026-10-25)

### How this was measured

**Local stack.**
- The React price chart ran as a Vite dev server (`react_price_preview`, port 5174) and Today Setup as a second one (`react_today_setup`, 5173, proxying `/api`).
- Both talked to `scripts/cr_bv_smoke_backend.py`: an auth-free Dash/Flask app that registers only the Ironbeam routes (`/api/ironbeam/*`, `/api/gex-landscape`) and the Today Setup routes.
- **The backend was read-only.** It connected as `BACKFILL_DATABASE_URL` (role `dash_backfill_writer`) with `PGOPTIONS='-c default_transaction_read_only=on'`.
- The scratch backend adds a CORS header for `localhost:517x` (port 5173 was taken when the chart server started), so no repo CORS change was needed.
- **No repo application code changed**, so every Current-mode response is untouched.

**Pixel values.**
- Measured in the browser with `getBoundingClientRect()` on the lightweight-charts pane cell and the landscape SVG.
- `series.priceToCoordinate()` came from PriceChart's candlestick series.
- The rendered panel elements are the tick-label `<text>` (y = yOf + 3) and the amber `<line y1>`.
- The snippet is in `specs/assets/cr-bv/step0_measure.js`. It reads props from the *current* React tree, starting at `#root`: on Today Setup a DOM node's fiber turned out to be a stale alternate holding an old range, so walking up from the node is not reliable.

**Q3/Q4 prototype.** `scripts/cr_bv_step0_q3q4.py` (read-only, scratch logic, no repo change).

**Environment notes.**
- `react_price_preview/node_modules` lacked `@rollup/rollup-darwin-arm64` (it had only the x64 binary). It was installed with `npm i --no-save`, leaving `package.json` and `package-lock.json` unchanged.
- `.claude/launch.json` (untracked) gained three local configs.
- **Branch incident.** At 05:19 another session checked out `feat/CR-BW-aggressor-absorption-at-levels` in the shared checkout. This commit was first made there by mistake (`53c466d`, saved as branch `cr-bv-save-1`). It was removed from CR-BW with a soft reset, touching neither its working tree nor its index, and re-applied here from a separate worktree.

**Verdict: no stop.**
- Q1 is a constant vertical offset, exactly `bodyRef.offsetTop` = 35 px, at every price, on both dates and both viewport sizes.
- Nothing contradicts L1–L10.
- Two clarifications need Ryan's decision before implementation (see "Decisions needed"); one is an unspecified detail in L2. Separately, the "Why" narrative is corrected on two points (Q2).

### Q1 — offset, measured: CONFIRMED (constant 35 px = `bodyRef.offsetTop`)

**Layout, both dates** (main page, landscape open):
- `chartPaneTop` = 0 (the pane `<td>` starts at the stage top).
- `panelBodyTop` = 35.
- `bodyRef.offsetTop` = 35 (the header row: regime badge, LIVE/OPEN).
- The pane `<td>` is 644 px tall at 1440×900 and 824 px at 1920×1080. The published `paneHeight` (PriceChart `getPlotHeight`) is 648 / 828. Both describe the same affine, because `coordinateToPrice(paneHeight)` extrapolates linearly, so this 4-px difference causes no error.
- The panel SVG is 600 / 780 px tall.

**Three prices per case** (10 %, 50 % and 90 % of the visible range). `y_chart` = `series.priceToCoordinate(p)`; the panel's `yOf(p)` = ((priceTop − p)/span)·paneHeight; page y adds the pane top and body top.

| date | viewport | price | y_chart | panel yOf | y_chart page | y_panel page | error |
|---|---|---|---|---|---|---|---|
| 2026-10-06 | 1440×900 | 8105 | 64.84 | 64.84 | 64.84 | 99.84 | **+35.00 px** (33.4 pts) |
| | | 7858 | 323.82 | 323.82 | 323.82 | 358.82 | +35.00 |
| | | 7611 | 582.79 | 582.79 | 582.79 | 617.79 | +35.00 |
| 2026-10-06 | 1920×1080 | 8048 | 82.76 | 82.76 | 82.76 | 117.76 | **+35.00 px** (23.0 pts) |
| | | 7830 | 413.96 | 413.96 | 413.96 | 448.96 | +35.00 |
| | | 7612 | 745.16 | 745.16 | 745.16 | 780.16 | +35.00 |
| 2026-10-23 | 1440×900 | 8007 | 64.39 | 64.39 | 64.39 | 99.39 | **+35.00 px** (32.6 pts) |
| | | 7765 | 324.18 | 324.18 | 324.18 | 359.18 | +35.00 |
| | | 7524 | 582.89 | 582.89 | 582.89 | 617.89 | +35.00 |
| 2026-10-23 | 1920×1080 | 8007 | 82.35 | 82.35 | 82.35 | 117.35 | **+35.00 px** (25.5 pts) |
| | | 7766 | 413.53 | 413.53 | 413.53 | 448.53 | +35.00 |
| | | 7525 | 744.72 | 744.72 | 744.72 | 779.72 | +35.00 |

**Cross-check against rendered elements:**
- The amber spot line is +35.00 px off on 2026-10-06 at both sizes, and on 2026-10-23 at 1440. On 2026-10-23 its rendered `y1` equals yOf(spot) = 436.09.
- The tick labels are +34.28 … +35.57 px off, the spread coming from integer label rounding.

**Conclusion:**
- y_chart = panel yOf exactly. The two share one affine: no scale or range difference.
- Every panel price is drawn exactly `offsetTop` px too low. In points that is 23–33 pts at these zooms, matching the ~33 px seen on Ryan's 2026-10-06 screenshot.
- L4's formula, y_panel = y_chart − (panelBodyTop − chartPaneTop), fixes it with a constant of 35 at present.
- **Consequence for L4:** after the shift, chart y ∈ [0, 35) maps to negative panel y, so the top 35 px of chart prices fall above the panel body and are clipped. The panel body (600 / 780 px) also ends 9 px (1440) or 9 px (1920) above the pane bottom. These are geometric facts of the current layout, not new requirements; L9 forbids layout edits.

Screenshot `specs/assets/cr-bv/step0-2026-10-19-panel-offset-1440.jpg` shows the 7829 confluence line on the chart and its panel label one header-height apart.

### Q2 — frame inventory

| drawing | frame | code site |
|---|---|---|
| chart GEX lines, **Current** | **B** (`discounted_level`, bucketed) | `apps/web/modules/Ironbeam/callbacks.py` `_fetch_gex_grouped_by_level` / `_fetch_gex_grouped_by_level_and_expir` |
| chart GEX lines, **Corrected** | **ES** (front contract at the A 15:45 ET snapshot): `place_levels(rows, ES close of the bar starting 15:45 ET on A)` | `callbacks.py` `_corrected_gex_tables` → `packages/shared/level_placement.py` |
| landscape curves (main page) | **B**: stored `orats_gex_landscape`, built from `discounted_level` by `compute_landscape` (`packages/shared/gex_landscape.py:61`). The grid is centred on `table_spot` (a frame-A-like S0) but its price axis is B | `/api/gex-landscape` **stored "low" path**. Measured: the main page requests `ticker=SPX&date=…&spot=…[&iv=…]` with **no `accuracy`**, and every payload returned `accuracy: "low"` (2026-10-06, 10-19, 10-23) |
| landscape `spot` (request and payload) | **ES**: `liveEsSpot` = close of the **last bar of `mergedBars`** (ES Ironbeam, every loaded session including `days_either_side` extras) | `react_price_preview/src/App.jsx:672`, fetch effect `:799–875` |
| amber line | **ES**: draws `data.spot` | `packages/web-shared/src/GexLandscape.jsx:436–455` |
| neg-zone bands | **B**: `neg_zones` from `find_proximate_negative_zones(find_walls(landscape))` | `react_price_preview/src/components/LandscapeChartOverlay.jsx`; builder `packages/shared/gex_landscape_api.py` |
| confluence price lines + intraday subtarget (on the chart) | **B**: `analyze_confluence(landscape)` and `find_intraday_subtarget` | `PriceChart.jsx:1975–2019` (`series.createPriceLine`) |
| panel header drift target ("MAGNET ABOVE → 7918") | **B** (`classify_regime(landscape, spot)`, with spot in ES) | `GexLandscape.jsx:294`; builder |
| Today Setup chart bars / landscape spot | **ES** (`/api/bars` reads `ironbeam_es_1m_bars`; the landscape request has no `spot`, so the route uses the ES RTH open, `_RTH_OPEN_SQL`) / landscape **B** (stored low path) | `apps/web/modules/Bars/service.py`; `apps/web/modules/Ironbeam/routes.py:29` |

**Corrections to "Why"** (wording only; L1–L10 unchanged):
- **OPEN mode does not freeze the ES RTH open.** It freezes the first `liveEsSpot` seen, which is the last merged bar's close at that moment. Measured on 2026-10-19: OPEN mode requests `spot=7759`, the last merged bar's close, the same value as LIVE.
- **On a historical date the "spot" is the last bar of all loaded sessions.** With the `days_either_side` extras, that can be after the centre session. Measured on 2026-10-19: amber 7759 = last merged bar close, while the chart's last-price label read 7734.75.

### Q3 — Corrected landscape feasibility

**Feasible without touching `compute_landscape`'s signature.** `compute_landscape(df, spot, range_pts, step_pts, spread_coef)` reads the kernel centre from the column `df["discounted_level"]` (`gex_landscape.py:56,61`). The smallest change is entirely inside `build_gex_landscape_response`, behind `levels=corrected`:

1. Fetch rows with `strike, expir_date, dte, stock_price, gex_call, gex_put` under **the same WHERE as CR-BT's** `_fetch_gex_rows_for_placement`: ticker, `trade_date = D`, `expir_date >= D`, `discounted_level IS NOT NULL`.
2. Fetch the ES close of the bar starting 15:45 ET on A (`es_snapshot_bar_start_utc`).
3. Apply the CR-BT A1.1/A1.2 gates: `pre_forward_era` / `no_es_snapshot` give a note and no curve.
4. Compute `place_levels(rows, p_es)`.
5. Pass a frame whose `discounted_level` column holds `level_corrected` into the existing request-time (CR-010 "high") branch. The params (`spread_coef`, `range_pts`, `step_pts`) are the stored row's, as today.

`compute_landscape`, the cron, the stored path and the other callers are untouched.

**Prototype on 2026-10-23** (spot 7660.75, in-process, read-only; the route is not built yet, so these are builder timings):

| path | seconds |
|---|---|
| Current, low (stored) | 0.18 |
| Current, high | 0.33 |
| **Corrected, end to end** (ES snapshot + 12,036 rows + placement + high-path builder) | **0.81** |

The browser-side Current request took 337 ms through the local backend.

### Q4 — worked day 2026-10-23

ES 15:45 ET close on A (2026-10-22) = **7686.00**. Spot 7660.75. Landscape local extrema of `gex_total` ($bn, raw ÷ 1e9):

| | Current (stored, B) | Corrected (ES) |
|---|---|---|
| negative peaks | **7651.4 (−1652.2)**, 7601.4 (−1324.8), 7699.4 (−1225.0) | **7685.75 (−1434.6)**, 7667.75 (−1226.9), 7635.75 (−1226.6) |
| positive peaks | 7908.4 (+574.5), 7828.4 (+475.5) (only two in the grid) | 7849.75 (+846.1) (only one) |
| regime | amplification | amplification |
| confluences (no iv) | none | 7841.7, 7924.6 |

The Current high path matches the stored path to within the 0.35-pt grid offset (7651.75 / 7601.75 / 7699.75).

**Against the Corrected chart lines** (CR-BT Q4 top by magnitude: 7681, 7781, 7687, 7843, 7737, 7636):
- The dominant Corrected landscape trough, **7685.75**, sits between the 7681 and 7687 lines. The kernel (σ = 8·√dte) merges those two adjacent strike lines into one trough. ✔
- **7635.75** matches the 7636 line. ✔
- **7667.75** is a kernel-merged trough between 7636 and 7687 with no single line of its own.
- **7781** (−266.6 B) is not a top-3 trough. The 7–DTE mass there is spread by the wider kernel and outweighed by the 0DTE cluster near spot.
- The positive peak **7849.75** sits next to the 7843 line, the 2027-06-17 LEAP mass that CR-BT brought into the band.

The Corrected landscape and the Corrected lines agree in the same ES frame. Current puts the dominant trough at 7651, 34 pts below Corrected's 7685.75, about the 0DTE under-placement CR-BT measured (+35.9).

### Q5 — Today Setup: no offset (0 px); L8 → Today Setup untouched

Measured on `/today-setup` in BROWSE mode on day 2026-10-12, at 1440×900. Analogue mode never finished loading locally: `/api/analogues` took > 120 s.
- `DayView` pads the chart wrapper by `LANDSCAPE_HEADER_PX = 35` (`react_today_setup/src/components/DayView.jsx:11,154`), so **`chartPaneTop` = 278 = `panelBodyTop` = 278** (`bodyRef.offsetTop` 35).
- The current-tree published range (7871.911 / 7819.564, paneHeight 421) equals the chart's `coordinateToPrice(0)` (7871.911).
- The rendered amber line `y1` = 69.66 = the chart's `priceToCoordinate(spot 7863.25)` = 69.66. **Error 0.00 px.**
- `showPriceAxis` is already `false` there.
- Today Setup does not have the main-page offset, so per L8 it is untouched. Its frame is B for the curves and ES for the spot, as on the main page; L8 keeps that.

### Q6 — crosshair hooks: confirmed, no conflict found

- **Chart → panel.** `handleCrosshairMove` (`PriceChart.jsx:1706`, subscribed at `:1863`) forwards `param` to `updateFloatingTooltip`, which already derives the cursor price as `seriesRef.current.coordinateToPrice(param.point.y)` (`:1571`). The price is available there.
  - The existing subscribers are the floating tooltip and `reportLinkedCrosshair` (time-only link to the flow chart). A panel-sync callback can be added beside them.
- **Panel → chart.** Installed lightweight-charts is **5.1.0** (`^5.0.8` in `package.json`). `chart.setCrosshairPosition(price, time, candleSeries)` and `chart.clearCrosshairPosition()` exist and ran without error on the live page:
  - The crosshair price label appeared at 7830.00 on the axis.
  - **0 `crosshairMove` events fired** for a programmatic set or clear (a probe subscriber was attached during the call). The floating tooltip and the linked flow crosshair are not triggered by the panel → chart path, so there is no feedback loop.
  - `time` is required. The panel path has no time of its own, so it needs one, for example the visible range's `to`.
- **Crosshair style** (for L5's "same color and dash"): `horzLine` colour `rgba(191, 219, 254, 0.45)`, width 1, style 2 (dashed), mode 0 (normal; no magnet snapping, so the line sits at the exact price).
- **Interaction note.** The panel overlays the chart's right side, and `pointerInfo(..., rightInset)` already tracks `overLandscapePanel` for the wheel. Over the panel, lightweight-charts receives no mouse events, so its own crosshair hides; the panel → chart path must set it explicitly, as L5 states.

### Q7 — overlays

- **Neg-zone bands** (`LandscapeChartOverlay`).
  - Frame **B**: they come from the landscape.
  - Pixel-aligned with the chart: the SVG is positioned at `top: 0` in the stage, the same as the pane top.
  - Measured on 2026-10-19 (chart zoomed to 7600–7900): zone 7656.48, band centre 493.93 px vs chart y 493.93 px, **error 0.00 px**.
  - In Corrected mode they must come from the Corrected landscape (L2). They follow automatically once the payload is ES-frame; no overlay code change is needed.
- **Confluence price lines and the intraday-subtarget line** on the chart (`series.createPriceLine`).
  - Frame **B**. Lightweight-charts draws them, so they are on the chart's axis by construction.
  - Measured on 2026-10-19: two price lines at 7829.475 ("★ × 2 soft", "→ 7829 confluence").
  - The panel's matching confluence label sits 32 px below the chart line (label text is drawn 3 px above its line), the same 35-px offset.
  - In Corrected mode they must follow L2; they too come from the landscape payload.
- **Other landscape-derived drawings on the chart:** none found. The `PriceChart` effects that read `landscapeData` are the confluence/subtarget effect (`:1975`) and the `LandscapeChartOverlay` mount (`:2913`). Panel-only items (regime badge, drift target, per-bucket legend) are in the panel.
- On 2026-10-06 and 2026-10-23 the payload had no `neg_zones`, confluences or subtarget. With no `iv`, `implied_move` is null and those are omitted; even with iv 0.136 on 10-06 there were none. So Q7 was measured on 2026-10-19.

### Byte-identity plan (Current mode)

Neither route was modified in Step 0. For implementation step 1, use CR-BT's L7 pattern:
- Record the DB reads of `build_gex_landscape_response` (the stored row, plus strikes and params for the high path) for 2026-10-23 into a fixture.
- Capture the golden `/api/gex-landscape` body from the unmodified module.
- Assert byte equality for absent `levels`, `levels=current`, unknown values, and `accuracy=high` with no `levels`.
- `/api/ironbeam/bars` keeps CR-BT's existing golden test.

### Decisions needed (before implementation)

1. **Corrected-mode `prior_spot` / `table_spot`.** The request-time branch sets `table_spot` (and so the classifier's `prior_spot`, used for broken-magnet detection) from `stock_price.iloc[0]`, the first row's per-expiry ORATS price, about S0 (SPX). In Corrected mode the landscape and spot are ES, so this one input would stay in another frame. L2 says "everything else unchanged" but also "both are in one frame".
   - Recommendation: in Corrected mode, `table_spot` = `prior_spot` = P_target, the ES 15:45 ET close on A, used for placement. Then all classifier inputs are ES.
2. **Clipping after the L4 shift.** With the 35-px shift, prices in the chart's top 35 px land above the panel body and are not drawn. The panel body also ends 9 px above the pane bottom.
   - Recommendation: accept, since L9 forbids layout changes. The L4 acceptance (three prices at 10 / 50 / 90 % of the visible range) is unaffected.
