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
