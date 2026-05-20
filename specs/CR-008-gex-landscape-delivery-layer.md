# CR-008 — GEX landscape delivery layer

Phase 1b of the GEX landscape build. Phase 1a (`[[2026-05-20 - CR-007 — GEX Landscape Data Pipeline]]`) put the data in `orats_gex_landscape`. CR-008 reads it back out through a backend endpoint and renders it as a right-docked panel beside the price chart. After CR-008 lands, the landscape is **on the chart**.

CR-008 also bumps the cron's `range_pts` default from 200 to 300 (a small data-side change made now so the first-day shipping experience surfaces the full structural backbone rather than edge-clipped artifacts — see CR-007's open question and the spot-check work documented in `[[2026-05-20 - CR-007 — GEX Landscape Data Pipeline]]`). The script's CLI default stays at 200 so the Phase 0 byte-identical regression test continues to pass.

Architecture reference: `[[gex-landscape]]` component spec (build sequence steps 5–6), `[[2026-05-20 - GEX Landscape Spot-Agnostic Storage]]` ADR.

## Problem

Phase 1a (CR-007) lands the GEX landscape data nightly into `orats_gex_landscape` — a JSONB-backed table containing the Gaussian-smoothed field, walls, and per-bucket peaks for each `(ticker, trade_date)`. The data is queryable via SQL but invisible to the user. CR-008 makes it visible: a backend endpoint that classifies the stored field at request time given a live spot + IV, and a right-docked chart panel that renders the result alongside the price chart.

Per `[[2026-05-20 - GEX Landscape Spot-Agnostic Storage]]`, the cron stores the spot-independent field and the endpoint performs spot-dependent classification on demand. CR-007's spot-check work surfaced a refinement of that split: the stored `walls` and `peaks_by_bucket` arrays are **edge-clipped** by `find_peaks` because the grid is centered at `table_spot` (prior-EOD reference), so features near the analytical session-spot may be partially missed. The fix lives in CR-008: recompute walls + peaks_by_bucket at request time from the stored `landscape` field. The stored extracted arrays become cron-side diagnostics; the `landscape` field is the analytical source of truth. CR-008 also bumps `range_pts` to 300 (piece 4 below) so the stored field itself covers a wider analytical window — reducing edge-clipping at the source.

## Proposed Solution

Four pieces:

**1. Endpoint** — `GET /api/gex-landscape` in `apps/web/modules/Ironbeam/callbacks.py`. Thin Flask route: parse args, delegate to a pure builder in `packages/shared/`, return JSON. Following the CR-006 pattern (`build_condor_pricing_payload` in `packages/shared/options_cache/pricing.py` for separation of HTTP/Flask coupling from analytical work).

The builder — `build_gex_landscape_response(conn, ticker, trade_date, spot, *, iv=None, implied_move=None)` — likely landing in `packages/shared/gex_landscape_api.py` (new file, kept separate from the analytical `gex_landscape.py` so the analytical module stays import-light for the cron). The builder:

- Reads the row from `orats_gex_landscape` for `(ticker, trade_date)`. Returns 404 if missing.
- **Recomputes walls + peaks_by_bucket from the stored `landscape` field** using `find_walls` and `find_peaks_per_bucket` from `packages.shared.gex_landscape`. Same algorithm, same thresholds — but called fresh against the JSONB-reconstructed landscape DataFrame.
- Runs the spot-dependent classifier chain — `classify_regime`, `_annotate_distance_class`, `classify_per_bucket`, `summarize_per_bucket`, `analyze_confluence`, `find_intraday_subtarget`, `find_proximate_negative_zones` — on top of those fresh walls/peaks plus the provided `(spot, iv)`.
- Returns the structured dict the Phase 0 script prints, JSON-serialized.

**Amendment — pre-implementation review (`conn` type):** `callbacks.py` has no psycopg connection path — it owns a module-level SQLAlchemy engine (`engine = create_engine(_get_db_url(), pool_pre_ping=True)`) and every DB-reading route uses `with engine.connect() as conn` (e.g. `_fetch_gex_grouped_by_level`). The builder's `conn` parameter is therefore a SQLAlchemy `Connection`, and the route passes `engine.connect()` into it — NOT a psycopg connection. This differs from `compute_and_upsert_landscape(conn, ...)` in `gex_landscape.py`, whose `conn` is psycopg (the cron/backfill path); the two functions share a parameter name but not a connection type. The builder reads its single `orats_gex_landscape` row with `conn.execute(text(...))`; psycopg's JSONB→Python adaptation still applies underneath SQLAlchemy, so `landscape` / `walls` / `peaks_by_bucket` come back as native lists/dicts ready to reconstruct into a DataFrame.

**Important about edge clipping**: recomputing peaks from the stored landscape does NOT add data outside the stored window. CR-008 mitigates this by bumping the cron's `range_pts` default from 200 to 300 (see piece 4 below) — wider stored grid means features previously sitting near the upper edge get prominence-qualifying buffer space on the rising side. For SPX with table_spot offsets typically in the 30–130pt range from session spot, range_pts=300 keeps the analytical window comfortably in-bounds. Edge-clipping is now a theoretical concern for very large overnight gaps rather than a practical one on typical days.

**2. Frontend panel** — `react_price_preview/src/components/GexLandscapePanel.jsx` (new). Right-docked panel matching the visual aesthetic of `outputs/landscape_2026-05-20_stacked.png`. Shares Y-axis range with `PriceChart`. Renders:

- 4 DTE bucket curves (`0DTE` red, `1-7 DTE` amber, `8-30 DTE` green, `30+ DTE` blue) with dominant bucket bolder and others muted. Curves rotated 90°: price on Y, GEX going right.
- Total landscape as a dashed overlay (matches Phase 0 plot's `total` line).
- Spot line — dashed amber, horizontal across both panels.
- Confluence horizontal lines styled by quality grade (pin-grade solid, drift-grade dashed, waypoint dotted) and colored by n_buckets (2=yellow, 3=orange, 4=green). Labels on the right edge: `<price> ★★★ <quality_short>`.
- Negative wall markers — cyan tick marks at neg-wall prices.
- Header — small chip showing current regime (e.g. `MAGNET-ABOVE`) and a two-segment **LIVE / OPEN** spot-mode switch.

LIVE/OPEN mode toggle:
- **LIVE** — panel uses the live ES price the chart is currently tracking. Re-fetches endpoint on spot move > 5pt (debounced). Regime, per-bucket, intraday-subtarget update as price moves.
- **OPEN** — captures the live ES price once when the LANDSCAPE pill is first opened *during* an RTH session (or at RTH open if the pill was already on). Freezes that captured spot for the rest of the session. Regime stays stable through the day.
- Mode preference persists to `localStorage` with key `GEX_LANDSCAPE_SPOT_MODE` (mirroring existing pattern: `FLOW_HEIGHT_STORAGE_KEY`).

**3. Wiring** — `react_price_preview/src/components/PriceChart.jsx` + `react_price_preview/src/App.jsx`.

In `PriceChart.jsx`, append to the existing `pills` array passed to `ChartToggleBar`:

```jsx
{ key: 'landscape',
  label: 'LANDSCAPE',
  isOpen: landscapeOpen,
  onToggle: () => setLandscapeOpen(o => !o),
  title: 'Show GEX landscape panel' }
```

Add `landscapeOpen` state to `PriceChart` (or hoist to `App.jsx` if `landscapeData` lives there).

In `App.jsx`, add `landscapeData` state. Fetch triggers:
- Pill toggled on → first fetch (use either live spot or the OPEN-captured spot per current mode)
- Date changes (chart loads a new session) → re-fetch
- In LIVE mode: spot moves > 5pt → debounced re-fetch (`SMILE_DATA_POLL_MS = 10000`-style cadence, but driven by spot delta not timer — landscape doesn't tick unless price moves materially)
- In OPEN mode: no re-fetch after initial capture

ATM IV is pulled from the same source `SmileChart` uses (Claude Code to verify this path during pre-implementation review — likely `window.parent.document.getElementById('live-data-mirror')` or a sibling state).

**4. Cron + backfill: bump `range_pts` default to 300** — `apps/cron/job_orats_eod.py` + `scripts/backfill_gex_landscape.py`.

Change the `range_pts` argument passed to `compute_and_upsert_landscape` from `200.0` to `300.0` in both the EOD cron call site and the backfill script's default. The helper's signature default in `packages/shared/gex_landscape.py` stays at 200 so the Phase 0 script and the byte-identical refactor regression test continue to work unchanged.

Then re-backfill the four Phase 0 validation dates (2026-05-06, 2026-05-07, 2026-05-18, 2026-05-20) so the table holds the wider window for the days we'll exercise the endpoint against. UPSERT semantics make this safe to re-run.

Expected impact:
- `n_landscape` per row: 401 → 601 grid points
- Cron compute time: ~50% increase (still milliseconds vs the ORATS fetch's seconds; immaterial)
- Storage per row: ~50% larger
- 5/20 row should now surface the +7520 structural magnet inside the stored grid; the endpoint's recomputed walls/peaks will include it (verifiable through the smoke check below)

## Affected Files

- `apps/web/modules/Ironbeam/callbacks.py` — add `GET /api/gex-landscape` route. Thin: parse args, CORS, open `engine.connect()` on the existing module-level SQLAlchemy engine, delegate to builder.
- `packages/shared/gex_landscape_api.py` — **new**. Houses `build_gex_landscape_response(conn, ticker, trade_date, spot, *, iv=None, implied_move=None)`. Pure function, no Flask coupling. Imports from `packages.shared.gex_landscape` for the analytical pipeline.
- `packages/shared/tests/test_gex_landscape_api.py` — **new**. Unit tests on `build_gex_landscape_response` with mocked DB returning known `orats_gex_landscape` rows. Verify response shape, regime classification matches expectation for known spot/IV inputs, 404 when row missing.
- `apps/cron/job_orats_eod.py` — change `range_pts` arg from `200.0` to `300.0` in the `compute_and_upsert_landscape` call.
- `scripts/backfill_gex_landscape.py` — change the script's default `range_pts` from `200.0` to `300.0`.
- `react_price_preview/src/components/GexLandscapePanel.jsx` — **new**. Right-docked panel component.
- `react_price_preview/src/components/PriceChart.jsx` — add LANDSCAPE pill to `pills` array, add `landscapeOpen` state (or lift to App.jsx), embed `<GexLandscapePanel>` when open.
- `react_price_preview/src/App.jsx` — `landscapeData` state, fetch lifecycle (pill-on, date change, debounced spot delta in LIVE mode), IV plumbing.
- `react_price_preview/src/styles.css` — minor additions if needed for panel layout.

## Acceptance Criteria

**Endpoint:**

- `GET /api/gex-landscape?ticker=SPX&date=2026-05-20&spot=7392&implied_move=40` returns 200 with the structured response shape from `[[gex-landscape]]` component spec.
- The response's `regime`, `per_bucket`, `confluences`, `intraday_subtarget`, and `neg_zones` fields match what `python scripts/explore_gex_landscape.py --date 2026-05-20 --spot 7392 --implied-move 40 --range-pts 300` prints (note: script invocation now needs `--range-pts 300` to match the re-backfilled stored grid).
- `walls` and `peaks_by_bucket` in the response are **recomputed from the stored `landscape` field**, not passed through from the stored arrays.
- With the re-backfilled 5/20 row (range_pts=300), the recomputed walls should include the **+7520 structural magnet** (previously missed at range_pts=200). Recomputed `peaks_by_bucket['30+ DTE']` should contain at least one peak around 7520–7530. This is the smoke test that the range_pts bump did its job.
- 404 returned when `(ticker, date)` row doesn't exist.
- 400 returned for invalid params (missing `spot`, malformed `date`, etc.).
- `iv` and `implied_move` are mutually exclusive; if neither is provided, distance classifications return `class: "unknown"` and intraday subtarget / neg zones are omitted.

**Cron + backfill:**

- `compute_and_upsert_landscape` call in `job_orats_eod.py` uses `range_pts=300.0`.
- `backfill_gex_landscape.py` defaults to `range_pts=300.0`.
- After re-backfilling the 4 Phase 0 dates, each row has `range_pts = 300` and `jsonb_array_length(landscape) = 601`.

**Frontend:**

- LANDSCAPE pill appears in `ChartToggleBar` next to existing pills.
- Toggling LANDSCAPE on opens the right-docked panel; toggling off closes it.
- Panel renders the 4 DTE bucket curves with the dominant bucket bolder, total overlay, spot line, confluence horizontal lines styled by quality, negative wall tick markers, and a regime chip + LIVE/OPEN switch in the header.
- LIVE mode: as live ES moves > 5pt from last fetch, endpoint is re-called and the panel updates (regime chip, dominance %, intraday subtarget, neg zone proximity).
- OPEN mode: spot is captured at pill-open (or RTH open if pill was already on), then frozen. Panel doesn't change as price moves.
- Mode preference persists across page reloads.
- Visual match to `outputs/landscape_2026-05-20_stacked.png` (Phase 0 reference). Won't be pixel-identical — different rendering stack (lightweight-charts vs matplotlib) — but the structural layout (curves, legends, confluence lines, spot bar) is the same.

## Verification

**Automated (Claude Code runs):**

- Existing test suite passes (152 baseline from CR-007).
- New tests in `test_gex_landscape_api.py` pass: at minimum, response-shape test, snapshot test on 5/20 with `spot=7392 implied_move=40`, 404 test, 400 tests for bad params.
- Test count: 152 → ~160 expected (give or take).

**Manual (Claude Code writes checklist into PR description):**

- **Re-backfill the 4 Phase 0 validation dates with the new range_pts=300 default** before testing the endpoint. From the worktree: `for d in 2026-05-06 2026-05-07 2026-05-18 2026-05-20; do python scripts/backfill_gex_landscape.py --date $d; done`. Confirm via psql that each row's `range_pts` is now 300 and `jsonb_array_length(landscape) = 601`.
- Hit the endpoint directly via curl: `curl "http://localhost:8060/api/gex-landscape?ticker=SPX&date=2026-05-20&spot=7392&implied_move=40" | jq .`. Confirm shape matches spec; eyeball regime is `magnet-above`, drift_target 7520, neg zone at 7343, and confluences include 7456 + ~7520 with 30+ DTE participating (this is the proof the range_pts bump did its job).
- Compare against script: capture script stdout for the same params (with `--range-pts 300`); verify all classifier outputs match.
- Open the Dash app, navigate to a session with `orats_gex_landscape` data (e.g. 5/20), toggle LANDSCAPE pill. Panel should render the field with regime chip showing `MAGNET-ABOVE` (assuming live ES is around 7390–7400 range matching the Phase 0 case).
- Toggle LIVE / OPEN. Move time-cursor to simulate spot movement (if possible) or wait for live tick; verify LIVE updates and OPEN stays stable.
- Visual smoke against `outputs/landscape_2026-05-20_stacked.png`.

## Out of Scope

Explicitly Phase 1.5 / Phase 2 (NOT CR-008):

- Confluence lines extended across the price chart panel itself
- Negative-zone bands behind candles on the price chart
- Intraday subtarget arrow/annotation on the price chart
- Heatmap halo behind candles
- Yesterday's landscape as a ghost outline
- Time-evolving landscape animation
- Containment-zone highlight bands spanning both panels
- Further `range_pts` tuning beyond 300 (revisit if production usage reveals edge-clipping on extreme overnight-gap days)
- Live intraday landscape recomputation (landscape itself still only updates nightly via cron)

## Handoff Prompt for Claude Code

> Read `specs/CR-008-gex-landscape-delivery-layer.md` and implement it on the current branch (`feat/CR-008-gex-landscape-delivery-layer`).
>
> **Before writing any code, do pre-implementation review (the contradiction-stop rule):**
>
> 1. Grep new symbol names (`build_gex_landscape_response`, `gex_landscape_api` module path, `GexLandscapePanel`) against the existing codebase to confirm they're net-new.
> 2. Read `packages/shared/options_cache/pricing.py` end-to-end as the structural template for the pure-builder pattern. Match it.
> 3. Read `apps/web/modules/Ironbeam/callbacks.py` for the route registration pattern, CORS handling, and how `/api/condor-pricing` was wired. The new route should mirror that exactly.
> 4. Read `react_price_preview/src/components/CondorPricingPanel.jsx` for the right-docked-panel pattern, dimensions, and prop API. The new `GexLandscapePanel.jsx` should follow the same conventions.
> 5. Read `react_price_preview/src/components/SmileChart.jsx` and `react_price_preview/src/App.jsx` to find:
>    - Where ATM IV is sourced from (likely a parent-frame element or shared state). Confirm the path before plumbing it into the landscape fetch.
>    - Where live ES price is tracked. Use the same source for the LANDSCAPE panel's LIVE mode.
>    - The existing `pills` array structure in `PriceChart.jsx`'s `ChartToggleBar` props.
> 6. Read `packages/shared/gex_landscape.py` to confirm `find_walls` and `find_peaks_per_bucket` accept a DataFrame (so reconstruction from JSONB works) and that `classify_regime` / `analyze_confluence` etc. are pure functions of `(landscape, spot, iv)`.
> 7. Verify the `orats_gex_landscape` table is populated for at least `2026-05-20` (used as the primary test fixture). Note: existing rows have `range_pts=200`. This CR re-backfills them at `range_pts=300` as part of the implementation — plan the re-backfill step before any endpoint smoke testing.
> 8. Confirm the helper's `compute_and_upsert_landscape` signature in `packages/shared/gex_landscape.py` accepts `range_pts` as a kwarg with default 200 (Phase 0 reproducibility), and that the cron + backfill call sites are the right places to override.
>
> If pre-implementation review surfaces contradictions between the spec and reality, **stop and amend the spec in its own commit before proceeding**. Each amendment is a separate commit so the audit trail shows what changed and why.
>
> When the endpoint piece is done, **smoke-test against real data with curl before starting the frontend**. The frontend assumes the endpoint returns the documented shape; catch shape issues at the boundary, not in the React tree.
>
> Run all verification commands listed in the Verification section. When finished, write the manual verification checklist into the PR description.
>
> Architectural context lives in the vault:
> - `[[gex-landscape]]` — component spec (build-sequence steps 5–6 are this CR; the design-input callout under step 5 captures the recompute-from-landscape decision)
> - `[[2026-05-20 - GEX Landscape Spot-Agnostic Storage]]` — ADR
> - `[[2026-05-20 - CR-007 — GEX Landscape Data Pipeline]]` — predecessor wrap-up; relevant decisions, lessons, and the edge-clipping open question
> - This session note (`[[2026-05-20 - CR-008 — GEX Landscape Delivery Layer]]`) for the CR-008 spec itself
