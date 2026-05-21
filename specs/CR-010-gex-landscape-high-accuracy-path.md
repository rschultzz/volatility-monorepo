# CR-010 — GEX Landscape High-Accuracy Path

### Problem

CR-008's endpoint reads the pre-computed `landscape` field from `orats_gex_landscape` and runs the classifier chain against the caller-supplied spot. The stored landscape was computed at EOD with `table_spot` (the previous day's close as observed by ORATS) as the grid center, with `range_pts=300` (post-CR-008) extending the analytical window ±150pt either side of `table_spot`.

When the analytical spot at request time drifts materially from `table_spot`, the stored grid is no longer centered on the level the user actually cares about. Two consequences:

1. **Total-field walls survive.** `find_walls` at request time picks them up from anywhere in the stored grid (post-CR-008's range_pts=300 widening). The +7520 magnet on 5/20 appears here.
2. **Per-bucket peak extraction is brittle near grid edges.** `find_peaks_per_bucket` runs on each DTE-bucket's smoothed gex column; `find_peaks`' `prominence` requirement suppresses peaks too close to the grid boundary. The 30+ DTE bucket — which carries the broadest structural features — is the most affected.

5/20 was the canonical demonstration. The 30+ DTE bucket has a magnet at 7520 that:
- With `range_pts=200` (Phase 1a default): completely edge-clipped from the [7257.625, 7457.625] window.
- With `range_pts=300` (CR-008): appears in the total-field walls but the 30+ DTE *bucket-level* peak still doesn't surface — 7520 sits near the upper edge of the [7207.625, 7507.625] window around `table_spot=7357.625`.
- With `range_pts=300` and a spot-centered grid at the analytical spot 7392 → window [7242, 7542] — the +7520 magnet sits well inside the prominent interior of the 30+ DTE bucket curve and surfaces cleanly.

CR-008's wrap-up flagged this as a documented limitation and proposed `accuracy=high` as the future enhancement. CR-010 ships that follow-up.

### Proposed Solution

Add an `accuracy` query param to `/api/gex-landscape` that switches the data source for the landscape field:

| `accuracy` value | Behavior | Latency |
|------------------|----------|---------|
| `low` (default) | Read stored `landscape` JSONB from `orats_gex_landscape`. Current CR-008 behavior. | ~ms |
| `high` | Recompute the landscape from raw `orats_oi_gamma` strikes at request time, using the caller's `spot` as the grid center. | Tens of ms to ~1s depending on the day's strike count |

Implementation lives in `packages/shared/gex_landscape_api.py`'s `build_gex_landscape_response`. New branch at the top of the function: if `accuracy == 'high'`, fetch raw strikes from `orats_oi_gamma` for the requested `(ticker, trade_date)`, call `compute_landscape(df, spot, range_pts, step_pts, spread_coef)` with the caller's spot as the grid center, then continue through the existing classifier chain (`find_walls`, `find_peaks_per_bucket`, `classify_regime`, `_annotate_distance_class`, `classify_per_bucket`, `summarize_per_bucket`, `analyze_confluence`, `find_intraday_subtarget`, `find_proximate_negative_zones`). Otherwise, take the existing fast path.

For `accuracy=high`, the computation parameters (`range_pts`, `step_pts`, `spread_coef`) come from the stored `orats_gex_landscape` row for the same `(ticker, trade_date)` when available — so the recomputed landscape is parameter-compatible with the stored one and only differs in grid centering. If `orats_gex_landscape` has no row, fall back to documented defaults (`range_pts=300`, `step_pts=1`, `spread_coef=8.0` — matching the current cron config). The response indicates which source was used.

Response shape adds three fields, all backward-compatible additions:

- `accuracy`: echoes the resolved accuracy mode (`"low"` or `"high"`).
- `recomputed_at`: ISO timestamp when `accuracy=high`. Null for `accuracy=low`.
- `params_source`: `"stored"` (params from `orats_gex_landscape` row), `"defaults"` (no stored row, fell back to documented defaults), or `null` (irrelevant on the `accuracy=low` path).

All other response fields keep their CR-008 shape — analytic outputs (regime, per_bucket, confluences, intraday_subtarget, neg_zones) come from the same classifier chain regardless of source. The high-accuracy path's `landscape` and `walls` fields will differ from the low-accuracy path's whenever the analytical spot ≠ stored `table_spot`; that's the point.

Param validation: `accuracy` accepts only `low` and `high` (case-insensitive). Unrecognized values → HTTP 400 with a descriptive error.

404 semantics:

- `accuracy=low`: 404 if `orats_gex_landscape` row missing (existing CR-008 behavior).
- `accuracy=high`: 404 if `orats_oi_gamma` has no rows for the requested `(ticker, trade_date)`. Missing `orats_gex_landscape` row is *not* a 404 on the high-accuracy path — params fall back to defaults.

### Affected Files

- `packages/shared/gex_landscape_api.py` — primary surface. Branch in `build_gex_landscape_response` for `accuracy=high`; helper to fetch raw strikes from `orats_oi_gamma`; helper to fetch params from `orats_gex_landscape` with default fallback; new response fields.
- `apps/web/modules/Ironbeam/callbacks.py` — accept `accuracy` query param on the `/api/gex-landscape` route; pass through to the builder; return HTTP 400 on invalid values.
- `packages/shared/tests/test_gex_landscape_api.py` — new tests covering the high-accuracy path. See verification section.

`packages/shared/gex_landscape.py` already exports `compute_landscape`, `find_walls`, `find_peaks_per_bucket`; CR-010 imports them as-is, no changes there. `apps/cron/job_orats_eod.py` is the existing canonical reader of `orats_oi_gamma`; cribbed for the query pattern (no changes to the cron itself).

No frontend changes. `react_price_preview` is untouched.

### Acceptance Criteria

**Endpoint contract:**

1. `GET /api/gex-landscape?ticker=SPX&date=2026-05-20&spot=7392` (no `accuracy` param) returns the exact same response as the current CR-008 behavior, with the addition of `accuracy: "low"`, `recomputed_at: null`, `params_source: null` in the response.
2. `GET /api/gex-landscape?...&accuracy=low` returns the same response as the no-param call (modulo the echoed `accuracy` field).
3. `GET /api/gex-landscape?...&accuracy=high` returns a response with `accuracy: "high"`, a non-null `recomputed_at` ISO timestamp, and `params_source` set to either `"stored"` or `"defaults"`.
4. Invalid `accuracy` values (e.g., `accuracy=medium`, `accuracy=`, `accuracy=foo`) return HTTP 400 with an error body describing the allowed values.
5. `accuracy` is case-insensitive: `accuracy=HIGH`, `accuracy=High`, `accuracy=high` all resolve to `"high"`.

**High-accuracy correctness:**

6. On the 5/20 canonical reference day with `spot=7392`, `accuracy=high` response has `peaks_by_bucket['30+ DTE']` containing a peak at approximately 7520 (the structural magnet documented in CR-008's wrap-up that is *not* present in `accuracy=low` for the same query).
7. The recomputed `landscape` field's grid covers `[spot - range_pts/2, spot + range_pts/2]` (e.g., `[7242, 7542]` for `spot=7392, range_pts=300`), independent of the stored row's `table_spot`.
8. The `walls` field on `accuracy=high` for 5/20 still includes the +7520 magnet (CR-008's range_pts=300 already surfaced this on `accuracy=low` — high-accuracy must not regress it).

**Edge cases:**

9. `accuracy=high` for a date where `orats_oi_gamma` has rows but `orats_gex_landscape` does not: succeeds using default params (`range_pts=300`, `step_pts=1`, `spread_coef=8.0`). Response has `params_source: "defaults"`.
10. `accuracy=high` for a date where `orats_oi_gamma` has no rows: returns HTTP 404 with a descriptive body.
11. `accuracy=high` without providing `spot` (already required on the endpoint pre-CR-010): returns HTTP 400 (same as current CR-008 behavior for missing required params).
12. `accuracy=low` for a date where `orats_gex_landscape` has no row: returns HTTP 404 (unchanged CR-008 behavior — `accuracy=high` is the way to get data when the stored landscape is missing).

**Performance:**

13. `accuracy=high` for a typical day's strike count returns in under 2 seconds against the production DB. If recompute consistently exceeds 2s on the 5/20 reference, raise it as a defect in the spec amendment cycle rather than shipping.

### Verification

**Automated:**

- `vite build` from `react_price_preview/` passes (sanity — no frontend changes).
- Backend test suite expands from 165 to ~172-176 (target: 7-11 new tests covering each AC path).
- New tests in `test_gex_landscape_api.py`:
  - `test_accuracy_default_is_low`: default (no param) → response has `accuracy: "low"`, matches no-param CR-008 response.
  - `test_accuracy_low_explicit_matches_default`: `accuracy=low` matches default behavior.
  - `test_accuracy_high_basic_shape`: returns response with `accuracy: "high"`, non-null `recomputed_at`, and the standard top-level shape.
  - `test_accuracy_high_recenters_grid`: on a synthetic dataset, `accuracy=high` produces a landscape with grid centered on `spot` (not `table_spot`); verify grid bounds.
  - `test_accuracy_high_5_20_30dte_bucket_peak`: on 5/20 reference data, `peaks_by_bucket['30+ DTE']` contains a peak at ~7520.
  - `test_accuracy_high_5_20_walls_no_regression`: on 5/20 reference data, the walls field still includes the +7520 magnet.
  - `test_accuracy_invalid_returns_400`: `accuracy=medium` → 400; payload includes allowed values.
  - `test_accuracy_case_insensitive`: `accuracy=HIGH` works same as `accuracy=high`.
  - `test_accuracy_high_missing_strikes_returns_404`: `accuracy=high` with no `orats_oi_gamma` rows → 404.
  - `test_accuracy_high_falls_back_to_default_params`: `accuracy=high` for a date with `orats_oi_gamma` but no `orats_gex_landscape` row → uses defaults, response shows `params_source: "defaults"`.
  - `test_accuracy_low_missing_landscape_returns_404`: `accuracy=low` (or default) for a date with no `orats_gex_landscape` row → 404 (CR-008 behavior, asserted unchanged).

**Manual smoke against production DB:**

- `curl 'http://localhost:8060/api/gex-landscape?ticker=SPX&date=2026-05-20&spot=7392&implied_move=40&accuracy=low'` → 200, matches the current 5/20 response, with new echo fields.
- Same with `accuracy=high` → 200, structurally similar response. Diff the two responses; `landscape` field's grid is different (recentered on 7392 → [7242, 7542]), and `peaks_by_bucket['30+ DTE']` contains the ~7520 peak that's missing in `accuracy=low`.
- Time the high-accuracy request (e.g., `curl -w '%{time_total}\n' -o /dev/null -s ...`); log the latency in the PR description.

### Out of Scope

Explicitly deferred:

- **Frontend integration.** The endpoint param ships available; React frontend continues polling with `accuracy=low` (default). A future CR can wire a UI toggle, an auto-trigger when 30+ DTE confluences are suspected but not surfaced on `accuracy=low`, or always-use-high mode. Not for CR-010.
- **Caching the recomputed landscape.** Each `accuracy=high` request recomputes from scratch. If `accuracy=high` becomes a hot path in production usage, a future CR can cache by `(date, spot bucket)` or memoize across small spot deltas. CR-010 measures latency and flags concerns; doesn't pre-optimize.
- **Caller override of `range_pts` / `step_pts` / `spread_coef`.** Caller can't override these via the request; they come from the stored row or defaults. A future CR can expose them as request params if a use case emerges.
- **Alternative grid-centering strategies.** Caller's `spot` is the only supported grid center. No volume-weighted, ATM-strike, session-VWAP, or other centering modes.
- **Live intraday recomputation of `orats_oi_gamma` itself.** The high-accuracy path still uses whatever strikes are in `orats_oi_gamma` for the requested date (today's row contains yesterday's strikes via the write-time-labeling convention). True intraday strike recomputation is a much bigger Phase 2 item.
- **Per-bucket grid centering.** Each bucket could in principle have a grid centered on its own peak rather than on `spot`; would catch even more bucket-level peaks but introduces non-uniform grids across buckets, which complicates downstream classification. Not for CR-010.

### Handoff prompt (implementation phase)

> **You are implementing CR-010 — GEX Landscape High-Accuracy Path.** The spec is in `specs/CR-010-gex-landscape-high-accuracy-path.md`. Branch `feat/CR-010-gex-landscape-high-accuracy-path` off `Main-Live`.
>
> **Step 0 — Read these in full before writing any code:**
> - `specs/CR-010-gex-landscape-high-accuracy-path.md` (this spec)
> - `packages/shared/gex_landscape_api.py` (existing `build_gex_landscape_response` — the function being extended)
> - `packages/shared/gex_landscape.py` (`compute_landscape`, `find_walls`, `find_peaks_per_bucket`, and the rest of the classifier chain)
> - `apps/web/modules/Ironbeam/callbacks.py` (the `/api/gex-landscape` route handler)
> - `packages/shared/tests/test_gex_landscape_api.py` (existing test patterns to mirror)
> - `apps/cron/job_orats_eod.py` (reference for how the cron reads `orats_oi_gamma` and calls `compute_and_upsert_landscape` — the high-accuracy path is essentially that computation, at request time with caller's spot as the grid center)
> - `specs/CR-008-gex-landscape-delivery-layer.md` (predecessor — endpoint contract and response shape)
>
> **Step 1 — Pre-implementation review. Run the contradiction-stop checklist before writing any code.** Amend the spec in its own commit for each contradiction found:
>
> 1. **Confirm the `orats_oi_gamma` query pattern.** The high-accuracy path needs to fetch raw strikes for `(ticker, trade_date)`. Look at how `job_orats_eod.py` does this (or whoever else queries `orats_oi_gamma` directly inside `packages/shared`). Reuse the existing pattern. If the cron uses psycopg but `build_gex_landscape_response` is called with a SQLAlchemy `Connection` (per CR-008 amendment `7ee3b3a`), adapt the query to the SQLAlchemy connection type — don't invent a new query pattern.
>
> 2. **Verify `compute_landscape` signature.** The spec assumes `compute_landscape(df, spot, range_pts, step_pts, spread_coef) -> landscape_df`. Confirm against the actual function in `packages/shared/gex_landscape.py`. If it differs (e.g., keyword-only args, different param names, an `iv` requirement), amend the spec.
>
> 3. **Verify the classifier chain operates on the recomputed landscape.** The chain (`find_walls`, `find_peaks_per_bucket`, `classify_regime`, etc.) needs to consume the freshly-computed landscape DataFrame in the same shape it consumes the stored-reconstructed one. The spec assumes shape parity because `compute_and_upsert_landscape` writes the same shape, but confirm by tracing the dtype and column set through both paths. If the recomputed landscape has any column shape divergence, amend.
>
> 4. **Resolve `params_source` semantics.** The spec introduces three new response fields (`accuracy`, `recomputed_at`, `params_source`). `params_source` is the most novel — it disambiguates whether the high-accuracy path used stored params or defaults. Confirm this is a sensible API surface: would it be better to merge into a richer `accuracy` value (e.g., `"high_stored_params"` vs `"high_default_params"`)? Or to drop entirely if the distinction doesn't matter to callers? Decide and amend the spec if the chosen representation differs from what's drafted.
>
> 5. **Performance check before final commit.** Run the high-accuracy path against 5/20 reference data and measure latency. If it exceeds 2s, that's a defect — flag in a spec amendment with the observed timing and either (a) optimize the recompute path, or (b) reduce scope (e.g., skip some classifier step that's not actually needed on the high-accuracy path).
>
> **Step 2 — Implementation.** Once the spec is reconciled with reality, implement per the Proposed Solution. Suggested commit slicing — each commit a working state:
>
> 1. Wire the `accuracy` query param through `callbacks.py` (validation, pass-through, 400 on invalid). Default behavior unchanged from CR-008 at this point.
> 2. Add the high-accuracy branch in `build_gex_landscape_response`. Fetch strikes from `orats_oi_gamma`, recompute landscape with caller's spot, run classifier chain. Returns the response with new fields.
> 3. Add the params-source-defaults fallback (the case where `orats_gex_landscape` row is missing on the high-accuracy path).
> 4. Tests for all AC paths.
>
> **Step 3 — Verification.** Backend test suite passes (was 165; target ~172-176). Manual smoke against the 5/20 reference data, both `accuracy=low` and `accuracy=high`, verifying the AC #6 30+ DTE bucket peak appears on high-accuracy. Time the high-accuracy request; note in the PR description.
>
> **Step 4 — PR.** Open against `Main-Live`. Title: `CR-010 — GEX Landscape High-Accuracy Path`. Body: AC checklist + the smoke output diff (low vs high for 5/20, with the 30+ DTE bucket peak highlighted) + latency measurement. `Create a merge commit` (not Squash).
>
> **Deployment caveat:** CR-010 only changes the web service (no cron-side code changes). The `volatility-web` service in `render.yaml` has `autoDeploy: true` and should pick up Main-Live changes normally. The cron worker's auto-deploy gap ([[orats-eod-cron-not-autodeploying]]) doesn't affect this CR, but if the endpoint behaves like it doesn't know about `accuracy` after merge, check the web service's deployed commit hash too.
