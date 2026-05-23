# CR-016 — Day Setup Audit and Browse Rebuild

## Problem

Three issues with the v1a `/today-setup` page as shipped in CR-015:

1. **Hardcoded `DEFAULT_SPOT = 7400`** in `react_today_setup/src/App.jsx` overrides the day's real spot for any historical date query. The 5/7 case documented in `open-questions/today-setup-5-7-not-classified-as-pin.md` is one instance; we suspect there are more. Because spot is the input to `classify_regime`, this propagates: wrong regime → wrong template selection → wrong proposal.

2. **No way to audit regime classifications.** The page shows a regime badge and a proposal, but there's no quick visual confirmation that either is right. We can't tell whether the proposal is misaligned because the day's regime is misclassified, or because the template logic itself is off, or because the day is just a genuine edge case the classifier handled correctly.

3. **No way to audit analogue rankings.** `AnaloguesPanel` and the `AnalogueDetailModal` show text stats but no visualization of either landscape structure or price action. You can read "max_gex 712B, top cluster 7353.4" and "EOD return −18.2 pts" and still not know whether 5/14 was a *real* analogue of 5/7 or whether the KNN matched on superficial feature overlap.

The pattern across all three: the system makes a structured judgment (regime label, ranked analogues, recommended trade) and gives the user no efficient way to inspect the underlying evidence to confirm or refute it.

## Goal

Rebuild `/today-setup` as a two-mode audit and recommendation surface.

**Mode 1 — Analogue:** Pick a date. See the day's context (date, open price, regime), the day's GEX landscape, the day's mini price chart, the top-K KNN analogues with feature-distance breakdowns, and the v1a proposal cards. Click any analogue → its landscape and mini chart render alongside the anchor's so the two are visible simultaneously. Two flag affordances: "regime label is wrong" (per day) and "this isn't a true analogue" (per pair).

**Mode 2 — Browse:** Pick a regime label and a date range. See *every* day in the corpus matching that filter, with a row list on the left and the same landscape + mini chart pair on the right. Arrow keys cycle through days. Same flag affordances on the per-day flag (the pair flag doesn't apply since there's no anchor in browse mode).

Both modes share the same landscape + mini-chart components and the same flag persistence. The page becomes a tool that does three things at once: surfaces v1a proposals, lets you vet whether the classifier got each day right, and lets you vet whether the KNN found true analogues — with the vetting work captured as structured data for downstream classifier and KNN-weight recalibration.

Also fix the spot-resolution bug as a side effect of the rebuild: when no `spot` param is sent, the backend resolves it from `ironbeam_es_1m_bars` (first RTH bar's open) rather than using a frontend default. Same applies to implied move.

## Changes

### New table — `bt_audit_flags`

DDL at `infra/sql/bt_audit_flags.sql`:

```sql
CREATE TABLE bt_audit_flags (
  flag_id BIGSERIAL PRIMARY KEY,
  flag_type TEXT NOT NULL
    CHECK (flag_type IN ('regime_wrong', 'not_a_true_analogue')),
  ticker TEXT NOT NULL,
  trade_date DATE NOT NULL,
  analogue_date DATE,
  auto_regime TEXT,
  corrected_regime TEXT,
  promoted BOOLEAN NOT NULL DEFAULT FALSE,
  note TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  CONSTRAINT pair_flag_needs_analogue_date
    CHECK (flag_type <> 'not_a_true_analogue' OR analogue_date IS NOT NULL),
  CONSTRAINT regime_flag_needs_corrected_regime
    CHECK (flag_type <> 'regime_wrong' OR corrected_regime IS NOT NULL)
);

CREATE UNIQUE INDEX bt_audit_flags_regime_unique
  ON bt_audit_flags (ticker, trade_date)
  WHERE flag_type = 'regime_wrong';

CREATE UNIQUE INDEX bt_audit_flags_pair_unique
  ON bt_audit_flags (ticker, trade_date, analogue_date)
  WHERE flag_type = 'not_a_true_analogue';
```

**Semantics:**

- `regime_wrong`: one row per (ticker, trade_date). `corrected_regime` is the user's assertion of what the regime should be. `auto_regime` is captured at flag-creation time so we can later analyze "of all days the auto-classifier said X but the user corrected to Y, what's the threshold pattern?"
- `not_a_true_analogue`: one row per (ticker, anchor_trade_date, analogue_date). Directional — flagging (today, 5/14) doesn't affect 5/14's appearance as an analogue for 5/15.
- `promoted`: when `true` on a `regime_wrong` flag, the corrected_regime becomes the effective regime for that day everywhere downstream (proposals, displayed labels). When `false`, the flag is metadata only — auto-regime continues to drive the system.

**Override resolution model:** overrides affect *display and template selection*, not the stored feature vector. `bt_daily_features.feature_vector` is not recomputed when a flag is promoted. This keeps KNN distances stable and the audit data pure for later analysis. The right way to "fix" misclassifications systematically is a follow-up CR that retunes classifier thresholds using the corrected_regime corpus — overrides are an escape valve for one-offs, not a feature.

### Schema change — `bt_daily_features.regime_at_classification`

**Step-0 discovery:** `bt_daily_features` does not currently have a `regime_at_classification` column. The spec requires `create_flag` to capture `auto_regime` from `bt_daily_features` at insert time. The effective-regime helper also needs to read the stored auto-regime when returning the fallback.

**Decision:** Add `regime_at_classification TEXT` column to `bt_daily_features` DDL and update `compute_and_upsert_daily_features` to populate it from the materialized `regime_block["regime"]`. The column is nullable for backwards-compatibility (existing rows have no regime stored; `get_effective_regime` will fall back to re-materializing from `orats_gex_landscape` if the column is NULL).

Migration: `ALTER TABLE bt_daily_features ADD COLUMN IF NOT EXISTS regime_at_classification TEXT;` — applied manually before deployment.

### New table DDL file — schema update for `bt_daily_features`

File: `infra/sql/bt_daily_features_add_regime.sql` — the `ALTER TABLE` migration to add `regime_at_classification`.

### Backend — new endpoints

**`POST /api/audit-flags`** — create a flag.
- Body: `{ flag_type, ticker, trade_date, analogue_date?, corrected_regime?, note? }`
- Server captures `auto_regime` from `bt_daily_features` at insert time.
- Returns the created flag row.

**`DELETE /api/audit-flags/:flag_id`** — remove a flag.

**`POST /api/audit-flags/:flag_id/promote`** — promote a `regime_wrong` flag (sets `promoted=true`). Errors on pair flags.

**`POST /api/audit-flags/:flag_id/demote`** — reverse a promotion (sets `promoted=false`). Symmetric to promote.

**`GET /api/audit-flags?date=YYYY-MM-DD&ticker=SPX`** — fetch all flags relevant to a day (regime_wrong for that day; not_a_true_analogue where trade_date=date OR analogue_date=date).

**`GET /api/days?regime=pin&from=YYYY-MM-DD&to=YYYY-MM-DD&ticker=SPX`** — flat filter for browse mode.
- Returns array of day objects, same row shape as analogues (trade_date, regime, landscape_summary, feature_vector, outcomes, labeled_signals).
- `regime` query value is interpreted against the *effective* regime (auto unless promoted override exists).
- Default `from`: 30 days back. Default `to`: today. Default ticker: SPX.

**`GET /api/bars?date=YYYY-MM-DD&ticker=SPX&session=rth`** — new clean endpoint for mini chart data.
- **Step-0 decision:** Create a new endpoint rather than extending the complex Ironbeam Dash callbacks route (`/api/ironbeam/bars`). The existing route serves the full Dash preview app (GEX overlays, ETH window, multi-day phase, live trade annotations) and is not appropriate to reuse for the mini chart's simple OHLC need. New endpoint at `apps/web/modules/Bars/`.
- RTH session filter: first bar timestamped ≥ 06:30 PT through 13:00 PT on `trade_date`. Convention matches `_fetch_session_outcomes` in Analogues routes: `(datetime AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')::date = trade_date` (first bar on the PT date ≈ 06:30 PT open).
- Returns JSON array of `{time, open, high, low, close}` where `time` is UTC epoch seconds.
- Empty array if no bars for the date; 200 status (not 404) so the mini chart can gracefully empty-state.

### Backend — modifications to existing endpoints

**`/api/setup/proposals`** (in `apps/web/modules/TodaySetup/routes.py`):
- When `spot` is not provided (or ≤ 0), resolve from `ironbeam_es_1m_bars`: open of the first RTH bar on the queried date. Falls back to `table_spot` from `orats_gex_landscape` if no bars exist. Further falls back to a server-side default (5000.0) only if both are unavailable.
- Echo the resolved `spot`, `implied_move`, and `spot_source` (one of `"bars"`, `"landscape"`, `"default"`) back in the response context.
- When serving proposals, check for a promoted `regime_wrong` override on the queried date and use the corrected regime for template selection if present.
- Remove the 400 validation that rejects missing spot — spot is now optional.

**`/api/analogues`** (in `apps/web/modules/Analogues/routes.py`):
- Same spot-resolution change as above.
- Exclude analogue days where a `not_a_true_analogue` flag exists for `(ticker, anchor_date, analogue_date)`.
- For each returned analogue, include a `feature_distances` field: list of the top 5 features by absolute σ-normalized contribution to the distance, with `feature_name`, `anchor_value`, `analogue_value`, `sigma_delta`, `weight`, `contribution`.
- For each returned analogue, include the effective regime label (promoted override applied if present).

### Backend — new modules

**`apps/web/modules/AuditFlags/`** — `__init__.py`, `service.py`, `routes.py`, `tests/`.
- Service is pure functions over a stub-able `conn`: `create_flag`, `delete_flag`, `promote_flag`, `demote_flag`, `list_flags_for_date`.
- Routes wires the five flag endpoints + the lookup.

**`apps/web/modules/DayBrowser/`** — `__init__.py`, `service.py`, `routes.py`, `tests/`.
- Service: `query_days_by_regime(conn, ticker, regime, date_from, date_to)`.
- Routes wires `GET /api/days`.

**`apps/web/modules/Bars/`** — `__init__.py`, `service.py`, `routes.py`, `tests/`.
- Service: `fetch_rth_bars(conn, ticker, trade_date) → list[dict]`.
- Routes wires `GET /api/bars`.

**Override resolution helper** — `packages/shared/audit_overrides.py`:
- `get_effective_regime(conn, ticker, trade_date) → str` — checks for promoted flag, returns corrected_regime or falls back to `bt_daily_features.regime_at_classification` (or re-materializes from landscape if column is NULL).
- `get_effective_regimes(conn, ticker, trade_dates) → dict[date, str]` — batched version for the day-list endpoints.

### Frontend — shared component package

Create `packages/web-shared/` — no existing shared frontend convention in the monorepo. Both `react_today_setup/` and `react_price_preview/` will reference it via a local path in their respective `package.json` (`"web-shared": "file:../../packages/web-shared"`).

**Move:** `GexLandscapePanel.jsx` → `packages/web-shared/src/GexLandscape.jsx`. Rename `GexLandscapePanel` → `GexLandscape` (drop "Panel"). Update both apps' imports. `LandscapeChartOverlay.jsx` stays in `react_price_preview/` (chart-overlay-specific).

**New:** `packages/web-shared/src/MiniPriceChart.jsx`.
- Library: `lightweight-charts` (already in `react_price_preview/`; added to `web-shared` devDependencies as peer).
- Renders RTH-only 1-min bars for a given date.
- Props: `date`, `ticker`, `apiBase`, `clusters` (array of `{center_price, max_gex}`), `height` (default 200).
- Cluster centers render as horizontal price lines, color-coded by GEX sign.
- No interactivity beyond hover crosshair.

### Frontend — `react_today_setup/` restructure

**Component decomposition:**
- `App.jsx` — top-level state, mode toggle, data fetching (significantly rewritten)
- `ContextStrip.jsx` — updated for new fields (open price, spot_source, effective regime)
- `DayListAnalogue.jsx` — new, analogue rows with feature-distance
- `DayListBrowse.jsx` — new, browse rows
- `DayView.jsx` — new, landscape + mini chart pair with flag controls
- `ProposalCard.jsx`, `LegTable.jsx` — exist, unchanged
- `AnalogueDetailModal.jsx` — kept, opened via drill-down "↗" icon

Remove `DEFAULT_SPOT = 7400` — backend resolves spot when none provided.

### Frontend — `react_price_preview/`

Update import for `GexLandscape` (rename from `GexLandscapePanel`). No other functional changes.

## Acceptance Criteria

1. Spot resolved from open on `/today-setup` for any date. `DEFAULT_SPOT = 7400` removed.
2. 5/7/2026 classifies as pin or magnetic-pin.
3. GEX landscape renders inline for both anchor and selected day.
4. Mini price chart renders inline with cluster center horizontal overlays.
5. Analogue mode works with feature-distance breakdown per row.
6. Browse mode returns all corpus days by regime + date range; arrow-key navigation.
7. `regime_wrong` flag: persists, badge updates, promotion flips effective regime, demotion reverts.
8. `not_a_true_analogue` flag: persists, day removed from current anchor's list; other anchors unaffected.
9. Graceful empty states for missing landscape/bars/features data.
10. Backend test suite ~260 tests passing.
11. `vite build` passes for both apps.
12. Production smoke per verification plan.

## Verification Plan

1. **Spot resolution.** Hit `/api/setup/proposals?date=2026-05-07` with no `spot` param. Confirm response context shows spot ~7362.825, regime in pin family.
2. **Open question close-out.** `/today-setup` → 2026-05-07 → context strip shows `pinned` or `magnetic-pin`. Proposals are pin-butterfly variants.
3. **Landscape + chart inline.** Same date. Landscape renders in right pane. Mini chart below/beside with cluster lines.
4. **Analogue swap.** Click analogue row. Selected-day view updates. Both anchor and selected remain visible.
5. **Feature-distance breakdown.** Analogue rows show top contributing features with σ-deltas.
6. **Browse mode.** Toggle to browse. Pick `regime=pin`, full date range. Count matches DB query. Arrow-key navigation.
7. **Regime flag.** Flag 5/7 (or similar), select corrected regime, submit. Badge updates. Persist after refresh. Promote → proposals reflect corrected regime. Demote → revert.
8. **Pair flag.** Flag analogue as "not a true analogue". Disappears from current anchor's list. Appears for different anchor that also had it ranked.
9. **AnalogueDetailModal** drill-down works.
10. **pytest + vite build** green for both apps.

## Out of Scope

- Reclassification CR using audit-flag corpus.
- KNN distance-weight tuning using pair flags.
- Recomputing `bt_daily_features` on flag promotion.
- Bulk-flag workflows.
- Free-form notes UI on flags (schema column reserved, no UI).
- Mobile/narrow-viewport optimization.
- Audit flag access control.

## Step-0 Validation Notes

**Branch state:** Created `feat/CR-016-day-setup-audit-browse-rebuild` off `Main-Live` (commit `77d3808`, CR-015 merged). Clean working tree.

**Shared component location:** No existing shared frontend package in the monorepo. Created `packages/web-shared/` per spec recommendation. Both apps reference via `"web-shared": "file:../../packages/web-shared"` in their `package.json`.

**Bar-series endpoint:** `GET /api/ironbeam/bars` exists but is a full Dash callback serving the price preview app (ETH window, GEX overlays, multi-day phase, live trade annotations). Not appropriate to extend for mini-chart's simple OHLC need. Decision: new clean `GET /api/bars` endpoint in `apps/web/modules/Bars/`. RTH filter: `(datetime AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')::date = trade_date` (matches convention in Analogues `_fetch_session_outcomes`).

**`ironbeam_es_1m_bars` coverage check:** Cannot verify directly without DB access. Mini chart implements graceful empty-state (empty array response → no chart rendered, message displayed). Gaps documented when found during smoke.

**Open price source:** Convention: first row returned by `SELECT ... FROM ironbeam_es_1m_bars WHERE (datetime AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')::date = trade_date ORDER BY datetime ASC` — this is the first bar timestamped on the PT date, nominally 06:30 PT. No existing RTH helper in `packages/shared/`; logic added inline to `Bars` service and `TodaySetup` routes.

**Effective-regime helper / schema discovery:** `bt_daily_features` has no `regime_at_classification` column — only `feature_vector` JSONB. **Schema add required:** `ALTER TABLE bt_daily_features ADD COLUMN IF NOT EXISTS regime_at_classification TEXT;`. `compute_and_upsert_daily_features` updated to populate it from the materialized `regime_block["regime"]`. Column is nullable for backward-compatibility; `get_effective_regime` re-materializes from `orats_gex_landscape` if NULL. Migration SQL at `infra/sql/bt_daily_features_add_regime.sql`.
