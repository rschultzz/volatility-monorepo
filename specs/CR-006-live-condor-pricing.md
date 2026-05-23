CR-006 — Live condor pricing on the dashboard
Context
Surfaces the price of a 1-σ condor on the live trading view at two timepoints: the entry timeslice the user selects on the smile popup, and a live evaluation point that polls until market close, then snaps to the close minute. Delta between the two is the running P&L of a hypothetical position. First user-facing consumer of the options cache infrastructure shipped in CR-003/004/005, and the structural sibling of Phase 5 P&L reconstruction (same data shape, different trigger, single-instance volume).
Discussed and resolved in chat before drafting: eval scope, click model (no drag), endpoint family (historical fetch_option_bars for both timepoints — the live monies pipeline is bucketed and doesn't carry OPRA quotes), shared helper for strike math, single endpoint shape, 10s poll cadence matching SMILE_DATA_POLL_MS.
Decision
Land four pieces:

New shared helper condor_strikes_from_smile(spx, iv_pct, minutes_to_expiry, *, wing_width_pts=10.0, strike_increment=5.0) → dict in packages/shared/options_cache/condor.py. Pure function. Same math currently inlined in App.jsx bandsLevels useMemo and (per the comment there) duplicated in apps/web/modules/BacktestsV2/service.py::_compute_hypothetical_condor. Becomes the single source of truth. Wing width and strike increment are keyword-only with defaults so the React frontend can pass three positional args and inherit the same numbers it currently hardcodes (wing=10, inc=5), while the backend _compute_hypothetical_condor preserves its existing pass-through of condor_wing_width_pts from the routes layer (default 10.0, overridable via condorWingWidthPts in the API params at apps/web/modules/BacktestsV2/routes.py:866). Defining WING_WIDTH/STRIKE_INCREMENT as module-level constants — the original draft of this spec — was rejected during pre-implementation review because it would silently strip backend configurability.
Refactor _compute_hypothetical_condor in BacktestsV2/service.py to call the new helper, passing condor_wing_width_pts and strike_increment through as keyword args (the existing call sites already accept these as parameters at lines 1325, 1357). The outer function's signature, return-dict shape, and the implied_sigma path through _implied_sigma_move are preserved — only the strike-rounding inner block delegates to the helper. Behavior unchanged; identical outputs verified by snapshot test against the existing scan cache.
New backend endpoint GET /api/condor-pricing in apps/web/modules/Ironbeam/callbacks.py, alongside the other React-facing routes (/api/ironbeam/smile-data, /api/ironbeam/atm-iv-series). The callback itself is a thin orchestrator (CORS, query-param parsing, session construction, JSON serialization) — all pricing logic lives in a new pure builder build_condor_pricing_payload(...) in packages/shared/options_cache/pricing.py. Builder computes strikes via the shared helper, calls fetch_option_bars for entry and eval minutes, returns strikes + leg prices + P&L summary. Splitting the pure builder out is what makes the endpoint logic unit-testable under the project's existing test convention (packages/shared/options_cache/tests/ — verified during pre-impl review as the only Python test location in the repo; apps/ has no tests/ subdirectories and no top-level pyproject.toml/pytest.ini wiring discovery there).
Frontend wiring in react_price_preview/: replace BOTH the strike computation AND the inline σ formula in App.jsx::bandsLevels with values from the new endpoint (strikes + sigma_pts). The display-coord band-line math (sigmaUpper = anchorEs + sigmaPts, sigmaLower = anchorEs - sigmaPts) stays in App.jsx — that's still a chart-coord concern with the ES basis correction. But the σ-in-SPX-coords formula itself (spx * (ivPct/100) * sqrt(minsToExp / (60*24*365)), at App.jsx:706) is duplicated against backend's _implied_sigma_move and consolidates here. Add poll loop matching the smile-data pattern; new CondorPricingPanel.jsx component rendered as a floating overlay inside PriceChart.jsx near the bands.

Behavior

Activating a smile anchor button (existing flow) sets bandsAnchor (existing state) AND fires an initial fetch to the new endpoint.
Response carries strikes (replaces local computation) + entry leg prices + eval leg prices + P&L summary.
If isLiveTradeDate is true and market is open, poll the endpoint every 10s with eval_pt=now. Each poll: backend translates "now" to the most recently completed minute, routes through fetch_option_bars. Cache-aware — first hit on a new minute makes an ORATS call (≈4 calls per minute when 4 unique OPRAs aren't yet cached for that minute), subsequent hits within the same minute return from cache.
When market closes for the selected session, stop polling. Final eval point = the close minute (12:59 PT).
When bandsAnchor is cleared (user clicks ✕ Off or deactivates), pricing panel disappears and polling stops.

API shape — GET /api/condor-pricing
Query params: trade_date, expiration_date, spx, iv_pct, minutes_to_expiry, entry_pt (HH:MM), eval_pt (HH:MM or "now").
Response payload:
json{
  "sigma_pts": 12.34,
  "strikes": {
    "short_put": 5780,
    "long_put":  5770,
    "short_call": 5810,
    "long_call":  5820
  },
  "opras": {
    "short_put":  "SPX  260519P05780000",
    "long_put":   "SPX  260519P05770000",
    "short_call": "SPX  260519C05810000",
    "long_call":  "SPX  260519C05820000"
  },
  "entry": {
    "snapshot_pt": "07:32",
    "legs": {
      "short_put":  {"bid": 1.20, "ask": 1.25, "mid": 1.225},
      "long_put":   {"bid": 0.85, "ask": 0.90, "mid": 0.875},
      "short_call": {"bid": 1.10, "ask": 1.15, "mid": 1.125},
      "long_call":  {"bid": 0.75, "ask": 0.80, "mid": 0.775}
    },
    "net_credit": 0.700
  },
  "eval": {
    "snapshot_pt": "08:42",
    "is_live": true,
    "legs": { /* same shape */ },
    "net_cost_to_close": 0.520
  },
  "pnl": {
    "gross": 0.180,
    "per_leg": { /* mid delta per leg */ }
  },
  "warnings": []
}
Sigma convention. sigma_pts is the raw 1σ in SPX price points, computed as spx * (iv_pct/100) * sqrt(minutes_to_expiry / (60*24*365)) via _implied_sigma_move. Returned both as a top-level field and (via the helper internally) consumed to round the strikes. Frontend uses it for the σ band lines; previously this formula was duplicated in App.jsx and consolidates here.
Pricing convention. Mid = (bid + ask) / 2 from orats_options_minute. Bid and ask exposed in the per-leg payload so the frontend can also show bid/ask spread if useful later. Net credit (entry) = mid(short_put) + mid(short_call) − mid(long_put) − mid(long_call). Net cost to close (eval) = same formula at eval prices. P&L = net_credit − net_cost_to_close (positive = profit).
Cache-miss handling. Synchronous fetch. If fetch_option_bars for the entry minute hits ORATS (rare — most clicks will be on minutes already in cache after the user's other workflows), the response blocks until the ORATS call returns (~500ms-1s per OPRA, parallelizable across the 4 legs). If ORATS returns 404 for an illiquid OPRA (the CR-005 e2e run saw 10 of these out of 1,698), add a warnings entry and return the legs that did fetch; frontend renders "—" for the missing leg and skips the P&L sum.
UX placement (recommendation, flag if wrong). Floating panel inside PriceChart.jsx, anchored top-right of the bands region. Shows: entry credit, eval net, gross P&L (color-coded), eval timestamp (with a live dot when polling is active). Disappears when the anchor is off. Rationale: bands and strike lines live on PriceChart and persist after the smile popup is dismissed; the pricing should travel with them, not the popup. Alternative considered: embedding inside the smile's "Bands @" button row — rejected because it dies when the popup closes.
Affected files

packages/shared/options_cache/condor.py — add condor_strikes_from_smile(spx, iv_pct, minutes_to_expiry, *, wing_width_pts=10.0, strike_increment=5.0) returning {short_put, long_put, short_call, long_call, sigma_pts}. Mirrors the math from App.jsx bandsLevels exactly: floor for short_put, ceil for short_call, round for the wings. sigma_pts is the raw 1σ in price points (pre-rounding) so callers can re-use it without recomputing. No module-level constants — wing width and strike increment are keyword-only function defaults instead, preserving the backend's runtime configurability for these knobs.
apps/web/modules/BacktestsV2/service.py — refactor _compute_hypothetical_condor at line 1734 to delegate the strike-rounding block to condor_strikes_from_smile. Existing signature (anchor_price, iv_atm_0dte_pct, horizon_minutes, wing_width_pts, strike_increment=5.0) and return-dict shape (short_put_strike / long_put_strike / short_call_strike / long_call_strike / implied_1sigma_pts / short_strike_width / wing_width_pts) preserved — both call sites at lines 1325 and 1357 are untouched. The helper's sigma_pts → outer dict's implied_1sigma_pts; rounded strike keys are renamed at the boundary. Module is private to service.py (not re-exported from BacktestsV2/__init__.py — verified during pre-impl review), so no public-surface impact.
apps/web/modules/Ironbeam/callbacks.py — register new /api/condor-pricing route as a thin endpoint, alongside the existing ironbeam_react_smile_data_api at line 2842. Reuses the registration-guard pattern (app.server._ironbeam_react_condor_pricing_route_registered = True) and CORS handling from the surrounding routes. Endpoint body: parse query params, build the SQLAlchemy session, delegate to build_condor_pricing_payload from packages.shared.options_cache.pricing, return (payload, status). No business logic in callbacks.py.
packages/shared/options_cache/pricing.py — new. Exposes build_condor_pricing_payload(session, *, trade_date, expiration_date, spx, iv_pct, minutes_to_expiry, entry_pt, eval_pt) → (payload_dict, http_status). Pure with respect to its injected session — accepts any SQLAlchemy session, which makes it mockable for unit tests. Internally: derive strikes via condor_strikes_from_smile, format OPRAs via existing format_opra in opra.py, call fetch_option_bars (CR-004) for entry and eval minutes, query orats_options_minute for the 4 OPRAs at each minute, compute mids + net_credit + net_cost_to_close + P&L, handle the "eval_pt = now" → most-recent-completed-minute translation, attach warnings on 404s.
react_price_preview/src/App.jsx — bandsLevels useMemo: delete the inline sigmaPts formula at line 706 and the strike computation at lines 715-718; pull both from condorPricing.sigma_pts and condorPricing.strikes. Keep the band-line math (sigmaUpper = anchorEs + sigmaPts, sigmaLower = anchorEs - sigmaPts) and the basis translation (shortPut = spxShortPut + basis, etc.) — these are display-coord transforms that still need to happen on the client. New useEffect for the polling loop, mirroring the smile-data poll structure (~lines 700-740) including the disposed/inFlight/activeController pattern. Stop condition: !bandsAnchor || !isLiveTradeDate (with snap-to-close handling when !is_market_hours).
react_price_preview/src/components/CondorPricingPanel.jsx — new. Receives condorPricing prop, renders entry/eval/delta. Styled to match existing chart overlays (~10px font, dark bg, positioned absolute).
react_price_preview/src/components/PriceChart.jsx — render <CondorPricingPanel> when condorPricing != null, positioned top-right inside the chart container.

Pre-implementation greps (CR-004/005 lesson — surface contradictions before writing code)

Grep repo for _compute_hypothetical_condor — enumerate callers. If anything besides the scan-row pipeline calls it, surface to chat before refactoring.
Grep bandsLevels and the four strike variable names (spxShortPut, etc.) in react_price_preview/ — confirm no other React file recomputes strikes locally.
Grep apps/web/modules/Ironbeam/callbacks.py for _ironbeam_react_*_route_registered pattern — confirm the registration-guard style and follow it for the new route. (Pre-impl: confirmed; existing guards live at lines 2680/2734/2736/2770/2772/2808/2810/2838/2840/2863/2865/2887. New guard name: _ironbeam_react_condor_pricing_route_registered.)
Grep orats_options_minute schema (infra/sql/ or repository.py) — confirm bid_price, ask_price, snapshot_pt are the canonical columns to read. Spec assumes these per OptionMinuteBar in models.py.
Grep apps/web/modules/BacktestsV2/__init__.py and any re-export surfaces for _compute_hypothetical_condor — if it's re-exported anywhere, the refactor must preserve the import path.

Test plan

Unit tests for condor_strikes_from_smile in packages/shared/options_cache/tests/test_condor.py (extend existing file): floor/ceil/round behavior at strike-increment boundaries, wing-width application, regression test that backend output equals the frontend's math for a known (spx, iv, minutes) tuple.
Snapshot test for _compute_hypothetical_condor refactor: sample 5-10 scan rows from bt2_scan_cache, capture current outputs, refactor, confirm outputs identical.
Mock-based tests for build_condor_pricing_payload in a new packages/shared/options_cache/tests/test_pricing.py: cache-hit path (orats_options_minute returns rows for all 4 OPRAs at both minutes), ORATS-fallback path (no rows for entry minute → fetch_option_bars invoked → rows present on re-query), partial-failure path (1 leg returns 404 → warnings entry, P&L sum skipped). Session is a unittest.mock.MagicMock; fetch_option_bars is patched at the import site in the pricing module. Tests live under the existing packages/shared/options_cache/tests/ convention (the project has no apps/ test directories or test-discovery wiring — verified pre-impl).
No new smoke test required — the eval-pt polling is exercised by manual e2e (see acceptance below).

Acceptance / e2e verification
Manual run with a recent trade date in the dashboard. Steps:

Load dashboard for a recent trade date (yesterday or earlier).
Open smile popup, select 2-3 timeslices, activate σ bands from one.
Confirm pricing panel renders with entry credit ≠ 0, eval net populated, P&L color-coded.
Confirm panel updates on subsequent 10s polls (eval_pt advances, eval prices change minute-over-minute).
After market close (or simulating closed state), confirm polling stops and panel freezes at close minute.
Switch anchors mid-session — confirm new fetch fires, panel updates to reflect new strikes.
Click ✕ Off — confirm panel disappears and polling stops.
Spot-check one row by hand: pull orats_options_minute for the 4 OPRAs at the entry minute, compute mid by hand, compare to displayed net credit (should match to floating-point).

Workflow notes
Branch: feat/CR-006-live-condor-pricing. Spec lands as its own commit. Per CR-004/005 pattern: hand off to Claude Code with explicit contradiction-stop instruction; expect 2-4 spec amendments before implementation starts.