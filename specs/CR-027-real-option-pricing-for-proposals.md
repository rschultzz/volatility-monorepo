CR-027 — Real Option Pricing for Proposals

## Goal

Make the proposal edge engine and P/L chart trade against numbers a human could actually transact at:

1. **Real entry cost** — each leg priced at its real ORATS bid/ask mid at an entry minute (not BSM, not T=0).
2. **Real market-implied distribution** — Breeden-Litzenberger over a dense band of real per-strike mids around the trade-thesis range (skew-bearing), feeding the unchanged `implied_distribution.py`.
3. **Correct P/L curves at both expiry and horizon** — BSM retained only for curve shape, calibrated so the curve passes through the real entry mid; render at-expiry and t1/t5/t15 horizon curves.

The structural/analogue half (`structural_distribution.py`, KNN) is already real and is **not touched** by this CR.

## Context

- The reusable infrastructure already exists and is in production via CR-006 (Live Condor Pricing): `build_condor_pricing_payload` (`packages/shared/options_cache/pricing.py`) prices option legs from `orats_options_minute` (mid = `(bid_price + ask_price)/2`) using `fetch_option_bars` — a gap-aware, write-through, idempotent primitive. The live condor panel already calls this from the web app in production, so the web app's DB role can already write the cache through this path. **This CR is reuse, not new pricing infrastructure.**
- Verified against real schema + a 2023-07-28 sample (2026-05-29): `bid_price`/`ask_price` present and tight; historical coverage exists at minute granularity; real skew visible in the data.
- This is the real-pricing dependency CR-G silently lacked. Everything downstream that consumes edge/P&L numbers — CR-F (debit/credit variants), CR-H (interactive editing), and the candidate position-aware-EV / multi-DTE CRs — should land *after* this so they build on real numbers.

## Critical design constraint (verified, load-bearing)

**Proposal `Leg.strike` is in ES discounted-forward space, NOT SPX cash.** `strategy_templates.py` sets `Leg.strike` directly from `drift_target` / `containment_zone` / cluster centers, all of which live in discounted-forward coordinates (`gex_landscape.py` computes the landscape over `discounted_level`; CR-G Step 8 documented this). `orats_options_minute` is keyed on real SPX cash strikes (e.g. 4600). So **every leg strike must be converted ES→SPX before OPRA construction**, via the existing `compute_spx_strike(strike_es, dte, r, q)` in `packages/shared/forward_math.py` (inverse of the ingest `compute_discounted_level`, `t = (dte+1)/252`, rounds to nearest 5). Pricing a leg off its raw ES strike would fetch the wrong (or a nonexistent) OPRA. This is the single most important correctness point in the CR.

## Verified facts the implementation relies on (from 2026-05-29 code read)

- `fetch_option_bars(opra_symbols: Sequence[str], start_pt, end_pt, *, source="historical_backfill") -> FetchOptionBarsSummary` — takes a **list** of OPRAs, gap-aware per OPRA, writes bars to `orats_options_minute` and windows to `orats_options_fetched_windows`, idempotent. Each ORATS row emits both call AND put bars at that strike+expir (so fetching a put also caches the call). Datetimes are **naive PT**; tz-aware raises `ValueError`.
- `repo.get_bars_for_contract(opra, start_pt, end_pt) -> list[OptionMinuteBar]` — the cache read.
- `format_opra(root, expir: date, "C"|"P", strike: float) -> str` — OPRA encoder. Strike encoded as `round(strike*1000)`, must be `< 10^8`. **Root is `"SPX"` for all SPX expirations including PM-settled weeklies; AM/PM distinguished by `expiry_tod`, not the root.** (This is an open accuracy question — see Step 0.)
- `OptionMinuteBar` fields available: `bid_price`, `ask_price`, `strike`, `option_type`, `expir_date_d`, `dte`, plus `bid_iv`/`mid_iv`/`ask_iv` and per-strike greeks.
- `compute_implied_pdf(option_chain, spot, r, tte)` consumes `[{"strike", "call_price"}]` where `call_price` is documented as a **mid-price** — it is already written for real input and needs no change. Sparse path triggers at `< 8 strikes` or `> 25pt` spacing.
- `OratsPermanentError` (404 / illiquid) and `OratsError` (transient) are the catch types; the condor path treats a permanent error as "leg unavailable → None + warning" rather than failing the request.
- Proposals route currently: `evaluation_time = build_evaluation_time(shortest_expir)` (= expiry); step 6 prices `initial_cost` at that time → T=0 → zero debit for OTM legs (the bug). `_fetch_smile_row` returns `(atmiv, risk_free_rate, yield_rate)`. Per-leg `strike_spx` is already computed for display in step 18 via `compute_spx_strike`.

## Pre-implementation greps (contradiction-stop)

Before writing any implementation code, run these and STOP on any contradiction:

1. Grep `compute_initial_cost` callers across `apps/` and `packages/` — confirm the only consumer of the BSM entry-cost is the proposals route (so demoting it to "curve calibration only" doesn't break another reader).
2. Grep `build_bsm_chain` callers — confirm proposals route step 9 is the only consumer before replacing its role.
3. Grep `compute_pl_curve` callers — Step 9 of CR-G made it regime-aware; confirm signature/callers before threading a real entry cost through it.
4. Confirm `orats_options_minute` is keyed/uniqued on `(opra_symbol, snapshot_pt)` and that historical rows for SPX exist for at least 3 test anchor dates spanning the corpus.
5. Confirm `expiry_tod` values present in `orats_options_minute` for SPX (AM vs PM).
6. Confirm `format_opra` + `opra_to_orats_ticker` round-trips for a known cached SPX OPRA from the sample.
7. Grep the proposals frontend (`react_today_setup` / `ProposalCard` / `LegTable`) for where `pl_curve` / `legs[].initial_value` / `edge` are consumed — Step 5/6 rendering changes must match the existing prop shape.

## Step 0 — Diagnosis and design lock

Resolve before any code:

1. **SPX vs SPXW root / AM-PM settlement.** Determine from `expiry_tod` in the cache whether the proposal's target expiration is AM or PM settled, and whether that affects which OPRA actually has quotes. Stop condition if ambiguous: surface rather than guess.
2. **Entry-time definition.** Define `entry_pt` for a proposal. Recommendation: the session open minute on `trade_date` (06:30 PT) or a fixed reference (e.g. 07:00 PT). Lock one convention.
3. **Strike band width + density for the implied distribution.** Cover at least `spot ± 1.5×IM` at 5pt spacing. Lock the band.
4. **BSM calibration choice for curve shape.** Calibrate each leg's BSM vol so the curve passes through the real entry mid (recommended). Lock one.
5. **Live vs historical scope for v1.** Confirm or fall back based on Step 0-B coverage findings.
6. **Per-leg failure semantics.** Mirror condor path: `OratsPermanentError` → warning + None, excluded from net debit. Lock.
7. **Fetch trigger — LOCKED:** prefetch on day-load, non-blocking, band-once-per-day.

## Step 0-A — Fix the zero-debit evaluation-time bug

**Commit:** `cr-t/step-0a: price proposal entry cost at entry time, not expiry (zero-debit fix)`

- Add `build_entry_time(trade_date)` helper or generalize `build_evaluation_time`.
- Entry time = session open on `trade_date` (06:30 PT or 07:00 PT per Step 0 lock).
- In `routes.py`, price `initial_cost` (and per-leg `initial_value`) at `entry_time`, NOT `evaluation_time`.
- This step still uses BSM; the point is that BSM at a correct entry T produces a non-zero debit and a loss region.

**Deliverable:** OTM debit spreads render with a negative left tail; `compute_key_levels` finds a true breakeven at long-strike + debit; max-loss line non-zero. Unit test: a known OTM debit spread has `initial_cost > 0` and `min(pnl) < 0`.

## Step 0-B — Historical coverage probe (no commits)

For 3 test anchors spanning the corpus, build the 4-leg OPRAs (post ES→SPX conversion) and check `count_bars_for_contract` at the entry minute. Record hit/miss. For misses, attempt one `fetch_option_bars` and confirm whether ORATS historical-intraday returns data. Output a coverage table into the CR status updates. Stop condition: if all 3 anchors miss AND fetch returns 404, fall back to live-only v1 and surface.

## Step 1 — Proposal leg → real mid pricing helper

**Commit:** `cr-t/step-1: price_proposal_legs helper (ES→SPX→OPRA→real mid via options cache)`

New helper (DB I/O via the cache layer, like `build_condor_pricing_payload`), location: extend `packages/shared/options_cache/pricing.py` or a sibling `proposal_pricing.py`.

```
def price_proposal_legs(
    legs,                  # list of {side, type ('call'/'put'), strike (ES), qty}
    *, trade_date, expiration_date, entry_minute_pt, eval_minute_pt | None,
    r, q,                  # per the leg expiration, from orats_monies_minute / smile row
) -> dict:
    # for each leg:
    #   spx_strike = compute_spx_strike(leg.strike_es, dte, r, q)
    #   opra = format_opra("SPX", expiration_date, "C"|"P", spx_strike)
    # fetch_option_bars(all_opras, entry_minute, entry_minute)   # one batched call
    # read mids from get_bars_for_contract per leg
    # net_debit = Σ sign(side)·qty·mid    (long +, short −) → positive = debit
    # returns { legs: [{role, spx_strike, opra, bid, ask, mid, ...}], net_debit, warnings }
```

- Reuse `fetch_option_bars` (batched list), the `_bar_to_quote` mid logic, and `OratsPermanentError`/`OratsError` handling from the condor path.
- Naive-PT datetimes only.
- Per-leg failure semantics per Step 0 #6.
- Self-contained, reusable unit: pure with respect to `(date, anchor, legs)` inputs, no UI assumptions, returns a plain dict.

**Deliverable:** helper importable + unit-tested with a mocked cache (cache-hit path, fetch-on-miss path, partial-404 path). Snapshot test: a known cached anchor's legs price to the expected mids.

## Step 2 — Real implied distribution chain

**Commit:** `cr-t/step-2: build real-mid strike band for Breeden-Litzenberger implied PDF`

- New helper to assemble the implied-distribution input from real mids: derive SPX strike band (Step 0 #3 width/density) around the thesis range, build call OPRAs across the band, `fetch_option_bars` the whole band in one batched call, read mids, produce `[{"strike", "call_price": mid}]` for `compute_implied_pdf`.
- Feed the band to the **unchanged** `compute_implied_pdf` / `compute_implied_prob_in_range`.

**Deliverable:** `implied_prob` for a real anchor computed from real skewed mids; unit test that a known skewed band produces a non-lognormal PDF (downside mass > symmetric lognormal).

## Step 3 — Wire real pricing into the proposals route

**Commit:** `cr-t/step-3: route /api/proposals/pl-data through real-mid pricing`

- Replace step 6 `compute_initial_cost`-via-BSM with `price_proposal_legs` (real entry debit).
- Replace step 9 `build_bsm_chain` with the Step 2 real-mid band feeding `compute_implied_pdf`.
- `r`/`q` per leg from the existing `_fetch_smile_row`; reuse per-expiration batching from CR-G Step 8.
- Thread warnings into the response `warnings` array.
- Preserve response schema (`pl_curve`, `iv_curve`, `trade_thesis`, `edge_zones`, `key_levels`, `legs`).

**Deliverable:** `POST /api/proposals/pl-data` returns real entry debit, real skewed `implied_prob`, meaningful `edge_ratio`; route tests updated.

## Step 4 — BSM demoted to curve shape; horizon + expiry curves

**Commit:** `cr-t/step-4: BSM curve-shape only; render expiry + t1/t5/t15 horizon P/L`

- `compute_pl_curve` keeps BSM but anchored to the real entry debit from Step 1.
- Render at-expiry curve AND horizon curves at t1/t5/t15 (BSM at intermediate `evaluation_time`s, calibrated to entry mid).
- Frontend: populate multiple curves via the existing `pl_curves` list contract from CR-G.

**Deliverable:** chart shows correct expiry payoff plus selectable horizon curves; t-selector drives the evaluation horizon.

## Step 5 — Frontend surfacing

**Commit:** `cr-t/step-5: surface real prices, per-leg mids, and data-quality warnings`

- `LegTable`: show real entry mid per leg (and optionally bid/ask spread) alongside existing ES Level / SPX Strike columns.
- Surface `warnings` as a small data-quality badge.
- No new chart component — reuse `ProposalEdgeChart`.

**Deliverable:** proposal card shows real per-leg pricing and flags degraded data; matches existing prop shape (grep #7).

## Step 5-A — Day-load prefetch wiring (frontend)

**Commit:** `cr-t/step-5a: prefetch pricing for all proposals on day-load (non-blocking, band-once)`

- On day-load, fire pricing for ALL proposals in the background; day renders immediately.
- Fetch the implied-distribution strike band **once per day** (shared `spot ± 1.5×IM`).
- Respect the existing read-through cache: re-loading an already-seen day fires zero ORATS calls.
- Expand-on-view reads already-fetched data; show loading state if prefetch still in flight.
- Do NOT block day render on pricing.

**Deliverable:** day-load populates all proposal pricing without blocking render; re-loading a cached day fires no ORATS calls; implied band fetched once per day.

## Step 6 — Edge threshold recalibration note (no code)

Real skewed implied will shift every `edge_ratio`. Capture a note (in status updates + a follow-up open-question) to re-tune the 1.3/2.0 zone thresholds against real values after a few days of observation.

---

# Order-entry surfacing (Steps 7–11 — added 2026-05-29, pre-merge)

> Steps 0-A–5-A shipped the real-pricing substrate but the card was not yet order-ready: greeks ≈0 (computed at expiry T=0), edge ratio/implied prob not rendered as text, no net debit/credit shown, no calendar expiry date, expanded chart overhanging the card. These five steps add surface only — no pricing, edge-engine, or curve changes.

## Scope decision (locked 2026-05-29)

**The P/L chart stays in ES space.** `spot` = `session_open_t0` (ES), `build_grid_bounds` + `drift_target` are ES, legs carry `raw["strike"]` (ES). The chart kink at the ES strike is correct. The apparent "offset" was a reading artifact: eye anchored on the SPX STRIKE column while the chart is ES. Ryan's call: keep chart ES — consistent with the rest of the app. No axis work in this CR.

## Step 7 — Greeks at entry (not expiry)

**Commit:** `cr-t/step-7: compute position greeks at entry time, not expiry`

`routes.py` step 15 calls `compute_position_greeks(legs_with_iv, spot, evaluation_time, market_state)` — `evaluation_time` is expiry (T≈0), so all greeks collapse to ≈0.

- **Greeks track the t1/t5/t15 horizon selector** — compute at the selected horizon (`entry_time + N days`, same horizon as the P/L curve). The selected `timeframe` is already a request param and drives `_TTE_BY_TIMEFRAME` + the P/L curves.
- Pass that horizon datetime (not `evaluation_time`) to `compute_position_greeks`.
- `legs_with_iv` already carry calibrated per-leg IVs (step 6b), so greeks reflect real-mid-anchored vols.

**Pre-step grep:** confirm proposals route is the only consumer of `compute_position_greeks` affected. Stop + surface on contradiction.

**Deliverable:** Δ/Γ/Θ/V/ρ show non-zero, sensible values for a 15-DTE call spread AND change when t1/t5/t15 selector is toggled. Test: non-zero and finite at each horizon, differ across horizons.

## Step 8 — Surface net debit/credit

**Commit:** `cr-t/step-8: echo net debit/credit into response and render on card`

`price_proposal_legs` returns `net_debit` (positive = debit, negative = credit, `None` if any leg mid missing); it becomes `initial_cost` in the route but is NOT echoed as its own response field.

- **Backend:** echo `net_cost` (= `initial_cost` or real `net_debit`) into the response, preserving the sign convention. One field, no new computation.
- **Frontend:** render on the card labeled by sign — "Net debit $X.XX" / "Net credit $X.XX". If `None`, show "—" with the existing data-quality warning, not a misleading 0.

**Deliverable:** card shows net debit/credit, correctly labeled by sign, matching per-leg mids by hand.

## Step 9 — Surface expiry calendar date

**Commit:** `cr-t/step-9: show expiry calendar date on card`

`legs_out[].expiration` is already an ISO date string in the payload; the card shows only "Target DTE: 15d (8-30 bucket)".

- **Frontend only:** show the actual expiration date ("Expiry: 2026-06-13") alongside or in place of the DTE/bucket. Keep DTE context. Calendar date a human punches into a ticket must be visible.
- Mixed-expiry structures: show per-leg expiry; single-expiry (common case) one date line suffices.

**Deliverable:** card shows calendar expiry date(s); user can read strike + expiry + side + qty + net cost without leaving the card.

## Step 10 — Render the edge block

**Commit:** `cr-t/step-10: render structural/implied/edge-ratio on card`

`trade_thesis` with `structural_prob`, `implied_prob`, `edge_ratio` is already in the response; not rendered as text.

- **Frontend only:** render as a small text block ("Struct 15.8% · Implied 11.2% · Edge 1.41×"), color-consistent with edge-zone semantics (green = struct > implied).
- If `implied_prob` is `None` (sparse band / fallback), show structural prob + "implied unavailable" rather than blank or divide-by-zero.
- **Display only** — do NOT recompute or re-threshold. Threshold recalibration is the separate observational follow-up (Step 6).

**Deliverable:** edge ratio + both probabilities visible as text, sign/color-consistent with the chart.

## Step 11 — Card fits chart width

**Commit:** `cr-t/step-11: proposal card grows to fit expanded edge chart`

- **CSS only:** when the edge chart is expanded, the card container grows to contain the chart (make the card fit the chart, not the reverse).
- Verify both collapsed (no chart) and expanded states render cleanly, and the two-card row (Directional / Debit side by side) doesn't break when one card is expanded.

**Deliverable:** expanded card has no floating chart overhang; collapsed and expanded states both look intentional.

---

## Smoke tests

1. **Zero-debit fix:** OTM debit spread → `initial_cost > 0`, `min(pnl) < 0`, one breakeven at long-strike + debit.
2. **ES→SPX→OPRA round-trip:** a known proposal leg converts to the expected SPX strike and OPRA; cache read returns the row.
3. **Real entry pricing:** for a cached anchor, leg mids match `(bid+ask)/2` from `orats_options_minute` by hand.
4. **Cache fetch-on-miss:** pricing an uncached anchor triggers `fetch_option_bars`, writes rows, second call is a pure cache read.
5. **Skewed implied PDF:** real band produces downside-skewed density vs the old flat-vol lognormal.
6. **Partial failure:** one leg's OPRA 404s → warning, leg renders `—`, net debit/edge computed from available legs or flagged.
7. **Horizon vs expiry:** expiry curve has sharp kinks at strikes; t5/t15 curves are smooth and pass near the real entry debit at spot.
8. **No regression:** structural side (`structural_prob`, KNN) unchanged vs CR-G for the same anchor.
9. **Day-load prefetch:** loading a day fires background pricing for all proposals without blocking render; cards fill progressively; re-loading the same day fires zero ORATS calls; the implied band is fetched once for the day, not per proposal.
10. **Greeks non-zero + horizon-tracking (Step 7):** the magnet-above debit spread shows non-zero, sensible Δ/Γ/Θ/V/ρ; values change across t1/t5/t15 (theta/vega shift with horizon). Not all ≈0, not static across the selector.
11. **Net cost correct + signed (Step 8):** card net debit/credit equals hand-summed per-leg mids with the long/short sign convention; credit structure labels "credit," debit labels "debit."
12. **Expiry date visible (Step 9):** card shows the calendar expiry matching `legs_out[].expiration`; mixed-expiry shows per-leg dates.
13. **Edge block visible (Step 10):** struct/implied/edge-ratio render as text, color-consistent with chart zones; `implied_prob = None` degrades gracefully (no NaN/∞, no blank).
14. **Card fits chart (Step 11):** expanding the chart grows the card to fit; no overhang; side-by-side row intact.
15. **No pricing/chart regression:** curve, edge zones, per-leg mids, and SPX strikes unchanged from Steps 0-A–5-A; chart stays ES.

## Wrap criteria

- Steps 0-A through 5-A **and Steps 7–11** committed on the branch; Step 0-B coverage table + Step 6 recalibration note in status updates.
- All smoke tests pass (1–9 pricing/substrate; 10–15 order-entry surfacing).
- A real anchor shows: real entry debit, real skewed `implied_prob`, meaningful `edge_ratio`, correct loss region, both horizon and expiry curves.
- The card reads as **order-ready**: SPX strikes, calendar expiry, net debit/credit, non-zero greeks, and the edge block all visible; chart unchanged (ES).
- Roadmap updated: CR-T complete; CR-F / CR-H confirmed unblocked to build on real numbers.
- ADR (2026-05-29 - Proposal Pricing Sources from Real ORATS Mids) status moved open → closed.
- Follow-up open-question filed for edge-threshold recalibration.

## Sequencing

Land BEFORE CR-F (debit/credit variants), CR-H (interactive editing), and the candidate position-aware-EV / multi-DTE CRs.

## Future (explicitly NOT in this CR)

Step 1's `price_proposal_legs` is built as the reusable, UI-free unit a future headless historical backfill would loop over — but the backfill itself is NOT built in this CR.

## Known frictions

- Filesystem MCP has timed out at session start before; retry or fall back.
- `DATABASE_URL` uses `postgresql+psycopg://`; strip with `psql "${DATABASE_URL//+psycopg/}"` for one-off queries.

## Related files

- **Read:** `packages/shared/options_cache/pricing.py`, `fetcher.py`, `opra.py`, `repository.py`, `models.py`; `packages/shared/forward_math.py`; `packages/shared/implied_distribution.py`; `packages/shared/pricing/engine.py`; `packages/shared/strategy_templates.py`; `apps/web/modules/Proposals/routes.py`, `service.py`
- **Write:** `packages/shared/options_cache/pricing.py` (or new `proposal_pricing.py`); `apps/web/modules/Proposals/routes.py`, `service.py`; `react_today_setup/.../LegTable.jsx`, proposal card; tests under `packages/shared/options_cache/tests/` and `apps/web/modules/Proposals/tests/`

---

## Step 0 — Diagnosis findings (2026-05-29)

### Pre-implementation grep results (all CLEAN — no contradictions)

1. **`compute_initial_cost`** — only in `Proposals/routes.py` (import + calls at lines 365, 489) and `Proposals/service.py` (definition). No other consumers. Safe to demote to entry-T BSM then replace with real mids.

2. **`build_bsm_chain`** — only in `Proposals/routes.py` (import + call at line 388) and `Proposals/service.py` (definition). No other consumers. Safe to replace for implied distribution.

3. **`compute_pl_curve`** — only in `Proposals/routes.py` (import + call at line 374) and `Proposals/service.py` (definition). `edge_zones.py:231` is a comment, not a caller. Safe.

4. **`orats_options_minute` uniqueness** — confirmed: `ON CONFLICT (opra_symbol, snapshot_pt) DO NOTHING` in `repository.py:156`. Correct unique key.

5. **`expiry_tod`** — present in `OptionMinuteBar` as `expiry_tod: Optional[str] = None` (models.py:140). Nullable, populated from ORATS data.

6. **SPX root** — `opra.py` docstring explicitly: "ORATS uses root `SPX` for all SPX expirations, including PM-settled weeklies (colloquially called SPXW). The `expiryTod` column distinguishes AM vs PM settlement." Condor path already uses `"SPX"` root. No ambiguity — use `"SPX"` for all.

7. **Frontend prop shapes** — `LegTable.jsx` reads `leg.type`, `leg.quantity`, `leg.strike`, `leg.strike_spx`, `leg.side` from **`proposal.legs`** (the generator output), not from the pl-data response. The pl-data response uses `leg.flag` and `leg.qty`. These are two distinct schemas — `LegTable` is always showing proposal generator legs, not pl-data legs. For Step 5, pricing will be shown via `chartData.legs` in the `ExpandedPanel` (which has `initial_value`), not by modifying the header `LegTable`.

**Critical spec assumption VERIFIED:** `Leg.strike` IS in ES discounted-forward space — confirmed at `routes.py:477–484` where `compute_spx_strike(leg["strike"], ...)` is called to convert for the response display. Spec assertion holds.

**Connection path verified:** `pricing.py` / `repository.py` use their own SQLAlchemy engine (reads `DATABASE_URL` from env). This is independent of the `psycopg.connect()` used in the proposals route. Both read the same `DATABASE_URL`. The condor panel writes via this path in production, so the web app's role can write `orats_options_minute` via this path. No role/connection mismatch.

### Step 0 design-lock decisions

1. **SPX root / AM-PM** — LOCKED: Use `"SPX"` root for all SPX option OPRAs. `expiry_tod` field in cache distinguishes AM/PM but OPRA format is always `SPX`. For 8-30 DTE proposals (standard weeklies/monthlies), all PM-settled SPX. No stop condition.

2. **Entry time** — LOCKED: `07:00 PT` (10:00 ET) on `trade_date`. Rationale: 30 minutes after open, market settled, ORATS has intraday data from open. Implementation: `build_entry_time(trade_date)` in `service.py` returning tz-aware UTC. For `fetch_option_bars` calls (which need naive PT), convert with `entry_time_utc.astimezone(PT).replace(tzinfo=None)` or use the naive PT `datetime(y, m, d, 7, 0)` directly.

3. **Strike band** — LOCKED: `spot ± 1.5 × implied_move` at 5pt spacing. Uses same bounds as `build_grid_bounds` default (`half_sigma=1.5`). Strikes generated as `round((spot + i * 5) / 5) * 5` in the band range. For a typical SPX spot=5500, IM=50 → 5425 to 5575 → 31 strikes per call band.

4. **BSM calibration** — LOCKED: Per-leg scipy `brentq` to find σ such that `BSM(spot, strike, T_entry, r, σ, flag) = real_mid`. Fallback to `atmiv` if calibration fails (tiny price, non-convergence). This σ feeds `compute_pl_curve` and horizon curves. Horizon curves are smooth BSM surfaces calibrated to entry reality.

5. **Historical scope** — LOCKED: include historical fetch-on-demand. `fetch_option_bars` uses ORATS `/datav2/hist/live/one-minute/strikes/option` (same path as condor panel). Step 0-B coverage table will confirm anchor coverage before declaring OK.

6. **Per-leg failure** — LOCKED: Mirror condor path exactly. `OratsPermanentError` → add warning, leg `mid = None`, excluded from `net_debit` (net_debit=None if any leg missing). Partial fills OK. No hard failure on 404.

7. **Fetch trigger** — ALREADY LOCKED per spec. Per-card prefetch on mount (non-blocking). Band once per day is naturally handled by the existing cache — first proposal for a day populates the band, subsequent proposals for the same day are cache reads. No explicit per-day coordination needed in frontend.

### Pre-existing observation (not a contradiction; pre-dates this CR)

`LegTable.jsx` renders `leg.type` and `leg.quantity` (from `proposal.legs` shape) while the `/api/proposals/pl-data` response uses `leg.flag` and `leg.qty`. These are two separate data schemas; the LegTable is not consuming pl-data response legs and this is intentional. No action needed in this CR.
