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

## Wrap criteria

- Steps 0-A through 5-A committed on the branch; Step 0-B coverage table + Step 6 recalibration note in status updates.
- All smoke tests pass.
- A real anchor shows: real entry debit, real skewed `implied_prob`, meaningful `edge_ratio`, correct loss region, both horizon and expiry curves.
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
