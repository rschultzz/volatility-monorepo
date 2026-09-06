# CR-AO — Listed-strike snapping; live quote sanity; holdout re-capture

> Authority: vault session note `Dash/sessions/2026-09-05 - CR-AO — Listed-Strike Snapping + Quote Sanity in Live Pricing.md`
> Branch: `feat/CR-AO-listed-strike-snapping` (off `origin/main` 5de7262, the CR-AN merge)
> Scope: one shared `snap_to_listed_strike` + `StrikeNotListed`, wired into the harness leg builder (width_actual / width_nominal), the capture script, CR-AI Stage 2 and the live Proposals leg builder; CR-AN's quote-sanity rule in the live pricing module and the CR-AI SQL-mid path; holdout re-capture of CR-AM's 8 404 dates. **No deploy.**
> Mode: unattended through PR; halts at the first STOP gate that misses.

## Problem

`[Certain]` CR-AN classified CR-AM's 8 holdout 404s: the target strike (always ending in 5) at a ~3-week non-monthly SPX expiry was not listed on the signal date; the same contract is served on later dates and at expiry. Every served target was a multiple of 10. The harness's `round5` picks strikes that don't exist yet.

`[Likely]` The live Proposals module uses the same target → strike logic, so proposal cards can name a strike that cannot be traded that day. Unverified until Step 0.

`[Certain]` CR-AN patched quote sanity in the shared harness path only. Two duplicates remain: CR-AI Stage 2's SQL mids, and the live pricing module in `apps/web`. The live cards can therefore show a spread price outside `[0, width]` at the open.

## Unknown to resolve from data, not assumption

SPX's listing schedule for non-monthly expiries is not known to us. `orats_oi_gamma` holds the full EOD chain per (trade_date, expiry, strike), so the listed strike set for any expiry on any prior close is in the DB. Step 0 derives the schedule empirically and, more importantly, the fix does not need the schedule at all: **snap to the nearest strike present in the prior-close chain for that expiry.**

## Locked decisions

| # | Decision | Value |
|---|---|---|
| 1 | Snapping rule | `snap_to_listed_strike(target, expiry, trade_date, conn)`: candidates = `DISTINCT strike FROM orats_oi_gamma WHERE ticker='SPX' AND trade_date = <prior close> AND expir_date = expiry`; return the candidate nearest `target` (ties → the one on the side of spot, i.e. toward the magnet direction). If the expiry is absent from the prior-close chain → raise `StrikeNotListed` (caller decides; the harness marks the trade `unlistable`, the live path shows "no listed strike" instead of a card). |
| 2 | Where it lives | One function in `packages/shared/options_cache/` (or `backtest/`, wherever `round5` currently lives), called by: the CR-AH harness leg builder, the CR-AM/CR-AN capture path, `cr_ai_stage2_backfill.py`, and the live Proposals leg builder. No duplicate implementations. |
| 3 | Spread width after snapping | Width is the distance between the two snapped legs, not a constant 10. If snapping widens a debit to 15 or narrows it to 5, the harness records `width_actual` and uses it for the `[0, width]` sanity bound and payoff. The 10-point *intent* is kept as `width_nominal`. |
| 4 | Quote sanity, live | Apply CR-AN's decision-2 rule (both legs `bid ≥ 0`, `ask > 0`, `bid ≤ ask`, spread price in `[0, width_actual]`) in the live pricing module; on failure the card shows the last valid price with a "stale quote" flag, never an out-of-range value. |
| 5 | Quote sanity, CR-AI | Patch the SQL-mid path to the shared validator. Do **not** re-run CR-AI here; list "CR-AI re-run with clean quotes + snapping" as a follow-on with the Step 0 contamination rate as its justification. |
| 6 | Historical train impact | Step 0 measures: of the CR-AN clean sample (debit 110 / credit 106), how many trades used a strike absent from the prior-close chain on the signal date, and for those, whether entry-day quotes nevertheless exist in `orats_options_minute`. **If > 10% of either structure traded unlisted strikes → the CR-AN reference needs a re-run with snapping (CR-AP). Do not re-run here; record the number and STOP the analysis question, continue the CR.** |
| 7 | Holdout re-capture | Re-run the CR-AM capture step for the 8 404'd dates with snapping. Capture only, no read, `cr_id='CR-AO-capture'`. Log dates / snapped legs / any remaining 404. |
| 8 | Deploy | Web service `autoDeploy: off`. After merge, the deploy is a separate attended step from chat via Render `trigger_deploy` on the web service, off-hours, with the previous deploy identified for rollback. This deploy also carries every `main` change since 2026-08-23 (CR-AH/AI/AK/AL/AM/AN, including the `compute_structural_probability` raise) — the pre-deploy check is that all routes passing `exclude_date` also pass `before_date` (CR-AL did this; re-verify by grep). |
| 9 | Not in scope | Any change to the edge threshold or direction gate; the entry-convention mismatch; CR-AI / CR-AN re-runs. |

## Expected values (gates)

| Gate | Expected | On miss |
|---|---|---|
| G0.1 — `main` contains the CR-AN merge | yes | STOP |
| G0.2 — empirical listing schedule: for non-monthly SPX expiries in 2026, fraction of 5-ending strikes present in the chain at 15 / 10 / 5 / 2 DTE | a step function (low at 15, near 1 at ≤ 5) | note; whatever it is |
| G0.3 — live Proposals leg builder located; uses `round5` or equivalent | yes/no recorded | STOP only if the module can't be located |
| G0.4 — decision-6 count | recorded | if > 10% → note "CR-AP required", continue |
| G1 — tests: snapping (nearest, tie toward magnet, unlisted expiry raises); live validator; width_actual propagation | pass; suites still pass | STOP |
| G2 — `git diff --stat` touches only: the new shared function + tests, harness leg builder, capture script, `cr_ai_stage2_backfill.py`, live pricing module, Proposals leg builder | yes | STOP if anything else |
| G3 — re-capture: 8 dates attempted, 404s remaining | 0 (or ≤ 2 with a stated reason) | note |
| G4 — no holdout P&L in logs | yes | STOP and redact |


## Step 0 findings (read-only, 2026-09-05 ~12:20 PT)

Interpreter: `apps/web/.venv/bin/python` for DB reads and the capture; Rosetta repo venv for the suites.

| Gate | Expected | Actual | Result |
|---|---|---|---|
| G0.1 `main` contains the CR-AN merge | yes | `origin/main` = 5de7262 (PR #46 merge); branch cut from it | PASS |
| G0.2 empirical listing schedule | step function | **yes** — non-monthly 5-ending completeness 0.20 at 15 bdays, 0.90 at 10, 1.00 at ≤ 7; monthly 1.00 throughout (table below) | note |
| G0.3 live Proposals leg builder located | yes/no | **yes** — `packages/shared/forward_math.py::compute_spx_strike` (`round(spx_raw / 5) * 5`) inside `options_cache/pricing.py::price_proposal_legs`; full inventory below | PASS |
| G0.4 decision-6 count | recorded | **debit 9 / 110 = 8.2 %, credit 12 / 106 = 11.3 %** — credit exceeds 10 % → **CR-AP required** (recorded; run continues) | note |

### G0.2 — SPX listing schedule from `orats_oi_gamma` (EOD chain, 2023-01-02 → 2026-09-04, 922 sessions)

Expiries 2026-01-02 → 2026-09-04: 163 non-monthly, 7 monthly (third Friday). For each expiry and dte ∈ {15, 10, 7, 5, 2} business days (the chain's own trade-date calendar), strikes within ±100 of that day's `stock_price`. "Share" = strikes ending in 5 ÷ all listed strikes in the window (0.50 = full 5-point grid); "completeness" = 5-ending strikes present ÷ 20 possible.

| dte (bdays) | non-monthly: median share | non-monthly: median completeness | non-monthly expiries with no 5-ending strike | monthly: median share | monthly: completeness |
|---|---|---|---|---|---|
| 15 | 0.17 | **0.20** | 0 / 163 | 0.50 | 1.00 |
| 10 | 0.47 | **0.90** | 0 / 163 | 0.50 | 1.00 |
| 7 | 0.50 | 1.00 | 0 / 163 | 0.50 | 1.00 |
| 5 | 0.50 | 1.00 | 0 / 163 | 0.50 | 1.00 |
| 2 | 0.50 | 1.00 | 0 / 163 | 0.50 | 1.00 |

At 15 business days a non-monthly expiry carries the 10-point grid plus a handful of 5-point strikes right at the money (~4 of 20); the 5-point grid fills in between 10 and 7 business days out. Monthlies are fully listed at 15. The chain tracks the CR-AM 404s to the day: SPX 7655 C exp 2026-08-03 is absent through 07-22 and present from 07-23 (the ORATS 404 window was ≤ 07-14 probed); SPX 7805 C exp 2026-09-02 absent through 08-13, present from 08-14 (ORATS: 404 on 08-12, 200 on 08-14). Zero-OI strikes are present in the chain (e.g. 3400 C on 07-10 with call_oi = put_oi = 0), so it is the listed set, not the open-interest set.

### G0.3 — target → strike logic inventory

| File : function | Rounding | Used by | CR-AO action |
|---|---|---|---|
| `scripts/cr_ah_step4_analysis.py : round5` (l.176) → `filter_clean_for_structure` (short = `round5(drift_target)`, other = short ± 10) → `build_legs` | nearest 5 | the CR-AH/CR-AM/CR-AN harness run path ("harness leg builder") | wire `snap_vertical_legs`; `width_nominal` / `width_actual` on `TradeData`; `unlistable` exclusion with reason |
| `scripts/cr_am_holdout_leg_capture.py : round5` (l.153) | nearest 5, ±10 | holdout capture | wire; `--dates` for the re-capture |
| `scripts/cr_ai_stage2_backfill.py : round5` (l.127) → `load_clean_dates` (`ss = round5(drift_target)`, `ls = ss − 10`; `SPREAD_WIDTH = 10.0` constant) | nearest 5 | CR-AI Stage 2 | wire snapping + `width_actual` on the entry; SQL mids → validator (decision 5); not re-run |
| `scripts/cr_ah_step2_stratified_backfill.py : round_to_5pt` (l.280) | nearest 5 | June's one-off backfill (superseded by the capture script) | listed, not patched — historical script |
| `packages/shared/forward_math.py : compute_spx_strike` (ES-forward → SPX, `round(spx_raw / 5) * 5`) | nearest 5 | **live**: `options_cache/pricing.py::price_proposal_legs` (l.288, the Proposals leg builder: legs from `strategy_templates.generate_proposals` with raw wall prices ± 10) and `fetch_horizon_delta` (l.488); `apps/web/modules/Proposals/routes.py` l.627 (falls back to it for `strike_spx` display); `apps/web/modules/TodaySetup/service.py` l.140 (`strike_spx` display on `/today-setup` cards, no DB in that module) | snap inside `price_proposal_legs` after the ES→SPX conversion (keeps `compute_spx_strike` pure); Proposals response carries `listed` / `stale_quote`; TodaySetup display-only `strike_spx` listed as follow-on (outside G2's file list) |
| `packages/shared/options_cache/pricing.py : build_condor_pricing_payload` (`condor_strikes_from_smile`, `strike_increment=5`) | 5-point grid | live condor pricing | quote validity applied to its legs (decision 4); strike snapping not applied (0-DTE condor; grid is complete at 0 DTE per G0.2) |

Shared function: `packages/shared/options_cache/strikes.py` — `prior_close`, `listed_strikes`, `snap_to_candidates` (pure), `snap_to_listed_strike(target, expiry, trade_date, conn)`, `snap_vertical_legs(...)` returning `width_actual` / `width_nominal`, and `StrikeNotListed`. Works with a psycopg connection or a SQLAlchemy connection (`exec_driver_sql`), so the live path and the scripts call the same code.

### G0.4 — decision 6 on the CR-AN sample (scratchpad `cr_ao_g04.py`, `cr_ao_g04.json`)

Legs regenerated exactly as CR-AN built them (`round5(drift_target)`, ± 10, expiry = 15 business days); "listed" = present in `orats_oi_gamma` for that expiry at the prior close before the signal date.

| structure | n | trades with ≥ 1 unlisted leg | % | short only / other only / both | expiry absent from chain | unlisted legs with entry-day rows in `orats_options_minute` | unlisted legs ending in 5 | by band near/mid/far | by year 23/24/25/26 |
|---|---|---|---|---|---|---|---|---|---|
| debit | 110 | **9** | **8.2 %** | 3 / 3 / 3 | 0 | 9 / 9 | 6 / 9 | 5 / 3 / 1 | 2 / 5 / 2 / 0 |
| credit | 106 | **12** | **11.3 %** | 0 / 8 / 4 | 0 | 12 / 12 | 6 / 12 | 6 / 5 / 1 | 1 / 6 / 4 / 1 |

Every unlisted leg nevertheless has entry-day minute rows in `orats_options_minute` — the contract was served once ORATS had it (listed intraday, or served from a later listing), which is exactly the "priced a leg with data from a later listing" risk the note anticipated. Half the unlisted legs are multiples of 10 (the 10-point grid is itself incomplete beyond ±100 of spot at 15 bdays). **Credit exceeds the 10 % line → the CR-AN reference needs a re-run with snapping (CR-AP). Recorded; not run here.**

### Pre-deploy check (decision 8)

CR-AL's route fix: `apps/web/modules/TodaySetup/routes.py` l.287 passes `before_date=trade_date.isoformat()` alongside `exclude_date`. Grep of every tracked `compute_structural_probability(` call site: `TodaySetup/routes.py` (before_date ✓), `scripts/cr_ah_step4_analysis.py` (mode → before_date or allow_lookahead ✓), `scripts/cr_z_step4b_k_display.py` (before_date ✓). The only call with `exclude_date` and neither kwarg is `scripts/cr_ah_posthoc_limit_entry.py`, which is untracked and not deployed. **Expect none → none.** Note: `render.yaml` still says `autoDeploy: true` for both services it lists; the vault records the web service as dashboard-managed with autoDeploy off and `render.yaml` as stale — nothing here deploys either way.
