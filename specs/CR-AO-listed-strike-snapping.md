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

