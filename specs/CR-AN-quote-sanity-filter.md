# CR-AN — Quote-sanity filter; clean-quote walk-forward reference; ORATS 404 classification

> Authority: vault session note `Dash/sessions/2026-09-05 - CR-AN — Quote-Sanity Filter in the Edge Harness.md`
> Branch: `feat/CR-AN-quote-sanity-filter` (off `origin/main` 1e3bd76, the CR-AM merge)
> Scope: minute-level quote validity in the edge harness's quote → spread-price path + tests; one re-run persisted as `cr_id='CR-AN'`; ORATS 404 classification (no fix).
> Mode: unattended; halts at the first STOP gate that misses.

## Problem

`[Certain]` CR-AM found that roughly a quarter of trades in both structures carry a spread price outside `[0, width]` at some minute, mostly at the first-minute baseline. A 10-point vertical cannot be worth less than 0 or more than 10; those minutes are bad quotes (crossed or stale legs, one-sided books at the open). The harness accepts them, so:

- every "beat vs baseline" in June and in CR-AM compares a gated fill to a baseline that may be a bad quote;
- the `bt_edge_backtest_results` rows persisted as `cr_id='CR-AM'` are contaminated the same way;
- CR-AM's clean-subset sensitivity table (in the spec) is the only uncontaminated read, and it was produced by *excluding trades*, which is the wrong fix — a bad first minute shouldn't erase a trade that had good quotes a minute later.

Two of CR-AM's four apparent reversals (credit far-band, debit gap-touch) were this contamination.

## Locked decisions

| # | Decision | Value |
|---|---|---|
| 1 | Where the filter lives | In the harness's quote → spread-price step, so every caller inherits it. Step 0 locates the function (in `scripts/cr_ah_step4_analysis.py` and/or `packages/shared/backtest/`). If the logic is duplicated across CR-AH and CR-AI scripts, CR-AN fixes the shared path and lists the duplicates as a follow-on; it does not fix two copies. |
| 2 | Minute-level rule | A minute is **valid** for a spread iff, for both legs, `bid ≥ 0`, `ask > 0`, `bid ≤ ask`, and the resulting spread price (mid-based, as the harness computes it today) is in `[0, width]` inclusive, with `width` from the structure. Invalid minutes are skipped, not clamped. |
| 3 | Baseline | First **valid** minute in the entry window, not the first minute. |
| 4 | Gated fill | Threshold crawl considers only valid minutes. |
| 5 | Settlement | Settlement price from the settlement window's valid minutes only; if none are valid, the trade is `unsettled` (existing status), not excluded. |
| 6 | Trade-level exclusion | Only if the entry window has **zero** valid minutes. Count and list these; expect few. |
| 7 | Observability | The harness records per trade: `n_minutes_total`, `n_minutes_valid`, `baseline_minute_offset` (minutes from window open to the first valid minute), and `had_invalid_quote` (bool). The run's smoke dict reports the distribution. |
| 8 | Re-persist | Re-run CR-AM's exact configuration with the filter (`--universe-end 2026-06-05 --split-date 2026-06-05 --structural-prob-mode walk-forward`) and persist with `cr_id='CR-AN'`. CR-AM and CR-AH rows remain as labeled comparison columns. Readers should prefer the latest `cr_id`; note for CR-Y. |
| 9 | Restatement | Same decision-7 rule as CR-AM (holds / weakens / reverses vs June, ≥ 50% magnitude) applied to CR-AN vs **CR-AM as-run** and vs **CR-AM clean-subset**. The clean-subset comparison is the one that should be near-identical; a large gap means decision 2 is doing something the trade-exclusion didn't, and the spec must explain it. |
| 10 | ORATS 404 diagnostic | Step 0b: pick one of CR-AM's 8 entry-day 404 dates and one fetched date; call the ORATS endpoint directly for each (same params the fetcher uses) and record status, and for the 404 also try the same date with a nearby strike and with `dte` widened. Classify: (i) date not served at all → retention/rolloff; (ii) that strike/expiry not served → strike-selection issue; (iii) transient. **No fix in this CR** — classification only, feeds [[debit-edge-needs-powered-holdout]] and CR-AG's reprioritisation. |
| 11 | Not in scope | Changing spread pricing from mid to bid/ask-aware fills; the entry-convention mismatch between CR-AH and CR-AI (open question); CR-AI re-run (only if the shared path is what CR-AI uses **and** Step 0 shows its quotes are contaminated at a similar rate — then list as follow-on, don't run). |

## Expected values (gates)

| Gate | Expected | On miss |
|---|---|---|
| G0.1 — `main` contains the CR-AM merge | yes | STOP |
| G0.2 — filter location identified; single shared code path or explicit list of duplicates | yes | STOP if the spread-price computation can't be located |
| G0.3 — pre-filter contamination count reproduced from CR-AM's log/spec (≈ 25% trades flagged per structure) | within ±5 pp | note |
| G1 — unit tests for decision 2 (valid / crossed / negative / over-width / one-sided) | pass; 420+ & 105 suites still pass | STOP |
| G2 — post-filter: trades with any accepted spread price ∉ `[0, width]` | **0** | STOP |
| G3 — trades excluded under decision 6 | ≤ 5% of clean sample per structure | STOP if > 10% |
| G4 — selection / clean counts | 50/50/50; debit 110 / credit 106 before decision-6 exclusion | STOP |
| G5 — baseline_minute_offset distribution | median 0, p95 ≤ 5 min | note; if p95 > 15 the open is systematically bad and decision 3 needs a window cap — flag, continue |
| G6 — restatement vs CR-AM clean-subset | all cells holds | any reverses → note with explanation; not a halt |
| G7 — no holdout date, no post-2026-06-05 P&L in logs | yes | STOP and redact |


## Step 0 findings (read-only, 2026-09-05 ~11:00 PT)

Interpreter: `apps/web/.venv/bin/python` for DB reads, the ORATS probes and the run; Rosetta repo venv for the test suites.

| Gate | Expected | Actual | Result |
|---|---|---|---|
| G0.1 `main` contains the CR-AM merge | yes | `origin/main` = 1e3bd76 (PR #45 merge); branch cut from it | PASS |
| G0.2 spread-price computation located; single shared path or explicit duplicate list | yes | located — one shared function plus three inline duplicates (table below); insertion point decided | PASS |
| G0.3 pre-filter contamination reproduced from CR-AM | ≈ 25 % per structure, ±5 pp | **debit 26 / 110 = 23.6 %, credit 20 / 106 = 18.9 %** (CR-AM spec Step 1, `range_diag.json`) | PASS (credit 6 pp under the ≈25 % guess; within the note's spirit — it is the same measurement CR-AM made) |

### G0.2 — where a two-leg spread price is computed from `orats_options_minute` rows

| File : function | How the price is built | Validates quotes? | Shared? | CR-AN action |
|---|---|---|---|---|
| `packages/shared/backtest/net_price.py : net_price_from_real_quotes(legs, QuoteMap)` | signed sum of per-leg **mids** from a `QuoteMap {(strike, 'C'/'P') → mid}` | no — returns None only if a leg is absent; no bid/ask, no range | **yes** — called by `VerticalPlugin.net_price`, `DebitVerticalPlugin.net_price`, and directly by the CR-AH Step 4 script | keep; it never sees bid/ask, so validity is applied one step earlier (below) |
| `packages/shared/backtest/harness.py : _run_entry_crawl / _touch_exit / _baseline_outcomes` | consumes `[(snapshot_pt, QuoteMap)]` built by the caller; baseline = first minute with a price (`first_quoted_minute`) | no | yes (generic harness; tests + verification) | **insertion point 1**: skip minutes whose spread value is out of range, baseline = first valid minute, record observability, exclude a trade only when no valid minute exists |
| `scripts/cr_ah_step4_analysis.py : build_entry_scan (≈ l.416) / get_touch_pos_val (≈ l.553)` | inline `mid = (bid + ask) / 2` per row → `QuoteMap` → `net_price_from_real_quotes` | no (NULL rows are simply absent) | no — an inline copy of the harness crawl (the harness is not called by this script) | **insertion point 2**: build the `QuoteMap` through the shared validity helper (bid/ask rule) and apply the same spread-range rule; this is the path Step 1 runs. Settlement here is the ES close (`get_settlement_price` reads `ironbeam_es_1m_bars`), so decision 5 (settlement-window valid minutes) has no option-quote settlement to apply to in this script — recorded, not invented |
| `scripts/cr_ai_stage2_backfill.py : get_net_credit_at / get_settlement_cost` | SQL `(bid_price + ask_price) / 2.0 AS mid`, `WHERE bid_price IS NOT NULL AND ask_price IS NOT NULL`; first / last quoted minute | NULL-check only | no — own SQL path | **duplicate, listed as follow-on** (decision 1: fix the shared path, don't fix two copies). CR-AI's entries are open-price / first-quoted-minute — the same minute class that carries most of the contamination here, so a CR-AI re-run on the filtered path is a candidate follow-on, not run in this CR (decision 11) |
| `packages/shared/options_cache/pricing.py : _bar_to_quote / _net_credit (condor payload), price_proposal_legs (≈ l.327), build_real_strike_band (≈ l.422)` | `mid = (bid + ask) / 2` from `OptionMinuteBar`; strike band keeps `mid > 0` only | NULL-check; no crossed-book or range check | live path (`apps/web/modules/Proposals/routes.py`, `Ironbeam/callbacks.py` import it); `apps/web` has no direct bid/ask arithmetic | **follow-on** (note's open question: a proposal card can show a bad quote at the open) |

Shared helper to add: `packages/shared/backtest/quote_validity.py` — leg-quote rule (decision 2: `bid ≥ 0`, `ask > 0`, `bid ≤ ask`, both present), `QuoteMap` builder that drops invalid legs, and the spread-range rule (`0 ≤ signed value in the structure's natural direction ≤ width`, derived from the legs for verticals; magnitude-only for multi-leg). Wired into the harness (insertion point 1) and the Step 4 script's inline crawl (insertion point 2). The two duplicates are listed, not patched.

### G0.3 — pre-filter contamination (CR-AM, T = 0.00, walk-forward)

From CR-AM's `range_diag.json`: debit 26 / 110 trades with ≥ 1 out-of-range value (baseline > width 22, fill sign 8, touch-exit > width 8, close > width 5); credit 20 / 106 (baseline > width 20, fill sign 1, close > width 1). The baseline (first quoted minute of RTH) is the dominant carrier in both structures.

### 0b — ORATS 404 diagnostic (decision 10; direct calls through `http_client.get_csv` with the fetcher's exact params; no writes)

Endpoint `/datav2/hist/live/one-minute/strikes/option`, `ticker` = side-stripped OPRA, `tradeDate` = ET `YYYYMMDDHHMM[,YYYYMMDDHHMM]` (06:30–13:00 PT → `…0930,…1600`).

| Probe | ticker | tradeDate | Result |
|---|---|---|---|
| A. CR-AM 404 date 2026-07-13, its 15-DTE target leg (7655 C, exp 08-03) | `SPX26080307655000` | `202607130930,202607131600` | **404** `Not Found.` |
| B. control: 2026-07-15, its leg (7650 C, exp 08-05) | `SPX26080507650000` | `202607150930,202607151600` | 200, 391 rows |
| C1. same contract, 07-14 | `SPX26080307655000` | `202607140930,…1600` | **404** |
| C2. same contract, 07-10 | `SPX26080307655000` | `202607100930,…1600` | **404** |
| C3. same contract on its expiry day 08-03 (settlement window) | `SPX26080307655000` | `202608031550,202608031600` | 200, 11 rows (as CR-AM captured) |
| D1. 07-13, same expiry, strike 7600 | `SPX26080307600000` | `202607130930,…1600` | 200, 391 rows |
| D2. 07-13, same expiry, strike 7700 | `SPX26080307700000` | `202607130930,…1600` | 200, 391 rows |
| E1. 07-13, 7650 C exp 07-17 (monthly) | `SPX26071707650000` | same | 200, 782 rows |
| E2. 07-13, 7650 C exp 07-31 | `SPX26073107650000` | same | 200, 391 rows |
| E3. 07-13, 7650 C exp 08-21 (monthly) | `SPX26082107650000` | same | 200, 782 rows |
| E4. 07-13, 7650 C exp 08-07 | `SPX26080707650000` | same | 200, 391 rows |
| F. 07-13, single minute 06:33 PT | `SPX26080307655000` | `202607130933` | **404** |
| G1. CR-AM 404 date 2026-08-12, its leg (7805 C, exp 09-02) | `SPX26090207805000` | `202608120930,…1600` | **404** |
| G2. same contract, 08-14 | `SPX26090207805000` | `202608140930,…1600` | 200, 391 rows |
| G3. 08-12, 7800 C exp 08-21 | `SPX26082107800000` | `202608120930,…1600` | 200, 782 rows |

**Classification: (ii) — that strike/expiry is not served on that date; not a retention/rolloff problem and not transient.** The 404 dates themselves are fully served for other contracts (D, E, G3), the 404 contracts are served on later dates (C3, G2), single-minute and day-range forms fail alike (A, F), and the failure is stable across adjacent days before listing (C1, C2). What is missing is the **5-point strike at a ~3-week weekly expiry**: on 07-13 the 08-03 expiry serves 7600 and 7700 but not 7655; on 08-12 the 09-02 expiry does not serve 7805, which is served from 08-14. Across CR-AM's 21 dates every served entry-day target at a multiple of 10 came back (7590, 7550, 7600, 7650, 7560, 7750, 7810, 7910) while the 404s are all strikes ≡ 5 mod 10 (7595, 7545, 7655, 7805 ×4, 7825) — with two exceptions where the 5-strike was already listed (08-10 → 7825 exp 08-31; 08-27 → 7805 exp 09-18). Strike granularity for non-monthly SPX expiries is added as expiry approaches; `round5(target)` lands on a strike that does not exist yet at 15 business days for roughly a third of dates.

Implication (no fix here): the CR-AH/CR-AM harness's strike selection (`round5`) is the cause, so the fix is a strike-availability fallback in the leg builder (nearest listed strike, or a chain query for the expiry) — a fetcher/harness change, not a capture cron. Feeds [[debit-edge-needs-powered-holdout]] (8 of 21 holdout dates are recoverable by re-selecting the strike, not lost) and de-prioritises the retention argument in CR-AG's reprioritisation. The 09-03 / 09-04 404s are the same class (7805 / 7825 at 15 DTE), not ingestion lag as CR-AM guessed.
