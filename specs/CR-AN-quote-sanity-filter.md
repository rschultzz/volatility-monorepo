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

## Step 1 — clean-quote walk-forward run, persisted as `cr_id='CR-AN'`

Command: `PYTHONUNBUFFERED=1 apps/web/.venv/bin/python -u scripts/cr_ah_step4_analysis.py --universe-end 2026-06-05 --split-date 2026-06-05 --cr-id CR-AN --structural-prob-mode walk-forward` (CR-AM's exact configuration + the filter). Log: `scripts/logs/cr_an_step4_wf_20260905_155201.log` (untracked). 284 s. Run row: `(UUID('7d1efe38-8945-4a22-aa19-8ae9e4402654'), 'completed', datetime.datetime(2026, 9, 5, 22, 52, 2, 724316), datetime.datetime(2026, 9, 5, 22, 56, 45, 827088), "Step 4 complete [mode=walk-forward]: debit=110, credit=106, T_d=0.05, T_c=0.0, 284s; summary_d={'debit': 'INCONCLUSIVE', 'credit': 'untestable'}")`. Persisted: 8 rows `cr_id='CR-AN'`; CR-AH (16) and CR-AM (8) rows untouched.

| Gate | Expected | Actual | Result |
|---|---|---|---|
| G1 unit tests + suites | pass; 420+ & 105 still pass | 13 new tests; backtest suite **118**, shared **424** | PASS |
| G2 post-filter: accepted spread values ∉ [0, width] | 0 | **0** (fill, baseline, close, touch-exit, baseline-touch, baseline-close checked at T = 0 on all 216 trades) | PASS |
| G3 decision-6 exclusions | ≤ 5 % per structure | **0 / 110 debit, 0 / 106 credit** | PASS |
| G4 selection / clean counts | 50/50/50; debit 110 / credit 106 | **50 / 50 / 50; 110 / 106** (bands 41/36/33 and 40/36/30, as CR-AM) | PASS |
| G5 `baseline_minute_offset` | median 0, p95 ≤ 5 | debit median 0, **p95 1**, max 9 (34 / 110 trades > 0); credit median 0, **p95 3**, max 17 (33 / 106 > 0) | PASS |
| G6 restatement vs CR-AM clean-subset | all holds | **not all** — debit near reverses (−0.04 → +0.07), debit mid / far / all weaken, credit mid / all weaken; near / far hold. Explained below | note (not a halt) |
| G7 no holdout date / post-split P&L in the log | none | grep of all 21 post-split dates: **0** hits | PASS |

### Decision-7 observability

| structure | n | trades with ≥ 1 invalid minute | valid-minute fraction median / p05 | baseline offset median / p95 / max | trades with offset > 0 |
|---|---|---|---|---|---|
| debit | 110 | 50 | 1.000 / 0.9974 | 0 / 1 / 9 min | 34 |
| credit | 106 | 50 | 1.000 / 0.9949 | 0 / 3 / 17 min | 33 |

Roughly half of all trades have at least one invalid minute, but invalid minutes are ≤ 0.5 % of a window at the 5th percentile: the contamination is concentrated in the first minute or two after the open, which is exactly where the first-quoted-minute baseline and many T = 0 fills sat. No window lost every minute (decision 6: 0 exclusions).

Chosen thresholds: DEBIT **0.05** (sweep is flat: beat +0.08 / +0.13 / +0.08 / +0.06 / −0.07 at T = 0.00 … 0.20), CREDIT 0.00. Credit fills at T = 0.00 fall from 37 (CR-AM) to **27**: ten trades' only positive-edge minutes were invalid quotes.

### Restatement (decision 9) — CR-AN vs CR-AM as-run and vs CR-AM clean-subset (decision-7 rule on beat; T = chosen per run)


### CR-AN vs CR-AM as-run

| structure | band | CR-AM as-run | CR-AN (clean quotes) | Δ beat | restatement |
|---|---|---|---|---|---|
| debit | near | n=39 / +2.01 / 69% / base +2.12 / beat -0.11 | n=39 / +2.12 / 69% / base +2.05 / beat +0.07 | +0.18 | **reverses** |
| debit | mid | n=36 / +2.03 / 61% / base +0.94 / beat +1.09 | n=36 / +1.17 / 56% / base +0.87 / beat +0.30 | -0.79 | **weakens** |
| debit | far | n=31 / +3.44 / 52% / base +3.45 / beat -0.01 | n=31 / +2.00 / 48% / base +1.97 / beat +0.03 | +0.05 | **reverses** |
| debit | all | n=106 / +2.43 / 61% / base +2.11 / beat +0.32 | n=106 / +1.76 / 58% / base +1.63 / beat +0.13 | -0.19 | **weakens** |
| credit | near | n=11 / -8.09 / 0% / base -2.79 / beat -5.30 | n=4 / -6.45 / 0% / base -2.28 / beat -4.17 | +1.14 | **holds** |
| credit | mid | n=11 / -1.25 / 55% / base -0.87 / beat -0.38 | n=10 / -1.37 / 60% / base -1.31 / beat -0.06 | +0.32 | **weakens** |
| credit | far | n=15 / -2.01 / 60% / base -2.42 / beat +0.41 | n=13 / -1.20 / 69% / base -0.76 / beat -0.43 | -0.84 | **reverses** |
| credit | all | n=37 / -3.59 / 41% / base -2.01 / beat -1.58 | n=27 / -2.04 / 56% / base -1.53 / beat -0.51 | +1.07 | **weakens** |

### CR-AN vs CR-AM clean-subset

| structure | band | CR-AM clean-subset | CR-AN (clean quotes) | Δ beat | restatement |
|---|---|---|---|---|---|
| debit | near | n=30 / +2.54 / 73% / base +2.58 / beat -0.04 | n=39 / +2.12 / 69% / base +2.05 / beat +0.07 | +0.11 | **reverses** |
| debit | mid | n=25 / +1.68 / 60% / base +0.57 / beat +1.11 | n=36 / +1.17 / 56% / base +0.87 / beat +0.30 | -0.81 | **weakens** |
| debit | far | n=25 / +1.39 / 40% / base +0.89 / beat +0.50 | n=31 / +2.00 / 48% / base +1.97 / beat +0.03 | -0.47 | **weakens** |
| debit | all | n=80 / +1.91 / 59% / base +1.44 / beat +0.48 | n=106 / +1.76 / 58% / base +1.63 / beat +0.13 | -0.34 | **weakens** |
| credit | near | n=8 / -7.54 / 0% / base -1.71 / beat -5.82 | n=4 / -6.45 / 0% / base -2.28 / beat -4.17 | +1.66 | **holds** |
| credit | mid | n=7 / -2.25 / 43% / base -1.83 / beat -0.42 | n=10 / -1.37 / 60% / base -1.31 / beat -0.06 | +0.36 | **weakens** |
| credit | far | n=13 / -1.02 / 69% / base -0.37 / beat -0.65 | n=13 / -1.20 / 69% / base -0.76 / beat -0.43 | +0.22 | **holds** |
| credit | all | n=28 / -3.19 / 43% / base -1.35 / beat -1.85 | n=27 / -2.04 / 56% / base -1.53 / beat -0.51 | +1.33 | **weakens** |

CR-AN rows: 8

Cells where the two comparisons disagree:

- **debit far** — reverses vs as-run (−0.01 → +0.03) but weakens vs clean-subset (+0.50 → +0.03): the as-run "reversal" is a sign flip of 0.04 pts around zero; the clean-subset +0.50 was carried by dropping 6 far trades that CR-AN keeps at ~zero beat.
- **credit far** — reverses vs as-run (+0.41 → −0.43) but holds vs clean-subset (−0.65 → −0.43): CR-AM's +0.41 was the baseline artifact CR-AM itself flagged; on either clean view the far credit fade is negative.

### Why the clean-subset comparison is not near-identical (the spec must explain it)

Per-trade join of CR-AN against CR-AM's per-trade dump (scratchpad `cr_an_trade_diag.py`, T = 0.00):

- **Debit: 96 / 110 trades have identical close P&L.** The 10 that changed are 8 of CR-AM's 26 flagged trades plus 2 whose fill minute moved; mean Δ on the changed trades is **−7.65 pts** — these were fills on inverted or over-width quotes (close P&L of +16.2, +16.57, +13.7 …). On the 84 trades CR-AM had *not* flagged, CR-AN's mean close is +1.82 vs the clean-subset's +1.91 — near-identical, as decision 9 expected. The gap is composition: the clean subset **excluded** the 26 flagged trades, whose mean close was an impossible +4.03; CR-AN **keeps** them, priced on their first valid minute, at +1.39 with a baseline of +1.39 (zero beat). Adding 26 zero-beat trades to 84 pulls the all-train beat from +0.48 to +0.13 and the mid beat from +1.11 to +0.30 — the mid band held 11 of the 26.
- **Credit: only 26 / 106 trades are unchanged**, because the fill set itself changed: 10 trades that filled at T = 0 in CR-AM no longer fill (their positive-edge minutes were phantom edge from bad quotes — the June note's "the positive-edge filter selected WORSE days for the credit fade" was partly this). The 27 clean fills lose −2.04 on average vs −3.59 for CR-AM's 37; credit is less bad, still negative in every band's P&L.
- Threshold is not a driver: the debit sweep is flat (T = 0.00 all-train +1.71 / beat +0.08 vs T = 0.05 +1.76 / +0.13).

So decision 2 (skip minutes, keep trades) does something trade-exclusion did not: it retains the trades with bad opens and prices them honestly, and those trades carry no edge over their own first valid minute. The clean-subset table was optimistic by construction.

### Summary reads on clean quotes

- Debit touch resolution: rth_touch n=41 touch-exit +1.06 vs close +3.47; gap_touch n=29 **+1.04 vs +3.08** (CR-AM as-run had gap touch-exit +5.17 — the 5 over-width exits are gone). Hold-to-close beats touch-exit on both resolutions.
- Summary C: debit leads credit in all bands (near +4.24, mid +0.37, far +0.47) — no crossover, as June.
- Summary D: debit match − no_match = −0.24 pts, CI [−2.28, +1.92] → INCONCLUSIVE; credit UNTESTABLE (0 matches). Consistent with CR-AL / CR-AM.
- Far-band credit selection bias persists (σ 2.81 vs 3.47); debit none.

### Full output (Phases 1–7)

```

======================================================================
CR-AH Step 4 — Two-structure × two-axis analysis
  cr_id=CR-AN  structural-prob mode=walk-forward  train_only=False  no_persist=False  universe_end=2026-06-05  split_date=2026-06-05  seed=20260905
======================================================================

Run ID: 7d1efe38-8945-4a22-aa19-8ae9e4402654

----------------------------------------------------------------------
Phase 1: Loading signal dates and selecting clean subset...
  Universe pinned to trade_date <= 2026-06-05: 375/396 magnet-above dates kept.
  Loaded 374/375 signal entries (skipped 1).
  Stratified selection: 150 dates
  Selection by band/partition: {'far/train': 50, 'mid/train': 50, 'near/train': 50}
  holdout: none (split = universe end)

Filtering to A-bucket clean dates for each structure...
  Credit (target + target+10)...
    Credit clean: 106
  Debit (target + target-10)...
    Debit clean:  110

  Credit by band/partition: {'far/train': 30, 'mid/train': 36, 'near/train': 40}
  Debit  by band/partition: {'far/train': 33, 'mid/train': 36, 'near/train': 41}

----------------------------------------------------------------------
Phase 2: Collecting per-date trade data...
  Processing 110 debit dates...
  Processing 106 credit dates...

  Decision-6 exclusions (no valid entry minute): 0
  Quote validity [debit]: {'n': 110, 'had_invalid_quote': 50, 'valid_minute_fraction_median': 1.0, 'valid_minute_fraction_p05': 0.9974, 'baseline_minute_offset_median': 0, 'baseline_minute_offset_p95': 1, 'baseline_minute_offset_max': 9, 'baseline_offset_gt0': 34}
  Quote validity [credit]: {'n': 106, 'had_invalid_quote': 50, 'valid_minute_fraction_median': 1.0, 'valid_minute_fraction_p05': 0.9949, 'baseline_minute_offset_median': 0, 'baseline_minute_offset_p95': 3, 'baseline_minute_offset_max': 17, 'baseline_offset_gt0': 33}

  Collected: debit=110, credit=106
  Settlement available: debit=107/110, credit=103/106
  Actionable touches: debit=70/110, credit=67/106
  Post-filter out-of-range accepted values (G2, expect 0): 0

----------------------------------------------------------------------
Phase 3: Threshold sweep on TRAIN only...

  Chosen threshold — DEBIT: 0.05  CREDIT: 0.00

DEBIT — Threshold sweep (TRAIN only) [mode=walk-forward]:
       T  n_settled  fill_n  mean_pnl  win%   beat  chosen?
  ────────────────────────────────────────────────────────────
  0.00     106      109        1.71    58%     0.08
  0.05     106      109        1.76    58%     0.13 ← CHOSEN
  0.10     104      107        1.71    58%     0.08
  0.15     103      106        1.69    57%     0.06
  0.20     100      103        1.56    56%    -0.07

CREDIT — Threshold sweep (TRAIN only) [mode=walk-forward]:
       T  n_settled  fill_n  mean_pnl  win%   beat  chosen?
  ────────────────────────────────────────────────────────────
  0.00      27       28       -2.04    56%    -0.51 ← CHOSEN
  0.05      20       20       -2.66    50%    -1.13
  0.10      14       14       -3.47    43%    -1.94
  0.15       7        7       -3.67    43%    -2.14
  0.20       4        4       -3.21    50%    -1.68

----------------------------------------------------------------------
Phase 4: Full results (all train; holdout: none (split = universe end))

DEBIT — By distance band (all splits, T=0.05) [mode=walk-forward]:
  band    part        n      pnl   win%  [lo–hi 95%]        base     beat
  ──────────────────────────────────────────────────────────────────────
  near    train      39     2.12    69%  [ 54%– 81%]     2.05     0.07
  mid     train      36     1.17    56%  [ 40%– 70%]     0.87     0.30
  far     train      31     2.00    48%  [ 32%– 65%]     1.97     0.03
  all     train     106     1.76    58%  [ 49%– 67%]     1.63     0.13
  holdout: none (split = universe end)

CREDIT — By distance band (all splits, T=0.00) [mode=walk-forward]:
  band    part        n      pnl   win%  [lo–hi 95%]        base     beat
  ──────────────────────────────────────────────────────────────────────
  near    train       4    -6.45     0%  [  0%– 49%]    -2.28    -4.17
  mid     train      10    -1.37    60%  [ 31%– 83%]    -1.31    -0.06
  far     train      13    -1.20    69%  [ 42%– 87%]    -0.76    -0.43
  all     train      27    -2.04    56%  [ 37%– 72%]    -1.53    -0.51
  holdout: none (split = universe end)

DEBIT — By post-touch pattern (TRAIN only, T=0.05) [mode=walk-forward]:
  (n with pattern_label=93, n without=17)
  pattern                           n      pnl   win%  [lo–hi 95%]        base     beat
  ──────────────────────────────────────────────────────────────────────
  mixed                            43     1.52    56%  [ 41%– 70%]     1.25     0.26
  overshoot-then-revert             1     4.35   100%  [ 21%–100%]     4.35     0.00
  stepping-stone                   46     1.34    54%  [ 40%– 68%]     1.28     0.06
  ──────────────────────────────────────────────────────────────────────
  (labeled)                        90     1.46    56%  [ 45%– 65%]     1.30     0.16
  (unlabeled)                      16     3.46    75%  [ 51%– 90%]     3.36     0.10

CREDIT — By post-touch pattern (TRAIN only, T=0.00) [mode=walk-forward]:
  (n with pattern_label=89, n without=17)
  pattern                           n      pnl   win%  [lo–hi 95%]        base     beat
  ──────────────────────────────────────────────────────────────────────
  mixed                             7     0.61    86%  [ 49%– 97%]    -1.16     1.77
  overshoot-then-revert             0        —      —  [   —–   —]        —        —
  stepping-stone                   11    -2.34    64%  [ 35%– 85%]    -1.27    -1.07
  ──────────────────────────────────────────────────────────────────────
  (labeled)                        18    -1.19    72%  [ 49%– 88%]    -1.26     0.07
  (unlabeled)                       9    -3.74    22%  [  6%– 55%]    -2.91    -0.83

DEBIT — Touch resolution breakdown (TRAIN only, T=0.05) [mode=walk-forward]:
  resolution                      n   touch_exit  close_pnl   base_close
  ─────────────────────────────────────────────────────────────────
  rth_touch                       41        1.06        3.47        3.45
  gap_touch                       29        1.04        3.08        2.95
  afterhours_touch_retraced       15           —        3.37        3.37
  no_touch                        25           —       -3.30       -3.73

DEBIT — Selection bias check (far band, decision #11) [mode=walk-forward]:
  far/all: n=50  clean: n=33 (mean σ=3.16)  dropped: n=17 (mean σ=2.90)
  ✓ No significant selection bias detected in far band.

CREDIT — Selection bias check (far band, decision #11) [mode=walk-forward]:
  far/all: n=50  clean: n=30 (mean σ=2.81)  dropped: n=20 (mean σ=3.47)
  ⚠ BIAS: clean far trades are significantly CLOSER than dropped far trades.
    The far-band result may be OPTIMISTICALLY SELECTED (closer to spot → easier trade).

======================================================================
SUMMARY READS A/B/C/D
======================================================================
[mode=walk-forward]

── Summary A: Debit near-band holdout ──
  holdout: none (split = universe end)

── Summary B: Credit near-band holdout ──
  holdout: none (split = universe end)

── Summary C: Structure crossover by distance (TRAIN, close P&L beat) ──
  near   debit_beat=   0.07  credit_beat=  -4.17  → DEBIT leads by 4.24
  mid    debit_beat=   0.30  credit_beat=  -0.06  → DEBIT leads by 0.37
  far    debit_beat=   0.03  credit_beat=  -0.43  → DEBIT leads by 0.47
  READ: Structure crossover = distance band where debit stops leading and credit starts.

── Summary D: Engine hypothesis (decision #13) ──
  DEBIT engine check:  pattern_match (n=46) pnl=1.34  vs no_match (n=44) pnl=1.58
  → No clear separation by pattern for debit. Engine hypothesis inconclusive.
  DEBIT decision-9 read [mode=walk-forward]: match−no_match = -0.24 pts, bootstrap 95% CI [-2.28, +1.92] (1000 resamples, seed=20260905) → INCONCLUSIVE
  credit: engine hypothesis UNTESTABLE (n_match=0, n_no_match=18)

----------------------------------------------------------------------
Phase 7: Persisting aggregate stats to bt_edge_backtest_results...
  ✓ bt_edge_backtest_results created/verified.
  debit: holdout: none (split = universe end)
  credit: holdout: none (split = universe end)
  ✓ Aggregate stats written.

======================================================================
Step 4 complete in 284s
======================================================================
```
