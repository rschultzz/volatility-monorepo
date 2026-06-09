# CR-AH — Per-Structure Edge Backtest (Thin)

Branch: `feat/CR-AH-per-structure-edge-backtest`
Vault note: `CRs/CR-AH — Per-Structure Edge Backtest (Thin).md`

## Scope

First end-to-end answer to "does the signal make money out of sample?" for the overhead-magnet vertical (SPX, credit call spread). Per-proposal thin backfill into `orats_options_minute`: entry-day crawl, ES-driven touch-window, expiry settlement. Structure-agnostic harness, vertical plugin ships first. Two parallel outcomes: touch-exit vs. hold-to-close, reported against a baseline on a locked train/holdout split.

## Locked design points

1. All prices from real `orats_options_minute` mids or expiry settlement — no BSM in the decision path.
2. Net structure price = simultaneous leg mids at one timestamp, never day-low pairing.
3. Entry = first minute where net credit ≥ edge threshold on the signal day.
4. Touch = ES-detected from `ironbeam_es_1m_bars`; exit window ~1 hr around touch minute (first valid mid at or after touch, not window's best tick).
5. Close = three-zone intrinsic payoff (below / between / above strikes), not a binary flag.
6. Touch-exit and close are two parallel strategies, not sequential — one entry, two outcome columns.
7. Baseline = same structure entered at first quoted minute on the same date (no edge filter).
8. Holdout split locked at 2025-08-12 (temporal 70/30). Threshold tuned on train only; holdout read once.

## Step 0 findings

### Power check (Step 0.0)

| Metric | Value |
|---|---|
| Total magnet-above dates (v0.6.0-openiv, SPX) | 375 |
| Train (≤ 2025-08-12) | 263 |
| Holdout (> 2025-08-12) | 112 |

Holdout n=112 → **HEALTHY**. One regime supports readable CI.

### Coverage probe (Step 0.1)

0/5 probe dates had pre-existing option data for the vertical's strikes. Heavy backfill required. ORATS does serve 2023+ SPX data once fetched. Entry-day: ≤ 292,500 rows (375 dates × 2 strikes × 390 min). Touch-window: ≤ 45,000 rows. Settlement: ≤ 750 rows.

### Distance distribution (Step 0.0b)

50-date stratified subsample (150 total): near (<1.5σ) n=50, mid (1.5–2σ) n=50, far (>2σ) n=50. Train 70% / holdout 30% per band. Far-holdout = 10 dates — flagged as thin before backfill.

## Step 2 backfill results

**Full 150-date run:** 103 fetched, 47 skipped (ORATS 404).

| Cell | Selected | Fetched | Skipped |
|---|---|---|---|
| near/train | 34 | 28 | 6 |
| near/holdout | 16 | 9 | 7 |
| mid/train | 31 | 27 | 4 |
| mid/holdout | 19 | 8 | 11 |
| far/train | 40 | 26 | 14 |
| far/holdout | 10 | 5 | 5 |

## Step 2 diagnostic — A/B/C/D expiry classification

### Blast radius

`nth_business_day()` is LOCAL to `scripts/cr_ah_step2_stratified_backfill.py:124`. Not shared.

Live path has the SAME holiday-blind bug (separate production concern — not in scope for CR-AH):
- `_add_trading_days()` at `packages/shared/options_cache/pricing.py:434` — holiday-blind
- `nearest_spx_expiration()` at `pricing.py:445` — also uses stale `_SPX_EXPIRY_WEEKDAYS = {0, 2, 4}` (M/W/F only; SPX has daily expirations since May 2022)

### A/B/C/D classification (NYSE calendar 2023–2026)

| Bucket | Definition | Total |
|---|---|---|
| A correct_expiry_fetched | No holiday in window; ORATS returned data | 56 |
| B wrong_expiry_fetched | Holiday in window → fetched DTE=14 instead of DTE=15; ORATS returned data | **47** |
| C wrong_expiry_404 | Blind expiry = NYSE holiday → 404; recoverable by fixing calc | 4 |
| D correct_expiry_404 | Valid trading day; ORATS has no data; NOT recoverable | 43 |

By band × split:

| Cell | A | B | C | D |
|---|---|---|---|---|
| near/train | 16 | 12 | 1 | 5 |
| near/holdout | 5 | 4 | 0 | 7 |
| mid/train | 11 | 16 | 1 | 3 |
| mid/holdout | 7 | 1 | 1 | 10 |
| far/train | 14 | 12 | 1 | 13 |
| far/holdout | 3 | 2 | 0 | 5 |

### Decision numbers

**RECOVERABLE (C = 4):** Fixing the holiday-aware calc recovers 4 dates (3 train, 1 holdout — mid/holdout only). Mid/holdout goes n=8 → n=9. Far/holdout: no change (stays n=5). Not material.

**CONTAMINATED (B = 47):** The dominant issue. 47 of 103 fetched dates silently backfilled a 14-DTE option instead of 15-DTE. The OPRA symbol encoded the wrong expiry date (blind calc counted a holiday as a business day). By split: train=40, holdout=7. Whether DTE=14 vs DTE=15 is economically material at ~14 DTE is the user's call.

**GENUINE GAP (D = 43):** 43 of 47 skips are valid trading days with no ORATS data — not recoverable by any calc fix. Confirmed by DB: some Thursday expirations (e.g. 2023-06-01, 2023-06-08) have 0 symbols in `orats_options_minute`. Latest DB snapshot = 2026-06-08, so holdout D-bucket is NOT a recency cutoff — specific expiry/trade-date combinations are simply absent from ORATS coverage.

**Awaiting user decision:** proceed to Step 3 accepting B-contamination + thin holdout, OR re-backfill the 47 B-dates at the correct (holiday-aware) expiry.
