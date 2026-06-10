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

### Decision numbers (pre-re-backfill)

**RECOVERABLE (C = 4):** Fixing the holiday-aware calc recovers 4 dates (3 train, 1 holdout — mid/holdout only). Mid/holdout goes n=8 → n=9. Far/holdout: no change (stays n=5). Not material.

**CONTAMINATED (B = 47):** The dominant issue. 47 of 103 fetched dates silently backfilled a 14-DTE option instead of 15-DTE. The OPRA symbol encoded the wrong expiry date (blind calc counted a holiday as a business day). By split: train=40, holdout=7. Whether DTE=14 vs DTE=15 is economically material at ~14 DTE is the user's call.

**GENUINE GAP (D = 43):** 43 of 47 skips are valid trading days with no ORATS data — not recoverable by any calc fix. Confirmed by DB: some Thursday expirations (e.g. 2023-06-01, 2023-06-08) have 0 symbols in `orats_options_minute`. Latest DB snapshot = 2026-06-08, so holdout D-bucket is NOT a recency cutoff — specific expiry/trade-date combinations are simply absent from ORATS coverage.

---

## B-bucket expiry fix — root cause, fix, re-backfill, re-audit

### Root cause (Part 1)

`nth_business_day(from_date, n)` at `scripts/cr_ah_step2_stratified_backfill.py:204` was a holiday-blind Mon–Fri counter. When a NYSE holiday fell in the 15-day window, the function returned the 14th true trading day. ORATS served data for a 14-DTE contract; the wrong OPRA symbol was written to `orats_options_minute` with no error.

Blast radius: LOCAL to the backfill script. The live engine uses separate functions (`_add_trading_days` / `nearest_spx_expiration` in `packages/shared/options_cache/pricing.py`) — same holiday-blind class of bug, different code path, not touched here.

B vs C reconciliation: both are the same root cause (holiday-blind counter). B = holiday at positions 1–14 (function returns 14th true trading day). C = holiday at position 15 (function returns a holiday date, ORATS 404). One bug, two failure modes.

### Fix (Part 2)

- Added `_NYSE_HOLIDAYS` frozenset (40 dates, 2023–2026, including Carter funeral and Inauguration Day 2025-01-20).
- `nth_business_day()` now skips weekends AND `_NYSE_HOLIDAYS`.
- `_blind_nth_weekday()` preserved (original buggy impl) for B-date identification in `REBACKFILL_B_ONLY` mode.
- `REBACKFILL_B_ONLY` config flag added: filters to only dates where `blind != aware` and not in `_PREVIOUSLY_SKIPPED_404`.
- 9/9 unit tests green in `packages/shared/backtest/tests/test_expiry_calc.py`. Tests cover: clean window, B-bucket (holiday at pos 1–14), C-bucket (holiday at pos 15), two-holiday windows, and spot-checks that aware result is always a valid trading day.
- Committed: `21e2847`.

Coexistence: old wrong-expiry bars (e.g., SPX230531*) remain in `orats_options_minute` as orphans. New correct-expiry OPRAs are different symbols — no conflict, no deletion needed.

### Re-backfill result (Part 3)

`REBACKFILL_B_ONLY=True, DRY_RUN=False`. Run ID: `928bcb41-3ab0-4132-ae7a-f8dc53e101eb`.

- Dates processed: 47 / Fetched (per script): 38 / Skipped 404: 9
- Total bars written: 62,512

9 dates that 404'd at correct expiry (now reclassified → D):

| Date | Cell | Correct expiry |
|---|---|---|
| 2023-11-10 | far/train | 2023-12-04 (long leg absent; only short in DB from prior load) |
| 2024-08-14 | mid/train | 2024-09-05 |
| 2024-08-30 | far/train | 2024-09-23 |
| 2025-01-31 | far/train | 2025-02-24 |
| 2025-05-09 | mid/train | 2025-06-02 |
| 2025-08-15 | near/holdout | 2025-09-08 |
| 2025-11-14 | far/holdout | 2025-12-08 |
| 2025-12-12 | near/holdout | 2026-01-06 |
| 2026-01-16 | far/holdout | 2026-02-09 |

Note on 2023-11-10: script reported 404, but a single-leg (short only) SPX231204C04510000 exists in DB from a prior load. The long leg (SPX231204C04520000) is absent — date is not usable for the vertical and is classified D.

### Re-audit (Part 4)

Post-fix A/B/C/D (DB-verified using `trade_date_d` + `expir_date_d` + ≥2 distinct call opras with ≥200 bars):

| Cell | A | B | C | D | Total |
|---|---|---|---|---|---|
| near/train | 28 | 0 | 1 | 5 | 34 |
| near/holdout | 7 | 0 | 0 | 9 | 16 |
| mid/train | 25 | 0 | 1 | 5 | 31 |
| mid/holdout | 8 | 0 | 1 | 10 | 19 |
| far/train | 23 | 0 | 1 | 16 | 40 |
| far/holdout | 3 | 0 | 0 | 7 | 10 |
| **Total** | **94** | **0** | **4** | **52** | **150** |

B = 0 confirmed. ✓ 38 B-dates recovered to A; 9 B-dates reclassified to D.

Effective clean sample (A only):

| Cell | Selected | Clean (A) | % |
|---|---|---|---|
| near/train | 34 | 28 | 82% |
| near/holdout | 16 | 7 | 44% |
| mid/train | 31 | 25 | 81% |
| mid/holdout | 19 | 8 | 42% |
| far/train | 40 | 23 | 58% |
| far/holdout | 10 | 3 | 30% |
| **Train total** | **106** | **76** | **72%** |
| **Holdout total** | **44** | **18** | **41%** |

Still-thin cells (D-bucket problem, NOT solved by this fix):
- **far/holdout: n=3** — extremely thin; wide CI, low power. Far holdout was already pre-flagged at n=10 before any backfill.
- **mid/holdout: n=8**, **near/holdout: n=7** — usable but CI will be wide.

The D-bucket gap (43 original + 9 reclassified = 52 total) reflects ORATS coverage holes for specific expiry/strike combinations — not a calc bug and not recoverable.

---

## D-bucket root cause diagnostic (read-only, 2026-06-09)

### Resolution function audit

`nth_business_day()` at `scripts/cr_ah_step2_stratified_backfill.py:204` **does not snap to specific weekdays**. It counts M–F business days (excluding `_NYSE_HOLIDAYS`) and returns whatever calendar date that lands on. No M/W/F filter, no listed-expiry snap. Since SPX has had daily M–F expirations since ~May 2022, every weekday in 2023–2026 is a real listed SPX expiry — the raw business-day offset always resolves to a valid contract cycle. The live path's stale `_SPX_EXPIRY_WEEKDAYS = {0,2,4}` filter in `pricing.py` is NOT present in the backfill.

### D1 / D2a / D2b classification

For each of the 52 D-dates, query: does ORATS have ANY SPX OPRA with `expir_date_d` = correct expiry (any trade date, any strike)? If yes → D1 (cycle present, our strike absent). If no → D2 (whole cycle absent). D2 subclassified by whether the expiry date is a real weekday.

| Bucket | Definition | Total |
|---|---|---|
| **D1** | Cycle present in ORATS (other strikes exist for that expiry); our specific proposed strike(s) absent | **17** |
| **D2a** | Whole expiry cycle absent from ORATS; expiry IS a real M–F weekday (SPX expiry that existed in reality) | **35** |
| **D2b** | Whole expiry cycle absent; expiry is NOT a real SPX expiry (resolution bug) | **0** |

By band × split:

| Cell | D1 | D2a | D2b | Total D |
|---|---|---|---|---|
| near/train | 1 | 4 | 0 | 5 |
| near/holdout | 3 | 6 | 0 | 9 |
| mid/train | 3 | 2 | 0 | 5 |
| mid/holdout | 4 | 6 | 0 | 10 |
| far/train | 5 | 11 | 0 | 16 |
| far/holdout | 1 | 6 | 0 | 7 |
| **Total** | **17** | **35** | **0** | **52** |

### D2b = 0 — no resolution bug

The "Thursday expirations absent" hypothesis in the Step 2 diagnostic was **wrong**. The D2a expiry weekday distribution is Mon=6, Tue=7, Wed=12, Thu=5, Fri=5 — all weekdays affected, Wednesdays most common. The issue is structural ORATS coverage, not a specific cycle type. ORATS covers only **317 of 1,003 SPX-eligible business days (31.6%)** in 2023–2026. The D2a dates are simply hitting the 68% of cycles where ORATS has no data.

### D1 pattern — strike range vs. far-OTM proposals

For the 17 D1 dates, the expiry cycle exists in ORATS but at a strike range that doesn't include our proposed strikes. For example:
- `2024-09-06` far/train: ORATS has calls at strikes 5370–5485 (12 distinct); our far-train proposal targets ~5570+ → outside ORATS' captured range.
- `2025-02-26` far/train: ORATS has calls at 5940–6020; our proposal targets ~6200+.
- Some cases have only 1 strike in ORATS at a very different level (e.g., `2023-11-10` expiry 2023-12-04: only strike 4510 = our short; long at 4520 is absent).

Pattern: **far-magnet proposals are systematically more likely to be D1** because the proposed strikes are further OTM than what ORATS captures for that cycle.

### 2023-11-01 anomaly

`2023-11-01` (far/train, expiry=2023-11-22 Wed) has no ORATS data for that expiry and no nearby expiries within ±7 days. ORATS data shows a complete gap for all SPX expiry cycles between `2023-11-03` and `2023-12-01` — a 4-week systematic absence in the ORATS DB for this date range. Classified D2a (real expiry, ORATS structural hole).

### Verdict

**D is overwhelmingly genuine and unrecoverable from ORATS.**

- D2b = 0: no third expiry-resolution bug. `nth_business_day()` is correct.
- D1 = 17: ORATS has the cycle at other strikes; our specific far-OTM strikes aren't covered. Unrecoverable from ORATS without changing strike selection.
- D2a = 35: ORATS covers only 32% of SPX expiry cycles. These 35 dates hit systematic ORATS coverage holes. Real trades existed; ORATS simply never captured them.

No change to expiry resolution would recover any D-bucket date. The only paths are: (1) accept D as unrecoverable from ORATS and proceed to Step 3 with n=94 clean dates, or (2) source missing cycles from Databento/other provider (separate CR scope).

---

## Root-symbol diagnostic — SPXW vs SPX (read-only, 2026-06-09)

### Hypothesis under test

User hypothesis: the D-bucket 404s are a root-symbol bug — ORATS serves PM-settled daily SPX expirations under root "SPXW" (not "SPX"), so requesting `SPX{yymmdd}{strike}` 404s even though the contract exists under `SPXW{yymmdd}{strike}`. The backfill always uses `OPRA_ROOT = "SPX"`.

### SPXW hypothesis verdict: REFUTED

Direct ORATS API tests on 5 D2a dates:

| Band | Trade date | Expiry | Chain ticker | expiryTod | SPX option | SPXW option |
|---|---|---|---|---|---|---|
| near/train | 2023-05-18 | 2023-06-09 Fri | **SPX** | pm | HTTP 200 ✓ | HTTP 404 ✗ |
| mid/train | 2024-05-23 | 2024-06-14 Fri | **SPX** | pm | HTTP 200 ✓ | HTTP 404 ✗ |
| far/train | 2023-11-01 | 2023-11-22 Wed | **SPX** | pm | HTTP 200 (at 4360) ✓ | HTTP 404 ✗ |
| near/holdout | 2025-08-15 | 2025-09-08 Mon | **SPX** | pm | HTTP 200 (at 6500) ✓ | HTTP 404 ✗ |
| far/holdout | 2025-12-09 | 2025-12-31 Wed | **SPX** | pm | HTTP 200 ✓ | HTTP 404 ✗ |

**ORATS uses "SPX" root for ALL SPX expirations**, including PM-settled dailies and weeklies. The `expiryTod` column (value `'pm'`) distinguishes AM/PM settlement; the root never changes. The `opra.py` docstring was correct. SPXW always 404s — ORATS does not serve under that root at all.

### Real discovery: 17 D-dates are recoverable — re-backfill filter bug

While refuting SPXW, a different bug was found: the ORATS chain endpoint returned data for the D2a expiry dates under "SPX" root (the correct root), but the OPTION endpoint at our **proposed strike** returned HTTP 200 on some and 404 on others. Re-examining the re-backfill filter revealed:

`_PREVIOUSLY_SKIPPED_404` was set to all 47 original 404 dates (C+D from the original blind-expiry backfill). The `REBACKFILL_B_ONLY` filter excluded these from re-processing:
```python
b_bucket = [e for e in all_selected
            if blind != aware AND e["trade_date"] not in _PREVIOUSLY_SKIPPED_404]
```

**Bug:** 30 of the 43 original D-dates have `blind ≠ aware` (a holiday fell in positions 1–14). For these dates:
1. The original backfill tried the BLIND expiry → ORATS 404'd (blind expiry has no data)
2. The date went into `_PREVIOUSLY_SKIPPED_404`
3. The re-backfill filter excluded it, so the **CORRECT expiry was NEVER tried**
4. Re-audit shows these as D (no data in cache), but that's cache absence, not ORATS absence

Tested all 30 at their correct expiry via the ORATS option endpoint:

- **17 of 30 → HTTP 200**: ORATS HAS the data; the correct expiry succeeds at the proposed strike. These are recoverable with a targeted re-backfill.
- **13 of 30 → HTTP 404**: The correct expiry also fails in ORATS at the proposed strike. Truly unrecoverable.

### Recovery breakdown by band × split (if authorized)

| Cell | Current A | Recoverable | New A | Selected | New % |
|---|---|---|---|---|---|
| near/train | 28 | +3 | 31 | 34 | 91% |
| near/holdout | 7 | +2 | 9 | 16 | 56% |
| mid/train | 25 | +2 | 27 | 31 | 87% |
| mid/holdout | 8 | +3 | 11 | 19 | 58% |
| far/train | 23 | +4 | 27 | 40 | 68% |
| far/holdout | 3 | +3 | 6 | 10 | 60% |
| **Total** | **94** | **+17** | **111** | **150** | **74%** |

**Holdout: 18 → 26 (+8).** far/holdout doubles from 3 to 6.

### Revised D-bucket decomposition

| Bucket | Count | Definition |
|---|---|---|
| True D (genuinely unrecoverable) | **35** | Correct expiry tried (or blind==aware, equivalent) AND ORATS 404s at proposed strike |
| Filter-miss D (re-backfill never tried correct expiry, ORATS has data) | **17** | Holiday in window, blind 404'd, in `_PREVIOUSLY_SKIPPED_404`, correct expiry untested → now confirmed available |
| **Total D** | **52** | |

The 17 recoverable dates:
- near/train (3): 2023-05-18, 2024-11-12, 2025-06-11
- near/holdout (2): 2026-01-29, 2026-05-21
- mid/train (2): 2023-05-11, 2024-05-23
- mid/holdout (3): 2025-08-19, 2026-02-12, 2026-05-19
- far/train (4): 2023-05-25, 2024-06-12, 2024-06-24, 2024-08-15
- far/holdout (3): 2025-08-21, 2025-12-09, 2025-12-17

### Revised verdict

**SPXW is wrong. The "68% coverage hole" was an artifact of the re-backfill filter.**

ORATS does cover most of the D-bucket expiries under "SPX" root. What actually happened: for 30 dates with holidays in the 15-day window, the original backfill tried the wrong (blind) expiry which 404'd, the date was incorrectly placed in `_PREVIOUSLY_SKIPPED_404`, and the correct expiry was never attempted.

17 of those 30 are confirmed ORATS-available at the correct expiry+strike. These can be recovered by a targeted re-backfill (fix `_PREVIOUSLY_SKIPPED_404` to only include dates where `blind==aware`, then run the filter-miss dates at their correct expiry).

Decision to user: authorize the targeted 17-date re-backfill? If yes, effective clean sample grows from A=94 to A=111, holdout from 18 to 26.
