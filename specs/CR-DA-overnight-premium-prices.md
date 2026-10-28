# CR-DA — Overnight premium on straddle prices (T-S1a′ on prices only, S1)

Read-only, pre-registered, one run. Road 2 step 2 of vault `Dash/research/two-roads.md`. Branch `feat/CR-DA-overnight-premium-prices` off `origin/main` `6bb2d81` (the CR-CY merge), worktree `.claude/worktrees/cr-da`.

No API calls, no credentials touched, no table writes, no cron. Holdout (2026-06-06 → 2026-10-26) is not read; nothing after 2026-06-05 is read.

Sources read for this freeze: CLAUDE.md; `components/field-conventions.md` (the CR-CY rows: coverage, and bid = ask = 0 = no quote); `research/standing-rulings.md` (B4 incl. the Road 2 exception, 2026-10-27); `specs/CR-BO-session-premium.md`; `specs/CR-BP-session-premium-live-open.md`; `specs/CR-CY-straddle-backfill.md`; vault `research/T-S1a-prime — draft pre-registration.md`.

## Question (kickoff)

CR-BO's T-S1a, S1 bucket, both ends now priced: does the SPX 1-DTE option market charge more implied variance for the overnight session than ES realises overnight, and less for the next RTH session?

## Rulings by Ryan in chat at freeze (2026-10-27)

Asked before the freeze; recorded here and in the rulings change log.

| # | Question | Ryan's answer |
|---|---|---|
| R-DA1 | CR-BO L8 kill (a) ("P_on CI spans 0 in every bucket" → KILL) evaluated over S1 alone would call a wide CI a KILL, against B1 | **B1 governs.** Verdict per V1 below; L8 kill (a)/(b) as written is printed as a readout. No deviation. |
| R-DA2 | The implied ends are 15:55 and 09:35 ET; CR-BO's RV windows are 16:00 → 09:30 and 09:30 → 16:00, so 10 RTH minutes sit in the implied decay but not in RV_on (biases P_on up, P_rth down) | **Match the clock.** Primary RV_on = 15:55 t → 09:35 t+1, RV_rth = 09:35 → 16:00 t+1 (CR-BP C6(i) approach). CR-BO's windows are a labelled readout. |
| R-DA3 | C1's ±0.5 IM strangle cannot be priced from CR-CY's K0 / K0 ± 5 strikes without a fetch | **±5 strangle readout:** put K0 − 5 + call K0 + 5, same solve, labelled "nearest-wing strangle, not C1's ±0.5 IM". |

## Locked definitions

| # | Decision |
|---|---|
| D1 | **Universe.** CR-BO's 621 ES-clean nights (`scripts/output/cr_bo/step0_pairs.csv`, `keep`; = `scripts/cr_cy_nights.csv`), all with t+1 ≤ 2026-06-05, **including 2025** (B4 Road 2 exception). Bucket S1 only: the SPX PM expiry settling on t+1 (1-DTE at the close, 0DTE at the open; one contract at both ends). |
| D2 | **Exclusions, in order, each counted.** (a) CR-CY window incomplete: either the close or the open window has fewer than 6 contracts with rows (`scripts/logs/cr_cy/full_windows.csv`; expect 26: 22 monthly-expiry AM collisions + 4 ORATS-404 nights). (b) No valid quote for an ATM leg (D3) at either end. (c) V_px solve fails at either end. |
| D3 | **Quotes.** `orats_options_minute`, ticker SPX, `expiry_tod = 'pm'`, `expir_date_d` = t+1. Close rows have `trade_date_d` = t; open rows `trade_date_d` = t+1. A quote is valid iff `leg_quote_is_valid(bid, ask)` (bid ≥ 0, ask > 0, bid ≤ ask). **bid = ask = 0 rows are missing** (field-conventions, CR-CY). Mid = (bid + ask)/2. A contract with no valid quote at the target minute takes the nearest valid minute within ±2 min (order 0, −1, +1, −2, +2; ties to the earlier minute), each contract separately. Else it is missing. |
| D4 | **Snapshots.** Close = t 15:55 ET (`snapshot_pt` 12:55). Open = t+1 09:35 ET (`snapshot_pt` 06:35). |
| D5 | **Strikes.** The CR-CY placement (`scripts/logs/cr_cy/full_plan.csv`): K0_c at the close, K0_o at the open (the middle strike of each window's three). Primary instrument = the **true ATM straddle**, call + put at K0 (CR-BP's primary). Readout = the ±5 strangle (R-DA3). |
| D6 | **V_px (CR-BP's K-aware Black solve, code reused from `scripts/cr_bp_step0.py`: `black`, `solve_v`).** Black(F = S, r = q = 0) in total variance v = σ²τ, one unknown: call(K, v) + put(K, v) = straddle mid. No τ convention enters V_px. **S = the ES-implied SPX spot at the snapshot minute** = ES print at that minute − CR-CY's basis_t (median ES − SPX cash over 12:30–12:54 PT on t). At the close this is CR-CY's S_c exactly (the 15:55 print); at the open it is the 09:35 ES print (close of the bar starting 09:34 ET) − basis_t. This is CR-BP's "S = the placement spot" convention moved to the snapshot minute. A parity forward F = K + C − P is a Step 0 diagnostic only. |
| D7 | **τ** (reporting only: annualised IV = √(V/τ)) = calendar time to 16:00 ET on t+1 / 365 days, as CR-BO. |
| D8 | **Premiums (total variance; variance points = × 10⁴).** V_c = V_px at the close; V_o = V_px at the open; implied variance at the PM settle = 0. D_on = V_c − V_o. D_rth = V_o − 0. **Primary (R-DA2):** RV_on = Σ r² over ES bars 15:55 t … 09:34 t+1 (bar-start ET stamps; the price path 15:55 print → 09:35 print). RV_rth = Σ r² over bars 09:35 … 15:59 t+1 (09:35 → 16:00). r = ln(close_i / close_{i−1}) on `ironbeam_es_1m_bars` exactly as CR-BO (`load_es`; the reopen gap belongs to the first bar after a halt). P_on = D_on − RV_on; P_rth = D_rth − RV_rth. **Readout (CR-BO windows):** RV_on_bo over bars 16:00 … 09:29, RV_rth_bo over 09:30 … 15:59; P_on_bo, P_rth_bo likewise. |
| D9 | **Population** = CR-BO's non-event weeknights: t Mon–Thu, t+1 the next calendar day, no CPI / PPI / NFP / PCE at 08:30 ET on t+1 (event dates from `scripts/cr_bo_common.py`). FOMC on t+1 stays in. |
| D10 | **Statistics.** Mean with 95 % percentile bootstrap CI, 10 000 resamples, seed 17 (CR-BO's `boot`); medians with the same bootstrap; sign test (two-sided exact binomial); the Muravyev–Ni paired contrast P_on − P_rth with CI. **No outlier rule** (CR-BP lesson). |

## V1 — primary verdict (L8 for S1, with B1 per R-DA1)

On the D9 population, S1, primary windows (D8):
- **PASS:** mean P_on CI lower bound > 0 AND (mean P_rth ≤ 0 OR the P_rth CI contains 0). This is CR-BO's L8 pass reading verbatim.
- **REFUTED:** mean P_on CI upper bound < 0; OR mean P_rth ≥ mean P_on AND the paired-contrast (P_on − P_rth) CI upper bound ≤ 0.
- **INCONCLUSIVE:** anything else.
- If PASS and REFUTED both hold, report REFUTED.
- **Readout:** CR-BO's L8 kill as written — kill (a) P_on CI spans 0 (S1, the only bucket); kill (b) S1 mean P_rth ≥ mean P_on — printed beside V1, not the verdict.

## Readouts (pre-registered, labelled "readout"; none changes V1)

1. The 5 largest |P_on| nights (date, V_c, V_o, RV_on, P_on); medians of P_on and P_rth with CI.
2. **Winsorised** P_on and P_rth at the population's 1 % / 99 % percentiles (each series separately), mean and CI.
3. **By year** of t: 2023 / 2024 / 2025 / 2026, each with n; 2025 shown in its own row so its role is visible.
4. **Splits** as CR-BO: weekday of t (Mon–Thu, Fri→Mon, holiday-span); 08:30 event nights vs not (and per release); FOMC on t+1.
5. **Placebo (CR-BO L9):** D_on(t) − RV_on(t+1 → t+2) with the primary window (15:55 t+1 → 09:35 t+2), on nights where t+1 is also in the 621 ES universe; the main P_on recomputed on the same nights; corr(D_on, RV_on) vs corr(D_on, RV_on_next).
6. **CR-BO windows** (R-DA2): P_on_bo / P_rth_bo through V1's arithmetic, labelled.
7. **±5 strangle** (R-DA3): V_px of put(K0 − 5) + call(K0 + 5) at each end, same D3 quote rules; P_on / P_rth and V1's arithmetic, labelled.
8. **Straddle bid–ask cost per night:** full spread Σ(ask − bid) of the ATM straddle at each end (median, IQR, mean), and the round-trip half-spread cost (spread_c + spread_o)/2.
9. **Tradeable readout (B6: a mechanical cousin, not Ryan's trade):** sell the K0_c straddle at 15:55 at mid − half-spread (the legs' bids), buy the same K0_c straddle back at 09:35 at mid + half-spread (the legs' asks), in index points per night, mean and CI, median, worst 1 % night (1st percentile and the list of nights at or below it).
   - **Buyback quote.** Any cached valid PM quote for K0_c at the open (D3 rules) — K0_c is among the fetched open strikes when |K0_c − K0_o| ≤ 5; CR-AS rows may also cover it. Otherwise a **model mark**: Black(F = S_o, K0_c, V_o) call + put, plus the half-spread of that night's open ATM straddle. Reported for all nights and for the quote-only subset; the share of model-marked nights is printed.
   - **Unhedged** P&L and **ES delta-hedged at entry:** hedge h = 2N(d1) − 1 (Black straddle delta at F = S_c, K0_c, V_c), long h ES points from the 15:55 print to the 09:35 print; ES cost |h| × 0.25 pts round trip.
   - On the D9 population and on all universe nights; by year.
10. Straddle-equivalent in index points: S_c × sign(v)·√|v| for mean D_on, RV_on, P_on (CR-BO L5 readability).

## Step 0 (no outcome statistics)

No RV, no premium, no P&L. Items:
1. Universe gate: `cr_cy_nights.csv` = CR-BO `keep` set (621); CR-CY completeness count (expect 595 complete).
2. Night counts after each D2 exclusion; by year of t; the D9 population count.
3. Quote-validity rates per window and leg role (ATM call, ATM put, wing put, wing call): valid at the target minute, valid via fallback (by offset), missing; bid = ask = 0 rows met at the target minute.
4. V_px solve success rate per end (straddle and strangle).
5. **CR-BP reproduction:** on CR-BP's primary-subset nights in the CR-BO universe (`scripts/output/cr_bp/vpx_primary.parquet`, `in_universe`; expect 22), recompute V_px from `orats_options_minute` with this CR's quote and solve code at CR-BP's K and S at 06:31 PT (CR-BP's v_b minute) and 06:35. Gate: every 06:31 value within 1 % of CR-BP's.
6. Diagnostics: median |F_parity − S| at each end; |K0 − S|; share of tradeable buybacks that are quoted vs model-marked.

**Step 0 gates.** S0-G1 the universe gate (item 1) holds. S0-G2 every reproduction night within 1 % (item 5). S0-G3 V_px straddle solve success ≥ 95 % at each end among nights with valid quotes. S0-G4 final primary n (all universe nights after D2) ≥ 500. If all pass: commit the run script and run once without stopping. If any fails: STOP and report.

## Implementation order

1. Spec freeze (this file).
2. Step 0 code + findings (`scripts/cr_da_lib.py`, `scripts/cr_da_step0.py`).
3. Run script committed before the run (`scripts/cr_da_run.py`); one run; outputs `scripts/output/cr_da/{report.md, summary.json, nights.parquet}`, log `scripts/logs/cr_da/`.
4. Results + wrap.

## Standing rulings conformance (`Dash/research/standing-rulings.md`, status current, read 2026-10-27)

| # | Ruling (short) | Mark | How |
|---|---|---|---|
| A1–A21 | the setup (levels, ranges, entries, stops, skew, flow, targets, sizing) | n/a | an option-pricing measurement; no level, range or intraday setup |
| B1 | three verdicts; wide CI = inconclusive | **honoured** | V1 per Ryan's R-DA1; CR-BO's kill-on-wide-CI is a readout only |
| B2 | many parameters disclosed | honoured | every readout pre-registered here and labelled |
| B3 | one setup at a time | n/a | one hypothesis |
| B4 | train ≤ 2026-06-05; holdout frozen; R-b 2025 untouched; Road 2 exception | **honoured (Road 2 exception)** | training = CR-BO's universe incl. 2025 per the exception (2026-10-27); 2025 reported as its own row. Holdout 2026-06-06 → 10-26 not read; forward log not read |
| B5 | Databento reference for bars | honoured | RV from `ironbeam_es_1m_bars` as CR-BO: 2023–2025 `source = 'databento'`; 2026 Ironbeam because Databento has no 2026 overnight (CR-CW precedent) |
| B6 | mechanical exits labelled | honoured | readout 9 labelled a mechanical cousin, not Ryan's trade |
| B7 | placebos matched EDGE / INTERIOR | n/a | the placebo is CR-BO's one-night time shift; no level or range position exists |
| B8 | Code labels ≠ Ryan's | n/a | |
| C1, C2 | UI | n/a | |
| Revoked R1–R7 | | not used | |

No "deviates". Proceeding to Step 0.

## Step 0 — findings (2026-10-27, `scripts/cr_da_step0.py`; full output `scripts/output/cr_da/step0.md`)

No outcome statistic was computed (no RV, premium or P&L).

- **Universe.** 621 = CR-BO `keep` (S0-G1 PASS). CR-CY complete: 595.
- **Exclusions (D2):**
  - (a) CR-CY incomplete: 26. From CR-CY's `missing.csv` these are **21** AM-collision nights (`am_only`: 20 third-Friday t+1 plus 2025-04-17) and **5** ORATS-404 nights (2023-05-23, 2025-01-24, 2025-05-12, 2025-05-13, 2025-11-26).
    - **Correction:** CR-CY's notes say "22 + 4". The total of 26 was right; the split is 21 + 5.
  - (b) No valid ATM leg quote within ±2 min: 6 nights.
  - (c) V_px solve failed: 1 night (2024-04-25 → 04-26).
  - **Final n = 588; D9 population = 374** (by year of t: 2023 78, 2024 119, 2025 124, 2026 53).
- **Quote validity** on CR-CY-complete nights:
  - At the target minute: 99.5 % of close legs, 98.7–99.0 % of open legs.
  - The ±2 fallback was used on 2–3 legs per window.
  - No bid = ask = 0 row sat at the target minute; they cluster at 06:30–06:33 PT (CR-CY).
- **Solve success.**
  - Straddle: close 594/594, open 589/590.
  - Strangle: the same. Strangle readout n = 588.
  - (S0-G3 PASS.)
- **CR-BP reproduction: 22/22** (S0-G2 PASS).
  - At 06:31, CR-BP has a value on 20 nights, and all 20 reproduce exactly (max relative difference 0).
  - On the other 2 (2024-05-13, 2024-07-08), CR-BP's own 06:31 solve is NaN because the straddle mid was ≤ intrinsic, and this code returns NaN too. These are why CR-BP had 20 V_px nights, not 22.
  - **Reading recorded:** "reproduced" = within 1 % or both NaN.
  - At 06:35: 22/22 exact.
- **Diagnostics.**
  - |F_parity − S|: median 1.10 pts at the close, 1.27 at the open.
  - |K0 − S|: median 1.20 at the close, 4.31 at the open. The open strike was placed on the 09:30 print, and the K-aware solve absorbs the 09:30 → 09:35 drift.
  - Annualised straddle IV (calendar τ): 0.128 at the close, 0.206 at the open. Calendar time understates the trading-time share of a 0DTE's remaining life; V_px itself needs no τ.
- **Tradeable buyback (readout 9).** K0_c is quoted at the open on **240 of 588** nights. **348 (59 %) are model-marked** (Black at V_o, the open ATM variance, plus the open ATM half-spread), as pre-registered. The quote-only subset is reported beside the full readout; the quote-only subset is selected on small overnight moves (|K0_c − K0_o| ≤ 5), so it is not a clean estimate either.

**Gate: S0-G1 … S0-G4 PASS — no amendment; commit the run script and run once.**

## Run summary (single run, 2026-10-27 19:13 PT; `scripts/output/cr_da/report.md`, `summary.json`, `nights.parquet`)

Run script committed before the run (`26cd975`); run once. Final nights 588; D9 population n = 374.

**V1 — INCONCLUSIVE.** S1, non-event weeknights, matched clock (15:55 → 09:35 → 16:00), variance points:

| n | P_on mean | P_on median | P_rth mean | P_rth median | P_on − P_rth |
|---|---|---|---|---|---|
| 374 | −0.041 [−0.126, +0.030] | +0.008 [−0.005, +0.020] | −0.015 [−0.082, +0.027] | +0.017 [+0.004, +0.032] | −0.026 [−0.088, +0.035] |

- **Implied vs realised (means).**
  - Overnight: D_on 0.310 vs RV_on 0.351, ratio 0.883.
  - RTH: D_rth 0.489 vs RV_rth 0.505, ratio 0.970.
  - P_on > 0 on 198/374 nights (sign p 0.28).
- **Against the priors.** The P_on CI's upper bound, +0.030, lies below CR-BP's P6 prior (+0.05) and far below CR-BO's +0.260.
- **Readout — CR-BO L8 as written:** KILL S1 (kill (a) and (b) both met). Not the verdict (R-DA1).
- **Readouts.**
  - **Winsorised 1/99:** P_on −0.020 [−0.048, +0.007]; P_rth +0.010 [−0.011, +0.031].
  - **Largest |P_on|:** 4 of the 5 are April 2025, the tariff shock: 04-08 −10.2, 04-07 +6.4, 04-02 −5.8, 04-15 −2.8. The fifth is 2024-11-05, election night, +1.7.
  - **By year (P_on mean):** 2023 −0.016 [−0.042, +0.010]; 2024 +0.040 [+0.003, +0.084]; 2025 −0.142 [−0.377, +0.063] (median +0.008); 2026 −0.024 [−0.113, +0.061]. Only 2024 sits above 0, and the years disagree in sign.
  - **Placebo:** D_on against the next night's RV gives −0.104 [−0.176, −0.039], vs main −0.043 on the same 360 nights. corr(D_on, RV_on) is +0.61; corr(D_on, RV_on_next) is +0.76.
  - **CR-BO windows:** P_on −0.017 [−0.102, +0.053], median +0.022; P_rth −0.030 [−0.097, +0.013]; INCONCLUSIVE. The 10 RTH minutes these windows leave out of RV_on carry +0.024 var pts on average.
  - **±5 strangle:** P_on −0.041 [−0.125, +0.030], indistinguishable from the straddle; INCONCLUSIVE.
  - **Spreads.** The ATM straddle's full spread has median 0.50 pts at 15:55 and 0.40 at 09:35. The round-trip half-spread cost has median 0.45, mean 0.49.
  - **Tradeable** (mechanical cousin; index points per night per straddle; population):

    | | mean | median | worst 1 % |
    |---|---|---|---|
    | unhedged | −0.30 [−1.69, +0.96] | +2.72 | −57.7 |
    | ES delta-hedged at entry | −0.28 [−1.67, +0.98] | +2.66 | −58.4 |

    - Mean entry credit 35.96 vs mean exit cost 36.26.
    - All 588 nights: −0.23 [−1.47, +0.94].
    - Quote-marked nights only: +1.71 [+0.24, +2.94], n 163. This subset is selected toward small overnight moves.
    - Worst nights: 2025-04-02 −93.5 (ES −179), 2026-04-07 −76.3 (+178), 2025-04-22 −69.2 (+143), 2026-03-02 −68.5 (−124).

**Post-run diagnostic (not registered; does not change V1).** On the 240 nights where K0_c is quoted at the open, the model mark Black(S_o, K0_c, V_o) + half-spread was compared with the actual ask. The median error is 0.00, but for |K0_c − S_o| ≥ 10 (n 98) the median is −0.69 and the mean −0.79: the model under-marks the buyback after a large move. The 348 model-marked nights have a median |K0_c − S_o| of 21 pts. So the tradeable mean above is probably **optimistic** by roughly 0.4–0.7 pts per night.

**Answer.** On real straddle prices at both ends, the 1-DTE SPX market does not charge measurably more implied variance for the overnight session than ES realises, and it does not undercharge RTH. Selling the straddle at 15:55 and buying it back at 09:35 did not make money after the bid–ask. The typical night earns about +2.7 pts, and a handful of large overnight moves take it all back.
