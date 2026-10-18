# CR-BS — T-S3a: does the 10:30 ET surface state predict the remaining-day variance premium?

Read-only, pre-registered, train only (t ≤ 2026-06-05). Single run. Branch `feat/CR-BS-morning-surface` off `origin/main` `95480e3` (the CR-BR merge, PR #69).

Authority: vault `Dash/research/Test plan — lean sequence and reconciliation protocol.md` (the T-S3a block and the Reconciliation protocol are binding); `Dash/research/strategies-proposed/S3 - Morning Surface.md` (incl. its `## Status` from CR-BM); the locked decisions L1–L8 below (kickoff, Ryan, 2026-10-18). Where L1–L8 and the test-plan block differ (pass/kill bands, regressor set), **L1–L8 govern**; the block is kept verbatim for the record.

No API calls, no credentials, no table writes, no cron. Holdout (> 2026-06-05) never read; DAILY-CAPTURE rows excluded.

---

## T-S3a block (verbatim from the test plan)

### T-S3a — 10:30 surface → remaining-day premium

- **Data.** `orats_monies_minute` at 10:30 ET: 0DTE and 1DTE `atmiv`, `slope`, and curvature from `vol25/vol50/vol75` (or the delta grid); ES 1-min 10:30 → 16:00.
- **Compute.** Remaining-day implied variance = 0DTE atmiv² × (time to 16:00 / year) × remaining-day share from the Todorov–Zhang 0DTE/1DTE ratio (document the exact formula in the spec). Realized 10:30→16:00 variance from ES. Premium = implied − realized. Regress premium on 10:30 IV, slope, curvature (22-day MA smoothing as in the source); report R² and coefficients; also split premium by upside vs downside using the smile (implied semivariance above/below the forward vs realized up/down semivariance).
- **Pass.** Joint R² ≥ 0.15 on train with the 22-day smoothing (the source reports 0.35 in-sample; half of that is the bar) AND the upside premium is ≥ the downside premium on average. **Kill S3 gate.** R² < 0.05 → the 10:30 state carries nothing here; S3 reduces to the unconditional wing sale, which T-S3c alone adjudicates.
- **Literature to reconcile against.** [[BandiFusariReno2023]] (R² 21 % / 35 %; ψ ≈ 1.9; fixed 10:30; loses significance excluding vol > 100 % days), [[Bandi2026]] (1–6-day term-structure oscillations), [[AlmeidaHizmeri2025]], [[TodorovZhang2024]], [[FournierJacobs2020]], [[Natividade2017]] (ΔIV gate), [[BekaertHoerova2014]].

---

## Locked decisions (kickoff, Ryan, 2026-10-18)

| # | Decision |
|---|---|
| L1 | Universe: train sessions t ≤ 2026-06-05 with (i) an `orats_monies_minute` snapshot at 07:30 PT (10:30 ET; first snapshot ≥ 07:30 and ≤ 07:33) carrying a PM-settled expiry with dte = 0 (ORATS dte convention — confirm it is the same-day expiry, per CR-BP's note that "smallest dte > 0" at 06:33 is the 0DTE) and the next-session expiry; (ii) ES RTH bars with ≥ 370 minutes; (iii) CR-BO's exclusions (roll weeks all years, holidays, early closes). Era E0 (t < 2024-03-01) / E1 mandatory in every table. Flag FOMC statement days (14:00 ET, after the snapshot) separately; CPI/NFP/PPI/PCE days print at 08:30 ET, before the snapshot, so they stay in the universe but are flagged. |
| L2 | State at 10:30 ET, from the monies row(s), with `atmiv` primary and `vol50` reported beside it (no price-based adjudication exists at 10:30 — say so): (a) level L = log(atmiv_0DTE²); (b) skew K = (vol75 − vol25)/vol50 on the 0DTE expiry (put-side minus call-side, per the confirmed delta mapping); (c) curvature C = (vol25 + vol75 − 2·vol50)/vol50 on 0DTE; (d) term T = log(atmiv_0DTE² / atmiv_1DTE²) (Todorov–Zhang's ratio object); (e) morning realized-vs-implied M = log(RV_{09:30→10:30} / (atmiv_0DTE² · τ_morning)) with τ on the same clock as L3. Report pairwise correlations and by-era means. |
| L3 | Outcome Y = log(RV_{10:30→16:00} / (atmiv_0DTE² · τ_rem)). Clock: τ_rem on the session clock, (5.5/6.5)·(1/252) — state it; the 1/365 calendar alternative is reported as a level shift only. Because the clock shifts every day equally, L6 is written to be clock-invariant (gradients and R², not levels). RV from roll-safe ES 1-min log returns, total variance. |
| L4 | Model: Y_t regressed on (L, K, C, T, M). In-sample R² on all train days (the Bandi–Fusari–Renò comparison; they report 21 % univariate / 35 % joint with a 22-day MA — replicate the MA version as a secondary column). Primary = walk-forward within train: fit on an expanding window starting at the first 250 days, predict each next day, OOS R² and the quintile gradient of the *predicted* premium (Q5 = richest predicted) against realized Y. Nothing from holdout. |
| L5 | Statistics: OOS R²; Q5 − Q1 of realized Y by predicted-premium quintile with bootstrap 95 % CI (10,000, seed 17), by era and pooled; in-sample R² with and without the 22-day MA; Spearman of each state variable with Y. Secondary, not gated: (i) upside vs downside — implied semivariance above/below the forward from the 0DTE smile vs realized up/down semivariance 10:30→16:00, premium per side with CI (the Almeida–Hizmeri side claim); (ii) realized skewness of 1-min returns by predicted-premium quintile (Vilkov); (iii) Fournier–Jacobs: Y_t vs the t−1 dealer-loss proxy (realized beyond implied in `bt_daily_outcomes`). |
| L6 | Pass: walk-forward OOS R² ≥ 0.05 AND Q5 − Q1 of realized Y < 0 with CI clear of zero pooled (richer predicted premium → realized falls further short of implied — the sign that pays a seller). Kill S3's gate (and, with CR-BM, S3 as designed): OOS R² < 0.02 AND Q5 − Q1 CI spans zero in both eras. Anything else: "not supported, not killed". The in-sample R² is reported against Bandi's 35 % but never gated. |
| L7 | Placebos: (i) state variables shuffled across days (seed 17); (ii) outcome shifted one day (Y_{t−1} on state_t) — contemporaneous fit is not prediction. |
| L8 | Output: parquet under `scripts/output/cr_bs/` (one row per day: state, Y on both clocks, predicted Y, quintile, side premia, skew) and a markdown report. Priors: Bandi 35 % in-sample at a fixed 10:30 with ψ ≈ 1.9; Almeida VRP⁺ > VRP⁻ at 0DTE; Vilkov "P&L is realized-skew driven"; Chong "fair net of costs"; CR-BM: neither wing earns at 09:33 held to close. |

### Readings fixed at the freeze (Code; mechanical from L1–L8, stated so the run is reviewable)

- **Sign of the "premium".** Y = log(realized / implied), so the variance premium a seller earns is −Y. "Q5 = richest predicted premium" = the quintile of **lowest predicted Y** (most negative). Q5 − Q1 is mean realized Y in Q5 minus mean realized Y in Q1; L6's pass sign (< 0) says the days predicted richest realize furthest below implied.
- **Quintile breakpoints in the walk-forward.** Each predicted Ŷ_t is ranked against the predictions made so far — the expanding set of out-of-sample predictions up to and including t, starting once 20 predictions exist, so no future prediction sets a breakpoint. The pooled-OOS-sample quintiles are printed beside it as a diagnostic.
- **OOS R².** 1 − Σ(Y − Ŷ)² / Σ(Y − Ȳ_expanding)², where Ȳ_expanding is the mean of Y over the same expanding fit window (the historical-mean benchmark, Campbell–Thompson). The OOS R² against the full-sample mean is printed beside it as a diagnostic.
- **Walk-forward start.** The first 250 universe days in date order form the first fit window; every later day is predicted from a fit on all earlier universe days. Refit daily.
- **Era tables.** The era of a day is set by t. Q5 − Q1 by era uses the walk-forward predictions inside that era. With a 250-day initial window most of E0 (≈ 10 months) falls inside the first fit window; the E0 OOS n is printed, and an E0 cell with n < 40 carries a "cell n < 40" note. The kill clause "CI spans zero in both eras" is evaluated literally on whatever n each era has.
- **Bootstrap.** Days are resampled i.i.d. within quintile (stratified on the realized quintile labels), 10,000 resamples, `numpy.random.default_rng(17)`; percentile CI. "Spans zero" is lower < 0 < upper.
- **22-day MA (Bandi footnote 17).** The source smooths the dependent variable and its vol-of-vol estimate β̃ with a 22-day moving average ("We use moving averages of estimates of αQt − αt and eβt with a window of 22 days", BFR 2024-03-15 draft p. 42 fn. 17). Secondary column (a), literal: trailing 22-universe-day MA of Y and of C (our vol-of-vol analogue), others raw. Column (b): MA22 on Y and all five regressors. Both in-sample OLS, never gated; overlapping MAs inflate R² mechanically, which the report states.
- **Reading of Bandi recorded at the freeze (from the source text, BFR draft 2024-03-15, Sec. 10, pp. 41–43).** Their "volatility premium" is αᵠ_t − α_t, the Q-drift minus the P-drift of *volatility* with α_t = (σ_{T′} − σ_t)/(T′ − t), T′ = 10:30 the next day — not implied minus realized variance over 10:30 → 16:00. "21.20 %" is the single-equation OLS of that premium on the leverage term and β̃² (two regressors), "34.81 %" is the GMM joint estimation of the return and volatility equations. The vol > 100 % exclusion applies to the *return*-premium regression (Table 5, ϕc), not the volatility one. None of this changes L1–L8 (the in-sample R² is never gated); it is carried to the reconciliation.
- **Implied semivariance (L5(i)).** On the 0DTE smile at 10:30: the delta grid (`vol5 … vol95`, call delta) is mapped to strikes by Black (F = `stock_price`, r = q = 0, τ = τ_rem, session clock), interpolated linearly in strike, flat beyond the grid; IV_up = 2∫_{F}^{∞} C(K)/K² dK, IV_down = 2∫_{0}^{F} P(K)/K² dK (the standard model-free semivariance split). Premium per side = log(RV_up / IV_up), log(RV_down / IV_down), RV_up/down = Σ r² over 1-min returns 10:30 → 16:00 with r > 0 / r < 0. The τ convention cancels to first order in the up/down comparison, not in each level; the report says so.
- **Dealer-loss proxy (L5(iii)).** `bt_daily_outcomes` realized-beyond-implied for session t−1, as defined by its stored columns; Step 0 names the columns and the frame. Not gated.

---

## Step 0 list (CR-BS)

1. L1 counts after each exclusion, by era.
2. The 07:30 PT `atmiv` / `vol50` ratio on 0DTE and 1DTE, by era. **If the E0 depression seen at 09:31 persists at 10:30, STOP and report — do not extend.**
3. The 0DTE dte-convention check with one worked day.
4. Confirm the `vol25` / `vol75` mapping.
5. Morning-RV availability (ES bars 09:30 → 10:30 ET).
6. The walk-forward window sizes.

If Step 0 requires a spec amendment, STOP and report. Otherwise continue without waiting.

## Implementation order

(1) spec freeze; (2) Step 0; (3) one commit for extraction; (4) one commit for statistics + report; (5) wrap.

Single run. Report pass/kill against L6 exactly as written, both placebos, the by-era tables, and the three secondaries. Then the Reconciliation protocol, all nine steps; STOP before any modification.

---

## Step 0 — diagnosis findings (2026-10-18)

Script `scripts/cr_bs_step0.py` (with `scripts/cr_bs_common.py`), SELECTs only, no run row, no API calls. Outputs in `scripts/output/cr_bs/`: `step0.md`, `step0_universe.csv`, `step0_gate.json`.

**Item 3 — dte convention.**
- ORATS stores the same-day expiry with **dte = 1**. There are 0 rows with dte = 0 in the window, and all 3,091 same-day PM rows carry dte = 1.
- `dte_pct` is 0.000 on same-day rows at 07:30 PT, so it is not a usable clock.
- Worked day 2025-03-04, 07:30 PT (quote 15:29:07Z): expir 2025-03-04 pm, dte 1, atmiv 0.4487, vol50 0.4303; next session 2025-03-05 pm, dte 2, atmiv 0.3306.
- L1(i)'s "dte = 0 (ORATS dte convention)" is implemented as `expir_date = trade_date`, PM. This is CR-BP's "smallest dte > 0" contract, so it confirms the convention and is not an amendment.

**Item 1 — L1 counts (E0 / E1 / total).**

| step | E0 | E1 | total |
|---|---|---|---|
| train sessions | 210 | 568 | 778 |
| after early closes | 208 | 562 | 770 |
| after roll weeks | 179 | 474 | 653 |
| snapshot present 07:30–07:33 PT | 177 | 473 | 650 |
| 0DTE PM smile | 176 | 473 | 649 |
| next-session PM atmiv | 169 | 459 | 628 |
| RTH ≥ 370 bars | 169 | 456 | **625** |

- Snapshot minute: 07:30 on 621 days.
- Third Fridays lost to the AM-only row: 0 (all fall in roll weeks or are otherwise excluded).
- Flags: FOMC 15 (E0 5, E1 10); 08:30 release days 108 (E0 33, E1 75).

**Item 2 — 10:30 ET era gate: PASS.** No captured quotes exist at 10:30 ET in train, so this is field vs field only.

| median | E0 | E1 |
|---|---|---|
| atmiv/vol50, 0DTE | 1.027 (IQR 1.004–1.050) | 1.051 (IQR 1.022–1.087) |
| atmiv/vol50, next session | 1.011 | 1.022 |
| vol50 10:30/10:50, 0DTE | 1.026 | 1.036 |
| atmiv 10:30/10:50, 0DTE | 1.014 | 1.034 |

- Monthly medians 2023-11 → 2024-06 run 1.021–1.055.
- The open-time E0 depression (CR-BP: vol50 09:33/09:50 E0 0.523) is absent at 10:30.

**Item 4 — sides.** `vol75` = 25Δ put, `vol25` = 25Δ call (`vol_features.py:21-22`, CR-BO/CR-BP). vol75 > vol25 on 97.8 % of days. K median: E0 0.085, E1 0.164.

**Item 5 — morning RV.** 60/60 morning bars on the median day; E1 minimum 46. 4 E1 days have < 57 morning bars, all kept.

**Item 6 — walk-forward sizes.**
- n 625.
- First fit window: days 1–250, **2023-05-01 → 2024-07-30**. First prediction 2024-07-31.
- **OOS n 375: E0 0, E1 375.**
- Expanding-rank quintiles: 356 ranked days.

### Gate: spec amendment required — STOP

The 250-day initial window covers all 169 E0 days. Three locked clauses are therefore not evaluable as written:
- L5's "Q5 − Q1 … by era";
- L6's kill clause, "Q5 − Q1 CI spans zero in both eras";
- L1's "E0/E1 mandatory in every table", for the walk-forward tables.

The freeze reading said the kill is "evaluated literally on whatever n each era has", but it assumed some E0 n. With E0 n = 0 there is no CI, so a literal reading can never meet the kill. That would turn the pre-registered test into pass-or-"not supported" by a window artifact, so it is not a mechanical reading.

Also recorded for the amendment and the reconciliation (from the source texts):
- Todorov–Zhang's estimator is QV⁽⁰⁾ / (QV⁽¹⁾ − QV⁽⁰⁾) in total variance (odp.pdf p. 7 eq. 8), not IV₀²/IV₁². L2(d)'s T is the cruder IV² ratio.
- Bandi2026 (p. 29) cites BFR as "Journal of Finance. Forthcoming" (2025). Our Sec. 10 numbers are from the 2024-03-15 draft.

---

## Amendment A1 — era clauses (Ryan, 2026-10-18; adopted after the Step 0 STOP, before any implementation run)

Adopted verbatim:

- **A1.1** The 10:30 era gate passed (atmiv/vol50 1.027 E0, 1.051 E1), so the era split is descriptive, not a gate. L6 kill clause becomes: "OOS R² < 0.02 AND Q5 − Q1 CI spans zero on the pooled walk-forward OOS set." Pass clause unchanged (pooled).
- **A1.2** Walk-forward first-fit window stays at 250 days as frozen (no tuning). The OOS set is therefore E1-only; state its n. E0 appears in the in-sample R² and the by-era descriptive tables only, labelled "in-sample (no OOS predictions by construction)".
- **A1.3** Add one descriptive column: the walk-forward OOS R² recomputed on the second half of the OOS set alone, so a drift in predictability across E1 is visible. Not gated.
- **A1.4** Record in field-conventions.md: ORATS dte = 1 denotes the same-day (0DTE) expiry in orats_monies_minute (confirmed CR-BS Step 0, consistent with CR-BP's note), and atmiv repeats across adjacent minutes 07:29–07:31 PT (update cadence coarser than one minute at that hour — measured, cause unknown).
- **A1.5** Nothing else changes. Proceed: commit the drafted extraction and stats scripts → single run → reconciliation (all nine steps; in the Bandi row note that his "volatility premium" is the Q-minus-P drift at next-day 10:30 with the 22-day MA on both the dependent variable and β̃, and that his 21.2 % / 34.8 % are single- vs joint-equation GMM — so our OOS R² and his in-sample are "same question, not the same statistic") → write-backs → verification agent → wrap → PR unmerged. STOP before any modification.

### A1 reading, recorded at adoption (Code)

- **Second half (A1.3).** The OOS set is split by date order at ⌊n/2⌋. The second-half OOS R² keeps the expanding-mean benchmark of each day, i.e. the same per-day terms restricted to the later half.
- **E0 descriptive tables (A1.2).** For each era: the in-sample OLS R² and the in-sample Q5 − Q1. The in-sample Q5 − Q1 uses fitted values from the full-sample OLS, cut into quintiles within the era. Every such cell is labelled "in-sample (no OOS predictions by construction)". The E1 OOS figures are the pooled OOS figures.
- **A1.4 measurement.** The extraction reports the share of universe days on which the 0DTE `atmiv` is identical at 07:30 and 07:31 PT, and at 07:30 → 07:33. 07:29 is outside the loaded window. Its one observed instance (2025-03-04, 0.448685 at 07:29, 07:30 and 07:31) comes from the Step 0 probe.
