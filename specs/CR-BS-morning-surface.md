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
