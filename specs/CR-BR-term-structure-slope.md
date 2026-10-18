# CR-BR — T-S4c: SPX IV term-structure slope at the close → next-day realized-vs-implied variance (read-only, pre-registered, train only)

> Authority: kickoff prompt, 2026-10-18 (Ryan) + vault `Dash/research/Test plan — lean sequence and reconciliation protocol.md` (T-S4c block and Reconciliation protocol, binding).
> Branch: `feat/CR-BR-term-structure-slope` off `origin/main` 88e2d6f (contains the CR-BQ merge).
> Data safety: **read-only**. No table writes, no cron, no API calls, no credentials. Train sessions t ≤ 2026-06-05 only; holdout never read. No open-time monies field is used — every implied input is the latest ≤ 12:46 PT snapshot (15:46 ET), the one CR-BL validated.
> Strategy: [[S4 - Vol-of-vol Switch]]. Constant-maturity method: CR-BL (`scripts/cr_bl_backfill.py` `total_variance_at`), reused, not re-derived. ES realized variance: CR-BO (`scripts/cr_bo_common.py`, `scripts/cr_bo_extract.py` `es_side` / `es_ok`).

## T-S4c block (verbatim from the Test plan)

### T-S4c — term-structure PC2, quotes-free

- **Data.** `orats_monies_minute` at 15:59: ATM IV for the 7, 14, 30, 60, 90-calendar-day interpolated expiries (or the listed ones nearest). ES next-day RV.
- **Compute.** Daily IV² vector across tenors → PCA on train → PC2 = slope. Quintiles of PC2. Next-day (and next-5-day) realized variance minus front implied variance by quintile (this is the quotes-free analogue of Johnson's straddle return).
- **Pass.** Monotone gradient across quintiles with top-minus-bottom CI clear of zero at 1 or 5 days, sign consistent with Johnson (low/inverted slope → long vol pays). **Kill S4's slope signal** (not S4): no gradient → drop (b) from the flip rule, keep vol-of-vol and the premium trajectory pending T-S4a.
- **Literature to reconcile against.** [[Johnson2017]] (0.29–1.81 % quintile spread; predicts straddle returns, *not* VIX changes), [[Cheng2019]], [[BekaertHoerova2014]], [[DewBecker2017]], [[Branger2018]], [[Vasquez2017]] (sign-convention warning in W3's contradictions), [[Mixon2007]].

## Locked decisions (kickoff, 2026-10-18)

| # | Decision |
|---|---|
| L1 | Universe: train sessions t ≤ 2026-06-05 with a ≤ 12:46 PT monies snapshot on t, PM expiries bracketing every tenor in L2, and roll-safe ES bars for t+1 (and t+1..t+5 for the 5-session read) per CR-BO's exclusion rules (roll weeks all years, holidays, early closes, ≥ 370 RTH minutes). Report counts after each exclusion. |
| L2 | Tenors: constant-maturity ATM IV at 7, 14, 30, 60, 90 calendar days by the CR-BL method. Vector x_t = (IV7², IV14², IV30², IV60², IV90²). |
| L3 | Slope, two definitions, both pre-registered: (a) PC2 of x_t from a PCA fitted on the train sample, with the sign fixed so that positive PC2 = back-end above front (upward-sloping); (b) simple slope s_t = IV30 − IV7 in vol points. (a) is primary (Johnson's object); (b) is reported beside it. Quintiles of each on the train sample (Q1 = most inverted). |
| L4 | Outcome: the quotes-free analogue of Johnson's straddle return. h = 1: log(RV_{t+1} / IV7²·(1/365)) where RV_{t+1} is the roll-safe realized variance of ES log returns from 13:00 PT on t to 13:00 PT on t+1 (overnight + RTH), total variance, no annualization on either side — the same construction CR-BO/CR-BP used. h = 5: log(ΣRV_{t+1..t+5} / IV7²·(5/365)). Report also the level-neutral version: the residual of the outcome regressed on log IV7² (so the slope effect is not a vol-level effect in disguise — the Bekaert–Hoerova point). |
| L5 | Statistics: mean outcome per quintile with bootstrap 95 % CI over days (10,000 resamples, seed 17); Q1 − Q5 difference with CI; Spearman of slope vs outcome; all at h = 1 and h = 5; all for raw and level-neutral outcomes; by year as a diagnostic. Overlapping h = 5 windows: block bootstrap with 5-day blocks. |
| L6 | Pass: Q1 − Q5 > 0 with CI clear of zero at h = 1 or h = 5 on the level-neutral outcome (inverted slope → realized exceeds implied more, i.e. long vol pays — Johnson's sign). Kill S4's slope signal (not S4): Q1 − Q5 CI spans zero at both horizons on both slope definitions. Anything else: "not supported, not killed". Secondary, not gated: the same for the raw outcome and for s_t. |
| L7 | Placebo: slope labels shuffled across days (seed 17); the quintile gradient should vanish. Second placebo: the same test with the outcome shifted one day earlier (outcome for t, slope from t) — a contemporaneous correlation is not predictability. |
| L8 | Output: parquet under scripts/output/cr_br/ (one row per day: tenors, PC2, s_t, quintiles, outcomes) and a markdown report. Priors: Johnson's quintile spread 0.29–1.81 % in straddle returns, predictive of straddle/variance returns but NOT of VIX changes; Dew-Becker: the premium lives ≤ 2 months; Cheng: the ex-ante premium falls before spikes; Vasquez/Jones–Wang use the opposite sign convention — state ours once and never flip it. |

**Sign convention (stated once, L8):** positive PC2 and positive s_t = upward-sloping term structure (back above front). Q1 = most inverted (lowest PC2 / s_t), Q5 = steepest upward. Johnson's sign = Q1 − Q5 > 0 on the outcome (realized exceeds implied more after an inverted curve).

## Sequence

1. Spec freeze (this commit).
2. Step 0 diagnosis (next commit):
   - L1 counts;
   - the ≤ 12:46 PT atmiv vs vol50 ratio by era (E0 < 2024-03-01, E1 ≥) at the close for the 7- and 30-day tenors — **if the CR-BP open-time era break is present at the close too, STOP and report**;
   - PCA loadings and variance explained on train;
   - the PC2 sign check (one worked day);
   - confirm every tenor is bracketed by PM expiries on ≥ 95 % of days.

   **If Step 0 requires a spec amendment, STOP and report.** Otherwise continue without waiting.
3. One commit for extraction.
4. One commit for statistics + report.
5. Wrap. Single run. Report pass/kill against L6 exactly as written, both placebos, and the by-year table. Then the Reconciliation protocol, all nine steps.
