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

---

## Step 0 — diagnosis findings (2026-10-18) — **STOP: L3(a) needs an amendment**

Script `scripts/cr_br_step0.py` (read-only; no outcome used anywhere); full output `scripts/output/cr_br/step0.md`. Shared definitions `scripts/cr_br_common.py`, which reuses CR-BL's `total_variance_at` and CR-BO's `es_side` / `es_ok` / `detect_switches` unchanged.

**Q3 — bracketing.** All five tenors are bracketed by PM expiries on **778 of 778** train sessions (100 %).
- 7 d, 14 d and 30 d mostly sit on an exact node (718 / 714 / 415 days).
- The 60 d and 90 d brackets have median widths of 21 and 31 days.
- The longest PM node is ≥ 283 d on every day.
- Snapshot: 12:46 PT on 765 days. The others are early-close days (10:06–10:15 PT), all removed by L1.

**Q2 — era gate (close-time atmiv vs vol50, constant maturity).** **PASS**: there is no close-time era break.

| tenor | E0 (< 2024-03-01, n 210) | E1 (≥ 2024-03-01, n 568) | E1 − E0 |
|---|---|---|---|
| 7 d | 1.0011 (IQR 0.996–1.008) | 1.0110 (IQR 1.000–1.025) | +0.0098 |
| 30 d | 0.9967 (IQR 0.993–1.001) | 1.0074 (IQR 0.999–1.017) | +0.0106 |

- Every value is inside CR-BP's 0.95–1.05 band. Monthly medians around 2024-03 are 0.997–1.024. Compare CR-BP's open-time ratios of 0.08 → 1.16.
- This is a new field-conventions row (to be added at wrap).

**Q1 — L1 counts.**
- 778 train sessions.
- → 778 with a ≤ 12:46 PT snapshot, and 778 with all tenors bracketed.
- → 777 with t+1 ≤ 2026-06-05 (holdout never read).
- → 761 after early closes.
- → **632** after roll weeks (all years).
- → 632 after contract switches (all 12 detected switches fall in roll weeks).
- → 631 after the ES anchor bar.
- → 623 after the overnight-minutes rule (≤ 5 % missing).
- → **621** after the RTH-minutes rule (≤ 5 % missing on t+1, so ≥ 371 of 390, which implies ≥ 370; minimum seen 376).
- **h = 5** (all five nights t … t+4 pass and t+5 ≤ split): **515**.
- By year (h = 1 / h = 5): 2023 131 / 104; 2024 201 / 170; 2025 199 / 167; 2026 90 / 74.
- The PCA and quintile sample is every x-complete train day (778), since the slope state does not depend on ES.

**Q4 — PCA of x_t = IV² levels (L3(a) as written).**
- Variance explained: PC1 96.94 %, PC2 **2.75 %**, PC3 0.22 %.
- PC2 loadings (7 / 14 / 30 / 60 / 90 d): −0.589 / −0.005 / +0.343 / +0.490 / +0.544. The sign is fixed so the 90 d loading exceeds the 7 d loading. The shape is close to Johnson's (≈ +0.60 at 1 m to −0.54 at 12 m, about 4 %; opposite sign convention).
- **Worked days:**
  - 2025-04-08 (IV 61.2 / 51.0 / 42.4 / 35.5 / 32.3) has PC2 −0.0574: inverted and negative, correct.
  - **2024-05-17 (IV 8.86 / 8.48 / 9.87 / 10.45 / 10.94, upward-sloping) has PC2 −0.0049: negative.**
  - The score measures slope *relative to the mean curve*, and in IV² levels it is mixed with the vol level.

**Q5 — what each definition sorts on. This is why Step 0 stops.**

| definition | share of days with IV90 < IV7, Q1 … Q5 | Spearman with log IV7² | Spearman with IV90 − IV7 |
|---|---|---|---|
| (a) PC2 of IV², L3 as written | 40 / 26 / 33 / 34 / 35 % | **+0.391** | +0.155 |
| (b) s_t = IV30 − IV7 | 97 / 60 / 10 / 2 / 0 % | **−0.757** | +0.934 |
| (c) candidate: PC2 of log IV² | **67 / 45 / 33 / 16 / 8 %** | −0.206 | +0.688 |
| (d) candidate: PC2 of IV² / row mean (shape only) | 35 / 32 / 35 / 34 / 33 % | +0.012 | +0.054 |

- **(a) is not a slope on this sample.** Its quintiles do not separate inverted from upward curves, and Q5 holds the *higher*-vol days (median IV7 14.3 vs 11.0 in Q1). L3's "positive PC2 = back-end above front" and "Q1 = most inverted" are false for (a)'s scores, so Johnson's sign cannot be read off it, and L6's pass clause loses its meaning.
- **(b)** sorts on inversion cleanly but is mostly a vol-level variable, which is why L4's level-neutral outcome matters.
- **(c)** sorts monotonically on inversion with a weak level link. Its PC2 explains 3.56 %, with loadings −0.645 / −0.156 / +0.266 / +0.453 / +0.534.
- **(d)** is a curvature factor, not slope.

**For approval (Ryan): choose L3(a).**
- (i) Keep PC2 of IV² as written, knowing Q1 ≠ inverted.
- (ii) Replace it with PC2 of log IV² (c). This is the recommended option: it is Johnson's object on a log scale, the only PCA variant that ranks inversion, and its sign convention holds.
- (iii) Make s_t primary.

L6 and everything else are unchanged under any choice. The run does not start until L3(a) is amended.
