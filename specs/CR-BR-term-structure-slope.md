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

---

## Amendment A1 (Ryan, 2026-10-18) — slope definition

Adopted verbatim:

| # | Amendment |
|---|---|
| A1.1 | L3(a) primary slope = PC2 of log IV² (x_t = log IV7², …, log IV90²), PCA fitted on train, sign fixed so positive = upward-sloping (back above front). Record the loadings and variance explained; the 2024-05-17 worked day must now pass the sign check — it is a gate, not a report line. |
| A1.2 | L3(b) s_t = IV30 − IV7 stays as secondary. The original PC2 of IV² levels is retained as a third, reported-only column labelled "frozen definition — failed sign check (Step 0)", with its +0.39 correlation to level and +0.19 to s_t stated. |
| A1.3 | Report Spearman of each slope definition against log IV7² and against the inversion indicator, so the reconciliation can say which object Johnson's SLOPE corresponds to here. |
| A1.4 | L6 pass/kill unchanged, applied to A1.1's quintiles on the level-neutral outcome. Everything else in the frozen spec unchanged: universe 621 / 515, L4 outcomes, L5 statistics, L7 placebos, priors. |
| A1.5 | Proceed: extraction → single run → reconciliation (all nine steps; note in the Johnson row that his PCA was on VIX-level curves and ours on log IV², and state whether that is "same object up to a monotone transform") → write-backs → verification agent → wrap → PR unmerged. STOP before any modification. |

**Operational definitions under A1.** These fill gaps A1 leaves open; they add no thresholds.

**Samples.**
- PCA and quintile cut points for every slope definition use all **778 x-complete train days**, as in Step 0: the slope state does not depend on ES.
- The h = 1 statistics use the 621-day universe; the h = 5 statistics use the 515-day universe.
- The "inversion indicator" is 1[IV90 < IV7].

**Sign-check gate (A1.1).** The extraction aborts unless both of these hold:
- PC2_log(2024-05-17) > 0;
- PC2_log(2025-04-08) < 0.

**Outcomes (L4).**
- **Level-neutral residual:** OLS of the outcome on [1, log IV7²], fitted separately per horizon on that horizon's universe.
- **h = 5 sum:** RV_{t+1..t+5} = Σ over the nights (t, t+1) … (t+4, t+5) of CR-BO's 13:00 → 13:00 PT RV. All five nights must pass `es_ok`.

**Bootstrap (L5).**
- **h = 1:** i.i.d. over days, 10,000 resamples, `default_rng(17)` per statistic.
- **h = 5:** moving-block bootstrap, 5-day blocks over the date-ordered universe, resampled within each quintile.
- **Q1 − Q5:** the two groups are resampled independently.

**Spearman CI.** The same bootstrap, applied to the (slope, outcome) pairs.

**Placebos (L7).**
- (i) One permutation (`default_rng(17)`) of the quintile labels across the universe days of each horizon.
- (ii) The outcome shifted one day earlier. The h = 1 outcome uses night (t−1, t), i.e. 13:00 PT t−1 → 13:00 PT t, normalised by IV7(t)²/365. That night must pass `es_ok`, and the day keeps its slope from t. This is contemporaneous, not predictive.
- (ii) at h = 5 uses nights (t−5 … t−1, t).

**By year.** Calendar years 2023, 2024, 2025 and 2026 H1, using the full-sample quintiles.

---

## A1 gate result (2026-10-18) — **STOP: the A1.1 sign-check gate FAILS**

`scripts/cr_br_extract.py` asserts the A1.1 gate before it loads any ES data or computes any outcome. It aborted:

| day | curve (IV 7 / 14 / 30 / 60 / 90 d) | PC2 of log IV² (centred) | required | percentile rank |
|---|---|---|---|---|
| 2024-05-17 | 8.86 / 8.48 / 9.87 / 10.45 / 10.94 (upward) | **−0.0324** | > 0 | 0.474 |
| 2025-04-08 | 61.18 / 51.01 / 42.37 / 35.45 / 32.25 (inverted by 28.9 pts, 90 d below 7 d) | −0.0492 | < 0 | 0.434 |

**Why it fails** (diagnostic; no outcome used).
- **The centring curve.** PCA scores are centred on the mean train curve. In vol points that curve is 12.84 / 13.02 / 13.29 / 13.60 / 13.81, already upward-sloping. A mildly upward day such as 2024-05-17 therefore scores below zero.
- **The level leak.** The PC2 loadings, −0.645 / −0.156 / +0.266 / +0.453 / +0.534, sum to **+0.452**, not 0. PC2 is orthogonal to PC1, and PC1 is front-heavy (+0.67 at 7 d), but PC2 is not orthogonal to a parallel shift. A high-vol day therefore gets a level push upward. That is why the sample's most inverted day, 2025-04-08, ranks mid-sample (0.43) rather than in Q1.
- **No sign meaning.** PC2 > 0 on 46.0 % of days, while 66.2 % of days slope upward (IV90 > IV7). The sign agrees with the slope direction on 66.5 % of days.
- **Not fixable within PCA.** The same mechanism made the frozen PC2 of IV² levels fail. On 2023–26, inversion is itself a vol-level event (s_t vs log IV7² −0.76), so a level-free *and* sign-correct centred PCA score does not exist on this sample.

**Monies-only alternatives, measured.**

| candidate | sign agrees with IV90 > IV7 | 2024-05-17 / 2025-04-08 | inverted share Q1 … Q5 | Spearman with log IV7² |
|---|---|---|---|---|
| (iii) s_t = IV30 − IV7 (pre-registered secondary) | sign-correct by construction | +1.01 / −18.81 | 97 / 60 / 10 / 2 / 0 % | −0.757 |
| (ii) row-demeaned log IV² curve projected on the PC2_log loadings (a loading-weighted slope; level removed per day) | 97.7 % | +0.367 / −1.003 | 100 / 69 / 0 / 0 / 0 % | −0.84 |
| (iv) PC2_log centred, gate dropped (sign = "relative to the mean curve") | 66.5 % | −0.032 / −0.049 | 67 / 45 / 33 / 16 / 8 % | −0.206 |

**For approval (Ryan): pick the primary slope.**
- (iii) Promote s_t to primary. It needs no new construction, but L6's kill clause then needs a second "definition".
- (ii) Use the weighted slope.
- (iv) Keep PC2_log with the gate replaced by a rank statement.

Every sign-correct option is strongly level-correlated, so L4's level-neutral outcome carries the Bekaert–Hoerova control under any choice. The run does not start until A1.1 is amended. The extraction script is committed with its gate unchanged.
