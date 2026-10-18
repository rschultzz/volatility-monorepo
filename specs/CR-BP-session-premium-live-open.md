# CR-BP — T-S1a′: session premium with the open-side implied variance adjudicated against captured option prices

Read-only, pre-registered, train only (t+1 ≤ 2026-06-05). Single run. Branch `feat/CR-BP-session-premium-live-open` off `origin/main` `08c1a20` (the CR-BO merge).

Authority: vault `Dash/research/T-S1a-prime — draft pre-registration.md` (copied verbatim below, from **Why** through the Step 0 list, incl. P3(g) adopted 2026-10-18 before this freeze); `Dash/research/Test plan — lean sequence and reconciliation protocol.md` (T-S1a block and the Reconciliation protocol are binding); `Dash/research/strategies-proposed/S1 - Night Shift.md`.

No API calls, no credentials touched, no table writes, no cron. Holdout (> 2026-06-05) never read; DAILY-CAPTURE / CR-AU rows excluded from P3 and P4 entirely (C2).

**Why.** CR-BO passed L8 as registered (S1 P_on +0.260 [+0.188, +0.355] var pts; P_rth −0.311 [−0.453, −0.226]). After the run, `atmiv` and `vol50` were found to disagree at 09:31 ET on short expiries:

| calendar DTE at the open | 0 | 1 | 2–3 | 4–7 | 8–15 |
|---|---|---|---|---|---|
| median atmiv / vol50 | 0.706 | 0.904 | 0.947 | 0.996 | 0.995 |

By 09:50 the ratio is ≈ 1.0 everywhere. Which field matches traded ATM IV is unmeasured, and so is the cause. CR-037's Background item 3 had already shown that the 06:33 PT `atmiv` is not the prior-close value. The S1 and S2 open snapshots depend on the answer; S7 (≥ 4 DTE at the open) does not.

**Hypothesis under test (P3):** at the open, `atmiv` ≠ traded ATM IV. Direction and cause unstated.

---

## T-S1a block (verbatim from the test plan, as frozen in CR-BO)

### T-S1a — session premium

- **Data.** `orats_monies_minute`: per expiry, ATM IV (`atmiv`) at 15:59 and next session 09:31; expiries 0/1/2 DTE (calendar DTE as listed) and 7-day. ES 1-min from `es_minutes`, roll-safe (exclude roll weeks per the Databento caveat; use `trading_calendar`). Train 2023-05 → 2026-06-05.
- **Compute.** Per night: implied variance at 15:59 = atmiv² × T (T in years to expiry); implied at 09:31 next day likewise; implied-variance *drop* overnight = IV₁₅:₅₉² T₁ − IV₀₉:₃₁² T₂. Realized overnight variance = Σ 1-min log returns² from 16:00 to 09:30 ET (continuous ES). Session premium = implied drop − realized overnight variance, in variance units and as a ratio. Also realized RTH variance and the implied RTH drop for the complementary test (Muravyev–Ni predicts the sign flips).
- **Primary statistic.** Median and mean session premium with bootstrap CI, by (a) weekday (Fri→Mon separately), (b) DTE bucket, (c) event nights vs non-event (hand-code FOMC/CPI/NFP dates for the train window in the spec — ~90 dates, no table needed).
- **Pass.** Overnight premium > 0 with CI clear of zero on non-event weeknights AND the intraday premium is ≤ 0 or indistinguishable from zero (the sign flip is the thesis). **Kill S1.** Overnight premium CI spans zero on non-event nights, or the intraday premium is as large as the overnight one (then there is no session asymmetry to trade).
- **Literature to reconcile against.** [[MuravyevNi2020]] (−1.0 %/day overnight vs +0.3 % intraday; < 3-week −2.6 % vs ≈ 0; the calendar-time mechanism), [[Terstegge2025]] (puts −2.49 % / +0.39 %; calls flat — so also split the premium by put-side vs call-side IV if `vol25`/`vol75` make that possible), [[JonesShemesh2018]] (weekend; weak for SPX index options in their WP), [[OptionMetrics2025]] (Mondays ≈ ⅔ post-2022), [[Gatheral2006]] ch. 5 (≈ 40 % overnight), [[Bennett2014]] (≈ 1/6), [[Augen2008]] ch. 8 (expiry-week decay), [[Chong2024]] (fair net of costs), [[LouPolkSkouras2019]].

---

## Locked block — the T-S1a′ change (P1–P8, as amended by C1–C5)

| # | Decision |
|---|---|
| P1 | **(amended by C1, C3, C4)** Open-side implied variance at 09:31 ET (v_b) = **V_px** (P3) on decision-subset days. On the remaining days, v_b = field²τ, using the monies field that P3's decision rule extends (`atmiv` or `vol50`; τ as in CR-BO L3). If neither field qualifies, P1 runs on the decision-subset days only and the report says so. The close side (v_a at 15:59 on t, v_c at 15:59 on t+1) stays `atmiv`²τ as in CR-BO, subject to P3(e). |
| P2 | Secondary read: `atmiv` at 09:50 ET, reported alongside P1 as a cross-check; the gated statistic is P1. (RV windows for this read: C6.) |
| P3 | **(amended by C1, C2, C4)** Step 0 gate — adjudicate the open-side implied variance against captured option prices. Parts (a)–(g), then the decision rule. |
| P4 | **(amended by C5)** The L5 straddle-equivalent in index points is mandatory per bucket. No 1-DTE train quotes exist, so the report gives the captured CR-AS 0DTE leg bid–ask widths at 09:31–09:33 ET as a **lower-bound proxy**, labelled as such. The real cost test is Gate B after quote capture. Nothing further is estimated. |
| P5 | L9 placebo interpretation: retaining most of the premium under a one-night shift (0.262 → 0.203 in CR-BO) is consistent with Muravyev–Ni's calendar-time mechanism (structural, not night-specific) and is **not** a kill criterion; L8 as frozen remains the only kill rule. |
| P6 | L8 unchanged, applied to the P1 read. **Prior** from CR-BO's rough correction (v_b scaled by the median atmiv/vol50 factor at 09:31, i.e. as if `vol50` were right): **S1 ≈ +0.05, S2 ≈ 0, S7 ≈ +0.14** var pts. The result is judged against these, not against CR-BO's +0.26. |
| P7 | Same universe, exclusions, seed, event list, splits and reconciliation source list as CR-BO, plus the CR-BO and CR-037 session notes as Dash priors. Read-only. |
| P8 | Write-back: P3's result closes or amends [[implied-move-1d-is-atm-iv-not-open-straddle]] with a measurement attached, whichever way it goes. |

### P3 parts (a)–(g)

**(a) Data (C2).** CR-AS 0DTE captures in `orats_options_minute`, train only (≤ 2026-06-05). **DAILY-CAPTURE / CR-AU rows are holdout (all ≥ 2026-09-08) and are excluded from P3 and P4 entirely.**

**(b) Primary subset (C1).** Train days with a captured CR-AS 0DTE call **and** put at the same strike, within 10 pts of the 06:33 spot (the ~196-pair set found in CR-BO Step 0 Q3); n is reported.
- S = the 06:33 PT spot CR-AS used for strike placement. Step 0 reads it from the CR-AS run record / code and states the field.
- Straddle mid at minute m = call mid + put mid, each (bid_price + ask_price)/2 from the captured legs at that minute.
- **V_px** is the remaining-life implied variance in that price. Invert straddle = 2S[2N(√v / 2) − 1] for v = σ²τ exactly, with no 0.8 approximation. In closed form: v = (2·N⁻¹((1 + straddle/(2S)) / 2))². *For P1 this closed form is superseded by (g), the K-aware solve; it is kept as a cross-check column.*
- V_px needs no τ convention.

**(c) Secondary set (C1).** All CR-AS capture days: the ±0.5 IM short put (K_p) and short call (K_c) taken together as an equidistant strangle.
- Use its **price, not its IVs**.
- V_px(strangle) = the total variance v that solves Black(F = S, r = 0) put(K_p, v) + call(K_c, v) = strangle mid at minute m. One unknown, no τ convention.
- Labelled in every table as wing-strike implied variance: it carries the smile at ±0.5 IM and is not an ATM quantity.

**(d) Ratios.** For minutes 09:31, 09:33, 09:35 and 09:45 ET, the median and IQR of atmiv²τ / V_px and vol50²τ / V_px, with τ per CR-BO L3 and the monies row for the captured `expir_date` at that minute.
- Computed by bucket and by set (primary, secondary).
- Only 0DTE contracts are captured, i.e. S1's open-side contract. S2's and S7's open-side contracts are not captured, so the field chosen for them rests on the 0DTE result, and the report states this.
- `mid_iv` (mean of the call's and put's) enters as mid_iv²τ / V_px, a **diagnostic column only, never used for P1**.

**(e) Close-side check (C3).** No 1–3 DTE pairs are captured, so the close side stays `atmiv`. Monies-only consistency check: median atmiv/vol50 at 15:59 ET by bucket over the universe. If a bucket's ratio is outside 0.95–1.05, its close side is flagged **"unadjudicated"** in the report and in the reconciliation table.

**(f) `atmiv` minute-to-minute.** The share of days on which `atmiv` changes at each minute step 09:31 → 09:40, for 0DTE and for the S1/S2 open-side contracts; the first-change minute (median, IQR).

**(g) V_px on the primary subset — K-aware solve (amendment, Ryan, 2026-10-18, adopted before the spec freeze).**
- On the primary subset, V_px is computed by the one-unknown Black–Scholes solve for σ²τ, using:
  - the captured pair's actual strike K;
  - S = the 06:33 placement spot;
  - r and q as the harness uses them;
  - the same solver as the secondary strangle.
  In symbols: call(K, v) + put(K, v) = straddle mid.
- The closed form v = (2·N⁻¹((1 + straddle/(2S))/2))² is kept only as a cross-check column. It is exact at K = S and is not used for P1.
- Report the median |K − S| on the primary subset, so the residual moneyness effect is visible.
- This supersedes the closed-form wording in (b) for P1 purposes; no τ convention enters V_px anywhere.

**Decision rule, written before running.**
1. **Decision subset:** the primary subset (b) if n ≥ 100, else the secondary set (c).
2. P1 open-side v_b = V_px on decision-subset days.
3. The monies field whose median ratio to V_px is within 10 % of 1.0 at 09:33 ET extends P1 to the remaining days. If both qualify, `atmiv` (continuity with CR-BO). If neither qualifies, P1 runs on the decision-subset days only, and the report says so.
4. `mid_iv` plays no part in the decision.

### How P1 maps onto the CR-BO rows

In L3, L4 and L6(d), "atmiv²τ at 09:31 ET" is read as "the P1 open-side variance v_b". "atmiv²τ at 15:59" is unchanged (C3).
- On decision-subset nights, the nights whose t+1 is a decision-subset day, v_b = V_px of t+1's 0DTE contract at 09:31. That is S1's open-side contract.
- S2 and S7 have no captured contract on those days, so their v_b uses the extended field. If no field is extended, S2 and S7 drop out, and the report says so.
- L6(d)'s side split keeps `vol75` / `vol25` at both snapshots, unchanged from CR-BO. It is not gated (fixed readings), and V_px has no put/call side.
- L10's output directory becomes `scripts/output/cr_bp/`. No other word of L1–L10 changes.

### Superseded P-text (kept for the record, not in force)

- **Original P1:** "Open-side implied variance at 09:31 ET uses whichever monies field P3 shows matches traded ATM IV (atmiv or vol50). Close-side (15:59) likewise per P3's 15:59 check. Not chosen in advance." Replaced by C1/C3/C4: price-based V_px on decision-subset days; close side `atmiv` with a consistency flag.
- **Original P3:** "… take the ATM call and put (strike nearest the 06:33 spot) from orats_options_minute at 09:31–09:33 ET and use ORATS' per-contract mid_iv (no Black–Scholes back-out); compare to atmiv and vol50 at the same minute. … Decision rule …: P1 uses the field whose median ratio is within 5 % of 1.0 by 09:33; if both are, atmiv (continuity with CR-BO); if neither is, P1 uses mid_iv directly on capture days …" Replaced by C4. The IV-vs-IV comparison becomes price-based, `mid_iv` is diagnostic only, and the tolerance is 10 % against V_px. C1 adds the subsets and C2 removes DAILY-CAPTURE.
- **Original P4:** "… set against a round-trip cost from captured 1-DTE put-spread leg quotes (DAILY-CAPTURE / CR-AU rows) where they exist; if none exist for a bucket, say so — do not estimate." Replaced by C5 and C2.

---

## Locked decisions carried from CR-BO (verbatim)

| # | Decision |
|---|---|
| L1 | Universe: every train session pair (t, t+1) with t+1 ≤ 2026-06-05 where `orats_monies_minute` has a 15:59 ET row on t and a 09:31 ET row on t+1 for the same expiry, and ES 1-min bars cover 16:00 ET on t → 09:30 ET on t+1 with ≤ 5 % missing minutes. Exclude ES roll weeks (the Databento `ES.c.0` dying-contract caveat) and the 2026-09 patched window if it falls in range (it does not), holidays and early closes per `trading_calendar`. **Addition:** exclude every night that spans a contract switch in the ES continuous series, all years 2023–2026, plus the 2025 `ES.c.0` dying-contract roll weeks. Step 0 lists the roll dates from `trading_calendar` / the contract schedule and the count of nights removed. |
| L2 | Contracts bucketed by SESSIONS to expiry at 15:59 ET on t, not calendar days: S1 = the expiry on the next trading session (Monday's on a Friday; after a holiday, the next session's), S2 = the expiry two sessions out, S7 = the nearest listed expiry ≥ 7 calendar days out. PM-settled only (`expiry_tod = 'pm'`), identified by `expirDate`, same `expirDate` at 09:31 on t+1. Record calendar days to expiry as a column so the Friday 3-day gap is visible in the splits. 0DTE is excluded because at 15:59 it has one minute of life and no overnight leg. |
| L3 | Implied variance = atmiv² × τ, τ = time from the snapshot to 16:00 ET on `expirDate` in years, using the same day-count ORATS uses for `atmiv` (confirm the convention in Step 0 and record it; if ambiguous, use calendar days/365 and state so). Implied overnight drop D_on = IV²τ\|15:59,t − IV²τ\|09:31,t+1. Implied intraday drop D_rth = IV²τ\|09:31,t+1 − IV²τ\|15:59,t+1 for the same contract. |
| L4 | Total variance over the window, no annualization on either side: D_on = atmiv²τ\|15:59,t − atmiv²τ\|09:31,t+1 and RV_on = Σ r² over the same clock window are directly comparable as they stand. Same for the RTH leg. Realized from ES 1-min log returns (basis cancels in log returns; no SPX conversion): RV_on = Σ r² over 16:00 ET t → 09:30 ET t+1 (including the 16:00–17:00 and 18:00+ sessions; the 17:00–18:00 halt contributes nothing); RV_rth = Σ r² over 09:30 → 16:00 ET t+1. Realized overnight share series: variance share RV_on/(RV_on+RV_rth) compared to Gatheral's 40 %; also the volatility-share form √RV_on/(√RV_on+√RV_rth) for Bennett's one-sixth — report both, never mix them. |
| L5 | Session premium P_on = D_on − RV_on; P_rth = D_rth − RV_rth. Primary units: variance points; secondary: the ratio D/RV. Also the ATM-straddle-equivalent in index points for readability (stock_price × √variance; `stock_price`, not `spot_price`, with the 2023 jitter caveat noted). |
| L6 | Splits: (a) weekday of t, Fri→Mon separate; (b) expiry bucket S1/S2/S7; (c) event nights vs not — event night = a CPI, PPI, NFP or PCE release at 08:30 ET on t+1; FOMC statement day on t+1 flagged separately as an RTH-leg event. Hand-code the dates for 2023-05 → 2026-06 from the BLS/BEA/Fed public schedules into the spec as a list (≈ 100 dates) with the source URL for each schedule; no table. (d) Terstegge split: put-side and call-side using the 25-delta put vol and 25-delta call vol — confirm in Step 0 which of `vol25`/`vol75` is which under ORATS' convention and record it; compute D_on and the premium per side with the same τ. |
| L7 | Statistics: median and mean of P_on and P_rth with bootstrap 95 % CI (10,000 resamples, seed 17), per split; sign test on P_on > 0 per night; the Muravyev–Ni contrast mean(P_on) − mean(P_rth) with CI, per bucket. |
| L8 | Pass: on non-event weeknights, P_on > 0 with CI clear of zero in bucket S1 AND P_rth ≤ 0 or CI spanning zero (the sign flip is the thesis). Kill S1: P_on CI spans zero on non-event weeknights in every bucket, OR P_rth is as large as P_on (no session asymmetry). Anything else: "not supported, not killed". |
| L9 | Pre-registered placebo: pair D_on on t with RV_on from the following night (t+1 → t+2); the premium should collapse toward the unconditional mean difference. Report alongside the main read. (Interpretation: P5.) |
| L10 | Output: parquet under `scripts/output/cr_bp/` (one row per night × bucket × side) and a markdown report. Read-only; holdout never read; no API calls. |

### Readings fixed before the run (verbatim from the `scripts/cr_bo_stats.py` header; reused unchanged)

- **Units.** Variance points = total variance × 10⁴ (i.e. %²). Ratio D/RV = ratio of means (mean D / mean RV); the median of the per-night ratio is printed alongside.
- **Primary side for L8 = atm.** In CR-BO this was `atmiv`; here it is **the P1 variance**. put / call (L6(d)) are reported, not gated.
- **"Non-event weeknight"** = t Mon–Thu with t+1 the next calendar day (cal_gap 1; this drops Fri→Mon and holiday-spanning nights) AND no CPI / PPI / NFP / PCE at 08:30 ET on t+1. FOMC on t+1 stays in (L6(c) makes it an RTH-leg event, not an event night); it is also reported as its own split.
- **L8 statistic = the MEAN**, 95 % percentile bootstrap CI (10,000 resamples, seed 17); medians reported.
  - Pass: S1 mean P_on CI lower bound > 0 AND (S1 mean P_rth ≤ 0 OR its CI contains 0).
  - Kill: (mean P_on CI contains 0 in every bucket S1, S2, S7) OR (S1 mean P_rth ≥ S1 mean P_on).
  - Otherwise "not supported, not killed". "Spans zero" is read literally (lower < 0 < upper).
- **Sign test.** Two-sided exact binomial on #(P_on > 0) vs n, p0 = 0.5.
- **Muravyev–Ni contrast.** Per-night paired P_on − P_rth, mean with bootstrap CI, per bucket.
- **Shares (L4).** Per night v_share = RV_on/(RV_on+RV_rth) and vol_share = √RV_on/(√RV_on+√RV_rth), over every distinct night in the universe; mean and median with bootstrap CI of the mean, plus the pooled ΣRV_on/Σ(RV_on+RV_rth). Benchmarks: Gatheral 40 % (variance share), Bennett 1/6 (volatility share).
- **L9 placebo.** P_on_pl = D_on(t) − RV_on(t+1 → t+2), on nights where the next night passes the ES-side exclusions; the main premium is recomputed on the same nights for comparison, with corr(D_on, RV_on) vs corr(D_on, RV_on_next).
- **Readability.** stock_price(15:59, t) × sign(v)·√|v| in index points, for mean D_on, mean RV_on, mean P_on. P4 makes this mandatory per bucket, beside the 0DTE bid–ask lower-bound proxy.

### Carried from CR-BO Step 0 (definitions reused, not re-derived)

- **Shared definitions.** `scripts/cr_bo_common.py` holds the event dates and source URLs, early closes, roll weeks (Monday of the week before the 3rd Friday → expiry Friday), the Globex grid, and τ = (16:00 ET on expir_date − snapshot label) / 365 days.
- **CR-BO universe for reference.** 621 ES-clean nights; S1 500, S2 498, S7 517.
- **Events.** 141 dates of 08:30 releases (CPI 36, PPI 36, NFP 37, PCE 32), plus 25 FOMC days. The PCE releases of 2024-11-27, 2025-04-30, 2025-12-05 and 2026-01-22 came at 10:00 and are not event nights.
- **Sides.** `vol75` = put side, `vol25` = call side.
- **Universe gate.** The extraction universe must equal the Step 0 universe; the gate is the same as CR-BO's.

---

## Resolved choices (Ryan, 2026-10-18)

| # | Resolution |
|---|---|
| C1 | **Primary P3 / P1 subset.** Train days with a CR-AS 0DTE call AND put whose strike is within 10 pts of the 06:33 spot (the ~196-pair set), n reported. **Secondary,** all CR-AS capture days: the ±0.5 IM short put and short call together as an equidistant strangle, using its price, not its IVs. Report both. The decision rule runs on the primary subset if n ≥ 100, else on the secondary. |
| C2 | DAILY-CAPTURE rows are holdout: excluded from P3 and P4 entirely. P3 / P4 use CR-AS captures only. |
| C3 | 15:59 side: no captured 1–3 DTE pairs, so the close side stays `atmiv` as in CR-BO. A monies-only consistency check (atmiv vs vol50 at 15:59, median ratio by bucket) is reported; outside 0.95–1.05 the close side is flagged "unadjudicated" in the report and the reconciliation table. |
| C4 | The IV-vs-IV comparison is replaced by a price-based one. V_px is the straddle-implied variance (exact inversion, S = the 06:33 placement spot, no τ convention). atmiv²τ and vol50²τ are compared to V_px at 09:31 / 09:33 / 09:35 / 09:45. The decision rule is as in P3. `mid_iv` is diagnostic only. |
| C5 | P4 cost reference: no 1-DTE train quotes exist. The report gives the CR-AS 0DTE leg bid–ask widths at 09:31–09:33 as a lower-bound proxy, labelled as such; the real cost test is Gate B. No further estimate. |
| C6 | The two remaining choices keep the drafted defaults (no conflict with P1–P8 or C1–C5). They are listed in Step 0 item 11 so they appear in the report. |

---

## Step 0 list (CR-BP)

CR-BO's Q1–Q7 are re-run with the same script logic and must reproduce CR-BO's counts exactly; the gate is an identical `step0_pairs.csv`. Then P3 and the items below. **STOP after Step 0 only if a spec amendment is required. P3's decision is mechanical and is not an amendment.**

1. **Q1–Q7 (CR-BO, unchanged).** The coverage table after each exclusion, contract switches and roll-week nights, τ convention, `vol25`/`vol75`, ES missing minutes by year, weekday S1 coverage, event list. Confirm that the CR-BO counts reproduce: 777 → 761 → 632 → 632 → 631 → 623 → 621; S1 500 / S2 498 / S7 517.
2. **P3(a)–(b), (g) — capture coverage and the primary subset** (incl. median |K − S|).
   - State the CR-AS placement-spot field S and where it was read (run record / code).
   - List the train CR-AS capture days; DAILY-CAPTURE is not read (C2).
   - Primary subset: days with a same-strike call AND put within 10 pts of S, both with a valid bid/ask at 06:31, 06:33, 06:35 and 06:45 PT (the CR-AN quote-sanity rules). Report n, the |K − S| distribution, and how many fall in the CR-BO universe as t+1.
3. **P3(c) — the secondary set.** All CR-AS capture days with both ±0.5 IM short legs valid at those minutes: n, the K_p / K_c distances from S, and how many fall in the CR-BO universe as t+1.
4. **P3(d), (g) — ratios at the open.** At 09:31, 09:33, 09:35 and 09:45 ET, the median and IQR of atmiv²τ / V_px and vol50²τ / V_px (V_px per (g) on the primary subset), with mid_iv²τ / V_px and closed-form-V_px / V_px as diagnostic columns. Each is given for the primary subset and the secondary set, by bucket (0DTE only is captured), with n per cell.
5. **P3(e) — close-side check.** Median atmiv/vol50 at 15:59 ET by bucket (S1, S2, S7 contracts on t, and the t+1 15:59 rows used for v_c) over the CR-BO universe. Flag any bucket outside 0.95–1.05 as "unadjudicated".
6. **P3(f) — `atmiv` minute-to-minute.** For 0DTE and for the S1/S2 open-side contracts: the share of days `atmiv` changes at each minute step 09:31 → 09:40, and the first-change minute (median, IQR).
7. **`vol50` minute-to-minute on five random days.** Days drawn with `numpy.random.default_rng(17)` from the CR-BO universe. Print `vol50` and `atmiv` per minute 09:30–09:50 and 15:50–15:59 for the S1, S2 and S7 contracts of each day.
8. **P3 decision line.** Apply the rule exactly as written and print:
   - which subset decided (primary or secondary) and why (n vs 100);
   - the 09:33 ratios that decided the extension field, or that neither qualified;
   - the resulting P1 coverage by bucket.
9. **P4 — cost proxy.** The CR-AS 0DTE leg bid–ask widths at 09:31–09:33 ET (median, IQR, in points per leg and per two-leg straddle / strangle), labelled "lower-bound proxy; real cost test is Gate B". State that no 1-DTE train quotes exist.
10. **Prior check (P6).** Print CR-BO's per-bucket P_on next to the P6 priors (S1 ≈ +0.05, S2 ≈ 0, S7 ≈ +0.14), so the run's report sets its result against them.
11. **Remaining defaults (C6), stated here and in the report.**
    - (i) **RV windows for the P2 read.** For P2 only, RV_on runs 16:00 ET t → 09:50 ET t+1 and RV_rth 09:50 → 16:00 ET t+1, so the implied and realized clocks match. The P1 read keeps L4's windows exactly.
    - (ii) **Small-n note.** If any L8 cell (non-event weeknights, by bucket) has n < 40, the L8 verdict is printed with a "cell n < 40" note naming the cell. The verdict itself is still computed exactly as L8 says.

**Known prior relevant to P3** (from CR-BO Step 0 Q3, `scripts/output/cr_bo/step0.md`). On cached 0DTE ATM pairs (|K − `stock_price`| ≤ 10, n 196) at 09:31 ET, the τ that reproduces each pair's mid at its own `mid_iv` was a median 3.889× calendar time (IQR 1.008–8.261). At 09:45 it was ≈ 1.02×. `mid_iv` at 09:31 is therefore not tightly tied to the 09:31 mid on 0DTE, which is why C4 makes P3 price-based and keeps `mid_iv` diagnostic only.

---

## Step 0 — diagnosis findings (2026-10-18)

Scripts `scripts/cr_bp_step0.py` (items 1–11) and `scripts/cr_bp_step0_era.py` (era split of P3). Both are SELECTs only, with no run row and no API calls. Outputs in `scripts/output/cr_bp/`:
- `step0.md`, `step0_era.md`, `step0_crbo_rerun.md`
- `step0_pairs.csv`, `vpx_primary.parquet`, `vpx_secondary.parquet`, `p3_decision.json`

**Item 1 — gate PASS.** The CR-BO Q1–Q7 re-run reproduces `step0_pairs.csv` exactly: 621 ES-clean nights; S1 500 / S2 498 / S7 517.

**Items 2–3 — P3(a)(b)(c)(g).**
- **S and the rates.** S = `spot_price` at the first monies row ≥ 06:33 PT (≤ 06:40), per CR-AS `SPX_OPEN_SQL`; found on 723 of 723 CR-AS train dates. r = q = 0, because the harness prices from quote mids and settles on intrinsic.
- **Primary subset: n = 44** (2023: 20, 2024: 20, 2025: 3, 2026: 1).
  - Median |K − S| = **9.09 pts** (IQR 8.28–9.42).
  - IM median 19.5 vs 44.8 on all CR-AS days: the subset is low-IM days only.
  - 13 days fall in 2023 `spot_price`-lagged windows. V_px with S = `stock_price` changes V_px by a median ×1.00.
- **Secondary set: n = 648** (445 in the CR-BO universe as t+1 with S1 ok). K_p − S median −22.5, K_c − S median +22.4 pts.

**Item 4 — P3(d), pooled.** At 09:33 ET:

| set | atmiv²τ / V_px | vol50²τ / V_px |
|---|---|---|
| primary | 0.132 | 0.104 |
| secondary | 0.358 | 0.908 |

The secondary IQRs are bimodal (vol50: 0.263–0.966).

**Era split (`step0_era.md`).** The bimodality is a regime break at **2024-03-01**:

| set | era | atmiv²τ / V_px @09:33 | vol50²τ / V_px @09:33 | low-mode days |
|---|---|---|---|---|
| secondary | 2023-05 → 2024-02 | 0.321 | **0.224** | 91 % |
| secondary | 2024-03 → 2026-06 | 0.368 | **0.931** | 2 % |
| primary | 2023-05 → 2024-02 | 0.129 | 0.083 | 96 % |
| primary | 2024-03 → 2026-06 | 0.133 | 1.158 | 5 % |

The primary rows are n 25 and 18. Low mode means vol50²τ / V_px < 0.5 at 09:33.

- **The quotes are steady across the break.** The strangle-implied vol is 0.224 vs 0.216, and leg-pair bid–ask is 0.30 vs 0.30 pts.
- **The monies fields move.** `vol50` at 09:33 is 0.102 before the break vs 0.213 after.
- **Over all train days** (all PM ≤ 15 DTE), median field(09:33) / field(09:50) by calendar DTE at the open:

  | field / era | 0 | 1 | 2–3 | 4–7 | 8–15 |
  |---|---|---|---|---|---|
  | `vol50`, 2023-05 → 2024-02 | 0.523 | 0.790 | 0.864 | 0.944 | 0.966 |
  | `vol50`, 2024-03 → 2026-06 | 1.035 | 1.014 | 1.012 | 1.004 | 1.000 |
  | `atmiv`, 2023-05 → 2024-02 | 0.622 | 0.834 | 0.890 | 0.969 | 0.976 |
  | `atmiv`, 2024-03 → 2026-06 | 0.625 | 0.841 | 0.900 | 0.977 | 0.982 |

  Same pattern at 09:31 and 09:35.

**Item 5 — P3(e), close side.** atmiv / vol50 at 15:59 on t: S1 1.017, S2 1.012, S7 1.007. All three are inside 0.95–1.05, so no "unadjudicated" flag.

**Item 6 — P3(f).** The 0DTE / S1 / S2 open-side `atmiv` first changes at a median 09:37 ET (IQR 09:34–09:39). 417 of 436 days change by 09:40.

**Item 8 — P3 decision line, applied mechanically.** The secondary set decides (primary n 44 < 100). The 09:33 medians are atmiv 0.358 (outside 10 %) and vol50 0.908 (inside), so the **extension field is `vol50`**. P1 coverage:

| bucket | v_b from V_px | v_b from vol50²τ |
|---|---|---|
| S1 | 445 | 55 |
| S2 | 0 | 498 |
| S7 | 0 | 517 |

**Item 9 — P4 proxy.** CR-AS 0DTE per-leg bid–ask at 09:31–09:33 ET: median 0.30 pts (IQR 0.10–0.70). Two-leg straddle 0.40; strangle 0.30. This is a lower-bound proxy; the real cost test is Gate B. No 1-DTE train quotes exist.

**Items 10–11.** CR-BO P_on vs the P6 priors: S1 +0.260 vs +0.05, S2 +0.255 vs 0, S7 +0.170 vs +0.14. C6(i) and (ii) are as drafted.

### Gate: spec amendment required — STOP

The P3 rule pools both eras into one median. That median (0.908) qualifies `vol50`, but `vol50` is itself depressed at the open on every day before 2024-03-01 (0DTE 0.52, 1 DTE 0.79, 2–3 DTE 0.86 of its 09:50 value). Mechanical application would put that field into v_b for S2 and S7 on all ~10 pre-break months, and for S1 on its 55 non-V_px nights. That is the bias this CR exists to remove. The spec did not anticipate an era break, so it needs an amendment before implementation.

Also recorded for the amendment:
- (i) `atmiv` is below the price-implied variance at the open in **both** eras: 0.13 against ATM straddles, 0.32–0.37 against the strangle. That is the P8 measurement.
- (ii) On S1's 445 V_px nights, v_b would be the ±0.5 IM strangle variance, a wing quantity. v_a stays ATM `atmiv`. In the clean era vol50²τ / V_px is 0.93 on the strangle, so the wing–ATM gap biases D_on down by about that much.
- (iii) The primary subset is n 44, on low-IM days only.

---

## Amendment A1 — era-aware P3 (Ryan, 2026-10-18; adopted after the Step 0 STOP, before any implementation run)

- **A1.1** Era split fixed at **2024-03-01**. It is data-determined from the Step 0 era table and recorded as an **in-sample choice**. Sensitivity at 2024-02-01 and 2024-04-01 is reported as a diagnostic column, not as a selection. Era E0 = nights with t < 2024-03-01; E1 = t ≥ 2024-03-01.
- **A1.2** P1 open-side variance = V_px (K-aware solve, P3(g)) on every primary-subset night, in both eras. It is the only convention-free measurement, and the quote-based vol shows no era break.
- **A1.3** The P3 decision rule (field within 10 % of the V_px median at 09:33) is applied separately per era and per bucket (S1/S2/S7).
  - A field qualifies for extension only in the era-bucket cells where it passes.
  - Where no field passes, that era-bucket runs on the primary subset only; n is reported per cell.
  - No field is extended across the break.
- **A1.4** Mandatory splits added to L6: era (E0/E1) for every statistic, including L8.
  - L8 is judged on the pooled P1 read, as frozen, and the report shows E0 and E1 beside it.
  - If the two eras disagree in sign, that is recorded as a finding, and the decision line must say "modify", not "pass".
- **A1.5** New diagnostic table (Step 0 addendum, committed): by era and bucket, the median and IQR of atmiv²τ / V_px and vol50²τ / V_px at 09:31, 09:33, 09:35 and 09:45, and at 15:59 where captures exist, plus the implied-vol-ratio form √(·).
  - This table is the measurement for `open-questions/implied-move-1d-is-atm-iv-not-open-straddle.md` (P8).
  - It is also the measurement for a new open question, "atmiv at the open understates traded straddle IV — implied_move_1d exposure by era" (status: created, priority: high). That note lists the downstream fields: `bt_daily_features.implied_move_1d` and the σ-normalised features, `bt_range_labels_daily` h=1, CR-AS/CR-BM condor strike placement, `bt_daily_outcomes.range_over_im`.
  - Exposure is stated as measured ratios per era; no cause.
- **A1.6** Two rows are appended to `components/field-conventions.md` (monies section): `atmiv` and `vol50` at the open, giving the per-era ratio to straddle-implied variance from A1.5 and the note "ORATS-side change at ~2024-03-01, cause unknown", established in CR-BP Step 0.
- **A1.7** Nothing else in the frozen spec changes. L8, L9, P5, the P6 priors, the seed, the universe and the event list are unchanged. Proceed to the extraction and the single run, then the reconciliation protocol with the CR-BO and CR-037 session notes as priors. STOP before any modification.
- **A1.8** The report states in one line the E1-era (t ≥ 2024-03-01) median of atmiv²τ / V_px and its square root at 09:33 for bucket S1, with n. This is the live-card `implied_move_1d` accuracy today. The E0 equivalent is stated beside it.

### A1 reading, recorded at adoption (Code)

These follow mechanically from A1 and the frozen spec; they are stated here so the run is reviewable.

- **(r1) Only 0DTE contracts are captured** (Step 0 item 4). So for S2 and S7 the A1.3 test has no V_px for the bucket's own contract and cannot pass in either era. The primary-subset V_px is the 0DTE contract's, not theirs. **So S2 and S7 have no P1 rows (n = 0) in both eras.**
  - L8's kill (a), "P_on CI spans zero … in every bucket", is evaluated over the buckets with n > 0, and the report says so.
  - S2 and S7 stay reported through the P2 read (`atmiv` at 09:50), which A1 does not restrict.
- **(r2) S1.** The A1.3 test uses the deciding set of C1. That is the secondary strangle, because the primary n is 44 < 100, and per era 25 / 18 < 100. P1 v_b is:
  - V_px (primary, A1.2) on primary-subset nights;
  - otherwise the field that passed in that era's S1 cell;
  - otherwise the night drops.
- **(r3) P2** (`atmiv` at 09:50, C6(i) windows) is computed on every CR-BO-universe night per bucket. It is reported both on its full sample and on the P1 nights.

---

## Run summary and reconciliation (2026-10-18)

Single run 08:21 PT. Outputs are `scripts/output/cr_bp/report.md` and `summary.json`; the session note is `2026-10-18 - CR-BP — Session Premium Live Open`.

**P1 coverage.** v_b = V_px (primary, K-aware) on 20 S1 nights (E0 14, E1 6); v_b = vol50²τ on 339 E1 S1 nights; 141 E0 S1 nights have no P1 row. S2 and S7 have no P1 rows (A1 reading r1).

**L8 — KILL S1 as written** (kill (b)). On the pooled P1 read, S1 non-event weeknights (n 225):
- P_on −0.818 [−2.288, −0.006], P_rth +0.748 [−0.056, +2.193].
- The mean is driven by one `vol50` print: on 2024-10-15, v_b = 144.8 var pts vs atmiv 0.077, contributing −0.643 of the mean.
- Without the worst 5 nights the mean is +0.012; the medians are P_on +0.012 and P_rth +0.035.

**A1.4.** E0 +0.002 (n 1) vs E1 −0.821. The signs disagree, so the decision line must be "modify".

**Convention-free S1 nights** (n 20): P_on +0.006 [−0.044, +0.055], P_rth +0.042 [+0.000, +0.085].

**P2** (atmiv at 09:50, RV split at 09:50): P_on ≈ 0 in every bucket, every CI spanning 0.

**Shares.** Variance share 38.7 % (E0 34.9, E1 40.4); volatility share 44.0 %.

**A1.8 — live card.** atmiv²τ / V_px at 09:33 is E1 0.133 (√ 0.365, n 18) and E0 0.129 (√ 0.359, n 25). On `implied_move_1d`'s √(1/252) clock it is E1 0.717 (√ 0.847) and E0 0.696 (√ 0.834).

**Reconciliation.**

| Source | Classification |
|---|---|
| Muravyev–Ni | does not replicate |
| Terstegge | not comparable |
| Jones–Shemesh | direction only (secondary) |
| OptionMetrics | does not replicate |
| Gatheral | replicates |
| Bennett | does not replicate |
| Augen | direction only |
| Chong | not comparable |
| Lou–Polk–Skouras | not comparable |
| CR-BO | does not replicate |
| CR-037 | does not replicate (its "not a snapshot artifact" reading) |

**Decision: modify** (A1.4). L8 as written returns KILL; the conflict is left to Ryan. STOP before any modification.

**Verification (step 9).** A separate agent re-checked about 110 numbers and 13 quotes. It fixed 7 wording and number discrepancies in the vault notes; none changes a verdict or classification.

**Follow-ups:**
- the `atmiv`-at-open exposure (new open question, high priority);
- `vol50` single-day extremes;
- Gate A's 09:31 ATM 0DTE straddle and 15:59 1-DTE capture, needed for a price-only re-test.

## Decision (Ryan, 2026-10-18) — supersedes the "Decision: modify" line above

**Not supported, not killed — S1 parked.**

- **(a) Governing number.** The convention-free P1 read: S1 P_on +0.006 [−0.044, +0.055], n = 20.
- **(b) L8's mechanical KILL is not adopted.** It is driven by the 2024-10-15 `vol50` print. No outlier rule was pre-registered, so neither the KILL nor the trimmed +0.012 is a result.
- **(c) CR-BO's PASS is withdrawn as a field artifact.**
- **(d) Augen is reclassified as Does not replicate** on the P2 S7 read: 5.8 % vs 19 %.
- **(e) What would unpark S1:** about 600 train nights of 1-DTE ATM straddle quotes at 15:55 and 09:35 ET via the ORATS one-minute history. This is a bounded backfill, deferred until a Gate A pass elsewhere.
