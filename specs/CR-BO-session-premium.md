# CR-BO — T-S1a: session premium — implied-variance decay close→open vs realized overnight variance

Read-only, pre-registered, train only (t+1 ≤ 2026-06-05). Single run. Branch `feat/CR-BO-session-premium` off `origin/main` `5704969` (contains `333f748`, the CR-BM merge, and `5704969`, the CR-BN merge).

Authority: vault `Dash/research/Test plan — lean sequence and reconciliation protocol.md` (the T-S1a block and the Reconciliation protocol section are binding); strategy note `Dash/research/strategies-proposed/S1 - Night Shift.md`. Dash prior: CR-BM (2026-10-17): neither condor wing earns at a 09:33 ET entry held to the close (713 dates), which Muravyev–Ni predict for the intraday leg.

No API calls, no credentials touched, no table writes, no cron. Holdout (> 2026-06-05) never read.

## T-S1a block (verbatim from the test plan)

### T-S1a — session premium

- **Data.** `orats_monies_minute`: per expiry, ATM IV (`atmiv`) at 15:59 and next session 09:31; expiries 0/1/2 DTE (calendar DTE as listed) and 7-day. ES 1-min from `es_minutes`, roll-safe (exclude roll weeks per the Databento caveat; use `trading_calendar`). Train 2023-05 → 2026-06-05.
- **Compute.** Per night: implied variance at 15:59 = atmiv² × T (T in years to expiry); implied at 09:31 next day likewise; implied-variance *drop* overnight = IV₁₅:₅₉² T₁ − IV₀₉:₃₁² T₂. Realized overnight variance = Σ 1-min log returns² from 16:00 to 09:30 ET (continuous ES). Session premium = implied drop − realized overnight variance, in variance units and as a ratio. Also realized RTH variance and the implied RTH drop for the complementary test (Muravyev–Ni predicts the sign flips).
- **Primary statistic.** Median and mean session premium with bootstrap CI, by (a) weekday (Fri→Mon separately), (b) DTE bucket, (c) event nights vs non-event (hand-code FOMC/CPI/NFP dates for the train window in the spec — ~90 dates, no table needed).
- **Pass.** Overnight premium > 0 with CI clear of zero on non-event weeknights AND the intraday premium is ≤ 0 or indistinguishable from zero (the sign flip is the thesis). **Kill S1.** Overnight premium CI spans zero on non-event nights, or the intraday premium is as large as the overnight one (then there is no session asymmetry to trade).
- **Literature to reconcile against.** [[MuravyevNi2020]] (−1.0 %/day overnight vs +0.3 % intraday; < 3-week −2.6 % vs ≈ 0; the calendar-time mechanism), [[Terstegge2025]] (puts −2.49 % / +0.39 %; calls flat — so also split the premium by put-side vs call-side IV if `vol25`/`vol75` make that possible), [[JonesShemesh2018]] (weekend; weak for SPX index options in their WP), [[OptionMetrics2025]] (Mondays ≈ ⅔ post-2022), [[Gatheral2006]] ch. 5 (≈ 40 % overnight), [[Bennett2014]] (≈ 1/6), [[Augen2008]] ch. 8 (expiry-week decay), [[Chong2024]] (fair net of costs), [[LouPolkSkouras2019]].

## Locked decisions

Amendments adopted before this freeze (Ryan, 2026-10-18): L2 and L4 replaced; L1 addition (contract-switch nights). They are written into the rows below; the original L2 / L4 text is kept under "Superseded" for the record.

| # | Decision |
|---|---|
| L1 | Universe: every train session pair (t, t+1) with t+1 ≤ 2026-06-05 where `orats_monies_minute` has a 15:59 ET row on t and a 09:31 ET row on t+1 for the same expiry, and ES 1-min bars cover 16:00 ET on t → 09:30 ET on t+1 with ≤ 5 % missing minutes. Exclude ES roll weeks (the Databento `ES.c.0` dying-contract caveat) and the 2026-09 patched window if it falls in range (it does not), holidays and early closes per `trading_calendar`. **Addition:** exclude every night that spans a contract switch in the ES continuous series, all years 2023–2026, plus the 2025 `ES.c.0` dying-contract roll weeks. Step 0 lists the roll dates from `trading_calendar` / the contract schedule and the count of nights removed. |
| L2 | **(replaced)** Contracts bucketed by SESSIONS to expiry at 15:59 ET on t, not calendar days: S1 = the expiry on the next trading session (Monday's on a Friday; after a holiday, the next session's), S2 = the expiry two sessions out, S7 = the nearest listed expiry ≥ 7 calendar days out. PM-settled only (`expiry_tod = 'pm'`), identified by `expirDate`, same `expirDate` at 09:31 on t+1. Record calendar days to expiry as a column so the Friday 3-day gap is visible in the splits. 0DTE is excluded because at 15:59 it has one minute of life and no overnight leg. |
| L3 | Implied variance = atmiv² × τ, τ = time from the snapshot to 16:00 ET on `expirDate` in years, using the same day-count ORATS uses for `atmiv` (confirm the convention in Step 0 and record it; if ambiguous, use calendar days/365 and state so). Implied overnight drop D_on = IV²τ\|15:59,t − IV²τ\|09:31,t+1. Implied intraday drop D_rth = IV²τ\|09:31,t+1 − IV²τ\|15:59,t+1 for the same contract. |
| L4 | **(replaced)** Total variance over the window, no annualization on either side: D_on = atmiv²τ\|15:59,t − atmiv²τ\|09:31,t+1 and RV_on = Σ r² over the same clock window are directly comparable as they stand. Same for the RTH leg. Realized from ES 1-min log returns (basis cancels in log returns; no SPX conversion): RV_on = Σ r² over 16:00 ET t → 09:30 ET t+1 (including the 16:00–17:00 and 18:00+ sessions; the 17:00–18:00 halt contributes nothing); RV_rth = Σ r² over 09:30 → 16:00 ET t+1. Realized overnight share series: variance share RV_on/(RV_on+RV_rth) compared to Gatheral's 40 %; also the volatility-share form √RV_on/(√RV_on+√RV_rth) for Bennett's one-sixth — report both, never mix them. |
| L5 | Session premium P_on = D_on − RV_on; P_rth = D_rth − RV_rth. Primary units: variance points; secondary: the ratio D/RV. Also the ATM-straddle-equivalent in index points for readability (stock_price × √variance; `stock_price`, not `spot_price`, with the 2023 jitter caveat noted). |
| L6 | Splits: (a) weekday of t, Fri→Mon separate; (b) expiry bucket S1/S2/S7; (c) event nights vs not — event night = a CPI, PPI, NFP or PCE release at 08:30 ET on t+1; FOMC statement day on t+1 flagged separately as an RTH-leg event. Hand-code the dates for 2023-05 → 2026-06 from the BLS/BEA/Fed public schedules into the spec as a list (≈ 100 dates) with the source URL for each schedule; no table. (d) Terstegge split: put-side and call-side using the 25-delta put vol and 25-delta call vol — confirm in Step 0 which of `vol25`/`vol75` is which under ORATS' convention and record it; compute D_on and the premium per side with the same τ. |
| L7 | Statistics: median and mean of P_on and P_rth with bootstrap 95 % CI (10,000 resamples, seed 17), per split; sign test on P_on > 0 per night; the Muravyev–Ni contrast mean(P_on) − mean(P_rth) with CI, per bucket. |
| L8 | Pass: on non-event weeknights, P_on > 0 with CI clear of zero in bucket S1 AND P_rth ≤ 0 or CI spanning zero (the sign flip is the thesis). Kill S1: P_on CI spans zero on non-event weeknights in every bucket, OR P_rth is as large as P_on (no session asymmetry). Anything else: "not supported, not killed". |
| L9 | Pre-registered placebo: pair D_on on t with RV_on from the following night (t+1 → t+2); the premium should collapse toward the unconditional mean difference. Report alongside the main read. |
| L10 | Output: parquet under `scripts/output/cr_bo/` (one row per night × bucket × side) and a markdown report. Read-only; holdout never read; no API calls. |

### Superseded (kept for the record, not in force)

- Original L2: "Contracts: for each t, the listed SPX expiries with 1, 2 and 7 (nearest) calendar days to expiry at 15:59 ET, PM-settled only (expiry_tod = 'pm'; AM-settled excluded, consistent with CR-BN). Identify the expiry by expirDate; use the same expirDate at 09:31 on t+1." — Replaced because a Friday has no 1-calendar-day expiry (Saturday); the Monday expiry is 3 calendar days out.
- Original L4: "… Both annualized the same way as L3 so units match. Report the realized overnight share RV_on/(RV_on+RV_rth) as a standalone series (Gatheral/Bennett benchmark)." — Replaced because atmiv²τ and Σr² are both total variance already; annualizing one side would break the match. Gatheral's figure is a variance share, Bennett's a volatility share.
- Test-plan "0/1/2 DTE": 0DTE dropped by L2 (one minute of life at 15:59, no overnight leg).

## Gates

| Gate | Requirement |
|---|---|
| Step 0 | All Step 0 questions answered in the second commit. If any requires a spec amendment, STOP and report; otherwise continue without waiting. |
| L8 | Pass / kill / "not supported, not killed", exactly as written. |
| L9 | Placebo reported alongside the main read. |
| Shares | Both realized-share series reported against their benchmarks (variance share vs Gatheral 40 %; volatility share vs Bennett 1/6). |

## Implementation order

1. Spec freeze (this file, amended block).
2. Step 0 diagnosis: L1 coverage counts after each exclusion (nights, by bucket); roll dates and nights removed; ORATS τ convention for `atmiv` (exact τ used, one worked example); the `vol25`/`vol75` delta mapping; ES overnight minutes missing by year; first date each weekday has a next-session expiry; the event-date list with sources.
3. Extraction script (monies snapshots + ES overnight / RTH sums), one commit.
4. Statistics + report (the single run), one commit.
5. Wrap commit (reconciliation summary, deltas, follow-ups).

## Step 0 — diagnosis findings (2026-10-17)

Script `scripts/cr_bo_step0.py` (SELECTs only, no run row; shared definitions in `scripts/cr_bo_common.py`). Full output `scripts/output/cr_bo/step0.md`; per-night flags `scripts/output/cr_bo/step0_pairs.csv`.

**Q1 — L1 coverage.** 777 session pairs → 761 after early closes (16 pairs touch the 8 early closes) → 632 after roll weeks (129 nights) → 632 after contract switches (all 12 switch nights already lie in roll weeks) → 631 with an ES anchor bar 15:55–15:59 on t → 623 with ≤ 5 % overnight minutes missing → **621 ES-clean nights** after also requiring ≤ 5 % of RTH t+1 minutes (needed to compute RV_rth at all; this applies L1's 5 % rule to the RTH leg — an interpretation, recorded here, not a new threshold). With the monies rows of L1/L3: **S1 500, S2 498, S7 517 nights.**
- Monies losses on ES-clean nights: the 09:31 ET snapshot is absent from `orats_monies_minute` on 92 t+1 days (an ingest gap concentrated in 2024-04/05 and 2025-01/02, 16 + 16 + 9 + 12 days). This takes ~15 % of every bucket. It is clustered in time, so it can bias the sample toward the remaining months; recorded, not repaired (read-only CR).
- On Thursday nights before a monthly third Friday the PM S1 row does not exist: the table is keyed (ticker, expir_date, quote_date), so the AM monthly and the PM weekly on the same date cannot both be stored, and the AM row is the one held (S1 21, S2 20 nights). L2's PM-only rule removes them as written.
- Weekdays of t in the ES-clean set: Mon 121, Tue 132, Wed 128, Thu 126, Fri 114; calendar gap t→t+1: 1 day 505, 3 days 103, 2 / 4 days 13 (holiday-spanning nights, kept per L2).

**Q2 — contract switches and roll weeks.** Detected from the ES − SPX basis at 15:59 (jump > 30 pts; non-switch |Δ| ≤ 27.4, sd 9.1): 12 switches, one per quarterly expiry, +35.6 to +64.3 pts. The 2023–2025 Databento `ES.c.0` data switch at the **Sunday 18:00 reopen after expiry Friday**, and print **no RTH bars on expiry Friday** (the dying contract stops at the 09:30 SOQ). This applies to 2023 and 2024 as well as 2025: the same `ES.c.0` backfill default produced all three years. The 2026-03 Ironbeam switch is at 09:30 on expiry Friday 2026-03-20. Roll weeks, under the CR-BM definition (Monday of the week before the third Friday → expiry Friday), remove 129 nights. That covers the 2025 `ES.c.0` dying-contract weeks and the same weeks in 2023, 2024 and 2026.

**Q3 — τ convention.** `orats_monies_minute` has no fractional time column: `dte` is calendar days + 1, and `dte_pct` = (dte − 1)/365.25 is constant through the day. τ was implied from cached `orats_options_minute` ATM call/put pairs (parity forward, Black, r = 0). For 4–30 calendar-DTE contracts, ORATS IV matches τ = calendar time to 16:00 ET on expiry / 365, with median ratio 1.000–1.020 at both 09:31 and 15:59. Worked example: 2025-10-24 09:31 ET, expiry 2025-11-13 PM, K 6800, F 6793.15, call mid 75.45, mid_iv 0.12338. Calendar time to expiry is 20.2701 days; the implied τ is 20.2682 days (ratio 0.9999). No 1–3 DTE pairs are cached, so for S1 / S2 the L3 fallback (calendar/365) applies and is stated as such. 0DTE IV departs from calendar time in the last ~90 minutes, because ORATS uses a longer clock near expiry. That affects only v_c of S1 (15:59 on its expiry day, τ = 1 min), which is ≈ 0 under either clock. **τ used: (16:00 ET on expir_date − snapshot label time) / 365 days.**
- **atmiv refresh cadence (finding).** `atmiv` refreshes every few minutes, unlike `vol50`, which changes every minute. At the 09:31 row it equals the 09:30 row's value on 85.9 % of rows. The 09:30 row is built from pre-open quotes (`quote_date` 09:15 / 09:25 ET), so v_b is in practice the IV as of ~09:15–09:30. The 15:59 row's quotes are from 15:56–15:59 (mostly 15:58). The implied side's clock therefore ends up to ~15 min before RV_on's 09:30 boundary. L3 is unchanged (atmiv at the 09:31 row). Median atmiv / vol50 at 09:31 is 0.989.

**Q4 — vol25 / vol75.** The ORATS smile is indexed by call delta. At 15:59, `vol75` > `vol25` on 96.5 % of PM rows ≤ 15 DTE, and `vol75` > `vol50` on 97.1 %. **Put side = `vol75` (75-delta call = 25-delta put); call side = `vol25`.** This agrees with `packages/shared/vol_features.py:20-21`.

**Q5 — ES overnight minutes missing.** On roll-, switch- and early-close-clean nights, the median missing share is 0.00 % in every year; p95 is 0.40 % (2023), 0.30 % (2024), 0.10 % (2025) and 0.66 % (2026). The nights that fail are holiday-schedule nights (Christmas, New Year, Good Friday: 27–38 % missing even with the holiday date dropped from the grid) and 2026-02-12 → 02-19 (Ironbeam gaps, including the RTH legs).

**Q6 — weekday coverage.** Every weekday of t has a next-session PM expiry from the first train session (Mon 2023-05-01, Tue 05-02, Wed 05-03, Thu 05-04, Fri 05-05). Thursday's lower S1 count (104 / 154 of all pairs) is the third-Friday AM collision (Q1).

**Q7 — events.** The dates are hand-coded in `scripts/cr_bo_common.py`, with the source URL for each schedule:
- 08:30 ET releases: CPI 36, PPI 36, NFP 37, PCE 32, giving 141 distinct dates.
- PCE released at 10:00 ET is not an event night: 2024-11-27, 2025-04-30, 2025-12-05, 2026-01-22.
- FOMC statement days: 25. The 2025-08-22 notation vote is not flagged.
- October 2025 CPI / NFP / PPI were never published (the appropriations lapse).
- ES-clean universe: 107 event nights and 15 FOMC-on-t+1 nights. The full date list is in `step0.md` Q7.

### Gate: no spec amendment required — continue

Every Step 0 finding is covered by L1–L10 as written. The exclusions above follow mechanically from L1/L2. τ follows L3: confirmed for 4–30 DTE, calendar/365 fallback for 1–3 DTE. The RTH-coverage requirement applies L1's 5 % rule to the leg L4 needs. Recorded for the reconciliation, not amended: the ~15 % monies loss from the missing 09:31 snapshot (time-clustered), and the atmiv cadence at the open (v_b ≈ pre-open IV).
