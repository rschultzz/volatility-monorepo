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
