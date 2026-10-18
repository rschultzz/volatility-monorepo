# CR-BQ — T-S2a: ES intraday return autocorrelation by dealer net-gamma decile (read-only, pre-registered, train only)

> Authority: kickoff prompt, 2026-10-18 (Ryan) + vault `Dash/research/Test plan — lean sequence and reconciliation protocol.md` (T-S2a block and Reconciliation protocol, binding).
> Branch: `feat/CR-BQ-gamma-autocorrelation` off `origin/main` 2d22998 (contains 961cf31 and 2d22998).
> Data safety: **read-only**. No table writes, no cron, no API calls, no credentials. Train sessions ≤ 2026-06-05 only; holdout never read. No ORATS monies fields at the open are used anywhere in this test.
> Strategy: [[S2 - Gamma Clock]]. Prior: CR-BK (`specs/CR-BK-corrected-gamma-profile-test.md`), State of Knowledge CR-BK section.

## T-S2a block (verbatim from the Test plan)

### T-S2a — autocorrelation by gamma decile

- **Data.** CR-BK parquet: net dealer gamma at the open per train day (the corrected profile, `scripts/cr_bk_profile.py` output). ES 1-min RTH bars.
- **Compute.** Per day: lag-1 autocorrelation of 5-min returns (78 obs) and of 60-min returns (6–7 obs — pool across days by decile instead), variance ratio VR(2) at 5-min, and the Baltussen regression (last-30-min return on first-30-min return) by gamma sign. Deciles of gamma on the train sample. Controls: same-window autocorrelation the previous day (Bogousslavsky), realized vol tercile, day-of-week.
- **Pass.** Bottom-decile minus top-decile 5-min autocorrelation > 0 with bootstrap CI clear of zero, OR the Baltussen last-half-hour coefficient positive and significant on negative-gamma days and not on positive-gamma days. **Kill S2.** Neither horizon shows a decile gradient with CI clear of zero AND the Baltussen coefficient is flat across gamma sign.
- **Literature to reconcile against.** [[Baltussen2021]] (6.63, t 4.78 vs 0.82 n.s.), [[BarbonBuraschi2021]] (−1.5 to −2 pp per sd at 5–60 min; Table VI), [[Barbon2022]] (82 % next-open reversal), [[DimErakerVilkov2024]], [[Adams2026]], [[Amaya2025]] (tails not means), [[Ni2021]] (identification), [[CarrLee2003]], [[Bogousslavsky2016]], [[Gao2018]], and Dash [[State of Knowledge — 2026-09-20]] CR-BK section (−0.115 range result, "gamma matters for volatility, not levels").

## Locked decisions (kickoff, 2026-10-18)

| # | Decision |
|---|---|
| L1 | Universe: train sessions (≤ 2026-06-05) in the CR-BK parquet with ES RTH 1-min bars (es_minutes, is_rth, 06:30–13:00 PT) having ≥ 370 of 390 minutes; exclude NYSE holidays, early closes, roll weeks in every year (CR-BO Step 0 rule), and the dates CR-BK excluded. Report counts after each exclusion. |
| L2 | Gamma state G = CR-BK's net dealer gamma at the open from the corrected profile (sign convention as the parquet defines it; state it). Deciles of G on the train sample (D1 = most negative). Also the sign split (G < 0 vs ≥ 0). No path-induced repricing in this CR (that is T-S2c). |
| L3 | Returns: 1-min log returns of ES RTH bar closes. Aggregates: 5-min returns (78 per day), 30-min returns (13 per day), 60-min returns (6 full + 1 half, use 6). |
| L4 | Statistics, all per decile and per sign, with bootstrap 95 % CI over days (10,000 resamples, seed 17): (a) ρ5 = within-day lag-1 autocorrelation of 5-min returns, averaged across days in the decile; (b) ρ60 = lag-1 autocorrelation of 60-min returns pooled across days in the decile (too few per day); (c) VR = variance ratio of 30-min to 5-min returns (VR = 1 random walk, > 1 momentum, < 1 reversal); (d) Baltussen regression: last-30-min return (12:30–13:00 PT) on first-30-min return (06:30–07:00 PT), slope and t-stat, by sign of G and unconditional (Gao et al. baseline); (e) secondary, not gated (this is T-S2b reported early because the data are identical): same-day realized skewness of 1-min returns by decile. |
| L5 | Controls: for (a) and (d) add the same statistic from the previous session as a regressor (Bogousslavsky infrequent-rebalancing channel), realized-vol tercile of the day, and day-of-week; report conditional and unconditional. |
| L6 | Pass: (D1 − D10) ρ5 > 0 with CI clear of zero, OR Baltussen slope positive with t ≥ 2 on G < 0 days and not on G ≥ 0 days. Kill S2: neither (a) nor (b) nor (c) shows a decile gradient with CI clear of zero AND the Baltussen slope is flat across gamma sign. Anything else: "not supported, not killed". |
| L7 | Placebo: the same statistics with G shuffled across days (seed 17) — the decile gradient should vanish. |
| L8 | Output: parquet under scripts/output/cr_bq/ (one row per day with G, decile, ρ5, VR, first/last 30-min returns, realized skew) and a markdown report. Priors to judge against: Baltussen 6.63 (t 4.78) vs 0.82 n.s.; Barbon–Buraschi −1.5 to −2 pp autocorrelation per +1 sd gamma at 5–60 min; effect is a tilt (stock-level R² ≈ 0.003) — judge by CI on the gradient, not by Sharpe. |

## Sequence

1. Spec freeze (this commit).
2. Step 0 diagnosis (next commit): parquet schema and gamma sign convention with one worked day; L1 counts after each exclusion; distribution of G (how many days < 0 — if fewer than 60, the sign split is reported but the decile test is primary); ES minute coverage by year; confirm the 12:30–13:00 PT window is the last 30 min on every included day. **If Step 0 requires a spec amendment, STOP and report.** Otherwise continue without waiting.
3. One commit for extraction (per-day statistics).
4. One commit for statistics + report.
5. Wrap. Single run. Report pass/kill against L6 exactly as written, L7, and (e) as secondary. Then the Reconciliation protocol, all nine steps.
