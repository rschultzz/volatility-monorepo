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

---

## Step 0 — diagnosis findings (2026-10-18) — **STOP: L2 needs an amendment**

Script `scripts/cr_bq_step0.py` (read-only; no outcome statistic computed); full output `scripts/output/cr_bq/step0.md`. Shared definitions `scripts/cr_bq_common.py`.

**Q1 — parquet schema and sign convention.**
- `scripts/logs/cr_bk/shape_summary.parquet`: 777 rows (2023-05-02 → 2026-06-05). Profiles in `profiles.parquet`: 769 dates, linear wings, S0 ± 300 in steps of 5.
- **Value:** `gex_total(price)` = Σ unit_gamma(F, K, σ, T) · F² · 100 · (call_oi − put_oi). Here F is the per-expiry forward, shifted in proportion to price. σ is sticky-strike, from the 12:46 PT prior-session smile.
- **Sign:** call OI +, put OI −, so dealers are assumed long calls and short puts (the SqueezeMetrics / Barbon–Buraschi eq. 4 convention). G > 0 means dealers are net long gamma.
- **Units:** $ of delta per unit (100 %) relative move. Divide by 100 for $ per 1 % move, so value / 1e11 = $bn per 1 % move.
- **`net_at_s0_linear`** is the only stored scalar. It is the profile at **S0 = the prior session's 15:46 ET** `orats_oi_gamma.stock_price`.
- **No stored column is gamma at the open.** CR-BK's T3 X was computed in the run and not saved: the profile interpolated at the SPX open from `packages/shared/spx_cash`. That open is the first `orats_monies_minute.stock_price` print ≥ 06:33 PT; on 2023 days before 2023-11-09 it is the 15-min-shifted `spot_price`.
- **Worked day 2025-02-11.** Recomputed from `orats_oi_gamma` and the 2025-02-10 12:46 PT smile:
  - calls +65.92e11, puts −61.42e11, net +4.50e11 (+$4.50bn per 1 %);
  - matches the parquet exactly (max grid difference 0);
  - flip −11.0 pts below S0. G(S0 − 25) = −5.57e11 and G(S0 + 25) = +14.90e11, so an ordinary overnight move changes the sign.

**Q2 — the conflict.** L2 asks for "CR-BK's net dealer gamma at the open". The kickoff says "no ORATS monies fields at the open are used anywhere in this test". CR-BK's value at the open needs exactly such a field, so the spec cannot be met as written. Two candidates use no monies field:
- **A:** `net_at_s0_linear`, G at the prior session's 15:46 ET price. Stored column.
- **B:** the profile at S0 · ES_open(d) / ES(prev session, 12:46 PT close). This is an ES-implied SPX open, uses ES only and is frame-neutral (a ratio). It cannot be computed on 12 universe days because the prior session has no 12:45 PT bar.
- **A vs B (n 629):** Spearman 0.932. Sign differs on 72 days (11.4 %). Same decile on only 46.3 %; decile differs by ≥ 2 on 13.5 %. D1 agrees on 53/63, D10 on 47/63. The S0 → open move has median |15.4| pts, p90 53.3. **Which G is used changes which days are in each decile, so it is a decision, not a detail.**

**Q3 — L1 counts.**
- 777 parquet rows. Every NYSE session in range is present, none missing.
- → 777 after dropping non-sessions.
- → **769** after the CR-BK exclusions (8 days with no 12:46 smile: 2023-07-05, 2023-11-27, 2024-07-05, 2024-12-02, 2024-12-26, 2025-07-07, 2025-12-01, 2025-12-26).
- → **761** after early closes.
- → **644** after roll weeks (all years).
- → **641** after requiring ≥ 370 bars (dropped 2026-02-13 with 361, 2026-02-17 with 282, 2026-02-18 with 307).

**Q4 — distribution of G (A; B is similar).**
- G < 0 on **243 of 641** days (37.9 %); B: 235 of 629. That is ≥ 60, so the decile test and the sign split are both powered as the spec intends.
- Decile cut points ($bn per 1 %): −29.05, −18.91, −6.67, +1.51, +8.72, +14.90, +21.82, +28.09, +37.05.
- **G is not stationary:**
  - median by year: 2023 +1.89, 2024 +9.22, 2025 +9.12, 2026 +25.48;
  - sd by year: 23.0 / 21.7 / 25.2 / 35.7;
  - **D10 = 1 day from 2023, 11 from 2024, 19 from 2025, 33 from 2026.** D1 is spread across years (21 / 13 / 22 / 9).
- So D1 − D10 on raw G is partly 2023–25 vs 2026. This is not an amendment under L2 as written, which says deciles on the train sample. It is flagged because Barbon–Buraschi normalise by ADV (Γ^IB) and CR-BK ranked over a trailing 250-session window; neither is in L2.

**Q5 — ES coverage by year.**
- **Non-early sessions with exactly 390 bars:** 2023 163/166, 2024 245/249, 2025 243/247. 2026 has 92 with 390 bars and 11 with 370–389.
- **0-bar sessions:** 10, all quarterly expiry Fridays (the dying contract stops at the SOQ; CR-BO Q2).
- **Other short sessions:** 2023-06-16 has 1 bar (expiry Friday); 2026-03-17 has 340 (roll week); 2026-02-13/17/18 are dropped above.
- Sources: databento 679 dates, ironbeam 111.

**Q6 — first and last 30 minutes.**
- All 641 included days have a 06:30 and a 12:59 PT bar, and all have the boundary bars 06:59 and 12:29.
- 12:30–12:59 is complete (30 bars) on every day. 06:30–06:59 has fewer than 30 bars on 5 days (minimum 24).
- No included day is an early close, so **12:30–13:00 PT is the last 30 minutes on every included day.**

**Comparability note (no amendment needed; recorded for the reconciliation).**
- Baltussen's 6.63 / 0.82 is the slope of the last half-hour on **rROD = prior close → 15:30 ET**.
- Gao et al.'s first half-hour runs **from the prior close to 10:00 ET**.
- L4(d)'s first-30 is 06:30–07:00 PT, RTH only, with no overnight leg. That matches neither paper's regressor exactly, so the L8 prior comparison is "direction only" by construction.

**Operational definitions to be fixed with the amendment (no new thresholds).**
- **Prices:** P0 = open of the 06:30 bar, P_k = close of bar k−1 (k = 1…390). Missing bars carry the last close forward.
- **Returns:**
  - 5-min r = ln(P_{5j}/P_{5(j−1)}), 78 per day;
  - 30-min returns: 13 per day;
  - 60-min returns: k = 60…360, 6 per day; the 12:30–13:00 half is dropped;
  - first-30 = ln(P30/P0), last-30 = ln(P390/P360).
- **ρ5:** sample lag-1 acf per day, mean across days.
- **ρ60:** pooled within-day pairs around the group mean.
- **VR:** per day Σr30² / Σr5², mean across days.
- **Skew:** √N Σr³ / (Σr²)^{3/2} on 1-min returns.
- **"Gradient":** D1 − D10, bootstrap over days, each group resampled separately, CI clear of zero in either sign.
- **Baltussen (L6):**
  - "t ≥ 2" means OLS with HC1 standard errors;
  - "flat across gamma sign" means the interaction (G < 0) × first-30 in the pooled regression has |t| < 2.
- **L5 previous-session statistic:** used only when the previous NYSE session itself passes L1; otherwise that day drops from the conditional regression.
- **RV tercile:** same-day Σ 1-min r², terciles on the universe.
- **Bootstrap:** `default_rng(17)` per statistic, 10,000 resamples.

**For approval (Ryan):** choose G under L2: A (stored `net_at_s0_linear`), B (ES-implied open), or C (CR-BK's exact X, which waives the no-monies-at-open rule for `stock_price` only). Also: keep raw-G deciles (L2 as written) or normalise. The run does not start until this is amended.
