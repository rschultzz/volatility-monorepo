# CR-BS — T-S3a report (single run, 2026-10-18 11:02 PT)

Universe n 625 (E0 169, E1 456), 2023-05-01 → 2026-06-05. Y = log(RV 10:30→16:00 / atmiv0²·τ_rem), session clock τ_rem = (5.5/6.5)/252. Premium a seller earns = −Y. `atmiv` primary; `vol50` beside it. No price-based adjudication exists at 10:30 ET (no train quotes at that minute).

## State at 10:30 ET (L2)

| var | E0 mean | E1 mean | pooled mean | sd |
|---|---|---|---|---|
| L | -3.291 | -3.149 | -3.187 | 0.731 |
| K | +0.101 | +0.169 | +0.150 | 0.097 |
| C | +0.032 | +0.072 | +0.061 | 0.060 |
| T | +0.656 | +0.651 | +0.652 | 0.515 |
| M | -1.117 | -1.051 | -1.069 | 0.454 |
| Y | -1.707 | -1.807 | -1.780 | 0.444 |
| Y_cal | -0.030 | -0.130 | -0.103 | 0.444 |

Pairwise correlations (Pearson above the diagonal, Spearman below):

| | L | K | C | T | M |
|---|---|---|---|---|---|
| L | 1 | -0.20 | -0.36 | +0.26 | -0.03 |
| K | -0.23 | 1 | +0.43 | -0.09 | +0.14 |
| C | -0.38 | +0.54 | 1 | -0.11 | +0.08 |
| T | +0.32 | -0.15 | -0.16 | 1 | +0.03 |
| M | +0.03 | +0.14 | +0.04 | +0.03 | 1 |

Clock: mean Y (session) -1.780 vs Y_cal (1/365) -0.103; the difference is the constant log(τ_session/τ_cal) = +1.677 on every day (level shift only). Unconditional RV/implied: session 0.190 (mean of ratio), exp(mean Y) 0.169; calendar exp(mean) 0.902.

## Primary — walk-forward (L4, L6)

- OOS predictions n 375 (E0 0, E1 375); ranked (quintile assigned) 356.
- **OOS R² (vs expanding historical mean) -0.0098**; vs full-OOS mean -0.0208 (diagnostic).
  - E1: OOS R² -0.0098 (n 375)

| quintile (Q5 = richest predicted premium) | n | mean ŷ | mean realized Y | E0 n / mean Y | E1 n / mean Y |
|---|---|---|---|---|---|
| Q1 | 63 | -1.652 | -1.706 | 0 / — | 63 / -1.706 |
| Q2 | 84 | -1.723 | -1.865 | 0 / — | 84 / -1.865 |
| Q3 | 77 | -1.765 | -1.792 | 0 / — | 77 / -1.792 |
| Q4 | 76 | -1.813 | -1.831 | 0 / — | 76 / -1.831 |
| Q5 | 56 | -1.886 | -1.794 | 0 / — | 56 / -1.794 |

- **Pooled Q5 − Q1 of realized Y: -0.087 [-0.259, +0.086]** (n 56 / 63).
- OOS set = E1 only (E0 OOS n 0 by construction: the 250-day first window covers all of E0; first prediction 2024-07-31).
- Diagnostic, pooled-OOS breakpoints: Q5 − Q1 -0.110 [-0.258, +0.037].
- A1.3 (descriptive): OOS R² first half +0.0053 (n 187, 2024-07-31 → 2025-07-09); **second half -0.0183** (n 188, 2025-07-10 → 2026-06-05).

**L6 (as amended by A1.1): KILL S3's gate (and, with CR-BM, S3 as designed).** Pass needs OOS R² ≥ 0.05 (-0.0098) AND pooled Q5 − Q1 < 0 with CI clear (-0.087 [-0.259, +0.086]). Kill needs OOS R² < 0.02 AND pooled Q5 − Q1 CI spans zero (True). The OOS set is E1-only by construction (n 375).

### By era — in-sample (no OOS predictions by construction for E0)

| era | n | in-sample R² (own-era OLS) | in-sample Q5 − Q1 (full-sample fit, quintiles within era) | label |
|---|---|---|---|---|
| E0 | 169 | 0.1115 | -0.237 [-0.416, -0.064] | in-sample (no OOS predictions by construction) |
| E1 | 456 | 0.0184 | -0.235 [-0.358, -0.107] | in-sample (no OOS predictions by construction) |

Full-sample OLS coefficients (= the last walk-forward fit + 1 day): const -1.517, L +0.002, K -0.403, C -0.109, T -0.066, M +0.138

## In-sample R² (reported against Bandi–Fusari–Renò; never gated)

| model | n | R² |
|---|---|---|
| raw, Y on (L, K, C, T, M) | 625 | 0.0288 |
| MA22 (a): Y and C smoothed (BFR fn. 17, literal analogue) | 604 | 0.0925 |
| MA22 (b): Y and all five smoothed | 604 | 0.2249 |
| univariate L (raw) | 625 | 0.0000 |
| univariate K (raw) | 625 | 0.0045 |
| univariate C (raw) | 625 | 0.0012 |
| univariate T (raw) | 625 | 0.0038 |
| univariate M (raw) | 625 | 0.0157 |
| raw, era E0 only | 169 | 0.1115 |
| raw, era E1 only | 456 | 0.0184 |
| noise benchmark: Y permuted across days (seed 17), MA22 (a) / (b) | — | 0.1009 / 0.1731 |

Overlapping 22-day moving averages on both sides inflate R² mechanically (serially correlated, effectively ~n/22 independent observations); the MA columns are the BFR comparison, not evidence. BFR's 21.20 % / 34.81 % are a different object (vol-drift premium α^Q − α to next-day 10:30; single equation vs GMM joint with the return equation).

## Spearman of each state variable with Y (bootstrap 2,000, seed 17)

| var | pooled | E0 | E1 |
|---|---|---|---|
| L | +0.034 [-0.049, +0.114] | -0.087 [-0.239, +0.071] | +0.073 [-0.016, +0.160] |
| K | -0.124 [-0.206, -0.046] | +0.079 [-0.070, +0.234] | -0.120 [-0.216, -0.022] |
| C | -0.037 [-0.114, +0.045] | +0.168 [+0.016, +0.317] | -0.035 [-0.124, +0.055] |
| T | -0.038 [-0.119, +0.040] | -0.047 [-0.201, +0.113] | -0.039 [-0.135, +0.055] |
| M | +0.126 [+0.045, +0.209] | +0.301 [+0.159, +0.439] | +0.076 [-0.020, +0.175] |

## `vol50` beside `atmiv` (reported, never gated)

- Y_v50 on (L_v50, K, C, T_v50, M_v50): OOS R² -0.0078 (n 375); Q5 − Q1 -0.148 [-0.317, +0.022]; in-sample R² 0.0217.
  - E0 Q5 − Q1 —
  - E1 Q5 − Q1 -0.148 [-0.317, +0.022]

## L7 placebos

- (i) State rows shuffled across days (seed 17): OOS R² -0.0158; Q5 − Q1 -0.052 [-0.226, +0.119]; in-sample R² 0.0163.
- (ii) Outcome shifted one day (Y_(t−1) on state_t; t−1 = the previous session, both in the universe; n 581): OOS R² +0.0581; Q5 − Q1 -0.345 [-0.507, -0.179]; in-sample R² 0.0819.
  - Spearman with Y_(t−1): L +0.235 [+0.157, +0.313]; K -0.107 [-0.185, -0.022]; C -0.140 [-0.224, -0.063]; T +0.140 [+0.060, +0.219]; M +0.152 [+0.072, +0.235]
  - Contemporaneous (past-outcome) fit is not prediction: a state that tracks Y_(t−1) as well as Y_t is persistence of the premium, not a 10:30 signal.
  - Persistence of Y itself: Spearman(Y_t, Y_(t−1)) +0.053 [-0.030, +0.133].

## L5 secondaries (not gated)

### (i) Upside vs downside (Almeida–Hizmeri)

- Implied up-share IV_up/(IV_up+IV_dn): median 0.416; realized up-share RV_up/RV: median 0.487, mean 0.493.
- (IV_up + IV_dn)/atmiv0²τ_rem median 1.042 (smile-integrated vs ATM; wings add variance).

| side premium (log RV_side / IV_side; seller earns −) | pooled | E0 | E1 |
|---|---|---|---|
| Y_up | -1.663 [-1.693, -1.632] | -1.673 [-1.716, -1.629] | -1.659 [-1.698, -1.620] |
| Y_dn | -1.980 [-2.020, -1.939] | -1.838 [-1.898, -1.778] | -2.032 [-2.083, -1.983] |
| Y_up − Y_dn (paired; < 0 = upside richer, Almeida's sign) | +0.317 [+0.291, +0.343] | +0.165 [+0.120, +0.210] | +0.373 [+0.343, +0.403] |

τ enters IV_up and IV_dn alike (session clock); the up − down difference is clock-robust to first order, the levels are not.

### (ii) Realized skewness of 1-min returns 10:30 → 16:00 by predicted-premium quintile (Vilkov)

| quintile | n | mean rskew |
|---|---|---|
| Q1 | 63 | +0.085 |
| Q2 | 84 | -0.104 |
| Q3 | 77 | -0.072 |
| Q4 | 76 | +0.035 |
| Q5 | 56 | -0.138 |

- Q5 − Q1 rskew -0.223 [-0.520, +0.072]; Spearman(rskew, Y) -0.253 [-0.326, -0.179].

### (iii) Fournier–Jacobs: Y_t vs the t−1 dealer-loss proxy

- Proxy: |close_move_over_im| − 0.798 on the previous session (`bt_daily_outcomes`, ES open→close over SPX `implied_move_1d`; the 06:33 PT `atmiv` it is built on understates the open straddle per CR-BP). n 590.
- Spearman(loss_(t−1), Y_t) +0.033 [-0.052, +0.112]. Mean Y after a loss day − after a no-loss day +0.025 [-0.052, +0.103] (n 147 / 443). FJ predict a richer premium (lower Y) after dealer losses: sign < 0.

## Flags (diagnostic)

- Q5 − Q1 excluding FOMC days -0.110 [-0.289, +0.075] (n 367); excluding 08:30 release days -0.183 [-0.364, +0.001] (n 314).
- Mean Y: FOMC days -1.748 (n 15); 08:30 days -1.750 (n 108); other -1.787.
