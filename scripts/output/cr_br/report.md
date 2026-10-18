# CR-BR — T-S4c: term-structure slope at the close → next-day realized vs implied (single run, 2026-10-18 10:11)

Spec `specs/CR-BR-term-structure-slope.md` (L1–L8, Step 0, A1, A2). Train only, read-only, holdout unread. Implied inputs: latest ≤ 12:46 PT monies snapshot on t, CR-BL constant-maturity method. Outcome y = log(RV / IV7²·h/365), RV = ES 13:00 PT → 13:00 PT (CR-BO), total variance. Level-neutral y_ln = residual of y on log IV7². Bootstrap 10,000, `default_rng(17)` per statistic; h = 5 moving 5-day blocks.

**Sign convention (once):** positive slope = upward (back above front); Q1 = most negative / most inverted. Johnson's sign = Q1 − Q5 > 0 (realized exceeds implied more after an inverted curve).

A2.4 gate: inverted share by r_t quintile 95.5 % / 61.9 % / 10.3 % / 1.3 % / 0.0 % → PASS; r_t is the gated primary. Samples: x-complete 778, h = 1 621, h = 5 515. Level-neutral fits y = a + b·log IV7²: y1 b = +0.075; y5 b = -0.093; y1_prev b = +0.102; y5_prev b = -0.083.

## 0. Unconditional realized vs implied

- h = 1: mean log(RV / implied) +0.221 [+0.181, +0.260] (n 621); RV / implied = 1.247; share of days RV > implied 67.3 %.
- h = 5: mean log(RV / implied) +0.329 [+0.255, +0.409] (n 515); RV / implied = 1.389; share of days RV > implied 76.3 %.

## 1. What each slope definition is (A1.3, A2.3; 778 x-complete days)

| definition | Spearman with log IV7² | Spearman with 1[IV90 < IV7] | Spearman with r_t | inverted share Q1 … Q5 |
|---|---|---|---|---|
| r_t = log(IV30/IV7) — primary (A2.1) | -0.779 | -0.766 | +1.000 | 96 % / 62 % / 10 % / 1 % / 0 % |
| s_t = IV30 − IV7 — secondary | -0.757 | -0.766 | +0.995 | 97 % / 60 % / 10 % / 2 % / 0 % |
| PC2 of log IV² — Johnson-object column, reported only (A2.3) | -0.206 | -0.447 | +0.675 | 67 % / 45 % / 33 % / 16 % / 8 % |
| PC2 of IV² — frozen definition, failed sign check (Step 0), reported only | +0.391 | -0.006 | +0.179 | 40 % / 26 % / 33 % / 34 % / 35 % |

PC2 (log IV²) vs r_t quintile agreement: same quintile on 39.1 % of days; Spearman of quintile labels +0.635. PC2 (log) loadings 7…90 d: -0.645 -0.156 +0.266 +0.453 +0.534; variance share 3.56 %.

## 2.1 r_t = log(IV30/IV7) — primary (A2.1)

| h | outcome | Q1 | Q2 | Q3 | Q4 | Q5 | **Q1 − Q5** [95 % CI] | Spearman(slope, y) [CI] |
|---|---|---|---|---|---|---|---|---|
| 1 | level-neutral | +0.006 (n 117) | -0.000 (n 128) | -0.033 (n 117) | +0.034 (n 129) | -0.009 (n 130) | **+0.015 [-0.119, +0.150]** | +0.004 [-0.076, +0.085] |
| 1 | raw | +0.282 (n 117) | +0.240 (n 128) | +0.182 (n 117) | +0.230 (n 129) | +0.174 (n 130) | **+0.108 [-0.027, +0.243]** | -0.064 [-0.143, +0.018] |
| 5 | level-neutral | +0.052 (n 102) | -0.088 (n 115) | +0.014 (n 102) | +0.026 (n 104) | +0.007 (n 92) | **+0.044 [-0.260, +0.398]** | +0.046 [-0.116, +0.215] |
| 5 | raw | +0.319 (n 102) | +0.220 (n 115) | +0.353 (n 102) | +0.388 (n 104) | +0.381 (n 92) | **-0.062 [-0.366, +0.286]** | +0.119 [-0.043, +0.284] |

`*` = CI clear of zero. Per-quintile 95 % CIs in summary.json.

## 2.2 s_t = IV30 − IV7 — secondary

| h | outcome | Q1 | Q2 | Q3 | Q4 | Q5 | **Q1 − Q5** [95 % CI] | Spearman(slope, y) [CI] |
|---|---|---|---|---|---|---|---|---|
| 1 | level-neutral | -0.001 (n 119) | +0.006 (n 126) | -0.033 (n 117) | +0.037 (n 127) | -0.012 (n 132) | **+0.010 [-0.125, +0.146]** | -0.003 [-0.084, +0.078] |
| 1 | raw | +0.281 (n 119) | +0.240 (n 126) | +0.180 (n 117) | +0.231 (n 127) | +0.175 (n 132) | **+0.106 [-0.029, +0.242]** | -0.070 [-0.149, +0.012] |
| 5 | level-neutral | +0.037 (n 104) | -0.077 (n 113) | +0.007 (n 102) | +0.052 (n 100) | -0.012 (n 96) | **+0.049 [-0.247, +0.383]** | +0.045 [-0.118, +0.215] |
| 5 | raw | +0.296 (n 104) | +0.240 (n 113) | +0.349 (n 102) | +0.415 (n 100) | +0.358 (n 96) | **-0.062 [-0.355, +0.269]** | +0.118 [-0.044, +0.284] |

`*` = CI clear of zero. Per-quintile 95 % CIs in summary.json.

## 2.3 PC2 of log IV² — Johnson-object column, reported only (A2.3)

| h | outcome | Q1 | Q2 | Q3 | Q4 | Q5 | **Q1 − Q5** [95 % CI] | Spearman(slope, y) [CI] |
|---|---|---|---|---|---|---|---|---|
| 1 | level-neutral | +0.033 (n 121) | +0.071 (n 135) | -0.015 (n 124) | -0.077 (n 107) | -0.025 (n 134) | **+0.059 [-0.069, +0.184]** | -0.080 [-0.161, -0.001] * |
| 1 | raw | +0.264 (n 121) | +0.297 (n 135) | +0.202 (n 124) | +0.150 (n 107) | +0.180 (n 134) | **+0.083 [-0.044, +0.210]** | -0.095 [-0.176, -0.016] * |
| 5 | level-neutral | +0.054 (n 108) | +0.019 (n 124) | -0.018 (n 91) | -0.070 (n 89) | -0.003 (n 103) | **+0.056 [-0.222, +0.322]** | -0.039 [-0.197, +0.129] |
| 5 | raw | +0.375 (n 108) | +0.345 (n 124) | +0.308 (n 91) | +0.251 (n 89) | +0.346 (n 103) | **+0.029 [-0.246, +0.285]** | -0.037 [-0.190, +0.128] |

`*` = CI clear of zero. Per-quintile 95 % CIs in summary.json.

## 2.4 PC2 of IV² — frozen definition, failed sign check (Step 0), reported only

| h | outcome | Q1 | Q2 | Q3 | Q4 | Q5 | **Q1 − Q5** [95 % CI] | Spearman(slope, y) [CI] |
|---|---|---|---|---|---|---|---|---|
| 1 | level-neutral | +0.022 (n 117) | +0.095 (n 125) | +0.029 (n 127) | -0.094 (n 122) | -0.053 (n 130) | **+0.075 [-0.063, +0.214]** | -0.103 [-0.184, -0.021] * |
| 1 | raw | +0.234 (n 117) | +0.299 (n 125) | +0.251 (n 127) | +0.130 (n 122) | +0.190 (n 130) | **+0.044 [-0.096, +0.186]** | -0.074 [-0.155, +0.008] |
| 5 | level-neutral | -0.006 (n 99) | +0.176 (n 95) | -0.006 (n 110) | -0.076 (n 108) | -0.070 (n 103) | **+0.065 [-0.184, +0.296]** | -0.116 [-0.261, +0.046] |
| 5 | raw | +0.336 (n 99) | +0.518 (n 95) | +0.324 (n 110) | +0.250 (n 108) | +0.235 (n 103) | **+0.100 [-0.139, +0.330]** | -0.158 [-0.298, +0.001] |

`*` = CI clear of zero. Per-quintile 95 % CIs in summary.json.

## 3. L6 verdict (as written, A2.5)

- Pass clause, r_t level-neutral: h = 1 Q1 − Q5 +0.015 [-0.119, +0.150]; h = 5 +0.044 [-0.260, +0.398] → not met (needs > 0 with CI clear of zero at h = 1 or h = 5).
- Kill clause, level-neutral, CI spans zero at both horizons on both r_t and s_t: r h1 spans; r h5 spans; s h1 spans; s h5 spans → met.

**L6: KILL S4's slope signal (not S4).**

## 4. L7 placebos

**(i) quintile labels permuted across days** (`default_rng(17)`, per horizon universe):

| slope | h | outcome | Q1 − Q5 shuffled [CI] |
|---|---|---|---|
| r | 1 | level-neutral | -0.076 [-0.199, +0.049] |
| r | 1 | raw | -0.079 [-0.201, +0.046] |
| r | 5 | level-neutral | -0.034 [-0.210, +0.146] |
| r | 5 | raw | -0.031 [-0.203, +0.145] |
| s | 1 | level-neutral | -0.059 [-0.178, +0.062] |
| s | 1 | raw | -0.062 [-0.180, +0.058] |
| s | 5 | level-neutral | -0.022 [-0.205, +0.153] |
| s | 5 | raw | -0.018 [-0.196, +0.154] |

**(ii) outcome shifted one day earlier** (night t−1 → t, or nights t−5 … t−1 → t, normalised by IV7(t)²; slope from t; contemporaneous, not predictive):

| slope | h | outcome | n | Q1 − Q5 [CI] | Spearman [CI] |
|---|---|---|---|---|---|
| r | 1 | level-neutral | 593 | -0.041 [-0.176, +0.089] | +0.035 [-0.046, +0.118] |
| r | 1 | raw | 593 | +0.083 [-0.053, +0.217] | -0.053 [-0.134, +0.030] |
| r | 5 | level-neutral | 422 | -0.183 [-0.417, +0.094] | +0.204 [+0.023, +0.369] * |
| r | 5 | raw | 422 | -0.278 [-0.513, -0.013] * | +0.279 [+0.104, +0.437] * |
| s | 1 | level-neutral | 593 | -0.031 [-0.164, +0.100] | +0.021 [-0.061, +0.103] |
| s | 1 | raw | 593 | +0.098 [-0.037, +0.230] | -0.066 [-0.147, +0.017] |
| s | 5 | level-neutral | 422 | -0.154 [-0.402, +0.121] | +0.193 [+0.012, +0.358] * |
| s | 5 | raw | 422 | -0.253 [-0.497, +0.012] | +0.268 [+0.092, +0.426] * |

## 5. By year (diagnostic; full-sample quintiles)

| year | slope | h | outcome | n Q1 / Q5 | Q1 − Q5 [CI] | Spearman |
|---|---|---|---|---|---|---|
| 2023 | r | 1 | level-neutral | 13 / 14 | -0.243 [-0.511, +0.005] | +0.207 |
| 2023 | r | 1 | raw | 13 / 14 | -0.161 [-0.426, +0.086] | +0.132 |
| 2023 | r | 5 | level-neutral | 12 / 4 | -0.285 [-0.444, -0.125] * | +0.269 |
| 2023 | r | 5 | raw | 12 / 4 | -0.364 [-0.510, -0.202] * | +0.352 |
| 2023 | s | 1 | level-neutral | 10 / 13 | -0.285 [-0.578, +0.002] | +0.202 |
| 2023 | s | 1 | raw | 10 / 13 | -0.199 [-0.488, +0.087] | +0.129 |
| 2023 | s | 5 | level-neutral | 10 / 6 | -0.342 [-0.406, -0.297] * | +0.267 |
| 2023 | s | 5 | raw | 10 / 6 | -0.421 [-0.501, -0.354] * | +0.351 |
| 2024 | r | 1 | level-neutral | 44 / 37 | -0.134 [-0.376, +0.103] | +0.053 |
| 2024 | r | 1 | raw | 44 / 37 | -0.054 [-0.291, +0.179] | -0.002 |
| 2024 | r | 5 | level-neutral | 40 / 28 | -0.055 [-0.383, +0.471] | +0.121 |
| 2024 | r | 5 | raw | 40 / 28 | -0.146 [-0.489, +0.388] | +0.178 |
| 2024 | s | 1 | level-neutral | 43 / 33 | -0.125 [-0.374, +0.120] | +0.034 |
| 2024 | s | 1 | raw | 43 / 33 | -0.042 [-0.286, +0.198] | -0.020 |
| 2024 | s | 5 | level-neutral | 38 / 24 | +0.016 [-0.335, +0.549] | +0.107 |
| 2024 | s | 5 | raw | 38 / 24 | -0.077 [-0.441, +0.464] | +0.162 |
| 2025 | r | 1 | level-neutral | 50 / 53 | +0.219 [+0.004, +0.446] * | -0.127 |
| 2025 | r | 1 | raw | 50 / 53 | +0.330 [+0.111, +0.561] * | -0.209 |
| 2025 | r | 5 | level-neutral | 41 / 42 | +0.375 [-0.146, +1.017] | -0.188 |
| 2025 | r | 5 | raw | 41 / 42 | +0.245 [-0.270, +0.873] | -0.104 |
| 2025 | s | 1 | level-neutral | 54 / 56 | +0.231 [+0.026, +0.445] * | -0.118 |
| 2025 | s | 1 | raw | 54 / 56 | +0.344 [+0.137, +0.560] * | -0.201 |
| 2025 | s | 5 | level-neutral | 45 / 45 | +0.333 [-0.175, +0.958] | -0.170 |
| 2025 | s | 5 | raw | 45 / 45 | +0.200 [-0.308, +0.818] | -0.085 |
| 2026 H1 | r | 1 | level-neutral | 10 / 26 | +0.013 [-0.304, +0.316] | +0.041 |
| 2026 H1 | r | 1 | raw | 10 / 26 | +0.112 [-0.198, +0.403] | -0.015 |
| 2026 H1 | r | 5 | level-neutral | 9 / 18 | -0.270 [-0.641, -0.066] * | +0.311 |
| 2026 H1 | r | 5 | raw | 9 / 18 | -0.396 [-0.777, -0.177] * | +0.412 |
| 2026 H1 | s | 1 | level-neutral | 12 / 30 | -0.112 [-0.458, +0.220] | +0.039 |
| 2026 H1 | s | 1 | raw | 12 / 30 | -0.013 [-0.352, +0.312] | -0.015 |
| 2026 H1 | s | 5 | level-neutral | 11 / 21 | -0.302 [-0.626, +0.013] | +0.307 |
| 2026 H1 | s | 5 | raw | 11 / 21 | -0.428 [-0.770, -0.093] * | +0.407 |

