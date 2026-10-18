# CR-BQ — T-S2a: ES intraday autocorrelation by dealer net-gamma decile (single run, 2026-10-18 08:54)

Spec `specs/CR-BQ-gamma-autocorrelation.md` (L1–L8, Step 0 definitions, amendment A1). Train only (≤ 2026-06-05), read-only, holdout unread. G = `net_at_s0_linear` (CR-BK corrected profile at the prior session's 15:46 ET price; call OI +, put OI −). Bootstrap 10,000 resamples over days, `default_rng(17)` per statistic. Baltussen t = OLS with HC1 SE.

Universe 641 days; trailing-250 rank decile (min 120 prior sessions) on **541** (first 2023-10-24; **100 days carry no decile** and are excluded from decile statistics). G < 0 on 243 of 641 days (sign split uses all 641).

## 1. Primary — by trailing-rank decile of G (D1 = lowest rank)

| decile | n | median G ($bn/1 %) | share G < 0 | ρ5 [95 % CI] | ρ60 pooled [CI] | VR [CI] | skew [CI] |
|---|---|---|---|---|---|---|---|
| D1 | 48 | -37.6 | 100 % | -0.0203 [-0.0550, +0.0156] | +0.1637 [-0.0519, +0.3125] | +0.9644 [+0.8511, +1.0829] | +0.1234 [-0.0676, +0.3499] |
| D2 | 37 | -24.5 | 100 % | -0.0038 [-0.0456, +0.0380] | -0.0255 [-0.2011, +0.1442] | +1.0753 [+0.9421, +1.2231] | +0.0287 [-0.1441, +0.2076] |
| D3 | 59 | -14.7 | 100 % | -0.0152 [-0.0492, +0.0186] | +0.0546 [-0.0252, +0.1635] | +1.1116 [+1.0061, +1.2220] | +0.1621 [-0.0860, +0.5046] |
| D4 | 55 | -3.0 | 76 % | -0.0352 [-0.0729, +0.0019] | -0.0011 [-0.1164, +0.1083] | +0.9925 [+0.8825, +1.1090] | -0.1233 [-0.2708, +0.0203] |
| D5 | 47 | +4.3 | 9 % | -0.0073 [-0.0439, +0.0280] | +0.1243 [-0.0064, +0.2369] | +0.9657 [+0.8446, +1.0901] | -0.0288 [-0.2344, +0.1572] |
| D6 | 48 | +10.1 | 0 % | -0.0481 [-0.0924, -0.0049] | -0.0206 [-0.1128, +0.0753] | +0.9748 [+0.8607, +1.0929] | -0.2095 [-0.4441, +0.0197] |
| D7 | 42 | +16.3 | 0 % | -0.0638 [-0.1095, -0.0172] | -0.0534 [-0.2108, +0.0885] | +0.9890 [+0.8574, +1.1353] | -0.0846 [-0.2433, +0.0719] |
| D8 | 54 | +22.0 | 0 % | -0.0203 [-0.0604, +0.0195] | -0.0412 [-0.1233, +0.0346] | +0.9971 [+0.8701, +1.1303] | -0.1881 [-0.3108, -0.0623] |
| D9 | 66 | +27.7 | 0 % | -0.0519 [-0.0856, -0.0194] | -0.0144 [-0.1298, +0.1021] | +0.9538 [+0.8490, +1.0603] | -0.0913 [-0.2252, +0.0524] |
| D10 | 85 | +43.5 | 0 % | -0.0305 [-0.0590, -0.0025] | +0.0414 [-0.0470, +0.1237] | +0.9303 [+0.8526, +1.0130] | -0.0362 [-0.2234, +0.1895] |

**Gradient D1 − D10** (groups resampled independently):

| statistic | D1 − D10 [95 % CI] | n D1 / D10 | CI clear of 0 |
|---|---|---|---|
| (a) ρ5 | +0.0102 [-0.0354, +0.0560] | 48 / 85 | no |
| (b) ρ60 pooled | +0.1222 [-0.1193, +0.2957] | 48 / 85 | no |
| (c) VR 30/5 | +0.0341 [-0.1062, +0.1785] | 48 / 85 | no |
| (e) skew (secondary) | +0.1596 [-0.1392, +0.4529] | 48 / 85 | no |

ρ5 per +1 sd of raw G (all 641 days): **-0.0095** (t -1.81) [-0.0198, +0.0008]; per +1 sd of trailing rank (541 days): -0.0096 (t -1.66) [-0.0210, +0.0017]. Spearman(rank, ρ5) -0.065. Prior (Barbon–Buraschi Table VI): −0.015 to −0.020 per +1 sd.

**L5 conditional (a):** ρ5 on decile dummies + previous-session ρ5 + RV tercile + day of week (n 530): D1 − D10 **-0.0466** (HC1 t -1.71) bootstrap [-0.0994, +0.0081]; previous-session ρ5 coefficient -0.103 (t -2.30).

## 2. Sign split (raw G, all days)

| statistic | G < 0 | G ≥ 0 | (G < 0) − (G ≥ 0) [CI] |
|---|---|---|---|
| (a) ρ5 | -0.0150 [-0.0315, +0.0009] (n 243) | -0.0362 [-0.0499, -0.0226] (n 398) | +0.0212 [-0.0003, +0.0426] |
| (b) ρ60 pooled | +0.0979 [-0.0055, +0.2077] (n 243) | +0.0109 [-0.0340, +0.0541] (n 398) | +0.0870 [-0.0297, +0.2035] |
| (c) VR 30/5 | +1.0541 [+1.0021, +1.1075] (n 243) | +0.9609 [+0.9198, +1.0034] (n 398) | +0.0932 [+0.0276, +0.1608] |
| (e) skew (secondary) | +0.0515 [-0.0350, +0.1520] (n 243) | -0.1317 [-0.1987, -0.0625] (n 398) | +0.1831 [+0.0715, +0.3029] |

## 3. Baltussen regression — last 30 min (12:30–13:00 PT) on the earlier return

Slope (raw log-return units; ×100 not applied), HC1 t, pairs-bootstrap 95 % CI. Primary regressor (A1.3) = rROD: prior-session 12:59 PT close → 12:30 PT.

| regressor | sample | slope | t (HC1) | 95 % CI | n |
|---|---|---|---|---|---|
| **rROD (primary)** | all | -0.0111 | -0.64 | [-0.0479, +0.0198] | 629 |
| **rROD (primary)** | G < 0 | -0.0008 | -0.04 | [-0.0529, +0.0398] | 237 |
| **rROD (primary)** | G ≥ 0 | -0.0308 | -1.68 | [-0.0667, +0.0040] | 392 |
| **rROD (primary)** | (G < 0) × x interaction | +0.0300 | +1.01 | — | 629 |
| first 30 min 06:30–07:00 (frozen L4(d)) | all | +0.0167 | +0.31 | [-0.0880, +0.1198] | 641 |
| first 30 min 06:30–07:00 (frozen L4(d)) | G < 0 | +0.0731 | +0.96 | [-0.0821, +0.2154] | 243 |
| first 30 min 06:30–07:00 (frozen L4(d)) | G ≥ 0 | -0.0587 | -0.91 | [-0.1844, +0.0640] | 398 |
| first 30 min 06:30–07:00 (frozen L4(d)) | (G < 0) × x interaction | +0.1318 | +1.32 | — | 641 |
| Gao: prior close → 07:00 | all | -0.0187 | -0.88 | [-0.0603, +0.0234] | 629 |
| Gao: prior close → 07:00 | G < 0 | -0.0077 | -0.28 | [-0.0618, +0.0500] | 237 |
| Gao: prior close → 07:00 | G ≥ 0 | -0.0400 | -1.29 | [-0.1005, +0.0181] | 392 |
| Gao: prior close → 07:00 | (G < 0) × x interaction | +0.0322 | +0.78 | — | 629 |

L5 conditional rROD, G < 0 (+ previous-session last-30, RV tercile, DOW; n 236): slope -0.0081 (t -0.30) [-0.0692, +0.0362]; previous last-30 coefficient +0.050 (t +0.67).
L5 conditional rROD, G ≥ 0 (+ previous-session last-30, RV tercile, DOW; n 392): slope -0.0304 (t -1.61) [-0.0666, +0.0063]; previous last-30 coefficient -0.140 (t -1.83).
L5 conditional rROD, all (+ previous-session last-30, RV tercile, DOW; n 628): slope -0.0152 (t -0.82) [-0.0538, +0.0168]; previous last-30 coefficient -0.009 (t -0.16).

## 4. L6 verdict (as written; A1.3 primary regressor)

- (a) D1 − D10 ρ5 = +0.0102 [-0.0354, +0.0560] → not > 0 with CI clear of zero.
- (d) rROD slope: G < 0 -0.0008 (t -0.04); G ≥ 0 -0.0308 (t -1.68) → pass clause not met.
- Kill test: gradient CI clear of zero on (a) False, (b) False, (c) False; interaction t +1.01 → flat across sign.

**L6: KILL S2.**

## 5. L7 placebo — decile and sign labels permuted across days (`default_rng(17)`)

| statistic | D1 − D10, shuffled [CI] | CI clear of 0 |
|---|---|---|
| (a) ρ5 | +0.0227 [-0.0215, +0.0674] | no |
| (b) ρ60 pooled | +0.0841 [-0.1029, +0.2603] | no |
| (c) VR 30/5 | -0.0071 [-0.1633, +0.1467] | no |
| (e) skew (secondary) | -0.0103 [-0.1928, +0.1663] | no |

Shuffled rROD slope: G < 0 +0.0003 (t +0.01), G ≥ 0 -0.0163 (t -0.74), interaction t +0.46.

## 6. Secondary / diagnostic decile constructions (reported only, not gated)

| construction | n D1 / D10 | ρ5 D1 − D10 | ρ60 D1 − D10 | VR D1 − D10 | skew D1 − D10 |
|---|---|---|---|---|---|
| raw-dollar deciles, full sample (A1.2) | 65 / 64 | +0.0122 [-0.0306, +0.0570] | +0.0981 [-0.1489, +0.2864] | +0.0536 [-0.0768, +0.1906] | +0.0561 [-0.2821, +0.3650] |
| G / ES ADV deciles (A1.2) | 63 / 63 | +0.0261 [-0.0195, +0.0740] | -0.0040 [-0.1437, +0.1374] | +0.0926 [-0.0465, +0.2335] | +0.0651 [-0.2679, +0.3575] |
| B: trailing rank of G at ES-implied open (A1.1) | 44 / 91 | +0.0131 [-0.0336, +0.0610] | -0.0484 [-0.1593, +0.0695] | -0.0012 [-0.1341, +0.1375] | +0.2439 [-0.0435, +0.5178] |

`*` = CI clear of zero.

B sign split, rROD slope: G_B < 0 -0.0127 (t -0.49, n 235); G_B ≥ 0 -0.0115 (t -0.73, n 394); interaction t -0.04.
B sign split, ρ5 (G_B < 0) − (G_B ≥ 0): +0.0309 [+0.0094, +0.0527].

## 7. By year (A1.5 diagnostic; deciles from the full run, not re-cut)

| year | ranked days | n D1 / D10 | ρ5 D1 − D10 | ρ60 D1 − D10 | VR D1 − D10 | rROD slope G < 0 (t, n) | rROD slope G ≥ 0 (t, n) | interaction t |
|---|---|---|---|---|---|---|---|---|
| 2023 | 35 | 4 / 3 | -0.0904 [-0.3225, +0.0944] | -0.0085 [-0.1947, +0.3921] | -0.3300 [-1.0661, +0.2816] | +0.003 (+0.11, 61) | -0.037 (-1.34, 71) | +1.06 |
| 2024 | 207 | 14 / 28 | +0.0284 [-0.0597, +0.1253] | +0.2051 [-0.0632, +0.4088] | +0.0201 [-0.2055, +0.2565] | -0.037 (-1.27, 71) | -0.023 (-0.65, 132) | -0.32 |
| 2025 | 205 | 21 / 25 | +0.0018 [-0.0664, +0.0724] | +0.1986 [-0.1459, +0.4128] | +0.0734 [-0.1798, +0.3103] | +0.011 (+0.31, 75) | -0.049 (-1.37, 126) | +1.20 |
| 2026 H1 | 94 | 9 / 29 | +0.0146 [-0.0806, +0.1133] | -0.0294 [-0.2037, +0.1159] | +0.0632 [-0.2465, +0.4171] | -0.007 (-0.28, 30) | -0.002 (-0.05, 63) | -0.13 |

Decile composition by year (primary deciles):

| decile | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|
| D1 | 4 | 14 | 21 | 9 |
| D2 | 1 | 17 | 13 | 6 |
| D3 | 1 | 24 | 25 | 9 |
| D4 | 1 | 25 | 20 | 9 |
| D5 | 2 | 20 | 17 | 8 |
| D6 | 2 | 21 | 20 | 5 |
| D7 | 5 | 16 | 19 | 2 |
| D8 | 9 | 22 | 16 | 7 |
| D9 | 7 | 20 | 29 | 10 |
| D10 | 3 | 28 | 25 | 29 |

