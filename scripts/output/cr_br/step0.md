# CR-BR — Step 0 diagnosis (read-only; 2026-10-18 09:29)

## Q3 — PM-expiry bracketing per tenor (CR-BL method, ≤ 12:46 PT snapshot on t)

Train sessions 778 (2023-05-01 → 2026-06-05); with a ≤ 12:46 PT snapshot 778.

| tenor | bracketed days | share of snapshot days | exact node | median bracket width (days) |
|---|---|---|---|---|
| 7 d | 778 | 100.0 % | 718 | 0 |
| 14 d | 778 | 100.0 % | 714 | 0 |
| 30 d | 778 | 100.0 % | 415 | 0 |
| 60 d | 778 | 100.0 % | 52 | 21 |
| 90 d | 778 | 100.0 % | 22 | 31 |

All five bracketed: 778 of 778 (100.0 %). Longest PM node: median 329 d, min 283 d. Shortest node ≥ 1 d: median 1 d, max 5 d.
Snapshot minute distribution (top 3): {'12:46:00': 765, '10:15:00': 6, '12:45:00': 3}; earliest 10:06:00.

## Q2 — atmiv vs vol50 at the ≤ 12:46 PT snapshot, constant maturity, by era (STOP gate)

Ratio = CM IV from `atmiv` nodes / CM IV from `vol50` nodes (same snapshot, same bracketing, CR-BL method). Gate (CR-BP item 5 / P3(e) band): both era medians inside 0.95–1.05 and |E1 − E0| ≤ 0.05; otherwise the open-time break is present at the close → STOP.

| tenor | era | n | median | IQR | p5–p95 | share outside 0.95–1.05 |
|---|---|---|---|---|---|---|
| 7 d | E0 (< 2024-03-01) | 210 | 1.0011 | 0.9964–1.0084 | 0.9764–1.0200 | 1.0 % |
| 7 d | E1 (≥ 2024-03-01) | 568 | 1.0110 | 0.9998–1.0246 | 0.9772–1.0505 | 6.9 % |
| 7 d | E1 − E0 | | +0.0098 | | | gate pass |
| 30 d | E0 (< 2024-03-01) | 210 | 0.9967 | 0.9930–1.0013 | 0.9764–1.0095 | 2.4 % |
| 30 d | E1 (≥ 2024-03-01) | 568 | 1.0074 | 0.9989–1.0172 | 0.9809–1.0333 | 2.1 % |
| 30 d | E1 − E0 | | +0.0106 | | | gate pass |

**Era gate: PASS — no close-time era break; continue.**

Monthly median ratio around the CR-BP break: 2023-11 1.005/1.000; 2023-12 1.001/0.997; 2024-01 1.003/0.997; 2024-02 1.000/0.997; 2024-03 1.007/0.999; 2024-04 1.002/1.005; 2024-05 1.024/1.002; 2024-06 1.009/1.007 (7 d / 30 d).

## Q1 — L1 universe, counts after each exclusion

ES bars 1093859 (2023-04-25 20:00:00 → 2026-06-05 15:59:00 ET); contract switches detected (CR-BO rule): 12.

| step (h = 1) | n after |
|---|---|
| train sessions t (2023-05-01 → 2026-06-05) | 778 |
| − no ≤ 12:46 PT monies snapshot on t | 778 |
| − 7 d not bracketed by PM expiries | 778 |
| − 14 d not bracketed by PM expiries | 778 |
| − 30 d not bracketed by PM expiries | 778 |
| − 60 d not bracketed by PM expiries | 778 |
| − 90 d not bracketed by PM expiries | 778 |
| − no next session ≤ 2026-06-05 (t+1 in holdout) | 777 |
| − early close on t or t+1 | 761 |
| − ES roll week on t or t+1 (all years) | 632 |
| − contract switch inside 13:00 PT t → 13:00 PT t+1 | 632 |
| − no ES anchor bar 15:55–15:59 ET on t | 631 |
| − > 5 % overnight minutes missing | 623 |
| − > 5 % RTH minutes missing on t+1 (i.e. < 371 of 390; implies ≥ 370) | 621 |
| **h = 5:** all five nights t … t+4 pass and t+5 ≤ 2026-06-05 | **515** |

x-complete train days (PCA / quintile sample): **778**. h = 1 universe **621**; h = 5 universe **515**.
RTH minutes on t+1 in the h = 1 universe: min 376 of 390.

By year (h = 1 / h = 5): 2023 131 / 104; 2024 201 / 170; 2025 199 / 167; 2026 90 / 74.

## Q4 — PCA of x_t = (IV7², IV14², IV30², IV60², IV90²) on the x-complete train sample

| tenor | PC1 loading | PC2 loading | PC3 loading |
|---|---|---|---|
| 7 d | +0.673 | -0.589 | -0.437 |
| 14 d | +0.517 | -0.005 | +0.687 |
| 30 d | +0.378 | +0.343 | +0.289 |
| 60 d | +0.281 | +0.490 | -0.220 |
| 90 d | +0.241 | +0.544 | -0.452 |

Variance explained: PC1 96.94 %, PC2 2.75 %, PC3 0.22 %. Sign fixed so that the PC2 loading on 90 d minus on 7 d is positive (L3).
Spearman(PC2, s_t = IV30 − IV7) on 778 days: +0.194. PC2 < 0 on 412 days; s_t < 0 on 312 days.

**Worked day 2025-04-08** (snapshot 2025-04-08 12:46:00): IV 7/14/30/60/90 = 61.18 / 51.01 / 42.37 / 35.45 / 32.25 vol pts; s_t = -18.81; PC2 = -0.05742 → inverted (front above back), negative PC2 as required — consistent: True. Brackets 7 d [7, 7], 90 d [83, 114].
**Worked day 2024-05-17** (snapshot 2024-05-17 12:46:00): IV 7/14/30/60/90 = 8.86 / 8.48 / 9.87 / 10.45 / 10.94 vol pts; s_t = +1.01; PC2 = -0.00486 → inverted (front above back), negative PC2 as required — consistent: False. Brackets 7 d [7, 7], 90 d [75, 105].

## Q5 — what each slope definition sorts on (train, x-complete days; no outcome used)

Candidate loadings (7…90 d): log IV² -0.645 -0.156 +0.266 +0.453 +0.534 (PC2 share 3.56 %); shape-only -0.570 +0.634 +0.390 -0.133 -0.321 (PC2 share 7.88 %).

| definition | share IV90 < IV7 by quintile Q1 … Q5 | median IV7 Q1 / Q5 | Spearman with log IV7² | Spearman with IV90 − IV7 |
|---|---|---|---|---|
| (a) PC2 of IV² (L3 as written) | 40 % / 26 % / 33 % / 34 % / 35 % | 11.0 / 14.3 | +0.391 | +0.155 |
| (b) s_t = IV30 − IV7 (L3) | 97 % / 60 % / 10 % / 2 % / 0 % | 18.5 / 10.2 | -0.757 | +0.934 |
| (c) candidate: PC2 of log IV² | 67 % / 45 % / 33 % / 16 % / 8 % | 12.7 / 11.1 | -0.206 | +0.688 |
| (d) candidate: PC2 of IV² / row-mean IV² (shape only) | 35 % / 32 % / 35 % / 34 % / 33 % | 12.3 / 12.1 | +0.012 | +0.054 |

L3 requires Q1 = most inverted and positive PC2 = back above front. A definition meets it when the inverted share falls from Q1 to Q5.

