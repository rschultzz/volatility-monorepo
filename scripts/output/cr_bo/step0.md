# CR-BO Step 0 — 2026-10-17 21:02 PT

Read-only. Train: pairs (t, t+1) with t ≥ 2023-05-01, t+1 ≤ 2026-06-05. Times ET unless tagged.

## Q2 ES contract switches and roll weeks

Detector: day-over-day change in (ES 15:59 bar close − SPX `stock_price` at the 15:59 row, nearest PM expiry) > 30 pts. Δbasis over 751 sessions: sd 9.09, largest non-switch |Δ| 27.36.

| switch detected on (session) | previous session | Δbasis (pts) | in a roll week? | switch minute (ET) | source |
|---|---|---|---|---|---|
| 2023-06-20 | 2023-06-15 | +35.58 | yes | 2023-06-18 Sun 18:00 (1-min r +0.0018) | databento |
| 2023-09-18 | 2023-09-14 | +46.85 | yes | 2023-09-17 Sun 18:00 (1-min r +0.0040) | databento |
| 2023-12-18 | 2023-12-14 | +46.80 | yes | 2023-12-17 Sun 18:00 (1-min r +0.0124) | databento |
| 2024-03-18 | 2024-03-14 | +60.29 | yes | 2024-03-17 Sun 18:00 (1-min r +0.0131) | databento |
| 2024-06-24 | 2024-06-20 | +60.06 | yes | 2024-06-23 Sun 18:00 (1-min r +0.0134) | databento |
| 2024-09-23 | 2024-09-19 | +53.29 | yes | 2024-09-22 Sun 18:00 (1-min r +0.0102) | databento |
| 2024-12-23 | 2024-12-19 | +64.34 | yes | 2024-12-22 Sun 18:00 (1-min r +0.0300) | databento |
| 2025-03-24 | 2025-03-20 | +40.43 | yes | 2025-03-23 Sun 18:00 (1-min r +0.0230) | databento |
| 2025-06-23 | 2025-06-18 | +46.54 | yes | 2025-06-22 Sun 18:00 (1-min r -0.0055) | databento |
| 2025-09-22 | 2025-09-18 | +53.98 | yes | 2025-09-21 Sun 18:00 (1-min r +0.0094) | databento |
| 2025-12-22 | 2025-12-18 | +49.42 | yes | 2025-12-21 Sun 18:00 (1-min r +0.0157) | databento |
| 2026-03-20 | 2026-03-19 | +56.81 | yes | 2026-03-20 Fri 09:30 (1-min r +0.0062) | ironbeam |

12 switches = one per quarterly expiry in the train window. Databento `ES.c.0` (2023–2025) switches at the Sunday 18:00 reopen after expiry Friday and prints **no RTH bars on expiry Friday** (dying contract stops at the 09:30 SOQ); the Ironbeam ingest (2026-03) switched at 09:30 on expiry Friday.


## Q1 L1 coverage — nights after each exclusion

Exclusions applied in this order (a night removed by an earlier step is not counted again):

| step | nights | S1 | S2 | S7 |
|---|---|---|---|---|
| all session pairs (t, t+1), t ≥ 2023-05-01, t+1 ≤ 2026-06-05 | 777 | 622 | 617 | 648 |
| − early close on t or t+1 | 761 | 622 | 617 | 648 |
| − ES roll week on t or t+1 (Mon of week before 3rd Fri → expiry Fri; Mar/Jun/Sep/Dec) | 632 | 509 | 506 | 526 |
| − window 16:00 t → 16:00 t+1 contains a contract switch (not already in a roll week) | 632 | 509 | 506 | 526 |
| − no ES anchor bar 15:55–15:59 on t | 631 | 508 | 506 | 525 |
| − ES overnight missing > 5% of expected minutes | 623 | 502 | 500 | 519 |
| − ES RTH (t+1) missing > 5% (needed for RV_rth; L1 interpretation) | 621 | 500 | 498 | 517 |

Bucket columns require the monies rows of L1/L3: atmiv > 0 at 15:59 on t, 09:31 on t+1 and 15:59 on t+1, same `expir_date`, PM. Monies-missing nights within the final ES-clean set: S1 121, S2 123, S7 104.

Final universe by weekday of t: {'Mon': 121, 'Tue': 132, 'Wed': 128, 'Thu': 126, 'Fri': 114}; holiday-spanning nights kept: 13; calendar-day gap t→t+1: {1: 505, 2: 1, 3: 103, 4: 12}.

Removed by roll week: 129 nights; by a switch outside a roll week: 0; switch nights total (any overlap): 12, all of which are also roll-week nights: True.

Early-close pairs removed: 2023-06-30→2023-07-03, 2023-07-03→2023-07-05, 2023-11-22→2023-11-24, 2023-11-24→2023-11-27, 2024-07-02→2024-07-03, 2024-07-03→2024-07-05, 2024-11-27→2024-11-29, 2024-11-29→2024-12-02, 2024-12-23→2024-12-24, 2024-12-24→2024-12-26, 2025-07-02→2025-07-03, 2025-07-03→2025-07-07, 2025-11-26→2025-11-28, 2025-11-28→2025-12-01, 2025-12-23→2025-12-24, 2025-12-24→2025-12-26

S1 monies-missing nights: 2023-05-12, 2023-05-18, 2023-05-19, 2023-05-22, 2023-05-23, 2023-07-20, 2023-08-17, 2023-10-11, 2023-10-12, 2023-10-19, 2023-11-16, 2023-11-28, 2024-01-18, 2024-02-15, 2024-02-16, 2024-02-20, 2024-02-21, 2024-04-02, 2024-04-03, 2024-04-04, 2024-04-05, 2024-04-08, 2024-04-09, 2024-04-10, 2024-04-11, 2024-04-12, 2024-04-16, 2024-04-17, 2024-04-18, 2024-04-22, 2024-04-23, 2024-04-25, 2024-04-26, 2024-04-29, 2024-05-01, 2024-05-02, 2024-05-03, 2024-05-06, 2024-05-07, 2024-05-08, 2024-05-09, 2024-05-13, 2024-05-14, 2024-05-16, 2024-05-17, 2024-05-21, 2024-05-23, 2024-05-24, 2024-05-28, 2024-05-29, 2024-05-30, 2024-05-31, 2024-06-03, 2024-06-04, 2024-06-06, 2024-07-10, 2024-07-18, 2024-07-24, 2024-07-25, 2024-08-01, 2024-08-06, 2024-08-07, 2024-08-08, 2024-08-09, 2024-08-12, 2024-08-15, 2024-09-03, 2024-09-04, 2024-09-05, 2024-10-01, 2024-10-02, 2024-10-03, 2024-10-07, 2024-10-17, 2024-10-24, 2024-10-25, 2024-10-28, 2024-10-29, 2024-10-31, 2024-11-01, 2024-11-04, 2024-11-14, 2024-11-15, 2024-11-26, 2024-12-03, 2025-01-02, 2025-01-07, 2025-01-08, 2025-01-10, 2025-01-13, 2025-01-16, 2025-01-21, 2025-01-22, 2025-01-24, 2025-01-27, 2025-01-31, 2025-02-03, 2025-02-04, 2025-02-07, 2025-02-10, 2025-02-11, 2025-02-12, 2025-02-13, 2025-02-20, 2025-02-21, 2025-02-24, 2025-02-26, 2025-02-28, 2025-03-03, 2025-03-04, 2025-03-05, 2025-04-16, 2025-05-07, 2025-05-08, 2025-10-16, 2025-10-21, 2025-11-25, 2026-01-15, 2026-02-19, 2026-04-16, 2026-05-14

Reasons, per bucket, on ES-clean nights (a night can have more than one):

| bucket | no 09:31 snapshot at all on t+1 | 15:59 row on t: only an AM row for that expir_date | 15:59 row on t absent otherwise | 09:31 row absent with snapshot present | 15:59 row on t+1 absent | no expiry fits (S7) |
|---|---|---|---|---|---|---|
| S1 | 92 | 21 | 6 | 7 | 6 | 0 |
| S2 | 92 | 20 | 6 | 20 | 26 | 0 |
| S7 | 91 | 0 | 0 | 1 | 6 | 6 |

ES-clean nights whose t+1 has no 09:31 snapshot: 92; by month: {'2024-04': 16, '2024-05': 16, '2025-02': 12, '2025-01': 9, '2024-08': 6, '2024-10': 6, '2024-11': 5, '2024-06': 4}. `orats_monies_minute` is keyed (ticker, expir_date, quote_date), so on a third-Friday expir_date the AM monthly and PM weekly cannot both be stored; the table holds the AM row and the PM S1 on Thursday nights before a monthly expiry is unavailable. L1/L2 exclude both cases as written (no amendment).

## Q5 ES minutes missing, by year (all pairs that pass early-close / roll / switch)

| year | nights | overnight expected min (median) | missing % median | p95 | max | nights > 5 % | RTH missing % median | RTH nights > 5 % |
|---|---|---|---|---|---|---|---|---|
| 2023 | 132 | 990 | 0.00 | 0.40 | 36.57 | 1 | 0.00 | 0 |
| 2024 | 203 | 990 | 0.00 | 0.30 | 36.36 | 2 | 0.00 | 0 |
| 2025 | 201 | 990 | 0.00 | 0.10 | 36.36 | 2 | 0.00 | 0 |
| 2026 | 96 | 990 | 0.00 | 0.66 | 38.18 | 4 | 0.00 | 4 |

Nights failing coverage: 2023-12-22→2023-12-26 on 36.6% rth 0.0%; 2023-12-29→2024-01-02 on 36.4% rth 0.0%; 2024-03-28→2024-04-01 on 26.7% rth 0.0%; 2024-12-31→2025-01-02 on 36.4% rth 0.0%; 2025-04-17→2025-04-21 on 26.7% rth 0.0%; 2025-12-31→2026-01-02 on 38.2% rth 5.1%; 2026-02-12→2026-02-13 on 0.0% rth 7.4%; 2026-02-13→2026-02-17 on 0.0% rth 27.7%; 2026-02-17→2026-02-18 on 17.6% rth 21.3%; 2026-02-18→2026-02-19 on 6.1% rth 0.0% no-anchor; 2026-05-07→2026-05-08 on 6.2% rth 4.1%

Expected-minute grid: CME equity Globex Sun 18:00 → Fri 17:00 ET, daily halt 17:00–17:59, bar-start stamps; on holiday-spanning nights the NYSE-holiday calendar date is dropped from the grid (CME runs a holiday schedule the grid does not model); bars that exist that day still enter RV.

## Q3 ORATS τ convention for atmiv

Method: cached `orats_options_minute` call/put pairs at the same strike (|K − stock_price| ≤ 10), forward from parity F = K + C − P, IV = mean of the pair's `mid_iv`; solve Black (r = 0) for the τ that reproduces the call mid; compare with calendar time to 16:00 ET on expiry / 365.

| calendar DTE | snapshot (ET) | n | median τ_ORATS / τ_calendar | IQR |
|---|---|---|---|---|
| 0 | 09:31 | 196 | 3.889 | 1.008–8.261 |
| 0 | 15:31 | 191 | 2.490 | 2.458–2.537 |
| 0 | 15:59 | 240 | 67.641 | 40.842–108.206 |
| 4–7 | 09:31 | 6 | 1.003 | 1.002–1.003 |
| 4–7 | 09:59 | 9 | 1.002 | 1.001–1.002 |
| 4–7 | 15:31 | 15 | 1.011 | 1.002–1.019 |
| 4–7 | 15:59 | 15 | 1.012 | 1.002–1.017 |
| 8–30 | 09:31 | 38 | 0.999 | 0.997–1.073 |
| 8–30 | 09:59 | 46 | 1.000 | 0.998–1.027 |
| 8–30 | 15:31 | 78 | 1.000 | 0.998–1.026 |
| 8–30 | 15:59 | 75 | 1.020 | 0.999–1.025 |
| 31+ | 09:31 | 1 | 0.996 | 0.996–0.996 |
| 31+ | 09:59 | 1 | 0.995 | 0.995–0.995 |

**Worked example.** 2025-10-24 09:31 ET, expiry 2025-11-13 PM, K 6800: parity F 6793.15, call mid 75.450, pair mid_iv 0.12338. Calendar time to 2025-11-13 16:00 ET = 20.2701 days; the τ that reproduces the mid = 20.2682 days (ratio 0.9999).

**Reading.** For 4–30 calendar-DTE contracts at both the 09:31 and 15:59 snapshots, ORATS IV is consistent with τ = calendar time to 16:00 ET on expiry / 365 (median ratio 1.000–1.020). No 1–3 DTE pairs are cached, so the S1 / S2 15:59 rows cannot be checked directly; calendar/365 is used for them (L3 fallback). 0DTE IV departs from calendar time in the last ~90 minutes (ORATS runs a longer clock near expiry); in this test that touches only v_c of S1 (the 15:59 row on its expiry day, τ = 1 minute), whose iv²τ is ≈ 0 under either clock. **L3 τ used: (16:00 ET on expir_date − snapshot label time) / 365 days.**

`orats_monies_minute` carries no fractional-τ column: `dte` = calendar days + 1 (1 on expiry day) and `dte_pct` = (dte − 1)/365.25, identical at 09:31 and 15:59.

**atmiv refresh cadence.** Over 7236 (date, PM expiry ≤ 15 DTE) rows: atmiv at the 09:31 row equals the 09:30 row on 85.9 %. Quote time (`quote_date`, ET) of the 09:31 row: {'09:30': 5790, '09:25': 795, '09:15': 316, '09:27': 240}; of the 15:59 row: {'15:58': 5408, '15:56': 894, '15:57': 539, '15:59': 222}. Median atmiv / vol50 at 09:31: 0.9888.

## Q4 vol25 / vol75 mapping

ORATS monies smile is indexed by **call delta** (`vol5` … `vol95`; repo convention `packages/shared/vol_features.py:20-21`: `vol25` = 25-delta call IV, `vol75` = 75-delta call = 25-delta put IV). Check on the data: at 15:59, `vol75 > vol25` on 96.5 % of PM rows ≤ 15 DTE (n 8355), `vol75 > vol50` on 97.1 %, `vol25 < vol50` on 93.2 % — the downward SPX skew puts the higher vol on the low-strike side, i.e. call delta 0.75. **L6(d): put side = `vol75`, call side = `vol25`.**

## Q6 first t per weekday with a next-session (S1) PM expiry

| weekday of t | first t with S1 rows | nights with S1 ok / all pairs |
|---|---|---|
| Mon | 2023-05-01 | 125 / 146 |
| Tue | 2023-05-02 | 130 / 161 |
| Wed | 2023-05-03 | 132 / 159 |
| Thu | 2023-05-04 | 104 / 154 |
| Fri | 2023-05-05 | 131 / 157 |

Daily SPX expiries for every weekday predate the train window; every weekday is covered from the first train session.

## Q7 event dates (L6(c)) — hand-coded in `scripts/cr_bo_common.py`

- **CPI** 08:30 ET (36): 2023-05-10, 2023-06-13, 2023-07-12, 2023-08-10, 2023-09-13, 2023-10-12, 2023-11-14, 2023-12-12, 2024-01-11, 2024-02-13, 2024-03-12, 2024-04-10, 2024-05-15, 2024-06-12, 2024-07-11, 2024-08-14, 2024-09-11, 2024-10-10, 2024-11-13, 2024-12-11, 2025-01-15, 2025-02-12, 2025-03-12, 2025-04-10, 2025-05-13, 2025-06-11, 2025-07-15, 2025-08-12, 2025-09-11, 2025-10-24, 2025-12-18, 2026-01-13, 2026-02-13, 2026-03-11, 2026-04-10, 2026-05-12
- **PPI** 08:30 ET (36): 2023-05-11, 2023-06-14, 2023-07-13, 2023-08-11, 2023-09-14, 2023-10-11, 2023-11-15, 2023-12-13, 2024-01-12, 2024-02-16, 2024-03-14, 2024-04-11, 2024-05-14, 2024-06-13, 2024-07-12, 2024-08-13, 2024-09-12, 2024-10-11, 2024-11-14, 2024-12-12, 2025-01-14, 2025-02-13, 2025-03-13, 2025-04-11, 2025-05-15, 2025-06-12, 2025-07-16, 2025-08-14, 2025-09-10, 2025-11-25, 2026-01-14, 2026-01-30, 2026-02-27, 2026-03-18, 2026-04-14, 2026-05-13
- **NFP** 08:30 ET (37): 2023-05-05, 2023-06-02, 2023-07-07, 2023-08-04, 2023-09-01, 2023-10-06, 2023-11-03, 2023-12-08, 2024-01-05, 2024-02-02, 2024-03-08, 2024-04-05, 2024-05-03, 2024-06-07, 2024-07-05, 2024-08-02, 2024-09-06, 2024-10-04, 2024-11-01, 2024-12-06, 2025-01-10, 2025-02-07, 2025-03-07, 2025-04-04, 2025-05-02, 2025-06-06, 2025-07-03, 2025-08-01, 2025-09-05, 2025-11-20, 2025-12-16, 2026-01-09, 2026-02-11, 2026-03-06, 2026-04-03, 2026-05-08, 2026-06-05
- **PCE** 08:30 ET (32): 2023-05-26, 2023-06-30, 2023-07-28, 2023-08-31, 2023-09-29, 2023-10-27, 2023-11-30, 2023-12-22, 2024-01-26, 2024-02-29, 2024-03-29, 2024-04-26, 2024-05-31, 2024-06-28, 2024-07-26, 2024-08-30, 2024-09-27, 2024-10-31, 2024-12-20, 2025-01-31, 2025-02-28, 2025-03-28, 2025-05-30, 2025-06-27, 2025-07-31, 2025-08-29, 2025-09-26, 2026-02-20, 2026-03-13, 2026-04-09, 2026-04-30, 2026-05-28
- **PCE at 10:00 ET, not event nights**: 2024-11-27, 2025-04-30, 2025-12-05, 2026-01-22
- **FOMC statement days** (25): 2023-05-03, 2023-06-14, 2023-07-26, 2023-09-20, 2023-11-01, 2023-12-13, 2024-01-31, 2024-03-20, 2024-05-01, 2024-06-12, 2024-07-31, 2024-09-18, 2024-11-07, 2024-12-18, 2025-01-29, 2025-03-19, 2025-05-07, 2025-06-18, 2025-07-30, 2025-09-17, 2025-10-29, 2025-12-10, 2026-01-28, 2026-03-18, 2026-04-29

Distinct 08:30 event dates: 141. In the final universe as t+1: 107 event nights, 15 FOMC-on-t+1 nights.

Sources: CPI https://www.bls.gov/bls/news-release/cpi.htm · PPI https://www.bls.gov/bls/news-release/ppi.htm · NFP https://www.bls.gov/bls/news-release/empsit.htm · PCE 2023 https://apps.bea.gov/scb/issues/2022/12-december/1222-news-releases-2023.htm, 2024 https://apps.bea.gov/scb/issues/2023/12-december/1223-news-releases-2024.htm, 2025 https://www.bea.gov/news/schedule/full-2025, 2026 https://bea.gov/index.php/news/blog/2026-01-15/economic-release-schedule-updates-gdp-personal-income-and-outlays + https://www.bea.gov/news/2026/personal-income-and-outlays-december-2025 + https://www.bea.gov/index.php/news/schedule/full · FOMC https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm

