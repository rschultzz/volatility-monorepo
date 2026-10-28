# CR-DA Step 0 — 2026-10-27 19:10 PT

Read-only; no outcome statistics (no RV, premium or P&L). Times ET unless tagged.

## 1. Universe

`cr_cy_nights.csv` = CR-BO `keep` set: **621 nights** (gate). CR-CY complete (6/6 contracts both windows): **595**; incomplete 26.

## 2. Night counts after each exclusion (D2)

| step | nights | removed |
|---|---|---|
| CR-BO ES-clean universe | 621 | |
| (a) CR-CY window incomplete | 595 | 26 |
| (b) no valid ATM leg quote within ±2 min | 589 | 6 |
| (c) V_px solve failed | 588 | 1 |

Final primary n = **588**; D9 population (non-event weeknights) **374**.

| year of t | universe | final | D9 population |
|---|---|---|---|
| 2023 | 131 | 125 | 78 |
| 2024 | 201 | 188 | 119 |
| 2025 | 199 | 189 | 124 |
| 2026 | 90 | 86 | 53 |

(a) nights: 2023-05-19, 2023-05-23, 2023-07-21, 2023-08-18, 2023-10-20, 2023-11-17, 2024-01-19, 2024-02-16, 2024-04-19, 2024-05-17, 2024-07-19, 2024-08-16, 2024-10-18, 2024-11-15, 2025-01-17, 2025-01-24, 2025-02-21, 2025-04-17, 2025-05-12, 2025-05-13, 2025-10-17, 2025-11-26, 2026-01-16, 2026-02-20, 2026-04-17, 2026-05-15 (t+1).

b_no_valid_atm_quote: 2024-04-15→2024-04-16, 2024-04-17→2024-04-18, 2024-05-02→2024-05-03, 2024-07-10→2024-07-11, 2025-04-04→2025-04-07, 2025-05-08→2025-05-09

c_solve_failed: 2024-04-25→2024-04-26

## 3. Quote validity per window and leg (CR-CY-complete nights)

| window | leg | valid at target | via ±1 | via ±2 | missing | bid = ask = 0 at target |
|---|---|---|---|---|---|---|
| close 15:55 | atmC | 592/595 (99.5 %) | 2 | 0 | 1 | 0 |
| close 15:55 | atmP | 592/595 (99.5 %) | 2 | 0 | 1 | 0 |
| close 15:55 | wingP | 592/595 (99.5 %) | 2 | 0 | 1 | 0 |
| close 15:55 | wingC | 592/595 (99.5 %) | 2 | 0 | 1 | 0 |
| open 09:35 | atmC | 588/595 (98.8 %) | 1 | 1 | 5 | 0 |
| open 09:35 | atmP | 588/595 (98.8 %) | 1 | 1 | 5 | 0 |
| open 09:35 | wingP | 589/595 (99.0 %) | 1 | 1 | 4 | 0 |
| open 09:35 | wingC | 587/595 (98.7 %) | 2 | 1 | 5 | 0 |

## 4. V_px solve success

| end | straddle (nights with valid ATM quotes) | ±5 strangle (nights with valid wing quotes) |
|---|---|---|
| close | 594/594 (100.0 %) | 594/594 (100.0 %) |
| open | 589/590 (99.8 %) | 589/590 (99.8 %) |

Strangle readout n (final nights with both ends solved): 588.

## 5. CR-BP reproduction (V_px at CR-BP's K and S)

CR-BP primary-subset nights in the CR-BO universe: **22**.

| minute PT | CR-BP has a value | within 1 % | both NaN | reproduced | max abs rel. diff |
|---|---|---|---|---|---|
| 06:31 | 20 | 20 | 2 | 22/22 | 0.00e+00 |
| 06:35 | 22 | 22 | 0 | 22/22 | 0.00e+00 |

## 6. Diagnostics (final nights)

- |F_parity − S|: close median 1.10 pts (IQR 0.49–2.23); open median 1.27 (IQR 0.56–2.43).
- |K0 − S|: close median 1.20; open median 4.31.
- Annualised straddle IV √(V/τ): close median 0.128, open median 0.206.
- Tradeable buyback marks (final nights): {'model': 348, 'quote': 240}.
- ES prints missing: close 0, open 0.

## Gates

- S0-G1 universe 621 = CR-BO keep: **PASS**
- S0-G2 CR-BP reproduction 06:31 all within 1 %: **PASS**
- S0-G3 straddle solve success >= 95 % each end: **PASS**
- S0-G4 final n >= 500: **PASS**

**Step 0: PASS — run once.**
