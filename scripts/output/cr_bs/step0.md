# CR-BS Step 0 — 2026-10-18 10:37 PT

Read-only. Train t ≤ 2026-06-05. Era E0 t < 2024-03-01, E1 t ≥ 2024-03-01. Times ET unless tagged PT.

## 3. 0DTE dte convention

- Rows with dte = 0 in the 07:30–07:33 PT window (all expiries, AM and PM): **0**.
- Same-day PM rows (expir_date = trade_date): dte values {1: 3091}.
- Next-session PM rows: dte values {2: 2277, 4: 566, 5: 80, 3: 40} (calendar gap + 1 on Fridays / before holidays).
- `dte_pct` on same-day PM rows at 07:30 PT: 0.000 (IQR 0.000–0.000, n 771) (ORATS' fraction field, diagnostic only; L3 fixes the clock).

Worked day 2025-03-04 (Tue), first snapshot ≥ 07:30 PT:

| snapshot_pt (PT) | quote_date (UTC) | expir_date | tod | dte | dte_pct | atmiv | vol25 | vol50 | vol75 | stock_price |
|---|---|---|---|---|---|---|---|---|---|---|
| 07:30 | 2025-03-04T15:29:07Z | 2025-03-04 | pm | 1 | 0.00000 | 0.4487 | 0.4408 | 0.4303 | 0.4385 | 5760.48 |
| 07:30 | 2025-03-04T15:29:07Z | 2025-03-05 | pm | 2 | 0.00274 | 0.3306 | 0.3079 | 0.3158 | 0.3280 | 5761.20 |
| 07:30 | 2025-03-04T15:29:07Z | 2025-03-06 | pm | 3 | 0.00548 | 0.3051 | 0.2779 | 0.2885 | 0.3017 | 5762.00 |
| 07:30 | 2025-03-04T15:29:07Z | 2025-03-07 | pm | 4 | 0.00821 | 0.3059 | 0.2726 | 0.2892 | 0.3126 | 5763.60 |
| 07:30 | 2025-03-04T15:29:07Z | 2025-03-10 | pm | 7 | 0.01643 | 0.2446 | 0.2159 | 0.2335 | 0.2542 | 5762.42 |

Reading: ORATS stores the same-day expiry with **dte = 1** (dte counts the expiry day). L1(i)'s "dte = 0 (ORATS dte convention — confirm it is the same-day expiry)" is implemented as `expir_date = trade_date`, PM-settled, which is CR-BP's "smallest dte > 0" contract. The next-session expiry is `expir_date = next_trading_day(t)`, PM-settled.

## 1. L1 universe — counts after each exclusion, by era

| step | E0 | E1 | total |
|---|---|---|---|
| train sessions | 210 | 568 | 778 |
| − early closes | 208 | 562 | 770 |
| − roll weeks | 179 | 474 | 653 |
| snapshot 07:30–07:33 PT | 177 | 473 | 650 |
| 0DTE PM row with smile | 176 | 473 | 649 |
| next-session PM row with atmiv | 169 | 459 | 628 |
| ES RTH bars ≥ 370 | 169 | 456 | 625 |
| morning bars ≥ 1 and remaining bars ≥ 1 | 169 | 456 | 625 |

- Third Fridays lost at the 0DTE step (the table keeps only the AM monthly row): 0 — []
- Snapshot minute used: {'07:30': 621, '07:32': 2, '07:31': 2} (PT).
- Flags in the universe: FOMC days 15 (E0 5, E1 10); 08:30 release days 108 (E0 33, E1 75). Both stay in.
- Non-roll, non-early-close days with no 07:30–07:33 PT snapshot: 3; by month {'2023-05': 1, '2023-10': 1, '2025-11': 1}.

## 2. `atmiv` / `vol50` at 10:30 ET by era (no price-based adjudication exists at 10:30)

No captured option quotes exist at 10:30 ET in train (`orats_options_minute` holds 06:30–06:45 PT condor legs and ≤ 2026-06-05 verticals only), so neither field can be adjudicated against a traded price here. The comparison is field vs field, plus each field against itself 20 minutes later.

| quantity | E0 | E1 |
|---|---|---|
| atmiv / vol50, 0DTE | 1.027 (IQR 1.004–1.050, n 169) | 1.051 (IQR 1.022–1.087, n 456) |
| atmiv / vol50, next-session expiry | 1.011 (IQR 0.999–1.018, n 169) | 1.022 (IQR 0.998–1.045, n 456) |
| vol50(10:30) / vol50(10:50), 0DTE | 1.026 (IQR 0.983–1.059, n 169) | 1.036 (IQR 0.994–1.073, n 455) |
| atmiv(10:30) / atmiv(10:50), 0DTE | 1.014 (IQR 0.985–1.043, n 169) | 1.034 (IQR 0.996–1.069, n 455) |
| vol50(10:30) / vol50(10:50), next-session | 1.010 (IQR 0.997–1.026, n 169) | 1.011 (IQR 0.992–1.041, n 455) |
| atmiv(10:30) / atmiv(10:50), next-session | 1.013 (IQR 0.997–1.025, n 169) | 1.019 (IQR 0.997–1.041, n 455) |

Monthly median atmiv/vol50 on 0DTE around the break: 2023-11 1.055, 2023-12 1.029, 2024-01 1.032, 2024-02 1.049, 2024-03 1.028, 2024-04 1.021, 2024-05 1.043, 2024-06 1.043.

Prior (CR-BP Step 0): at 09:33 ET, 0DTE vol50(09:33)/vol50(09:50) E0 0.523, E1 1.035; atmiv E0 0.622, E1 0.625; CR-BO median atmiv/vol50 at 09:31 on 0DTE 0.706.

**Gate:** (a) False — E0 1.027, E1 1.051 vs band 0.95–1.05; (b) False — E0/E1 − 1 = -0.023; (c) False — vol50 10:30/10:50 E0 1.026, E1 1.036.

**Gate PASS — the E0 depression does not persist at 10:30.**

## 4. `vol25` / `vol75` mapping

- Code: `vol_features.py:21-22` — `vol25` = 25Δ call, `vol75` = 25Δ put (the delta grid is call delta; call Δ 0.75 sits at a strike below spot, i.e. the put wing). CR-BO / CR-BP Step 0: `vol75` = put side, `vol25` = call side.
- Data, 0DTE at the snapshot: vol75 > vol25 on 97.8 % of universe days; K = (vol75 − vol25)/vol50 by era: E0 0.085 (IQR 0.045–0.124, n 169), E1 0.164 (IQR 0.117–0.211, n 456). Positive K = put wing richer, consistent with the mapping (index skew).
- So L2(b) K = (vol75 − vol25)/vol50 is put-side minus call-side, as written.

## 5. Morning-RV availability (ES `ironbeam_es_1m_bars`, bar-start ET)

- E0: morning bars 09:30 → 10:30 60.000 (IQR 60.000–60.000, n 169) (expected 60; min 60); remaining bars 10:30 → 16:00 330.000 (IQR 330.000–330.000, n 169) (expected 330; min 330); days with < 57 morning bars: 0.
- E1: morning bars 09:30 → 10:30 60.000 (IQR 60.000–60.000, n 456) (expected 60; min 46); remaining bars 10:30 → 16:00 330.000 (IQR 330.000–330.000, n 456) (expected 330; min 312); days with < 57 morning bars: 4.
- Returns are log close / prior stored close on the continuous series; roll weeks are excluded (L1(iii)), so no contract switch falls inside an RTH window.

## 6. Walk-forward window sizes

- Universe n = 625 (E0 169, E1 456).
- First fit window: days 1–250, 2023-05-01 → 2024-07-30. First prediction 2024-07-31.
- OOS predictions: 375 (E0 0, E1 375). Final fit window 624 days.
- Expanding-rank quintiles start after 20 predictions: first ranked day 2024-08-28; ranked OOS days 356.
- **E0 OOS n < 40**: the by-era E0 Q5 − Q1 carries a "cell n < 40" note (spec reading).
