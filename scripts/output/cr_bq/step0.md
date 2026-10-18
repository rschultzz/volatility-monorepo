# CR-BQ — Step 0 diagnosis (read-only; 2026-10-18 08:46)

## Q1 — CR-BK parquet schema, sign convention, units

`scripts/logs/cr_bk/shape_summary.parquet`: 777 rows × 21 columns, d 2023-05-02 → 2026-06-05.
`scripts/logs/cr_bk/profiles.parquet` (linear wings): 93049 rows, 769 dates, 121 grid points per date (S0 ± 300, step 5).

| column | dtype | non-null |
|---|---|---|
| `d` | object | 777 |
| `usable` | bool | 777 |
| `S0` | float64 | 769 |
| `remapped` | object | 769 |
| `snap` | datetime64[ns] | 769 |
| `n_pos_linear` | float64 | 769 |
| `n_neg_linear` | float64 | 769 |
| `top_pos_linear` | float64 | 554 |
| `pos_prices_linear` | object | 769 |
| `n_flip_linear` | float64 | 769 |
| `flip_nearest_linear` | float64 | 753 |
| `net_at_s0_linear` | float64 | 769 |
| `exposure_share_without_smile` | float64 | 769 |
| `am_rows` | float64 | 769 |
| `n_pos_flat` | float64 | 769 |
| `n_neg_flat` | float64 | 769 |
| `top_pos_flat` | float64 | 559 |
| `pos_prices_flat` | object | 769 |
| `n_flip_flat` | float64 | 769 |
| `flip_nearest_flat` | float64 | 753 |
| `net_at_s0_flat` | float64 | 769 |

Per `scripts/cr_bk_profile.py` (`profile`, `load_day`, `shape_of`):
- `gex_total(price)` = Σ_rows unit_gamma(F, K, σ, T) · F² · 100 · (call_oi − put_oi), with F = S_e · price / S0 (per-expiry forward shifted proportionally), σ sticky-strike from the 12:46 PT prior-session call-delta smile, T = calendar/365 to 13:00 PT on expiry.
- **Sign:** call OI counts +, put OI counts − (CR-BK L5 'net = call − |put|'), i.e. dealers assumed long calls and short puts (the SqueezeMetrics / Barbon–Buraschi eq. 4 index convention). G > 0 = dealers net long gamma (hedging dampens); G < 0 = net short.
- **Units:** dollars of delta change for a unit (100 %) relative move, Γ·S²·100·OI. ÷ 100 → $ per 1 % move. ÷ 1e9 → $bn.
- `S0` = the nearest live expiry's `orats_oi_gamma.stock_price` = ORATS' per-expiry 15:46 ET (12:46 PT) snapshot of the **prior** session (field-conventions; CR-BK A1).
- `net_at_s0_linear` = gex_total interpolated at S0. **No column holds gamma at the open of day d.** CR-BK's T3 X (`scripts/cr_bk_placebo.py` `x_at`) was computed in the run, not stored: the profile interpolated at the SPX open from `packages/shared/spx_cash` = first `orats_monies_minute.stock_price` print ≥ 06:33 PT (`spot_price` shifted 15 min on 2023 days before 2023-11-09).

### Worked day 2025-02-11 (store date 2025-02-11, smile snapshot 2025-02-10 12:46:00)

- Rows with OI on live expiries: 8615; with a smile: 8615; S0 = 6067.32.
- Recomputed now from `orats_oi_gamma`: call side +65.922e11, put side −61.419e11 → net at S0 +4.5034e11 (= +4.50 $bn per 1 % move).
- Parquet `net_at_s0_linear` +4.5034e11; profile grid at S0 +4.5034e11; max |recomputed − stored| over the grid 0.00e+00e11.
- Zero-gamma flip 6056.3 (-11.0 vs S0); top positive peak 6212.3.
- Profile slope near S0: G(S0 − 25) -5.573e11, G(S0 + 25) +14.902e11 — a 25-pt overnight move changes G by this much.

## Q2 — 'gamma at the open' without monies fields at the open

L2 asks for CR-BK's net dealer gamma **at the open**; the kickoff forbids ORATS monies fields at the open. CR-BK's own value at the open needs the SPX open from `orats_monies_minute` (Q1). Two monies-free candidates, both from the stored profile:

- **A** `net_at_s0_linear` — G at S0, the prior session's 15:46 ET price. Stored column; no construction.
- **B** G interpolated at S0 · ES_open(d) / ES(prev session, 12:46 PT) — the profile read at an ES-implied SPX open (ES 06:30 bar open; ES close of the 12:45 PT bar on the prior session = the CR-BK snapshot minute). ES only, frame-neutral (a ratio).

On the L1 universe (Q3, n 641): B computable on 629 days; missing on 12 (no 12:45 PT bar on prev session: 12).
ES-implied overnight move S0 → open (pts): median +4.2, |move| median 15.4, p90 53.3, max 240.1.

| A vs B (n 629) | value |
|---|---|
| Spearman(G_A, G_B) | 0.932 |
| days with a different sign | 72 (11.4 %) |
| same decile | 46.3 % |
| decile differs by ≥ 2 | 13.5 % |
| D1 under A also D1 under B | 53 of 63 |
| D10 under A also D10 under B | 47 of 63 |

## Q3 — L1 universe, counts after each exclusion

NYSE sessions 2023-05-02 → 2026-06-05: 777; not in the CR-BK parquet: 0 [].

| step | n after |
|---|---|
| CR-BK parquet rows (effective train dates ≤ 2026-06-05) | 777 |
| − not an NYSE session (holiday / weekend) | 777 |
| − dates CR-BK excluded (usable = False: no 12:46 PT prior-session smile) | 769 |
| − NYSE early closes | 761 |
| − ES roll weeks, every year (CR-BO/CR-BM rule: Mon of week before 3rd Fri → expiry Fri, Mar/Jun/Sep/Dec) | 644 |
| − ES RTH bars < 370 of 390 | 641 |

CR-BK-excluded dates: ['2023-07-05', '2023-11-27', '2024-07-05', '2024-12-02', '2024-12-26', '2025-07-07', '2025-12-01', '2025-12-26'].
Dropped for < 370 bars: ['2026-02-13 (361)', '2026-02-17 (282)', '2026-02-18 (307)'].

## Q4 — distribution of G on the L1 universe ($bn per 1 % move = value / 1e11)

**A (at S0):** n 641, G < 0 on **243** days (37.9 %); median +8.72, p10 -29.05, p90 +37.05.
decile cut points: -29.05, -18.91, -6.67, +1.51, +8.72, +14.90, +21.82, +28.09, +37.05

| year | n | G < 0 | median | sd |
|---|---|---|---|---|
| 2023 | 135 | 62 | +1.89 | 23.01 |
| 2024 | 207 | 73 | +9.22 | 21.66 |
| 2025 | 205 | 77 | +9.12 | 25.21 |
| 2026 | 94 | 31 | +25.48 | 35.74 |

Days per decile by year (decile 1 = most negative):

| decile | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|
| D1 | 21 | 13 | 22 | 9 |
| D2 | 18 | 20 | 20 | 6 |
| D3 | 15 | 20 | 20 | 9 |
| D4 | 13 | 24 | 18 | 9 |
| D5 | 10 | 25 | 22 | 7 |
| D6 | 18 | 19 | 21 | 6 |
| D7 | 21 | 28 | 15 | 0 |
| D8 | 10 | 26 | 25 | 3 |
| D9 | 8 | 21 | 23 | 12 |
| D10 | 1 | 11 | 19 | 33 |

**B (at ES-implied open):** n 629, G < 0 on **235** days (37.4 %); median +10.07, p10 -30.75, p90 +40.93.
decile cut points: -30.75, -19.49, -5.95, +1.80, +10.07, +17.18, +23.23, +30.76, +40.93

| year | n | G < 0 | median | sd |
|---|---|---|---|---|
| 2023 | 132 | 60 | +3.10 | 23.64 |
| 2024 | 203 | 71 | +12.39 | 22.78 |
| 2025 | 201 | 75 | +11.74 | 26.66 |
| 2026 | 93 | 29 | +21.04 | 37.69 |

Days per decile by year (decile 1 = most negative):

| decile | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|
| D1 | 22 | 11 | 18 | 12 |
| D2 | 15 | 21 | 21 | 6 |
| D3 | 13 | 21 | 23 | 6 |
| D4 | 14 | 22 | 19 | 8 |
| D5 | 16 | 24 | 17 | 6 |
| D6 | 17 | 23 | 18 | 4 |
| D7 | 18 | 23 | 17 | 5 |
| D8 | 11 | 25 | 23 | 4 |
| D9 | 6 | 25 | 23 | 9 |
| D10 | 0 | 8 | 22 | 33 |

## Q5 — ES RTH minute coverage by year (es_minutes, is_rth)

| year | NYSE sessions | 390 bars | 370–389 | 1–369 | 0 bars | of which early close | mean bars (non-early) |
|---|---|---|---|---|---|---|---|
| 2023 | 168 | 163 | 0 | 3 | 2 | 2 | 383.0 |
| 2024 | 252 | 245 | 0 | 3 | 4 | 3 | 383.7 |
| 2025 | 250 | 243 | 0 | 3 | 4 | 3 | 383.7 |
| 2026 | 107 | 92 | 11 | 4 | 0 | 0 | 386.7 |

Sessions with 0 bars: ['2023-09-15', '2023-12-15', '2024-03-15', '2024-06-21', '2024-09-20', '2024-12-20', '2025-03-21', '2025-06-20', '2025-09-19', '2025-12-19'].
Non-early sessions with 1–369 bars: ['2023-06-16 (1)', '2026-02-13 (361)', '2026-02-17 (282)', '2026-02-18 (307)', '2026-03-17 (340)'].

## Q6 — first and last 30 minutes on included days

Included days 641: last bar starts 12:59 PT on 641; first bar 06:30 PT on 641.
Days whose last bar is not 12:59: [].
Days whose first bar is not 06:30: [].
Bars present in 12:30–12:59: min 30, days < 30: 0. Bars present in 06:30–06:59: min 24, days < 30: 5.
Boundary bars present — 06:59 (first-30 end): 641/641; 12:29 (last-30 start): 641/641; 12:59: 641/641.
No included day is an early close (Q3), so 12:30–13:00 PT is the session's last 30 minutes on every included day (es_minutes is_rth ends at the 12:59 bar on every non-early session).

Sources in the window (es_minutes.source): databento 679, ironbeam 111 dates.

