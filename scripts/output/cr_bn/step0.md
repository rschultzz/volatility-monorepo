# CR-BN Step 0 — 2026-10-17 19:03 PT

Read-only. `orats_options_minute`, ticker SPX, 0DTE = `expir_date_d = trade_date_d`, 2023-05-01 → 2026-10-17. Train ≤ 2026-06-05, holdout after. Counts only.

## Q2 expiry_tod inventory (0DTE SPX rows)

| expiry_tod | rows | dates |
|---|---:|---:|
| am | 1,056 | 8 |
| pm | 1,021,686 | 801 |

Dates with any 0DTE row: 801 (2023-05-01 → 2026-10-16).

## Q1 third Fridays

Third Fridays in range: 42 (train 37, holdout 5).

| date | split | 0DTE rows am / non-am | in 06:30–06:45 am / non-am | opras am / non-am | mixed |
|---|---|---|---|---|---|
| 2023-05-19 | train | 112 / 152 | 112 / 144 | 16 / 24 | **mixed** |
| 2023-06-16 | train | 96 / 214 | 96 / 96 | 12 / 22 | **mixed** |
| 2023-07-21 | train | 128 / 128 | 128 / 128 | 16 / 16 | **mixed** |
| 2023-08-18 | train | 128 / 128 | 128 / 128 | 16 / 16 | **mixed** |
| 2023-10-20 | train | 128 / 176 | 128 / 128 | 16 / 20 | **mixed** |
| 2023-11-17 | train | 144 / 160 | 144 / 112 | 16 / 20 | **mixed** |
| 2023-12-15 | train | 0 / 48 | 0 / 0 | 0 / 4 |  |
| 2024-01-19 | train | 160 / 96 | 160 / 96 | 16 / 16 | **mixed** |
| 2024-02-16 | train | 160 / 144 | 160 / 96 | 16 / 20 | **mixed** |
| 2024-04-19 | train | 0 / 374 | 0 / 256 | 0 / 26 |  |
| 2024-05-17 | train | 0 / 304 | 0 / 256 | 0 / 20 |  |
| 2024-06-21 | train | 0 / 118 | 0 / 0 | 0 / 10 |  |
| 2024-07-19 | train | 0 / 256 | 0 / 256 | 0 / 16 |  |
| 2024-08-16 | train | 0 / 256 | 0 / 256 | 0 / 16 |  |
| 2024-09-20 | train | 0 / 92 | 0 / 0 | 0 / 8 |  |
| 2024-10-18 | train | 0 / 304 | 0 / 256 | 0 / 20 |  |
| 2024-11-15 | train | 0 / 3616 | 0 / 256 | 0 / 30 |  |
| 2024-12-20 | train | 0 / 48 | 0 / 0 | 0 / 4 |  |
| 2025-01-17 | train | 0 / 256 | 0 / 256 | 0 / 16 |  |
| 2025-02-21 | train | 0 / 1188 | 0 / 256 | 0 / 36 |  |
| 2025-05-16 | train | 0 / 4736 | 0 / 256 | 0 / 24 |  |
| 2025-07-18 | train | 0 / 256 | 0 / 256 | 0 / 16 |  |
| 2025-08-15 | train | 0 / 326 | 0 / 256 | 0 / 22 |  |
| 2025-09-19 | train | 0 / 48 | 0 / 0 | 0 / 4 |  |
| 2025-10-17 | train | 0 / 9344 | 0 / 256 | 0 / 50 |  |
| 2025-11-21 | train | 0 / 12798 | 0 / 256 | 0 / 70 |  |
| 2026-01-16 | train | 0 / 256 | 0 / 256 | 0 / 16 |  |
| 2026-02-20 | train | 0 / 350 | 0 / 256 | 0 / 24 |  |
| 2026-03-20 | train | 0 / 256 | 0 / 256 | 0 / 16 |  |
| 2026-04-17 | train | 0 / 5606 | 0 / 256 | 0 / 34 |  |
| 2026-05-15 | train | 0 / 256 | 0 / 240 | 0 / 16 |  |
| 2026-09-18 | holdout | 0 / 128 | 0 / 120 | 0 / 8 |  |
| 2026-10-16 | holdout | 0 / 128 | 0 / 128 | 0 / 8 |  |

- Third Fridays with any 0DTE row: 33; with any `am` row: 8; **mixed (am + non-am): 8** — train 8, holdout 0.
- Mixed dates: ['2023-05-19', '2023-06-16', '2023-07-21', '2023-08-18', '2023-10-20', '2023-11-17', '2024-01-19', '2024-02-16']
- `am`-only third Fridays: []
- Holdout third Fridays and whether 0DTE rows exist: [('2026-06-19', False), ('2026-07-17', False), ('2026-08-21', False), ('2026-09-18', True), ('2026-10-16', True)]

DAILY-CAPTURE runs (`bt_backfill_runs`):

- `DAILY-CAPTURE`: 22 runs, 2026-09-08 → 2026-10-16
- `DAILY-CAPTURE-MATURE`: 8 runs, 2026-09-22 → 2026-10-16

## Q3 non-third-Friday 0DTE dates with `am` rows

0: []

## Q4 minute slots taken by `am` rows on mixed dates (06:30–06:45 PT)

PK is `(opra_symbol, snapshot_pt)` and `insert_bars` is `ON CONFLICT DO NOTHING`, so each (contract, minute) holds one row — AM or PM, not both.

| date | opras | slots am | slots non-am | opras with no non-am bar |
|---|---:|---:|---:|---:|
| 2023-05-19 | 16 | 112 | 144 | 0 |
| 2023-06-16 | 12 | 96 | 96 | 0 |
| 2023-07-21 | 16 | 128 | 128 | 0 |
| 2023-08-18 | 16 | 128 | 128 | 0 |
| 2023-10-20 | 16 | 128 | 128 | 0 |
| 2023-11-17 | 16 | 144 | 112 | 0 |
| 2024-01-19 | 16 | 160 | 96 | 0 |
| 2024-02-16 | 16 | 160 | 96 | 0 |

