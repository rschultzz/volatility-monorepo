# CR-BM Step 0 — 2026-10-17 16:35 PT

Source: `scripts/logs/cr_as_analysis_full_fee0.65.json` (CR-AS run 85b690ec, $0.65). Rows 1441; date status {'priced_date': 728, 'bad_spx_open': 3, 'no_spx_open_snapshot': 1}; box status {'pm0.5:priced': 722, 'pm1:priced': 719, 'pm1:unlistable': 9, 'pm0.5:unlistable': 6}.

## Q1 coverage

- CR-AS universe 732 dates; priced dates 728; dates with ≥ 1 priced box 723.
- ±0.5 boxes 722; ±1.0 boxes 719; **both structures on the same date: 718**.
- Dates with only one structure: ['2023-05-25', '2023-06-16', '2024-08-06', '2025-05-09', '2025-05-13']

### Q1b both wings captured per priced box

- Priced boxes missing a leg in the 06:30–06:45 window: none (CR-AS priced a box only if all four legs had bars).
- Dates whose captured legs carry `expiry_tod = 'am'` (AM-settled SPX monthly): **8** ['2023-05-19', '2023-06-16', '2023-07-21', '2023-08-18', '2023-10-20', '2023-11-17', '2024-01-19', '2024-02-16']
- Bars on those dates, (am rows, non-am rows) in the window: {'2023-05-19': (56, 72), '2023-06-16': (32, 32), '2023-07-21': (64, 64), '2023-08-18': (64, 64), '2023-10-20': (64, 64), '2023-11-17': (72, 56), '2024-01-19': (80, 48), '2024-02-16': (80, 48)}

## Q2 CR-AS close vs harness settlement (spx_cash.settlement_prints)

- Dates with a harness settlement: 720 / 723; absent: 3 ['2023-05-19', '2023-07-03', '2024-07-03']
- harness − CR-AS close (pts): exact (|Δ| < 0.005) 124; |Δ| ≥ 0.01 594; ≥ 1 pt 343; ≥ 5 pts 97; median |Δ| 0.89; p95 |Δ| 10.41; max |Δ| 68.29
- Disagreeing dates (|Δ| ≥ 0.01) by segment: {'2023 lagged': 94, '2023': 30, '2024': 229, '2025': 169, '2026': 72}
- Boxes whose wing intrinsic changes under the harness settlement: {'pm0.5:call': 90, 'pm0.5:put': 57, 'pm1:call': 32, 'pm1:put': 22}
- Ten largest |Δ|: [('2026-03-02', 68.29), ('2026-05-19', -65.39), ('2026-04-07', 64.11), ('2025-04-10', 45.62), ('2026-02-18', 40.41), ('2026-04-27', 38.74), ('2026-03-10', 38.63), ('2026-03-25', -38.47), ('2026-03-06', 38.13), ('2026-02-13', 36.99)]

## Q3 roll weeks, holidays, early closes

- Non-trading days (NYSE calendar) in the universe: []
- NYSE holidays in the universe: []
- Early-close days in the universe: ['2023-07-03', '2024-07-03']
- ES roll-week dates (week before third Friday through expiry, quarterly): 105
- Third Fridays in the universe: 26

## Q4 joins

- CR-BK `profiles.parquet` (wing = linear) dates: 769; universe dates joined: **719 / 723**; missing: ['2023-05-01', '2023-11-27', '2024-12-02', '2025-12-01']
- `bt_vol_state_daily` cr-bl-v1 rows ≤ split: 777; universe dates with a row 722; with non-NULL `iv_7d_pct` **609 / 723**; first non-NULL 2023-10-23
- CR-AS regime labels (from the CR-AS rows): {'magnet-above': 369, 'amplification': 166, 'untethered': 79, 'magnetic-pin': 90, 'bounded': 19}

## Q5 DAILY-CAPTURE (CR-AU) runs

- `CR-AS-analysis`: 3 runs, 2026-09-07 → 2026-09-07
- `CR-AS-analysis-smoke`: 1 runs, 2026-09-07 → 2026-09-07
- `CR-AS-capture`: 2 runs, 2026-09-07 → 2026-09-07
- `CR-AS-step0`: 1 runs, 2026-09-07 → 2026-09-07
- `DAILY-CAPTURE`: 22 runs, 2026-09-08 → 2026-10-16
- `DAILY-CAPTURE-MATURE`: 8 runs, 2026-09-22 → 2026-10-16
- Train dates with any 0DTE bar in 06:30–06:45 PT outside the CR-AS priced set: 5
