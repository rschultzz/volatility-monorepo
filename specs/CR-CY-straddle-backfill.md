# CR-CY — Backfill 1-DTE SPX ATM straddle quotes at the close and the open

Road 2 of `Dash/research/two-roads.md` (step 1, "the unparking backfill"); CR-BP step 7(e).
Type: **write_backfill**, bounded, unattended after a 5-night trial. Data capture only — no outcome,
no premium, no verdict is computed or read.

Branch `feat/CR-CY-straddle-backfill` off `origin/main` 625d2ec, own worktree `.claude/worktrees/cr-cy`.

## Why

CR-BP's price-based read of S1 had n = 20 nights (P_on +0.006 [−0.044, +0.055]), because no 1-DTE
quotes exist in the train cache. CR-BP (e): "about 600 train nights of 1-DTE ATM straddle quotes at
15:55 and 09:35 ET via the ORATS one-minute history" would give a price-based close and open on
every night. This CR fetches those quotes into the existing cache. A later CR re-runs T-S1a′ on
them (pre-registered, train only).

## Scope (fixed by the kickoff)

| # | Item | Definition |
|---|---|---|
| S1 | Nights | CR-BO's ES-clean universe (CR-BO Step 0 Q1, `scripts/output/cr_bo/step0_pairs.csv`, `keep = True`) with t+1 ≤ 2026-06-05: **621 nights**, t 2023-05-01 → t+1 2026-06-05. Frozen into `scripts/cr_cy_nights.csv` (t, t1). Nothing after 2026-06-05 is fetched (B4 holdout 2026-06-06 → 2026-10-26). |
| S2 | Expiry | The SPX **PM** expiry with `expir_date = t+1` (1-DTE at t's close, 0DTE at t+1's open). Root `SPX` (ORATS has no `SPXW`); PM identified at read time by `expiry_tod = 'pm'`. |
| S3 | Strikes | Per window: K0 = the strike nearest the ES-implied SPX spot at that snapshot, plus K0 − 5 and K0 + 5; call and put each → **6 contracts per window**, 12 per night. |
| S4 | Windows (ET) | Close: t 15:50 → 16:00 (12:50–13:00 PT, 11 minutes). Open: t+1 09:30 → 09:45 (06:30–06:45 PT, 16 minutes). One-minute rows as the option endpoint returns them (bid, ask, sizes, IVs, greeks — every column the cache stores). |
| S5 | Fetch path | `options_cache.fetch_option_bars` (gap-aware, option endpoint, range `tradeDate`) with `format_opra` / `opra_to_orats_ticker`, writing `orats_options_minute` and `orats_options_fetched_windows`. **`record_empty_windows=False`**: an empty ORATS response is never recorded as a fetched window (the condor leg-miss lesson, decision 2026-05-22 / CR-AE); it goes to the missing list. |
| S6 | Protocol | `get_backfill_db_conn` → `assert_role_or_die` → `backfill_run(conn, "CR-CY")` → `update_run_smoke`. The options_cache repository reads `DATABASE_URL`, so the script sets it to `BACKFILL_DATABASE_URL` in-process before import (CR-AH / CR-AP precedent): every write goes through `dash_backfill_writer` (INSERT/SELECT on both cache tables; no DELETE). Insert-only (`ON CONFLICT DO NOTHING`). |
| S7 | Rate / resume | Sequential, paced ≤ 80 ORATS calls per minute (limit 100). 429 / 5xx: `http_client`'s retry with backoff, then an outer 60 s sleep and retry (≤ 5). Resume-safe: a (strike, window) whose call and put already have a PM row on every minute of the window is skipped with no call; otherwise `fetch_option_bars`' own gap detection decides. Re-running the script is a no-op on completed windows. |

## Strike placement (data available at the snapshot only)

- **SPX cash** = `packages/shared/spx_cash` (CR-BH series: `spot_price` shifted on the 2023 lagged
  days, `stock_price` from 2023-11-09, bad-print filter).
- **Basis on t** = median over 15:30–15:54 ET (12:30–12:54 PT) of ES − SPX cash, ES = close of the
  `ironbeam_es_1m_bars` bar ending at the print's minute (bar start m − 1). All of it precedes 15:55.
- **Close spot** S_c = ES close of the bar ending 15:55 ET (start 15:54) − basis_t. This is the
  kickoff's "prior close for 15:55": the last completed ES minute before the snapshot.
- **Open spot** S_o = ES close of the bar ending 09:30 ET on t+1 (start 09:29; the latest bar
  ≤ 09:29 within 5 min if missing) − basis_t. This is "the 09:30 spot". The front-month contract is
  unchanged overnight (roll weeks and contract switches are outside the universe, CR-BO L1).
- Why ES-implied rather than the ORATS print at the snapshot: the 09:30–09:32 cash prints are mostly
  the prior close (CR-BH), and the R-a `quote_date` lag is ≈ 2 min (field-conventions).
- **Listed strike** = 5-pt grid (SPX near-ATM PM strikes are listed every 5 pts in the whole
  universe). Listing is not looked up separately: a strike that is not listed returns empty and is
  logged as missing. K0 = 5·round(S/5).

## Missing list

Every (night, window, strike, side) with no PM row after the fetch is written to
`scripts/logs/cr_cy/missing.csv` with a reason: `empty_response` (ORATS returned no rows; nothing
recorded as fetched), `http_<code>` (permanent error), `am_only` (the third-Friday AM contract holds
the `(opra_symbol, snapshot_pt)` key — CR-BN collision), or `covered_no_rows` (an older fetched
window covers the range but no PM rows exist). Nothing is retried in a loop; the list is the record.

**Known risk, `am_only`:** 24 universe nights have a monthly third-Friday t+1 (2023-05-19 … 2026-05-15).
The SPX AM monthly and the PM weekly share `opra_symbol` under the cache PK. If the AM row is
inserted first, the PM row is dropped. Step 0 has no such night. The full run reports their outcome
separately.

## Step 0 — 5-night trial (automatic, no attendance)

One night per era, chosen where possible so t+1 is a CR-BP primary-subset day
(`scripts/output/cr_bp/vpx_primary.parquet`, `in_universe`), so the gate (G3) has data to run on:

| era | t → t+1 | CR-BP overlap |
|---|---|---|
| 2023 | 2023-07-14 → 2023-07-17 | yes |
| 2024 | 2024-05-10 → 2024-05-13 | yes |
| early 2025 | 2025-02-10 → 2025-02-11 | no primary day in early 2025 |
| late 2025 | 2025-10-03 → 2025-10-06 | yes |
| 2026 | 2026-01-09 → 2026-01-12 | yes |

Gates. All four must pass, or the run stops and the reason is written to the vault:

| # | Gate |
|---|---|
| G1 | Every one of the 10 windows has PM rows for ≥ 5 of its 6 contracts. |
| G2 | Mids are inside the bid–ask: on every fetched row 0 ≤ bid ≤ ask and ask > 0 (mid = (bid + ask)/2). |
| G3 | On overlap nights, the 09:35 ET K0 straddle mid (call mid + put mid at 06:35 PT) is within **0.5–2.0×** CR-BP's captured straddle at the same minute. Reference = the Black (F = S, r = 0) call + put at CR-BP's K and S with total variance `vpx_06:35`, i.e. CR-BP's price-implied straddle (not `implied_move_1d`, which CR-BP showed is not a straddle price). Where CR-BP's K equals our K0, the rows are the same cache rows and the ratio is ≈ 1 by construction. That is stated, not hidden. |
| G4 | No empty window was written as fetched: `orats_options_fetched_windows` has 0 rows with `row_count = 0` and `fetched_at` ≥ the run start. |

If all pass, the full backfill runs unattended (the remaining 616 nights; the 5 trial nights are
cache hits).

## Report (end of run)

Nights complete (12/12 contracts with rows) / partial / missing (0 rows in both windows); ORATS
calls made; rows inserted; the `bt_backfill_runs` row id(s); smoke results (G1–G4 recomputed over
the full set, plus G3 on all CR-BP overlap nights). Push, PR (script + spec), don't merge. Session
note in the vault.

## Standing rulings conformance

Data capture only: no outcome, no session premium, no signal is computed or read. ES bars are read
only to centre strikes. CR-BP's `vpx_primary` (option prices, train) is read for G3.

| Ruling | Status | Note |
|---|---|---|
| A1–A21 (the setup) | n/a | no setup, level, range, entry, stop, target or flow read |
| B1 (three verdicts) | n/a | no test verdict; gates are data-quality gates |
| B2 (disclosed multiple testing) | n/a | |
| B3 (one setup at a time) | n/a | |
| B4 (train ≤ 2026-06-05; holdout 06-06 → 10-26 frozen; forward log ≥ 10-27; R-a; R-b 2025 untouched) | **honoured, with a note** | Nothing after 2026-06-05 is fetched or read. The universe includes 2025 nights (199). **R-b is "untouched" for tests.** This CR caches 2025 quotes but reads no outcome, so no R-b result is seen. The caching itself does not spend R-b. Recorded so the later T-S1a′ spec decides explicitly whether its universe includes 2025. CR-BO / CR-BP (already run) used 2023–2026 train including 2025 for S1. |
| B5 (Databento reference for flow and bars in tests) | n/a | ES bars place strikes only; not a test. 2023–2025 bars are `source = 'databento'` anyway |
| B6–B8 | n/a | |
| C1–C2 (UI) | n/a | no UI |
| Revoked R1–R7 | none used | |

No "deviates". No stop.

## Out of scope

The T-S1a′ re-run, any premium or variance computation, any read of a fetched quote beyond the G2 / G3
sanity checks, any change to `options_cache` code, any fetch after 2026-06-05.

## Step 0 findings (2026-10-27, run `0aff7767-db30-49a0-baac-490b07b4f58f`)

Night list frozen: `scripts/cr_cy_nights.csv`, 621 nights (t+1 by year: 2023 131, 2024 201, 2025 199,
2026 90; last t+1 2026-06-05). Strike plan for the trial (`scripts/logs/cr_cy/step0_plan.csv`):

| t → t+1 | basis_t | S_c | K_c | S_o | K_o |
|---|---|---|---|---|---|
| 2023-07-14 → 07-17 | 32.5 | 4503.5 | 4500/4505/4510 | 4504.3 | 4500/4505/4510 |
| 2024-05-10 → 05-13 | 24.1 | 5225.9 | 5220/5225/5230 | 5238.1 | 5235/5240/5245 |
| 2025-02-10 → 02-11 | 21.6 | 6065.1 | 6060/6065/6070 | 6043.4 | 6040/6045/6050 |
| 2025-10-03 → 10-06 | 48.6 | 6719.9 | 6715/6720/6725 | 6740.2 | 6735/6740/6745 |
| 2026-01-09 → 01-12 | 38.6 | 6971.2 | 6965/6970/6975 | 6930.9 | 6925/6930/6935 |

S_c is within 2–4 pts of the published SPX closes (4505.42, 6066.44, 6715.79). The close window
uses the last ES minute before 15:55, so it is not the 16:00 print.

| Gate | Result |
|---|---|
| G1 ≥ 5/6 contracts per window | **pass**: 10/10 windows have 6/6 contracts; 5/5 nights complete (12/12) |
| G2 mid inside bid–ask | **pass**: 810/810 rows (= 5 × (6 × 11 + 6 × 16), every minute present), 0 null quotes, 0 crossed or non-positive asks |
| G3 09:35 straddle vs CR-BP | **pass**: 4 overlap nights, ratios 1.010 / 0.920 / 0.969 / 1.064 (median 0.990). CR-BP's K differs from K0 by 5 pts on all four, so these are independent rows, not a self-comparison |
| G4 no empty window written | **pass**: 0 `row_count = 0` windows since the run start |

Cost: 25 ORATS calls, 650 rows inserted, 5 strike-windows skipped as already cached (CR-AS 0DTE
legs), 0 transient / permanent errors, 1.1 min. Projection for the remaining 616 nights:
≈ 3,700 calls, ≈ 50 min at the 80/min pace.

All gates pass → the full run proceeds unattended (kickoff rule). No amendment.

## Full run (2026-10-27)

Two run rows in `bt_backfill_runs`:

- **`6b107f41-40f1-4851-a6b7-aeedafc9799b` run 1, `aborted`.** It fetched all 621 nights (3,589 ORATS calls, 95,498 rows inserted, 161.6 min, 0 transient errors). It then crashed in the smoke step on a script bug (a set built from dicts). Fixed in the script (`n_am_only` count). The run's logs are kept as `scripts/logs/cr_cy/full_run1.out`, `missing_run1.csv`, `full_windows_run1.csv`.
- **`76fc3d80-031d-4d8a-a209-57e302839b02` run 2, `completed`.** This is the same command re-run for the smoke step. Everything was a cache hit (3,476 strike-windows skipped, 474 contracts cache-hit) except the 13 permanent-404 strike-windows, which were re-asked (13 calls, 0 rows). Smoke written to the row and to `scripts/logs/cr_cy/full_smoke.json`.

Totals: **3,614 ORATS calls** (25 Step 0 + 3,589 run 1; run 2's 13 are re-asks of 404s), **96,148 rows inserted** (650 + 95,498).

| | |
|---|---|
| Nights complete (12/12 contracts) | **595 / 621** |
| Nights partial | **26**, of which 22 are close windows lost to the AM/PM key collision and 4 are 404 windows |
| Nights missing (no rows in either window) | **0** |
| Rows checked | 97,746 (every minute of every window present wherever a contract has rows) |

Smoke over the full set (the kickoff's gates were for Step 0; recomputed here as a report):

| Gate | Full-set result |
|---|---|
| G1 ≥ 5/6 contracts | 1,214 / 1,242 windows. **28 short.** 22 close windows with 0/6 = **`am_only`**: t+1 a monthly third Friday (21 of the 24 such nights) plus **2025-04-17** (Thursday before Good Friday, which held the April AM monthly). The AM monthly holds `(opra_symbol, snapshot_pt)`, so the PM rows were not stored. This is the known CR-BN collision, not fixable without a schema change. 6 windows (2023-05-23 open; 2025-01-24 close + open; 2025-05-12 open; 2025-05-13 close + open; 2025-11-26 open) are short because of ORATS **404** on 13 strike-windows (`missing.csv`, reason `http_404`). Nothing was recorded as fetched for them. |
| G2 mid inside bid–ask | **300 / 97,746 rows (0.31 %) are bid = ask = 0**, ORATS's no-quote marker. 0 crossed, 0 negative. By minute: 06:30 PT 66, 06:31 98, 06:32 113, 06:33 12, 06:37 6, close 12:56 5. The 09:35 ET minute is clean. Readers must treat bid = ask = 0 as missing, never as a mid of 0. |
| G3 09:35 straddle vs CR-BP | **pass, 22 / 22 overlap nights** in 0.5–2.0×: median 1.00, range 0.61–1.14. Four nights share CR-BP's strike (ratio 1.00 by construction). Ratios < 1 (0.61–0.85) are nights where CR-BP's K sat 10–20 pts off our K0, so its straddle includes intrinsic value. |
| G4 no empty window written | **pass**: 0 `row_count = 0` windows written by either run. |

Missing list: `scripts/logs/cr_cy/missing.csv`, 152 contract-windows = 126 `am_only` + 26 `http_404`. Git-ignored and machine-local; the counts are recorded here.
