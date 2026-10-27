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
