# CR-AS — 0DTE condor quote capture + first priced read (train only)

> Authority: vault session note `Dash/sessions/2026-09-06 - CR-AS — 0DTE Condor Quote Capture + First Priced Read.md`
> Branch: `feat/CR-AS-0dte-condor-capture` (off `origin/main` a076d49, the CR-AQ merge)
> Scope: capture of 0DTE SPXW condor legs (06:30–06:45 PT windows) into `orats_options_minute` for the ≤ 2026-06-05 universe under a wall-clock budget, then a read-only priced analysis. Nothing persisted except quotes and `bt_backfill_runs` rows.
> Mode: unattended; halts at the first STOP gate that misses.

Drafted in chat 2026-09-06. Unattended overnight. Two steps: capture (data), then a read-only priced analysis on the ≤ 2026-06-05 universe. Nothing persisted except quotes and run records.

## Problem

CR-AQ's containment columns gave the P-side of a 0DTE condor: on the ≤ 2026-06-05 universe, expected breach cost beyond a ±0.5 IM box is 0.19–0.23 IM in every regime with n > 50, and 0.05–0.09 beyond ±1 IM. `[Guessing]` A normal-model Q puts the ±0.5 IM strangle near 0.5–0.6 IM, i.e. the market may pay ~2× the realized breach cost. That is a claim about prices we don't have. `orats_options_minute` holds no 0DTE contracts for these dates.

## Pre-registered hypotheses (read after the run, not before)

Per regime, per box, mean net P&L in IM units at settlement, with a bootstrap 95% CI:

- **H1** ±0.5 IM condor, 10-wide wings, sold at the first valid 06:33+ PT minute: mean P&L > 0 on the pooled sample. Expected: yes (VRP), CI excludes 0 on the pooled sample only.
- **H2** Magnetic-pin mean P&L (±0.5 box) exceeds magnet-above by ≥ 0.03 IM. Expected: sign yes, CI overlaps.
- **H3** Credit collected / realized breach cost ratio ≥ 1.5 on the pooled ±0.5 box. Expected: yes; this is the VRP measurement.
- **H4** Down-wing losses exceed up-wing losses on magnet-above and untethered (from CR-AQ's asymmetry). Expected: yes.
- **H5** ±1 IM condor mean P&L per unit of max loss is lower than ±0.5 (wider box = less credit, similar tail). Expected: yes.
- **H6 (Ryan's rule)** Trading only `magnetic-pin` + `bounded` days beats trading all days, on mean net P&L per trade in IM units, ±0.5 box. Expected from the P-side table: yes for pin by ~0.03–0.04 IM, bounded unknown (n=20). Also report the rule's *opportunity cost*: total P&L captured by the rule vs. all-days, since a per-trade improvement on 14% of days can be a worse strategy. Secondary: `amplification` and `magnet-above` are each individually worse than pooled. Expected from P-side: **no** — both are within 0.01 IM of pooled on breach cost. If prices reverse that, say so.

Read rule: report each with sign, magnitude, CI. "Supported" only if CI excludes 0 in the stated direction; otherwise directional. No per-regime claim on n < 40.

## Locked decisions

| # | Decision | Value |
|---|---|---|
| 1 | Universe | Canonical dates ≤ 2026-06-05 with `close_move_over_im IS NOT NULL` and `implied_move_1d > 0` (excludes the 2 zero-IM dates and the 9 no-bar dates). Train only; no holdout dates fetched or read. |
| 2 | Boxes | Two per date. Shorts at `round5(open ± 0.5·IM)` and `round5(open ± 1.0·IM)`; long wings 10 points further out. `open` = `session_open_t0`. Same-day expiry (SPXW PM). Puts below, calls above. Up to 8 distinct contracts per date (wings may coincide across boxes at small IM; dedupe). |
| 3 | Listing | 0DTE SPXW is fully listed on the 5-point grid (CR-AO G0.2: completeness 1.0 at ≤ 7 DTE). `round5` per leg; if a leg is absent from the chain for that date, mark the box `unlistable` and skip. No pair-snapping dependency. |
| 4 | Entry window | 06:30–06:45 PT for each contract into `orats_options_minute` via `fetch_option_bars`. Entry price = first minute ≥ 06:33 where all four legs pass the CR-AN quote rule and the condor credit ∈ `(0, min wing width]`. If none by 06:45 → box `no_valid_entry`, skipped. |
| 5 | Settlement | No quotes. SPX close from `orats_monies_minute` last RTH snapshot (`stkPx` / underlying field the runner uses; Step 0 names it). Payoff = intrinsic of the four legs at the close. Sanity: SPX close within 30 points of `session_close_t0` (ES) ± basis; Step 0 measures typical basis and sets the tolerance. |
| 6 | Sampling & budget | Step 0 times one full date (8 fetches). Budget = 8 hours of wall time. Pre-registered order: all `bounded`; then `magnetic-pin`, `magnet-above`, `amplification`, `untethered` in **interleaved** round-robin of stride-selected dates (so a budget cut leaves balanced groups) until 40 each; then continue round-robin over the remaining dates until the budget is spent. Log the sample achieved. |
| 7 | Throughput safety | Use the existing watchdog pattern (`run_cr_ai_stage2_watchdog.sh`) — checkpoint per date, resumable, `--max-hours 8`. |
| 8 | Fees | Report gross and net at $1.30/contract/side (Schwab per the CSV: $0.65 per contract, 4 legs × 2 sides… Step 0 reads the CSV fee lines and states the per-condor round-trip fee). Fees in points = fee / 100 per contract. |
| 9 | Persistence | Quotes only (`orats_options_minute`) + `bt_backfill_runs` (cr_id CR-AS-capture, CR-AS-analysis). Analysis prints tables and writes them to the spec. No results table. |
| 10 | Not in scope | Wall-conditioned strikes; holdout; live cards; anything after 2026-06-05. |

## Gates

| Gate | Expected | On miss |
|---|---|---|
| G0.1 — capture path reusable with a date list + arbitrary contract list (CR-AM/AP capture script) | yes | STOP if a new fetcher is needed |
| G0.2 — one-date timing | recorded; sample size derived | note |
| G0.3 — SPX close field in `orats_monies_minute` named; ES–SPX basis measured on 10 dates | recorded | STOP if no SPX close source |
| G0.4 — ORATS serves 0DTE contracts for a 2023 date and a 2025 date (one probe each) | 200 | STOP if 404 on both (retention) |
| G1 — capture: 0 exceptions other than 404; 404 rate < 10% of legs | yes | STOP if > 25% |
| G2 — sample achieved: bounded 20, ≥ 40 in each other regime | yes | note if budget cut it |
| G3 — analysis: boxes with `no_valid_entry` < 15% | yes | note |
| G4 — settlement sanity violations | 0 | STOP |
| G5 — no date > 2026-06-05 in any log | yes | STOP and redact |

## Step 0 findings (2026-09-06 ~21:00 PT)

Interpreter: `apps/web/.venv/bin/python` for DB reads, the ORATS probes and the capture; Rosetta repo venv (`arch -x86_64 .venv/bin/python -m pytest`) for the suites. Branch cut from `origin/main` a076d49 (the CR-AQ merge).

Universe (decision 1): **732 dates** ≤ 2026-06-05 (743 canonical rows minus 9 no-bar roll Fridays and 2 zero-IM dates). By `regime_kind_at_classification`: magnet-above 373, amplification 169, magnetic-pin 91, untethered 79, bounded 20. Existing 0DTE coverage in `orats_options_minute` for the 06:30–06:45 window on these dates: **3 dates** (the 837k "0DTE" rows in the cache are earlier CRs' 12:50–13:00 settlement windows), so every entry window is a fresh fetch.

| Gate | Expected | Actual | Result |
|---|---|---|---|
| G0.1 capture path reusable with a date list + arbitrary contract list | yes | **yes** — `fetch_option_bars([opra], start_pt, end_pt)` (CR-AP's pattern), one contract per call so a 404 on one leg cannot abort the others; no new fetcher | PASS |
| G0.2 one-date timing | recorded | **~46 s per date** (8 legs, 5.8 s per fetch, 256 bars = 8 contracts × 16 minutes × call+put rows): 47.6 / 46.6 / 48.7 / 43.7 / 45.0 s over 5 dates, run `80e2bc37-5fa5-446d-99f5-e6538aaa45c0` (`cr_id='CR-AS-step0'`, 40 legs, 0 404s, 1280 bars) | PASS |
| G0.3 SPX close field named; ES–SPX basis measured | recorded | **`orats_monies_minute.spot_price`** at the last snapshot ≤ 13:00 PT (`stock_price` is the per-expiry discounted level — differs from spot by 1–3 pts; not used). Basis measured on **all 732** dates (table below), not 10 | PASS |
| G0.4 ORATS serves 0DTE on a 2023 and a 2025 date | 200 | **200 / 200** — 2023-06-13 (magnetic-pin) and 2025-06-05 (magnet-above): all 8 legs each, 16 minutes each, `dte=1`, `expiry_tod='pm'` | PASS |

### ES–SPX basis (`session_open_t0` / `session_close_t0` vs `orats_monies_minute.spot_price`), 732 dates

| quantity | median | p01 | p05 | p95 | p99 | min | max |
|---|---|---|---|---|---|---|---|
| ES open − SPX spot @ 06:30 snapshot | +27.4 | −22.9 | −0.7 | +60.9 | +85.0 | −178.3 | +1134.5 |
| ES open − SPX spot @ **06:33** snapshot | **+26.4** | −11.1 | −1.1 | +58.3 | +83.7 | −32.7 | +645.7 |
| ES close − SPX spot @ last ≤ 13:00 | +25.4 | −7.5 | +1.6 | +56.0 | +67.7 | −43.7 | +100.9 |
| \|basis@06:33 − basis@close\| | 5.6 | 0.05 | 0.6 | 25.0 | 39.7 | 0.0 | 603.9 |

Median IM is 45 pts (p05 21, p95 111). On **547 / 732** dates the open basis exceeds IM/4: centring the strikes on the ES open would shift the box by ~0.6 IM on a median day — the ±0.5 IM condor would have its short call ~0.1 IM from the SPX open. The 06:30 snapshot is stale at the bell (p99 +85, max +1134); the 06:33 snapshot — the one CR-037 pins the implied move to — is clean except three dates. Outliers at 06:33: 2023-09-21 (+646), 2023-11-29 (+570), 2024-07-31 (+128) — bad monies spot; 2023-12-04 (−32.7) is a legitimate negative-basis day. Close outliers: 2026-04-07 (+100.9), 2026-05-19 (−43.7). Dates whose first ≥ 06:33 snapshot is later than 06:40: one (2025-11-26, 09:27). Bar `spot_price` in the captured 0DTE rows equals the monies 06:33 `spot_price` exactly on all three dates inspected.

### Listing (decision 3) — the prior-close chain under-lists next-day expiries

`orats_oi_gamma` at the prior close carries the 0DTE expiry on **732 / 732** dates, but the 5-point grid near spot is incomplete on some: 2025-05-13 → expiry 05-14 lacks 5835, 5845, 5855, 5865, 5885, 5895, 5905, 5910, 5915, 5920 within ±100 of spot (10 of 41), while the dry-run plan flagged 2025-05-02 (5735) and 2025-05-13 (5915) too. Strikes for a same-day expiry are added on the morning of expiry, after the EOD chain. Probe: 2025-05-14 fetched with the chain check bypassed — **all 8 legs served, 16 minutes each**, including 5835 / 5845 / 5930 / 5940 / 5960 / 5970 that the prior-close chain lacks. Over the first 10 bounded dates the chain rule would have discarded 2 boxes and one whole date for no reason.

### Third Fridays

26 universe dates are monthly (3rd-Friday) expiries, where the AM-settled SPX and the PM-settled SPXW share the date and ORATS keys both under root `SPX`. Probe 2025-05-16: the option endpoint returned **one contract per strike, `expiry_tod='pm'`**, 16 minutes each — the PM SPXW, which is the instrument decision 2 names. No ambiguity in the response; included.

### Fees (decision 8)

The brokerage export (Schwab transactions CSV, 2026-09-06) could not be read in this session — the sandbox classifier refused every read of the file, including a column-filtered one. Fee used: the note's stated convention, **$1.30 per contract per side** ($0.65 commission + exchange/OCC/regulatory fees) → per condor round trip 4 legs × 2 sides × $1.30 = **$10.40 = 0.104 SPX points** (`CONDOR_ROUND_TRIP_FEE_PTS`). If Schwab does not charge on expiring legs the true figure is $5.20 = 0.052 pts; gross is reported next to net so either reading is available. **Ryan to confirm against the CSV.**

### Spec amendments (Step 0, before any implementation commit)

| # | Was | Now | Why |
|---|---|---|---|
| 2 | `open` = `session_open_t0` (ES) | `open` = SPX `spot_price` at the first `orats_monies_minute` snapshot ≥ 06:33 PT (must be ≤ 06:40, else `no_spx_open_snapshot`); sanity **ES open − SPX open ∈ [−40, +100]**, else `bad_spx_open` (3 dates) | +26 pt median basis vs 45 pt median IM; strikes must be in SPX space |
| 3 | `round5` per leg; absent from the chain for that date → `unlistable` | Listing truth is the ORATS response: a leg that 404s **or returns no bars** makes its box `unlistable`. The prior-close chain is recorded per box as `chain_missing` (information only) | `orats_oi_gamma` lacks strikes added on the expiry morning; ORATS served all of them |
| 5 | SPX close within 30 pts of `session_close_t0` ± basis; Step 0 sets the tolerance | Settlement = `spot_price` at the last monies snapshot ≤ 13:00 PT. Sanity (G4): **\|basis@close − basis@06:33\| ≤ 50 pts and basis@close ∈ [−50, +110]** | p99 of the drift is 39.7; the only dates over 50 are the three `bad_spx_open` dates, which never enter |
| 8 | fee from the CSV | $1.30/contract/side from the note; CSV unreadable here | see Fees |
| — | 3rd Fridays unspecified | included (ORATS returns the PM contract) | probe |
| — | files | `packages/shared/backtest/condor_0dte.py` (box / dedupe / credit / payoff / sampling, shared by both scripts + tests) and `scripts/run_cr_as_capture_watchdog.sh` added to files touched | one box definition for capture and analysis |

### Sample plan (decision 6, from G0.2)

46 s per date → 8 h ≈ **620 dates** of the 728 eligible (732 − 3 `bad_spx_open` − 1 `no_spx_open_snapshot`); the 5 Step 0 dates are cached. Pre-registered order (`sample_order`): all 20 `bounded`; then magnetic-pin / magnet-above / amplification / untethered round-robin over 40 stride-selected dates each (160 dates, ≈ 2.3 h in — G2 met there); then round-robin over the remaining 552 until the budget is spent. Expected achieved: bounded 20, ≥ 40 in each other regime, ~600 dates total; the cut, if any, lands in the remainder phase and leaves the four regimes within one date of each other.

## Step 1 — capture (2026-09-06 21:17 → 2026-09-07 05:18 PT, watchdog, `--max-hours 8`)

Command: `MAX_HOURS=8 bash scripts/run_cr_as_capture_watchdog.sh` (one attempt, no restarts). Run **`2abf9ff7-d367-4454-b8ba-d21a53c9cc38`** (`cr_id='CR-AS-capture'`), `completed`, 28 810 s. Log `scripts/logs/cr_as_capture_watchdog.log`, checkpoint `scripts/logs/cr_as_capture_checkpoint.json` (both untracked). Step 0's 5 dates (run `80e2bc37`) were cache hits (40 legs).

Budget exhausted at 05:18:01 before 2025-10-10, in the remainder phase: **658 dates checkpointed** (3 `bad_spx_open` skips, 655 fetched), 5128 legs planned, 5109 fetched, **19 × 404 (0.37 %)**, 0 exceptions, 160 756 bars written, median 44.8 s per date (max 112.7 s). 72 universe dates not reached (all in the remainder phase, so the regime groups stayed balanced). The 404s are all 5-mod-10 strikes on a handful of dates (2024-08-05 P 5085/5095, 2024-08-06 P 5065, 2025-05-09 C 5720, 2025-05-12 C 5845/5855/5885, 2026-04-09 C 6815/6845/6855, …): 13 boxes `unlistable` (pm0.5 5, pm1 8), decision 3 as amended.

| Gate | Expected | Actual | Result |
|---|---|---|---|
| G1 capture: 0 exceptions other than 404; 404 rate < 10 % of legs | yes | **0 exceptions; 19 / 5128 = 0.37 %** | PASS |
| G2 sample achieved: bounded 20, ≥ 40 in each other regime | yes | **bounded 19** (2026-04-09: the short call 6815 / 6845 and long call 6855 404 — both boxes unlistable, not a budget cut), **magnetic-pin 90, magnet-above 297, amplification 166, untethered 79** (651 dates with ≥ 1 captured box) | note (bounded 19 / 20) |

Sample achieved by regime (dates with at least one fully captured box): bounded 19 / 20, magnetic-pin 90 / 91, magnet-above 297 / 373, amplification 166 / 169, untethered 79 / 79.

## Step 2 — priced read (2026-09-07 05:21 PT, read-only)

Command: `apps/web/.venv/bin/python -u scripts/cr_as_condor_analysis.py --cr-id CR-AS-analysis --out scripts/logs/cr_as_analysis_final.md`. Run **`b5152407-5742-4760-bf3e-183c79a4a438`** (`cr_id='CR-AS-analysis'`), `completed`. (A code-path smoke run mid-capture is recorded as `0d6babc8…`, `cr_id='CR-AS-analysis-smoke'`, 11 boxes; not a read.) Nothing persisted except the run row and the report file.

| Gate | Expected | Actual | Result |
|---|---|---|---|
| G3 boxes with `no_valid_entry` | < 15 % | **0 / 1297 = 0.0 %** — every attempted box had all four legs valid with credit ∈ (0, 10] at 06:33 (entry offset p95 = 0 min) | PASS |
| G4 settlement sanity violations | 0 | **0** (656 dates; rule: \|basis@close − basis@06:33\| ≤ 50 and basis@close ∈ [−50, +110]) | PASS |
| G5 no date > 2026-06-05 in any log | yes | max date in results **2026-06-03**; universe assertion and results assertion both hold; checkpoint max 2026-06-03 | PASS |

### Tables (verbatim from the analysis report)

#### Per regime × box (net P&L in IM units at settlement; mean [95 % bootstrap CI])

| box | regime | n | credit (IM) | realized loss (IM) | gross P&L (IM) | net P&L (IM) | net / max loss | win % | breach below / above | put-wing loss (IM) | call-wing loss (IM) | entry offset p50/p95 (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pm0.5 | pooled | 650 | 0.101 | 0.099 | +0.003 | +0.000 [-0.009, +0.009] | -0.025 | 59 | 136 / 189 | 0.042 | 0.057 | 0 / 0 |
| pm0.5 | magnetic-pin | 90 | 0.137 | 0.108 | +0.029 | +0.026 [-0.005, +0.056] | +0.106 | 68 | 17 / 18 | 0.055 | 0.053 | 0 / 0 |
| pm0.5 | magnet-above | 296 | 0.106 | 0.110 | -0.004 | -0.006 [-0.021, +0.007] | -0.059 | 57 | 62 / 99 | 0.044 | 0.066 | 0 / 0 |
| pm0.5 | amplification | 166 | 0.081 | 0.088 | -0.007 | -0.009 [-0.023, +0.005] | -0.099 | 53 | 36 / 52 | 0.033 | 0.054 | 0 / 0 |
| pm0.5 | untethered | 79 | 0.089 | 0.079 | +0.010 | +0.007 [-0.016, +0.030] | +0.053 | 61 | 20 / 15 | 0.047 | 0.032 | 0 / 0 |
| pm0.5 | bounded | 19 | 0.094 | 0.059 | +0.035 | +0.033 [-0.015, +0.079] | +0.212 | 74 | 1 / 5 | 0.009 | 0.050 | 0 / 0 |
| pm1 | pooled | 647 | 0.043 | 0.037 | +0.005 | +0.003 [-0.004, +0.010] | +0.006 | 83 | 50 / 63 | 0.015 | 0.022 | 0 / 0 |
| pm1 | magnetic-pin | 89 | 0.064 | 0.061 | +0.004 | +0.000 [-0.028, +0.027] | -0.015 | 81 | 7 / 10 | 0.027 | 0.033 | 0 / 1 |
| pm1 | magnet-above | 296 | 0.043 | 0.036 | +0.008 | +0.005 [-0.005, +0.015] | +0.022 | 83 | 20 / 31 | 0.012 | 0.024 | 0 / 0 |
| pm1 | amplification | 165 | 0.033 | 0.035 | -0.002 | -0.004 [-0.016, +0.007] | -0.031 | 80 | 15 / 17 | 0.014 | 0.021 | 0 / 0 |
| pm1 | untethered | 79 | 0.036 | 0.023 | +0.013 | +0.011 [-0.004, +0.024] | +0.040 | 87 | 8 / 3 | 0.019 | 0.003 | 0 / 0 |
| pm1 | bounded | 18 | 0.042 | 0.034 | +0.007 | +0.005 [-0.041, +0.042] | +0.014 | 89 | 0 / 2 | 0.000 | 0.034 | 0 / 0 |

#### P-side (ES, `bt_daily_outcomes`) vs priced (SPX quotes) on the same dates — ±0.5 IM box

| regime | n | P-side breach cost beyond ±0.5 IM (uncapped, IM) | beyond ±1.0 IM | wing-capped ±0.5 (IM) | priced realized loss ±0.5 (IM) | priced credit ±0.5 (IM) |
|---|---|---|---|---|---|---|
| pooled | 650 | 0.206 | 0.066 | 0.088 | 0.099 | 0.101 |
| magnetic-pin | 90 | 0.194 | 0.087 | 0.084 | 0.108 | 0.137 |
| magnet-above | 296 | 0.212 | 0.066 | 0.099 | 0.110 | 0.106 |
| amplification | 166 | 0.221 | 0.069 | 0.079 | 0.088 | 0.081 |
| untethered | 79 | 0.200 | 0.049 | 0.082 | 0.079 | 0.089 |
| bounded | 19 | 0.069 | 0.013 | 0.049 | 0.059 | 0.094 |

#### Hypotheses (pre-registered; read rule: supported only if the CI excludes the threshold in the stated direction)

| # | statement | estimate [95 % CI] | read |
|---|---|---|---|
| H1 | ±0.5 IM condor, pooled mean **net** P&L (IM) > 0 (n=650) | +0.000 [-0.009, +0.009] | directional (sign agrees) |
| H1 (gross) | same, gross (n=650) | +0.003 [-0.006, +0.012] | directional (sign agrees) |
| H2 | magnetic-pin − magnet-above mean net P&L (±0.5) ≥ 0.03 IM (n=90 / 296) | +0.032 [-0.001, +0.065] | directional (sign agrees) |
| H3 | credit / realized breach cost ≥ 1.5, pooled ±0.5 (n=650) | +1.03 [+0.94, +1.13] | directional (sign disagrees) |
| H4 magnet-above | put-wing loss − call-wing loss > 0 (±0.5, n=296) | -0.022 [-0.040, -0.004] | directional (sign disagrees) |
| H4 untethered | put-wing loss − call-wing loss > 0 (±0.5, n=79) | +0.015 [-0.012, +0.043] | directional (sign agrees) |
| H5 | ±1.0 minus ±0.5 mean net P&L per unit max loss < 0 (paired dates n=646) | +0.032 [-0.019, +0.083] | directional (sign disagrees) |
| H5 (levels) | net / max loss: ±0.5 -0.026 [-0.087, +0.034] · ±1.0 +0.005 [-0.026, +0.036] |  |  |
| H6 | rule (pin + bounded) − all days, mean net P&L per trade (±0.5) > 0 (n=109 / 650) | +0.027 [-0.001, +0.054] | directional (sign agrees) |
| H6 opportunity cost | total net P&L (IM): rule +2.95 over 109 trades vs all-days +0.15 over 650 (17% of days); rule captures +1994% of the all-days total |  |  |
| H6 secondary amplification | amplification − pooled mean net P&L (±0.5) < 0, i.e. individually worse (n=166) | -0.009 [-0.025, +0.007] | directional (sign agrees) |
| H6 secondary magnet-above | magnet-above − pooled mean net P&L (±0.5) < 0, i.e. individually worse (n=296) | -0.007 [-0.023, +0.010] | directional (sign agrees) |

#### 10 sample rows (±0.5 box)

| date | regime | strikes | entry | credit (pts) | SPX close | put loss | call loss | net (pts) | net (IM) | breach |
|---|---|---|---|---|---|---|---|---|---|---|
| 2023-05-01 | magnet-above | 4145/4155/4175/4185 | 06:33 | 4.75 | 4167.35 | 0.00 | 0.00 | +4.65 | +0.218 | — |
| 2023-08-09 | amplification | 4470/4480/4515/4525 | 06:33 | 3.43 | 4482.20 | 0.00 | 0.00 | +3.32 | +0.096 | — |
| 2023-11-14 | magnet-above | 4445/4455/4505/4515 | 06:33 | 2.60 | 4492.97 | 0.00 | 0.00 | +2.50 | +0.051 | — |
| 2024-02-29 | magnet-above | 5060/5070/5115/5125 | 06:33 | 3.12 | 5099.96 | 0.00 | 0.00 | +3.02 | +0.067 | — |
| 2024-06-07 | magnet-above | 5305/5315/5370/5380 | 06:33 | 3.07 | 5347.58 | 0.00 | 0.00 | +2.97 | +0.057 | — |
| 2024-09-19 | magnet-above | 5670/5680/5750/5760 | 06:33 | 2.18 | 5711.83 | 0.00 | 0.00 | +2.07 | +0.031 | — |
| 2024-12-30 | amplification | 5860/5870/5910/5920 | 06:33 | 6.05 | 5911.43 | 0.00 | 1.43 | +4.52 | +0.120 | above |
| 2025-04-14 | amplification | 5390/5400/5510/5520 | 06:33 | 4.60 | 5412.88 | 0.00 | 0.00 | +4.50 | +0.041 | — |
| 2025-07-29 | untethered | 6375/6385/6415/6425 | 06:33 | 3.50 | 6372.82 | 10.00 | 0.00 | -6.60 | -0.213 | below |
| 2025-11-19 | amplification | 6565/6575/6670/6680 | 06:33 | 3.70 | 6641.98 | 0.00 | 0.00 | +3.60 | +0.038 | — |

### Hypothesis reads (pre-registered; CI = 95 % bootstrap, B = 10 000, seed 20260906; "supported" only if the CI excludes the threshold in the stated direction)

- **H1 — not supported; directional, sign agrees, magnitude nil.** Pooled ±0.5 IM condor net P&L +0.000 IM [−0.009, +0.009] (n = 650); gross +0.003 [−0.006, +0.012]. The condor sold at the first valid 06:33 minute breaks even after fees. The VRP the note guessed (~2× breach cost) is not there at this entry.
- **H2 — not supported; directional, sign agrees.** magnetic-pin − magnet-above = +0.032 IM [−0.001, +0.065] (n = 90 / 296): the point estimate meets the ≥ 0.03 threshold; the CI does not clear it (or zero).
- **H3 — not supported; sign disagrees.** Credit / realized breach cost = **1.03** [0.94, 1.13] pooled ±0.5. The market prices the ±0.5 IM box at almost exactly its realized breach cost. The P-side table explains why the note's guess was wrong: the *uncapped* breach cost beyond ±0.5 IM is 0.206 IM (matching CR-AQ's 0.19–0.23), but the condor's loss is wing-capped, 0.088 IM on ES / 0.099 IM priced on SPX — and the credit is 0.101 IM. The 2× ratio compared credit to the wrong (uncapped) cost.
- **H4 — magnet-above: reverses (sign disagrees, CI excludes 0):** put-wing − call-wing loss = −0.022 IM [−0.040, −0.004] (n = 296) — magnet-above days lose on the **call** wing (99 breaches above vs 62 below). Untethered: +0.015 [−0.012, +0.043] (n = 79), directional, sign agrees. CR-AQ's downside asymmetry does not carry to the priced condor on magnet-above days; the reverse is supported there.
- **H5 — not supported; sign disagrees.** ±1.0 minus ±0.5 net P&L per unit max loss = +0.032 [−0.019, +0.083] (paired n = 646); levels ±0.5 −0.026 [−0.087, +0.034], ±1.0 +0.005 [−0.026, +0.036]. The wider box is not worse per unit of risk; if anything slightly better.
- **H6 (Ryan's rule) — not supported; directional, sign agrees.** Trading only magnetic-pin + bounded days: mean net P&L per trade +0.027 IM [−0.001, +0.054] above all-days (n = 109 / 650). **Opportunity cost:** the rule's 109 trades sum to **+2.95 IM** versus **+0.15 IM** for all 650 trades — the rule captures essentially the entire pooled total on 17 % of the days because the other 541 days net to ≈ −2.8 IM. Per-trade the improvement is +0.027, in line with the P-side expectation (+0.03–0.04). Secondary: amplification − pooled −0.009 [−0.025, +0.007] (n = 166), magnet-above − pooled −0.007 [−0.023, +0.010] (n = 296): both individually worse than pooled in sign, neither supported — as the P-side predicted (within 0.01 IM). Prices do not reverse that.

Per-regime (n ≥ 40) net P&L, ±0.5 box: magnetic-pin **+0.026** [−0.005, +0.056] (68 % wins), untethered +0.007 [−0.016, +0.030], magnet-above −0.006 [−0.021, +0.007], amplification −0.009 [−0.023, +0.005]. Bounded (n = 19, no claim) +0.033 [−0.015, +0.079]. No per-regime CI excludes zero.
