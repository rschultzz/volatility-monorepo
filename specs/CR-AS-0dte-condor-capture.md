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

