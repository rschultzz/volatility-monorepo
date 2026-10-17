# CR-BM — T-S3c: retired 0DTE condor legs split by wing

Read-only, pre-registered, train only (≤ 2026-06-05). Single run. Branch `feat/CR-BM-condor-legs-by-wing` off `origin/main` `f29c557`.

Authority: vault `Dash/research/Test plan — lean sequence and reconciliation protocol.md` (T-S3c block and the Reconciliation protocol section are binding); strategy note `Dash/research/strategies-proposed/S3 - Morning Surface.md`.

## T-S3c block (verbatim from the test plan)

### T-S3c — condor legs by wing (free)

- **Data.** `orats_options_minute` condor legs captured 06:30–06:45 PT on 732 train dates (CR-AS/CR-AU), SPX-cash settlement already in the harness.
- **Compute.** For the ±0.5 IM and ±1.0 IM condors: P&L of the call spread alone and the put spread alone, per date, at the 06:33 entry, settled at the close. Pooled mean and Wilson/bootstrap CI per wing; also by `bt_vol_state_daily` IV tercile and by CR-BK gamma sign.
- **Pass.** Call-wing mean P&L > 0 with 95 % CI clear of zero AND put-wing CI spans zero or is negative. **Kill S3 (as designed).** Put wing is the positive one with CI clear of zero → side-selection thesis is backwards for this entry time; S3 is rewritten around the put side or dropped.
- **Literature to reconcile against.** [[AlmeidaHizmeri2025]] (VRP⁺ > VRP⁻ at 0DTE), [[BandiFusariReno2023]] (wings = jump premium), [[Vilkov2023]] (P&L realized-skew driven), [[Sinclair2020]] (condor 30 % max-loss at fair), plus the Dash session notes for CR-AS (the break-even and the "magnet-above days lose on the call wing" reading — note this *contradicts* the thesis and must be confronted, not skipped).

## Locked decisions

| # | Decision |
|---|---|
| L1 | Universe: every train date (≤ 2026-06-05) with a complete 0DTE condor leg capture (CR-AS/CR-AU, cr_id DAILY-CAPTURE / CR-AS runs), both the ±0.5 IM and ±1.0 IM structures, 10-wide wings, as captured. |
| L2 | Entry: 06:33 PT quote per leg exactly as the CR-AS read priced it (reuse that script's fill rule; do not re-derive). Settlement: SPX cash close, the harness's existing settlement function. Fees: the harness's existing per-leg commission. |
| L3 | Decompose each condor into its call spread and its put spread. Per date per structure: call_pnl, put_pnl, condor_pnl (= sum; must reproduce the CR-AS pooled figures to the cent — that is gate G1). |
| L4 | Statistics: per wing, mean and median P&L in points and in IM units, bootstrap 95 % CI (10,000 resamples, seed 17), win rate with Wilson CI. Splits: all dates; by `bt_vol_state_daily` 7-day IV tercile; by CR-BK net-gamma sign at the open (from the CR-BK parquet); by regime label as CR-AS used it (magnet-above vs other). |
| L5 | Pass: call-wing mean P&L > 0 with CI clear of zero AND put-wing CI spans zero or is negative, on all dates. Kill S3-as-designed: put wing is the one positive with CI clear of zero. Anything else: "not supported, not killed" — record and stop. |
| L6 | Read-only. No table writes, no cron. Output: a parquet under `scripts/output/cr_bm/` and a markdown report. Holdout never read. |
| L7 | Pre-registered placebo: the same decomposition on a date-shuffled pairing of call legs and put legs (break the same-day pairing) to show the wing difference is not a pricing artefact. |

## Gates

| Gate | Requirement |
|---|---|
| G1 | condor_pnl = call_pnl + put_pnl reproduces the CR-AS pooled figures to the cent. |
| L5 | Pass / kill / not-supported-not-killed, exactly as written, on all dates. |
| L7 | Placebo reported alongside. |

## Commit plan

1. Spec freeze (this file).
2. Step 0 diagnosis: coverage (how many of the 732 dates have both wings on both structures); dates where the CR-AS fill rule and the harness settlement disagree; roll-week and holiday dates in the universe; CR-BK parquet join count. Stop if a spec amendment is needed.
3. Single run (script + parquet + report), one commit.
4. Reconciliation (session note) and wrap.
