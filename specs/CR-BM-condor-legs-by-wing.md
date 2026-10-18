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

## Step 0 — diagnosis findings (2026-10-17)

Script `scripts/cr_bm_step0.py` (SELECTs only, no run row); full output `scripts/output/cr_bm/step0.md`. Universe source: the CR-AS full-sample read as run, `scripts/logs/cr_as_analysis_full_fee0.65.json` (run `85b690ec`, $0.65), whose rows carry strikes, IM, regime and the CR-AS settlement.

**Q1 Coverage.** 732 CR-AS universe dates → 728 priced dates (3 `bad_spx_open`, 1 `no_spx_open_snapshot`) → 723 with ≥ 1 priced box. ±0.5 boxes 722, ±1.0 boxes 719; **both structures on the same date: 718**. One structure only: 2023-05-25, 2023-06-16, 2024-08-06, 2025-05-09, 2025-05-13. Every priced box has all four legs (both wings) in the 06:30–06:45 window. DAILY-CAPTURE (CR-AU) runs are all 2026-09-08 → 2026-10-16, i.e. holdout; they add no train dates. 5 train dates have some 0DTE bar in the window outside the priced set (the skipped CR-AS dates).

**Q2 CR-AS fill / settlement vs the harness settlement — they disagree.**
- CR-AS settled on `orats_monies_minute.spot_price` at the last snapshot ≤ 13:00 PT. The harness settles on `spx_cash.settlement_prints` (CR-BI: last clean print 12:50–13:00 PT of the CR-BH hybrid series — `spot_price` shifted −15 min on the lagged 2023 windows, `stock_price` from 2023-11-09).
- 720 / 723 dates have a harness settlement; absent: 2023-05-19, 2023-07-03, 2024-07-03 (the last two are early closes, no 12:50–13:00 print).
- harness − CR-AS close: exact on 124 dates; |Δ| ≥ 0.01 on 594; ≥ 1 pt on 343; ≥ 5 pts on 97; median |Δ| 0.89, p95 10.41, max 68.29 (2026-03-02). The large ones are the 2026 `spot_price` drift and the 2023 lagged windows (94 dates).
- The wing intrinsic changes on **90 ±0.5 call wings, 57 ±0.5 put wings, 32 ±1.0 call, 22 ±1.0 put**.
- **AM/PM contamination on 8 third Fridays** (2023-05-19, 06-16, 07-21, 08-18, 10-20, 11-17, 2024-01-19, 02-16): the captured bars under the same OPRA symbols include `expiry_tod = 'am'` rows (AM-settled SPX monthly) alongside the PM rows, roughly half each. The CR-AS fill builds a per-minute `(strike, type)` dict, so the entry mid on those dates may be an AM-contract quote. The harness excludes AM-settled expiries (CR-BI A1.4).

**Q3 Calendar.** No NYSE holiday or non-trading day in the universe. Early closes: 2023-07-03, 2024-07-03 (CR-AS took the ~10:00 PT last print; the harness has no settlement). ES roll-week dates (week before the quarterly third Friday through expiry): 105 — SPX legs and SPX-cash settlement do not use ES; ES entered CR-AS only through the open-basis filter and the regime label. Third Fridays: 26.

**Q4 Joins.** CR-BK `scripts/logs/cr_bk/profiles.parquet` (the CR-BK output lives under `scripts/logs/cr_bk/`, not `scripts/output/`; wing = linear): 769 dates; **719 / 723** universe dates join (missing 2023-05-01, 2023-11-27, 2024-12-02, 2025-12-01). `bt_vol_state_daily` `cr-bl-v1`: 722 / 723 have a row; non-NULL `iv_7d_pct` on **609 / 723** (starts 2023-10-23). Regime labels as CR-AS used them: magnet-above 369, amplification 166, magnetic-pin 90, untethered 79, bounded 19.

### Gate: spec amendment required — STOP

L2 (settle on the harness's SPX-cash function) and G1 (condor_pnl reproduces the CR-AS pooled figures to the cent) cannot both hold: CR-AS settled on a different price on 594 of 720 dates and its wing intrinsic changes on 201 boxes. Separately, 8 dates carry AM-settled quotes the CR-AS fill cannot tell apart from PM, and 3 dates have no harness settlement. The spec does not say which wins, so no run is made until it is amended.

**Proposed amendment A1 (superseded by the adopted A1 below):**
- A1.1 Primary read (L4, L5, L7) uses the harness settlement, as L2 says.
- A1.2 G1 is a code check, run on the CR-AS rows under CR-AS's own settlement: per-row call_pnl + put_pnl = CR-AS `net_pts` to the cent, and the pooled ±0.5 / ±1.0 mean net IM reproduces CR-AS's printed −0.001 / +0.002 (run 85b690ec, $0.65). The harness-settled condor totals are then printed next to CR-AS's as a restatement, not a gate.
- A1.3 Primary universe excludes the 8 AM-contaminated third Fridays and the 3 dates with no harness settlement (2023-05-19 is in both) → 713 dates.
- A1.4 Fee: `packages/shared/config.FEE_PER_CONTRACT_PER_LEG` = $0.65; per wing 2 legs × 2 sides = 0.026 pts; condor 0.052.
- A1.5 Splits: IV tercile = `iv_7d_pct` ≤ 33 / 33–67 / ≥ 67 (CR-BL cuts; 609 dates, the rest reported as "no IV rank"); gamma sign = sign of CR-BK `x_at` on the linear profile at the SPX-cash open (CR-BL's verbatim CR-BK convention; 719 dates); regime = CR-AS row label, magnet-above vs other.
- A1.6 No roll-week exclusion (the test uses no ES price).

## Amendment A1 — adopted (Ryan, 2026-10-17), before any wing P&L was computed

- **A1.1** Main read settles on the harness function (`spx_cash.settlement_prints`). L5 pass/kill applies to this read only.
- **A1.2** G1 is split in two. **G1a:** under CR-AS's own settlement (`spot_price` ≤ 13:00 PT), call_pnl + put_pnl must equal the CR-AS net per row to the cent, and the pooled means must reproduce −0.001 (±0.5) and +0.002 (±1.0) IM from the $0.65 run — this proves the decomposition. **G1b:** report the pooled condor P&L under both settlements side by side, with the 201 changed boxes broken out (count, direction, mean change in IM). G1b is diagnostic, not a gate; its result is a finding about CR-AS, not about S3.
- **A1.3** Universe: drop the 8 mixed-quote third Fridays (AM/PM under the same OPRA symbol) and the 3 dates with no harness settlement (2023-05-19, 2023-07-03, 2024-07-03). 713 dates. Apply the same universe to both settlements so G1b is like-for-like. List the 11 dates in the report.
- **A1.4** Fees $0.65/contract/leg = 0.026 pts per wing.
- **A1.5** Gamma from the `scripts/logs/cr_bk/` parquet; report the 4 unjoined dates by name. IV tercile on the 609 dates with an IV rank; the gamma and IV splits report their own n.
- **A1.6** No roll-week exclusion — the test uses no ES price (SPX legs, SPX-cash settlement); ES entered CR-AS only through its open-basis filter and the regime label, which are taken as CR-AS recorded them.
- **A1.7** Reconciliation gets a row for the CR-AS session note stating that its break-even and its "magnet-above days lose on the call wing" reading were computed on `spot_price` settlement, with G1b's numbers as the re-settled version. A line goes into State of Knowledge — 2026-09-20 under a new "## Updates — 2026-10-17 (CR-BM)" heading only if G1b moves the CR-AS pooled condor P&L by more than its CI half-width; otherwise the session note records "re-settlement did not change the read."

Nothing else in the spec changes.
