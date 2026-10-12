# CR-BK — Corrected Gamma Profile: Placebo and Volatility Test (test-only, no pipeline)

> Authority: vault CR `Dash/CRs/CR-BK — Corrected Gamma Profile Placebo Test.md` (last reviewed 2026-10-11) + kickoff prompt, 2026-10-11
> Branch: `feat/CR-BK-corrected-gamma-profile-test` (off `origin/main` 9949d24). CR id confirmed unused (no branch, spec or commit named CR-BK).
> Data safety: **read-only** against the database. No writes to any table. Train dates only (`trade_date <= 2026-06-05`); the holdout is not read.
> **Locked decisions L1–L10 and the pass criteria were confirmed by Ryan on 2026-10-11.**

## Why

Chat-side DB checks on 2026-10-11 (exploratory, SPX cash, 2023-05 → 2026-10):

- Magnet-above days do not beat other days on forward 15-session SPX return in an uptrend (+1.29% vs +1.27%) and do worse outside one (+0.32% vs +1.2–1.8%). Holds across 5/10/20/40/60-session trend lookbacks.
- 1–7 DTE wall touch depends on recent momentum (about 79% vs 49%), not on the wall.
- Day range / implied move is flat across regimes (1.13–1.17), including amplification.
- The 2026-09-20 pre-registered placebo (commit `17008e3`) already found walls as built do not separate from equal-distance levels.

All of that is a verdict on the landscape as built. Reading `packages/shared/gex_landscape.py` shows two construction problems:

1. Kernel centre is `discounted_level`, which moves strikes UP by carry, while the cash-space gamma peak is slightly below the strike (about 70 pts apart at 42 DTE).
2. `compute_landscape` spreads each row's gamma exposure, measured at the table spot, as a Gaussian of width `8 × sqrt(dte)` points. True gamma width is roughly spot × IV × sqrt(T): about 70 pts for 1 DTE at SPX 7700 / 15% vol, versus 8 in the code. The same factor of about nine holds at longer DTE.

This CR answers one question before any pipeline is built: **do levels from a correctly computed gamma profile separate from placebo?**

## Locked decisions (confirmed by Ryan, 2026-10-11)

| # | Decision | Value |
|---|---|---|
| L1 | Scope | Test only. No new production table, no cron, no frontend, no change to `orats_gex_landscape`, `bt_daily_features`, `bt_daily_outcomes`. |
| L2 | Profile method | Full repricing: for each grid price S, recompute Black-Scholes gamma for every option row at S, multiply by OI, sum. No `discounted_level`, no Gaussian kernel. |
| L3 | Axis | SPX cash. All prices, levels and outcomes in SPX cash. No ES anywhere in this CR. |
| L4 | IV dynamics | Sticky strike: each option keeps its IV as S moves across the grid. |
| L5 | Dealer sign | Unchanged: net = call − \|put\| (dealers long calls, short puts). |
| L6 | Grid | S0 ± 300 pts, step 5, where S0 is the row `stock_price` (prior close). |
| L7 | Dates | Train only: trade_date ≤ 2026-06-05. Holdout is not read. |
| L8 | Run discipline | Thresholds and pass criteria below are committed before any outcome is computed. One run. No re-tuning after seeing results. |
| L9 | Sessions | NYSE calendar via `packages/shared/sessions.py`. SPX price series = CR-BH hybrid series (cutover 2023-11-09). |
| L10 | ES conversion | Out of scope. If the test passes, a follow-up CR converts SPX levels to ES at read time using observed ES − SPX basis. |

## Step 0 — diagnosis gate (answer all before implementation; contradiction-stop)

1. Confirm how `gex_call` / `gex_put` are computed at ingest and that `gamma` is evaluated at `stock_price` = prior close. (Chat check on one row: `gex_call ≈ gamma × call_oi × 100 × stock_price²`.)
2. IV source. `orats_oi_gamma` has no IV column. Preferred: interpolate each strike's IV from the `orats_monies_minute` smile for the same expiry at the last snapshot of the prior session. Validate by reproducing the stored `gamma` at S0: report median and p90 absolute percent error on 20 random train dates. Gate: median error ≤ 5%. If it fails, report and stop.
3. Forward per expiry: use the per-expiry `stock_price` in `orats_monies_minute` (it is expiry-specific) or r/q from the row; say which and why. Flag AM-settled expiries.
4. Confirm no lookahead: everything used for date t is available before the 06:30 PT open of t.
5. Confirm the `17008e3` placebo harness can take an alternate level source without changing its matching logic. If it cannot, describe the smallest change and stop for approval.
6. Shape report, no outcomes: on train dates, distribution of the number of positive peaks and negative troughs in the corrected net profile (same `find_walls` prominence, 10%), and how often a zero-gamma flip level exists within the grid. **Stop if more than 80% of days have a single positive peak and no trough** — then "walls" are not defined in the corrected profile and T2 is dropped; report before continuing.
7. Count of usable train dates per test.

## Tests (pre-registered)

All in SPX cash, train dates, one run. Report the old-landscape result from the same harness alongside each.

**T1 — touch at the peak-gamma level.** Level = highest positive peak of the corrected net profile. Split into above-open and below-open. Outcome: SPX trades to the level within 5 sessions and within 20 sessions. Placebo: the `17008e3` equal-distance design on non-level days. Stat: touch-rate difference with block-bootstrap 95% CI.

**T2 — containment.** Nearest positive peaks on each side of the open (when both exist). Outcome: close inside and full range inside, same session. Placebo: `17008e3` equal-shape design.

**T3 — volatility.** X = corrected net gamma evaluated at the SPX open, as a percentile rank within the trailing 250 train sessions. Y1 = same-day SPX (high − low) / open. Y2 = same-day range / open-straddle implied move. Stat: Spearman correlation with block-bootstrap 95% CI; also top-tercile vs bottom-tercile mean of Y1 and Y2. Also report Y1 and Y2 for days opening below vs above the zero-gamma flip level, where one exists.

## Pass criteria (locked, confirmed by Ryan 2026-10-11)

| Test | Pass |
|---|---|
| T1 | Touch difference ≥ +8 pts at either horizon, CI lower bound > 0 |
| T2 | Containment difference ≥ +5 pts on close-inside, CI lower bound > 0 |
| T3 raw (Y1) | Spearman ≤ −0.10, CI excludes 0 |
| T3 priced (Y2) | Reported, not a pass/fail. Tells us whether any vol effect is already in the implied move. |

Verdicts:

- **T1 or T2 passes** → levels carry information; follow-up CR builds the parallel pipeline (new table, ES conversion at read time, crons) and confirms once on the holdout.
- **Only T3 raw passes** → gamma matters for volatility, not for levels; redirect toward vol-conditioned structures, no wall targets.
- **Nothing passes** → wall-as-level is retired; record in State of Knowledge and stop work on the landscape.

## Outputs

- Results appended to this spec (tables above plus the Step 0 answers).
- Corrected profiles saved locally (parquet, not committed, path recorded here). No DB writes.
- Session note in `Dash/sessions/` with the verdict.

## Not in scope

- Any change to live cards, regimes, features or outcomes.
- Holdout reads.
- 0DTE intraday flow (OI is prior-day; a known limit, recorded, not fixed here).
- Testing the dealer-sign assumption.

## Step-0 findings (2026-10-11, read-only, train only)

Session `default_transaction_read_only = on` on every connection; no table written; no date > 2026-06-05 read; no outcome computed. Scripts: `scripts/cr_bk_step0_*.py`.

> [!danger] **STOP — gate 2 fails (median 5.47 % > 5 %).** Gate 1 also contradicts the spec's wording ("stock_price = prior close"). Q6 (shape report) was not run because it would build profiles on a failed IV method. Nothing implemented. Decisions needed: see "For approval" at the end.

**1. `gex_call` / `gex_put` at ingest — identity confirmed; "prior close" is wrong.** `[Certain]` `apps/cron/job_orats_eod.py` and `scripts/backfill_orats_oi_gamma.py` use the same formula: `gex = gamma × stockPrice² × OI × 100` (ORATS `hist/strikes` fields `gamma`, `stockPrice`, `callOpenInterest`, `putOpenInterest`). The row is stored under `trade_date = next_trading_day(API date)`. Check on every train row 2023-05 → 2026-06-05 with gamma > 0 and both OIs > 0 (3,756,473 rows): `gex_call / (gamma·call_oi·100·stock_price²)` ∈ [1 − 2.4e-10, 1 + 2.4e-10]; `gex_put` likewise.
   - **Contradiction:** ORATS `stockPrice` is **not the prior close.** It is the **12:46 PT (15:46 ET) EOD snapshot** of the API date. The expiring-expiry `stock_price` matches `orats_monies_minute.stock_price` at 12:46 PT exactly on 4 of 6 dates checked and at 12:47–12:50 PT on the other 2. It differs from the 13:00 PT print by 12–60 pts (2025-03-11: 5601.10 vs 5572.24; 2026-05-19: 7356.80 vs 7416.45).
   - It is also **per expiry**: 53 distinct values on one date, rising with expiry at about the r − q carry (2025-03-12: 5601.10 for the expiring 03-11 expiry → 5606.85 at 12 DTE). It equals monies' per-expiry `stock_price` at 12:46 (03-21: 5606.85 vs 5606.875).
   - So the stored gamma is evaluated at the per-expiry ORATS price at 15:46 ET of the prior session. L6's substance ("S0 = row `stock_price`") is still implementable; its parenthetical "(prior close)" is wrong, and "the row `stock_price`" needs a rule because it varies by expiry. Proposed amendment A1 below.

**2. IV source — FAILS the gate as written.** Method: for each (store date, expiry), take the `orats_monies_minute` call-delta smile (`vol5`…`vol95`) of the prior session. Map each delta point to a strike with K = F·exp(−N⁻¹(Δ)·σ_Δ·√T + ½σ_Δ²·T), with T in calendar days/365 to 13:00 PT on expiry. Interpolate σ linearly in K, flat beyond Δ5/Δ95. Black gamma φ(d1)/(S·σ·√T). Compare with stored `gamma` on every row with OI > 0, using 20 random train store dates (seed 20261011; 157,041 rows).

   | snapshot | forward | median \|err\| | p90 | GEX-weighted median | GEX-weighted p90 |
   | --- | --- | --- | --- | --- | --- |
   | last of session (13:00 PT) — as specified | F = per-expiry S | **5.64 %** | 99.5 % | 2.96 % | 16.1 % |
   | last of session (13:00 PT) | F = S·e^{(r−q)T} | 5.85 % | 99.4 % | 3.14 % | 16.2 % |
   | 12:46 PT (= ORATS EOD time) | F = per-expiry S | **5.47 %** | 99.7 % | 2.61 % | 16.0 % |
   | 12:46 PT | F = S·e^{(r−q)T} | 5.69 % | 99.7 % | 2.79 % | 16.0 % |

   Gate: median ≤ 5 % → **fails on all four variants.** Where the error is (12:46, F = per-expiry S):
   - **By moneyness z = ln(K/F)/(σ√T):** |z| ≤ 1: median 1.5–2.4 % (73 % of GEX). 1 < |z| ≤ 2: 2.9–8.8 % (23 %). 2 < |z| ≤ 3: 17–55 % (2.9 %). |z| > 3: 65–100 % (0.3 %). Inside the Δ5–Δ95 strike span: median 2.5 % (93 % of GEX); outside: 56 % (7 % of GEX).
   - **By DTE (row median / GEX-weighted median):** 0 DTE 55 % / 1.4 %; 1–7 DTE 21 % / 2.4 %; 8–30 DTE 4.6 % / 2.4 %; 31–90 DTE 3.9 % / 2.3 %; 91–400 DTE 5.9 % / 3.8 %. Short-dated chains have most of their OI > 0 strikes beyond Δ5/Δ95.
   - **Time and forward conventions are right.** Near-ATM (|z| ≤ 0.5) ratio of reproduced to stored gamma is 0.97–1.01 at every calendar DTE from 1 to 40, and implied ORATS T matches calendar days. There is no business-day or 252 effect.
   - **Diagnostic only (same sample, not a gate result):** linear-in-log-K wing extrapolation instead of flat gives median 5.20 %, GEX-weighted 2.59 %. It does not clear the gate.
   - **Why it matters beyond the gate metric:** under L4 (sticky strike), when the grid moves S by ±300, short-dated wing strikes become at-the-money and carry these extrapolated IVs. For 1 DTE, ±300 pts is about ±4–5σ. The profile near the grid edges for 0–7 DTE is therefore built on the least reliable IVs.

**3. Forward per expiry.** Use the **per-expiry `stock_price` on the `orats_oi_gamma` row itself** as the expiry's forward. It is ORATS' parity-implied price for that expiry: it rises with expiry at about the r − q carry and equals monies' per-expiry `stock_price` at 12:46. Treated as the forward (no extra carry), it reproduces gamma better than S·e^{(r−q)T} with the row's `short_rate` / `div_yield`: 5.47 vs 5.69 % median, 2.61 vs 2.79 % weighted. The row r/q are a poor second choice anyway: `div_yield` = 0 on most rows. Grid mapping: F_e(S) = S_e · S / S0 (each expiry's forward shifts proportionally with cash).
   - **AM-settled:** monies tags `expiry_tod = 'am'`, about 20 expiries per date (SPX monthlies / quarterlies). `orats_oi_gamma` merges AM and PM chains on third-Friday dates under one `expir_date`, because the PK `(ticker, trade_date, expir_date, strike)` lets one overwrite the other. Example: store 2025-03-12, expiry 03-21: 465 rows at S 5606.85 plus 2 rows with OI 0 at 5605.95. Effect: a 6.5 h T error, material only at ≤ 1 DTE. Proposal: T to 06:30 PT for `am`-tagged expiries; flag merged dates and count them in the results.

**4. Lookahead.** `[Certain]` The profile inputs for store date t are ORATS EOD of the previous session (OI, gamma, per-expiry price at 15:46 ET t−1) and the monies smile from the same session. The cron writes them at 10:00 UTC on t, before the open. The two quantities that are *not* pre-06:30 are part of the test definitions and match the CR-BI harness:
   - the SPX open (first clean print ≥ 06:33 PT; used for the above/below split and as T3's X evaluation point);
   - `implied_move_1d` (06:33 PT ATM IV, smallest dte > 0, × prior-session S × √(1/252); used as the placebo distance scale and the Y2 denominator).

   Outcome windows start at that open. The T3 percentile rank uses only prior sessions.
   - **Calendar artefact:** 32 store dates are NYSE holidays (old next-weekday convention, e.g. 2023-05-29, 2024-07-04, 2024-12-25). Each holds the prior session's EOD, which belongs to the next trading day, and that trading day has no row of its own (745 + 32 = 777 = the NYSE trading days 2023-05-02 → 2026-06-05). Remapping them to `next_trading_day` is lookahead-safe and recovers all 32.

**5. Placebo harness.** `17008e3` is the pre-registration text only. The code that ran it is `scripts/cr_bi_followup_placebo.py` (commit `b05954d`). **It cannot take an alternate level source as is.** Levels are built inline in `main()` from `orats_gex_landscape.walls` plus `wall_carry()`; the P1 signal set is hard-wired to `magnet-above` rows with `horizon_sessions = h`.
   - **Smallest change:** a new `scripts/cr_bk_placebo.py` that imports `boot_ci` / `line` from the CR-BI script. It re-states the two matching cores verbatim as functions taking `{date: level}` and `{date: (below, above)}`. P1: leave-one-out pool, touch iff max high over h sessions ≥ open + z·IM. P2: band [open − b·IM, open + a·IM], close-inside / range-inside on t0. Same month-cluster bootstrap, 10,000 resamples, seed 20260920.
   - The CR-BI script stays untouched. **Control before the one run:** fed CR-BI's strike-axis magnet-above walls, the new script must reproduce the published cells exactly (68.1 / 66.6, 89.4 / 87.2, 84.8 / 84.7, 75.4 / 72.8).

**6. Shape report — not run** (stopped at gate 2).

**7. Usable train dates (upper bounds; T2 depends on Q6).**
   - NYSE trading days 2023-05-02 → 2026-06-05: 777.
   - With an `orats_oi_gamma` row on the trading day: 745 (777 with the holiday remap).
   - With the prior-session 12:40–12:50 PT monies snapshot: 741. Missing: 2023-11-27, 2024-12-02, 2025-05-09, 2025-12-01.
   - With `implied_move_1d` > 0 (`v0.6.1-nyse-sessions`): **736**.
   - **T1:** ≤ 736 at 5 sessions and ≤ 736 − 19 at 20 sessions, before the above/below split.
   - **T2:** unknown until Q6.
   - **T3:** ≤ 736 − 250 ≈ **486** if the trailing-250 percentile requires a full window (first full window ≈ 2024-05). It would be ≈ 736 with an expanding window and a minimum of N sessions.

### For approval (nothing below is implemented)

- **A1 (gate 1 wording):** S0 = per-expiry `stock_price` of the nearest live expiry (= ORATS 15:46 ET snapshot of the prior session), not "prior close". IV smile from the 12:46 PT snapshot (ORATS' own time) instead of the session's last snapshot.
- **Gate 2, pick one:**
  - (a) Re-state the gate on the GEX-weighted median, since the profile is a GEX-weighted sum and the row median is dominated by far-wing rows that weigh almost nothing. That passes at 2.6 %. Then re-validate once on a *fresh* 20-date sample before T1–T3.
  - (b) Keep the row-median gate and get per-strike IV from ORATS directly (`hist/strikes` `smvVol`, the IV ORATS itself used for `gamma`; an API read, no DB write). This means a new fetch path and a re-validation.
  - (c) Stop the CR here.
- **Interpretation questions for T1–T3** (the spec is silent on these):
  - T1's placebo pool on "non-level days": every day has a highest positive peak in a full-repricing profile. Proposal: leave-one-out over all other train days, as in CR-BI P1, run separately for above-open and below-open.
  - "Block bootstrap" = CR-BI's month-cluster bootstrap.
  - T3's trailing-250 window: full window (n ≈ 486) or expanding with a minimum?
  - Holiday-stamped `orats_oi_gamma` rows remapped to the next trading day.

## Amendment A1 (Ryan, 2026-10-11, after Step 0) — Step-0 decisions

Supersedes the Step-0 gates and fills the T1–T3 gaps listed under "For approval" above. **Pass criteria and L1–L10 are otherwise unchanged.**

**Correction to L6 / gate 1 wording:** `stock_price` in `orats_oi_gamma` is **not the prior close**. It is ORATS' 15:46 ET (12:46 PT) EOD snapshot of the prior session, and it is per expiry.

| # | Item | Decision |
|---|---|---|
| A1.1 | S0 | S0 = `stock_price` of the **nearest live expiry** (expir_date ≥ trade_date) — the 15:46 ET snapshot of the prior session. Grid S0 ± 300, step 5 (L6 otherwise unchanged). |
| A1.2 | IV snapshot | Smile from the `orats_monies_minute` **12:46 PT** snapshot of the prior session (latest snapshot ≤ 12:46), not the session's last snapshot. |
| A1.3 | Gate 2 restated | Gate = **exposure-weighted median** absolute gamma error ≤ 5 % (weight = gamma × OI × S², the row's contribution to the profile). Re-validate **once** on a fresh random 20-date train sample not used so far (excluding the Step-0 sample and any date carrying the same data through the holiday remap). If it fails, stop and report. No further variants. |
| A1.4 | Wing method | **Primary:** linear extension in log-strike beyond the smile's Δ5 / Δ95 points (σ floored at 0.03). **Robustness diagnostic only:** flat extension. Report the share of train days on which the top positive peak moves by more than 10 pts between the two methods. Not a second run of T1–T3. |
| A1.5 | Gate 3 | Forward = the row's own per-expiry `stock_price`. Under a grid move each forward shifts proportionally: F_e(S) = S_e · S / S0. AM-settled expiries (`expiry_tod = 'am'`) are flagged and counted, not re-timed. The third-Friday AM/PM overwrite in `orats_oi_gamma` is a **known limit, not fixed**. |
| A1.6 | Gate 4 | The 32 holiday-stamped `orats_oi_gamma` store dates are moved to `next_trading_day(store date)` (their content is the prior session's EOD). |
| A1.7 | Gate 5 | New `scripts/cr_bk_placebo.py`, reusing the matching logic and bootstrap of `scripts/cr_bi_followup_placebo.py`. It must reproduce CR-BI's published cells exactly as a control before the one run; if it does not, stop. |
| A1.8 | Gate 6 | Run the shape report now, on the amended method, before any outcome is computed. The 80 % single-peak stop rule stands. Report gate 7's T2 count with it. |
| A1.9 | T1 placebo | Pool = all other train days (leave-one-out), separately for above-open and below-open levels. Distance matched in units of the open implied move (`implied_move_1d`), not points. |
| A1.10 | T1 secondary readout | Not a pass criterion: T1 reported separately for days with trailing 20-session SPX return > 3 % at the prior close, and for all other days. |
| A1.11 | Bootstrap | CR-BI month-cluster bootstrap (calendar months of the signal date, 10 000 resamples, seed 20260920, 95 % percentile) for every CI. |
| A1.12 | T3 window | Percentile rank over a trailing 250-session window, starting once 120 prior sessions are available. |

**Sequence:** amendment commit → gate 2 re-validation → gate 6 shape report (+ gate 7 T2 count) → **stop and report**, before T1–T3.

## Post-A1 gates (2026-10-11, read-only, train only; no outcome computed)

Code: `scripts/cr_bk_profile.py` (`validate`, `shape`). Profiles saved locally, not committed: `scripts/logs/cr_bk/profiles.parquet` (date × grid × wing) and `scripts/logs/cr_bk/shape_summary.parquet`. One `validate` invocation crashed before any number was computed (pandas `.T` = transpose, not the `T` column). It was fixed and run once.

### Gate 2 re-validation — **PASS**

Fresh sample, seed 20261012, drawn from the effective train dates. It excludes the 20 Step-0 dates and 2024-07-05 / 2024-12-26, which carry the Step-0 holiday-stamped data. Dates: 2023-05-15, 07-12, 09-01, 10-04; 2024-04-12, 07-09, 07-29, 08-15, 10-09; 2025-03-10, 03-17, 04-28, 05-08, 07-10, 08-28, 11-24; 2026-01-16, 01-30, 05-06, 05-22. Method: A1 (12:46 PT smile, per-expiry forward, linear wings). 161,353 rows with OI > 0.

| | value |
| --- | --- |
| **exposure-weighted median \|gamma error\|** | **2.33 %** (gate ≤ 5 %) |
| exposure-weighted p90 | 15.10 % |
| row median (info only) | 4.95 % |
| weighted median by DTE: 0 · 1–7 · 8–30 · 31–90 · 91–400 | 1.39 · 1.61 · 2.22 · 2.34 · 3.56 % (exposure share 9 · 21 · 24 · 14 · 32 %) |

### Gate 6 — shape report (linear wings, `find_walls` prominence 10 %, grid S0 ± 300 step 5)

Effective train dates: 777, including 32 holiday-stamped store dates moved forward. **Usable: 769.** The 8 unusable dates have no prior-session smile at 12:30–12:46 PT:
- 5 follow NYSE half-days (2023-07-05, 2024-07-05, 2024-12-26, 2025-07-07, 2025-12-26). The session closed at 10:00 PT, so there is no 12:46 snapshot; that is a known limit of A1.2.
- 3 are monies gaps (2023-11-27, 2024-12-02, 2025-12-01).

Exposure dropped for a missing expiry smile: 0.00 % on every day. AM-flagged rows: median 2,780 per day.

| positive peaks \ troughs | 0 | 1 | all |
| --- | --- | --- | --- |
| 0 | 186 | 29 | 215 (28.0 %) |
| 1 | 540 | 14 | 554 (72.0 %) |
| ≥ 2 | 0 | 0 | **0** |
| all | 726 | 43 | 769 |

- **Stop rule (> 80 % with exactly one positive peak and no trough): 70.2 % → does not trigger.**
- **No day has more than one positive peak.** The corrected net profile is a single smooth hump or a monotone curve. 186 days have no peak or trough at all: monotone over ±300, net gamma negative at S0 on 82 % of them; by year 40 · 48 · 92 · 35 days in 2023 · 2024 · 2025 · 2026.
- Troughs on 5.6 % of days.
- **Top positive peak sits above S0 on effectively every day:** peak − S0 median +107.5 pts, p10 +55, p90 +180.
- **Zero-gamma flip** inside the grid on 97.9 % of days, never more than one. Flip − S0: median −24.6, p10 −100.9, p90 +108.5 pts. Net gamma at S0 < 0 on 36.8 % of days.
- **Wing robustness (A1.4):** the top positive peak moves > 10 pts between linear and flat wings on **0.9 %** of 552 days (median shift 0.0 pts).

### Gate 7 — usable counts (SPX session + `implied_move_1d` > 0: 737)

| test | n |
| --- | --- |
| T1 5-session, peak above open / below open | **523 / 3** (window inside train: 733) |
| T1 20-session, above / below | **512 / 3** (719) |
| **T2 (positive peak on both sides of the open)** | **0** |
| T3 (≥ 120 prior usable sessions) | 617 |

**Reading (no outcome computed):**
- **T2** is undefined by construction: there are never two positive peaks.
- **T1** reduces to an above-open test. Below-open n = 3 cannot be estimated.
- **T3** is unaffected.

The literal stop rule passes, but its stated purpose ("walls are not defined in the corrected profile → T2 dropped") applies in full. Under A1 and L2, T2 is dropped unless Ryan decides otherwise. Stopped here for review, before T1–T3, per A1 "Sequence".

## Amendment A2 (Ryan, 2026-10-11, after gates 2 and 6) — scope of the one run

Recorded before `scripts/cr_bk_placebo.py` exists or anything is run. Pass criteria and L1–L10 are unchanged.

- **Finding in its own right (gate 6):** on the train sample, **no day has two or more positive peaks** in the corrected net gamma profile (769 usable days: 554 with one positive peak, 215 with none). The multi-wall structure of the old landscape does not survive full repricing. That structure came from the 8·√dte kernel.
- **T2 dropped:** 0 usable days.
- **T1 runs above-open only.** The 3 below-open days are dropped.
- **Go:** build `scripts/cr_bk_placebo.py`. Reproduce CR-BI's published placebo output as the control: the full text of `scripts/logs/cr_bi_followup_placebo.md` except its timestamp line; **stop if it does not match**. Then run T1 (above-open, 5 and 20 sessions, with the trend-up / other secondary readout) and T3 **once**.
- No additional tests, cuts or re-runs. The trough-only days are not tested. Half-day snapshot loss (5 days) accepted as a known limit.

**Operational definitions, fixed before the run** (they fill details the spec leaves open; none is a new test):

1. **Harness base = CR-BI's, unchanged:** `bt_daily_features` `v0.6.0-openiv` (IM = `implied_move_1d`), the CR-BH hybrid SPX-cash daily OHLC (open = first clean print ≥ 06:33 PT), and the leave-one-out pool of all train days whose h-session window ends ≤ 2026-06-05.
2. **T1 signal:** usable corrected-profile day d in the base, with top positive peak L_d > open_d and the window available. z_d = (L_d − open_d) / IM_d. Actual = max high over sessions d…d+h−1 ≥ L_d. Placebo = mean over the pool minus d of 1[max high over h sessions from e ≥ open_e + z_d·IM_e]. The difference, with the CR-BI month-cluster bootstrap (10 000 resamples, seed 20260920).
3. **T1 secondary (readout only):** trend-up = SPX-cash close_{d−1} / close_{d−21} − 1 > 3 %, i.e. the trailing 20-session return at the prior close. Days without 21 prior sessions in the series are left out of the split. Same placebo pool for both halves.
4. **Old landscape alongside T1:** the same harness with L_d = the highest positive wall of `orats_gex_landscape.walls`, put on the strike axis with CR-BI's `wall_carry` (P − carry), above-open days only, all regimes.
5. **T3:**
   - X_d = corrected net gamma (linear wings) interpolated on the grid at open_d; missing if open_d is outside S0 ± 300.
   - Percentile rank = share of the previous ≤ 250 usable X values strictly below X_d, plus half the ties, on (0, 1). Defined once ≥ 120 prior X values exist.
   - Y1 = (high_d − low_d) / open_d. Y2 = (high_d − low_d) / IM_d.
   - Spearman(rank, Y) with the month-cluster bootstrap (same seed and resamples; Spearman recomputed on each resample).
   - Top-tercile (rank > 2/3) minus bottom-tercile (rank < 1/3) mean of Y1 and Y2, with CI.
   - Flip readout: Y1 and Y2 for days with open_d below vs above the nearest zero-gamma flip (where one exists), with CI on the difference.
6. **Old landscape alongside T3:** X from `orats_gex_landscape.landscape` `gex_total` interpolated at open_d, as built (frame B, no carry correction); otherwise identical.

## Results — the one run (2026-10-11 17:47, read-only, train only)

`scripts/cr_bk_placebo.py` at commit `ca06350`, run once. Raw output: `scripts/logs/cr_bk_placebo_run.txt` (git-ignored).
- **Control: PASS.** The CR-BI P1/P2 output, regenerated through the new matching functions, is identical to `scripts/logs/cr_bi_followup_placebo.md` on all 41 content lines. A pre-run `--control-only` check first failed on a comparison artefact: the CR-BI timestamp header `"# …\n"` spans two lines and only one was stripped. Every content line already matched. The strip was fixed and the control passed before the script was committed.
- **Incomplete run:** the run crashed *after* T1 and every corrected-profile T3 statistic had been computed and printed. It failed on the old-landscape T3 comparison (`np.interp` on an object-dtype array built from the `orats_gex_landscape.landscape` JSON). The old-landscape T3 comparison and the script's verdict block were therefore not produced, and `scripts/logs/cr_bk_placebo.md` was not written.
- **Not re-run, per A2.** The verdict below is derived from the printed numbers under the locked criteria. The old-landscape T3 comparison is a report-only item and cannot change any pass/fail.

### T1 — touch at the corrected peak-gamma level (above-open only)

| horizon · level source · subset | n | actual | placebo | difference | 95 % CI |
| --- | --- | --- | --- | --- | --- |
| **5-session · corrected · all above-open** (pool 737) | 523 | 19.1 % | 24.6 % | **−5.5 pts** | [−10.0, −1.0] |
| 5 · corrected · secondary: trend-up (20-session return > 3 %) | 232 | 24.6 % | 30.4 % | −5.9 | [−13.9, +1.0] |
| 5 · corrected · secondary: other days | 273 | 15.4 % | 20.4 % | −5.1 | [−11.5, +3.1] |
| 5 · old landscape (strike-axis walls) · all above-open | 455 | 56.7 % | 56.0 % | +0.7 | [−6.8, +8.0] |
| **20-session · corrected · all above-open** (pool 723) | 512 | 64.8 % | 61.3 % | **+3.5 pts** | [−8.2, +14.5] |
| 20 · corrected · secondary: trend-up | 221 | 84.2 % | 66.9 % | +17.3 | [+5.8, +26.4] |
| 20 · corrected · secondary: other days | 273 | 48.0 % | 57.6 % | −9.6 | [−24.1, +5.3] |
| 20 · old landscape · all above-open | 447 | 77.2 % | 81.4 % | −4.2 | [−15.2, +5.5] |

- **Level distance:** the corrected level sits a median +105.8 pts above the open (z median 2.62, p10 1.25, p90 4.61). The old wall sits at z median 1.10.
- **Trend split coverage:** 18 signal days have no 21-session history and are not in the split.
- **The corrected level is touched *less* often than an arbitrary level at the same IM distance at 5 sessions** (CI excludes 0 on the negative side). At 20 sessions the difference is indistinguishable from 0.
- **The 20-session trend-up readout (+17.3, CI [+5.8, +26.4]) is a secondary readout, not a pass criterion.** Its placebo pool is all days, not trend-up days. A trend-up day reaching a level above the open is exactly the momentum effect the chat checks found (touch tracks momentum, not the wall). The "other days" half goes the opposite way (−9.6). Treat it as a lead at most.

### T2 — dropped (A2: 0 usable days)

### T3 — volatility vs corrected net gamma at the open

n 617 (first ranked day 2023-10-30; X missing on 4 base days with the open outside S0 ± 300).

| statistic | Y1 = range / open | 95 % CI | Y2 = range / IM | 95 % CI |
| --- | --- | --- | --- | --- |
| **Spearman(rank X, Y)** | **−0.554** | [−0.629, −0.454] | −0.115 | [−0.200, −0.025] |
| top-tercile mean (n 257) | 0.683 % | | 1.013 | |
| bottom-tercile mean (n 182) | 1.404 % | | 1.111 | |
| top − bottom | −0.721 pp | [−1.119, −0.435] | −0.098 | [−0.202, +0.005] |
| open below zero-gamma flip (n 189) | 1.239 % | | 1.104 | |
| open above zero-gamma flip (n 413) | 0.774 % | | 1.085 | |
| below − above | +0.466 pp | [+0.324, +0.614] | +0.018 | [−0.099, +0.141] |

- **Old landscape alongside:** not produced (crash, see above).
- **Range is strongly lower on high-net-gamma days in raw terms.** The bottom tercile has about twice the range of the top tercile, and opens below the zero-gamma flip have about 1.6× the range of opens above it.
- **Almost all of that is already in the implied move.** On range / IM, Spearman is −0.115 (CI just excludes 0), the tercile gap is −0.10 IM with a CI touching 0, and the flip gap is +0.02 IM with a CI around 0.
- **The priced effect is small.** It amounts to about 10 % of an implied move between terciles.

### Pass / fail (locked criteria)

| test | result | criterion | verdict |
| --- | --- | --- | --- |
| T1 5-session | −5.5 pts, CI lower −10.0 | ≥ +8, lower > 0 | fail |
| T1 20-session | +3.5 pts, CI lower −8.2 | ≥ +8, lower > 0 | fail |
| **T1** | | either horizon | **FAIL** |
| T2 | — | — | dropped (A2) |
| **T3 raw (Y1)** | Spearman −0.554, CI [−0.629, −0.454] | ≤ −0.10, CI excludes 0 | **PASS** |
| T3 priced (Y2) | Spearman −0.115, CI [−0.200, −0.025] | reported only | — |

**Verdict (spec rules): only T3 raw passes → gamma matters for volatility, not for levels; redirect toward vol-conditioned structures, no wall targets.**
- **Levels:** under correct full repricing, the profile has at most one positive peak per day, sitting a median 106 pts above the open. It is *not* touched more often than placebo. Combined with the CR-BI result for the old landscape, wall-as-level has no support in either construction.
- **Volatility (Y2 qualifier):** the volatility relationship is real in raw terms but mostly priced by the open implied move. Any vol-conditioned structure must earn its edge on the small residual (about −0.1 Spearman on range / IM), not on the raw −0.55.
