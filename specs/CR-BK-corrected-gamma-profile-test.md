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
