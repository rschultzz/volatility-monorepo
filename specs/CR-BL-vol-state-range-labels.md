# CR-BL — Vol State Table and Realized-over-Implied Range Labels

> Authority: vault CR `Dash/CRs/CR-BL — Vol State and Realized-over-Implied Labels.md` (last_reviewed 2026-10-11), kickoff prompt 2026-10-11.
> Branch: `feat/CR-BL-vol-state-range-labels` (off `origin/main` 9949d24). CR-BL confirmed unused (no branch, spec or other vault reference).
> Locked decisions L1–L14 and pass criteria transcribed verbatim from the vault CR; treated as confirmed by Ryan via the kickoff prompt.


## Why

Chat review on 2026-10-11 of SpotGamma Founder's Notes ahead of the year's largest ES moves (18 notes read, 7 move windows; 2025-10-10, 2025-11-13/20, 2025-12-16, 2026-01-20, 2026-03-26, 2026-03-31) found no directional calls. What was flagged in advance each time was a combination: implied vol at the floor, gamma zero or negative, and a level whose break flipped the desk risk-off.

Dash stores the gamma half and almost none of the vol half:

- `atm_iv_percentile`, `skew_percentile`, `term_structure_slope`, `vol_risk_premium` are NULL on all 242 active `v0.6.1-nyse-sessions` rows of `bt_daily_features` since 2025-10-10.
- The only realized-vs-implied label is the one-day `range_over_im` in `bt_daily_outcomes`.

A mispriced range is realized versus implied. Gamma alone does not give one: on train dates, a crude call-minus-put tilt from `orats_oi_gamma` roughly doubles raw ES range from the most positive to the most negative quintile, but `range_over_im` is nearly flat (1.07 to 1.16; close beyond the implied move 12.8 % vs 18.1 %, n ≈ 148 per quintile). CR-BK's "Why" records the same flatness across regimes.

This CR builds the missing vol-state data and multi-horizon realized-over-implied labels, then runs one pre-registered test of whether vol state, alone and combined with gamma, identifies days where the range is mispriced.

## Prior looks (disclosed)

- Chat ran the crude-tilt vs `range_over_im` table above on train dates (≤ 2026-06-05). It used gamma only. The T1 variable in this CR (implied-vol percentile vs realized-over-implied) has not been looked at.
- An earlier chat table of crude tilt vs raw ES range ran through 2026-10-09, so ES ranges by tilt in the post-2026-06-05 window have been seen. No trade outcomes were read. Any later holdout confirmation of a range strategy must note this.

## Locked decisions (Ryan to confirm before spec freeze)

| # | Decision | Value |
|---|---|---|
| L1 | Scope | Two new tables, a backfill, one test run. No change to `bt_daily_features`, `bt_daily_outcomes`, `orats_gex_landscape`, any card, any existing cron. No new cron in this CR. |
| L2 | Tables | `bt_vol_state_daily` (one row per ticker, trade_date, version) and `bt_range_labels_daily` (one row per ticker, trade_date, horizon_sessions, version). SPX only. |
| L3 | As-of rule | Every vol-state field for date t is observable before the 06:30 PT open of t: implied vol from the prior session's 12:46 PT `orats_monies_minute` snapshot (same snapshot rule as CR-BK A1.2), realized vol through the prior close. |
| L4 | Implied vol | Constant-maturity ATM IV at 7 and 30 calendar days, interpolated linearly in total variance between the bracketing expiries' `atmiv`. One-day implied move = the existing open-straddle `implied_move_1d` (ADR locked), reused as stored, not recomputed. |
| L5 | Realized vol | SPX cash, CR-BH hybrid series (cutover 2023-11-09). Close-to-close and Parkinson (high-low), each over 5 and 21 sessions, annualized on 252. |
| L6 | Premium | `vrp_30` = iv_30d − rv_21_cc. `vrp_7` = iv_7d − rv_5_cc. Vol points. |
| L7 | Percentiles | Rank within the trailing 250 sessions, starting once 120 prior sessions exist (same window as CR-BK A1.12). Stored for iv_7d, iv_30d, rv_21_cc, vrp_30, vrp_7. |
| L8 | Labels | Horizons 1 and 5 sessions. `range_over_implied` = (max high − min low over the horizon, SPX cash RTH) / implied move. `abs_close_over_implied` = \|close at horizon end − open of t\| / implied move. Implied move: horizon 1 = `implied_move_1d`; horizon 5 = SPX open × iv_7d × sqrt(5/252). The two horizons use different implied bases and are not compared with each other. |
| L9 | Axis | SPX cash for every price, range and label. No ES in this CR. |
| L10 | Sessions | NYSE calendar via `packages/shared/sessions.py`. |
| L11 | Dates | Tables are populated for every date with inputs. Tests read train only: trade_date ≤ 2026-06-05, and for horizon 5 the horizon must end ≤ 2026-06-05. Holdout label rows are written but not read, summarised or plotted. |
| L12 | Writes | Unattended backfill protocol from `CLAUDE.md` (`get_backfill_db_conn`, `assert_role_or_die`, `backfill_run`), INSERT-only, `version` column on every row. Table creation by migration in this CR. |
| L13 | Run discipline | Thresholds and pass criteria below are committed before any label is joined to any feature. One run. Month-cluster bootstrap as CR-BI (10 000 resamples, seed 20260920, 95 % percentile). No re-tuning. |
| L14 | Gamma input for T2/T3 | CR-BK's corrected net gamma percentile at the SPX open (A1.12), read from CR-BK's saved output. If CR-BK stops before producing it, T2 and T3 are not run. Chat's crude tilt is not a substitute. |

## Step 0 — diagnosis gate (answer all before implementation; contradiction-stop)

1. Confirm CR-BL is unused.
2. `orats_monies_minute` (and `_archive`) inventory: date range, expiries per day, share of sessions with a snapshot at or before 12:46 PT, share of days where expiries bracket both 7 and 30 calendar days. Usable date count.
3. Find the code that should populate the four vol columns in `bt_daily_features` and say why they are NULL (`vol_risk_premium` is known NULL since 2026-05-22; the other three are NULL across the last year). Report only. Do not fix here.
4. State exactly how `implied_move_1d` and the existing `range_over_im` are computed and in which frame (ES or SPX). If the existing label is ES-based, report the difference against the SPX-cash horizon-1 label on overlapping train dates (distribution only, no feature joins).
5. SPX RTH high/low source: confirm the CR-BH hybrid series gives session high, low, open, close for all train dates; quantify the known minute-sampling understatement (median 0.75 / 1.38 / 2.25 pts on high / low / range) against the implied move.
6. AM-settled expiries (`expiry_tod = 'am'`): say whether they are excluded from the interpolation or kept, and how many days it changes.
7. Sanity report on the built series, no outcomes: distribution of iv_7d, iv_30d, rv_21_cc, vrp_30 by year, and ten hand-checked dates.
8. Confirm no lookahead for every stored field (L3).
9. Usable train dates for T1 at each horizon. Stop if fewer than 500.
10. Correlation index feasibility only: is CBOE 1-month implied correlation daily history obtainable and usable under its terms. Report. No ingest in this CR.

## Tests (pre-registered)

Train dates, SPX cash, one run. Y at each horizon: P(`abs_close_over_implied` > 1), mean `range_over_implied`, and P(`abs_close_over_implied` < 0.5).

**T1 — vol state alone.** X = iv_7d percentile. Compare bottom tercile (cheap) against top tercile (rich). Also Spearman of X against `range_over_implied`. Secondary readout, not a pass criterion: the same split on vrp_7 percentile.

**T2 — cheap vol with low gamma (gated on CR-BK, L14).** Cell = iv_7d percentile ≤ 33 and CR-BK gamma percentile ≤ 33. Compare against all other train days.

**T3 — rich vol with high gamma (gated on CR-BK, L14).** Cell = iv_7d percentile ≥ 67 and CR-BK gamma percentile ≥ 67. Compare against all other train days.

## Pass criteria (locked)

| Test | Pass |
|---|---|
| T1 | P(close beyond implied) in the cheap tercile exceeds the rich tercile by ≥ 8 pts at either horizon, CI lower bound > 0 |
| T2 | P(close beyond implied) in the cell exceeds other days by ≥ 8 pts at either horizon, CI lower bound > 0, cell n ≥ 40 |
| T3 | P(close inside half implied) in the cell exceeds other days by ≥ 8 pts at horizon 1, CI lower bound > 0, cell n ≥ 40 |

Verdicts:

- **T1 or T2 passes** → cheap-range days are identifiable; follow-up CR specifies a long-range structure, adds a daily cron for both tables, and confirms once on the holdout.
- **T3 passes** → entry filter for the parked 0DTE condor; follow-up CR re-runs the condor with it, train then holdout.
- **Nothing passes** → the tables stay as data; record in State of Knowledge that neither vol state nor vol-with-gamma separates realized from implied at these horizons.

## Running alongside CR-BK

- Separate worktree and branch. No shared files are edited: this CR adds new scripts, a migration and two tables, and reads `packages/shared/sessions.py` without changing it.
- CR-BK is read-only against the database and writes parquet locally; this CR writes only to its own two tables.
- Both read `orats_monies_minute` heavily on a 1 GB Postgres plan. Do not start this CR's backfill while CR-BK's single T1–T3 run is executing.
- Step 0, the build and T1 do not need CR-BK. T2 and T3 wait for CR-BK's output.

## Outputs

- Repo spec `specs/CR-BL-vol-state-range-labels.md` with Step 0 answers, results tables and verdict.
- `bt_vol_state_daily` and `bt_range_labels_daily` populated, run recorded in `bt_backfill_runs`.
- Session note in `Dash/sessions/` with the verdict.

## Not in scope

- Fixing the NULL vol columns in `bt_daily_features`.
- Any gamma computation (CR-BK owns it).
- Correlation index ingest, intraday flow, skew and term-structure features.
- Holdout reads, trade structures, P&L.
- Daily cron for the new tables.

