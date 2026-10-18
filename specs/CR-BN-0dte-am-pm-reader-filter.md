# CR-BN — 0DTE quote readers: drop AM-settled SPX bars on third Fridays

> Authority: kickoff prompt from the CR-BM session, 2026-10-17 (follow-up filed in CR-BM's session wrap, PR rschultzz/volatility-monorepo#63). No vault CR note exists yet.
> Branch: `feat/CR-BN-0dte-am-pm-reader-filter` (off `origin/main` f29c557). CR-BN unused (no branch or spec; latest is CR-BM).
> Read-only against the DB. No writes, no deploy, no re-run of any pre-registered test.

## Why

CR-BM Step 0 (`scripts/output/cr_bm/step0.md` on the CR-BM branch) found that on 8 train third Fridays (2023-05-19, 06-16, 07-21, 08-18, 10-20, 11-17, 2024-01-19, 02-16) `orats_options_minute` holds bars with `expiry_tod = 'am'` (AM-settled SPX monthly) under the same OPRA symbols as the PM-settled SPXW bars — roughly one AM and one PM row per leg per minute. ORATS files both under root `SPX`, so the OPRA symbol does not tell them apart; only `expiry_tod` does.

`scripts/cr_as_condor_analysis.py` builds a per-minute `(strike, type)` dict from `BARS_SQL` with no `expiry_tod` filter, so on those dates the 0DTE condor entry mid can be an AM-contract quote (whichever row lands last in the dict). CR-BM dropped the 8 dates from its own universe (A1.3) but did not fix the readers. The CR-AH harness (`scripts/cr_ah_step4_analysis.py`, `is_am_settled`) already excludes AM-settled expiries for multi-day spreads (CR-BI A1.4).

## Scope (locked by the kickoff prompt)

| # | Decision | Value |
|---|---|---|
| L1 | Step 0 | Read-only counts: third Fridays 2023-05 → today whose 0DTE bars in `orats_options_minute` mix `expiry_tod` values, including holdout dates written by the CR-AU live capture (`scripts/cron_daily_leg_capture.py`, cr_id `DAILY-CAPTURE`). Counts only — no holdout P&L is read or analysed. Inventory every reader of 0DTE bars from that table. |
| L2 | Fix | 0DTE readers exclude `expiry_tod = 'am'` rows. Unit test in `packages/shared/backtest/tests`. |
| L3 | Archive | CR-AS's archived results (`scripts/logs/cr_as_*`, the CR-AS session note) are not changed and no pre-registered test is re-run. The fix changes what a *future* run of the reader would select. |
| L4 | Fee | `condor_0dte.FEE_PER_CONTRACT_SIDE` (1.30) is **not** changed unless Ryan says so — see Q-fee below. |
| L5 | Visible change | None. No card, cron schedule or table change. |

## Open question for Ryan

**Q-fee.** `packages/shared/backtest/condor_0dte.FEE_PER_CONTRACT_SIDE` is still $1.30/contract/side (CR-AS decision 8); `packages/shared/config.FEE_PER_CONTRACT_PER_LEG` is $0.65 and is what CR-BM and the harness use. Pointing the condor constant at config changes `CONDOR_ROUND_TRIP_FEE_PTS` from 0.104 to 0.052 and therefore the default `fee_pts` of `settle_condor` (CR-AS's analysis passes `fee_pts` explicitly from its `--fee` flag, so its archived $0.65 run is unaffected either way). Should CR-BN make that change, or leave it for a separate CR? **Default if unanswered: leave it — not done in this CR.**

## Step 0 — diagnosis gate

1. Third Fridays 2023-05-01 → 2026-10-17 with 0DTE bars (expir_date = trade_date) in `orats_options_minute`; of those, how many carry both `am` and non-`am` rows; split train (≤ 2026-06-05) vs holdout; split by writer (CR-AS capture window 06:30–06:45 vs DAILY-CAPTURE dates).
2. `expiry_tod` value inventory on 0DTE SPX rows (am / pm / NULL) — decides whether the filter is `= 'pm'` or `IS DISTINCT FROM 'am'`.
3. Does any non-third-Friday 0DTE date carry `am` rows?
4. Reader inventory: every place in `scripts/`, `packages/shared/backtest/` (and `packages/shared/options_cache/`, `apps/`) that SELECTs bars from `orats_options_minute`, classed as 0DTE reader / non-0DTE reader / count-only / writer.

## Implementation order

1. Spec freeze (this file).
2. Step 0 findings appended here.
3. Shared predicate + pure row filter in `condor_0dte`, unit tests.
4. Apply to the 0DTE readers.
5. Wrap: deltas, decisions, follow-ups.
