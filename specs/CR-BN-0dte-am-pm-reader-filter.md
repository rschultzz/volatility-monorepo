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

## Step 0 findings — 2026-10-17 19:03 PT

Script `scripts/cr_bn_step0.py` (read-only transaction, counts only); full output `scripts/output/cr_bn/step0.md`. Range 2023-05-01 → 2026-10-17, ticker SPX, 0DTE = `expir_date_d = trade_date_d`.

**Q1 Third Fridays.** 42 in range (train 37, holdout 5); 33 have 0DTE rows. **Mixed (`am` + `pm` rows): 8, all train** — 2023-05-19, 06-16, 07-21, 08-18, 10-20, 11-17, 2024-01-19, 02-16, the same 8 CR-BM found. **Holdout: 0 mixed.** The two holdout third Fridays with 0DTE rows (2026-09-18, 2026-10-16, both written by DAILY-CAPTURE, 22 runs 2026-09-08 → 10-16) carry `pm` rows only; 2026-06-19 / 07-17 / 08-21 have no 0DTE rows. No third Friday after 2024-02-16 carries an `am` row.

**Q2 `expiry_tod` inventory** on 0DTE SPX rows: `pm` 1,021,686 rows / 801 dates; `am` 1,056 rows / 8 dates; **NULL: none.** → the filter selects PM explicitly (`expiry_tod = 'pm'`); with no NULLs this is identical today to `IS DISTINCT FROM 'am'`, and a future NULL row shows up as a missing leg rather than an unlabelled quote.

**Q3** Non-third-Friday 0DTE dates with `am` rows: **0**.

**Q4 (new — not in CR-BM).** The table's PK is `(opra_symbol, snapshot_pt)` and `insert_bars` is `ON CONFLICT DO NOTHING`. So on a mixed date each (contract, minute) holds **one** row — AM or PM — not both. In the 06:30–06:45 PT window the AM row took 112 / 96 / 128 / 128 / 128 / 144 / 160 / 160 slots against 144 / 96 / 128 / 128 / 128 / 112 / 96 / 96 PM slots on the 8 dates; every contract still has ≥ 1 PM bar. Consequence: the reader fix stops AM quotes being priced, but it cannot recover the PM quote for the minutes the AM row occupies — on these 8 dates the PM series has gaps (roughly half the window's minutes). The CR-AS capture also records the window as fetched, so a re-fetch would be a cache hit. Recovering the PM minutes needs a writer/schema change (`expiry_tod` in the key, or a delete-and-refetch) — out of scope here (read-only CR); filed as a follow-up.

**Q4b Reader inventory** (every SELECT from `orats_options_minute` in `scripts/`, `packages/shared/`, `apps/`):

| file | what it reads | 0DTE? | action |
|---|---|---|---|
| `scripts/cr_as_condor_analysis.py` `BARS_SQL` | per-minute bid/ask for the condor legs, 06:30–06:45 | **yes — prices** | add `expiry_tod = 'pm'` |
| CR-BM `scripts/cr_bm_condor_legs_by_wing.py` (PR #63, unmerged) | imports `BARS_SQL` from the CR-AS script | **yes — prices** | inherits the fix; column list of `BARS_SQL` kept unchanged so its 5-tuple unpack still works; its universe already drops the 8 dates |
| `scripts/cr_as_capture_0dte_condor_legs.py` `BARS_IN_WINDOW_SQL` | count per leg → leg status ok/empty | yes — count | add predicate (status counts PM bars) |
| `scripts/cron_daily_leg_capture.py` (DAILY-CAPTURE) | count per leg → leg status ok/empty | yes (condor legs) and debit legs | add predicate. Never prices; the status now reflects usable PM bars |
| `scripts/cr_ah_step4_analysis.py` (harness) | debit-vertical quotes; `is_am_settled` | no (multi-day) | none — already excludes AM-settled trades (CR-BI A1.4) |
| `scripts/cr_bi_step0b_resettle.py` | selects `expiry_tod` to flag AM | no | none — diagnostic |
| `scripts/cr_ah_step2_stratified_backfill.py`, `cr_ai_stage2_backfill.py`, `cr_ah_step25b_debit_recovery_probe.py`, `cr_ai_step0_diagnosis.py`, `cr_ai_step0b_coverage_probe.py`, `cr_ae_cleanup_empty_windows.py` | debit-vertical captures / one-off probes (CR-AH/AI/AE) | no | none — archived one-offs, non-0DTE |
| `scripts/cr_am_holdout_leg_capture.py`, `cr_aa_sweep_pending_outcomes.py`, `cr_ap_capture_snapped_legs.py` | writers (`fetch_option_bars`) | — | none |
| `packages/shared/options_cache/repository.py` `get_bars_for_contract` / `count_bars_for_contract` | generic per-contract read | not 0DTE-specific; no callers in repo | none — noted |
| `packages/shared/backtest/net_price.py` | docstring reference only | — | none |
| `apps/` | no reader | — | — |

**Archive (L3).** CR-AS's archived logs and session note are untouched and nothing is re-run. A future re-run of `cr_as_condor_analysis.py` would now price the 8 dates on PM bars only (fewer minutes) and so would not reproduce run 85b690ec on those dates — intended. CR-BM's G1a re-derivation runs on its 713-date universe, which excludes the 8, so it is unaffected.

**Gate:** no contradiction with the spec. One amendment, recorded here before code: the fix is reader-side only and leaves PM gaps on the 8 dates (Q4); the writer-side recovery is a follow-up. Q-fee unanswered → fee not changed.
