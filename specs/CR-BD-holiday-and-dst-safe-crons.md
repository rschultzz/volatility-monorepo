# CR-BD — Holiday- and DST-safe crons; orphan repair

> Authority: vault session note `Dash/sessions/2026-09-08 - CR-BD — Holiday-Safe Crons and Orphan Repair.md`
> Branch: `feat/CR-BD-holiday-safe-crons` (off `origin/main` 69efc0f, the CR-AU date fix)
> Scope: shared trading-day calendar, cron call sites, snapshot poll in the implied-move and outcomes crons, condor-without-feature-row in the daily capture, repair of the holiday-mis-stamped rows. **No deploy.**
> Mode: unattended through PR; halts at the first STOP gate that misses.

## Problem

`[Certain]` The EOD job stamps ORATS date D onto `next_business_day(D)` using a calendar with no market holidays. After Labor Day it stamped Friday 09-04's data onto Monday 09-07 (closed) and produced **no row of any kind for Tuesday 09-08** — no features, no landscape, no outcome, and the new daily capture found nothing to do. Same on 06-22 and 07-06. The implied-move and outcome crons then fail on the phantom date. Every market holiday does this; the DST shift on 2026-11-01 will do it silently every day (13:35 UTC becomes 05:35 PT, before the 06:33 pin).

## Locked decisions

| # | Decision | Value |
|---|---|---|
| 1 | Calendar | One shared helper `next_trading_day(d)` / `prev_trading_day(d)` using an NYSE holiday calendar (a maintained package if already in requirements, else a hardcoded list through 2027 with a test that fails after 2027-12-31 so it gets extended). All `next_business_day` / `previous_business_day*` call sites in the cron path switch to it. |
| 2 | Snapshot polling | `open-implied-move` and `backfill_outcomes` adopt the CR-AU poll: wait for the day's 06:33 PT SPX monies snapshot up to 120 min before proceeding; exit 0 logged if none. Schedules stay in UTC. |
| 3 | Daily capture | Default `trade_date` = today in America/Los_Angeles (already fixed on main); condor box captured whenever the snapshot exists, debit only when a magnet-above feature row exists. |
| 4 | Repair | Under the backfill role where possible, owner role for deactivation: (a) deactivate the mis-stamped feature/outcome rows for 06-19, 07-03, 09-07 (`active=false`, reason `holiday-mis-stamp`); (b) re-run the EOD job with `--date` = ORATS date and the store date forced to the true next trading day for 06-22, 07-06, 09-08; (c) run the IV fill and outcome insert for those three; (d) run the daily capture for 09-08 (past-date single probe). |
| 5 | Verification | After redeploy, the next holiday-adjacent day is checked by a query in the vault (features/landscape row exists for every NYSE trading day since 2026-06-01, none for closed days). |
| 6 | Not in scope | Auto-deploy; anything else in the loose-ends list. |

## Gates

| Gate | Expected | On miss |
|---|---|---|
| G0 — every call site of the old business-day helpers listed | yes | STOP |
| G1 — tests: Labor Day 2026, Juneteenth, July 4 observed, Thanksgiving, Christmas; DST-drift poll case; 2027-12-31 expiry guard | pass; suites pass | STOP |
| G2 — repair: 06-22, 07-06, 09-08 each have exactly one active feature row + landscape row + outcome row; 06-19, 07-03, 09-07 have none active | yes | STOP |
| G3 — daily capture for 09-08 wrote a run row with the condor legs (and the debit legs if 09-08 is magnet-above) | yes | note |

## Kickoff prompt

```
CR-BD — holiday- and DST-safe crons; orphan repair. Unattended through PR.
Authority: vault note "Dash/sessions/2026-09-08 - CR-BD — Holiday-Safe
Crons and Orphan Repair.md". Halt only at STOP gates. DO NOT deploy.

Branch: git fetch; git checkout -b feat/CR-BD-holiday-safe-crons origin/main.
Commit 1 spec freeze. Step 0 → Commit 2: G0 call-site list; state which
  calendar source is used and why.
Commit 3 shared trading-day helper + tests; Commit 4 call sites switched;
  Commit 5 snapshot poll in open-implied-move and backfill_outcomes + tests;
  Commit 6 daily capture: condor-without-feature-row + test.
Step 1 repair (decision 4), in order, logging each command; Commit 7 with
  the G2 query output before/after.
Step 2 G3 run; Commit 8.
Wrap Commit 9: spec What changed; vault run log; update
  next-business-day-skips-holidays (status → resolved pending redeploy);
  loose-ends Q3 → done; Sessions MOC. Push, open PR "CR-BD — Holiday/DST-safe
  crons; orphan repair". DO NOT MERGE. Print PR URL, G2 output, G3 run row.
```

## Step 0 — diagnosis findings (2026-09-08, before any implementation code)

Run on the main checkout (branch `feat/CR-BD-holiday-safe-crons` off `origin/main` 69efc0f). Interpreters: `apps/web/.venv/bin/python` (arm64) for runs, `arch -x86_64 .venv/bin/python -m pytest` (Rosetta) for suites.

### G0 — call sites of the old business-day helpers — PASS (complete list)

**Cron path (switched in Commit 4):**

| file:line | symbol | role |
|---|---|---|
| `apps/cron/job_orats_eod.py:101` | `def next_business_day` — weekend-only skip | defines the store-date stamp |
| `apps/cron/job_orats_eod.py:148` | `store_trade_date = … else next_business_day(api_trade_date)` | **the bug**: stamps ORATS date D onto the next weekday, holiday or not; `FORCE_STORE_DATE` env overrides |
| `apps/cron/job_orats_eod.py:93` | `def previous_business_day_with_data` — walks back calendar days probing ORATS | data-driven, correct by construction; switched to walk `prev_trading_day` so it never probes a closed day |
| `scripts/backfill_orats_oi_gamma.py:185, 295, 303, 352, 428–429` | own `next_business_day` copy + 5 uses | same stamp for the historical `orats_oi_gamma` backfill; switched |

**Not in the cron path (listed, left alone — follow-up):** `scripts/cr_ah_step4_analysis.py:240 next_weekday` (gap-touch → next RTH open; holiday-blind, harness), the same copies in `cr_ai_stage1_days_to_touch.py:88`, `cr_ah_step2_stratified_backfill.py:272`, `cr_ai_stage2_backfill.py:123`; eight private `_NYSE_HOLIDAYS` copies (harness, CR-AH/AI scripts, `packages/shared/backtest/tests/test_expiry_calc.py`) and `apps/web/modules/SetupV2/service.py:428–449` (web path, its own copy). All should eventually import the shared calendar; outside this CR's files.

### Calendar source

No holiday package is in `apps/cron/requirements.txt` or `apps/web/requirements.txt`, and neither venv has `holidays`, `pandas_market_calendars` or `exchange_calendars`. Decision 1 therefore takes the second branch: **hardcoded NYSE list 2023–2027** in `packages/shared/trading_calendar.py` (2023–2026 verbatim from the harness's `_NYSE_HOLIDAYS`, incl. 2025-01-09 national day of mourning; 2027 added: Jan 1, Jan 18, Feb 15, Mar 26, May 31, Jun 18 (Juneteenth observed), Jul 5 (Independence Day observed), Sep 6, Nov 25, Dec 24 (Christmas observed)), `CALENDAR_VALID_THROUGH = 2027-12-31`, and `test_calendar_not_expired` that fails once today passes it. Adding a dependency to the cron image for ten dates a year was not worth a requirements change on all five crons.

### Data facts (G2 "before")

| date | NYSE | `bt_daily_features` | `orats_gex_landscape` | `bt_daily_outcomes` (canonical) | 06:33 monies |
|---|---|---|---|---|---|
| 2026-06-19 Juneteenth | closed | `v0.5.0-rebuilt` active (computed 06-22 03:01 from 06-18 data); no canonical row | row (7504.25) | none | none |
| 2026-06-22 | open | **none** | **none** | **none** | 7523.28 |
| 2026-07-03 Independence Day obs. | closed | `v0.5.0-rebuilt` active (computed 07-06 from 07-02 data) | row (7460.175) | none | none |
| 2026-07-06 | open | **none** | **none** | **none** | 7524.13 |
| 2026-09-07 Labor Day | closed | `v0.6.0-openiv` active, IM NULL (computed 09-08 03:01 from 09-04 data) | row (7715.275) | `na_data` active | none |
| 2026-09-08 | open | **none** | **none** | **none** | 7705.27 |

`orats_oi_gamma` also carries the three holiday dates (the same stamp). ES traded shortened RTH sessions on all three holidays (210 / 171 / 210 bars), so the outcomes runner's session list counts them (pre-existing, out of scope per decision 6). Grants: `dash_backfill_writer` has INSERT/SELECT on features and landscape, INSERT/SELECT/UPDATE on outcomes — so feature deactivation needs the owner URL (decision 4 anticipates this); outcome deactivation can use the backfill role. Both tables have `active`, `deactivated_at`, `deactivated_reason`; `orats_gex_landscape` has **no** `active` flag.

### Repair mechanics (decision 4)

- (b) `apps/cron/job_orats_eod.py --date <ORATS date>` with `FORCE_STORE_DATE=<true next trading day>` (existing env override at line 148): 06-18 → 06-22, 07-02 → 07-06, 09-04 → 09-08. The job runs on `DATABASE_URL` (owner, `apps/cron/db.get_conn`) and `ORATS_TOKEN`; it hard-upserts `orats_oi_gamma` for the store date, upserts the landscape and the non-IV canonical feature row.
- (c) `scripts/cr_ab_open_implied_move.py --date <d>` (backfill role) then `scripts/cr_b_backfill_outcomes.py --from-date <d> --to-date <d>`.
- (d) `scripts/cron_daily_leg_capture.py --date 2026-09-08 --max-wait-min 0`.

### Amendments

- **A1 — landscape rows for the three holidays stay.** `orats_gex_landscape` has no `active` column; removing them means an owner-role DELETE, which decision 4 does not authorise (it names deactivation only). The rows are harmless to the fixed crons (nothing enumerates landscape dates forward) and are listed here for Ryan to delete or keep. Decision 5's post-redeploy query should therefore check *features* per NYSE trading day and treat the landscape holiday rows as known.
- **A2 — `previous_business_day_with_data` keeps its data probe** but walks `prev_trading_day`, so a Tuesday run probes Friday, not Monday. Same result on every day it has ever run; fewer ORATS calls after a holiday.
- **A3 — the outcomes cron's poll targets today in PT**: `cr_b_backfill_outcomes.py` has no date argument for "today" (it inserts every canonical feature row lacking an outcome), so the poll waits for *today's* 06:33 PT snapshot when today is a trading day and exits 0 logged when it is not (`--no-wait` for backfills, where today's snapshot is irrelevant). `cr_ab_open_implied_move.py` polls for its *target* date's snapshot (auto-detected NULL-IM row or `--date`), single probe when that date is in the past, and exits 0 immediately when the target date is not a trading day (a mis-stamped row can no longer stall it for 120 min).
- **A4 — daily capture condor without a feature row** (decision 3): with no feature row there is no `table_spot` either, so the implied move for the box is computed from the 06:33 snapshot's own `spot_price` × ATM IV × √(1/252) (`im_source = open_straddle_0633 × spx_open`); the debit pair still requires a magnet-above feature row.
- Tests live in `packages/shared/tests/test_trading_calendar.py`, `packages/shared/tests/test_snapshot_poll.py`, `scripts/tests/`; the poll moves from `cron_daily_leg_capture.py` into `packages/shared/snapshot_poll.py` so the three crons share one implementation. A small repo script `scripts/cr_bd_repair_orphans.py` performs step (a) so the deactivation is logged in `bt_backfill_runs`.
