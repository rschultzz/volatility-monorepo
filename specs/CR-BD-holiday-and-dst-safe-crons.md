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

## Step 1 — repair (decision 4), 2026-09-08 21:28 → 21:31 PT

Executed by `scripts/cr_bd_repair_orphans.py` (new; every command logged, run rows `CR-BD-repair`). Logs: `scripts/logs/cr_bd_repair_*.log` (untracked; connection strings scrubbed).

**Run 1 — `6d208e3f-0ee2-437e-bfc3-9a591339f2d5`: step (a) succeeded, step (b) failed before any write.**
- (a) owner URL: `UPDATE bt_daily_features SET active=false, deactivated_at=now(), deactivated_reason='holiday-mis-stamp'` for (2026-06-19, v0.5.0-rebuilt) → 1 row, (2026-07-03, v0.5.0-rebuilt) → 1, (2026-09-07, v0.6.0-openiv) → 1. Backfill role: same on `bt_daily_outcomes` → 0, 0, 1 (only 09-07 had an outcome row, `na_data`).
- (b) `apps/cron/job_orats_eod.py --date …` exited 1 at `db.get_conn`: `apps/cron/db.py` normalises `postgres://` but not the SQLAlchemy scheme `postgresql+psycopg://` that the local `.env` carries (Render supplies `postgres://`, so production never hits this). **Surfaced, not changed**: the repair script hands the subprocess the plain `postgresql://` form instead (`apps/cron/db.py` is outside this CR's files). (c) then failed as expected on "no landscape row" / "Target dates: 0".

**Run 2 — `9af35280-247c-48f6-95ad-9f39c8c073fc`: steps (b), (c), verify — all exit 0.**
```
FORCE_STORE_DATE=2026-06-22 python apps/cron/job_orats_eod.py --date 2026-06-18   # landscape 601 pts, 2 walls, table_spot 7504.25; features v0.6.0-openiv IM=NULL; 12 566 oi_gamma rows
FORCE_STORE_DATE=2026-07-06 python apps/cron/job_orats_eod.py --date 2026-07-02   # 3 walls, table_spot 7460.175; 12 190 rows
FORCE_STORE_DATE=2026-09-08 python apps/cron/job_orats_eod.py --date 2026-09-04   # 2 walls, table_spot 7715.275; 12 078 rows
python scripts/cr_ab_open_implied_move.py --date 2026-06-22   # implied_move 37.1088 (single probe, past date)
python scripts/cr_b_backfill_outcomes.py --from-date 2026-06-22 --to-date 2026-06-22   # run 30ecb71c, 1 inserted (poll skipped: historical)
python scripts/cr_ab_open_implied_move.py --date 2026-07-06   # implied_move 30.4526
python scripts/cr_b_backfill_outcomes.py --from-date 2026-07-06 --to-date 2026-07-06   # run 1f94c9aa, 1 inserted
python scripts/cr_ab_open_implied_move.py --date 2026-09-08   # implied_move 62.1615
python scripts/cr_b_backfill_outcomes.py --from-date 2026-09-08 --to-date 2026-09-08   # run abc17c7a, 1 inserted (today: polled once, snapshot present)
```

### G2 — before / after — PASS

```
BEFORE (21:28:41 PT)
  date        | act.feat | feature versions                 | landscape | act.outc | outcomes                 | IM
  2026-06-19  |        1 | v0.5.0-rebuilt                   |         1 |        0 | -                        | 0.0
  2026-06-22  |        0 | -                                |         0 |        0 | -                        | None
  2026-07-03  |        1 | v0.5.0-rebuilt                   |         1 |        0 | -                        | 0.0
  2026-07-06  |        0 | -                                |         0 |        0 | -                        | None
  2026-09-07  |        1 | v0.6.0-openiv                    |         1 |        1 | na_data                  | None
  2026-09-08  |        0 | -                                |         0 |        0 | -                        | None

AFTER (21:30:50 PT)
  date        | act.feat | feature versions                 | landscape | act.outc | outcomes                 | IM
  2026-06-19  |        0 | v0.5.0-rebuilt(inactive)         |         1 |        0 | -                        | None
  2026-06-22  |        1 | v0.6.0-openiv                    |         1 |        1 | computed                 | 37.108780313581704
  2026-07-03  |        0 | v0.5.0-rebuilt(inactive)         |         1 |        0 | -                        | None
  2026-07-06  |        1 | v0.6.0-openiv                    |         1 |        1 | na_regime                | 30.452556014261408
  2026-09-07  |        0 | v0.6.0-openiv(inactive)          |         1 |        0 | na_data(inactive)        | None
  2026-09-08  |        1 | v0.6.0-openiv                    |         1 |        1 | pending_history          | 62.16152845842874
```

Each orphan now has exactly one active feature row, a landscape row and an outcome row; the closed days have no active feature or outcome rows. The landscape rows for the three closed days remain (amendment A1: no `active` flag, deletion not authorised). `orats_oi_gamma` now holds the chain under both the holiday date and the true session (same data; harmless to `listed_strikes`, which takes `max(trade_date) < d`).

## Step 2 — G3 daily capture for 2026-09-08 — PASS

`scripts/cron_daily_leg_capture.py --date 2026-09-08 --max-wait-min 0` at 21:31 PT (after the repair, so the day has a canonical magnet-above feature row; the condor-only path was exercised in Commit 6's dry run before the repair: `regime=None IM=62.08 (open_straddle_0633 × spx_open)`, condor 7665P/7675P/7735C/7745C, no debit).

```
2026-09-08  regime=magnet-above  IM=62.16 (feature_vector.implied_move_1d)  SPX06:33=7705.27@06:33:00  ES_open=7712.25  basis=6.98  window=06:30–06:45 PT
  debit: long 7810 / short 7825 C  expiry 2026-09-29  width 15
  condor ±0.5 IM: 7665P / 7675P / 7735C / 7745C  expiry 2026-09-08 (0DTE)
legs planned=6  to_fetch=6  already_covered=0
Run ID: 4fd93a8e-6cb0-45a6-b191-509093dccbb1
  6 legs: bars_in_window=15 each, written=32 each, 0 × 404, 0 empty, 0 exceptions
SUMMARY: 2026-09-08 magnet-above: legs planned=6 covered=0 fetched=6 404=0 empty=0 exceptions=0 bars_written=192; debit=captured condor=captured; no P&L computed
```

`bt_backfill_runs` → `run_id 4fd93a8e-6cb0-45a6-b191-509093dccbb1 · cr_id DAILY-CAPTURE · completed · 2026-09-09 04:31:48 → 04:32:09 UTC · smoke: feature_row true, regime magnet-above, im_source feature_vector.implied_move_1d, structures {debit: captured, condor: captured}, debit {long 7810, short 7825, expiry 2026-09-29, width 15}, condor {7665, 7675, 7735, 7745, expiry 2026-09-08}`.

## G1 — tests and suites — PASS

New: `packages/shared/tests/test_trading_calendar.py` (9: Labor Day 2026, Juneteenth 2026 + orphaned Monday, July 4 observed, Thanksgiving + half-day after, Christmas / New Year 2027, weekends, `nth_trading_day` = harness expiry rule, list sanity, **expiry guard**), `packages/shared/tests/test_snapshot_poll.py` (5: **DST-drift case** 05:35 → 06:34 PT, 120-min timeout with the logged line, past-date single probe, budget selection, clipped last sleep), `scripts/tests/test_job_orats_eod_calendar.py` (3), `scripts/tests/test_cron_snapshot_gates.py` (3), daily capture +2 (condor without a feature row; IM fallback). Suites (Rosetta venv): `packages/shared/tests` **460** (+14), `packages/shared/backtest/tests` **142**, `packages/shared/options_cache/tests` **201** (1 skipped, pre-existing), `apps/web/modules` **265**, `scripts/tests` **44**. All pass.

## What changed

- **`packages/shared/trading_calendar.py`** (new, decision 1): `NYSE_HOLIDAYS` 2023–2027, `is_trading_day`, `next_trading_day`, `prev_trading_day`, `nth_trading_day`, `CALENDAR_VALID_THROUGH = 2027-12-31`; `test_calendar_not_expired` fails once today is past it.
- **`apps/cron/job_orats_eod.py`**: `store_trade_date = next_trading_day(api_trade_date)`; the weekend-only `next_business_day` is deleted; `previous_business_day_with_data` walks `prev_trading_day` (A2). **`scripts/backfill_orats_oi_gamma.py`**: its `next_business_day` delegates to `next_trading_day` (text-level switch; the script imports `pandas_market_calendars`, present in no venv, so it was compiled but not run).
- **`packages/shared/snapshot_poll.py`** (new, decision 2): `poll_until` (injectable clock/sleep), `fetch_open_snapshot`, `poll_budget_for` (120 min for today or later on the PT clock, one probe for a past date), `wait_for_open_snapshot`, `no_snapshot_line`. **`scripts/cr_ab_open_implied_move.py`**: polls for the target date's 06:33 PT snapshot before filling, exits 0 logged without it; a non-trading target date is skipped immediately (`skip_reason_for`); `--max-wait-min`, `--poll-seconds`. **`scripts/cr_b_backfill_outcomes.py`**: polls for *today's* snapshot on a trading-day nightly run (`wait_needed`), exits 0 logged without it; `--no-wait`, `--max-wait-min`, `--poll-seconds`; historical `--to-date` and non-trading days skip the poll. **`scripts/cron_daily_leg_capture.py`**: imports the shared poll; condor box captured whenever the snapshot exists — without a feature row the IM is the open straddle × the snapshot's own spot (`implied_move_fallback`, A4); the debit still needs a magnet-above row (decision 3).
- **`scripts/cr_bd_repair_orphans.py`** (new): decision 4 with logged commands, `--dry-run`, `--only`, G2 verify; connection strings never echoed.
- **Data**: runs `6d208e3f` / `9af35280` (repair, G2 PASS), `30ecb71c` / `1f94c9aa` / `abc17c7a` (outcome inserts), `4fd93a8e` (G3 capture). Not deployed.

## Decisions

- **Hardcoded calendar, not a package** — no calendar dependency exists in any requirements file; the expiry-guard test is the maintenance contract.
- **Holiday landscape rows stay (A1)** — no `active` flag; deletion is an owner DELETE the note did not authorise. Ryan's call.
- **Outcomes cron proceeds without a poll on non-trading days** — nothing to wait for; the backlog should still be inserted.
- **Implied-move fill refuses a non-trading target immediately** — a mis-stamped row must not hold the cron for 120 minutes.
- **`apps/cron/db.py` untouched** — the SQLAlchemy URL scheme is a local-`.env` quirk (Render supplies `postgres://`); the repair script passes the plain form.
- **Repair split across two run rows** — run 1 recorded the failed step (b) honestly rather than being retried in place.

## Open questions

- Delete the three holiday `orats_gex_landscape` rows and the duplicate holiday-dated `orats_oi_gamma` chains (owner DELETE)? Harmless if left; decision 5's query must allow for them.
- `apps/cron/db.py`: accept `postgresql+psycopg://` (one line) so the EOD job runs locally without the workaround.
- Outcomes runner counts holiday partial ES sessions as RTH sessions (decision 6 excluded it).
- Eight private `_NYSE_HOLIDAYS` copies + the harness's holiday-blind `next_weekday` (gap-touch → next RTH open) should import the shared calendar.
- Redeploy: all five data crons (`orats-eod-gamma`, `sweep_pending_outcomes`, `open-implied-move`, `backfill_outcomes`, `daily-leg-capture`) — chat-side after merge; the fix is inert until then. Next holiday-adjacent check: 2026-11-27 (day after Thanksgiving); the DST shift is 2026-11-01.
