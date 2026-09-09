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
