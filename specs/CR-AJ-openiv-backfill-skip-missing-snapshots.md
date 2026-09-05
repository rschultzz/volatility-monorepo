# CR-AJ — openiv backfill: `--skip-missing-snapshots` flag

> Authority: kickoff prompt, session 2026-09-04 (vault session note to follow at wrap)
> Branch: `feat/CR-AJ-openiv-backfill-skip-missing-snapshots` (off `origin/main` 9b77191)
> Scope: one script flag + one data run. No other code changes.

## Context

The `orats-eod-gamma` cron ran a pre-PR#39 build from 2026-06-08 through 2026-09-04,
writing `bt_daily_features` rows under `v0.5.0-rebuilt` (with `implied_move_1d = 0`)
instead of the canonical `v0.6.0-openiv`. Redeployed to 9b77191 on 2026-09-05; a manual
re-run produced the `v0.6.0-openiv` row for 2026-09-04 with `implied_move_1d = NULL`.

Goal: backfill `v0.6.0-openiv` features + IV + outcomes for 2026-06-08 → 2026-09-04 (SPX)
using the existing CR-AB / CR-022 scripts.

## Problem

Gate 0.6 in `scripts/cr_ab_backfill_openiv.py` (`_MISSING_SNAPSHOTS_SQL`) scans the
**whole** `v0.5.0-rebuilt` corpus for dates lacking a 06:33 PT `orats_monies_minute`
snapshot, ignoring `--from-date` / `--to-date`, and exits 1 if any exist.

Two such dates exist today: **2026-06-19** and **2026-07-03**. Both are market holidays
whose rows were mis-stamped by `next_business_day()` (the row computed on 06-22 was
stamped 06-19; the row computed on 07-06 was stamped 07-03). See open question
`next-business-day-skips-holidays`. These rows must **not** be deactivated here — that is
a separate decision. The orphaned real sessions (06-22, 07-06) have no landscape row and
are out of scope for this CR.

## Change

Add `--skip-missing-snapshots` to `scripts/cr_ab_backfill_openiv.py`.

When set:
- Gate 0.6 logs each missing date at WARNING instead of exiting 1.
- The missing dates are excluded from the target list.
- The `bt_backfill_runs.smoke_test_results` payload gains
  `skipped_missing_snapshot: [<ISO dates>]`.

When not set: behavior is byte-identical to today (gate exits 1).

Docstring note: the flag exists for the mis-stamped-holiday case, not as a general
tolerance for missing data.

## Non-goals

- No changes to `v0.5.0-rebuilt` rows, to 2026-06-19 / 2026-07-03 / 2026-06-22 /
  2026-07-06, or to any manually-tampered row.
- No deactivations.
- No changes to `cr_ab_open_implied_move.py`, `cr_b_backfill_outcomes.py`,
  `backfill_daily_features.py`, or `packages/shared/`.
- No layout / frontend changes.

## Step 0 findings (2026-09-04, read-only)

| Script | Date range | Safety scaffolding | Version written |
|---|---|---|---|
| `cr_ab_backfill_openiv.py` | `--from-date` / `--to-date` / `--limit` | yes (CR-AB) | hardcoded `v0.6.0-openiv`, sourced from `v0.5.0-rebuilt` date list |
| `backfill_daily_features.py` | `--since` / `--date` only | **no** (plain `DATABASE_URL`, UPSERT) | `day_features.FEATURE_VERSION` = `v0.5.0` unless `--version` |
| `cr_ab_open_implied_move.py` | single `--date` or auto-detect newest NULL | yes (CR-AB) | `CANONICAL_FEATURE_VERSION` |
| `cr_b_backfill_outcomes.py` | `--from-date` / `--to-date` / `--limit` | yes (CR-022) | `CANONICAL_FEATURE_VERSION` |

Coverage, SPX, 2026-06-08 → 2026-09-04:

| Metric | Count |
|---|---|
| Weekdays | 65 |
| `orats_gex_landscape` dates | 63 |
| `v0.5.0-rebuilt` active rows (all `implied_move_1d = 0`) | 63 |
| `v0.6.0-openiv` rows in range | 1 (2026-09-04, IM NULL) |
| Clean targets (v0.5.0 + landscape + 06:33 snapshot, no v0.6.0) | 60 |
| `bt_daily_outcomes` rows in range (any version) | 0 |
| Gate 0.6 missing dates (whole corpus) | 2 (06-19, 07-03) |

Other findings surfaced (not fixed here):
- `sweep_pending_outcomes` cron is live at 851ba52 (PR #38), where the canonical was
  still `v0.5.0-rebuilt`; its runs 06-29 → 08-27 swept v0.5.0 rows. 12 `v0.6.0-openiv`
  `pending_history` rows (2026-04-02 → 06-05) have never been swept. Needs redeploy.
- `open-implied-move` and `backfill_outcomes` crons are at 642ca21 (PR #39); every cron
  has `autoDeploy: yes` but none redeployed after June merges (open question
  `orats-eod-cron-not-autodeploying`).

## Locked expected counts

| Step | Expectation |
|---|---|
| 1 dry run | 60 dates, 2026-06-08 … 2026-09-03; 06-19 and 07-03 skipped by flag; 09-04 excluded by existing NOT IN |
| 2 real run | inserted=60 skipped=0 failed=0; `v0.6.0-openiv` active 744 → 804 |
| 3 IV fill | `cr_ab_open_implied_move.py` no args fills 2026-09-04 only |
| 4 outcomes | "Target dates: 61" |
| 5 final query | `v0.6.0-openiv` 804 rows, min 2023-05-01, max 2026-09-04, IM NULL count 0 |

## Implementation order

1. Spec freeze (this file).
2. Add `--skip-missing-snapshots` to `cr_ab_backfill_openiv.py`. No other edits.
3. Data run Steps 1–5, each pasted and confirmed before the next.
4. Smoke + wrap in the session note.

## Smoke + wrap (2026-09-04, run 2026-09-05 03:44–03:49 UTC)

Interpreter: `apps/web/.venv/bin/python` (native arm64; `python3` and the repo `.venv`
carry x86_64 psycopg wheels and cannot connect on this machine).

| Step | Run id | Locked | Actual |
|---|---|---|---|
| 1 dry run | a22509a6 | 60 dates 06-08 … 09-03 | 60 dates 06-08 … 09-03; 06-19 / 07-03 skipped; `skipped_missing_snapshot` present |
| 2 real | c2d60482 | inserted=60 skipped=0 failed=0; 744 → 804 | inserted=60 skipped=0 failed=0; 744 → 804; 60 rows carry run_id |
| regime parity | — | 0 mismatches | 0 mismatches (60/60); IM 26.4–120.2 pts, none NULL/zero |
| 3 IV fill | b7e971e9 | fills 09-04 only | auto-detected 09-04, implied_move=68.22, n_features=35 |
| 4 outcomes | 5847d6a8 | Target dates: 61 | Target dates: 61; inserted=61; computed 18 / pending 6 / na_regime 37 / na_data 0 |
| 5 final | — | 804 rows, IM NULL 0 | `v0.6.0-openiv` 804, 2023-05-01 → 2026-09-04, IM NULL 0 |

### Deltas from spec

- None in scope. The 37 na_regime rows equal the 32 amplification + 2 bounded + 3 untethered dates from the dry run.

### Deferred / surfaced (not fixed here)

- `sweep_pending_outcomes` cron was redeployed from chat tonight (dep-dadov7vqj5pc7391rkq0, live on 9b77191). Verify Monday 2026-09-07 11:05 UTC smoke row reports the `v0.6.0-openiv` split (18 pending in scope), not the v0.5.0 457/13/273.
- Outcomes runner counts holiday partial ES sessions (06-19, 07-03) as RTH sessions for horizon counting. Appended to vault open question `next-business-day-skips-holidays`.
- `bt_backfill_runs` fidelity: dry-run opens a run row claiming "inserted N/N"; `rows_inserted` flushes every 50 and never at end. New vault open question `bt-backfill-runs-audit-record-fidelity`; fix in `backfill_safety.py` / callers, separate CR.
- 06-22 / 07-06 orphan sessions (no landscape row) deferred to the holiday open question.
- Labor Day 2026-09-07 will reproduce the mis-stamp pattern on 09-07 / 09-08.
- `feature_config_hash='tampered'` row (2023-06-01, v0.5.0-rebuilt) is a test fossil; untouched, out of scope.

Session note: `Dash/sessions/2026-09-04-cr-aj-openiv-gap-backfill.md`.
