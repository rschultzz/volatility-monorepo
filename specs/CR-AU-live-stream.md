# CR-AU — Live out-of-sample stream

> Authority: vault session note `Dash/sessions/2026-09-07 - CR-AU — Live Out-of-Sample Stream.md`
> Branch: `feat/CR-AU-live-stream` (off `origin/main` 04e13a2, the CR-AV merge)
> Scope: daily leg capture cron script, matured-trade capture in the outcome sweep, monthly reference re-run wrapper, holdout read rule in the harness, fees in the debit harness, `render.yaml` documentation entries. **No Render cron creation, no deploy.**
> Mode: unattended through PR; halts at the first STOP gate that misses.

## Problem

The corpus grows one honest row per day (features, IM, outcomes, containment) but nothing captures the *prices* a new signal date would have traded at. Every post-2026-06-05 date has an outcome and no priceable trade unless someone fetches its legs by hand (CR-AM/AO/AP did, once). The reference backtest is a snapshot (`cr_id='CR-AR'`, universe ≤ 2026-06-05) with no re-run schedule, and the holdout has no read rule. "Self-updating" exists on the outcome side and not on the price side.

## Locked decisions

| # | Decision | Value |
|---|---|---|
| 1 | Daily leg capture | New script `scripts/cron_daily_leg_capture.py`, new Render cron **06:35 PT (13:35 UTC) Mon–Fri**, after `open-implied-move`. For today's canonical feature row: if regime is magnet-above → build the debit pair (target−10 / target calls, `snap_spread_to_listed`, 15-DTE expiry chosen as the harness does) and fetch the 06:30–06:45 PT window for both legs. **Every day regardless of regime** → build the symmetric ±0.5 IM condor box on the SPX 06:33 spot (CR-AS convention) and fetch its 4 legs, same window. Dedupe against existing contract-windows. Backfill role, `backfill_run` cr_id `DAILY-CAPTURE`. Log legs / 404s / `unlistable`. |
| 2 | Matured-trade capture | In `cr_aa_sweep_pending_outcomes.py`: after promoting a magnet-above outcome to `computed`, fetch the debit's settlement window (expiry day 12:50–13:00 PT) and, if `reached_touch`, the touch window the harness uses (from `detect_touch`, 90 min). Same dedupe. cr_id `DAILY-CAPTURE-MATURE`. |
| 3 | Monthly reference re-run | `scripts/run_reference_rerun.py`: wraps `cr_ah_step4_analysis.py` with `--universe-end <last calendar month end> --split-date 2026-06-05 --structural-prob-mode walk-forward --cr-id REF-YYYY-MM`. New Render cron **1st of month 14:00 UTC**. Persists cells; the card's stamp reads the latest `cr_id` matching `REF-%` (falling back to `CR-AR` until the first run). |
| 4 | Holdout read rule | In the harness: when `partition='holdout'` rows exist, print only `holdout n per band` and `holdout dates matured`. Print holdout P&L **only** if `--holdout-read <path-to-preregistration.md>` is passed **and** holdout magnet-above computed count ≥ 60. Otherwise print "holdout: n=<k>, unread (threshold 60)". Record the rule in the ADR [[2026-09-05 - Holdout Split Moves to 2026-06-05]] as an amendment. |
| 5 | Fees | `FEE_PER_CONTRACT_PER_LEG = 0.65` from shared config (CR-AS) applied in the debit harness too: net P&L = gross − 4 × 2 × fee / 100 per vertical. Persisted cells gain `mean_pnl_net`; existing `mean_pnl` stays gross. |
| 6 | Cron creation | Chat-side via Render MCP after merge (permission per cron). Build/start commands mirror the existing `cr_ab_open_implied_move` cron (rootDir "", `pip install -r apps/cron/requirements.txt`). `render.yaml` gains both entries for documentation, `autoDeploy: false` noted as dead in practice. |
| 7 | Not in scope | Wall-aware or pin-only condor boxes (add to the capture list after CR-AT); the card; any holdout read. |

## Gates

| Gate | Expected | On miss |
|---|---|---|
| G0 — `main` contains CR-AV merge (`04e13a2`); capture path and `snap_spread_to_listed` callable from a cron-style script with no web imports | yes | STOP |
| G1 — tests: leg builder for magnet/non-magnet days; condor box builder; dedupe; holdout gate (prints n only without flag; raises if flag without file; prints P&L only when n ≥ 60 and flag) | pass; suites pass | STOP |
| G2 — dry run of daily capture for 2026-09-04 (a magnet day) and 2026-09-02 (non-magnet): correct leg list printed, nothing fetched | yes | STOP |
| G3 — real run of daily capture for 2026-09-04: legs fetched, run row present | yes; 404s listed | STOP on exception |
| G4 — reference re-run dry: `--universe-end 2026-08-31` resolves; selection prints; **no holdout P&L in the log** | yes | STOP and redact |
| G5 — `git diff --stat` confined to the files listed | yes | STOP |

## Kickoff prompt

```
CR-AU — live out-of-sample stream. Unattended through PR. Authority: vault
note "Dash/sessions/2026-09-07 - CR-AU — Live Out-of-Sample Stream.md".
Halt only at STOP gates; on halt write "## Halt", commit, push, end.
DO NOT create Render crons; DO NOT deploy. Work in worktree
.claude/worktrees/cr-au if another session holds the main checkout.

Branch: git fetch; git checkout -b feat/CR-AU-live-stream origin/main.
Commit 1 — spec freeze from the note.
Step 0 → Commit 2: G0; read the cron scripts' env handling (ORATS token,
  DATABASE_URL vs BACKFILL_DATABASE_URL in the crons' env group) and record
  which URL a cron-run capture must use.
Commit 3 — cron_daily_leg_capture.py + tests (decision 1). --dry-run.
Commit 4 — sweep matured-trade capture (decision 2) + tests.
Commit 5 — fees in the debit harness (decision 5) + tests; run_reference_rerun.py
  (decision 3); holdout read gate in cr_ah_step4_analysis.py (decision 4) + tests.
Commit 6 — render.yaml entries for the two crons (documentation).
Step 1 — G2 dry runs, G3 real run for 2026-09-04, G4 reference dry run.
  Commit 7 with logs summarised.
Wrap — Commit 8: spec What changed / Decisions; ADR amendment (holdout read
  rule) appended to "2026-09-05 - Holdout Split Moves to 2026-06-05"; vault
  run log; Sessions MOC. Push, open PR "CR-AU — Live out-of-sample stream".
  DO NOT MERGE. Print PR URL, the leg lists from G2, the G3 run row, and
  the exact build/start commands + schedules the two crons need.
```

## Decisions

- **Capture the condor box every day, not just pin/bounded.** The holdout has to price the days the rule would skip, or the rule can't be tested on it.
- **Reference re-run monthly, not nightly.** 30 minutes and it changes by tenths of a point; the card's stamp says which month.
- **Holdout read needs a file, not a flag.** The pre-registration document is the consent; a bare flag is too easy to pass.

## Step 0 — diagnosis findings (2026-09-07, before any implementation code)

Run from the git worktree `.claude/worktrees/cr-au` (branch `feat/CR-AU-live-stream` off `origin/main` 04e13a2). Interpreters are the main checkout's: `/Users/ryan/code/volatility-monorepo/apps/web/.venv/bin/python` (arm64, runs) and `arch -x86_64 /Users/ryan/code/volatility-monorepo/.venv/bin/python -m pytest` (Rosetta, suites); `.env` is an untracked symlink to the main checkout's file.

### G0 — PASS

- `origin/main` tip is `04e13a2` (PR #52, the CR-AV merge); the branch is cut from it.
- Capture path from a cron-style script with no web imports: `scripts/cr_am_holdout_leg_capture.py`, `scripts/cr_ap_capture_snapped_legs.py` and `scripts/cr_as_capture_0dte_condor_legs.py` already call `snap_vertical_pair` / `snap_spread_to_listed` (`packages/shared/options_cache/strikes.py`) and `fetch_option_bars` (`packages/shared/options_cache/fetcher.py`) under the backfill role. `grep` of `packages/shared/options_cache`, `packages/shared/backtest`, `day_features.py`, `probability.py`, `gex_landscape.py`, `outcomes.py` and `scripts/cr_ah_step4_analysis.py` for `apps.`, `flask` or `dash` imports: none. `_materialize_payload` is documented as "kept local to avoid a SQLAlchemy dependency in the cron path".

### Cron env handling (kickoff question)

- Every `scripts/cr_*` backfill script: reads `.env` upward from the script (`os.environ.setdefault`, so a real env always wins), **requires `BACKFILL_DATABASE_URL`**, and sets `os.environ["DATABASE_URL"] = BACKFILL_DATABASE_URL` in-process because `packages/shared/options_cache/repository.py` reads `DATABASE_URL`. The ORATS token is read by `packages/shared/options_cache/http_client.py` from **`ORATS_API_KEY`** (not `ORATS_TOKEN`).
- `apps/cron/db.py` (the EOD-cron family) uses `DATABASE_URL` directly — the owner role. Not used by any `scripts/cr_*` script.
- Live Render cron jobs (read-only listing, workspace `tea-d2r4jkndiees73ds5vn0`, region oregon, plan starter, branch `main`, autoDeploy on commit, rootDir `""`, build `pip install -r apps/cron/requirements.txt`):

| service | id | start command | schedule (UTC) |
|---|---|---|---|
| `orats-eod-gamma` | crn-d43779gdl3ps73a7l4lg | `python job_orats_eod.py` (rootDir `apps/cron`, build `pip install -r requirements.txt`) | `0 10 * * 1-5` |
| `sweep_pending_outcomes` | crn-d8guhf8jo6nc73e1iou0 | `python scripts/cr_aa_sweep_pending_outcomes.py` | `5 11 * * 1-5` |
| `open-implied-move` | crn-d8ime5k8aovs738fef60 | `python scripts/cr_ab_open_implied_move.py` | `35 13 * * 1-5` |
| `backfill_outcomes` | crn-d8gtgougvqtc73arti5g | `python scripts/cr_b_backfill_outcomes.py` | `40 13 * * 1-5` |

  Env var *values* were not read. Per the CR-AB shipped note the post-open cron carries `BACKFILL_DATABASE_URL` (+ `PYTHON_VERSION=3.13.4`); `DATABASE_URL` is not needed by these scripts and must not be the owner URL.
- **A cron-run capture must use `BACKFILL_DATABASE_URL`** (role `dash_backfill_writer`; INSERT on `orats_options_minute` / `orats_options_fetched_windows` proven by the CR-AM/AP/AS runs) **and needs `ORATS_API_KEY`** in its env. The daily-capture cron therefore needs two env vars; the existing `sweep_pending_outcomes` cron needs `ORATS_API_KEY` **added** for decision 2 (chat-side, with the cron creations). Until it is, the sweep must keep promoting outcomes and skip the capture with a warning (implemented, see A2).
- `bt_backfill_runs` today: `CR-AB` completed 13:35:45 UTC, `CR-022` (backfill_outcomes) 13:40:47 UTC — i.e. the jobs fire on the schedule minute and finish within a minute.

### Data facts for the gates

| date | regime (`bt_daily_features.regime_at_classification`) | IM (`implied_move_1d`) | SPX 06:33 spot (`orats_monies_minute.spot_price`) | ES `session_open_t0` | outcome |
|---|---|---|---|---|---|
| 2026-09-04 (G2 magnet, G3) | magnet-above | 68.22 | 7742.90 @ 06:33 | 7742.25 | pending_history, bucket 8-30 DTE |
| 2026-09-02 (G2 non-magnet) | amplification | 51.99 | 7635.27 @ 06:33 | 7650.50 | na_regime |
| 2026-09-07 (today, Labor Day) | row exists (EOD forward-stamp onto a holiday) | — | no monies snapshot | — | — |

- Holdout stream (> 2026-06-05, magnet-above): **22 dates, 15 `computed`** → the decision-4 gate (≥ 60) is far from unlocked; the harness prints `holdout: n=15, unread (threshold 60)`.
- `orats_options_minute` already holds 6 contracts on 2026-09-04 (the CR-AM/AP entry-day captures); the G3 run will show cache hits for the debit legs and new fetches for the condor legs.
- `bt_edge_backtest_results` has no `mean_pnl_net` column (columns end at the CR-AR per-point set); `cr_id`s present: CR-AH, CR-AM, CR-AN, CR-AP, CR-AR — no `REF-%` yet, so the card's fallback to `CR-AR` is the live case.
- `packages/shared/config.py` **does not exist**. The CR-AS fee constant is `FEE_PER_CONTRACT_SIDE = 1.30` in `packages/shared/backtest/condor_0dte.py` (the $0.65 figure was applied by CLI flag in CR-AS Step 2b, never persisted as a constant).

### Spec amendments (recorded before implementation)

- **A1 — daily-capture schedule `50 13 * * 1-5` (06:50 PDT), not 13:35.** Two reasons. (i) Decision 1 says "after `open-implied-move`", which itself fires at 13:35, and the outcomes insert fires at 13:40. (ii) The capture window is 06:30–**06:45** PT and `fetch_option_bars(..., record_empty_windows=True)` records the requested window as covered: a fetch at 06:35 would mark 06:36–06:45 as fetched with no bars — the exact cache-poisoning failure diagnosed in "Condor Pricing Leg-Miss Diagnosis". The script also refuses to fetch a window whose end is less than `MIN_LAG_MIN = 5` minutes in the past (exit 1, nothing recorded). Same PST caveat as CR-AB (fixed UTC schedule drifts one hour in winter; not in scope).
- **A2 — matured-trade capture is a scan, not a one-shot.** The outcome horizon (`bucket_sessions`: 1 / 5 / 20 / 60 sessions) is unrelated to the debit's 15-business-day expiry, so at promotion time the settlement window is often still in the future (`1-7 DTE` days) or long past (`30+ DTE`). Each sweep run therefore scans **every** post-split magnet-above date with `outcome_status = 'computed'`, plans the debit pair (payload target, `snap_vertical_pair(..., 'debit')`, expiry `nth_business_day(td, 15)`), and fetches only the windows that are (a) due — window end ≤ latest closed RTH session — and (b) not already covered in `orats_options_fetched_windows` (`find_gaps`). Touch window: `detect_touch` → `[touch_pt, touch_pt + 90 min]` when `rth_touch` / `gap_touch`, regardless of the outcome row's `reached_touch` flag (the harness's definition is the one the read will use). Runs under its own `backfill_run` (`DAILY-CAPTURE-MATURE`) after the promotion run, never inside it, so a capture exception cannot abort a promotion. If `ORATS_API_KEY` is absent the capture is skipped with a warning and the sweep exits as before.
- **A3 — fees.** New `packages/shared/config.py` with `FEE_PER_CONTRACT_PER_LEG = 0.65` (confirmed 2026-09-07, CR-AS decision 8). A vertical has 2 legs × 2 sides = **4 contract-sides**: net = gross − 4 × fee / 100 = gross − 0.026 pts. The note's literal "4 × 2 × fee" is the condor's count (4 legs × 2 sides) and is read as the vertical's 4 contract-sides here; flagged for review. `condor_0dte.FEE_PER_CONTRACT_SIDE` (1.30) is outside this CR's files and is left alone — follow-up to point it at the shared constant.
- **A4 — `mean_pnl_net` column.** Added to `bt_edge_backtest_results` by the idempotent `ALTER TABLE … ADD COLUMN IF NOT EXISTS` in `ensure_catalog_table` (owner URL, same pattern as CR-AR's per-point columns). Existing `mean_pnl` stays gross.
- **A5 — `--selection-only` on the harness.** G4 asks for a dry reference re-run that resolves the universe and prints the selection without a full 30-minute run; the harness gains `--selection-only` (Phase 1 only: universe, stratified selection, clean filter, unlistable list; no Phase 2–7, no cells). `run_reference_rerun.py --dry-run` passes it with `--no-persist` and `cr_id = REFDRY-YYYY-MM` so no `REF-%` match is ever created by a dry run.
- **A6 — holdout cells are not persisted while the read is locked.** Decision 4 governs the printed read; persisting holdout `mean_pnl` cells would be the same read through the DB. Phase 7 writes holdout cells only when the read is unlocked (`--holdout-read <file>` present **and** n ≥ 60). Holdout **n per band** is still printed.
- **A7 — where "today's canonical feature row" comes from.** `bt_daily_features` (`regime_at_classification`, `feature_vector.implied_move_1d`) at `CANONICAL_FEATURE_VERSION`, most recent `trade_date ≤ today` (or `--date`). The `bt_daily_outcomes` row for the day does not exist until 13:40 UTC. IM: the feature row's `implied_move_1d`; if NULL (the CR-AB fill missed), the same 06:33 open-straddle computation CR-AB uses, with `im_source` logged.
- **Tests** live in a new `scripts/tests/` package (the scripts stay scripts; no new shared module), plus `packages/shared/tests/test_config.py` for the fee constant.
