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
