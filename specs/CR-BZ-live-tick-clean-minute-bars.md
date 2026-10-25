---
type: cr
cr_id: CR-BZ
title: Price Chart Live Tick and Clean Minute Bars — restore the live-ticking candle, stop stale contract symbols in the web service, and stop the minute-bar ingest from storing partial bars
status: created
run_mode: interactive
size: small-medium
estimated_days: 1-2
data_safety_class: writes_reviewed
dependencies: []
soft_dependencies: [CR-AC, CR-BV]
depended_on_by: []
branch_name: feat/CR-BZ-live-tick-clean-minute-bars
last_reviewed: 2026-10-25
related_decisions: []
related_open_questions:
  - "[[ironbeam-1m-bars-miss-trades-in-fast-minutes]]"
related_sessions:
  - "[[2026-06-07 - Render Env Centralization and Branch Reconciliation]]"
tags: [dash, cr, created, price-chart, ironbeam, ingest, symbols, render, data-quality]
---

# CR-BZ — Price Chart Live Tick and Clean Minute Bars

## Why

**1. The live candle stopped ticking.** Ryan: it broke around the expired-contract incident and the move to the shared Render env group, about June 2026. Chat's read on 2026-10-25 (code at `0e943d8`, Render API, DB):

- **The live tick comes from `/api/ironbeam/live-trades-overlay`.** It reads `ironbeam_es_trades` with an exact match, `WHERE symbol = :sym` (`_fetch_trades_utc`, `apps/web/modules/Ironbeam/callbacks.py`). `:sym` is `TRADES_SYMBOL = IRONBEAM_TRADES_SYMBOL or IRONBEAM_SYMBOL or "XCME:ES.H26"`, read once at import, with no fallback when no rows match.
- **The web service is not on the shared group.** The 2026-06-07 env centralization linked only the three Ironbeam workers to the "Volatility Dash Shared" group (`IRONBEAM_SYMBOL`). "Dash - MonoRepo" (the web service) is not among them.
- **The trades table holds the live contract.** Over the last 10 days `ironbeam_es_trades` holds only `XCME:ES.Z26`, which is correct.
- **Hypothesis:** the web service's symbol is a stale contract (an old per-service value, or the `XCME:ES.H26` default). The overlay query then matches no rows, so the candle doesn't tick. Bars still draw, because `ironbeam_es_1m_bars` has no symbol column. The flow panel survives only through its dominant-symbol fallback (CR-BW Step 0).
- **CR-AC is unfinished.** It removes the H26 defaults (branch `feat/CR-038-central-contract-env`, `a7fdd90`) but is still in progress and unmerged. Its plan for the web service was an empty-string fallback, which would break the tick the same way.

**2. Minute bars store partial bars.**
- The live writer is the "Ironbeam Cron" job (`ironbeam_minute_snapshot.py`, every minute). It inserts only bars **newer than `max(datetime)`** (`write_snapshot_to_db`). If a snapshot contains the still-forming minute, that partial bar is stored and never updated.
- The `ironbeam_ingest.py` worker, the one CR-BW's open question named, is suspended on Render and is not the live writer.
- **Measured against `ironbeam_es_flow_1s`, 2026, RTH:**
  - high/low off by > 0.25 pt on 0.4–2.6 % of minutes per month, worst in fast months (Feb 2.6 %, Apr 1.8 %);
  - Oct 2026 0.2 %;
  - Sep 2026 20 %, which is the CR-BF patch window (Z26 bars vs the U26 flow): expected, not the defect.
- The table's primary key is `datetime`, so an upsert is possible.

## Locked decisions (Ryan to confirm before spec freeze)

| # | Decision | Value |
|---|---|---|
| L1 | Scope | (a) The web service resolves the ES contract from data, not a frozen env value. (b) The minute-bar writer stops storing partial bars and re-writes recent bars. (c) A one-off repair of 2026 minute bars from the 1-second flow, logged and reversible. No change to research tables, outcomes or crons other than "Ironbeam Cron". |
| L2 | Contract resolution in the web service | Per request: the symbol with the most rows in the requested window, first in `ironbeam_es_trades`, else in `ironbeam_es_flow_1s`. If there are no rows in the window: `IRONBEAM_SYMBOL` / `IRONBEAM_TRADES_SYMBOL` if set, else return empty with a payload `symbol_note`. **No hardcoded contract literal.** The resolved symbol is echoed in the payload (`symbol`) for the overlay and flow routes. Applies to `/api/ironbeam/live-trades-overlay`, `/api/ironbeam/flow` and every other `TRADES_SYMBOL`/`FLOW_SYMBOL` read in `callbacks.py`. |
| L3 | Render | Link "Dash - MonoRepo" to the "Volatility Dash Shared" group, and remove any per-service `IRONBEAM_SYMBOL` / `IRONBEAM_TRADES_SYMBOL` / `IRONBEAM_FLOW_SYMBOL`. Ryan does this in the dashboard or authorizes chat to. Code reports what it finds; it does not change Render. |
| L4 | Minute-bar writer | `ironbeam_minute_snapshot.py`: (1) drop any bar whose end time is later than now minus 2 s (still forming); (2) **upsert** the last 3 completed bars on every run (`INSERT … ON CONFLICT (datetime) DO UPDATE`), so a bar stored early is corrected on the next runs; (3) older bars: insert-only, as today. Replace the deprecated `utcfromtimestamp` (CR-AC open question). |
| L5 | One-off repair | For 2026-01-01 → now, RTH and ETH: where `ironbeam_es_1m_bars` and the minute built from `ironbeam_es_flow_1s` (same contract, Step 0 Q3) differ by > 1 tick in high, low or close, or by > 10 % in volume, UPDATE the bar to the flow-built values. **Every changed row's old values go first to a new table `ironbeam_es_1m_bars_repair_log`** (datetime, old OHLCV, new OHLCV, run id). The CR-BF patch window (2026-09-13 22:00 → 09-18 20:59 UTC, source `databento`, Z26 by design) is **excluded**. Backfill protocol (CLAUDE.md): `get_backfill_db_conn`, `assert_role_or_die`, `backfill_run`, `update_run_smoke`. Dry-run first, with counts; then a go from Ryan. |
| L6 | Downstream | No recompute of `es_minutes`, outcomes or features in this CR. Step 0 lists which consumers read `ironbeam_es_1m_bars` and whether `es_minutes` is derived from it; recompute is a follow-up decision. |
| L7 | Verification | Live tick: on a live session, the overlay returns `symbol = XCME:ES.Z26` and trades within the last minute, and the chart's last candle updates between minute boundaries (screen recording or two screenshots ≥ 10 s apart). Bars: after 1 trading day with the new writer, the share of RTH minutes off by > 1 tick vs flow is ≤ 0.1 %. |

## Step 0 — diagnosis gate (answer all before implementation)

1. **Confirm the tick failure.**
   - Call the overlay route for 2026-10-23 locally against the DB (read-only) with the env the web service would have (report what `TRADES_SYMBOL` resolves to under each plausible env).
   - Ryan checks the "Dash - MonoRepo" env vars in the Render dashboard and reports any `IRONBEAM_*` values.
   - Confirm the frontend's polling path for the live candle (`App.jsx` around line 1456) and that an empty overlay is the only reason it doesn't tick.
2. **CR-AC state:** what's on `feat/CR-038-central-contract-env` that overlaps this CR. Recommend merge, supersede or cherry-pick; don't merge it here.
3. **Repair feasibility** (no writes):
   - per month: minutes differing, and by how much;
   - whether `ironbeam_es_flow_1s` holds one contract per minute (use the dominant symbol per session; skip minutes with two symbols);
   - whether the flow volume definition matches the bar volume definition (compare on calm minutes);
   - the dry-run change count.
4. **Writer behaviour:** from the "Ironbeam Cron" logs (Render), how often a snapshot contains the forming minute. Confirm the partial-bar mechanism or refute it.
5. **Consumers:** every reader of `ironbeam_es_1m_bars` (chart, `es_minutes`, outcomes, scripts), and whether `es_minutes` is a copy or a view.

## Implementation order (one commit each)

1. Contract resolution in `callbacks.py` (L2), with unit tests: a window with Z26 rows resolves Z26 whatever the env says; an empty window returns empty with a note.
2. Minute-bar writer (L4), with unit tests on the forming-bar filter and the upsert.
3. Repair script (L5), dry-run mode and log table DDL.
4. Smoke: run the repair dry-run; Ryan approves; run the repair; report counts and the log table row count.

**Deploys** (by chat after merge): "Dash - MonoRepo" and "Ironbeam Cron". L3's Render group change happens before the web deploy.

## Not in scope

- Recomputing `es_minutes`, outcomes or any research table.
- CR-AC's broader `render.yaml` reconciliation.
- Chart visuals (CR-BV).

## Kickoff prompt for Claude Code

```
Start CR-BZ in its own git worktree (another CR's worktree may be active; do not switch the main checkout's branch). First action: git fetch && git worktree add <scratch>/crbz-wt -b feat/CR-BZ-live-tick-clean-minute-bars origin/main (confirm CR-BZ is unused; rename if not).
Read the vault CR "/Users/ryan/My Drive/Obsidian Vault/Anonymous/Dash/CRs/CR-BZ — Price Chart Live Tick and Clean Minute Bars.md", components/field-conventions.md, the CR-AC CR note and the 2026-06-07 env centralization session note.
Commit 1: spec freeze to specs/CR-BZ-live-tick-clean-minute-bars.md, verbatim.
Commit 2: Step 0 findings — answer all five questions. Read-only DB; no writes, no Render changes. For Q1 tell me exactly what to look for in the Render dashboard. Stop and report.
```
