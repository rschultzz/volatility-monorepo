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

---

## Step 0 findings (2026-10-25, read-only)

Branch `feat/CR-BZ-live-tick-clean-minute-bars` off `origin/main` `3f595ec`. Every DB session was `SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY`. Render: reads only (service list, deploy list, logs). No Render changes. The L1–L7 table above is still marked "Ryan to confirm"; it was frozen as written. **Four findings below contradict L4, L5 and L6 as written (see "Amendments needed").**

### Q1 — Tick failure

**Code path (the only source of the forming candle).**
- `react_price_preview/src/App.jsx:1456` polls `/api/ironbeam/live-trades-overlay` every 1 s (`LIVE_OVERLAY_POLL_MS`, line 18), but only while `isLiveTradeDate` is true, i.e. the selected session equals the browser's current session (15:00 PT roll, weekday roll-forward).
- The response's `bars` go to `liveTradeBars` → `PriceChart.jsx` `liveTradeCandles` → a separate series (`liveTradeSeriesRef`, ~line 2246).
- The full-bars poll (`FULL_BARS_LIVE_POLL_MS` = 30 s) only redraws stored minutes.
- **Nothing else updates the forming candle.** An empty `bars` array is therefore exactly "no tick".

**The server (`_build_react_live_trades_overlay_payload`, `callbacks.py:1227`) returns empty bars when any one of these holds:**
- (a) it is not the live session day;
- (b) the stored bars already cover the current bucket (`start_bucket > current_bucket`);
- (c) `ironbeam_es_trades` has no rows for `TRADES_SYMBOL` in the window;
- (d) the trades table has no rows at all in the window (feed down).

**Replay against the DB.** Script `scripts/cr_bz_step0/q1_overlay.py`: real module, real DB, `now` pinned to a live minute on 2026-10-23, stored bars truncated to before `now`.

| Web env | `TRADES_SYMBOL` | `bars` at 11:15:25 PT |
|---|---|---|
| none set | `XCME:ES.H26` (code default) | 0 |
| `IRONBEAM_SYMBOL=XCME:ES.H26` | H26 | 0 |
| `IRONBEAM_SYMBOL=XCME:ES.M26` | M26 | 0 |
| `IRONBEAM_SYMBOL=XCME:ES.U26` | U26 | 0 |
| `IRONBEAM_SYMBOL=XCME:ES.Z26` | Z26 | **1** (O 7669.25 H 7671.0 L 7668.75 C 7670.75) |
| `IRONBEAM_SYMBOL=` (empty; CR-AC's fallback) | `""` | 0 |
| `IRONBEAM_SYMBOL=Z26` + `IRONBEAM_TRADES_SYMBOL=U26` | U26 | 0 (TRADES wins) |
| `IRONBEAM_SYMBOL=U26` + `IRONBEAM_TRADES_SYMBOL=Z26` | Z26 | **1** |

- Only Z26 ticks.
- `IRONBEAM_TRADES_SYMBOL` takes precedence over `IRONBEAM_SYMBOL`, so a stale per-service `IRONBEAM_TRADES_SYMBOL` breaks the tick even after the group is linked.
- `FLOW_SYMBOL` defaults to `TRADES_SYMBOL`.
- The local `.env` holds `XCME:ES.M26` for all three variables, so local runs fail the same way.

**Other findings for Q1.**
- **(d) is a second, independent cause.** On 10-23, `ironbeam_es_trades` (and so `ironbeam_es_flow_1s`, which is built from it) has **no rows 14:52:48 → 16:56:14 UTC (07:52–09:56 PT)**, while `ironbeam_es_1m_bars` has all 123 minutes.
  - The trades-ws worker logged nothing for 2 h.
  - At 16:56:22 it logged `Connection reset by peer`, then reconnected. The socket hung silently and the worker has no staleness watchdog.
  - Same pattern on 2026-09-22: 130 RTH minutes with bars and no flow.
  - During such a stall the tick fails whatever the symbol. Not in L1; see "Out of scope findings".
- **(b) does not occur with the current writer.** Snapshots never include the forming minute (Q4), so the overlay always has the current bucket to fill.
- **Raw trades are kept 2 days.** `prune-ironbeam-trades` runs at 01:00 UTC with `IRONBEAM_TRADES_RETENTION_DAYS` default 2. The table currently holds only 10-23 01:02 → 20:59 UTC.
- **Flow route mislabels the symbol.** `_fetch_flow_utc` falls back to the dominant symbol, but `_build_react_flow_payload` still echoes `FLOW_SYMBOL` (`callbacks.py:1557/1612`).
- **Dead prior art.** `apps/web/modules/gamma/callbacks_replacement_autoresolve_symbols.py` (`1c818f4`, 2026-03-30) already has a per-window dominant-symbol resolver, `_resolve_symbol_for_window`. Nothing imports it; `app.py:42` registers `modules.Ironbeam.callbacks`. It is usable prior art for L2.

**What Ryan checks in the Render dashboard:**
1. Open **"Dash - MonoRepo"** (web service `srv-d4370d6uk2gs738ps5t0`) → **Environment**.
2. Under **Environment Variables**, report the exact value, or "absent", of:
   - `IRONBEAM_SYMBOL`
   - `IRONBEAM_TRADES_SYMBOL`
   - `IRONBEAM_FLOW_SYMBOL`
   - plus, if present: `IRONBEAM_ES_SYMBOL`, `IRONBEAM_TRADES_TABLE`, `IRONBEAM_FLOW_TABLE`, `IRONBEAM_BARS_TABLE`, `IRONBEAM_ROLL_SESSION_AFTER_15PT`, `IRONBEAM_LIVE_TRADES_MAX_BARS`, `IRONBEAM_REACT_ALLOWED_ORIGINS`.
3. Under **Linked Environment Groups** (same page, below the variables): is **"Volatility Dash Shared"** (`evg-d8ipl5tckfvc73c39g2g`) linked?
4. Open the group (Dashboard → Env Groups) and confirm its `IRONBEAM_SYMBOL` is `XCME:ES.Z26`.
5. How to read the result:
   - On Render a service-level variable overrides a group variable of the same name.
   - In code, `IRONBEAM_TRADES_SYMBOL` overrides `IRONBEAM_SYMBOL`.
   - So the tick works today only if the effective `IRONBEAM_TRADES_SYMBOL`, or else `IRONBEAM_SYMBOL`, is `XCME:ES.Z26`.
   - Absent, H26, M26 or U26 all give 0 bars (table above).
6. Variables are read at import. The live deploy is `dep-dbehju8m7kps73dtuah0`, finished 2026-10-24 20:52 UTC (autoDeploy is off for this service), so the values shown are the ones running.
7. Change nothing for now (L3 is a later step).

The hypothesis is confirmed in code and against the DB. Whether the web service actually holds a stale value is up to Ryan's dashboard reading.

### Q2 — CR-AC state (`feat/CR-038-central-contract-env`, head `a7fdd90`)

- **State.** 5 commits on merge-base `642ca21` (2026-06-07). Not in `origin/main`, no PR. `origin/main` is 299 commits ahead.

| Commit | Content | Overlap with CR-BZ | Recommendation |
|---|---|---|---|
| `6a8d3e0` step-1 | H26 defaults removed in 4 files: flow aggregator and trades-ws (loud `SystemExit`); `Ironbeam/callbacks.py` (`or ""`); the dead gamma replacement file | `callbacks.py` line 78: the same line L2 rewrites. The empty-string fallback gives 0 bars (Q1 table) | **Supersede** the `callbacks.py` hunk with L2. The worker hunks are good but outside L1; leave them on CR-AC |
| `81b3422` step-2 | `ORATS_TOKEN` → `ORATS_API_KEY` | none | leave on CR-AC |
| `443205b`, `a7fdd90` step-3/3a | `render.yaml` group + service entries (M26, wrong service names, grouped `DATABASE_URL`) | none in code; L3 is dashboard-only | **do not merge**: stale (M26), conflicts with 3 later `render.yaml` commits on main (`7016dd1`, `727cc49`, `b548eaf`), and is the duplicate-service hazard the 2026-06-07 session named |
| `b44561b` spec | CR-038 spec | none | — |

- **Recommendation: supersede for the web service; don't merge CR-AC here.**
  - After CR-BZ, re-scope CR-AC to the worker loud-failure hunks plus the ORATS key rename.
  - Cherry-pick those onto a fresh branch in its own CR, and drop its `render.yaml` and `callbacks.py` changes.
- **Snapshot writer.** `ironbeam_minute_snapshot.py` is untouched on CR-AC. Its `utcfromtimestamp` is now at line 279; L4 replaces it.

### Q3 — Repair feasibility (2026-01-01 → 2026-10-23, CR-BF window excluded)

**Method.** Flow-built minute = first open, max high, min low, last close and summed volume of `ironbeam_es_flow_1s`, keyed on `date_trunc('minute', ts_utc)` = `ironbeam_es_1m_bars.datetime`. Scripts: `scripts/cr_bz_step0/q3.sql`, `scripts/cr_bz_step0/q3_shape.py`.

- **One contract per minute:** `count(DISTINCT symbol) > 1` on **0** minutes in every month. The dominant-symbol rule is never needed inside a minute. The flow symbols are M26 to 06-12, U26 06-14 → 09-18, Z26 from 09-20.

**Volume definitions do not match.**
- Median bar/flow volume ratio is **1.019–1.071** in every month and session.
- On price-identical October minutes (bar H/L = flow H/L, n 22,250), bar volume exceeds flow volume on **18,403**, is below it on **13**, and equals it on the rest.
- The bar counts more than the trades feed delivers. Flow volume is not a substitute for bar volume.

**Price differences > 1 tick (max over H, L, C):**

| Month | RTH n | RTH diff | RTH % | RTH bar vol < flow | ETH n | ETH diff | ETH % | ETH bar vol < flow | RTH med / max pts |
|---|---|---|---|---|---|---|---|---|---|
| 2026-01 | 7,985 | 39 | 0.49 | 26 | 20,693 | 53 | 0.26 | 1 | 2.0 / 19.0 |
| 2026-02 | 7,471 | 213 | 2.85 | 169 | 19,466 | 114 | 0.59 | 10 | 1.75 / 24.75 |
| 2026-03 | 7,380 | 132 | 1.79 | 97 | 18,934 | 274 | 1.45 | 25 | 2.0 / 17.5 |
| 2026-04 | 8,115 | 198 | 2.44 | 172 | 21,684 | 205 | 0.95 | 75 | 1.75 / 33.25 |
| 2026-05 | 7,892 | 91 | 1.15 | 32 | 20,658 | 103 | 0.50 | 2 | 1.5 / 22.25 |
| 2026-06 | 8,331 | 142 | 1.70 | 61 | 21,341 | 215 | 1.01 | 8 | 2.25 / 25.0 |
| 2026-07 | 7,741 | 102 | 1.32 | 30 | 19,204 | 127 | 0.66 | 11 | 0.75 / 15.25 |
| 2026-08 | 8,184 | 40 | 0.49 | 22 | 20,907 | 72 | 0.34 | 0 | 1.5 / 11.25 |
| 2026-09 | 6,320 | 42 | 0.66 | 27 | 16,623 | 37 | 0.22 | 1 | 1.625 / 14.5 |
| 2026-10 | 6,506 | 16 | 0.25 | 13 | 16,708 | 48 | 0.29 | 1 | 2.125 / 16.25 |
| **Total** | 75,925 | **1,015** | 1.34 | **649** | 196,218 | **1,248** | 0.64 | **134** | |

**Two opposite defects.**
- **Bar volume < flow volume (783 minutes; RTH 649, ETH 134): the bar is a fragment and the flow is right.**
  Per-second classification of these minutes (does the bar equal the flow over seconds `[0, s)`, `[s, 60)`, or neither):

  | Shape | RTH | ETH | Median cut (s) | Median bar/flow volume | Where |
  |---|---|---|---|---|---|
  | HEAD: the first `s` seconds only (early partial kept) | 230 | 104 | 37.5 / 43 | 0.59 / 0.70 | **episodic:** 2026-04-01 120, 04-02 69, 02-26 35, 03-31 33 (257 / 334); by month Feb 55, Mar 71, Apr 190, then 1, 4, 12, 0, 0, 0; last 2026-07-15 |
  | TAIL: the last `60 − s` seconds only (start of minute missing) | 311 | 15 | 26 | 0.63 / 0.48 | **all year, every month** (13–101 / month); concentrated at **12:59 PT (97)** and **06:30 PT (33)** |
  | NONE | 108 | 15 | — | 0.64 / 0.71 | — |

  Example TAIL bar, 2026-10-08 16:17 UTC: bar O/H/L/C 7847 / 7850 / 7842 / 7842, volume 8,469. It equals the flow from second :33 onward. The flow's first 33 s (7826 → 7848, about 15k contracts) are missing.
- **Bar volume > flow volume (1,478 minutes; ETH 1,112, RTH 366): the flow is the series missing trades.**
  - These are mostly thin ETH minutes, median 0.5 pt; the bar's range is wider than the flow's.
  - Example: 2026-10-12 15:03 UTC, flow 1 second / 62 contracts vs bar 909.
  - **Overwriting these with the flow would corrupt correct bars.**

**Change counts:**

| Rule | Minutes |
|---|---|
| **L5 as written** (> 1 tick in H/L/C **or** > 10 % volume) | **50,480** (RTH 10,940, ETH 39,540); 18.5 % of 272,143 joined minutes, almost all from the volume clause (the ±2–7 % definitional offset) |
| L5 price clause only | 2,263 (RTH 1,015, ETH 1,248) |
| Price clause **and** bar volume < flow volume (fragment bars only) | **783** (RTH 649, ETH 134) |

- Bars with no flow at all (feed gaps, e.g. 10-23 07:53–09:56 PT, 09-22 130 min) are outside the join and can't be repaired from flow.

### Q4 — Writer behaviour (Ironbeam Cron `crn-d4ni0sfdiees73fk5900`)

**The writer.**
- `ironbeam_minute_snapshot.py` subscribes `loadSize` 2000, collects for about 10 s, `drop_duplicates(subset=["datetime"])` (keeps the **first** copy), then inserts rows with `datetime > max(datetime)` via `to_sql(append)`.
- Stored timestamps are bar start = Ironbeam `t` − 1 min.
- The 97–99 % same-minute agreement with the flow (CR-BW, and the table above) confirms the label is not shifted by a minute.

**Logs.**
- Render keeps about 7 days of logs for this cron: 2026-10-08 returned nothing, 2026-10-21 onward is present. The February–April HEAD episodes cannot be traced in logs.
- **100 consecutive runs, 2026-10-23 13:29–15:09 UTC (the open):**
  - every run's snapshot `max datetime` = the minute that had just closed;
  - **0 / 100 contained the forming minute**;
  - runs flush 10–46 s after the minute boundary;
  - overlapping instances happen (e.g. 14:02:21 and 14:02:47 both saw max 14:01), and the insert-above-max rule makes the second a no-op.
- 10-21 16:39 TAIL defect: written by the run that flushed at 16:40:46, 46 s after the bar closed, as a completed bar. The bar was already missing the first part of the minute.

**Verdict.**
- **Refuted** for October: the current writer does not store forming bars.
- **Plausible but unverifiable** for the Feb–Apr HEAD episodes (334 minutes, 77 % on four days). Logs are gone and the cause is unknown: Render cron timing, or the vendor including a forming bar on those days.
- **TAIL fragments are not explained by the writer.** The vendor's time bar, as delivered at least 10–46 s after the close, already lacks the start of the minute, most often 12:59 and 06:30 PT.
- L4's "re-write the last 3 bars" helps only if Ironbeam revises a bar in later snapshots. That is **unmeasured**: `IRONBEAM_DEBUG_WS_RAW` is off, so no bar values are logged.

### Q5 — Consumers

- **`es_minutes` is a BASE TABLE, filled by trigger `trg_es_minutes_from_ironbeam`.**
  - Definition: `AFTER INSERT ON ironbeam_es_1m_bars FOR EACH ROW EXECUTE FUNCTION es_minutes_from_ironbeam()`.
  - The function computes `ts_utc`, `ts_pt`, `trade_date`, `is_rth` (06:30–13:00 PT) and `bar_index`, then `INSERT … ON CONFLICT (ts_utc) DO NOTHING`. It does not set `source`.
  - **There is no UPDATE trigger.** An `ON CONFLICT DO UPDATE` (L4) or an `UPDATE` (L5) changes `ironbeam_es_1m_bars` only; `es_minutes` keeps the old bar. After CR-BZ the two tables would disagree on every corrected minute.
- **Views on `es_minutes`:** `es_minutes_with_features`, `es_minutes_with_features_bt`, `es_minutes_with_features_v2`, `es_gamma_smile_minutes`.
  - Readers: `apps/web/modules/Backtests/callbacks.py`, `BacktestsV2/service.py`, `packages/backtests/gex_fade.py`, `scripts/run_gex_fade_backtest.py`.
  - Direct `es_minutes` readers: `scripts/backfill_databento_es.py`, `cr_bh_diff.py`, `cr_bo_step0.py`, `cr_bq_common.py`, `cr_bq_extract.py`, `cr_bq_step0.py`.
- **Direct readers of `ironbeam_es_1m_bars`:**
  - **Web:** `Ironbeam/callbacks.py` (chart bars, live overlay base, CR-BT P_target `_fetch_es_snapshot_close`), `Ironbeam/routes.py` (landscape RTH open), `Bars/service.py`, `DayBrowser/service.py`, `SetupV2/routes.py`, `TodaySetup/routes.py`, `Analogues/routes.py`, plus the dead `gamma/callbacks.py` and `gamma/callbacks_replacement_autoresolve_symbols.py`.
  - **Shared:** `packages/shared/sessions.py`, `vol_features.py`, `level_placement.py`.
  - **Production crons (Render):** `sweep_pending_outcomes` (`scripts/cr_aa_sweep_pending_outcomes.py`, 11:05 UTC) and `backfill_outcomes` (`scripts/cr_b_backfill_outcomes.py`, 13:40 UTC). Both build `bt_daily_outcomes` from RTH OHLC through `packages/shared/outcomes_runner.py`.
  - **Writers:** `ironbeam_minute_snapshot.py` (live); `ironbeam_ingest.py` (worker `srv-d4jftq75r7bs73fdvm3g`, suspended since 2025-12-15, branch `price-chart-controlled-by-trade-date`).
  - **CR scripts:** about 30 one-off or Step 0 scripts (`cr_ah_*`, `cr_ai_*`, `cr_bh_*`, `cr_bi_*`, `cr_bo_step0`, `cr_br_common`, `cr_bs_step0`, `cr_bt_step0`, `cr_bu_*`, `cr_g_*`, `cr_x`, `cr_z`, exports and backfills).

### Amendments needed before implementation (Ryan's call; spec not amended yet)

1. **L4's premise is refuted for the current writer.** The forming-bar filter and upsert are harmless, but they won't fix TAIL fragments unless Ironbeam revises bars.
   - Proposal: have L4 also **log** every snapshot bar that differs from the stored row (datetime, old, new). Then measure after 1 trading day before claiming the effect.
   - L7's "≤ 0.1 % after 1 day" bar is not reachable by the writer alone if TAIL fragments persist (Oct RTH: 0.25 %).
2. **The trigger is insert-only.** Choose one:
   - (a) add an `AFTER UPDATE` trigger mirroring OHLCV into `es_minutes` (touches `es_minutes`, against L6);
   - (b) accept the divergence and record it in field-conventions;
   - (c) have the repair update both tables.
3. **L5's volume clause must go.** It hits 50,480 minutes because of a definitional offset. Restrict L5 to **price diff > 1 tick and bar volume < flow volume** (783 minutes, the fragment bars) and require a continuous flow minute. Also decide the repaired bar's volume: flow volume is a lower bound and runs 2–7 % below bar volume on clean minutes.
4. **"The flow is correct" holds only for fragment bars.** On 1,478 of 2,263 price-diff minutes the flow is missing trades. Update the field-conventions row for `ironbeam_es_flow_1s`.

### Out of scope findings (proposed follow-ups)

- **trades-ws silent stall.** No watchdog. 2026-10-23: 2 h with no trades (tick, flow panel and flow table all empty); 2026-09-22: 130 min. Needs a stale-feed reconnect (e.g. no trade for > N s during RTH → reconnect). Not in L1.
- **Flow payload `symbol` echo.** Covered by L2's "echo the resolved symbol".
- **Dead module.** `gamma/callbacks_replacement_autoresolve_symbols.py` registers the same routes, under the same guard flags, if ever imported. Delete it or keep it as reference.

---

## Amendment A1 (Ryan, 2026-10-25, after Step 0)

Verbatim:

- **A1.1** L2 stands, with data-derived symbol taking precedence over every env var (IRONBEAM_TRADES_SYMBOL / IRONBEAM_SYMBOL / IRONBEAM_FLOW_SYMBOL are fallbacks only). This supersedes CR-AC's callbacks.py change.
- **A1.2** L4 replaced: keep the forming-bar filter (harmless); drop the 3-bar upsert. Instead, log to a new table ironbeam_es_1m_bars_revision_log every case where a later snapshot's bar differs from the stored row (datetime, stored OHLCV, snapshot OHLCV, seen_at), insert-only. No change to stored bars. Purpose: measure whether Ironbeam revises bars. L7's ≤ 0.1 % target is dropped.
- **A1.3** L5 narrowed: repair only the fragment bars (price differs > 1 tick AND bar volume < flow volume), 2026, excluding the CR-BF patch window. OHLC from flow; volume = flow volume (lower bound, noted). Log old values first. Update BOTH ironbeam_es_1m_bars and the matching es_minutes rows in the same transaction, logging both. Dry-run counts first, then stop for my go.
- **A1.4** L6: es_minutes stays insert-trigger-only otherwise; no new trigger.
- **A1.5** Out of scope, filed as new vault open questions: (a) trades-ws worker silent stalls (10-23 07:52–09:56 PT, 09-22 130 min) → watchdog CR; (b) CR-AC re-scope (worker loud-failure + ORATS key rename; its render.yaml is stale — do not merge). Mark CR-AC's vault note status cancelled, superseded by CR-BZ (callbacks) and the new re-scope question.

### Implementation notes under A1 (Claude Code)

- **"Price differs > 1 tick"** = max(|ΔH|, |ΔL|, |ΔC|) > 0.25. This is the Step 0 Q3 definition (783 minutes as of 2026-10-23).
- **Grants.**
  - `dash_backfill_writer` has only INSERT/SELECT on `ironbeam_es_1m_bars` and `es_minutes`, measured on `information_schema.role_table_grants` 2026-10-25.
  - The live repair therefore needs a column-level `GRANT UPDATE (open, high, low, close, volume)` on both tables. It ships in `scripts/cr_bz_ddl.sql` together with the two new tables, and is **not applied** in this CR's implementation commits.
  - New tables owned by `rschultz` give `dash_backfill_writer` INSERT/SELECT through the existing default ACL.
- **The revision log table must exist before "Ironbeam Cron" is deployed with step 2.** The writer degrades to a printed warning if the table is missing, and still inserts bars.
- **Revision log key.** The PK is `(datetime, snapshot OHLCV)`, with `ON CONFLICT DO NOTHING`. Each distinct revision is logged once; repeated identical snapshots on later runs are not logged again. `seen_at` = first sighting.
