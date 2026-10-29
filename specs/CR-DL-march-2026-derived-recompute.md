# CR-DL — Recompute derived tables for the March 2026 patched window

Branch `feat/CR-DL-march-2026-derived-recompute` (worktree `../volatility-monorepo-cr-dl`) off `origin/main` `ce13906` (CR-DJ merge).
Kickoff: chat 2026-10-29 (Ryan). Data CR (`data_safety_class: write_backfill`).
Source: vault open-question `march-2026-roll-patch-derived-recompute`.

## Problem (from the open-question)

CR-DJ (run `2adcc833-5f7f-4406-b645-0ce1b7d61bbc`) replaced the bars of 2026-03-15 22:00 → 03-20 20:59 UTC (dying ES H26, ~50 pts below the front month) with Databento ES M26 in `ironbeam_es_1m_bars` and `es_minutes`. Rows computed from the old H26 bars before the patch still hold H26-derived values. CR-DJ Step 0d inventory:

| Candidate | Status |
| --- | --- |
| `bt_daily_outcomes` | **84 active rows** (21 trade dates 2025-12-19 → 2026-03-20 × 4 feature versions) whose horizon includes a window session — recompute in place, CR-BG pattern (`scripts/cr_bg_recompute_missed_roll.py`) |
| `bt_daily_features.vol_risk_premium` | 23 non-null rows 2026-03-16 → 04-17 (ES daily closes, written by `cr_d_backfill_vol_features.py`) — recompute the affected ones |
| Views (`es_minutes_with_features*`), live chart / REVIEW readers | self-heal — no action |
| `bt_signals`, `es_minute_features` | 0 rows — no action |
| `ironbeam_es_flow_1s` H26 rows | symbol-tagged; stay (CR-BF / CR-BG precedent) |

## Kickoff (frozen)

Branch first (after merging CR-DJ). Spec from the vault open-question. This is a data CR: standing-rulings conformance table required.

**STEP 0 (commit before any write; STOP and report):**
- a. Exactly which `bt_daily_outcomes` rows (84) and `vol_risk_premium` rows are derived from 2026-03-16 → 03-20 bars, how each was computed (script / CR), and whether the same code can recompute them unchanged.
- b. List the past research CRs whose inputs included those sessions (from specs / logs / run rows). For each: how many sessions in its sample, and whether 03-16 → 03-19 could matter to its verdict (e.g. a sample under 50 sessions, or a result near its bar). Report only; don't re-run any research.
- c. B4: these sessions are train (≤ 2026-06-05), so reading outcomes there is allowed. Confirm nothing in the holdout is touched.
- d. Before/after for each recomputed row (dry run), and the write plan under the unattended backfill protocol. If any table needs grants the writer lacks, say so; don't apply grants without Ryan's approval.

Don't recompute until Ryan approves Step 0.

## Standing rulings conformance

| Ruling | Status | Note |
|---|---|---|
| A1–A22 (the setup) | n/a | Data repair of stored derived rows; no setup, level, entry, exit or event filter is defined or tested. |
| B1, B2, B3 | n/a | No verdict, parameter sweep or setup optimisation. Step 0b lists past CRs and their sample sizes; no research is re-run. |
| B4 (data split) | honoured (to be confirmed in Step 0c) | The window sessions 2026-03-16 → 03-20 are train. Outcome rows whose horizon runs past 2026-06-05 would read holdout bars on recompute — Step 0c lists them and whether any exist. |
| B5 (Databento is the reference) | honoured | The recompute reads the CR-DJ-patched bars (Databento ES M26 in the window). |
| B6, B7, B8 | n/a | No exits, placebos or labels. |
| C1, C2 | n/a | No UI or display change. |
| Revoked R1–R7 | none used | R6 (`discounted_level`) is the stored `bt_daily_outcomes` target frame; the recompute reproduces the existing computation unchanged and does not introduce R6 into any new test. |

## Step 0 findings (2026-10-29, read-only)

Method: `scripts/cr_dl_step0_dryrun.py` (read-only transaction as `dash_backfill_writer`, no run row; output `scripts/logs/cr_dl/step0_dryrun.{json,log}`, git-ignored). Every candidate row is computed by **the code that wrote it**, three times:

- **pre:** the bars as they were when the row was computed. This is the current table with every logged repair undone (the earliest `old_*` per minute in `ironbeam_es_1m_bars_repair_log`: CR-BZ on 2026-10-25 and CR-DJ on 2026-10-29) and CR-DJ's 124 inserted minutes removed.
- **mid:** CR-DJ undone only.
- **post:** the bars now.

How to read the three states:
- **pre = stored** is the reproduction check: it shows the code is the writer.
- **mid → post** is the CR-DJ effect.
- **pre → mid** is the CR-BZ effect, which this CR's kickoff doesn't cover; it is reported in F2.

Candidate rows: every active row with trade_date from 2025-11-03 to 2026-03-20, in each ES-based version (89 dates per version), plus `vol_risk_premium` from 2026-02-01 to 05-22.

### a. Which rows, how computed, and whether the same code recomputes them unchanged

**Writers** (`bt_daily_outcomes.backfill_run_id` → `bt_backfill_runs.cr_id`, plus git history):

| feature_version | Wrote the outcome / t0 columns | Wrote `session_*_t{1,5,15}` / `position_t*_post_touch` | Frame | Reproduces stored (pre = stored)? |
|---|---|---|---|---|
| `v0.6.1-nyse-sessions` (**canonical**) | CR-BI `compute_row` (run `a44f185c`, 2026-09-21) | the same function | `packages/shared/sessions` (PT 06:30–13:00 window, NYSE-calendar sessions) | **yes, 89 / 89 rows, every column** |
| `v0.6.0-openiv` | `outcomes_runner` via cr_b / CR-AA, CR-AQ containment | CR-G / CR-I before CR-BI (runs `d605b950`, `4e55a538`) | the old daily frame (same RTH SQL, bar-present days); CR-G / CR-I fixed 13:30–20:00 UTC (verbatim from `5531453^`) | **yes, 89 / 89** |
| `v0.5.0-rebuilt` | cr_b / CR-AA (runs `8d0a8663`, `5cef55c5`); `session_open_t0` by CR-G Step 2.5a (first bar ≥ 13:30 UTC); containment never filled | CR-G / CR-I (old frames) | as `v0.6.0-openiv` | **84 / 89**: 5 rows aren't reproduced (2026-03-02, 03-03, 03-05, 03-11, 03-12). These are the CR-AA sweep rows of 2026-06-04: CR-G / CR-I never filled their t1/t5/t15 / position columns (stored NULL), and 3 values differ by 5.5–17.75 pts, an unlogged bar state at 06-04. Those columns are left as stored. |
| `v0.6.0-openiv-spxcash` | CR-BH (run `ba5d92f4`) | CR-BH | SPX cash (`packages/shared/spx_cash`, `orats_monies_minute`); **reads no ES bar** | not affected; excluded |
| `bt_daily_features.vol_risk_premium` (`v0.5.0-rebuilt` only; NULL on every other version) | CR-D `_compute_vol_features` (2026-05-24/25; `fetch_es_closes_before` = last bar per UTC `DATE(datetime)`, 21 closes) | — | — | **yes, 76 / 76** |

**Rows the CR-DJ patch actually changes (mid → post): 15 trade dates in each of the 3 ES versions, 45 outcome rows, plus 18 `vol_risk_premium` rows.** That is not the 84 in CR-DJ's inventory:
- The 21 `-spxcash` rows read no ES bar.
- The other 60-session rows whose horizon spans the window don't change in any column: their touch, excursion and final close fall outside it.
- 2026-03-20 doesn't change: Ironbeam was already on M26 from 13:29 UTC, before the RTH open.

The 15 dates:
- **Own session (t0 columns):** 2026-03-16 → 03-19, about +48 to +51 pts.
- **T+1 / T+5 / T+15 session columns:** 2026-02-23 → 02-26, 03-09 → 03-13 and 03-16 → 03-18 (the Nth session lands in 03-16 → 03-19).
- **Final close of a 60-session horizon:** 2025-12-19 (horizon end 03-16) and 12-22 (horizon end 03-17); `final_close_distance_from_target` +50.5. On `v0.6.1` `actual_realized_em_pct` also changes.

**No `outcome_status`, `reached_touch`, `reached_close`, `days_to_reach`, `horizon_*`, `max_excursion_in_direction` or `position_t*_post_touch` value changes because of CR-DJ in any version.** All 15 window-session rows are `na_regime`, except that the t15 / t5 rows include `computed` ones, whose labels don't move.

**The same code recomputes them unchanged:**
- `v0.6.1`: `scripts/cr_bi_backfill_nyse_sessions.compute_row`, imported as is.
- `v0.6.0-openiv` / `v0.5.0-rebuilt`: the pre-CR-BI frames. CR-BI (`5531453`) replaced them in the CR-G / CR-I / cr_b scripts, so they can't be called from `main`. The dry run carries verbatim copies (`_fetch_rth_minute_bars`, `_compute_session_ohlc`, the CR-I UTC-date close) and reproduces 89 / 89 with them.
- `vol_risk_premium`: `scripts/cr_d_backfill_vol_features._compute_vol_features`, imported as is. Its own write path is `COALESCE` (it never overwrites), so the recompute must UPDATE the column directly.

### b. Past research CRs whose inputs included 2026-03-16 → 03-19 (report only; nothing re-run)

Swept by a read-only subagent across `specs/`, `scripts/cr_*` and the main checkout's `scripts/logs/`. Spot-checked: `scripts/cr_cl_lib.py:262-285` (P_target from `ironbeam_es_1m_bars`, symbol guard on `prev_trading_day`), `scripts/logs/cr_ct/step0.json` `level_missing` ("06:29 Ironbeam 6697.25 vs Databento 6747.0"), and the CR-CB L14a roll-week exclusion 03-09 → 03-20.

| CR | Where the window touched its input | Sample / headline | Could 03-16 → 03-19 change the verdict? |
|---|---|---|---|
| CR-CT (approach speed) | Only the overnight level F2 used Ironbeam bars; its guard (Ironbeam vs Databento at 06:29 PT within 2 pts) **dropped** F2 on 03-16 → 03-20, and F1 on 03-16 (symbol switch). No H26 price entered. | 101 sessions; FAST − SLOW +2.2 pp [−15.2, +19.5], INCONCLUSIVE | **No.** About 4–5 F2 touches lost; the CI is ±17 pp. |
| CR-CW (momentum path) | reuses CR-CT's touches, so the same dropout | 101 sessions; H1 −1.4 pp [−13.7, +11.4], H2 −0.6 [−13.5, +12.4], INCONCLUSIVE | **No.** |
| CR-CV (scalping scale) | S4 reuses CR-CT's touches; S1–S3 use Databento only | 0 PASS / 9 REFUTED / 7 INCONCLUSIVE; S4 CI upper −0.49 / −0.53 | **No.** S4 is far from its bar; the near-bar cell (S3, −0.04) is Databento only. |
| CR-CL (skew → next move) | Primary uses Databento only. The GEX-tag **readout** anchors P_target on the Ironbeam 15:45 ET close, so levels for 03-17 → 03-20 sat about 50 pts low (03-16 dropped by the symbol guard). | G1-down 49 legs / 36 sessions, +17.4 pp [−1.2, +37.3], INCONCLUSIVE | **No for the verdict.** Maybe for the descriptive GEX-tag readout: 4 of 99 sessions misplaced, sub-cells ≤ 13 legs. |
| CR-AQ (containment columns) / CR-AS (0DTE condor) | `bt_daily_outcomes` `session_*_t0` (canonical before CR-BI), 4 of 795 / 732 dates | CR-AS H1 n 650, +0.000 IM [−0.009, +0.009]; H2 +0.032 [−0.001, +0.065] | **No.** CR-AS P&L is SPX legs settled on SPX; ES is only the open-basis filter / sanity gate. CR-AQ is descriptive. |
| CR-AH family (AH/AL/AM/AN/AP/AR) | 2 CR-AR debit trades **expired** 03-18 / 03-19 on Ironbeam settlement | CR-AR debit 103 trades | **No.** Both are full losses on H26, M26 or SPX (target about 7019 vs settlement about 6610–6680); CR-BI re-settled them on SPX anyway. CR-AI's exact trade list isn't determinable from the repo; immaterial for the same reason. |

**Unaffected**, with the reason:
- **Roll weeks excluded** (L14a / CR-BO L1: 03-09 → 03-20 dropped): CR-BO, BP, BQ, BR, BS, DA, CY, BT (headline), BU, BW, BX, BY, CB, CE, CF, CM.
- **Databento outright bars, front symbol = M26 from 03-16:** CR-CJ, CK Part 1, CP, CS.
- **No 2026 bars:** CR-CN, CR, CO, CU.
- **SPX only:** CR-BK, BL, BM, BI placebo.
- **No verdict:** CR-CX (forward log; offset-invariant swing check), CR-CZ and CR-DC → DI (display / data).

`vol_risk_premium` (`v0.5.0-rebuilt`) is read by no research spec from CR-BI onward. **Bottom line: no past research verdict is plausibly changed.**

### c. B4

- All candidate rows are trade_date ≤ 2026-03-20; the script asserts every stored `horizon_end_date` ≤ 2026-06-05 (latest is 06-05, the 03-12 60-session row).
- Every bar read is capped at 2026-06-05: the fetch end is `min(…, TRAIN_END)`, and the pre / mid overlay is bounded to `datetime < 2026-06-06`.
- `vol_risk_premium` rows end at 2026-05-22 and read closes on or before their own date.
- **Nothing in the holdout (2026-06-06 → 10-26) or the forward log is read or written.** Not one changed row's horizon reaches 06-06.

### d. Before / after (dry run) and the write plan

Values below are mid → post, the CR-DJ effect. Columns a version never filled (`v0.5.0-rebuilt` 03-11 / 03-12 t1/t5/t15 / positions; `v0.5.0` containment) are excluded and stay as stored.


#### `v0.6.1-nyse-sessions` — CR-DJ changes (bars with CR-DJ undone → now)

| trade_date | status | column: before (bars with CR-DJ undone) → after (bars now) |
|---|---|---|
| 2025-12-19 | computed | final_close_dist -354.77 → -304.27; realized_em_pct 6.4653 → 5.7764 |
| 2025-12-22 | computed | final_close_dist -379.25 → -328.75; realized_em_pct 14.78 → 13.21 |
| 2026-02-23 | computed | open_t15 6697.25 → 6746.75; high_t15 6734.25 → 6784.75; low_t15 6684.50 → 6734.50; close_t15 6703.00 → 6753.50 |
| 2026-02-24 | na_regime | open_t15 6739.75 → 6787.75; high_t15 6757.50 → 6808.50; low_t15 6716.00 → 6764.25; close_t15 6721.50 → 6773.00 |
| 2026-02-25 | computed | open_t15 6695.25 → 6746.50; high_t15 6709.75 → 6759.75; low_t15 6623.00 → 6673.75; close_t15 6626.00 → 6676.75 |
| 2026-02-26 | computed | open_t15 6579.25 → 6629.25; high_t15 6640.75 → 6692.00; low_t15 6561.75 → 6611.75; close_t15 6611.00 → 6661.50 |
| 2026-03-09 | na_regime | open_t5 6697.25 → 6746.75; high_t5 6734.25 → 6784.75; low_t5 6684.50 → 6734.50; close_t5 6703.00 → 6753.50 |
| 2026-03-10 | na_regime | open_t5 6739.75 → 6787.75; high_t5 6757.50 → 6808.50; low_t5 6716.00 → 6764.25; close_t5 6721.50 → 6773.00 |
| 2026-03-11 | computed | open_t5 6695.25 → 6746.50; high_t5 6709.75 → 6759.75; low_t5 6623.00 → 6673.75; close_t5 6626.00 → 6676.75 |
| 2026-03-12 | computed | open_t5 6579.25 → 6629.25; high_t5 6640.75 → 6692.00; low_t5 6561.75 → 6611.75; close_t5 6611.00 → 6661.50 |
| 2026-03-13 | na_regime | open_t1 6697.25 → 6746.75; high_t1 6734.25 → 6784.75; low_t1 6684.50 → 6734.50; close_t1 6703.00 → 6753.50 |
| 2026-03-16 | na_regime | open_t0 6697.25 → 6746.75; high_t0 6734.25 → 6784.75; low_t0 6684.50 → 6734.50; close_t0 6703.00 → 6753.50; close_move_over_im 0.0784 → 0.092; open_t1 6739.75 → 6787.75; high_t1 6757.50 → 6808.50; low_t1 6716.00 → 6764.25; close_t1 6721.50 → 6773.00 |
| 2026-03-17 | na_regime | open_t0 6739.75 → 6787.75; high_t0 6757.50 → 6808.50; low_t0 6716.00 → 6764.25; close_t0 6721.50 → 6773.00; range_over_im 0.5088 → 0.5425; close_move_over_im -0.2237 → -0.1808; open_t1 6695.25 → 6746.50; high_t1 6709.75 → 6759.75; low_t1 6623.00 → 6673.75; close_t1 6626.00 → 6676.75 |
| 2026-03-18 | na_regime | open_t0 6695.25 → 6746.50; high_t0 6709.75 → 6759.75; low_t0 6623.00 → 6673.75; close_t0 6626.00 → 6676.75; open_t1 6579.25 → 6629.25; high_t1 6640.75 → 6692.00; low_t1 6561.75 → 6611.75; close_t1 6611.00 → 6661.50 |
| 2026-03-19 | na_regime | open_t0 6579.25 → 6629.25; high_t0 6640.75 → 6692.00; low_t0 6561.75 → 6611.75; close_t0 6611.00 → 6661.50 |

#### `v0.6.0-openiv` — CR-DJ changes (bars with CR-DJ undone → now)

| trade_date | status | column: before (bars with CR-DJ undone) → after (bars now) |
|---|---|---|
| 2025-12-19 | computed | final_close_dist -262.77 → -212.27 |
| 2025-12-22 | computed | final_close_dist -287.25 → -236.75 |
| 2026-02-23 | computed | open_t15 6697.25 → 6746.75; high_t15 6734.25 → 6784.75; low_t15 6684.50 → 6734.50; close_t15 6703.25 → 6753.75 |
| 2026-02-24 | na_regime | open_t15 6739.75 → 6787.75; high_t15 6757.50 → 6808.50; low_t15 6716.00 → 6764.25; close_t15 6720.50 → 6771.75 |
| 2026-02-25 | computed | open_t15 6695.25 → 6746.50; high_t15 6709.75 → 6759.75; low_t15 6623.00 → 6673.75; close_t15 6626.25 → 6677.25 |
| 2026-02-26 | computed | open_t15 6579.25 → 6629.25; high_t15 6640.75 → 6692.00; low_t15 6561.75 → 6611.75; close_t15 6608.25 → 6658.75 |
| 2026-03-09 | na_regime | open_t5 6697.25 → 6746.75; high_t5 6734.25 → 6784.75; low_t5 6684.50 → 6734.50; close_t5 6703.25 → 6753.75 |
| 2026-03-10 | na_regime | open_t5 6739.75 → 6787.75; high_t5 6757.50 → 6808.50; low_t5 6716.00 → 6764.25; close_t5 6720.50 → 6771.75 |
| 2026-03-11 | computed | open_t5 6695.25 → 6746.50; high_t5 6709.75 → 6759.75; low_t5 6623.00 → 6673.75; close_t5 6626.25 → 6677.25 |
| 2026-03-12 | computed | open_t5 6579.25 → 6629.25; high_t5 6640.75 → 6692.00; low_t5 6561.75 → 6611.75; close_t5 6608.25 → 6658.75 |
| 2026-03-13 | na_regime | open_t1 6697.25 → 6746.75; high_t1 6734.25 → 6784.75; low_t1 6684.50 → 6734.50; close_t1 6703.25 → 6753.75 |
| 2026-03-16 | na_regime | open_t0 6697.25 → 6746.75; high_t0 6734.25 → 6784.75; low_t0 6684.50 → 6734.50; close_t0 6703.00 → 6753.50; close_move_over_im 0.0784 → 0.092; open_t1 6739.75 → 6787.75; high_t1 6757.50 → 6808.50; low_t1 6716.00 → 6764.25; close_t1 6720.50 → 6771.75 |
| 2026-03-17 | na_regime | open_t0 6739.75 → 6787.75; high_t0 6757.50 → 6808.50; low_t0 6716.00 → 6764.25; close_t0 6721.50 → 6773.00; range_over_im 0.5088 → 0.5425; close_move_over_im -0.2237 → -0.1808; open_t1 6695.25 → 6746.50; high_t1 6709.75 → 6759.75; low_t1 6623.00 → 6673.75; close_t1 6626.25 → 6677.25 |
| 2026-03-18 | na_regime | open_t0 6695.25 → 6746.50; high_t0 6709.75 → 6759.75; low_t0 6623.00 → 6673.75; close_t0 6626.00 → 6676.75; open_t1 6579.25 → 6629.25; high_t1 6640.75 → 6692.00; low_t1 6561.75 → 6611.75; close_t1 6608.25 → 6658.75 |
| 2026-03-19 | na_regime | open_t0 6579.25 → 6629.25; high_t0 6640.75 → 6692.00; low_t0 6561.75 → 6611.75; close_t0 6611.00 → 6661.50 |

#### `v0.5.0-rebuilt` — CR-DJ changes (bars with CR-DJ undone → now)

| trade_date | status | column: before (bars with CR-DJ undone) → after (bars now) |
|---|---|---|
| 2025-12-19 | computed | final_close_dist -262.77 → -212.27 |
| 2025-12-22 | computed | final_close_dist -287.25 → -236.75 |
| 2026-02-23 | computed | open_t15 6697.25 → 6746.75; high_t15 6734.25 → 6784.75; low_t15 6684.50 → 6734.50; close_t15 6703.25 → 6753.75 |
| 2026-02-24 | na_regime | open_t15 6739.75 → 6787.75; high_t15 6757.50 → 6808.50; low_t15 6716.00 → 6764.25; close_t15 6720.50 → 6771.75 |
| 2026-02-25 | computed | open_t15 6695.25 → 6746.50; high_t15 6709.75 → 6759.75; low_t15 6623.00 → 6673.75; close_t15 6626.25 → 6677.25 |
| 2026-02-26 | computed | open_t15 6579.25 → 6629.25; high_t15 6640.75 → 6692.00; low_t15 6561.75 → 6611.75; close_t15 6608.25 → 6658.75 |
| 2026-03-09 | na_regime | open_t5 6697.25 → 6746.75; high_t5 6734.25 → 6784.75; low_t5 6684.50 → 6734.50; close_t5 6703.25 → 6753.75 |
| 2026-03-10 | na_regime | open_t5 6739.75 → 6787.75; high_t5 6757.50 → 6808.50; low_t5 6716.00 → 6764.25; close_t5 6720.50 → 6771.75 |
| 2026-03-11 | computed | — (only never-filled columns; no write) |
| 2026-03-12 | computed | — (only never-filled columns; no write) |
| 2026-03-13 | na_regime | open_t1 6697.25 → 6746.75; high_t1 6734.25 → 6784.75; low_t1 6684.50 → 6734.50; close_t1 6703.25 → 6753.75 |
| 2026-03-16 | na_regime | open_t0 6697.25 → 6746.75; open_t1 6739.75 → 6787.75; high_t1 6757.50 → 6808.50; low_t1 6716.00 → 6764.25; close_t1 6720.50 → 6771.75 |
| 2026-03-17 | na_regime | open_t0 6739.75 → 6787.75; open_t1 6695.25 → 6746.50; high_t1 6709.75 → 6759.75; low_t1 6623.00 → 6673.75; close_t1 6626.25 → 6677.25 |
| 2026-03-18 | na_regime | open_t0 6695.25 → 6746.50; open_t1 6579.25 → 6629.25; high_t1 6640.75 → 6692.00; low_t1 6561.75 → 6611.75; close_t1 6608.25 → 6658.75 |
| 2026-03-19 | na_regime | open_t0 6579.25 → 6629.25 |

#### `bt_daily_features.vol_risk_premium` (`v0.5.0-rebuilt`) — CR-DJ changes

| trade_date | stored | new | Δ |
|---|---|---|---|
| 2026-03-16 | -0.03733 | -0.03208 | +0.00525 |
| 2026-03-17 | -0.02380 | -0.01842 | +0.00538 |
| 2026-03-18 | -0.04230 | -0.03680 | +0.00550 |
| 2026-03-19 | -0.04517 | -0.03934 | +0.00583 |
| 2026-03-20 | -0.08971 | -0.07938 | +0.01033 |
| 2026-03-23 | -0.03533 | -0.02576 | +0.00957 |
| 2026-03-24 | -0.04708 | -0.03741 | +0.00967 |
| 2026-03-25 | -0.04325 | -0.03335 | +0.00990 |
| 2026-03-26 | -0.07039 | -0.06058 | +0.00981 |
| 2026-03-27 | -0.08952 | -0.08023 | +0.00930 |
| 2026-03-30 | -0.08112 | -0.07157 | +0.00955 |
| 2026-03-31 | -0.00314 | 0.00484 | +0.00798 |
| 2026-04-01 | -0.01176 | -0.00315 | +0.00862 |
| 2026-04-02 | -0.01539 | -0.00674 | +0.00865 |
| 2026-04-07 | -0.01365 | -0.00857 | +0.00508 |
| 2026-04-08 | 0.03281 | 0.03771 | +0.00490 |
| 2026-04-09 | 0.04873 | 0.05370 | +0.00497 |
| 2026-04-10 | 0.03843 | 0.04391 | +0.00548 |

`vol_risk_premium` moves by +0.005 to +0.010 (vol points as a decimal), because the 20-session realized vol now sees M26 closes. **Caveat:** the recompute doesn't remove the roll seam from these windows. Before the patch the H26 → M26 step sat in the 03-19 → 03-20 close-to-close return; now it sits in the 03-13 → 03-15 (UTC Sunday) return. Either way one ~0.75 % spurious return is in every window covering the roll. That is the same seam-bias class as `outcome-target-vs-es-roll-seam-and-basis-bias` (CR-BG Q2), and it is not fixed here.

**Proposed write plan (unattended protocol; nothing written until Ryan approves):**

1. `scripts/cr_dl_recompute_derived.py`: `get_backfill_db_conn` → `assert_role_or_die` → `backfill_run(conn, "CR-DL")`.
2. Compute pre / mid / post exactly as the dry run, and refuse to write if any reproduction check fails on a column it would write.
3. **Archive first:** the complete old rows (45 outcome rows; 18 `vol_risk_premium` values with their keys) go as JSON into `bt_backfill_runs.smoke_test_results` via `update_run_smoke`, before the write.
4. **One transaction, in-place UPDATE, column-scoped** (CR-BG A1 pattern, minus the full-row overwrite):
   - For each (version, trade_date) above, set **only the columns in the CR-DJ diff** to the post value.
   - Guard each UPDATE on the PK plus every written column `IS NOT DISTINCT FROM` its archived value, and require rowcount 1, so any concurrent change rolls the whole write back.
   - `bt_daily_outcomes` rows also get `backfill_run_id` = this run and `computed_at` = NOW() (CR-BG precedent; the old values are in the archive).
   - `bt_daily_features.vol_risk_premium`: UPDATE that one column only, guarded the same way. `backfill_run_id` is not stamped (CR-D convention: feature-row attribution is preserved); the run row records the 18 keys.
   - Writes: 45 outcome rows (`v0.6.1` 15, `v0.6.0-openiv` 15, `v0.5.0-rebuilt` 13 + 2 with nothing to write → 43 UPDATEs) + 18 feature rows. `active`, `deactivated_*`, the PK and every other column are untouched. No INSERT, no DELETE.
5. **Smoke** in the same run:
   - Every written column equals post; every unwritten column equals its archived value.
   - Recompute on the stored rows now gives mid → post = ∅ (nothing left for CR-DJ).
   - Row counts per version are unchanged.
   - `-spxcash` is untouched.
   - Results go to `update_run_smoke`.
6. **Grants:** none needed. `dash_backfill_writer` has UPDATE on every `bt_daily_outcomes` column (checked: `session_open_t1`, `position_t1_post_touch`, `session_open_t0`, `wall_above_price`, `backfill_run_id`, `computed_at` all true) and on `bt_daily_features.vol_risk_premium` (true).

**Choices for Ryan:**
- **D1, versions:** all three ES versions (recommended: the same code reproduces each one, and research may read older versions) or canonical `v0.6.1` only.
- **D2, column scope:** CR-DJ columns only (recommended: matches the kickoff). The alternative is a full-row recompute on current bars, which would also pull CR-BZ changes (F2) into these 45 rows only, a patchy mix.
- **D3, `vol_risk_premium`:** recompute the 18 (recommended for consistency; tiny change; seam caveat above) or leave them and record.

### Approval and amendment 1 (Ryan, chat 2026-10-29)

**Approved:** D1 all three ES-based versions; D2 only the columns CR-DJ changes; D3 recompute the 18 `vol_risk_premium` rows.

**45 changed rows vs 43 UPDATEs.** The two rows with no write are `v0.5.0-rebuilt` **2026-03-11** and **2026-03-12**. They are *not* skipped because they already match. CR-DJ changes their `session_{open,high,low,close}_t5` (T+5 = 03-18 / 03-19), but those columns are **stored NULL**, as is every t1 / t5 / t15 and position column on those rows. Both were inserted by the CR-AA sweep run `5cef55c5` (2026-06-04), after CR-G / CR-I had run, so those columns were never filled (03-03 and 03-05 from the same run are NULL too; 03-02, run `8d0a8663`, is filled). **Ruling: leave both rows as stored and write 43.** CR-DL corrects values that were computed on H26 bars; it does not fill gaps. Whether to backfill the four CR-AA sweep rows is a separate open question (session note).

**Write mechanics (refines plan steps 2–5).** The full pre / mid / post recompute takes ~30 min on the overlays, so the write script does not re-run it. It takes the Step 0 dry run (`scripts/logs/cr_dl/step0_dryrun.json`) as the frozen plan: for each (version, trade_date, column) in the CR-DJ diff whose stored value is not NULL, `before` = the dry run's stored value and `after` = its post value. Before writing it (i) recomputes post on the current bars with the same writer code and requires it to equal every planned `after`, and (ii) reads the stored rows and requires every planned column to equal its `before`. The dry run already showed no written column is also changed by CR-BZ (stored = mid on every written column), so post − stored on those columns is exactly the CR-DJ effect. Archive, guards, one transaction and stamping as in step 4.

**Smoke:** after the commit, re-run the post recompute and require: every written column equals the dry run's expected value; every row's remaining stored-vs-post difference equals the dry run's, minus the written columns (only the CR-BZ / never-filled differences remain); labels unchanged (`outcome_status`, `reached_touch`, `reached_close`, `days_to_reach`, `max_excursion_in_direction`, `position_t*_post_touch`, `horizon_*`) against the archive; row counts per version unchanged; `-spxcash` untouched; the 18 `vol_risk_premium` values equal the dry run's post.

Standing-rulings conformance re-checked: unchanged (B4 honoured — every read capped at 2026-06-05; rows ≤ 03-20).

### F2 — the CR-BZ fragment repair (2026-10-25) was never propagated to derived rows (out of scope; follow-up)

The same dry run shows that CR-BZ's repair of 783 fragment minutes in 2026 (many at 06:30 / 12:59 PT) changes stored outcome rows that were computed before it.

- **Train rows ≤ 2026-03-20 alone:** `v0.6.1` 43 rows, `v0.6.0-openiv` 29, `v0.5.0-rebuilt` 26 (10–13 of them overlap the CR-DJ rows).
- **Columns:** mostly `session_open_t0/t1/t5/t15` (06:30 TAIL fragments), plus `max_excursion_in_direction` (7 rows per version), `close_move_over_im`, `range_over_im`, and one `final_close_distance_from_target`.
- `vol_risk_premium` is unaffected (it uses no 06:30 bar).

CR-BZ's repairs run to 2026-10-22, so holdout and forward-log rows are affected too. Recomputing those reads holdout bars: a B4 question for its own CR. Proposed as a vault open-question, not done here.
