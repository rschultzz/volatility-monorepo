# CR-DO — Recompute derived rows after CR-BZ's fragment repair (train full, holdout blind)

Branch `feat/CR-DO-crbz-derived-recompute` (worktree `../volatility-monorepo-cr-do`) off `origin/main` `a2c2216` (CR-DL merge).
Kickoff: chat 2026-10-29 (Ryan). Data CR (`data_safety_class: write_backfill`).
Source: vault open-question `crbz-fragment-repair-derived-recompute` (CR-DL F2).

## Problem

CR-BZ (run `88139854-3605-4d94-ab03-7bc34af5da5c`, 2026-10-25) repaired 783 partial ("fragment") minutes in `ironbeam_es_1m_bars` and `es_minutes` (2026-01-02 → 2026-10-22; many at 06:30 / 12:59 PT). No derived table was recomputed afterwards. CR-DL's dry run measured the effect only on rows ≤ 2026-03-20: `v0.6.1-nyse-sessions` 43 rows, `v0.6.0-openiv` 29, `v0.5.0-rebuilt` 26.

## Kickoff (frozen)

Read first: CLAUDE.md (Unattended backfill protocol); vault `components/field-conventions.md`; `research/standing-rulings.md` (B4 incl. the 2026-10-29 blind-maintenance clarification); vault open-question `crbz-fragment-repair-derived-recompute` (F2); `specs/CR-DL-march-2026-derived-recompute.md` (same method: archive first, compare-and-update, only changed columns, one transaction). Data CR: conformance table required.

NEVER print connection strings: catch DB errors and print the error type only.

**Scope:** correct every derived value that depends on a repaired bar, in every live derived table (`bt_daily_outcomes` in all ES-based feature versions, `vol_risk_premium`, and anything else Step 0 finds), using the code that wrote each row (as CR-DL did, including exact copies from git history for retired code paths).

**Two regimes (B4):**
- **TRAIN** (trade dates ≤ 2026-06-05, horizons ending ≤ 06-05): full CR-DL treatment. Per-row before / after tables in the spec, values visible.
- **HOLDOUT** (any row whose trade date or horizon touches 2026-06-06 → 2026-10-26): **BLIND.** Scripts may read and write these rows, but nothing shown to Ryan or in any output, log, spec, commit or session note may contain an outcome value, row, diff, example or summary statistic from them. Allowed output only: row counts per table / version, the names of columns changed, counts of changed values per column, and checksums (e.g. md5 of the sorted recomputed column values) with a second independent run reproducing the same checksums. Logs for holdout steps are written in that form from the start; never write values and redact later.
- **Forward log** (≥ 2026-10-27) and **R-b** (2025): confirm in Step 0 (counts only) that no repaired bar touches them. If any does, STOP and report.

**Step 0** (commit before any write; stop and report):
- a. Every derived table and column that reads Ironbeam bars, and which code wrote each feature version's rows. Confirm each path recomputes stored rows unchanged on the bars as they were before CR-BZ (train: match counts; holdout: checksum equality only).
- b. Train: per table / version, the rows and columns the repair changes, with before / after values, and confirm no outcome label changes; list any that do.
- c. Holdout: counts only — rows per table / version that would change, columns changed, and the number of labels (status / touch / reached) that would change, as a COUNT, no values.
- d. Interaction with CR-DL / CR-DJ: rows already rewritten for March 2026 must come out identical or differ only in CR-BZ columns. Show that for train; checksum only for any holdout overlap.
- e. Write plan under the unattended backfill protocol, any grants needed (don't apply without Ryan), and the blind smoke design for holdout rows.

Ryan decides after Step 0, especially if any label count (b or c) is non-zero.

**Wrap:** session note (holdout parts in counts / checksums only), PR (don't merge), logs copied (holdout logs are already count-only), no deploy.

## Standing rulings conformance

| Ruling | Status | Note |
|---|---|---|
| A1–A22 (the setup) | n/a | Data repair of stored derived rows; no setup, level, entry, exit or event filter is defined or tested. |
| B1, B2, B3 | n/a | No verdict, parameter sweep or setup optimisation. |
| B4 (data split) | honoured, under the 2026-10-29 blind-maintenance clarification | Train rows: full treatment. Holdout rows (trade date or any bar read in 2026-06-06 → 10-26) are read and written by scripts only; every output for them is counts, column names and checksums, from the first run. Forward log and R-b: counts-only confirmation that no repaired bar touches them; no writes there. No research result is read or re-run. |
| B5 (Databento is the reference) | n/a | No test. The recompute reproduces each stored row's own Ironbeam-bar computation on the repaired bars (CR-BZ rebuilt the fragments from Ironbeam's own 1-s flow). |
| B6, B7, B8 | n/a | No exits, placebos or labels. `setup_candidates` / `setup_reviews` are not modified. |
| C1, C2 | n/a | No UI or display change. |
| Revoked R1–R7 | none used | R6 (`discounted_level`) is the stored `bt_daily_outcomes` target frame; the recompute reproduces the existing computation unchanged and introduces it into no new test. |

## Step 0 findings (2026-10-29, read-only)

Scripts (read-only transactions as `dash_backfill_writer`, no run row; logs in `scripts/logs/cr_do/`, git-ignored):
- `scripts/cr_do_step0_dryrun.py`: every active row of the three ES-based versions with trade_date ≥ 2025-09-01 (the earliest that can reach a 2026 bar over 60 sessions), plus every non-null 2026 `vol_risk_premium`.
- `scripts/cr_do_step0_inventory.py`: the other tables, counts only.

Each row is computed by **the code that wrote it** (CR-DL's writers: `cr_bi_backfill_nyse_sessions.compute_row` for `v0.6.1`; the verbatim pre-CR-BI frames from `5531453^` for `v0.6.0-openiv` / `v0.5.0-rebuilt`; CR-D `_compute_vol_features`) on two bar states:
- **pre:** CR-BZ undone. This is the earliest `old_*` per minute of run `88139854` in `ironbeam_es_1m_bars_repair_log`. The one exception is minutes CR-DJ later rewrote: they keep CR-DJ's values, which CR-DL already carried into the derived rows.
- **post:** the bars now.

Reproduction compares stored with **pre** for rows computed before CR-BZ finished (2026-10-25 17:17), and for rows stamped by CR-DL (which wrote only CR-DJ columns). It compares stored with **post** for rows computed later. The CR-BZ effect is pre → post, on columns the row stores. A stored NULL means the writer never filled that column: it is not compared and not written (CR-DL rule). Values are compared at 1e-6, or at 4 dp for the float4 columns. That is tighter than CR-DL's 0.013, which matters for d.

**Regimes (row level):**
- **train:** trade_date ≤ 2026-06-05, and the row computes identically in both states with every bar read capped at 06-05, so it reads no holdout bar.
- **holdout:** trade_date 06-06 → 10-26, or a train-dated row that reads a holdout bar. Every T+15 row from about 05-11 on falls here, as does every 60-session row from about 03-12 on.
- **rb:** trade_date in 2025.
- **fwd:** trade_date ≥ 10-27.

Holdout, R-b and forward rows are blind: the script prints and stores only counts, column names and md5 checksums for them. The checksums are md5 of the sorted `version|date|column|value` lines:
- `keys`, `before` and `after` cover the planned write.
- `all_post` covers every filled column of the group.

**Second-run equality:** run2 and run3 (independent processes, final code) print identical logs. Every holdout / R-b / forward checksum below is reproduced, and `all_post` also matches the two earlier runs.

### a. Derived tables that read Ironbeam bars, their writers, and reproduction

CR-BZ's 783 repaired minutes fall on train sessions (620) and holdout sessions (163). None falls on a 2025 or forward-log session.

| Table / version | Reads ES bars? | Writers of rows ≥ 2025-09-01 (`backfill_run_id` → `cr_id`) | Reproduction (pre = stored, or post for rows computed after CR-BZ) |
|---|---|---|---|
| `bt_daily_outcomes` `v0.6.1-nyse-sessions` (canonical) | yes (shared PT session frame) | CR-BI `compute_row` 237; CR-022 daily insert 17; CR-AA sweep 14; CR-DL 15 | train **78 / 81**; holdout 117 / 118 (column-level: `close_move_over_im` 1); R-b 81 / 81; fwd 3 / 3 |
| `bt_daily_outcomes` `v0.6.0-openiv` | yes (pre-CR-BI frames) | CR-G 117; CR-I 91; CR-022 12; CR-AA 14; CR-BG 6; CR-DL 15 | train **80 / 83**; holdout 88 / 91 (`outcome_status` 3); R-b 81 / 81 |
| `bt_daily_outcomes` `v0.5.0-rebuilt` | yes (pre-CR-BI frames) | CR-G 142; CR-AA 23; CR-022 5; CR-DL 13 | train **80 / 83**; holdout 17 / 19 (`session_low_t5` 1, `session_close_t5` 1, `outcome_status` 1); R-b 81 / 81 |
| `bt_daily_outcomes` `v0.6.0-openiv-spxcash` | no (SPX cash) | CR-BH | not affected |
| `bt_daily_features.vol_risk_premium` (`v0.5.0-rebuilt` only) | yes (last bar per UTC date) | CR-D | **94 / 94** reproduce; **0 change** (no repaired minute is a UTC-date last bar). All 2026 rows ≤ 05-22 (train). **No write.** |
| `bt_daily_features` other columns, `feature_vector` | no (ORATS) | — | not affected |
| `es_minutes` | copy of the bars | CR-BZ updated it with the bars | 783 / 783 repaired minutes equal `ironbeam_es_1m_bars` (OHLCV) |
| `es_minutes_with_features*` views | via `es_minutes` | — | self-heal |
| `es_minute_features` | from bars | backfill script | trade dates 2025-11-23 → 12-21 only: not affected |
| `bt_vol_state_daily`, `bt_range_labels_daily`, `orats_gex_landscape`, `gex_walls_daily_rounded` | no (ORATS / SPX cash) | — | not affected |
| `forward_log_*` (CR-CX) | no (Databento only, B5) | `forward_log_g1down.py` | not affected; sessions 10-27 → 10-28 |
| `setup_candidates` | research event sets (some prices from bars) | CR-CB / CR-CC / CR-CZ loads, **immutable by design** (`ON CONFLICT DO NOTHING`; a correction is a new `set_name`) | train: 141 / 209 events sit on a session with a repair, and 3 have a stamped test / confirm / entry / retest minute that is a repaired minute. Holdout: 0 / 19. **Not modified**; see D6. |
| `bt_signals` | live signal-panel snapshots | live UI | 4 train rows on repaired sessions (6 R-b rows, none on a repaired session). Records of what the panel showed; **not recomputed.** |
| `bt_edge_backtest_results` | CR-AH research aggregates | CR-AH | frozen research output; not recomputed (CR-DL precedent: research verdicts are listed, not re-run) |

**Expected state:** every train row was computed before CR-BZ (pre: 81 / 83 / 83). The 4 + 1 `v0.6.1` rows the cron wrote after 10-25 are holdout or forward and reproduce on post.

**Train reproduction fails (values shown; their failing columns are not written):**
- **`v0.6.1` and `v0.6.0-openiv`, 2026-03-16 / 03-18 / 03-19** (the CR-DL rows): `range_over_im` and `close_move_over_im` differ from the recompute by 0.004–0.011. These are **CR-DJ** effects (H26 → M26 t0 range over the same IM) that CR-DL's 0.013 comparison tolerance hid, so CR-DL didn't write them. They are not CR-BZ effects; see d and D4.
  - 03-16: `range_over_im` stored 0.67808 vs 0.68489.
  - 03-18: `range_over_im` 1.09334 vs 1.08389; `close_move_over_im` −0.87278 vs −0.87909.
  - 03-19: `range_over_im` 0.68830 vs 0.69919; `close_move_over_im` 0.27663 vs 0.28098.
- **`v0.5.0-rebuilt`, 2026-03-02 / 03-03 / 03-05:** the CR-AA sweep rows of 2026-05-25 / 06-04, from an unlogged bar state (as CR-DL found). The differences:
  - 03-02: `final_close_distance_from_target` 499.125 vs 504.625.
  - 03-03: `session_open_t0` 6775.75 vs 6782.5.
  - 03-05: `session_open_t0` 6842.5 vs 6860.25.

  Left as stored.

**Holdout reproduction fails (counts only):**
- `v0.6.1`: 1 value (`close_move_over_im`); it is excluded from the write.
- `v0.6.0-openiv`: 3 rows (`outcome_status`).
- `v0.5.0-rebuilt`: 2 rows (`outcome_status` 1; `session_low_t5` + `session_close_t5` 1).

Rule for the plan: a row whose label column fails reproduction is left whole as stored; any other failing column is left as stored. None of the four `outcome_status` rows has a CR-BZ change, so the rule excludes 1 + 1 values in all.

### b. Train — what CR-BZ changes (values below)

| Version | Train rows | Rows changed | Values to write | Label changes |
|---|---|---|---|---|
| `v0.6.1-nyse-sessions` | 81 | 61 | 162 | `outcome_status`, `reached_touch`, `reached_close`, `days_to_reach`, horizons, positions: **0**. `max_excursion_in_direction`: 9 rows. `final_close_distance_from_target`: 1. |
| `v0.6.0-openiv` | 83 | 61 | 179 | same: status / touch / reached **0**; `max_excursion_in_direction` 9; `final_close_distance_from_target` 1 |
| `v0.5.0-rebuilt` | 83 | 50 | 106 | same: **0**; `max_excursion_in_direction` 9; `final_close_distance_from_target` 1 |

**No outcome label changes in train.** `max_excursion_in_direction` moves by 0.25–4.75 pts on 9 dates (2026-02-02, 02-11, 02-18, 02-20, 02-25, 02-27, 03-02, 04-28, 04-30). It feeds the labels, but no status, touch or reach flips. `final_close_distance_from_target` moves by 1.5 pts on one row (`v0.6.1` 03-12; the legacy versions 05-11).

Columns:
- Mostly `session_open_t0/t1/t5/t15`: 06:30 PT TAIL fragments; the open moves by up to 10 pts, median 2–2.5.
- `session_close_t*` on the legacy versions: their 13:30–20:00 UTC frame ends on the 12:59 PT fragment.
- `close_move_over_im` / `range_over_im` follow the t0 open, high and low.
- `close_pos_in_band` (3, containment).

#### `v0.6.1-nyse-sessions` — train, CR-BZ changes (stored, pre-CR-BZ → bars now)

| trade_date | status | column: before → after |
|---|---|---|
| 2026-01-05 | na_regime | session_open_t0 6941.50 → 6934.25; session_low_t0 6933.75 → 6932.75; range_over_im 1.091 → 1.1276; close_move_over_im 0.0642 → 0.33 |
| 2026-01-09 | computed | session_open_t15 6943.25 → 6943.50; session_low_t15 6942.50 → 6941.75 |
| 2026-01-12 | computed | session_open_t5 6881.75 → 6871.75; session_low_t5 6822.50 → 6822.25 |
| 2026-01-13 | computed | session_open_t15 6948.50 → 6949.25 |
| 2026-01-14 | computed | session_open_t15 6849.00 → 6853.50 |
| 2026-01-16 | computed | realized_em_pct 3.8838 → 3.8892; session_open_t1 6881.75 → 6871.75; session_low_t1 6822.50 → 6822.25 |
| 2026-01-21 | na_regime | session_open_t15 7003.00 → 7003.75 |
| 2026-01-26 | na_regime | session_open_t5 6943.25 → 6943.50; session_low_t5 6942.50 → 6941.75 |
| 2026-01-27 | na_regime | session_open_t15 6870.50 → 6873.75 |
| 2026-01-28 | na_regime | session_open_t5 6948.50 → 6949.25 |
| 2026-01-29 | computed | session_open_t5 6849.00 → 6853.50; session_open_t15 6848.00 → 6852.75; session_low_t15 6847.50 → 6847.25 |
| 2026-01-30 | computed | session_open_t1 6943.25 → 6943.50; session_low_t1 6942.50 → 6941.75 |
| 2026-02-02 | computed | **max_excursion** 268.00 → 267.75; session_open_t0 6943.25 → 6943.50; session_low_t0 6942.50 → 6941.75; range_over_im 1.5267 → 1.542; close_move_over_im 1.1131 → 1.108 |
| 2026-02-03 | computed | session_open_t1 6948.50 → 6949.25; session_open_t15 6932.25 → 6930.50; session_low_t15 6930.25 → 6929.75 |
| 2026-02-04 | na_regime | session_open_t0 6948.50 → 6949.25; close_move_over_im -0.5329 → -0.5437; session_open_t1 6849.00 → 6853.50; session_open_t5 7003.00 → 7003.75 |
| 2026-02-05 | na_data | session_open_t0 6849.00 → 6853.50; close_move_over_im -0.2507 → -0.295 |
| 2026-02-06 | na_regime | session_open_t15 6813.00 → 6811.00 |
| 2026-02-10 | computed | session_open_t1 7003.00 → 7003.75; session_open_t5 6870.50 → 6873.75 |
| 2026-02-11 | computed | **max_excursion** 8.5 → 7.75; session_open_t0 7003.00 → 7003.75; close_move_over_im -0.7093 → -0.7215 |
| 2026-02-12 | computed | session_open_t5 6848.00 → 6852.75; session_low_t5 6847.50 → 6847.25 |
| 2026-02-18 | computed | **max_excursion** 613.25 → 610.00; session_open_t0 6870.50 → 6873.75; close_move_over_im 0.212 → 0.1696; session_open_t5 6932.25 → 6930.50; session_low_t5 6930.25 → 6929.75 |
| 2026-02-19 | computed | session_open_t1 6848.00 → 6852.75; session_low_t1 6847.50 → 6847.25 |
| 2026-02-20 | computed | **max_excursion** 692.00 → 687.25; session_open_t0 6848.00 → 6852.75; session_low_t0 6847.50 → 6847.25; range_over_im 1.0372 → 1.0403; close_move_over_im 0.9292 → 0.8705; session_open_t5 6860.25 → 6855.75; session_high_t5 6892.50 → 6894.00; session_close_t5 6881.50 → 6883.75 |
| 2026-02-23 | computed | session_open_t5 6813.00 → 6811.00 |
| 2026-02-24 | na_regime | session_open_t1 6932.25 → 6930.50; session_low_t1 6930.25 → 6929.75 |
| 2026-02-25 | computed | **max_excursion** 607.75 → 609.50; session_open_t0 6932.25 → 6930.50; session_low_t0 6930.25 → 6929.75; range_over_im 0.4957 → 0.5027; close_move_over_im 0.377 → 0.4015 |
| 2026-02-26 | computed | session_open_t1 6860.25 → 6855.75; session_high_t1 6892.50 → 6894.00; session_close_t1 6881.50 → 6883.75 |
| 2026-02-27 | computed | **max_excursion** 679.75 → 684.25; session_open_t0 6860.25 → 6855.75; session_high_t0 6892.50 → 6894.00; session_close_t0 6881.50 → 6883.75; close_pos_in_band 0.3261 → 0.3351; range_over_im 0.7049 → 0.7257; close_move_over_im 0.2937 → 0.387; session_open_t1 6813.00 → 6811.00 |
| 2026-03-02 | computed | **max_excursion** 742.25 → 744.25; session_open_t0 6813.00 → 6811.00; close_move_over_im 1.0682 → 1.0974; session_open_t15 6658.25 → 6658.00 |
| 2026-03-03 | computed | session_open_t5 6801.25 → 6795.00; session_open_t15 6590.25 → 6589.25 |
| 2026-03-04 | na_regime | session_open_t15 6668.75 → 6661.75 |
| 2026-03-05 | computed | session_low_t15 6520.50 → 6520.25 |
| 2026-03-09 | na_regime | session_open_t1 6801.25 → 6795.00; session_close_t15 6387.50 → 6383.25 |
| 2026-03-10 | na_regime | session_open_t0 6801.25 → 6795.00; close_move_over_im -0.0937 → -0.0404; session_open_t15 6459.75 → 6457.25; session_close_t15 6570.00 → 6572.25 |
| 2026-03-11 | computed | session_close_t15 6620.00 → 6620.50 |
| 2026-03-12 | computed | final_close_dist 373.62 → 375.12; session_high_t15 6644.25 → 6644.50 |
| 2026-03-13 | na_regime | session_open_t15 6624.75 → 6621.75 |
| 2026-03-16 | na_regime | session_open_t5 6658.25 → 6658.00; session_high_t15 6659.50 → 6659.25 |
| 2026-03-17 | na_regime | session_open_t5 6590.25 → 6589.25; session_open_t15 6829.75 → 6829.50; session_high_t15 6831.75 → 6836.75 |
| 2026-03-18 | na_regime | session_open_t5 6668.75 → 6661.75 |
| 2026-03-19 | na_regime | session_low_t5 6520.50 → 6520.25 |
| 2026-03-20 | na_regime | session_open_t1 6658.25 → 6658.00 |
| 2026-03-23 | na_regime | session_open_t0 6658.25 → 6658.00; close_move_over_im -0.5352 → -0.5312; session_open_t1 6590.25 → 6589.25; session_close_t5 6387.50 → 6383.25 |
| 2026-03-24 | na_regime | session_open_t0 6590.25 → 6589.25; close_move_over_im 0.1504 → 0.1589; session_open_t1 6668.75 → 6661.75; session_open_t5 6459.75 → 6457.25; session_close_t5 6570.00 → 6572.25 |
| 2026-03-25 | na_regime | session_open_t0 6668.75 → 6661.75; close_move_over_im -0.2008 → -0.1403; session_low_t1 6520.50 → 6520.25; session_close_t5 6620.00 → 6620.50 |
| 2026-03-26 | na_regime | session_low_t0 6520.50 → 6520.25; range_over_im 0.9081 → 0.9104; session_high_t5 6644.25 → 6644.50 |
| 2026-03-27 | na_regime | session_close_t1 6387.50 → 6383.25; session_open_t5 6624.75 → 6621.75 |
| 2026-03-30 | na_regime | session_close_t0 6387.50 → 6383.25; close_pos_in_band -0.1346 → -0.1891; close_move_over_im -0.9304 → -0.9805; session_open_t1 6459.75 → 6457.25; session_close_t1 6570.00 → 6572.25; session_high_t5 6659.50 → 6659.25 |
| 2026-03-31 | na_regime | session_open_t0 6459.75 → 6457.25; session_close_t0 6570.00 → 6572.25; close_pos_in_band 1.9455 → 1.968; close_move_over_im 0.9542 → 0.9953; session_close_t1 6620.00 → 6620.50; session_open_t5 6829.75 → 6829.50; session_high_t5 6831.75 → 6836.75 |
| 2026-04-01 | na_regime | session_close_t0 6620.00 → 6620.50; close_move_over_im 0.1523 → 0.1578; session_high_t1 6644.25 → 6644.50 |
| 2026-04-09 | na_regime | session_open_t15 7198.50 → 7200.00; session_high_t15 7251.25 → 7251.75 |
| 2026-04-21 | na_regime | session_open_t5 7171.50 → 7172.00 |
| 2026-04-22 | computed | session_open_t15 7427.25 → 7429.25 |
| 2026-04-23 | computed | session_open_t5 7198.50 → 7200.00; session_high_t5 7251.25 → 7251.75 |
| 2026-04-24 | computed | session_open_t15 7455.25 → 7459.50 |
| 2026-04-27 | na_regime | session_open_t1 7171.50 → 7172.00 |
| 2026-04-28 | computed | **max_excursion** 129.25 → 128.75; session_open_t0 7171.50 → 7172.00; close_move_over_im -0.0218 → -0.0267 |
| 2026-04-29 | computed | session_open_t1 7198.50 → 7200.00; session_high_t1 7251.25 → 7251.75; session_high_t15 7456.75 → 7458.00 |
| 2026-04-30 | computed | **max_excursion** 388.00 → 386.50; session_open_t0 7198.50 → 7200.00; session_high_t0 7251.25 → 7251.75; range_over_im 0.9325 → 0.9375; close_move_over_im 0.4315 → 0.4167; session_open_t15 7420.50 → 7423.75 |
| 2026-05-06 | computed | session_open_t5 7427.25 → 7429.25 |
| 2026-05-08 | na_regime | session_open_t5 7455.25 → 7459.50; session_open_t15 7583.25 → 7583.00 |

#### `v0.6.0-openiv` — train, CR-BZ changes (stored, pre-CR-BZ → bars now)

| trade_date | status | column: before → after |
|---|---|---|
| 2026-01-05 | na_regime | session_open_t0 6941.50 → 6934.25; session_low_t0 6933.75 → 6932.75; range_over_im 1.091 → 1.1276; close_move_over_im 0.0642 → 0.33 |
| 2026-01-08 | computed | realized_em_pct 4.7848 → 4.7905 |
| 2026-01-16 | computed | realized_em_pct 3.8838 → 3.8892 |
| 2026-01-27 | na_regime | session_close_t15 6878.25 → 6877.25 |
| 2026-01-30 | computed | session_low_t15 6847.50 → 6847.25 |
| 2026-02-02 | computed | **max_excursion** 257.25 → 257.00; session_open_t0 6943.25 → 6943.50; session_low_t0 6942.50 → 6941.75; range_over_im 1.5267 → 1.542; close_move_over_im 1.1131 → 1.108 |
| 2026-02-04 | na_regime | session_open_t0 6948.50 → 6949.25; close_move_over_im -0.5329 → -0.5437 |
| 2026-02-05 | na_data | session_open_t0 6849.00 → 6853.50; close_move_over_im -0.2507 → -0.295 |
| 2026-02-10 | computed | session_close_t5 6878.25 → 6877.25 |
| 2026-02-11 | computed | **max_excursion** 8.5 → 7.75; session_open_t0 7003.00 → 7003.75; close_move_over_im -0.7093 → -0.7215 |
| 2026-02-13 | na_regime | session_low_t5 6847.50 → 6847.25 |
| 2026-02-18 | computed | **max_excursion** 584.25 → 581.00; session_open_t0 6870.50 → 6873.75; close_move_over_im 0.212 → 0.1696; session_open_t15 6801.25 → 6795.00 |
| 2026-02-19 | computed | session_low_t1 6847.50 → 6847.25 |
| 2026-02-20 | computed | **max_excursion** 692.00 → 687.25; session_open_t0 6848.00 → 6852.75; session_low_t0 6847.50 → 6847.25; range_over_im 1.0372 → 1.0403; close_move_over_im 0.9292 → 0.8705; session_low_t15 6675.75 → 6675.25 |
| 2026-02-25 | computed | **max_excursion** 607.75 → 609.50; session_open_t0 6932.25 → 6930.50; session_low_t0 6930.25 → 6929.75; range_over_im 0.4957 → 0.5027; close_move_over_im 0.377 → 0.4015 |
| 2026-02-27 | computed | **max_excursion** 679.75 → 684.25; session_open_t0 6860.25 → 6855.75; session_high_t0 6892.50 → 6894.00; session_close_t0 6881.50 → 6883.75; close_pos_in_band 0.3261 → 0.3351; range_over_im 0.7049 → 0.7257; close_move_over_im 0.2937 → 0.387 |
| 2026-03-02 | computed | **max_excursion** 756.75 → 758.75; session_open_t0 6813.00 → 6811.00; close_move_over_im 1.0682 → 1.0974; session_close_t5 6801.50 → 6801.75; session_open_t15 6658.25 → 6658.00 |
| 2026-03-03 | computed | session_open_t5 6801.25 → 6795.00; session_open_t15 6590.25 → 6589.25 |
| 2026-03-04 | na_regime | session_open_t15 6668.75 → 6661.75 |
| 2026-03-05 | computed | session_low_t5 6675.75 → 6675.25; session_low_t15 6521.00 → 6520.25; session_close_t15 6522.75 → 6523.00 |
| 2026-03-06 | na_regime | session_close_t1 6801.50 → 6801.75; session_close_t15 6410.00 → 6407.25 |
| 2026-03-09 | na_regime | session_open_t1 6801.25 → 6795.00; session_close_t15 6389.50 → 6387.75 |
| 2026-03-10 | na_regime | session_open_t0 6801.25 → 6795.00; close_move_over_im -0.0937 → -0.0404; session_open_t15 6459.75 → 6457.25; session_close_t15 6569.50 → 6569.25 |
| 2026-03-11 | computed | session_low_t1 6675.75 → 6675.25; session_close_t15 6615.25 → 6619.50 |
| 2026-03-12 | computed | session_high_t15 6644.25 → 6644.50 |
| 2026-03-13 | na_regime | session_open_t15 6624.75 → 6621.75 |
| 2026-03-16 | na_regime | session_open_t5 6658.25 → 6658.00; session_high_t15 6659.50 → 6659.25; session_close_t15 6655.75 → 6655.50 |
| 2026-03-17 | na_regime | session_open_t5 6590.25 → 6589.25; session_open_t15 6829.75 → 6829.50; session_high_t15 6831.75 → 6836.75; session_close_t15 6824.25 → 6823.50 |
| 2026-03-18 | na_regime | session_open_t5 6668.75 → 6661.75; session_close_t15 6862.50 → 6861.75 |
| 2026-03-19 | na_regime | session_low_t5 6521.00 → 6520.25; session_close_t5 6522.75 → 6523.00; session_close_t15 6857.75 → 6855.50 |
| 2026-03-20 | na_regime | session_open_t1 6658.25 → 6658.00; session_close_t5 6410.00 → 6407.25; session_close_t15 6923.75 → 6921.25 |
| 2026-03-23 | na_regime | session_open_t0 6658.25 → 6658.00; close_move_over_im -0.5352 → -0.5312; session_open_t1 6590.25 → 6589.25; session_close_t5 6389.50 → 6387.75 |
| 2026-03-24 | na_regime | session_open_t0 6590.25 → 6589.25; close_move_over_im 0.1504 → 0.1589; session_open_t1 6668.75 → 6661.75; session_open_t5 6459.75 → 6457.25; session_close_t5 6569.50 → 6569.25; session_close_t15 7060.50 → 7060.00 |
| 2026-03-25 | na_regime | session_open_t0 6668.75 → 6661.75; close_move_over_im -0.2008 → -0.1403; session_low_t1 6521.00 → 6520.25; session_close_t1 6522.75 → 6523.00; session_close_t5 6615.25 → 6619.50 |
| 2026-03-26 | na_regime | session_low_t0 6520.50 → 6520.25; range_over_im 0.9081 → 0.9104; session_close_t1 6410.00 → 6407.25; session_high_t5 6644.25 → 6644.50; session_close_t15 7161.25 → 7161.50 |
| 2026-03-27 | na_regime | session_close_t1 6389.50 → 6387.75; session_open_t5 6624.75 → 6621.75; session_close_t15 7148.50 → 7148.25 |
| 2026-03-30 | na_regime | session_close_t0 6387.50 → 6383.25; close_pos_in_band -0.1346 → -0.1891; close_move_over_im -0.9304 → -0.9805; session_open_t1 6459.75 → 6457.25; session_close_t1 6569.50 → 6569.25; session_high_t5 6659.50 → 6659.25; session_close_t5 6655.75 → 6655.50; session_close_t15 7098.25 → 7100.50 |
| 2026-03-31 | na_regime | session_open_t0 6459.75 → 6457.25; session_close_t0 6570.00 → 6572.25; close_pos_in_band 1.9455 → 1.968; close_move_over_im 0.9542 → 0.9953; session_close_t1 6615.25 → 6619.50; session_open_t5 6829.75 → 6829.50; session_high_t5 6831.75 → 6836.75; session_close_t5 6824.25 → 6823.50 |
| 2026-04-01 | na_regime | session_close_t0 6620.00 → 6620.50; close_move_over_im 0.1523 → 0.1578; session_high_t1 6644.25 → 6644.50; session_close_t5 6862.50 → 6861.75; session_close_t15 7142.00 → 7141.75 |
| 2026-04-09 | na_regime | session_close_t1 6857.75 → 6855.50; session_open_t15 7198.50 → 7200.00; session_high_t15 7251.25 → 7251.75 |
| 2026-04-10 | na_regime | session_close_t1 6923.75 → 6921.25; session_close_t5 7161.25 → 7161.50 |
| 2026-04-13 | computed | session_close_t5 7148.50 → 7148.25 |
| 2026-04-14 | computed | session_close_t1 7060.50 → 7060.00; session_close_t5 7098.25 → 7100.50 |
| 2026-04-15 | computed | session_close_t15 7387.00 → 7386.75 |
| 2026-04-16 | na_regime | session_close_t1 7161.25 → 7161.50; session_close_t5 7142.00 → 7141.75 |
| 2026-04-17 | computed | session_close_t1 7148.50 → 7148.25 |
| 2026-04-20 | na_regime | session_close_t1 7098.25 → 7100.50 |
| 2026-04-21 | na_regime | session_open_t5 7171.50 → 7172.00; session_close_t15 7428.00 → 7428.25 |
| 2026-04-22 | computed | session_close_t1 7142.00 → 7141.75; session_close_t5 7168.75 → 7170.25; session_open_t15 7427.25 → 7429.25 |
| 2026-04-23 | computed | session_open_t5 7198.50 → 7200.00; session_high_t5 7251.25 → 7251.75; session_close_t15 7525.75 → 7526.00 |
| 2026-04-24 | computed | session_open_t15 7455.25 → 7459.50; session_close_t15 7432.25 → 7432.50 |
| 2026-04-27 | na_regime | session_open_t1 7171.50 → 7172.00 |
| 2026-04-28 | computed | **max_excursion** 129.25 → 128.75; session_open_t0 7171.50 → 7172.00; close_move_over_im -0.0218 → -0.0267; session_close_t1 7168.75 → 7170.25 |
| 2026-04-29 | computed | session_open_t1 7198.50 → 7200.00; session_high_t1 7251.25 → 7251.75; session_close_t5 7387.00 → 7386.75; session_high_t15 7456.75 → 7458.00; session_close_t15 7451.50 → 7451.75 |
| 2026-04-30 | computed | **max_excursion** 371.25 → 369.75; session_open_t0 7198.50 → 7200.00; session_high_t0 7251.25 → 7251.75; range_over_im 0.9325 → 0.9375; close_move_over_im 0.4315 → 0.4167; session_open_t15 7420.50 → 7423.75; session_close_t15 7465.25 → 7465.50 |
| 2026-05-05 | computed | session_close_t1 7387.00 → 7386.75; session_close_t5 7428.00 → 7428.25 |
| 2026-05-06 | computed | session_open_t5 7427.25 → 7429.25; session_close_t15 7539.75 → 7540.00 |
| 2026-05-07 | computed | session_close_t5 7525.75 → 7526.00 |
| 2026-05-08 | na_regime | session_open_t5 7455.25 → 7459.50; session_close_t5 7432.25 → 7432.50 |
| 2026-05-11 | computed | final_close_dist -28.25 → -26.75; session_close_t1 7428.00 → 7428.25; session_open_t15 7583.25 → 7583.00 |
| 2026-05-15 | na_regime | session_open_t0 7455.25 → 7459.50; close_move_over_im -0.4387 → -0.4948 |

#### `v0.5.0-rebuilt` — train, CR-BZ changes (stored, pre-CR-BZ → bars now)

| trade_date | status | column: before → after |
|---|---|---|
| 2026-01-08 | computed | realized_em_pct 5.2287 → 5.235 |
| 2026-01-16 | computed | realized_em_pct 4.7395 → 4.7461 |
| 2026-01-27 | na_regime | session_close_t15 6878.25 → 6877.25 |
| 2026-01-30 | computed | session_low_t15 6847.50 → 6847.25 |
| 2026-02-02 | computed | **max_excursion** 257.25 → 257.00 |
| 2026-02-10 | computed | session_close_t5 6878.25 → 6877.25 |
| 2026-02-11 | computed | **max_excursion** 8.5 → 7.75 |
| 2026-02-13 | na_regime | session_low_t5 6847.50 → 6847.25 |
| 2026-02-18 | computed | **max_excursion** 584.25 → 581.00; session_open_t15 6801.25 → 6795.00 |
| 2026-02-19 | computed | session_low_t1 6847.50 → 6847.25 |
| 2026-02-20 | computed | **max_excursion** 692.00 → 687.25; session_low_t15 6675.75 → 6675.25 |
| 2026-02-25 | computed | **max_excursion** 607.75 → 609.50 |
| 2026-02-27 | computed | **max_excursion** 679.75 → 684.25 |
| 2026-03-02 | computed | **max_excursion** 756.75 → 758.75; session_close_t5 6801.50 → 6801.75; session_open_t15 6658.25 → 6658.00 |
| 2026-03-04 | na_regime | session_open_t15 6668.75 → 6661.75 |
| 2026-03-06 | na_regime | session_close_t1 6801.50 → 6801.75; session_close_t15 6410.00 → 6407.25 |
| 2026-03-09 | na_regime | session_open_t1 6801.25 → 6795.00; session_close_t15 6389.50 → 6387.75 |
| 2026-03-10 | na_regime | session_open_t0 6801.25 → 6795.00; session_open_t15 6459.75 → 6457.25; session_close_t15 6569.50 → 6569.25 |
| 2026-03-13 | na_regime | session_open_t15 6624.75 → 6621.75 |
| 2026-03-16 | na_regime | session_open_t5 6658.25 → 6658.00; session_high_t15 6659.50 → 6659.25; session_close_t15 6655.75 → 6655.50 |
| 2026-03-17 | na_regime | session_open_t5 6590.25 → 6589.25; session_open_t15 6829.75 → 6829.50; session_high_t15 6831.75 → 6836.75; session_close_t15 6824.25 → 6823.50 |
| 2026-03-18 | na_regime | session_open_t5 6668.75 → 6661.75; session_close_t15 6862.50 → 6861.75 |
| 2026-03-19 | na_regime | session_low_t5 6521.00 → 6520.25; session_close_t5 6522.75 → 6523.00; session_close_t15 6857.75 → 6855.50 |
| 2026-03-20 | na_regime | session_open_t1 6658.25 → 6658.00; session_close_t5 6410.00 → 6407.25; session_close_t15 6923.75 → 6921.25 |
| 2026-03-23 | na_regime | session_open_t0 6658.25 → 6658.00; session_open_t1 6590.25 → 6589.25; session_close_t5 6389.50 → 6387.75 |
| 2026-03-24 | na_regime | session_open_t0 6590.25 → 6589.25; session_open_t1 6668.75 → 6661.75; session_open_t5 6459.75 → 6457.25; session_close_t5 6569.50 → 6569.25; session_close_t15 7060.50 → 7060.00 |
| 2026-03-25 | na_regime | session_open_t0 6668.75 → 6661.75; session_low_t1 6521.00 → 6520.25; session_close_t1 6522.75 → 6523.00; session_close_t5 6615.25 → 6619.50 |
| 2026-03-26 | na_regime | session_close_t1 6410.00 → 6407.25; session_high_t5 6644.25 → 6644.50; session_close_t15 7161.25 → 7161.50 |
| 2026-03-27 | na_regime | session_close_t1 6389.50 → 6387.75; session_open_t5 6624.75 → 6621.75; session_close_t15 7148.50 → 7148.25 |
| 2026-03-30 | na_regime | session_open_t1 6459.75 → 6457.25; session_close_t1 6569.50 → 6569.25; session_high_t5 6659.50 → 6659.25; session_close_t5 6655.75 → 6655.50; session_close_t15 7098.25 → 7100.50 |
| 2026-03-31 | na_regime | session_open_t0 6459.75 → 6457.25; session_close_t1 6615.25 → 6619.50; session_open_t5 6829.75 → 6829.50; session_high_t5 6831.75 → 6836.75; session_close_t5 6824.25 → 6823.50 |
| 2026-04-01 | na_regime | session_high_t1 6644.25 → 6644.50; session_close_t5 6862.50 → 6861.75; session_close_t15 7142.00 → 7141.75 |
| 2026-04-09 | na_regime | session_close_t1 6857.75 → 6855.50; session_open_t15 7198.50 → 7200.00; session_high_t15 7251.25 → 7251.75 |
| 2026-04-10 | na_regime | session_close_t1 6923.75 → 6921.25; session_close_t5 7161.25 → 7161.50 |
| 2026-04-13 | computed | session_close_t5 7148.50 → 7148.25 |
| 2026-04-14 | computed | session_close_t1 7060.50 → 7060.00; session_close_t5 7098.25 → 7100.50 |
| 2026-04-15 | computed | session_close_t15 7387.00 → 7386.75 |
| 2026-04-16 | na_regime | session_close_t1 7161.25 → 7161.50; session_close_t5 7142.00 → 7141.75 |
| 2026-04-17 | computed | session_close_t1 7148.50 → 7148.25 |
| 2026-04-20 | na_regime | session_close_t1 7098.25 → 7100.50 |
| 2026-04-21 | na_regime | session_open_t5 7171.50 → 7172.00; session_close_t15 7428.00 → 7428.25 |
| 2026-04-22 | computed | session_close_t1 7142.00 → 7141.75; session_close_t5 7168.75 → 7170.25; session_open_t15 7427.25 → 7429.25 |
| 2026-04-23 | computed | session_open_t5 7198.50 → 7200.00; session_high_t5 7251.25 → 7251.75; session_close_t15 7525.75 → 7526.00 |
| 2026-04-24 | computed | session_open_t15 7455.25 → 7459.50; session_close_t15 7432.25 → 7432.50 |
| 2026-04-27 | na_regime | session_open_t1 7171.50 → 7172.00 |
| 2026-04-28 | computed | **max_excursion** 129.25 → 128.75; session_open_t0 7171.50 → 7172.00; session_close_t1 7168.75 → 7170.25 |
| 2026-04-30 | computed | **max_excursion** 371.25 → 369.75; session_open_t0 7198.50 → 7200.00 |
| 2026-05-08 | na_regime | session_open_t5 7455.25 → 7459.50; session_close_t5 7432.25 → 7432.50 |
| 2026-05-11 | computed | final_close_dist -28.25 → -26.75 |
| 2026-05-15 | na_regime | session_open_t0 7455.25 → 7459.50 |

### c. Holdout — counts only (BLIND)

| Version | Holdout rows | Rows changed | Values planned (excluded by reproduction) | Label changes (`outcome_status` / `reached_touch` / `reached_close` / `days_to_reach`) | `max_excursion_in_direction` changes |
|---|---|---|---|---|---|
| `v0.6.1-nyse-sessions` | 118 | 66 | 172 (1) | **0** | 10 |
| `v0.6.0-openiv` | 91 | 67 | 196 (0) | **0** | 10 |
| `v0.5.0-rebuilt` | 19 | 6 | 12 (1) | **0** | 5 |

Columns changed, count of planned values per column:
- **`v0.6.1`:**
  - `session_open_t0` 22, `session_open_t1` 22, `session_open_t5` 18, `session_open_t15` 17
  - `close_move_over_im` 22, `range_over_im` 13, `max_excursion_in_direction` 10
  - `session_high_t0` 7, `session_low_t0` 6, `session_low_t5` 6, `session_low_t1` 5, `session_low_t15` 5, `session_high_t1` 5, `session_high_t5` 4, `session_high_t15` 3
  - `actual_realized_em_pct` 2
  - `session_close_t0` 1, `session_close_t1` 1, `session_close_t5` 1, `session_close_t15` 1, `wall_below_price` 1
- **`v0.6.0-openiv`:**
  - `close_move_over_im` 21, `session_open_t0` 20, `session_close_t1` 20, `session_close_t5` 18, `session_open_t1` 17, `session_open_t5` 14
  - `range_over_im` 13, `session_close_t15` 13, `session_open_t15` 13, `max_excursion_in_direction` 10
  - `session_high_t0` 7, `session_low_t0` 6, `session_low_t1` 6, `session_low_t5` 5, `session_low_t15` 4
  - `actual_realized_em_pct` 2, `session_high_t1` 2, `session_high_t15` 2
  - `session_close_t0` 1, `session_high_t5` 1, `wall_below_price` 1
- **`v0.5.0-rebuilt`:** `max_excursion_in_direction` 5, `session_open_t0` 5, `session_open_t1` 1, `session_close_t1` 1.

Checksums of the planned write (identical in run2 and run3):

| Version | md5 keys | md5 before | md5 after | md5 all_post (every filled holdout column, post) |
|---|---|---|---|---|
| `v0.6.1` | `d2eb66097ca13d1a439f52d56636a68c` | `be466141050a08dd4ca0250d0a46c011` | `7ff06bbd652d9957794f199d349c9b20` | `44e112bc73f5ff886cef154e321c99dd` |
| `v0.6.0-openiv` | `c3f179722bcaa13fbe45bbb9f20ec70a` | `de3047c8f7e376a30ffd84131238e693` | `d1d2cb34374213a31da926bac91849cc` | `3636b3e12273c8d7657b16e34a02a22d` |
| `v0.5.0-rebuilt` | `808ee4a85d9e364c0a332ee5b4f01fb9` | `33c277f56de5a203cf3de921a19e8b9d` | `32592b1ea72a85d8ff9c873939d35a8a` | `4d3848f9fff94ca383d9bd8d57545f52` |

**Forward log:** `v0.6.1` 3 rows (the only version with rows ≥ 10-27): 0 changed; `all_post` `d5efbf1bdde884af5555de2507c5f1aa`. No repaired bar touches the forward log.

**R-b (2025) — STOP item:**
- **`v0.6.1-nyse-sessions`: 1 row of 81 changes, 2 values (`session_open_t15`, `session_low_t15`)**, because its T+15 session is a repaired 2026 train session. No label column changes.
  - md5 keys `d257cae8cfce37ede61d0a4af9ca20ed`, before `9a13d9a48daa2e9f6c19d299bc661fb6`, after `9a71a3be1edb0874f9f470dc6343bca5`.
- `v0.6.0-openiv` / `v0.5.0-rebuilt`: 0 of 81 each. Their T+15 columns on the 2025 rows aren't affected.
- No 2025 *bar* was repaired (CR-BZ scope starts 2026-01-01). The row is touched only through a horizon column.
- The kickoff says STOP and report; Ryan decides (D1).

### d. Interaction with CR-DL / CR-DJ

**Train CR-DL rows (15 / 15 / 13):** stored vs post differs **only in CR-BZ columns** on every row except 03-16 / 03-18 / 03-19 in `v0.6.1` and `v0.6.0-openiv`. There, the remaining difference is the sub-0.013 CR-DJ residual in `range_over_im` / `close_move_over_im` listed in a. No CR-DJ column is also a CR-BZ column on any row. CR-BZ changes these CR-DL rows:

| Version | CR-DL row: CR-BZ columns |
|---|---|
| `v0.6.1` | 02-23 `session_open_t5`; 02-24 `session_open_t1`, `session_low_t1`; 02-25 t0 open / low, both ratios, `max_excursion_in_direction`; 02-26 t1 open / high / close; 03-09 `session_open_t1`, `session_close_t15`; 03-10 `session_open_t0`, `close_move_over_im`, t15 open / close; 03-11 `session_close_t15`; 03-12 `session_high_t15`, `final_close_distance_from_target`; 03-13 `session_open_t15`; 03-16 `session_high_t15`, `session_open_t5`; 03-17 t15 open / high, `session_open_t5`; 03-18 `session_open_t5`; 03-19 `session_low_t5` |
| `v0.6.0-openiv` | 02-25, 03-09 → 03-19 (t15 / t5 / t1 columns, as the tables above); 02-23 / 02-24 / 02-26: none |
| `v0.5.0-rebuilt` | 02-25 `max_excursion_in_direction`; 03-09, 03-10, 03-13, 03-16 → 03-19 (t1 / t5 / t15 columns) |

**Holdout / R-b overlap:** no CR-DL row is holdout (0 / 0 / 0). The 2 CR-DL rows in R-b (2025-12-19, 12-22, already written by CR-DL) differ from post **only in CR-BZ columns** (2 / 2 per version; checked blind, no values). `all_post` for R-b: `50bc1e34f7d6a496e949cfc23928663e` / `422c9cd7d195176d5c2a2e01efa4a1ec` / `cce358ec1d21789c8c516c6d6cc05508`.

> B4 note: CR-DL's spec printed values for two R-b rows (2025-12-19 / 12-22 `final_close_dist`, `realized_em_pct`) before the 2026-10-29 blind clause existed. They are not repeated here.

### e. Write plan (nothing written until Ryan approves)

1. **Script `scripts/cr_do_recompute_derived.py`:** `get_backfill_db_conn` → `assert_role_or_die` → `backfill_run(conn, "CR-DO")`. DB errors print the type only.
2. **Plan.**
   - Train: frozen from `scripts/logs/cr_do/step0_run2.json`, CR-DL style. For each (version, date, planned column), before = stored and after = post.
   - Holdout (and R-b if D1 = write): recomputed in process with the same code. The planned key / before / after md5s must equal the Step 0 checksums above, or nothing is written.
   - Preflight: the post recompute equals every train `after`, and stored equals every `before`.
3. **Archive first.**
   - Train: the complete old rows go as JSON in `bt_backfill_runs.smoke_test_results` (CR-DL pattern).
   - Holdout / R-b: old values must not land anywhere Ryan reviews (`smoke_test_results` is the morning-review surface). **Proposed:** a new insert-only table `bt_derived_repair_archive (run_id, table_name, feature_version, trade_date, column_name, old_value jsonb, archived_at)`, with `dash_backfill_writer` INSERT / SELECT only. All archived rows (train too) go there, and the run row records only counts and an md5 of the archived set. **Needs DDL + grant from Ryan (D5).** Alternative: archive holdout into `smoke_test_results` under a `blind_archive` key that no review query selects (weaker).
4. **Write: one transaction, guarded, column-scoped UPDATEs** (CR-DL pattern).
   - Set only the planned columns.
   - Guard: PK + `computed_at` + every written column `IS NOT DISTINCT FROM` its archived value, cast to the column type (the CR-DL float4 lesson). Rowcount must be 1, or the whole write rolls back.
   - Stamp `backfill_run_id` and `computed_at`.
   - Writes (D1–D4 as recommended):
     - train 61 + 61 + 50 = 172 rows / 447 values;
     - holdout 66 + 67 + 6 = 139 rows / 380 values;
     - R-b 0 or 1 row (D1);
     - optional CR-DJ residual, 6 rows / 10 values (D4).
   - No INSERT, no DELETE. `vol_risk_premium`: no write (0 changes).
5. **Grants:** none for the UPDATE. `dash_backfill_writer` has UPDATE on all 40 owned / stamp columns of `bt_daily_outcomes` (checked 2026-10-29). Only the D5 archive table needs DDL / grant.
6. **Blind smoke** (same run, after commit; holdout output = counts / booleans / md5 only):
   - Re-run the post recompute.
   - Train: every written column equals the plan's `after`. The remaining stored-vs-post differences equal Step 0's minus the written columns. Every unwritten column equals the archive.
   - Holdout / R-b:
     - md5 of the stored planned columns equals Step 0 `md5 after`, and the recompute's planned set has empty md5 keys (nothing left to change).
     - md5 of every stored filled column equals Step 0 `all_post`, except the excluded values: a count (2), and the excluded values' md5 equals their archive md5.
     - md5 of all unwritten columns, and of `outcome_status` / `reached_touch` / `reached_close` / `days_to_reach` separately, is unchanged before vs after.
   - Row and active counts per version are unchanged. `-spxcash`, `vol_risk_premium` and the forward-log rows are untouched (`max(computed_at)` and md5).
   - A **second independent smoke process** re-reads and must print the same md5s.
7. **Race with the crons:**
   - The CR-022 daily insert (13:40 UTC) writes new rows only.
   - The CR-AA sweep (11:05 UTC) promotes `pending_history` rows in `v0.6.1`. Whether any planned holdout row is pending is not measured (blind). The `computed_at` + column guards roll the whole write back if the sweep touched a planned row in between.
   - Run outside 11:00–14:00 UTC.

### Decisions for Ryan

- **D1 — R-b (STOP item):** 1 `v0.6.1` row in 2025 has 2 changed values (`session_open_t15`, `session_low_t15`) through a repaired 2026 T+15 session. No label changes.
  - (a) **Blind-write it**, as the 2026-10-29 B4 clause allows for R-b. Recommended: the same correction as everywhere else, counts / md5 only.
  - (b) Leave it, recorded.
- **D2 — Holdout label-feeding changes:** status / touch / reached / days_to_reach changes = **0** in every version; `max_excursion_in_direction` changes on 10 / 10 / 5 holdout rows.
  - (a) **Write all planned holdout values blind** (recommended).
  - (b) Train only, holdout at the holdout read.
- **D3 — Versions:**
  - (a) **All three ES versions** (recommended, as CR-DL).
  - (b) Canonical `v0.6.1` only.
- **D4 — CR-DJ residual** (train, 3 rows × `v0.6.1` / `v0.6.0-openiv`, `range_over_im` / `close_move_over_im`, 0.004–0.011; CR-DL's tolerance hid them):
  - (a) **Write them here** (recommended; they are from CR-DJ, not CR-BZ, so this is a scope amendment).
  - (b) Leave them and log an open question.
- **D5 — Archive location:**
  - (a) **New insert-only `bt_derived_repair_archive` table** (DDL + grant by Ryan; recommended).
  - (b) `smoke_test_results` `blind_archive` key.
- **D6 — `setup_candidates` / `bt_signals`:** not modified (immutable research sets, live snapshots). 141 train events sit on repaired sessions, and 3 have a stamped minute that was repaired. Whether any affected set needs a corrected re-load (a new `set_name`) is proposed as an open question, not done here.

## Amendment 1 — Ryan's decisions (chat 2026-10-29)

D1–D6 approved as recommended, with these conditions:

- **D1 (a):** blind-write the R-b row.
- **D2 (a):** write all planned holdout values blind.
- **D3 (a):** all three ES-based versions.
- **D4 (a):** write the 10 CR-DJ leftover values (`v0.6.1` / `v0.6.0-openiv`, 03-16 `range_over_im`; 03-18 and 03-19 `range_over_im` and `close_move_over_im`). Scope amended to include them.
  - **Every comparison in this CR is exact in the column's own type, with no tolerance.** That covers reproduction, the regime test, the pre-write guard and the smoke. `real` columns compare as float4 (both sides rounded to float32 the way PostgreSQL stores them). `double precision` compares as float8. Integer, boolean, date and text compare as values. Checksums hash the same canonical form. The Step 0 numbers above used 1e-6 / 4 dp and are superseded by the exact re-run (Amendment 2).
  - **Why CR-DL used 0.013:** CR-DL's `_same(a, b, tol=0.013)` (`scripts/cr_dl_step0_dryrun.py`) is a copy of CR-BI's (`scripts/cr_bi_backfill_nyse_sessions.py`, commented "REAL columns round-trip at ~1e-3 relative"). CR-BI chose it as an absolute tolerance for ES *prices* stored as float4: at ~7000 their resolution is about 0.0005, and 0.013 is far below the 0.25 tick. CR-DL applied it to every column, including the `real` ratios `range_over_im` / `close_move_over_im` (around 1, where 0.013 is about 1 %). So it hid CR-DJ's 0.004–0.011 changes to those ratios on 03-16 / 03-18 / 03-19, and CR-DL left them unwritten. A note goes into CR-DL's session note.
  - D4 proof: on each of the 10 values, stored must equal (exactly) a recompute on bars with **both** CR-DJ and CR-BZ undone (CR-DL's overlay of both runs), and **post** is the value written.
- **D5 (a):** new archive table `bt_derived_repair_archive`, created by the owner role. `dash_backfill_writer` gets INSERT only (no SELECT, UPDATE or DELETE), and no other role gets SELECT. Marked BLIND in vault `components/field-conventions.md` ("contains holdout / R-b values; never queried by Ryan or any Claude session"). The DDL and grant times are recorded below and in the run notes.
- **D6:** open question, counts only: how many CR-CZ cards (`setup_candidates`) sit on CR-BZ-repaired sessions (train only), and whether their outcomes were computed from Databento or Ironbeam bars.

Write under the unattended backfill protocol: archive first, then guarded, column-only UPDATEs in one transaction. Smoke: train with values; holdout and R-b blind (counts, columns, checksums, second-run equality). Then wrap, open a PR (don't merge), copy the logs (holdout parts count-only). No deploy.
