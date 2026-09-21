# CR-BG — Recompute derived rows contaminated by the missed U26→Z26 roll

> Branch: `feat/CR-BG-missed-roll-derived-recompute` (off `origin/main` 741f848)
> `data_safety_class: write_backfill`
> Source: vault open-question `missed-roll-2026-09-derived-data-rebuild-inventory` + kickoff prompt (2026-09-20).
> Do not touch `ironbeam_es_1m_bars`, `es_minutes`, or the `_bak_20260920` tables.

## Problem

The ES U26 → Z26 roll (due Mon 2026-09-14) was missed. From 2026-09-13 22:00 UTC to 2026-09-18 13:29 UTC the live feed ingested the dying **ES U26** contract. CR-BF repaired the *bars* (`ironbeam_es_1m_bars`, `es_minutes` now hold ES Z26, `source='databento'`, 2026-09-13 22:00 → 2026-09-18 20:59 UTC). Everything *stored* that was computed from those bars during the week was computed on ES U26 and has not been rebuilt.

## Inventory — verified from chat (read-only SQL, 2026-09-20)

| Candidate | Finding | Action |
| --- | --- | --- |
| `es_minutes` feature views (`es_minutes_with_features`, `_bt`, …) | plain views | self-healed — none |
| `bt_signals`, `bt_strategy_instances`, `trade_log_trades`, `bt_audit_flags`, `es_minute_features` | 0 rows for 2026-09-14..18 | none |
| `orats_gex_landscape` | SPX/ORATS-derived | none |
| `bt_daily_outcomes` trade_date 2026-09-14..18 (5 rows, ticker SPX) | `session_*` columns are ES U26 prices (09-14 `session_open_t0`=7606.50 = ES U26 open; ES Z26 open was 7673.50) | **recompute** |
| `bt_daily_outcomes` trade_date 2026-09-08 | `computed`, `horizon_end_date` 2026-09-14, computed 2026-09-15 on ES U26 bars | **recompute** |
| `ironbeam_es_flow_1s` 09-14..18 | ~102K rows tagged `XCME:ES.U26` | out of scope — known gap; rows stay, consumers filter on symbol |

## Step-0 diagnosis gate (all answered in this file before any write)

1. Does anything in `bt_daily_features.feature_vector` (or the scalar columns) read ES price/volume, or is it SPX/ORATS-only? If ES-dependent, the 5 rows for 09-14..18 join the recompute list.
2. How does the outcome code compare targets to ES price across sessions: target fixed in ES points at t0, or SPX level converted with a same-day basis? State plainly whether a horizon that crosses a futures roll is biased by the U/Z spread (~67 pts here). If yes, do NOT fix it in this CR; write an open-question and list which `pending_history` rows cross the seam (08-27, 09-03, 09-04, 09-14).
3. Why are 2026-09-11 and 2026-09-16 `outcome_status = na_data`? 09-16 may be a casualty of thin ES U26 bars; 09-11 had full data.
4. Which script legitimately recomputes an outcome row for a given trade_date (`cr_b_backfill_outcomes.py` / `cr_aa_sweep_pending_outcomes.py`), and what are the table's PK and active-row semantics?

## Implementation

- Follow the CR-BD repair pattern: deactivate the affected rows (`active=false`, `deactivated_reason='missed-roll-u26'`), then recompute to new active rows through the unattended backfill protocol (`get_backfill_db_conn`, `assert_role_or_die`, `backfill_run`). No DELETE.
- Smoke: for each recomputed row, print old vs new `session_open_t0` / `session_close_t0` and any changed status, `reached_touch`, `reached_close`. Expect t0 opens to move up ~60–70 pts for 09-14..17.
- Wrap: close the open-question if everything on the list is resolved or explicitly deferred; otherwise leave it `in-progress` with what remains.

## Implementation order

1. Step 0 — diagnosis findings appended to this file (gate).
2. Step 1 — repair script (`scripts/cr_bg_recompute_missed_roll.py`), dry-run output.
3. Step 2 — run the repair; smoke appended.
4. Smoke + wrap.

## Step 0 — diagnosis findings (2026-09-20, read-only SQL under `dash_backfill_writer` with `default_transaction_read_only=on`)

### Q1 — are `bt_daily_features` rows ES-dependent? **No. SPX/ORATS-only; the 5 feature rows do not join the recompute list.**

- Writers: `apps/cron/job_orats_eod.py` → `packages/shared/day_features.compute_and_upsert_daily_features` reads only `orats_gex_landscape` (`landscape`, `table_spot`) and `orats_monies_minute`; `scripts/cr_ab_open_implied_move.py` fills `implied_move_1d` from `orats_monies_minute`. Neither file references `es_minutes` / `ironbeam_es_1m_bars`. "spot" in the feature vector is `table_spot` (prior-day ORATS EOD), not ES.
- The one ES-reading feature is `vol_risk_premium` (`packages/shared/vol_features.py` → 20-session realized vol from `ironbeam_es_1m_bars` daily closes). It is written only by the one-off `scripts/cr_d_backfill_vol_features.py`; last non-NULL `vol_risk_premium` is trade_date 2026-05-22. For 2026-09-08..18 all five scalar columns and `feature_vector->>'vol_risk_premium'` are NULL. Nothing to recompute.
- Forward note (not this CR): any future VRP backfill spanning 2026-09-11→09-14 will see the U/Z seam (+~67 pts, ≈ +0.9% fake log-return) in the daily-close series. Folded into the new open-question under Q2.

### Q2 — how are targets compared to ES across sessions? **Target is fixed at t0 as an SPX forward-space level; ES front-contract bars are compared to it raw, with no basis conversion and no roll adjustment. Yes — a horizon that crosses a futures roll is biased by the U/Z spread.**

- `pick_drift_target(walls)` returns the price of the largest positive-GEX wall in `orats_gex_landscape.walls` for trade_date. Wall prices are `discounted_level = strike × exp((r−q)×T)` per option expiry (SPX forward space, mostly short-dated → close to SPX cash). The value is fixed once at t0.
- `compute_outcome` compares that fixed level to daily RTH OHLC aggregated from `ironbeam_es_1m_bars` (whatever front contract is stored) for every session of the horizon: `high >= target` (magnet-above), `low <= target` (magnet-below), band overlap (pin); `reached_close` uses `|final_close − target| ≤ 0.25 × implied_move_1d`. `packages/shared/outcomes.py` docstring (E6) asserts the residual is "< 5 pts".
- Consequences, stated plainly:
  1. **Seam-crossing bias.** At a roll the stored series jumps by the calendar spread (ES U26→Z26: 09-14 open 7606.50 vs 7673.50, ≈ +67 pts) while the target stays put. For a magnet-above row whose horizon spans the seam, a +67-pt step makes `reached_touch` / `reached_close` spuriously easier (magnet-below: harder), and `max_excursion_in_direction` / `actual_realized_em_pct` absorb a 67-pt non-move. 67 pts is 0.7–1.8× the week's `implied_move_1d` (38–104), i.e. far outside the 0.25×EM tolerance.
  2. **Level bias after every roll (broader than the seam).** A fresh front contract carries ~3 months of carry over SPX cash, so right after a roll ES sits ~60–70 pts above wall space, decaying toward ~0 at expiry. The "< 5 pts" claim in E6 only holds near expiry. This also affects `compute_session_containment` (`nearest_walls(walls, ES open)`).
- **Not fixed in this CR.** New open-question: `outcome-target-vs-es-roll-seam-and-basis-bias`.
- Active `pending_history` rows (all `v0.6.0-openiv`): **2026-08-18 (60 sessions), 2026-08-27 (60), 2026-09-03 (20), 2026-09-04 (20)** start on ES U26 and cross the U/Z seam. (08-18 was not in the kickoff list; it is the same case.) **2026-09-14 (20 sessions, ends ≈ 2026-10-09)**: after this CR's recompute its whole horizon is ES Z26, so it does *not* cross a seam — it is subject only to the level bias (2). 08-18 and 08-27 (60 sessions → mid-November) do not reach the Z26→H27 roll (2026-12-14).
- 2026-09-08 (`computed`, 5 sessions 09-08→09-14): after the recompute its horizon is 4 ES U26 sessions + 1 ES Z26 session (09-14) — a seam-crosser by construction. Recomputed here as instructed (the alternative, U26 for 09-14, is no longer in the bars table); flagged in the new open-question.

### Q3 — why are 2026-09-11 and 2026-09-16 `na_data`? **Not a bar problem. Both are `magnet-above` with zero positive-GEX walls → `drift_target is None` → `na_data`.**

- `orats_gex_landscape.walls`: 09-11 = one wall, 7501.425, sign −1; 09-16 = one wall, 7524.475, sign −1. `pick_drift_target` → None → `compute_outcome` step 3 returns `na_data` before any bar is read.
- 09-16 is therefore *not* a casualty of thin ES U26 bars; its recompute will stay `na_data` (only `session_*_t0` change). 09-11 is outside the contaminated window and is not touched.
- Underlying oddity (not this CR): the feature row says `is_magnet_day=1, magnet_direction_signed=1, n_clusters_total=0` while the landscape has no positive wall — the classifier's magnet-above and the outcome runner's wall pick disagree. Same on 09-10/09-17 shape-wise (no positive wall, but those are `amplification` → `na_regime`). Recorded in the new open-question list below; possibly related to vault `regime-classification-may-be-distance-blind`.

### Q4 — legitimate recompute script, PK, active-row semantics

- **PK: `(ticker, trade_date, feature_version)`** — `active` is *not* part of the key, and there is no other unique index. No FK to `bt_daily_features`. Consumers (`SetupV2`, `Proposals`, `probability.py`, `edge_today.py`) read the view `bt_daily_outcomes_active`.
- `scripts/cr_b_backfill_outcomes.py`: INSERT-only. Targets = active feature rows with **no outcome row of any `active` state** for `(ticker, trade_date, feature_version)`; `ON CONFLICT (ticker, trade_date, feature_version) DO NOTHING`. It cannot recompute an existing date.
- `scripts/cr_aa_sweep_pending_outcomes.py`: in-place UPDATE, guarded `WHERE outcome_status='pending_history'` (+ a containment null-fill guarded `session_close_t0 IS NULL`). It never touches `computed` / `na_*` rows and never rewrites a non-NULL `session_*_t0`.
- **So no existing script recomputes an outcome row for a given trade_date**, and the spec's "deactivate, then recompute to new active rows" **cannot be done as written**: a second row for the same `(SPX, date, v0.6.0-openiv)` violates the PK. CR-BD's pattern worked because its deactivated rows (closed days) and its new rows (true trading days) had *different* trade_dates.
- Role: `dash_backfill_writer` has INSERT/SELECT/UPDATE on `bt_daily_outcomes` (every column, incl. `active`, `feature_version`); no DELETE.
- Both computation paths share `packages/shared/outcomes_runner.compute_outcome_for_date` + `compute_session_containment`; the CR-BG script must call those, not re-implement.

### Before-state of the 6 rows (all `v0.6.0-openiv`, `active=true`)

| trade_date | status | regime / bucket | open_t0 | high | low | close | hed | touch/close |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-09-08 | computed | magnet-above / 1-7 DTE (5) | 7712.25 | 7717.75 | 7674.25 | 7681.25 | 2026-09-14 | F / F |
| 2026-09-14 | pending_history | magnet-above / 8-30 DTE (20) | 7606.50 | 7652.00 | 7595.25 | 7624.25 | — | — |
| 2026-09-15 | na_regime | amplification | 7616.75 | NULL | NULL | NULL | — | — |
| 2026-09-16 | na_data | magnet-above / 8-30 DTE | 7609.25 | NULL | NULL | NULL | — | — |
| 2026-09-17 | na_regime | amplification | 7646.00 | NULL | NULL | NULL | — | — |
| 2026-09-18 | na_regime | amplification | NULL | NULL | NULL | NULL | — | — |

Current ES Z26 RTH daily bars (391 bars/day, `source='databento'`): 09-14 O 7673.50 H 7719.75 L 7662.25 C 7692.00 · 09-15 O 7684.50 C 7659.75 · 09-16 O 7676.50 C 7623.00 · 09-17 O 7713.50 C 7705.25 · 09-18 O 7710.00 C 7719.25. 09-08's own session (ES U26, `ironbeam`) is unchanged: only its horizon metrics can move.

Side findings: (a) containment columns for 09-15..18 were never filled (sweep null-fill has not landed them), and `wall_above_price`/`wall_below_price` are NULL on every row since 09-08 even where walls exist on both sides of the open (09-08, 09-09, 09-14) — the recompute fills them for the 6 rows; the wider gap (09-09..11) goes to the open-question list. (b) 09-18 `session_open_t0` is NULL (insert ran at 13:40 UTC right after the 13:29 symbol fix).

## Amendment A1 (2026-09-20, Ryan's decision after Step 0 / Q4) — in-place UPDATE replaces deactivate-and-reinsert

The "Implementation" bullet "deactivate the affected rows … then recompute to new active rows" is **superseded**: the PK `(ticker, trade_date, feature_version)` excludes `active`, so a deactivated row and a new active row for the same date cannot coexist. Instead:

1. **In-place UPDATE** of the 6 rows (`SPX`, `v0.6.0-openiv`, trade_date 2026-09-08, 09-14, 09-15, 09-16, 09-17, 09-18) under `dash_backfill_writer` inside `backfill_run` (cr_id `CR-BG`). No DELETE, no INSERT.
2. **Full-row recompute.** Every computed column is overwritten, including explicit NULLs where the new computation has no value; no column may retain an ES U26-derived value. Computed columns = the 11 `compute_outcome` fields, `session_open_t0`, the 11 CR-AQ containment columns, and the CR-G/CR-I columns (`session_{open,high,low,close}_t{1,5,15}`, `position_t{1,5,15}_post_touch`). The CR-G/CR-I columns are NULL on all 6 rows today (their one-off backfills last reached 2026-09-02 / 2026-08-14) and are written as explicit NULL — this CR does not extend those backfills.
3. **Archive first.** The 6 complete old rows go, as JSON, into `bt_backfill_runs.smoke_test_results` and into this spec *before* the write. Six rows only; no bulk data in the repo.
4. **`active`, `deactivated_at`, `deactivated_reason` are not touched.** The repair is recorded via `backfill_run_id` and `computed_at` only. `deactivated_reason='missed-roll-u26'` is not used.
5. **Smoke diffs all columns** old vs new (not just `session_*`).
6. New vault open-question (priority low, `status: future-project`): PK on `bt_daily_outcomes` excludes `active` → deactivate-and-reinsert impossible for same-date rows; candidate fix = partial unique index on active rows; CR-BD only worked because its new rows had different trade_dates.

Guard: each UPDATE is keyed on the PK and additionally requires `computed_at` and `session_close_t0` to equal the archived values (catches a sweep promotion or a containment null-fill landing between archive and write); the six UPDATEs run in one transaction, so any guard miss rolls back the whole write.

## Step 1 — repair script + dry-run (2026-09-20)

`scripts/cr_bg_recompute_missed_roll.py` (new). Computation via `packages/shared/outcomes_runner` (shared with CR-022 / CR-AA); bars + landscape loaders imported from `scripts/cr_b_backfill_outcomes.py`. `--dry-run` prints the archive and the planned diff, writes nothing, creates no run row.

### Archive — the 6 complete old rows (A1 item 3; one JSON object per line, NULL-valued keys included)

```json
{"ticker":"SPX","trade_date":"2026-09-08","feature_version":"v0.6.0-openiv","regime_kind_at_classification":"magnet-above","dominant_bucket_at_classification":"1-7 DTE","horizon_sessions":5,"horizon_end_date":"2026-09-14","outcome_status":"computed","reached_touch":false,"reached_close":false,"days_to_reach":null,"max_excursion_in_direction":5.5,"final_close_distance_from_target":-202.02499999999964,"actual_realized_em_pct":2.1275216887313793,"active":true,"deactivated_at":null,"deactivated_reason":null,"backfill_run_id":"0cd0e6a2-b036-4264-93a9-fe77d93c344e","computed_at":"2026-09-15T11:05:51.946702","position_t1_post_touch":null,"position_t5_post_touch":null,"position_t15_post_touch":null,"session_open_t1":null,"session_high_t1":null,"session_low_t1":null,"session_close_t1":null,"session_open_t5":null,"session_high_t5":null,"session_low_t5":null,"session_close_t5":null,"session_open_t15":null,"session_high_t15":null,"session_low_t15":null,"session_close_t15":null,"session_open_t0":7712.25,"session_high_t0":7717.75,"session_low_t0":7674.25,"session_close_t0":7681.25,"wall_above_price":null,"wall_below_price":null,"contained_close":null,"contained_range":null,"close_pos_in_band":null,"range_over_im":0.69978976,"close_move_over_im":-0.49870074,"breach_side":null}
{"ticker":"SPX","trade_date":"2026-09-14","feature_version":"v0.6.0-openiv","regime_kind_at_classification":"magnet-above","dominant_bucket_at_classification":"8-30 DTE","horizon_sessions":20,"horizon_end_date":null,"outcome_status":"pending_history","reached_touch":null,"reached_close":null,"days_to_reach":null,"max_excursion_in_direction":null,"final_close_distance_from_target":null,"actual_realized_em_pct":null,"active":true,"deactivated_at":null,"deactivated_reason":null,"backfill_run_id":"9d5252b1-5757-4a54-bda7-5ce83aca6adf","computed_at":"2026-09-14T13:41:07.044158","position_t1_post_touch":null,"position_t5_post_touch":null,"position_t15_post_touch":null,"session_open_t1":null,"session_high_t1":null,"session_low_t1":null,"session_close_t1":null,"session_open_t5":null,"session_high_t5":null,"session_low_t5":null,"session_close_t5":null,"session_open_t15":null,"session_high_t15":null,"session_low_t15":null,"session_close_t15":null,"session_open_t0":7606.5,"session_high_t0":7652.0,"session_low_t0":7595.25,"session_close_t0":7624.25,"wall_above_price":null,"wall_below_price":null,"contained_close":null,"contained_range":null,"close_pos_in_band":null,"range_over_im":1.4936396,"close_move_over_im":0.46717364,"breach_side":null}
{"ticker":"SPX","trade_date":"2026-09-15","feature_version":"v0.6.0-openiv","regime_kind_at_classification":"amplification","dominant_bucket_at_classification":"8-30 DTE","horizon_sessions":null,"horizon_end_date":null,"outcome_status":"na_regime","reached_touch":null,"reached_close":null,"days_to_reach":null,"max_excursion_in_direction":null,"final_close_distance_from_target":null,"actual_realized_em_pct":null,"active":true,"deactivated_at":null,"deactivated_reason":null,"backfill_run_id":"31b47b1e-d40b-43ff-8765-cfc7aec1f61c","computed_at":"2026-09-15T13:40:50.731692","position_t1_post_touch":null,"position_t5_post_touch":null,"position_t15_post_touch":null,"session_open_t1":null,"session_high_t1":null,"session_low_t1":null,"session_close_t1":null,"session_open_t5":null,"session_high_t5":null,"session_low_t5":null,"session_close_t5":null,"session_open_t15":null,"session_high_t15":null,"session_low_t15":null,"session_close_t15":null,"session_open_t0":7616.75,"session_high_t0":null,"session_low_t0":null,"session_close_t0":null,"wall_above_price":null,"wall_below_price":null,"contained_close":null,"contained_range":null,"close_pos_in_band":null,"range_over_im":null,"close_move_over_im":null,"breach_side":null}
{"ticker":"SPX","trade_date":"2026-09-16","feature_version":"v0.6.0-openiv","regime_kind_at_classification":"magnet-above","dominant_bucket_at_classification":"8-30 DTE","horizon_sessions":20,"horizon_end_date":null,"outcome_status":"na_data","reached_touch":null,"reached_close":null,"days_to_reach":null,"max_excursion_in_direction":null,"final_close_distance_from_target":null,"actual_realized_em_pct":null,"active":true,"deactivated_at":null,"deactivated_reason":null,"backfill_run_id":"fc3cc3e5-4b70-489e-a32b-c6ea0bffa943","computed_at":"2026-09-16T13:40:48.025298","position_t1_post_touch":null,"position_t5_post_touch":null,"position_t15_post_touch":null,"session_open_t1":null,"session_high_t1":null,"session_low_t1":null,"session_close_t1":null,"session_open_t5":null,"session_high_t5":null,"session_low_t5":null,"session_close_t5":null,"session_open_t15":null,"session_high_t15":null,"session_low_t15":null,"session_close_t15":null,"session_open_t0":7609.25,"session_high_t0":null,"session_low_t0":null,"session_close_t0":null,"wall_above_price":null,"wall_below_price":null,"contained_close":null,"contained_range":null,"close_pos_in_band":null,"range_over_im":null,"close_move_over_im":null,"breach_side":null}
{"ticker":"SPX","trade_date":"2026-09-17","feature_version":"v0.6.0-openiv","regime_kind_at_classification":"amplification","dominant_bucket_at_classification":"8-30 DTE","horizon_sessions":null,"horizon_end_date":null,"outcome_status":"na_regime","reached_touch":null,"reached_close":null,"days_to_reach":null,"max_excursion_in_direction":null,"final_close_distance_from_target":null,"actual_realized_em_pct":null,"active":true,"deactivated_at":null,"deactivated_reason":null,"backfill_run_id":"2ea340e1-2395-4992-a4e4-58570e491120","computed_at":"2026-09-17T13:40:58.835575","position_t1_post_touch":null,"position_t5_post_touch":null,"position_t15_post_touch":null,"session_open_t1":null,"session_high_t1":null,"session_low_t1":null,"session_close_t1":null,"session_open_t5":null,"session_high_t5":null,"session_low_t5":null,"session_close_t5":null,"session_open_t15":null,"session_high_t15":null,"session_low_t15":null,"session_close_t15":null,"session_open_t0":7646.0,"session_high_t0":null,"session_low_t0":null,"session_close_t0":null,"wall_above_price":null,"wall_below_price":null,"contained_close":null,"contained_range":null,"close_pos_in_band":null,"range_over_im":null,"close_move_over_im":null,"breach_side":null}
{"ticker":"SPX","trade_date":"2026-09-18","feature_version":"v0.6.0-openiv","regime_kind_at_classification":"amplification","dominant_bucket_at_classification":"30+ DTE","horizon_sessions":null,"horizon_end_date":null,"outcome_status":"na_regime","reached_touch":null,"reached_close":null,"days_to_reach":null,"max_excursion_in_direction":null,"final_close_distance_from_target":null,"actual_realized_em_pct":null,"active":true,"deactivated_at":null,"deactivated_reason":null,"backfill_run_id":"12422b1a-fd24-457a-8139-38d0a1843547","computed_at":"2026-09-18T13:40:50.457703","position_t1_post_touch":null,"position_t5_post_touch":null,"position_t15_post_touch":null,"session_open_t1":null,"session_high_t1":null,"session_low_t1":null,"session_close_t1":null,"session_open_t5":null,"session_high_t5":null,"session_low_t5":null,"session_close_t5":null,"session_open_t15":null,"session_high_t15":null,"session_low_t15":null,"session_close_t15":null,"session_open_t0":null,"session_high_t0":null,"session_low_t0":null,"session_close_t0":null,"wall_above_price":null,"wall_below_price":null,"contained_close":null,"contained_range":null,"close_pos_in_band":null,"range_over_im":null,"close_move_over_im":null,"breach_side":null}
```

### Dry-run planned diff (summary)

| trade_date | status | session_open_t0 old → new | session_close_t0 old → new | other changed columns |
| --- | --- | --- | --- | --- |
| 2026-09-08 | computed (unchanged); touch/close F/F unchanged | 7712.25 → 7712.25 (own session is ES U26, legit) | 7681.25 → 7681.25 | max_excursion 5.5 → 7.5; final_close_distance −202.02 → −134.27; realized_em_pct 2.128 → 2.160; walls/containment NULL → filled |
| 2026-09-14 | pending_history (unchanged) | 7606.50 → 7673.50 (+67.00) | 7624.25 → 7692.00 | high/low; range_over_im; close_move_over_im; walls/containment filled |
| 2026-09-15 | na_regime (unchanged) | 7616.75 → 7684.50 (+67.75) | NULL → 7659.75 | high/low, walls, containment, IM ratios filled |
| 2026-09-16 | na_data (unchanged — no positive wall, Q3) | 7609.25 → 7676.50 (+67.25) | NULL → 7623.00 | high/low, wall_below, IM ratios filled |
| 2026-09-17 | na_regime (unchanged) | 7646.00 → 7713.50 (+67.50) | NULL → 7705.25 | high/low, wall_below, IM ratios filled |
| 2026-09-18 | na_regime (unchanged) | NULL → 7710.00 | NULL → 7719.25 | high/low, walls, containment, IM ratios filled |

No `outcome_status`, `reached_touch` or `reached_close` changes. 09-08's `final_close_distance_from_target` moving +67.75 with an unchanged own session is the Q2 seam artifact in miniature (its final horizon session, 09-14, is now ES Z26).
