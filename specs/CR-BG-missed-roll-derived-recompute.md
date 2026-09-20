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
