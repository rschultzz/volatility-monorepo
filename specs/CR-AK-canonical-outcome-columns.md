# CR-AK — Canonical outcome column backfill (session OHLC + post-touch)

> Authority: vault session note `Dash/sessions/2026-09-05 - CR-AK — Canonical Outcome Column Backfill.md`
> Branch: `feat/CR-AK-canonical-outcome-columns` (off `origin/main` 7a63825)
> Scope: `--feature-version` arg on two backfill scripts (default canonical) + two data runs. No other code changes.
> Mode: unattended single-swing run; halts at the first STOP gate that misses.

## Problem

`[Certain]` Under the canonical feature version `v0.6.0-openiv`, all 804 active `bt_daily_outcomes` rows have NULL in every session-OHLC column (`session_{open,high,low,close}_t{1,5,15}`) and every post-touch position column (`position_t{1,5,15}_post_touch`). Only `session_open_t0` (795/804), `reached_touch`, `days_to_reach`, `max_excursion_in_direction`, `final_close_distance_from_target` are populated.

Under the retired `v0.5.0-rebuilt` those columns are filled (697 OHLC, 360 post-touch). CR-G (`cr_g_backfill_session_ohlc.py`) and CR-I (`cr_i_backfill_post_touch_positions.py`) were run against the old version in May; CR-037 created fresh outcome rows under the new version on 2026-06-07 and the two null-fill backfills were never re-run. Both scripts hardcode `FEATURE_VERSION = "v0.5.0-rebuilt"`, have no argparse, and do not read `CANONICAL_FEATURE_VERSION`. The `cr_g` script is also untracked in git.

**Live consequence.** `TodaySetup/routes.py` and `Proposals` call `compute_structural_probability` at canonical, which reads the post-touch and OHLC columns from `bt_daily_outcomes_active` at that version. `aggregate_post_touch_distribution` treats None as missing → the post-touch block has no fractions → `apply_direction_qualification` either badges every magnet-day proposal "low-confidence — post-touch sample insufficient" or passes through unfiltered. **The CR-I direction gate and the CR-G session-OHLC edge inputs have been inert in the live Dash since 2026-06-07.**

**Research consequence.** CR-AH Step 4 recorded the engine hypothesis (decision #13, post-touch pattern split) as UNTESTABLE — 0/216 `pattern_label`s — and attributed it to ~37% t15 coverage. The true coverage at canonical was 0%. The hypothesis was never tested. [[engine-qualification-untested-needs-corpus]] is mis-diagnosed.

## Why recompute, not copy

Compared 743 dates present under both versions: 21 rows differ. 12 are staleness (the pre-PR#39 sweep cron only promoted `v0.5.0` rows; redeployed 2026-09-05, should self-resolve). 1 is 2023-05-23 (see below). **8 are genuine IV-basis differences** — the open-straddle σ changes the 0.25σ touch tolerance, so `reached_touch` flips (2023-06-13, 2023-09-15, 2023-12-15) or `days_to_reach` shifts by 1–2 sessions (2023-06-16, 2023-11-15, 2024-03-13, 2024-07-15, 2026-04-28). Post-touch positions at the old version are therefore the wrong answer for those days at canonical. Session OHLC is version-independent in principle, but recomputing both through the same scripts keeps one provenance (`backfill_run_id`) per column set.

## The 2023-05-23 `na_data` flip — explained, accepted

`[Certain]` Two canonical feature rows carry `implied_move_1d = 0.0` (not NULL): **2023-05-23** (magnet-above → outcome `na_data`) and **2024-04-26** (amplification → `na_regime`, outcome unaffected, σ-features wrong). Cause: `_OPEN_STRADDLE_SQL` guards `atmiv IS NOT NULL` but not `atmiv > 0`; on both dates the pinned 06:33 PT snapshot carries `atmiv = 0`. The first snapshot with `atmiv > 0` is 06:34 on 2024-04-26 but **09:33** on 2023-05-23 — three hours after the open, not an open straddle by any definition.

**Decision:** accept 2023-05-23 as `na_data` for CR-AK. The `atmiv > 0` guard lives in `packages/shared/day_features.py`, which is a cron import → changing it means a cron redeploy → out of scope for an overnight run. Filed as a follow-on below. Cost: one magnet-above date out of 387.

## Locked decisions

| # | Decision | Value |
|---|---|---|
| 1 | Recompute vs copy | **Recompute** via the CR-G / CR-I scripts pointed at canonical. No cross-version column copy. |
| 2 | Version selection | Both scripts gain `--feature-version` (argparse), **default = `CANONICAL_FEATURE_VERSION`** imported from `packages.shared.canonical_version`. The hardcoded `FEATURE_VERSION` constant is removed; every query (targets, `_load_implied_moves`, smoke) uses the resolved value. Version logged at start. |
| 3 | WHERE clauses | Unchanged. Both scripts remain null-fill idempotent (`session_open_t1 IS NULL` / `position_t1_post_touch IS NULL`). |
| 4 | Order | CR-G (OHLC, 773 targets) first, then CR-I (positions, 387 targets). Independent, but OHLC is the larger and simpler run — if it fails, CR-I doesn't run. |
| 5 | Provenance | Commit the untracked `cr_g_backfill_session_ohlc.py` **as-is** before modifying it, so the diff of the version-arg change is reviewable against the file that actually ran in May. |
| 6 | 2023-05-23 | Accepted as `na_data`. `atmiv > 0` guard deferred (touches cron import). |
| 7 | Not in scope | `na_regime` outcome design (37/60 summer dates produce no outcome); holiday mis-stamps (06-19, 07-03, 09-07); orphans 06-22 / 07-06; `bt_backfill_runs` fidelity; CR-AH Step 4 re-run (that is a separate CR — CR-AK only makes it *possible*). |
| 8 | Merge | PR opened, **not merged** overnight. Chat reviews diff + run records in the morning; user smoke on `/today-setup` badges before merge. |

## Expected values (gates)

Measured from the live DB 2026-09-05 03:50 UTC via read-only connector. No crons run Sat/Sun; Monday 09-07 is Labor Day (EOD cron will run and mis-stamp, but does not touch outcome rows). These should be exact at run time.

| Gate | Expected | Tolerance | On miss |
|---|---|---|---|
| G0.1 — CR-G target count (`computed`+`na_regime`, `session_open_t1 IS NULL`, canonical) | **773** | exact | STOP |
| G0.2 — CR-I target count (`reached_touch = TRUE`, `position_t1_post_touch IS NULL`, canonical) | **387** | exact | STOP |
| G0.3 — canonical feature rows with `implied_move_1d <= 0` | **2** (2023-05-23, 2024-04-26) | exact | STOP |
| G0.4 — both scripts import cleanly under the chosen interpreter (`python -c "import ..."` or `--help`) | clean | — | STOP |
| G1.1 — CR-G `n_failed` | 0 | exact | STOP before CR-I |
| G1.2 — CR-G `n_skipped` (no bars) | 0 | ≤ 2 | note, continue |
| G1.3 — CR-G `high_lt_low_violations_t1` | 0 | exact | STOP |
| G1.4 — CR-G `rows_t1_populated_t15_null` (corpus-end: 15th session after trade_date > 2026-09-04) | ~15 | 10–25 | note, continue |
| G1.5 — CR-G rows updated | ≥ 770 | — | STOP if < 760 |
| G2.1 — CR-I `n_failed` | 0 | exact | STOP |
| G2.2 — CR-I `out_of_range` | 0 | exact | STOP |
| G2.3 — CR-I `skip_reasons` | `{}` or only `no_bars` for touch dates whose T+1 is past 2026-09-04 | any `no_implied_move` or `no_drift_target` → STOP | |
| G2.4 — CR-I `remaining_null_t1` | small (touches in late Aug/Sep) | ≤ 15 | STOP if larger |
| G3 — live-path check (Step 4) | `post_touch.filter_mode` no longer `insufficient` for at least 3 of 5 probe dates; `pattern_label` non-null for ≥ 1 | — | note, continue (this is diagnostic, not a gate) |

## Smoke tests (Step 5, all read-only SQL)

1. `SELECT count(*) FROM bt_daily_outcomes WHERE feature_version = canonical AND active AND outcome_status IN ('computed','na_regime') AND session_open_t1 IS NOT NULL` → ≥ 770.
2. Same with `session_close_t15 IS NOT NULL` → ≈ 758 (773 − corpus-end NULLs).
3. `SELECT position_t1_post_touch, count(*) ... WHERE reached_touch GROUP BY 1` → all values in {−1, 0, 1, NULL}, NULL count = G2.4.
4. Cross-version sanity on 5 dates where touch outcomes agree between versions (pick from the 722 agreeing dates): `session_close_t5` identical between `v0.5.0-rebuilt` and `v0.6.0-openiv` (OHLC is version-independent). Post-touch positions should also agree on those 5 unless tolerance differences flip a borderline — report, don't gate.
5. `v0.5.0-rebuilt` rows untouched: 697 OHLC / 360 positions, unchanged.
6. Two `bt_backfill_runs` rows (CR-G, CR-I) with `status = 'completed'`, `smoke_test_results` populated.


## Step 0 findings (read-only, 2026-09-05 04:23 UTC)

Interpreter: `apps/web/.venv/bin/python` (native arm64; pandas 2.3.3, psycopg 3.2.13, dotenv all import cleanly — no Rosetta fallback needed).
Branch: `feat/CR-AK-canonical-outcome-columns` off `origin/main` 7a63825. Backfill role check: `BACKFILL_DATABASE_URL` connects as `dash_backfill_writer`.

| Gate | Expected | Actual | Result |
|---|---|---|---|
| G0.1 CR-G targets (canonical, computed+na_regime, `session_open_t1 IS NULL`) | 773 | **773** | PASS |
| G0.2 CR-I targets (canonical, `reached_touch`, `position_t1_post_touch IS NULL`) | 387 | **387** | PASS |
| G0.3 canonical feature rows with `implied_move_1d <= 0` | 2023-05-23, 2024-04-26 | **2023-05-23, 2024-04-26** | PASS |
| G0.4 both scripts import under the chosen interpreter | clean | **clean** (both report `FEATURE_VERSION = v0.5.0-rebuilt` pre-change) | PASS |

Context counts at canonical (active, SPX): computed 463 / na_data 13 / na_regime 310 / pending_history 18 = 804.
`v0.5.0-rebuilt` baseline for smoke test 5: 697 rows with OHLC, 360 with positions.

### Step 4 baseline — `compute_structural_probability` at canonical (k=200, exclude_date=trade_date, regime inferred from feature flags)

Probe script: scratchpad `probe_sp.py`; JSON saved as `step0_baseline.json`. Badges are what `apply_direction_qualification` returns for a credit + debit magnet-spread pair at DTE 2 / 7 / 15 (t1 / t5 / t15 bands).

| Date | Regime (canonical flags) | k / k_out | touch_rate | filter_mode | same_bucket_n / total_touchers | denominators t1/t5/t15 | pattern_label | Badge (all DTE bands) |
|---|---|---|---|---|---|---|---|---|
| 2026-09-03 | magnet-above | 20 / 18 | 0.611 | pooled-fallback | 5 / 11 | 0 / 0 / 0 | None | mixed pattern — no clear direction |
| 2026-08-27 | magnet-above | 38 / 31 | 0.903 | strict | 9 / 28 | 0 / 0 / 0 | None | mixed pattern — no clear direction |
| 2026-07-30 | amplification | 27 / 0 | None | insufficient | 0 / 0 | 0 / 0 / 0 | None | low-confidence — post-touch sample insufficient |
| 2026-06-24 | amplification | 28 / 0 | None | insufficient | 0 / 0 | 0 / 0 / 0 | None | low-confidence — post-touch sample insufficient |
| 2026-04-28 | magnetic-pin | 0 / 0 | None | insufficient | 0 / 0 | 0 / 0 / 0 | None | low-confidence — post-touch sample insufficient |

Observations (recorded, not gated):

- The two true magnet dates confirm the live-path diagnosis: `filter_mode` is `strict` / `pooled-fallback` (bucket filter works, touchers exist) but every denominator is 0 and `pattern_label` is None because `position_t*_post_touch` is NULL on all canonical rows. `apply_direction_qualification` falls through to "mixed pattern — no clear direction" on both proposals — neither credit nor debit ever qualifies.
- The vault note labels all five probe dates "magnet". At canonical, 2026-07-30 and 2026-06-24 are amplification (`na_regime` outcome) and 2026-04-28 is magnetic-pin (computed, touched at day 0). `bt_audit_flags` rows for the five dates: (0,) — no promoted regime override changes this. Only 09-03 and 08-27 (both `pending_history` at canonical, so excluded from their own corpus anyway) exercise the direction gate; G3 "≥3 of 5 not insufficient" is therefore structurally capped at 2 of 5 unless the amplification/pin dates gain touchers. Recorded as a spec delta; G3 is diagnostic, run continues.

## Step 1 — CR-G run (session OHLC at canonical)

Command: `apps/web/.venv/bin/python -u scripts/cr_g_backfill_session_ohlc.py` (no args → resolved `v0.6.0-openiv`, logged on the line after role verification). Log: `scripts/logs/cr_ak_g_20260904_212611.log`. Wall time 04:26:12 → 05:19:23 UTC (53 min, ~4 s/row — each row fetches a 90-calendar-day 1-minute bar window).

`bt_backfill_runs` row (latest CR-G):

```
(UUID('d605b950-b12b-4f3e-8f69-9579b6621c8d'), 'CR-G', 'completed', datetime.datetime(2026, 9, 5, 4, 26, 12, 252884), datetime.datetime(2026, 9, 5, 5, 19, 23, 973207), 750, "773/773 rows updated; 11 T+15 NULLs (corpus-end); by_status={'computed': 460, 'na_regime': 307}; high<low violations=0", {'by_outcome_status': {'computed': 460, 'na_regime': 307}, 'rows_with_t1_ohlc': 767, 'high_lt_low_violations_t1': 0, 'rows_t1_populated_t15_null': 11})
```

| Gate | Expected | Actual | Result |
|---|---|---|---|
| G1.1 `n_failed` | 0 | **0** | PASS |
| G1.2 `n_skipped` (no bars in window) | 0, ≤ 2 | **0** | PASS |
| G1.3 `high_lt_low_violations_t1` | 0 | **0** | PASS |
| G1.4 `rows_t1_populated_t15_null` (corpus-end) | ~15, 10–25 | **11** | PASS (note) |
| G1.5 rows updated | ≥ 770, STOP < 760 | **773 / 773** updated; **767** carry non-NULL `session_open_t1` | PASS (see delta) |

Smoke dict: `rows_with_t1_ohlc = 767`, `rows_t1_populated_t15_null = 11`, `by_outcome_status = {computed: 460, na_regime: 307}`, `high_lt_low_violations_t1 = 0`. Script tally: 17 T+15 NULLs = 11 corpus-end + 6 below.

### Delta — 6 quarterly roll Fridays have no RTH bars on the trade date

`ironbeam_es_1m_bars` ends at 13:29 UTC on **2023-09-15, 2024-03-15, 2024-06-21, 2024-09-20, 2025-03-21, 2025-09-19** (third-Friday ES roll dates; the Databento-era continuous series has no RTH bars for the expiring session, and the 2025-09-19 Ironbeam-era gap is the same shape). For those rows the script finds `trade_date` absent from the session list and writes all twelve OHLC columns as NULL while still stamping `backfill_run_id` and counting the row as updated. The same six dates are NULL under `v0.5.0-rebuilt` (its remaining 27 NULL-t1 rows are 2026-03 → 06 rows created after the May run). Because the null-fill `WHERE session_open_t1 IS NULL` still matches them, they stay re-targetable and G0.1 will read 6, not 0, on any future run. This is a pre-existing bar-coverage gap, not a CR-AK regression; smoke test 1 will read 767 (< the spec's ≥ 770) for this reason. Not gated; recorded for the roll-date open question.

G1.1 / G1.3 / G1.5 STOP conditions not met → proceeding to Step 2 (CR-I).

## Step 2 — CR-I run (post-touch positions at canonical)

Command: `apps/web/.venv/bin/python -u scripts/cr_i_backfill_post_touch_positions.py` (no args → resolved `v0.6.0-openiv`). Log: `scripts/logs/cr_ak_i_20260904_221959.log` (untracked, like all `scripts/logs/`). Wall time 05:20:02 → 05:45:53 UTC (26 min). Landscape 811 rows, implied moves 804 rows (canonical), targets 387.

`bt_backfill_runs` row (latest CR-I):

```
(UUID('4e55a538-bb9b-42d0-853f-6383430b7c39'), 'CR-I', 'completed', datetime.datetime(2026, 9, 5, 5, 20, 1, 593544), datetime.datetime(2026, 9, 5, 5, 45, 53, 910843), 350, 'Pass 1 complete. updated=387, skipped=0, failed=0. remaining_null_t1=0 (expected: rows where T+1 bar unavailable). out_of_range=0 (expect 0). Status: clean.', {'n_failed': 0, 'n_skipped': 0, 'n_updated': 387, 'out_of_range': 0, 'skip_reasons': {}, 'value_dist_t1': {'0': 90, '1': 196, '-1': 101}, 'value_dist_t15': {'0': 16, '1': 279, '-1': 91, 'None': 1}, 'remaining_null_t1': 0})
```

| Gate | Expected | Actual | Result |
|---|---|---|---|
| G2.1 `n_failed` | 0 | **0** | PASS |
| G2.2 `out_of_range` | 0 | **0** | PASS |
| G2.3 `skip_reasons` | `{}` or only `no_bars`; any `no_implied_move` / `no_drift_target` → STOP | **`{}`** | PASS |
| G2.4 `remaining_null_t1` | ≤ 15 | **0** | PASS |

Notes: `rows_inserted` on both run rows reflects the last 50-row heartbeat (750 for CR-G, 350 for CR-I), not the final tally — the known `bt_backfill_runs` fidelity issue from CR-AJ, unchanged here. The six roll-Friday dates from Step 1 did not appear as skips: `_fetch_rth_daily_bars` returns the surrounding sessions, and `classify_post_touch_positions` indexes from `days_to_reach`, so positions were classified for any of them that touched (checked in Step 3 smoke 4 detail).

No G2 STOP condition met → proceeding to Step 3.

## Step 3 — smoke tests (read-only, 2026-09-05 05:48 UTC)

Script: scratchpad `smoke_step3.py`. Smoke 4 dates picked by rank (35 / 105 / 175 / 245 / 315) from the 351 dates where touch outcome and `days_to_reach` agree between versions and the old version has both OHLC and t15 position.

| # | Test | Expected | Actual | Result |
|---|---|---|---|---|
| 1 | canonical computed+na_regime with `session_open_t1` | ≥ 770 | **767** | FAIL |
| 2 | same with `session_close_t15` | ≈ 758 (773 − corpus-end) | **756** (t15 NULL = 17) | PASS |
| 3 | `position_t1_post_touch` over touched rows | values ⊂ {−1,0,1,NULL}, NULL = G2.4 | **{'-1': 101, '0': 90, '1': 196}**; t15: {'-1': 91, '0': 16, '1': 279, 'None': 1} | PASS |
| 4 | cross-version on 5 agreeing dates: `session_close_t5` identical | 5/5 | **5/5** close equal; positions equal on 5/5 | PASS |
| 5 | `v0.5.0-rebuilt` untouched (OHLC / positions) | 697 / 360 | **697 / 360** | PASS |
| 6 | two `bt_backfill_runs` rows completed with smoke | CR-G + CR-I completed | **[('CR-G', 'd605b950', 'completed', True), ('CR-I', '4e55a538', 'completed', True)]** | PASS |
| 7 | (extra) canonical rows carrying each new run_id | 773 / 387−skips | [('4e55a538', 387), ('d605b950', 386)] | info |

Smoke 4 detail (old vs canonical):
| date | close_t5 old | close_t5 canon | p1 old/new | p5 old/new | p15 old/new |
|---|---|---|---|---|---|
| 2023-07-17 | 4584.25 | 4584.25 | 1/1 | 1/1 | 0/0 |
| 2024-02-08 | 5044.75 | 5044.75 | 1/1 | 1/1 | 1/1 |
| 2024-07-23 | 5472.75 | 5472.75 | 1/1 | 1/1 | 1/1 |
| 2025-05-16 | 5817.0 | 5817.0 | 0/0 | 1/1 | 1/1 |
| 2025-12-01 | 6843.0 | 6843.0 | -1/-1 | -1/-1 | 1/1 |


Smoke 1 reads **767**, three below the spec's ≥ 770. The shortfall is exactly the six roll-Friday rows from the Step 1 delta (no RTH bars on the trade date → all-NULL OHLC); 767 + 6 = 773. Not a STOP condition (G1.5 is measured on rows updated = 773, and 767 is above the 760 floor either way). Smoke 2's 17 t15 NULLs = 11 corpus-end + 6 roll Fridays.

Provenance note (row 7): `backfill_run_id` is a single column, so CR-I's UPDATE overwrote CR-G's id on the 387 touched rows. Final split: 386 rows carry the CR-G run id, 387 the CR-I run id — every one of the 773 targets carries one of the two. The "one `backfill_run_id` per column set" framing in decision #1 is therefore only recoverable from `bt_backfill_runs` timestamps, not per row.

Roll-Friday spot check (old vs canonical positions on the six dates): 2024-09-20 and 2025-09-19 touched under both versions and classify identically (d=6 → 1/1/1; d=3 → 1/1/1). 2023-09-15 is one of the eight IV-basis flips from the vault note (old: no touch; canonical: touch at day 0 → −1/−1/−1). 2024-03-15, 2024-06-21, 2025-03-21 are `na_regime` under both.

## Step 4 — live-path before / after

Same probe as Step 0 (`compute_structural_probability` at canonical, k=200, `exclude_date` = the date, regime inferred from canonical feature flags; badge = what `apply_direction_qualification` returns for a credit + debit magnet-spread pair). JSON: scratchpad `step0_baseline.json` / `step4_after.json`.

| Date | Regime | filter_mode | denominators t1/t5/t15 before → after | pattern_label before → after | Badge before → after |
|---|---|---|---|---|---|
| 2026-09-03 | magnet-above | pooled-fallback (unchanged) | 0/0/0 → **11/11/11** | None → **stepping-stone** | dte2, dte15: mixed pattern (both kept) → unchanged; **dte7 (t5): mixed pattern → "debit-to-target supported", credit proposal dropped** |
| 2026-08-27 | magnet-above | strict (unchanged) | 0/0/0 → **9/9/9** | None → **mixed** | mixed pattern — no clear direction on both, all bands (unchanged, now on real data) |
| 2026-07-30 | amplification | insufficient | 0/0/0 → 0/0/0 | None → None | low-confidence — post-touch sample insufficient (unchanged) |
| 2026-06-24 | amplification | insufficient | 0/0/0 → 0/0/0 | None → None | low-confidence — post-touch sample insufficient (unchanged) |
| 2026-04-28 | magnetic-pin | insufficient (k=0) | 0/0/0 → 0/0/0 | None → None | low-confidence — post-touch sample insufficient (unchanged) |

After-run fractions for the two magnet dates:

- 2026-09-03 (11 touchers): t1 below/at/above 0.09/0.27/0.64; t5 0.18/0.00/0.82; t15 0.27/0.09/0.64 → stepping-stone. Wilson lower bound on `above` at t5 clears the 0.40 floor, so only the t5 band (DTE 4–9) qualifies debit; t1 and t15 lower bounds fall short → mixed.
- 2026-08-27 (9 same-bucket touchers, strict): t1 0.11/0.44/0.44; t5 0.11/0.22/0.67; t15 0.11/0.22/0.67 → mixed.

G3 (diagnostic): "`filter_mode` no longer `insufficient` for ≥ 3 of 5" — **not met, 2 of 5**, and structurally could not be: three of the five probe dates are not magnet at canonical (Step 0 delta), so they never had touchers to lose. "`pattern_label` non-null for ≥ 1" — **met, 2 of 2** magnet dates. The direction gate is live again: before CR-AK every magnet-day proposal pair fell through to "mixed pattern"; now the badge is data-driven, and on 09-03 it actually selects a direction.
