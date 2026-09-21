# CR-BH — Shadow-recompute `bt_daily_outcomes` on SPX cash; diff against `v0.6.0-openiv`

> Branch: `feat/CR-BH-spx-cash-shadow-outcomes` (off `origin/main` a11e958)
> `data_safety_class: write_backfill`
> Source: vault open-question `outcome-target-vs-es-roll-seam-and-basis-bias` (incl. its "Evidence — quantified from chat" section, taken as given — not re-derived) + kickoff prompt (2026-09-20).
> INSERT only. Do not promote the canonical version. Do not change `cr_b_backfill_outcomes`, the CR-AA sweep, the Proposals card, or any live cron. `vol_risk_premium` is out of scope.

## Why

`compute_outcome` compares an SPX forward-space wall target, raw, to front-contract ES bars. The measured ES−SPX basis is a sawtooth: 33–49 pts in the first month after each roll, 15–28 mid-quarter, mean ≈ 28. The 5-session `magnet-above` touch rate runs 44 % / 52 % / 79 % across low / mid / high basis buckets (p ≈ 0.003). All directional `computed` rows are `magnet-above`, so the bias inflates touch one-sidedly. SPX cash removes basis and roll seams entirely.

## Verified from chat (read-only SQL, 2026-09-20)

- `orats_monies_minute` has 0DTE SPX rows on all 815 active outcome dates: 850 days, 816 with ≥ 380 RTH minutes (median 391, 06:30–13:00 PT), 34 partial, 0 sparse.
- SPX cash per minute = `avg(spot_price) WHERE expir_date = trade_date AND ticker IN ('SPX','SPXW')`, grouped by `date_trunc('minute', snapshot_pt)`. This predicate matches partial index `idx_omm_front_minute_pt_ticker`. Use `spot_price`, not `stock_price`.
- Known bad prints exist (basis of 710.8 in 2023-09, 548.4 in 2023-11).
- `bt_daily_outcomes` PK = `(ticker, trade_date, feature_version)`. Versions present: `v0.6.0-openiv` (816 rows), `v0.5.0-rebuilt` (743).

## Step-0 diagnosis gate (all answered in this file before any write)

1. **Version leakage.** Grep every reader of `bt_daily_outcomes` (`apps/`, `scripts/`, `packages/`, SQL views). Does each one filter on `feature_version` or the canonical version constant? If ANY reader does not, a new version would pollute it: **STOP and ask** (the alternative is a separate shadow table, which needs owner DDL).
2. Does anything join `bt_daily_outcomes` to `bt_daily_features` on `feature_version` such that a new outcomes version needs matching feature rows? State how the shadow version avoids breaking that join.
3. Document the exact session window the current code uses for `session_*_t0` OHLC. Stored closes differ from a 06:30–13:00 PT RTH close by 0.25 to 7 pts and the 09-18 stored high exceeds the RTH high, so the window is not plain RTH. SPX cash ends 13:00 PT: state the window to be used and the mismatch it implies.
4. List every place ES price enters an outcome row: `compute_outcome` (touch / close / excursion / realized EM), `compute_session_containment` (`nearest_walls(walls, ES open)`), t1/t5/t15 session and post-touch columns. All of them switch to SPX cash in the shadow version.
5. Minute-sampled SPX understates true high/low. Quantify with ES as proxy: for 50 random days, compare range from ES 1m closes only vs true ES 1m high/low. Report the median understatement in pts (expect small vs a 28-pt basis).
6. Bad-print filter: propose a rule (`spot_price > 0` plus a robust neighbor-median check) and list every minute it drops. List the 34 partial days and classify each: early close vs data outage.

## Implementation

- Build SPX-cash session OHLC per trade_date in code (no new table unless Step 0 forces it).
- Recompute ALL rows (`computed`, `pending_*`, `na_*`) into a new `feature_version`, proposed `v0.6.0-openiv-spxcash`: same targets, same regime labels, same horizons, same tolerance rules. Only the price series changes.
- INSERT only, under the unattended backfill protocol (`get_backfill_db_conn`, `assert_role_or_die`, `backfill_run`). No UPDATE or DELETE of existing rows.
- Do NOT promote the canonical version. `v0.6.0-openiv` keeps running untouched.

## Deliverable — the diff (session note + append to the open-question)

- Per-day basis: `es_minutes.close` at 07:00 PT minus SPX cash at the same minute; buckets < 20, 20–35, 35+; drop basis outside −50..120.
- Old vs new `reached_touch` and `reached_close`: confusion matrix overall and by regime × horizon × basis bucket. List every trade_date whose label flips.
- Pooled touch/close rates old vs new, by regime × horizon.
- Confound test: in the NEW version, does 5-session `magnet-above` touch still vary by basis bucket? If the gradient disappears, it was the basis. If it persists, there is a real post-OPEX effect worth its own open-question.
- Status transitions (`computed` ↔ `na_*`, pending) caused by the switch.
- Leave the open-question `status: in-progress`. Promotion to canonical is a separate decision (ADR) made from chat after reading the diff.

## Implementation order

1. Step 0 — diagnosis findings appended to this file (gate).
2. Step 1 — SPX-cash session builder + shadow recompute script (`scripts/cr_bh_shadow_spxcash_outcomes.py`), dry-run output.
3. Step 2 — run the INSERT under the backfill protocol; smoke appended.
4. Step 3 — diff script + results.
5. Smoke + wrap.

## Step 0 — diagnosis findings (2026-09-20, read-only SQL under `dash_backfill_writer` with `default_transaction_read_only=on`)

Scripts: `scripts/cr_bh_step0_diagnosis.py` (Q3/Q5/Q6), `scripts/cr_bh_spx_cash.py` (series builder, draft filter). **Gate result: Q1–Q5 pass; Q6 surfaced Q7, which blocks the write — see Q7.**

### Q1 — version leakage. **None. Every live reader filters on `feature_version`; a new version pollutes nothing.**

- Live readers (all `WHERE … feature_version = %s` with `CANONICAL_FEATURE_VERSION` or a caller-passed version): `apps/web/modules/SetupV2/routes.py:133`, `apps/web/modules/Proposals/routes.py:141`, `packages/shared/probability.py:588`, `scripts/cron_daily_leg_capture.py:256`, `scripts/cr_aa_sweep_pending_outcomes.py` (all 7 statements, incl. both UPDATEs — the first UPDATE's WHERE is 18 lines below the statement head), `scripts/cr_b_backfill_outcomes.py` (NOT EXISTS is correlated on `o.feature_version = f.feature_version`, so shadow rows do not suppress canonical inserts). `packages/shared/edge_today.py` only mentions the view in comments.
- One-off scripts (`cr_ah_*`, `cr_ai_*`, `cr_am_*`, `cr_aq_backfill_containment`, `cr_as_*`, `cr_bg_*`, `cr_g_*backfill*`, `cr_i_*`): all filter on `feature_version` (several additionally inner-join `bt_daily_features` on it).
- Unfiltered statements — all in completed one-off migration/diagnostic scripts, none live: `cr_aq_run_migration.py:68`, `cr_g_run_ddl_step0a.py:59/134`, `cr_g_backfill_session_ohlc.py:312` (smoke counts), `cr_bd_repair_orphans.py:61–62` (per-date row counts for six named dates). They already see two versions today.
- DB side: the only dependent view is `bt_daily_outcomes_active` (`SELECT * … WHERE active`) — no version filter in the view, but every reader of it filters. No `LIKE` / prefix / `max(feature_version)` matching anywhere, so the suffix `-spxcash` cannot be picked up by a `v0.6.0-openiv%` pattern. Two versions (`v0.5.0-rebuilt` 743, `v0.6.0-openiv` 816) already coexist without incident.

### Q2 — `bt_daily_features` join. **Several readers inner-join features on `f.feature_version = o.feature_version`; the shadow version needs no feature rows because nothing reads it through that join.**

The sweep (`_PENDING_ROWS_SQL`, `_CONTAINMENT_TARGETS_SQL`), `cr_as_capture`, `cr_am`, `cr_ah` join on version but also filter `o.feature_version = canonical`, so shadow rows are never candidates. The shadow script reads regime / `feature_vector` from `bt_daily_features` at `v0.6.0-openiv` and writes outcomes under `v0.6.0-openiv-spxcash`; the diff joins old↔new on `(ticker, trade_date)` with each version named explicitly. No `bt_daily_features` rows are written. Consequence: the sweep never promotes shadow `pending_history` rows — the shadow version is a point-in-time snapshot (INSERT-only, `ON CONFLICT DO NOTHING`).

### Q3 — session window. **ES: bars whose bar-open timestamp is 06:30:00–13:00:00 PT inclusive = 391 bars, i.e. 06:30:00 → 13:00:59. The 13:00 bar (the minute *after* the cash close) is included.**

- `cr_b_backfill_outcomes._RTH_BARS_SQL` (shared by the sweep and CR-BG): `(datetime AT TIME ZONE 'UTC' AT TIME ZONE 'America/Los_Angeles')::time BETWEEN '06:30:00' AND '13:00:00'`, DST-safe. open = first bar open, close = **close of the 13:00 bar (≈ 13:01 PT)**, high/low over all 391 bars. Verified 2026-09-18: stored high 7719.25 = high of the 13:00 bar (06:30–12:59 high is 7718.25); stored close 7719.25 = 13:00 bar close (12:59 bar close 7712.25). That is the 0.25–7 pt close difference and the high > RTH high.
- Side finding (not this CR): `cr_g_backfill_session_ohlc.py` (t1/t5/t15 OHLC) and `cr_i_backfill_post_touch_positions.py` use a **fixed 13:30 ≤ t < 20:00 UTC** window — 06:30–12:59 PT in PDT but **05:30–11:59 PT in PST months**, and they exclude the 13:00 bar. The stored t1/t5/t15 columns are therefore on a different window from t0, and wrong by an hour in winter.
- Shadow window: snapshots with minute 06:30–13:00 PT inclusive (last print = 13:00:00, the cash close). One window for t0, horizon bars, t1/t5/t15 and post-touch closes. Mismatch vs ES: SPX ends 13:00:00 where ES ends 13:00:59, so the ES series sees one extra minute of post-close futures trading (incl. MOC drift); SPX `|13:00 − 12:59|` is median 1.3, p90 4.1, max 17 pts (closing-auction print). The opening print is a second mismatch: the 06:30 SPX print is largely the prior close (constituents not yet open); `|06:33 − 06:30|` median 3.3, p90 12.6, max 33 pts. Project precedent (CR-AB, memory `project_es_spx_basis`) is 06:33. **Proposed: open = first print ≥ 06:33, high/low/close over 06:33–13:00** (a stale prior-close print on a gap-down day would otherwise become the session high and a spurious magnet-above touch).

### Q4 — every place ES price enters an outcome row

| Column(s) | Code | ES input |
| --- | --- | --- |
| `reached_touch`, `days_to_reach` | `outcomes.compute_outcome` §7–8 | horizon daily `high` / `low` vs fixed SPX-space `drift_target` |
| `reached_close`, `final_close_distance_from_target` | §8 | last horizon `close` |
| `max_excursion_in_direction` | §7 | horizon `high`/`low` vs first `open` |
| `actual_realized_em_pct` | §8 | horizon high − low |
| `outcome_status` (`pending_history` / `na_data`), `horizon_end_date` | §4–6 | bar *availability* (session calendar) |
| `session_open_t0` | `outcomes_runner.compute_outcome_for_date` | t0 `open` (also the Proposals card spot and `cron_daily_leg_capture` ES open) |
| `session_high/low/close_t0`, `wall_above/below_price`, `contained_close`, `contained_range`, `close_pos_in_band`, `breach_side`, `range_over_im`, `close_move_over_im` | `compute_session_containment` | t0 OHLC; `nearest_walls(walls, ES open)` |
| `session_{open,high,low,close}_t{1,5,15}` | `cr_g_backfill_session_ohlc.py` | Nth-session OHLC (fixed-UTC window) |
| `position_t{1,5,15}_post_touch` | `cr_i_…` → `probability.classify_post_touch_positions` | session close at `days_to_reach + N` vs target ± 0.25×IM |

Not ES: `drift_target` (`pick_drift_target(walls)`), regime, `dominant_bucket`, `implied_move_1d`, `direction_sanity` (`table_spot`). All ES inputs switch to the SPX series; the shadow script computes the CR-G / CR-I columns itself from the same daily frame (they are NULL-filled only to 2026-09-02 / 2026-08-14 in canonical, so old-vs-new on those columns is compared only where canonical is non-NULL).

### Q5 — minute-sampling understatement (ES proxy, 50 random outcome dates, seed 20260920)

1m closes only vs true 1m high/low, 06:30–13:00 PT: high understated **median 0.75** (mean 1.29, p90 1.85, max 11.5); low understated **median 1.38** (mean 1.82, p90 3.85, max 5.0); range **median 2.25** (p90 5.8, max 13.75). Small against a 28-pt basis; it biases touch slightly *down* (the opposite direction to the basis bias).

### Q6 — bad prints and partial days

- One row per minute everywhere (no SPX/SPXW duplicates), so `avg()` is a no-op.
- Draft rule (`spot_price > 0`; > 12 % from session median; > 0.5 % from a centered 11-print median) drops 327 minutes on 51 days: 231 gross (2023-09-21 06:31–08:08 ×77 prints ≈ 3 680–3 815 vs SPX ≈ 4 350; 2023-11-29 ×150; 2024-04-26 ×4 prints of 22.76–22.78 — a VIX-like value), 1 NULL (2025-11-11 10:48), 95 neighbor. **Rejected as drafted:** 22 of the neighbor drops are real moves confirmed by ES (2025-04-07 07:17–07:35 headline spike, 2025-04-09 10:22–10:45 tariff-pause rally); the session-median guard inverts when most of a day is bad (2023-11-29: it keeps the bad half); and it cannot see frozen runs (Q7). 46 neighbor drops are genuine (mostly stale 06:31–06:33 opening prints 25–87 pts off, plus 2024-08-06 12:04). Threshold sensitivity: 0.15 % → 1 822 drops, 0.3 % → 524, 0.5 % → 327, 1 % → 249.
- Replacement rule (to be finalised with Q7, full dropped-minute list appended then): (1) `> 0`; (2) gross guard vs the centered 5-session median of session medians (robust to a mostly-bad day); (3) frozen runs — ≥ 3 consecutive identical prints, drop all but the first; (4) isolated spike — a run of ≤ 2 prints > 0.3 % beyond *both* the preceding and following prints (time-adjacent, ≤ 3 min) on the same side; trends and multi-minute moves are kept.
- Partial days (< 380 clean minutes): **37** (34 before filtering), 31 of them outcome dates. Early closes (8, 10:00 PT): 2023-07-03, 2023-11-24, 2024-07-03, 2024-11-29, 2024-12-24, 2025-07-03, 2025-11-28, 2025-12-24 (only 2023-07-03 and 2024-07-03 are outcome dates). Outages (29) — late start: 2023-05-23 (09:33), 2023-09-21 (07:08 after filter), 2025-11-26 (09:27), 2026-07-22 (07:35), 2026-08-18 (06:51), 2026-08-19 (07:57), 2026-08-28 (07:05); early end: 2023-05-19 (12:50), 2023-11-29 (10:29 after filter), 2025-05-08 (12:30); intraday gaps (max gap 3–29 min): 2023-08-18, 2023-10-25, 2023-11-09, 2024-02-22, 2024-05-09, 2024-08-08, 2024-10-28, 2024-10-31, 2025-04-07, 2025-04-09, 2025-05-20, 2025-07-15, 2025-09-26, 2025-10-22, 2025-10-29, 2025-11-11, 2025-11-24, 2026-02-26, 2026-08-25. 2026-09-07 (the one inactive row, Labor Day) has no SPX minutes. Late-start / early-end days get a truncated session (wrong open or close); proposal: compute them but flag, and report the diff with and without them.

### Q7 (new, surfaced by Q6) — `spot_price` is not a clean SPX cash series. **BLOCKS the write; needs a decision.**

ES 1m closes used only as an independent *witness* (intraday ES−SPX basis should be flat to a few pts):

1. **2023-05 → ~2023-11: `spot_price` is 15 minutes delayed.** Best cross-correlation lag vs ES is exactly 15 min on every sampled day May–Aug 2023, mixed Sep–Nov, ≤ 1 from 2023-12. The first 15 prints of those sessions are frozen at the prior close (e.g. 2023-07-06 06:30–06:45 = 4445.01, then 4409.00). Data ends 13:00, so the true last 15 minutes are absent. 2024-04 → 2025-02 both columns lag 2–3 min (minor). 132 of 169 days in 2023 have ≥ 15 frozen prints; 2023-10-24 is frozen 06:30–09:15 (165 min).
2. **2026: `spot_price` drifts away from the index on 48 of 179 days** (intraday basis p95−p5 > 15 pts; `stock_price`: 0 days). Example 2026-04-07: ES − `spot_price` goes 47 → 87 → 105 pts through the day while ES − `stock_price` stays ≈ 40; at 13:00 `spot_price` 6551.12 vs `stock_price` 6615.23 (next-day `table_spot` 6599.5). Daily closes built from the two columns differ by > 15 pts on **37 / 179** days in 2026, highs on 36, lows on 29 (2024: 2 / 6 / 8; 2025: 4 / 5 / 6). That is the same size as the basis bias this CR is trying to remove.
3. **`stock_price` is not a drop-in fix:** in 2023 it is unlagged but jitters ± 15 pts against ES minute to minute (2023-08-10), 56 of 169 days with basis spread > 15; on 2026-09-10 it sits ~8 pts under `spot_price` for the last hour and both converge at 13:00 (7600.82) — consistent with `spot_price` being an option-implied spot and `stock_price` the index feed.
4. 2024 and 2025 are clean in both columns (1–2 days each with spread > 15).

The chat evidence (basis at 07:00 from `spot_price`) is not invalidated — the 2026 drift builds through the session and a 15-min lag is unbiased noise at one sample — but session high/low/close from `spot_price` are not trustworthy in 2023-05→11 and on ~27 % of 2026 days. Options: (a) `spot_price` as specified + frozen-run filter + 15-min timestamp shift in the lagged period — leaves the 2026 drift unfixed; (b) `stock_price` — clean 2024–2026, noisy 2023; (c) per-period hybrid: `stock_price` from 2023-12 on, `spot_price` shifted −15 min before, with an ES-witness QA flag per day (flag only — ES never enters a price); (d) restrict the shadow to 2023-12 → present. No write until one is chosen.

## Amendment A1 (Ryan, 2026-09-20, after Step 0) — per-period hybrid series

Q7 decision: option (c), amended. Supersedes "Use `spot_price`, not `stock_price`" in *Verified from chat*.

1. **Segments.** `series_segment = 'spot_shifted_2023'` for trade_date ≤ 2023-11-08; `'stock_price'` for trade_date ≥ **2023-11-09 (cutover)**. The tag is per outcome row (by its trade_date) and lives in the diff output only, not in the table. A 2023 row's horizon may run into `stock_price` sessions; the diff also carries `horizon_crosses_cutover`.
2. **`spot_shifted_2023` sessions.** Path / high / low from `spot_price`, timestamps shifted −15 min **on lagged days** (windows below; unlagged days in the segment use `spot_price` unshifted — both columns are clean there). On lagged days the true 12:46–13:00 is absent, so **session close = median `stock_price` over 12:56–13:00 PT** (entered as the 13:00 print, so it also bounds high/low). Same rule for any t1/t5/t15 or horizon close landing on a lagged day — it is one daily frame. Measured quality of that close: `stock_price`'s ES-basis spread is 11–24 pts on lagged days (2–4 on unlagged days), so expect an error of a few pts; every lagged close is checked by the ES QA flag (item 4).
3. **Lag evidence (measured, not eyeballed).** Per day, lag L ∈ 0..20 min maximising corr(Δ`spot_price`(t), ΔES 1m close(t − L)), 07:15–12:45 PT; `frozen_open` = number of leading session prints identical to the first.

| month | days | L = 15 | L ≤ 1 | other | n/a | median corr at best L | median corr at L = 0 | median `frozen_open` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-05 | 22 | 22 | 0 | 0 | 0 | 0.66 | −0.01 | 16 |
| 2023-06 | 21 | 20 | 0 | 0 | 1 | 0.66 | 0.01 | 16 |
| 2023-07 | 20 | 20 | 0 | 0 | 0 | 0.62 | −0.02 | 16 |
| 2023-08 | 23 | 23 | 0 | 0 | 0 | 0.63 | 0.00 | 16 |
| 2023-09 | 20 | 4 | 15 | 0 | 1 | 0.58 | 0.48 | 2 |
| 2023-10 | 22 | 6 | 16 | 0 | 0 | 0.54 | 0.45 | 2 |
| 2023-11 | 21 | 6 | 14 | 1 | 0 | 0.54 | 0.48 | 2 |
| 2023-12 | 20 | 0 | 20 | 0 | 0 | 0.53 | 0.48 | 2 |
| 2024-01 → 03 | 61 | 0 | 61 | 0 | 0 | 0.53–0.55 | 0.48–0.53 | 2 |

   The lag is all-or-nothing per day and contiguous: **lagged windows = 2023-05-01 → 2023-09-07 and 2023-10-24 → 2023-11-08** (every day in them has L = 15 and `frozen_open` = 16, except 2023-06-16 — quarterly OPEX, correlation undefined, `frozen_open` = 16 → treated as lagged — and 2023-10-24, frozen 165 min). Unlagged: 2023-09-08 → 2023-10-23 and from 2023-11-09 (first unlagged day after the last lagged day = the cutover). Exceptions inside unlagged time: 2023-09-15 (quarterly OPEX, `frozen_open` 42, corr undefined) and 2023-11-29 (bad-print day, L = 4 at corr 0.14) — left unshifted, caught by the filter / QA flag.
4. **ES is a per-day QA flag only, never a price.** Flag when, for the chosen series: intraday basis spread (p95 − p5 after 06:45) > 15 pts, or |close basis − day-median basis| > 10, or |ES-implied high/low − series high/low| > 10. Flagged days are listed in the diff; nothing is corrected from ES.
5. **Basis buckets recomputed with the chosen series:** median over 06:58–07:02 PT of (`es_minutes.close` − series), buckets < 20 / 20–35 / 35+, drop outside −50..120. These supersede both the original `spot_price` table and the `stock_price` 2023-12+ correction in the open-question.
6. **Every diff table is reported twice:** full corpus, and `stock_price` segment only.
7. **Filter (final, replaces the Q6 draft), applied to the chosen column per day:** (i) `> 0`; (ii) gross guard — > 10 % from the centered 5-session median of session medians; (iii) frozen runs — ≥ 3 consecutive identical prints: a run that starts at the session's first print is dropped whole (pre-open stale), otherwise all but its first print are dropped; (iv) isolated spike — a run of ≤ 2 prints more than 0.3 % beyond both the preceding and following kept prints (each within 3 min) on the same side. Every dropped minute is written to `scripts/logs/` and summarised in Step 1.
8. **Window (Q3 proposal adopted):** after any shift, prints 06:33–13:00 PT; open = first print ≥ 06:33.
9. **Scope:** active canonical rows only (815; the inactive 2026-09-07 holiday mis-stamp is not shadowed). Step 1 also recomputes every row on **ES through the same code path as a control**, so flips caused by anything other than the series (landscape drift since the original compute, the holiday-session calendar) are separated from series flips.
10. **Out of scope, new low-priority open-question:** the 2026 `spot_price` drift, naming every repo reader of `spot_price`. Not investigated here.

## Step 1 — series builder, shadow script, dry run (2026-09-20)

Files: `scripts/cr_bh_spx_cash.py` (A1 series; no ES), `scripts/cr_bh_shadow_spxcash_outcomes.py` (`--dry-run` writes nothing), `scripts/cr_bh_diff.py`, `scripts/cr_bh_step0_diagnosis.py` (ported to the final series). Full per-minute drop list: `scripts/logs/cr_bh_dropped_minutes.csv`; daily frame: `scripts/logs/cr_bh_spx_daily.csv` (both git-ignored).

**Filter as built (A1.7, one change):** the spike / edge thresholds are `max(0.3 % / 0.5 %, 10 × the session's median |1-min return|)`. With a flat 0.3 % the rule dropped 20 prints on 2025-04-07 / 04-09 / 04-10 that ES confirms were real moves; with the volatility floor it drops 25 spike/edge prints in the whole corpus and **none** is ES-confirmed as real.

**Every dropped minute (2 302 on 159 days):**
- `gross` 297: 2023-09-21 06:31–08:08 ×97 (prints 3 678–3 873 vs SPX ≈ 4 350; session starts 08:09); 2023-11-29 06:32–10:29 ×198 (3 451–4 057 vs ≈ 4 550; session starts 10:31); 2024-04-26 06:30, 06:34 (22.78 — a VIX-like value).
- `nonpositive` 1: 2025-11-11 10:48 (NULL).
- `spike` 10: 2024-02-29 06:40–06:41; 2024-04-02 06:40–06:41; 2024-05-02 06:31; 2024-07-31 06:31; 2024-09-26 06:31; 2024-10-28 06:31; 2024-12-18 12:47; 2026-02-25 06:32 (24–86 pts off ES).
- `edge_first` 10 (stale first print; all before the 06:33 window except none): 2023-09-26 06:31; 2023-09-29, 2023-10-17, 2024-03-06, 2024-04-04, 2024-07-26, 2025-04-03, 2025-11-28, 2026-08-07 06:30; 2024-05-15 06:32.
- `edge_last` 5 (bad 13:00 `stock_price` print, 31–60 pts off ES; close falls back to 12:59): 2026-02-18, 2026-03-06, 2026-03-10, 2026-04-27, 2026-05-19.
- `frozen_open` 1 856 on 113 days: 97 lagged days × 16 prints (06:30–06:45, the 15-min delay — these map to before 06:30 after the shift anyway); 2023-10-24 ×165; 2023-09-15 ×42; 2023-05-15 ×13; 2023-05-24 ×11; 2024-01-19, 2024-02-16 ×10; 2023-11-17, 2023-12-15 ×9; 2023-05-05 ×8; 2023-09-11, 09-18, 10-02, 10-09, 10-16, 10-23 ×4; 2023-09-25 ×3.
- `frozen` (interior, duplicates of a kept print) 123 on 26 days — mostly the post-10:00 tail on early-close days (2023-11-24 ×13, 2024-07-03 ×12, 2024-12-24 ×12, 2025-07-03 ×14) and 2023-06-26 ×9, 2025-10-30 ×8, 2023-06-27 ×6, 2023-09-15 ×5; the rest ×2–4.

**Dry run** (`--dry-run`, 815 active source rows, 2023-05-01 → 2026-09-18):
- SPX sessions 850; ES "sessions" 866. **ES has 26 RTH-window "sessions" with no SPX session** — market holidays with Globex trading (Memorial Day, Juneteenth, July 4, Labor Day, Thanksgiving, MLK, Presidents' Day…) plus 2026-02-21 (a Saturday — stray ES bars) and 2026-09-07. The canonical horizons count those as sessions; the shadow version does not.
- Status, new = control: computed 479, na_regime 316, na_data 15, pending_history 5 (no status transitions).
- Pooled computed rows: touch **83.9 % (ES) → 73.3 % (SPX)**; close 8.1 % → 8.1 %.
- **Control:** recomputing all 815 rows on ES through the same path reproduces the stored canonical rows exactly — 0 mismatches in status, touch, close, `days_to_reach`, `horizon_end_date`; `session_open_t0` / `session_close_t0` / `final_close_distance` max |diff| 0.0. Landscape and features have not drifted; every old → new difference is the price series (incl. its session calendar).

## Step 2 — shadow INSERT (2026-09-21 01:49 UTC)

Run **`ba5d92f4-2e97-4c99-a99b-07877fa71681`** (`bt_backfill_runs`: `completed`, `rows_inserted` 815, self-assessment "OK: one shadow row per active canonical row; canonical untouched"). Interpreter `apps/web/.venv/bin/python`; minutes fetched live (no dev cache). Log: `scripts/logs/cr_bh_run_*.log`.

- Inserted **815** rows under `v0.6.0-openiv-spxcash` (computed 479, na_regime 316, na_data 15, pending_history 5). INSERT only, `ON CONFLICT DO NOTHING`, one transaction.
- After-state: `v0.5.0-rebuilt` 743 · `v0.6.0-openiv` 815 active + 1 inactive (unchanged; max `computed_at` still 2026-09-21 00:02:04 = CR-BG) · `v0.6.0-openiv-spxcash` 815. No `bt_daily_features` rows written. Canonical constant untouched.
- The shadow version is a snapshot: the CR-AA sweep filters on the canonical version and will not promote its 5 `pending_history` rows.

## Step 3 — the diff (`scripts/cr_bh_diff.py` → `scripts/logs/cr_bh_diff.md`, per-row CSV `scripts/logs/cr_bh_diff_rows.csv`)

Per-row tags in the CSV: `series_segment`, `lagged_day`, `horizon_crosses_cutover`, `qa_flag_t0`, `bucket`, `basis`. Full tables (confusion by regime × horizon × bucket, all 112 flips, both scopes) are in the md file; headline numbers:

**Control.** 0 of 815 stored canonical rows differ from an ES recompute today → every difference below is the price series (incl. its session calendar: ES counts 26 market-holiday / stray Globex "sessions", so `horizon_end_date` moved on 248 of 479 computed rows — the SPX horizon is the same number of *real* sessions and therefore ends later; that works *against* T→F flips).

**Status transitions.** None (computed 479, na_regime 316, na_data 15, pending 5 in both versions; same in the `stock_price` segment: 403 / 266 / 11 / 5).

**Basis buckets (chosen series, median 06:58–07:02 PT):** < 20: 325 days (mean 10.7) · 20–35: 220 (27.4) · 35+: 249 (47.2) · no basis 21. These supersede both earlier tables. On the `stock_price` segment the *old* 5-session magnet-above touch rates by bucket are 47.4 / 45.8 / 81.6 % — identical to the correction already appended to the open-question.

**Confusion, 479 rows computed in both (old → new):** touch T→T 342 · **T→F 60** · F→T 9 · F→F 68 (83.9 % → 73.3 %); close T→T 10 · T→F 29 · F→T 29 · F→F 411 (8.1 % → 8.1 %). 112 rows flip at least one label (touch 69, close 58). `stock_price` segment (403): touch T→F 52 / F→T 8 (84.6 → 73.7 %), close 26 / 25 (8.7 → 8.4 %); 99 flip rows. Touch T→F by bucket: 35+ 30, 20–35 15, < 20 14, no basis 1. The 9 F→T are 7 magnetic-pin rows (ES sat above the pin band; SPX is inside it) and 2 magnet-above rows (2023-05-26, 2024-08-22: longer real-session horizon).

**Pooled rates old → new (full | `stock_price` segment):**

| regime | horizon | n | touch % | close % | days_to_reach | final dist (pts) |
| --- | --- | --- | --- | --- | --- | --- |
| magnet-above | 5 | 99 \| 85 | 61.6 → 43.4 \| 62.4 → 42.4 | 17.2 → 13.1 \| 16.5 → 11.8 | 1.1 → 1.9 \| 1.0 → 1.8 | −20.6 → −48.2 \| −18.9 → −48.8 |
| magnet-above | 20 | 176 \| 153 | 90.9 → 76.7 \| 91.5 → 77.8 | 8.0 → 8.0 \| 9.2 → 8.5 | 3.0 → 5.0 \| 3.2 → 5.2 | +28.7 → −2.4 \| +33.7 → +2.1 |
| magnet-above | 60 | 103 \| 83 | 88.3 → 84.5 \| 91.6 → 86.7 | 3.9 → 4.9 \| 4.8 → 6.0 | 13.5 → 15.0 \| 14.4 → 14.4 | +215.0 → +189.0 \| +215.5 → +190.0 |
| magnetic-pin | 5 | 32 \| 26 | 78.1 → 71.9 \| 76.9 → 76.9 | 12.5 → 18.8 \| 11.5 → 19.2 | 0.4 → 0.5 \| 0.6 → 0.5 | +21.4 → −12.4 \| +24.5 → −10.2 |
| magnetic-pin | 20 | 59 \| 46 | 96.6 → 94.9 \| 95.7 → 93.5 | 0.0 → 1.7 \| 0.0 → 2.2 | 1.2 → 0.9 \| 1.4 → 0.9 | +55.4 → +33.7 \| +48.5 → +24.2 |

(magnet-above 1-session n = 3, magnetic-pin 1-session n = 2, 60-session n = 5 omitted here.) Mean final distance falls by ≈ 26–34 pts everywhere — the mean basis, as it should.

**Confound test — magnet-above touch by basis bucket (< 20 · 20–35 · 35+; z, p for < 20 vs 35+):**

| horizon | version | full corpus | `stock_price` segment |
| --- | --- | --- | --- |
| 5 | old (ES) | 13/27 48.1 % · 14/28 50.0 % · 32/40 80.0 % — z 2.72, p 0.007 | 9/19 47.4 · 11/24 45.8 · 31/38 81.6 — z 2.66, p 0.008 |
| 5 | **new (SPX)** | 9/27 33.3 % · 12/28 42.9 % · 20/40 50.0 % — **z 1.35, p 0.18** | 6/19 31.6 · 9/24 37.5 · 19/38 50.0 — **z 1.32, p 0.19** |
| 20 | old (ES) | 78/87 89.7 · 37/41 90.2 · 44/47 93.6 — z 0.77, p 0.44 | 66/73 90.4 · 30/33 90.9 · 43/46 93.5 — p 0.56 |
| 20 | **new (SPX)** | 73/87 83.9 · 30/41 73.2 · 31/47 66.0 — **z −2.38, p 0.017** | 61/73 83.6 · 27/33 81.8 · 30/46 65.2 — **z −2.30, p 0.022** |
| 60 | old / new | 92.0 · 77.1 · 100 / 92.0 · 65.7 · 100 | 100 · 74.1 · 100 / 100 · 59.3 · 100 |

Reading: (1) the 5-session gradient drops from +32 pts (p 0.007) to +17 pts (p 0.18) — most of it was the basis; what remains is the same sign, not significant at n = 27 / 40, and excluding ES-QA-flagged t0 days does not change it (8/26 · 10/26 · 20/40, p 0.12). (2) At 20 sessions the ES version showed no gradient because touch was saturated by the bias; on SPX cash a **reverse** gradient appears — high-basis (first month after quarterly OPEX) rows touch *less* (66 % vs 84 %, p ≈ 0.02). Both scopes agree, so the 2023 segment is not driving either result. The basis explains the inflation; it does not explain everything — there is an OPEX-cycle dependence with opposite sign at 5 vs 20 sessions that deserves its own open-question (small n, two tests; treat as a lead, not a finding).

**ES QA flag (flag only): 30 of 850 sessions** — `spot_shifted_2023` 3 (all lagged days: 2023-05-18 high +10, 2023-07-11 high +11, 2023-08-24 low −13), `stock_price` 27: 2023-11-14 (low +53), 2023-11-29 (high +21; 149-minute session), 2023-12-04 (high −36), 2023-12-07, 2023-12-19, 2024-01-03 (high −19), 2024-01-17 (high −31), 2024-02-07, 2024-02-13 (high −40), 2024-03-05 (high −21), 2024-03-07, 2024-03-18, 2024-03-22 (high −13), 2024-05-31, 2024-09-18 (spread 18), 2024-10-31, 2024-12-18 (spread 15; low −22), 2025-04-07 (spread 16), 2025-04-10, 2025-06-23, 2025-11-26 (low −30; session starts 09:27), 2026-02-27 (high −19), 2026-06-29, 2026-07-08 (low +28), 2026-07-10 (low −25), 2026-07-15 (low +28), 2026-08-25. Sign: (ES-implied − SPX); "high −36" = the SPX series high is 36 pts above what ES implies — a residual stale opening print surviving past 06:33. The seven "high −" days could only *add* touches to the new version; checked: they set `days_to_reach` on three rows (2023-11-28 / 29 / 30, touch day 2023-12-04) and change **no** touch label (each also touches on another day). Lagged-day closes (median `stock_price` 12:56–13:00) vs the day's own basis, 101 days: median |err| 3.1, p90 10.7, max 19.0, mean −0.2 pts — unbiased, a few pts of noise against a 0.25 × IM (≈ 10–14 pt) close tolerance.

**Known limitations of the shadow version.** Residual stale opening prints on the 27 flagged `stock_price` days (rows are INSERT-only; not corrected); truncated sessions on outage days (2023-09-21 from 08:09, 2023-10-24 from 09:01, 2023-11-29 from 10:31, 2025-11-26 from 09:27, 2023-05-23, 2026-07-22, 2026-08-19, 2026-08-28 late starts); minute sampling understates highs by ≈ 0.75 pts (biases touch slightly down); SPX close is the 13:00:00 print vs ES 13:00:59.

## Wrap (2026-09-20)

- Delivered: shadow version `v0.6.0-openiv-spxcash` (815 rows, run `ba5d92f4-2e97-4c99-a99b-07877fa71681`), diff (Step 3), session note `2026-09-20 - CR-BH — SPX-Cash Shadow Outcomes` (incl. all 112 flip dates), open-question `outcome-target-vs-es-roll-seam-and-basis-bias` appended and left `in-progress`, new low-priority open-question `orats-monies-spot-price-drift-2026`.
- Deltas vs the frozen spec: series is the A1 hybrid, not `spot_price`; SPX window starts 06:33; spike threshold is volatility-scaled; active rows only (815, not 816); an ES control recompute was added (0 diffs); basis buckets use a 5-minute median.
- Not done / deferred: canonical promotion (ADR, Ryan); residual stale opening prints on 27 ES-flagged `stock_price` days (reported, not fixed — ES may not alter a price and the CR is INSERT-only); the shadow version is a snapshot (5 pending rows will not be swept).
- Unfiled side findings: ES holiday / stray "sessions" counted in canonical horizons (26 days; `horizon_end_date` differs on 248 computed rows); `cr_g` / `cr_i` fixed-UTC session window (an hour early in PST months); possible post-OPEX dependence of magnet-above touch (reverse 20-session gradient, p ≈ 0.02).
- Nothing to deploy. No live code path touched: `packages/`, `apps/`, `cr_b_backfill_outcomes.py`, `cr_aa_sweep_pending_outcomes.py` unchanged.
