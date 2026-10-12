# CR-BL — Vol State Table and Realized-over-Implied Range Labels

> Authority: vault CR `Dash/CRs/CR-BL — Vol State and Realized-over-Implied Labels.md` (last_reviewed 2026-10-11), kickoff prompt 2026-10-11.
> Branch: `feat/CR-BL-vol-state-range-labels` (off `origin/main` 9949d24). CR-BL confirmed unused (no branch, spec or other vault reference).
> Locked decisions L1–L14 and pass criteria transcribed verbatim from the vault CR; treated as confirmed by Ryan via the kickoff prompt.


## Why

Chat review on 2026-10-11 of SpotGamma Founder's Notes ahead of the year's largest ES moves (18 notes read, 7 move windows; 2025-10-10, 2025-11-13/20, 2025-12-16, 2026-01-20, 2026-03-26, 2026-03-31) found no directional calls. What was flagged in advance each time was a combination: implied vol at the floor, gamma zero or negative, and a level whose break flipped the desk risk-off.

Dash stores the gamma half and almost none of the vol half:

- `atm_iv_percentile`, `skew_percentile`, `term_structure_slope`, `vol_risk_premium` are NULL on all 242 active `v0.6.1-nyse-sessions` rows of `bt_daily_features` since 2025-10-10.
- The only realized-vs-implied label is the one-day `range_over_im` in `bt_daily_outcomes`.

A mispriced range is realized versus implied. Gamma alone does not give one: on train dates, a crude call-minus-put tilt from `orats_oi_gamma` roughly doubles raw ES range from the most positive to the most negative quintile, but `range_over_im` is nearly flat (1.07 to 1.16; close beyond the implied move 12.8 % vs 18.1 %, n ≈ 148 per quintile). CR-BK's "Why" records the same flatness across regimes.

This CR builds the missing vol-state data and multi-horizon realized-over-implied labels, then runs one pre-registered test of whether vol state, alone and combined with gamma, identifies days where the range is mispriced.

## Prior looks (disclosed)

- Chat ran the crude-tilt vs `range_over_im` table above on train dates (≤ 2026-06-05). It used gamma only. The T1 variable in this CR (implied-vol percentile vs realized-over-implied) has not been looked at.
- An earlier chat table of crude tilt vs raw ES range ran through 2026-10-09, so ES ranges by tilt in the post-2026-06-05 window have been seen. No trade outcomes were read. Any later holdout confirmation of a range strategy must note this.

## Locked decisions (Ryan to confirm before spec freeze)

| # | Decision | Value |
|---|---|---|
| L1 | Scope | Two new tables, a backfill, one test run. No change to `bt_daily_features`, `bt_daily_outcomes`, `orats_gex_landscape`, any card, any existing cron. No new cron in this CR. |
| L2 | Tables | `bt_vol_state_daily` (one row per ticker, trade_date, version) and `bt_range_labels_daily` (one row per ticker, trade_date, horizon_sessions, version). SPX only. |
| L3 | As-of rule | Every vol-state field for date t is observable before the 06:30 PT open of t: implied vol from the prior session's 12:46 PT `orats_monies_minute` snapshot (same snapshot rule as CR-BK A1.2), realized vol through the prior close. |
| L4 | Implied vol | Constant-maturity ATM IV at 7 and 30 calendar days, interpolated linearly in total variance between the bracketing expiries' `atmiv`. One-day implied move = the existing open-straddle `implied_move_1d` (ADR locked), reused as stored, not recomputed. |
| L5 | Realized vol | SPX cash, CR-BH hybrid series (cutover 2023-11-09). Close-to-close and Parkinson (high-low), each over 5 and 21 sessions, annualized on 252. |
| L6 | Premium | `vrp_30` = iv_30d − rv_21_cc. `vrp_7` = iv_7d − rv_5_cc. Vol points. |
| L7 | Percentiles | Rank within the trailing 250 sessions, starting once 120 prior sessions exist (same window as CR-BK A1.12). Stored for iv_7d, iv_30d, rv_21_cc, vrp_30, vrp_7. |
| L8 | Labels | Horizons 1 and 5 sessions. `range_over_implied` = (max high − min low over the horizon, SPX cash RTH) / implied move. `abs_close_over_implied` = \|close at horizon end − open of t\| / implied move. Implied move: horizon 1 = `implied_move_1d`; horizon 5 = SPX open × iv_7d × sqrt(5/252). The two horizons use different implied bases and are not compared with each other. |
| L9 | Axis | SPX cash for every price, range and label. No ES in this CR. |
| L10 | Sessions | NYSE calendar via `packages/shared/sessions.py`. |
| L11 | Dates | Tables are populated for every date with inputs. Tests read train only: trade_date ≤ 2026-06-05, and for horizon 5 the horizon must end ≤ 2026-06-05. Holdout label rows are written but not read, summarised or plotted. |
| L12 | Writes | Unattended backfill protocol from `CLAUDE.md` (`get_backfill_db_conn`, `assert_role_or_die`, `backfill_run`), INSERT-only, `version` column on every row. Table creation by migration in this CR. |
| L13 | Run discipline | Thresholds and pass criteria below are committed before any label is joined to any feature. One run. Month-cluster bootstrap as CR-BI (10 000 resamples, seed 20260920, 95 % percentile). No re-tuning. |
| L14 | Gamma input for T2/T3 | CR-BK's corrected net gamma percentile at the SPX open (A1.12), read from CR-BK's saved output. If CR-BK stops before producing it, T2 and T3 are not run. Chat's crude tilt is not a substitute. |

## Step 0 — diagnosis gate (answer all before implementation; contradiction-stop)

1. Confirm CR-BL is unused.
2. `orats_monies_minute` (and `_archive`) inventory: date range, expiries per day, share of sessions with a snapshot at or before 12:46 PT, share of days where expiries bracket both 7 and 30 calendar days. Usable date count.
3. Find the code that should populate the four vol columns in `bt_daily_features` and say why they are NULL (`vol_risk_premium` is known NULL since 2026-05-22; the other three are NULL across the last year). Report only. Do not fix here.
4. State exactly how `implied_move_1d` and the existing `range_over_im` are computed and in which frame (ES or SPX). If the existing label is ES-based, report the difference against the SPX-cash horizon-1 label on overlapping train dates (distribution only, no feature joins).
5. SPX RTH high/low source: confirm the CR-BH hybrid series gives session high, low, open, close for all train dates; quantify the known minute-sampling understatement (median 0.75 / 1.38 / 2.25 pts on high / low / range) against the implied move.
6. AM-settled expiries (`expiry_tod = 'am'`): say whether they are excluded from the interpolation or kept, and how many days it changes.
7. Sanity report on the built series, no outcomes: distribution of iv_7d, iv_30d, rv_21_cc, vrp_30 by year, and ten hand-checked dates.
8. Confirm no lookahead for every stored field (L3).
9. Usable train dates for T1 at each horizon. Stop if fewer than 500.
10. Correlation index feasibility only: is CBOE 1-month implied correlation daily history obtainable and usable under its terms. Report. No ingest in this CR.

## Tests (pre-registered)

Train dates, SPX cash, one run. Y at each horizon: P(`abs_close_over_implied` > 1), mean `range_over_implied`, and P(`abs_close_over_implied` < 0.5).

**T1 — vol state alone.** X = iv_7d percentile. Compare bottom tercile (cheap) against top tercile (rich). Also Spearman of X against `range_over_implied`. Secondary readout, not a pass criterion: the same split on vrp_7 percentile.

**T2 — cheap vol with low gamma (gated on CR-BK, L14).** Cell = iv_7d percentile ≤ 33 and CR-BK gamma percentile ≤ 33. Compare against all other train days.

**T3 — rich vol with high gamma (gated on CR-BK, L14).** Cell = iv_7d percentile ≥ 67 and CR-BK gamma percentile ≥ 67. Compare against all other train days.

## Pass criteria (locked)

| Test | Pass |
|---|---|
| T1 | P(close beyond implied) in the cheap tercile exceeds the rich tercile by ≥ 8 pts at either horizon, CI lower bound > 0 |
| T2 | P(close beyond implied) in the cell exceeds other days by ≥ 8 pts at either horizon, CI lower bound > 0, cell n ≥ 40 |
| T3 | P(close inside half implied) in the cell exceeds other days by ≥ 8 pts at horizon 1, CI lower bound > 0, cell n ≥ 40 |

Verdicts:

- **T1 or T2 passes** → cheap-range days are identifiable; follow-up CR specifies a long-range structure, adds a daily cron for both tables, and confirms once on the holdout.
- **T3 passes** → entry filter for the parked 0DTE condor; follow-up CR re-runs the condor with it, train then holdout.
- **Nothing passes** → the tables stay as data; record in State of Knowledge that neither vol state nor vol-with-gamma separates realized from implied at these horizons.

## Running alongside CR-BK

- Separate worktree and branch. No shared files are edited: this CR adds new scripts, a migration and two tables, and reads `packages/shared/sessions.py` without changing it.
- CR-BK is read-only against the database and writes parquet locally; this CR writes only to its own two tables.
- Both read `orats_monies_minute` heavily on a 1 GB Postgres plan. Do not start this CR's backfill while CR-BK's single T1–T3 run is executing.
- Step 0, the build and T1 do not need CR-BK. T2 and T3 wait for CR-BK's output.

## Outputs

- Repo spec `specs/CR-BL-vol-state-range-labels.md` with Step 0 answers, results tables and verdict.
- `bt_vol_state_daily` and `bt_range_labels_daily` populated, run recorded in `bt_backfill_runs`.
- Session note in `Dash/sessions/` with the verdict.

## Not in scope

- Fixing the NULL vol columns in `bt_daily_features`.
- Any gamma computation (CR-BK owns it).
- Correlation index ingest, intraday flow, skew and term-structure features.
- Holdout reads, trade structures, P&L.
- Daily cron for the new tables.


## Step-0 findings (2026-10-11, read-only)

All queries were read-only (`default_transaction_read_only = on`). The scratch scripts and parquet are in the session scratchpad and are not committed. No holdout label was computed: SPX bars and implied moves were pulled through 2026-06-05 only. Proposed new table names `bt_vol_state_daily` and `bt_range_labels_daily` do not exist yet.

**Verdict: no gate fails. Gate 9 passes (626 / 653 ≥ 500). STOP for decisions D1–D4 below:** one is a design issue in L8 that biases T1 toward passing at horizon 5.

### Q1 — CR-BL unused
Confirmed. There is no local or remote branch, no file in `specs/`, and the only vault reference is this CR note.

### Q2 — `orats_monies_minute` inventory
- **Main table:** 865 trade dates, 2023-05-01 → 2026-10-09. That equals the NYSE calendar exactly: no missing sessions, no extra non-sessions. Ticker is `SPX` only (SPXW expiries are stored under SPX). Per day: median 391 snapshots (min 181) and 54.5 expiries (47–59).
- **`_archive`:** 71 dates, 2025-11-03 → 2026-02-13. All 71 are also in the main table, so the archive is not needed.
- **Snapshot ≤ 12:46 PT:** exists on 865 / 865 days.
  - At exactly 12:46 on 856 days.
  - 2025-05-08: 12:30 (16 min stale).
  - Early closes: last print of the session (10:06–10:47 PT) on 2023-07-03, 2023-11-24, 2024-07-03, 2024-11-29, 2024-12-24, 2025-07-03, 2025-11-28, 2025-12-24.
- **Brackets:** expiries (pm, `atmiv` > 0, ≥ 1 calendar day) bracket both 7 and 30 calendar days on **865 / 865** days. The max gap between listed expiries within 45 days is 3–8 days (median 4). `atmiv` NULL/≤ 0: 7 rows in total.
- **Usable dates:** 865 snapshot days → vol state for trade dates 2023-05-02 → 2026-10-12 (as-of t−1).
- **Convention, stated here:** constant maturity is measured in calendar days from the **snapshot date** (t−1) to the expiry date: `cal = expir_date − trade_date(snapshot)`. ORATS `dte` is that value + 1. 0-calendar-day expiries are excluded as nodes.

### Q3 — why the four `bt_daily_features` vol columns are NULL (report only)
They are NULL on **all 829 active `v0.6.1-nyse-sessions` rows** (2023-05-01 → 2026-10-09), not only since 2025-10-10. They are also NULL on all 815 active `v0.6.0-openiv` rows. They are populated only on `v0.5.0-rebuilt`: `atm_iv_percentile`, `skew_percentile` and `term_structure_slope` on 678 rows, `vol_risk_premium` on 735, last non-NULL 2026-05-22.
- **Compute:** the functions are in `packages/shared/vol_features.py` (`compute_atm_iv_percentile` :90, `compute_skew_percentile` :121, `compute_term_structure_slope` :147, `compute_vol_risk_premium` :222).
  - IV comes from the EOD (MAX snapshot) near-30 / near-90 DTE `orats_monies_minute` rows.
  - Realized vol is **ES** (`ironbeam_es_1m_bars`, `fetch_es_closes_before` :467).
  - The percentile window is 60 sessions.
- **Only writer:** the one-off `scripts/cr_d_backfill_vol_features.py`, hard-coded to `FEATURE_VERSION = "v0.5.0-rebuilt"` (:57), using a COALESCE UPDATE. 2026-05-22 is simply the end of the v0.5.0-rebuilt corpus when CR-D ran.
- **Cron path:** `apps/cron/job_orats_eod.py:274` → `compute_and_upsert_daily_features` (`day_features.py:264–274` upserts feature_vector/hash/regime only). The post-open `scripts/cr_ab_open_implied_move.py` (`day_features.py:338–346`) also never writes them. Neither did `cr_ab_backfill_openiv.py` when it created `v0.6.0-openiv`.
- **Carry-over:** `scripts/cr_bi_backfill_nyse_sessions.py:79–90` copies the openiv rows verbatim, NULLs included.
- **Root cause:** the columns were filled once on a version that is no longer canonical, and nothing computes them going forward. There is no input-data cause. Not fixed here (out of scope).

### Q4 — `implied_move_1d` and `range_over_im`
- **`implied_move_1d`** (feature_vector key, `v0.6.1-nyse-sessions`): `table_spot × atmiv × sqrt(1/252)` (`gex_landscape.py:244–249`).
  - `atmiv` is the first `orats_monies_minute` row with `snapshot_pt ≥ 06:33` PT on t, smallest `dte > 0`.
  - `table_spot` = `orats_oi_gamma.stock_price`, which is the prior session's 12:46 PT per-expiry snapshot (CR-BK Step 0).
  - Written by `compute_and_upsert_open_implied_move` (`day_features.py:482–577`), locked by CR-037.
  - **Frame: SPX.** It is an ATM-IV move, not a straddle price, despite the "open-straddle" name. It is observed at 06:33 on t, which is fine for a label denominator. It is not a vol-state field and will not be stored in `bt_vol_state_daily`.
- **Coverage gap:** 743 / 778 train sessions have it. **The 35 missing sessions are all first-session-after-a-closure days**: 2023-05-30, 2023-06-20, 2023-07-05, 2023-09-05, 2023-11-24, 2023-12-26, 2024-01-02, 2024-01-16, 2024-02-20, 2024-04-01, 2024-05-28, 2024-06-20, 2024-07-05, 2024-09-03, 2024-11-29, 2024-12-24, 2024-12-26, 2025-01-02, 2025-01-10, 2025-01-21, 2025-02-18, 2025-04-21, 2025-05-27, 2025-06-20, 2025-07-03, 2025-07-07, 2025-09-02, 2025-11-28, 2025-12-24, 2025-12-26, 2026-01-02, 2026-01-20, 2026-02-17, 2026-04-06, 2026-05-26. This is the holiday-remap gap CR-BK hit. Under L4 ("reused as stored") these days have no horizon-1 label (D3).
- **`range_over_im`** (`bt_daily_outcomes`, canonical `v0.6.1-nyse-sessions`): `(h − l) / implied_move_1d`; `close_move_over_im = (c − o) / implied_move_1d` (`outcomes_runner.py:227–229`).
  - h/l/o/c are **ES** RTH bars, 06:30–13:00 PT (`sessions.fetch_es_daily_bars`).
  - So the ratio is ES points over an SPX denominator. The SPX-cash shadow `v0.6.0-openiv-spxcash` (CR-BH) is do-not-promote.
- **ES label vs SPX-cash horizon-1 label**, 732 overlapping train dates, distributions only:

  | | ES (stored) | SPX cash (CR-BL h1) | SPX − ES, median [p10, p90] |
  |---|---|---|---|
  | range / IM, median | 1.038 | 0.972 | −0.073 [−0.173, −0.029] |
  | range / IM, mean | 1.164 | 1.089 | |
  | \|close − open\| / IM, median | 0.474 | 0.494 | +0.002 [−0.185, +0.182] |
  | P(\|close−open\|/IM > 1) | 15.0 % | 14.5 % | 32 days flip |
  | P(\|close−open\|/IM < 0.5) | 52.3 % | 50.7 % | 78 days flip |

  Spearman ES vs SPX: range 0.962, |close| 0.908.

### Q5 — SPX RTH OHLC source
`packages/shared/spx_cash.load_spx_daily_bars` (CR-BH A1 hybrid series, cutover 2023-11-09, prints 06:33–13:00 PT, lagged-2023 close = median `stock_price` 12:56–13:00) returns open/high/low/close for **778 / 778** NYSE train sessions (2023-05-01 → 2026-06-05). Minutes per session: median 388, min 149 (known outage days, CR-BH).
- **Minute-sampling understatement vs the median `implied_move_1d` (45.2 pts; p10 27.1):**
  - High: 0.75 pts = 1.7 % of IM.
  - Low: 1.38 pts = 3.1 %.
  - Range: 2.25 pts = **5.0 %** (p90 5.8 pts = 12.8 %).
- **Effect:** this lowers every `range_over_implied` by about 0.05 at horizon 1, which is consistent with the SPX − ES median of −0.07 above. It does not touch `abs_close_over_implied` (open and close are single prints). The T1–T3 pass criteria use only the close label.

### Q6 — AM-settled expiries
At the 12:46 snapshot, 16 489 of 47 127 rows are `expiry_tod = 'am'` (Fridays and a few Thursdays; within 45 days 1–2 per day, i.e. the monthly).
- **One row per expiry date:** on an AM monthly date there is no separate PM row, so excluding AM removes that date as a node.
- **Effect of excluding:** both maturities stay bracketed on every day either way.
  - iv_7d: the bracketing pair changes on **44 days**; |Δ| median 0.80 vol pt, p90 1.38, max 3.50.
  - iv_30d: changes on **114 days**; |Δ| median 0.13, p90 0.22, max 0.29.
- **Recommendation (D2):** exclude AM. Its settlement is about 6.5 h earlier (~4 % of T at 7 days) and it carries the SOQ open, so mixing it with PM dailies in a calendar-day interpolation is the source of the 7-day differences. Scratch values below use AM-excluded.

### Q7 — sanity report (train dates only, no outcomes; vol points)

| field | 2023 (n 168/147) | 2024 (252) | 2025 (250) | 2026 ≤ 06-05 (107) |
|---|---|---|---|---|
| iv_7d p10 / p50 / p90 (min–max) | 9.1 / 11.6 / 15.8 (7.7–19.8) | 8.8 / 11.2 / 16.2 (7.0–31.5) | 10.0 / 12.9 / 22.6 (6.5–61.2) | 10.5 / 14.3 / 22.6 (9.2–27.2) |
| iv_30d | 10.2 / 11.7 / 15.6 (9.4–17.8) | 10.0 / 11.6 / 15.6 (9.0–27.3) | 11.5 / 13.6 / 20.1 (10.4–42.4) | 12.4 / 14.8 / 20.8 (11.3–25.8) |
| rv_21_cc | 9.0 / 11.0 / 14.0 (6.7–16.3) | 8.4 / 11.5 / 16.0 (5.9–21.7) | 8.6 / 13.1 / 24.1 (5.9–49.2) | 9.3 / 12.3 / 18.9 (8.4–20.4) |
| vrp_30 | −2.3 / 1.0 / 3.8 (−5.2–6.1) | −3.0 / 0.3 / 4.2 (−9.7–8.8) | −6.4 / 0.7 / 5.3 (−30.7–11.0) | −2.6 / 2.4 / 7.1 (−5.7–10.6) |

- **Hand checks (10 dates: seed 20261011 random 7 plus 3 stress days):** for every date, the interpolation was recomputed by hand from the printed bracketing expiries and matched to the stored value, and rv_21_cc was recomputed independently from the 21 SPX closes strictly before t. All 10 match exactly. Examples:
  - 2024-08-06 (snap 08-05): iv_7d 31.5, iv_30d 27.3, rv21 18.5.
  - 2025-04-07 (snap 04-04): iv_7d 45.7, iv_30d 33.3.
  - 2025-04-09 (snap 04-08): iv_7d 61.2, iv_30d 42.4, vrp_30 +11.0.
  - 2024-05-17: iv_7d 9.9, iv_30d 10.3 (29d/32d bracket).
  - 2023-06-27, 2024-07-31, 2024-09-09, 2024-11-15, 2025-06-09, 2025-07-03: all match.
  - ATM-30d sits roughly 10–12 pts under the recalled VIX closes on the stress days (VIX includes skew). That is plausible, and only a coarse check.
- **Lowest iv_7d** values are all holiday weeks (2025-12-23/24/26/29, 2024-05-28: 6.5–7.1). See Q9 / D1.

### Q8 — lookahead (L3)
- **Implied vol:** the snapshot date is strictly < t for every row (asserted), and the snapshot time is ≤ 12:46 PT (max 12:46:00).
- **Realized vol:** rv uses closes and high/low strictly before t (hand-check: last close used = t−1).
- **Percentiles:** `pct_t = 100 × share of {x_{t−k}: 1 ≤ k ≤ 250, non-NULL} ≤ x_t`, required ≥ 120 prior values. x_t is itself as-of, and the window excludes t.
- **Derived fields:** vrp_* combine only those inputs.
- **No stored vol-state field uses t's open or later.** Label denominators (`implied_move_1d` at 06:33, SPX open of t) are label-side only.

### Q9 — usable train dates for T1
- **iv_7d percentile** exists from **2023-10-23** (657 train dates); vrp_7 percentile from 2023-10-30 (652).
- **Horizon 1** (needs `implied_move_1d`): **626**.
- **Horizon 5** (sessions t..t+4, end ≤ 2026-06-05, last start 2026-06-01): **653**.
- Both ≥ 500, so **pass**. Terciles: about 209 / 218 days.
- **Finding that bears on L8 (D1):** iv_7d spans 7 calendar days from the t−1 snapshot. On 137 of the 657 dates that window holds only **4** sessions (holiday weeks), against 5 on the other 520.
  - Those 137 days fall in the cheap tercile 54 % of the time (74 / 34 / 29 cheap / mid / rich), against 27 % for the rest (141 / 171 / 208).
  - The horizon-5 window on those days runs 8–11 calendar days from the snapshot, but L8 scales iv_7d by sqrt(5/252) as if the 7 days held 5 sessions.
  - So the horizon-5 implied move is understated by up to ~sqrt(4/5) (≈ 11 %) on days that cluster in the cheap tercile. That mechanically raises cheap-tercile `range_over_implied` and P(close beyond implied) at horizon 5. It biases T1 (and T2) **toward passing**.
  - Horizon 1 is unaffected (it uses `implied_move_1d`).

### Q10 — CBOE 1-month implied correlation (COR1M), report only
- **Obtainable:** Cboe serves a free daily CSV (`cdn-api.cboe.com/api/global/us_indices/daily_prices/COR1M_History.csv`; DATE/OPEN/HIGH/LOW/CLOSE), 2006-01-03 → 2026-10-09 (last close 6.93).
- **Terms:** cboe.com website terms allow viewing/downloading one copy for personal non-commercial use. Storing in a retrieval system, creating derivative works, or redistributing needs Cboe's prior written consent.
  - Index data for redistribution is licensed through Cboe Data Services (Customer Agreement – Index Data).
  - Third-party vendors (FirstRate Data, Optuma) sell it under their own licenses.
- **Usability:** a Postgres ingest is "storing in a retrieval system", so a durable ingest needs written consent or a licensed vendor feed. A one-off local personal-research read is arguably within the personal-use clause. Ryan's call; no ingest in this CR.

### Decisions needed before implementation
- **D1 (L8, horizon-5 implied basis).** Options:
  - (a) Keep L8 as written and add a `holiday_window` flag, reporting T1/T2 also on non-holiday days as a secondary readout.
  - (b) Amend L8: horizon-5 implied move = SPX open × σ_h × sqrt(T_h / 365), where σ_h² T_h is total variance interpolated (same L4 method, t−1 12:46 snapshot) to the calendar maturity of the horizon-end session's close. On a normal week this is within ~2 % of L8 (7/365 vs 5/252), and it removes the holiday understatement.
  - (c) (b) plus excluding holiday-window days from T1.
  - **Recommendation: (b).** It stays as-of t−1, needs no new data, and leaves T1's X (iv_7d percentile) unchanged.
- **D2 (L4, AM expiries):** exclude (recommended) or keep. 44 / 114 days affected (Q6).
- **D3 (L4/L8, 35 post-closure sessions without `implied_move_1d`):** leave horizon-1 labels NULL there (recommended; L4 says "as stored") or recompute.
- **D4 (L7, percentile definition):** confirm `pct_t` = share of the prior ≤ 250 non-NULL values ≤ x_t, min 120, t excluded (as Q8).
