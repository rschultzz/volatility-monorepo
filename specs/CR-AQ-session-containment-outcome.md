# CR-AQ — Session containment outcome (all regimes)

> Authority: vault session note `Dash/sessions/2026-09-06 - CR-AQ — Session Containment Outcome.md`
> Branch: `feat/CR-AQ-session-containment-outcome` (off `origin/main` 122a763, the CR-AP merge)
> Scope: schema addition on `bt_daily_outcomes` (owner-role migration), `compute_session_containment` in the outcomes runner + tests, null-fill backfill script and one run under the backfill role. No analysis.
> Mode: unattended; halts at the first STOP gate that misses.

## Problem

Two problems, one fix.

`[Certain]` 310 of 804 canonical outcome rows are `na_regime`: amplification / bounded / untethered days have no outcome definition. 37 of the 60 summer-2026 dates fell there. Half the corpus is invisible to any outcome-conditioned analysis, and the mining pass would otherwise run on the magnet half only.

`[Certain]` Ryan has been trading 0DTE iron condors around the walls (6 in the brokerage export: 4 wins, 2 losses, net −$40 after fees; the 05/06 loss was on a magnet-above day with the call side sold 50 pts above the wall). Nothing in Dash tests that trade. Its outcome — does the session close inside the walls, and how much of the implied move did the range use — is computable for every date from ES bars, the landscape, and the open straddle, with no option quotes.

## Outcome definition

Per (ticker, trade_date, feature_version) on `bt_daily_outcomes`, new nullable columns:

| Column | Definition |
|---|---|
| `session_high_t0`, `session_low_t0`, `session_close_t0` | Trade-date RTH high / low / close, using the **outcomes runner's existing session logic** (same source as `session_open_t0`). Not ad hoc SQL — a chat query with a mis-set timezone produced wrong RTH ranges on 2026-09-06; the runner's session boundaries are the convention. |
| `wall_above_price`, `wall_below_price` | Nearest landscape wall strictly above / below `session_open_t0` from `orats_gex_landscape.walls` for the trade date (any sign). NULL if none on that side. |
| `contained_close` | `wall_below_price < session_close_t0 < wall_above_price`. NULL if either wall is NULL. |
| `contained_range` | `wall_below_price < session_low_t0 AND session_high_t0 < wall_above_price`. NULL if either wall NULL. |
| `close_pos_in_band` | `(session_close_t0 − wall_below_price) / (wall_above_price − wall_below_price)`; < 0 or > 1 means the close breached. NULL if either wall NULL. |
| `range_over_im` | `(session_high_t0 − session_low_t0) / implied_move_1d` from the canonical feature row. NULL if IM NULL or ≤ 0. |
| `close_move_over_im` | `(session_close_t0 − session_open_t0) / implied_move_1d`. Signed. |
| `breach_side` | `'above'` / `'below'` / `'both'` / `NULL` for `contained_range = false`. |

The Q-side (straddle-implied probability of containment) is **not stored** — it is derived at analysis time from `implied_move_1d` and the wall distances, so the model choice (normal, empirical, etc.) stays in the analysis, not the data.

## Locked decisions

| # | Decision | Value |
|---|---|---|
| 1 | Where | Columns on `bt_daily_outcomes` (every canonical date already has a row, including `na_regime` and null-horizon rows). Not a new table. |
| 2 | Migration | Owner-role migration adding the columns, in whatever migrations convention the repo uses (Step 0 locates; if none, `scripts/migrations/` with a dated SQL file and a `-- applied <date>` note). The backfill role cannot DDL — migration is applied by Claude Code under the owner URL as an explicit, logged step. |
| 3 | Computation | A function in the outcomes runner (`compute_session_containment(...)`) so the nightly `backfill_outcomes` cron fills these for new dates automatically after the next cron redeploy. The backfill script calls the same function. |
| 4 | Backfill | `scripts/cr_aq_backfill_containment.py`, `--feature-version` default canonical, null-fill idempotent (`session_close_t0 IS NULL`), backfill role, `backfill_run` cr_id CR-AQ, standard scaffolding. All 804 rows regardless of `outcome_status`. |
| 5 | Walls | From `orats_gex_landscape.walls` for the trade date (the EOD landscape stored under trade_date D is the pre-open landscape for D — same convention the regime classifier uses; Step 0 confirms). |
| 6 | Cron | The nightly cron will only fill these after a redeploy (auto-deploy is dead — [[orats-eod-cron-not-autodeploying]]). Redeploy of `backfill_outcomes` is a chat-side step after merge, noted in Next session. Until then, the backfill script can be re-run to fill new dates. |
| 7 | Not in scope | Any analysis of these columns (that is the mining pass); condor P&L; option quotes; changing the regime classifier. |

## Gates

| Gate | Expected | On miss |
|---|---|---|
| G0.1 — migrations convention located; owner URL available in env | yes | STOP |
| G0.2 — landscape-under-trade-date convention confirmed from the classifier's read path | pre-open for D | STOP if it's post-close for D (then wall source must shift to D+1) |
| G0.3 — session logic reused from the runner (function name recorded) | yes | STOP if the runner has no reusable session extractor |
| G1 — tests: containment true/false/both/NULL-wall; `close_pos_in_band` at 0, 1, breach; `range_over_im` NULL when IM ≤ 0 | pass; suites pass | STOP |
| G2 — migration applied; columns present; 0 non-null before backfill | yes | STOP |
| G3 — backfill: rows updated | 804 minus the 6 roll-Friday no-bar dates and any no-bar dates → ≥ 795 | STOP if < 780 |
| G4 — `contained_close` non-null | ≥ 700 (dates with a wall on each side) | note |
| G5 — sanity: `session_open_t0` unchanged for all rows (compare pre/post checksum) | identical | STOP |
| G6 — sanity: `session_low_t0 ≤ session_open_t0 ≤ session_high_t0` and same for close | 0 violations | STOP |
| G7 — 2026-05-06 (known condor loss, magnet-above): `contained_close = false`, `breach_side = 'above'` | yes | note — if not, check wall selection before trusting the column |


## Step 0 findings (read-only, 2026-09-06)

Interpreter: `apps/web/.venv/bin/python` for DB reads, the migration and the backfill; Rosetta repo venv for the suites.

| Gate | Expected | Actual | Result |
|---|---|---|---|
| G0.1 migrations convention; owner URL | yes | **`infra/sql/<table>_<change>.sql`**: dated SQL with its own `BEGIN … COMMIT`, an `-- Applied: <date>` note, column-level `GRANT UPDATE (…) TO dash_backfill_writer` (column grants do not auto-extend — CR-I lesson), and a `CREATE OR REPLACE VIEW bt_daily_outcomes_active` refresh because views snapshot their column list (CR-G ran it as `scripts/cr_g_ddl_step0a.sql` + `cr_g_run_ddl_step0a.py` under `DATABASE_URL`). `DATABASE_URL` connects as `rschultz`, which is `pg_tables.tableowner` for `bt_daily_outcomes` → DDL allowed | PASS |
| G0.2 landscape under `trade_date` D is the pre-open landscape for D | pre-open | **confirmed**: `apps/cron/job_orats_eod.py` computes from `api_trade_date` = previous business day with data and stores under `store_trade_date = next_business_day(api_trade_date)` (l.142–150); the classifier reads `_LANDSCAPE_ROW_SQL` with `trade_date = D` (`day_features.py` l.276–281, used by `compute_and_upsert_daily_features` l.437) | PASS |
| G0.3 session logic reused from the runner | yes | **`packages/shared/outcomes_runner.py::compute_outcome_for_date`** takes `daily_bars` (RTH daily OHLC, index = session date) and sets `session_open_t0 = daily_bars.loc[trade_date, "open"]`. The same frame carries high / low / close for `trade_date`, so the containment function reads `daily_bars.loc[trade_date]` — one source. The frame is built by `_fetch_daily_bars` in `scripts/cr_b_backfill_outcomes.py` (duplicated in `cr_aa_sweep_pending_outcomes.py`) from `_RTH_BARS_SQL`: 1-minute `ironbeam_es_1m_bars` aggregated per session with RTH bounded in UTC (bounds recorded below) | PASS |

`orats_gex_landscape` (SPX, ≤ 2026-09-04): 811 dates; **807 with ≥ 1 wall, 360 with ≥ 2 walls.** A containment value needs a wall on each side of the open, so the upper bound on `contained_*` non-null is ~360 — G4's "≥ 700" cannot be met by the locked wall source (`walls`, any sign); recorded as a note gate. 2026-05-06 (G7) has a single wall (7306.2, above spot 7271.2): by the locked definition `wall_below_price` is NULL and `contained_close` is NULL, not false — G7 will read as a note with that explanation.

Canonical outcome rows: 804 active; `session_open_t0` non-null on 795 (the 9 NULLs: 6 roll Fridays with no RTH bars + 3 others). Existing columns end at `session_open_t0` (35 columns); none of the CR-AQ columns exist yet.

### Design consequence found in Step 0 — when the cron can fill these

The nightly `backfill_outcomes` cron (`scripts/cr_b_backfill_outcomes.py`) runs at 13:40 UTC on D and inserts D's outcome row with `session_open_t0` from bars that are still accumulating; D's high / low / close are not final until 20:00 UTC. Computing containment at that insert would stamp a partial session as final. Therefore: `compute_session_containment` is called (a) by the CR-AQ backfill (null-fill over completed sessions), (b) by `cr_b` at insert **only when `trade_date` is before the run date** (catch-up dates), and (c) as a null-fill pass in the 11:05 UTC sweep (`cr_aa_sweep_pending_outcomes.py`) for rows whose `session_close_t0` is still NULL and whose session has closed — that is the path that makes new dates fill automatically after the redeploy. Both scripts already build the same `daily_bars` frame.

## Step 1 — migration applied (owner role) · Step 2 — checksum

**Step 2 (taken first, before any DDL):** `SELECT md5(string_agg(session_open_t0::text, ',' ORDER BY trade_date)) FROM bt_daily_outcomes WHERE feature_version='v0.6.0-openiv' AND active` → `29a56df6abab8f0691d8e05232e22b40`.

**Step 1:** `apps/web/.venv/bin/python -u scripts/cr_aq_run_migration.py` under `DATABASE_URL` (connected as the table owner `rschultz`). File: `infra/sql/bt_daily_outcomes_session_containment.sql` (ALTER TABLE … ADD 11 columns; GRANT UPDATE on them to `dash_backfill_writer`; COMMIT; CREATE OR REPLACE VIEW `bt_daily_outcomes_active` with the 35 existing columns in their ordinal order plus the 11 new ones). Runner output (statement as executed, then `\d`-style column list, grants, view columns, pre-backfill non-null count):

```
Connected as rschultz; bt_daily_outcomes owner = rschultz

Applying infra/sql/bt_daily_outcomes_session_containment.sql (3824 bytes)
--- statement ---
BEGIN;
ALTER TABLE bt_daily_outcomes
  ADD COLUMN session_high_t0    REAL,
  ADD COLUMN session_low_t0     REAL,
  ADD COLUMN session_close_t0   REAL,
  ADD COLUMN wall_above_price   REAL,
  ADD COLUMN wall_below_price   REAL,
  ADD COLUMN contained_close    BOOLEAN,
  ADD COLUMN contained_range    BOOLEAN,
  ADD COLUMN close_pos_in_band  REAL,
  ADD COLUMN range_over_im      REAL,
  ADD COLUMN close_move_over_im REAL,
  ADD COLUMN breach_side        VARCHAR(8);
GRANT UPDATE (
  session_high_t0, session_low_t0, session_close_t0,
  wall_above_price, wall_below_price,
  contained_close, contained_range, close_pos_in_band,
  range_over_im, close_move_over_im, breach_side
) ON bt_daily_outcomes TO dash_backfill_writer;
COMMIT;
CREATE OR REPLACE VIEW bt_daily_outcomes_active AS
SELECT ticker,
       trade_date,
       feature_version,
       regime_kind_at_classification,
       dominant_bucket_at_classification,
       horizon_sessions,
       horizon_end_date,
       outcome_status,
       reached_touch,
       reached_close,
       days_to_reach,
       max_excursion_in_direction,
       final_close_distance_from_target,
       actual_realized_em_pct,
       active,
       deactivated_at,
       deactivated_reason,
       backfill_run_id,
       computed_at,
       position_t1_post_touch,
       position_t5_post_touch,
       position_t15_post_touch,
       session_open_t1,
       session_high_t1,
       session_low_t1,
       session_close_t1,
       session_open_t5,
       session_high_t5,
       session_low_t5,
       session_close_t5,
       session_open_t15,
       session_high_t15,
       session_low_t15,
       session_close_t15,
       session_open_t0,
       session_high_t0,
       session_low_t0,
       session_close_t0,
       wall_above_price,
       wall_below_price,
       contained_close,
       contained_range,
       close_pos_in_band,
       range_over_im,
       close_move_over_im,
       breach_side
FROM bt_daily_outcomes
WHERE active = true;
--- end ---
applied ✓

\d bt_daily_outcomes — 46 columns:
  ticker                               character varying            nullable=NO
  trade_date                           date                         nullable=NO
  feature_version                      character varying            nullable=NO
  regime_kind_at_classification        character varying            nullable=YES
  dominant_bucket_at_classification    character varying            nullable=YES
  horizon_sessions                     integer                      nullable=YES
  horizon_end_date                     date                         nullable=YES
  outcome_status                       character varying            nullable=NO
  reached_touch                        boolean                      nullable=YES
  reached_close                        boolean                      nullable=YES
  days_to_reach                        integer                      nullable=YES
  max_excursion_in_direction           double precision             nullable=YES
  final_close_distance_from_target     double precision             nullable=YES
  actual_realized_em_pct               double precision             nullable=YES
  active                               boolean                      nullable=NO
  deactivated_at                       timestamp without time zone  nullable=YES
  deactivated_reason                   text                         nullable=YES
  backfill_run_id                      uuid                         nullable=YES
  computed_at                          timestamp without time zone  nullable=YES
  position_t1_post_touch               smallint                     nullable=YES
  position_t5_post_touch               smallint                     nullable=YES
  position_t15_post_touch              smallint                     nullable=YES
  session_open_t1                      real                         nullable=YES
  session_high_t1                      real                         nullable=YES
  session_low_t1                       real                         nullable=YES
  session_close_t1                     real                         nullable=YES
  session_open_t5                      real                         nullable=YES
  session_high_t5                      real                         nullable=YES
  session_low_t5                       real                         nullable=YES
  session_close_t5                     real                         nullable=YES
  session_open_t15                     real                         nullable=YES
  session_high_t15                     real                         nullable=YES
  session_low_t15                      real                         nullable=YES
  session_close_t15                    real                         nullable=YES
  session_open_t0                      real                         nullable=YES
  session_high_t0                      real                         nullable=YES
  session_low_t0                       real                         nullable=YES
  session_close_t0                     real                         nullable=YES
  wall_above_price                     real                         nullable=YES
  wall_below_price                     real                         nullable=YES
  contained_close                      boolean                      nullable=YES
  contained_range                      boolean                      nullable=YES
  close_pos_in_band                    real                         nullable=YES
  range_over_im                        real                         nullable=YES
  close_move_over_im                   real                         nullable=YES
  breach_side                          character varying            nullable=YES

GRANT UPDATE on new columns to dash_backfill_writer: 11/11
bt_daily_outcomes_active columns: 46
non-null new-column rows before backfill: 0 (expect 0)
```

| Gate | Expected | Actual | Result |
|---|---|---|---|
| G2 migration applied; columns present; 0 non-null before backfill | yes | 11 columns added (46 total), 11/11 column grants, view at 46 columns, 0 non-null | PASS |
