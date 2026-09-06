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

