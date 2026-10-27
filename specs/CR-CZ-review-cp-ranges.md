# CR-CZ — Blind review set from CR-CP's flow-called ranges

Write: `setup_candidates` only, backfill protocol. Train only. Branch `feat/CR-CZ-review-cp-ranges` off `origin/main`
`625d2ec`, own worktree (`.claude/worktrees/cr-cz`). Kickoff: Ryan, 2026-10-27 (chat). Frozen 2026-10-27 before any
CR-CZ data was read.

---

## Kickoff (verbatim)

```
CR-CZ — Load a blind review set from CR-CP's flow-called ranges (write: setup_candidates only,
backfill protocol). Train only.

Branch first: git checkout -b feat/CR-CZ-review-cp-ranges off origin/main. Own worktree.
Read CLAUDE.md (backfill protocol), standing-rulings.md (B4, B8), the CR-CC spec (review tool,
setup_candidates schema, loader scripts/load_setup_candidates.py) and the CR-CP spec.
Set: CR-CP's D2 flow-called ranges on 2026 train (the 235 calls as frozen after A1), random sample
of 120 (seed 20261027), stratified half short-calls / half long-calls. One card per range:
  entry_ts_pt = the decision bar (so replay stops BEFORE the outcome), direction = the D2 side,
  level/zone meta = the box (high/low by closes), range start, approach leg, in-range delta share,
  progress, contracts/pt ratio; outcome = both races (called side AND opposite side, ±10 and ±20),
  hidden until reveal as CR-CC does.
set_name 'cr-cz:cp-ranges', cr_id 'CR-CZ'. Insert-only, run under backfill_run('CR-CZ').
Labels: Ryan uses Take / Skip / Unsure as built; reasons must include two new tags available in the
panel's reason list: "take continuation (other edge)" and "take fade (called side)". If adding reason
tags needs a frontend change, make it the smallest change (reason list only), behind no toggle,
Playwright unchanged elsewhere.
No analysis in this CR. Report counts loaded; PR (loader + any reason-tag change); don't merge.
```

## Standing rulings conformance (`Dash/research/standing-rulings.md`, status current, read 2026-10-27)

This CR builds review cards; it tests nothing. The A block describes setups and tests, so it is n/a except where the
cards carry CR-CP's definitions unchanged.

| # | Mark | How |
|---|---|---|
| A1–A21 | n/a | no test. The cards carry CR-CP's frozen A1 definitions (D0–D5) unchanged; nothing is re-defined or tuned |
| B1, B2, B3 | n/a | no verdict, no parameter search |
| B4 | honoured | train only: every card's session ≤ 2026-06-05 (asserted); CR-CP's train reads only; holdout, R-a, R-b and the forward log not read |
| B5 | honoured | outcomes recomputed from the Databento 1-min bars CR-CP used (`es_flow_1m_databento`); the chart replay uses the review tool's bars as built |
| B6 | honoured | the ±10 / ±20 races are CR-CP's stop-free direction races, labelled as such in `outcome`, not a model of Ryan's exit |
| B7 | n/a | no baseline |
| B8 | honoured | Code writes no reviews; no Playwright writing test is run; cards load unreviewed |
| C1 | honoured | no colours changed |
| C2 | **deviates — pre-approved in the kickoff** | the two new reason chips appear in the existing panel's list for every set, with no toggle ("behind no toggle", Ryan). Recorded, not stopped for |

Nothing from "Revoked / superseded" is used (R1–R7: no entry rule is modelled; the races start at CR-CP's D4 edge fill).

---

## Step 0 findings (2026-10-27; code read at `625d2ec`; DB catalog reads only, read-only transaction; no outcome read)

1. **Reason tags need no frontend change.** The panel's chips come from `GET /api/setup-review/candidates/<id>`
   → `service.known_reasons()` = `DEFAULT_REASONS` (`apps/web/modules/SetupReview/service.py:22`) plus every reason
   already used in `setup_reviews`. `SetupReviewPanel.jsx` renders `body.reasons` as given. The change is two strings
   in `DEFAULT_REASONS`, inserted before "other". Backend-only, so it shows after the "Dash - MonoRepo" deploy. No
   Playwright spec asserts the chip list; none is changed.
2. **Loader.** `scripts/load_setup_candidates.py` hard-codes `backfill_run(conn, "CR-CC")`. Smallest change: a
   `--run-cr` option (default `CR-CC`, so earlier loads behave as before); this CR loads with `--run-cr CR-CZ`.
   Validation, insert-only `ON CONFLICT (id) DO NOTHING` and the outcome-key deny-list are unchanged.
3. **Table.** Live `setup_candidates` constraints: PK and the direction check only (the id-shape check in the CR-CC
   draft DDL is not on the table), so a set name with a colon works, as `cr-cb:round-1` does. Ids will be
   `CR-CZ:cr-cz:cp-ranges:<n>`. `dash_backfill_writer` has INSERT. No `CR-CZ` rows exist. Existing sets: CR-CB
   round-1 (63), round-2 (40), CR-CC `cc:fixture` (5).
4. **The frozen set.** CR-CP's 235 A1-primary calls are `scripts/logs/cr_cp/run_primary_trades.parquet` (main
   checkout, git-ignored; copied with `m1_train.parquet` into this worktree's logs). `scripts/cr_cz_build_gallery.py`
   rebuilds the schedule from `cr_cp_lib` (101 sessions, IM_train 65.71) and asserts it equals the frozen file:
   same 235 keys (short 108, long 127), box, leg, delta share, progress, both fills, and every frozen ±10 race result
   (`ch_res`, `op_res`). All asserts pass. Every session ≤ 2026-06-05 (B4).
5. **Sample.** `np.random.default_rng(20261027)`; 60 of the 108 short calls, then 60 of the 127 long calls, without
   replacement; cards numbered 1..120 by session and decision bar. 120 cards on 66 sessions, 2026-01-06 → 2026-06-03.
   The sample includes calls whose D4 limit never filled (the outcome says so); the set is "the calls", not "the fills".
6. **Card mapping** (all pre-decision; nothing in `meta` is known only after the close of decision bar t):

   | field | value |
   |---|---|
   | `entry_ts_pt` | close of decision bar t (CR-CC A1.2), so the replay's last candle is t |
   | `direction` | the D2 side |
   | `entry_price` | the D4 limit at the called edge (short box high − 1, long box low + 1); a limit, not a fill |
   | `stop_price` | none (CR-CP is stop-free) |
   | `extreme_price` | the approach leg's running wick extreme at the qualifying bar |
   | `confirm_ts_pt` | close of the qualifying bar q (the range has held 15 min); `test` / `retest` unused |
   | `level_*` | none: CR-CP's primary is level-free; `level_frame` = `es` (Databento ES front month, as the chart's bars) |
   | `meta.range` / `meta.zone` | the box on closes: top, bottom, height, `formed_ts_pt` = range start r0, `qualified_ts_pt` |
   | `meta` (flat) | approach (up/down, start time and price, start-confirmed time, anchor time, extreme, pts, swing_d), range number in the session, delta share % and progress (pts, buying direction) at t, contracts/pt up, down and ratio over r0..t, the D4 entry rule |
   | `outcome` | `called` and `opposite`: D4 fill status / time / price and the ±10 and ±20 races (result, exit time, exit price; C9 / C10). `pnl_pts`, `exit_*` = the called side's ±10 race; `result` = one line with all four races, which is what the panel shows on reveal |

   The panel draws the zone band (box low → high). It draws no range rectangle: that needs `meta.edge_touch_ts_pt`,
   which would read as "price touched the edge", so it is left out rather than misnamed.
7. **Blindness.** The gallery JSON holds the outcomes, so it is written to `scripts/logs/cr_cz/` (git-ignored), not
   `research/gallery/`. The API keeps `outcome` back until a review exists (CR-CC L3). Code writes no review (B8).

No stop: nothing contradicts the kickoff.

---

## Load and wrap (2026-10-27)

| commit | what |
|---|---|
| `d04414a` | spec freeze |
| `29ba8d3` | Step 0 findings |
| `b31734f` | step 1: the two reason chips in `DEFAULT_REASONS`; unit test (SetupReview 9 tests OK, 2 DB tests skipped) |
| `cd647d1` | step 2: loader `--run-cr`; `scripts/cr_cz_build_gallery.py` (loader tests 10 OK) |
| (this commit) | load record, this section |

- **Build.** `scripts/cr_cz_build_gallery.py` (read-only, train): every parity assert held → `scripts/logs/cr_cz/cr-cz-cp-ranges.json`,
  120 events, short 60 / long 60, 66 sessions, 2026-01-06 → 2026-06-03.
- **Load.** `load_setup_candidates.py … --run-cr CR-CZ` under the backfill protocol: `bt_backfill_runs`
  `dc4210ca-d7ed-4960-a6d0-07e9bd455422`, status completed, **inserted 120, skipped 0**; set `CR-CZ` / `cr-cz:cp-ranges`
  holds 120 (long 60 on 40 sessions, short 60 on 44), outcome stored on all 120, 0 reviews.
- **Blind check** (local Flask test client on this branch, GET only): the list returns 120 candidates with no `outcome`,
  `pnl_pts` or race key; the detail of card 1 has no `outcome` (entry 2026-01-06 08:43, short); the reason list carries
  both new chips.
- **No analysis, no outcome read.** Nothing was reviewed or revealed; no Playwright test was run (B8).
- **Visible to Ryan after deploy only:** the two chips are backend; the cards are in the DB now and appear in REVIEW on
  the deployed app, but the chips show only after "Dash - MonoRepo" deploys this branch.
- **Not verified:** the deployed panel (Render); the zone band on these cards in the browser.
