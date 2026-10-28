---
type: cr
cr_id: CR-DH
title: GEX levels — Corrected by default on the Dashboard chart and the price chart
status: in-progress
run_mode: interactive
data_safety_class: read_only
dependencies: [CR-BT, CR-CC]
branch_name: feat/CR-DH-gex-levels-corrected-default
tags: [dash, cr, gex, level-placement, dashboard, price-chart, frontend, display-only]
---

# CR-DH — GEX levels: Corrected by default (Dashboard + price chart)

Display only, read-only DB. Branch `feat/CR-DH-gex-levels-corrected-default` off `origin/main` `0b4eb05`, own
worktree (`../volatility-monorepo-cr-dh`). Kickoff: Ryan, 2026-10-28 (chat). Frozen 2026-10-28 before any code.
Session note: vault `sessions/2026-10-28 - CR-DH — GEX Levels Corrected Default.md`.

---

## Kickoff (verbatim)

```
CR-DH — GEX levels: Corrected by default on the Dashboard chart and the price chart

Branch first, before reading anything: git fetch origin && git worktree add ../volatility-monorepo-cr-dh -b feat/CR-DH-gex-levels-corrected-default origin/main. Work only in that worktree. Run git branch -v before every commit.

Read first: CLAUDE.md; vault components/field-conventions.md; vault research/standing-rulings.md (R6: discounted_level placement is revoked, replaced by Corrected placement; C2).

CONCURRENCY: CR-DG (feat/CR-DG-absorption-side-colours) is in flight in another worktree and also edits react_price_preview/src/App.jsx.
- Don't touch CR-DG's worktree or branch.
- Only ONE DB-backed local server at a time. Before starting any local server, check that no other Dash/Flask/smoke server is running (ps, open ports). If CR-DG's is up, do code and unit tests only and wait. Never stop another session's server without asking Ryan.
- Before opening the PR: if CR-DG has merged, rebase onto origin/main, resolve any App.jsx conflict keeping both changes, and re-run the full smoke.
- Never print connection strings or env values.

CONTEXT
- Dashboard tab: apps/web/modules/gamma/callbacks.py builds "GEX by Discounted Level" from orats_oi_gamma grouped by ROUND(discounted_level) (expir_date >= trade_date, optional ticker).
- Price chart: corrected placement already exists, _corrected_gex_tables(session_date) in apps/web/modules/Ironbeam/callbacks.py (CR-BT; L = strike × ES 15:45 ET close on the prior API date ÷ stock_price, packages/shared/level_placement.py), returning (df_gex, df_by_expir, reason). The React price chart has a Levels: Current / Corrected switch (App.jsx, key 'ib-react-gex-levels-mode', default 'current'; review mode forces corrected).

SPEC
1. Dashboard: add a "Levels: Current / Corrected" switch to the GEX chart, same labels as the price chart. Default Corrected, remembered per browser (dcc persistence, local). Independent of the React chart's switch.
2. Dashboard Current = today's discounted_level chart, byte-identical to origin/main (title, axes, levels).
3. Dashboard Corrected = corrected placement: same rows and filters, puts left / calls right, levels from the corrected table with the price chart's bucket (GEX_LEVEL_BUCKET). Reuse _corrected_gex_tables. If importing it from Ironbeam callbacks makes a circular import, move it and its two fetch helpers (_fetch_gex_rows_for_placement, _fetch_es_snapshot_close) into a shared module both import. Don't copy the code. Title and x/y axis say "Corrected Level".
4. Dashboard Corrected unavailable (reason pre_forward_era / no_es_snapshot / no_rows): show the Current chart with the reason in the title, e.g. "GEX by Discounted Level (corrected unavailable: no ES snapshot)". Never draw both at once.
5. Nothing else on the Dashboard changes: layout, colours, other controls, the ES 15:01 marker.
6. React price chart: default Levels mode becomes 'corrected'. The old key 'ib-react-gex-levels-mode' already holds 'current' in browsers that have loaded the page, because the effect saves on every mount. So use a new key 'ib-react-gex-levels-mode-v2' with default 'corrected', ignore the old key, and remember the viewer's choice as before. Review mode still forces corrected (unchanged). Current stays selectable and unchanged.
7. Pre-forward-era / missing-snapshot days on the price chart in Corrected: keep CR-BT's existing handling (no new behaviour).

STANDING RULINGS (UI CR)
- C2 DEVIATION APPROVED by Ryan in chat 2026-10-28: the default of an existing control changes (Levels: Current → Corrected) on both charts, and the Dashboard gets a new switch defaulting to Corrected. Add a change-log line to standing-rulings.md saying exactly that.
- R6 honoured: Corrected is the default placement; discounted is only the explicit Current option or the labelled fallback.
- C1 honoured: no new colours.
- Visible changes limited to items 1–7.

COMMITS: spec freeze (specs/CR-DH-gex-levels-corrected-default.md, spec copied from the session note) → Step 0 → one commit per item group (Dashboard switch + Current; Dashboard Corrected + fallback; price chart default + key) → smoke + wrap.

STEP 0 (commit before implementation; no local server needed for code reading, queries read-only):
a. For 2026-10-27, list the top 10 |net| levels under discounted vs corrected placement, and the price chart's corrected lines for the same session. Dashboard corrected levels must equal the price chart's. If they don't, STOP and report.
b. Report what the price chart shows today in Corrected for a pre-forward-era date and for a no-ES-snapshot date.
c. Report-only: grep the repo for every other reader of discounted_level (backtests, features, gamma/callbacks_replacement_autoresolve_symbols.py, scripts). List each with file, purpose, and whether it's live code (imported by app.py, a cron, or a current script). Don't change any of them in this CR.

SMOKE (one DB-backed server at a time):
- Fresh browser profile: Dashboard and price chart both open on Corrected.
- Switch each to Current → survives reload; Dashboard Current pixel/byte-identical to origin/main.
- A browser with the old key set to 'current' opens the price chart on Corrected the first time.
- 2026-10-27 Dashboard Corrected levels equal the price chart's corrected lines.
- A pre-forward-era date: Dashboard shows the labelled fallback.
- App tests and existing Playwright specs green; update only assertions that test the old default, and list them.

WRAP: session note (What changed / Decisions / Open questions incl. the Step 0c list), PR opened, not merged; Ryan merges. Before removing the worktree, copy scripts/logs/cr-dh/ into the main checkout's scripts/logs/cr-dh/ and verify the copies.
```

---

## Standing rulings conformance (UI CR; read 2026-10-28)

| Ruling | Status | Note |
|---|---|---|
| A1–A11, A13–A21 | n/a | No setup, entry, exit, flow or skew test |
| A12 | honoured | Levels drawn with Corrected placement by default; colours / filters unchanged |
| B1–B8 | n/a | No test of a hypothesis; Step 0 reads 2026-10-27 (forward log) only to compare level tables, no outcome read (B4) |
| C1 | honoured | No new colours |
| C2 | **deviates — APPROVED** | Ryan in chat 2026-10-28: the default of an existing control changes (Levels: Current → Corrected) on both charts, and the Dashboard gets a new switch defaulting to Corrected. Logged in standing-rulings.md change log |
| R1–R5, R7 | not used | |
| R6 | honoured | Corrected is the default; `discounted_level` only as the explicit Current option or the labelled fallback |

## Step 0 findings

Script `scripts/cr_dh_step0/q_levels.py` (read-only; calls the Dashboard's `_fetch_gex_grouped_by_level`, Ironbeam's
`_corrected_gex_tables` and `_build_react_preview_bars_payload` directly, no server). Output
`scripts/logs/cr-dh/step0_levels.json` (machine-local).

### 0a — 2026-10-27: discounted vs corrected, and the price chart's corrected lines

P_target = ES 15:45 ET bar close on 2026-10-26 = 7717.25. Rows: discounted (Dashboard `ROUND(discounted_level)`) 4720 levels,
corrected (`_corrected_gex_tables`, bucket `GEX_LEVEL_BUCKET` = 1) 4489 levels; reason None.

Top 10 by |net| over **all** levels (the Dashboard draws every level; $bn):

| # | discounted level | net | corrected level | net |
|---|---|---|---|---|
| 1 | 7655 | -252.0 | 7682 | -250.1 |
| 2 | 8101 | +188.5 | 8029 | +188.4 |
| 3 | 7605 | -184.6 | 7632 | -187.3 |
| 4 | 8212 | +176.4 | 8138 | +176.5 |
| 5 | 8162 | +172.1 | 7531 | -172.1 |
| 6 | 7505 | -171.0 | 8088 | +171.8 |
| 7 | 8222 | +166.1 | 8148 | +167.1 |
| 8 | 7705 | -152.2 | 7732 | -153.0 |
| 9 | 8111 | +145.4 | 8039 | +149.4 |
| 10 | 8242 | +141.8 | 8168 | +141.8 |

Corrected sits ≈ 27 pts above discounted below spot and ≈ 72–74 pts below it far above spot (the CR-BT shape: discounted
over-shifts long-dated strikes upward). Price chart (Levels: Corrected, 1min, default filter) draws 63 lines on
2026-10-27 (band = session high/low ± padding, |net| ≥ threshold, spacing); its top 10 by |net|:

| # | price-chart level | net |
|---|---|---|
| 1 | 7682 | -250.1 |
| 2 | 7632 | -187.3 |
| 3 | 7732 | -153.0 |
| 4 | 7832 | +98.8 |
| 5 | 7696 | -92.4 |
| 6 | 7676 | -85.9 |
| 7 | 7848 | +84.2 |
| 8 | 7622 | -83.5 |
| 9 | 7581 | -80.9 |
| 10 | 7822 | +71.9 |

**Every one of the 63 price-chart lines equals the corrected table row at the same level** (level, call, put and net to
1e-3; mismatches: none). The price chart's top three (7682 / 7632 / 7732) are the
corrected table's #1 / #3 / #8; the far-OTM 8000+ call levels are outside the chart band. The Dashboard Corrected view
will draw that same table (same function, no copy), so its levels equal the price chart's by construction. **Gate passed.**

### 0b — what the price chart shows today in Corrected when corrected is undefined

| case | date | `_corrected_gex_tables` reason | payload `levels_notes` | corrected lines | current lines |
|---|---|---|---|---|---|
| pre-forward-era | 2024-03-15 | pre_forward_era | `[{'session_date': '2024-03-15', 'reason': 'pre_forward_era'}]` | 0 | 59 |
| no ES snapshot (2025-11-28 early close, no 15:45 ET bar) | 2025-12-01 | no_es_snapshot | `[{'session_date': '2025-12-01', 'reason': 'no_es_snapshot'}]` | 0 | 57 |

So the price chart in Corrected draws **no GEX lines** on those days (bars, EM levels etc. unchanged). The React app
does not read `levels_notes` (grep: no consumer in `react_price_preview/src`), so there is no on-chart label either.
Item 7 keeps that behaviour unchanged. With the default flipping to Corrected, a pre-2024-04-03 date opened in a fresh
browser now shows no GEX lines until the viewer picks Current — listed as an open question, not changed here.

### 0c — other readers of `discounted_level` (report only; none changed)

| file | purpose | live? |
|---|---|---|
| `apps/cron/job_orats_eod.py` | **writer**: computes `discounted_level` = strike × exp((r−q)(dte+1)/252) on ingest; also calls `compute_and_upsert_landscape` | **live** — Render cron `orats-cron` |
| `packages/shared/gex_landscape.py` | `compute_landscape` aggregates GEX by `discounted_level` into `orats_gex_landscape` (walls, `drift_target`, frame B) | **live** — called by `job_orats_eod.py`; `compute_implied_move` etc. imported by TodaySetup / SetupV2 / Analogues / day_features |
| `packages/shared/gex_landscape_api.py` | `/gex-landscape` endpoint rows; Current reads `discounted_level`, Corrected substitutes L into the same column (CR-BU) | **live** — registered via `modules/Ironbeam/routes.py` (app.py) |
| `apps/web/modules/Ironbeam/callbacks.py` | price chart Current queries (`_fetch_gex_grouped_by_level[_and_expir]`) and the `discounted_level IS NOT NULL` row filter of the corrected query | **live** (app.py) |
| `apps/web/modules/gamma/callbacks.py` | Dashboard GEX chart (this CR: becomes the Current option / fallback) | **live** (app.py) |
| `apps/web/modules/BacktestsV2/service.py` | `_load_signed_gex_for_date`: signed / gross GEX at target levels by closest `discounted_level` (`target_level_gex_bn` colouring) | **live** (app.py imports BacktestsV2) |
| `packages/shared/forward_math.py` | `compute_spx_strike` — inverse of the discounted formula (comment only; no column read) | **live** — TodaySetup, Proposals, options_cache pricing |
| `packages/shared/outcomes.py` | docstring explaining why `drift_target` (frame B) compares to ES bars | live module; comment only |
| `apps/web/modules/gamma/callbacks_replacement_autoresolve_symbols.py` | stale copy of an old Ironbeam callbacks file (header says `modules/Ironbeam/callbacks.py`) | **dead** — imported nowhere (only mentioned in specs CR-BZ / CR-CQ) |
| `react_price_preview/src/App.jsx` | comment on the Levels mode | comment only |
| `scripts/backfill_orats_oi_gamma.py` | **writer**: historical backfill of `orats_oi_gamma` incl. `discounted_level` | one-off backfill, not scheduled |
| `scripts/backfill_es_minute_features_gex_walls.py` | GEX-wall features per ES minute grouped by `discounted_level` | one-off backfill, not scheduled |
| `scripts/explore_gex_landscape.py`, `scripts/gamma_pin_study.py` | exploratory studies | not live |
| `scripts/cr_bi_followup_placebo.py`, `cr_bi_step0b_frame_test.py`, `cr_bi_step0b_resettle.py`, `cr_bt_step0.py`, `cr_bu_step0.py`, `cr_bu_support.py`, `cr_bv_step0_q3q4.py`, `cr_cb_t0_ra_r1.py`, `cr_cm_lib.py` | past CR research (frame comparisons; `cr_cm_lib` only uses `discounted_level IS NOT NULL` as the row filter for corrected placement) | not live (completed CRs) |
| `packages/shared/tests/test_forward_math.py`, `test_gex_landscape.py`, `test_gex_landscape_api.py` | unit tests of the above | tests |

Not changed in this CR. The live frame-B readers (landscape / `drift_target`, BacktestsV2 target GEX) are listed in
the session note's open questions.

### Step 0 decisions

- No circular import: `modules.Ironbeam` never imports `modules.gamma`, and app.py imports both. The Dashboard imports
  `_corrected_gex_tables` from `modules.Ironbeam.callbacks` directly; nothing is moved or copied.
