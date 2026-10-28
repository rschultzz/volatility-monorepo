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

_(appended in the Step 0 commit)_
