---
type: cr
cr_id: CR-CC
title: Setup Review Mode on the Price Chart — load candidate setups, replay each one to its entry, label Take / Skip / Unsure with reasons, then reveal the outcome
status: created
run_mode: interactive
size: medium
estimated_days: 2-3
data_safety_class: writes_reviewed
dependencies: [CR-BV, CR-BZ]
soft_dependencies: [CR-CB]
depended_on_by: []
branch_name: feat/CR-CC-setup-review-mode
last_reviewed: 2026-10-25
related_decisions: []
related_open_questions: []
related_sessions: []
tags: [dash, cr, created, review, labeling, discretion, price-chart, frontend]
---

# CR-CC — Setup Review Mode on the Price Chart

## Why

**Ryan (2026-10-25):** the goal is rules for his discretion, so that mid-session he doesn't convince himself of patterns that aren't real, and so that he stops taking the "no" setups he sees all the time.

**The problem.** The order-flow and level tests (CR-BU/BW/BX/BY) never showed Ryan the setups they traded. A test may have refuted a definition that didn't match his eye.

**What this CR adds.** Every pattern CR gets a visual review step:
- Ryan labels candidate setups **blind to the outcome**, on the same chart he trades from;
- his labels tighten the rules;
- then outcomes are compared for Take vs Skip, which is the real-vs-imagined check on his own discretion.

**Existing machinery to reuse.** The Backtests trade log already sends the price chart to a trade:
- `react_backtests_v2/src/components/TradeLog.jsx` POSTs `/api/trade-log/annotation-state`;
- `PriceChart.jsx` polls it every 1.5 s and opens `TradeAnnotationPanel` on that date.

The state is in-memory (`_annotation_state` in `apps/web/modules/TradeLog/routes.py`). This CR adds a persistent review queue next to it, rather than overloading it.

## Locked decisions (Ryan to confirm before spec freeze)

| # | Decision | Value |
|---|---|---|
| L1 | Scope | (a) two new tables; (b) API routes; (c) a REVIEW panel and replay mode in `react_price_preview`; (d) a loader for candidate JSON files. No change to existing routes' default responses (byte-identical tests as in CR-BT/BV), and none to research tables. |
| L2 | Tables | `setup_candidates` (id text PK `<cr>:<set>:<n>`, cr_id, set_name, session_date, cell, direction, level_price, level_net_bn, level_gross_bn, entry_ts_pt, entry_price, stop_price, extreme_price, test_ts_pt, confirm_ts_pt, retest_ts_pt, meta jsonb, outcome jsonb, loaded_at). `setup_reviews` (candidate_id FK, label in {take, skip, unsure}, reasons text[], comment text, reviewed_at, revealed_at). Insert-only, plus UPDATE of `revealed_at`. DDL applied once by Code with `DATABASE_URL` after Ryan's go; grants to the web role as needed. |
| L3 | Loader | `scripts/load_setup_candidates.py` reads a gallery JSON (format below) into `setup_candidates`. `outcome` (P&L / race result) is stored in the table and **never** sent to the frontend before a review exists for that candidate. |
| L4 | API | `GET /api/setup-review/candidates?cr=&set=&cell=&status=unreviewed\|reviewed\|all` (list, no outcomes). `GET /api/setup-review/candidates/<id>` (detail; includes `outcome` only if reviewed). `POST /api/setup-review/candidates/<id>/review` (label, reasons, comment). `POST …/<id>/reveal` (sets `revealed_at`, returns the outcome). `GET /api/setup-review/summary?cr=&set=` (counts, Take vs Skip outcome stats, reasons tally). |
| L5 | Replay mode | `/api/ironbeam/bars` gains an optional `until=<ISO PT>`: bars after it are omitted, and so is the live overlay. Absent = today's behaviour, byte-identical. The flow route gets the same `until`. The landscape and levels are prior-day (no lookahead) and stay as they are. |
| L6 | REVIEW panel | A REVIEW chip next to SMILE / SIGNALS / GEX / ANALOGUES opens a side list, filterable by CR / set / cell / status. Clicking a candidate (or **N** / **P** for next / previous):<br>• loads that session in **Corrected** levels mode, with `until = entry_ts_pt`;<br>• draws the level (line + label with net/gross $bn), the extreme H/L, the entry and the stop as price lines, and markers at the test / confirm / retest / entry times;<br>• shows metadata (cell, overshoot, any CR-specific fields).<br>No P&L or post-entry bars are shown. |
| L7 | Labelling | Buttons **Take / Skip / Unsure** (keys **T / S / U**). Reason chips, multi-select, with a free-text "add reason" that becomes a chip for later candidates: overshoot too big, no clean extreme, level too weak, trend day, chop, too late in day, flow against, other. A comment box. Save is required before reveal. |
| L8 | Reveal | After Save, a **Reveal** button (key **R**) re-fetches without `until` (the full session), draws exit markers and the outcome from `outcome` (e.g. +34 pts / stopped −3.5), and records `revealed_at`. A review can't be edited after reveal; a new review can be added and is flagged as post-reveal. |
| L9 | Summary view | In the panel: per CR/set, counts by label, and for revealed candidates the mean outcome for Take vs Skip vs Unsure, with n, plus the reasons tally on Skips. Chat reads `/summary` (or the tables) to propose rule amendments. |
| L10 | Gallery JSON format | `{cr_id, set_name, generated_at, events: [{id, session_date, cell, direction, level_price, level_net_bn, level_gross_bn, entry_ts_pt, entry_price, stop_price, extreme_price, test_ts_pt, confirm_ts_pt, retest_ts_pt, meta: {...}, outcome: {exit_ts_pt, exit_price, pnl_pts, result, by_exit: {...}}}]}`. Bars are **not** embedded: the chart loads them from the DB with `until`. |
| L11 | Visible-change discipline | Only the REVIEW chip and panel, replay truncation when a candidate is active, and the candidate price lines and markers. Nothing else changes. |

## Step 0 — diagnosis gate

1. **Annotation path.** Confirm how the trade log drives the chart (annotation-state poll, the date switch) and say whether REVIEW should reuse that poll or set the date directly in the price app. It is the same app, so a direct state change is expected.
2. **`until` feasibility.** Where `/api/ironbeam/bars` builds bars and segments, and the smallest change that truncates them, including the live overlay and the flow route. The byte-identity plan.
3. **Price-line API.** Confirm lightweight-charts 5 `createPriceLine` and series markers for the candidate overlays, and that they don't collide with the existing GEX segments.
4. **DDL** text and roles: which DB role the web service uses, and the grants needed.
5. **First data:** whether `research/gallery/cr-cb-stage1.json` (and `cr-bu.json`, `cr-bw.json`) exist yet; if so, validate them against L10.

## Implementation order (one commit each)

1. DDL and loader (L2, L3, L10), with tests.
2. API routes (L4), with tests: outcome never present before review.
3. `until` replay (L5), with byte-identity tests.
4. REVIEW panel, labelling and reveal (L6–L8).
5. Summary view (L9).
6. Smoke: load the CR-CB gallery (or a 5-event fixture if it isn't ready); Playwright walks label → reveal on 2 candidates; screenshots.

**Deploy:** "Dash - MonoRepo" only.

## Kickoff prompt for Claude Code

```
Start CR-CC in its own git worktree: git fetch && git worktree add <scratch>/crcc-wt -b feat/CR-CC-setup-review-mode origin/main (confirm CR-CC is unused; rename if not).
Read the vault CR "/Users/ryan/My Drive/Obsidian Vault/Anonymous/Dash/CRs/CR-CC — Setup Review Mode on the Price Chart.md", components/field-conventions.md, specs/CR-BV-landscape-chart-one-space.md, specs/CR-BZ-live-tick-clean-minute-bars.md, and the TradeLog module (apps/web/modules/TradeLog, react_backtests_v2/src/components/TradeLog.jsx, react_price_preview TradeAnnotationPanel/PriceChart annotation code).
Commit 1: spec freeze to specs/CR-CC-setup-review-mode.md, verbatim. Commit 2: Step 0 findings (all five). No DB writes, no DDL yet. Stop and report.
```
