---
type: cr
cr_id: CR-CD
title: Order Flow Pane v2 — CVD vs price, delta per bar, delta share, absorption band and rule markers, using the same feature code CR-CB tests
status: created
run_mode: interactive
size: medium
estimated_days: 2-3
data_safety_class: read_only
dependencies: [CR-CA, CR-BZ, CR-CC]
soft_dependencies: [CR-CB]
depended_on_by: []
branch_name: feat/CR-CD-order-flow-pane-v2
last_reviewed: 2026-10-25
related_decisions: []
related_open_questions:
  - "[[ironbeam-es-flow-1s-undercounts-trades]]"
related_sessions: []
tags: [dash, cr, created, order-flow, aggressor, price-chart, frontend, indicator]
---

# CR-CD — Order Flow Pane v2

## Why

**Ryan (2026-10-25):** he doesn't know how to read the flow. All he has had is the current indicator, and he knows it isn't right. What the CR-CB tests calculate needs to be visible on the chart so he can see it and learn the read.

**The current pane** (`AggressorFlowPanel.jsx`, `/api/ironbeam/flow` via `_build_react_flow_payload` in `apps/web/modules/Ironbeam/callbacks.py`):
- it plots EMA(buy, 840 s) − EMA(sell, 840 s), a 14-minute-smoothed delta that mostly restates recent price direction;
- it reads `ironbeam_es_flow_1s`, which CR-CA measured at about 0.94× true volume and about 0.7× true buy − sell, with the worst gaps in busy minutes.

`es_flow_1s_databento` (CR-CA) now holds the accurate feed for 2024-04-03 → 2026-10-23, 06:15–13:00 PT.

## Locked decisions (Ryan to confirm before spec freeze)

| # | Decision | Value |
|---|---|---|
| L1 | Scope | A new flow pane mode **"Flow v2"** beside the existing indicator: a toggle "Flow: Current \| v2", **default Current**, remembered in localStorage (try/catch). With the toggle on Current, or without the new route parameter, `/api/ironbeam/flow` is byte-identical (golden test; CR-CC already recorded a flow golden). No table writes, no cron changes. |
| L2 | Data source | Per session: `es_flow_1s_databento` when it covers the session and window, otherwise `ironbeam_es_flow_1s`. Live sessions always use Ironbeam. The payload carries `source` (`databento` / `ironbeam`), and the pane shows it as a small badge. Contract from the data (CR-BZ L2). Ironbeam's ETH and post-13:00 PT data is used for those hours, since Databento covers only 06:15–13:00 PT. |
| L3 | Shared feature code | A new module `packages/shared/flow_features.py` holds the computations. CR-CB's scripts import it, so the pane shows exactly what is tested. If CR-CB's Step 0 code already defines these features, move it here unchanged and have CR-CB import it (no logic change; a test proves identical outputs on 3 sessions). |
| L4 | Series (v2) | All on 1-minute bars aligned to the price chart's bars, RTH-anchored (reset at 06:30 PT); ETH shown when the chart shows ETH, anchored at the 15:00 PT ETH open.<br>**(a) CVD:** cumulative (buy − sell), as a line.<br>**(b) Delta per bar:** buy − sell per minute, as a histogram (green/red).<br>**(c) Delta share:** (buy − sell) / volume, EMA over 3 bars, range −1…+1, as a line on its own scale.<br>**(d) Absorption band:** over a rolling 5 min, \|price move\| ÷ \|net aggressor volume\|, relative to the session's prior-20-session baseline (CR-BW L7 / CR-CB A5). Bars in the lowest 20 % of that ratio, with \|delta share\| ≥ 0.15, are shaded as "absorption" behind the histogram. |
| L5 | Markers (v2), computed causally (no lookahead) | **Divergence:** price makes a new 15-min low (high) while CVD's corresponding low (high) is higher (lower) by ≥ 10 % of the 15-min CVD range. **Push decay:** at a repeated test of a 30-min high or low, the push's delta is ≤ 0.6× the prior push's (CR-CB A5 PD logic). **Fading:** delta share's last-third vs first-third over the trailing 30 min crosses below 0 (CR-CB A5 LE logic). Each marker has a tooltip in plain English ("Sellers fading: price made a new low, but selling pressure was weaker than on the last low"). |
| L6 | Legend and help | A "?" in the pane opens a short guide: what each series means, how to read divergence and absorption against price, and the data-source caveat (live Ironbeam understates buy − sell by about 30 %, and the ratios are less affected). |
| L7 | Review mode | Respects CR-CC `until`: everything is truncated at the entry; markers after `until` are not drawn; all restored on reveal. |
| L8 | Layout | Same pane position and height as today. The v2 series stack inside it (histogram plus CVD on the left axis, delta share on a right axis). The pane gets a modest height increase only if Step 0 shows v2 is unreadable at the current height; Step 0 decides and reports it. No other visible change (CLAUDE.md visible-change discipline). |
| L9 | Performance | One session of v2 served in ≤ 1.5 s from a cold cache (Databento is about 23k rows per session). Cache per session for completed days. |

## Step 0 — diagnosis gate

1. **Current pane path:** where the frontend fetches and draws flow (`App.jsx` around lines 1326 and 1568, `AggressorFlowPanel.jsx`), and the smallest change that adds the v2 mode and toggle.
2. **Feature code:** whether CR-CB's Step 0 scripts define these features; propose the `flow_features.py` API; how equality with CR-CB will be proven.
3. **Source switching:** coverage checks for `es_flow_1s_databento` per session; behaviour at session boundaries and on days with ETH plus RTH.
4. **A worked day, 2026-05-01** (CR-CC fixture 5): v2 series and markers around the 07:39 entry and the 08:00 dip to about 7271–72, as a screenshot. Does divergence or absorption fire near the low?
5. **Byte-identity plan** for Current mode, and the payload shape for v2.

## Implementation order (one commit each)

1. `packages/shared/flow_features.py`, with unit tests on synthetic data (divergence, push decay, absorption, fading, and causality: no feature at t uses data after t).
2. Backend: a `mode=v2` parameter on `/api/ironbeam/flow` (source switching, series, markers, `until`); Current byte-identical.
3. Frontend: the toggle, the v2 rendering, tooltips and the "?" guide.
4. Smoke: 2026-05-01, 2026-04-01 and 2026-10-23 (the Ironbeam gap day, Databento-served), plus a live session if available; screenshots; Playwright toggle test.

**Deploy:** "Dash - MonoRepo".

## Kickoff prompt for Claude Code

```
Start CR-CD in its own git worktree: git fetch && git worktree add <scratch>/crcd-wt -b feat/CR-CD-order-flow-pane-v2 origin/main (confirm CR-CD is unused; rename if not). Another session runs CR-CB in its own worktree — do not touch it.
Read the vault CR "/Users/ryan/My Drive/Obsidian Vault/Anonymous/Dash/CRs/CR-CD — Order Flow Pane v2.md", components/field-conventions.md, specs/CR-CA-databento-es-flow-backfill.md, specs/CR-CC-setup-review-mode.md, specs/CR-BZ-live-tick-clean-minute-bars.md, and the CR-CB spec's A4/A5 flow-feature definitions.
Commit 1: spec freeze to specs/CR-CD-order-flow-pane-v2.md, verbatim. Commit 2: Step 0 findings (all five), with the 2026-05-01 screenshot. Read-only DB. Stop and report.
```
