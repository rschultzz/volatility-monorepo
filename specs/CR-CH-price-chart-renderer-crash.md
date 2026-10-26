---
type: cr
cr_id: CR-CH
title: Price Chart tab renderer crash ("Aw, Snap! Error code 5") with Flow v2 shown
status: in-progress
run_mode: interactive
size: small
data_safety_class: read_only
dependencies: [CR-CD]
branch_name: feat/CR-CH-price-chart-renderer-oom
related_open_questions:
  - "[[price-chart-tab-renderer-crash-error-5]]"
tags: [dash, cr, frontend, price-chart, flow-v2, crash]
---

# CR-CH — Price Chart renderer crash with Flow v2 shown

## Why

Since 2026-10-25 16:50 PT, Ryan's Chrome (155, macOS) crashes the Dash Price Chart tab ("Aw, Snap! Error code: 5"). His three crash dumps and every local reproduction stop at the same instruction: renderer main thread, `EXC_BREAKPOINT` at `Google Chrome Framework` + `0x8628b7c`. That is a deliberate `CHECK` (SIGTRAP), not an out-of-memory kill. It happens while Flow v2 is shown and the price chart is being zoomed and dragged. Flow Current is unaffected.

## Scope (locked)

| # | Decision |
|---|---|
| L1 | The smallest frontend change that makes Flow v2 survive the headed repro: 10 min of hard zoom/pan in headed Chrome 155, twice. |
| L2 | Flow **Current** is not changed (no edits to `AggressorFlowPanel.jsx` or the shared time helpers). Flow v2's output (pane pixels, markers, effort labels) is unchanged. |
| L3 | No layout, colour or sizing changes. No backend changes. No DB writes. |
| L4 | A regression test: a Playwright spec that fails on `origin/main` and passes with the fix. Existing Playwright suites (cr-bv, cr-cd, cr-cd-a5) still pass with byte-identical screenshots. |

## Repro (from the vault investigation)

- Full Dash page locally (read-only DB), Flow v2, live session.
- Seeded hard interaction on the price chart: wheel zoom, pans, price-axis and time-axis drags, price-axis resets.
- Headed Google Chrome 155 crashes in 27–226 s. Tooling: `scripts/cr_ch_repro/` (local, untracked).
