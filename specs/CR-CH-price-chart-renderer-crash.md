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

## Step 0 — diagnosis findings (2026-10-26)

All runs: headed Google Chrome 155.0.8059.85 on Ryan's Mac, the full Dash page locally, read-only DB, the same seeded hard sequence, Flow v2 with Effort off, 600 s budget. "Clean" means 10 min without a crash, twice.

**Same crash as Ryan's.** His three dumps (2026-10-25 16:50:39 / 16:51:23 / 17:26:43 PT) and every reproduction share one site: `EXC_BREAKPOINT`, `Google Chrome Framework` + `0x8628b7c`.

**What does not matter** (each still crashes):
- Effort on or off; swing 8 or 12; the build before A5 (`0bb6003`).
- The v2 request schedule: poll off, 60 s poll, no abort/overlap, visible session only. All v2 requests finish within ~10 s of load.
- v2 data at all: `nofetch` crashed 2/2.
- The v2 pane's chart, series, price lines, markers, session bands, time sync, decorations and absorption band.
- Pane DOM: a bare empty container (`p-container`) still crashes; so do header-only and stage-only.
- GPU raster, accelerated 2D canvas, GPU compositing; Maglev or Turbofan off alone.
- Geometry stays sane: nothing > 32,767 px, NaN, Infinity or negative.

**What does matter:**
- `nopane` (App renders no v2 pane): clean 2/2.
- `m-noalign` (the pane's `aligned` and `snapped` memos return `[]`): clean 2/2.
- Disabling just one of the two memos still crashes (align on: 328 s; snap on: 49.8 s).
- Chromium 130 (non-official): clean 2/2. The CHECK is reachable in Chrome 155, not 130.

**Root cause in our code.** `App.jsx` passes `payloads={flowV2Payloads.map((p) => p.payload)}`, a new array on every App render. In `AggressorFlowV2Panel`:
- `allBars` / `allMarkers` are memoized on `payloads`, so they change on every App render;
- `aligned = alignV2ToCandles(allBars, candles)` and `snapped = snapMarkers(allMarkers, candles)` then recompute on every App render as well.

Measured in the baseline crash run, during drag/zoom:
- `alignV2ToCandles` ran 3–6×/s (~100 ms each, 2652 candles) and `snapMarkers` 3–6×/s (~50 ms each), while `candles` itself changed 4 times in the whole run;
- that is **500–950 ms of main-thread work per second**;
- each call builds thousands of fresh `Intl.DateTimeFormat`s (`partsForZone` constructs one per timestamp).

The Current pane gets state arrays whose identity is stable, so its memos don't re-run.

A minimal page building 13 M `Intl.DateTimeFormat`s in 300 s did not crash. The CHECK needs the combination with live interaction and rendering; the exact Chrome-side assertion stays unidentified (no public macOS symbols).

**Fix candidates, gated in a debug build of `f302498`:**

| candidate | change | result |
|---|---|---|
| `fix-memo` | App memoizes the payloads array: `useMemo(() => flowV2Payloads.map((p) => p.payload), [flowV2Payloads])` | **clean 2/2** (2626 / 2689 interactions) |
| `fix-fmt` | one cached `Intl.DateTimeFormat` inside the v2 file | clean 1/1 (2017 interactions) |

**Chosen fix: `fix-memo`.**
- It removes the redundant recomputation at its source with one memo in `App.jsx`.
- It doesn't touch Current or any shared helper (L2).
- v2's output is identical: same data, computed once per real change.

`fix-fmt` stays a possible follow-up for performance, not needed for the crash.

## Implementation order

1. `App.jsx`: memoize the Flow v2 payloads passed to the pane.
2. Regression spec `tests/e2e/cr-ch.spec.js`: with Flow v2 shown, interaction that re-renders App leaves the pane's `payloads` prop and its `aligned` / `snapped` memos referentially unchanged.
3. Smoke: the new spec fails on `origin/main` and passes on the branch. Existing suites cr-bv, cr-cd and cr-cd-a5 pass with byte-identical screenshots. Headed repro on the branch build: clean 10 min ×2 (and the unfixed `origin/main` build still crashes).

## Smoke (2026-10-26)

**Regression spec** `tests/e2e/cr-ch.spec.js` (dev server, `cr_cd_smoke_backend.py` on 8062, 2026-05-01, Flow v2):

| frontend | result |
|---|---|
| `origin/main` `8f1226b` (unfixed) | **fails** at `expect(cmp.samePayloads).toBe(true)`; the re-render assertion before it passes |
| branch (fixed) | **passes** (10.6 s) |

**Existing suites on the branch:**
- cr-bv 9/9, cr-cd 4/4, cr-cd-a5 4/4 pass. cr-cd-a5's own byte comparisons against the pre-A5 frontend (`0bb6003`) pass.
- 9 screenshots in `specs/assets/` come out different from the committed copies (5 × cr-bv, 4 × cr-cd v2). Unfixed `origin/main` produces the **same bytes** as the branch for all 9, so the drift is environmental (data and frontend changes since the assets were captured), not this change. The assets were left as committed.
- Read-only throughout: the review routes ran on a read-only connection, not `cr_cc_smoke_backend.py`.

**Headed repro.** Google Chrome 155, full Dash page locally on a read-only DB, live session, production builds, same seeded hard zoom/pan sequence, date changed twice:

| build | Flow | Effort | result |
|---|---|---|---|
| `origin/main` `8f1226b` (control) | v2 | off | **crash 84.9 s** |
| branch `4e4f88d` run 1 | v2 | off | **clean 600 s** (2692 interactions) |
| branch `4e4f88d` run 2 | v2 | off | **clean 600 s** (2763 interactions) |
| branch `4e4f88d` | v2 | on, swing 8 (26 labels) | **clean 600 s** (2752 interactions) |

Before the fix, Flow v2 crashed in every headed run under this sequence (27–226 s, 20+ runs across builds and gates).

## Deferred

- `partsForZone` (shared with Current) builds a new `Intl.DateTimeFormat` per timestamp. Caching it would make each alignment ~10× cheaper (`fix-fmt` was clean 1/1). Not needed for the crash, and out of scope here because it touches Current's helper (L2).
- The Chrome-side `CHECK` (`+0x8628b7c`) stays unidentified. Ryan's uploaded crash reports could go to crbug if it shows up elsewhere.
