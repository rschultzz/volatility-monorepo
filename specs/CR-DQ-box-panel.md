# CR-DQ — Box panel: one clean draggable panel with a verdict line, unusual-flow percentiles, faint box shading, no in-chart labels

Branch `feat/CR-DQ-box-panel`, worktree `../volatility-monorepo-cr-dq`, off `origin/main` `82028ca` (CR-DP merge, PR #124).
UI CR with one small offline baseline script; no research claims. Spec: Ryan, chat 2026-10-31 (frozen below, lightly
formatted).

## Why (Ryan, chat 2026-10-31)

In-chart box labels are hard to read and overlap other marks. Ryan likes the popup and wants it to be the one clean place
for box stats, telling him at a glance whether there's anything worth acting on (or a conflict, which means no action). A
raw Δ like +787 says nothing without context; it needs comparing against volume and against what's normal for a box that
long.

## Spec

1. **Remove in-chart box text.** Remove the per-box labels drawn on the chart (e.g. "Δ +787 · +10.0 pts · 41 m ·
   20.2k/pt") and the large hover panel on frozen boxes. Probe ticks, edge lines, absorption levels and their labels are
   unchanged.

2. **One box panel.** With Edges on and an anchor set, CR-DP's popup becomes a panel listing every drawn box (the same
   at-most-4 as the chart), one section per box.
   - Order: oldest at top, current at the bottom, highlighted (brighter border or header).
   - If boxes are hidden beyond the 4 drawn: one line at the top, "n older boxes not shown".
   - Current box section:
     - first, the verdict pill and one evidence line (item 7);
     - then its stats, collapsed by default (click to expand): Δ · Δ share (+ percentile) · pts · minutes ·
       contracts/pt (+ percentile); Top price · buyers / sellers · held · state; Bottom price · sellers / buyers · held ·
       state; probes; mixed-sources note when it applies.
   - Frozen box sections: one line each ("Box hh:mm–hh:mm · broke ↑ hh:mm · share +x% · pNN"), expandable to the same
     stats.
   - The same numbers as today's popup and box hover (one source). The current section updates live.
   - Max height about 50% of the price chart plot; scrolls inside beyond that.
   - Clean styling: the chart's dark tag background, compact type, generous line spacing, no clutter.

3. **Placement and dragging.**
   - Default position: bottom-left inside the price chart plot.
   - Draggable by its header. The position is remembered per browser and reused for later anchors and sessions.
   - Double-clicking the header returns it to the default. It always stays inside the plot on resize.
   - × hides it until the next anchor; clear / Esc clear / Edges off remove it, as today.

4. **Faint box shading.** Each drawn box gets a faint fill between its top and bottom over its time span, behind the
   candles:
   - existing magenta at very low opacity: current box about 10%, frozen boxes about 5%;
   - edge lines stay as today;
   - the set-end window shading (CR-DN) stays distinguishable (state how in Step 0).

5. **"Is this unusual?" — percentile badges.**
   - a. For each box, show Δ share and contracts/pt as percentiles against a baseline of RTH windows of similar length
     (length buckets, e.g. 5 / 10 / 15 / 30 / 60 / 90+ min, matched to the box's minutes), e.g. "share +4.1% · p94",
     "20.2k/pt · p71".
   - b. Tiers by brightness, no new colours: below p80 = normal text; p80–p95 = "notable" (brighter); above p95 =
     "strong" (brightest, bold). The cut-offs are display defaults in ⚙, labelled "display aid — untested".
   - c. Baseline: train sessions only (≤ 2026-06-05; B4: no holdout read), computed once by a script into a small stored
     table or JSON (percentiles per length bucket, separately for |Δ share| and contracts/pt; split by time of day only if
     Step 0 shows it matters a lot). Refreshed only by re-running the script.
   - d. Source matching: live boxes use Ironbeam (es_flow_price_1m), which understates delta by about 30% vs Databento.
     Step 0 must state which source the baseline uses and either build it from the same source as the live data or apply
     a documented adjustment, showing both on 3 example boxes.
   - e. Hover on a badge: what the percentile means, the bucket used, baseline sessions and date range.

6. **Nothing else changes:** delta-share line, Edge effort strip, header, ⚙ (apart from the new settings above and in 7).

7. **Verdict line** (Ryan's draft rules, applied mechanically; display aid, untested).
   - a. The current box's section starts with one verdict pill plus one short evidence line.
   - b. Inputs (all already computed): Δ share and its percentile (5), contracts/pt percentile, price progress, edge
     counts by side, absorption levels in or within 2 pts of the box, and where price sits in the box. Checked in this
     order:
     - **NO TRADES — event day:** A22 (FOMC decision day; hard-coded 2026 FOMC date list in config until an events table
       exists).
     - **CONFLICT → NO ACTION:** flow lean and absorption point opposite ways (A24), e.g. Δ share ≥ notable positive while
       a sellers-absorbed level sits at the bottom, or the mirror. List the conflicting items.
     - **SHORT READ at top** (A19 + A23): price within 2 pts of the top; buyers leaning (Δ share ≥ notable, positive)
       and/or a buyers-absorbed level at the top; no advance (contracts/pt ≥ notable, or progress ≤ 2 pts); nothing
       contradicting.
     - **LONG READ at bottom:** the mirror.
     - **WATCH BREAK ↑/↓** (idea only, labelled "idea"): the pushing side winning at an edge: Δ share ≥ notable in its
       direction, contracts/pt below notable (moving easily), no absorption against it. Text: "wait for an accepted
       break".
     - Otherwise: **NO ACTION — nothing notable.**
   - c. Every verdict shows, on hover, the rule IDs used, the exact thresholds and "draft rules, untested". All
     thresholds live in one config object, editable in ⚙ under "Verdict rules".
   - d. Colours: existing palette only (C1). NO ACTION in muted slate; reads in brighter text; no red / green.

## Standing rulings conformance

Read `research/standing-rulings.md` (2026-10-31 copy: last ruling A22; last change-log line CR-DP deviation (C2),
2026-10-29) and `research/order-flow-reading-notes.md` (draft rules A23–A28) before the freeze.

| Ruling | Status | Note |
|---|---|---|
| A1–A18, A20, A21 | n/a | no range, level, setup, entry, exit or outcome logic changes; boxes are CR-DN's, unchanged |
| A19 — in-range confirmation (starting definition, untested) | honoured (display only) | the SHORT / LONG READ pills apply A19's wording mechanically as a display aid, labelled "draft rules, untested"; nothing tested or shortlisted |
| A22 — skip FOMC decision days | honoured (display only) | NO TRADES pill on FOMC decision days (hard-coded 2026 list); the baseline also leaves FOMC days out (event days not pooled, A22 "Tests") |
| A23–A28 (Ryan's drafts) | n/a — not rulings | A23 / A24 used as drafts in the verdict, labelled so; A25–A28 not used |
| B1–B3, B5–B7 | n/a | no hypothesis tested; percentiles describe flow, not results. B5: the baseline's source is matched to the box's source (Step 0, 5d), no pooling of Databento and Ironbeam |
| B4 — data split | honoured | baseline from train sessions only (≤ 2026-06-05); no holdout, R-a, R-b or forward-log outcome read; no outcome read anywhere |
| B8 — Code smoke labels are never Ryan's | honoured | no review labels written; smoke runs against a read-only backend |
| C1 — no green / red; Dash palette | honoured | existing magenta (shading), slate, light grey / white text, tag background; tiers by brightness only |
| C2 — new display features off by default, existing views unchanged | **deviates — APPROVED by Ryan in chat 2026-10-31** | visible changes limited to items 1–5 and 7 (removes in-chart box labels and the frozen-box hover panel, changes the popup into the box panel, adds box shading). Change-log line added to `standing-rulings.md` |

Nothing in "Revoked / superseded" (R1–R7) is used. Percentiles, tiers and verdicts are display aids with no research
meaning.

## Step 0 (commit before implementation)

- Where the in-chart box labels and the frozen-box hover panel are drawn.
- How CR-DP's popup is built and positioned (reuse it).
- How to fill boxes behind candles with the existing primitives, and how the set-end shading stays distinct.
- Baseline: source and coverage, length buckets, whether time of day needs a split, and the 3-box source comparison (5d).
- Where each verdict input comes from, so the panel computes nothing twice.

## Commits

spec freeze → Step 0 → one commit per item (1; 2+3; 4; 5 incl. baseline script; 7) → smoke + wrap.

## Smoke (read-only backend)

- 2026-10-30, anchor 11:57 (2+ boxes): no in-chart box text; panel opens bottom-left, one section per box, current last
  and highlighted, verdict on top; numbers equal origin/main's popup and hover for the same boxes.
- Drag, reload: position kept. Double-click header: back to bottom-left.
- Boxes visibly shaded (current stronger than frozen), candles on top; set-end shading still distinct.
- More than 4 boxes: the "older boxes not shown" line.
- Percentiles: the 3 example boxes match the baseline by hand; tiers render as specced.
- Verdicts: 3 hand-checked boxes on 2026-10-29/30 match the rule table (one read, one conflict, one nothing-notable);
  2026-10-28 shows NO TRADES (FOMC).
- Existing specs green; only assertions on the removed labels / hover updated (list them).

## Wrap

Session note, PR (don't merge), logs copied. Don't deploy. Only ONE DB-backed local server at a time; check first, never
stop another session's server without asking Ryan. Never print connection strings: catch DB errors and print the error
type only.
