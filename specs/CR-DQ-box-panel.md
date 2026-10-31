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
- **Smoke addition (Ryan, chat 2026-10-31) — report only.** Replay the panel on 2026-10-30 at Ryan's entry times 06:11:46, 07:06:47, 07:49:17, 09:22:50, anchored at the start of the range he was trading (06:11 → the overnight box from ~05:00; 07:06 → 06:58; 07:49 → 07:40; 09:22 → 09:15), and record the verdict and its evidence for each. Thresholds are NOT changed to make these match; a disagreement with chat's expectation is reported as is.

## Wrap

Session note, PR (don't merge), logs copied. Don't deploy. Only ONE DB-backed local server at a time; check first, never
stop another session's server without asking Ryan. Never print connection strings: catch DB errors and print the error
type only.

## Step 0 — findings (2026-10-31, on `82028ca`)

### S0.1 In-chart box labels and the frozen-box hover panel (item 1)

- **Labels:** `EdgeLinesPrimitive.paint` (`components/flowV3Primitives.js:472–486`, CR-DN item 3b) draws `b.label` on a tag
  pill inside each drawn box's top-left corner. `b.label` = `boxLabelText(readout)`, set in PriceChart's `v3Edges` memo
  (`PriceChart.jsx:2704`). Removal: drop that block and the `label` field. Edge lines and probe ticks / "probe hh:mm" text
  are drawn by the same primitive and stay.
- **Hover panel:** `EdgeLinesPrimitive` hit-tests the box edge lines (`_hit`, `onHover`) → `setV3BoxHover`
  (`PriceChart.jsx:2723, 2834, 2841`) → the `flow-v3-box-tooltip` div (`:3892–3925`). Removal: drop `onHover` / `_hit`,
  the state, its relink and the div. The probe hover (`onProbeHover`, `flow-v3-probe-tooltip`) is unchanged.
- Absorption levels (`AbsorptionLevelsPrimitive`) and their labels are untouched.

### S0.2 CR-DP's popup (items 2 and 3 reuse it)

- Built in PriceChart (`:2725–2813`, JSX `:3927–3960`): `v3CurrentBox` = the non-frozen entry of `v3Edges.boxes`; text
  from `flowV3.boxPopupLines(box, hhmm, segments)` (= `boxLabelParts` + `edgeStateText` + `mixedSourcesText`), i.e. the
  same objects the label and the hover read. Every drawn box already carries `box` = drawnBoxes' box + `readout`
  (`v3Segments.readouts[k]`), `ended`, `pending`, `dropped`, so the panel's per-box sections need no new computation.
- Position today: a rAF loop follows the box's top-right (`timeToCoordinate` / `priceToCoordinate`), clamped to the plot
  (`pane` rect = the time scale's width × pane 0's height, in stage px); drag by the body (pointer capture), kept until the
  next anchor.
- Item 3 changes: the rAF loop keeps measuring the plot rect only (needed for the clamp on resize); default place =
  bottom-left of the plot (4 px inset); drag by the header only; the position stored in localStorage
  (`ib-react-box-panel-pos`) as px from the plot's left and bottom edges (so it stays bottom-anchored when the plot
  height changes), re-clamped every frame; double-click the header removes the key. The "reset on a new anchor" goes
  (spec: reused for later anchors); × still hides until the next anchor.
- Item 2 text: a new pure `flowV3.boxPanelSections(boxes, …)` builds every section from the same fields; the current
  section's expanded stats reuse `boxPopupLines` (head / top / bottom / mixed) verbatim plus the hover's "n probes" line,
  so the numbers are origin/main's popup and hover by construction. Max height `0.5 × plot height`, `overflow-y: auto`.

### S0.3 Box shading (item 4) and the set-end shade

- New `BoxShadePrimitive` in `flowV3Primitives.js`, `zOrder 'bottom'` (behind the candles, like `HeatmapPrimitive` and
  `WindowShadePrimitive`): for each drawn box, `fillRect` from its `from` to `to` x and between `priceToCoordinate(top)` /
  `(bottom)`, existing magenta `232,121,249` at α 0.10 (current) / 0.05 (frozen). Fed from the same `v3Edges` data as
  `EdgeLinesPrimitive`; attached after the heatmap so it sits over the heat and under the candles.
- **Set-end shade stays distinct:** `WindowShadePrimitive` is **slate** (`148,163,184` α 0.08) and spans the **full plot
  height** over anchor → end; the box fills are **magenta** and only between each box's top and bottom (and bounded by its
  dashed edge lines). Where they overlap the two blend; the slate band stays visible above and below the boxes.

### S0.4 Baseline (item 5) — `scripts/cr_dq_step0.py`, output `scripts/logs/cr-dq/step0.json`

**Correction to 5d's premise.** A box's Δ and volume do **not** come from `es_flow_price_1m`. They are the v2 rows
(`flowV3Segments` → `anchorReadout`): RTH from `es_flow_1m_databento` for sessions ≤ 2026-06-05 (CR-CD A1.1), else
Ironbeam `ironbeam_es_flow_1m` (+ the last 5 min from `ironbeam_es_flow_1s`); ETH always Ironbeam. `es_flow_price_1m` feeds
the edge contracts and the absorption marks (cells payload). So live boxes (≥ 2026-06-06) are Ironbeam `ironbeam_es_flow_1m`
and train-date boxes on the chart are Databento.

**Window definition** = the box readout: anchor minute a, length L; Σ delta (buy − sell) and Σ volume over a+1 … a+L;
progress = close[a+L] − close[a]; |Δ share| = |Σ delta| ÷ Σ volume; contracts/pt = Σ volume ÷ max(|progress|, 0.25) (one
tick floor: a flat box is the most effortful, not undefined; the panel still shows "–/pt" as today). Every window with all
minutes present, every anchor, L = 3 … 240.

**Coverage (train only, B4).** NYSE sessions 2026-01-02 → 2026-06-05 = 107; left out FOMC decision days (A22: 01-28,
03-18, 04-29) and the March expiry week 03-16 → 03-20 (Ironbeam flow on the dying H26) → 100 eligible. Databento 100
sessions, no missing RTH minute. Ironbeam 99 sessions; 18 have missing minutes (trades-ws stalls, max 277 min), windows
touching them are skipped. Per session the contract with the larger RTH volume. R-a, R-b, holdout and forward log: not read.

**Length buckets** (box minutes → bucket): 5 = 3–7 · 10 = 8–12 · 15 = 13–22 · 30 = 23–45 · 60 = 46–75 · 90+ = 76+. Boxes under
3 min: no badge. |Δ share| falls with length as expected (Ironbeam p80: 6.7 % at 5 min → 2.1 % at 90+); contracts/pt rises
(p80 10.8k → 54.8k).

| bucket | Ironbeam \|share\| p50 / p80 / p95 | Ironbeam c/pt p50 / p80 / p95 | Databento \|share\| p50 / p80 / p95 | Databento c/pt p50 / p80 / p95 |
|---|---|---|---|---|
| 5 | 3.4 / 6.7 / 11.1 % | 4.1k / 10.8k / 32.5k | 4.1 / 8.0 / 12.9 % | 4.4k / 11.6k / 34.9k |
| 10 | 2.6 / 5.2 / 8.5 % | 5.9k / 15.5k / 53.7k | 3.1 / 6.1 / 9.8 % | 6.3k / 16.6k / 57.7k |
| 15 | 2.1 / 4.2 / 6.9 % | 7.8k / 20.5k / 76.8k | 2.5 / 4.9 / 7.9 % | 8.4k / 22.0k / 82.5k |
| 30 | 1.7 / 3.4 / 5.4 % | 10.7k / 28.0k / 111.6k | 1.9 / 3.8 / 6.0 % | 11.5k / 30.1k / 120.0k |
| 60 | 1.4 / 2.8 / 4.4 % | 13.8k / 35.5k / 141.3k | 1.5 / 2.9 / 4.7 % | 14.8k / 38.1k / 151.2k |
| 90+ | 1.1 / 2.1 / 3.3 % | 21.0k / 54.8k / 218.9k | 1.1 / 2.1 / 3.4 % | 22.4k / 58.3k / 231.9k |

**Time of day** (window start: open 06:30–07:30, mid 07:30–11:00, late 11:00–13:00). Where the pooled p80 / p95 cut-offs
land inside each group (Ironbeam; Databento within 1–2 points):

| bucket | \|share\| open | \|share\| mid | \|share\| late | c/pt open | c/pt mid | c/pt late |
|---|---|---|---|---|---|---|
| 5 | p88 / p98 | p80 / p95 | p76 / p94 | p77 / p92 | p81 / p96 | p79 / p95 |
| 15 | p88 / p98 | p80 / p95 | p76 / p93 | p78 / p94 | p81 / p95 | p79 / p95 |
| 30 | p87 / p98 | p80 / p95 | p75 / p93 | p78 / p95 | p81 / p95 | p80 / p95 |
| 60 | p87 / p98 | p79 / p95 | p76 / p93 | p81 / p96 | p80 / p95 | p80 / p95 |
| 90+ | p91 / p99 | p77 / p94 | p69 / p88 | p82 / p96 | p79 / p95 | p83 / p96 |

- **|Δ share| needs the split:** the opening hour is one-sided less often. A pooled "notable" (p80) is p87–91 within the
  open, and p69–76 late. So a pooled badge would flag about 12 % of opening boxes and 24–31 % of late ones as notable,
  where the spec means 20 %. **Contracts/pt doesn't:** within 3–4 points everywhere.
- **Recommendation (needs Ryan's OK, item 5c):** split |Δ share| by time of day (box start in open / mid / late);
  contracts/pt pooled. The JSON stays small (2 sources × 6 buckets × (3 + 1) × 101 values).

**5d — source matching.** **Recommendation: same-source baselines, no adjustment.** An Ironbeam box reads the Ironbeam
baseline, a Databento box (train dates on the chart) the Databento one. A box spanning two sources (the mixed-sources
note), or in ETH, gets no badge.
- CR-CA's ratios (delta slope 0.71, volume 0.943) over-correct on these 99 sessions: Ironbeam |share| here is ~0.83–0.97 ×
  Databento's at equal percentiles, not 0.75 ×. Against the Databento baseline (85k windows, both sources, same window),
  Ironbeam boxes' share percentiles move **−3.9 points raw / +4.6 adjusted** on average (contracts/pt −2.4 raw / −0.5
  adjusted). Same-source is unbiased by construction.
- The same window read from the two feeds differs in **share percentile by a median 8.7 points (p90 28)**: per-box noise
  from Ironbeam's dropped prints (which follow the dominant side), not a bias. Contracts/pt: median 0.4 points. An Ironbeam
  share badge near a tier cut-off can therefore flip; the badge hover says so.

Three fixed train windows, chosen before looking at values (percentile vs own-source baseline; then Ironbeam read
against Databento's baseline raw / CR-CA-adjusted):

| window | bucket | Ironbeam Δ / vol / pts → share · c/pt | p (own) | Databento Δ / vol / pts → share · c/pt | p (own) | Ironbeam vs Databento baseline raw / adjusted |
|---|---|---|---|---|---|---|
| 02-10 07:05 + 25 m | 30 | −1,001 / 85,659 / +4.75 → −1.17 % · 18.0k | share p37 · c/pt p69 | −1,718 / 92,876 / +4.75 → −1.85 % · 19.6k | p49 · p69 | p33 · p67 / p42 · p69 |
| 04-15 09:12 + 12 m | 10 | +320 / 17,630 / +4.5 → +1.82 % · 3.9k | p36 · p30 | +725 / 18,923 / +4.5 → +3.83 % · 4.2k | p59 · p30 | p31 · p27 / p40 · p30 |
| 05-20 10:40 + 55 m | 60 | −131 / 62,331 / +1.0 → −0.21 % · 62.3k | p8 · p89 | −154 / 68,938 / +1.0 → −0.22 % · 68.9k | p8 · p89 | p7 · p88 / p10 · p89 |

(Percentiles above are against the pooled baseline; with the time-of-day split they are recomputed in the item-5 commit.)
- Ironbeam's undercount since the CR-BZ ingest work (2026-10) is not measured: the 2026 train ratio is assumed to hold
  live. Noted on the badge hover ("Ironbeam baseline: 2026-01-02 → 06-05").

**Storage:** the item-5 script writes `react_price_preview/src/data/boxBaseline.json` (committed; the frontend imports it,
no backend change). Lookup = linear interpolation in the 101-point quantile grid (the same `np.interp` as the script, so a
hand check matches).

### S0.5 Verdict inputs (item 7) — each computed once

| Input | Source (already computed) | Note |
|---|---|---|
| Δ share | `readoutShare(v3Segments.readouts[k])` (App `flowV3Segments`) | the panel's share |
| Δ share / c/pt percentiles | new `flowV3.boxPercentiles(readout, startUtc, source)` once per drawn box, in the `v3Edges` memo | baseline JSON; null for ETH / mixed / < 3 min |
| price progress | `readout.dPrice` | signed: "no advance" for a short = dPrice ≤ +2; for a long ≥ −2 |
| edge counts by side | `box.top.contracts / opposite`, `box.bottom.contracts / opposite` (`edgeReadout`) | shown in the evidence; the rule table doesn't threshold them |
| absorption levels in / within 2 pts of the box | `absorptionLevels(v3Payload, …)` — today only computed with the Absorption levels layer on | moved to one memo computed whenever Edges or Levels is on; the layer draws it, the verdict reads it. Same top-8 set as drawn. Counted if unbroken at the box's last bar and started ≤ it; "at the top" = within 2 pts of the top (likewise bottom) |
| where price sits in the box | last close of the box's last bar | `edgeReadout` gains `lastClose` (the loop already has `bars.c[last]`); within 2 pts of top / bottom |
| event day | trade date ∈ `FOMC_2026` (config in `flowV3.js`) | 01-28, 03-18, 04-29, 06-17, 07-29, 09-16, 10-28, 12-09: last day of each 2026 meeting, checked against federalreserve.gov/monetarypolicy/fomccalendars.htm 2026-10-31 (first three also in `scripts/cr_bo_common.py`) |

Rule table as implemented (one config object `BOX_RULES_DEFAULTS`: tiers notable 80 / strong 95, near 2 pts, progress 2
pts, absorption distance 2 pts; ⚙ groups "Percentile tiers" and "Verdict rules"); "notable" in the rules is the tier
cut-off:
1. FOMC day → NO TRADES — event day (A22).
2. CONFLICT → NO ACTION (A24): share p ≥ notable and share > 0 with a sellers-absorbed level in / near the box, or share
   < 0 with a buyers-absorbed level; lists both items.
3. SHORT READ at top (A19 + A23): last close within 2 of top; (share > 0 with p ≥ notable) or a buyers-absorbed level at
   the top; (c/pt p ≥ notable) or dPrice ≤ +2. LONG READ: mirror.
4. WATCH BREAK ↑ (idea): last close within 2 of top; share > 0 with p ≥ notable; c/pt p < notable; no buyers-absorbed level
   at the top. ↓ mirror. Text "wait for an accepted break".
5. Else NO ACTION — nothing notable.

### S0.6 Points for Ryan before implementation

1. **Time-of-day split** for |Δ share| (recommended, S0.4); contracts/pt pooled.
2. **Baseline source:** same-source (Ironbeam for live, Databento for train dates on the chart), no CR-CA adjustment
   (recommended, S0.4). The spec's "es_flow_price_1m" is the cells source, not the box Δ source.
3. **Conflict scope:** the spec defines conflict as flow vs absorption. Buyers- *and* sellers-absorbed levels both at the
   box with no flow lean are not a conflict under that wording (→ a read if one side's other conditions hold, else NO
   ACTION). Keep as written, or also treat two-sided absorption as CONFLICT?
4. **Absorption set for the verdict:** the same top-8 levels the chart would draw, also when the Absorption levels layer is
   off (the panel says which level it used).

## Step 0 decisions (Ryan, chat 2026-10-31)

1. **Time-of-day split:** yes, for Δ share only (box start in open 06:30–07:30 / mid 07:30–11:00 / late 11:00–13:00 PT).
   Contracts/pt stays all-day.
2. **Same-source baselines, no adjustment:** yes. The badge hover states the cross-feed noise (median ~9 percentile points
   on Δ share), so a badge near a cut-off can flip.
3. **Conflict scope:** A24 as written (flow lean vs absorption = CONFLICT). New separate no-action reason, checked right
   after CONFLICT and before the reads: when both a buyers-absorbed and a sellers-absorbed level sit at the box (within
   2 pts) and there is no notable flow lean, the verdict is **"NO ACTION — two-sided absorption"**, naming both levels.
   Not labelled A24.
4. **Absorption levels for the verdict:** the same top 8 the chart draws, used even when the Absorption levels layer is off;
   the panel names the level it used.

Verdict order as built: NO TRADES (A22) → CONFLICT (A24) → NO ACTION two-sided absorption → SHORT / LONG READ (A19 + A23)
→ WATCH BREAK (idea) → NO ACTION nothing notable.

## Implementation notes (deltas found while building; no spec change)

- **Item 2:** the current section's stats are collapsed like the frozen ones (spec); the lines stay in the DOM
  (`display: none`), so the popup's test ids (`flow-v3-box-popup-head / -top / -bottom / -mixed`) still name the current
  box's lines. Max width 460 px (380 wrapped most lines).
- **Item 3:** "× hides it until the next anchor" keeps CR-DP's meaning: picking the same anchor again is not a new anchor.
- **Item 5:** `boxPopupLines` gains `parts` (the head's pieces) so a badge sits after the share and after contracts/pt; the
  head string itself is unchanged. ⚙ holds both groups ("Box panel — percentile tiers", "Verdict rules") in one stored
  object (`ib-react-flow-v3-settings.boxRules`, defaults `BOX_RULES_DEFAULTS`).
- **Item 7 fix (found by the smoke scan):** the panel's boxes carry edge objects (`top: { price, … }`), so the first
  build's verdict read NaN distances and always fell to "price mid-box". `boxVerdict` now takes edge prices; unit test on
  that shape.
- **Evidence wording:** "price 2.00 from the top" is the distance either side of the edge (a close up to 2 pts beyond the
  top counts as "near", as a test of the edge); it doesn't say above / below.

## Smoke (2026-10-31, read-only backend `scripts/cr_dp_smoke_backend.py` :8068; branch :5193, origin/main `82028ca` :5194)

`tests/e2e/cr-dq.spec.js` **9 / 9** (logs `scripts/logs/cr-dq/smoke/`, screenshots `specs/assets/cr-dq/`). Unit **91 / 91**.
GEX lines are hidden in these loads (min |GEX| 9999 $bn): an armed anchor click on a GEX line opens its popup instead.

| Smoke item | Result |
|---|---|
| 2026-10-30, anchor 11:57: no box text; panel bottom-left; one section per box; current last, highlighted, verdict first; numbers = origin/main | 2 boxes (11:57–12:39 broke ↑, 12:39–12:59 current). No drawn box has a label (origin/main: both); no box hover. Branch box objects = origin/main's (plus `lastClose`). Current section's head (badges removed) / Top / Bottom = origin/main's popup text exactly; the frozen section has origin/main's label pieces and Top / Bottom contracts. Panel 4 px in from the plot's left and bottom, ≤ half the plot. Verdict: NO ACTION — nothing notable (price mid-box · share +4.05 % p73 · c/pt p62) |
| Drag, reload: kept; double-click: bottom-left | drag +200 / −150 px stored (`ib-react-box-panel-pos`); same place after reload and for a new anchor; double-click → bottom-left, key removed; 900 × 700 viewport: inside the plot; × until the next anchor; clear removes |
| Shading: current stronger than frozen, candles on top; set-end distinct | magenta lift (r + b − 2g, shade on vs off) current 24, frozen 12; candle body pixels unchanged; set end 12:40: the window's pixel above the boxes lifts 10 / 10 / 10 (slate, no magenta cast) |
| > 4 boxes | 2026-10-29 anchor 06:31: 5 sections, "3 older boxes not shown" |
| Percentiles: 3 example boxes by hand; tiers | Train windows as anchor + set end (Databento on the chart), one box each: 02-10 07:05–07:30 Δ −1,718 / vol 92,876 → share p58.41 · c/pt p69.38; 04-15 09:12–09:24 +725 / 18,923 → p57.74 · p30.33; 05-20 10:40–11:35 −154 / 68,938 → p7.69 · p88.99 = Step 0's minute-table sums and numpy.interp on the JSON (to 0.01). 2026-10-30 anchor 06:31 (Ironbeam): every badge = `boxPercentiles` of its readout; tiers seen normal / notable / strong with weights < 600 / ≥ 600 / ≥ 800 |
| Verdicts: 3 hand-checked boxes; 10-28 NO TRADES | below |

### Verdict scan (2026-10-29 / 10-30, replay to every 10 min 07:00 → 12:50, anchor 30 min earlier; 72 replays)

65 NO ACTION nothing notable · 3 WATCH BREAK ↑ (idea) · 2 SHORT READ · 1 CONFLICT · 1 NO ACTION two-sided (`item-7-scan.json`).
Hand checks against the rule table (defaults: notable p80, near 2, no advance 2, absorption 2):

- **Read — 10-30 replay 09:30, anchor 09:00:** box 09:00–09:29, top 7629.50, last close 7631.50 (2.00 beyond the top ≤ 2:
  near); share 3,962 / 80,300 = +4.93 % p93 ≥ 80 (buyers leaning); c/pt p86 ≥ 80 (effort; also dPrice +2.0 ≤ 2); the only
  sellers-absorbed level (7640.00) broke at 08:30, before the box → no conflict → **SHORT READ at top** (A19). ✓
- **Conflict — 10-30 replay 08:30, anchor 08:00:** box 08:00–08:29, bottom 7639.75; share 3,949 / 103,311 = +3.82 % p84 ≥ 80
  (buyers leaning); sellers absorbed 7640.00 from 08:20, unbroken at the box's last bar, 0.25 from the bottom → **CONFLICT →
  NO ACTION** (A24). ✓
- **Nothing notable — 10-29 replay 07:20, anchor 06:50:** box 06:50–07:19, top 7756.25 / bottom 7738.50, last 7753.50
  (2.75 from the top > 2, 15 from the bottom): not at an edge; share −0.58 % p23 (no lean); no levels → **NO ACTION —
  nothing notable**. ✓
- Also seen: **two-sided** 10-29 replay 12:40 (buyers absorbed 7746.50 0.75 from the top 7745.75, sellers absorbed 7738.25
  2.00 from the bottom 7740.25, share p4) ✓; **watch break ↑** 10-29 12:10, 10-30 07:50, 12:50 (share p85–97, c/pt p35–44,
  at the top, no buyers absorbed there).
- 10-30 replay 10:20 (anchor 09:50) reads SHORT on a box whose top = bottom = 7609.00 (a box after a break starts at one
  close and widens only on a close > 2 pts away, CR-DN). The rule reads it as written; noted, not changed.
- **2026-10-28 (FOMC):** NO TRADES — event day, hover "Rules: A22" ✓.

### Smoke addition — Ryan's 2026-10-30 entries (report only; thresholds unchanged)

`smoke-ryan-entries.json`, screenshots `ryan-entry-*.png`. Replays cut at the entry minute (the backend floors `until` to
the minute).

| Entry | Anchor | Verdict | Evidence / inputs |
|---|---|---|---|
| 06:11:46 | 05:00 (FULL) | NO ACTION — nothing notable | 2 boxes (05:00–05:34 broke ↑; 05:34–06:10). Last 7664.75, 0.50 below the bottom 7665.25; share +2.40 % — **no badges: ETH box (baseline is RTH only)**, so no lean and no effort read; no levels (ETH: absorption is RTH-only) |
| 07:06:47 | 06:58 | NO ACTION — nothing notable | box 06:58–07:05, 7658.50–7667.00, last 7657.00 (1.50 below the bottom); share −0.05 % p1 (no lean), c/pt p60, dPrice −7.0; no levels |
| 07:49:17 | 07:40 | WATCH BREAK ↑ (idea) | box 07:40–07:48, 7652.25–7656.50, last 7657.00 (0.50 above the top); share +8.14 % p94 (buyers), c/pt p65 (moving easily), dPrice +4.75; buyers absorbed 7660.25 (07:05) is 3.75 above the top — beyond the 2-pt absorption distance, so not counted |
| 09:22:50 | 09:15 | NO ACTION — nothing notable | box 09:15–09:21, 7624.75–7627.50, last 7624.50 (0.25 below the bottom); share +3.97 % p56 (no notable lean), c/pt **p98 (strong)**, dPrice −0.25. A LONG READ needs sellers leaning or sellers absorbed at the bottom; neither (the effort was buyers', positive share, below notable) |

### Existing specs (`scripts/logs/cr-dq/existing-specs.log`)

All 18 other spec files against the branch (:5193, backend :8068, origin/main :5194): **62 passed, 16 skipped, 26 failed
(47 min)**. Every failure is either not runnable here or fails on origin/main as well:
- **Not runnable here (other CRs' fixed ports, refused):** `cr-ck` ×2 (:5178 / :5179), `cr-db` ×5 (:5176 / :5177; T/S/U also
  fails on main, CR-DN), `cr-de` ×7 (:5172).
- **Need a live session (today, 2026-10-31, is a Saturday; the chart never loads):** `cr-dc` live, `cr-df` live, `cr-dg`
  live, `cr-dp` item 8 ("a live bar arrived").
- **Need the Dash app:** `cr-dh` ×4 (as in CR-DP).
- **Fail on origin/main too:** `cr-dn` item 6 "57k closing minute" (stale since CR-DN merged, as in CR-DP); `cr-dp` item 4
  and item 6 — both compare with the *pre-CR-DP* main (its header edge readout / raw CVD line), which origin/main `82028ca`
  no longer has; run with origin/main's own spec copy against origin/main they fail the same way; `cr-dm` "review replay
  2026-10-29" — times out waiting for the chart on origin/main as on the branch (15 min each; its hand-check file copied
  from the main checkout's logs).

**Updated (assertions on the removed labels / hover / old popup placement only):**
- `cr-dn`: `drawn()` labels → `boxLabelText` of each box's readout (the label left the chart; the hand-sum checks now check
  the panel's numbers, one source).
- `cr-di`: the frozen-box hover → the oldest frozen section of the box panel, expanded ("Top … · buyers …" instead of
  "Top …: buyers …"; "n older boxes not shown" instead of "not drawn").
- `cr-dp` item 4: label read from the box readout; default place bottom-left (was right of the box top-right); drag by
  the header, clamped in the plot; a new anchor reuses the remembered place (was: reset); ≤ half the plot (was the 8 %
  compactness check). Amendment 2: popup head `toBeAttached` (the current section is collapsed by default).
