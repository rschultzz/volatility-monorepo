# CR-DI — Edge effort strip in Flow v3

Aggressive volume at each range edge over time, both sides. Spec copied from the vault session note
`sessions/2026-10-28 - CR-DI — Edge Effort Strip.md` (Ryan's kickoff, chat 2026-10-28).

Branch `feat/CR-DI-edge-effort-strip` off `origin/main` `24a8f6f`, worktree `../volatility-monorepo-cr-di`.
CR-DH is in flight in its own worktree and is not touched.

## Why (Ryan, chat 2026-10-28)

The Edges readout gives one-sided totals since the anchor ("Top 7769.25 · 16.4k buyers", "Bottom 7762.00 · 16.9k
sellers"). Ryan wants to see that effort over time: which visit to each edge was heavier, whether visits grow or
fade, and whether a big push held. One-sided counts mislead. At 07:49–08:02 PT the bottom showed ~10.7k aggressive
sellers, but aggressive buyers traded ~15.3k across all prices in the same minutes (net −339). So the low was
actively defended, not one-way selling. The strip must show both sides.

## Spec

1. An **"Edge effort" chip** in the Flow v3 header, next to Edges.
   - Off by default (C2), remembered per browser (new localStorage key).
   - It only does anything when Edges is on and an anchor is set. Otherwise it's disabled, with a tooltip saying why.
2. When on, a **strip** (about 60–70 px) appears in the Flow v3 pane below the existing delta-per-bar bars.
   - It is time-aligned with the price chart and crosshair-linked like the rest of the pane.
   - It has one entry per 1-minute bar from the anchor to now:
   - a. TOP edge, drawn upward: aggressive BUY contracts toward the top (main bar, blue), and aggressive SELL
     contracts toward the top (thinner, lighter grey bar beside it).
   - b. BOTTOM edge, drawn downward: aggressive SELL contracts toward the bottom (main bar, grey), and aggressive BUY
     contracts toward the bottom (thinner, lighter blue bar beside it).
   - c. Colours from the existing BUY / SELL palette at different opacities (C1, no new colours).
3. **Counting rule.**
   - The main bars use exactly the rule behind the header's edge counts, so they sum to the header numbers at every
     moment, including after an edge moves on a break.
   - Reuse that code path; don't reimplement the band logic.
   - The opposite-side bars use the same rule with the side flipped.
4. **Visit totals.**
   - A visit is a run of minutes with a non-zero main bar at that edge, joined across gaps of up to 2 minutes.
   - Label each visit once with both sides, main first (e.g. top "8.9k / 8.4k", bottom "10.7k / 9.9k").
   - Labels are small text in the pane's label style, above the top bars and below the bottom ones.
   - If labels collide, drop the smaller visit's label; the hover still shows it.
5. **Hover a minute.** It shows:
   - the time;
   - at the top, buyers / sellers / net;
   - at the bottom, sellers / buyers / net;
   - the running totals of the visit it belongs to.
6. **Exact data only**, like the edge counts.
   - On approximate (1-s) sessions the strip isn't drawn, and the chip's tooltip says why.
   - Replay / review respects the until time.
7. With the chip off, Flow v3 and the price chart are **pixel-identical to origin/main**.
8. **Legend (?) entry** for Edge effort: what the bars are, plus this reading guide, labelled "how to read, untested":
   - heavy aggressive selling at the low, CVD falls, price holds = passive defense (resting bids absorb);
   - heavy aggressive selling at the low, CVD flat, price holds = active defense (buyers lift offers as hard as
     sellers hit bids);
   - heavy selling, CVD falls, price falls = sellers winning;
   - balanced flow, CVD flat, price falls = bids pulled / thin book;
   - mirror for the top edge.

   It also notes that visit sizes are effort, not a tested signal, and that Ironbeam live data understates volume.

## Standing rulings conformance (UI CR)

| Ruling | Status | Note |
|---|---|---|
| C1 — no green / red; Dash palette | honoured | BUY blue `96,165,250` and SELL light grey `229,231,235` at different opacities; no new colour |
| C2 — new display features off by default, existing views unchanged | honoured | chip off by default; with it off Flow v3 and the price chart are pixel-identical to origin/main |
| A19 — in-range confirmation, both sides | n/a (context) | the strip shows effort at the edges; it is not a confirmation rule |
| A20 — trade the range edges | n/a (context) | display only; no entry logic |
| A2 — a range is measured against the prior move, no fixed duration | n/a (display simplification) | amendment 9's hold time N is a ⚙ drawing default (15 min, 5–60), not a range definition or a tested threshold (Ryan, chat 2026-10-28) |
| C2 (amendment 9) — the Edges box restarts on a break | **deviates — APPROVED** by Ryan in chat 2026-10-28 | an existing view's behaviour changes (Edges box, header readout); change-log line added to `standing-rulings.md` |
| C1 (amendment 9) | honoured | frozen boxes use the existing dashed magenta at lower opacity |
| all other A* / B* rulings | n/a | no research claim, no outcome, no threshold |

- Visible change in scope (the chip, the strip, the legend entry), approved by Ryan in chat 2026-10-28.
- No research claims; nothing here is a tested threshold. Nothing in "Revoked / superseded" is used.

## Step 0 (commit before implementation)

- a. Find and quote the exact rule behind the header's edge counts:
  - which edge price applies to which minute (current edge vs the edge as it stood then);
  - the band (within 1 pt, inclusive or exclusive);
  - what happens to the counts when an edge breaks and moves.
- b. For 2026-10-28 with the anchor at 07:00 PT:
  - Compute the per-minute series (all four: top buy, top sell, bottom sell, bottom buy) by that rule.
  - Show the main-side totals equal the header's at 08:09 PT (top 16.4k buyers, bottom 16.9k sellers, as Ryan saw
    them).
  - Chat's rough version (fixed bands ≥ 7768.25 / ≤ 7763 around the final edges) gave about 17.6k / 17.6k; explain
    the gap.
  - List the per-visit totals for both sides.
- c. Confirm whether the cells payload already carries what's needed (side and price per minute), or whether a
  backend change is required. Prefer no backend change; if one is needed, stop and report before implementing.

## Commits

1. spec freeze;
2. Step 0;
3. one commit per item group:
   - chip + state;
   - strip with main and opposite bars;
   - visit labels + hover;
   - approximate-data handling + legend;
4. smoke + wrap.

## Smoke (one read-only server)

- Chip off → pixel-identical to origin/main on 2026-10-28 and 2026-10-27.
- 2026-10-28, anchor 07:00, Edges on, chip on:
  - main-side strip sums equal the header counts at replay times 07:40, 08:09 and the end;
  - visit labels match Step 0b's per-visit sums for both sides.
- An edge break moves the edge: strip and header still agree after it.
- An approximate-data session: no strip; chip tooltip explains.
- Existing cr-dc / cr-df / cr-dg specs green.

## Step 0 findings (read-only: one DB connection via `scripts/cr_di_step0/dump_cells.py`, no server)

Payloads: `/flow/cells` for 2026-10-28 RTH, built by `flow_v3.build_cells_payload` with until 07:40, 08:09 and live
(last bar 08:24 PT, dumped 15:24 UTC). The series come from `scripts/cr_di_step0/series.mjs`, which runs the shipped
`flattenBars` + `edgeReadout`. Outputs are in `scripts/logs/cr-di/step0/` (git-ignored).

### 0a — the rule behind the header's edge counts (`flowV3.edgeReadout`, `react_price_preview/src/flowV3.js`)

Quoted from the function's comment:

> "The edges start at the anchor minute's close; a later close > EDGE_BREAK_PTS beyond an edge is a break and the
> edge moves to it, a close within EDGE_BREAK_PTS is a test (edge unchanged). Contracts: aggressor volume of cells
> since the anchor within EDGE_NEAR_PTS of the current edge (buys at the top, sells at the bottom), only when every
> bar since the anchor is on an exact segment (else null and `exact` false)."

What the code does:

- **Which edge price applies to which minute: the current (final) edge, for every minute.**
  - The loop first ratchets `top` / `bottom` over all closes from the anchor bar to the last bar.
  - Only then does it sum cells in `[t0, t1]` (anchor bar start → last bar start, inclusive) against that one final
    price.
  - The edge as it stood at the time is never used.
- **Band: within 1 pt, inclusive, two-sided.**
  - `Math.abs(p − round(edge / 0.25)) <= 4` in ticks (`EDGE_NEAR_PTS / TICK`).
  - So the band is edge − 1.00 … edge + 1.00 inclusive: 9 price cells.
  - Trades more than 1 pt beyond the edge (wicks) are not counted.
- **On a break the counts are recomputed from the anchor against the new edge.**
  - Minutes that were near the old edge drop out if they are more than 1 pt from the new one.
  - Minutes near the new edge count even if they came before the break.
  - Measured live: at the 08:16 PT top break (7769.25 → 7772.25, its third), the header top went from 16.4k buyers
    (08:09) to 4.5k (08:24).
- **Consequence for the strip (spec item 3).**
  - The main bars computed per minute against the current edge sum to the header at every moment.
  - Also, after a break the strip redraws: earlier visits at the old edge shrink or vanish, as the header does.
  - This follows from "reuse that code path" and is not a separate choice.

### 0b — 2026-10-28, anchor 07:00 PT

| until (last bar) | Top edge | header buyers | Σ strip top buy | Σ top sell | Bottom edge | header sellers | Σ strip bottom sell | Σ bottom buy |
|---|---|---|---|---|---|---|---|---|
| 07:40 (07:39) | 7769.25 | 15,421 | 15,421 | 14,160 | 7762.00 | 5,287 | 5,287 | 4,903 |
| 08:09 (08:08) | 7769.25 | 16,446 (16.4k) | 16,446 | 15,041 | 7762.00 | 16,613 (16.6k) | 16,613 | 15,437 |
| live (08:24) | 7772.25 (broke 08:16) | 4,526 | 4,526 | 4,276 | 7762.00 | 16,613 | 16,613 | 15,437 |

- **The main sides equal the header at every until.**
- **Ryan's 16.4k / 16.9k** at 08:09 reproduces exactly with the anchor at **06:58 PT**, not 07:00:
  - Top 7769.25 · 16,446; Bottom 7762.00 · 16,931.
  - The bottom started at the 06:58 close and broke to 7762.00 at 07:00. The extra 318 sellers traded 06:58–06:59
    near 7762.
  - With the anchor at 07:00 the bottom is 16.6k.
  - Other anchors give other bottoms: 06:59 → 16.6k, 06:57 → 11.4k (edge 7761.50), 07:01 → 21.1k (edge 7763.00).
  - So the anchor Ryan used was most likely 06:58 (to confirm with Ryan; it changes nothing in the build).
- **Chat's fixed bands** (≥ 7768.25 / ≤ 7763, same minutes 07:00–08:08) give 17,335 / 17,550 (17.3k / 17.6k):
  - The gap is the wicks. A one-sided band also counts trades more than 1 pt beyond the edge, which the two-sided
    ±1 pt rule drops.
  - Top buys at 7770.50–7771.25: 889 = 17,335 − 16,446.
  - Bottom sells at 7759.25–7760.75: 937 = 17,550 − 16,613.
  - Chat's ~17.6k on the top likely ran a minute or two further; at 08:08 it is 17.3k.
  - The same wick effect explains chat's "10.7k" bottom visit at 07:49–08:02, against 9.8k by the rule.

**Per-visit totals.** A visit is a run of minutes with a non-zero main bar, joined across up to 2 empty minutes.
Shown as main / opposite:

| until | Edge | Visit (PT) | Main | Opposite | Label | Net (buy − sell) |
|---|---|---|---|---|---|---|
| 08:09, anchor 07:00 | top | 07:04–07:06 | 2,502 buy | 2,275 sell | 2.5k / 2.3k | +227 |
| | top | 07:13–07:24 | 8,602 | 7,776 | 8.6k / 7.8k | +826 |
| | top | 07:28–07:36 | 4,317 | 4,109 | 4.3k / 4.1k | +208 |
| | top | 08:05–08:07 | 1,025 | 881 | 1.0k / 881 | +144 |
| | bottom | 07:00–07:02 | 3,259 sell | 3,167 buy | 3.3k / 3.2k | −92 |
| | bottom | 07:11–07:12 | 1,679 | 1,554 | 1.7k / 1.6k | −125 |
| | bottom | 07:39–07:41 | 1,909 | 1,226 | 1.9k / 1.2k | −683 |
| | bottom | 07:49–08:02 | 9,766 | 9,490 | 9.8k / 9.5k | −276 |
| 08:09, anchor 06:58 | bottom | 06:58–07:02 | 3,577 | 3,393 | 3.6k / 3.4k | −184 |
| | (others as anchor 07:00) | | | | | |
| live 08:24, anchor 07:00 | top (7772.25) | 07:34 | 9 | 0 | 9 / 0 | +9 |
| | top | 08:14–08:24 | 4,517 | 4,276 | 4.5k / 4.3k | +241 |
| | bottom | as 08:09 | | | | |

Visits sum to the header: top 2,502 + 8,602 + 4,317 + 1,025 = 16,446; bottom 3,259 + 1,679 + 1,909 + 9,766 = 16,613.

The per-minute series (all four, plus all-price buy / sell) is in `scripts/logs/cr-di/step0/series_0809.txt`.

### 0c — payload

- No backend change is needed. The cells payload already has, per segment, `cells.t` (minute start, UTC), `cells.p`
  (price in ticks), `cells.buy` and `cells.sell` (aggressor contracts), plus `exact` and `bars`.
- `edgeReadout` already reads exactly these.
- The strip is computed client-side by the same function, extended to return the per-minute arrays. The header
  numbers become the sums of those arrays, so there is one code path.
- Replay `until` already truncates the payload server-side.

## Amendment 1 (Ryan, chat 2026-10-28) — box restarts on a break

Received after the four item-group commits and before the smoke. Nothing that depends on edges is implemented
under it until this amendment is committed.

9. **Box restarts on a break.**
   - Today a close more than 2 pts beyond an edge moves that edge out and keeps the box, so a held range and the
     range after its break merge.
   - On 2026-10-28, 7762–7769.25 held 07:04 → the 08:16 break, and the box became 7762–7775 by 08:39.
   - New behaviour:
     - a. On a break, the current box **freezes**:
       - it is drawn faded (same dashed magenta at lower opacity), ending at the break minute;
       - its hover gives its own stats: top / bottom, counts per edge and side, minutes held, "broke hh:mm ↑/↓".
     - b. A **new box starts at the break bar**, with top = bottom = that bar's close, growing by the existing test /
       break rule (as ruled below).
     - c. The **header readout and the Edge effort strip use the current box only**.
       - The strip draws each box's bars in its own time span.
       - Visits never join across a break.
     - d. At most the **last 4 frozen boxes** are drawn; older ones are dropped from the drawing (the hover shows the
       count dropped).
     - e. Clearing or moving the anchor removes all boxes, as today.
10. Step 0a must also answer: do the header's edge counts reset when an edge moves, or recount at the new price since
    the anchor? (2026-10-28: top showed 16.4k buyers before the break and 8.7k at 7775 after it.)

Rulings:
- **C2 deviation approved** by Ryan in chat 2026-10-28 (the Edges box behaviour changes); change-log line added to
  `standing-rulings.md`.
- C1 honoured (existing magenta at lower opacity).
- The Edges legend text is updated to describe restart-on-break.
- Smoke addition: 2026-10-28 anchor 07:00 shows a frozen box ending at the first break of 7762–7769.25 (↑), then
  later boxes; the header matches the current box only.

### Amendment Step 0 (read-only, `scripts/cr_di_step0/box_rules.mjs` on the 09:07 PT live dump; output in `scripts/logs/cr-di/step0/box_rules.txt`)

**9b as written can't grow a box.**
- A new box has top = bottom = the break close, and under the test / break rule a close up to 2 pts beyond is a test
  (no change) while one more than 2 pts beyond is a break.
- Every move of more than 2 pts would therefore restart it. On 2026-10-28 anchor 07:00 that is **32 boxes**, none
  wider than one price.
- Today the only way an edge moves *is* a break, so "growing" and "breaking" were the same event.

**The first box breaks at 08:16, not 08:39.**
- With today's 2-pt rule, 08:16 closed 7772.25 > 7769.25 + 2.
- 08:39 (close 7775) is the second break of the top: 7772.25 → 7775.

**Ruling (Ryan, chat 2026-10-28): hold time N.**
- A close more than 2 pts beyond an edge **grows** the box (the edge moves to the close, today's ratchet) while that
  edge has stood less than N minutes since it was set.
- Once the edge has stood ≥ N minutes, such a close is a **break**: the box freezes and a new one starts at the
  break bar.
- N is a **⚙ setting**: "Box edge must hold N min before a break freezes it", default **15**, range **5–60**.
- Labelled a display default, not a tested threshold.
- A2 measures ranges against the prior move, not fixed time, so N is a simplification for the drawing only.
- Both edges of a new box count as set at its first bar.

Candidates measured on 2026-10-28, anchor 07:00 (bars to 09:07 PT):

| Rule | Boxes |
|---|---|
| literal 9b | 32 |
| only tested edges break | 13 (long box 07:11–08:16) |
| N = 5 | 8 |
| N = 10 | 07:00–08:16 7762–7769.25 ↑ · 08:16–08:29 7772.25 ↓ · 08:29– current |
| **N = 15 (default)** | **07:00–08:16 7762.00–7769.25 ↑ · 08:16–08:39 7769.75–7772.25 ↑ · 08:39– current 7770.25–7775.00** |
| N = 30 | 07:00–08:16 ↑ · 08:16– current 7769.75–7775 |

**Smoke expectation (replaces "frozen box 07:00–08:39").** At N = 15:
- frozen 07:00–08:16 (7762.00–7769.25, broke 08:16 ↑);
- frozen 08:16–08:39 (broke 08:39 ↑);
- current box from 08:39;
- the header matches the current box only.

**Item 10 (answered by Step 0a): they recount; they don't reset.**
- `edgeReadout` counts all cells since the anchor against the current edge price.
- When the top moved 7769.25 → 7772.25 at 08:16, the top count went 16.4k (08:09) → 4.5k (08:24): only cells within
  1 pt of 7772.25 since 07:00.
- Ryan's 8.7k at 7775 is the same recount at the 08:39 edge.
- Under amendment 9 the counts instead start at the current box's first bar.

**Counting per box (implementation rule).**
- A box spans its first bar to the bar before its break. The break bar opens the next box, so no minute is counted
  twice.
- Within a box, cells are counted against that box's final edges with the unchanged ±1 pt inclusive rule, through
  one shared function. So the strip's main bars over the current box still sum to the header.
- The strip draws the minutes of the drawn boxes (the last 4 frozen + current), each against its own box. Visits
  are computed per box.

## Amendment 2 (Ryan, chat 2026-10-28) — the pane grows by the strip

11. When the Edge effort strip is drawn, the Flow v3 pane grows by the strip's height (64 px), so CVD, rolling and
    delta keep their current heights.
    - The 64 px come from the price chart above it (the price pane flexes; the flow pane has a fixed height).
    - With the chip off, the pane height and everything in it are pixel-identical to origin/main (C2).
    - "Drawn" is one shared condition (`flowV3.edgeStripDrawn`): chip on, Edges on, an anchor, and exact data. App sizes
      the pane by it, and the panel draws the strip by it, so the two can't disagree.
    - Re-run cr-di and the chip-off pixel comparisons on 2026-10-27 and 2026-10-28; record the results.

Standing rulings: C2 honoured (chip off = origin/main); the visible change with the chip on is approved by Ryan in
chat 2026-10-28. C1 n/a.
