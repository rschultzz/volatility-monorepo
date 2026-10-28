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
