---
type: cr
cr_id: CR-DC
title: Flow v3 — anchored CVD, aggressor-side volume heatmap, volume profile and absorption marks (display only, off by default)
status: in-progress
run_mode: interactive
data_safety_class: read_only
dependencies: [CR-CD, CR-CH]
branch_name: feat/CR-DC-flow-v3
tags: [dash, cr, order-flow, aggressor, price-chart, frontend, heatmap, display-only]
---

# CR-DC — Flow v3

Display only, read-only DB. Branch `feat/CR-DC-flow-v3` off `origin/main` `0fdcd6d`, own worktree
(`.claude/worktrees/cr-dc`). Kickoff: Ryan, 2026-10-27 (chat). Frozen 2026-10-27 before any code.

---

## Kickoff (verbatim)

```
CR-DC — Flow v3: anchored CVD, aggressor-side volume heatmap, volume profile and absorption marks
(display only, off by default).

Branch first: git checkout -b feat/CR-DC-flow-v3 off origin/main. Own worktree.
Read CLAUDE.md (visible-change discipline), field-conventions.md, standing-rulings.md, the CR-CD spec
(Flow v2 pane, swing labels) and the CR-CH spec (the Chrome crash and its fix — the reason for the
performance rules below). Freeze specs/CR-DC-flow-v3.md before code; list any choice the brief doesn't
fix (C1, C2, …) first.
Rulings: C1 (Dash palette only: blue = buyers, light grey = sellers, slate = neutral/marks; no green/red);
C2 (everything new is off by default; Flow Current and v2 stay pixel-identical); B4 (Databento served only
through 2026-06-05, as v2). Every new element's '?' guide says "visual aid — untested". No research claims.

1. Flow toggle: Current | v2 | v3 (default unchanged; new localStorage key).

2. v3 pane (below the price chart, bar-aligned like v2):
   - Anchored CVD: with v3 on, clicking a bar on the price chart sets the anchor (buttons: "anchor =
     session open", "anchor = last range start"; Esc / "clear" resets). CVD is drawn from 0 at the anchor.
     Header readout: Δ delta since anchor, Δ price since anchor, minutes since anchor, contracts per point.
   - Rolling net delta: bars of Σ(buy − sell) over the last N minutes (⚙: 5 / 10 / 15, default 10).
   - The v2 delta histogram underneath, so the pane stands alone.

3. Aggressor-side heatmap on the price pane (v3 header toggle "Heatmap", off by default):
   - Grid behind the candles: one cell per 1-min bar × 0.25 pt (0.5 pt if crowded). Each cell has two
     values: aggressive SELL volume (market sells hitting the bid at that price) and aggressive BUY volume
     (market buys lifting the offer at that price). Two layers, each toggleable: sell heat (light grey),
     buy heat (blue). Opacity from the cell's percentile in the session's per-cell distribution.
   - Absorption marks (aggressor side):
     • "Sellers absorbed" at P: aggressive sell volume at P ≥ K × the session median cell volume, and no
       trade below P − 1 tick for the next M minutes. Up-pointing marker below the cell plus a short slate
       line at P that extends until price trades through it.
     • "Buyers absorbed" at P: the mirror (aggressive buys; nothing prints above P + 1 tick); down-pointing
       marker above the cell.
     • Defaults K = 3, M = 3, both in ⚙ (display defaults, not tested thresholds).
     • Hover: time, price, aggressor contracts, opposite-side contracts, minutes held, and whether / when
       it later broke.

4. Volume profile (v3 header toggle "Profile", off by default): horizontal histogram at the right edge of
   the price pane, per 0.25 pt (or 0.5 pt), split by aggressor side (sell grey / buy blue), over
   [anchor or session open → now / replay `until`].

5. Data:
   - Live session: ironbeam_es_trades, each trade at its own price with its aggressor side (exact). Check
     the prune-ironbeam-trades schedule and report how many days are retained; dates beyond retention fall
     back to the 1-s method.
   - Past dates: 1-s Databento — each second's sell volume at that second's LOW, buy volume at its HIGH;
     badge "approximate: 1-s data". In Step 0, report the share of seconds whose range is > 1 tick.
   - REVIEW replay never uses data after `until`. B4 respected.
   - Extend the existing flow route or add a read-only endpoint; no DB writes.

6. Performance (CR-CH lesson): the heatmap, profile and marks each draw in ONE canvas primitive (no
   series or price lines per cell); compute and draw only the visible range; memoise inputs so nothing
   recomputes on re-render unless the data or the visible range changed.

Tests: unit tests for anchored CVD, the 1-s aggressor-price assignment and the absorption rule; Playwright:
Current and v2 pixel-identical to origin/main; v3 renders on 2026-05-01 and on the live date; Heatmap and
Profile off by default; replay shows nothing after `until`; headed Chrome 155, CR-CH-style 10-min
drag/zoom with v3 + Heatmap + Profile on — no crash, no heap growth. PR, don't merge; Ryan merges, chat
deploys Dash. Session note; copy logs before removing the worktree.
```

---

## Standing rulings conformance (`Dash/research/standing-rulings.md`, status current, read 2026-10-27)

CR-DC is a display CR, not a research CR; the table is filled anyway because the kickoff names three rulings.
Nothing in "Revoked / superseded" (R1–R7) is used.

| ruling | status | how |
|---|---|---|
| A1–A4, A6–A16, A18–A21 (setup) | n/a | no setup is defined, tested or shortlisted. The "last range start" anchor (DC4) is a display anchor, not a setup |
| A5, A17 (flow decides fade vs break; flow first) | n/a | the pane shows flow; it makes no fade/break call. Every new "?" guide says "visual aid — untested" |
| B1–B3, B6–B8 (tests) | n/a | no test of a setup; no verdict |
| B4 (data split) | **honoured** | Databento is served only for sessions ≤ 2026-06-05, as v2 (A1.1). Later sessions use Ironbeam (exact trades while retained, else 1-s). Step 0 value reads: Databento train only; the one Ironbeam-trades comparison (2026-10-27) is mechanical (where volume lands on the price grid), no direction, outcome or feature statistic |
| B5 (Databento is the reference) | honoured | display only; the badge names the source; the guide repeats CR-CA's Ironbeam undercount |
| C1 (palette) | **honoured** | sell = light grey `#e5e7eb`, buy = blue `#60a5fa`, marks / lines = slate `#94a3b8`; no green, no red |
| C2 (off by default; existing views unchanged) | **see DC1** | everything new is off by default. The kickoff's own item 1 adds a "v3" button to the flow-mode toggle that both existing headers show — DC1 asks how that is reconciled with "pixel-identical" |

---

## Choices the brief does not fix (DC1 …) — Ryan to rule before implementation

Numbered **DC1 …** (the kickoff says "C1, C2, …"; prefixed so they don't clash with standing rulings C1 / C2).
Each has a recommendation; Step 0 measurements behind them are below.

| # | Choice | Options | Recommendation |
|---|---|---|---|
| DC1 | **The "v3" toggle button vs "Current and v2 pixel-identical".** `FlowModeToggle` is shared by both existing pane headers; adding "v3" to it changes those headers by one button | (a) add "v3" to the shared toggle; pixel identity is asserted on the price pane, the flow pane's plot canvas and the header **outside the toggle group**, and the toggle group's change is the only diff (the kickoff asks for it); (b) put "v3" somewhere else (chart toggle bar) so the Current / v2 panes are byte-identical, at the cost of a split control | **(a)** |
| DC2 | **"New localStorage key"** | (a) `ib-react-flow-mode-v3` = `'1'` while v3 is chosen; the old `ib-react-flow-mode` keeps its last `'current'` / `'v2'`, so a rollback to pre-v3 code lands on that mode, never on an unknown value; (b) store `'v3'` in a new key that replaces the old one | **(a)** |
| DC3 | **Anchor click vs the existing click.** A click on a price bar today toggles a timeslice (published to the other panels) or opens the GEX-line popup | (a) a "📍 set anchor" button arms the next click; that click sets the anchor and is consumed (no timeslice toggle); plain clicks keep today's behaviour; (b) with v3 on, every click sets the anchor *and* still toggles the timeslice; (c) with v3 on, clicks set the anchor only (timeslices not selectable while v3 is on); (d) Alt/Option-click sets the anchor | **(a)** — the click keeps its meaning everywhere else |
| DC4 | **"anchor = last range start" — no range detector exists in the chart.** CR-CP's range rule (`scripts/cr_cp_lib.py` `approach_legs` + `find_range`: vol-scaled approach swing, 15-min hold) lives only in a research script and needs the session implied move | (a) port CR-CP's range rule into shared code (causal; the anchor = r0 of the last *qualified* range before now / `until`; route also reads the session implied move); (b) use CR-CD A5's swing legs: the anchor = the last confirmed swing extreme (start of the current, unconfirmed leg), with the ⚙ swing, labelled **"anchor = last swing"** | **(b)** now, honestly labelled; (a) as a follow-up if Ryan wants CR-CP's range specifically |
| DC5 | **Anchor rules** | anchor bar's CVD = 0 and accumulation starts with the next bar; Δ price = close now − close at the anchor bar; minutes = bars since anchor × interval; contracts per point = Σ volume since anchor ÷ \|Δ price\| ("–" when Δ price = 0), as A5's cpp. Bars before the anchor are blank. The anchor is per selected session; a date change or Esc / "clear" resets it to none (CVD then from the session open, as v2). "session open" = 06:30 PT in RTH view, 15:00 PT (prior day) in FULL view. Not stored | as written |
| DC6 | **Rolling net delta on 5-min charts** | N stays in minutes (5 / 10 / 15); on a 5-min chart each candle shows the sum over the last N minutes ending at its close | as written |
| DC7 | **"0.5 pt if crowded"** | client-side: 0.25 pt when one tick is ≥ 3 px tall at the current price scale, else 0.5 pt (two 0.25 cells summed); re-evaluated on zoom. Marks always use 0.25 pt | as written (3 px) |
| DC8 | **Heatmap opacity distribution** | per side, over all of that side's non-zero 0.25-pt cells of the selected session up to now / `until`; opacity = 0.08 + 0.62 × percentile (so light cells stay visible but candles stay readable). Not causal within the day (a cell's shade can change later in the day); it is shading only, and replay never reads past `until` | as written |
| DC9 | **Absorption rule details (marks).** "Session median cell volume" and the window need fixing | **Median:** the running median of the same side's non-zero 0.25-pt cells in the session **before the cell's minute** (causal; no mark until 15 cells exist). **Hold window:** the cell's own minute plus the next M minutes — the low of that window ≥ P − 1 tick (sellers; buyers: high ≤ P + 1 tick); the mark appears at the window's end. **Dedupe:** per side per minute, only the extreme qualifying price (lowest for sellers, highest for buyers); a new mark within 1 tick of an unbroken earlier mark of the same side is not drawn (its hover adds the volume). **Break:** first bar with low < P − 1 tick (sellers) / high > P + 1 tick (buyers); the slate line ends there, or at now / `until`. **Hover:** time, price, aggressor contracts, opposite-side contracts at P that minute, minutes held (to break or now / `until`), "broke at hh:mm" or "not broken". RTH only (as v2's markers, A1 D10) | as written |
| DC10 | **Marks' own toggle** | (a) a separate v3 header toggle "Absorbed", off by default, drawn on the price pane with or without Heatmap; (b) marks are part of Heatmap | **(a)** |
| DC11 | **1-s assignment on approximate data** — see Step 0 Q3: the extreme rule puts sells on each second's low and buys on its high, which piles sell volume onto local lows — the very pattern the "sellers absorbed" rule looks for | (a) the brief's rule as written, badge "approximate: 1-s data", and the guide says marks on approximate data lean toward lows / highs; the smoke reports mark counts exact vs approximate on a retained session; (b) as (a), but marks are hidden on approximate data; (c) a different assignment (each second's VWAP, rounded to the tick, both sides: 87 % vs 77 % same-cell, no side bias, but it loses the bid / offer split inside the second) | **(a)** — Ryan sees the caveat and the measured difference before deciding whether (b) is needed |
| DC12 | **Sources per segment** | per session segment, like v2 (A1.4 contract by bar match): **exact** (`ironbeam_es_trades`) when the trades table covers the whole segment window (its oldest row ≤ window start); otherwise **1-s**: Databento for RTH ≤ 2026-06-05 (06:30–13:00 PT only, A1.1), Ironbeam 1-s otherwise (ETH; RTH after 2026-06-05). Before 2025-12-26 there is no ETH flow (as v2). Badge per segment: "exact: trades" / "approximate: 1-s data" + source. Opacity percentiles and the mark median are computed per segment, so Databento and Ironbeam scales are never pooled (field-conventions: Databento ≈ 1.06× volume) | as written |
| DC13 | **Endpoint** | (a) a new read-only `GET /api/ironbeam/flow/cells` (cells + marks + per-segment source), fetched only while Heatmap, Profile or Absorbed is on; the v3 pane's series reuse the existing `mode=v2` request unchanged (CVD, delta, volume per bar), so the v2 route and its goldens are untouched; (b) `mode=v3` on `/api/ironbeam/flow` | **(a)** |
| DC14 | **Live refresh** | cells poll with the v2 poll (15 s) and pause during interaction (as v2). Completed minutes are cached in-process; each poll reads only from the last cached minute − 2 min. On 2026-10-27 the trades table held 394 k RTH+ETH rows for the session; a full re-aggregation each poll is avoided | as written |
| DC15 | **5-min charts** | heatmap cells are summed per candle (one column per 5-min candle); marks snap to their candle; the profile is unaffected | as written |
| DC16 | **Profile geometry** | right edge of the plot area, inside the price pane, max bar width 18 % of the plot width, scaled to the largest price row in range; sell (grey) left of buy (blue) in one stacked bar per row; 0.25 / 0.5 pt as DC7 | as written |

---

## Step 0 findings (2026-10-27, read-only)

**Method.** `get_backfill_db_conn()` (role asserted) with `default_transaction_read_only = on` (asserted); SELECTs
only. Scripts and outputs in `scripts/cr_dc_step0/`. Render read via the Render MCP (list services). No code
outside `scripts/cr_dc_step0/` and `specs/` was changed.

### Q1 — trades retention (`prune-ironbeam-trades`)

- **Schedule:** Render cron `prune-ironbeam-trades` (`crn-d57k9n2li9vc739htvqg`), `0 1 * * *` (daily 01:00 UTC),
  `python apps/cron/prune_ironbeam_trades.py`; last success 2026-10-28 01:01:48 UTC.
- **Rule:** `DELETE … WHERE ts_utc < now() − IRONBEAM_TRADES_RETENTION_DAYS days`, default **2**.
- **Measured:** oldest row 2026-10-26 01:01:45 UTC at 2026-10-28 02:56 UTC, i.e. exactly 2 days before the last
  prune — the env var is at its default (or 2).
- **What that retains, by session:** the live session in full; the **previous session's RTH** until ~01:00 UTC two
  days later (its ETH from 15:00 PT is partly gone after the next prune); nothing older. On a Monday the Friday
  session is already gone (pruned Mon 01:00 UTC). So: **≈ 1 full session, sometimes 2 for RTH**. Older dates fall
  back to the 1-s method (DC12 decides per segment by coverage, not by date arithmetic).
- **Rows:** 2026-10-26 (PT) 460,786 trades / 1.53 M contracts; 2026-10-27 394,155 / 1.36 M (to 19:56 PT).
  `aggressor_side` = `BUY` / `SELL` / `0` (2 rows, 110 contracts = unknown, not drawn).

### Q2 — share of 1-s seconds whose range is > 1 tick (RTH 06:30–13:00 PT)

| data | seconds | seconds > 1 tick | **volume in those seconds** | seconds > 2 ticks | volume > 2 ticks | single-price seconds |
|---|---|---|---|---|---|---|
| Databento 2026 train (01-02 → 06-05) | 2.49 M | **31.0 %** | **61.2 %** | 12.0 % | 35.9 % | 28.7 % |
| Databento R-a (2024-04-03 → 12-31) | 4.28 M | 17.9 % | 46.8 % | 5.2 % | 22.8 % | 40.8 % |
| Databento 2026-05-01 | 21.6 k | 20.4 % | 50.9 % | 5.0 % | 25.1 % | 33.7 % |
| Ironbeam 1-s 2026 train | 2.29 M | 29.2 % | 57.8 % | 11.0 % | 32.6 % | 29.6 % |

(`q_ranges.py`, `q_ranges.out.json`.) Most of the volume trades in seconds that span more than one tick, because
busy seconds are the wide ones.

### Q3 — exact vs the 1-s rule, on one retained session (2026-10-27 RTH, Ironbeam, ES Z26)

Mechanical: where volume lands on the price grid only (B4: the session is forward-log; no direction, outcome or
feature statistic computed). Trades 279,515, 1,069,166 contracts = the 1-s table's buy + sell exactly.

| assignment | sell: same 0.25 cell (per minute) | buy: same 0.25 cell | sell 0.5 | buy 0.5 |
|---|---|---|---|---|
| **extreme (brief)**: sell at low, buy at high | **77.4 %** | **77.1 %** | 84.1 % | 84.7 % |
| second's close | 83.4 % | 83.5 % | 90.5 % | 90.4 % |
| second's VWAP, rounded to the tick | 87.5 % | 87.2 % | 93.2 % | 93.2 % |

- Per trade, the extreme rule puts **55.7 %** of sell and **54.1 %** of buy volume at the exact traded price.
- It is **biased by side**: sells land on average **−0.16 pts** below their traded price, buys **+0.17 pts** above.
- So on approximate data sell heat is pushed toward each second's low — at a swing low, onto the low itself — which
  is the condition the "sellers absorbed" rule tests. This is why DC11 exists. (`q_exact_vs_approx.py`, `.out.json`.)

### Q4 — code paths (smallest change)

- **Flow v2** (`apps/web/modules/Ironbeam/flow_v2.py`, `/api/ironbeam/flow?mode=v2`): segments, sources, A1.4
  contract choice, `until`, caches. v3's cells endpoint reuses `_segment_minutes`-style contract choice and the
  same source rules; the v2 builder is not edited.
- **1-s reads:** both 1-s tables have per-second `high` / `low` / `buy_vol` / `sell_vol` (and `notional`), so the
  1-s rule is one `GROUP BY minute, low` + one `GROUP BY minute, high` per segment.
- **Exact reads:** `ironbeam_es_trades` (`ts_utc`, `symbol`, `price`, `size`, `aggressor_side`); one
  `GROUP BY minute, price, aggressor_side` per segment.
- **Frontend:** `App.jsx` holds `flowMode` (`ib-react-flow-mode`), `flowV2Payloads` and CR-CH's memoised
  `flowV2PanePayloads`; `FlowModeToggle` (`AggressorFlowPanel.jsx:272`) is shared by both pane headers (DC1).
  `PriceChart.jsx` uses lightweight-charts **5.1.0**; it has no series primitive yet. Heatmap, profile and marks
  will each be one `ISeriesPrimitive` attached to the candle series (heatmap `zOrder: 'bottom'`, behind the candles),
  drawing visible cells only (time-sorted arrays, binary search on the visible range), with inputs memoised on
  (payload identity, interval) — no series, price lines or markers per cell. Price-bar clicks go through
  `handleClick` (`PriceChart.jsx:1850`: GEX popup, else timeslice toggle) — DC3.
- **CR-CH repro tooling** is in the CR-CH worktree (`/Users/ryan/code/volatility-monorepo-CR-CH/scripts/cr_ch_repro/`,
  untracked); it will be copied for the headed 10-min run.

### Gate

No stop on rulings. **Implementation waits for Ryan's rulings on DC1–DC16** (DC1, DC3, DC4 and DC11 change the
design most; the rest are operational readings that stand unless Ryan objects).

## Implementation order (one commit each, after the rulings)

1. `packages/shared/flow_heat.py` — cells (exact and 1-s), absorption marks, anchored-CVD reference; unit tests
   (anchored CVD, the 1-s assignment, the absorption rule incl. causality and `until`).
2. Backend: `GET /api/ironbeam/flow/cells` (DC13), sources per segment (DC12), caches (DC14); tests; v2 and Current
   goldens unchanged.
3. Frontend: toggle + new key (DC1, DC2), v3 pane (anchored CVD, rolling delta, v2 histogram), anchor control (DC3–DC5).
4. Frontend: heatmap, absorbed marks and profile primitives (DC7–DC10, DC15, DC16), ⚙, "?" guides, badges.
5. Smoke: Playwright (pixel identity vs `origin/main`, v3 on 2026-05-01 and live, defaults off, replay `until`),
   headed Chrome 155 10-min drag/zoom with heap samples, exact vs approximate mark counts on a retained session; PR.

---

## Amendment A1 (Ryan, 2026-10-27, after Step 0) — rulings on DC1–DC16, new DC17

| # | Ruling |
|---|---|
| A1.1 | **Accepted as recommended:** DC1 (a), DC3 (a), DC4 (b, "anchor = last swing"), and the operational readings DC2, DC5–DC10, DC12–DC16 as written. |
| A1.2 (DC11, **changed**) | The 1-s low/high rule stays for the **heat layers and the profile**, with the "approximate: 1-s data" badge. **Absorption marks are NOT drawn on approximate data** (Step 0 Q3: the rule shifts sells to lows and buys to highs, which manufactures "sellers absorbed" at lows). **Marks only on exact data.** On an approximate segment the payload carries no marks and the pane says "Absorbed: exact data only". |
| A1.3 (DC17, **new**) | The cells endpoint reads, per segment, in this order: **(1)** a new exact table **`es_flow_price_1m`** (minute × 0.25-pt price × aggressor side; being built by CR-DD), schema `(minute_utc, symbol, price, buy_vol, sell_vol, trade_count, source)`; **(2)** `ironbeam_es_trades`; **(3)** the 1-s fallback. Coded against that name and schema; **until the table exists the endpoint skips it** (`to_regclass`, re-checked as v2's 1-minute tables are). Exact sources (1) and (2) carry marks; (3) does not. |

### A1 operational readings (Claude Code, fixed before implementation)

- **`es_flow_price_1m` use (A1.3).** `source` must be the segment's source by the v2 rules: `databento` only for RTH of sessions ≤ 2026-06-05 (B4, A1.1 of CR-CD); `ironbeam` otherwise. `symbol` = the segment's contract chosen as v2 (A1.4 bar match). The table **covers** a segment when its distinct minutes for that symbol and source are ≥ 99 % of the minutes with ≥ 1 trade in the segment's 1-minute flow (`es_flow_1m_databento` / `ironbeam_es_flow_1m`, via v2's `fetch_flow_minutes`); otherwise it is skipped for that segment. Its `price` is taken as the 0.25-pt cell floor; rows off the 0.25 grid are floored.
- **`ironbeam_es_trades` use.** Covers a segment when its oldest row (cached 60 s) ≤ the segment start and the segment's symbol is present. `aggressor_side` `BUY` → buy, `SELL` → sell; anything else (e.g. `0`) is not drawn.
- **Hold / break extremes for marks** come from the segment's 1-minute flow highs and lows (the same trades as the exact cells for Ironbeam; the same feed for Databento).
- **Badge per segment:** `exact: es_flow_price_1m` / `exact: trades` / `approximate: 1-s data`, plus the source.
