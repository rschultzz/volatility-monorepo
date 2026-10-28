---
type: cr
cr_id: CR-DF
title: Flow v3 — absorption levels and an edge readout (display only; replaces the heatmap as the main view)
status: in-progress
run_mode: interactive
data_safety_class: read_only
dependencies: [CR-DC, CR-DD]
branch_name: feat/CR-DF-absorption-levels
tags: [dash, cr, order-flow, aggressor, price-chart, frontend, absorption, display-only]
---

# CR-DF — Flow v3: absorption levels and an edge readout

Display only, read-only DB. Branch `feat/CR-DF-absorption-levels` off `origin/main` `4db4173` (CR-DC merged), own
worktree (`.claude/worktrees/cr-df`). Kickoff: Ryan, 2026-10-27 (chat). Rulings DF1–DF4: Ryan, 2026-10-27 (chat),
before freeze. Frozen 2026-10-27 before any code.

---

## Kickoff (verbatim)

```
CR-DF — Flow v3: absorption LEVELS and an edge readout (display; replaces the heatmap as the main view).

Branch first: git checkout -b feat/CR-DF-absorption-levels off origin/main. Own worktree.
Read CLAUDE.md, standing-rulings.md (C1, C2), the CR-DC spec (cells endpoint, exact-data rule, absorption
rule, performance rules). Ryan, 2026-10-27: the heatmap is hard to see and looks like the bars; he wants
to see levels that have been absorbing recently, and whether we are absorbing at the range edges.
C1 deviation APPROVED by Ryan: absorption uses a new distinct colour (magenta, e.g. #e879f9) so it can't be
confused with candles (blue/grey) or GEX lines (yellow/green/blue/indigo); record it in the rulings change
log. Still no red/green.
Build (exact data only — es_flow_price_1m / ironbeam_es_trades; nothing on 1-s approximate data):
1. "Absorption levels" toggle in the v3 header (off by default): a horizontal magenta line at each price
   that meets CR-DC's absorption rule, starting at the minute the volume built up and ending when price
   trades through it (or continuing to now if it hasn't). Thickness ∝ contracts absorbed (log scale,
   capped); opacity fades with age (full at 0 min → 25 % at 60 min). Small label at the right end:
   "sellers absorbed 3.2k · 4 min" / "buyers absorbed …". Up to the 8 most recent/largest at once.
   Hover: start time, contracts, opposite-side contracts, held minutes, broke at.
2. Edge readout in the v3 header, when an anchor is set: box = highest / lowest 1-min CLOSE since the
   anchor; for each edge, aggressor contracts traded within 1 pt of it since the anchor (buyers at the top,
   sellers at the bottom), whether price closed beyond it by > 2 pts (held / broke), and minutes since the
   edge was first touched. Format: "Top 7724.75 · 1.8k buyers · held 9 m | Bottom 7715.25 · 4.1k sellers ·
   held 6 m". Thin dashed magenta lines mark the two edges when the readout is on.
3. Heatmap: keep as is but rename "Heatmap (cells)" and move it under ⚙; still off by default.
Performance as CR-DC (one canvas primitive per layer, visible range only, memoised). '?' guide: "visual aid
— untested". REVIEW replay never past `until`.
Tests: unit tests for level start/end, fading, edge stats; Playwright on 2026-05-01 and the live date with
levels and the readout on; Current and v2 pixel-identical; the headed-Chrome 10-min drag/zoom test with all
v3 layers on — no crash, no heap growth. PR, don't merge; Ryan merges, chat deploys.
```

---

## Rulings before freeze (Ryan, 2026-10-27, chat)

The brief left four choices open; Claude Code listed them with recommendations and Ryan ruled before freeze:
"DF1–DF4 approved as recommended."

| # | Choice | Ruling |
|---|---|---|
| DF1 | CR-DC's **Absorbed** toggle (slate triangles + slate line) | **Absorption levels replace it as the main view** — same rule (DC9), same ⚙ K / M. The slate marks move under ⚙ as "Absorbed (marks)", off by default, like the heatmap. |
| DF2 | Level start, end and age | The line **starts at the cell's minute** (`time`, the minute the volume built up) but **appears only once the hold window has ended** (`confirm_time` ≤ now / `until`) — causal; replay never shows a level early. It **ends at the break bar** (DC9 break: first bar with low < P − 1 tick / high > P + 1 tick) or runs to now / `until`. **Age** = now / `until` − start minute; broken levels fade too. |
| DF3 | "Up to the 8 most recent/largest" | The **top 8 by contracts × fade weight** (fade weight = the opacity of DF2's age), so old levels drop out by themselves and big recent ones win. |
| DF4 | The brief's edge rule contradicts itself (a highest close can never be closed beyond) | **The edge moves only with the 2-pt buffer:** a close ≤ 2 pts beyond the edge is a test (edge stays, still "held"); a close **> 2 pts** beyond shows **"broke hh:mm"** and the box moves to that close. **"held N m"** counts from the first minute whose high (top) / low (bottom) came within 1 pt of the edge. **Edge contracts from exact cells only** ("exact data only" otherwise). Ryan: this matches **CR-CP's A1 close-based box and void rule** (CR-CP A1.1: box edges on 1-min closes, wicks neither define nor void it, void only on a close > 2 pts beyond an edge; A1.2 / A1.a: the extreme moves only when a close is > 2 pts beyond the running extreme). |

Kickoff-approved deviations: **C1** — magenta `#e879f9` for absorption levels and edge lines (Ryan, kickoff).
**C2** — item 3 moves "Heatmap" from the v3 header into ⚙ (and DF1 moves "Absorbed" there), which changes the v3
header CR-DC built; both stay off by default, and Current / v2 stay pixel-identical. Recorded in the rulings change
log, not stopped for.

---

## Standing rulings conformance (`Dash/research/standing-rulings.md`, status current, read 2026-10-27)

CR-DF is a display CR, not a research CR; the table is filled anyway because the kickoff names C1 and C2.
Nothing in "Revoked / superseded" (R1–R7) is used.

| ruling | status | how |
|---|---|---|
| A1–A4, A6–A16, A18–A21 (setup) | n/a | no setup is defined, tested or shortlisted. The edge readout's box follows CR-CP A1's close rule (DF4) as a display, not a range qualification (no 15-min hold, no 12-pt width) |
| A5, A17, A19, A20 (flow at the range edges) | n/a | the readout shows aggressor contracts at the edges; it makes no fade / break call. Every "?" guide says "visual aid — untested" |
| B1–B3, B6–B8 (tests) | n/a | no test of a setup; no verdict |
| B4 (data split) | **honoured** | sources exactly as CR-DC (Databento served only for sessions ≤ 2026-06-05). Step 0 reads are mechanical (counts of levels, contract sizes for the line-width scale, timing) on train sessions and the live / forward session — no direction, outcome or feature statistic |
| B5 (Databento is the reference) | honoured | display only; badges name the source, as CR-DC |
| C1 (palette) | **deviates — approved by Ryan in the kickoff** | magenta `#e879f9` for absorption levels, their labels and the dashed edge lines, so they can't be confused with candles (blue / grey) or GEX lines (yellow / green / blue / indigo). No red / green. Logged in the rulings change log |
| C2 (off by default; existing views unchanged) | honoured for Current / v2; **v3 header change pre-approved in the kickoff** | everything new is off by default; Heatmap and Absorbed (marks) move into ⚙ (kickoff item 3, DF1), still off by default; Flow Current and v2 pixel-identical to `origin/main` (Playwright) |

---

## Design (as ruled)

**Data — exact only.** Levels come from CR-DC's marks unchanged (`flow_heat.absorption_marks`, DC9 rule, ⚙ K / M,
RTH only, exact tiers only — A1.2). The edge readout's contracts come from the payload's exact cells; if any segment
the anchor window touches is approximate, the readout prints "exact data only" in place of the contracts (the box,
held / broke and minutes still show, because they need only prices). The box uses the segment's **1-minute flow
closes** — the same series the marks' hold / break use (CR-DC A1 readings) and the same feed as the cells — which the
cells payload gains as `bars: {t, h, l, c}` per segment (the only backend change; cut at now / `until` as the cells
are). On 5-minute charts the readout still uses 1-minute closes (the brief says 1-min CLOSE).

**Levels (item 1).** For each mark with `confirm_time` ≤ now / `until`:
- start = `time`; end = `break_time` or now / `until` (DF2); contracts = `aggressor + merged_vol`;
- age (min) = (now / `until` − start) / 60; opacity = 1 − 0.75 × min(age, 60) / 60 (100 % → 25 % at 60 min,
  25 % after);
- thickness = log-scaled contracts, capped (scale fixed in Step 0 from the measured contract distribution);
- shown: the top 8 by contracts × opacity (DF3);
- label at the right end: "sellers absorbed 3.2k · 4 min" / "buyers absorbed …" (minutes = held minutes, DC9);
- hover: start time, contracts, opposite-side contracts, held minutes, "broke at hh:mm" / "not broken".
One canvas primitive draws all levels and labels; visible range only; inputs memoised.

**Edge readout (item 2)**, with an anchor set, from the anchor bar's minute to the last minute with data:
- top = bottom = the anchor minute's close; each later close c: c > top + 2 → **break**, top = c (record the
  break time); c ≤ top + 2 → test, top unchanged. Mirror for the bottom (c < bottom − 2);
- contracts = Σ aggressor contracts of cells since the anchor with |price − edge| ≤ 1 pt (buyers at the top,
  sellers at the bottom), against the current edge;
- first touch = the first minute since the anchor with high ≥ top − 1 pt (bottom: low ≤ bottom + 1 pt); "held N m"
  = last minute − first touch;
- status: "held N m"; after a break since the anchor the edge text adds "broke hh:mm" (the latest break);
- dashed magenta lines (thin) at the two edges from the anchor to now / `until`, drawn by one primitive.

**Header and ⚙ (item 3, DF1).** v3 header: "Absorption levels" toggle (off by default), "Edge readout" toggle (off by
default; shown with an anchor), Profile (unchanged). ⚙ gains "Heatmap (cells)" and "Absorbed (marks)" switches, both
off by default; the existing sell / buy layer switches and K / M stay. Stored layer keys: an old stored `heat` /
`marks` value carries over (a viewer who had them on keeps them on, now under ⚙).

**REVIEW replay** never past `until`: the payload is already cut at `until` (CR-DC); levels and edges read only it.

---

## Implementation order (one commit each)

1. Backend: `bars` per segment in `GET /api/ironbeam/flow/cells`; test (cut at `until`, equal to the minutes used for
   marks). No other route touched.
2. `flowV3.js`: `absorptionLevels` (start / end / fade / top 8 / labels) and `edgeReadout` (DF4 box, contracts, first
   touch, breaks, exact-only); unit tests.
3. Frontend: levels and edge primitives (magenta, one each), header toggles, ⚙ Heatmap (cells) / Absorbed (marks),
   hover, "?" guides; layer-key carry-over.
4. Smoke: Playwright (Current / v2 pixel identity vs `origin/main`; 2026-05-01 and live with levels and readout on;
   defaults off; replay `until`); headed Chrome 10-min drag / zoom with all v3 layers on, heap samples; PR.

---

## Step 0 findings (2026-10-27, read-only)

**Method.** `scripts/cr_df_step0/q_levels.py` (output `q_levels.out.json`): the real cells builder
(`flow_v3.build_cells_payload`, RTH, K 3 / M 3) on `BACKFILL_DATABASE_URL` with `default_transaction_read_only = on`
and the role asserted. Sessions: every 10th train session with Databento `es_flow_price_1m` rows (11 sessions,
2026-01-06 → 06-01) and the forward session 2026-10-27 (Ironbeam). Mechanical counts only (levels, sizes, timing) —
no direction, outcome or feature statistic.

**B4 disclosure.** The script's first run also included **2026-10-26** (a retained Ironbeam session, which is in
the frozen holdout 2026-06-06 → 10-26). It computed the same mechanical counts only (11 marks, sizes, timing); the
script was then changed to exclude the holdout and re-run, and the numbers below and the stored output do not use
2026-10-26. Nothing was read from it beyond those counts, and nothing in the design depends on it.

| session | tier | marks (levels) | sellers | contracts p50 / max | held p50 (min) | broken | minutes with > 8 confirmed | max levels < 60 min old |
|---|---|---|---|---|---|---|---|---|
| 2026-01-06 | price_1m (Databento) | 19 | 11 | 484 / 1,399 | 10 | 84 % | 30 % | 8 |
| 2026-01-21 | price_1m | 5 | 1 | 317 / 379 | 45 | 60 % | 0 | 3 |
| 2026-02-04 | price_1m | 0 | 0 | – | – | – | 0 | 0 |
| 2026-02-19 | price_1m | 8 | 4 | 252 / 343 | 14 | 50 % | 0 | 4 |
| 2026-03-05 | price_1m | 1 | 1 | 231 / 231 | 13 | 0 | 0 | 1 |
| 2026-03-19 (roll week) | price_1m | 17 | 13 | 32 / 170 | 32 | 71 % | 42 % | 6 |
| 2026-04-02 | price_1m | 6 | 2 | 259 / 992 | 6.5 | 67 % | 0 | 3 |
| 2026-04-17 | price_1m | 3 | 1 | 371 / 619 | 12 | 100 % | 0 | 1 |
| 2026-05-01 | price_1m | 5 | 4 | 296 / 509 | 11 | 100 % | 0 | 5 |
| 2026-05-15 | price_1m | 6 | 4 | 316 / 419 | 9 | 83 % | 0 | 3 |
| 2026-06-01 | price_1m | 20 | 11 | 339 / 1,426 | 12 | 65 % | 31 % | 9 |
| 2026-10-27 | price_1m (Ironbeam) | 17 | 6 | 425 / 851 | 11 | 71 % | 18 % | 6 |

Pooled contracts per level (n 107): p10 39, p25 259, **p50 350**, p75 444, p90 656, p99 1,375, max 1,426.

**What this fixes (readings, within DF1–DF4):**
- **Thickness scale (DF2):** width px = clamp(1, 5, 1 + 4 × ln(contracts / 50) / ln(30)) — 1 px at ≤ 50 contracts,
  ≈ 3.3 px at the median (350), 5 px at ≥ 1,500 (cap, above the observed max). Log, capped, as the brief.
- **The 8-cap (DF3)** binds on 4 of 12 sessions for 18–42 % of the session (all confirmed levels count, since a
  level never disappears, it fades to 25 %); levels under 60 min old exceed 8 on one minute in one session, so the
  cap mostly removes faded old levels, as intended.
- **Most levels break** (median ≈ 70 %) after a median ≈ 11 min, so the line usually ends well before now; the label
  sits at the line's right end (the break bar or now / `until`).
- **The 2026-03-19 roll week** shows small levels (p50 32): the RTH contract chosen by v2's bar match (A1.4) is the
  thin one there, which is CR-DC's existing behaviour, not changed here.
- **Bars for the edge readout:** the segment minutes v2 already picks (`_segment_minutes`: `o, h, l, c, …` indexed
  by minute start, cut at now / `until`) have 390 minutes on every RTH session above, the same as the minutes with
  cells. Adding `bars: {t, h, l, c}` per segment is ≈ 390 × 4 numbers per segment.

**Gate.** No ruling changes and no stop. Implementation proceeds in the frozen order.

---

## Implementation and smoke (2026-10-27/28)

| step | commit | what |
|---|---|---|
| freeze | `5e343b9` | kickoff verbatim, DF1–DF4, conformance, design |
| Step 0 | `bdb594c` | levels per session, contract sizes → width scale, 8-cap, B4 disclosure |
| 1 | `3cb9af9` | cells payload: `bars {t, h, l, c}` per segment |
| 2 | `d179e35` | `flowV3.absorptionLevels`, `edgeReadout`, `edgeText` + 11 unit tests |
| 3 | `e97eb57` | magenta level / edge primitives, header toggles, ⚙ Heatmap (cells) / Absorbed (marks), guide |
| fix | `e10d54b` | level labels avoid each other |
| harness | `778e57a` | Playwright `cr-df.spec.js`, `cr-dc.spec.js` adapted, smoke backend, headed driver |
| fix | `fd6a705` | App: cells payload only from the current request (see "App guard") |
| harness | `11f848d` | headed driver waits for the anchor reset; v3dc control; review-banner wait 30 s |

### Readings fixed during implementation (Claude Code; within DF1–DF4, no ruling changed)

- **"Now"** for levels and the readout is the end of the payload's last 1-minute bar: live = the last completed data,
  REVIEW replay = `until` (the payload is already cut there). CR-DC's server-side `held_min` is not used for levels
  (see Observations).
- **Anchor minute.** The readout starts at the first 1-minute bar at or after the anchor candle (on 5-minute charts,
  the candle's first minute); the edges start at that minute's close.
- **Contracts within 1 pt** = cells with |price − edge| ≤ 1.00 (four ticks either side, inclusive), from the anchor
  minute through the last minute, against the current edge.
- **After a break** the edge text keeps "held N m" (from the first touch of the new edge) and adds "broke hh:mm" (the
  latest break), so both are visible.
- **Labels** that would overlap an already drawn label are nudged one line up / down, else not drawn (the line and
  its hover remain). Found in the smoke on 2026-05-01, where several levels end at now.
- **Edge lines** run from the anchor candle to the last bar's candle; on 5-minute charts both ends snap to candles.
- **Header.** "Absorption levels", "Profile", then the anchor buttons, "Edges" and the readout text (the header still
  scrolls when narrow, CR-DC). The marks note ("Absorbed: exact data only") also shows with levels on.
- **Smoke backend** caps the Ironbeam engine at 3 + 2 connections (Render DB: 103 slots).

### Tests

- **Python:** `test_flow_cells` 16/16 (3 new: bars sorted, inside the segment, before `until`, every mark break on a
  bar; RTH cell minutes ⊆ bar minutes; CR-DC golden equal apart from `bars`); Ironbeam suite 102/102; `flow_heat` 16/16.
- **Frontend unit:** 28/28 (`flowV3Levels.test.js` 11: fade, width, now, start / end, causal confirm and break, top 8,
  approximate → none, edge tests vs breaks incl. repeated breaks, first touch / held, ±1 pt contracts, exact-only,
  text format).
- **Playwright `cr-df.spec.js` 6/6** (branch dev server vs `origin/main` `4db4173` dev server, read-only backend):
  - *unchanged:* Current and v2 on 2026-05-01 — same flow requests, no cells request; price chart, flow plot and the
    whole flow header byte-identical (pixels and DOM) to `origin/main`.
  - *defaults:* Absorption levels / Profile / Edges off; ⚙ Heatmap (cells) / Absorbed (marks) unchecked; no cells
    request; the price chart with v3 and every layer off is byte-identical to Current's; guide has the new entries,
    ≥ 7 "Visual aid — untested."; a stored CR-DC layers object `{heat, marks: true}` keeps both on in ⚙.
  - *2026-05-01* (RTH exact, `es_flow_price_1m`): 5 marks → 5 levels; header text equal to `edgeText(edgeReadout(…))`
    of the page's payload and anchor (last swing 12:26; an RTH click 09:45: "Top 7280.25 · 12.9k buyers · held 159 m ·
    broke 10:28 | Bottom 7257.50 · 8.8k sellers · held 0 m · broke 12:59"); hover on the largest level ("sellers
    absorbed 509 · 6 min": from 12:36, opposite 176, broke at 12:42).
  - *live 2026-10-28* (before the RTH open): ETH exact (trades); no RTH segment yet → no levels (DC9 RTH only); edge
    readout from "session open" (15:00) with contracts — the header follows each 15 s poll (checked against the last
    payload the page received).
  - *replay* (CR-CC fixture 5, `until` 07:39): every bar < `until`, every mark confirmed ≤ `until`, every level ends
    ≤ `until`, the readout ends before `until` and equals `edgeText` of the replay payload.
- **Playwright `cr-dc.spec.js`** with Heatmap / Absorbed switched in ⚙: 5/5 (v3 defaults, 2026-05-01, exact marks
  2026-10-27, live, replay). Its "unchanged" test compares against a **pre-v3** `origin/main` (DC1: the extra "v3"
  button) and no longer applies now that `origin/main` has v3; CR-DF's identity test replaces it. CR-DC's committed
  assets were restored after the run.

- **Re-run on the final build (after `fd6a705`)**: `cr-df.spec.js` 6/6 and `cr-dc.spec.js` 5/5. During that re-run the
  laptop → Render DB round trip rose to ≈ 0.64 s per query (`/api/ironbeam/bars` 19–64 s, candidate detail ≈ 9.6 s; no
  code of this branch on that path), and three tests hit timeouts: CR-DC "v3 defaults" (120 s test timeout on a
  reload; passed with `--timeout 420000`) and both replay tests (the review banner took ≈ 8 s against a 5 s wait). The
  same CR-DC replay test passed against the `origin/main` frontend at that time; in the browser the branch showed the
  banner after 8 s, so the wait was raised to 30 s in both specs (test only) and both replay tests passed.

Screenshots and JSON: `specs/assets/cr-df/`.

### Headed Chrome, CR-CH / CR-DC-style 10-min drag / zoom

`scripts/cr_df_headed/headed_v3.mjs` (CR-DC's driver + an anchor after every load) against the full Dash app locally
(`scripts/cr_df_headed/dash_app.py`, read-only, production build of this branch), real Google Chrome, live session
2026-10-28, date → 2026-05-01 at 300 s and back at 420 s, the same seeded wheel-zoom / pan / price-axis / time-axis /
reset sequence. Cells: **v3all** = every v3 layer on (Heatmap, Profile, Absorbed, Absorption levels, Edges) with
"session open" as anchor; **v3dc** = control, CR-DC's layers only, same branch and sequence.

| run | actions | crash | page errors | JS heap after forced GC (MB): first → range → last |
|---|---|---|---|---|
| v3all (run 2) | 1,717 | no | 0 | 48.7 → 71–114 → 90.9 |
| v3dc control | 1,706 | no | 0 | 63.9 → 59–130 → 106.4 |
| v3all (final, after the App guard below) | 875 | no | 0 | 56.2 → 63–128 → 81.8 |

The heap rises and falls with the session loaded and with the forced GC — the same sawtooth with and without the CR-DF
layers, no monotonic growth over the run. The final run made fewer actions because its harness waits (up to 2 min)
for each date switch to clear the anchor before re-anchoring, and the input loop pauses meanwhile. Edge readouts in the
final run: live "Top 7766.00 · 859 buyers · held 257 m · broke 20:50 | Bottom 7754.75 · 2.8k sellers · held 568 m ·
broke 15:02"; 2026-05-01 (anchor = 15:00 ETH, approximate → "exact data only") "Top 7299.25 · exact data only · held
356 m · broke 07:21 | Bottom 7246.75 · exact data only · held 1266 m · broke 02:03". Logs (git-ignored):
`scripts/logs/cr_df/` (`run2-*`, `headed-v3dc*`, `headed-v3all*`; runs 1, 3, 4 kept, see below).

**Harness history (disclosed).** Run 1 clicked "session open" before the live session had loaded (button disabled), so
the edges were not on for its first 300 s (no crash, heap 49–131 MB). Runs 2 and 3 logged, right after each date
switch, a readout of the *previous* session. Claude Code first read that as the app using the previous session's
cells and added the App guard below; run 3 (with the guard) logged the same, so the cause was the harness reading the
page before the switch took effect. Run 4 tried to wait for an iframe URL with the new date and found none (the
harness threw at 400 s; no crash). The guard is kept on its own merits (next bullet); the final harness waits for the
anchor reset (CR-DC DC5) instead.

- **App guard (`flowCells` only from the current request URL).** While a new cells request is loading, the previous
  response is no longer used — so after a date / session / K / M / `until` change, levels, edges, heat, marks and
  profile disappear until the new payload arrives instead of drawing the old one. Within the standalone app a date
  change keeps React state, and a later session's levels would otherwise snap onto the new date's last candle for a
  moment. The CR-DC layers are covered by it too (they briefly blank instead of showing stale cells); their content is
  unchanged.

### Observations (not changed; Ryan's call)

- **CR-DC `held_min` on live sessions.** The server computes an unbroken mark's `held_min` to the segment end, which on
  a live session (no `until`) is 13:00 PT, not now — so the **Absorbed (marks)** hover overstates "held" for unbroken
  live marks until the close. Absorption levels compute held minutes client-side to now and are not affected. A
  one-line server fix (stop at the last bar) would change CR-DC's hover; left for Ryan.
- **Price scale.** With GEX lines on, the price axis spans all of them (≈ 7,075–7,525 on 2026-05-01), so the session
  occupies a thin band and the levels and their labels sit close together; zooming the price axis separates them.
- **Levels are RTH only** (DC9), so the live session shows none before 06:30 PT; the edge readout works in ETH.
