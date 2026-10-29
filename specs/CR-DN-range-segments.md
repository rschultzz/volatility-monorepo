# CR-DN — Flow v3 ranges as segments: set end, breaks need acceptance, CVD and stats per box

Spec copied from the vault session note `sessions/2026-10-29 - CR-DN — Range Segments.md` (Ryan's kickoff, chat
2026-10-29).

Branch `feat/CR-DN-range-segments`, worktree `../volatility-monorepo-cr-dn`. **Started off `origin/main` `78a50fb`
before CR-DM merged**, by Ryan's instruction in chat 2026-10-29: spec freeze and Step 0 only, no code. Implementation
starts only after CR-DM is merged: rebase onto `origin/main`, re-check Step 0 against CR-DM's code, record that here,
then items 1–3. CR-DM's worktree and branch are not touched.

## Why (Ryan, chat 2026-10-29)

Ryan anchors ranges by hand and wants to study each range on its own: its delta trend and stats, start to finish,
and to set where a window ends. Today one CVD line runs across every box, and a failed break splits a range in two.
On 2026-10-29, anchored at 06:49, a 07:34 close above the top froze the box though price came straight back inside,
and the new box grew back over the old range.

## Spec (Ryan, chat 2026-10-29)

1. **Set end.**
   - a. A "set end" button beside 📍 set anchor, enabled once an anchor is set: click it, then click a price bar
     after the anchor. The end must be after the anchor and not after the payload's last bar (live: now; replay /
     review: the until time). Esc while picking cancels; "clear" removes anchor and end; a small "clear end"
     removes only the end.
   - b. With an end set, everything anchored uses anchor → end only: anchored CVD and readout, Edges box(es) (the
     last one ends at the end), the Edge effort strip, the volume Profile.
   - c. After the end: CVD isn't drawn; boxes and the strip stop. Per-bar series (delta bars, volume bars, rolling
     delta if on) and absorption levels / marks are unchanged.
   - d. The window is shaded lightly on the price chart (existing slate, low opacity, C1); the header shows e.g.
     "06:49 → 06:57 · 8 min".
   - e. Same persistence as the anchor today (stated in Step 0).
2. **A break needs acceptance** (replaces the single-close break from CR-DI item 9).
   - a. A close more than 2 pts beyond an edge that has held N min starts a pending break. The box freezes only if
     no 1-minute close comes back inside within A minutes (⚙ "Break must hold A min", default 5, range 1–15). The
     freeze time and the new box's start are the original break bar.
   - b. If a close comes back inside within A minutes, the break is cancelled: the box is unchanged (edges don't
     move), and a "probe hh:mm" tick is drawn at that edge (existing magenta, low opacity). Hover: probe extreme,
     minutes outside, contracts traded beyond the edge.
   - c. While pending, the header shows "testing break hh:mm"; nothing freezes or grows until it resolves.
   - d. Edge growth for a box younger than N min (CR-DI) is unchanged.
3. **Boxes as segments.**
   - a. With Edges on and an anchor set, each box is a segment: the anchored CVD restarts at 0 at each segment
     start and is drawn per segment, with a small gap or tick at each boundary.
   - b. Each box gets a small label inside its top-left: "Δ +1.2k · +7.5 pts · 45 m · 1.8k/pt". Hover: top /
     bottom, edge counts by side (main / opposite, as the Edge effort strip), held minutes per edge, probe count,
     and how it ended (broke ↑/↓ hh:mm, set end hh:mm, or current).
   - c. The header readout shows the current box (or the last one, if an end is set).
   - d. At most the 4 most recent boxes are drawn (as CR-DI); older segments' CVD is not drawn either, and the
     hover notes how many are hidden.
4. **Edges off:** CVD and readout behave exactly as today (one anchored line, anchor → now / end), apart from
   item 1.

## Standing rulings conformance (UI CR)

Read `research/standing-rulings.md` (2026-10-29 copy, last ruling A22, last change-log line CR-DM) before the freeze.

| Ruling | Status | Note |
|---|---|---|
| A1 — a range develops in a zone around the level | n/a (context) | display of hand-anchored ranges; no level or zone logic |
| A2 — 2–3 min is not a range; no duration limit | n/a (display simplification) | N (hold), A (break must hold) and the end are ⚙ / hand-set drawing aids, not range definitions; the box label's minutes are a readout |
| A3–A18 | n/a | no setup, level, entry, exit or outcome logic |
| A19 — in-range confirmation, both sides | n/a (context) | per-box CVD and edge counts display flow inside a range; not a confirmation rule |
| A20 — trade the range edges | n/a (context) | display only; no entry logic |
| A21, A22 | n/a | |
| A27 (draft) — "a failed probe can strengthen the fade; untested" | **not in the rulings file** | the file ends at A22 and no vault note mentions A27. The legend quotes the kickoff's wording, labelled untested; nothing is added to `standing-rulings.md` (only Ryan changes a line). Flagged to Ryan |
| B1–B7 | n/a | no hypothesis tested, no data split read beyond what the chart already displays |
| B8 — Code smoke labels are never Ryan's | honoured | no review labels written; smoke runs against a read-only backend |
| C1 — no green / red; Dash palette | honoured | end window in the existing slate at low opacity; probe ticks in the existing magenta (`#e879f9`, CR-DF) at low opacity; per-box CVD in the existing CVD colour; labels in the pane's label style |
| C2 — new display features off by default, existing views unchanged | item 1 honoured; items 2–3 **deviate — APPROVED by Ryan in chat 2026-10-29** | item 1 is a new control: nothing changes until an end is set. Items 2 and 3 change existing behaviour with Edges on (a break needs acceptance; CVD restarts per box; per-box labels). Change-log line added to `standing-rulings.md` |

Nothing in "Revoked / superseded" (R1–R7) is used. A, M and N (kickoff wording; M = CR-DM's live effort min move) and
the end are display settings, not tested thresholds; the legend says so and notes A27's draft wording ("a failed probe can strengthen the fade; untested").

## Step 0 (commit before implementation)

- How the anchor is stored and persisted today.
- Where CR-DI's break / restart and header counting live.
- How the anchored CVD and readout are computed, so the per-segment version reuses that code path.
- On 2026-10-29 anchored at 06:49: every break CR-DI's current rule produces and what each becomes under item 2
  (A = 5) — probe or held.
- (Ryan, chat 2026-10-29) Anything that depends on CR-DM's pane changes (volume row, rolling delta in ⚙), to be
  re-checked after the rebase.

## Commits

spec freeze → Step 0 → (after CR-DM merges: rebase + Step 0 re-check) → one commit per item (1, 2, 3) → smoke + wrap.

## Smoke (read-only backend)

- No end, Edges off → identical to origin/main.
- 2026-10-29, anchor 06:49, Edges on:
  - the 07:34 close above the top becomes a probe; one box runs from 06:49 to now, unless a later break holds;
  - with A = 1 the old split appears, with the CVD restarting at 07:34;
  - per-box labels equal hand sums from the minute data;
  - set end at 06:57 gives a single box 06:49–06:57, with the readout, strip and profile using only those minutes.
- A date with a break that holds still freezes and restarts.
- Replay never uses bars after until.
- Existing specs green.

## Constraints

- Work only in `../volatility-monorepo-cr-dn`; CR-DM's worktree and branch are not touched.
- One DB-backed local server at a time; check first and never stop another session's server without asking Ryan.
- Never print connection strings: catch DB errors and print the error type only.
- Rebase onto origin/main before the PR. PR not merged; not deployed. Logs copied to the main checkout at wrap.

## Step 0 findings (read-only, against `origin/main` `78a50fb`; CR-DM not merged)

Data: `/flow/cells` payloads built by `flow_v3.build_cells_payload` through one read-only connection, no server
(`scripts/logs/cr-dn/step0/dump_cells.py`, CR-DI's dumper with DB errors printed as type only). 2026-10-29 RTH
dumped live at 15:23 UTC (bars 06:30–08:23 PT, 114, exact on every bar); 2026-10-28 RTH complete. Breaks from
`scripts/logs/cr-dn/step0/breaks.mjs`, which runs the shipped `flattenBars` + `edgeReadout` and simulates item 2.
Outputs in `scripts/logs/cr-dn/step0/` (git-ignored; copied to the main checkout at wrap).

### 0a — how the anchor is stored and persisted

- **App state only, no persistence.** `App.jsx:593` `const [flowAnchor, setFlowAnchor] = useState(null)` and
  `flowAnchorArmed` (`:594`). No localStorage / URL key (the localStorage keys in App are the Flow mode, v3 on,
  `FLOW_V3_SETTINGS`, `FLOW_V3_LAYERS`, Edge effort chip, heights; not the anchor).
- **Value:** a *shifted* PT epoch (the price chart's candle time). Set by `handleAnchorPick` (`App.jsx:1275`) from
  the price chart's click while armed (`PriceChart.jsx:1933`: an armed pick consumes the click, no timeslice / GEX
  popup), or by the panel's "session open" / "last swing" buttons (`setAnchorTo`, panel `:519`).
- **Cleared** on a change of trade date, Flow session or v3 on/off (`App.jsx:1279–1283`, DC5), by Esc when not armed
  (panel `:315–326`; Esc while armed only disarms) and by "clear". A page reload loses it. Not cleared when the review
  `until` changes.
- **So item 1e = the same: the end is `useState` in App beside the anchor, cleared by everything that clears the
  anchor** (and by "clear end"), not persisted.

### 0b — where CR-DI's break / restart and header counting live

All in `react_price_preview/src/flowV3.js`, computed once in App and passed down:

- `edgeReadout(payload, bars, anchorShifted, shift, holdMin)` (`flowV3.js:484`): the box loop over the cells
  payload's 1-minute closes (`flattenBars`, UTC bar starts). A close > `EDGE_BREAK_PTS` (2) beyond an edge grows the
  box while that edge has stood < `holdMin` (⚙ N, `DEFAULT_V3_SETTINGS.boxHoldMin` 15) since it was set; otherwise it
  is a break, pushed to `spans` as `[j0, i−1, dir]`, and the new box starts at bar i with top = bottom = its close
  (both edges set there). **This loop is where item 2's pending state goes**; nothing else decides a break.
- `boxStats(...)` (`:437`): one box's edges, first touches, growth moves, and the per-minute edge cells from
  `edgeCounts` (`:396`, ±1 pt inclusive against the box's final edges). Frozen boxes add `breakAt`, `dir`,
  `heldMin`, per-edge `heldMin`.
- `drawnBoxes` (`:548`, last `FROZEN_BOXES_MAX` = 4 frozen + current), `edgeEffortSeries` (`:557`, the strip's
  minutes per drawn box), `edgeVisits`, `edgeStripColumns`, `edgeText` (`:663`, header text "Box from hh:mm ↑ · Top …
  | Bottom …").
- App: `flowV3Edges = edgeReadout(flowCells, flowCellsBars, flowAnchor, flowV3Shift, flowV3Settings.boxHoldMin)`
  (`App.jsx:2004–2007`) → the panel (`edges` prop: header text, strip) and the price chart (`flowV3Chart.edges` →
  `PriceChart.jsx:2645–2662` → `EdgeLinesPrimitive` in `flowV3Primitives.js:348`, boxes + frozen-box hover).
- The volume Profile: `profileTotals(flat, fromShifted, toShifted = Infinity)` (`flowV3.js:213`), called in
  `PriceChart.jsx:2635–2641` with `flowV3Chart.profileFrom = flowAnchor` and no `to`. **It already takes an end**;
  item 1b passes `to` = the end bar's start + 60 s (shifted; cells are minute starts, `[from, to)`).

### 0c — how the anchored CVD and readout are computed

- Both run in the panel (`AggressorFlowV3Panel.jsx:254–262`) on **the v2 flow payload's bars** aligned to the candles
  (`alignV2ToCandles(allBars, candles)` → `aligned` rows with shifted `time`, `delta`, `vol`, `seg`), **not on the
  cells payload** the boxes come from. Both payloads are truncated server-side at the review `until` (every flow URL
  in App sets `until`, e.g. `App.jsx:1958`, `:2063`).
- `anchoredCvd(rows, anchorTime)` (`flowV3.js:50`): `floorIndex` of the anchor in the rows; `out[a] = 0`; adds
  `delta` from bar a + 1 on; null before the anchor. **The anchor bar's own delta is not counted.**
- `anchorReadout(rows, closes, anchorTime)` (`flowV3.js:68`): over bars a + 1 … last with data: Σ delta, Σ vol;
  dPrice = close[last] − close[a]; minutes = (t[last] − t[a]) / 60; cpp = vol / |dPrice|. Header (panel `:611–613`):
  "since hh:mm: Δ … · ±x.xx pts · m min · k/pt". `vol` is the flow volume (CR-DM Step 0: buy + sell + unknown).
- **Per-segment version, reusing that path:** add an optional end (`endTime`) to `anchoredCvd` / `anchorReadout`
  (null = today's behaviour, so Edges off and no end are byte-identical), and build the per-box series by calling them
  once per box with (box start, box end), concatenated into one array aligned to `aligned` (null at gaps). Box spans
  come from `edgeReadout` (UTC) mapped with `shift`. Consequences, stated so the smoke's hand sums use the same rule:
  - each box's CVD starts at 0 on its first bar and adds bars first + 1 … last (the CR-DC anchor convention);
  - a frozen box ends on the bar before its break, so **the break bar's own delta is in neither box** (it is the
    0-point of the next one) — exactly what anchoring at the break bar does today;
  - box label numbers = `anchorReadout` over (box start, box end): "Δ {delta} · {dPrice} pts · {minutes} m ·
    {cpp}/pt".
- **The per-box readout must move up to App** (or a pure helper both call with the same inputs): the label is drawn on
  the price chart (inside the box) but the CVD rows live in the panel. Plan: App aligns the v2 bars once (the same
  `alignV2ToCandles` call), computes the box readouts with `edgeReadout`'s spans, and passes them to both.

### 0d — 2026-10-29, anchor 06:49 (close 7736.00), N = 15, bars to 08:23 PT

**CR-DI's current rule produces one break:**

| Box | Edges | Ended | Top buy / sell | Bottom sell / buy |
|---|---|---|---|---|
| 06:49–07:33 | 7736.00–7756.25 | **broke 07:34 ↑** (close 7760.00, 3.75 beyond) | 11,839 / 10,424 | 5,130 / 4,730 |
| 07:34– (current) | 7738.25–7760.00 | current | 7,887 / 6,479 | 4,839 / 4,729 |

Closes after the 07:34 break: 07:35 **7756.25** (exactly at the top) · 07:36 7758.00 · 07:37 7760.25 · 07:38
**7756.00** · 07:39 7759.50 · 07:40 7756.75 · 07:41 7751.00 · …

**Under item 2 (A = 5) it is a probe.** The answer depends on what "a close back inside" means at the edge price
itself, and only the strict reading matches the kickoff's A = 1 expectation:

| "Inside" | 07:34 at A = 5 | 07:34 at A = 1 | Item 2 from 06:49, A = 5 |
|---|---|---|---|
| **strict** (bottom < close < top) | probe, back 07:38, 4 min outside | **held → old split at 07:34** (as the smoke expects) | one box 06:49– 7736.00–7756.25; probes 07:34 (ext 7761.00, 4 min, 8,025 buy / 7,432 sell beyond the edge), 07:39 (7760.50, 2 min, 3,031 / 2,789), 08:10 (7760.50, 5 min, 7,605 / 7,058) |
| inclusive (bottom ≤ close ≤ top) | probe, back 07:35, 1 min outside | probe (07:35 closed at 7756.25) → no split until 07:39 | one box; probes 07:34, 07:37, 07:39, 08:10, 08:14 |

With strict "inside" and A = 1 the split is identical to CR-DI's (frozen 06:49–07:33, current 07:34– 7738.25–
7760.00). **Proposed: strict** (a close exactly on the edge is not back inside); Ryan to confirm. As of 08:23 PT no
later break holds; the smoke re-checks on the complete session.

**Held-break date for the smoke: 2026-10-28, anchor 07:00** (complete session, bars to 12:51 PT). Under item 2,
A = 5, strict: 08:16 ↑, 08:39 ↑, 09:54 ↑ and 11:05 ↓ still freeze and restart exactly as CR-DI (same boxes); the
later CR-DI breaks 11:29 ↑, 12:12 ↓ and 12:36 ↑ become probes, and the boxes after 11:05 differ (frozen 11:05–11:32
↑, 11:33–12:05 ↓, 12:06–12:45 ↑, current from 12:46). Output: `breaks_1028_0700_N15_strict.txt`.

**Definitions used (to implement exactly, unless Ryan says otherwise):**
- *Within A minutes* = the closes of the A bars after the break bar (break + 1 … break + A); a close inside on bar
  break + A still cancels. If none is inside, the break holds, the box freezes at the bar before the break bar and the
  new box starts at the break bar and replays from there with the normal growth rule.
- *Pending:* closes during the A minutes neither grow nor break either edge. A close beyond the *other* edge while
  pending is ignored until the pending break resolves (none on 2026-10-29 / 10-28).
- *After a probe* the edge is unchanged and keeps its set time, so the next close > 2 pts beyond starts a new pending
  break at once (07:39 right after the 07:34 probe).
- *Probe hover:* extreme = highest high (↑) / lowest low (↓) from the break bar to the bar before the return;
  minutes outside = return bar − break bar; contracts beyond = aggressor buy and sell of cells strictly beyond the edge
  price over the same minutes (exact data only, as the edge counts).
- *An end inside a pending window:* the box ends at the end unchanged and its hover says "break pending at end";
  nothing freezes.
- *Live:* a pending break at the last bar shows "testing break hh:mm"; it resolves on later refreshes.

### 0e — items that depend on CR-DM's pane changes (re-check after the rebase)

CR-DM (`feat/CR-DM-volume-strip-live-effort`, read with `git diff 78a50fb …`, not touched) changes the same panel.
On `78a50fb` rolling net delta is always drawn and there is no volume series.

1. **Item 1c's "unchanged" list** names volume bars and "rolling delta if on": both exist only after CR-DM (volume
   series `s.vol`, `settings.rollingOn`). Re-check that an end leaves both untouched.
2. **CVD series and `setData` block** (panel `:340–428`): CR-DM adds `s.vol`, moves rolling to its own scale
   (`priceScaleId: 'rolling'`, `rolling` null when off) and rewrites the same `setData` effect. Per-segment CVD edits
   that block → implement only on the rebased code.
3. **Pane slot layout:** CR-DM's `paneSlotMargins` / `applySlots` set the `cvd` scale's margins. Per-segment CVD stays
   one series on `priceScaleId: 'cvd'` (one array with resets and whitespace gaps), so `applySlots` keeps working;
   re-check with rolling on (pane grows) and with the Edge effort strip on.
4. **⚙ settings:** CR-DM adds `rollingOn` and `liveMinMove` to `DEFAULT_V3_SETTINGS` and the form. Item 2's
   `breakHoldMin` (A, default 5, 1–15) goes into the same object and form → merge there.
5. **Header readout / hover JSX:** CR-DM changes the no-anchor text ("CVD from session open · … volume per minute")
   and the minute hover (`minuteFlow`). Item 3c changes the anchored text in the same block; the hover keeps
   `cvd[hoverIdx]` (the per-segment array is still aligned to rows).
6. **App:** CR-DM's `flowPaneExtra` adds `rollingExtraHeight`; CR-DN changes no heights. `flowLiveEffort` (live
   effort label) is swing-based, not anchored, so an end does not change it (not in item 1c's list; re-check).
7. **PriceChart:** CR-DM adds the live-effort hover and an import; CR-DN adds the end shading, probe ticks and box
   labels near `EdgeLinesPrimitive` — no overlap beyond the import line.
8. **Volume field:** the box label's `/pt` uses `aligned[i].vol`, already on `78a50fb` (CR-DM uses the same field
   for its bars). No backend change for CR-DN.

### 0f — other notes

- **A27 is not in `standing-rulings.md`** (it ends at A22; no vault note mentions A27). The legend will quote the
  kickoff's wording, labelled untested; nothing is added to the rulings file.
- 5-minute chart: a click picks a 5-min candle; the end is taken as **the last 1-minute bar inside that candle**
  (so the box, strip and profile include the whole candle). On the 1-minute chart it is that minute.
- Replay: if `until` moves before the end (the payload shrinks), the end is dropped (as if "clear end") rather than
  clamped.
- No backend change is needed for items 1–3.
