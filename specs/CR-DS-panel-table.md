# CR-DS — Box panel as a colour-coded table: verdict, evidence rows, read history

Branch `feat/CR-DS-panel-table`, worktree `../volatility-monorepo-cr-ds`, off `origin/main` `3ec3a42` (CR-DQ merge, PR #125;
deployed 2026-10-31 13:34 PT, restarted 13:40). UI CR, no research claims. Spec: Ryan, chat 2026-10-31 (two pastes, frozen
below, lightly formatted).

## Why (Ryan, chat 2026-10-31)

The panel reads as sentences and is hard to scan. Ryan wants a table where colour shows what each item says: all green = a
clean read; green and red together = contradiction = no action. Also, 2026-10-30 07:39–08:02 (set end 08:02) said "nothing
notable" despite share p93, c/pt p93 and buyers absorbed at the top, because price was 1.25 below the bottom at 08:02. The
decision was right; the label was wrong.

## Spec

1. **Verdict pill** (top of the current box's section):
   - SHORT READ / LONG READ in muted green;
   - CONFLICT in muted red;
   - WATCH BREAK ↑/↓ (idea) in muted amber;
   - NO ACTION and NO TRADES in slate.
   One short reason under the pill, e.g. "short read, but price not at the top".

2. **Evidence table** (current box), one row per check, 3 columns: Check · Value · Reads. Colour by agreement with the
   candidate side:
   - Candidate side: the read the flow points to (buyers leaning → short at the top; sellers leaning → long at the bottom).
     If no flow lean, the side of any absorption at the box; if none, no candidate, and all rows are slate.
   - Muted green: supports the candidate side. Muted red: argues against it. Slate: neutral / not notable / informational.
   Rows:
   - a. Event / time: FOMC (A22) or after the late cutoff, if set → red with "no trades"; otherwise green "ok".
     *(Answer 1: FOMC only; no cutoff control yet — added when Ryan sets the time.)*
   - b. Box formed: minutes and height vs the minimums → green if met, red "still forming" if not.
   - c. Location: where price is vs the box (e.g. "0.50 below the top", "1.25 below the bottom") → green if within 2 pts of
     the candidate edge, red if not.
   - d. Δ share: value + percentile → green if ≥ notable in the candidate direction; red if ≥ notable against it; slate
     otherwise.
   - e. Effort (contracts/pt): value + percentile → green if ≥ notable (no advance); slate otherwise.
   - f. Progress: pts since box start → green if ≤ 2 pts toward the pushing side; red if more than 2 (answer 2); slate only
     with no candidate side.
   - g. Absorption at the box (wicks included, CR-DQ): level, side, time → green if it supports the candidate side, red if
     against; slate if none. Both sides present → one red row, "two-sided absorption".
   - h. Edge counts: Top buyers / sellers, Bottom sellers / buyers → slate (informational).
   The verdict logic and thresholds stay exactly as CR-DQ built them. The table only displays what the verdict already uses,
   plus the edge-count row.

3. **Wording fix:** when the flow checks qualify for a read but location (or box formed, or time) fails, the pill says NO
   ACTION with the reason ("short read, but price not at the top"), never "nothing notable". "Nothing notable" only when no
   flow check passes. *(Answer 3: evaluate "would be a read except for location / box still forming / time" for display
   only; verdict category and order unchanged. "Short read, but box still forming" is the intended wording.)*

4. **Read history:** under the table, "Read live: 07:49–07:58 (short)". These are minute ranges in which a read verdict
   held, replayed minute by minute without using later bars, from anchor → now / set end. Shown for the current box, and for
   frozen boxes on expand.

5. **Frozen boxes:** one compact row each (time span · how it ended · share pNN · c/pt pNN · any read-live window),
   expandable to the same table as of the box's end.

6. **Colours:** flat, muted, not bright, readable on the dark panel in both themes. Green #6f8f72, red #a86464, amber
   #b08a4a, slate (existing). Text stays light; colour goes on a small left bar or chip per row, not full-row fills. Used
   only inside this panel.

7. **Nothing else changes:** chart, shading, header, ⚙ thresholds.

## Standing rulings conformance

Read `research/standing-rulings.md` (2026-10-31 copy: last ruling A22; last change-log line CR-DQ deviation (C2),
2026-10-31) before the freeze.

| Ruling | Status | Note |
|---|---|---|
| A1–A18, A20, A21 | n/a | no range, level, setup, entry, exit or outcome logic changes; boxes are CR-DN's, verdict is CR-DQ's, unchanged |
| A19 — in-range confirmation | honoured (display only) | the table displays the existing short / long read checks; no threshold change |
| A22 — skip FOMC decision days | honoured (display only) | event row (a) red "no trades" on FOMC days (CR-DQ's 2026 list) |
| A23 / A24 (order-flow-reading-notes.md) | draft, not a ruling | used as CR-DQ used them (A24 = CONFLICT), labelled draft |
| B1–B8 | n/a | no hypothesis tested, no data read beyond what the panel already loads; read history replays the loaded session only; B8: no review labels written |
| C1 — no green / red | **deviates — APPROVED by Ryan in chat 2026-10-31** | muted green #6f8f72 / red #a86464 / amber #b08a4a inside the box panel only, as status colours for agreement / contradiction / idea; everything else keeps C1. Change-log line added to `standing-rulings.md` |
| C2 — new display features off by default, existing views unchanged | **deviates — APPROVED by Ryan in chat 2026-10-31** | visible changes limited to the box panel (items 1–6). Change-log line added |

Nothing in "Revoked / superseded" (R1–R7) is used. Display aid, untested; no threshold changes.

## Step 0 (commit before implementation)

- Where the verdict inputs live (reuse; compute nothing twice).
- Read-history replay cost per box (boxes and absorption rebuilt as of each minute, no later bars); if heavy, cap to the
  current box + expanded frozen box, or cache per minute, and say which.
- Colour contrast in both themes.

## Commits

spec freeze → Step 0 → one commit per item group (1–3 table + wording; 4 history; 5 frozen rows; 6 colours) → smoke + wrap.

## Smoke (read-only backend)

- 2026-10-30 anchor 07:39, set end 08:02: pill "NO ACTION — short read, but price not at the top"; table shows share, c/pt
  and absorption green, location red; read history includes 07:49 (short).
- Same box with set end 07:49: SHORT READ, relevant rows green.
- A CONFLICT example (2026-10-30 08:30 from CR-DQ) shows green and red rows.
- 2026-10-28: NO TRADES, event row red.
- Frozen boxes collapse to one row and expand to the table.
- Existing specs green; only assertions on the old panel text updated (list them).

## Wrap

Session note, PR (don't merge), logs copied. Don't deploy. Only ONE DB-backed local server at a time; check first, never
stop another session's server without asking Ryan. Never print connection strings: catch DB errors and print the error type
only.

Open questions to record (not this CR): the GEX-line click stealing the anchor click (pre-existing); the live Ironbeam
undercount ratio since CR-BZ (unmeasured; the percentile baseline assumes the 2026 train ratio).

## Step 0 — findings (2026-10-31, on `358f27b` = origin/main `3ec3a42` + spec)

### S0.1 Where the verdict inputs live (reused, nothing computed twice)

| Input | Where (today) | CR-DS use |
|---|---|---|
| boxes (current + ≤ 4 frozen), edges, `high` / `low`, `lastClose`, edge counts | `App.flowV3Edges` = `edgeReadout(...)`; PriceChart `v3Edges` (`drawnBoxes` + readouts + ended / pending) | table rows b, c, g, h for the current box; frozen boxes need their last close → `boxStats` gains `lastClose` (= `bars.c[j1]`, additive, not visible elsewhere) |
| per-box readout (Δ, vol, dPrice, minutes) | `App.flowV3Segments.readouts` (`segmentReadouts` over the aligned v2 rows) | rows d, e, f; the replay needs the same rows and closes → `flowV3Segments` also carries `rows` and `closes` (already built there) |
| percentiles | PriceChart `v3BoxPanel` → `boxPercentiles` per drawn box | rows d, e (value + pNN) |
| absorption levels (top 8) | PriceChart `v3LevelsAll` = `absorptionLevels(...)` | row g via `boxLevels` (same function the verdict uses) |
| rules / thresholds | `boxRules(flowV3.boxRules)` (⚙) | unchanged; the table reads the same `r` |
| verdict | `boxVerdict(...)` | unchanged category and order. Refactor only: the checks it already makes are returned as `checks` (candidate side, location, lean, effort, progress, levels, forming, event), and the table is drawn from them, so the table can't disagree with the pill. The item-3 reason is a display string added to the existing NO ACTION outcomes: when `forming` / `none` would have been a SHORT / LONG READ with location (or forming / time) ignored, the reason names it ("short read, but price not at the top" / "short read, but box still forming"); "nothing notable" only when no flow check passes. On an FOMC day NO TRADES stays first (A22), with "short read, but event day" in the reason when the rest would read |

### S0.2 Read-history replay cost

Replay = for each minute *t* from the anchor to now / set end: `edgeReadout(…, end = t)` (boxes as of *t*, no later bars;
an unresolved pending break at *t* keeps the old box, as the live view did), the current box's readout over the v2 rows
up to *t*, its percentiles, the absorption levels as of *t* (`absorptionLevels` with `now = t`: marks confirmed ≤ *t*, broken
only if `break_time` ≤ *t*, top 8 by score at *t*), then `boxVerdict`. Read minutes (SHORT / LONG READ) are grouped into ranges
per box (box start = the replayed current box's `from`).

Measured in node on real payloads fetched from the read-only backend (2026-10-30, `scratchpad/bench.mjs`, full
`edgeReadout` incl. edge counts per minute — the naive version):

| Payload | minutes replayed | cells | boxes (max) | time |
|---|---|---|---|---|
| RTH, anchor 06:30 | 389 | 6,266 | 8 | 37–39 ms |
| FULL (ETH + RTH), anchor at the Globex open | 1,319 | 13,475 | 26 | 287 ms |

The cost is the per-box edge counts (cells × boxes × minutes), which the verdict never reads. **Choice:** replay with a
light `edgeReadout` option that skips edge counts and probe contracts (prices, edges, high / low, pending identical —
unit-tested against the full readout), over the whole anchor → now / set end span, memoised on (payload, anchor, end,
hold settings, rows, rules). No per-minute cache unless the light replay is still heavy in the smoke (time recorded
there). Frozen boxes' histories come from the same single replay (grouped by box start), so expanding a frozen box costs
nothing.

**Caveat (stated, not fixed):** a mark's `merged_vol` (packages/shared/flow_heat.py) adds the volume of later marks
merged into it, up to the payload's end. It only changes a level's contracts and therefore its rank for the top 8; which
marks exist, when they confirm and when they break are causal. So a replayed minute can pick a slightly different top 8
than the live chart did at that minute when more than 8 marks stand. The smoke compares the replay at a few minutes with
the payload fetched with `until = t`.

### S0.3 Colour contrast

The React price chart has one (dark) theme: no light theme or theme switch exists in `react_price_preview` (grep:
no `prefers-color-scheme` / theme state); chart background `#1f2937`, panel `TAG_BG(0.92)` = `rgba(30,41,59,0.92)` → ≈
`#1d283a`. WCAG ratios (computed):

| Colour | vs panel (non-text, needs ≥ 3:1) | light text `#e2e8f0` on it as a fill |
|---|---|---|
| green `#6f8f72` | 4.2 | 2.9 |
| red `#a86464` | 3.3 | 3.7 |
| amber `#b08a4a` | 4.7 | 2.6 |
| slate `#94a3b8` | 5.9 | 2.1 |
| text `#e2e8f0` | 12.2 | — |

So colour goes on non-text marks only (spec item 6): a 3 px left bar per table row and the pill's border + left bar;
all text stays light on the panel background (12:1). Filled pills with light text would fall below 4.5:1, so none are
used. On a white page behind the chart (not a current state) the bars would still be ≥ 2.5:1 against the panel.

## Implementation notes (no spec change)

- `boxVerdict` computes its existing checks first and returns them as `checks`, plus `pill` (the label's head) and
  `reason` (the line under it). `label` keeps CR-DQ's strings except for the item-3 renames. Category and order are unchanged; unit
  tests for CR-DQ's verdict pass as they were.
- Item 3's "would be a read" is the existing short / long flow checks (lean or absorption on that side, plus effort or no
  advance) with location / box formed / event ignored. It is not offered when the flow is opposed by absorption (that
  would be CONFLICT once formed).
- Evidence rows, no candidate: event and box-formed rows still show red when they fail (they stop any trade) and slate
  when met; every side row is slate. Two-sided absorption: one red row.
- Read history labels minutes by **candle time (bar start)**: "07:49" = the verdict once the 07:49 bar closed (the live
  view at 07:50). Stated on the line's hover.
- `boxStats` gains `lastClose` (frozen boxes' location row as of their end). A frozen box's table uses the absorption levels
  as of its break minute.
- Colours: 3 px left bar per row and the pill's border / left bar; neutral = existing slate `#64748b` (ReviewMeta); text
  stays light.

## Smoke (2026-10-31, read-only backend `scripts/cr_dp_smoke_backend.py` :8068; branch :5195; origin/main `3ec3a42` :5196)

`react_price_preview/tests/e2e/cr-ds.spec.js` **6 / 6 passed**; logs in `scripts/logs/cr-ds/smoke/`, screenshots
`specs/assets/cr-ds/`. Unit 106 / 106. Build clean.

| Check | Result |
|---|---|
| 1. 10-30 anchor 07:39, set end 08:02 | ✓ **NO ACTION**, reason "short read, but price not at the top". Rows: event ✓ green, formed (23 min · 7.00 pts) green, **location "1.25 below the bottom" red**, **Δ share +5.02 % p93 green**, **effort 83.4k/pt p93 green**, progress −1.25 green, **absorption buyers 7660.25 (07:05) at the top green**, edge counts slate. Bar colours = the specced hex. **Read history: "Read live: none"** — differs from the kickoff's expectation (07:49 short); see 2 |
| 2. same box, set end 07:49 | **Reported as is (differs from the kickoff):** anchored at 07:39 the verdict is **NO ACTION — nothing notable** (share +6.19 % p86 green, absorption green, location green, effort p60 slate, **progress +7.0 red**: the no-advance check fails, so no read). Anchored at 06:30 the box starts at 07:38 and reads **SHORT READ at top** ("buyers absorbed at the top, effort"; c/pt p82), history "Read live: 07:49 (short)"; event / formed / location / effort / absorption green, share slate (p<80), progress +3.5 red. Thresholds not changed |
| 3. CONFLICT 10-30 until 08:30, anchor 08:00 | ✓ CONFLICT (red pill border), reason "buyers leaning vs sellers absorbed at the bottom"; Δ share p84 green, absorption sellers 7640.00 at the bottom red, location red |
| 4. 10-28 (FOMC) | ✓ NO TRADES, reason "event day (FOMC)"; event row red "no trades" |
| 5. frozen boxes (10-30 anchor 06:30) | ✓ 4 frozen rows collapsed ("09:10–09:46 · broke ↓ 09:46 · share p67 · c/pt p88 · read 09:20 (short) · 09:23–09:26 (short)"); each expands to its pill + table as of its end + history + stats. Replay in the browser: **11 ms** (anchor 06:30 → 13:00) |
| 6. no later bars | Full-day history (box 07:38: 07:49; box 09:10: 09:20, 09:23–09:26) vs loads cut with `until` hh:mm:59 at 07:48 / 07:49 / 07:50 / 09:22 / 09:23 / 09:26 / 09:27: each live verdict equals the replay at that load's last bar, and each cut load's history is the full one truncated (e.g. until 09:26:59 → "09:20 · 09:23–09:25", SHORT READ). The merged_vol caveat did not show (5 marks on 10-30) |

### Existing specs (`scripts/logs/cr-ds/existing-*.log`)

A (cr-bv, cc, cd-a5, cd, ch, ck, cq): 23 passed, 8 skipped. B (cr-db, dc, de, df, dg, dh): 17 passed, 5 skipped, 14
failed — live-date tests (Saturday, no live session: cr-dc / df / dg live), the Dashboard app not running (cr-dh × 4), review
saves (cr-db:124, cr-de × 6 — **also fail on origin/main**, `existing-B-review-on-main.log`). C (cr-di, dk, dm): 14 passed, 2
skipped; cr-di:250 failed on the old frozen text (updated, now passes); cr-dm:280 needs CR-DM's `hand_check.json` log (not
in this worktree; listed failing on origin/main in CR-DQ). D (cr-dn, dp): 14 passed, 1 skipped; cr-dn:647 (57k minute),
cr-dp:158 / 307 **also fail on origin/main** (`existing-D-on-main.log`), cr-dp:491 needs a live bar. E (cr-dq): 8 passed
(items 1–3 after the stale-baseline fix below); the 72-replay verdict scan (report only) not re-run.

**Assertions updated (old panel text / stale baseline only):**
- `cr-dq.spec.js` panelText: stats lines read from the stats blocks (`[data-testid$=-stats] > div`; expanded frozen sections
  now show a table first); evidence from `data-evidence`; verdict hover from the pill.
- `cr-dq.spec.js` items 1–3: frozen summary "hh:mm–hh:mm · broke ↓ hh:mm · share …" (no "Box " prefix).
- `cr-dq.spec.js` item 7: pill "NO TRADES" (label unchanged, reason under it).
- `cr-dq.spec.js` items 1–3, **stale baseline (Ryan, chat 2026-10-31)**: line 226 asserted origin/main draws in-chart box
  labels, true only against pre-CR-DQ main; now it asserts no labels on both sides and that both show the box panel (popup).
  Same stale cause, same test: origin/main's popup head carries badges (stripped on both sides), its boxes have no `label`
  (numbers from `boxLabelText` on the box readout), and frozen boxes nested in the current box gain `lastClose` (stripped,
  as the top-level one already was).
- `cr-di.spec.js:356`: frozen row text without "Box ".

## Amendment (Ryan, chat 2026-10-31, before merging) — no-advance legs

No advance is "effort (c/pt ≥ notable) OR progress ≤ 2 pts toward the pushing side". When either leg qualifies, the
other row is slate "not needed", never red, both ways: effort met + progress > 2 → Progress slate "not needed (effort)";
progress ≤ 2 + effort below notable → Effort slate "not needed (no advance)". Progress is red only when neither leg
qualifies. Display only; verdict unchanged. Unit 107 / 107 (new test covers both legs, the long mirror and neither).
cr-ds smoke re-run **6 / 6** (`scripts/logs/cr-ds/smoke-run-2.log`); smoke 2 now asserts 10-30 06:30 → 07:49 (SHORT READ via
effort p82, +3.5 pts) shows Progress slate "not needed (effort)".
