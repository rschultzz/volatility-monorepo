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
