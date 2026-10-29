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
