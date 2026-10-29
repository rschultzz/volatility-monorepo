# CR-DM — Flow v3: volume per minute replaces rolling net delta; live effort readout for the leg in progress

Spec copied from the vault session note `sessions/2026-10-29 - CR-DM — Volume Strip and Live Effort.md` (Ryan's
kickoff, chat 2026-10-29).

Branch `feat/CR-DM-volume-strip-live-effort` off `origin/main` `78a50fb`, worktree `../volatility-monorepo-cr-dm`.

## Spec (Ryan, chat 2026-10-29)

1. **Volume per minute.** In the Flow v3 pane, the middle series (rolling net delta, bars) is replaced by volume
   per 1-minute bar: plain grey bars (existing palette, C1), same position and height, so the pane's total height
   is unchanged.
   - Hover: time, volume, delta, delta share for that minute.
   - Values come from the same flow data as the delta bars (state which volume field and whether unknown-side
     volume is included).
2. **Rolling net delta moves into ⚙** as "Rolling net delta (N min)", OFF by default, same behaviour as today when
   on.
   - When on, it draws in its old place and the volume bars move to the slot below it; the pane grows by one slot
     height rather than squeezing the others.
3. **Live effort readout for the leg in progress** (only while Effort is on):
   - a. The current, unconfirmed leg (from the last confirmed leg's end to the running extreme since) gets a
     provisional label at its running extreme, dimmed, e.g. "↑ +6.5 pts so far · 1.2× vs last ↑". Same
     contracts-per-point and ratio as confirmed legs: cpp vs the previous leg in the same direction
     (`packages/shared/flow_features.leg_effort`).
   - b. Shown only once the leg has moved at least M pts (⚙ "Live effort min move", default 4, range 2–12) and a
     previous same-direction leg exists; otherwise nothing.
   - c. Updates on each refresh; the label moves with the running extreme. When the swing rule confirms the leg,
     the provisional label is replaced by the normal confirmed label. In replay / review it respects the until
     time and never uses bars after it.
   - d. Hover on the provisional label: pts, volume, delta, cpp, dpp, ratio, minutes so far, and the words
     "provisional — leg not confirmed".
4. **Nothing else changes:** CVD, delta bars, Edges, Edge effort strip, absorption, confirmed effort labels.

## Standing rulings conformance (UI CR, no research claims)

Read `research/standing-rulings.md` (2026-10-29) before the freeze.

| Ruling | Status | Note |
|---|---|---|
| A1–A6, A8–A22 | n/a | no setup, level, range, entry, exit or outcome logic |
| A7 — approach = vol-scaled ≈ 12-pt swing | n/a | a test ruling; the live label uses the same swing rule and size as the existing confirmed effort labels (Flow v2 ⚙ swing, default 8), unchanged. No test is run |
| B1–B7 | n/a | no hypothesis tested, no data split read beyond what the chart already displays |
| B8 — Code smoke labels are never Ryan's | honoured | no review labels written; smoke runs against a read-only backend |
| C1 — no green / red; Dash palette | honoured | volume bars in the existing light grey (`SELL_RGBA` / slate family); provisional label in the existing blue / grey at reduced opacity |
| C2 — new display features off by default, existing views unchanged | **deviates — approved by Ryan in chat 2026-10-29** | rolling net delta, visible today, becomes off by default (moved to ⚙); volume bars take its place. The live readout appears only with Effort on (an existing switch) — it changes what Effort-on shows, covered by the same approval. Change-log line added to `standing-rulings.md` |

Nothing in "Revoked / superseded" (R1–R7) is used. The live readout and the min-move setting are display aids, not
tested thresholds; the legend says so.

## Step 0 (commit before implementation)

- Which volume field the flow payload carries per minute, and whether it includes unknown-side volume.
- How the in-progress leg can be derived from `swing_legs`' state without a second swing algorithm.
- Confirm that confirmed effort labels are unchanged by this work on 2026-10-27 and 2026-10-29.

## Commits

spec freeze → Step 0 → items 1+2 together → item 3 → smoke + wrap.

## Smoke (read-only backend)

- Default view shows volume bars where rolling delta was; pane height unchanged.
- ⚙ rolling delta on → old series back, pane grows.
- 2026-10-29 replay at several times shows a provisional label on the current leg that matches a hand computation
  from the minute data, and becomes the confirmed label once the retrace arrives.
- Review mode never uses bars after `until`.
- Existing specs green.

## Constraints

- Work only in `../volatility-monorepo-cr-dm`; CR-DL (data recompute) may be in Step 0 in its own worktree — not
  touched.
- One DB-backed local server at a time; never stop another session's server without asking Ryan.
- Never print connection strings (catch DB errors, print the error type only).
- Rebase onto origin/main before the PR if anything merged. PR not merged; not deployed.
