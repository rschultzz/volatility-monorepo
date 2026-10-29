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

## Step 0 findings (2026-10-29, read-only; `scripts/cr_dm_step0.py`, output in `scripts/logs/cr-dm/step0/`)

### Q1 — the volume field

- The v2 payload's bars carry `vol` per 1-minute bar (`flow_v2._bars`), alongside `buy`, `sell`, `delta`, `share`.
  It is the same row the delta bars are drawn from: `delta = buy − sell` of the same minute.
- `vol` is the flow table's `volume` column summed over the minute: `es_flow_1m_databento.volume` /
  `ironbeam_es_flow_1m.volume` (read by `flow_1m.v2_select`), or `sum(volume)` of the 1-s table for minutes after
  the 1-minute table's cutoff (`flow_v2._FLOW_SQL_1S`). Quiet minutes are 0 (`complete_minutes`).
- **Unknown-side volume is included by construction**: the 1-s aggregator sets `volume = Σ size` of all prints and
  `unknown_vol` = size of prints with neither BUY nor SELL (`apps/cron/ironbeam_flow_1s_aggregate.py`); Databento
  side `N` likewise (CR-CA). So `vol = buy + sell + unknown`.
- **Measured on RTH 06:30–13:00 PT** (1-minute tables): volume = buy + sell + unknown on every minute, and
  unknown = 0 on every minute of all five sessions checked:

  | table | session | minutes | volume | unknown |
  |---|---|---|---|---|
  | `es_flow_1m_databento` (ESM6) | 2026-05-01 | 390 | 1,008,323 | 0 |
  | `es_flow_1m_databento` (ESH6) | 2026-01-06 | 390 | 941,656 | 0 |
  | `ironbeam_es_flow_1m` | 2026-10-27 | 390 | 1,069,166 | 0 |
  | `ironbeam_es_flow_1m` | 2026-10-28 | 380 | 1,041,671 | 0 |
  | `ironbeam_es_flow_1m` | 2026-10-29 (to 07:32 PT) | 62 | 381,851 | 0 |

  So in practice volume = buy + sell in RTH; the hover says "volume (all prints, incl. unknown side)". Delta share
  for the minute = (buy − sell) ÷ volume, i.e. the raw per-minute share (the payload's `share` is its 3-bar EMA, so
  the hover computes the raw value from the same row).
- Sources as today: Databento for sessions ≤ 2026-06-05, Ironbeam after (Ironbeam undercounts ≈ 6 %, B5; the guide
  already says so).

### Q2 — the in-progress leg from `swing_legs`' own state

- `swing_legs` already carries everything at the end of its loop: `dirn` (+1 = current leg is up, −1 = down,
  0 = no extreme confirmed yet), `start_i` / `start_px` (the last confirmed leg's end = the current leg's start) and
  the running extreme of the current direction (`hi` / `hi_i` while up, `lo` / `lo_i` while down). It just doesn't
  return them ("The last, unconfirmed leg is not returned").
- Plan: move the loop into `_swing_scan(H, L, swing)` returning `(legs, tail)`, where `tail` = `{dir, start_i,
  start_px, end_i = running-extreme bar, end_px = running extreme}` or `None` while `dirn == 0`. `swing_legs` returns
  `_swing_scan(...)[0]` — **the same loop, the same legs** (no second swing algorithm).
- The live leg's effort = `leg_effort(legs + [tail])[-1]`: the same bars `(start_i, end_i]`, the same cpp / dpp and
  the ratio to the previous leg in the same direction, exactly as a confirmed leg. Once the swing rule confirms it,
  that leg *is* the next confirmed leg (same start, same extreme, same bars), so the provisional label is replaced
  by the confirmed one with the same numbers, and a new tail (the opposite direction) starts.
- `pane_live_leg(m, start_epoch, swing)` returns it on the same RTH bars as `pane_effort_legs` (bars ≥ the RTH open,
  bars < `end`); the payload gets `live_leg` (only with `effort=1`) next to `legs`. In review / replay `end = until`,
  so bars after `until` are never read — the same cut the confirmed legs already use.
- Payload fields: times as confirmed legs (`start_time`, `end_time` = running-extreme bar), `pts`, `vol`, `delta`,
  `cpp`, `dpp`, `ratio`, plus `prev_cpp` (the previous same-direction leg's cpp, null if none), `last_time` (the last
  bar used) and `minutes` (= bars from the start to the running extreme, the window pts / volume cover).
- The frontend applies (b): drawn only when |pts| ≥ M (⚙, default 4, 2–12) and a previous same-direction leg
  exists (`prev_cpp` not null). Nothing is filtered server-side, so M changes don't refetch.
- The live label is drawn in **Flow v3 only** (the CR's pane; ⚙ M lives in Flow v3 settings); Flow v2 with Effort on
  is unchanged.

### Q3 — confirmed effort labels are the baseline

- Saved the v2 payload's confirmed legs (effort=1, swing 8 = the App default) from origin/main code:
  2026-10-27 full session (13 legs), until 08:00 (6) and 10:15 (10); 2026-10-29 until 07:00 (8) and 07:30 (12)
  (`legs_*.json`). After implementation the same five calls must give byte-identical `legs` (checked in the item-3
  commit). The frontend label code (`effortChartMarkers`, `effortLabel`) is not edited; the live marker is appended
  to the list separately.
