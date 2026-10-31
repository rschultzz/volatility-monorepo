# CR-DT — Would a nearby GEX level have been a better entry than the box edge?

Read-only analysis on the CR-CZ learning set. No DB writes; no server. Branch `feat/CR-DT-gex-level-entry` off
`origin/main` `3ec3a42`, own worktree (`../volatility-monorepo-cr-dt`). Kickoff: Ryan, 2026-10-31 (chat). Frozen
2026-10-31 before any CR-DT outcome was computed.

---

## Kickoff (verbatim)

```
CR-DT — Would a nearby GEX level have been a better entry than the box edge? (read-only analysis, learning set)

Branch first: git fetch origin && git worktree add ../volatility-monorepo-cr-dt -b feat/CR-DT-gex-level-entry origin/main. Work only there; git branch -v before every commit.

Read first: CLAUDE.md; vault components/field-conventions.md; vault research/standing-rulings.md (incl. A3, A4, A12, B1, B2, B4, R2, R6); vault research/CR-CZ learning set — results 2026-10-31.md. Research CR: standing-rulings conformance table required; any "deviates" is a STOP.
- Read-only. No DB writes; no server needed.
- Never print connection strings: catch DB errors and print the error type only.

QUESTION (Ryan, chat 2026-10-31): on the 119 CR-CZ cards (set cr-cz:cp-ranges, card 31 dropped as a duplicate of 30), would a resting limit at a nearby GEX level (gross ≥ 40 $bn) have given better entries than the card's edge limit?

DATA AND DEFINITIONS (fixed before any outcome is read)
- Levels: corrected placement only (R6), via the app's existing function (_corrected_gex_tables in modules/Ironbeam/callbacks.py), per session; the same buckets the chart draws. Days with no corrected levels are reported and excluded.
- Bars: Databento front month by daily RTH volume (as the learning-set queries), races to 12:59 PT, same-minute ties reported.
- A level qualifies if gross |GEX| ≥ 40 $bn (Ryan's chart filter). Readout: A12 strong levels (net beyond the day's p95 |net|).
- Called direction, from the card. For a short at the top: the nearest qualifying level at or above the fade edge, within D pts; for a long at the bottom: the mirror.
- Variants (all reported, none chosen by result; B2):
  - D = 2, 4, 6 pts beyond the edge;
  - a level inside the box within 2 pts of the edge ("level just inside").
- Entry: a limit at the level, resting from the card's call time until 12:59 or until the box breaks the other way (close > 2 pts beyond the far edge). No fill = no trade.
- Outcome: ±10 race from the fill (same as the cards), plus net pts at a 10-pt stop.
- Baseline: the card's own edge limit, same race.

REPORT (by label: takes / skips / all)
- How many cards have a qualifying level per variant.
- Fill rate (level vs edge).
- Won / lost, win rate with 95% interval, net pts.
- The same on the A12-strong readout.
- Paired view: on cards where both filled, level entry vs edge entry; on cards where only one filled, which one.
- Verdicts per B1 (pass / inconclusive / refuted) vs the edge baseline. This is the learning set (69 of 119 revealed), so any pass is a hypothesis for blind set 2, not a finding.

STEP 0 (commit before computing outcomes): confirm the level function can be called per session date; list the sessions without corrected levels; freeze the variant table above. Then run once.

WRAP: findings appended to vault research/CR-CZ learning set — results 2026-10-31.md (new section) and the session note; PR with the script and spec (don't merge). Nothing to deploy.
```

## Frozen definitions (Claude, at spec freeze; read from code at `3ec3a42`, nothing measured yet)

| # | Item | Definition |
|---|---|---|
| T1 | Cards | `setup_candidates` where `set_name = 'cr-cz:cp-ranges'`, card 31 (`CR-CZ:cr-cz:cp-ranges:31`) dropped → 119. Direction, `session_date`, `entry_ts_pt` (close of decision bar t) and `meta.range.top / bottom` (the box by closes, CR-CP A1.1) from the card |
| T2 | Labels | the blind label = the first pre-reveal `setup_reviews` row per card (CR-DE semantics, as `service.py`). Groups: **takes** (label `take`), **skips** (label `skip`, called side, incl. the "take continuation (other edge)" / "take opposite" reason cards — they are skips of the called side), **all** (119, incl. `unsure`) |
| T3 | Bars | `cr_cp_lib.build` on the cached CR-CP train minutes (`es_flow_1m_databento`, the session's most-RTH-volume symbol, C18 flat fill): the bars the cards were built and raced on. RTH bar i = 06:30 + i PT, i = 0..389 |
| T4 | Level tables | `_corrected_gex_tables(session_date)` from `apps/web/modules/Ironbeam/callbacks.py`, called once per card session; a non-None reason (`pre_forward_era` / `no_es_snapshot` / `no_rows`) excludes the session's cards from every level variant (reported) |
| T5 | Drawn levels | per card, exactly as the React chart draws them in the card's review replay: `_select_levels(df_gex, band_min, band_max, 40e9)` with band = the chart's session-window bars (`_session_window_pt`, `_fetch_bars_pt`, 1m) cut at the card's `entry_ts_pt` (CR-CC L5 `until_pt`) ± `GEX_LEVEL_PADDING`; gross = \|call\| + \|put\| ≥ 40 $bn, 1-pt buckets, greedy 5-pt spacing, ≤ 80 levels. These are the chart's frame (ES, CR-BT P_target) |
| T6 | A12 strong | a drawn level with \|net\| ≥ the chart colour span = p95 of \|net\| over that card's drawn levels (`_compute_color_span`) |
| T7 | Fade edge | short: E = box top; long: E = box bottom (the card's `meta.range`) |
| T8 | Level choice | short, variant D ∈ {2, 4, 6}: qualifying levels with E ≤ L ≤ E + D, the one nearest E (lowest). "Just inside": E − 2 ≤ L < E, nearest E (highest). Long: mirror (E − D ≤ L ≤ E, highest; E < L ≤ E + 2, lowest). No such level = the card has no entry in that variant |
| T9 | Level entry | a limit at L itself, live from bar t + 1 to bar 389 (12:59 PT); fills on the first bar with high ≥ L (short) / low ≤ L (long); cancelled at the first bar whose close is > 2 pts beyond the far edge (short: close < bottom − 2; long: close > top + 2); a same-bar touch and cancel is a cancel (as CR-CP C9 `fill`). Same loop as `cr_cp_lib.fill` with px = L |
| T10 | Baseline | the card's own D4 edge limit (short top − 1, long bottom + 1) through the same `cr_cp_lib.fill`; asserted equal to the card's stored `outcome.called` fill status, fill price and ±10 result |
| T11 | Race | `cr_cp_lib.race_from` (C10): the fill bar's adverse extreme ≥ 10 = loss, then the CR-CL race from bar j + 1 to 12:59 PT. `win` / `loss` / `tie` (same minute, reported) / `unresolved` |
| T12 | Net pts, 10-pt stop, +10 target | win +10, loss −10, tie −10 (conservative; ties also counted), unresolved 0, no fill / no level 0. Readout: unresolved marked to the 12:59 close |
| T13 | Win rate interval | Wilson 95 % on won / (won + lost), ties counted as lost |
| T14 | B1 verdict | per variant × label × level set (gross 40, A12 strong): over cards with a qualifying level, paired difference = net(level) − net(edge) per card; 95 % bootstrap CI of the mean (10,000 resamples, seed 20261031). **pass** = lower bound > 0; **refuted** = upper bound < 0; else **inconclusive**. Learning set: a pass is a hypothesis for blind set 2, never a finding |
| T15 | Paired view | on cards with a level: both filled (level net vs edge net: better / same / worse), level only, edge only, neither |

Variant table (frozen; all reported, none chosen by result — B2): {D2, D4, D6, just inside} × {gross ≥ 40 (primary), A12 strong (readout)} × {takes, skips, all} = 24 cells. Disclosed: 24 cells looked at on one learning set.

## Standing rulings conformance (`Dash/research/standing-rulings.md`, status current, read 2026-10-31)

| # | Mark | How |
|---|---|---|
| A1 | honoured | levels are where price may slow; the test asks whether a level near the range edge is a better entry, not whether a level is a trade |
| A2 | honoured | the box is CR-CP's frozen range (≥ 15-bar hold); unchanged |
| A3 | honoured | entry stays after the range has formed and the side is called (from the card's call time); the level is one entry variant, not automatic |
| A4 | honoured | stop is the cards' fixed 10-pt race, unchanged; no stop placed at the level |
| A5 | honoured | side = the card's flow call (D2), unchanged |
| A6, A14, A16 | n/a | no skew |
| A7 | honoured | CR-CP's vol-scaled swing legs as built; unchanged |
| A8 | n/a | put / call level split not requested; level kind not tested (could be a readout later) |
| A9, A10 | n/a | not tested |
| A11 | honoured | ±10 race primary, as the cards |
| A12 | **deviates — pre-approved in the kickoff** | A12 makes the yellow / darkest (p95 net) population primary for level tests; the kickoff makes gross ≥ 40 $bn the qualifying set ("Ryan's chart filter") and A12 strong the readout. Recorded, not stopped for (CR-CZ precedent); logged in the rulings change log. Corrected placement honoured. A12 itself unchanged |
| A13 | n/a | no sizing |
| A15 | honoured | fixed ±10 bracket, no tuning |
| A17, A19, A20 | honoured | the in-range flow call decides the side (unchanged); the edge entry is the baseline |
| A18 | n/a | range position vs level not split |
| A21 | n/a | no second-touch setup |
| A22 | n/a | the cards are the frozen CR-CZ set; FOMC days not re-filtered (none removed or added) |
| B1 | honoured | pass / inconclusive / refuted on a CI (T14) |
| B2 | honoured | 24 cells, all reported, disclosed; none chosen by result |
| B3 | honoured | one setup (the CR-CZ cards), one change (entry price) |
| B4 | honoured | train only: every card session ≤ 2026-06-03 (asserted ≤ 2026-06-05); holdout, R-a, R-b, forward log not read. Learning set, labelled as such |
| B5 | honoured | Databento bars (T3) |
| B6 | honoured | the ±10 race is the cards' direction race, labelled; net at a 10-pt stop is a readout, not Ryan's exit |
| B7 | n/a | no placebo |
| B8 | honoured | labels are Ryan's blind labels from `setup_reviews`; Code writes none |
| C1, C2 | n/a | no UI |

Revoked / superseded: R1 not used. **R2**: entry at the level price is used as one labelled entry variant against the
card's edge entry, after the range and the call (not the automatic entry at the touch) — as R2 allows. **R6**: not used;
corrected placement only. R3–R5, R7 not used.

---
