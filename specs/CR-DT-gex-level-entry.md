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

## Step 0 findings (2026-10-31; code at `3ec3a42`; read-only; no outcome read — `outcome` not selected)

Script: `scripts/cr_dt_level_entry.py step0` (output `scripts/logs/cr_dt/step0.json`, `levels.json`, git-ignored).

1. **Cards and labels.** 120 rows in `cr-cz:cp-ranges`, card 31 dropped → **119** on **66 sessions**, 2026-01-06 →
   2026-06-03 (all ≤ 2026-06-05, B4). Short 60, long 59. Blind labels (first pre-reveal row): **take 44, skip 75**, no
   `unsure`, none missing. So "all" = takes + skips.
2. **Level function per session.** `_corrected_gex_tables(d)` was called for each of the 66 sessions, imported from
   `apps/web/modules/Ironbeam/callbacks.py` unchanged. 65 return tables; **one session has no corrected levels:
   2026-02-19 (`no_es_snapshot`**: no Ironbeam ES bar starting 15:45 ET on 2026-02-18, the CR-BT P_target). Its one card
   (**#37**) is excluded from every level variant and kept in the edge baseline totals (reported both ways at the run).
   No `pre_forward_era` / `no_rows` / errors.
3. **Drawn levels per card** (T5, band cut at `entry_ts_pt`): 17–52, median 34; adjacent drawn levels median 8 pts
   apart (min 5, the chart's spacing rule). A12 strong (T6, p95 of the card's drawn \|net\|): 1–3 per card, median 2.
   So gross ≥ 40 is a dense grid near any edge; A12 strong is sparse.
4. **Coverage** — cards with a qualifying level (takes / skips / all of 44 / 75 / 119):

   | Variant | gross ≥ 40 | A12 strong |
   |---|---|---|
   | D2 (≤ 2 pts beyond) | 11 / 24 / 35 | 2 / 4 / 6 |
   | D4 | 23 / 46 / 69 | 4 / 6 / 10 |
   | D6 | 33 / 58 / 91 | 5 / 6 / 11 |
   | just inside (≤ 2 pts) | 14 / 15 / 29 | 0 / 2 / 2 |

   The A12 cells are too small for a verdict (≤ 11 cards); they will be reported, and will be inconclusive by
   construction unless every card goes one way.
5. **Variant table frozen** as in the definitions above (24 cells, T14 verdict rule, seed 20261031). No change from
   the kickoff. Nesting disclosed: D2 ⊂ D4 ⊂ D6 cards do not all pick the same level (nearest to the edge in each), and
   the variants are not independent tests.
6. **Baseline check planned at the run** (T10): the edge limit recomputed through the same loop must reproduce every
   card's stored fill status, fill time, fill price and ±10 result; the run asserts it before any variant is summarised.

## Run (once, 2026-10-31) and wrap

`scripts/cr_dt_level_entry.py run` → `scripts/logs/cr_dt/{trades,results}.json`, `run_stdout.log` (git-ignored).
**Baseline check passed:** the edge limit through the T9 loop reproduces all 119 cards' stored fill status, fill time,
fill price and ±10 result. Edge baseline: takes 17 / 12 (59 %, 95 % 41–75), net +50; skips 19 / 26, −70; all 36 / 38,
−20 (= the learning-set note). Without card 37 (no levels; a skip, edge fill, loss): skips 19 / 25, all 36 / 37. No
same-minute ties anywhere.

### Primary — gross ≥ 40 $bn levels (cards with a level; level entry vs the edge entry on the same cards)

| Variant | Group | Cards | Level fill / won / lost (win %, 95 %) / net | Edge fill / won / lost / net | Both filled: better / same / worse | Edge only (edge net) | Mean Δ net / card (95 %) | B1 |
|---|---|---|---|---|---|---|---|---|
| D2 | takes | 11 | 9 / 4 / 3 (57 %, 25–84) / +10 | 9 / 3 / 4 / −10 | 1 / 8 / 0 | 0 | +1.8 (0.0, 5.5) | inconclusive |
| D2 | skips | 24 | 17 / 7 / 6 (54 %, 29–77) / +10 | 19 / 6 / 10 / −40 | 5 / 12 / 0 | 2 (+20) | +2.1 (−0.4, 5.0) | inconclusive |
| D2 | all | 35 | 26 / 11 / 9 (55 %, 34–74) / +20 | 28 / 9 / 14 / −50 | 6 / 20 / 0 | 2 (+20) | +2.0 (0.0, 4.3) | inconclusive |
| D4 | takes | 23 | 14 / 7 / 5 (58 %, 32–81) / +20 | 17 / 8 / 7 / +10 | 2 / 12 / 0 | 3 (+30) | +0.4 (−2.2, 3.5) | inconclusive |
| D4 | skips | 46 | 28 / 10 / 10 (50 %, 30–70) / 0 | 35 / 10 / 17 / −70 | 7 / 21 / 0 | 7 (+40) | +1.5 (−0.4, 3.7) | inconclusive |
| D4 | all | 69 | 42 / 17 / 15 (53 %, 36–69) / +20 | 52 / 18 / 24 / −60 | 9 / 33 / 0 | 10 (+70) | +1.2 (−0.4, 2.9) | inconclusive |
| D6 | takes | 33 | 20 / 9 / 9 (50 %, 29–71) / 0 | 25 / 12 / 11 / +10 | 2 / 18 / 0 | 5 (+50) | −0.3 (−2.4, 2.1) | inconclusive |
| D6 | skips | 58 | 32 / 12 / 12 (50 %, 31–69) / 0 | 45 / 15 / 22 / −70 | 9 / 23 / 0 | 13 (+80) | +1.2 (−0.9, 3.3) | inconclusive |
| D6 | all | 91 | 52 / 21 / 21 (50 %, 36–65) / 0 | 70 / 27 / 33 / −60 | 11 / 41 / 0 | 18 (+130) | +0.7 (−0.9, 2.2) | inconclusive |
| inside 2 | takes | 14 | 11 / 5 / 5 / 0 | 11 / 5 / 5 / 0 | 0 / 11 / 0 | 0 | 0.0 | inconclusive |
| inside 2 | skips | 15 | 13 / 2 / 7 / −50 | 13 / 2 / 7 / −50 | 0 / 13 / 0 | 0 | 0.0 | inconclusive |
| inside 2 | all | 29 | 24 / 7 / 12 / −50 | 24 / 7 / 12 / −50 | 0 / 24 / 0 | 0 | 0.0 | inconclusive |

"Level only" is 0 in every cell by construction (a level at or beyond the edge can only be touched on or after the
bar that touches the edge limit). Unresolved fills count 0 in net; marking them to the 12:59 close moves no cell by
more than 13 pts (results.json `net_mtc`).

**Where the difference comes from (both-filled vs edge-only, all cards):** at D4 the level entry is a median 3.0 pts
better and fills a median 3 min later. On the 42 cards where both filled it never did worse: 6 edge losses became
wins, 2 became unresolved, 1 unresolved became a win (+150 pts). But on 10 cards price turned before reaching the
level and the edge trade it missed went 8 won / 1 lost / 1 unresolved (+70). Net +80 over 69 cards. The same
shape at D2 (+90 / −20) and D6 (+190 / −130): the deeper the level, the more of the edge's winners it misses. On
takes the missed winners nearly cancel the gain (D4 +40 / −30; D6 +40 / −50). "Just inside" levels sit a median
0.5 pt from the edge limit and change nothing.

### Readout — A12 strong levels

6 / 10 / 11 / 2 cards (D2 / D4 / D6 / inside), 0–5 on takes. Every cell inconclusive (all-card means 0.0, +2.0,
+1.8, 0.0 pts / card; CIs straddle 0). On these cards both entries mostly lose (edge all-card D6: 1 / 7). Too few to
read.

### Verdicts (B1)

**24 of 24 cells inconclusive; none pass, none refuted.** The level entry is ahead on net in most "all" and "skips"
cells (better price, never worse when both fill) but gives back most of it on the edge trades it misses; on takes
it is flat. Learning set (69 of 119 revealed), 24 cells disclosed (B2): nothing here is a finding. **Hypothesis for
blind set 2 (H-DT1):** a resting limit at the nearest gross ≥ 40 level 0–4 pts beyond the fade edge improves the
edge entry's net per card — test as "D4, all cards" with the same definitions; note its cost is the edge's quick
winners.
