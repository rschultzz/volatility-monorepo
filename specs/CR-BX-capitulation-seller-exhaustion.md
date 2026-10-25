# CR-BX — Capitulation and Seller Exhaustion

Spec freeze. Branch `feat/CR-BX-capitulation-seller-exhaustion` off `origin/main` `0e943d8`. Read-only against the database; no table writes, no cron, no frontend.

Source: vault `Dash/CRs/CR-BX — Capitulation and Seller Exhaustion.md` (last_reviewed 2026-10-25). The body below is transcribed verbatim from that note, frontmatter omitted. The kickoff prompt (Ryan, 2026-10-25) confirmed:
- locked decisions L1–L16;
- tests T1–T4 and their pass criteria;
- secondary readouts S1–S14;
- Step 0 gates 1–7.

Kickoff additions:
- k under L5 is fixed from family A counts only;
- a family under 150 per tercile is dropped, not loosened;
- Step 0 also reports box counts for the S11(b) 10- and 30-minute windows;
- no race outcome is computed outside the A/A splits;
- train only, 2026-01-06 .. 2026-06-05; holdout not read.

CR-BW is unmerged (branch `feat/CR-BW-aggressor-absorption-at-levels`), so its code is copied rather than imported.

---

## Why

**Ryan (2026-10-25):** beyond absorption, test buying after a move down (and selling after a move up).

**What is already known:**
- CR-BQ: 5-minute ES returns mean-revert slightly at every gamma level (ρ5 −0.004 to −0.064). That is about a 0.3-pt average bounce after a 10-pt 5-minute drop, roughly one round-trip cost.
- CR-BW: an 8-tick fade at levels wins about 50 % before costs and loses after costs.

**What is untested:**
1. Reversal conditional on **unusually large, fast moves** (capitulation), where the average says little.
2. **Seller exhaustion:** the move makes a new extreme while aggressive flow in its direction is *fading*.
3. **Ryan (2026-10-25): move → consolidation → break.** Price often moves, consolidates at a level, then reverses or continues. Does aggressor flow inside the consolidation predict which?

Two event families share the impulse definition: **A** fades at the extreme, **B** trades the break of the consolidation box.

**Multiple-testing note:** this is the third order-flow / level test on the same 88 train sessions (CR-BU, CR-BW). A pass here earns a single pre-registered holdout read, nothing more.

## Locked decisions (Ryan to confirm before spec freeze)

| # | Decision | Value |
|---|---|---|
| L1 | Scope | Test only. Read-only, no table writes, no cron, no frontend. |
| L2 | Data | ES only. `ironbeam_es_flow_1s` 1-second OHLC and buy/sell volume, front contract (CR-BW Q1/Q2: the flow is the correct series; the 1-minute bars are not used). |
| L3 | Sessions | CR-BW A2 universe: train 2026-01-06 → 2026-06-05, CR-BT exclusions (roll weeks, no ES snapshot, roll mismatch), RTH 06:30–13:00 PT. Event-level flow-gap rule: drop an event if its 300 s pre-window or race window has an RTH gap > 15 s. Holdout (after 2026-06-05) not read. |
| L4 | Move scale | σ5_d = median \|5-minute price change\| over all non-overlapping RTH 300 s windows of the prior ≥ 5 (up to 20) usable sessions. No lookahead. |
| L5 | Capitulation event | **Down:** at second t, P(t) = the minimum over [t − 300 s, t], and P(t) ≤ max over [t − 300 s, t] − k·σ5_d. **Up:** the mirror. After an event, no further event in the same direction for 15 minutes. No events before 06:35 or after 12:45 PT. **k** = the largest of {3.0, 2.5, 2.0} giving ≥ 150 eligible events in every exhaustion tercile (L8), fixed in Step 0 from counts only, before any outcome. |
| L6 | Trade | Fade the move. **Entry:** market at the next second's open + 1 tick slippage. **Exit:** target R ticks in the fade direction, filled only on a trade 1 tick through it; stop R ticks against, filled at stop + 1 tick slippage. Both in the same second → loss. Unresolved by 13:00 PT → excluded and counted. **Primary R = 8 ticks.** Cost $4.50 round trip (0.36 tick). |
| L7 | Exhaustion predictor | D(a, b) = net aggressor volume over seconds [t−a, t−b], signed in the move direction (selling into a down move is positive). **X = D(60, 0) / (D(300, 0) / 5)**: last-minute flow intensity relative to the 5-minute average. Defined only when D(300, 0) > 0. X < 1 means flow is fading at the extreme; X ≤ 0 means flow has flipped. |
| L8 | Terciles | X terciles over all eligible train events, both directions pooled. **Low tercile = exhausted.** |
| L9 | Placebo (random entries) | For each event: 5 entries in the same session at random seconds within ±30 min of the event time, with no capitulation event in the prior 300 s, in the same trade direction (seed 20261025). Same L6 trade. |
| L10 | Bootstrap | Session-cluster, 10 000 resamples, seed 20260920, 95 % percentile. |
| L11 | Run discipline | Committed before any outcome. One run. |
| L12 | Family B: box | After a family-A-qualifying impulse (L5, same k, both directions; the 15-minute cooldown applies to impulses), search the next 15 minutes for the **first** 180 s window with high − low ≤ 0.5·σ5_d that starts after the impulse extreme. Event time t_B = the end of that window. Box = [low, high] of that window. One box per impulse. No box ending after 12:45 PT. |
| L13 | Family B: outcome | From t_B, race: **reversal** = a trade 8 ticks beyond the box boundary **against** the impulse direction first; **continuation** = 8 ticks beyond the boundary **with** the impulse first. Same second → counted as neither, excluded and counted. Unresolved by 13:00 PT → excluded and counted. |
| L14 | Family B: predictor | P = D_box / V_box: net aggressor volume inside the box window, signed in the impulse direction (selling inside a box after a down move is positive), over total box volume. Terciles of P over all eligible train boxes, directions pooled. **High tercile = pressure into the box without progress** (absorption inside the base). |
| L15 | Family B: placebo | For each box: up to 5 boxes in the same session meeting the same range rule (180 s, ≤ 0.5·σ5_d) whose window ends within ±30 min of t_B and which had **no** qualifying impulse in the prior 15 minutes. The "impulse direction" is taken from the sign of the 300 s price change before the placebo box. Same race. |
| L16 | Family B: trade readout | Stop entry 1 tick beyond the box boundary in the T3-predicted direction (high P → reversal side, low P → continuation side), + 1 tick slippage; target 8 ticks, stop 8 ticks, same fill rules and cost as L6. |

## Tests and pass criteria (locked)

| Test | Statistic | Pass |
|---|---|---|
| **T1 exhaustion → reversal** | win rate, low X tercile − high X tercile | ≥ +8 pts, CI lower > 0, ≥ 150 resolved per tercile |
| **T2 exhausted capitulation vs random entry** | low-tercile win rate − matched placebo win rate | ≥ +8 pts, CI lower > 0 |
| **T3 box pressure → reversal** (family B) | reversal rate, high P tercile − low P tercile | ≥ +8 pts, CI lower > 0, ≥ 150 resolved per tercile |
| **T4 impulse boxes vs placebo boxes** (family B) | reversal rate of high-P impulse boxes − reversal rate of matched placebo boxes in the same P tercile | ≥ +8 pts, CI lower > 0 |

**Verdicts (per family):**
- **Family A:** T1 and T2 both pass → candidate edge. T1 only → exhaustion sorts events but does not beat random entries. Neither → recorded.
- **Family B:** T3 and T4 both pass → candidate edge. T3 only → flow in the box predicts the break, but not specifically after impulses. Neither → recorded.
- **Multiple testing:** four gated tests. Only a family passing both of its gates earns the single pre-registered holdout read. If both families pass, the holdout read covers both, pre-registered together.
- **Neither family passes** → the order-flow scalping line stops here on train.

## Secondary readouts (not gated)

- **S1 costs:** net expectancy per trade (ticks) at R = 4, 6, 8 and break-even win rates, by tercile and for the placebo.
- **S2 move size:** events at 2–3σ, 3–4σ and ≥ 4σ (whatever k is), low tercile vs placebo.
- **S3 direction:** buying after down moves vs selling after up moves.
- **S4 at levels:** events within 2 pts of a CR-BW A1.1 level vs not.
- **S5 time of day:** before vs after 10:00 PT.
- **S6 flow flip:** X ≤ 0 events vs placebo.
- **S7 window:** X from a 30 s last window (D(30, 0) / (D(300, 0) / 10)).
- **S8 family B costs:** L16 net expectancy per trade in ticks, by P tercile and for placebo boxes; break-even win rate.
- **S9 family B at levels:** boxes whose range lies within 2 pts of a CR-BW A1.1 level (≥ 40 $bn magnitude) vs not; and ≥ 100 $bn.
- **S10 family B continuation:** continuation rate by P tercile (the complement, reported for readability).
- **S11 family B longer boxes** (Ryan, 2026-10-25: consolidations often last much longer than 3 minutes):
  - **(a) holding time:** the same 180 s boxes, split by how long they held before the 8-tick break (< 10 min, 10–30 min, ≥ 30 min). P is recomputed over the whole box up to the break, which is known at break time, so there is no lookahead for a break-entry trade. T3-style difference and L16 net expectancy per bucket.
  - **(b) longer qualifying windows:** boxes defined by a 600 s window with range ≤ 0.5·σ10_d and by an 1800 s window with range ≤ 0.5·σ30_d, where σw_d is the median \|w-minute change\| over the prior sessions (as L4). Search window 60 minutes after the impulse. Same race, predictor, placebo and trade rules.
- **S12 eventual outcome that day:** from t_B to the 13:00 PT close, the signed move in the reversal direction, in σ5_d units. Report the share closing on the reversal side, by P tercile, vs placebo boxes.
- **S13 did a move happen that lines up with the flow:** the signed move in the reversal direction over the next 30 and 60 minutes and to the close, against P: Spearman correlation with CI, and top − bottom tercile mean, vs placebo boxes. Also the maximum favorable and adverse excursion (MFE / MAE, in ticks) over 60 minutes after the break, by tercile.
- **S14 family A eventual outcome:** for family A events, the same signed move to +30 min, +60 min and the close, by X tercile vs placebo entries.

## Step 0 — diagnosis gate (answer all before the run; contradiction-stop)

1. **Universe.** Sessions after each rule; σ5_d distribution across sessions.
2. **Counts at k = 3.0 / 2.5 / 2.0** (no outcomes), for family A and, separately, family B:
   - events per session, both directions;
   - eligible (D > 0) per tercile;
   - flow-gap drops;
   - family B: impulses with a qualifying box, boxes per tercile of P, placebo boxes per box;
   - the k chosen under L5 (from family A counts). Family B uses the same k. **A family whose counts at that k are below 150 per tercile is dropped from the run, not loosened.** Stop if both are dropped.
3. **X distribution** and the share with X ≤ 0.
4. **Placebo feasibility:** median placebo entries per event, and events with fewer than 5 eligible placebo seconds.
5. **Worked event** (seed 20261025): path, σ5_d, D windows, X, entry second. The outcome stays hidden.
6. **A/A calibration** (CR-BW A1.3 design), for each family kept: 200 random splits of the placebo entries (A) or placebo boxes (B) into two halves by event; the T1-style (A) or T3-style (B) difference using each event's tercile. Pass if the CI excludes 0 in ≤ 10 % of splits and the median is within ±3 pts.
7. **Box geometry:** median box range in points and ticks, and the median box duration found. Report boxes narrower than 4 ticks (8-tick breaks then sit close to noise).

## Outputs

- Spec `specs/CR-BX-capitulation-seller-exhaustion.md`, with Step 0, results and verdict.
- `scripts/cr_bx_capitulation.py`, committed before its run.
- Session note; State of Knowledge update.

## Kickoff prompt for Claude Code

```
Start CR-BX. First action: git fetch && git checkout -b feat/CR-BX-capitulation-seller-exhaustion origin/main (confirm CR-BX is unused; rename if not).
Read the vault CR "/Users/ryan/My Drive/Obsidian Vault/Anonymous/Dash/CRs/CR-BX — Capitulation and Seller Exhaustion.md", components/field-conventions.md, and specs/CR-BW-aggressor-absorption-at-levels.md (reuse its session universe, flow loader, gap rule, fill/cost model and A/A design; copy code rather than importing from an unmerged branch if needed).
Commit 1: spec freeze to specs/CR-BX-capitulation-seller-exhaustion.md, verbatim.
Commit 2: Step 0 findings. Answer all seven gates; fix k under L5 from counts only. Compute NO race outcomes except inside the A/A splits. Stop and report; stop immediately if any gate fails.
Read-only DB, train only (2026-01-06 .. 2026-06-05), no holdout. Do not push.
```

---

## Step 0 findings (2026-10-25)

Script `scripts/cr_bx_step0.py`; log `scripts/logs/cr_bx/step0.log` (git-ignored).

**How it ran.**
- Read-only: `get_backfill_db_conn()` (role asserted), `default_transaction_read_only = on` (asserted), `statement_timeout = 600s`, 3 h Python alarm. SELECTs only.
- Train only: sessions ≤ 2026-06-05, flow `ts_utc` < 2026-06-05 21:00 UTC. Holdout not read.
- CR-BW's session universe, flow loader, gap rule and fill model are copied from `scripts/cr_bw_absorption.py` (unmerged branch, `51f3e3a`). Every operational choice the spec leaves open is fixed in the script docstring.

**No race outcome was computed at any real event.** Races ran only on the family A placebo entries inside the A/A splits, and only the split statistic is reported.

**Run history.**
- **Run 1** (log `step0_run1_signbug.log`) had the aggressor-flow sign inverted: down moves counted buy − sell as positive. That showed up as D(300, 0) ≤ 0 on 70 % of events. It also crashed in the S11(b) report when the 30-minute window found no boxes. No race had run.
- **Run 2** (this report) fixes both. Box counts depend only on price range, so they are identical in both runs.
- One print line was fixed after run 2, with no effect on any number. The log's last line says "family B PASS" for gate 6, but family B was dropped and never raced; it now prints "n/a (dropped)".

**Verdict: Step 0 PASSES for family A at k = 3.0. Family B is DROPPED** (7 eligible boxes; kickoff rule: drop, do not loosen). The run is not started.

### Gate 1 — universe

| rule | removed | remaining |
|---|---|---|
| trading sessions 2025-12-26 → 2026-06-05 | | 111 |
| L14a roll week (2026-03-09 → 03-20) | 10 | 101 |
| L14b no ES 15:45 ET bar on A (2025-12-26, 2026-02-18, 02-19) | 3 | 98 |
| L14c roll mismatch | 0 | 98 |
| no flow rows / RTH open (2026-03-03) | 1 | 97 |
| no canonical `bt_daily_features` row (01-02, 01-20, 02-17, 04-06, 05-26) | 5 | 92 |
| IM null / no `orats_oi_gamma` rows | 0 | 92 |
| < 5 prior scale sessions (2025-12-29, 12-30, 12-31, 2026-01-05) | 4 | **88** |

**88 sessions, 2026-01-06 → 2026-06-05**: the CR-BW A2 universe exactly.

| scale | median | IQR | range | day-to-day ratio p5 / p95 |
|---|---|---|---|---|
| σ5_d (pts) | **2.25** | 2.00–3.00 | 1.75–3.50 | 0.911 / 1.111 |
| σ10_d | 3.25 | 2.84–4.25 | 2.00–4.75 | 0.911 / 1.100 |
| σ30_d | 5.75 | 5.34–7.00 | 3.75–8.25 | 0.934 / 1.071 |

σ5_d monthly medians: Jan 1.75, Feb 2.75, Mar 3.25, Apr 3.00, May 2.25, Jun 2.25.

### Gate 2 — counts (no outcomes)

**Family A:**

| k | events (down / up) | per session median (IQR) | pre-window gap drops | D(300,0) ≤ 0 | eligible | per X tercile |
|---|---|---|---|---|---|---|
| **3.0** | 1,831 (885 / 946) | 20 (13–29) | 30 | 510 | **1,291** | **431 / 430 / 430** |
| 2.5 | 2,169 (1,057 / 1,112) | 26 (18–33) | 31 | 671 | 1,467 | 489 / 489 / 489 |
| 2.0 | 2,652 (1,309 / 1,343) | 32 (26–36) | 36 | 877 | 1,739 | 580 / 579 / 580 |

- **k = 3.0 chosen under L5**: the largest k with ≥ 150 per tercile, from family A counts only. Family B uses the same k.
- X tercile cut points: 1.757 / 3.767.
- Eligible splits:
  - S2: 3–4σ 1,017, ≥ 4σ 274. There is no 2–3σ bucket at k = 3.
  - S3: buy-after-down 622 / sell-after-up 669.
  - S5: before / after 10:00 PT 841 / 450.
- The race-window part of the gap rule needs the race end, an outcome, so it is not applied to real events here. On the A/A placebo races it removed 16 of 6,121 entries (0.3 %).

**Family B (k = 3.0) — DROPPED:**
- Of 1,831 impulses, **11 find a 180 s window with range ≤ 0.5·σ5_d within 15 minutes**. 4 are dropped for gaps, leaving **7 eligible boxes**; per P tercile 3 / 2 / 2.
- After a ≥ 3σ5 move, a 3-minute range of half a typical 5-minute move (median 1.25 pts, 5 ticks) almost never forms inside 15 minutes. Family B is dropped from the run under the kickoff rule (not loosened). Its gate 4, 6 and 7 figures below are reported only for completeness.
- **S11(b) longer boxes:**

| window | rule | found | gap drops | eligible | per tercile | median range | median wait |
|---|---|---|---|---|---|---|---|
| 10 min | range ≤ 0.5·σ10_d, within 60 min | 7 | 6 | **1** | 1 / 0 / 0 | 1.75 pts | 1,532 s |
| 30 min | range ≤ 0.5·σ30_d, within 60 min | 0 | 0 | **0** | 0 / 0 / 0 | — | — |

  A longer window does not rescue family B: the range rule scales with σ_w, and post-impulse ranges stay wide.

### Gate 3 — X distribution (family A eligible, n 1,291)

- Percentiles 5 / 10 / 25 / 50 / 75 / 90 / 95: −0.49 / 0.36 / 1.45 / **2.61** / 4.67 / 9.37 / 17.76. Mean 7.10, sd 35.7, range −89 → 925.
- **X ≤ 0 (flow flipped): 93 (7.2 %).** X < 1 (fading): 18.1 %.
- S7: X30 median 3.84.
- At a 3σ extreme, last-minute flow usually runs above the 5-minute average (median X 2.6). The "exhausted" low tercile (X ≤ 1.76) is relatively fading, not mostly flipped.

### Gate 4 — placebo feasibility

- **Family A:** 6,121 placebo entries for 1,291 eligible events; per event median 5. **71 events have fewer than 5 eligible placebo seconds.**
- Family B (dropped): 6 placebo boxes for 7 boxes; 5 boxes have none.

### Gate 5 — worked event (seed 20261025; outcome hidden)

- Session 2026-01-26, `XCME:ES.H26`. σ5_d 2.00, k 3.0. Event 09:04:34 PT, **up move → fade short**.
- 300 s window low 6984.00, high 6990.50: a move of 6.50 pts = 3.25 σ5.
- The 30 s path: buy outweighs sell in 9 of 11 slots, and price climbs 6984.25 → 6990.00. The last slot is the event second itself (359 buy / 284 sell).
- **D(300, 0) 1,335; D(60, 0) 469; D(30, 0) 398. X = 469 / (1,335 / 5) = 1.757 → low (exhausted) tercile**, just at the cut. X30 2.981.
- Entry: next second 09:04:35 PT, open 6990.00, sell at **6989.75** (1 tick slippage); target and stop ±8 ticks.

### Gate 6 — A/A calibration — family A PASS

- 200 splits of the placebo entries by parent event, using each parent's X tercile. Statistic: [low − high win rate](half 1) − (half 2). Session-cluster bootstrap, 2,000 per split.
- Race-window gap drops −16; unresolved 0. Resolved placebo 6,105 in 1,227 parent events.

| | value |
|---|---|
| statistic median | **+0.03 pts** (gate ±3) |
| mean / sd | +0.04 / 3.97 |
| p5 / p95 | −6.41 / +6.42 |
| **CI excludes 0** | **5.0 % (10 / 200)** (gate ≤ 10 %) |

Family B: not run (dropped).

### Gate 7 — box geometry (family B, for the record)

- Median box range 1.25 pts (5 ticks), IQR 1.12–1.38. 0.5·σ5_d median 1.25.
- Time from impulse to box end: median 621 s (IQR 477–660). The box window is fixed at 180 s.
- Boxes narrower than 4 ticks: 0.

### Consequences for the run (no redefinition proposed)

- **Family A runs:** T1, T2 and S1–S7, S14.
- **Family B does not:** T3, T4 and S8–S13 are not computed.
- Under the verdict table, "neither family passes" then rests on family A alone.

---

## Results (single run, 2026-10-25) — family A only

**The run.**
- `scripts/cr_bx_capitulation.py` at commit `f18a928` (sha256 `6f37d5b0…95cd49`), one run, exit 0.
- Read-only: `default_transaction_read_only = on`, `statement_timeout = 600s`, 3 h alarm. Train only; holdout not read.
- Every statistic was written to `scripts/logs/cr_bx/results.json` before anything was printed. The printout is `scripts/logs/cr_bx/run.log` (git-ignored).
- The script asserted Step 0 consistency before any race: 88 sessions, 1,831 events, 1,291 eligible, 431 / 430 / 430. All held.

**Family B: dropped at Step 0, no outcome computed.**
- The box rule is too narrow: 0.5·σ5_d ≈ 1.1 pts (median 1.25 pts, 5 ticks).
- Only 11 of 1,831 impulses found a qualifying 180 s box within 15 minutes; 7 were eligible.
- S11(b): 10-minute window 1 eligible, 30-minute window 0.
- T3, T4 and S8–S13 were not computed.

**Universe.**
- 88 sessions, k = 3.0, 1,291 eligible events.
- X cut points 1.757 / 3.767.
- 6,121 placebo entries, 5 per event within ±30 min, same trade direction.

**Drops (R = 8):**
- Race-window gap rule: real 1 / 0 / 2 by tercile; placebo 6 / 9 / 1.
- Unresolved by 13:00 PT: 0 for both.

### T1 — exhaustion → reversal — **FAIL**

| X tercile | resolved | win rate (R = 8) |
|---|---|---|
| low (exhausted) | 430 | 39.30 % |
| mid | 430 | 46.74 % |
| high | 428 | 39.25 % |

**Low − high = +0.05 pts, 95 % CI [−7.74, +7.45].**
- ≥ +8: no.
- CI lower bound > 0: no.
- ≥ 150 resolved per tercile: yes.

### T2 — exhausted capitulation vs random entry — **FAIL**

Matched placebo: the 2,012 placebo entries whose parent event is in the low tercile, win rate 36.43 %.

**T2 = 39.30 − 36.43 = +2.87 pts, 95 % CI [−2.79, +8.33].**
- ≥ +8: no.
- CI lower bound > 0: no.

### Verdict (locked rules)

**Family A: neither T1 nor T2 passes. Recorded.** Family B was dropped at Step 0. **Neither family passes, so under the verdict table the order-flow scalping line stops here on train.** No holdout read.

### Secondary readouts (not gated; no conclusion drawn)

**S1 cost reality.** Expectancy is per resolved trade, in ticks: win +R, loss −(R + 1), minus 0.36 tick.

| R | break-even WR | real WR low / mid / high | real exp. low / mid / high | real all (n, WR, exp.) | placebo all (n, WR, exp.) |
|---|---|---|---|---|---|
| 4 | 59.56 % | 34.57 / 32.09 / 29.53 | −2.25 / −2.47 / −2.70 | 1,291, 32.07 %, −2.47 | 6,115, 30.22 %, −2.64 |
| 6 | 56.62 % | 38.98 / 40.93 / 36.05 | −2.29 / −2.04 / −2.67 | 1,291, 38.65 %, −2.34 | 6,109, 34.90 %, −2.82 |
| 8 | 55.06 % | 39.30 / 46.74 / 39.25 | −2.68 / −1.41 / −2.69 | 1,288, 41.77 %, −2.26 | 6,105, 37.31 %, −3.02 |

Every cell is negative. Fading a 3σ5 move loses 1.4–2.7 ticks per trade after costs.

**S2 move size:**

| move | real resolved (low / mid / high) | T1 | T2 |
|---|---|---|---|
| 3–4σ | 303 / 367 / 344 | +0.34 [−8.52, +9.03] | +2.25 [−4.54, +8.98] |
| ≥ 4σ | 127 / 63 / 84 | −5.12 [−16.27, +6.00] | +3.73 [−6.98, +13.95] |

There are no 2–3σ events at k = 3.

**S3 direction:**

| direction | T1 | T2 |
|---|---|---|
| buy after down | −1.21 [−11.79, +9.00] | +0.01 [−8.61, +8.65] |
| sell after up | +0.71 [−9.18, +10.17] | +5.28 [−2.76, +12.60] |

**S4 at levels** (event extreme within 2 pts of a CR-BW A1.1 level; 256 events):

| | T1 | T2 |
|---|---|---|
| near a level (resolved 95 / 83 / 78) | +3.87 [−12.01, +19.38] | +2.87 [−8.87, +15.66] |
| not near | −0.91 [−9.60, +7.31] | +2.86 [−3.83, +9.40] |

**S5 time of day:**

| | T1 | T2 |
|---|---|---|
| before 10:00 PT | +0.78 [−8.19, +9.45] | +2.26 [−4.01, +8.37] |
| after 10:00 PT | −1.95 [−14.15, +9.16] | +4.00 [−5.00, +12.71] |

**S6 flow flipped (X ≤ 0).** Real 48.39 % (n 93) vs matched placebo 32.51 % (n 443): **diff +15.88 [+3.24, +28.78].** This is the only interval clear of 0 among the win-rate readouts. It is unregistered, n 93, and one of many readouts. Even at 48.4 % it sits below the 55.1 % break-even at R = 8.

**S7 30 s window.** X30 cut points 2.337 / 5.865. T1 −0.09 [−6.42, +6.12]; T2-form +3.96 [−0.88, +8.67].

**S14 eventual outcome.** Signed move in the fade direction from the entry price, in σ5_d units (mean by X tercile low / mid / high).

| horizon | real | placebo | low − high | low − matched placebo | share on fade side, real (placebo) |
|---|---|---|---|---|---|
| +30 min | −0.23 / +0.06 / −0.19 | −1.48 / −0.72 / −0.69 | −0.05 [−0.93, +0.78] | +1.24 [+0.78, +1.70] | 46–48 % (38–43 %) |
| +60 min | −0.35 / −0.10 / −0.02 | −1.42 / −0.84 / −0.96 | −0.33 [−1.50, +0.80] | +1.07 [+0.48, +1.62] | 46–49 % (41–43 %) |
| close | −0.49 / −0.91 / +0.24 | −1.30 / −1.69 / −0.82 | −0.73 [−2.15, +0.71] | +0.81 [+0.23, +1.31] | 48–51 % (44–47 %) |

**How to read S14.**
- Real capitulation fades drift slightly against the trade on average, and X does not sort them.
- The placebo entries drift against the trade by more, about 1–1.5 σ5. They trade the same fade direction at random seconds within ±30 min of a 3σ move, so they sit inside the trend that produced it.
- So the real-minus-placebo margin, and the T2 margin, reflect continuation drag on the placebo more than any edge at the extreme. Fading a capitulation is less bad than fading a random point of the same trend window, but it is still negative.
