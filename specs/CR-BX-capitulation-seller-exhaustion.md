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
