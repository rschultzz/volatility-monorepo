# CR-CB — Failed Retest at a Level

Spec freeze. Branch `feat/CR-CB-failed-retest-at-level` off `origin/main` `d97f3bf` (PR #76 merge), in its own worktree. Read-only against the database; no table writes, no cron, no frontend.

Source: vault `Dash/CRs/CR-CB — Failed Retest at a Level.md` (last_reviewed 2026-10-25). Body below is transcribed verbatim from that note (frontmatter omitted); the locked decisions L1–L11, Stage 1 readouts, Stage 2 tests T1/T2 and Step 0 questions were confirmed by the Stage 1 kickoff prompt (Ryan, 2026-10-25). Kickoff additions: reuse CR-BW's universe (A2), level selection A1.1, flow loader, gap rule, fill/cost model and A/A design (A1.3); Step 0 fixes O_c per cell from pre-entry paths only and computes no exit outcome except the A/A hold-to-close on placebo setups; train only (2026-01-06 .. 2026-06-05); 2026-10-23 read only for the Q4 detector illustration (no outcome).

---

> [!summary] Framework (Ryan, 2026-10-25) — read this first; the amendments below are the detailed record
> **Layer 1 — where:** major GEX levels (Corrected, ≥ 40 $bn gross) are where price should **slow down**. Levels give **no direction** by themselves, at call or put levels alike. Treating call levels as automatic shorts was a mistake; Ryan recalls about 50/50 at call levels without flow, and CR-BU/CR-BI agree.
> **Layer 2 — the setup:** price stalls and a range forms at the level (L5–L6, A4.1). Which side of the level the stall sits on may matter (SS, A5.1).
> **Layer 3 — direction:** **order flow** across the range (A5: push decay, late vs early, absorption, divergence, stall side) plus **put-skew divergence** from the move (A8.2: down move with falling put skew, or up move with rising put skew) and the **skew trend during the range** (A8.3) decides fade vs break at the range edge (A4). Put skew is read at put and call levels alike; call skew is a secondary check everywhere.
> **Tests in order:** **T0** do levels slow price down (A7.1) → **stall side alone** (A7.2) → flow / skew direction rules vs always-fade / always-break baselines (**T-flow**, A4.7) → the named confluences **H-PNB** (long at put levels, gate T5) and **H-PSD** (short at call levels, gate T5b) (A6, A8) → profitability after costs (T1), real vs placebo (T2), stop-free move (T3). Stage 1 (2026, Ironbeam flow) searches and shapes the rules with Ryan in the review tool (CR-CC); **Stage 2 (2024-04 → 2025-12, Databento, CR-CA) decides.**

## Why

**Ryan (2026-10-25)** describes a discretionary pattern he sees often. Worked on 2026-10-23, a holdout day used only to illustrate the detector:
1. ES ran into the Corrected 7686 put-heavy node from below and overshot about 9–11 pts.
2. It set a local high around 7694–97 and pulled back.
3. It retested that high around 7694 and failed.
4. A short there with a ~3.5-pt stop caught about 34 pts.

Entering on the first touch at 7686 would have lost.

**This is untested.** Every earlier CR entered on the first touch or the first extreme. This one enters on the **failed retest**.

**Ryan's requirements:**
- puts and calls separate (he expects bigger overshoots at negative-gamma levels);
- no fixed target: find the exit that maximises profit;
- the CR's job is to discover what that is.

**Design consequence: two stages.**
- **Stage 1 (train, 2026):** the exit is chosen from a pre-declared grid. That is a search, so its result is in-sample.
- **Stage 2 (replication set R, 2024-04-03 → 2025-12-31, after CR-CA lands):** the chosen exit and everything else are frozen and run once. Only Stage 2 decides.

## Locked decisions (Ryan to confirm before spec freeze)

| # | Decision | Value |
|---|---|---|
| L1 | Scope | Test only. Read-only, no table writes, no cron, no frontend. |
| L2 | Data | ES 1-second OHLC and flow: `ironbeam_es_flow_1s` (Stage 1, train 2026-01-06 → 2026-06-05, the CR-BW A2 universe and exclusions); `es_flow_1s_databento` (Stage 2, set R, CR-BT exclusions). RTH 06:30–13:00 PT. Holdout not read. |
| L3 | Levels | CR-BW A1.1, unchanged: Corrected lines, ≤ 30 DTE, gross magnitude ≥ 40 $bn, chart spacing, the 6 largest within ±1·IM of the open. Each level is tagged **put-heavy** (net < 0) or **call-heavy** (net > 0). |
| L4 | Cells | 4 cells, reported separately: {put-heavy, call-heavy} × {approached from below → short setup, approached from above → long setup}. |
| L5 | Test and overshoot zone | A level is **tested** when price trades within 2 pts of it after being ≥ 0.10·IM away (CR-BW re-arm). Overshoot allowance O_c **per cell** = the 80th percentile of the maximum overshoot beyond the level before price first moves 0.10·IM back. Measured on Stage 1 **pre-entry price paths only**, in Step 0, and frozen there. The zone runs from the level to level + O_c in the approach direction. |
| L6 | Local extreme | Short setups: H = the highest high reached in the zone after the test, **confirmed** when price pulls back ≥ 6 pts (and ≥ 0.10·IM, whichever is larger) without exceeding H. If price exceeds level + O_c first, there is no setup: the level is broken. Longs mirror this with L, the low. |
| L7 | Failed retest (entry) | Within 90 minutes of confirmation, price returns to ≥ H − 1 pt without trading above H + 1 tick. **Entry:** short limit at H − 1 pt, filled only on a trade 1 tick through it (CR-BW fill rule). Same 1-second bar hitting both entry and stop → loss. Longs mirror this. One setup per level per session. |
| L8 | Stop | H + 2 pts, filled at stop + 1 tick slippage. Fixed; not searched. |
| L9 | Exit grid (Stage 1 search) | Pre-declared, nothing else: fixed targets 5, 8, 10, 15, 20, 30 pts; trailing stops of 4, 6, 10 pts (activated after +4 pts); time exits at +15, +30, +60 min; hold to 13:00 PT. Each with the L8 stop. Cost $4.50 round trip. |
| L10 | Placebo | The **same pattern** (test zone, extreme, failed retest, using the cell's O_c) at pseudo-levels: same session, prices ≥ 15 pts from any real level, at matched distances from the open (CR-BW A1.2 style, 2 per real level). Same entries and exits. |
| L11 | Bootstrap | Session-cluster, 10 000 resamples, seed 20260920, 95 % percentile. |

## Amendment A1 (Ryan, 2026-10-25, before Step 0 outcomes): sweep the setup parameters too

Ryan: test all the parameters; the concern is discarding real patterns because they are defined too tightly. Stage 1 therefore searches the setup parameters as well as the exit. Stage 2 still freezes one configuration per cell and decides.

| # | Clause |
|---|---|
| A1.1 | **Stage 1 grid** (pre-declared, nothing else): overshoot allowance O_c at the {60th, 80th, 95th} percentile; confirmation pullback {4, 6, 10} pts (with the ≥ 0.10·IM floor); retest window {30, 90, 180} min; stop buffer beyond the extreme {1, 2, 4} pts; retest tolerance {0.5, 1, 2} pts below the extreme. That is 243 setups, each crossed with the 13 L9 exits. |
| A1.2 | **Choice rule: a plateau, not a peak.** For each cell, score each configuration by the **mean net expectancy of it and its immediate grid neighbours** (one step in any one parameter). Pick the configuration with the best neighbourhood score among those with ≥ 40 trades. Report the raw best too, and how far it sits from the plateau choice. |
| A1.3 | **Overfitting report:** for each cell, the distribution of net expectancy across all configurations, and the same distribution at placebo levels. A real effect shifts the whole real distribution above placebo, not just its top. |
| A1.4 | **Stage 2 unchanged:** one frozen configuration plus exit per cell (the A1.2 choice), run once on R. |
| A1.5 | **Step 0 Q3/Q4** report counts and the detector table at the L5–L8 defaults (the centre of the grid). Counts at the grid corners too: fewest and most setups. |

## Amendment A2 (Ryan, 2026-10-25, before Step 0 outcomes): a stop-free "is it real" readout

**Ryan:** futures first; the first question is whether the pattern is real. A fixed stop mixes two things: whether price moves the predicted way, and whether the overshoot hunts the stop first. A2 separates them.

| # | Clause |
|---|---|
| A2.1 | **S1-d, stop-free move (the "is it real" readout).** Per setup, the signed move in the setup direction from entry to +15, +30 and +60 min and to 13:00 PT, in points, with **no stop**. Reported per cell, at real levels vs placebo, as mean with CI, and as the share positive. Grid centre and the A1.2 plateau configuration. |
| A2.2 | **S1-e, adverse excursion before the move.** Per setup, the maximum adverse excursion before the first +10 pts (or by the close). Its distribution per cell answers how much overshoot room a real move needs, and informs the stop grid. |
| A2.3 | **Interpretation rule, fixed now:** "real" (for Stage 2 to confirm) means the stop-free move at +30 or +60 min beats placebo with a CI lower bound > 0 in a cell, whatever the stopped P&L does. A real move that the stop grid loses on points to a stop / expression problem (wider stop, scaled entry, or options), not to "no edge". |
| A2.4 | Stage 2 adds T3 per cell: the A2.3 stop-free move at the Stage 1 horizon, real > placebo, CI lower bound > 0. |

## Amendment A3 (Ryan, 2026-10-25, before Step 0 outcomes): both resolutions — fade the failure, follow the break

**Ryan:** the level only says price will slow down there; something else must give the direction. The direction comes from how the stall resolves. A3 tests both resolutions on the same setups.

| # | Clause |
|---|---|
| A3.1 | **Branch F (fade, unchanged):** L7 failed retest → trade against the approach (short below a level approached from below). |
| A3.2 | **Branch K (break, new):** after the L6 extreme is confirmed, within the same retest window, price trades through the extreme by ≥ 1 pt (above H for a level approached from below; below L for one approached from above) → trade **with** the approach. **Entry:** stop order at H + 1 pt, filled at + 1 tick slippage. **Stop:** below the pullback low between confirmation and the break, or 4 pts, whichever is tighter, + 1 tick slippage. Same exit grid (L9), cost model, A1 parameter grid and A2 stop-free readouts. |
| A3.3 | **Resolution share:** per cell, the share of confirmed extremes that resolve F, K, or neither within the window. Reported in Step 0 from price paths only. |
| A3.4 | **Cells become 8:** {put, call} × {from below, from above} × {F, K}. Placebo, A/A and the A1.2 plateau rule apply per cell. Stage 2 freezes one configuration and exit per cell and tests T1–T3 per cell. |
| A3.5 | **Flow stays a readout** (S1-c, now per branch): net aggressor flow in the 60 s before entry, signed with the trade. A later CR may promote it to a filter on R. |

## Amendment A4 (Ryan, 2026-10-25, before Step 0 outcomes): order flow chooses the direction at the range edge

**Ryan:** order flow must be a major signal. The level and the local highs/lows only set up a developing range; order flow says whether to go short or long off its top or bottom. A4 makes flow the direction signal. A3's fade (F) and break (K) entries become the two possible actions at an edge, and A3's always-fade and always-break become **baselines**.

| # | Clause |
|---|---|
| A4.1 | **Range.** After the L5 test and the L6 extreme (H for a level approached from below, L from above) is confirmed, the opposite range boundary is the pullback extreme reached before price returns toward H (PL, the pullback low; or PH for approaches from above). Range = [PL, H]. It is valid until either boundary is broken by > the stop buffer, or the A1 retest window ends. |
| A4.2 | **Edge events.** Price trades within the retest tolerance (A1 grid) of **either** boundary. The first touch of each boundary after the range forms is one event; at most one event per boundary per range. Each event belongs to a cell: {put, call} × {approached from below, from above} × {top edge, bottom edge}. |
| A4.3 | **Flow features at the edge** (1-second flow, from the 120 s before the touch; same sign conventions as CR-BW L7):<br>• **Push P** = net aggressor volume toward the edge (buyers into the top, sellers into the bottom);<br>• **Impact A_rel** = (price move toward the edge / P) ÷ the session's prior-20-session baseline (CR-BW L7/A1.5), defined when P > 0;<br>• **Second-push ratio Q** = P ÷ the push into the same boundary when it was set (the 120 s before H or PL was made). |
| A4.4 | **Direction rules (Stage 1 grid dimension; pick by the A1.2 plateau rule).** Tercile cut points are computed on Stage 1 events per cell and frozen for Stage 2.<br>• **FR1 impact:** A_rel low tercile → fade; high tercile → break; else no trade.<br>• **FR2 divergence:** Q < 0.75 → fade; Q > 1.25 → break; else no trade.<br>• **FR3 push:** P ≤ 0 → fade; P high tercile → break; else no trade.<br>• **FR4 agree:** FR1 and FR2 give the same action; else no trade. |
| A4.5 | **Actions.** **Fade** = limit at the edge −/+ 1 pt against the push (A3 F entry), stop = edge ± the stop buffer. **Break** = stop order 1 pt beyond the edge (A3 K entry), stop = back inside the range by the stop buffer (max 4 pts). Exits from the L9 grid, with the cost model and A2 stop-free readouts per action. |
| A4.6 | **Baselines at the same events:** always-fade and always-break (A3 F / K) with the same entries, stops and exits. |
| A4.7 | **Primary question — does flow add direction?** T-flow per cell: net expectancy of flow-directed trades − the better of the two baselines, CI lower bound > 0. Also flow-directed at real levels vs placebo ranges. Stage 2 freezes one rule, configuration and exit per cell and tests: **T1** net > 0, **T2** real > placebo, **T3** stop-free move > placebo (A2), **T4 = T-flow**. A cell is a **candidate** only if T1 and T4 pass. |
| A4.8 | **Step 0 additions:** edge-event counts per cell; the distributions of P, A_rel and Q; how often each FR rule says fade / break / no-trade per cell (price and flow only, no outcomes); the detector table (Q4) shows the range, both edges, the flow features and the rule decision for each event. |
| A4.9 | **Data.** Flow exists from 2025-12-26 for Stage 1; Stage 2 uses `es_flow_1s_databento` (CR-CA). Same conventions; the Databento side sign is verified in CR-CA Q2. |

## Amendment A5 (Ryan, 2026-10-25, before Step 0 outcomes): flow read across the whole range, plus a skew readout

**Ryan:** once a range forms, read the **trend of flow through the range**, which may run 30 minutes or more, not only the last 2. The questions: are buyers or sellers running out; is the edge absorbing; is price stalled above or below the level? He also asked whether the put-skew change (`orats_monies_minute`) was used correctly earlier; he hopes flow is enough. A5 adds range-wide flow features and rules. A4's 120-second features and FR1–FR4 stay as variants.

| # | Clause |
|---|---|
| A5.1 | **Range-wide flow features**, computed at each A4 edge event from range formation (the confirmation time) to the edge touch, on 1-second flow, with CR-BW sign conventions (positive = toward the edge being touched):<br>• **PD push decay:** P on each successive push into this edge (a push = an approach to within the retest tolerance of the edge, separated by a ≥ 2-pt retreat), as the slope of P across pushes per push, ÷ the mean P. Defined with ≥ 2 pushes; otherwise NA.<br>• **LE late vs early:** delta share (net aggressor / volume) in the last third of the range time vs the first third.<br>• **RA range absorption:** (net price move toward the edge over the range / net aggressor volume toward the edge) ÷ the session baseline (CR-BW L7). Low = absorption.<br>• **DV divergence:** cumulative delta share over the range, signed toward the edge; ≤ 0 while price is at the edge = divergence.<br>• **SS stall side:** share of range seconds on the far side of the level (beyond the level in the approach direction), and (range VWAP − level) in points, signed in the approach direction. |
| A5.2 | **Range-wide rules** (added to the A4.4 grid; the plateau rule still picks one per cell). Tercile cuts per cell on Stage 1 events, frozen for Stage 2.<br>• **FR5 exhaustion:** PD < 0 and LE in its low tercile → fade.<br>• **FR6 absorption:** P > 0 and RA in its low tercile → fade; RA high tercile and LE ≥ 0 → break.<br>• **FR7 acceptance:** SS share ≥ 0.6 and DV > 0 → break; SS share ≤ 0.4 and DV ≤ 0 → fade.<br>• **FR8 divergence:** DV ≤ 0 → fade; DV high tercile and PD ≥ 0 → break.<br>• Else no trade. |
| A5.3 | **FR9 model:** a regularised logistic model predicting "fade wins vs break wins" (stop-free move sign at +30 min, A2.1) from PD, LE, RA, DV, SS and the A4.3 features. Fitted on Stage 1 with leave-one-session-out cross-validation; coefficients frozen. Act when the predicted probability is > 0.6 (fade) or < 0.4 (break). Its Stage 1 score is the cross-validated one only. |
| A5.4 | **Skew readout (S1-f, not a rule):** 0DTE put skew from `orats_monies_minute` (vol75 − vol50, PM expiry, `expir_date = trade_date`). Change from range formation to the edge touch, **net of the change predicted by the price move**: regress Δskew on Δspot (log) over all 5-minute RTH windows of the prior 20 sessions, and use the residual. Exclude events before 07:00 PT (open-time field quirks; field-conventions). Report residual-skew terciles vs the flow-directed outcome, per cell. |
| A5.5 | **Step 0 additions:** distributions of PD, LE, RA, DV, SS; push counts per range; range duration (median, IQR; share ≥ 30 min); how often each of FR5–FR9 says fade / break / no trade per cell; the residual-skew availability count. The Q4 detector table and the gallery JSON include all features and each rule's decision. |

## Amendment A6 (Ryan, 2026-10-25, before Step 0 outcomes): the named confluence — "puts not getting bid into a decline"

**Ryan's hypothesis:** a big move down into a **put** level, with order flow supporting a turn (sellers exhausting or being absorbed) **and** the put wing collapsing (puts not bid into the decline), is a strong long. The mirror (a rally into a call level, buyers exhausting, the call wing not bid) is the short. A6 pre-registers it as one named rule with its own gate, so it is not lost in the A1–A5 search.

| # | Clause |
|---|---|
| A6.1 | **H-PNB (long), all must hold at a bottom-edge event of a range at a put-heavy level approached from above:**<br>(a) **Big move in:** the decline into the level test is ≥ 3·σ5_d within 15 min (CR-BX impulse), or ≥ 0.5·IM_d from the session high within 60 min. Either counts.<br>(b) **Flow supports the turn:** FR5 (exhaustion) **or** FR6 (absorption) says fade at this edge.<br>(c) **Put wing collapsing:** residual 25Δ put skew (A5.4 method, vol75 − vol50, net of the price-move-predicted change) from the start of the decline to the edge touch is in its **lowest tercile** (Stage 1 cut per cell, frozen). |
| A6.2 | **H-CNB (short, mirror):** a top-edge event at a call-heavy level approached from below; a rally ≥ 3·σ5_d within 15 min or ≥ 0.5·IM_d from the session low within 60 min; FR5 or FR6 says fade; residual 25Δ call skew (vol25 − vol50, net of the move-predicted change) in its lowest tercile. |
| A6.3 | **Entry/exit:** the A4.5 fade action; exits from the L9 grid with the A1.2 plateau choice; A2 stop-free readouts. Placebo: the same three conditions at placebo ranges. |
| A6.4 | **Readouts:** each condition's contribution (a)+(b) without (c), (a)+(c) without (b), (b)+(c) without (a), so it is visible which leg carries the result; the 10Δ wing (vol90 − vol50 for puts, vol10 − vol50 for calls) instead of 25Δ; events before 07:00 PT excluded (A5.4). |
| A6.5 | **Stage 2 gate T5 (per rule):** H-PNB and H-CNB each: net expectancy > 0 after costs, CI lower bound > 0, **and** above the same-cell always-fade baseline, CI lower bound > 0. Minimum 40 events per rule in R; below that, the verdict is **inconclusive (underpowered)**, not a fail. |
| A6.6 | **Step 0:** event counts for H-PNB and H-CNB, and for each two-condition subset; skew-field coverage on those events. No outcomes. |

## Amendment A7 (Ryan, 2026-10-25, before Step 0 outcomes): T0 "do levels slow price down", stall side alone, skew at both level types

| # | Clause |
|---|---|
| A7.1 | **T0, levels slow price down** (price paths only; Stage 1 and Stage 2). For real levels vs the A1.2-style placebo prices (same session, distance-matched, ≥ 15 pts from real levels), at each first test (L5):<br>(a) **range-formation rate:** the share of tests followed by a confirmed L6 extreme and a range (A4.1) at the grid-centre parameters;<br>(b) **dwell:** seconds spent within ±2 pts of the price in the 30 min after the test;<br>(c) **velocity drop:** absolute price speed (pts/min) in the 5 min after the test ÷ the 5 min before.<br>Per cell (put/call × approach side). **Pass (Stage 2):** (a) or (b) real > placebo, CI lower bound > 0, in a cell. If T0 fails in a cell, that cell's results are read as structure + flow alone; levels add nothing there. |
| A7.2 | **Stall side alone** (readout in Stage 1; Stage 2 gate T6): the direction rule "stall on the far side of the level → break, stall on the near side → fade" (SS share ≥ 0.6 / ≤ 0.4), with no flow or skew. Its net expectancy vs the always-fade / always-break baselines per cell, so it is visible whether stall side carries direction before flow is added. |
| A7.3 | **Skew at both level types:** compute **both** residual 25Δ put skew (vol75 − vol50) and residual 25Δ call skew (vol25 − vol50) (A5.4 method) for every edge event, at put and call levels. Add both to FR9 (A5.3). Report each skew's tercile vs the flow-directed outcome per cell, so "put skew is more telling at put levels" is measured. |
| A7.4 | **H-CNB is a readout, not a gate** (amends A6.5): T5 applies to H-PNB only. H-CNB assumed call skew behaves like put skew, which is now an open question (A7.3). |
| A7.5 | **Step 0 additions:** the T0 descriptive counts: tests, range-formation shares and dwell, real vs placebo, per cell. These come from price paths and are allowed in Step 0, because T0 is not a trade outcome. Report them before any trade outcome is computed. |

## Amendment A8 (Ryan, 2026-10-25, before Step 0 outcomes): put skew everywhere, skew divergence as Ryan defines it, skew trend in the range

**Ryan:**
- look at **put skew at call levels** too, not only at put levels;
- call skew is worth checking everywhere but is probably less telling;
- the skew signal is **divergence from the move**: a move down with *falling* put skew is a divergence, and so is a move up with *climbing* put skew;
- also check whether the **trend of skew during range formation** matters.

| # | Clause |
|---|---|
| A8.1 | **Put skew is the primary skew everywhere.** PS = vol75 − vol50 (25Δ put minus ATM), 0DTE PM expiry, `orats_monies_minute`, at put **and** call levels. Call skew CS = vol25 − vol50 is computed everywhere as a secondary readout. Events before 07:00 PT are excluded (A5.4). |
| A8.2 | **Skew divergence, Ryan's definition (primary):**<br>• over the **move into the level** (from the move's start, per A6.1(a), or the session extreme within 60 min, to the level test): ΔP = price change, ΔPS = put-skew change;<br>• **bullish divergence** = ΔP < 0 and ΔPS < 0 (puts not bid into a decline);<br>• **bearish divergence** = ΔP > 0 and ΔPS > 0 (protection bid into a rally);<br>• otherwise no divergence.<br>Magnitude: ΔPS per 10 pts of move. The **residual** version (A5.4: ΔPS net of the move-predicted change) is reported alongside as a variant, not the primary. |
| A8.3 | **Skew trend in the range:** the slope of PS (raw, and residual) from range formation to the edge touch, per 10 minutes, signed with the setup (falling PS at a bottom edge counts as positive for a long). Plus the change in PS slope between the move-in and the range (did the divergence continue, stop or reverse while the range formed?). |
| A8.4 | **New direction rules** (added to the grid; the plateau choice still applies):<br>• **FR10 skew divergence:** bullish divergence on the move in → fade a bottom edge (long); bearish divergence → fade a top edge (short); else no trade.<br>• **FR11 flow + skew agree:** FR5 or FR6 says fade **and** FR10 gives the same direction.<br>• **FR12 skew trend:** the range PS trend in its favourable tercile (A8.3 sign) → fade at that edge; unfavourable tercile → break. |
| A8.5 | **H-PNB condition (c) redefined** (amends A6.1): (c) = **bullish divergence** on the move in (A8.2, primary). The A6.1 residual-tercile version becomes a readout variant. |
| A8.6 | **H-PSD (short), new named rule, replacing H-CNB:** a top-edge event at a **call-heavy** level approached from below; (a) a rally ≥ 3·σ5_d within 15 min or ≥ 0.5·IM_d off the session low within 60 min; (b) FR5 or FR6 says fade; (c) **bearish put-skew divergence** on the rally (A8.2). Same handling as H-PNB, with its own Stage 2 gate **T5b** (≥ 40 events in R, else inconclusive). H-CNB (call skew) stays a readout. |
| A8.7 | **Readouts:**<br>• divergence by level type: put-skew divergence at call levels and at put levels, both directions;<br>• call-skew divergence (same definitions with CS) for comparison;<br>• raw vs residual divergence;<br>• FR12 trend alone vs FR10 alone vs FR11, per cell. |
| A8.8 | **Step 0 additions:** divergence counts (bullish, bearish, none) per cell on the move in; PS trend distributions in the range; event counts for H-PNB (new (c)) and H-PSD; PS/CS field coverage. No outcomes. |

## Amendment A9 (Ryan, 2026-10-25, after the pre-A2 Step 0; no trade outcome computed)

**Context.** The first Step 0 ran on the pre-A2 design (L5–L8 plus A1). It found:
- the put-heavy cells never reach 40 fills on the A1 grid (max 31 / 29);
- the 0.10·IM floor on the confirmation pullback, together with keeping the first confirmed extreme, kept the detector from matching Ryan's 2026-10-23 trade;
- adjacent levels sometimes share one fill;
- the Ironbeam flow has a 07:52 → 09:56 PT hole on 2026-10-23.

A9 settles these before Step 0 is re-run on the full design (A1–A8).

| # | Clause |
|---|---|
| A9.1 | **Flow source.** Stage 1 (2026-01-06 → 2026-06-05) uses `es_flow_1s_databento` (CR-CA). `ironbeam_es_flow_1s` is a fallback only, for seconds or sessions Databento lacks, and every fallback use is counted. All flow thresholds and baselines are on the Databento scale throughout: the CR-BW-style impact baseline, terciles, and the A1.4/A2.3 gap rule. This amends A4.9 and the Framework's "Stage 1 (2026, Ironbeam flow)". |
| A9.2 | **Pooling for the exit choice.** Under A1.2, when a cell has < 40 trades, its short and long sides are pooled within the same level type (put-heavy with put-heavy, call-heavy with call-heavy) for the configuration and exit choice. **Stage 2 stays per cell.** |
| A9.3 | **O_c reading confirmed.** O_c is measured from the running extreme after the test: overshoot up to the first 0.10·IM pullback from it (the first Step 0's reading). |
| A9.4 | **No 0.10·IM floor on the confirmation pullback.** L6 confirmation uses the A1 grid value P ∈ {4, 6, 10} pts alone. |
| A9.5 | **The range keeps developing (amends L6 / A4.1).** Until a fill, a later, more extreme high (low for approaches from above) inside the zone replaces H (L). The range [PL, H] updates accordingly. |
| A9.6 | **One trade per fill.** When adjacent levels produce the same fill (same session, side and fill second), it is one trade, not one per level. |
| A9.7 | **2026-10-23 illustration** uses Databento flow (`es_flow_1s_databento`), still detector-only with no outcome. |

### A9.5′ — ruling on A9.5 (Ryan, 2026-10-25)

As written, A9.5 is nearly inert. A limit fade at H − tol fills before price can make a new high, so after confirmation H could only move on a jump between seconds. Ryan chose re-arming after a stop.

| # | Clause |
|---|---|
| A9.5′ | **Re-arm after a stop (replaces A9.5; amends L7's one setup per level).** After a fade is stopped, if price is still inside the cell's zone (level + O_c), H (or L) resets to the new extreme and a new setup at the same level is allowed. At most **3 attempts per level per session**. Every attempt is a trade and is counted. |
| A9.5′.1 | **Range-edge events (A4.2) re-arm the same way.** After a stopped trade, the range boundaries re-form from the new extreme. |
| A9.8 | **Accepted assumptions (Ryan, 2026-10-25):**<br>• the gallery `outcome` is null in Step 0 (the review reveal shows the price path; P&L is backfilled after the Stage 1 run);<br>• a fill shared by adjacent levels is one trade, credited to the higher-magnitude level, which also sets its cell;<br>• FR9's inputs are reported in Step 0, but the model is not fitted. |
| A9.9 | **Gallery export (Step 0).** `research/gallery/cr-cb-stage1.json` per CR-CC L10, `outcome` null. Contents: 40 random edge events stratified across cells (seed 20261024); all H-PNB / H-PSD events; and the 2026-10-23 7686/7687 events (all attempts). `meta` carries: range bounds, attempt number, O_c, flow features (P, A_rel, Q, PD, LE, RA, DV, SS), put/call skew divergence and trend, and each FR rule's decision. |

## Amendment A10 (Ryan, 2026-10-25, before Step 0 outcomes): each variation is a candidate setup; promotion pipeline and data split

**Ryan:** the cells and rules are potentially **different trade setups**. If one works among several that don't, focus on optimizing that one. The goal is several setups, taken one at a time.

| # | Clause |
|---|---|
| A10.1 | **A setup** = one cell (level type × approach × edge) + one direction rule (FR1–FR12, H-PNB, H-PSD, always-fade, always-break) + its plateau-chosen configuration and exit. Each setup is tracked on its own in `research/setup-registry.md`. |
| A10.2 | **Data split, fixed now** (R from CR-CA): **R-a** = 2024-04-03 → 2024-12-31 (confirmation); **R-b** = 2025-01-01 → 2025-12-31 (final confirmation of optimized setups); **H** = after 2026-06-05 (holdout, untouched). |
| A10.3 | **Pipeline per setup:**<br>**(1) Screen:** Stage 1 on 2026 train. A setup is **shortlisted** if its stop-free move (A2) or its net expectancy beats placebo and both baselines with a CI lower bound > 0. Cap: 5 setups.<br>**(2) Confirm:** Stage 2 runs on **R-a only**, frozen as chosen in Stage 1. A Holm correction applies across the shortlisted setups.<br>**(3) Optimize:** for each confirmed setup, a dedicated CR tunes it (finer parameter and exit grids, an optional flow or skew filter, sizing) on 2026 train + R-a, with the plateau rule.<br>**(4) Final confirm:** the optimized setup runs once on **R-b**.<br>**(5) Holdout and paper:** one read on H, then paper trading via the live indicator. |
| A10.4 | **Amends Stage 2:** "Stage 2 on R" means Stage 2 on R-a. R-b is not read until a setup reaches pipeline step 4. |
| A10.5 | **Losers are recorded too:** every non-shortlisted setup is logged in the registry as refuted or inconclusive (three-verdict rule), with its key numbers, so it is not re-tested by accident. |

## Amendment A11 (Ryan, 2026-10-25, before Step 0 outcomes): structure-based stops

**Ryan, reviewing the CR-CC fixture on 2026-02-02.** A long at a put-heavy level, 6991.50, with its stop 1 pt below the extreme at 6988.25. It was stopped at 08:05 on a tick-level flush, then ran about 20–25 pts the right way. This is a fixture example (a CR-BW-style touch), not CR-CB evidence. But CR-CB Step 0 measured put-level overshoot O_c at about 9–10 pts, and the L8 / A1 stops (1 / 2 / 4 pts beyond the extreme) are a fraction of that.

| # | Clause |
|---|---|
| A11.1 | **Stop family** (replaces the A1 stop-buffer dimension; searched in the Stage 1 grid, plateau rule per cell):<br>**(S-x) extreme + b:** b ∈ {1, 2, 4} pts, as before;<br>**(S-z) zone edge:** beyond the level by the cell's O_c (the overshoot allowance) + 1 tick, so "past the normal overshoot means the level failed";<br>**(S-v) volatility:** entry ∓ m·σ5_d, m ∈ {1.0, 1.5, 2.0}.<br>All filled at stop + 1 tick slippage. For break trades (A3/A4), S-z means back inside the range by the cell's O_c, capped at the range width. |
| A11.2 | **Report per stop family:** win rate, average loss, net expectancy and max drawdown per cell, next to the A2 stop-free move, so it is visible whether a wider stop turns a real move into a profitable setup. |
| A11.4 | **Record R-multiples for later sizing work** (Ryan: the long-term goal is position sizing; fewer contracts and wider stops in high-vol cells such as put levels, more contracts and tighter stops for smaller 5–10 pt moves in low-vol cells such as call levels; not tested now). Every trade row stores its stop distance in points, the result in points, and **R = result / stop distance**, plus σ5_d and IM_d at entry. A later sizing CR can then simulate fixed-risk sizing with no re-run. |
| A11.3 | **Grid size:** the stop dimension grows from 3 to 7 values; the other A1 dimensions are unchanged. The plateau neighbourhood for stops runs within each family only. |

## Amendment A12 (Ryan, 2026-10-25, before Step 0 outcomes): P&L per episode

**Ryan, reviewing fixture 2026-04-01.** A short at call level 6641 (net +60 / gross 140 $bn), entry 6638, stopped at 6643 at 08:45. Price overshot to about 6650 (Ryan's reading; about 9 pts past 6641, about 1.5× the call-level O_c), held about 90 min, then broke down to about 6605. "Maybe this is where you get cut up, but the mechanism seems to exist." No A11 stop saves the first entry; the A9.5′ re-arm (attempt 2 at the new extreme) is what would catch it.

| # | Clause |
|---|---|
| A12.1 | **Episode** = all attempts (A9.5′, up to 3) at one level in one session, plus any range-edge trades (A4) at that level. **Episode P&L** = the sum of its trades' net points (and R, per A11.4). |
| A12.2 | **Report per setup (Stage 1, and Stage 2 per cell):** episode count; episode P&L mean with CI; the share of episodes that are net positive; the distribution of attempts per episode; and the mean P&L by attempt number (1st, 2nd, 3rd). This shows whether the cut-ups outweigh the eventual winner. |
| A12.3 | **Max attempts as a grid dimension:** {1, 2, 3} attempts per level per session, searched with the plateau rule. "1" reproduces the no-re-arm behaviour. |
| A12.4 | **The gate stays per trade** (T1–T5 unchanged). Episode P&L is a readout, but it is required in the setup registry entry for any shortlisted setup. |

## Amendment A13 (Ryan, 2026-10-25, before Step 0 outcomes): wait for the second level in a cluster

**Ryan, on fixture 2026-04-01:** with a second large call level just above (6641 and 6646), he usually waits for the second, because price tends to get there. It did that day. Ryan's correction: the session high was **about 6650**, only about 4 pts past 6646. That is inside the call-level O_c (about 6 pts), so a setup armed at 6646 survives the S-z zone-edge stop (6646 + O_c ≈ 6652). A retest-of-6650 short with a stop above 6652 risks about 3 pts against the drop to about 6605.

| # | Clause |
|---|---|
| A13.1 | **Cluster:** levels from the A1.1 selection (≥ 40 $bn gross, Corrected) whose prices lie within **d** pts of each other, chained transitively, form one zone. **d** is a Stage 1 grid dimension: {0 (off), 5, 10} pts. |
| A13.2 | **The far level arms the setup:** the cluster's level farthest in the approach direction (the highest when price approaches from below, the lowest from above). The L5 test, the O_c overshoot (A1/L5), the L6 extreme, the S-z zone-edge stop (A11) and the range (A4) are all measured from it. Touches of nearer cluster levels do not create setups. |
| A13.3 | **Cluster attributes for cells and readouts:** the cluster's type is the sign of the summed net (put-heavy / call-heavy), with a "mixed" flag when members disagree in sign. Recorded: member count, span (pts), summed gross and net $bn. |
| A13.4 | **Readouts:** clustered vs single-level setups per cell (net expectancy, stop-free move, episode P&L); and, for d > 0, the share of clusters where price reached the far level after touching the near one. That is Ryan's "good chance you will get there", measured from price paths. |
| A13.5 | **Step 0 additions:** the cluster count and span distribution at each d; the "reached far level" share (price paths only); event counts per cell at each d. The gallery meta carries the cluster members. |

## Amendment A14 (Ryan, 2026-10-25, before Step 0 outcomes): same-day reference levels form confluence zones with GEX levels

**Ryan, reviewing fixture 2026-05-01** (call-heavy 7277, net +158 / gross 181 $bn; fixture long 7279.25, stop 7273.50, stopped 08:01, then +24 to about 7296 by 08:45):
- he would look for longs in the 71–72 handle, because that is where price found support earlier in the day, right below the GEX level;
- put skew was falling as price dropped (the skew table shows Δput about −75 % from 07:24 to 07:38), a bullish divergence per A8.

The fixtures are random 2026 CR-BW touches with a fixed stop; CR-BW's full sample won about 50 % before costs at 8 ticks, so the CR-CB stop-free and episode readouts decide, not the cards.

| # | Clause |
|---|---|
| A14.1 | **Reference levels**, known at the time with no lookahead: the prior session's high, low and close; the overnight (ETH) high and low; and today's **confirmed swing highs and lows** so far. A swing is confirmed when price retraces ≥ 4 pts (and ≥ 0.05·IM_d) from it. Each is recorded with its type and timestamp. |
| A14.2 | **Confluence zone:** a GEX level (or A13 cluster) and any reference levels within **e** pts of it, chained, form one zone. **e** is a Stage 1 grid dimension: {0 (off), 4, 8} pts. As in A13.2, the zone's **far edge** in the approach direction arms the setup, and the test, O_c, extreme, S-z stop and range are measured from it. |
| A14.3 | **Readouts:** zone type (GEX only / GEX + same-day swing / GEX + prior-day / GEX + overnight); per cell: net expectancy, stop-free move, episode P&L by zone type; the share of setups whose extreme held within 1 pt of a reference level. |
| A14.4 | **Step 0 additions:** reference-level counts per session by type; confluence-zone counts and spans at each e; how often the far edge is a reference level rather than the GEX level. Gallery meta carries the zone members. |

## Amendment A15 (Ryan, 2026-10-25, after Step 0 v2; no trade outcomes seen): ranges that look like the ones Ryan trades

**Step 0 v2 findings** (Databento, 89 sessions):
- **T0:** pooled no; **put-heavy from below yes** (range formation +11.0 [+1.7, +21.0]); call-heavy from below, price **speeds up**.
- **Ranges** last a median of 60 s; only 1 % reach 30 min.
- **A14** saturates: about 90 swings per session, zone spans up to 52 pts.
- The **break branch** never fires first, by construction.
- **Big-move leg (a)** is near-universal.
- **Placebo** is thin.

| # | Clause |
|---|---|
| A15.1 | **Reference levels (amends A14):** a swing counts only if it retraces ≥ max(8 pts, 0.15·IM_d) **and** is not violated for ≥ 10 min after it forms. Prior-day H/L/C and overnight H/L are unchanged. **Zone = the GEX level (or A13 cluster) + at most one reference level**, the nearest one beyond it in the approach direction within e. No chaining through reference levels. Zone span ≤ 10 pts (if longer, the reference level is not added). |
| A15.2 | **Orders are exclusive per rule (fixes A3/A4):** each direction rule places only the order for its chosen action. Always-fade places only the fade limit; always-break places only the break stop order; flow and skew rules place whichever their decision says. **The A3.3 resolution share is now measured from price paths:** "fail" = price retreats ≥ the confirmation pullback from H before trading H + 1 pt; "break" = it trades H + 1 pt first. |
| A15.3 | **Range minimum duration:** edge events count only after the range [PL, H] has held (no break per A15.4) for **≥ R_min**, a Stage 1 grid dimension: **{10, 20, 30} min** (Ryan: a 2–3 minute stall is not a range). The range bottom PL is fixed only after a rebound ≥ max(4 pts, 0.10·IM_d), not 2 pts. Range-wide flow features (A5) are computed from range formation; PD needs ≥ 2 pushes as before. |
| A15.4 | **Range invalidation:** a range breaks only when price trades beyond a boundary by ≥ max(3 pts, 0.5 × range height). The break branch's trigger (A4.5) stays at 1 pt beyond the edge, so a break *trade* can start before the range counts as broken. |
| A15.5 | **Big move (amends A6.1(a) and A8.6(a)):** a decline (rally) into the level of ≥ 0.5·IM_d from the session high (low) within 60 min, **or** ≥ 5·σ5_d within 15 min. |
| A15.6 | **Placebo density (amends L10):** 4 pseudo-levels per real level, ≥ 10 pts from any real level (was 2 at ≥ 15). |
| A15.7 | **Round structure:** gallery `cr-cb-stage1.json` (A1–A14 definitions) is loaded as review set `cr-cb:round-1`. Ryan's labels and comments on it feed a possible A16 before the round-2 gallery (A15 definitions) is built. Stage 1 trade outcomes are computed only after round 2's definitions are frozen. |
| A15.8 | **Step 0 re-run (v3) under A15:** T0 again with the A15.9 zone measures (time to escape first); range-duration distribution; reference-level and zone counts; break/fail shares from price paths; counts per cell at each R_min; placebo counts; A/A. Gallery round 2: `research/gallery/cr-cb-stage1-r2.json`, same composition as round 1. |
| A15.9 | **T0 measured on the zone, both sides of the level (amends A7.1).** Ryan: the slowdown may happen before or after the level, so the range must be seen developing and traded from there. For each first test, the **zone** = [level − O_c, level + O_c] (the cell's overshoot allowance on **both** sides; A13/A14 cluster or confluence span added).<br>**(a′) Range formation:** a range per A15.3/A15.4 (≥ R_min) whose boundaries both lie inside the zone, forming within 60 min of the test.<br>**(b′) Zone dwell:** seconds inside the zone in the 60 min after the test (replaces the ±2-pt dwell).<br>**(d) Time to escape (new, primary T0 measure):** minutes from the test until price is ≥ 0.25·IM_d from the level in either direction, censored at the 13:00 PT close.<br>**(c) Velocity drop:** unchanged.<br>All vs the A15.6 placebo prices, per cell; Stage 2 pass = (d) or (a′) real > placebo, CI lower bound > 0. |
| A15.10 | **Edge events come from the developed range:** a range's edges (A4.2) count only once the range meets A15.3 (≥ R_min, both boundaries inside the zone). Touching the level itself is not an entry trigger. Entries come from the range edges wherever the range sits relative to the level (above, below or straddling), and **SS (stall side)** records which. |

## Stage 1 — train (search)

**S1-a, exit choice.**
- Per cell: for every L9 exit, net expectancy per trade (pts and $ per contract), win rate, trades, max drawdown in pts, and the same at placebo.
- **Chosen exit per cell** = the highest net expectancy among exits with ≥ 40 trades, ties going to the simpler exit.
- Pooled-across-cells choice too.

**S1-b, MFE/MAE.** Per trade, the maximum favourable excursion before the stop, its distribution per cell, and the full expectancy-vs-target curve.

**S1-c, flow on the retest** (readout). Net aggressor flow in the 60 s before entry, signed with the setup: sellers into a short retest count positive. Terciles vs expectancy.

**Stage 1 verdict is descriptive only:** "promising" if the chosen exit's net expectancy beats placebo with a CI lower bound > 0 in a cell, else "not promising". Even a promising Stage 1 is in-sample and decides nothing.

## Stage 2 — replication set R (decides)

Run **after** CR-CA has loaded R, and after Stage 1 results are written and the chosen exits are recorded in this note. Everything is frozen: L3–L11 and the Stage 1 chosen exit per cell, plus the pooled exit.

| Test | Pass |
|---|---|
| **T1 (per cell with a Stage 1 "promising")** | net expectancy per trade > 0 after costs, CI lower bound > 0, ≥ 100 trades |
| **T2 (same cells)** | real − placebo net expectancy > 0, CI lower bound > 0 |

**Verdicts per cell:**
- **Pass both** → a candidate strategy; next is the holdout read and paper trading.
- **T1 only** → the pattern pays, but levels add nothing.
- **Neither** → recorded.

The three-verdict rule applies: **refuted** if the CI's upper bound is < 0 net, **inconclusive** otherwise.

## Step 0 (Stage 1) — before any exit is evaluated

1. **Universe** (CR-BW A2): sessions, levels per session, levels per cell.
2. **O_c per cell** (L5) from pre-entry paths, with its distribution. Is the put-heavy overshoot larger than the call-heavy one (Ryan's expectation)?
3. **Counts** (no outcomes): tests, confirmed extremes, retests, filled entries per cell, real and placebo. Cells with < 40 filled entries are reported and excluded from the exit choice.
4. **Detector check:** for 5 random train sessions (seed 20261025), list every setup with its times and prices, plus 2026-10-23's 7686 setup as a detector illustration. No outcomes; Ryan eyeballs whether the detector matches what he would call the pattern.
5. **A/A calibration** (CR-BW A1.3) on placebo setups, using the pooled hold-to-close exit only.

## Outputs

- Spec `specs/CR-CB-failed-retest-at-level.md` (Stage 1 and later Stage 2).
- `scripts/cr_cb_retest.py`, committed before each stage's run.
- Session notes per stage; State of Knowledge after Stage 2.

## Kickoff prompt for Claude Code (Stage 1)

```
Start CR-CB in its own git worktree (other CR worktrees may be active): git fetch && git worktree add <scratch>/crcb-wt -b feat/CR-CB-failed-retest-at-level origin/main (confirm CR-CB is unused; rename if not).
Read the vault CR "/Users/ryan/My Drive/Obsidian Vault/Anonymous/Dash/CRs/CR-CB — Failed Retest at a Level.md", components/field-conventions.md, and specs/CR-BW-aggressor-absorption-at-levels.md (reuse its universe, level selection A1.1, flow loader, gap rule, fill/cost model and A/A design).
Commit 1: spec freeze to specs/CR-CB-failed-retest-at-level.md, verbatim.
Commit 2: Step 0 (Stage 1) — all five questions; fix O_c per cell from pre-entry paths only. Compute NO exit outcomes except the A/A hold-to-close on placebo setups. For Q4 print the setups as a table I can check against charts. Stop and report.
Read-only DB, train only (2026-01-06 .. 2026-06-05); 2026-10-23 only for the Q4 detector illustration (no outcome). Do not push.
```

---

## Step 0 findings (Stage 1, 2026-10-25)

Script `scripts/cr_cb_step0.py` (sha256 `d9b270fa…ba060e1`; log `scripts/logs/cr_cb/step0.log`, JSON `step0.json`, git-ignored). Read-only: `get_backfill_db_conn()` (role asserted) + `default_transaction_read_only = on` (asserted), `statement_timeout = 600s`, 3 h alarm; SELECTs only. Train reads bounded to flow `ts_utc` < 2026-06-05 21:00 UTC. 2026-10-23 read in one separate block (flow, 1m bars, `orats_oi_gamma`, IM, P_target for that date) for the Q4 illustration only. `place_levels` / `bucket_levels` imported unchanged; the CR-BW loaders, A1.1 level selection, gap rule and fill rule are copied from `scripts/cr_bw_absorption.py`. Every operational choice the spec leaves open is fixed in the script docstring (summarised below).

**No exit outcome was computed at real levels, for placebo setups as a set, or on 2026-10-23.** The only outcome is the hold-to-close P&L of placebo setups inside the Q5 A/A splits, reported only as the split statistic's distribution.

**Runs.** Two, same code path for everything reported. Run 1 printed all five answers. Inspecting the 10-23 row showed a flow hole, so I added a 1-minute-bar proxy for the illustration, per-cell counts for all 81 configurations, and a shared-fill count. Run 2 is the one reported. The O_c values, counts and A/A are identical across the two runs: Q5 is deterministic and was not changed.

**Verdict: Step 0 answers all five questions; the A/A passes. But the put-heavy cells never reach the 40-trade floor anywhere on the A1 grid (max 31 / 29), so under A1.2 Stage 1 can choose an exit only for the two call-heavy cells.** That is Ryan's decision (below). Stage 1 is not started.

### Operational choices fixed before the run (not stated in the spec)

- **Universe.** CR-BW A2's 88 sessions. CR-BW's "≥ 5 prior baseline sessions" rule only removed sessions before 2026-01-06, so the window simply starts there (asserted n = 88). No B_d (CR-CB has no A_rel).
- **Test (L5).** CR-BW arming with a 2-pt band. Armed below when a 1-s bar's high ≤ L − 0.10·IM (above: low ≥ L + 0.10·IM). The test is the first later bar with high ≥ L − 2 (from below) or low ≤ L + 2 (from above). The level then disarms until price is again ≥ 0.10·IM away.
- **O_c (L5), "before price first moves 0.10·IM back".** Read as a 0.10·IM pullback from the running extreme after the test: the same construction L6 uses for H. Overshoot = max(0, extreme − L), measured over every real test with no RTH flow gap > 15 s in the measurement window. The alternative reading (until price is 0.10·IM back on the approach side of L) is reported but not used. **This reading is a choice Ryan should confirm.**
- **Zone (L5/L6).** [L, L + O_c] as written. A confirmed extreme that stays below L (the test reached the 2-pt band but not the level) is "short of the level": no setup.
- **Extreme (L6).** H = running max from the test bar. Confirmed at the first bar with low ≤ H − max(P, 0.10·IM) whose high is below H (the max was set earlier). Broken if the running max exceeds L + O_c first.
- **Retest and fill (L7).** Window (confirmation, confirmation + W], capped at 13:00 PT. Retest = high ≥ H − tol; fill = a trade 1 tick through the entry H − tol. The "no trade above H + 1 tick" condition is checked on bars before the fill. It is vacuous on continuous 1-s data (it never failed: 0 "overrun").
- **One setup per level per session** = the first filled entry; detector episodes on a level run sequentially.
- **Gap rule.** Drop a setup with an RTH flow gap > 15 s in [test, fill]. The A/A also drops on [fill, exit] (0 dropped).
- **Placebo (L10).** For each real level, the 2 nearest 5-pt bins on the same side of the open, within ±1·IM and ≥ 15 pts from every real level. Ties go to the lower bin; dedup per session; the parent's sign is kept (it decides the cell).
- **Counts depend on (O pct, P, W, tol) only.** The stop buffer does not change counts, so 81 configurations are counted.
- **Q5.** Placebo setups at the grid centre; L8 stop at H + 2 (filled + 1 tick; a fill bar that also reaches the stop is stopped). Held to the last RTH print before 13:00; net of 0.09 pts ($4.50 / $50). 200 splits by pseudo-level (`default_rng(20261025)`); 2,000 session-cluster resamples per split (one `default_rng(20260920)` stream). Pass: ≤ 10 % of splits exclude 0 and median within ±3 **pts** (CR-BW's ±3 was percentage points of win rate).

### Q1 — universe

| rule | removed | remaining |
|---|---|---|
| trading sessions 2026-01-06 → 2026-06-05 | | 105 |
| L14a roll week | 10 | 95 |
| L14b no ES 15:45 ET bar on A (2026-02-18, 02-19) | 2 | 93 |
| L14c roll mismatch | 0 | 93 |
| no flow rows (2026-03-03) | 1 | 92 |
| no canonical `bt_daily_features` row (01-20, 02-17, 04-06, 05-26) | 4 | **88** |

- 88 sessions, `XCME:ES.H26` then `XCME:ES.M26` (one symbol per session), same as CR-BW A2.
- Chart lines (≥ 40 $bn, 5-pt spacing) within ±1·IM: median 15 per session. After the top-6 cut: **522 levels**, median 6 (IQR 6–6, min 1); **put-heavy 168, call-heavy 354**.
- Placebo pseudo-levels (≥ 15 pts from every real level): **283** (put-parent 129, call-parent 154), median 2 per session. With 6 levels inside ±1·IM, each with a 30-pt exclusion band, little of the window is left (CR-BW at 10 pts had 427).
- Armed tests: real **1,666** (put/below 225, put/above 246, call/below 627, call/above 568); placebo 614.

### Q2 — O_c per cell (pre-entry paths only) — fixed here

Overshoot = max(0, extreme − L) up to the first 0.10·IM pullback, in pts. 20 tests are dropped for a > 15 s flow gap in the window. "Censored" = no pullback by 13:00.

| cell | n | censored | share 0 | p25 | p50 | **p60** | p75 | **p80** | p90 | **p95** | p99 | p50 / IM |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| put-heavy, from below (short) | 222 | 2 | 24.3 % | 0.25 | 2.62 | 4.65 | 8.50 | 9.75 | 15.75 | 21.95 | 32.84 | 0.044 |
| put-heavy, from above (long) | 246 | 2 | 28.0 % | 0.00 | 2.38 | 3.50 | 6.50 | 8.75 | 14.00 | 19.00 | 27.94 | 0.036 |
| call-heavy, from below (short) | 615 | 10 | 36.3 % | 0.00 | 1.00 | 2.35 | 5.00 | 6.00 | 9.50 | 12.57 | 23.22 | 0.021 |
| call-heavy, from above (long) | 563 | 10 | 39.6 % | 0.00 | 1.00 | 2.00 | 4.25 | 5.25 | 8.50 | 10.97 | 25.92 | 0.020 |

**Frozen O_c (tick-rounded) for Stage 1:**

| cell | 60th | **80th (centre)** | 95th |
|---|---|---|---|
| put / short | 4.75 | **9.75** | 22.00 |
| put / long | 3.50 | **8.75** | 19.00 |
| call / short | 2.25 | **6.00** | 12.50 |
| call / long | 2.00 | **5.25** | 11.00 |

**Put vs call (Ryan's expectation).** **Yes, put-heavy levels overshoot more**, from both sides. Session-cluster bootstrap, 2,000 resamples:
- from below: p80 put − call **+3.75 [+2.00, +7.55]**; median **+1.62 [+0.75, +2.50]**;
- from above: p80 **+3.50 [+0.35, +6.00]**; median **+1.38 [+0.50, +2.25]**.

This describes the overshoot only, not whether a trade pays. Put-heavy levels carry larger magnitudes on average, and that is not controlled for.

Alternative reading (excursion until 0.10·IM back on the approach side), p80: put/short 23.6, put/long 17.3, call/short 10.1, call/long 8.7. Roughly 2× wider; not used.

### Q3 — counts (no outcomes)

Format: tests used / confirmed extremes in the zone / retests / filled / **kept after the gap rule**. "Tests used" is below Q1's total because a level stops after its first fill and episodes don't overlap.

| config (O pct, P, W, tol) | arm | put / short | put / long | call / short | call / long |
|---|---|---|---|---|---|
| **centre (80, 6, 90, 1)** | real | 69 / 37 / 31 / 31 / **29** | 77 / 31 / 22 / 22 / **21** | 211 / 96 / 69 / 69 / **66** | 166 / 68 / 48 / 47 / **46** |
| | placebo | 61 / 32 / 30 / 30 / **26** | 54 / 25 / 23 / 22 / **21** | 57 / 23 / 15 / 15 / **15** | 43 / 23 / 19 / 19 / **19** |
| fewest (60, 10, 30, 0.5) | real | 101 / 36 / 23 / 23 / **21** | 120 / 26 / 10 / 10 / **9** | 323 / 52 / 22 / 22 / **22** | 290 / 51 / 20 / 20 / **19** |
| | placebo | 91 / 27 / 19 / 19 / 17 | 102 / 17 / 11 / 11 / 11 | 85 / 13 / 6 / 5 / 5 | 65 / 10 / 7 / 7 / 7 |
| most (95, 4, 180, 2) | real | 46 / 35 / 33 / 33 / **30** | 57 / 39 / 31 / 31 / **29** | 168 / 105 / 85 / 85 / **84** | 112 / 68 / 60 / 58 / **58** |
| | placebo | 47 / 34 / 31 / 31 / 27 | 42 / 28 / 25 / 25 / 24 | 44 / 28 / 20 / 20 / 20 | 34 / 23 / 20 / 20 / 19 |

Centre episode outcomes (real, pre-entry): broken (beyond L + O_c) 21 / 25 / 60 / 54; extreme short of the level 11 / 20 / 52 / 43.

- **≥ 40-trade floor across the 81 configurations:** put/short **0 of 81** (max 31), put/long **0 of 81** (max 29), call/short 68 of 81 (max 84), call/long 52 of 81 (max 58).
  - **Both put-heavy cells are excluded from the exit choice everywhere on the grid.** At the centre both call cells clear 40 (66 and 46).
- **Placebo arm is thin.** The call-parent placebo cells hold 5–20 setups across the grid (centre 15 / 19). A per-cell real − placebo comparison in Stage 1 cannot say much for the call cells.
- **Shared fills.** Adjacent real levels (5–6 pts apart) can share one extreme and one fill. At the centre, 162 kept real setups map to 157 distinct fills (5 duplicates); placebo 81 → 75. The spec counts per level; the duplicates are kept and flagged.
- Fill rate after a retest: at the centre, 1 of 170 real retests did not fill (call/long) and 1 of 87 placebo retests (put/long). On 1-s data the retest and the 1-tick-through fill almost always happen together.

### Q4 — detector check (centre config; no outcomes)

Sessions drawn with `default_rng(20261025)`: 2026-01-26, 01-27, 01-28, 04-17, 06-05. Every test that reached confirmation is listed, plus the filled setups. Times PT; prices ES (H26 / M26). For longs, "extreme" is the low L and the zone runs downward. "Stop" is the L8 stop at the centre buffer (H ± 2).

| session | level | sign | mag_bn | setup | O_c | zone | test_t | test_px | extreme | extreme_t | confirm_t | confirm_px | status | retest_t | fill_t | entry | stop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-01-26 | 6956.0 | call | 493.6 | long | 5.25 | 6956.00–6950.75 | 06:30:17 | 6957.75 | 6952.0 | 06:30:45 | 06:31:06 | 6958.0 | noretest |  |  |  |  |
| 2026-01-26 | 6951.0 | put | 79.2 | long | 8.75 | 6951.00–6942.25 | 06:30:34 | 6953.0 | 6952.0 | 06:30:45 | 06:31:06 | 6958.0 | short |  |  |  |  |
| 2026-01-26 | 6971.0 | call | 63.3 | short | 6.0 | 6971.00–6977.00 | 06:35:40 | 6969.25 | 6976.0 | 06:41:09 | 06:42:09 | 6969.0 | filled | 06:43:12 | 06:44:05 | 6975.0 | 6978.0 |
| 2026-01-26 | 6976.0 | call | 369.0 | short | 6.0 | 6976.00–6982.00 | 06:41:03 | 6974.25 | 6976.0 | 06:41:09 | 06:42:09 | 6969.0 | filled | 06:43:12 | 06:44:05 | 6975.0 | 6978.0 |
| 2026-01-26 | 6986.0 | call | 59.3 | short | 6.0 | 6986.00–6992.00 | 07:19:42 | 6984.0 | 6988.0 | 07:23:48 | 07:27:23 | 6982.0 | filled | 07:53:02 | 07:53:29 | 6987.0 | 6990.0 |
| 2026-01-27 | 7001.0 | call | 170.0 | long | 5.25 | 7001.00–6995.75 | 07:05:04 | 7002.75 | 6998.0 | 07:05:51 | 07:09:17 | 7004.25 | noretest |  |  |  |  |
| 2026-01-27 | 7015.0 | call | 150.5 | short | 6.0 | 7015.00–7021.00 | 07:45:17 | 7013.25 | 7018.5 | 08:40:50 | 09:00:27 | 7012.5 | noretest |  |  |  |  |
| 2026-01-27 | 7010.0 | call | 162.7 | long | 5.25 | 7010.00–7004.75 | 09:08:50 | 7012.0 | 7006.5 | 09:19:12 | 09:25:49 | 7012.5 | filled | 09:56:37 | 09:56:58 | 7007.5 | 7004.5 |
| 2026-01-27 | 7001.0 | call | 170.0 | long | 5.25 | 7001.00–6995.75 | 10:02:11 | 7001.5 | 7001.5 | 10:02:11 | 10:13:20 | 7007.5 | short |  |  |  |  |
| 2026-01-27 | 7015.0 | call | 150.5 | short | 6.0 | 7015.00–7021.00 | 10:32:13 | 7013.25 | 7017.25 | 12:13:55 | 12:30:05 | 7011.25 | noretest |  |  |  |  |
| 2026-01-28 | 7021.0 | call | 212.8 | long | 5.25 | 7021.00–7015.75 | 06:36:04 | 7022.75 | 7021.5 | 06:37:47 | 06:42:12 | 7027.5 | short |  |  |  |  |
| 2026-01-28 | 7021.0 | call | 212.8 | long | 5.25 | 7021.00–7015.75 | 06:47:44 | 7023.0 | 7021.25 | 06:48:06 | 06:52:19 | 7027.25 | short |  |  |  |  |
| 2026-01-28 | 7026.0 | call | 368.0 | short | 6.0 | 7026.00–7032.00 | 07:02:37 | 7024.0 | 7025.75 | 07:02:48 | 07:07:15 | 7019.25 | short |  |  |  |  |
| 2026-01-28 | 7011.0 | call | 148.2 | long | 5.25 | 7011.00–7005.75 | 07:15:00 | 7012.5 | 7012.5 | 07:15:00 | 07:15:37 | 7018.5 | short |  |  |  |  |
| 2026-01-28 | 7021.0 | call | 212.8 | short | 6.0 | 7021.00–7027.00 | 07:15:41 | 7019.0 | 7023.25 | 07:24:29 | 07:28:14 | 7017.25 | noretest |  |  |  |  |
| 2026-01-28 | 7011.0 | call | 148.2 | long | 5.25 | 7011.00–7005.75 | 07:42:02 | 7013.0 | 7007.25 | 07:59:58 | 08:04:43 | 7013.25 | filled | 08:21:59 | 08:22:00 | 7008.25 | 7005.25 |
| 2026-01-28 | 7001.0 | call | 189.3 | long | 5.25 | 7001.00–6995.75 | 08:42:07 | 7002.75 | 6997.0 | 08:54:33 | 09:03:04 | 7003.25 | filled | 09:31:33 | 09:33:39 | 6998.0 | 6995.0 |
| 2026-04-17 | 7136.0 | call | 165.7 | short | 6.0 | 7136.00–7142.00 | 06:36:18 | 7134.0 | 7135.5 | 06:37:15 | 06:39:43 | 7129.25 | short |  |  |  |  |
| 2026-04-17 | 7136.0 | call | 165.7 | short | 6.0 | 7136.00–7142.00 | 06:46:02 | 7134.0 | 7134.5 | 06:46:04 | 06:46:47 | 7128.25 | short |  |  |  |  |
| 2026-04-17 | 7136.0 | call | 165.7 | long | 5.25 | 7136.00–7130.75 | 06:58:23 | 7137.0 | 7137.0 | 06:58:23 | 06:59:11 | 7143.25 | short |  |  |  |  |
| 2026-06-05 | 7510.0 | put | 137.4 | long | 8.75 | 7510.00–7501.25 | 06:59:06 | 7512.0 | 7505.75 | 07:01:28 | 07:02:12 | 7514.75 | filled | 08:15:08 | 08:15:09 | 7506.75 | 7503.75 |

Things to check against the charts:
- 01-26 06:44:05: the 6971 and 6976 call lines share one extreme (6976.00) and one fill. This is the shared-fill case.
- Many tests show "short": the extreme stayed below the level (for longs, above it). This happens when price reverses inside the 2-pt band.
- 04-17 has no filled setup; 06-05 has one put long, filled 74 min after confirmation.

**2026-10-23 illustration (holdout day; detector only, no outcome).** Inputs: `XCME:ES.Z26`, IM 83.89, P_target 7686.00 (A = 10-22), RTH open 7661.25. Top-6 levels: 7681, 7687, 7636, 7737, 7731, 7630. Ryan's "7686" line is the 1-pt bucket **7687** (put-heavy, mag 315 $bn, net −272 $bn). 7681 (put, 400 $bn) is also a top-6 level; 7697 (put, 97 $bn) is a chart line but not in the top 6.

- **The 1-s flow has an RTH hole on 10-23 from 07:52:46 to ~09:56 PT (7,408 s), covering the whole move.** 14,773 of ~23,400 RTH seconds are present. On the flow, the detector sees H = 7691.75 at 07:52:32 and "confirms" only across the hole:

| session | level | sign | mag_bn | setup | O_c | zone | test_t | test_px | extreme | extreme_t | confirm_t | confirm_px | status | retest_t | fill_t | entry | stop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-23 | 7687.0 | put | 315.0 | short | 9.75 | 7687.00–7696.75 | 07:46:06 | 7687.25 | 7691.75 | 07:52:32 | 09:56:14 | 7677.75 | noretest |  |  |  |  |

- **1-minute proxy** (`ironbeam_es_1m_bars`; each minute treated as one bar, t = minute start; illustration only):

| session | level | sign | mag_bn | setup | O_c | zone | test_t | test_px | extreme | extreme_t | confirm_t | confirm_px | status | retest_t | fill_t | entry | stop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-23 | 7687.0 | put | 315.0 | short | 9.75 | 7687.00–7696.75 | 07:46:00 | 7687.25 | 7693.5 | 07:56:00 | 08:02:00 | 7684.0 | filled | 08:25:00 | 08:25:00 | 7692.5 | 7695.5 |
| 2026-10-23 | 7681.0 | put | 400.2 | long | 8.75 | 7681.00–7672.25 | 08:03:00 | 7682.75 | 7679.75 | 08:10:00 | 08:13:00 | 7688.5 | filled | 09:23:00 | 09:23:00 | 7680.75 | 7677.75 |
| 2026-10-23 | 7697.0 | put | 97.2 | short | 9.75 | 7697.00–7706.75 | 08:26:00 | 7696.5 | 7697.25 | 08:34:00 | 08:37:00 | 7685.5 | noretest |  |  |  |  |

  At 7687 the detector takes the *first* local high: 7693.50 at 07:56, confirmed by the 08:02 dip to 7684 (≥ 8.39 = 0.10·IM). Its retest fills at 7692.50 at 08:25, the minute before the run to 7696.50–7697. That is the first setup on the level, so the episode ends there (L7: one setup per level). Ryan's trade (high ~7697 at 08:27, retest ~7694 at 08:34) is not detected at 7687.
- **Which clause binds.** Readout only: the same proxy with the confirmation pullback = P alone (6 pts, no 0.10·IM floor):

| session | level | sign | mag_bn | setup | O_c | zone | test_t | test_px | extreme | extreme_t | confirm_t | confirm_px | status | retest_t | fill_t | entry | stop |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-10-23 | 7681.0 | put | 400.2 | short | 9.75 | 7681.00–7690.75 | 07:44:00 | 7680.0 | 7687.25 | 07:46:00 | 07:47:00 | 7680.5 | filled | 07:50:00 | 07:50:00 | 7686.25 | 7689.25 |
| 2026-10-23 | 7687.0 | put | 315.0 | short | 9.75 | 7687.00–7696.75 | 07:46:00 | 7687.25 | 7687.25 | 07:46:00 | 07:47:00 | 7680.5 | filled | 07:50:00 | 07:50:00 | 7686.25 | 7689.25 |
| 2026-10-23 | 7697.0 | put | 97.2 | short | 9.75 | 7697.00–7706.75 | 08:26:00 | 7696.5 | 7697.0 | 08:27:00 | 08:29:00 | 7690.5 | filled | 08:34:00 | 08:34:00 | 7696.0 | 7699.0 |

  - Ryan's trade appears, but at the **7697** line (not top 6): H 7697.00 at 08:27, confirmed by the 08:29 dip to 7690.50 (6.50 pts: ≥ 6, < 8.39), filled 7696.00 at 08:34.
  - At 7687 the looser rule fires earlier still, at 07:50, and shares a fill with 7681.
  - So two clauses keep the detector from matching Ryan's read:
    1. The 0.10·IM floor on the confirmation pullback: the 08:27 → 08:29 pullback is 6.5 pts, below the 8.39 floor.
    2. One setup per level, first fill. At 7687 an earlier, smaller extreme is taken first.

  Ryan's own description ("overshot 9–11 pts, local high 7694–97") fits a zone of 9.75 at 7687. The detector, though, keys H to the first confirmed extreme, not the eventual one.

### Q5 — A/A calibration (pooled hold-to-close, placebo only) — PASS

- 81 placebo setups at the grid centre on 81 pseudo-levels; 0 dropped by the exit-window gap rule. 200 splits by pseudo-level, ~40 setups per half.

| | value |
|---|---|
| statistic (mean net pts, half 1 − half 2), median | **+0.05 pts** (gate ±3) |
| mean / sd | −0.07 / 1.05 |
| p5 / p25 / p75 / p95 | −1.71 / −1.29 / +0.47 / +1.76 |
| CI half-width, median | 2.17 pts |
| split sd ÷ (median half-width / 1.96) | 0.95 (≈ 1 if calibrated) |
| **share of splits with CI excluding 0** | **6.5 % (13 / 200)** (gate ≤ 10 %) |

The session-cluster bootstrap is calibrated for this statistic. At this n, though, each half is ~40 trades, and a ±2-pt half-width is the resolution of a real − placebo comparison on the full arms.

### Decisions needed (before Stage 1)

1. **Put cells cannot meet the ≥ 40-trade rule anywhere on the grid** (max 31 / 29 on 88 sessions). Options:
   - (a) Run Stage 1's choice on the call cells and the pooled choice only; report the put cells descriptively.
   - (b) Pool put-heavy short + long (≈ 50 at the centre) as one "put" cell for the choice.
   - (c) Lower the floor for put cells.
   - (d) Defer the put question to Stage 2's larger set R, after CR-CA.
2. **O_c reading.** Confirm "0.10·IM back" = a pullback from the running extreme (used here) rather than a return to 0.10·IM on the approach side of L (≈ 2× wider zones).
3. **Detector vs Ryan's read on 10-23.** On the 1-minute proxy the detector takes the first confirmed extreme and stops at one setup per level; with the 0.10·IM floor it misses the 08:27 high. Options:
   - (a) Keep as specified. P = 4 is then always replaced by the 0.10·IM floor (IM median ≈ 68, so ≈ 7 pts), and P = 6 is too on most days, so the P dimension is partly inert.
   - (b) Drop the floor, so P is the pullback.
   - (c) Allow a later extreme in the zone to replace H while no fill has happened.

   Any of these needs an amendment before Stage 1.
4. **Shared fills on adjacent levels** (5 of 162 at the centre). Options: keep them (per level, as written), or de-duplicate to one trade per fill.
5. **Placebo thinness.** 81 placebo setups at the centre (15–26 per cell). T2-style comparisons per cell will be wide. The ≥ 15-pt separation is the main cost (283 pseudo-levels vs CR-BW's 427 at 10 pts).
6. **2026-10-23 flow hole** (07:52:46 → 09:56 PT). This is a holdout-period feed gap in `ironbeam_es_flow_1s`. It matters for any future holdout read of this pattern and is not checked further here.

---

## Step 0 under A1–A14 (Stage 1, Databento flow, 2026-10-25)

**Script and logs.** `scripts/cr_cb_step0_v2.py` (sha256 `954c7dee…62bb522`). Logs (git-ignored) in `scripts/logs/cr_cb/`: `step0_v2.log` and `step0_v2.json`; run 1 is kept as `step0_v2_run1.*`. Gallery: `research/gallery/cr-cb-stage1.json`.

**Read-only setup.** `get_backfill_db_conn()` (role asserted) with `default_transaction_read_only = on` (asserted); `statement_timeout = 600s`; SELECTs only.
- Flow is `es_flow_1s_databento` (A9.1); the Ironbeam fallback was used on 0 sessions.
- Train reads stop at `ts_utc` < 2026-06-05 21:00 UTC. No 2025 data is read: R-b, A10.
- 2026-10-23 is read in one block for the A9.7 illustration: flow, `orats_oi_gamma`, IM, P_target, 0DTE skew, and its own overnight bars. Its prior session (10-22) is not read.

**Every open choice is fixed in the docstring before the run.** That covers the range bottom PL, edge events, re-arm mechanics, feature windows, skew reads, cluster and zone construction, T0 metrics, A/A and gallery sampling. The main ones are below.

**No trade outcome is reported.** A stop is used internally in two places, both required by the spec:
- the A9.5′ re-arm needs to know whether a fade was stopped, so train counts are for **attempt 1 only**; attempts 2–3 appear only in the gallery (attempt number in meta, per A9.9) and on 10-23;
- the Q5 A/A computes placebo hold-to-close P&L inside the splits, and only the split statistic is reported.

**Runs.**
- **Run 1:** all of T0, Q1–Q5 and the additions, plus the gallery.
- **A14** arrived during run 1. It was committed to the spec after T0 had been computed, and the A14 block was added for run 2.
- **Run 2:** T0, Q1, Q2, Q3, the additions and Q5 are **byte-identical** to run 1. Q4 and the 10-23 tables gain a shared-fill column and the reference-level list. The detector itself does not use A14 zones in Step 0 (counts only, A14.4).

### Fixed choices that matter for reading the numbers

- **Centre:** O_c 80th, P 6, W 90 min, tol 1, attempts 2, **d 5**, e 4 (A14 counts only), stop S-x 2.
  - Counts depend on (O, P, W, tol, d) for attempt 1: 243 configurations.
- **Range (A4.1).** Top = H.
  - The bottom PL is the running low after H, fixed at the first bar that rebounds ≥ 2 pts off it (A5.1's push separation).
  - The range is invalid beyond H + 2 or PL − 2 (S-x 2), or at window end.
  - Top / bottom edge event = first touch within tol while the range is valid.
- **The L7 fade fill does not need a valid range.** A retest can fill after the range was invalidated at PL − 2, so filled entries exceed top-edge events.
- **A9.5′ re-arm:** only a stopped top fade inside the zone re-arms. A level's later tests are used only until a top fade on it fills.
- **Shared fills (A9.6 / A9.8):** de-duplicated by (session, direction, fill second), credited to the higher-magnitude arming level.
- **Scales (σ5_d, B_d, skew β_d).** Taken from the prior 5–20 usable **train** sessions.
  - The first 5 sessions (01-06 → 01-12) have none, so their A_rel, RA, residual skew and the 3·σ5 leg are NA.
- **Skew.** 0DTE PM, the latest snapshot ≤ t (within 5 min).
  - Edge touches before 07:00 PT have NA skew.
  - The move in = from the opposite extreme within 60 min to the test.
- **A14 references:**
  - prior-session RTH H/L/C from Databento;
  - overnight H/L from `ironbeam_es_1m_bars`, 15:00 PT prior → 06:30 PT (Databento holds RTH only);
  - swings by zigzag, threshold max(4, 0.05·IM), known from their confirmation time.

  Zones chain refs known at time t to the d = 5 cluster within e.

### T0 — do levels slow price down? (A7.1; price paths only; real − placebo, session-cluster 95 % CI)

d = 0 (single levels): real 1,687 tests, placebo 627.

| cell | n real / plc | (a) range formed: real / plc, diff | (b) dwell s within ±2 pts, 30 min: real / plc, diff | (c) log speed ratio after/before: diff |
|---|---|---|---|---|
| put, from below | 231 / 170 | 64.5 % / 53.5 %, **+11.0 [+1.7, +21.0]** | 326 / 312, +13 [−60, +91] | **−0.066 [−0.136, −0.004]** |
| put, from above | 252 / 193 | 57.9 % / 53.9 %, +4.1 [−6.6, +14.9] | 299 / 302, −3 [−68, +71] | +0.040 [−0.049, +0.119] |
| call, from below | 630 / 148 | 45.9 % / 46.0 %, −0.1 [−8.1, +8.0] | 451 / 433, +18 [−111, +126] | +0.084 [+0.024, +0.153] |
| call, from above | 574 / 116 | 44.4 % / 48.3 %, −3.9 [−14.2, +6.5] | 425 / 408, +17 [−110, +119] | +0.065 [−0.005, +0.121] |
| all | 1,687 / 627 | 49.7 % / 50.9 %, −1.1 [−5.9, +4.2] | 402 / 355, +47 [−16, +103] | +0.037 [+0.000, +0.071] |

At d = 5 (real 1,455 / placebo 389), the same pattern with wider CIs:
- put from below: formed +9.3 [−1.3, +20.3], speed −0.036 [−0.121, +0.041];
- call from above: formed −10.4 [−22.7, +2.6];
- all: formed −2.4 [−8.1, +3.6], dwell +46 [−27, +109].

**Reading (descriptive, not the Stage 2 gate):**
- Pooled, real levels do not form ranges more often than placebo prices, and dwell is not reliably longer.
- **Put-heavy levels approached from below** are the one cell where both (a) and (c) favour "slows down" with a CI clear of 0 at d = 0.
- At call-heavy levels approached from below, price speeds up *more* after the test than at placebo (+0.084).

### Q1 — universe

| rule | removed | remaining |
|---|---|---|
| trading sessions 2026-01-06 → 2026-06-05 | | 105 |
| L14a roll week (03-09 → 03-20) | 10 | 95 |
| L14b no ES 15:45 ET bar on A (02-18, 02-19) | 2 | 93 |
| L14c roll mismatch | 0 | 93 |
| no flow rows | 0 | 93 |
| no canonical `bt_daily_features` row (01-20, 02-17, 04-06, 05-26) | 4 | **89** |

- **89 sessions.** 2026-03-03, which had no Ironbeam flow, is now in. All 89 are on Databento (`XCME:ES.H26` / `ES.M26`, the session's largest-volume symbol).
- **Levels:** 528 (put 172, call 356), median 6 per session. Placebo pseudo-levels: 292.
- **Scales (Databento):**
  - σ5_d median 2.50 pts (p10 2.0, p90 3.5);
  - B_d median 3.60 pts / 1k contracts (CR-BW's Ironbeam B_d was 4.14: the 1.4× delta scale);
  - skew β_d (ΔPS on Δlog spot) median −2.25;
  - 5 sessions without scales (01-06 → 01-12).
- 0DTE skew present on 89 / 89 sessions.

### Q2 — O_c per cell (A9.3 reading; real armed tests at arming levels; frozen for Stage 1)

| d | cell | n | share 0 | p50 | **p60** | **p80** | **p95** |
|---|---|---|---|---|---|---|---|
| 0 | put / below | 231 | 23.8 % | 2.75 | 5.00 | 9.75 | 22.00 |
| 0 | put / above | 252 | 27.8 % | 2.25 | 3.50 | 8.25 | 18.75 |
| 0 | call / below | 630 | 35.6 % | 1.25 | 2.75 | 6.25 | 13.75 |
| 0 | call / above | 574 | 39.4 % | 1.25 | 2.00 | 5.25 | 11.50 |
| 5 | put / below | 205 | 22.9 % | 3.25 | 5.00 | 9.75 | 21.75 |
| 5 | put / above | 225 | 26.2 % | 2.25 | 3.50 | 8.25 | 19.50 |
| 5 | call / below | 559 | 36.9 % | 1.00 | 2.00 | 6.00 | 14.00 |
| 5 | call / above | 466 | 41.6 % | 1.25 | 2.25 | 5.50 | 11.75 |
| 10 | put / below | 156 | 20.5 % | 3.25 | 6.00 | 11.75 | 24.00 |
| 10 | put / above | 170 | 25.3 % | 2.50 | 3.50 | 9.25 | 19.75 |
| 10 | call / below | 371 | 35.6 % | 1.00 | 2.00 | 6.00 | 14.50 |
| 10 | call / above | 338 | 38.2 % | 1.25 | 2.25 | 6.75 | 15.50 |

- Versus the Ironbeam Step 0 at d = 0: p80 within 0.5 pt (put / above 8.25 vs 8.75, call / below 6.25 vs 6.00); p95 up to 1.25 pts higher (call / below 13.75 vs 12.50).
- **Put-heavy levels still overshoot more** (d = 0). From below: p80 +3.50 [+1.55, +7.25], median +1.50 [+0.75, +2.75]. From above: p80 +3.00 [+0.25, +5.75], median +1.00 [+0.50, +2.12].

### Q3 — counts (attempt 1, no outcomes)

Centre (80, 6, 90, 1), filled L7 entries after the gap rule and shared-fill de-dup. Each cell shows real / placebo.

| d | put / short | put / long | call / short | call / long |
|---|---|---|---|---|
| 0 | 51 / 42 | 41 / 41 | 95 / 29 | 68 / 26 |
| **5** | **46 / 27** | **36 / 27** | **81 / 15** | **56 / 16** |
| 10 | 36 / 27 | 31 / 22 | 57 / 14 | 46 / 14 |

Breakdown at d = 5, real, by cell (put/short, put/long, call/short, call/long):

| | put / short | put / long | call / short | call / long |
|---|---|---|---|---|
| episodes | 86 | 96 | 224 | 202 |
| confirmed extremes in the zone | 55 | 49 | 105 | 80 |
| top-edge events (valid range) | 31 | 21 | 47 | 29 |
| bottom-edge events | 46 | 43 | 96 | 77 |

- **≥ 40 kept, of 81 configurations per d** (put/short, put/long, call/short, call/long):
  - d 0: 66, 38, 72, 67;
  - d 5: 61, 26, 70, 52;
  - d 10: 27, 9, 57, 46.
- Max per cell: d 0 62 / 56 / 129 / 109; d 5 56 / 51 / 111 / 92; d 10 45 / 41 / 77 / 65.
- **Pooled within level type (A9.2), ≥ 40:** put 81 / 80 / 75 and call 81 / 79 / 73 of 81 (d 0 / 5 / 10). With pooling, every cell type is choosable almost everywhere.
- **Corners:** fewest = d 10, (60, 10, 30, 0.5): 20 / 10 / 20 / 10. Most = d 0, (95, 4, 180, 2): 62 / 56 / 129 / 109.
- **Shared fills:** 2 + 2 duplicates at d 0 (put/short, call/short), 0 at d 5 and 10.
- Compared with the Ironbeam Step 0, counts rose, mainly because the 0.10·IM floor was dropped (A9.4).
- **The placebo arm is thin at d = 5:** 15–27 per cell.

### A3.3 — resolution of confirmed ranges (attempt 1, centre)

| cell | confirmed | F first | K first | neither |
|---|---|---|---|---|
| put / below | 55 | 46 | **0** | 9 |
| put / above | 49 | 36 | **0** | 13 |
| call / below | 105 | 81 | **0** | 24 |
| call / above | 80 | 56 | **0** | 24 |

**The break never comes first.** The fade limit (H − tol) sits inside the break trigger (H + 1), so on continuous 1-s data price always fills the fade first. "Which resolution" can only be read after the fade's outcome, so the resolution share is degenerate as defined.

### A4.8 / A5.5 — edge events, flow features and rule decisions (centre, d 5, attempt 1, kept)

**Edge events (real / placebo).** "top" is the level-side extreme (the L7 retest edge); "bot" is PL.

| | from below, top | from below, bot | from above, top | from above, bot |
|---|---|---|---|---|
| put | 31 / 16 | 46 / 27 | 21 / 10 | 43 / 30 |
| call | 47 / 7 | 96 / 20 | 29 / 9 | 77 / 14 |

Real total 390.

**Feature distributions, pooled real** (p10 / p25 / p50 / p75 / p90):

| feature | p10 | p25 | p50 | p75 | p90 | n |
|---|---|---|---|---|---|---|
| P (contracts, 120 s) | −764 | −216 | +115 | +428 | +809 | 390 |
| A_rel | 0.14 | 1.03 | 1.96 | 3.49 | 7.76 | 228 |
| Q | −0.41 | 0.02 | 0.56 | 1.06 | 1.79 | 279 |
| PD | −3.06 | −1.78 | −0.65 | +0.12 | +1.23 | 152 |
| LE | −0.29 | −0.12 | +0.02 | +0.11 | +0.20 | 378 |
| RA | −3.70 | −0.69 | +0.89 | +3.09 | +6.63 | 247 |
| DV | −0.054 | −0.014 | +0.032 | +0.081 | +0.148 | 390 |
| SS share | 0.00 | 0.00 | 0.00 | 0.11 | 0.64 | 390 |
| SS VWAP − level (pts) | −5.57 | −4.41 | −2.90 | −1.09 | +0.42 | 390 |

- **NA counts.** A_rel 162 (P ≤ 0 or no B_d), Q 111 (P_set ≤ 0), PD 238, RA 143.
  - PD needs ≥ 2 pushes. Push counts: 1 push 205, 2 pushes 162, 3 pushes 19, 4 pushes 4.
  - Q is unstable when P_set is small (single values beyond ±60 occur).
- **Ranges are short:** range time (confirmation → edge touch) median **60.5 s** (p25 22, p75 147, p90 343); **1 % last ≥ 30 min**. The first touch of each boundary comes quickly. The 30-minute ranges Ryan described (A5) are rare under A4.2's first-touch rule.
- **Rule decisions, pooled real** (fade / break / none):

  | rule | fade | break | none |
  |---|---|---|---|
  | FR1 | 78 | 76 | 236 |
  | FR2 | 163 | 53 | 174 |
  | FR3 | 149 | 130 | 111 |
  | FR4 | 27 | 13 | 350 |
  | FR5 | 32 | 0 | 358 |
  | FR6 | 62 | 39 | 289 |
  | FR7 | 107 | 29 | 254 |
  | FR8 | 128 | 17 | 245 |
  | FR10 | 31 | — | 359 |
  | FR11 | 16 | — | 374 |
  | FR12 | 26 | 28 | 336 |
  | stall side (A7.2) | 325 | 45 | 20 |

  The per-cell tables are in the JSON. FR9 inputs are reported but not fitted (A9.8).
  - The stall-side rule says "fade" on 83 %: price usually stalls on the near side, and the SS share median is 0.

### A6.6 / A8.8 — skew, divergence, named confluences (centre, d 5)

- **Skew coverage:** 288 / 390 edge events have raw PS and CS (all 102 NA are touches before 07:00 PT); 273 have the residual.
  - **Range trend: only 86 / 390** (≥ 3 snapshots needs a range ≥ 3 min). FR12 and A8.3 are thin.
- **Raw put-skew divergence on the move in, by level type and real edge** (bull / bear / none / NA):
  - put, top edge: 12 / 10 / 30 / 22;
  - put, bottom edge: 6 / 16 / 22 / 23;
  - call, top edge: 24 / 10 / 62 / 28;
  - call, bottom edge: 5 / 23 / 68 / 29.

  Bullish divergence needs a decline into the level, so it occurs only in from-above cells; bearish only in from-below cells.
- **H-PNB** (put, from above, level-side edge; 21 cell events):

  | condition | events |
  |---|---|
  | (a) big move | 21 |
  | (b) FR5 or FR6 fade | 5 |
  | (c) bullish divergence | 6 |
  | (a)+(b) | 5 |
  | (a)+(c) | 6 |
  | (b)+(c) | 3 |
  | **all three** | **3** |

  Skew NA on 10. Placebo: 1 of 10 cell events.
- **H-PSD** (call, from below, level-side edge; 47 cell events):

  | condition | events |
  |---|---|
  | (a) big move | 35 |
  | (b) FR5 or FR6 fade | 17 |
  | (c) bearish divergence | 10 |
  | (a)+(b) | 13 |
  | (a)+(c) | 10 |
  | (b)+(c) | 5 |
  | **all three** | **5** |

  Skew NA 12, σ5 NA 4. Placebo: 1 of 7.
- **H-CNB readout:** 4 events.
- **Leg (a) barely discriminates.** 3·σ5_d (median 7.5 pts) within 15 min is met by almost every test (|move| / σ5 median 5.3). The named rules are carried by (b) and (c).
- At ~3–5 events in 2026, T5 / T5b (≥ 40 in R-a) is likely **inconclusive (underpowered)** unless R-a is much denser.

### A13.5 — clusters

| d | clusters | multi-member | members 1 / 2 / 3 / 4+ | span median / p90 / max | types put / call / mixed | reached far after near |
|---|---|---|---|---|---|---|
| 0 | 528 | 0 | 528 / 0 / 0 / 0 | 0 / 0 / 0 | 172 / 356 / 0 | — |
| 5 | 461 | 58 | 403 / 49 / 9 / 0 | 5 / 10 / 10 | 158 / 303 / 8 | **86 / 207 = 41.5 %** (below 44/88 = 50.0 %, above 42/119 = 35.3 %) |
| 10 | 348 | 120 | 228 / 82 / 26 / 12 | 10 / 20 / 33 | 134 / 214 / 21 | **117 / 408 = 28.7 %** (below 65/207 = 31.4 %, above 52/201 = 25.9 %) |

- "Reached" = a trade within 2 pts of the far member after a test of the near member, before price is 0.10·IM back. No placebo comparison is defined in A13.
- With 5-pt chart spacing, d = 5 joins only levels exactly 5 pts apart.

### A14.4 — reference levels and confluence zones (counts only)

- **Refs per session:** prior high, low and close 1 each (89 / 89); overnight high and low 1 each (89 / 89).
- **Confirmed swings: median 45 highs and 45 lows per session** (p10 21, p90 ≈ 99). The A14.1 threshold max(4, 0.05·IM) ≈ 4 pts on almost every day, which is about 1.6·σ5_d. That is intraday noise, not structure.

| e | zones at 13:00 | GEX clusters merged into another zone | span median / p75 / p90 (pts) | GEX-only zones | tests whose zone has a ref | **far edge is a ref** |
|---|---|---|---|---|---|---|
| 0 | 461 | 0 | 0 / 0 / 5 | 388 | 397 / 1,455 | **173 (11.9 %)** |
| 4 | 342 | 119 | 1.75 / 30.75 / 52.5 | 162 | 1,365 / 1,455 | **1,025 (70.4 %)** |
| 8 | 254 | 207 | 5.0 / 58.2 / 82.8 | 113 | 1,424 / 1,455 | **839 (57.7 %)** |

- Far-edge ref types at e = 4: swing high 407, swing low 379, overnight low 58, overnight high 51, prior close 49, prior low 42, prior high 39.
- Zone types at e = 4: GEX only 162; GEX + swing + prior + overnight 80; GEX + swing 31; GEX + prior 25; GEX + overnight 14; GEX + swing + overnight 11; GEX + swing + prior 11; GEX + prior + overnight 8.
- **A14 as written saturates at e ≥ 4.** With ~90 swings a day, chaining within 4 pts links most of the session's traded range into one zone (p90 span 52 pts), so the "far edge" is usually a swing far from the GEX level. This is counted, not fixed. See the decisions below.

### Q4 — detector table (5 random train sessions, seed 20261025; centre, d 5, attempt 1; no outcomes)

Sessions: 2026-01-26, 01-27, 01-29, 04-17, 06-05. Times PT; ES prices. For long setups the extreme is the low and PL is the range's upper boundary. Rule codes: F = fade, B = break; only rules that fire are shown. Full flow features per event are in the JSON and the gallery.

| session | arm_level | members | type | approach | O_c | test_t | status | extreme | extreme_t | confirm_t | PL | top_t | top_rules | bot_t | bot_rules | fade_fill_t | entry | stop_Sx2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2026-01-26 | 6956.0 | 6946/6951/6956 | put | below | 9.75 | 06:30:50 | broken |  |  |  |  |  |  |  |  |  |  |  |
| 2026-01-26 | 6976.0 | 6971/6976 | call | below | 6.0 | 06:41:03 | filled | 6976.0 | 06:41:09 | 06:42:09 | 6968.75 | 06:43:12 | FR2:F FR3:F FR7:F FR8:F STALL:F |  |  | 06:44:05 | 6975.0 | 6978.0 |
| 2026-01-26 | 6971.0 | 6971/6976 | call | above | 5.5 | 06:42:09 | nofill | 6968.75 | 06:42:10 | 06:43:12 | 6977.25 |  |  | 06:45:22 | FR2:B FR3:B FR6:B STALL:F |  |  |  |
| 2026-01-26 | 6986.0 | 6986 | call | below | 6.0 | 07:19:42 | filled | 6988.0 | 07:23:48 | 07:27:23 | 6980.75 |  |  | 07:28:22 | FR2:F FR3:B STALL:F | 07:53:29 | 6987.0 | 6990.0 |
| 2026-01-26 | 6986.0 | 6986 | call | above | 5.5 | 09:10:03 | nofill | 6982.75 | 10:09:41 | 10:20:29 | 6991.0 |  |  | 10:27:41 | FR2:F FR3:F STALL:F |  |  |  |
| 2026-01-26 | 6986.0 | 6986 | call | above | 5.5 | 12:00:30 | broken |  |  |  |  |  |  |  |  |  |  |  |
| 2026-01-27 | 7001.0 | 7001 | call | below | 6.0 | 06:50:15 | broken |  |  |  |  |  |  |  |  |  |  |  |
| 2026-01-27 | 7001.0 | 7001 | call | above | 5.5 | 07:05:04 | nofill | 6998.0 | 07:05:51 | 07:09:17 | 7006.75 |  |  | 07:13:35 | FR1:B FR2:F FR12:F STALL:F |  |  |  |
| 2026-01-27 | 7015.0 | 7010/7015 | call | below | 6.0 | 07:45:17 | nofill | 7018.5 | 08:40:50 | 09:00:27 | 7012.5 |  |  | 09:02:18 | FR2:F FR3:F FR7:F FR8:F STALL:F |  |  |  |
| 2026-01-27 | 7010.0 | 7010/7015 | call | above | 5.5 | 07:50:20 | short | 7010.5 | 07:57:34 | 08:18:59 |  |  |  |  |  |  |  |  |
| 2026-01-27 | 7010.0 | 7010/7015 | call | above | 5.5 | 09:08:50 | filled | 7006.5 | 09:19:12 | 09:25:49 | 7014.25 | 09:56:37 | FR2:F FR3:F FR6:B FR10:F | 09:28:21 | FR2:F FR3:F FR12:F STALL:F | 09:56:58 | 7007.5 | 7004.5 |
| 2026-01-27 | 7001.0 | 7001 | call | above | 5.5 | 10:02:11 | short | 7001.5 | 10:02:11 | 10:13:20 |  |  |  |  |  |  |  |  |
| 2026-01-27 | 7015.0 | 7010/7015 | call | below | 6.0 | 10:32:13 | nofill | 7017.25 | 12:13:55 | 12:30:05 | 7011.25 |  |  | 12:33:47 | FR2:F FR3:F FR7:F FR8:F FR12:B STALL:F |  |  |  |
| 2026-01-29 | 7001.0 | 7001 | call | above | 5.5 | 06:36:41 | filled | 6999.25 | 06:37:36 | 06:37:45 | 7007.0 | 06:41:24 | FR2:B FR8:B STALL:F | 06:39:07 | FR3:F FR7:F FR8:F STALL:F | 06:41:24 | 7000.25 | 6997.25 |
| 2026-01-29 | 6976.0 | 6976 | call | above | 5.5 | 06:45:16 | filled | 6973.0 | 06:47:06 | 06:47:41 | 6986.25 |  |  | 06:48:24 | FR1:B FR2:F STALL:F | 06:53:08 | 6974.0 | 6971.0 |
| 2026-01-29 | 6956.0 | 6956 | put | above | 8.25 | 07:00:34 | broken |  |  |  |  |  |  |  |  |  |  |  |
| 2026-01-29 | 6956.0 | 6956 | put | below | 9.75 | 07:04:57 | filled | 6959.25 | 07:05:33 | 07:05:49 | 6952.5 |  |  | 07:06:14 | FR3:F FR7:F FR8:F STALL:F | 07:11:14 | 6958.25 | 6961.25 |
| 2026-01-29 | 6956.0 | 6956 | put | above | 8.25 | 09:26:28 | short | 6957.5 | 09:26:51 | 09:28:05 |  |  |  |  |  |  |  |  |
| 2026-01-29 | 6956.0 | 6956 | put | above | 8.25 | 09:41:39 | filled | 6952.25 | 09:43:31 | 09:44:03 | 6959.0 | 09:45:50 | FR1:B FR2:F FR5:F FR6:B FR10:F FR11:F STALL:F | 09:44:55 | FR3:F FR7:F FR8:F STALL:F | 09:45:51 | 6953.25 | 6950.25 |
| 2026-01-29 | 6976.0 | 6976 | call | below | 6.0 | 12:38:17 | broken |  |  |  |  |  |  |  |  |  |  |  |
| 2026-01-29 | 7001.0 | 7001 | call | below | 6.0 | 12:58:41 | short | 6999.0 | 12:58:41 | 12:59:33 |  |  |  |  |  |  |  |  |
| 2026-04-17 | 7136.0 | 7136 | call | below | 6.0 | 06:36:18 | short | 7135.5 | 06:37:15 | 06:39:42 |  |  |  |  |  |  |  |  |
| 2026-04-17 | 7136.0 | 7136 | call | below | 6.0 | 06:46:02 | short | 7134.5 | 06:46:04 | 06:46:47 |  |  |  |  |  |  |  |  |
| 2026-04-17 | 7136.0 | 7136 | call | below | 6.0 | 06:50:34 | broken |  |  |  |  |  |  |  |  |  |  |  |
| 2026-04-17 | 7136.0 | 7136 | call | above | 5.5 | 06:58:23 | short | 7137.0 | 06:58:23 | 06:59:11 |  |  |  |  |  |  |  |  |
| 2026-06-05 | 7510.0 | 7510 | put | above | 8.25 | 06:59:06 | filled | 7505.75 | 07:01:28 | 07:02:09 | 7514.75 |  |  | 07:02:13 | FR3:F FR6:B STALL:F | 08:15:09 | 7506.75 | 7503.75 |
| 2026-06-05 | 7510.0 | 7510 | put | below | 9.75 | 08:23:29 | short | 7508.0 | 08:23:29 | 08:24:27 |  |  |  |  |  |  |  |  |

### 2026-10-23 illustration (Databento; detector only; all attempts; no outcome)

Inputs: `XCME:ES.Z26`, IM 83.89, P_target 7686.00, open 7661.25, top 6 levels 7681 / 7687 / 7636 / 7737 / 7731 / 7630. **The Databento flow has no RTH gap > 15 s** (21,966 rows); the Ironbeam 07:52 → 09:56 hole is absent. Overnight high / low 7714.25 / 7660.00; prior-session refs not read.

Attempt ≥ 2 rows exist only because the previous attempt's fade was stopped (A9.5′). This is the one place in Step 0 where that is visible, as Ryan asked.

**d = 5** (7681 and 7687 are 6 pts apart, so they stay separate):

| arm_level | members | approach | O_c | att | test_t | status | extreme | extreme_t | confirm_t | PL | top_t | top_rules | fade_fill_t | entry | stop_Sx2 | shared_fill_dup |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 7681.0 | 7681 | below | 9.75 | 1 | 07:44:49 | filled | 7687.25 | 07:46:06 | 07:46:09 | 7679.75 | 07:50:05 | FR2:F FR3:B FR7:B FR12:B STALL:B | 07:50:05 | 7686.25 | 7689.25 | False |
| 7681.0 | 7681 | below | 9.75 | 2 | 07:44:49 | broken |  |  |  |  |  |  |  |  |  |  |
| 7687.0 | 7687 | below | 9.75 | 1 | 07:46:06 | filled | 7687.25 | 07:46:06 | 07:46:09 | 7679.75 | 07:50:05 | FR2:F FR3:B FR12:B STALL:F | 07:50:05 | 7686.25 | 7689.25 | True |
| 7687.0 | 7687 | below | 9.75 | 2 | 07:46:06 | filled | 7693.5 | 07:56:02 | 07:57:47 | 7686.75 | 07:59:57 | FR2:B FR3:B FR7:B FR8:B STALL:B | 08:25:38 | 7692.5 | 7695.5 | False |
| 7687.0 | 7687 | below | 9.75 | 3 | 07:46:06 | broken |  |  |  |  |  |  |  |  |  |  |
| 7681.0 | 7681 | above | 8.25 | 1 | 08:03:21 | short | 7682.5 | 08:04:01 | 08:06:28 |  |  |  |  |  |  |  |
| 7681.0 | 7681 | above | 8.25 | 1 | 08:09:48 | filled | 7679.75 | 08:10:06 | 08:11:39 | 7688.0 |  |  | 09:23:45 | 7680.75 | 7677.75 | False |
| 7681.0 | 7681 | above | 8.25 | 2 | 08:09:48 | filled | 7676.25 | 09:26:05 | 09:30:45 | 7683.0 |  |  | 09:53:00 | 7677.25 | 7674.25 | False |
| 7681.0 | 7681 | above | 8.25 | 3 | 08:09:48 | broken |  |  |  |  |  |  |  |  |  |  |
| 7687.0 | 7687 | above | 8.25 | 1 | 08:37:03 | filled | 7681.25 | 08:46:12 | 08:47:13 | 7689.0 |  |  | 09:21:07 | 7682.25 | 7679.25 | False |
| 7687.0 | 7687 | above | 8.25 | 2 | 08:37:03 | broken |  |  |  |  |  |  |  |  |  |  |

**d = 10** (7681 and 7687 form one cluster; from below it arms at 7687, from above at 7681):

| arm_level | members | approach | O_c | att | test_t | status | extreme | extreme_t | confirm_t | PL | top_t | top_rules | fade_fill_t | entry | stop_Sx2 | shared_fill_dup |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 7687.0 | 7681/7687 | below | 11.75 | 1 | 07:46:06 | filled | 7687.25 | 07:46:06 | 07:46:09 | 7679.75 | 07:50:05 | FR2:F FR3:B FR12:B STALL:F | 07:50:05 | 7686.25 | 7689.25 | False |
| 7687.0 | 7681/7687 | below | 11.75 | 2 | 07:46:06 | filled | 7693.5 | 07:56:02 | 07:57:47 | 7686.75 | 07:59:57 | FR2:B FR3:B FR7:B FR8:B STALL:B | 08:25:38 | 7692.5 | 7695.5 | False |
| 7687.0 | 7681/7687 | below | 11.75 | 3 | 07:46:06 | filled | 7697.0 | 08:27:00 | 08:29:08 | 7690.5 | 08:34:33 | FR2:F FR3:B FR8:F FR12:B STALL:B | 08:34:37 | 7696.0 | 7699.0 | False |
| 7681.0 | 7681/7687 | above | 9.25 | 1 | 08:03:21 | short | 7682.5 | 08:04:01 | 08:06:28 |  |  |  |  |  |  |  |
| 7681.0 | 7681/7687 | above | 9.25 | 1 | 08:09:48 | filled | 7679.75 | 08:10:06 | 08:11:39 | 7688.0 |  |  | 09:23:45 | 7680.75 | 7677.75 | False |
| 7681.0 | 7681/7687 | above | 9.25 | 2 | 08:09:48 | filled | 7676.25 | 09:26:05 | 09:30:45 | 7683.0 |  |  | 09:53:00 | 7677.25 | 7674.25 | False |
| 7681.0 | 7681/7687 | above | 9.25 | 3 | 08:09:48 | broken |  |  |  |  |  |  |  |  |  |  |

- **At d = 10, attempt 3 at 7687 is Ryan's trade:**
  - H 7697.00 at 08:27:00, confirmed by the 6-pt pullback at 08:29:08;
  - the retest fills at 7696.00 at 08:34:37, with the S-x 2 stop at 7699.00.

  It is reached through the two earlier attempts (fills 07:50:05 at 7686.25 and 08:25:38 at 7692.50).
- **At d = 5 the same attempt is "broken":** 7697.00 > 7687 + O_c 9.75 = 7696.75, by one tick.
- At d = 5, 7681 and 7687 share the 07:50:05 fill. It is credited to 7681 (400 vs 315 $bn); the 7687 row shows `shared_fill_dup`.

### Q5 — A/A calibration (placebo top-fade trades, pooled hold-to-close, centre d 5, attempts ≤ 2) — PASS

| | value |
|---|---|
| placebo trades / pseudo-level clusters | 109 / 56 |
| statistic (mean net pts, half 1 − half 2), median | **+0.26** (gate ±3) |
| mean / sd | −0.01 / 2.26 |
| p5 / p25 / p75 / p95 | −3.89 / −1.81 / +1.89 / +3.11 |
| CI half-width, median | 4.64 pts |
| split sd ÷ (median half-width / 1.96) | 0.96 |
| **share of splits with CI excluding 0** | **8.5 % (17 / 200)** (gate ≤ 10 %) |

The bootstrap is calibrated. At ~55 placebo trades per half, the resolution of a per-cell real − placebo comparison is ±4–5 pts.

### Gallery (A9.9) — `research/gallery/cr-cb-stage1.json`

- **64 events,** CR-CC L10 format, `outcome` null on all; all L10 fields present; 215 KB.
  - 40 random edge events, 5 per edge cell (seed 20261024);
  - 7 tagged named events (H-PSD 5, one of them an attempt-2 event; H-PNB 2). Two more named events (attempt 1) were already drawn into the random 40, so all 3 attempt-1 H-PNB and all 5 attempt-1 H-PSD events are in the gallery;
  - 17 from 2026-10-23: d 5: 9, d 10: 8.
- **Attempts:** 1: 47, 2: 15, 3: 2.
- `meta` carries:
  - cell, attempt, O_c, arming level, cluster members;
  - the range (top / bottom), the edge and its touch time;
  - flow P, A_rel, Q, PD, LE, RA, DV, SS share / VWAP;
  - put / call skew divergence (raw and residual), PS trend (raw, residual, change);
  - big move, every FR decision and the named-rule flags;
  - the A14 e = 4 zone members known at the test time.
- `direction` = the fade direction at that edge; `entry_price` / `stop_price` = the fade limit and the S-x 2 stop.
- One random event has no `entry_ts_pt` (an unfilled bottom edge).

### Decisions needed before Stage 1

1. **A14 saturation.** About 90 swings a day at the 4-pt threshold chain into zones with p90 span 52 pts, and the far edge is a swing on 70 % of tests at e = 4. Options:
   - (a) a much larger swing threshold, e.g. ≥ 0.25·IM or ≥ 3·σ5_d sustained for N minutes;
   - (b) only the session's running high / low and swings that held ≥ 15 min;
   - (c) no chaining through swings, and only refs within e of the GEX level itself;
   - (d) cap the zone span (e.g. ≤ O_c).
2. **A3 break branch (K).** The break-first share is 0 by construction, because the fade limit fills before the break trigger. If K is to be an alternative action at the same event, it needs its own trigger. Options: K only if price reaches H + 1 without first trading the fade limit, which never happens; or K as a separate setup (a break of H after the range forms), independent of F. A4's flow rules choose fade vs break *before* entry, so the A4 baselines still work; only the A3.3 readout is degenerate.
3. **Ranges are short** (median 60 s; 1 % ≥ 30 min). The A5 range-wide features (PD needs ≥ 2 pushes; A8.3 skew trend needs ≥ 3 min) are NA on 40–80 % of events. Options: define the edge event as a later touch (e.g. the first touch after ≥ N minutes or ≥ 2 pushes); or accept the thin coverage.
4. **PL fixed after a 2-pt rebound** makes the range fragile: range invalidations at PL − 2 cut valid top-edge events to 31 vs 46 L7 fills (put / short). Options: keep (A4 as written); or fix PL only after a rebound ≥ P (the confirmation size).
5. **Leg (a) of H-PNB / H-PSD** (3·σ5 in 15 min) is met by almost every test. Options: raise it (e.g. ≥ 5·σ5_d) or rely on the 0.5·IM leg.
6. **O_c at d = 5 breaks the 10-23 trade by one tick** (7697.00 vs 7696.75). The d = 10 cluster has O_c 11.75 and catches it. No change proposed: this is what the grid's d and O dimensions are for. Noted so the review doesn't read it as a detector fault.
7. **Placebo thinness at d = 5** (15–27 per cell; A/A ±4.6 pts). Per-cell T2-style comparisons in Stage 1 will be wide; pooled-within-type comparisons (A9.2) are about twice the n.
