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
