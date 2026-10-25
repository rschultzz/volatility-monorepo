# CR-CB — Failed Retest at a Level

Spec freeze. Branch `feat/CR-CB-failed-retest-at-level` off `origin/main` `d97f3bf` (PR #76 merge), in its own worktree. Read-only against the database; no table writes, no cron, no frontend.

Source: vault `Dash/CRs/CR-CB — Failed Retest at a Level.md` (last_reviewed 2026-10-25). Body below is transcribed verbatim from that note (frontmatter omitted); the locked decisions L1–L11, Stage 1 readouts, Stage 2 tests T1/T2 and Step 0 questions were confirmed by the Stage 1 kickoff prompt (Ryan, 2026-10-25). Kickoff additions: reuse CR-BW's universe (A2), level selection A1.1, flow loader, gap rule, fill/cost model and A/A design (A1.3); Step 0 fixes O_c per cell from pre-entry paths only and computes no exit outcome except the A/A hold-to-close on placebo setups; train only (2026-01-06 .. 2026-06-05); 2026-10-23 read only for the Q4 detector illustration (no outcome).

---

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
