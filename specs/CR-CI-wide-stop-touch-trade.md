# CR-CI — Wide-Stop Touch Trade

Spec freeze. Branch `feat/CR-CI-wide-stop-touch-trade` off `origin/main` `1c71c44` (PR #87 merge, CR-CF), in its own worktree (`.claude/worktrees/cr-ci`). Read-only against the database; no table writes, no cron, no frontend.

Source: vault `Dash/CRs/CR-CI — Wide-Stop Touch Trade.md` (last_reviewed 2026-10-26). Body below is transcribed verbatim from that note (frontmatter and title omitted). L0–L6 are Ryan's kickoff (2026-10-26); C1–C10 are pre-run choices fixed by Code before Step 0.

---


## Why

The 5 `cc:fixture` cards are CR-BW touches. Note: Code drew them from CR-BW's 2026 train touches for the CR-CC smoke test; Ryan did not pick them (corrected 2026-10-26, see [[2026-10-25 - Chat — Discretionary Setup Research Arc (CR-BT to CR-CF)]]). Under CR-BW's exit (1-pt stop pad, 8-pt target) they went 4 stops and 1 target. But 4 of the 5 reached +20 with at most ~16 pts of heat, over 1–3 hours.

CR-BW tested a tight stop. CR-CE tested a 60-minute horizon with no stop. Neither tested the trade Ryan actually takes: a stop beyond the zone, a 20+ pt target, held to the close.

Prior evidence (CR-CE L8b): reaching +10 by 13:00 happens at 67.1 % of real touches and 69.8 % of placebo touches. So the bar is real − placebo and real − random, not the raw win rate.

## Locked decisions (Ryan, kickoff 2026-10-26)

| # | Decision | Value |
|---|---|---|
| L0 | Data rules | Train only: the CR-CE universe (86 sessions, 2026-01-09 → 2026-06-05; 803 re-armed touches with D > 0; CR-BW direction rule). The holdout (after 2026-06-05) is never read. R-a (2024-04-03 → 2024-12-31) and R-b (2025) are not read. Read-only connection; no connection strings in output. |
| L1 | Entry | The CR-BW/CR-CE entry price and time per event, the same as the `setup_candidates` rows. |
| L2 | Exits, fixed grid (6 cells) | Stop distance S is measured beyond the **level**, not the entry: S ∈ {10, 15, 20} pts × target T ∈ {20, 30} pts from entry. Otherwise exit at 13:00 PT. If the stop and the target are hit in the same bar, the stop counts first. Cost: 0.5 pt per round trip. Readout, not gated: IM-scaled stops S ∈ {0.25, 0.4}·IM with T = 20 and 30. |
| L3 | Baselines | The same exit rules applied to (a) CR-CE placebo levels, re-matched on EDGE/INTERIOR position in the session range at the touch (CR-CF lesson: a 63 % vs 24 % EDGE imbalance), reporting the EDGE share real vs placebo after matching; and (b) CR-CE random entries (same session, same time-of-day window, same direction). |
| L4 | Primary | Mean P&L per trade (pts, after cost), real − placebo and real − random, pooled, per grid cell. Session-cluster bootstrap, seed 20260920, 95 % CI. Also report: win rate, average win and loss, MAE before target, minutes to target, and P&L in R (R = S + level-to-entry distance). |
| L5 | Verdict (pooled, three-verdict rule) | **PASS** (shortlist for R-a) if the **pre-named centre cell S = 15 / T = 20** has CI lower bound > 0 vs **both** placebo and random, **and** at least 4 of the 6 cells have point estimates > 0 vs both (a plateau, not a peak). **REFUTED** if the centre cell's CI upper bound is < 0 vs placebo. Otherwise **INCONCLUSIVE**. Holm across the 6 cells is reported as a readout. |
| L6 | Readouts (not gated) | The 4 CR-CE cells (put/call-heavy × from below/above). First touch of a level per session only, to cap the A16 re-arming concentration; also report the top-3-session share of events. The 5 `cc:fixture` events individually, reproducing: 01-06 long, heat 15.75 → +20 at 19:00Z; 02-02, heat 4.75 → +20 at 17:05Z; 03-02, heat 10.25 → +20 at 16:09Z; 04-01 short, heat ≤ 15.75 → +20 at 18:18Z; 05-01, heat 23.75, never +20 (max +16.5). **If these don't reproduce, stop and report before the run.** |

**Step 0 (no outcomes at real levels):**
- counts per cell;
- placebo and random coverage after EDGE matching;
- an A/A check (placebo vs placebo split) on the centre cell's exit mechanics.

**Stop for Ryan's review** if coverage is < 80 % of events in any cell, or if the A/A difference CI excludes 0.

**After the run:**
- write the results and verdict here;
- update the Layer-2 row in [[setup-registry]];
- write a session note from the template;
- rebase, push, and open a PR. Don't merge.

## Pre-run choices fixed by Code (2026-10-26, before Step 0; not in the kickoff)

Each choice below resolves a point the kickoff left open. All were fixed before any outcome at a real level was computed. The one fixture probe (the five published fixture paths, run to settle C1) is disclosed under C1.

| # | Choice | Value |
|---|---|---|
| C1 | **Which entry L1 means.** CR-CE enters at the level price L at the touch second. A `setup_candidates` row (the CR-CC fixture rule, `scripts/cr_cc_build_fixture.py`) enters at the close of a confirm bar, up to 10 one-minute bars after the touch. The two can be far apart (fixture 3: entry 6858.5, level 6845). | **The `setup_candidates` rule.** The touch bar is the 1-minute bar containing the CR-CE touch second t. The confirm bar is the first of the next 10 bars (k+1 … k+10) whose close is back on the approach side of L (long: close > L; short: close < L). Entry = that bar's close; entry time = the confirm bar's start + 1 min. No confirm → no trade (counted). **Evidence:** Ryan's fixture heat/time numbers reproduce to the tick from the confirm-bar entry, and not from L. From L they come out 13.5 / 4.25 / 0.0 / 12.75 / 21.5 heat and +20 at 18:46Z / 17:02Z / 15:03Z / 18:17Z. "R = S + level-to-entry distance" also implies entry ≠ L. **Disclosure:** that probe read the 5 fixture paths, which Ryan had already published. |
| C2 | **Direction** | The CR-BW/CR-CE side, as stated in the kickoff ("CR-BW direction rule"): s = +1 (long) if approached from above, −1 (short) if from below, from the 1-s bar before the touch. The CR-CC builder instead used the 1-minute close 5 bars before the touch. Step 0 reports how many events the two rules disagree on. |
| C3 | **Price series for the trade path** | 1-minute OHLC built from `es_flow_1s_databento` (the session's CR-CE symbol; the reference feed, CR-CA), bars on [06:30, 13:00) PT: open = first, high = max, low = min, close = last. A minute with no 1-s row has no bar. It is skipped in the path and counted. The fixtures reproduce on these bars and on `ironbeam_es_1m_bars` alike; Step 0 re-checks the fixture entry prices. |
| C4 | **Exit mechanics** | Exits are checked on bars from the one after the confirm bar through the 12:59 PT bar. Long: stop SP = L − S, target TP = entry + T. Short: SP = L + S, TP = entry − T. A stop fills at the worse of SP and the bar open; a target fills at TP. Both in one bar → stop. Otherwise exit at the close of the last bar before 13:00 PT. A confirm bar at 12:59 PT gives no trade (no bar left to exit on). P&L = s·(exit − entry) − 0.5. R = S + \|entry − L\|. |
| C5 | **EDGE / INTERIOR** (CR-CF A2 D3) | EDGE if L ≥ running RTH high − 2 or L ≤ running RTH low + 2. Running extremes come from the 1-s flow, from 06:30 PT through the touch second inclusive. Otherwise INTERIOR. Applied identically to real and placebo touches. |
| C6 | **Placebo matching** | Post-stratification. Strata are the 4 CR-CE cells × {EDGE, INTERIOR}. Matched placebo mean = Σ_k w_k · (placebo mean in stratum k), where w_k = the real trades' share in stratum k, recomputed in each bootstrap resample. Per-cell comparisons use the 2 strata within the cell; pooled uses all 8. A real trade is **covered** if its stratum holds ≥ 5 placebo trades (full sample). Uncovered real trades drop from the vs-placebo comparison only. The EDGE share is reported for real, raw placebo and matched placebo. |
| C7 | **Random entries** | Per real trade, 5 draws using CR-CE's generator: `default_rng(20261025)`, u ~ U[0,1), one row of 5 per trade in event order. The random entry bar r is drawn uniformly from the bar starts in [max(06:30, entry time − 60 min), min(entry time + 60 min, 12:58)] PT. Entry is r's close. Direction is the same s. **Stop distance from entry = the paired real trade's R** (the same risk in points). Target T, same exit rules. A draw counts only if r's bar exists. Event random value = mean P&L of its valid draws; the vs-random comparison is paired (real − random value). Covered = ≥ 1 valid draw. |
| C8 | **Bootstrap, p, Holm** | Session-cluster, 10 000 resamples, seed 20260920. One multinomial session-weight matrix for every statistic; real, placebo and random resampled jointly by session. 95 % percentile CI. One-sided p = share of resamples ≤ 0. Holm (readout) is applied over the 6 grid cells, separately for vs-placebo and vs-random. A cell "passes after Holm" if both adjusted p < 0.025. |
| C9 | **A/A (Step 0)** | Placebo trades at S = 15 / T = 20. Each placebo level in each session is assigned to half A or B by `default_rng(20261026)`. Statistic: the A − B mean P&L difference, with B post-stratified to A's 8-stratum mix as in C6, plus the session-cluster bootstrap CI. **The primary A/A is that single split** (stop if its CI excludes 0). Readout: 200 splits (`default_rng(20261026)` continued), 2 000 resamples each, with the share of splits whose CI excludes 0 (expect ≈ 5 %) and the median difference. |
| C10 | **Readout definitions** | Win = P&L > 0 after cost. MAE before target = the maximum adverse excursion from entry over the bars before the target bar (the target bar's own adverse print is not counted), for target exits. Minutes to target = target bar start − entry time. The IM-scaled readout uses IM = canonical `implied_move_1d` (the CR-CE IM). First-touch-only = CR-CE `first` (k = 0). Top-3-session share = the share of primary real trades in the 3 sessions with the most trades. |

## Kickoff prompt

```
CR-CI — Wide-stop, big-target trade at CR-BW level touches (read-only, pre-registered, train only).
Branch first: git checkout -b feat/CR-CI-wide-stop-touch-trade off origin/main. Own worktree.
(... full kickoff in the 2026-10-26 Code session; L0–L6 above are its content ...)
```

---

## Amendment A1 (Ryan, 2026-10-26)

Ryan sent A1 as "before Step 0". An uncommitted Step 0 run had already finished when it arrived. That run computed no outcome at a real level (only real entries, EDGE tags and random-draw validity, plus placebo P&L for the A/A), so A1 is still fixed before any real-level outcome. Step 0 is re-run with A1 included.

| # | Clause |
|---|---|
| A1.1 | **Baseline (c), opposite direction.** Each real touch is traded the other way (break instead of fade), at the same entry price and time, with the same S/T grid, the same 13:00 PT exit and the same cost. |
| A1.2 | **Primary readout adds real − opposite** per grid cell, bootstrapped as in L4 (paired, session-cluster, seed 20260920). It also adds the share of touches where the fade hits its target first vs where the break hits its target first. |
| A1.3 | **The verdict rule (L5) is unchanged.** The opposite-direction result is reported alongside it, and is flagged in the plain-English summary as the direct answer to "can we pick which way". |

**Pre-run choices for A1 (Code, before Step 0 re-run):**

| # | Choice | Value |
|---|---|---|
| C11 | **Break trade's stop** | **The same risk in points as the paired fade trade.** For a fade with side s, R = S + \|entry − L\|. The break has side −s: stop = entry + s·R, target = entry − s·T, exits as in C4. Rejected alternative: a stop S beyond the level on the break's adverse side. When entry sits past the level (e.g. fixture 3, entry 13.5 above L), that leaves a 1.5-pt stop on a 15-pt S, so risk would differ from the fade by construction rather than by direction. Same-R isolates direction. |
| C12 | **Target race** | Per real trade and per T ∈ {20, 30}, over the bars from entry through 12:59 PT, with no stops: the first bar whose range reaches entry + s·T (fade) or entry − s·T (break). The race is a **tie** if one bar reaches both, and **neither** if no bar does. Reported per T: the shares fade-first / break-first / tie / neither, and fade-first − break-first with the session-cluster CI. Pooled and per CR-CE cell. |
| C13 | **Real − opposite statistic** | Paired: the mean over real trades of (fade P&L − break P&L), per grid cell, with CI and one-sided p (C8). Holm over the 6 cells is a readout. It also goes into the plain-English summary: "can we pick which way". |
