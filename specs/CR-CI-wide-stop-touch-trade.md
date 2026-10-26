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

---

## Step 0 (2026-10-26) — no outcome at a real level

Scripts: `scripts/cr_ci_lib.py` (CR-CE `cr_ce_stage1.py` lines 52–400 copied unchanged, plus the CR-CI machinery) and `scripts/cr_ci_step0.py`. Read-only connection. Universe asserted: 86 sessions, 803 real touches with D > 0. Run on both minute-bar series (`CR_CI_BARS=databento` per C3, and `ironbeam`, the `setup_candidates` series). Logs are in `scripts/logs/cr_ci/` (git-ignored).

**Result: STOP for Ryan's review.** Under C3 (Databento bars), fixture 4 does not reproduce to the tick. Every other gate passes on both series.

### Counts (trades per CR-CE cell; the same in all 6 grid cells and in the IM readout)

| cell | real events → trades (Databento) | placebo (Databento) | real (Ironbeam) | placebo (Ironbeam) |
|---|---|---|---|---|
| put-heavy, from below | 138 → 106 | 162 → 119 | 105 | 116 |
| put-heavy, from above | 136 → 113 | 148 → 119 | 113 | 119 |
| call-heavy, from below | 285 → 229 | 161 → 123 | 229 | 123 |
| call-heavy, from above | 244 → 203 | 131 → 111 | 203 | 111 |
| **pooled** | **803 → 651** | **602 → 472** | **650** | **469** |

- **Real touches with no trade (Databento):** 151 have no confirm bar within 10 bars, and 1 confirms at 12:59. Placebo: 128 and 2. So 81 % of real touches and 78 % of placebo touches become trades.
- **First touch only:** 170 real / 138 placebo trades.
- **Concentration:** the top-3 sessions hold 10.6 % of real trades (05-18 25, 05-04 23, 05-29 21; 84 sessions have trades).
- **C2:** the CR-CC 5-bar direction disagrees with the CR-BW side on 13 of 651 trades, and is undefined on 19.
- **Entry distance from the level |entry − L|** (p5/25/50/75/95): real 0.25 / 0.75 / 1.5 / 3.25 / 7.0; placebo 0.25 / 1.0 / 2.0 / 3.5 / 8.7.

### Coverage after EDGE matching (gate: ≥ 80 % in every cell)

- **Placebo:** 100 % in every cell. Every cell × EDGE stratum holds ≥ 30 placebo trades (smallest: call_above|EDGE, 30).
- **Random:** 100 % in every cell (3 255 of 3 255 draws valid).
- **Opposite direction (A1):** 100 % by construction.
- **Pass.**

**EDGE share, real vs raw placebo** (Databento):

| cell | real | raw placebo |
|---|---|---|
| put below | 0.26 | 0.30 |
| put above | 0.27 | 0.29 |
| call below | 0.37 | 0.37 |
| call above | 0.28 | 0.27 |
| **pooled** | **0.31** | **0.31** |

After matching, placebo equals real by construction. Unlike CR-CF (63 % vs 24 %), the CR-CE placebo touches are already balanced on EDGE, so the matching changes little.

### A/A, centre cell S = 15 / T = 20, placebo vs placebo (gate: the CI must include 0)

| | primary split: A − B (matched) | 200 splits: share of CIs excluding 0 | 200 splits: median |
|---|---|---|---|
| Databento | −2.49 [−6.66, +2.12] (253 / 219 trades) | 5.0 % | +0.06 |
| Ironbeam | −2.81 [−7.00, +1.90] | 4.5 % | +0.05 |

**Pass on both.** The 200-split spread of A − B (p5–p95) is −3.7 to +3.3 pts: the noise floor for a real − placebo difference at this n.

**Placebo descriptive, centre cell** (Databento, not a test): mean −1.45 pts per trade, with 212 stops, 169 targets and 91 time exits out of 472.

### Fixture reproduction (L6)

| # | session | in the trade set | Databento entry → heat / +20 | Ironbeam entry → heat / +20 | Ryan |
|---|---|---|---|---|---|
| 1 | 01-06 long | no (before 01-09; D ≤ 0) | 6966.25 → 15.75 / 19:00Z | same | 15.75 / 19:00Z |
| 2 | 02-02 long | yes | 6991.5 → 4.75 / 17:05Z | same | 4.75 / 17:05Z |
| 3 | 03-02 long | yes | 6858.5 → 10.25 / 16:09Z | same | 10.25 / 16:09Z |
| 4 | 04-01 short | yes | **6637.75** → **16.0** / 18:18Z | 6638.0 → 15.75 / 18:18Z | ≤ 15.75 / 18:18Z |
| 5 | 05-01 long | yes | 7279.25 → 23.75 / never (max +16.5) | same | 23.75 / never (+16.5) |

**Fixture 4 on Databento:**
- The 08:38 PT confirm bar closes at 6637.75 on Databento and at 6638.0 on Ironbeam. That one tick moves the entry, and the heat becomes 16.0.
- This is a feed difference, not a logic error: the cards were built on Ironbeam bars.
- Every timestamp, direction and the other four fixtures match exactly on both series.

### The choice for Ryan (the run waits on it)

|  | Databento minute bars (C3 as frozen) | Ironbeam 1-min bars (`setup_candidates` series) |
|---|---|---|
| Fixtures | 4 of 5 to the tick; fixture 4 one tick off | 5 of 5 to the tick |
| Missing RTH minutes in the 86 sessions | **0** | **92 on 11 sessions** (02-13 29, 05-08 16, 04-02 14, 05-07 11, …) |
| Feed status | reference (CR-CA); same feed as CR-CE's touches | live feed; fragment bars repaired 2026-10-25; known undercounting in flow |
| Trades | 651 real / 472 placebo | 650 / 469 |

**Code's recommendation: keep Databento (C3).**
- It is the reference feed and has no holes.
- The touches and D already come from it.
- The fixture-4 miss is one tick on the entry price, caused by the card's source series.

Switching to Ironbeam buys an exact fixture match, at the cost of 92 missing minutes inside trade paths.

---

## Ruling on the Step 0 stop (Ryan, 2026-10-26) — A1.4

| # | Clause |
|---|---|
| A1.4 | **Use the Databento minute bars, as frozen (C3).** Fixture tolerance is **0.25 pt (one tick)** for cards built on Ironbeam bars. Fixture 4 (entry 6637.75 vs 6638.0, heat 16.0 vs ≤ 15.75) passes under it, so the L6 fixture gate passes. Ironbeam is kept as a readout only if it's cheap; it is not required. |

## Amendment A2 (Ryan, 2026-10-26) — Ryan's direction rule, stop-free

**A2 becomes the PRIMARY question.** The L2 stop/target grid (with A1's opposite-direction baseline) is demoted to a readout. It runs after the A2 primary, in the same committed run script. The amendment was written into the repo spec and this note before any A2 code.

| # | Clause |
|---|---|
| A2.0 | **Entry for A2:** the close of the **touch** bar, for every one of the 803 touches. Not the fade-confirm bar: that conditions on fade behaviour and would bias the break side. |
| A2.1 | **Approach leg:** the last swing leg into the touch at swing size 12, by the CR-CD A5 rule: from the last swing high/low **confirmed** before the touch, to the touch bar. Causal: nothing after the touch is used. Readouts at swing sizes 8 and 16. |
| A2.2 | **Delta read on the approach leg** (Databento flow). **Push** = net delta (buy − sell) in the approach direction: selling into a level from above, buying into it from below. Push > 0 is required for either call. **Effort ratio** = the leg's contracts/pt ÷ the previous same-direction leg's (the A5 ratio). **ABSORBED** if ratio ≥ 1.5; **FREE** if ratio ≤ 1.0; neutral in between. |
| A2.3 | **Skew read on the approach leg.** Dash residual put skew = actual − the Skew module's expected curve (BETA 4.5, THETA −638, as computed in `apps/web/modules/Skew/callbacks.py`), as the change from leg start to touch. **Divergent** if it moves against price (price down and residual put skew falling, or price up and residual rising). **Confirming** if it moves with price. Readout: the same with raw 25Δ put − ATM IV (nearest expiry), no expected curve. |
| A2.4 | **Call:** **FADE** if ABSORBED and skew divergent. **BREAK** if FREE and skew confirming. Otherwise **NO TRADE**. Readouts: a delta-only call and a skew-only call. |
| A2.5 | **Outcome (stop-free):** a race from entry, first to +20 or −20 pts in the called direction by 13:00 PT. A bar reaching both is a tie; unresolved races are reported separately. Also the signed move at +60 min, +120 min and 13:00. |
| A2.6 | **Population:** primary = the CR-CE 803 touches (train). Readout = every swing-12 low/high touch at any price (no level): the CR-CG "any price" version. |
| A2.7 | **Primary, paired on the same called touches:** the race win rate of the rule's call vs always-fade and vs always-break at those touches, and vs 50 %. Session-cluster bootstrap, seed 20260920. **PASS** (shortlist for R-a) if rule − max(always-fade, always-break) has CI lower bound > 0 **and** n called ≥ 60. **REFUTED** if the CI upper bound is < 0. Otherwise **INCONCLUSIVE**. Report n FADE / BREAK / NO TRADE and the accuracy per side. |
| A2.8 | **Re-run Step 0 for A2 (no outcomes):** skew coverage at the touch timestamps; approach-leg counts; the distributions of ratio and skew change; n per call; and whether R-a (2024) has the flow and skew to confirm. **Stop for Ryan's review after Step 0.** |

**Plain-English summary for Ryan at the end:** when delta and skew agree, is the direction right more often than always-fade or always-break?

### Pre-run choices for A2 (Code, 2026-10-26, before any A2 code)

| # | Choice | Value |
|---|---|---|
| C14 | **Bars and entry** | **Bars:** C3 Databento minute bars, completed as in Flow v2 (`flow_features.complete_minutes`: a quiet minute takes the prior close, with zero volume), from 06:30 PT. Volume = buy + sell (Databento `unknown_vol` ≈ 0). **Entry (A2.0):** touch bar k = the minute containing the touch second t; entry = close of bar k; entry time = bar k start + 60 s. Every one of the 803 touches has an entry. |
| C15 | **Approach leg (A2.1)** | `flow_features.swing_legs(H, L, swing)` on bars 06:30 … k inclusive (nothing after the touch bar's close). The last confirmed swing extreme is the start of the in-progress leg (`start_i`, the end of the last returned leg). The approach leg runs from that extreme to bar k. Its direction is the in-progress leg's (the opposite of the last confirmed leg). **Up** must match an approach from below and **down** an approach from above. Otherwise there is **no approach leg** at that swing size, the call is NO TRADE, and the reason is counted. Also NO TRADE: no confirmed extreme yet in the session. The leg's end is its running extreme through bar k (A5 is extreme to extreme); pts = \|end − start\|. The leg's bars are (start_i, k], as in `leg_effort`. Swing size 12 is primary; 8 and 16 are readouts. |
| C16 | **Push and effort (A2.2)** | **Push** = Σ(buy − sell) over the leg's bars × (+1 up, −1 down). **cpp** = leg volume / pts. **Ratio** = cpp ÷ the cpp of the last confirmed leg in the same direction (A5 `leg_effort`). No earlier same-direction leg, or previous cpp 0 → ratio undefined → neutral. ABSORBED: ratio ≥ 1.5. FREE: ratio ≤ 1.0. Neutral in between. |
| C17 | **Skew snapshots** | `orats_monies_minute`, SPX. Expiry = the 0DTE PM row (`expir_date = trade_date`, `expiry_tod = 'pm'`). Where the date has no PM row (third-Friday AM monthlies, field-conventions), the nearest PM `expir_date` > trade_date is used, and counted. Snapshot for a bar = the latest `snapshot_pt` ≤ that bar's close (start + 60 s). It counts as missing if more than 5 min older. Leg-start snapshot: for the extreme bar `start_i`. Touch snapshot: for bar k. |
| C18 | **Residual put skew (A2.3)** | The `callbacks.py` Expected-SS block, run unchanged on the pair (prev = leg-start row, now = touch row). The functions and constants are imported from `apps/web/modules/Skew/callbacks.py` (`_skews_from_row`, `_prev_smile_interp`, `_interp_linear_extrap`, `_T_from_row_snapshot`, BETA 4.5, BETA_MAX_SHIFT 6.0, THETA −638), with `k_for_abs_delta`. **Residual change** = put_skew_pp(touch) − exp_put_skew_pp(touch \| leg-start surface, spot shift), in vol pts. At the leg start itself it is 0 by construction, so this is the change from leg start to touch. An exception in the module block → missing. |
| C19 | **Skew direction** | Price direction = the approach leg's direction. **Divergent:** down & residual < 0, or up & residual > 0. **Confirming:** down & residual > 0, or up & residual < 0. Residual = 0 or missing → no skew read. **Raw readout:** Δ(vol75 − vol50)·100 from leg start to touch on the same rows, same signs. |
| C20 | **`snap_shot_date` clock** | The module reads `snap_shot_date` as UTC to build T. That field is not in field-conventions, so Step 0 measures snap_shot_date − snapshot_pt against the PT→UTC offset and reports it. The module convention is kept as it is (A2.3: "as computed in callbacks.py"). A mismatch is reported, not fixed. |
| C21 | **Calls (A2.4)** | **Primary:** FADE = push > 0 & ABSORBED & divergent. BREAK = push > 0 & FREE & confirming. Otherwise NO TRADE. **Delta-only readout:** push > 0 & ABSORBED → FADE; push > 0 & FREE → BREAK. **Skew-only readout:** divergent → FADE; confirming → BREAK. Fade direction = CR-BW side s; break = −s. |
| C22 | **Race and moves (A2.5)** | Bars k+1 … 12:59 PT. Up hit: high ≥ entry + 20. Down hit: low ≤ entry − 20. The first bar with either hit decides; a bar with both is a **tie**; none by 12:59 is **unresolved**. **Win rate** = wins / (wins + losses); ties and unresolved are excluded and reported. **Signed moves:** called direction × (close of the bar ending at entry + H − entry) for H = 60 and 120 min, using only events whose horizon ends by 13:00 (CR-CE A1.3); and at 13:00 (the close of 12:59). |
| C23 | **Primary statistic (A2.7)** | Over touches with a FADE or BREAK call and a resolved, non-tie race. **Rule** = the share where the called direction won. **Always-fade** = the share where the fade direction won (on the same touches); **always-break** = 1 − always-fade. Statistic: rule − max(always-fade, always-break), with the max taken inside each resample. Session-cluster bootstrap, 10 000 resamples, seed 20260920. n called = FADE + BREAK touches with a resolved non-tie race; the ≥ 60 gate uses it. Also reported: rule − 50 %, accuracy per side, n FADE / BREAK / NO TRADE, ties and unresolved. |
| C24 | **Any-price readout population (A2.6)** | Per session, every swing-12 extreme confirmed in RTH (the ends of `swing_legs` legs, plus the in-progress leg's start) becomes a level from its confirm bar's close. Touches follow the CR-BW/CR-CE rule unchanged (`events_for`: re-arm at 0.10·IM, then a 1-s bar containing the level, D > 0, pre-touch gap ≤ 15 s), counted only after the confirm time. Cells: from below / from above. The same A2 reads and calls apply. A readout, with no gate. |
| C25 | **R-a availability (A2.8)** | Counts only, no prices. On 2024-04-03 → 2024-12-31 NYSE sessions: the number with `es_flow_1s_databento` RTH rows, and the number with 0DTE-PM (or fallback) `orats_monies_minute` RTH snapshots (≥ 300 distinct minutes). |
| C26 | **Run order** | One committed run script. First the A2 primary, with all A2 readouts (swing 8 and 16, delta-only, skew-only, raw skew, any-price, the 4 CR-CE cells, first touch only). Then the demoted L2 grid with A1 (verdict wording kept as written, labelled as a readout). Every statistic is written to JSON before anything is printed. |

---

## Step 0 for A2 (2026-10-26) — no outcomes

Script: `scripts/cr_ci_step0_a2.py`. The A2 machinery is in `scripts/cr_ci_lib.py`. Read-only. The 803 touches are asserted. No price after a touch bar's close enters any statistic. Logs and reads are in `scripts/logs/cr_ci/step0_a2.*` (git-ignored).

**Stop for Ryan's review (A2.8).**

### Entry (A2.0)

The touch-bar close sits on the fade side of the level by a median of +0.25 pts (p5 −3.25, p95 +4.0). Every one of the 803 touches has an entry.

### Approach legs and effort (A2.1, A2.2)

| | swing 8 | **swing 12** | swing 16 |
|---|---|---|---|
| approach leg defined | 696 | **569** | 428 |
| no leg: in-progress leg against the approach | 61 | 139 | 206 |
| no leg: no confirmed leg yet | 39 | 88 | 162 |
| no leg: touch in the first bar | 7 | 7 | 7 |
| push > 0 | 677 | 552 | 413 |
| leg pts, p25 / 50 / 75 | 11.8 / 15.5 / 21.5 | 15.5 / 20.3 / 28.8 | 20.3 / 25.3 / 34.5 |
| leg minutes, p25 / 50 / 75 | 5 / 9 / 18 | 7 / 15 / 33 | 11 / 25 / 59 |
| ratio defined | 654 | 495 | 345 |
| ratio, p5 / 25 / 50 / 75 / 95 | 0.32 / 0.63 / 0.99 / 1.63 / 3.63 | 0.29 / 0.62 / 1.05 / 1.77 / 4.24 | 0.27 / 0.58 / 1.03 / 1.87 / 5.20 |
| ABSORBED / FREE / neutral | 185 / 330 / 181 | **149 / 238 / 182** | 113 / 169 / 146 |

### Skew (A2.3)

**Coverage:**
- A snapshot within 5 min exists at 100 % of touch bars and leg starts.
- Every session has a 0DTE PM row (no fallback expiry needed).
- 389–390 RTH snapshots per session.

**C20, `snap_shot_date`:**
- It is ISO text in UTC ("…Z"), equal to `snapshot_pt` + the PT→UTC offset on 33 490 of 33 490 rows. The module's T is built on the right clock.
- `quote_date` lags the snapshot label by 0–20 s (median 10 s). A snapshot never post-dates its label.
- Added to [[field-conventions]].

**Swing-12 reads:**
- Divergent 423, confirming 146.
- Residual change in pp, p5 / 25 / 50 / 75 / 95: −5.41 / −1.15 / +0.21 / +1.28 / +3.49.
- Raw 25Δ − ATM change: −5.09 / −0.84 / −0.04 / +0.69 / +3.11.
- Residual and raw agree in sign on 64 %; correlation 0.74.

> [!warning] Finding for review: "divergent" is the residual's default state
> At swing 12 the residual reads divergent on **74 %** of approach legs, against **43 %** for raw skew.
> - By leg-size quartile the residual's divergent share is 0.82 / 0.79 / 0.71 / 0.65, and raw's 0.46 / 0.55 / 0.37 / 0.34.
> - Down legs 0.71, up legs 0.77.
> - Signed residual vs leg pts: Spearman 0.01.
>
> So the expected curve (BETA 4.5, THETA −638; uncalibrated, see [[skew-expected-curve-calibration]]) predicts more put-skew movement *with* price than actually happens, by a roughly constant amount. "Residual divergent" mostly means "the model over-predicted", whatever the leg size.
>
> Consequences:
> - It does not bias A2.7: the rule is compared with always-fade and always-break on the same called touches.
> - It does tilt the calls toward FADE, about 2 : 1.
> - It means the skew leg of the rule carries less information than its label suggests. The raw-skew readout is the cross-check.
>
> **Code's recommendation: run A2 as specified.** The residual is what Ryan reads on his screen, and that is the thing under test. Keep raw as the readout.

### Calls (A2.4), n of 803

| | FADE | BREAK | NO TRADE |
|---|---|---|---|
| **primary, swing 12** | **116** | **58** | **629** |
| swing 8 | 132 | 71 | 600 |
| swing 16 | 83 | 48 | 672 |
| delta-only, swing 12 | 142 | 231 | 430 |
| skew-only, swing 12 | 423 | 146 | 234 |
| raw-skew rule, swing 12 | 62 | 126 | 615 |
| first touch only, swing 12 | 27 | 12 | 179 |

**Primary by cell, swing 12** (FADE / BREAK): put below 22 / 15, put above 23 / 9, call below 42 / 16, call above 29 / 18.

**n called:** 174 at swing 12 before ties and unresolved races drop out. The A2.7 n ≥ 60 gate is likely met pooled. Per-cell n are 25–58, readouts only.

### Any-price population (A2.6 readout, C24)

- **Touches:** 2 549 swing-12 extreme touches over 82 sessions, after the extreme is known; 1 997 with D > 0 (below 1 078, above 919; first touch 681).
- **Approach legs:** defined for 1 744; push > 0 on 1 686.
- **Effort:** ABSORBED 548, FREE 862, neutral 334.
- **Skew:** divergent 1 435, confirming 309.
- **Calls:** FADE 441, BREAK 159, NO TRADE 1 397.

### R-a (2024-04-03 → 2024-12-31): can it confirm? (C25, counts only)

Of 189 NYSE sessions:
- 189 have `es_flow_1s_databento` RTH rows;
- 186 have ≥ 300 0DTE-PM `orats_monies_minute` snapshots;
- 186 have both.

**Yes, R-a has the flow and the skew.** Corrected levels exist there from CR-CB A17.
