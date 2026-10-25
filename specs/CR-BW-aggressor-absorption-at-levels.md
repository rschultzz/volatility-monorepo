# CR-BW — Aggressor Absorption at Levels

Spec freeze. Branch `feat/CR-BW-aggressor-absorption-at-levels` off `origin/main` `0e943d8` (the CR-BT merge, PR #71). Read-only against the database; no table writes, no cron, no frontend.

Source: vault `Dash/CRs/CR-BW — Aggressor Absorption at Levels.md` (last_reviewed 2026-10-25). Body below is transcribed verbatim from that note (frontmatter omitted); the locked decisions L1–L11, tests T1/T2 with their pass criteria, and secondary readouts S1–S7 were confirmed by the kickoff prompt (Ryan, 2026-10-25). Kickoff additions: train only, sessions 2025-12-26 .. 2026-06-05 with the CR-BT exclusions (CR-BU L14: roll weeks, no ES 15:45 ET bar on A, roll mismatch); holdout not read; `place_levels` imported unchanged; Step 0 computes no race outcome at real or placebo levels except inside the A/A control, which reports only its difference and CI.

---

## Why

**Ryan's hypothesis (2026-10-25):** price should slow down where open interest sits, and price action against aggressor flow should show what traders are doing there, and so whether the move continues or reverses.

**The order-flow version of this is absorption.**
- Heavy aggressive selling into a level while price barely moves means passive buyers are absorbing it, so expect a reversal.
- The same flow moving price easily means nobody is defending the level, so expect continuation.

**What CR-BU leaves open.** CR-BU found put nodes do not act as support *on average* (T1 −1.45 [−9.03, +6.50]). Flow may still separate the touches that are defended from those that aren't.

**Ryan also asked** whether a system can catch many 4–8 tick moves rather than a few large ones. Hence a tick-scale primary outcome with an explicit cost and fill model.

**State of the inputs** (chat, 2026-10-25, read-only):
- **Aggressor data is exchange-tagged and correctly signed.** `ironbeam_es_flow_1s` sums buy/sell from the Ironbeam trade feed's `as` field (1 = BUY, 2 = SELL); 0.01 % of volume is untagged. ES Z26, 2026-10-12 → 10-23, 13,555 one-minute bars:
  - corr(minute delta, minute Δprice) = **+0.49**;
  - mean Δprice +0.57 pts on buyer-dominated minutes, −0.62 on seller-dominated ones.
  - Coverage 2025-12-26 → 2026-10-23, 258 sessions.
- **The chart's flow indicator is not built for this.** It plots EMA(buy, 840 s) − EMA(sell, 840 s): a 14-minute-smoothed delta that mostly restates recent price direction. Its default symbol `XCME:ES.H26` is expired, and it works through the dominant-symbol fallback. Out of scope here; a display follow-up only if this test passes.

## Locked decisions (Ryan to confirm before spec freeze)

| # | Decision | Value |
|---|---|---|
| L1 | Scope | Test only. Read-only, no table writes, no cron, no frontend. |
| L2 | Frame | ES throughout. Levels from CR-BT `place_levels` with P_target = the ES 15:45 ET bar close on A (the chart's Corrected frame). Prices, touches and races from `ironbeam_es_flow_1s` 1-second OHLC (front contract). |
| L3 | Dates | Train: sessions 2025-12-26 → 2026-06-05. CR-BT exclusions: roll weeks, no ES 15:45 ET bar on A, roll mismatch (CR-BU L14). Holdout (after 2026-06-05) not read. |
| L4 | Levels | `orats_oi_gamma` rows for session D with `expir_date` in [D, D + 7 calendar days]; corrected L binned to 5 pts; net = `gex_call` − \|`gex_put`\| summed per bin. **Level** = any bin with \|net\| ≥ 25 $bn (stored gex ÷ 1e9), both signs, within ±1.0·IM_d of the RTH open. IM_d = `implied_move_1d`, canonical `bt_daily_features` (scale only). |
| L5 | Touch event | RTH 06:30–13:00 PT. A level is **armed** once price has been ≥ 0.10·IM_d away from it. Event = the first 1-second bar with low ≤ L ≤ high while armed. Approach side = the side price came from. The level then disarms until price is again ≥ 0.10·IM_d away. |
| L6 | Fade trade (outcome) | **Entry:** limit at L against the approach, filled only if price trades **1 tick through** L within 60 s of the touch (else no trade, counted). **Exit:** target R ticks back in the fade direction, filled only on a trade 1 tick through it. Stop R ticks through L, filled at stop + 1 tick slippage. **Win** = target before stop. Same 1-second bar hits both → loss. Neither by 13:00 PT → unresolved. **Primary R = 8 ticks (2.00 pts).** |
| L7 | Absorption predictor | Over the 300 s before the touch: D = net aggressor volume (buy − sell) signed in the approach direction, so sellers into a level approached from above are positive; ΔP = price move over the same 300 s in the approach direction. **A = ΔP / D**, points per 1,000 contracts, defined only when D > 0. Baseline B_d = median A over all non-overlapping 300 s RTH windows with D > 0 in the **prior 20 sessions** (no lookahead). **A_rel = A / B_d.** Events with D ≤ 0 (flow against the approach) are reported separately, not in T1/T2. |
| L8 | Terciles | A_rel terciles over all T1-eligible train events. **Low tercile = absorption**, high = easy continuation. |
| L9 | Placebo levels | For each session: pseudo-levels at the same set of signed distances from the open as that session's real levels, taken from another randomly drawn usable session (seed 20261025), dropping any pseudo-level within 5 pts of a real level. Same touch, trade and predictor rules. |
| L10 | Bootstrap | Session-cluster bootstrap, 10 000 resamples, seed 20260920, 95 % percentile. |
| L11 | Run discipline | All of the above committed before any outcome is computed. One run. |

## Tests and pass criteria (locked)

| Test | Statistic | Pass |
|---|---|---|
| **T1 absorption → reversal** (real levels) | win rate, low A_rel tercile − high tercile | ≥ +8 pts, CI lower bound > 0, ≥ 150 resolved events in each tercile |
| **T2 levels matter** | (T1 difference at real levels) − (the same difference at placebo levels) | ≥ +5 pts, CI lower bound > 0 |

**Verdicts:**
- **T1 and T2 pass** → absorption at OI levels carries information. Next: a holdout confirmation, then a cost-aware strategy CR.
- **T1 passes, T2 fails** → absorption works but the GEX levels add nothing. That points to a pure order-flow system; levels are display-only.
- **T1 fails** → this absorption measure does not predict reversals at this scale.

## Secondary readouts (not gated)

- **S1 cost reality** (the 4–8 tick system question):
  - for R = 4, 6, 8 ticks: win rate and **net expectancy per trade in ticks** after a $4.50 round-trip cost (0.36 tick) plus the L6 fill rules, by tercile, real vs placebo;
  - break-even win rate for each R.
- **S2 target ladder:** R = 4, 6 ticks and 0.20·IM, same rules.
- **S3 level sign:** negative-net (put) vs positive-net (call) levels.
- **S4 time of day:** before vs after 10:00 PT.
- **S5 no-fill rate:** share of touches with no entry fill, by tercile. Adverse selection shows up here.
- **S6 flow against the approach** (D ≤ 0 events): win rate vs placebo.
- **S7 window length:** A over 60 s and 900 s instead of 300 s. Readout only.

## Step 0 — diagnosis gate (answer all before the run; contradiction-stop)

1. **Flow OHLC.** `ironbeam_es_flow_1s` open/high/low/close is unmeasured (field-conventions). Compare its per-minute high/low/close against `ironbeam_es_1m_bars` on 10 train sessions. Report median and p90 differences; stop if the median exceeds 0.25 pt.
2. **Symbol.** Per session, the symbol(s) present and the front contract used. Confirm one contract per kept session after L3 exclusions.
3. **Universe and counts** (no outcomes):
   - sessions after each exclusion;
   - levels per session (median, IQR);
   - armed touch events per session;
   - T1-eligible events (D > 0), and per-tercile counts;
   - entry-fill rate (L6), from price paths only, not outcomes;
   - placebo event counts.
   - **Stop if fewer than 150 eligible events per tercile could resolve.**
4. **Baseline B_d.** Distribution across sessions and its stability; sessions without 20 prior sessions (the first ~20 of the window) are excluded and counted.
5. **Worked event.** One random event (seed 20261025): the level, its rows, the 300 s path, D, ΔP, A, B_d and A_rel. The race outcome stays hidden.
6. **A/A control.** Placebo-vs-placebo (two independent placebo draws) through the full harness. The T1-style difference must have a CI covering 0; otherwise stop.

## Outputs

- Spec `specs/CR-BW-aggressor-absorption-at-levels.md` with Step 0, results and verdict.
- `scripts/cr_bw_absorption.py`, committed before its run. Raw output in `scripts/logs/cr_bw/` (git-ignored).
- Session note in `Dash/sessions/`; State of Knowledge update; field-conventions row for the `ironbeam_es_flow_1s` OHLC fields.

## Not in scope

- Order-book data (no bid/ask or depth in the Ironbeam path). Absorption is inferred from price impact, not observed.
- Changes to the chart's flow indicator (follow-up only if T1 passes).
- Holdout reads.

## Kickoff prompt for Claude Code

```
Start CR-BW. First action: git fetch && git checkout -b feat/CR-BW-aggressor-absorption-at-levels origin/main (confirm CR-BW is unused; rename if not).
Read the vault CR "/Users/ryan/My Drive/Obsidian Vault/Anonymous/Dash/CRs/CR-BW — Aggressor Absorption at Levels.md", components/field-conventions.md and specs/CR-BT-corrected-level-placement.md.
Commit 1: spec freeze to specs/CR-BW-aggressor-absorption-at-levels.md, locked decisions, tests, pass criteria and S1–S7 verbatim.
Commit 2: Step 0 findings appended. Answer all six gate questions. Compute NO race outcomes at real or placebo levels except inside the A/A control (report only its difference and CI). Stop and report; stop immediately if any gate fails.
Read-only DB (get_backfill_db_conn, default_transaction_read_only on, statement_timeout). Train only: sessions 2025-12-26 .. 2026-06-05 with CR-BT exclusions; do not read the holdout. Import place_levels unchanged. Do not push.
```

---

## Step 0 findings (2026-10-25)

Script `scripts/cr_bw_step0.py` (log `scripts/logs/cr_bw/step0.log`, git-ignored). Read-only: `get_backfill_db_conn()` (role asserted) + `default_transaction_read_only = on` (asserted), `statement_timeout = 600s`, 3 h Python alarm; SELECTs only. Every read bounded to sessions ≤ 2026-06-05 (flow `ts_utc` < 2026-06-05 21:00 UTC); holdout not read. `place_levels` / `bucket_levels` imported unchanged. All operational definitions (touch, arming, approach side, fill, D, ΔP, baseline windows, placebo construction, A/A statistic, seeds) are fixed in the script docstring before the run. The script ran twice with identical output (the first run crashed inside Q6 on a tuple-access bug, before any race result was printed).

**No race outcome was computed at real levels or at the L9 placebo levels.** The race ran only on the two A/A placebo draws; only their difference and CI are reported.

**Verdict: STOP — gate Q6 (A/A) FAILS.** Q1–Q3 pass; Q4 and Q5 answered. The run is not started.

### Q1 — flow OHLC vs `ironbeam_es_1m_bars` — PASS

10 usable sessions (seed 20261025: 2026-02-09, 02-11, 02-12, 03-06, 04-13, 04-15, 04-21, 05-06, 05-21, 06-01). Flow 1-s rows aggregated to minutes (open = first, high = max, low = min, close = last) and joined to the 1m bar with the same UTC start, RTH only: 3,899 joined minutes (one bar missing on 05-06).

| field | median \|diff\| | p90 | p99 | max | exact |
|---|---|---|---|---|---|
| open | 0.000 | 0.000 | 0.50 | 11.75 | 96.9 % |
| high | 0.000 | 0.000 | 0.25 | 11.50 | 98.3 % |
| low | 0.000 | 0.000 | 0.25 | 8.00 | 98.4 % |
| close | 0.000 | 0.000 | 0.25 | 2.50 | 98.9 % |

Median 0 ≤ 0.25 → pass. The flow OHLC is the same price series as the 1m bars to the tick on ~98 % of minutes; rare multi-point outliers (max 11.75) exist and were not traced.

### Q2 — symbol

- One RTH symbol on every session (0 sessions with > 1): `XCME:ES.H26` 2025-12-26 → 2026-03-19 (03-20 has no flow), `XCME:ES.M26` from 2026-03-23. The switch falls inside the excluded March roll week (and 03-20 has no flow). The flow table's `trade_date` is the UTC date; RTH lies inside one UTC date.
- Every usable session uses one contract. P_target (1m bar starting 15:45 ET on A) vs the same symbol's last flow print in that minute: n 52, median |diff| 0.00, p90 0.00, max 1.75 (2026-04-08); one session (of 53) has no flow print in that minute. Levels and the traded path are in the same contract frame.

### Q3 — universe and counts (no outcomes) — PASS

Sequential exclusions over 111 trading sessions 2025-12-26 → 2026-06-05:

| step | removed | remaining |
|---|---|---|
| L14a roll week (2026-03-09 → 03-20) | 10 | 101 |
| L14b no ES 15:45 ET bar on A (2025-12-26, 2026-02-18, 02-19) | 3 | 98 |
| L14c roll mismatch | 0 | 98 |
| flow incomplete (Step 0 data rule: first RTH row ≤ 06:30:59, last ≥ 12:59:00, no gap > 60 s) | 22 | 76 |
| no canonical `bt_daily_features` row (01-20, 02-17, 04-06, 05-26) | 4 | 72 |
| IM null / ≤ 0 | 0 | 72 |
| no `orats_oi_gamma` rows | 0 | 72 |
| < 20 prior baseline sessions (B_d) | 19 | **53** |

- **Usable: 53 sessions, 2026-01-29 → 2026-06-02.**
- Flow-incomplete detail: 2026-03-03 no flow; 03-02 ends 08:22 PT; 03-04 starts 07:57; 04-27 and 05-14 have 1–1¼ h holes; the rest have one or more 61–619 s gaps in RTH (01-02, 02-04/05/06/13, 04-16, 05-08/11/13/15/18/19/20/29, 06-03/04/05). ES trades every second in RTH, so these are feed gaps, not quiet tape. The 60 s threshold is a Step 0 choice; 13 of the 22 have a maximum gap of 61–120 s.
- Levels: per session median 17 (IQR 13–20, min 5, max 32); 906 total; 287 negative-net, 619 positive-net; |net| median 57.7 $bn; |distance from open| median 33.8 pts; IM median 68.3.
- Armed touch events: **1,572** (per session median 28, IQR 17–37, max 88). Approach from above 769, from below 803, undetermined 0. Crossings between consecutive 1-s bars with no print at L (not events): 7. Touches before 06:35 PT whose 300 s window reaches pre-open flow: 49.
- T1-eligible (D > 0): **1,144**; D ≤ 0 (S6): 428.
- A_rel tercile cut points 1.608 / 3.397. A_rel percentiles 1/5/25/50/75/95/99: −6.44 / 0.25 / 1.31 / 2.34 / 4.16 / 19.2 / 84.4. Median A_rel 2.3 > 1: events are selected on price having just travelled ≥ 0.10·IM toward the level, so their impact runs above the unconditional baseline. Terciles are within-event, so this does not bias T1.

| tercile | eligible | entry filled | fill rate |
|---|---|---|---|
| low (absorption) | 382 | 347 | 0.908 |
| mid | 381 | 343 | 0.900 |
| high (continuation) | 381 | 348 | 0.913 |

- Entry-fill rate: all events 0.912, eligible 0.907. Gate: ≥ 150 could-resolve (filled) eligible events per tercile → **pass (min 343)**.
- S3 / S4 counts (eligible, filled): negative-net 397 / positive-net 641; before / after 10:00 PT 722 / 316.
- **Placebo (L9 draw, seed 20261025):** 261 pseudo-levels (vs 906 real); 430 events; 315 eligible; 293 filled; per tercile (eligible / filled) 119/109, 97/91, 99/93. **Below 150 per tercile.** At 17 levels per session inside ±1·IM (≈ 27 five-point bins), about two thirds of the bins near the open are "levels", so the 5-pt separation rule drops ~70 % of pseudo-levels. The placebo arm is about a quarter the size of the real arm, and T2 is correspondingly under-powered.

### Q4 — baseline B_d

- B_d (pts per 1,000 contracts, median A_w pooled over 20 prior flow-complete, non-roll-week sessions, 78 windows each): median **4.25**, IQR 3.98–5.03, range 3.32–5.71; none ≤ 0.
- Stable: day-to-day ratio median 1.000, p5 0.963, p95 1.063. Monthly medians: Jan 3.33, Feb 3.94, Mar 5.13, Apr 4.92, May 3.97, Jun 4.15.
- Single-session median A_w (78 baseline-eligible sessions): median 4.20, IQR 3.27–5.62. corr(B_d, same-session median A_w) = 0.07: the trailing baseline carries the regime level but not the day's own impact.
- First session with 20 prior baseline sessions: 2026-01-29. 22 sessions of the window fall before it; 19 of them are removed by the B_d clause after the earlier exclusions.

### Q5 — worked event (seed 20261025; outcome hidden)

- Session 2026-02-20 (A = 02-19), `XCME:ES.H26`. P_target 6866.25, RTH open 6852.75, IM 80.98.
- Level L = 6895 (bin net −25.23 $bn, just over the 25 $bn floor). Rows: 6880 0DTE (L 6893.60, −13.65 $bn), 6885 02-26 (6896.53, −9.17), 6880 02-23 (6893.15, −3.16), 6885 02-25 (−0.67), 6880 02-24 (+0.03), 6885 02-27 (+1.39).
- Touch 09:11:06 PT, approached from above (fade long); entry filled.
- 300 s window: reference close 6908.00 → last close before touch 6895.50. Buy 6,360 / sell 6,909 over ten 30 s slots.
- **D = 549, ΔP = +12.50 pts, A = 22.77 pts/1k, B_d = 4.18, A_rel = 5.44 → high tercile** (easy continuation). S7: A_60 45.28 (D 127), A_900 9.78 (D 2,403).

### Q6 — A/A control — **FAIL**

Two independent placebo draws (`SeedSequence(20261025).spawn(2)`), full harness: touch, fill, predictor, real-level tercile cut points, R = 8-tick race. Statistic in T2 form: [low − high tercile win rate](draw 1) − [same](draw 2). Session-cluster bootstrap, 10,000 resamples, seed 20260920.

| | resolved events (low / mid / high) |
|---|---|
| draw 1 | 317 (116 / 93 / 108) |
| draw 2 | 321 (107 / 114 / 100) |

**A/A = +16.22 pts, 95 % CI [+2.91, +31.08] — does not cover 0 → STOP.**

No further outcome was computed. What the A/A says, without diagnosing further:
- With ~100 resolved events per tercile per arm on 53 sessions, a placebo-vs-placebo difference of 16 pts falls outside the bootstrap CI. Either the session-cluster bootstrap understates the variance of this statistic, or chance (a 5 % event). The harness cannot currently tell a T2 of ±15 pts from noise.
- The placebo arm's size is the binding constraint (Q3): T2 compares ~1,040 filled real events with ~290 placebo events.

### Decisions needed (before any run)

1. **A/A failure.** Options: (a) repeat the A/A over many independent draw pairs (e.g. 200) to measure the A/A statistic's actual spread and calibrate the CI. This reads placebo outcomes only. (b) Change the bootstrap, e.g. cluster by week, or resample donors as well as host sessions so donor-draw variance enters the CI. (c) Enlarge the placebo arm (next item). The Step 0 gate is defined on one A/A pair, so (a) or (b) changes the gate and needs an amendment.
2. **Placebo thinness.** At 25 $bn, real levels fill most of the ±1·IM grid. Options: (a) average over K placebo donors per session (e.g. K = 10), which raises n and lowers donor variance; (b) relax the 5-pt separation rule; (c) raise the level threshold (e.g. 50 $bn), which leaves more room for pseudo-levels but cuts real n.
3. **Sample size.** 53 usable sessions. Over half the losses are the B_d warm-up (19) and flow gaps (22). Options: relax the flow-gap rule (13 of 22 have a maximum gap ≤ 120 s), or shorten the baseline warm-up (e.g. 10 sessions).
4. **Field-conventions row** for `ironbeam_es_flow_1s` OHLC (Q1 measured: equals `ironbeam_es_1m_bars` per minute to the tick on ~98 %; median 0; p99 0.25–0.5; rare outliers to 11.75 pts). Also: one symbol per session, `trade_date` = UTC date, RTH feed gaps on 22 of 111 train sessions. To be written at session end.

---

## Amendment A1 (Ryan, 2026-10-25, after Step 0; no outcome computed at real or run-placebo levels)

**Step 0 found:** at 25 $bn, real levels occupy about two-thirds of the 5-pt bins within ±1·IM. The cross-session placebo kept only 261 pseudo-levels against 906 real. One A/A pair gave +16.22 [+2.91, +31.08]. This amendment makes levels sparse, moves the placebo into the same session, and calibrates the bootstrap. T1/T2 pass criteria, L6 (trade, R = 8 ticks primary, fills, costs), L7 (absorption), L8 (real-event terciles, applied to placebo too), L10–L11 and S1–S7 are unchanged.

| # | Clause |
|---|---|
| A1.1 | **Levels = the chart's Corrected lines, as Ryan filters them** (replaces L4's selection). Rows: `orats_oi_gamma` for session D with `expir_date` in [D, D + 30 calendar days]. Placement: CR-BT `place_levels`, ES frame. Aggregation and selection exactly as the chart: 1-pt buckets (`bucket_levels`, half away from zero); magnitude = \|call gex\| + \|put gex\| (the chart's Min \|GEX\| filter is on magnitude, not net); **magnitude ≥ 40 $bn** (Ryan's chart filter, 2026-10-25); the chart's 5-pt minimum spacing by descending magnitude. Then keep those within ±1.0·IM_d of the open, and of those the **6** largest by magnitude. Net (call − \|put\|) is kept per level for S3. |
| A1.2 | **Placebo** (replaces L9): same session, distance-matched. For each real level, the **2 nearest** 5-pt bins on the same side of the open that are not real levels and are ≥ 10 pts from every real level, and within ±1.0·IM_d. Same touch (L5), trade (L6) and predictor (L7) rules. |
| A1.3 | **A/A calibration** (replaces gate 6): 200 random splits (seed 20261025) of the placebo events into two halves, split by pseudo-level, not by event. Each split: the T2-style statistic (half-1 tercile difference − half-2 tercile difference) with its session-cluster bootstrap CI (L10, 2,000 resamples per split). **Pass** if the CI excludes 0 in ≤ 10 % of splits (nominal 5 %) **and** the median statistic is within ±3 pts. Otherwise stop: the bootstrap is miscalibrated. |
| A1.4 | **Flow gaps** (replaces the session-level 60 s rule): keep sessions; drop an **event** if its 300 s pre-touch window or its race window contains an RTH gap > 15 s in `ironbeam_es_flow_1s`. Count drops. |
| A1.5 | **Baseline warm-up** (amends L7): B_d needs ≥ **5** prior usable sessions (trailing up to 20). Q4 showed B_d stable (ratio p5–p95 0.96–1.06). |
| A1.6 | **Q1 outliers:** before the run, trace the 1-second-vs-1-minute differences > 1 pt (count, sessions, times, symbol). If any falls inside a kept event window, drop that event and count it. |
| A1.7 | **S8 strong levels** (readout only, not gated): T1 and S1 restricted to levels with magnitude ≥ 100 $bn (Ryan: "100 BN is a strong level", untested). Report n per tercile; no conclusion below 50 per tercile. |

---

## Step 0 under A1 (2026-10-25)

Script `scripts/cr_bw_step0_a1.py` (log `scripts/logs/cr_bw/step0_a1.log`, git-ignored). It used the same read-only connection and holdout guard as the first Step 0 (`default_transaction_read_only = on`, `statement_timeout = 600s`, sessions ≤ 2026-06-05). `place_levels` / `bucket_levels` are imported unchanged. The docstring fixes every operational choice the amendment leaves open (listed below) before the run. One run.

**No race outcome was computed at real levels or at the run's placebo levels as a set.** The race ran only on placebo events inside the A1.3 splits, and only the split statistic's distribution is reported.

**Verdict: Step 0 under A1 PASSES.** The tercile gate passes (min 210 filled), and so does the A/A calibration (3.5 % of splits exclude 0; median −0.14 pts). The run is not started; that awaits Ryan.

### Operational choices fixed before the run (not stated in A1)

- **A1.1 chart band.** The chart applies its price band (the session's own bars) before the 5-pt spacing. That band is lookahead, so spacing runs over every bucket of the [D, D+30] universe; then the ±1·IM window; then the top 6. `_select_levels` is re-implemented in the script (same loop: mag ≥ threshold, descending mag, keep if ≥ 5 pts from every kept level). Importing `callbacks.py` would open the app's own DB engine, which is not the read-only role.
- **RTH open.** The open of the first flow row in [06:30:00, 06:31:00) PT. If there is none, the open of the `ironbeam_es_1m_bars` bar starting 06:30 PT; this applied only to 2026-03-04.
- **A1.4 gap measure.** The largest of: (first row − window start), consecutive row gaps, and (window end − last row). The window start is clipped to 06:30 PT.
  - Real events: Step 0 applies the rule to the pre-touch window only. The race window ends where the race resolves, and that is an outcome. So for real events the race-window part is reported only as an upper bound (gap or outlier anywhere in [touch, 13:00)).
  - Placebo events in the A/A: the rule is applied in full.
- **A1.5 baseline sessions.** Sessions passing L14 (a)–(c) with flow. A window is dropped if it has an RTH gap > 15 s or net = 0.
- **A1.2 placebo.** Ties go to the lower bin. Pseudo-levels are deduplicated per session, and each keeps its parent's mag/net for S3/S8.
- **A1.3 split.** Over the pseudo-levels that carry ≥ 1 eligible filled event: 169 of the 427. They are shuffled with `default_rng(20261025)` and cut into halves 200 times. The bootstrap is one `default_rng(20260920)` stream across splits.

### A1.6 — Q1 outlier trace

- Over the 88 usable sessions (RTH), **756 (minute, field) pairs differ by > 1 pt**. That is 465 distinct minutes on 84 sessions: O 226, H 187, L 174, C 169; H26 262, M26 494. |diff| median 2.5, p90 7.5, max 33.25.
- **Timing.** 108 pairs fall in the 12:59 PT minute and 50 in 06:30. The rest cluster on fast days: 2026-04-01 89, 04-02 71, 03-31 49, 02-26 34.
- **Which series is wrong: the 1-minute bars, not the flow.** In every traced minute the flow is continuous: each minute's open equals the prior close, with 60 one-second rows per minute. The bar is not. Example, 2026-04-02 07:36 PT:
  - bar O/H/L/C 6601.0 / 6601.0 / 6599.25 / 6599.75, volume **584**;
  - flow 6601.0 / 6634.25 / 6597.0 / 6626.25, volume **27,329**;
  - the next bar opens at 6626.5, the flow's close.

  In the 12:59 PT minute, bar volume is about half the flow's (2026-05-18: 24,298 vs 44,971). The other four traced minutes show the same pattern (bar opens at the minute's later price; the flow holds the first-second prints).
- **Raw trades.** `ironbeam_es_trades` has no rows retained for these minutes.
- **Cause.** Most consistent with the 1-minute bar feed missing trades in fast or boundary minutes. Not measured further.
- **Rule applied as written (A1.6):** an event is dropped when an outlier minute overlaps its pre-touch window. This removes 138 real and 144 placebo events, although the flow series those events use is not the defective one.

### Sessions after each rule

| rule | removed | remaining |
|---|---|---|
| trading sessions 2025-12-26 → 2026-06-05 | | 111 |
| L14a roll week | 10 | 101 |
| L14b no ES 15:45 ET bar on A (2025-12-26, 2026-02-18, 02-19) | 3 | 98 |
| L14c roll mismatch | 0 | 98 |
| no flow rows (2026-03-03) | 1 | 97 |
| no RTH open | 0 | 97 |
| no canonical `bt_daily_features` row (01-02, 01-20, 02-17, 04-06, 05-26) | 5 | 92 |
| IM null / ≤ 0, no `orats_oi_gamma` rows | 0 | 92 |
| < 5 prior baseline sessions (12-29, 12-30, 12-31, 01-05) | 4 | **88** |

**Usable: 88 sessions, 2026-01-06 → 2026-06-05** (53 before A1). Baseline windows dropped per session: median 0 of 78, max 57.

### Levels (A1.1)

- Chart lines (mag ≥ 40 $bn, 5-pt spacing) within ±1·IM before the top-6 cut: median 15 per session (IQR 11–19, max 31).
- After the top-6 cut: median 6 (IQR 6–6, min 1), 6 on 86 of 88 sessions, **522 levels** in total.
- Magnitude median 139.2 $bn (IQR 102–200). Net sign −/+ 169 / 353. |L − open| median 26.5 pts.
- **≥ 100 $bn: 399 levels on 86 sessions.**
- Placebo (A1.2): **427 pseudo-levels** (target 2 × 522 = 1,044 before dedup and the ≥ 10-pt rule), median 4 per session (IQR 3–6).

### Events (no outcomes)

| | armed touches | − A1.6 outlier in pre-window | − A1.4 gap > 15 s in pre-window | kept | eligible (D > 0) | D ≤ 0 (S6) |
|---|---|---|---|---|---|---|
| real | 1,071 | 138 | 4 | 929 | 711 | 218 |
| placebo | 757 | 144 | 8 | 605 | 455 | 150 |

Real kept events per session: median 10 (IQR 5–14, max 38). A_rel tercile cut points (real eligible): 1.632 / 3.144.

| real (eligible) | eligible | filled | fill rate | filled at ≥ 100 $bn (S8) | race-window upper-bound drop | filled left at that bound |
|---|---|---|---|---|---|---|
| low (absorption) | 237 | **224** | 0.945 | 169 | 192 | 32 |
| mid | 237 | 213 | 0.899 | 168 | 175 | 38 |
| high (continuation) | 237 | **210** | 0.886 | 163 | 172 | 38 |

| placebo (eligible, real cut points) | eligible | filled | fill rate | filled at ≥ 100 $bn |
|---|---|---|---|---|
| low | 143 | 129 | 0.902 | 103 |
| mid | 133 | 119 | 0.895 | 104 |
| high | 179 | 166 | 0.927 | 133 |

- Entry-fill rate: real 0.912 (all kept) / 0.910 (eligible); placebo 0.902 / 0.910.
- Real eligible filled, S3 neg/pos net 212 / 435; S4 before/after 10:00 PT 442 / 205.
- **Gate: every real tercile ≥ 150 eligible filled → PASS (min 210).**
- **The race-window upper bound is not informative.** It counts any > 15 s gap or outlier minute between the touch and 13:00, and the 12:59 PT bar defect sits on most sessions. So the bound drops almost every event. Measured on the A/A placebo races (actual race windows), the race-window rule removed **2 of 414** (0.5 %). The 8-tick races resolve long before the close. At that rate the real terciles stay near 209–223.
- **S8 (≥ 100 $bn):** 163–169 filled real events per tercile, above A1.7's 50 floor.
- The placebo terciles lean high (179 vs 143 eligible): with the real cut points, pseudo-level touches show easier continuation more often. This is a predictor distribution, not an outcome.

### Q4 — baseline B_d (A1.5)

Median 4.138 pts / 1,000 contracts (IQR 3.704–4.792, range 3.01–5.71). Day-to-day ratio p5 / median / p95 0.956 / 1.000 / 1.050. 14 sessions rest on fewer than 20 prior baseline sessions (min 5).

### A1.3 — A/A calibration — PASS

- Placebo eligible filled: 414. Race-window drops (gap or outlier): −2. Unresolved: 0. Resolved: 412.
- 200 splits of 169 pseudo-levels; 2,000 bootstrap resamples per split.

| | value |
|---|---|
| statistic, median | **−0.14 pts** (gate ±3) |
| mean / sd | +0.88 / 11.42 |
| p5 / p25 / p75 / p95 | −17.74 / −7.24 / +8.62 / +18.44 |
| min / max | −28.07 / +37.15 |
| CI half-width, median | 23.37 pts |
| split sd ÷ (median half-width / 1.96) | 0.96 (≈ 1 if calibrated) |
| **share of splits with CI excluding 0** | **3.5 % (7 / 200)** (gate ≤ 10 %) |

Histogram (5-pt bins, −30 → +40): 1, 3, 10, 18, 33, 36, 28, 26, 20, 16, 4, 3, 1, 1.

The session-cluster bootstrap is calibrated on this harness: the spread of the split statistic matches the bootstrap CI width (ratio 0.96), and it excludes 0 at about the nominal rate. The first Step 0's single-pair failure (+16.22) sits inside the A1 split distribution (p95 +18.4).

**Power, for the record (not a redefinition):** each A/A half holds about 200 placebo events, and the split CI half-width is about 23 pts. T2 compares about 640 real events with about 410 placebo events, so its CI will be narrower than the A/A's, but likely still above ±10 pts. T2 (≥ +5 with lower bound > 0) is under-powered for effects smaller than about 15 pts.

---

## Amendment A2 (Ryan, 2026-10-25, after the A1 Step 0; no outcome computed at real or run-placebo levels)

| # | Clause |
|---|---|
| A2.1 | A1.6's event drop is revoked. Step 0 showed the 1-minute bars, not the 1-second flow, are the defective series in fast minutes, and CR-BW uses only the flow (L2). Keep all events A1.6 dropped. Take every session's RTH open from the flow (first 1-second print >= 06:30 PT), including 2026-03-04. |
| A2.2 | New readout S9 (not gated): T1 and T2 recomputed with the A1.6 drop applied, for comparison. |
| A2.3 | The race-window gap rule (A1.4) applies as written to real and placebo events: drop an event whose race window, from entry to resolution or 13:00 PT, contains an RTH flow gap > 15 s. Count drops per tercile. |

Everything else unchanged (A1.1–A1.5, A1.7, L6 trade with R = 8 ticks primary, T1/T2 pass rules, 150 per tercile).

### Tercile counts under A2 (no outcomes)

Produced by `scripts/cr_bw_absorption.py --counts-only` (sha256 `e426a959…b7bd84`). That mode stops after building the events and runs no race. The file is committed unchanged in the next commit, before the outcome run. Log: `scripts/logs/cr_bw/a2_counts.log`.

- **Sessions.** 88 (2026-01-06 → 2026-06-05), unchanged. The 2026-03-04 open now comes from the flow (first print ≥ 06:30 PT, at 07:57:31): 6876.75.
- **Levels.** 522 real, of which 400 are ≥ 100 $bn (399 under A1; the 03-04 open moved). 427 placebo.
- **Events.** Real: 1,071 touches; 1,057 kept after the A1.4 pre-window gap rule (no A1.6 drop); 801 eligible (D > 0); 256 with D ≤ 0. Placebo: 757 touches, 742 kept, 543 eligible.
- **A_rel cut points.** 1.655 / 3.187.

| tercile | real eligible | real filled | placebo eligible | placebo filled |
|---|---|---|---|---|
| low (absorption) | 267 | **251** | 178 | 163 |
| mid | 267 | 244 | 152 | 138 |
| high (continuation) | 267 | **238** | 213 | 199 |

**Gate: every real tercile ≥ 150 filled → PASS (min 238).** The A2.3 race-window drops depend on where each race ends, so they are counted per tercile in the run.

---

## Results (single run, 2026-10-25)

Run of `scripts/cr_bw_absorption.py` at commit `51f3e3a` (sha256 `e426a959…b7bd84`), one run, exit 0. It used a read-only connection (`default_transaction_read_only = on`, `statement_timeout = 600s`, 3 h alarm), train data only, and did not read the holdout. Every statistic was written to `scripts/logs/cr_bw/results.json` before anything was printed; the printout is in `scripts/logs/cr_bw/run.log` (both git-ignored).

Universe: 88 sessions (2026-01-06 → 2026-06-05), 522 real levels, 427 placebo pseudo-levels. Bootstrap: session-cluster, 10,000 resamples, seed 20260920, with real and placebo resampled jointly by session.

### A2.3 drops and unresolved trades (R = 8 ticks; eligible and filled)

| | low | mid | high |
|---|---|---|---|
| real eligible filled | 251 | 244 | 238 |
| real race-window gap drop (A2.3) | 1 | 0 | 0 |
| real unresolved by 13:00 PT | 3 | 0 | 0 |
| **real resolved** | **247** | **244** | **238** |
| placebo eligible filled | 163 | 138 | 199 |
| placebo drop / unresolved | 0 / 0 | 0 / 0 | 0 / 0 |

### T1 — absorption → reversal (real levels, R = 8 ticks) — **FAIL**

| tercile | resolved | win rate |
|---|---|---|
| low A_rel (absorption) | 247 | 50.20 % |
| mid | 244 | 42.21 % |
| high A_rel (continuation) | 238 | 49.58 % |

**Low − high = +0.62 pts, 95 % CI [−8.43, +9.92].**
- ≥ +8: no.
- CI lower bound > 0: no.
- ≥ 150 resolved per tercile: yes.

### T2 — levels matter — **FAIL**

Placebo win rates by tercile: 43.56 % (163) / 42.03 % (138) / 46.73 % (199). Placebo low − high: −3.18 [−12.66, +6.21].

**T2 = (+0.62) − (−3.18) = +3.80 pts, 95 % CI [−9.65, +17.61].**
- ≥ +5: no.
- CI lower bound > 0: no.

### Verdict (locked rules)

**T1 fails → this absorption measure does not predict reversals at this scale.** Under the verdict table, T2 has no separate reading once T1 fails. No holdout confirmation and no strategy CR follow.

### Secondary readouts (not gated; no conclusion drawn from any of them)

**S1 cost reality.** Expectancy is per resolved trade, in ticks: win +R, loss −(R + 1), minus 0.36 tick.

| R | break-even WR | real low / mid / high WR | real exp. low / mid / high | real all (n, WR, exp.) | placebo all (n, WR, exp.) |
|---|---|---|---|---|---|
| 4 | 59.56 % | 39.60 / 32.79 / 35.71 | −1.80 / −2.41 / −2.15 | 732, 36.07 %, −2.11 | 500, 34.60 %, −2.25 |
| 6 | 56.62 % | 46.99 / 37.70 / 41.18 | −1.25 / −2.46 / −2.01 | 731, 42.00 %, −1.90 | 500, 41.00 %, −2.03 |
| 8 | 55.06 % | 50.20 / 42.21 / 49.58 | −0.83 / −2.18 / −0.93 | 729, 47.33 %, −1.32 | 500, 44.40 %, −1.81 |

Every cell is negative after costs. All win rates sit 5–27 pts below break-even. Part of that is the fill rules themselves: entry needs a trade 1 tick through L, and the target needs a trade 1 tick through L ± R.

**S2 target ladder** (T1 real / T1 placebo / T2):

| R | T1 real | T1 placebo | T2 |
|---|---|---|---|
| 4 ticks | +3.89 [−4.60, +11.83] | −4.00 | +7.88 [−5.68, +20.58] |
| 6 ticks | +5.81 [−2.79, +15.22] | −3.45 | +9.26 [−3.76, +22.89] |
| 0.20·IM | −10.14 [−20.00, −0.46] | −5.61 | −4.53 [−19.82, +10.57] |

At 0.20·IM (≈ 14 pts), resolved n is 217 / 222 / 217 and the high tercile reverts more often (57.1 %). That is the opposite sign to the hypothesis. It is one readout among many, and its CI just excludes 0.

**S3 level sign:**
- Negative-net levels (n 66 / 81 / 100): T1 +7.06 [−9.60, +25.25]; T2 +20.05 [−1.87, +43.56].
- Positive-net levels (n 181 / 163 / 138): T1 −1.93 [−11.73, +8.08]; T2 −8.62 [−24.50, +7.26].

**S4 time of day:**
- Before 10:00 PT (n 174 / 168 / 158): T1 +4.02 [−8.16, +16.04]; T2 +6.11 [−13.22, +24.66].
- After 10:00 PT (n 73 / 76 / 80): T1 −7.65 [−23.48, +7.93]; T2 −2.40.

**S5 no-fill rate** (eligible), low / mid / high:
- real 5.99 / 8.61 / 10.86 %;
- placebo 8.43 / 9.21 / 6.57 %.

The absorption tercile fills slightly more often at real levels (no adverse-selection signature in the low tercile).

**S6 flow against the approach** (D ≤ 0, R = 8): real 41.53 % (n 236) vs placebo 36.11 % (n 180); diff +5.41 [−3.80, +14.44].

**S7 window length:**
- 60 s (n 266 / 270 / 277): T1 −2.18 [−10.16, +5.86]; T2 +2.74 [−10.72, +16.15].
- 900 s (n 203 / 199 / 199): T1 −1.95 [−11.43, +7.54]; T2 +4.89 [−7.85, +17.73].

**S8 strong levels (≥ 100 $bn).**
- Real n 185 / 190 / 180 (above the 50 floor).
- T1 +4.68 [−5.30, +14.72]; placebo T1 −5.50; T2 +10.18 [−4.52, +25.45].
- S1 at R = 8: real all 45.41 % (exp. −1.64 ticks), placebo 46.32 % (−1.49). All negative.

**S9 with the A1.6 drop applied** (A2.2). The drop removes 96 real and 90 placebo eligible events.
- Real n 223 / 217 / 200.
- T1 −1.67 [−11.55, +8.45]; T2 +0.82 [−13.97, +15.57]. Same verdict.

### Notes

- Win rates at R = 8 average 47 % (real) and 44 % (placebo), below the symmetric 50 %. As in CR-BU, the touch-and-fade has a continuation base rate. Here the 1-tick-through entry and target rules add a further handicap.
- Nothing here bears on the holdout, which remains unread.
