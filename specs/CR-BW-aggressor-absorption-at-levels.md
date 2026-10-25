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
