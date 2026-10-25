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
