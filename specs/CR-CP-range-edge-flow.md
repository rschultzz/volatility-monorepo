# CR-CP — Range edge, order flow picks the side

Read-only, pre-registered, train first. Branch `feat/CR-CP-range-edge-flow` off `origin/main` `77879ec`, own worktree.
Kickoff: Ryan, 2026-10-26 (chat). Spec drafted in chat (vault `Dash/CRs/CR-CP — Range Edge, Flow Picks the Side.md`,
as updated 2026-10-26 18:02 for the A7 change). Frozen 2026-10-26 **before any CR-CP data was read**.

---

## Spec body (copied verbatim from the vault draft)

> Spec drafted in chat 2026-10-26. **D1 / D2 approved by Ryan as starting values** ("I don't really know. I guess we
> can start here"); the readout grid covers wider/narrower values so the result doesn't hinge on the pick. Written
> against [[standing-rulings]] (status current). Read-only, pre-registered, train first.

### Question

When a range forms after a larger move, at any price, does the **order flow inside the range** pick the right side to
trade from the range edge? (A1, A2, A17–A20.) Levels are a readout, not a requirement (CR-CB / CR-CF / CR-CM: levels
don't slow price on their own).

### Definitions (fixed before any outcome)

| # | Item | Primary | Readouts |
|---|---|---|---|
| D0 | Approach | a **vol-scaled swing** leg (A7 as changed 2026-10-26: 12 × session IM ÷ 2026-train median IM; A5 swing code), confirmed before the range starts; the range must start within 5 min of the leg's end | fixed swing 12; swing 8 / 16 |
| **D1** | **Range** | from the approach leg's end, a trailing box with **high − low ≤ 12 pts**, held for **≥ 15 min**. It qualifies at the first bar where the box has held 15 min. Box = [low, high] at that bar; if a later bar breaks it by > 2 pts before entry, the range is void | width 8 / 16 pts; 10 / 30 min; width ≤ 0.5 × approach leg |
| **D2** | **Flow picks the side** (A19), over range start → decision bar (Databento): **short** if net delta (buy − sell) ≥ **+2 % of range volume** and price has advanced ≤ **2 pts** in the buying direction since range start; **long** = mirror (≤ −2 %, declined ≤ 2 pts); otherwise **no trade** | 1 % / 3 %; progress 1 / 4 pts; contracts/pt of in-range pushes (A5 ratio) |
| D3 | Decision bar | first bar at or after the range qualifies where D2 calls a side; no later than 60 min after range start | — |
| D4 | Entry (A20, A3) | after the decision bar, **first touch within 1 pt of the chosen edge** (short at box high, long at box low), filled at the edge −/+ 1 pt. No return to the edge before the box breaks or 13:00 PT → no trade (counted) | entry at decision-bar close |
| D5 | Outcome (A15, A11) | stop-free race from entry: first to **±10 pts** by 13:00 PT; ties / unresolved excluded and counted | ±20 race; move at +30 / +60 min; max favourable before −10; reach the last same-day pivot (A10) |

Worked example (2026-10-26, forward, not test data): approach 07:30→07:50 up into 7731; range ~7729–7739.5 from 07:50;
in-range buying (≈ +2 % of volume on Ironbeam) with ≈ 2 pts progress → short at the top; it broke down at 08:49 and
reached −20 at 09:00.

### Comparison (primary)

Paired on the same ranges: race win rate of the **flow-chosen side minus the opposite side** (long the bottom vs short
the top of the same box), plus each vs 50 %. Session-cluster bootstrap, seed 20260920. One primary, pooled. Three
verdicts (B1): PASS if CI lower bound > 0; REFUTED if CI upper bound < 0; else INCONCLUSIVE; ≥ 40 trades, else
underpowered.

Secondary baselines (readouts): same-time random entries (B7 EDGE-matched); "no-trade" ranges traded at both edges.

### Splits (readouts, none gated)

- Range position vs the nearest strong line (A12 yellow/darkest): short of / straddling / beyond (A18), by approach
  direction; strong vs weak vs no line within 10 pts; put vs call (A8).
- Time of day; first range after the open vs later.
- Skew (A17, optional): CR-CK stickiness ratio on the approach leg, beyond-noise vs not.

### Data (B4, B5)

Train: 2026-01-06 → 2026-06-05 (Databento flow + bars). Confirmation if PASS: R-a (2024-04-03 → 12-31; Databento flow
available per CR-CN Step 0). R-b (2025) and holdout not read.

### Step 0 (no outcomes)

Range counts per session, decision counts (short / long / no trade), entry-fill share, distribution of in-range delta
share and progress, counts per split, A/A on the random baseline. Stop for review.

### Conformance (pre-check by chat)

A1, A2, A3, A7, A11, A15, A17, A18, A19, A20 honoured; A12 / A8 as readouts; A21 n/a (second touch is its own setup);
A4 n/a (stop-free); B1, B4, B5, B7 honoured; B6: this models Ryan's trade (entry after the range, at the edge); skew
block (A6, A14, A16) used as readout only.

---

## Kickoff scope (Ryan, 2026-10-26)

- Data: train 2026-01-06 → 2026-06-05 only: `es_flow_1m_databento` / `es_flow_1s_databento` and Databento ES 1-min
  bars. **R-a, R-b and the holdout are not read in this CR.**
- Reuse: `packages/shared/flow_features.py` (A5 `swing_legs`), `level_placement.py` + the chart's strong-line rule from
  CR-CM (`scripts/cr_cm_lib.py` `chart_levels`), CR-CK stickiness (`skew_expected.stickiness_read` via
  `scripts/cr_cl_lib.py` `read`) for the skew readout, CR-CL's race / bootstrap helpers (`race`, `mfe_before`,
  `move_at`, `boot_weights`).
- Step 0 (no outcomes) as listed; **stop for Ryan's review after Step 0**. Then: commit the run script before running,
  run once, results to the vault CR note, setup-registry row, session note. PR, don't merge.
- Note before Step 0 (Ryan, 2026-10-26): A7 changed — primary approach swing is vol-scaled; fixed 12 is a readout.
  The spec body above already carries it.

## Standing rulings conformance (Dash/research/standing-rulings.md, status current, read 2026-10-26 18:02)

| # | Ruling (short) | Mark | How |
|---|---|---|---|
| A1 | levels slow price, not a trade; range in a zone around a level | honoured | level-agnostic primary; level position is a readout |
| A2 | 2–3 min is not a range; pause relative to the prior move, up to ~30 min | honoured | ≥ 15 min (readouts 10 / 30); width ≤ 0.5 × approach leg as a readout (D1 approved as a fixed 12 pts by Ryan) |
| A3 | don't auto-enter at the first touch; entry after the range forms | honoured | entry only after the range qualifies and D2 calls a side (D4) |
| A4 | stops beyond the zone | n/a | stop-free direction test (A15) |
| A5 | flow and skew decide fade vs break at range edges; never "always fade" | honoured | flow picks the side per range; both sides possible |
| A6 | skew divergence direction | n/a (parked) | skew only as an A17 readout |
| A7 | approach = larger move, swing 12 × IM ÷ 2026-train median IM; fixed 12 a readout | honoured | D0 primary vol-scaled (C2); fixed 12, 8, 16 readouts |
| A8 | put levels overshoot; analyse put and call separately | honoured | put vs call split in the level readout; never pooled into the primary as a level effect |
| A9 | two close levels → wait for the second | n/a | no level in the primary |
| A10 | earlier same-day S/R matters | honoured (readout) | "reach the last same-day pivot" readout |
| A11 | target open; ±10 primary, ±20 readout, favourable-excursion distribution | honoured | D5 + MFE readout |
| A12 | levels = chart's yellow / darkest (net ≥ day p95 \|net\|) | honoured (readout) | CR-CM `chart_levels` |
| A13 | sizing | n/a | deferred |
| A14 | skew relative to expected; stickiness ratio, beyond noise | n/a (parked) | CR-CK read, shipped band, readout only |
| A15 | direction first, stop-free | honoured | D5 |
| A16 | skew as a trend over the leg | n/a (parked) | readout uses the CR-CK leg read (trend over the leg) |
| A17 | two stages; flow in the range decides; skew optional, test flow alone first | honoured | primary is flow alone; skew is a readout |
| A18 | range before or beyond the level; split by position | honoured (readout) | short of / straddling / beyond by approach direction |
| A19 | in-range confirmation, both sides | honoured | D2 short / long mirror; contracts/pt as a readout |
| A20 | trade the range edges | honoured | D4 |
| A21 | second touch is a different setup | n/a | not tested here |
| B1 | three verdicts | honoured | |
| B2 | many parameters fine if disclosed; plateau not peak | honoured | readout grid reported in full; no re-selection of the primary |
| B3 | one setup at a time | honoured | |
| B4 | train ≤ 2026-06-05; holdout, R-b never read | honoured | every read asserted ≥ 2026-01-01 and < 2026-06-05 21:00 UTC (C17) |
| B5 | Databento is the reference for flow and bars | honoured | bars and flow both from Databento (C3); Ironbeam bars read only for the chart's P_target, as the chart does (CR-CM) |
| B6 | non-Ryan exits are a mechanical cousin, label them | honoured | the ±10 race is a direction test, not a model of Ryan's exit; labelled so in the results |
| B7 | placebos matched on EDGE vs INTERIOR | honoured | random baseline matched on EDGE (C13) |
| B8 | Code labels ≠ Ryan's labels | n/a | no labels |
| C1, C2 | UI | n/a | no UI |

**No "deviates".** C1 below is an interpretation of D0's wording that Ryan should confirm; it is listed as a choice,
not as a deviation from a ruling.

## Choices the draft doesn't fix (Code, 2026-10-26, before any data)

- **C1 — what "confirmed before the range starts" means (D0).** In the A5 swing code a leg's *end* is confirmed only
  when price later retraces a full swing from it. Read literally, the range (box ≤ 12 pts) would have to start after
  a ≥ swing retracement, within 5 min of the extreme: the box would contain both the extreme and the confirming bar,
  so it is ≥ swing wide and essentially never qualifies; it also contradicts the worked example (the 07:50 top is
  inside the range). **Reading used:** the approach is the swing leg whose **start** is confirmed, i.e. price has
  moved ≥ swing from the prior opposite extreme (the A5 code's confirmation of the previous leg). The leg's end is
  its running extreme (highest high of an up leg so far). This is causal and matches the worked example. Stop for
  Ryan's OK on this reading before Step 0 data is loaded.
- **C2 — vol-scaled swing.** swing_d = 12 × IM_d ÷ median(IM) where IM = canonical `bt_daily_features`
  `implied_move_1d` (SPX pts) and the median is over the CR-CP train universe (2026-01-06 → 2026-06-05, after C4's
  exclusions). One number, reported in Step 0. Sessions with no IM are excluded (counted). D1 width (12), D2 progress
  (2), D4 (1 pt), D5 (±10) stay fixed in points as drafted; only the swing scales.
- **C3 — bars and flow.** 1-min bars and flow from `es_flow_1m_databento` (o/h/l/c, `volume`, `buy_vol`, `sell_vol`),
  RTH 06:30–13:00 PT, the session's most-RTH-volume symbol (CR-CJ / CR-CM rule; roll weeks hold both contracts).
  `es_flow_1m_databento` is not in field-conventions, so Step 0 measures it against the minute aggregate of
  `es_flow_1s_databento` (bar-start convention, o/h/l/c, volume, buy/sell) before it is used, and adds the row.
  Missing minutes (no trades) are NaN bars.
- **C4 — universe.** NYSE trading days 2026-01-06 → 2026-06-05; excluded (counted): no Databento RTH flow; early close
  (last RTH bar < 12:55 PT); no canonical IM. Roll weeks are kept (outright contract, no intraday splice), and Step 0
  reports their counts. Level readouts use CR-CM's universe; sessions outside it fall in "no levels".
- **C5 — legs.** `flow_features.swing_legs` on RTH bars from 06:30 PT (no overnight). Each up / down leg whose start is
  confirmed (C1) is an approach candidate from its confirming bar on; at most **one range per approach leg** (the
  first that qualifies). A leg stops being a candidate once the next leg's start is confirmed.
- **C6 — range start and box.** Leg end e = bar of the leg's running extreme (resets if a new extreme is made). The
  range starts at r0 ∈ [e, e + 5] (within 5 min). At bar k the box is [min L, max H] over bars r0..k; it qualifies at
  the first k with k − r0 + 1 ≥ 15 and width ≤ 12; if several r0 qualify at the same k, the earliest. Box frozen at
  the qualifying bar q. Void if any bar after q and before entry has H > high + 2 or L < low − 2 (counted).
- **C7 — D2 quantities.** Over bars r0..t (inclusive): delta share = Σ(buy − sell) ÷ Σ volume (unknown volume stays in
  the denominator). Progress in the buying direction = c[t] − o[r0] for buying (short call), o[r0] − c[t] for selling
  (long call). "Advanced ≤ 2 pts" includes declines (buying with no advance).
- **C8 — decision bar.** First t ∈ [q, r0 + 60] where D2 calls a side, evaluated bar by bar; a void before a call ends
  the range (counted "void before decision"). No call by r0 + 60 → "no trade" range.
- **C9 — entry fill.** Short: first bar j > t with H[j] ≥ high − 1, filled at high − 1 (the limit, even if the bar
  opened above it — conservative). Long mirror. Searched before 13:00 PT and before a void. If the fill bar itself
  reaches the box break the fill stands (price passes the limit first).
- **C10 — race.** CR-CL `race` from p = fill, bars j + 1 … 12:59 PT, plus the fill bar's own adverse extreme beyond the
  fill counted as a loss if ≥ 10 (its favourable extreme is not credited: order inside the bar unknown). Tie (same
  bar both) and unresolved excluded and counted; > 5 consecutive missing bars → "gap", excluded and counted.
- **C11 — primary statistic.** Population = ranges where D2 called a side. Chosen side = the D2 side at its edge; opposite
  side = the other edge after the same decision bar (its own C9 fill and C10 race). Win rate of each over its own
  resolved fills; primary = chosen − opposite, session-cluster bootstrap (10 000 draws, seed 20260920, percentile
  95 % CI). The ≥ 40 rule applies to resolved chosen-side trades. Readout: the subset where both sides filled
  (true pairs). Each side vs 50 % also reported.
- **C12 — readout definitions.** Contracts/pt of in-range pushes (D2 readout): over r0..t, up bars (c > o) and down
  bars (c < o); cpp = Σ volume ÷ Σ |c − o| per kind; short if cpp_up ÷ cpp_down ≥ 1.5 and progress ≤ 2; long mirror
  (≤ 1 ÷ 1.5). Entry at decision-bar close: race from c[t]. Last same-day pivot (A10): the last swing_d extreme
  confirmed before entry on the target side of the fill (for a short, a confirmed low below the fill), reached before
  −10. Width ≤ 0.5 × approach leg replaces 12 pts. Readouts swap one parameter at a time; all reported.
- **C13 — random baseline (B7).** For each chosen-side trade (fill minute m, side h, EDGE/INTERIOR): pool = every other
  usable session's entry at the close of minute m, side h, with the same EDGE tag (EDGE = entry price within 2 pts of
  the running RTH high for a short / low for a long, CR-CF D3 / CR-CL C7). Matched difference = trade win − pool win
  rate, averaged; session-cluster bootstrap. **A/A (Step 0):** sessions split at random (seed 20261025, 200 splits)
  into halves; pseudo-trades = the C13 keys of the *range schedule* (minute, h, EDGE) placed in half A, baseline from
  half B; the matched difference should centre on 0. Only random-entry outcomes are read for the A/A; no range
  outcome is read.
- **C14 — level split.** CR-CM `chart_levels` (displayed Corrected lines at the RTH open, strong = |net| ≥ p95, within
  ±1·IM of the open). Nearest strong line within 10 pts of the box: for an up approach, line above the box high =
  short of, inside [low, high] = straddling, below the low = beyond (mirror for down). Strong / weak (displayed, not
  strong) / no displayed line within 10 pts. Put / call by sign of net.
- **C15 — skew readout.** CR-CL `read` on the approach leg (s = leg start + 1, e = leg end + 1), shipped band; beyond
  noise vs in noise vs no read. G1-down flag as CR-CN's next-session note defines it (swing 12, ratio < 0, CR-CK band,
  end ≤ 11:00 PT), readout only. Uses CR-CJ's `y26` cache loader (2026-01-02 → 06-05).
- **C16 — time splits.** Decision-bar hour (PT); first range of the session vs later.
- **C17 — guards.** Every DB read asserted ≥ 2026-01-01 00:00 UTC (prior-session P_target for 2026-01-06 is in range)
  and < 2026-06-05 21:00 UTC; read-only transaction; `dash_backfill_writer` role checked.

## Step 0 (no range outcomes) — stop for Ryan's review

1. `es_flow_1m_databento` vs the `es_flow_1s_databento` minute aggregate (C3); field-conventions row.
2. Universe funnel; median IM and the swing_d distribution (C2).
3. Range counts per session (primary and each readout definition); void counts; legs per session.
4. Decision counts short / long / no trade; distribution of in-range delta share and progress.
5. Entry-fill share (chosen and opposite), no-fill reasons.
6. Counts per split (level position, strong / weak / none, put / call, hour, first vs later, skew).
7. A/A on the random baseline (C13).
