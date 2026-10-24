# CR-BU — Put-Node Support Test

Spec freeze. Branch `feat/CR-BU-put-node-support-test` off `origin/main` `0e943d8` (the CR-BT merge, PR #71). Read-only against the database; no table writes, no cron, no frontend.

Source: vault `Dash/CRs/CR-BU — Put-Node Support Test.md` (last_reviewed 2026-10-24; amended pre-freeze 2026-10-24: L3 date floor 2024-04-03, L14 session exclusions). Body below is transcribed verbatim from that note (frontmatter omitted); locked decisions L1–L14, the pass criterion and secondary readouts S1–S7 were confirmed by the kickoff prompt (Ryan, 2026-10-24). S7 = target ladder R = 0.10/0.30/0.40·IM and a fixed 15 SPX pts; only R = 0.20·IM is gated.

---


# CR-BU — Put-Node Support Test

## Why

**Ryan's hypothesis (2026-10-24):** large negative-gamma nodes give support. Put holders monetize near the strike, where their puts have gained the most. The dealer's short-put hedge (short futures) is then bought back, which supports price.

**Prior tests did not cover this case:**
- CR-BI tested landscape walls as multi-day touch targets and containment bands.
- CR-BK tested the repriced profile's single peak.

Neither tested **near-dated strike clusters below the open as intraday support**. That is also the one case where the physics allows sharp levels: near-dated gamma is narrow, and 0DTE becomes very narrow late in the day.

**The two mechanisms predict different phases:**
- Gamma hedging (dealers short gamma below spot) predicts the move *accelerates* into the node.
- Monetization predicts it *stops and bounces* there.
- Monetization also predicts that **put OI at the node falls the next day**.

**Prompt:** a single-day observation on 2026-10-23. That date is in the holdout and is not used here.

Levels are placed with CR-BT's corrected function, so this tests the hypothesis, not the `discounted_level` misplacement.

## Locked decisions (Ryan to confirm before spec freeze)

| # | Decision | Value |
|---|---|---|
| L1 | Scope | Test only. Read-only, no table writes, no cron, no frontend. |
| L2 | Frame | SPX cash (frame A) throughout: levels, prices, outcomes. Levels from CR-BT `place_levels` with P_target = S0 (the nearest live expiry's `stock_price`, prior session 15:46 ET; CR-BK A1.1). No ES except in secondary readout S4. |
| L3 | Dates | Train only: session date D in **[2024-04-03, 2026-06-05]**. 2024-04-03 is the first session on which `orats_oi_gamma.stock_price` is a per-expiry forward (CR-BT Step 0 Q1); `place_levels` is undefined before it (CR-BT A1.1). Holiday-stamped store dates remapped to `next_trading_day` (CR-BK A1.6). Holdout not read. |
| L4 | Price series | CR-BH hybrid SPX-cash minute series (`packages/shared/spx_cash.py`, cutover 2023-11-09). RTH 06:30–13:00 PT via `packages/shared/sessions.py`. Open = first clean print ≥ 06:33 PT (CR-BI base). |
| L5 | Scale unit | IM_d = `implied_move_1d` from the canonical `bt_daily_features` version (Step 0 names it). It is a scale unit only, applied identically to actual and placebo days. It runs ≈ 85 % of the traded straddle (field-conventions); that is accepted. |
| L6 | Node rows (primary) | `orats_oi_gamma` rows for effective date d with `expir_date` in [d, d + 7 calendar days]. Row net = `gex_call` − \|`gex_put`\| (stored; gamma at the prior 15:46 ET). |
| L7 | Node | Bin corrected L to 5 pts; sum net per bin. **N_d = the bin with the most negative net** among bins with L in [open_d − 1.5·IM_d, open_d − 0.25·IM_d]. No negative bin in that window → no node that day. One node per day. |
| L8 | Entry (trade-shaped) | Entry price E_d = N_d + 0.05·IM_d (a resting buy just above the node). Event = the first RTH minute after the open with SPX-cash low ≤ E_d. No touch → not an event. |
| L9 | Outcome (primary) | From the entry minute, a symmetric race measured from E_d. **Win (bounce)** = high ≥ E_d + R before low ≤ E_d − R, with **R = 0.20·IM_d** (≈ 13–16 SPX pts at 2026 vol levels; Ryan 2026-10-24: a ~15-pt capture is tradeable). Both barriers in the same minute → loss. Neither by 13:00 PT → **unresolved**: reported, excluded from the rate. Symmetric from entry, so the win rate reads directly as a trade's win rate (break-even 50 % before costs). |
| L10 | Placebo | Leave-one-out across days (CR-BK T1 design). z_d = (open_d − N_d)/IM_d. For every other usable day e: P_e = open_e − z_d·IM_e, with the same entry rule (L8, E_e = P_e + 0.05·IM_e) and the same race (L9, R = 0.20·IM_e). Placebo rate for d = win share over touched, resolved e. Statistic = mean(actual − placebo) over event days. |
| L11 | Bootstrap | CR-BI month-cluster bootstrap: calendar months of the event date, 10 000 resamples, seed 20260920, 95 % percentile. |
| L12 | Run discipline | Everything here is committed before any outcome is computed. One run. No re-tuning, no added cuts. |
| L13 | Dealer sign | Unchanged: dealers long calls, short puts. Not tested. |
| L14 | Session exclusions (CR-BT Step 0) | Excluded from every test and readout, and counted in Step 0: (a) **roll weeks**: Monday of the week before the quarterly third Friday through that Friday (CR-BM/BO definition), all years; (b) sessions whose API date A has **no ES 15:45 ET bar** (CR-BT A1.2); (c) **roll-mismatch sessions**: a contract switch between the A 15:45 ET snapshot and the end of session D (CR-BT A1.4). (b) and (c) bear on S4 only (ES), but they are excluded everywhere so that T1 and S4 run on the same days. |

## Pass criterion (locked)

| Test | Pass |
|---|---|
| T1 bounce at the put node (R = 0.20·IM) | actual − placebo win rate **≥ +8 pts**, CI lower bound > 0, with **n ≥ 150** resolved events |

**Verdicts:**
- **Pass** → put nodes carry intraday support information. The follow-up is a holdout confirmation (pre-registration file, n ≥ 60) and then CR "C" (intraday recast of gamma size).
- **Fail** → near-dated put nodes do not separate from placebo at this power. Record in State of Knowledge; the corrected levels stay display-only.

## Secondary readouts (reported, not gated, cannot change the verdict)

- **S1 time of day:** T1 split by first touch before vs after 10:00 PT. Physics says late-day nodes should be stronger.
- **S2 approach speed (gamma mechanism):** \|SPX move\| over the 15 minutes before the touch, in IM units, actual vs placebo.
- **S3 node definitions:** 0DTE-only rows; all expiries.
- **S4 old placement, as displayed** (same L14 universe):
  - nodes from `discounted_level` mapped to cash at the snapshot: L_old = `discounted_level` × S0 ÷ ES(15:46 ET), with ES as in CR-BT L4;
  - this is the direct old-vs-corrected comparison.
- **S5 monetization (OI):** for node-bin rows whose expiry survives to d+1, the change in put OI from d to d+1 at those strikes:
  - compare bounce days with non-bounce days;
  - compare against the matched-distance strike on non-event days.
  - 0DTE rows expire and cannot be measured.
- **S6 close:** share of event days closing above N_d, actual vs placebo.
- **S7 target ladder:** the same race at R = 0.10, 0.30 and 0.40·IM, and at a **fixed 15 SPX pts**. For each: win rate vs placebo, and mean points per resolved event, (2p − 1)·R, vs placebo. Only R = 0.20·IM is gated; the ladder shows where any edge lives.

## Step 0 — diagnosis gate (answer all before the run; contradiction-stop)

1. **Dependency.** CR-BT's `place_levels` is merged on origin and imported unchanged. CR-BT Step 0 gate 2 (F_matched vs ES) passed. If not, stop.
2. **Counts (no outcomes).**
   - Usable train days.
   - Days with a node, and the distribution of z_d.
   - Days whose node is touched. The touch count uses only the low path, not the race outcome, so it reveals nothing about bounces.
   - Sessions removed by L3's floor and by each L14 clause, separately.
   - **Stop if the touched count is under 150.** Expected universe ≈ 2024-04-03 → 2026-06-05 less roll weeks, roughly 450 sessions; whether 150 touched events survive is the first thing Step 0 answers.
3. **Node anatomy.** For the node bins: share of net from 0DTE vs 1–7 DTE, median node \|net\| ($bn), and median node distance in points.
4. **Worked day** (a random train date, seed 20261024). Node rows, bin, corrected L vs `discounted_level`, touch minute. The race result stays hidden.
5. **A/A control.** Build pseudo-nodes on random train days at z values drawn from the actual z distribution, run them through the full harness, and report actual − placebo. It must have a CI covering 0. If it doesn't, the harness is biased: stop.
6. **IM source.** Name the canonical `bt_daily_features` version and the count of days with IM null or 0 (excluded).

## Outputs

- Spec `specs/CR-BU-put-node-support-test.md` with Step 0, the results tables and the verdict.
- `scripts/cr_bu_support.py`, committed before its run. Raw output in `scripts/logs/cr_bu/` (git-ignored).
- Session note in `Dash/sessions/`. State of Knowledge update.

## Not in scope

- Positive-gamma nodes as resistance. They are a candidate for a separate pre-registration if T1 passes.
- Holdout reads.
- Intraday OI or flow (OI is prior-day; a known limit).
- Any live-card or regime change.

## Kickoff prompt for Claude Code

```
Start CR-BU only after CR-BT is merged on origin. First action: git fetch && git checkout -b feat/CR-BU-put-node-support-test origin/main (confirm CR-BU is unused; rename if not).
Read the vault CR at "/Users/ryan/My Drive/Obsidian Vault/Anonymous/Dash/CRs/CR-BU — Put-Node Support Test.md" and components/field-conventions.md.
Commit 1: spec freeze to specs/CR-BU-put-node-support-test.md, locked decisions and pass criterion transcribed verbatim.
Commit 2: Step 0 findings appended to the spec. Answer all six gate questions; compute no bounce outcomes in Step 0; stop and report to me. Stop immediately if any gate fails.
Read-only against the database. Train dates only (session date 2024-04-03 .. 2026-06-05, L3/L14 exclusions); do not read the holdout. Import place_levels from CR-BT unchanged.
After I approve Step 0: commit scripts/cr_bu_support.py before running it, run T1 and S1-S6 once, append results and the verdict to the spec, write the session note.
```
