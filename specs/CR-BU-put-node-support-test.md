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

---

## Step 0 findings (2026-10-24)

Script `scripts/cr_bu_step0.py` (committed with this section). Connection `get_backfill_db_conn()` (role `dash_backfill_writer` asserted) with `SET default_transaction_read_only = on`; SELECTs only. Log `scripts/logs/cr_bu/step0.log` (git-ignored). Definitions fixed in the script docstring before the run. Session D = NYSE trading day in [2024-04-03, 2026-06-05]. Holiday-stamped `orats_oi_gamma.trade_date` was remapped to `next_trading_day`: 23 store dates in range, 0 collisions with a properly stamped row. Levels come from `place_levels` (imported unchanged) with P_target = S0 = MIN(`stock_price`) of the nearest expiry ≥ D. Bins use `bucket_levels(L, 5)`. Price series: the `spx_cash` hybrid series, open = first clean print ≥ 06:33 PT. No race was run at any actual node. The race harness ran only for the Q5 A/A pseudo-nodes and their placebo levels, and only actual − placebo is reported.

**Verdict: STOP. Gate 2 fails: 99 touched node days, against a minimum of 150.** Gates 1, 5 and 6 pass. Q3 and Q4 are answered below. L12 rules out re-tuning, so this needs Ryan's decision (see "Decisions needed").

### Q1 — dependency: PASS

- `packages/shared/level_placement.py` on this branch is byte-identical to `origin/main`, checked with `git diff --quiet origin/main`.
- Its introducing commit `0fc88d3` is an ancestor of `origin/main` `0e943d8` (PR #71).
- The script imports `place_levels` and `bucket_levels` unchanged.
- CR-BT Step 0 gate 2 (F_matched vs ES) passed: median 0.70 pts on the gate sample. In the ≥ 2024-04-03 era without roll weeks it was 0.23 pts (p90 0.85, n 536). Source: `specs/CR-BT-corrected-level-placement.md` Q2.

### Q2 — counts (no outcomes): **FAIL (touched 99 < 150)**

**L3 floor.** It removes 232 sessions: the trading days 2023-05-01 → 2024-04-02, all of which have `orats_oi_gamma` rows. That leaves 546 trading sessions in [2024-04-03, 2026-06-05].

**L14, each clause on its own** (on the 546 sessions):

| clause | sessions |
|---|---|
| (a) roll week | 78 |
| (b) no ES 15:45 ET bar on A | 15. All 15 fall outside (a): 2024-06-24, 07-05, 09-23, 12-02, 12-23, 12-26; 2025-03-24, 06-23, 07-07, 09-22, 12-01, 12-22, 12-26; 2026-02-18, 02-19 |
| (c) roll mismatch (CR-BT A1.4) | 8: the quarterly expiry Fridays 2024-06-21 … 2026-03-20. 2026-06-15 is in a roll week. All 8 are inside (a), so (c) removes 0 sessions beyond (a)∪(b) |

**Sequential exclusions:**

| step | removed | remaining |
|---|---|---|
| start (2024-04-03 → 2026-06-05) | | 546 |
| L14a roll week | 78 | 468 |
| L14b no ES 15:45 ET bar on A | 15 | 453 |
| L14c roll mismatch | 0 | 453 |
| no canonical `bt_daily_features` row | 19 | 434 |
| IM null or ≤ 0 | 1 (2024-04-26, IM = 0.0) | 433 |
| no SPX-cash open | 0 | 433 |
| no node rows | 0 | 433 |
| **usable** | | **433** |

**Missing canonical rows.** 25 of the 546 sessions have no canonical row (546 − 521). Almost all are the first session after a market closure: 2024-05-28, 06-20, 07-05, 09-03, 11-29, 12-24, 12-26; 2025-01-02, 01-10, 01-21, 02-18, 04-21, 05-27, 06-20, 07-03, 07-07, 09-02, 11-28, 12-24, 12-26; 2026-01-02, 01-20, 02-17, 04-06, 05-26. Six of them were already removed by L14a or L14b, so 19 are removed at this step. The rows are absent, not null.

**Nodes** (L7): 365 of the 433 usable days have one. The other 68 have no negative 5-pt bin inside [open − 1.5·IM, open − 0.25·IM].

**z_d distribution** (n 365): mean 0.952, median 1.006, p5 0.316, p25 0.624, p75 1.303, p95 1.461. The distribution is skewed toward the far edge of the window: 114 days have z in (1.25, 1.5], against 30 in the first bin, [0.25, 0.375].

| z band | 0.25–0.375 | –0.5 | –0.625 | –0.75 | –0.875 | –1.0 | –1.125 | –1.25 | –1.375 | –1.5 |
|---|---|---|---|---|---|---|---|---|---|---|
| days | 30 | 26 | 36 | 28 | 30 | 32 | 34 | 35 | 63 | 51 |

**Touched** (L8, low path only):
- **99 of 365 node days (27.1 %). Gate requires ≥ 150: FAIL.**
- Median z: 0.609 on touched days, 1.114 on untouched days.
- By year: 2024 35 of 120, 2025 49 of 168, 2026 15 of 77.
- S1 split (count only): 72 first touches before 10:00 PT, 27 after.
- Minute prints understate the session low by a median 1.38 pts (field-conventions, CR-BH Q5). That biases touches slightly downward, by an amount far too small to close a 51-event gap.

### Q3 — node anatomy (365 node bins)

- **Share of node net:** 0DTE 38.2 %, 1–7 DTE 61.8 % (aggregate of signed sums).
  - Per-day 0DTE share: median 26.7 %, IQR 5.7–63.0 %.
  - 0DTE carries the majority on 121 of 365 days.
- **Median node |net|:** 60.3 $bn, IQR 27.3–105.5. Units: stored gex ÷ 1e9, the chart's "B".
- **Median node distance (open − N):** 51.1 pts, IQR 35.5–69.0. Median IM is 52.9 pts.
- **Typical node bin:** 6 rows, 6 expiries, 2 strikes. A node is one or two strikes stacked across the week's expiries.
- **Touched subset:** 0DTE share 43.4 %, median |net| 82.5 $bn, median distance 36.6 pts.
- **By year**, median |net| / distance / 0DTE share: 2024 69.4 / 46.2 / 30 %, 2025 48.9 / 54.3 / 25 %, 2026 70.2 / 51.1 / 25 %.
- `gex_put` is stored positive (500,831 of 746,995 node-universe rows are > 0, the rest 0). net = `gex_call` − |`gex_put`| as L6 specifies.

### Q4 — worked day 2026-01-12 (random node day, `default_rng(20261024)`)

- A = 2026-01-09. S0 = 6973.40, open = 6944.10, IM = 24.12.
- Window: [6907.93, 6938.07].
- Node: N = 6920, net −9.70 $bn, z = 0.999, E = 6921.21.
- **Not touched:** no print after the open reached ≤ E. No race exists to hide.

| strike | expiry | dte | `stock_price` | `discounted_level` | corrected L | L − `discounted_level` | net $bn |
|---|---|---|---|---|---|---|---|
| 6925 | 2026-01-16 | 4 | 6977.55 | 6930.09 | 6920.88 | −9.20 | −6.58 |
| 6920 | 2026-01-13 | 1 | 6973.92 | 6922.03 | 6919.48 | −2.55 | −2.00 |
| 6920 | 2026-01-12 | 0 | 6973.40 | 6921.02 | 6920.00 | −1.02 | −1.22 |
| 6920 | 2026-01-14 | 2 | 6974.55 | 6923.05 | 6918.86 | −4.19 | −0.69 |
| 6920 | 2026-01-15 | 3 | 6975.08 | 6924.07 | 6918.34 | −5.73 | +0.79 |

- **Other window bins:** 6915 +2.11, 6910 +3.56, 6935 +8.36, 6925 +8.63 $bn. 6920 is the only negative bin in the window.
- **Under `discounted_level` binning** the same OI splits differently. The most negative bins are 6915 (−6.71) and 6910 (−3.72), while 6920 is +9.80. This day shows the S4 old-vs-corrected contrast directly.
- In SPX cash, the corrected 0DTE row sits at its strike, as it should: F_e = S0.

### Q5 — A/A control: PASS

- 365 pseudo-nodes. Days were drawn without replacement from the 433 usable days and z was drawn with replacement from the actual z_d, both from `default_rng(20261024)` after the Q4 draw.
- They went through the full L8–L11 harness: entry, race at R = 0.20·IM, leave-one-out placebo, and the month-cluster bootstrap (seed 20260920, 10 000 resamples).
- 87 pseudo-events were touched and resolved, across 26 months. The median placebo n per event is 158.
- **actual − placebo = +0.78 pts, 95 % CI [−7.08, +9.38]. The CI covers 0, so the harness shows no bias at this n.**
- The CI is about 16 pts wide at n 87. At n ≈ 100, the +8-pt pass bar would sit inside the noise band.

### Q6 — IM source: PASS

- Canonical version: `v0.6.1-nyse-sessions` (`packages/shared/canonical_version.py`), ticker SPX, `active`. IM is `feature_vector->>'implied_move_1d'`.
- 521 rows in [2024-04-03, 2026-06-05], with 0 duplicate dates.
- IM is null on 0 rows and ≤ 0 on 1: 2024-04-26 stores 0.0, not NULL.
- 25 sessions have no row at all (listed under Q2).
- IM on the usable sessions: median 50.96 pts, IQR 39.07–69.88.

### Decisions needed (Ryan)

The train universe cannot produce 150 touched events under L3 + L7 + L8. Before 2024-04-03 L3 has no valid forward (CR-BT A1.1), so the train window cannot be extended without the re-ingest or forward reconstruction raised in open-question `orats-oi-gamma-stock-price-not-forward-before-2024-04`. Options:

1. **Close CR-BU as underpowered on train.** Record "not testable at n ≥ 150 on 2024-04-03 → 2026-06-05" in State of Knowledge.
2. **Re-register with a lower n floor.** n ≈ 99 at most, and fewer once unresolved races drop out. Q5 suggests a CI half-width of about ±8 pts at that n, the size of the pass bar.
3. **Re-register the node or entry definition** (window, bin width, DTE range). Any change must be fixed now, before any outcome is computed. Touch counts reveal nothing about bounces, but choosing a definition by its touch count is still a specification change and needs a new freeze.
4. **Extend the date range backward** after resolving the pre-2024-04-03 `stock_price` open question. That is a separate CR.

---

## Amendment A1 (Ryan, 2026-10-24, after Step 0; no actual-node outcome has been computed)

A1 replaces L7/L8/L10's single-node rule; all else (L1–L6, L9 race with R = 0.20·IM, L11–L14, pass criterion incl. n ≥ 150, S1–S7) is unchanged. S1–S6 refer to the touched node.

| # | Clause |
|---|---|
| A1.1 | **Candidates:** up to 3 per day — the most negative-net 5-pt bins of corrected L (0–7 DTE rows, P_target = S0) with L in [open − 1.5·IM, open − 0.25·IM], greedy by most negative net, each ≥ 10 pts from any already chosen. Only bins with net < 0. |
| A1.2 | **Event:** scan RTH minutes from the open; the event is the first candidate whose entry E = N + 0.05·IM satisfies low ≤ E. One event per day. Race (L9) runs from that minute and that E. |
| A1.3 | **Placebo:** for event day d with candidate z-set {z_1..z_k} (z_i = (open_d − N_i)/IM_d), on each other usable day e place pseudo-levels P_i = open_e − z_i·IM_e, apply the same first-touch rule and race. Placebo rate for d = win share over touched, resolved e. Statistic = mean(actual − placebo) over event days. |

---

## Step 0 re-count under A1 (2026-10-24)

Script `scripts/cr_bu_step0_a1.py` (committed with this section). Connection: `get_backfill_db_conn()` with `SET default_transaction_read_only = on`, SELECTs only. Log: `scripts/logs/cr_bu/step0_a1.log` (git-ignored). The universe, the loads and L1–L6/L14 are unchanged from Step 0; the script asserts **433 usable sessions**.

Definitions were fixed in the docstring before the run:
- **Greedy order:** net ascending; on a tie, the higher bin first.
- **Same-minute tie:** if several candidates are touched in the same minute, the one with the highest E wins (the first level the falling path crosses). This tie occurred **0 times**.

The pipeline was run twice, before and after the pseudo unresolved counter was added. Both runs produced the same numbers.

No race was run at any actual node. The race harness ran only at A/A pseudo-levels and their placebo levels.

**Verdict: the touched gate passes, at 151 against a minimum of 150.** The A/A control passes. Two power caveats are listed at the end of this section; neither is a gate.

### Counts (low path only)

**Days with ≥ 1 candidate: 365 of 433.** This equals Step 0's node days: a day has candidates exactly when it has a negative window bin.

| candidates per day | days |
|---|---|
| 1 | 36 |
| 2 | 24 |
| 3 | 305 |

Mean: 2.74 candidates per day.

**Touched days: 151 of 365 (41.4 %).** Under the single-node rule it was 99.

**Rank of the touched candidate:**

| rank | days |
|---|---|
| 1 (most negative) | 61 |
| 2 | 44 |
| 3 | 46 |

**z of the touched candidate:** median 0.441, IQR 0.334–0.622, range 0.252–1.465.

**First touch:** 125 before 10:00 PT, 26 after.

**Touched node |net|** (stored gex ÷ 1e9, the chart's $bn): median 45.32, IQR 25.18–87.51.

| year | n | median | p25 | p75 |
|---|---|---|---|---|
| 2024 | 56 | 47.45 | 31.85 | 81.34 |
| 2025 | 71 | 36.39 | 19.37 | 85.94 |
| 2026 | 24 | 55.06 | 40.21 | 121.69 |

Candidate days by year: 2024 120, 2025 168, 2026 77.

### Net-floor counts (counts only; no floor adopted)

In these variants the same greedy selection runs only on window bins that clear the floor.

| floor | candidate days | touched | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|
| net ≤ −100 $bn | 101 | 43 | 15 | 18 | 10 |
| net ≤ −50 $bn | 210 | 81 | 30 | 36 | 15 |

### A/A control under A1: PASS

**Design:**
- 365 pseudo z-sets, drawn with replacement from the actual candidate z-sets.
- They were placed on 365 usable days, drawn without replacement, using `default_rng(20261024)`.
- They ran through the full A1 harness: first touch, race at R = 0.20·IM, leave-one-out z-set placebo, and the month-cluster bootstrap (seed 20260920, 10 000 resamples).

**Result:**
- Of the 365 pseudo z-sets, 151 were touched and 1 of those ended unresolved, leaving 150 resolved events over 27 months. The median placebo n per event was 203.
- **actual − placebo = +1.65 pts, 95 % CI [−8.05, +11.45]. The CI covers 0.**

### Power caveats (not gates; the spec is unchanged)

1. **Resolved n.** The pass criterion requires n ≥ 150 *resolved* events, and there are 151 touched. If 2 or more actual events end unresolved, T1 cannot pass whatever the win rate. In the A/A, 1 of 151 pseudo touches was unresolved.
2. **CI width.** At n 150 the A/A CI half-width is about ±9.8 pts. A true +8-pt edge would therefore probably not clear "CI lower bound > 0". An actual − placebo of roughly ≥ +10 pts is needed for the lower bound to clear 0.

---

## Addendum — S8 (Ryan, 2026-10-24, after the A1 re-count; added before any outcome is computed)

Secondary readout added to S1–S7:

- **S8 node size** (readout only, not gated, cannot change the verdict): T1 (R = 0.20·IM) restricted to events whose touched node has net ≤ −50 $bn, and separately ≤ −100 $bn (stored gex ÷ 1e9, same units as the chart). Same placebo design (A1.3); report n, actual, placebo, difference and CI, plus the same split by year.
