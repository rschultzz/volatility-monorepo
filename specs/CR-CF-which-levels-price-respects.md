# CR-CF — Which Levels Does Price Respect

Spec freeze. Branch `feat/CR-CF-which-levels-price-respects` off `origin/main` `2f7cd79` (PR #83 merge, CR-CB), in its own worktree. Read-only against the database; no table writes, no cron, no frontend.

Source: vault `Dash/CRs/CR-CF — Which Levels Does Price Respect.md` (last_reviewed 2026-10-25). Body below is transcribed verbatim from that note (frontmatter omitted); the locked decisions L1–L12 and the Step 0 questions were confirmed by the kickoff prompt (Ryan, 2026-10-25). Kickoff additions: reuse the CR-CB A17 T0 harness (`scripts/cr_cb_t0_ra_a17.py`, itself `scripts/cr_cb_t0_ra.py` + A17), copied, not imported; Step 0 reads train only (2026-01-06 .. 2026-06-05); do not push.

---

## Why

CR-CB's T0 (A17, R-a; train v3) found **no evidence that GEX levels slow price**, Corrected or Current. Every cell was inconclusive, leaning toward faster escape.

**Ryan's question (2026-10-25):** if not open interest, what makes ranges develop? Candidates are the reference levels traders watch: prior-day high/low/close, the overnight high/low, the opening range, VWAP, round numbers, the prior-week high/low and same-day swings. The 2026-05-01 review card hinted at this: the support Ryan would have bought was earlier same-day support, not the GEX line.

**Purpose.** Find which level families price actually respects. Those that pass become the "where" layer for the level-agnostic range + flow setups (a later CR). This CR reuses CR-CB A17's T0 harness unchanged except for the level source.

## Locked decisions (Ryan to confirm before spec freeze)

| # | Decision | Value |
|---|---|---|
| L1 | Scope | Test only, read-only, price paths only. No trade outcomes. |
| L2 | Data | ES front contract. RTH seconds from `es_flow_1s_databento` (OHLC + volume); ETH extremes from `ironbeam_es_1m_bars` (contract consistent with the session's front contract; roll weeks excluded per CR-BT/CR-CB). |
| L3 | Level families, all causal (known at the time of the test) | **F1** prior RTH session high, low and close. **F2** overnight high and low (15:00 PT prior → 06:30 PT). **F3** opening range high and low (06:30–07:00 PT; armed from 07:00). **F4** session VWAP from 06:30, a moving level, updated each second. **F5** round numbers, as three sub-families: multiples of 100, 50 (not 100) and 25 (not 50). **F6** prior-week high and low (RTH). **F7** today's confirmed swing highs and lows (CR-CB A15.1: retrace ≥ max(8, 0.15·IM), unviolated ≥ 10 min; armed once confirmed). **Reference row:** Corrected GEX (CR-CB A17 numbers, not re-run). |
| L4 | Level window | Levels within ±1·IM_d of the RTH open (F4 always included). |
| L5 | Test mechanics (CR-CB A17) | One test per level per session (the first). Arming distance max(0.10·IM_d, 4 pts), always beyond the 2-pt test band. Session-weighted. Early closes excluded; IM < 25 flagged, with an exclusion readout. |
| L6 | Measures | **Primary:** time to escape 0.25·IM_d (restricted mean, capped at 120 min or 13:00). **Secondary:** zone dwell, with zone = level ± max(3 pts, 0.05·IM_d), over 60 min; and velocity drop (CR-CB A7.1(c)). |
| L7 | Placebo | Static families: same session, distance-matched pseudo-levels, 4 per real level, ≥ 10 pts from **any level of that family**. F4 (VWAP): VWAP + a random offset of ±(10–30) pts, re-drawn per session (seed 20261025). F5: placebo prices must not fall on a 25-multiple ± 3 pts. |
| L8 | Approach split | Per family, approached from below vs above, plus pooled. No put/call split, since these are not option levels. |
| L9 | Stage 1 (screen) | Train 2026-01-06 → 2026-06-05. A family is **shortlisted** if the primary measure (slower escape, real > placebo) has CI lower bound > 0, pooled or in either approach cell. Holm across families × {pooled, below, above}, one-sided α 0.025. |
| L10 | Stage 2 (confirm) | Shortlisted families only, frozen, on R-a = 2024-04-03 → 2024-12-31. Disclosure: R-a has been read for GEX-level T0 (CR-CB A16/A17) only; these families' paths were not examined. Read nothing from 2025 or the holdout. |
| L11 | Readouts | Each family's overlap with GEX levels (share of its levels within 5 pts of a Corrected GEX level), and T0 split by GEX-overlap vs not, which shows whether GEX adds anything **on top of** a respected level. Also time of day (before / after 10:00 PT). |
| L12 | Registry | Results go to `research/setup-registry.md` (vault) as "Layer 1 candidates": family, stage, verdict, key numbers. |

## Step 0

1. Per family: levels per session (median and IQR), tests per session, and placebo counts.
2. A/A on placebo (CR-CB design, 200 splits) for the primary measure.
3. A worked day, 2026-05-01: list every family's levels near 7271–7277 at 07:30–08:10 PT. Was the 71–72 support Ryan described a family member, and which one? Price paths only; no trade outcome.

Stop after Step 0 for Ryan's go.

## Kickoff prompt

```
Start CR-CF in its own worktree: git fetch && git worktree add <scratch>/crcf-wt -b feat/CR-CF-which-levels-price-respects origin/main (confirm CR-CF is unused). Other sessions may be running in their own worktrees — do not touch them.
Read the vault CR "/Users/ryan/My Drive/Obsidian Vault/Anonymous/Dash/CRs/CR-CF — Which Levels Does Price Respect.md", components/field-conventions.md, and the CR-CB spec's A15.9/A16/A17 (reuse scripts/cr_cb_t0_ra.py's harness; copy, don't import from an unmerged branch).
Commit 1: spec freeze, verbatim. Commit 2: Step 0 (counts, A/A, the 2026-05-01 worked day). Read-only DB, train only for Step 0. Stop and report. Do not push.
```

---

## Step 0 findings (Stage 1, train, 2026-10-25)

**Script and outputs.**
- Script: `scripts/cr_cf_step0.py` (sha256 `f3ea6c2f…cb9507d9`). Logs (git-ignored): `scripts/logs/cr_cf/step0.log` and `step0.json`.
- Read-only: `get_backfill_db_conn()` (role asserted) + `default_transaction_read_only = on` (asserted); SELECTs only. Every flow / bar read is asserted inside [2026-01-01 00:00 UTC, 2026-06-05 21:00 UTC). No 2025 data, no R-a, no holdout.
- Harness: copied (not imported) from `scripts/cr_cb_t0_ra_a17.py` (universe, contract check, A1.1 GEX levels, A17 first test with arming max(0.10·IM, 4) and 2-pt band, restricted-mean time to escape, session weighting); `overnight()` / `swings15()` and the A/A split design from `scripts/cr_cb_step0_v3.py`. The only change is the level source, plus a per-row level array (for VWAP) and an armed-from time (F3 07:00, F7 known time). Every choice the spec leaves open is fixed in the script docstring before the run.
- **No real-level T0 measure is computed** (asserted in the script): Q1 counts tests, Q2 uses placebo tests only, Q3 lists levels and test times. The Stage 1 screen is unseen.

**Runs: three.**
- Run 1 completed. Its worked-day table lacked each level's first-test time, so it could not say whether the 08:02 dip would count under L5; that field was added. Q1/Q2 code unchanged.
- Run 2 crashed on an SSL disconnect from the database (`psycopg.OperationalError`) before any output.
- **Run 3 is the one reported.** Its Q1 and Q2 output is byte-identical to run 1's. Run 1's outputs are kept as `run1_step0.*`.

**Choices worth knowing (fixed in the docstring):**
- F1 / F6 use today's front symbol, and only prior sessions whose own front is that symbol. F2 comes from `ironbeam_es_1m_bars` (no symbol column) 15:00 PT prior → 06:30 PT, kept only if the 06:29 PT bar close is within 2 pts of the Databento close.
- F6 is missing when the prior week has a day before 2026-01-01 (keeps 2025 unread).
- F4 VWAP = cumulative per-second typical price × volume from 06:30, through the previous second (causal). Arming, test, escape are measured against VWAP at that second.
- Within a family, levels < 0.5 pt apart are merged.
- Placebo (L7): the 4 nearest 5-pt bins per real level, same side of the open, within ±1·IM, ≥ 10 pts from every level of that family that session (in or out of window; F7: every kept swing of the day). Placebos inherit the parent's armed-from time. F5: bins > 3 pts from any 25-multiple. F4: 4 offsets per session, ±U[10, 30], `default_rng(20261025)`.
- A/A: per family × {pooled, below, above} and all families pooled; 200 splits of pseudo-levels (`default_rng(20261025)`), session-weighted half-1 − half-2 mean, 2,000 session resamples per split (`default_rng(20260920)`). Pass: CI excludes 0 in ≤ 10 % of splits and median within ±3 min (the CR-BW ±3 tolerance, in minutes).

### Universe

| rule | removed | remaining |
|---|---|---|
| train sessions 2026-01-06 → 06-05 | — | 105 |
| roll week (03-09 → 03-20) | 10 | 95 |
| no ES 15:45 ET bar on A (02-18, 02-19) | 2 | 93 |
| quarterly-expiry Friday / no Databento RTH flow / early close | 0 | 93 |
| no canonical IM (01-20, 02-17, 04-06, 05-26) | 4 | 89 |
| no `orats_oi_gamma` rows / contract check | 0 | **89** |

- 89 sessions (ES H26 / M26), the same universe as CR-CB Step 0 v3. No early closes on train.
- IM median 61.3 (IQR 50.3–84.2). **One session has IM < 25:** 2026-01-12 (24.1). Arming = max(0.10·IM, 4) is ≥ 4 pts everywhere.
- Families missing: F2 on 1 session (2026-03-05: no 06:29 PT bar); F6 on 4 (01-06 → 01-09, prior week in 2025).

### Q1 — counts per family (no outcomes)

| family | levels / session in window, median [IQR] (share of sessions with 0) | placebo levels / session | tests real / placebo | sessions with a test, real / placebo | real tests below / above (sessions) | placebo tests below / above (sessions) | real tests before / after 10:00 PT | GEX overlap (≤ 5 pts), levels; tests real / placebo |
|---|---|---|---|---|---|---|---|---|
| F1 prior H/L/C | 3 [2, 3] (0.01) | 8 | 133 / 281 | 79 / 70 | 74 (53) / 59 (44) | 145 (48) / 136 (45) | 123 / 10 | 0.49; 0.47 / 0.52 |
| F2 overnight H/L | 2 [2, 2] (0.02) | 8 | 96 / 325 | 82 / 81 | 52 (51) / 44 (44) | 167 (65) / 158 (60) | 90 / 6 | 0.44; 0.44 / 0.46 |
| F3 opening range | 2 [2, 2] (0.00) | 8 | 116 / 317 | 84 / 88 | 65 (63) / 51 (51) | 176 (66) / 141 (62) | 112 / 4 | 0.49; 0.52 / 0.46 |
| F4 VWAP | 1 | 4 | 88 / 230 | 88 / 85 | 50 / 38 | 115 (68) / 115 (58) | 88 / 0 | —; 0.50 / 0.42 |
| F5 R100 | 1 [1, 2] (0.07) | 4 | 53 / 199 | 53 / 67 | 28 / 25 | 100 (39) / 99 (38) | 47 / 6 | 0.31; 0.38 / 0.46 |
| F5 R50 | 1 [1, 2] (0.04) | 4 | 63 / 217 | 61 / 68 | 34 / 29 | 110 (36) / 107 (38) | 55 / 8 | 0.30; 0.33 / 0.48 |
| F5 R25 | 3 [2, 4] (0.00) | 10 | 121 / 442 | 84 / 87 | 64 (54) / 57 (52) | 249 (75) / 193 (64) | 104 / 17 | 0.44; 0.50 / 0.47 |
| F6 prior week H/L | 1 [1, 1] (0.25) | 4 | 35 / 119 | 35 / 48 | 18 / 17 | 66 (30) / 53 (19) | 27 / 8 | 0.55; 0.51 / 0.41 |
| F7 confirmed swings | 12 [9, 17] (0.00) | 6 | **803 / 46** | 87 / **18** | 421 (86) / 382 (78) | 26 (10) / 20 (8) | 480 / 323 | 0.48; 0.48 / 0.33 |

Real tests by kind: prior close 51, prior high 46, prior low 36; overnight high 53, low 43; OR high 63, low 53; VWAP 88; prior-week high 22, low 13; swing high 419, swing low 384.

**Readings (counts only):**
- **The F7 placebo is degenerate: 46 placebo tests on 18 sessions against 803 real tests on 87.** With ~15 kept swings a day, "≥ 10 pts from any level of that family" excludes almost every price the session visited, so the surviving bins are prices price did not come back to after the parent's known time. An F7 real − placebo comparison under L7 as written would compare levels in the traded range with levels outside it. **Needs a decision before Stage 1** (decision 1).
- **Most tests come before 10:00 PT** for every family except F7 (F1 123 / 10, F2 90 / 6, F3 112 / 4, F4 88 / 0). First-test-only spends a static level early, so the L11 time-of-day readout has almost no after-10:00 sample outside F7.
- **Thin cells.** F6 has 35 real tests (18 / 17 sessions per side); F5 R100 and R50 have 25–34 sessions per side. Cells with < ~30 sessions will have wide CIs.
- **The L11 GEX-overlap split is near chance.** 44–55 % of family levels sit within 5 pts of a Corrected GEX level, and placebo tests overlap at about the same rate (0.41–0.52). With ~6 GEX levels per session at 5-pt spacing inside ±1·IM, a random price is within 5 pts of one about half the time. The split still answers L11's question, but "overlap" is close to a coin flip, not a property of the family.
- **About 20 % of non-F5 placebo levels sit on a 25-multiple** (F1 0.21, F2 0.19, F3 0.22, F6 0.20, F7 0.22), because placebo bins are 5-pt multiples. If round numbers are respected, that biases those families' placebo toward "respected" (decision 2).
- Swings: median 17 per session before the 10-minute hold, 15 after (as CR-CB v3).

### Q2 — A/A on placebo, primary measure (time to escape)

| family | pooled: tests, CI-excludes-0 share, median (min) | below | above |
|---|---|---|---|
| F1 | 281, 8.5 %, +0.23 PASS | 145, 2.5 %, −0.38 PASS | 136, 5.5 %, −0.03 PASS |
| F2 | 325, 4.0 %, +0.51 PASS | 167, 6.5 %, +0.63 PASS | 158, 6.0 %, −0.11 PASS |
| F3 | 317, 6.0 %, +0.25 PASS | 176, 6.0 %, −0.05 PASS | 141, 3.0 %, −0.76 PASS |
| F4 | 230, 5.5 %, −0.09 PASS | 115, 8.0 %, −0.02 PASS | 115, 3.5 %, −0.19 PASS |
| F5 R100 | 199, 3.5 %, −0.37 PASS | 100, 4.5 %, +0.21 PASS | 99, 6.0 %, +0.53 PASS |
| F5 R50 | 217, 8.0 %, −0.07 PASS | 110, 7.0 %, +0.24 PASS | 107, 6.0 %, −1.34 PASS |
| F5 R25 | 442, 6.0 %, +0.28 PASS | 249, 7.0 %, +0.77 PASS | 193, 8.0 %, +0.35 PASS |
| F6 | 119, 3.5 %, +0.49 PASS | 66, 7.5 %, −1.45 PASS | **53, 12.0 %, −0.59 FAIL** |
| F7 | 46, 5.5 %, −1.10 PASS | 26, 8.0 %, −0.41 PASS | 20, 5.5 %, +2.25 PASS |
| all families | 2,176, 5.0 %, +0.07 PASS | 1,154, 6.5 %, −0.03 PASS | 1,022, 5.0 %, −0.16 PASS |

- **The bootstrap is calibrated in 26 of 27 family cells and pooled** (5.0 % against nominal 5 %).
- **F6 / above fails** (12.0 %, 53 placebo tests on 19 sessions). Small-sample miscalibration, consistent with the thin-cell note above (decision 3).
- F7 passes, but on 46 placebo tests; the A/A says nothing about the F7 comparison while its placebo is degenerate.

### Q3 — worked day 2026-05-01 (price paths only; no outcome)

ES M26; RTH open 7264.50; IM 70.2 (arming 7.0 pts). Price path 07:30 → 08:10 PT: from 7290 down to **7274.50 at 07:37:53**, back up to 7285.50 (07:47), then the **08:00 dip to 7272.00 at 08:02:18**; VWAP 7282.2–7283.4 throughout (≈ 10 pts above the low).

**Family levels in 7266–7282 known by 08:10 PT:**

| family | level | known from | first A17 test (the only one L5 counts) |
|---|---|---|---|
| F2 overnight high | **7271.50** | 06:30 | **06:32:14**, from below |
| F5 R25 | 7275.00 | 06:30 | 06:33:53, from below |
| F7 swing low (the 07:37:53 low) | 7274.50 | 07:47:53 | **08:00:27, from above** |
| *(reference)* Corrected GEX | 7272.0 / 7277.0 | 06:30 | 06:32:48 / 06:34:58, from below |

Nothing from F1 (7157.25 / 7238.25 / 7251.75), F3 (OR 7258.75–7293.25), F4 or F6 (7079.25 / 7200.50) is in the band.

**Earlier same-day support.** Minute lows 06:32 → 06:55 PT sat at 7270.25–7275.00 (06:40 7271.50, 06:46–06:47 7270.75, 06:50 7270.25): price held the overnight high from above for about 20 minutes after the open. That shelf is **not an F7 swing**: its retraces (≤ ~7 pts) are below the A15.1 threshold max(8, 0.15·70.2) = 10.5.

**Answer.**
- **The 71–72 support is the overnight high (F2, 7271.50)**, which was the 06:32–06:55 shelf. The 08:02 low (7272.00) held 0.50 above it. The Corrected GEX 7272 line sits on the same price, and the 07:37 swing low (F7 7274.50) and the 7275 round number are 2.5–3 pts above.
- **Under L5, the 08:02 dip is not a test of F2, GEX 7272 or 7275.** Each had its one test at 06:32–06:35 from below, at the open. The dip counts only as the **first test of the F7 swing low 7274.50 (08:00:27, from above)**.
- So on this day, first-test-only scores the support Ryan would have bought as an F7 event. The level Ryan described (the overnight high, retested from above two hours later) is not tested at that moment under the A17 mechanics (decision 4).

### Decisions for Ryan before Stage 1

1. **F7 placebo.** As written (≥ 10 pts from any swing of the day), it leaves 46 placebo tests on 18 sessions. Options: (a) exclude only swings **known at the parent's armed-from time**; (b) distance-match to the parent swing (e.g. parent ± 10–30 pts, like F4) and ignore other swings; (c) drop F7 from the Stage 1 screen.
2. **Placebo on round numbers.** About 20 % of non-F5 placebo bins are 25-multiples. Keep as written, or apply F5's "> 3 pts from a 25-multiple" rule to every family's placebo.
3. **Thin cells.** F6 / above fails the A/A (12 %, n 53). Keep F6 in the Holm family (it is unlikely to pass anyway) or screen F6 pooled only.
4. **First test only vs retests from the other side.** On 2026-05-01 the overnight high was tested at the open from below. The later retest from above (the one Ryan would trade) is not counted. L5 is CR-CB A17's fix for re-arming degeneracy. A variant "first test from each side" (≤ 2 per level per session) would keep that fix and count role-reversal retests. Keep L5 as frozen, or amend before Stage 1.

---

## Amendment A1 (Ryan, 2026-10-25, after Step 0; no outcomes seen): rulings on the four Step 0 decisions

**Rulings (verbatim):**
1. F7 swings placebo: exclude only swings confirmed at or before the placebo's test time (causal), and distance-match each placebo to its parent swing (same signed distance from the open as the parent, ≥ 10 pts from any swing confirmed so far).
2. All non-round families' placebos must avoid multiples of 25 ± 3 pts (as F5's already do).
3. F6 prior week: readout only, removed from the Holm family.
4. One test per level per side (from below, from above), with the A17 arming rule within each side; first-side-only is a readout.

Re-run Step 0 counts and the A/A under A1 (no outcomes); the A/A must pass for every family cell in the Holm set before Stage 1 runs.

| # | Clause (implementation, fixed before the A1 runs) |
|---|---|
| A1.1 | **F7 placebo.** Per in-window real swing (the parent), candidates are the 5-pt bins on the parent's side of the open, within ±1·IM_d, ordered by distance to the parent. This is the distance match: the literal "same signed distance from the open" is the parent's own price, so the nearest bins on the same side are used. At the parent's known time, bins < 10 pts from any swing confirmed at or before that time (the parent included) are excluded, and so are bins ≤ 3 pts from a 25-multiple; the 4 nearest remaining bins are the parent's placebos. A placebo is armed from its parent's known time; a bin claimed by several parents takes the earliest. **At each placebo test, the test is kept only if the bin is ≥ 10 pts from every swing confirmed at or before the test second**; otherwise that test is dropped (counted), and the bin stays a placebo level. |
| A1.2 | **Placebo off round numbers.** F1, F2, F3, F6 and F7 placebo bins must be > 3 pts from every multiple of 25 (F5 unchanged). F4 (a moving level): a placebo test is dropped when VWAP + offset at the test second is ≤ 3 pts from a 25-multiple (counted). |
| A1.3 | **Holm family.** F1, F2, F3, F4, F5 R100, F5 R50, F5 R25, F7 × {pooled, below, above} = **24 hypotheses**, one-sided α 0.025. F6 is a readout with the same statistics and no gate. |
| A1.4 | **Tests (amends L5).** Per level and session, the first test **from below** (first arming h ≤ L − arm after the armed-from time, then the first h ≥ L − 2) and the first test **from above** (first l ≥ L + arm, then the first l ≤ L + 2), found independently: ≤ 2 tests per level per session. Arming max(0.10·IM_d, 4 pts) within each side. Session-weighted means (A17.3) pool both sides in "pooled". **Readout "first side only"** = the earlier of the two tests, which is L5 as frozen. |
| A1.5 | **A/A gate.** Step 0 counts and the A/A (Step 0 design: 200 pseudo-level splits, 2,000 session resamples, CI excludes 0 in ≤ 10 % of splits, median within ±3 min) are re-run under A1 on placebo tests only. Stage 1 runs only if all 24 Holm cells pass. |
| A1.6 | **Stage 1 run.** The script is committed before its single run, and every statistic is written to JSON before printing. **Primary** (L6): restricted mean time to escape 0.25·IM_d, real − placebo, session-weighted, 10,000 session-multinomial resamples (seed 20260920), 95 % percentile CI, one-sided p = share of resamples ≤ 0 (floor 1e-4), Holm step-down over A1.3. **Shortlisted** (L9) = at least one of the family's three hypotheses Holm-rejected with CI lower bound > 0. Each hypothesis is otherwise inconclusive, or refuted when the CI upper bound is < 0. **Secondary, no gate:** zone dwell (seconds within level ± max(3, 0.05·IM_d) over 60 min, forward-filled 1-s closes, capped at 13:00) and velocity drop (log of the 10-s-sampled path length in the 5 min after the test over the 5 min before, as CR-CB v3). **Readouts, no gate:** L11 GEX overlap (real − placebo within overlap / non-overlap tests, and real overlap − real non-overlap) and time of day (before / after 10:00 PT); first side only; F6; excluding IM < 25. |

## Step 0 under A1 (train, 2026-10-25; counts and A/A only, no outcomes)

**Run.**
- `scripts/cr_cf_step0_a1.py` (sha256 `d6a69354…7df7cfb1`) on the shared construction `scripts/cr_cf_lib.py` (sha256 `abff78f2…bc393563`); one run, exit 0. Logs `scripts/logs/cr_cf/step0_a1.{log,json}` (git-ignored).
- The library is Step 0's code with the A1 changes; it is the same file the Stage 1 script imports.
- Read-only, train only; no real-level measure (asserted).

**Check.** First-side-only real tests equal the frozen Step 0 counts in every family (F1 133, F2 96, F3 116, F4 88, R100 53, R50 63, R25 121, F6 35, F7 803). So the A1.4 readout reproduces L5.

### Counts under A1

| family | real tests (sessions): below / above | real first-side-only | real before / after 10:00 PT | placebo tests (sessions): below / above | GEX overlap, real / placebo tests |
|---|---|---|---|---|---|
| F1 | 218 (79): 112 (71) / 106 (65) | 133 | 193 / 25 | 423 (72): 215 (65) / 208 (69) | 0.48 / 0.54 |
| F2 | 152 (82): 78 (71) / 74 (67) | 96 | 141 / 11 | 486 (80): 246 (78) / 240 (77) | 0.43 / 0.47 |
| F3 | 185 (84): 94 (77) / 91 (78) | 116 | 170 / 15 | 453 (86): 236 (76) / 217 (80) | 0.56 / 0.46 |
| F4 | 166 (88): 84 (84) / 82 (82) | 88 | 160 / 6 | 287 (83): 146 (77) / 141 (70) | 0.48 / 0.39 |
| F5 R100 | 87 (53): 44 / 43 | 53 | 70 / 17 | 326 (67): 169 (62) / 157 (60) | 0.41 / 0.45 |
| F5 R50 | 102 (61): 51 / 51 | 63 | 88 / 14 | 353 (68): 180 (59) / 173 (62) | 0.34 / 0.49 |
| F5 R25 | 194 (84): 98 (77) / 96 (74) | 121 | 160 / 34 | 719 (87): 375 (86) / 344 (85) | 0.50 / 0.50 |
| F6 (readout) | 55 (35): 29 / 26 | 35 | 41 / 14 | 193 (48): 107 (43) / 86 (37) | 0.53 / 0.47 |
| F7 | 1,228 (87): 613 (87) / 615 (84) | 803 | 705 / 523 | **324 (76)**: 173 (62) / 151 (62) | 0.49 / 0.46 |

- **F7 placebo is usable now:** 324 tests on 76 sessions, up from 46 on 18. Placebo levels per session median 10. **282 F7 placebo tests were dropped** because a swing confirmed by the test second lay < 10 pts away (A1.1).
- **78 F4 placebo tests were dropped** because VWAP + offset sat ≤ 3 pts from a 25-multiple (A1.2). No static placebo bin sits on a 25-multiple (share 0.0 in every family).
- Per-side tests add 30–65 % to the real counts. Most tests are still before 10:00 PT, except F7.

### A/A under A1 (placebo only; CI-excludes-0 share, median min)

| family | pooled | below | above | first side only |
|---|---|---|---|---|
| F1 | 7.0 %, +0.72 | 3.0 %, −0.76 | 6.5 %, −0.03 | 6.5 %, +0.59 |
| F2 | 5.0 %, −0.12 | 6.5 %, −0.15 | 5.0 %, −0.24 | 4.0 %, −0.06 |
| F3 | 6.0 %, −0.74 | 4.0 %, −0.16 | 4.0 %, −0.02 | 6.0 %, −0.63 |
| F4 | 3.5 %, +0.17 | 3.5 %, −0.39 | 4.5 %, +0.20 | 6.5 %, +0.48 |
| F5 R100 | 3.0 %, −0.51 | 5.0 %, −0.37 | 6.5 %, +1.05 | 3.5 %, −0.37 |
| F5 R50 | 5.5 %, +0.68 | 5.5 %, +0.43 | 8.0 %, +0.52 | 8.0 %, −0.07 |
| F5 R25 | 3.5 %, +0.53 | 4.0 %, −0.30 | 5.5 %, +0.37 | 6.0 %, +0.28 |
| F6 (readout) | 5.0 %, −0.06 | 5.5 %, +0.11 | 8.0 %, −1.23 | 7.0 %, +0.31 |
| F7 | 8.0 %, −0.36 | 4.5 %, +0.20 | **10.0 %**, −0.03 | 6.0 %, −0.05 |
| all families | 4.5 %, +0.13 | 8.0 %, −0.16 | 6.5 %, +0.24 | 8.0 %, −0.04 |

**A1.5 gate: PASS.** All 24 Holm cells pass. F7 / above passes at the limit (10.0 %, i.e. 20 of 200 splits). F6 passes in every cell under A1 (it is a readout regardless). Stage 1 may run.

## Stage 1 result — screen on train (single run, 2026-10-25)

**Run.**
- `scripts/cr_cf_stage1.py`, commit `8ada987` (sha256 `3f9f6275…f45142768a`), on `scripts/cr_cf_lib.py` unchanged since the A1 gate run (`b9414a0`).
- One run, exit 0, 129 s. All statistics were written to `scripts/logs/cr_cf/stage1.json` before printing (log `stage1.log`; both git-ignored).
- Read-only, train only (89 sessions, 2026-01-06 → 06-05). No trade outcome.

**Tests.** 2,387 real, 3,564 placebo. Escapes censored at 13:00: 309 / 470. Placebo tests dropped: F7 282 (swing confirmed within 10 pts, A1.1), F4 78 (at a 25-multiple, A1.2).

### Primary — time to escape 0.25·IM, session-weighted (restricted mean, min; real − placebo; Holm over 24, one-sided α 0.025)

| family | pooled | below | above | verdict |
|---|---|---|---|---|
| F1 prior H/L/C | −5.24 [−14.88, +3.79] | −7.36 [−17.48, +2.49] | −1.12 [−14.01, +11.65] | inconclusive |
| F2 overnight H/L | −3.30 [−11.60, +5.33] | −7.17 [−17.19, +3.32] | +3.65 [−6.08, +13.68] | inconclusive |
| F3 opening range | +0.50 [−8.48, +9.48] | −2.82 [−12.19, +6.83] | +3.65 [−8.01, +15.19] | inconclusive |
| F4 VWAP | −1.95 [−10.48, +6.57] | −4.68 [−14.50, +5.24] | +0.92 [−10.86, +12.51] | inconclusive |
| F5 R100 | +4.08 [−4.37, +12.86] | +5.21 [−6.46, +17.10] | +2.85 [−10.02, +16.10] | inconclusive |
| F5 R50 | −5.45 [−13.68, +2.88] | −10.08 [−21.03, +0.90] | −6.42 [−16.54, +3.41] | inconclusive |
| F5 R25 | −1.62 [−8.76, +5.71] | +0.47 [−9.50, +10.51] | −1.34 [−9.73, +7.22] | inconclusive |
| **F7 swings** | **+8.22 [+2.00, +14.44]**, p 0.0052 | +8.29 [−0.37, +16.83], p 0.030 | +8.60 [−0.06, +16.83], p 0.026 | inconclusive (not Holm-rejected) |
| *F6 prior week (readout)* | +7.54 [−6.22, +21.99] | +8.27 [−7.15, +24.47] | +10.17 [−9.63, +30.72] | no gate |
| *Corrected GEX (reference, CR-CB A17, R-a; not re-run)* | −5.98 [−13.35, +1.21] | | | inconclusive |

Sessions real / placebo per cell: 43–88 / 59–87. Real / placebo means: F7 pooled 55.6 vs 47.4 min; other families 36.5–52.5 vs 39.2–55.4.

**Stage 1: no family is shortlisted.** No hypothesis is Holm-rejected; the smallest p is F7 / pooled at 0.0052, against a first-step threshold of 0.025 / 24 = 0.0010. Under L9, Stage 2 (R-a) has nothing to confirm. No hypothesis is refuted either (no CI upper bound < 0).

### Secondary (no gate)

- **Zone dwell:** no CI excludes 0 except **F2 / above +211 s [+13, +413]**. F5 R50 and R25 lean negative (−134 to −189 s, CIs touching 0). F7 pooled +10 [−119, +139].
- **Velocity drop** (log path-length ratio, after / before; negative = slows down):
  - F3 / below −0.11 [−0.18, −0.03] and F5 R25 / below −0.09 [−0.16, −0.01] slow down;
  - **F5 R50 speeds up**: pooled +0.23 [+0.09, +0.39], above +0.32;
  - F4 pooled +0.10 [+0.02, +0.17] speeds up;
  - F6 pooled −0.20 [−0.37, −0.05] slows down;
  - F7 pooled −0.03 [−0.07, +0.02].
  - These signs are mixed and 27 cells per measure are uncorrected.

### Readouts (no gate)

- **First side only (L5):** close to the per-side picture.
  - F7 pooled +8.19 [+1.57, +14.88], above +11.24 [+1.26, +20.88].
  - F4 / above +22.76 [+7.63, +38.22] (38 sessions).
  - F1 / below −17.94 [−31.45, −4.92]: faster escape.
- **Excluding IM < 25** (one session, 2026-01-12): unchanged to two decimals except F4 and F5 R25 (≤ 0.6 min).
- **GEX overlap (L11, pooled):** no subset CI excludes 0.
  - Real overlap − real non-overlap: F1 +3.8, F2 −5.8, F3 −6.8, F4 −9.3, R100 +8.4, R50 +0.5, R25 −9.6, F7 −3.8, F6 +4.0. All CIs cover 0.
  - **On train, being near a Corrected GEX level adds nothing measurable on top of any family.**
- **Time of day (L11, pooled):**
  - F7 before 10:00 PT +9.78 [+2.05, +17.54] (705 / 265 tests); after 10:00 −3.87 [−19.57, +11.71] (523 / 59).
  - Every other family has ≤ 34 real tests after 10:00, so no after-10:00 read.

### Reading (descriptive; the verdict is the Holm table)

- **On train, none of the seven families slows price measurably relative to its placebo** once 24 hypotheses are corrected for.
  - Prior day, overnight, opening range, VWAP and the 50 / 25 round numbers all have point estimates within ±6 min of placebo, mostly negative (faster escape), like Corrected GEX in CR-CB.
- **F7 (same-day confirmed swings) is the one consistent positive.** Real swings hold price about 8 min longer than placebo in both approach cells, on first-side-only too, before 10:00 PT. The pooled 95 % CI excludes 0, but the result does not survive Holm. It is a lead, not a finding.
- **Construction caveat on F7, stated before any follow-up:**
  - A1.1 drops a placebo test whenever a swing has been confirmed within 10 pts by the test second. That happened to 282 tests, against 324 kept.
  - A swing confirming near a placebo price is evidence that price stalled there. So the drop rule may remove the placebo tests that escape slowly, making the placebo faster and inflating F7's real − placebo.
  - Whether it does is unmeasured. Measuring it needs the dropped placebo tests' escape times: a post-hoc diagnostic, to be disclosed as such, not run in this pass.
- The 2026-05-01 support (overnight high, Step 0 Q3) belongs to F2, which shows no slowdown on train (−3.3 pooled; from above, the side Ryan would buy, +3.65 [−6.08, +13.68]).

## Amendment A2 (Ryan, 2026-10-25, after Stage 1): A1.1 approved; post-hoc F7 diagnostic and its decision rule (recorded before D1–D4 run)

**Ruling 1.** The A1.1 implementation is **approved as written**: the 4 nearest 5-pt bins on the parent's side of the open; bins within 10 pts of a swing confirmed by then are skipped; a placebo test is dropped when a swing confirmed by the test second is within 10 pts.

**Ruling 2: post-hoc F7 diagnostic.** Every number is labelled **"post-hoc, not a test"**.
- Train only. No Holm, and no p-values reported as evidence. No parameter changes to Stage 1. Holdout, R-a and R-b untouched.
- Session-cluster bootstrap (seed 20260920), 95 % CIs.

| # | Readout |
|---|---|
| D1 | Escape-time distribution of the 282 dropped placebo tests vs the 324 kept (median, IQR, mean; session-weighted dropped − kept with CI). Are the dropped tests slow escapers? |
| D2 | F7 real − placebo with the dropped tests restored to the placebo pool. Conservative: some restored tests sit near real swings. Read it as a lower bound. |
| D3 | Range-position confound. Every real and placebo test is tagged **EDGE** (the level is at or beyond the session's running RTH high or low at the test second, within 2 pts) or **INTERIOR**. Report F7 real − placebo within EDGE only and within INTERIOR only, plus the EDGE / INTERIOR share in real vs placebo. |
| D4 | D2 and D3 on first-side-only tests (consistency check). |

- One table, with counts per cell. A cell with < 30 tests or < 15 sessions in either arm is printed but flagged **underpowered**, not read.

**Ruling 3: decision rule.**
- **F7 is a CANDIDATE** for a new single-hypothesis CR on R-a if **the D2 pooled CI lower bound > 0** AND **the D3 EDGE-only pooled effect is ≥ 50 % of +8.22 (≥ 4.11 min) with CI lower bound > 0**.
- **Otherwise** F7 is logged as **"placebo artifact likely"**, and CR-CF closes **INCONCLUSIVE** for all families.
- The follow-up CR is not drafted here.

**Operational choices, fixed before the run:**
- **Running high / low:** the RTH (from 06:30 PT) maximum high and minimum low of the 1-s flow **through the test second inclusive**. EDGE if L ≥ running high − 2 or L ≤ running low + 2. "Beyond" is included, because L ≥ running high − 2 covers L above the high.
- **D3 for the rule** uses the Stage 1 placebo pool (kept tests, A1.1 as approved), because D3 is listed separately from D2. D3 on the restored pool is printed too, as a readout.
- **"Pooled"** (both approach sides) is the cell the rule reads. Below / above are printed.
- **Dropped tests are recovered** by running the unchanged `cr_cf_lib.run` under a family label that bypasses the A1.1 drop; dropped = recovered tests not in the Stage 1 kept set. `scripts/cr_cf_lib.py` is not modified.

## A2 result — F7 diagnostic (post-hoc, not a test; train; single run, 2026-10-25)

**Run.** `scripts/cr_cf_f7_posthoc.py`, commit `293efb3` (sha256 `a267234e…30a1dd25`), on the unchanged `cr_cf_lib.py`. One run, exit 0. JSON `scripts/logs/cr_cf/f7_posthoc.json` written before printing (git-ignored). Counts reproduced the Stage 1 run exactly (asserted): real 1,228, kept placebo 324, dropped 282. Kept-pool check: +8.22 [+2.00, +14.44], as Stage 1.

**Every number below is post-hoc, not a test.** Cells: tests real / placebo, sessions real / placebo. Session-weighted means in minutes (restricted mean time to escape 0.25·IM); diff [95 % CI].

| readout (post-hoc, not a test) | pooled | below | above |
|---|---|---|---|
| **D1** dropped − kept placebo (282 / 324; 71 / 76 sessions) | **+8.51 [−0.31, +17.35]** (55.86 vs 47.35) | +6.95 [−4.27, +18.63] | +8.20 [−1.80, +18.21] |
| Stage 1 check: real − kept placebo | +8.22 [+2.00, +14.44] (1,228 / 324; 87 / 76) | +8.29 [−0.37, +16.83] | +8.60 [−0.06, +16.83] |
| **D2** real − restored placebo | **+3.54 [−1.32, +8.46]** (1,228 / 606; 87 / 85) | +7.01 [+0.58, +13.53] (613 / 317) | +2.55 [−3.75, +8.66] (615 / 289) |
| **D3** EDGE, real − kept placebo | **+8.83 [+0.09, +17.50]** (291 / 203; 84 / 67) | +8.48 [−3.29, +20.35] (161 / 112; 67 / 40) | +9.87 [−0.26, +19.93] (130 / 91; 53 / 34) |
| D3 INTERIOR, real − kept placebo | +2.42 [−7.21, +11.58] (937 / 121; 86 / 52) | +5.18 [−9.06, +19.01] (452 / 61) | −1.80 [−14.21, +9.59] (485 / 60) |
| D3 EDGE, real − restored placebo | +4.88 [−1.95, +11.58] (291 / 298) | +4.77 [−4.69, +14.29] | +7.56 [−0.70, +15.60] |
| D3 INTERIOR, real − restored placebo | +3.31 [−3.28, +9.92] (937 / 308) | +11.00 [+1.56, +20.60] (452 / 153) | −3.23 [−11.41, +4.42] |
| **D4** first side: real − kept | +8.19 [+1.57, +14.88] (803 / 255) | +6.85 [−2.66, +16.43] | +11.24 [+1.26, +20.88] |
| D4 first side: real − restored (D2) | +4.31 [−0.70, +9.18] (803 / 388) | +2.57 [−5.20, +10.17] | +10.35 [+2.07, +18.59] |
| D4 first side: EDGE, real − kept (D3) | +8.83 [+0.09, +17.50] (291 / 203) | +8.48 | +9.87 |
| D4 first side: INTERIOR, real − kept (D3) | −1.39 [−16.43, +12.71] (512 / 52; 82 / 31) | −2.10 (27 placebo tests, 21 sessions) **underpowered** | −1.02 (25 / 18) **underpowered** |

**EDGE share** (post-hoc): real 0.237, kept placebo **0.627**, restored 0.492, dropped 0.337. On first-side tests: real 0.362, kept 0.796. Every real EDGE test is a first-side test: a second-side test comes only after price has crossed the level, so the level is interior then. That is why the D4 EDGE cells equal the all-test ones.

D1 distribution (post-hoc): dropped median 34.2 min (IQR 14.6–93.3), mean 50.4; kept median 29.1 (IQR 10.7–100.1), mean 49.3. Censored at 13:00: 53 / 48; at the 120-min cap: 51 / 66.

### Verdict (A2 ruling 3, applied as recorded)

- D2 pooled CI lower bound = **−1.32**, not > 0. **Fails.**
- D3 EDGE pooled = +8.83 (≥ 4.11) with lower bound +0.09 > 0. Would pass, but the rule needs both.
- **F7: placebo artifact likely. CR-CF closes INCONCLUSIVE for all families.**

**Reading (post-hoc, not a test):**
- **The dropped placebo tests are slow escapers.** Session-weighted they are +8.5 min slower than the kept ones, the same size as the Stage 1 F7 effect. The A1.1 drop rule did make the placebo faster.
- **Restoring them roughly halves the F7 effect** to +3.5, and its CI covers 0.
- **The kept placebo is dominated by EDGE tests** (63 %, against 24 % of real tests). Within EDGE the real − kept gap stays about +9, while INTERIOR is about +2.
- Both readings point to construction (which placebo tests survive, and where they sit in the range), not to swings holding price.
