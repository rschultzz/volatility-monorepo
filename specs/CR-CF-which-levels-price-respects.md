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
