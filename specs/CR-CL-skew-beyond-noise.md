# CR-CL — Does a beyond-noise skew move predict the next move? (read-only, pre-registered, train only)

Spec freeze. Branch `feat/CR-CL-skew-beyond-noise` off `origin/main` `8f1226b` (PR #90 merge, CR-CK), in its own worktree (`.claude/worktrees/cr-cl`). Read-only against the database. No table writes, no cron, no deploy, no display change. PR, don't merge.

Source: the kickoff prompt (Ryan, 2026-10-26), transcribed as K1–K10, and Amendment A1 (Ryan, 2026-10-26, received before the spec was first committed, so it is folded into K6 here and recorded below). C1–C16 are choices Code fixed before any data was pulled.

---

## Why

CR-CK measured the stickiness ratio (A14) on 2026 train and left one question open: when put skew moves **beyond noise** over a swing-12 leg, does that say which way price goes next? CR-CK read no outcomes. This CR is the pre-registered, stop-free direction test (A15) of that question, on train only.

## Kickoff (Ryan, 2026-10-26)

| # | Decision | Value |
|---|---|---|
| K1 | Scope | Read-only, pre-registered, **train only**. Branch off `origin/main` (8f1226b); own worktree. Freeze with the conformance table; stop on any "deviates". |
| K2 | Population | Every swing-12 leg on 2026 train sessions (≤ 2026-06-05), from the CR-CK leg set; Databento ES bars; raw 0DTE PM put skew at quote time. **R-a, R-b and the holdout are not read.** |
| K3 | Skew reading per leg | CR-CK definitions: actual (trend), sticky-strike expected, ratio, beyond-noise flag. Legs ending after 11:00 PT: readout only. |
| K4 | Groups (beyond-noise legs only), split by leg direction (up / down) | **G1 "steepened against the move":** ratio < 0 → hypothesis: the next move goes **against** the leg. **G2 "relaxed as mechanics imply":** ratio > 1 → hypothesis: the leg's direction **continues**. |
| K5 | Baseline | In-noise legs matched on leg size (pts, tercile) and time of day (30-min bucket); report match coverage. |
| K6 | Outcome (as amended by A1) | Stop-free (A15, A11), from the leg's **confirmation bar close** (causal). **Primary = race to ±10 pts** in the hypothesised direction by 13:00 PT (ties, unresolved reported). **Readouts:** race to ±20; signed move at +30 and +60 min in the hypothesised direction; distribution of the maximum favourable move before −10 (share reaching +15 and +20 after +10). |
| K7 | Primary | Per group × direction (4 cells): hypothesised-direction race win rate minus the matched baseline's; session-cluster bootstrap, seed 20260920; Holm across the 4 cells. **Three verdicts (B1):** PASS if the Holm-adjusted CI lower bound > 0; REFUTED if the upper bound < 0; else INCONCLUSIVE. **≥ 40 legs per cell**, else inconclusive (underpowered). |
| K8 | Step 0 (no outcomes) | Counts per cell; beyond-noise share by time of day; baseline match coverage; A/A (baseline split in halves). **Stop for Ryan's review after Step 0.** |
| K9 | Readouts (not gated) | Swing 8 and 16; ratio cut at −0.5 / 1.5; legs into a Corrected GEX level (top 6, ≥ 40 $bn) vs not. |
| K10 | Delivery | Commit the run script before running; **one run**; results to the CR note, setup-registry row, session note. PR, don't merge. Plain-English summary: when skew moves beyond noise, does it tell you which way price goes next? |

## Pre-run choices fixed by Code (2026-10-26, before any data pull)

| # | Choice | Value |
|---|---|---|
| C1 | Sessions and loaders | CR-CJ period `y26` (2026-01-02 → 2026-06-05) through `scripts/cr_cj_lib.py`, which asserts every read inside the period and before 2026-06-05 21:00 UTC. Only `y26` is loaded: **R-a, 2025 and the holdout are never queried.** Skew = 0DTE PM `orats_monies_minute`, put skew (vol75 − vol50) × 100; ES = Databento outright minute bars from `es_flow_1s_databento`, the session's most-volume symbol (B5); a snapshot pairs with the ES close at its `quote_date` minute (CR-CJ A1). |
| C2 | Legs | `flow_features.swing_legs(H, L, swing)` on each session's Databento bars, swing 12 (primary). Skew window endpoints s = start_i + 1, e = end_i + 1 (snapshot-minute index; CR-CK C7 alignment: the snapshot at e carries the close of the extreme bar). Domain: s ≥ 07:00 PT (CR-CK). **Primary legs:** leg end (extreme bar) ≤ 11:00 PT. **Late legs** (end after 11:00, and ≤ 12:45 as CR-CK): readout only (K3). |
| C3 | Skew read | `skew_expected.stickiness_read` exactly as CR-CK Part 1 (`scripts/cr_ck_stickiness.py` `read`): trend actual (OLS slope × L, ≥ 80 % coverage, ≥ 5 points), sticky-strike expected from the start smile, ratio when \|ES move\| ≥ 0.15 % and expected ≠ 0, beyond noise = \|actual\| > band(L, start bucket). Band table = the shipped `NOISE_BAND_PP` (CR-CK, measured on these same sessions — disclosed, not re-fitted). |
| C4 | Groups | Among primary legs with a ratio and beyond noise: **G1** = ratio < 0, hypothesised direction h = −dir (against the leg); **G2** = ratio > 1, h = +dir (continuation). Beyond-noise legs with 0 ≤ ratio ≤ 1 belong to neither group (counted). Cells: G1-up, G1-down, G2-up, G2-down (leg direction). |
| C5 | Baseline pool | Primary legs with a ratio (\|ES move\| ≥ 0.15 %) and **in noise**. Legs with "move too small" or "insufficient data" are in neither group nor baseline (counted). |
| C6 | Match strata | (a) leg direction; (b) leg size tercile, cut points = terciles of \|end_px − start_px\| over **all primary legs with a ratio** (directions pooled, fixed in Step 0 from counts only); (c) 30-min bucket of the **confirmation bar's start time** (the outcome's clock), 07:00 … 12:30; (d) EDGE / INTERIOR at confirmation (C7; approved by Ryan 2026-10-26, B7). A group leg's baseline rate = the in-noise win rate in its stratum; the cell's baseline = the mean of those stratum rates over the cell's covered legs (exact stratification, group-leg weighted). **Coverage** = share of a cell's legs whose stratum has ≥ 1 resolved baseline leg; uncovered legs are dropped from the primary and counted. |
| C7 | EDGE / INTERIOR (approved, B7) | At the confirmation bar, inclusive: up leg EDGE if end_px ≥ running RTH high (from 06:30) − 2 pts; down leg EDGE if end_px ≤ running RTH low + 2. Else INTERIOR. Adapts CR-CF D3 (level → the leg's extreme). |
| C8 | Outcome start | P0 = close of the Databento bar confirm_i (the bar whose low / high confirmed the leg). Known at that bar's close; the skew window ends at snapshot e = end_i + 1 ≤ confirm_i, whose quotes (median lag 10 s in 2026) precede it. **Causal.** |
| C9 | Race (primary ±10, readout ±20) | Bars confirm_i + 1 … the 12:59 bar. Win = h × (price − P0) reaches +X (high / low) before −X. Both in the same minute → **tie**: excluded from the rate and counted. Neither by the 12:59 bar close → **unresolved**: excluded and counted. Win rate = wins / (wins + losses). Bars missing from the series are skipped; a missing run > 5 min inside the race drops the leg (counted). |
| C10 | Readouts on the move | Signed move h × (close(confirm_i + k) − P0) at k = 30, 60 min (NaN past the 12:59 bar; counted). MFE before −10 = the largest h-signed favourable excursion from P0 before the first −10 adverse touch (or 13:00); reported as quantiles, and among legs reaching +10, the shares reaching +15 and +20. |
| C11 | Inference | Session-cluster bootstrap, **10,000** draws, seed 20260920: resample sessions with replacement; recompute stratum rates and the cell difference. Two-sided bootstrap p = 2 × min(share ≤ 0, share ≥ 0). **Holm:** cells ranked by p; adjusted p_(j) = max_{i ≤ j} min(1, (5 − i) p_(i)); the cell at rank j gets the percentile CI at level 1 − 0.05/(5 − j). **PASS** = Holm CI lower bound > 0 and adjusted p < 0.05 (step-down); **REFUTED** = Holm CI upper bound < 0; else **INCONCLUSIVE**. |
| C12 | Minimum n | A cell needs **≥ 40 legs resolved and covered** at the run (race outcome excludes ties / unresolved), else INCONCLUSIVE (underpowered). Step 0 reports the pre-outcome count (legs in the cell and covered), which is an upper bound. |
| C13 | A/A (Step 0) | Outcomes are computed **only on baseline legs, only inside the splits**. 200 random session-level halves of the baseline pool, seed 20260920. Per split and direction, half A is scored as if it were a group against half B as baseline with the C6 matching and C9 ±10 race, for both h = +dir and h = −dir. Reported: median and 2.5 / 97.5 % of the split differences, and the share of splits whose 500-draw session bootstrap 95 % CI excludes 0 (expected ≈ 5 %). No group outcome is computed or printed in Step 0. |
| C14 | Readouts K9 | Each re-runs C2–C11 unchanged except: **swing 8 / 16** (legs from `swing_legs` at 8 / 16; ratio still needs \|move\| ≥ 0.15 %, so many swing-8 legs drop as "move too small"; terciles re-cut per swing); **ratio cuts** G1 ratio < −0.5, G2 ratio > 1.5; **GEX**: leg "into a level" if \|end_px − L\| ≤ 5 pts for any of the session's Corrected levels, split put-level (net < 0) vs call-level (net > 0) (A8). Levels = CR-CF `select_levels` (CR-BT placement, P_target = `ironbeam_es_1m_bars` 15:45 ET close on the prior session, ≤ 30 DTE, gross ≥ 40 $bn, 5-pt spacing, within ±1·IM of the Databento RTH open, top 6), copied, not imported (`cr_cf_lib` loads all flow at import). Sessions without levels are counted. **Late legs** (C2) as their own readout. Readouts are reported with unadjusted 95 % CIs and no verdict. |
| C15 | Run discipline | `scripts/cr_cl_step0.py` (Step 0) and `scripts/cr_cl_run.py` (the run) committed before they are executed; the run executes **once**. A crash before any number is printed or saved allows a fix-and-rerun (CR-BY precedent), logged. Outputs `scripts/logs/cr_cl/` (git-ignored). |
| C16 | Connection | `get_backfill_db_conn()` via the CR-CJ loader, `read_only = True`. SELECTs only. |

## Steps

- **Step 0** (K8, C13): counts per cell (swing 12 primary; late legs; swing 8/16), funnel (move too small / insufficient / in neither group), beyond-noise share by 30-min bucket, size terciles, baseline match coverage per cell, EDGE share per cell and baseline, A/A. **Stop for Ryan's review.**
- **Step 1** (run): `scripts/cr_cl_run.py`, committed before it runs; one run; tables below.
- **Wrap**: vault CR note, setup-registry row, session note, PR.

## Standing rulings conformance (at spec freeze, re-checked after A1)

| Ruling | Status | Note |
|---|---|---|
| A1, A2, A4, A5, A9, A10, A13, A18, A19, A20 | n/a | No levels traded, ranges, stops, entries or sizing; this is a direction test on legs. |
| A3 | honoured | No entry is modelled. The outcome clock starts at the leg's confirmation bar close, not at a touch. |
| A6 | honoured | G1 is A6's divergence in ratio units: on an up leg ratio < 0 = put skew rose (bearish → against); on a down leg = put skew fell (bullish → against). |
| A7 | honoured | Swing 12 primary; 8 / 16 readouts. |
| A8 | honoured | The GEX readout splits put levels and call levels. |
| A11 | **honoured (after A1)** | ±10 race primary, ±20 readout, favourable-excursion distribution reported (A11 as rewritten 2026-10-26). |
| A12 | honoured | Corrected placement, gross ≥ 40 $bn, top 6 (readout only). |
| A14 | honoured | CR-CK's definitions, functions and shipped noise band, unchanged. |
| A15 | honoured | Stop-free direction test; no stops, no exits tuned. |
| A16 | honoured | Trend Δ over the leg, not endpoints. |
| **A17** | **deviates — approved by Ryan 2026-10-26** | A17: "Skew alone is never the trigger … **Test order flow alone first**; skew is an add-on readout." This CR's primary tests skew **alone** as a direction predictor, and order flow alone has not yet been tested in this frame (the level-agnostic range / flow CR, CR-CG, has not run). Nothing here makes skew a trigger, but the sequencing differs. **Proposed:** Ryan approves running this as a measurement of skew's information content (no trigger, no setup shortlisted from it), logged in standing-rulings. |
| B1 | honoured | Three verdicts, wide CI = inconclusive, n < 40 = inconclusive (underpowered). |
| B2 | honoured | All readouts (swing 8/16, ratio cuts, GEX, late legs, ±20, moves, MFE) disclosed and unadjusted; nothing chosen from them. |
| B3 | n/a | Not a setup optimisation. |
| B4 | honoured | `y26` train only; R-a, R-b and holdout not read (loader asserts). |
| B5 | honoured, with a note | Databento bars for legs and outcomes. The GEX readout's P_target uses `ironbeam_es_1m_bars` at 15:45 ET (the CR-BT definition, as in CR-CE/CF). |
| B6 | honoured | Labelled: a stop-free direction test, **not a test of Ryan's trade**. |
| **B7** | **deviates as kicked off — resolved by Ryan 2026-10-26 (EDGE / INTERIOR added, C6d)** | B7: "Placebos are matched on where price sits in the session range (EDGE vs INTERIOR)." K5's baseline matches size and time only. A leg that just made a session extreme vs one inside the range plausibly differs on continuation, so the comparison could be confounded. **Proposed:** add EDGE / INTERIOR as a third match key (C6d, C7), and report the EDGE share per cell and baseline. Alternative: keep K5 as written and report EDGE / INTERIOR as a readout split only. |
| B8 | n/a | No labels. |
| C1, C2 | n/a | No display. |

Two "deviates" (A17, B7) stopped the freeze; both approved by Ryan in chat 2026-10-26 and logged in the standing-rulings change log (see Approvals below).

---

## Amendment A1 (Ryan, 2026-10-26, before Step 0 and before the run script is committed)

Kickoff, transcribed: replace the Outcome line (standing ruling A11 rewritten 2026-10-26: target size is open). Outcome (stop-free, A15, A11), from the leg's confirmation bar close (causal): **primary = race to ±10 pts** in the hypothesised direction by 13:00 PT (ties, unresolved reported); **readouts** = race to ±20, signed move at +30 and +60 min, and the distribution of the maximum favourable move before −10 (share reaching +15 and +20 after +10). Everything else unchanged.

Folded into K6, C9, C10, C12, C13 above (the frozen text above is the amended version; there was no earlier committed version). Conformance re-checked: A11 honoured; no other row changed.

---

## Approvals (Ryan, 2026-10-26, after spec freeze `2285be3`, before any data)

1. **A17:** approved as a **measurement** of what skew alone tells you about the next move. **No setup is shortlisted from CR-CL. A PASS means only "skew earns a place as an add-on readout in the flow-first setup test", nothing more.** The verdict section must say so.
2. **B7:** EDGE / INTERIOR is a third baseline match key (C6d, C7: leg extreme within 2 pts of the session's running RTH high / low at confirmation, as CR-CF D3). Match coverage is reported; **a cell that drops below 40 because of the extra key is reported as underpowered; the match is not relaxed.**
3. C-choices confirmed as written (ties / unresolved excluded and counted; 10,000 draws; CR-CK noise band disclosed).

Both logged in `Dash/research/standing-rulings.md` change log. Next: Step 0, then stop.
