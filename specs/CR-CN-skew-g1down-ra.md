# CR-CN — Confirm CR-CL G1-down on R-a (read-only, pre-registered, single hypothesis)

Spec freeze. Branch `feat/CR-CN-skew-g1down-ra` off `origin/main` `1d9c71a` (PR #92 merge, CR-CL), in its own worktree (`.claude/worktrees/cr-cn`). Read-only against the database. No table writes, no cron, no deploy, no display change. PR, don't merge.

Source: the kickoff prompt (Ryan, 2026-10-26), transcribed as K1–K9. N1–N10 are choices Code fixed before any R-a data was pulled. Everything not listed here is CR-CL as shipped (`specs/CR-CL-skew-beyond-noise.md`, K1–K10, C1–C16, A1, A2, Approvals).

---

## Why

CR-CL's single primary, G1-down (a swing-12 decline leg during which raw 0DTE PM put skew **fell** beyond noise → next move up; Ryan's put-monetization theory), came out **INCONCLUSIVE, leaning toward** on 2026 train: ±10 race +17.4 pp [−1.2, +37.3], n 47. This CR re-runs exactly that test, unchanged, on R-a (2024-04-03 → 2024-12-31), the confirmation period (B4), where no price outcome has been read for this question.

## Kickoff (Ryan, 2026-10-26)

| # | Decision | Value |
|---|---|---|
| K1 | Scope | Read-only, pre-registered, single hypothesis. Branch off `origin/main` after #92 merged; own worktree. Freeze with the conformance table. |
| K2 | Hypothesis | **Exactly CR-CL G1-down as run, no changes:** decline swing-12 leg, raw 0DTE PM put skew fell beyond noise (ratio < 0; CR-CK definitions and noise band **as shipped, not re-measured**) → next move **UP**. |
| K3 | Outcome | Same as CR-CL: race ±10 from the confirmation bar close, by 13:00 PT, ties / unresolved excluded and counted. |
| K4 | Baseline, inference | Same 3-key matched in-noise baseline; same session-cluster bootstrap (seed 20260920). Legs ending after 11:00 PT: readout only. |
| K5 | Code | Reuse CR-CL's scripts with the date range switched; **no other change**. |
| K6 | Data | **R-a only** (2024-04-03 → 2024-12-31), ORATS aligned on **quote time** (CR-CJ A1). R-a was read for skew vs price only (CR-CJ); no price outcomes have been read on it for this question. **Train, R-b (2025) and the holdout are not read.** |
| K7 | Verdict | **PASS** if the 95 % CI lower bound > 0 (one hypothesis, no Holm); **REFUTED** if the upper bound < 0; else **INCONCLUSIVE**; 40-leg minimum (else INCONCLUSIVE, underpowered). |
| K8 | Readouts | Same as CR-CL: ±20 race, +30 / +60 min move, favourable-move distribution, GEX-level split, stricter ratio cut (−0.5). |
| K9 | Step 0 (no outcomes) | Leg counts, G1-down count, baseline match coverage, A/A, and the **2024 noise vs the shipped band (report only — the band is NOT changed)**. **Stop for Ryan's review after Step 0.** Then: results to the CR note, setup-registry row, session note. PR, don't merge. Plain-English summary: did the put-monetization bounce show up in 2024? |

## Pre-run choices fixed by Code (2026-10-26, before any R-a data was pulled)

| # | Choice | Value |
|---|---|---|
| N1 | Sessions and loaders | CR-CJ period `ra` (2024-04-03 → 2024-12-31) through `scripts/cr_cj_lib.py`: NYSE trading days minus the early closes 2024-07-03, 11-29, 12-24 (as CR-CJ). The loader's `guard` asserts every read sits inside one period and never in 2025. Only `ra` is loaded: **2026 train, 2025 and the holdout are never queried by this CR.** No R-a parquet cache exists, so the loads are fresh SELECTs (cached to `scripts/logs/cr_cj/{skew,es}_ra.parquet`, git-ignored). |
| N2 | The period switch (K5) | `scripts/cr_cl_lib.py` gains one constant, `PERIOD = os.environ.get("CR_CL_PERIOD", "y26")`, used where `"y26"` was hard-coded (`load`, `gex_levels`), and `OUT` = `scripts/logs/cr_cl/` for `y26`, `scripts/logs/cr_cn/` for `ra`. `scripts/cr_cl_step0.py`'s `PERIODS["y26"]` assert reads `PERIODS[C.PERIOD]`. Default `y26` leaves CR-CL unchanged; **parity check:** CR-CL Step 0 re-run on `y26` must reproduce its `step0.json` exactly. `scripts/cr_cl_run.py`'s CR-CK parity assert (1,233 legs / 764 ratio-defined) is y26-specific; under `ra` it asserts the CR-CN Step 0 counts instead (one line, committed with this CR's Step 0 findings, before the run). No other line of the CR-CL code changes. |
| N3 | Skew alignment on R-a | As CR-CJ A1 in `cr_cj_lib.skew_minutes`: a snapshot pairs with the ES close at its `quote_date` minute; snapshots whose quotes are > 5 min from the label are dropped. R-a quotes lag the label by a median 129 s (field-conventions), so the skew trend window [s, e] (snapshot labels start_i + 1 … end_i + 1, CR-CK C7) holds quotes ≈ 2 min older than the leg it reads, and px[s], px[e] are the ES closes at those quote minutes, not at the leg's start / extreme. Causality (C8) holds with more margin. Unchanged by K5; disclosed. Step 0 reports the snapshot drop share and the quote lag at e on the primary legs. |
| N4 | Noise band (K2, K9) | `skew_expected.NOISE_BAND_PP` as shipped (measured on 2026 train by CR-CK C6) decides beyond-noise. Step 0 measures the 2024 band with the CR-CK C6 method (SD of the endpoint Δ put skew by window length × 30-min start bucket, starts on a 5-min grid inside 07:00–12:45, ≥ 30 windows per cell, unmeasured lengths carry the longest measured value), code copied from `scripts/cr_ck_stickiness.py` (that script loads `y26` at import). Reported as a table and as the ratio 2024 / shipped. **Report only: the band is not changed and no count under a 2024 band is computed.** |
| N5 | Size terciles | CR-CL C6b's code computes the cut points from the loaded legs (primary legs with a ratio, directions pooled), so under K5 they are **re-cut on R-a**. Step 0 reports them beside CR-CL's train cuts (20.0 / 28.5 pts). |
| N6 | Primary | CR-CL A2.1 unchanged: G1-down, h = up, swing 12, primary legs (end ≤ 11:00 PT), baseline = in-noise down legs, strata direction × size tercile × confirmation 30-min bucket × EDGE / INTERIOR; ±10 race; group win rate − matched baseline win rate; 95 % percentile CI from 10,000 session-cluster draws, seed 20260920; ≥ 40 legs resolved and covered. |
| N7 | Readouts | CR-CL A2.3 / A2.4 unchanged, all from the same `cr_cl_run.py`: ±20 race, +30 / +60 min move, MFE before −10 for every cell; G1-up, G2-up, G2-down; ratio cut −0.5 / 1.5 (the "stricter cutoff"); swing 8 / 16; late legs; G1-down by GEX tag. Unadjusted 95 % CIs, no verdicts. K8 does not name swing 8 / 16 or the other three cells; they come out of the unchanged script and are reported as readouts. |
| N8 | GEX readout on R-a | `gex_levels` unchanged: `orats_oi_gamma` (R-a dates), canonical `implied_move_1d`, P_target = `ironbeam_es_1m_bars` close of the 15:45 ET bar on the prior session. In 2024 that table is Databento `ES.c.0` (calendar roll, dying contract in roll weeks; open-question `databento-2025-backfill-calendar-roll-dying-contract` covers 2025 but the same fetch applies). `gex_levels` already drops sessions whose prior session has a different Databento front symbol, and 2024-04-03 has no prior R-a session (no P_target). Sessions without levels are counted. |
| N9 | A/A | CR-CL C13 unchanged (200 session halves of the baseline pool, 500-draw bootstrap each, ±10 race on baseline legs only). |
| N10 | Run discipline, connection | CR-CL C15 / C16: Step 0 = `scripts/cr_cn_step0.py` (runs `cr_cl_step0.py` under `ra` for counts, coverage, A/A; adds the N3 lag and N4 band measurements), committed before it runs. The run = `CR_CL_PERIOD=ra apps/web/.venv/bin/python scripts/cr_cl_run.py`, **once**. A crash before any number is printed or saved allows a fix-and-rerun, logged. `get_backfill_db_conn()`, `read_only = True`, SELECTs only. Outputs `scripts/logs/cr_cn/` (git-ignored). |

## Steps

- **Step 0** (K9): parity check of the period switch on `y26`; R-a funnel, cells, G1-down count, coverage, EDGE share, terciles, GEX counts, A/A; quote lag; 2024 noise band vs shipped. **Stop for Ryan's review.**
- **Step 1** (run): `cr_cl_run.py` under `ra`, parity assert set from Step 0, committed before it runs; one run.
- **Wrap**: vault CR note, setup-registry row, session note, PR.

## Standing rulings conformance (at spec freeze)

| Ruling | Status | Note |
|---|---|---|
| A1, A2, A4, A5, A9, A10, A13, A18, A19, A20, A21 | n/a | No levels traded, ranges, stops, entries, second touches or sizing; a direction test on legs. |
| A3 | honoured | No entry modelled; the clock starts at the leg's confirmation bar close. |
| A6 | honoured | G1-down is A6's "price down with put skew falling = bullish". |
| A7 | honoured | Swing 12 primary; 8 / 16 readouts. |
| A8 | honoured | GEX readout splits put vs call levels. |
| A11 | honoured | ±10 race primary, ±20 readout, favourable-excursion distribution. |
| A12 | honoured, with a note | GEX readout uses CR-CL's population (Corrected placement, gross ≥ 40 $bn, top 6), which is A12's "old ≥ 40 $bn gross top-6" readout, not the yellow / darkest primary. Unchanged by K5; readout only. |
| A14 | honoured | CR-CK definitions and shipped band, unchanged (K2). The band was measured on 2026; Step 0 reports the 2024 band for information only (N4). |
| A15 | honoured | Stop-free direction test. |
| A16 | honoured | Trend Δ over the leg. |
| **A17** | **deviates — needs Ryan's approval** | Same deviation as CR-CL: skew alone as a direction predictor before the flow-first test has run. CR-CL's approval (2026-10-26) was given for CR-CL. **Proposed:** the same terms — a **measurement** of what skew alone says about the next move; no setup shortlisted; a PASS means only that G1-down earns a place as a pre-registered add-on readout in the flow-first setup test. |
| B1 | honoured | Three verdicts; < 40 = inconclusive (underpowered). |
| B2 | honoured | Readouts disclosed, unadjusted, nothing chosen from them. |
| B3 | n/a | Not an optimisation. |
| B4 | honoured | R-a used for confirmation; train, R-b, holdout not read (loader asserts). R-a has been read before by CR-CJ (skew vs price) and CR-CB A17 (price paths at GEX levels, a different question); no price outcome of a skew-leg rule has been read on it. |
| B5 | honoured, with a note | Databento bars for legs and outcomes; GEX P_target uses `ironbeam_es_1m_bars` (N8). |
| B6 | honoured | Labelled: a stop-free direction test, not a test of Ryan's trade. |
| B7 | honoured | EDGE / INTERIOR is the third match key (CR-CL approval 2). |
| B8 | n/a | No labels. |
| C1, C2 | n/a | No display. |

One "deviates" (A17). Step 0 reads no outcomes on group legs, so Code proceeds to Step 0 and stops there; the run waits on Ryan's A17 approval and his Step 0 review.

---

## Amendment A1 (Code, 2026-10-26, before any data was pulled): N2 parity check without reading train

N2's parity check ("CR-CL Step 0 re-run on `y26` must reproduce its `step0.json`") would load 2026 train, which K6 forbids in this CR. Replaced by a **code-level parity check**: the diff to `cr_cl_lib.py` / `cr_cl_step0.py` replaces only the literal `"y26"` (5 sites) by `PERIOD`, whose default is `"y26"`, plus the `OUT` switch (`cr_cl` under `y26`); a test asserts that under the default the module resolves `PERIOD == "y26"` and `OUT == scripts/logs/cr_cl`, and that no `"y26"` literal remains outside the default. No train data is loaded. Everything else unchanged.

---

## Step 0 findings (2026-10-26, `scripts/cr_cn_step0.py` @ `a37218e` → `scripts/logs/cr_cn/step0.json`, `step0_extra.json`; no group outcome computed)

R-a only: **186 sessions, 2024-04-03 → 2024-12-31**, none missing (the loader's `guard` held; 2026 train, 2025 and the holdout were not loaded). Loads were fresh SELECTs (no R-a cache existed). One launch, no errors. A1 parity held: under the default the module resolves `PERIOD = y26`, `OUT = logs/cr_cl`, and no `"y26"` literal remains outside the default.

### Funnel, swing 12, primary legs (end ≤ 11:00 PT) — R-a vs CR-CL train

| Step | R-a (186 sessions) | Train (107 sessions) |
|---|---|---|
| Legs in domain | 1,036 | 1,233 |
| Primary legs | 681 (late 355) | 972 (late 261) |
| Ratio defined | 399 (move too small 222; insufficient 55; no read 5) | 597 (219; 155; 1) |
| In noise → baseline pool | 324 (up 152, down 172) | 448 (227, 221) |
| Beyond noise | 75 (18.8 % of ratio-defined): **G1 25**, G2 28, neither 22 | 149 (25.0 %): G1 76, G2 45, neither 28 |

Size tercile cuts (N5, re-cut on R-a): **20.92 / 29.5 pts** (train 20.0 / 28.5).

R-a has 74 % more sessions than train but **fewer legs** (1,036 vs 1,233). Per session: 3.7 primary legs vs 9.1 on train, and 2.1 ratio-defined vs 5.6. 2024 moved less: swing-12 legs are rarer and a larger share is "move too small".

### Cells (pre-outcome; covered = the stratum has ≥ 1 baseline leg)

| Cell | Legs | Sessions | Covered | EDGE share | Can reach 40? |
|---|---|---|---|---|---|
| G1-up | 14 | 13 | 13 | 64 % | no |
| **G1-down (primary)** | **11** | **11** | **11** | 82 % | **no — the primary is INCONCLUSIVE (underpowered) before any outcome** |
| G2-up | 14 | 13 | 14 | 43 % | no |
| G2-down | 14 | 12 | 14 | 50 % | no |
| Baseline up / down | 152 / 172 | | | 47 % / 62 % | |

- **The primary cannot carry a verdict.** 11 legs against a 40-leg minimum (train had 49). Under K7 the run would return INCONCLUSIVE (underpowered) whatever the outcomes.
- The EDGE key costs nothing here (coverage without it: 11 / 11). Baseline strata: 108 occupied, median 3 legs, 23 with one leg.
- Every readout cell is also small: ratio cut −0.5 G1-down 7; swing 8 G1-down 14; swing 16 G1-down 7; late G1-down 1. Pooling swings is not in the spec (CR-CL "never pooled").
- GEX: levels on 172 / 186 sessions (median 6; 10 without P_target, 4 without IM). G1-down legs: put level 4, call 1, none 6.

### Beyond-noise share by 30-min start bucket (ratio-defined legs, primary + late)

| Start | 07:00 | 07:30 | 08:00 | 08:30 | 09:00 | 09:30 | 10:00 | 10:30 | 11:00 | 11:30 | 12:00 | 12:30 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n | 105 | 73 | 81 | 57 | 46 | 43 | 45 | 36 | 44 | 41 | 29 | 6 |
| Beyond | 15 % | 10 % | 24 % | 25 % | 28 % | 5 % | 24 % | 22 % | 11 % | 10 % | 0 % | 17 % |
| G1 / G2 | 7 / 5 | 3 / 3 | 4 / 11 | 7 / 3 | 3 / 5 | 0 / 0 | 1 / 7 | 1 / 3 | 0 / 4 | 1 / 2 | 0 / 0 | 0 / 1 |

Train's early-morning G1 cluster (07:00–07:59: 31 legs) is mostly absent (10 on R-a).

### Quote lag (N3)

- All 72,709 R-a 0DTE PM RTH rows: label − `quote_date` median **129 s** (p5 79, p95 179); 0.3 % > 5 min (dropped). Matches field-conventions.
- At the leg's end snapshot e: median 108 s on ratio-defined primary legs (G1-down 85 s; baseline down 111 s).
- **Consequence:** the ES price paired with snapshot e is the close **1–4 bars before the leg's extreme bar** (1: 110 legs, 2: 237, 3: 51, 4: 1), against 0 in 2026. So on R-a the skew window and its ES move both stop about 2 min short of the leg's extreme. This is CR-CJ A1 working as specified (skew and price stay paired), but it means the R-a "leg" read is the leg minus its last ~2 minutes, which lowers |ES move| (more "move too small") and can change the ratio. K5 forbids changing it; disclosed.

### 2024 noise band vs the shipped band (N4; report only — the band was not changed)

SD of the endpoint Δ put skew, vol pts, CR-CK C6 method on R-a, and its ratio to the shipped (2026) band:

| Start | 5 min | 15 | 30 | 60 | 120 | ratio 5 / 15 / 30 / 60 / 120 |
|---|---|---|---|---|---|---|
| 07:00 | 0.62 | 0.69 | 0.79 | 0.84 | 0.96 | 0.85 / 0.61 / 0.62 / 0.78 / 0.95 |
| 07:30 | 0.55 | 0.68 | 0.77 | 0.89 | 1.02 | 0.62 / 0.54 / 0.65 / 0.73 / 0.73 |
| 08:00 | 0.63 | 0.65 | 0.76 | 0.81 | 0.85 | 0.96 / 0.76 / 0.87 / 0.92 / 0.66 |
| 08:30 | 0.63 | 0.70 | 0.73 | 0.86 | 0.99 | 1.10 / 0.85 / 0.89 / 0.73 / 0.93 |
| 09:00 | 0.52 | 0.61 | 0.70 | 0.85 | 1.03 | 0.90 / 0.77 / 0.63 / 0.72 / 0.81 |
| 09:30 | 0.53 | 0.67 | 0.69 | 0.89 | 1.26 | 0.58 / 0.46 / 0.51 / 0.67 / 0.48 |
| 10:00 | 0.57 | 0.68 | 0.81 | 0.87 | 1.30 | 0.63 / 0.64 / 0.68 / 0.62 / 0.43 |
| 10:30 | 0.71 | 0.86 | 0.91 | 1.37 | 1.77 | 0.90 / 0.91 / 0.69 / 0.54 / 0.39 |
| 11:00 | 0.75 | 0.88 | 1.17 | 1.32 | 1.81 | 0.52 / 0.34 / 0.47 / 0.44 / 0.39 |
| 11:30 | 1.09 | 1.43 | 1.39 | 1.80 | 1.80 | 0.37 / 0.47 / 0.38 / 0.34 / 0.34 |
| 12:00 | 1.04 | 1.23 | 2.00 | 3.21* | 3.21* | 0.33 / 0.24 / 0.40 / 2.19 / 2.19 |
| 12:30 | 1.73 | 2.78 | 2.78* | 2.78* | 2.78* | 0.32 / 0.31 / 0.31 / 0.31 / 0.31 |

\* carried (C6 clamp). **2024 noise is lower than the shipped band:** median ratio 0.71 (IQR 0.61–0.85) over 07:00–10:30; after 11:00 it is a third of 2026's. So the shipped band is **conservative** on R-a (fewer legs flagged beyond noise than a 2024-measured band would flag). R-a median 0DTE put skew level 1.87 vol pts.

### A/A (N9; ±10 race on baseline legs only, 200 session halves, 500-draw bootstrap each)

| Direction scored | Up legs: median [2.5, 97.5 %] | CI excludes 0 | Down legs: median [2.5, 97.5 %] | CI excludes 0 |
|---|---|---|---|---|
| Continuation | −1.4 pp [−21.6, +23.8] | 2.0 % | +1.4 pp [−19.0, +17.6] | 1.0 % |
| Against | +0.8 pp [−21.8, +21.7] | 2.0 % | −1.2 pp [−19.1, +20.9] | 3.5 % |

Unbiased (medians within 1.5 pp of 0), slightly conservative (1–3.5 % vs nominal 5 %), wider than train (±20 pp on ~80-leg halves).

### Gate summary for Ryan

- **No data gate failed.** Sessions complete, loader guard held, coverage 93–100 %, A/A clean.
- **The primary is underpowered before any outcome:** G1-down has **11 legs** on R-a (train 49). By K7 the run can only return INCONCLUSIVE (underpowered). Every readout cell is ≤ 14 legs.
- Two R-a differences drive the low count, both outside K5's reach: 2024 has fewer and smaller swing-12 legs, and the 2-min quote lag cuts the last ~2 minutes off each leg's skew / price read.
- The shipped band is wider than 2024's own noise (≈ 0.7×), so it under-flags on R-a. Not changed (K2).
- **A17 approval is still pending.**

Stopped for Ryan's review. No run script change has been made; the run has not been executed.

---

## Ruling (Ryan, 2026-10-26, after Step 0): closed at Step 0, not run

Transcribed:
- **Close CR-CN at Step 0. Do not run.**
- **A17 deviation approved for the record** on the CR-CL terms (measurement only, nothing shortlisted). Moot, since nothing runs.
- **R-a outcomes for this question stay unread.**
- **Verdict: not testable on R-a.** G1-down has 11 legs on R-a (vs 49 on train), because 2024 moved less and the 2024 quote lag cuts off the end of each leg.
- **Keep** A1 and the date-range switch in `cr_cl_lib.py` / `cr_cl_step0.py` (harmless; defaults to train).

`scripts/cr_cl_run.py` was never executed under `ra`, and its parity assert was not changed. No group outcome was computed on R-a. Logs are copied to the main checkout's `scripts/logs/cr_cn/` (git-ignored, machine-local).

Setup registry, G1-down: **train: inconclusive +17 [−1, +37]; R-a: not testable.**

Conformance re-checked after A1 and this ruling: no row changes; A17 now "deviates — approved for the record by Ryan 2026-10-26 (moot, not run)". Revoked / superseded R1–R7: none used. Process note: the frozen table had A17 as "deviates" and Step 0 ran before Ryan's approval, on the grounds that Step 0 computes no group outcome (disclosed at freeze). Under the CLAUDE.md "Standing rulings" rule (merged after this freeze, #94) a "deviates" is a stop before any further step.
