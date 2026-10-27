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
