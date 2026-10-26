# CR-CJ — Calibrate the Skew Expected Curve

Spec freeze. Branch `feat/CR-CJ-skew-expected-calibration` off `origin/main` `1c71c44` (PR #87 merge, CR-CF), in its own worktree (`.claude/worktrees/cr-cj`). Read-only against the database. No table writes, no cron, no frontend change, no deploy. PR, don't merge.

Source: vault `Dash/CRs/CR-CJ — Skew Expected Calibration.md`. The kickoff prompt (Ryan, 2026-10-26) is transcribed as K1–K9. C1–C14 are choices Code fixed before any data was pulled.

---

## Why

Standing ruling A14: skew is read **relative to what is expected for the price move**. The Skew table's "Expected SS" curve (`apps/web/modules/Skew/callbacks.py`) is uncalibrated and must be fixed before it decides a test. CR-CI Step 0 (`f487f63`) measured the bias: on 569 swing-12 approach legs, the residual read "divergent" on **74 %** of legs vs **43 %** for raw put skew, roughly constant across leg sizes and both directions. CR-CI A2 was superseded by A17 and paused until this curve is fixed.

### What the current curve does (code read, `1c71c44`)

For a previous snapshot `prev` and the current one `now`, on the selected expiry:

1. Previous smile in log-moneyness, k-grid from `prev` vol50 and T_prev (delta → k by `k_for_abs_delta`).
2. Sticky-strike shift: k_shift = log(S_now / S_prev), with S = `stock_price` (SPX cash).
3. Expected ATM = smile_prev(k_shift) + clip(−4.5 × ret %, ±6) vol pts + (−638) × (√T_prev − √T_now) vol pts.
4. Expected 25Δ put = smile_prev(k_p25(now) + k_shift) + the **same** vertical shift, where k_p25(now) uses **now**'s vol50 and T_now.
5. Expected put skew = expected 25Δ put − expected ATM.

**The 4.5 beta and the −638 theta cancel out of expected skew** (step 4 adds the same shift as step 3). Expected skew is sticky strike plus a remap: as T and vol50 change, the 25Δ put's k moves, and the old smile is read at the new k. The beta and theta only affect expected ATM (the "ATM residual" column).

## Kickoff (Ryan, 2026-10-26)

| # | Decision | Value |
|---|---|---|
| K1 | Scope | Read-only. **No price-outcome data**: only skew vs contemporaneous ES price. |
| K2 | Data | Train only (≤ 2026-06-05), from 2024-04-03 if 0DTE PM coverage allows; report coverage. No holdout. **No R-b (2025)**: if 2025 is needed for coverage, stop and ask. |
| K3 | Measures | Raw 0DTE PM put skew (vol75 − vol50) and ATM vol50 from `orats_monies_minute`; ES 1-min close (`ironbeam_es_1m_bars` / Databento). |
| K4 | Step 1, noise | SD of 1-min skew changes by time of day. |
| K5 | Step 2, fit | Change in put skew vs ES % move (and vs time to expiry), over 5/15/30-min windows and over swing-12 legs. OLS with session-clustered SEs, by time of day and by price-up vs price-down. Report whether a single beta suffices. |
| K6 | Step 3, compare | Implied beta of the 4.5/−638 model vs the fitted one. |
| K7 | Step 4, balance | On random 15–30-min windows and on swing-12 legs, the share of "divergent" under (a) the current curve, (b) the fitted curve, (c) raw. Target ≈ 50 % for (b). |
| K8 | Step 5, out of sample inside train | Fit on 2024-04 → 2025-?? if allowed, else on the first half of the 2026 train; check balance on the other half. |
| K9 | Deliverable | Recommended expected-curve parameters + a shared function in `packages/shared` (pure, unit-tested), **not** wired into the Dash chart. Close the open question with the numbers. PR, don't merge, no deploy. Plain-English summary for Ryan. |

## Pre-run choices fixed by Code (2026-10-26, before any data pull)

| # | Choice | Value |
|---|---|---|
| C1 | Sample | Train sessions in **R-a 2024-04-03 → 2024-12-31** and **2026-01-02 → 2026-06-05**. 2025 is never queried (K2 is met without it; Step 0 reports the counts). Every DB read is asserted inside those two date ranges and before 2026-06-05 21:00 UTC. Early closes are excluded, because the 0DTE clock (16:00 ET expiry) is wrong on them. |
| C2 | Skew rows | `orats_monies_minute`, ticker SPX, `expir_date = trade_date`, `expiry_tod = 'pm'` (field-conventions: select 0DTE by expiry date, not `dte`). Snapshots 06:30 ≤ `snapshot_pt` < 13:00 PT. Put skew = (vol75 − vol50) × 100 vol pts (vol75 = 25Δ put). ATM = vol50 × 100. Step 0 reports per-session minute coverage and third-Friday PM availability. |
| C3 | ES price | **Databento** (B5): `es_flow_1s_databento`, RTH, the session's most-volume outright symbol (asserted one per session), built into 1-min bars as CR-CI `make_bars` (first open, max high, min low, last close). No `ES.c.0` and no roll-week artifacts. Cross-check in Step 0: close against `ironbeam_es_1m_bars` on a sample of 2026 minutes. |
| C4 | Alignment | Snapshot labelled minute T (quotes 0–20 s before T, field-conventions `quote_date`) pairs with the ES bar **starting T − 1 min** (close at T). This is CR-CI C17's `snap_at` (latest snapshot ≤ the bar's close) with zero tolerance. A window needs both endpoint snapshots and both ES closes. |
| C5 | Time of day | 30-min buckets by the window's start, 06:30 → 13:00 PT. **Fit domain** 07:00 ≤ start, end ≤ 12:45 PT (open-time ORATS quirks before ~06:50 PT, CR-BP; 0DTE degenerates as T → 0). 06:30–07:00 and 12:45–13:00 are reported as readouts and are outside the recommended curve's domain. |
| C6 | Step 1 noise | SD of 1-min Δ put skew and Δ ATM by 30-min bucket, pooled over sessions. Also the variance ratio VR(h) = Var(h-min Δ) / (h · Var(1-min Δ)) for h = 5, 15, 30: VR < 1 means the 1-min changes are partly measurement noise that reverses. |
| C7 | Step 2 windows | Non-overlapping 5/15/30-min windows on a grid from 07:00 PT; swing-12 legs = `packages.shared.flow_features.swing_legs(H, L, 12)` on the RTH Databento bars (the Flow pane's legs, A7), endpoints at the legs' `start_i` / `end_i` bars, fit domain as C5. Model, vol pts: **Δskew = α_b + β · r**, r = 100 · (C_now / C_prev − 1) ES %, α_b one per 30-min start bucket (absorbs the time-to-expiry drift, which is fixed for a given bucket and length). Variants: β per bucket (β_b), β_up / β_down by sign of r, and a time term **γ · Δ√T** (T in years to 16:00 ET) in place of α_b. Same for Δ ATM. Session-clustered SEs (statsmodels `cov_type='cluster'`). |
| C8 | Single beta | A single β **suffices** if the single-β fitted curve's divergent share (C11) is within **45–55 %** in every 30-min bucket in the fit domain and in both directions on the random windows. Otherwise the recommended curve carries β per bucket and/or per direction, whichever brings every cell inside 45–55 %. The Wald tests of equal β are reported alongside, not used for the call. |
| C9 | Drift in the function | The recommended curve's drift is a per-minute rate θ_b per 30-min bucket (fitted on 1-min changes after removing β · r), so expected drift over any window = Σ θ_b × minutes spent in bucket b. One parameter set serves every window length. |
| C10 | Recommended β | From the 5/15/30-min window fits if their β agree within ± 25 % of the 15-min β (a plateau, B2); else from the swing-12 legs (the A17 use case), disclosed. |
| C11 | Divergent | A6: price up with put skew rising vs expected, or price down with it falling. Read = **divergent** if sign(residual) = sign(ES move), **confirming** if opposite, **none** if either is 0 or missing. Residual = Δ put skew − expected Δ put skew. (a) current = CR-CI `residual_put_skew` (the `callbacks.py` block, imported, not copied; prev = window start, now = window end); (b) fitted = C7–C10; (c) raw = Δ put skew. Share = divergent / (divergent + confirming). |
| C12 | Balance samples | **Random windows:** per session, 20 windows; length uniform on 15..30 min, start uniform on the minute grid with start ≥ 07:00 and end ≤ 12:45 PT; seed 20261026. **Swing-12 legs:** every confirmed leg in the fit domain. Reported pooled, by direction, by |ES move| tercile, by bucket. CIs: session-cluster bootstrap, 2,000 draws, seed 20261026. **Balanced** = point estimate inside 45–55 % pooled and in each direction. |
| C13 | Step 5 OOS | Primary (K8): fit on the first half of the 2026 sessions (by count), balance on the second half. Readouts: fit R-a → check 2026; fit 2026 → check R-a. The R-a-free curve (2026-only fit) is published alongside, for any later R-a confirmation (see B4 note). |
| C14 | Function | `packages/shared/skew_expected.py`: pure (no I/O, no pandas required), parameters as a frozen dataclass with the recommended values, `expected_put_skew_change_pp(ret_pct, t_prev_pt, t_now_pt)`, `put_skew_residual_pp(...)` and the ATM counterparts; unit tests in `packages/shared/tests/test_skew_expected.py`. Not imported by the Dash app. |

## Steps

- **Step 0** (diagnosis): coverage per period; third-Friday PM rows; ES symbol uniqueness; ES Databento vs Ironbeam close cross-check; alignment check (snapshot `stock_price` vs ES close, lag correlation at −2..+2 min to confirm C4); count of windows and legs per sample. Stop and ask if 2025 would be needed, or if any period has < 80 % of sessions usable.
- **Step 1–5** (run): one script, `scripts/cr_cj_calibrate.py`, writes `scripts/logs/cr_cj/results.json` and the tables below.
- **Function**: C14, tests.
- **Wrap**: close the open question; vault CR note, session note; PR.

## Standing rulings conformance (at spec freeze)

| Ruling | Status | Note |
|---|---|---|
| A1–A5, A8–A13, A15, A18–A20 | n/a | No levels, entries, exits or trades. |
| A6 | honoured | C11 is A6's definition. |
| A7 | honoured | Swing-12 legs from `flow_features.swing_legs` (the Flow pane rule). |
| A14 | honoured | This CR is the calibration; raw put skew is reported as the cross-check throughout. |
| A16 | honoured | Step 1 measures the snapshot noise; every read is a change over a window or leg, never a single snapshot. |
| A17 | honoured | Calibration only; skew is not made a trigger. |
| B1 | honoured | CIs on every share and β; no edge verdict is drawn. |
| B2 | honoured | Three window lengths, legs, and per-bucket variants are all disclosed; C10 chooses by plateau. |
| B3 | n/a | Not a setup optimisation. |
| B4 | honoured, with a note | Holdout and R-b not read. **R-a is read** for skew vs contemporaneous price only (no outcomes), per the kickoff ("from 2024-04-03"). Consequence: a later R-a confirmation that uses the pooled curve is not fully independent of it on the skew side; C13 publishes a 2026-only curve for that case. |
| B5 | honoured | Databento outright minute bars (C3). |
| B6, B7, B8 | n/a | No exits, no placebos, no labels. |
| C1, C2 | honoured | Not wired into any view (K9). |

---

## Step 0 findings (2026-10-26, `scripts/cr_cj_step0.py` → `scripts/logs/cr_cj/step0.json`, no fits)

| Q | Finding |
|---|---|
| Coverage (K2) | **R-a: 186 / 186** sessions (2024-04-03 → 12-31, early closes 07-03, 11-29, 12-24 excluded) and **2026: 107 / 107** sessions (01-02 → 06-05) have 0DTE PM skew and Databento ES on ≥ 90 % of minutes. Median 387 / 390 skew minutes and 390 ES minutes per session. **2025 is not needed** and was not queried. |
| Third Fridays | The PM 0DTE row exists on all 14 third Fridays in the sample (379–390 minutes), so no session is lost to the AM/PM key collision. |
| ES symbol | Most-volume outright per session: M24, U24, Z24, H25 (from 2024-12-16), H26, M26 (from 2026-03-16). 35 sessions have two symbols; the kept one carries ≥ 59.3 % of the volume. |
| ES cross-check | Databento minute close vs `ironbeam_es_1m_bars`, 10 random 2026 sessions: 9 sessions equal to the tick on 97.7–100 % of minutes; **2026-03-16 is 50.5 pts off on every minute**, the roll Monday where the most-volume contract (M26) differs from Ironbeam's (H26). Databento (B5) is kept. |
| **Alignment (C4)** | **C4 is wrong for R-a.** `quote_date` lags the snapshot label by median **129 s** in R-a (p5 79, p95 179; 220 / 72,709 rows > 5 min) vs **10 s** in 2026 (p5 0, p95 20; field-conventions measured 2026 only). 1-min Δ `stock_price` vs Δ ES correlation by lag, label pairing: R-a peaks at **−2 min (0.556)**, 0.004 at lag 0; 2026 peaks at 0 (0.844). With quote-time pairing (A1): R-a **0.736 at lag 0** (next 0.160 at +1), 2026 unchanged 0.844. |
| Samples (fit domain, A1) | R-a: 12,731 / 4,238 / 2,027 windows of 5 / 15 / 30 min; 1,027 swing-12 legs (of 1,399). 2026: 7,364 / 2,457 / 1,174 windows; 1,231 legs (of 1,606). |
| Implementation note | statsmodels is not installed in the run interpreter; the session-clustered OLS is the same CR1 sandwich written in numpy (`ols_cl` in `scripts/cr_cj_calibrate.py`): V = G/(G−1) · (N−1)/(N−K) · (X′X)⁻¹ (Σ_g X_g′u_g u_g′X_g) (X′X)⁻¹. |

## Amendment A1 (Code, Step 0, before any fit) — align on quote time

C4 is replaced: a snapshot pairs with the ES close at its **quote time**, `quote_date` rounded to the nearest minute Q (the close of the bar starting Q − 1). Snapshots whose `quote_date` is > 5 min from the label are dropped (R-a 220 rows, 2026 none). Time-of-day buckets, the fit domain and the 0DTE clock stay on the snapshot label (in R-a the quotes are ≈ 2 min older than the bucket says; immaterial at 30-min buckets). The current-curve comparison (C11a) uses the module's own `stock_price`, which is consistent with the quotes in both periods. No standing ruling is affected. A field-conventions row is added for `quote_date` in R-a.

---

## Run as frozen (C1–C14 + A1) — `scripts/cr_cj_calibrate.py` → `scripts/logs/cr_cj/results.json`

293 sessions (R-a 186, 2026 107). Fit domain 07:00–12:45 PT. Vol pts; β in vol pts per 1 % ES move. CIs 95 %, session-clustered (β) or session bootstrap (shares).

### Step 1 — noise (K4), put skew

| 30-min bucket (PT) | 06:30 | 07:00 | 08:00 | 09:00 | 10:00 | 11:00 | 11:30 | 12:00 | 12:30 |
|---|---|---|---|---|---|---|---|---|---|
| SD of 1-min Δ | 6.05 | 0.48 | 0.47 | 0.45 | 0.55 | 0.59 | 1.01 | 1.49 | 1.55 |
| SD of 5-min Δ | 3.58 | 0.66 | 0.64 | 0.55 | 0.72 | 1.05 | 2.00 | 2.07 | 2.94 |
| SD of 15-min Δ | 2.81 | 0.83 | 0.73 | 0.69 | 0.85 | 1.55 | 1.48 | 3.89 | 5.76 |
| VR(15) | 0.01 | 0.20 | 0.17 | 0.16 | 0.16 | 0.47 | 0.14 | 0.45 | 0.92 |

Variance ratios far below 1 (07:00–11:00: VR(5) 0.29–0.44, VR(15) 0.16–0.32, VR(30) 0.08–0.15): **most 1-min skew movement is fit noise that reverses**. A 5-min change has SD ≈ 0.55–0.74 vol pts on 07:00–11:00 (A16's "~0.7"). Noise jumps after 11:30 and is extreme in the first 30 min. ATM vol50: 1-min SD 0.47–0.82 on 07:00–11:00, VR(15) 0.37–0.59.

### Step 2 — fits (K5), α-model (α per start bucket + β·r), OLS

| Sample | n | Put skew β | β up / β down | Wald equal β by bucket, p | ATM β | ATM β up / down |
|---|---|---|---|---|---|---|
| 5-min | 19,362 | **−1.79** [−2.43, −1.16] | −3.69 / −0.02 (p 0.001) | 0.007 | −6.24 [−7.77, −4.72] | +1.06 / −13.09 |
| 15-min | 6,541 | **−2.33** [−4.12, −0.55] | −5.34 / +0.36 (p 0.13) | < 0.001 | −4.87 [−9.26, −0.48] | +6.42 / −14.95 |
| 30-min | 3,169 | **−1.54** [−3.63, +0.54] | −4.29 / +0.92 (p 0.21) | 0.002 | −4.23 [−9.14, +0.68] | +4.70 / −12.27 |
| Swing-12 legs (minutes model) | 2,246 | **−1.61** [−2.26, −0.96] | −3.71 / −0.80 | — | −5.45 [−7.55, −3.35] | +1.55 / −8.09 |

- R² 0.01–0.05: the ES move explains little of the skew change; noise dominates.
- By period, put β (15-min): R-a −1.49, 2026 −3.64; legs R-a −1.13, 2026 −2.11.
- Per-bucket put β (15-min) runs from about 0 at 07:00–09:00 to −3 to −4 at 09:30–11:00, and −13 at 12:00.
- Readouts outside the domain: open 06:30–07:00 put β −12.2, late 12:45–13:00 +5.1.
- Time term (15-min, Δ√T in place of buckets): put γ −266, ATM γ −1,806 vol pts per √year.
- **C10:** 5/15/30-min β −1.79 / −2.33 / −1.54 is not a ± 25 % plateau, so β comes from the legs (−1.61).

### Step 3 — current curve vs fitted (K6)

| Sample | Put skew: current implied β | fitted β | ATM: current implied β | fitted β |
|---|---|---|---|---|
| 5-min | −6.40 | −1.79 | −9.11 | −6.24 |
| 15-min | −5.72 | −2.33 | −9.42 | −4.87 |
| 30-min | −5.40 | −1.54 | −9.65 | −4.23 |
| Legs | −4.46 | −1.61 | −9.69 | −5.45 |

- The beta/theta add-ons cancel out of expected skew exactly (max |difference| 6e−15 vol pts).
- **The sticky-strike skew response is 3–4× too steep.**
- On ATM, the curve's −9.4 is roughly sticky strike (≈ −5) plus the 4.5 add-on. The fitted ATM β (−4.2 to −6.2) is about sticky strike alone, so **the 4.5 double-counts**.

### Step 4 — balance (K7), share "divergent"

| Sample | Raw | Current | Fitted, single β (C9) | Fitted, β up/down (C9) | Fitted, β by bucket | Fitted, bucket × up/down |
|---|---|---|---|---|---|---|
| Random 15–30 min (n 5,730) | 0.479 [0.464, 0.495] | **0.760** [0.748, 0.772] | 0.568 [0.553, 0.584] | 0.539 | 0.563 | 0.543 |
| … up / down | 0.464 / 0.496 | 0.857 / 0.655 | 0.573 / 0.563 | **0.303 / 0.796** | 0.571 / 0.555 | 0.420 / 0.677 |
| … small / mid / large \|move\| | 0.49 / 0.48 / 0.46 | 0.64 / 0.79 / 0.85 | 0.52 / 0.57 / 0.61 | | | |
| Swing-12 legs (n 2,246) | 0.481 [0.449, 0.510] | **0.780** [0.759, 0.800] | 0.625 [0.594, 0.654] | 0.606 | 0.600 | 0.574 |
| … up / down | 0.484 / 0.478 | 0.829 / 0.733 | 0.629 / 0.621 | 0.484 / 0.722 | 0.621 / 0.580 | 0.494 / 0.650 |

**C8: no variant is inside 45–55 % in every cell; the pre-registered procedure yields no recommended curve.** Two causes, found in the tables, not tuned:
1. **Mean vs median.** OLS fits the mean change; "divergent" is a sign, which tracks the median. A few large skew jumps on down moves pull the OLS slope away from the typical response. So the single-β residual still leans divergent, more so on large moves (0.52 → 0.61).
2. **C9's 1-min drift can't carry an asymmetric slope.** θ is estimated after removing β_up·r⁺ + β_dn·r⁻ minute by minute. Over a window, the sum of 1-min r⁺ is much larger than the window's r⁺ (path cancellation), so the drift absorbs a spurious term. That is why the up/down variants split 0.30 / 0.80.

### Step 5 — out of sample (K8), single-β curve

| Split | Random windows: fitted / current / raw | Legs: fitted / current / raw |
|---|---|---|
| **2026 H1 (01-02 → 03-19, 53) → H2 (03-20 → 06-05, 54)** | 0.530 [0.487, 0.575] / 0.790 / 0.479 | 0.561 [0.495, 0.621] / 0.787 / 0.462 |
| R-a → 2026 (readout) | 0.546 / 0.787 / 0.499 | 0.620 / 0.804 / 0.529 |
| 2026 → R-a (readout) | 0.594 / 0.745 / 0.468 | 0.608 / 0.750 / 0.423 |

---

## Amendment A2 (Code, after the frozen run, before any A2 fit) — median fit, joint drift

**Post-hoc: chosen after seeing the C8 failure. Disclosed as such (B2); needs Ryan's OK before the curve decides a test.** K5's OLS results above stand as the answer to K5. A2 only adds an estimator aimed at K7's target, a sign-balanced residual.

| # | Choice | Value |
|---|---|---|
| A2.1 | Model | Δ put skew = Σ_b θ_b · minutes in bucket b + β · r (**single**), or + β_up · r⁺ + β_dn · r⁻ (**up/down**). Drift fitted **jointly** with β on the same windows (replaces C9's 1-min θ, which cannot carry an asymmetric slope). |
| A2.2 | Estimator | **LAD** (median regression, scipy `linprog` HiGHS). OLS with the same joint model is a readout. β CIs: session-cluster bootstrap, 200 draws, seed 20261026. |
| A2.3 | Fit sample | Swing-12 legs (C10's outcome). Readout: LAD on the 15-min grid windows. |
| A2.4 | Recommendation rule | The simpler LAD-on-legs variant (single before up/down) whose **K8 primary out-of-sample** balance (fit 2026 H1, test H2) is inside 45–55 % pooled **and** per direction, on **both** random windows and legs. If neither passes, no fitted curve is recommended, and raw put-skew change is the reading until a better model exists. In-sample bucket cells and the cross-era readouts are reported, not gating. |
| A2.5 | Published parameters | The recommended variant fitted on all 293 sessions, plus the same fitted on 2026 only (for a later R-a confirmation, B4 note). β and θ outside 07:00–12:45 are copied from the nearest domain bucket and flagged out of domain. |
| A2.6 | ATM | The same A2.1–A2.3 for Δ ATM vol50, reported. The shared function carries the ATM curve with the same variant choice. |

---

## A2 result (post-hoc, disclosed) — the typical skew response to price is ≈ 0

**Median (LAD) slope, swing-12 legs, joint drift.** Put skew β = **−0.32** [−0.53, −0.01] (bootstrap). Up/down: −0.63 [−1.18, −0.20] / −0.09 [−0.28, +0.33]. LAD on 15-min windows −0.25. OLS on the same model is −1.61.

**Slope by subsample, legs, single β:**

| Subsample | n legs | Put skew LAD | Put skew OLS | ATM LAD | ATM OLS |
|---|---|---|---|---|---|
| All | 2,246 | −0.32 | −1.61 | **−7.02** | −5.45 |
| R-a | 1,019 | −0.55 | −1.13 | −5.84 | −6.71 |
| 2026 | 1,227 | +0.12 | −2.11 | −8.32 | −4.13 |
| 2026 H1 | 701 | **+0.58** | −1.15 | −8.95 | −4.59 |
| 2026 H2 | 526 | **−0.65** | −3.48 | −7.25 | −3.26 |

**Balance, share divergent.**

| Curve | In sample: random / legs | K8 OOS (2026 H1 → H2): random / legs | R-a → 2026: random / legs | 2026 → R-a: random / legs |
|---|---|---|---|---|
| LAD single | 0.498 / 0.510 | 0.467 [0.429, 0.505] / **0.426** [0.364, 0.483] | 0.522 / 0.579 | 0.474 / 0.406 |
| LAD up/down | 0.496 / 0.509 | 0.458 / **0.432** | 0.527 / 0.583 | 0.476 / 0.405 |
| OLS single (joint drift) | 0.565 / 0.623 | — | — | — |
| Raw (from the frozen run) | 0.479 / 0.481 | 0.479 / 0.462 | 0.499 / 0.529 | 0.468 / 0.423 |
| Current curve | 0.760 / 0.780 | 0.790 / 0.787 | 0.787 / 0.804 | 0.745 / 0.750 |

**A2.4 outcome: neither LAD variant passes** (legs 0.426 / 0.432 out of sample). **No fitted put-skew curve is recommended.**
- The median put-skew response to a 15–30-min or swing-12 ES move is about zero and flips sign between subsamples.
- What a fitted β buys in one sample it gives back in the next.
- Raw put-skew change is as balanced out of sample as any fitted curve.
- **Recommended expected put-skew change = 0** (flat), i.e. read the raw change.
- Not flat in two places: the 12:00–13:00 drift (LAD θ ≈ −0.025 vol pts/min) and the first 30 minutes are outside the domain.

**ATM is different.** The median β is stable at −5.8 to −9.0 in every subsample. The published ATM curve is LAD single: β **−7.02** vol pts per 1 % [−7.92, −6.29], with drift θ by bucket (results.json `a2.lad_single_atm_theta`): −0.034 /min at 07:00, about 0 on 09:00–11:30, −0.13 at 12:00, −0.34 at 12:30. It is reported, not gated (A2.6): A6's divergence read is about put skew.

---

## Wrap (2026-10-26)

**Deliverable (K9).**
- `packages/shared/skew_expected.py`: pure, 18 unit tests, parity with the run (legs 0.511 vs 0.510, random 0.498 vs 0.498).
- Recommended `PUT_SKEW_PARAMS` = flat (expected change 0; residual = raw Δ put skew, A2.4).
- `PUT_SKEW_LAD_A2_PARAMS` (reference, not recommended).
- `ATM_PARAMS` = LAD β −7.02 with per-bucket drift.
- Not imported by the Dash app. The Skew module is unchanged.

**Plain English.**
- The current curve expects put skew to move about 4× more with price than the typical move: −5.7 vol pts per 1 % against a median of ≈ −0.3, which isn't reliably different from zero. So the residual reads "divergent" about three times out of four.
- Typical put skew barely responds to the ES move inside 07:00–12:45, and the small slope it has flips sign between samples. The best "expected" put-skew change is zero, so read the raw change.
- Raw is close to balanced (≈ 48 %; out of sample 42–53 % depending on the split).
- ATM vol does respond: about +7 vol pts per 1 % sell-off.

**Deferred / follow-ups.**
- Wiring a raw-Δ put-skew read and the fitted ATM curve into the Skew table is a UI CR (C2: off by default).
- CR-CI A2's successor can use raw Δ put skew over the leg as the A14 residual.
- With 5-min noise at SD ≈ 0.6 vol pts, any divergence read needs a size threshold or a leg-length trend (A16). That is not measured here.
- A14's wording ("relative to what's expected for the price move") is Ryan's to revisit given this result. Standing rulings are not edited by Code.
