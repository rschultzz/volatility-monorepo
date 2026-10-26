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
