# CR-CK — Skew Stickiness Ratio: measure it, then show it

Spec freeze. Branch `feat/CR-CK-skew-stickiness` off `origin/main` `4f84484` (PR #89 merge, CR-CJ), in its own worktree (`.claude/worktrees/cr-ck`). Read-only against the database. No table writes, no cron, no deploy. One small display change, off by default. PR, don't merge.

Source: the kickoff prompt (Ryan, 2026-10-26), transcribed as K1–K8. C1–C14 are choices Code fixed before any data was pulled.

---

## Why

Standing ruling A14 (rewritten 2026-10-26 after CR-CJ): the baseline for intraday 0DTE put skew is **no change** (the smile re-forms around the new ATM), and the **scale** is the sticky-strike prediction. The stickiness ratio = actual Δ put skew ÷ sticky-strike Δ reads ≈ 0 for normal re-forming, ≈ 1 when skew relaxed as the mechanics imply, and < 0 when put skew steepened against the move. Nobody has yet seen what the ratio looks like on ordinary windows, so a reading such as the 2026-10-26 approach into 7731 can't be judged. This CR measures it on train, computes it for that day, and adds an optional display.

## Kickoff (Ryan, 2026-10-26)

| # | Decision | Value |
|---|---|---|
| K1 | Scope | Read-only research plus a small display. Branch off `origin/main` after #89; own worktree. |
| K2 | Definition (A14) | Over a window (a swing-12 leg, or the last N min), on raw 0DTE PM put skew (vol75 − vol50) at ORATS quote time (CR-CJ A1): **actual** = trend change in put skew over the window (OLS slope × window length, not endpoints; A16); **expected** = sticky-strike predicted change for the same ES move; **ratio** = actual ÷ expected, defined only when \|ES move\| ≥ 0.15 % (else "move too small"); **noise** = whether \|actual\| exceeds the CR-CJ noise band scaled to the window length. |
| K3 | Reading | ≈ 0 normal re-forming; ≈ 1 skew relaxed as mechanics imply; < 0 put skew steepened against the move (bearish divergence on a rally, mirror on a decline). |
| K4 | Part 1 (no outcomes) | Train only ≤ 2026-06-05, 2026 sessions; **do not read R-a or R-b**. Distribution of the ratio on swing-12 legs and on random 15/30-min windows (median, p10/p25/p75/p90), by leg direction and time of day; share with \|actual\| beyond noise; share < 0 and > 1. Pure function in `packages/shared/skew_expected.py` with unit tests. |
| K5 | Part 2 (live, not a test) | 2026-10-26: the approach leg into ES 7731 (~07:30 → ~07:50 PT) and the range 07:50 → 08:48 PT; report actual, expected, ratio, noise flag. |
| K6 | Part 3 (display, off by default; C1/C2) | In the Skew overview, a toggle "Stickiness" that adds a row per selected time showing actual Δ put skew, sticky-strike expected Δ, ratio and a noise flag, in the Dash palette (no green/red). Nothing else on the panel changes; Playwright pixel-identical with the toggle off. |
| K7 | Delivery | PR, don't merge, don't deploy. Session note. |
| K8 | Summary | Plain English for Ryan: what does a normal ratio look like, and what did 7731 show? |

## Pre-run choices fixed by Code (2026-10-26, before any data pull)

| # | Choice | Value |
|---|---|---|
| C1 | Sample (Part 1) | 2026 train sessions 2026-01-02 → 2026-06-05 (CR-CJ period `y26`, 107 sessions). Loaders are CR-CJ's (`scripts/cr_cj_lib.py`), which assert every read inside the period and before 2026-06-05 21:00 UTC. **R-a, 2025 and the holdout are never queried** in Part 1 (the script loads only `y26`). Window domain as CR-CJ C5: start ≥ 07:00, end ≤ 12:45 PT; time of day = the window's start. |
| C2 | Skew and ES | As CR-CJ C2/C3 + A1: 0DTE PM `orats_monies_minute` rows (`expir_date = trade_date`, `expiry_tod = 'pm'`); put skew = (vol75 − vol50) × 100 vol pts; ES = Databento outright minute bars (B5), the session's most-volume symbol; a snapshot pairs with the ES close at its `quote_date` minute. |
| C3 | Actual | OLS slope of put skew on snapshot minute, over every valid snapshot minute in [s, e] inclusive, × (e − s). Valid when ≥ 80 % of the minutes in [s, e] have a snapshot and ≥ 5 points. Endpoint Δ is reported alongside as a readout. |
| C4 | Expected (sticky strike) | `skew_expected.py` has no sticky-strike part today (the sticky-strike block lives in `apps/web/modules/Skew/callbacks.py`). CR-CK adds one, pure: the **start** snapshot's smile (vol5 … vol95 by delta, mapped to log-moneyness with the start vol50 and start T, exactly as `callbacks._prev_smile_interp`, linear inside and linear extrapolation outside) is held fixed in strike; spot moves by the ES move, k_shift = ln(1 + r/100). Expected put skew after the move = σ(k_p25 + k_shift) − σ(k_shift), with k_p25 the start 25Δ-put node. **Expected Δ = that − the start put skew.** T is held at the start, so expected Δ = 0 when the ES move is 0 (A14's baseline is no change; the clock is not a price effect). This differs from the callbacks block, which re-derives the 25Δ-put k from the **end** snapshot's vol50 and T; CR-CK reports both implied βs on the same windows as a parity readout (CR-CJ measured the callbacks version at −5.7 vol pts per 1 % on 15-min windows). |
| C5 | Ratio | ratio = actual ÷ expected when \|r\| ≥ 0.15 % (r = ES % move between the endpoint closes at quote time) and expected ≠ 0; otherwise "move too small" (no ratio). |
| C6 | Noise band | The CR-CJ band is the SD of the endpoint change in put skew over a window (Step 1: ≈ 0.55–0.74 vol pts at 5 min on 07:00–11:00). CR-CJ showed it does **not** scale as √L (VR(15) 0.16–0.32), so the band is scaled to length **empirically**: SD of the endpoint Δ put skew on 2026 train windows (C1) of length L ∈ {5, 10, 15, 20, 30, 45, 60, 90, 120} min, starts on a 5-min grid from 07:00, by 30-min start bucket (07:00 … 12:30). Band(L, start) = linear interpolation in L within the start bucket, clamped to 5 and 120 min. Flag **beyond noise** = \|actual\| > 1 × band (A14: "changes must beat noise"). The table is shipped as constants in `skew_expected.py` so the display uses the same band. Readout: SD of the trend Δ (C3) on the same windows. |
| C7 | Samples (Part 1) | **Swing-12 legs:** `flow_features.swing_legs(H, L, 12)` on each session's Databento bars, endpoints s = start_i + 1, e = end_i + 1 (CR-CJ's alignment), inside the C1 domain. **Random windows:** per session, 20 windows of 15 min and 20 of 30 min, start uniform on the minute grid inside the domain, seed 20261026 (fresh draws; not CR-CJ's windows). |
| C8 | Reported (Part 1) | Per sample (legs, rand15, rand30): n, share "move too small"; on ratio-defined windows: median, p10, p25, p75, p90; share < 0, share in [0, 1], share > 1; share beyond noise (all windows, and ratio-defined only). Split by direction (ES up / down) and by hour of start (07, 08, 09, 10, 11, 12:00–12:45). Session-bootstrap 95 % CIs (2,000 draws, seed 20261026) on pooled medians and shares. Descriptive only: no outcomes, no pass/fail. |
| C9 | Function | In `packages/shared/skew_expected.py` (pure, no I/O, no pandas): `sticky_strike_put_skew_change_pp`, `trend_change_pp`, `noise_band_pp`, `stickiness_read` (returns actual, expected, ratio or None, the reason when None, band, beyond-noise flag). Unit tests in `packages/shared/tests/test_skew_expected.py`. Parity with the Part 1 script on a sample of windows. |
| C10 | Part 2 data | 2026-10-26 is a live session after the holdout cut: read for this one readout only, labelled live, never pooled with Part 1. Databento ends 2026-10-23, so ES = `ironbeam_es_1m_bars` closes at quote time, cross-checked against the `ironbeam_es_flow_1s` minute close (field-conventions: the flow is the correct series where they differ; any > 1-tick endpoint difference is reported and the flow is used). |
| C11 | Part 2 windows | (a) **Approach, as named:** 07:30 → 07:50 PT. (b) **Swing-12 approach leg** on the Ironbeam bars containing 07:30–07:50 (its own start/end reported) and the same leg cut at the first bar that trades 7731. (c) **Range:** 07:50 → 08:48 PT. Each reports actual, endpoint Δ, expected, ES move, ratio (or "move too small"), band, flag. |
| C12 | Display endpoint | New read-only route `GET /api/ironbeam/skew-stickiness?trade_date&expiration_date&times[&until]`. For each selected time t after the first, the window is **the previous selected time → t** (the panel's other Δ columns compare against the previous selected time); the first time has no window. 0DTE only (expiration = trade date), else the rows say "0DTE only". ES from `ironbeam_es_1m_bars` (the price chart's own source; live, not a test). The existing `/api/ironbeam/skew-data` route is untouched. No live-row stickiness. |
| C13 | Display UI | In the price chart's floating **Overview** panel (smile + skew table, `PriceChart.jsx`), a "Stickiness" checkbox in the settings dialog, stored in `localStorage` (`ib-react-skew-stickiness`), **default off**. Off: no request to the new route, no DOM change. On: under each time row of the skew table, one extra row: "Δput {actual} · SS {expected} · ratio {r \| move too small} · {beyond noise \| in noise}". Colours from the Dash palette only: text `#94a3b8` (light grey), values beyond noise `#60a5fa` (blue), within noise `#64748b` (slate). The existing green/red cells are not touched (C2: existing views unchanged). |
| C14 | Pixel check | Playwright on a local read-only backend (the CR-BV/CR-CD smoke pattern): the Overview panel screenshotted on `origin/main` and on this branch with the toggle off, same day/times/viewport; byte-identical PNGs required. A second test turns the toggle on and checks the rows and colours. |

## Steps

- **Step 0** (diagnosis): 2026 coverage via CR-CJ loaders; 2026-10-26 coverage (0DTE PM snapshots, Ironbeam bars and flow 06:30–13:00 PT); sticky-strike parity readout (C4 vs callbacks βs on a sample); window/leg counts. Stop and ask if 2026 train has < 80 % usable sessions or 2026-10-26 lacks the 07:30–08:48 data.
- **Step 1** (function): C9 + tests.
- **Step 2** (Part 1 run): `scripts/cr_ck_stickiness.py` → `scripts/logs/cr_ck/results.json` (git-ignored); tables below.
- **Step 3** (Part 2): `scripts/cr_ck_live_7731.py` → tables below.
- **Step 4** (Part 3): route + toggle + Playwright (C12–C14).
- **Wrap**: vault CR note, session note, PR.

## Standing rulings conformance (at spec freeze)

| Ruling | Status | Note |
|---|---|---|
| A1–A5, A8–A13, A15, A19, A20 | n/a | No levels, entries, exits or trades are tested. |
| A6 | honoured | K3's reading (< 0 = put skew steepened against the move) is A6's divergence in the ratio's units. |
| A7 | honoured | Swing-12 legs from `flow_features.swing_legs` (C7, C11b). |
| A14 | honoured | This CR implements A14's definition: baseline no change, sticky-strike as the scale, ratio, noise. C4 adds the sticky-strike part to `skew_expected.py` because it was not there; the callbacks version is reported for parity. |
| A16 | honoured | Actual is the OLS trend over the window (C3); endpoints are a readout only. |
| A17 | honoured | Descriptive and display only; the ratio is never a trigger. |
| A18 | n/a | Part 2's range window is the one Ryan named. |
| B1 | honoured | CIs on pooled medians and shares; no verdicts drawn. |
| B2 | honoured | Every window type, band length and split is disclosed; no parameter is chosen from the results. |
| B3 | n/a | Not a setup optimisation. |
| B4 | honoured, with a note | Part 1 reads 2026 train only; **R-a, R-b and the holdout are not read**. Part 2 reads one live session (2026-10-26) because Ryan's kickoff asks for it, labelled "live, not a test", not pooled, no outcomes. The display reads whatever date the user selects, as the panel already does. |
| B5 | honoured, with a note | Part 1 uses Databento. Part 2 and the display use Ironbeam (no Databento after 2026-10-23; neither is a test), with the flow cross-check in C10. |
| B6, B7, B8 | n/a | No exits, placebos or labels. |
| C1 | honoured | New row in blue / light grey / slate only (C13). |
| C2 | honoured | Off by default; existing route and panel unchanged; pixel check C14. |

No "deviates".

---

## Step 0 findings (2026-10-26, `scripts/cr_ck_step0.py` → `scripts/logs/cr_ck/step0.json`, no fits)

| Q | Finding |
|---|---|
| 2026 train coverage | 107 sessions (2026-01-02 → 06-05); **106** have 0DTE PM put skew and Databento ES on ≥ 90 % of the 07:00–12:45 domain minutes. Only `y26` was loaded; R-a and 2025 were not queried. |
| Swing-12 legs | 1,226 legs inside the domain. Length (start → end bar) p10 **2 min**, median 13, p90 52, max 278. Legs shorter than 4 min cannot meet C3 (≥ 5 snapshot points), so they drop out of the ratio sample; the count is reported in Step 2. The band table (C6) reaches 120 min; longer legs use the 120-min band (clamped, as frozen). |
| 2026-10-26 (Part 2) | 390 0DTE PM snapshots 06:30 → 12:59 PT (91 in 07:20–08:50), quote lag median 10 s; `ironbeam_es_1m_bars` 390 RTH bars (`source` ironbeam), `ironbeam_es_flow_1s` on 390 RTH minutes. Part 2 can run. |
| Sticky-strike parity | Moved to Step 2: it needs the C9 function, which lands in Step 1. No ruling or choice is affected. |

---

## Part 1 result (run as frozen, C1–C9) — `scripts/cr_ck_stickiness.py` → `scripts/logs/cr_ck/results.json`

107 sessions, 2026-01-02 → 06-05 (y26 only; R-a, 2025 and the holdout were not loaded). Domain 07:00–12:45 PT. Vol pts. 95 % CIs from a session bootstrap (2,000 draws). Descriptive: no outcomes, no verdicts.

### Noise band (C6) — SD of the endpoint Δ put skew, vol pts (shipped as `skew_expected.NOISE_BAND_PP`)

| Start bucket | 5 min | 15 min | 30 min | 60 min | 120 min |
|---|---|---|---|---|---|
| 07:00 | 0.73 | 1.13 | 1.26 | 1.07 | 1.02 |
| 07:30 | 0.89 | 1.26 | 1.19 | 1.21 | 1.39 |
| 08:00 | 0.65 | 0.85 | 0.88 | 0.88 | 1.29 |
| 08:30 | 0.57 | 0.82 | 0.82 | 1.18 | 1.06 |
| 09:00 | 0.58 | 0.79 | 1.11 | 1.17 | 1.27 |
| 09:30 | 0.91 | 1.44 | 1.36 | 1.33 | 2.59 |
| 10:00 | 0.91 | 1.06 | 1.19 | 1.41 | 3.02 |
| 10:30 | 0.79 | 0.94 | 1.31 | 2.53 | 4.48 |
| 11:00 | 1.43 | 2.56 | 2.50 | 3.02 | 4.60 |
| 11:30 | 2.97 | 3.05 | 3.69 | 5.23 | 5.23* |
| 12:00 | 3.14 | 5.13 | 5.03 | 1.47* | 1.47* |
| 12:30 | 5.34 | 8.83 | 8.83* | 8.83* | 8.83* |

\* carried from the bucket's longest measurable length (C6 clamp). **The 12:00 row drops to 1.47 at 45 min** (the only 45-min window from 12:00 is 12:00 → 12:45, which skips the noisier 12:30–12:45 intermediate endpoints); it is measured, not tuned, but makes the 12:00 band for ≥ 45-min windows unreliable. After 11:00 the band is 2–9 vol pts, so almost nothing reads "beyond noise" there.

- A14's "~0.6 vol pts per 5 min" holds at 5 min on 08:00–09:30. **For the 15–30 min windows and legs it is 0.8–1.4 vol pts on 07:00–10:30**, and the band does not grow like √L (CR-CJ's VR < 1).
- The trend Δ (C3) has about the same SD as the endpoint Δ at these lengths (`noise_band_cells.sd_trend`): the OLS trend does not shrink the band, because most of the spread is genuine wandering over 15–60 min, not 1-min fit noise.

### Sticky-strike parity (C4, moved from Step 0)

On the 2,134 random 15-min windows: CR-CK's sticky-strike expected Δ has implied β **−5.54** vol pts per 1 % ES; the Skew module's version (end-snapshot ATM and T) **−6.93** with intercept −0.14 (the clock), correlation 0.67. CR-CJ's pooled figure for the module was −5.7. Same mechanics, same order of magnitude; CR-CK's has no clock term by construction.

### The ratio

| Sample | n | ratio defined | move too small | insufficient | median [95 % CI] | p10 | p25 | p75 | p90 | < 0 | 0–1 | > 1 | beyond noise (ratio defined) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Swing-12 legs | 1,231 | 764 | 22.7 % | 15.3 % | **−0.12** [−0.19, −0.02] | −2.04 | −0.78 | 0.49 | 1.53 | 55.2 % [50.5, 59.6] | 28.8 % | 16.0 % [13.2, 19.0] | 23.8 % [20.5, 27.1] |
| Random 15 min | 2,134 | 379 | 82.1 % | 0.1 % | **−0.03** [−0.27, +0.11] | −2.64 | −0.96 | 0.80 | 2.56 | 50.4 % [44.1, 57.1] | 27.4 % | 22.2 % [17.0, 27.4] | 30.1 % [23.8, 36.3] |
| Random 30 min | 2,135 | 569 | 73.2 % | 0.1 % | **+0.02** [−0.09, +0.15] | −1.79 | −0.70 | 0.58 | 1.71 | 49.0 % [44.0, 54.0] | 33.4 % | 17.6 % [14.0, 21.2] | 25.1 % [20.3, 30.4] |

"Insufficient" on legs = legs shorter than ~4 min (C3 needs ≥ 5 snapshots). Median |actual| 0.63–0.80 vol pts; median |expected| 0.87–1.13.

**By direction (ES move over the window):**

| Sample | Up: median | Up: < 0 / > 1 | Down: median | Down: < 0 / > 1 |
|---|---|---|---|---|
| Legs | +0.09 | 46.9 % / 19.5 % | −0.21 | 63.7 % / 12.4 % |
| Random 15 | +0.30 | 39.1 % / 26.6 % | −0.35 | 62.0 % / 17.6 % |
| Random 30 | +0.26 | 39.1 % / 21.5 % | −0.15 | 59.3 % / 13.6 % |

**By hour of start (legs, median ratio / share < 0):** 07:00 −0.19 / 58 % · 08:00 −0.22 / 62 % · 09:00 −0.08 / 54 % · 10:00 +0.06 / 46 % · 11:00 −0.13 / 57 % · 12:00 +0.91 / 33 % (n 30). Random windows by hour are in `results.json`; no hour stands apart except 12:00 (n 30, band 3–9 vol pts).

**Readout (post-hoc, descriptive):** median actual Δ put skew by ES-move bin. On moves ≥ 0.15 % *either way*, put skew's median change is slightly **negative** (down moves −0.24 to −0.38 vol pts, up moves −0.13 to −0.15 on random windows; legs +0.09 up). Sticky strike says + on declines and − on rallies. So on declines the typical window already has ratio < 0 (put skew fell while price fell, A6's "bullish" pattern) — about 60 % of the time. That asymmetry is the normal state in 2026 train, not a signal.

### What a normal ratio looks like (plain English)

- The middle half of ordinary windows runs from about **−0.8 to +0.6**, centred on **≈ 0**. A reading in that range is the market's normal "smile re-forms around the new price".
- About **half of all readings are negative** and **1 in 6 is above 1**. So a negative ratio by itself is not unusual, and neither is > 1.
- Only about **1 in 4** ratio-defined windows has a skew change bigger than the noise band. A ratio is worth reading only when its "beyond noise" flag is set.
- Declines skew negative: on a sell-off, put skew *falling* is the more common outcome (≈ 60 %).
