# CR-CM — Do the STRONG GEX levels slow price? (T0 re-run on the chart's strongest lines)

Read-only, price paths only, no trade outcome. Branch `feat/CR-CM-strong-level-t0`, own worktree.
Kickoff: Ryan, 2026-10-26 (chat). Frozen 2026-10-26 before any CR-CM data was read.

## Question

CR-CB's T0 (time to escape 0.25·IM, real − placebo, zone dwell) used Corrected GEX, top 6 per session with gross
≥ 40 $bn, and came back inconclusive, leaning against (A17 pooled −5.98 [−13.35, +1.21] min on R-a; train v3
pooled −4.7 [−11.9, +2.0]). Ryan trades only the chart's strongest lines (standing ruling A12): call levels =
yellow, put levels = darkest blue / indigo. Does price slow more at those lines than at fake levels, and than at
the weak lines?

## Strong levels — the chart's rule (A12)

Read from `apps/web/modules/Ironbeam/callbacks.py` (Levels: Corrected; `_corrected_gex_tables`,
`_build_gex_level_segments` path, `_select_levels`, `_compute_color_span`, `_color_for_net_gex`) at `origin/main`
`8f1226b`:

| step | chart code | CR-CM |
|---|---|---|
| rows | `orats_oi_gamma`, `trade_date = D`, `discounted_level IS NOT NULL`, `expir_date >= D` (every live expiry; no DTE cap) | same; ticker `SPX` |
| placement | `place_levels(rows, P_es)`, P_es = close of the `ironbeam_es_1m_bars` bar starting 15:45 ET on A = `prev_trading_day(D)`; 1-pt `bucket_levels` | same |
| aggregation | per level: call = Σ gex_call, put = −Σ\|gex_put\|, net = call + put | same |
| band | [min low − 150, max high + 150] of the session window's bars (prior 15:00 PT → 13:00 PT, or → `until`) | **bars prior 15:00 PT → 06:30 PT** (what the chart showed at the RTH open; causal). Full-session band reported as a Step 0 count |
| display filter | gross = \|call\| + \|put\| ≥ `gex_min_abs_b` (A12: 40 $bn is Ryan's display filter) | 40 $bn |
| spacing / cap | sort by gross desc; keep if ≥ 5 pts from every kept level; ≤ 80 | same |
| colour span | `GEX_COLOR_ABS_MAX` > 0 → span = that; else span = `nanpercentile(\|net\|, GEX_COLOR_PERCENTILE=95)` over the **displayed** levels | p95 rule (see env note) |
| colour | t = (clip(net, −span, span) + span) / 2·span; t = 1 → `#fef08a` (yellow), t = 0 → `#312e81` (indigo) | **strong call** = net ≥ +span; **strong put** = net ≤ −span |

**Env note.** `GEX_COLOR_ABS_MAX` and `GEX_COLOR_PERCENTILE` on the Render web service `Dash - MonoRepo`
(`srv-d4370d6uk2gs738ps5t0`) could not be read from this session: the Render MCP exposes no env-var read, the
live `/api/ironbeam/bars` endpoint returns "Login Required", and neither key is in `render.yaml` or the local
`.env`. The code defaults (0 → percentile rule; 95) are used. **Ryan to confirm from the Render dashboard**; if
`GEX_COLOR_ABS_MAX` is set, the strong set is redefined as |net| ≥ that value before any outcome is computed.

**Window.** As CR-CB A1.1, levels are kept only within ±1·IM_d of the RTH open (the A16.4 placebo lives in that
window). Strong levels outside it are counted in Step 0.

## Level sets

| set | role | definition |
|---|---|---|
| **S (strong)** | **primary** | the strong calls + strong puts above, ±1·IM_d of the open; type = sign of chart net |
| X100 | readout | displayed levels (same chart build) with \|net\| ≥ 100 $bn, ±1·IM_d; type = sign of net |
| W (weak) | readout | the CR-CB A1.1 set (Corrected, ≤ 30 DTE, gross ≥ 40 $bn, 5-pt spacing, ±1·IM_d, top 6 by gross; A17 code unchanged) minus every level within 2 pts of a strong level (S, before the window cut) |
| S-first | readout | S tests kept only when price had not traded within the 2-pt band of the level earlier in the RTH session (the level's first touch of the session) |

## Test, measures, statistics — CR-CB A17, unchanged

Copied from `scripts/cr_cb_t0_ra_a17.py` (A16 + A17): one test per level per session (first armed test, either
side; its side sets the approach cell); arming max(0.10·IM_d, 4 pts); 2-pt test band; Databento 1-s flow of the
session's largest-RTH-volume symbol; primary = restricted mean time to escape 0.25·IM_d (min(escape, 120 min,
time to 13:00 PT)); secondary = zone dwell, seconds within [L − O_c, L + O_c] over 60 min, O_c frozen from train
(put/below 9.75, put/above 8.25, call/below 6.25, call/above 5.25); A16.4 / A15.6 placebo (per real level, the 4
nearest 5-pt bins on the same side of the open, within ±1·IM_d, ≥ 10 pts from every real level **of the set
under test**, deduplicated, parent's sign sets the type); session-weighted means (A17.3); 10,000 session-multinomial
resamples, seed 20260920, real and placebo resampled jointly; 95 % percentile CI; IM < 25 flagged, primary
includes, exclusion readout. Universe funnel as A16.1 (roll week; no 15:45 ET bar on A; quarterly-expiry Friday; no
Databento RTH flow — **no Ironbeam fallback**; early close; no canonical IM; no `orats_oi_gamma` rows; contract
check).

## Changes from A17 (only these)

| # | Change |
|---|---|
| M1 | Level set (above). |
| M2 | **EDGE / INTERIOR matching (B7), as CR-CF A2 D3 tagged it.** Each test is EDGE if, at the test second, L ≥ running RTH high − 2 or L ≤ running RTH low + 2 (running extremes of the 1-s flow from 06:30 PT through the test second inclusive), else INTERIOR. **Matched statistic** per hypothesis: Σ_s w_s · (R_s − P_s), s ∈ {EDGE, INTERIOR}, with R_s, P_s the A17 session-weighted means within stratum s and w_s the real arm's test share in s; recomputed in every resample. A resample where a stratum carrying real tests has no placebo session is undefined (NaN, dropped by the percentile; its share is reported). If the point estimate is undefined, the hypothesis is "unmatchable" → inconclusive. The unmatched A17 statistic, the within-EDGE and within-INTERIOR differences and the EDGE share real vs placebo are reported for every hypothesis. |
| M3 | **Cells and Holm (A8; "Holm across calls / puts").** Holm family = {strong puts, strong calls}, each pooled over approach side (2 hypotheses), one-sided α 0.025, H0 real − placebo ≤ 0 (matched primary). One-sided p = share of defined resamples ≤ 0 (floor 1e-4). Approach-side cells (put/below, put/above, call/below, call/above) and all-pooled are reported, not gated. |
| M4 | **Data.** **Train** = 2026-01-06 → 2026-06-05 (CR-CB Step 0 v3 train window; Corrected placement needs D ≥ 2024-04-03, R-b is 2025) — **primary verdict**. **R-a** = 2024-04-03 → 2024-12-31 — **confirmation**, flagged: *R-a was already read for T0 with the broader set (CR-CB A16 / A17); this is the third T0 read on R-a.* R-b (2025) and holdout (after 2026-06-05) are not read: the scripts assert every read is ≥ 2026-01-01 and < 2026-06-05 21:00 UTC (train) or < 2025-01-01 (R-a). |
| M5 | **A/A (CR-CF Step 0 design)** on placebo tests of S, per Holm hypothesis and pooled: 200 pseudo-level splits (default_rng 20261025), 2,000 session resamples per split (default_rng 20260920), both the unmatched and the M2-matched statistic; PASS if the CI excludes 0 in ≤ 10 % of splits and \|median\| ≤ 3 min. **The outcome run happens only if both Holm hypotheses pass on train.** |

## Verdict (B1)

Per hypothesis on train: **pass** = Holm-rejected with 95 % CI lower bound > 0 (real escapes more slowly);
**refuted** = CI upper bound < 0; else **inconclusive** (including underpowered or unmatchable). The same
pre-registered statistic is computed once on R-a. CR-CM's overall reading for a type: **confirmed** if it passes
on train and on R-a (Holm-rejected, CI lower bound > 0); **train-only** if train passes and R-a does not;
otherwise the train verdict stands. Readouts (X100, W, S-first, unmatched, approach cells, secondary, excl. IM <
25) carry no gate. Comparison with the weak lines is descriptive: S vs W real − placebo differences side by side,
plus S − W on real tests (session-weighted, same bootstrap), labelled readout.

## Step 0 (no outcomes on real levels) — stop for Ryan's review

Train and R-a: universe funnel; strong levels per session (calls / puts; median, IQR, zero-strong sessions);
displayed-level count and p95 span per session; strong levels outside ±1·IM; strong set under the full-session band
(differences); X100 / W counts; overlap S ∩ CR-CB set; tests per cell (real / placebo, sessions) for every set;
EDGE / INTERIOR counts per cell (real / placebo) with the underpowered flag (< 30 tests or < 15 sessions in an
arm, CR-CF); placebo coverage (placebo levels per real level, share of real levels with ≥ 1 placebo, share of S
placebo bins within 10 pts of a CR-CB-set level); A/A (M5). No real-level escape or dwell is computed.

## Run discipline

Each script is committed before its single run; statistics go to JSON before printing; outputs under
`scripts/logs/cr_cm/` (git-ignored). Read-only: `dash_backfill_writer` role with `default_transaction_read_only`.

## Standing rulings conformance (Dash/research/standing-rulings.md, 2026-10-26)

| # | status | note |
|---|---|---|
| A1 | honoured | zone dwell spans both sides of the level; escape is two-sided |
| A2 | n/a | T0 measures escape, not ranges |
| A3 | n/a | no entries |
| A4 | n/a | no stops |
| A5 | n/a | Layer 1 only; no direction |
| A6 | n/a | no skew |
| A7 | n/a | no approach-leg rule; the A17 arming distance is a test mechanic, unchanged |
| A8 | honoured | puts and calls separate (Holm family), pooled reported |
| A9 | n/a | single-level slowdown, as A17; Step 0 counts strong levels with another S / CR-CB level within 10 pts |
| A10 | n/a | no reference levels |
| A11 | n/a | no targets |
| A12 | **honoured** | strong = chart yellow / darkest (net beyond the day's p95 \|net\| of displayed lines); readouts 100 $bn net and the ≥ 40 gross top-6 (minus strong); Corrected placement; env value unverified (Ryan to confirm) |
| A13 | n/a | no sizing |
| A14 | n/a | no skew |
| A15 | n/a | direction first — no direction here |
| A16–A17 | n/a | no skew; Layer 1 only |
| A18 | honoured | escape and dwell symmetric around the level, so overshoot ranges count |
| A19–A20 | n/a | no entries |
| B1 | honoured | three verdicts |
| B2 | honoured | readouts disclosed as such; no parameter search |
| B3 | n/a | no setup optimisation |
| B4 | honoured | train primary; R-a confirmation (third R-a T0 read, disclosed); R-b and holdout not read (asserted) |
| B5 | honoured | Databento 1-s flow; no Ironbeam flow fallback (`ironbeam_es_1m_bars` only for P_target and the chart band, as the chart) |
| B6 | n/a | no exits |
| B7 | **honoured** | M2 EDGE / INTERIOR matching; EDGE share reported |
| B8 | n/a | no labels |
| C1–C2 | n/a | no UI |
| R1–R7 | honoured | none reintroduced; `discounted_level` used only as the chart's `IS NOT NULL` row filter, not for placement (R6) |

**No "deviates".**

## Choices fixed at freeze that Ryan should confirm at the Step 0 review

1. Band frozen at the RTH open (causal) rather than the full session's bars.
2. Strong levels restricted to ±1·IM_d of the open (A17 window, needed by the placebo).
3. Placebo exclusion = ≥ 10 pts from the set under test only (A17 literal); the share of S placebos near CR-CB-set
   lines is reported, so a stricter exclusion can be chosen before outcomes.
4. M2 matching by post-stratification (EDGE status is known only at the test second, so it cannot be matched at
   construction).
5. "First touch of the session" read as the level's first touch (no earlier RTH trade within 2 pts).
6. Train window 2026-01-06 → 2026-06-05.

---

## Step 0 findings (2026-10-26; no real-level outcome computed)

**Run.** `scripts/cr_cm_step0.py` + `scripts/cr_cm_lib.py` at commit `c9da013` (sha256 `6acdc730dc96…` /
`eb83f042ffe5…`); one run, exit 0. JSON `scripts/logs/cr_cm/step0.json` (git-ignored), written before printing.
Read-only; reads asserted inside the train (2026-01-01 → 2026-06-05 21:00 UTC) and R-a (< 2025-01-01) guards.
Escape was computed on placebo tests only (A/A).

### Universe

| | train | R-a |
|---|---|---|
| NYSE sessions | 105 | 191 |
| roll week | −10 | −29 |
| no 15:45 ET bar on A | −2 | −6 |
| early close | 0 | −3 |
| no canonical IM | −4 | −3 |
| contract check | 0 | −1 |
| **usable** | **89** (2026-01-06 → 06-05; ES H26 / M26) | **147** (2024-04-04 → 12-31; same as A17) |
| IM < 25 | 1 (2026-01-12, 24.1) | 14 (the A17 Mondays) |

### Strong levels (chart rule, band at the open)

| | train | R-a |
|---|---|---|
| displayed lines per session, median (IQR) | 35 (30–40) | 30 (27–33) |
| colour span = p95 \|net\|, $bn, median (IQR) | 115 (89–144) | 103 (80–132) |
| strong lines per session (any distance), median; range | 2; 1–3 | 2; 1–3 |
| strong calls / puts per session, mean | 1.62 / 0.56 | 1.29 / 0.67 |
| strong net $bn, median: calls / puts | +143 / −149 | +155 / −106 |
| strong lines outside ±1·IM of the open | 66 of 194 | 140 of 289 |
| **S (in window): levels, calls / puts** | **128: 93 / 35** | **149: 103 / 46** |
| sessions with no S level / with an S call / with an S put | 13 / 60 / 24 | 38 / 76 / 38 |
| S distance from open / IM, median | 0.52 | 0.39 |
| full-session band vs open band: sessions differing | 9 (9 levels only-full, 4 only-open) | 7 (7 / 6) |

- By construction p95 of ~30–35 displayed lines leaves ~2 strong lines per session; they are mostly calls.
- **S is almost a subset of the CR-CB set:** 123 / 128 (train) and 135 / 149 (R-a) S levels sit within 2 pts of a
  CR-CB top-6 level. So W = CR-CB minus S is a clean "weak" set: 405 (train) / 733 (R-a) levels.
- X100: 224 / 206 levels (calls 177 / 163, puts 47 / 43); 108 / 110 S levels are also X100.
- S levels with another S or CR-CB line within 10 pts (A9 count): 72 / 128 train, 91 / 149 R-a.
- Worked day 2026-05-01 (open 7264.5, IM 70.2, span 141 $bn): strong = 7227 (net +518) and **7277 (net +158)** —
  the 7277 call level of Ryan's 2026-05-01 review card is a yellow line.

### Tests per Holm hypothesis (first armed test per level per session)

| set | hyp | train real tests / sessions | train placebo | R-a real | R-a placebo |
|---|---|---|---|---|---|
| **S** | **put** | **14 / 12** | 50 / 16 | 30 / 27 | 83 / 27 |
| **S** | **call** | **42 / 34** | 135 / 47 | 61 / 52 | 119 / 53 |
| S | pooled | 56 / 45 | 185 / 59 | 91 / 77 | 202 / 79 |
| S-first | put / call | 14 / 38 | 46 / 124 | 23 / 49 | 74 / 102 |
| X100 | put / call | 21 / 85 | 73 / 124 | 30 / 94 | 71 / 107 |
| W | put / call | 72 / 143 | 155 / 174 | 164 / 259 | 145 / 143 |

Approach cells for S (real / placebo tests), train: put/below 5 / 20, put/above 9 / 30, call/below 32 / 82,
call/above 10 / 53; R-a: 14 / 27, 16 / 56, 33 / 78, 28 / 41.

**Power.** By the CR-CF flag (< 30 tests or < 15 sessions in an arm), **S put on train is underpowered** (14 real
tests on 12 sessions). S call on train (42 / 34) and both S hypotheses on R-a clear it.

### EDGE / INTERIOR (B7)

| S | EDGE share real / placebo, train | R-a |
|---|---|---|
| put | 1.00 / 0.96 | 0.93 / 0.95 |
| call | 0.93 / 0.96 | 0.92 / 0.97 |
| pooled | 0.95 / 0.96 | 0.92 / 0.96 |

- With **first-test-only**, almost every test is EDGE in both arms: price reaches a level it was armed away from by
  making a new session extreme. INTERIOR holds 0–7 tests per arm (underpowered everywhere).
- So the CR-CF imbalance (real 24 % vs placebo 63 % EDGE, two tests per level and swings) does not occur here; the
  M2 matched statistic will be close to the EDGE-only difference and to the unmatched one.
- Every S-first test is EDGE (share 1.00), as expected.

### Placebo coverage

| | train | R-a |
|---|---|---|
| S placebo levels / S real levels | 433 / 128 | 403 / 149 |
| S real levels with ≥ 1 placebo | 90.6 % | 83.2 % |
| **S placebo bins within 10 pts of a CR-CB-set line** | **49.9 %** | **70.2 %** |

Half (train) to 70 % (R-a) of the "fake" levels for S sit within 10 pts of a weaker GEX line from the CR-CB set,
because A17's exclusion applies only to the set under test (choice 3).

### A/A (M5; S placebo tests; 200 splits × 2,000 resamples)

| | train unmatched: excl share, median, PASS | train matched | R-a unmatched | R-a matched |
|---|---|---|---|---|
| put | 0.065, **+3.005**, **FAIL** | 0.079, +2.93, PASS (139 of 200 splits defined) | 0.065, +1.03, PASS | 0.054, +0.79, PASS |
| call | 0.050, −0.29, PASS | 0.062, −0.84, PASS | 0.060, +0.25, PASS | 0.053, −0.23, PASS |
| pooled | 0.050, −0.40, PASS | 0.045, −0.61, PASS | 0.055, −0.43, PASS | 0.055, −0.57, PASS |

**Gate (M5, as frozen): FAIL on train** — the put unmatched median is 3.005 min against a ±3 tolerance; every other
cell passes. The put placebo arm is 50 tests on 16 sessions. Under the spec the outcome run does not happen
without Ryan's ruling.

### Decisions for Ryan before any outcome

1. **A/A gate.** Train put unmatched fails by 0.005 min on the median (excl share 6.5 % is fine). The matched
   statistic (the primary) passes. Options: (a) gate on the primary (matched) statistic only, unmatched becomes a
   readout — proceed; (b) hold the put hypothesis to "inconclusive (A/A fail)" and run calls only; (c) stop.
2. **Placebo exclusion.** 50–70 % of S placebos sit near a CR-CB line. Keep A17's literal rule (fake = not a strong
   line), or exclude placebo bins within 10 pts of any CR-CB-set line too (cleaner "no GEX line" fake, fewer
   placebos; counts would be re-reported before outcomes).
3. **Power.** S put on train is underpowered (14 tests / 12 sessions); its train verdict will almost surely read
   inconclusive. Accept, or name R-a (30 / 27) as the put read — that would change the pre-registered train /
   confirmation roles and needs your explicit call.
4. **Env.** Confirm `GEX_COLOR_ABS_MAX` is unset (and `GEX_COLOR_PERCENTILE` = 95) on `Dash - MonoRepo`.
5. Freeze-time choices 1, 2, 5, 6 (band at the open; ±1·IM window — 34 % / 48 % of strong lines fall outside it;
   first touch; train window) stand unless you change them.

---

## Amendment A1 (Ryan, 2026-10-26, from Step 0 counts only; no outcome computed)

**Rulings (Ryan, verbatim in substance):**
1. A/A gate applies to the **primary (matched)** statistic, which passes in every cell. The train-put unmatched
   median (+3.005 vs ±3) is recorded as a near-miss in an underpowered cell; proceed.
2. Placebos must also sit ≥ 10 pts from **any CR-CB line** (gross ≥ 40 $bn), not just the strong set. A17's literal
   rule is a readout. Report the resulting placebo counts per cell.
3. Puts: the primary put read = **train + R-a pooled**, labelled **EXPLORATORY** — no untouched confirmation; R-b
   (2025) is reserved for that. Calls unchanged: train verdict, R-a confirmation. Holm across calls / puts.
4. Render env (checked by Ryan): `GEX_COLOR_ABS_MAX` unset, `GEX_COLOR_PERCENTILE` unset → code defaults (95th
   percentile, no fixed scale), as Step 0 assumed.

| # | Clause (implementation, fixed before the run) |
|---|---|
| A1.1 | **Gate.** M5 reads the matched statistic only; it passed in every cell on train and R-a (Step 0). Unmatched A/A is a readout. |
| A1.2 | **Placebo (amends the A17 placebo for every set: S, X100, W).** As A16.4, plus: a candidate bin is dropped if it is < 10 pts from any level of the session's CR-CB set (A1.1: Corrected, ≤ 30 DTE, gross ≥ 40 $bn, 5-pt spacing, ±1·IM_d, top 6 — the "CR-CB line" measured in Step 0) or of the in-window strong set S. The 4 nearest surviving bins per real level, deduplicated, parent's sign. **Readout:** the same statistics with the A17 literal placebo (≥ 10 pts from the set under test only). Placebo counts per cell are reported for both. |
| A1.3 | **Hypotheses and Holm.** H_call: strong calls, **train** (2026-01-06 → 06-05). H_put: strong puts, **train + R-a pooled** (236 sessions; one session-multinomial bootstrap over the union, seed 20260920, 10,000), labelled **EXPLORATORY**. Holm across {H_call, H_put}, one-sided α 0.025, matched statistic (M2). **Call confirmation:** strong calls on R-a alone, same statistic, one-sided p ≤ 0.025 and CI lower bound > 0. Readouts (no gate): put train-only and R-a-only; call pooled; all-type pooled per scope. |
| A1.4 | **Verdicts.** H_call: B1 three verdicts on train; "confirmed" if it passes and the R-a confirmation passes, "train-only" if only train passes. H_put: B1 three verdicts, always tagged EXPLORATORY (R-a is no longer untouched for it; R-b reserved for confirmation). |
| A1.5 | **Run.** `scripts/cr_cm_t0.py` (imports `scripts/cr_cm_lib.py`; the lib gains an optional `avoid` argument to `placebo`, default behaviour unchanged) is committed before its single run. JSON written before printing. Everything else as frozen (M1–M4, readouts, secondary dwell, IM < 25 exclusion readout, S − W comparison). |
