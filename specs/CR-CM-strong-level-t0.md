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
