# CR-CO — Replicate CR-CL G1-down on 2023-01-03 → 2024-04-02 (read-only, pre-registered, single hypothesis)

Spec freeze. Branch `feat/CR-CO-skew-g1down-2023` off `origin/main` `65c2214` (PR #94 merge), in its own worktree (`../volatility-monorepo-cr-co`). Read-only against the database. No table writes, no cron, no deploy, no display change. PR, don't merge.

Source: the kickoff prompt (Ryan, 2026-10-26), transcribed as K1–K10. O1–O12 are choices Code fixed before any 2023 data was pulled. Everything not listed here is CR-CL as shipped (`specs/CR-CL-skew-beyond-noise.md`, K1–K10, C1–C16, A1, A2, Approvals), re-used through CR-CN's period switch (`specs/CR-CN-skew-g1down-ra.md` N2 / A1; PR #95, open).

---

## Why

CR-CL's single primary, G1-down (a swing-12 decline leg during which raw 0DTE PM put skew **fell** beyond noise → next move up; Ryan's put-monetization theory), came out **INCONCLUSIVE, leaning toward** on 2026 train: ±10 race +17.4 pp [−1.2, +37.3], n 47. CR-CN tried to confirm it on R-a and closed at Step 0 (11 legs, not testable). This CR re-runs the same test, unchanged, on the train-era window 2023-01-03 → 2024-04-02, where no price outcome has been read for this question.

## Kickoff (Ryan, 2026-10-26)

| # | Decision | Value |
|---|---|---|
| K1 | Scope | Read-only, pre-registered, single hypothesis. Branch off `origin/main`; own worktree. Freeze with the conformance table. **A17 deviation pre-approved** on the CR-CL terms (measurement only, nothing shortlisted): record it, don't stop for it. Stop on any other "deviates". |
| K2 | Hypothesis | **Exactly CR-CL G1-down as run:** decline swing-12 leg, raw 0DTE PM put skew fell beyond noise (ratio < 0; CR-CK definitions and noise band **as shipped**) → next move **UP**. |
| K3 | Outcome | Same ±10 race from the confirmation bar close by 13:00 PT. |
| K4 | Baseline, inference | Same 3-key matched in-noise baseline; same session-cluster bootstrap (seed 20260920); legs ending after 11:00 PT readout only; 40-leg minimum; three verdicts (B1). |
| K5 | Code | Reuse CR-CL / CR-CN scripts with the date range switched; **no other change to the primary**. |
| K6 | Data | **2023-01-03 → 2024-04-02 only** (train era, never read for this question). R-a (from 2024-04-03), R-b (2025), 2026 train and the holdout are **not read**. |
| K7 | Step 0 (no outcomes) | (1) coverage of 0DTE PM SPX skew at minute resolution and of Databento ES 1-min bars, gaps listed, third Friday PM only (CR-BN); (2) ORATS quote time vs label by month, and the CR-CJ A1 alignment correlation on quote time — a period whose quote time can't be established is excluded and reported; (3) ES bars: Databento source rows, roll weeks (dying-contract risk) — roll-week sessions excluded if contaminated, reported; (4) counts: swing-12 legs, legs with a ratio, G1-down, all readout cells, baseline match coverage, A/A, era noise vs the shipped band (report only; band unchanged); (5) a **volatility-scaled swing readout, counts only for now**: swing = 12 × the session's implied move ÷ the 2026 train median implied move, everything else the same. **STOP for review after Step 0.** |
| K8 | Power gate | If G1-down < 40, report it and stop; don't run. |
| K9 | Verdict | As CR-CL / CR-CN K7: PASS if the 95 % CI lower bound > 0; REFUTED if the upper bound < 0; else INCONCLUSIVE; < 40 legs resolved and covered → INCONCLUSIVE (underpowered). |
| K10 | Wrap | Results (if run) to the CR note, setup-registry G1-down row, session note. PR, don't merge. Copy run logs before removing the worktree. Plain-English summary for Ryan: did the put-monetization bounce show up in 2023–early 2024? |

## Pre-run choices fixed by Code (2026-10-26, before any 2023 data was pulled)

| # | Choice | Value |
|---|---|---|
| O1 | Period and guard | New CR-CJ period `y23` = 2023-01-03 → 2024-04-02 in `scripts/cr_cj_lib.py` `PERIODS`, sessions = NYSE trading days minus the early closes 2023-07-03 and 2023-11-24 (added to `EARLY_CLOSES`, as CR-CJ treats R-a's). The loader's `guard` (every read inside one period, never 2025, never past the holdout cut) applies unchanged; `y23` ends the day before R-a starts, so no R-a row can be read. Only `y23` is loaded (plus O8's one train statistic). Caches `scripts/logs/cr_cj/{skew,es}_y23.parquet` (git-ignored). |
| O2 | Period switch (K5) | CR-CN's switch (`PERIOD = os.environ.get("CR_CL_PERIOD", "y26")`), cherry-picked from PR #95 commit `a37218e` (code only), with `OUT` widened to a map `{y26: cr_cl, ra: cr_cn, y23: cr_co}`. Default `y26` unchanged (CR-CN A1 code-level parity re-asserted; no train data loaded for it). |
| O3 | Skew rows (K7.1) | `load_skew` unchanged: `orats_monies_minute`, SPX, `expir_date = trade_date`, `expiry_tod = 'pm'`, 06:30 ≤ `snapshot_pt` < 13:00. Third Fridays: PM only (CR-BN); where the table holds only the AM row under the shared key (field-conventions), the session has no 0DTE PM skew and drops out (listed). Sessions with no 0DTE PM rows, or with < 80 % of the 390 RTH minutes present, are listed as gaps; sessions with no rows drop (as CR-CJ); thin sessions stay in and the per-leg ≥ 80 % coverage rule of `stickiness_read` handles them (unchanged). No backfill (read-only). |
| O4 | ES bars (K7.1, K7.3) | CR-CL reads `es_flow_1s_databento`, which starts 2024-04-03 (field-conventions). For `y23` the only Databento ES series is **`ironbeam_es_1m_bars` `source = 'databento'`** (fetched as `ES.c.0`, calendar roll; no contract column). `load_es("y23")` reads those RTH bars (06:30 ≤ start < 13:00 PT) into the same frame with the pseudo-symbol `ES.c.0`; everything downstream (legs, race, EDGE, P0) is unchanged. Field-conventions: minute OHLC built from the Databento 1-s table equals the Databento `ironbeam_es_1m_bars` outside roll weeks, so the substitution is like-for-like outside roll weeks. Step 0 counts `source` per session and lists sessions with no Databento RTH bars or any `source = 'ironbeam'` row (excluded). |
| O5 | Roll weeks (K7.3) | Roll window per quarterly expiry (H/M/U/Z, third Friday) = the Monday of the week before expiry week (L3) → expiry Friday. Within the window `ES.c.0` (calendar roll) is the dying contract. **Contamination test, per session, on bars only (no outcome):** (a) RTH volume < 25 % of the median RTH volume of the period's non-window sessions; or (b) more than 5 missing RTH minutes, or any missing run > 5 bars (C9's gap rule). Every window session is reported with volume ratio and missing minutes. **Rule:** a window session flagged by (a) or (b) is excluded; if ≥ half the window sessions of a quarter are flagged, the whole window is excluded for that quarter. Expiry Fridays drop anyway (no 0DTE PM skew, O3). Step 0 also reports the G1-down count with every window session excluded (count only). |
| O6 | Quote timing (K7.2) | Per calendar month: label − `quote_date` (median, p5, p95, share > 5 min, share unparsed) on 0DTE PM RTH rows. **Alignment (CR-CJ A1 method):** 1-min Δ `stock_price` vs 1-min Δ ES close, correlation at lags −20 … +5 min, on the label axis and on the quote-time axis (snapshot paired with the ES close at its `quote_date` minute, as `es_close_at_snapshot`). Because `stock_price` jitters ±15 pts before 2023-11-09 (CR-BH), a second check: 1-min Δ `vol50` vs Δ ES (expected negative). **Quote time is established for a month** if on the quote-time axis the stock_price correlation peaks at lag 0 with ρ ≥ 0.3, **or** (when stock_price fails ρ ≥ 0.3) the `vol50` correlation is most negative at lag 0 with ρ ≤ −0.2. A month failing both, or with > 20 % of rows unparsed / > 5 min from the label, is **excluded** and reported (K7.2). The A1 pairing itself (`skew_minutes`, drop > 5 min) is unchanged. `spot_price` is not used anywhere. |
| O7 | Noise band (K7.4) | As CR-CN N4: the CR-CK C6 band measured on `y23`, reported beside the shipped `NOISE_BAND_PP` and as a ratio. **Report only: the band is not changed and no count under a 2023 band is computed.** |
| O8 | Vol-scaled swing readout (K7.5) | swing_d = 12 × IM_d ÷ IM_train, where IM_d = canonical `bt_daily_features.implied_move_1d` for the session and IM_train = the median of the same field over the 2026 train sessions (`y26`, 2026-01-02 → 06-05). **Reading IM_train is the one train read in this CR:** a single pre-open statistic required by the kickoff's definition; no 2026 price, skew or outcome is read. Sessions without IM_d are excluded from this readout and counted. Legs from `swing_legs(H, L, swing_d)`, everything else (read, groups, strata with terciles re-cut on these legs, baseline) as the primary. **Step 0 reports its counts only**; whether it gets an outcome is Ryan's call at review. |
| O9 | Size terciles | Re-cut on `y23` by CR-CL C6b's code (as CR-CN N5); reported beside train's 20.0 / 28.5 and R-a's 20.92 / 29.5. |
| O10 | GEX readout | CR-CL's `gex_levels` places levels with the corrected level L (CR-BT), which needs `orats_oi_gamma.stock_price` to be a forward; it is **not** a forward before session 2024-04-03 (field-conventions, CR-BT era split). So on `y23` the GEX readout (A2.3 "G1-down by GEX tag") is **not computed**: `gex_levels` returns no levels with reason "pre-2024-04-03: placement undefined (CR-BT)". Readout only; the primary does not use it. |
| O11 | A/A | CR-CL C13 unchanged (200 session halves of the baseline pool, 500-draw bootstrap each, ±10 race on baseline legs only). |
| O12 | Run discipline, connection | Step 0 = `scripts/cr_co_step0.py` (runs `cr_cl_step0.py` under `y23` for counts, coverage, A/A; adds O3–O8), committed before it runs. If Ryan clears the run: `CR_CL_PERIOD=y23 apps/web/.venv/bin/python scripts/cr_cl_run.py`, **once**, with its parity assert set to the Step 0 counts and the O5 / O6 exclusions applied through the loader, both committed before the run. A crash before any number is printed or saved allows a fix-and-rerun, logged. `get_backfill_db_conn()`, `read_only = True`, SELECTs only. Outputs `scripts/logs/cr_co/` (git-ignored); copied to the main checkout before the worktree is removed. |

## Steps

- **Step 0** (K7): coverage; quote timing by month; ES roll weeks; funnel, cells, G1-down count, coverage, terciles, A/A; era noise band vs shipped; vol-scaled swing counts. **Stop for Ryan's review.** If G1-down < 40: report and stop (K8).
- **Step 1** (run, only if cleared): `cr_cl_run.py` under `y23`, one run.
- **Wrap**: vault CR note, setup-registry row, session note, PR.

## Standing rulings conformance (at spec freeze)

| Ruling | Status | Note |
|---|---|---|
| A1, A2, A4, A5, A9, A10, A13, A18, A19, A20, A21 | n/a | No levels traded, ranges, stops, entries, second touches or sizing; a direction test on legs. |
| A3 | honoured | No entry modelled; the clock starts at the leg's confirmation bar close. |
| A6 | honoured | G1-down is A6's "price down with put skew falling = bullish". |
| A7 | honoured (as changed 2026-10-26, Amendment A1) | Frozen: swing 12 primary; 8 / 16 and the vol-scaled swing (O8) readouts. **After A1:** A7 now reads a volatility-scaled swing (12 × session IM ÷ 2026 train median IM); the primary uses it; fixed 12 / 8 / 16 are readouts. |
| A8, A12 | n/a | The GEX readout that splits put vs call levels cannot be computed before 2024-04-03 (O10); no level population is used. |
| A11 | honoured | ±10 race primary, ±20 readout, favourable-excursion distribution. |
| A14 | honoured | CR-CK definitions and shipped band, unchanged (K2). Step 0 reports the 2023 band for information only (O7). |
| A15 | honoured | Stop-free direction test. |
| A16 | honoured | Trend Δ over the leg. |
| **A17** | **deviates — pre-approved by Ryan 2026-10-26** (kickoff K1) | Same deviation as CR-CL / CR-CN: skew alone as a direction predictor before the flow-first test has run. Terms: a **measurement** of what skew alone says about the next move; nothing shortlisted; a PASS means only that G1-down earns a place as a pre-registered add-on readout in the flow-first setup test. Logged in standing-rulings at wrap. |
| B1 | honoured | Three verdicts; < 40 = inconclusive (underpowered); K8 power gate. |
| B2 | honoured | Readouts disclosed, unadjusted, nothing chosen from them. |
| B3 | n/a | Not an optimisation. |
| B4 | honoured | 2023-01-03 → 2024-04-02 is train (≤ 2026-06-05, outside R-a / R-b); R-a, R-b and the holdout are not read (loader guard). The only 2026 train read is O8's median implied move (a pre-open feature, no outcome). |
| B5 | honoured, with a note | Databento bars for legs and outcomes. In this era Databento exists only as the `ES.c.0` minute bars in `ironbeam_es_1m_bars` (O4), with roll weeks tested (O5). |
| B6 | honoured | Labelled: a stop-free direction test, not a test of Ryan's trade. |
| B7 | honoured | EDGE / INTERIOR is the third match key (CR-CL approval 2). |
| B8 | n/a | No labels. |
| C1, C2 | n/a | No display. |
| R1–R7 | not used | |

One "deviates" (A17), pre-approved in the kickoff. No other "deviates": Code proceeds to Step 0 and stops there.

---

## Step 0 findings (2026-10-26, `scripts/cr_co_step0.py` @ `25e7fd4` → `scripts/logs/cr_co/step0.json`, `step0_extra.json`, `exclusions.json`; no group outcome computed)

One launch, exit 0. O2 parity held: under the default the module resolves `PERIOD = y26`, `OUT = logs/cr_cl`. Only `y23` was loaded, plus O8's one statistic (median 2026 train `implied_move_1d`, 102 sessions). R-a, 2025 and the holdout were not read.

### K8 power gate: **G1-down = 22 legs (19 covered) < 40, so stop; not run**

| G1-down count (swing 12, primary) | Legs |
|---|---|
| With the O5 / O6 exclusions (the test as frozen, 200 sessions) | **22** (19 covered) |
| No exclusions (227 sessions) | 22 |
| Every roll-window session excluded | 20 |

The exclusions cost no G1-down legs; the shortfall is the era itself. Train had 49, R-a 11.

### 1. Coverage (O3 / O4)

- **Calendar:** 311 sessions (2023-01-03 → 2024-04-02, early closes 2023-07-03 / 11-24 removed).
- **Skew:** `orats_monies_minute` has **no SPX rows of any expiry before 2023-05-01**, so 81 sessions (2023-01-03 → 04-28) are a table gap (ORATS holds them; not backfilled, read-only). From 2023-05-01: 230 sessions with 0DTE PM skew, monthly median 388–390 of 390 RTH minutes; one thin session (2023-05-23, 207 minutes; kept, per-leg coverage rule applies).
- **Third Fridays:** the 0DTE **PM** row is present on every third Friday 2023-05 → 2024-03 (366–386 RTH rows, AM 7–10 rows). The field-conventions line "PM weeklies on the same date are lost" does not hold for these `orats_monies_minute` sessions. PM only was used (O3).
- **ES:** `ironbeam_es_1m_bars` holds Databento rows on 307 sessions, with **no non-Databento rows** in the period. There are no RTH bars on the four quarterly expiry Fridays (2023-03-17, 09-15, 12-15, 2024-03-15), and 2023-06-16 has 1 bar: `ES.c.0` expires at the open. Non-roll sessions: 0 missing minutes (median), none with > 5.
- **Both series present:** 227 sessions.

### 2. Quote timing (O6)

- **2023 quotes are fresh: label − `quote_date` median 21–23 s every month** (p5 9–11, p95 41–51; > 5 min 0.2–0.3 %, unparsed 0 %). That is close to 2026 (10 s), **not** R-a's 129 s. At the primary legs' end snapshot the lag is median 21 s (G1-down 18 s), and the ES price paired with snapshot e is the leg's extreme bar on 278 / 341 legs (1 bar behind on 62, 2 on 1). So the R-a truncation does not apply here.
- **Alignment:** 1-min Δ `stock_price` vs Δ ES on the quote-time axis peaks at lag 0 in every month (ρ 0.40–0.66), and quote time beats the label slightly (2024-01: 0.663 vs 0.623). Lags −20 … +5 show no 15-min shift in `stock_price` (the CR-BH 2023 delay is in `spot_price`, which is not used).
- **Excluded by rule: 2023-09** (20 sessions). Both checks peak at lag 0, but `stock_price` ρ is 0.109 (< 0.3) and `vol50` ρ is −0.159 (needs ≤ −0.2), so the O6 rule fails. Probable cause is `stock_price` jitter that month (CR-BH ±15 pts); the quote lag itself is normal (22 s). Excluded as frozen; it would add no G1-down legs (the no-exclusion count is the same 22).

### 3. Roll weeks (O5)

`ES.c.0` volume vs the non-window median (1.20 M contracts per RTH session):

| Expiry | Prior week (Mon–Fri) | Expiry-week Mon / Tue | Wed / Thu | Fri |
|---|---|---|---|---|
| 2023-03-17 | 0.98–1.61 | 0.78 / 0.35 | 0.23 / 0.13 | no bars |
| 2023-06-16 | 1.00–1.31 | 0.31 / 0.31 | 0.22 / 0.12 | 1 bar |
| 2023-09-15 | 0.72–1.09 | 0.37 / 0.20 | 0.13 / 0.13 | no bars |
| 2023-12-15 | 0.91–1.24 | 0.30 / 0.16 | 0.22 / 0.14 | no bars |
| 2024-03-15 | 0.77–1.39 | 0.43 / 0.29 | 0.15 / 0.14 | no bars |

- **Contamination:** the calendar-roll series is the dying contract for the whole expiry week, and its volume falls to about 0.3× from the Monday, after the market rolled on the prior Thursday. No window session has missing minutes. Non-window sessions have a volume p5 of 0.69×.
- **Excluded by the O5 rule (< 0.25×, or the gap rule):** 17 sessions, the Wed–Fri of each expiry week plus 2023-09-12. No quarter reached the whole-window threshold. Expiry-week Mondays and Tuesdays at 0.30–0.43× pass the rule and stay in.
- **Count sensitivity with every window session excluded:** G1-down 20.

### 4. Counts (swing 12; 200 sessions after 33 excluded + 81 with no skew)

| Step | y23 (200 sessions) | R-a (186) | Train (107) |
|---|---|---|---|
| Legs in domain | 547 | 1,036 | 1,233 |
| Primary legs | 360 (late 187) | 681 | 972 |
| Ratio defined | 341 (move too small 13; insufficient 5; no read 1) | 399 | 597 |
| In noise → baseline | 241 (up 109, down 132) | 324 | 448 |
| Beyond noise | 100 (29.3 %): G1 38, G2 43, neither 19 | 75 (18.8 %) | 149 (25.0 %) |

Size tercile cuts (O9): **16.0 / 22.67 pts** (train 20.0 / 28.5; R-a 20.92 / 29.5).

The era is quiet: swing-12 legs are rare (median RTH range 36.5 pts; 4.0 swing-12 legs per session, 2.7 in the domain). But because SPX is lower (~4,500), almost none fail the 0.15 % "move too small" rule.

| Cell (pre-outcome; covered = stratum has ≥ 1 baseline leg) | Swing 12 | Ratio cut −0.5 / 1.5 | Late | Swing 8 | Swing 16 |
|---|---|---|---|---|---|
| G1-up | 16 (14) | 15 (13) | 4 | 36 (34) | 7 (6) |
| **G1-down** | **22 (19)** | 21 (18) | 5 | 43 (43) | 9 (7) |
| G2-up | 25 (18) | 21 (16) | 9 | 44 (40) | 8 (4) |
| G2-down | 18 (14) | 16 (13) | 12 | 46 (46) | 12 (8) |

- **Baseline coverage:** G1-down 19 / 22 covered with the EDGE key, 21 / 22 without it. Baseline strata: 94 occupied, median 2 legs, 33 with one leg.
- **Beyond-noise share by start bucket:** 18–33 %. The G1 legs cluster early again (07:00–07:59: 22 of 47 ratio-defined G1 legs, primary + late).
- **GEX readout:** not computed (O10). All 200 sessions show "pre-2024-04-03: placement undefined".

### A/A (O11; ±10 race on baseline legs only, 200 session halves, 500-draw bootstrap each)

| Direction scored | Up legs: median [2.5, 97.5 %] | CI excludes 0 | Down legs: median [2.5, 97.5 %] | CI excludes 0 |
|---|---|---|---|---|
| Continuation | −0.5 pp [−25.5, +32.0] | 4.5 % | −0.0 pp [−29.8, +25.8] | 2.5 % |
| Against | −1.0 pp [−36.3, +28.6] | 3.0 % | +0.7 pp [−30.0, +27.1] | 4.0 % |

Unbiased (medians within 1 pp of 0), at or below nominal 5 %, wide (±30 pp on ~60-leg halves).

### 2023 noise vs the shipped band (O7; report only, band not changed)

- **07:00–10:30:** the 2023 band is **about the same as the shipped band**, median ratio 1.02 (IQR 0.89–1.23). It runs higher at 07:00 (1.2–1.65×) and slightly lower at 09:30–10:00 (0.8–0.9×). So unlike R-a (0.7×), the shipped band is neither loose nor tight in this era.
- **After ~11:00 and on long windows from 10:30:** the 2023 SD explodes (30–84 vol pts) because near-expiry 0DTE `vol75 − vol50` blows up late in the session. These cells are beyond the primary (e ≤ 11:00) and are reported only.
- Median 0DTE put skew level 0.96 vol pts (R-a 1.87).

### 5. Vol-scaled swing readout (O8; counts only)

- **Scale:** swing_d = 12 × IM_d ÷ 65.71 (median 2026 train IM, n 102). The y23 median IM is 34.62, so swing_d is median **6.3 pts** (p5 3.4, p95 10.9). Nine sessions have no IM (excluded).
- **Funnel:** 2,351 legs; 1,644 primary; 738 ratio-defined (711 "move too small" — at a 6-pt swing many legs fall under the 0.15 % ES-move floor); baseline 531; beyond noise G1 87, G2 93.
- **Cells:** G1-up 37 (36), **G1-down 50 (50 covered)**, G2-up 45 (41), G2-down 48 (47). Terciles 12.0 / 16.5 pts.
- It is the only version that clears 40 in this era, but it is **not** the frozen primary. Running it would be a new pre-registered hypothesis (Ryan's call, K7.5).

### Gate summary for Ryan

- **K8: G1-down has 22 legs (< 40), so the CR stops here and is not run.** The primary cannot carry a verdict in this era: 2023-05 → 2024-04 moved little (4 swing-12 legs per session, against 11.5 on train).
- **Data gates:**
  - Skew coverage starts 2023-05-01 (81 sessions lost, a table gap).
  - Quote time is established in 14 of 15 months, and the 2023 lag is ~22 s (no R-a truncation). 2023-09 was excluded by rule.
  - Roll-week Wed–Fri excluded (17 sessions).
  - Third-Friday PM present; A/A clean; band ≈ shipped.
  - None of these change the G1-down count materially (22 / 22 / 20).
- **What could reach 40:**
  - Vol-scaled swing readout: 50 legs.
  - Swing 8 readout: 43 legs.
  - Backfilling ORATS minute monies for 2023-01 → 04 (+81 sessions; a write, outside this read-only CR).
  - Each would be a new pre-registration, not CR-CL as run.
- No outcome was computed on any group leg. `cr_cl_run.py` was not run or changed.

Stopped for Ryan's review.

---

## Amendment A1 (Ryan, 2026-10-26, from the Step 0 counts only — no outcome read): vol-scaled primary

Transcribed:
- **Standing-rulings A7 changed 2026-10-26** to a volatility-scaled swing: swing = 12 × the session's implied move ÷ the 2026 train median implied move.
- **The primary becomes G1-down on the vol-scaled swing** (50 legs, 50 covered at Step 0), i.e. the O8 legs.
- **Everything else exactly as frozen:** CR-CK definitions and shipped band, ratio < 0 beyond noise, h = up, ±10 race from the confirmation bar close by 13:00 PT, 3-key matched in-noise baseline (terciles cut on the vol-scaled primary legs, as O8), session-cluster bootstrap 10,000 draws seed 20260920, ≥ 40 legs resolved and covered, three verdicts (K9), legs ending after 11:00 PT readout only, O5 / O6 exclusions.
- **Fixed swing 12 (22 legs, underpowered) and swing 8 become readouts.**

How Code applies it (fixed before the run):

| # | Item | Value |
|---|---|---|
| A1.1 | Legs | `cr_cl_lib.vol_scaled_legs(D)`: per session `swing_legs(H, L, 12 × IM_d ÷ IM_train)`, IM as O8 (IM_train = 65.71, median of 102 train sessions); 9 sessions without IM contribute no legs. Same result as Step 0's O8 block (parity assert: 2,351 legs, 738 ratio-defined). |
| A1.2 | Readouts (no verdicts, unadjusted) | On the vol-scaled legs: G1-up, G2-up, G2-down; ratio cut −0.5 / 1.5; late legs. Fixed swing 12 (all four cells incl. G1-down), swing 8, and swing 16 (it comes out of the unchanged loop; disclosed). GEX readout not computed (O10). |
| A1.3 | Run | `CR_CL_PERIOD=y23 apps/web/.venv/bin/python scripts/cr_cl_run.py`, once, after this amendment and the script change are committed. `y26` path unchanged. |

Conformance after A1: A7 row updated above (honoured, as changed by Ryan). A17 still deviates, pre-approved. No new "deviates".

**Run log (A1.3).** Launch 1 crashed on the parity assert before any number was computed or saved: the assert compared *all* ratio-defined vol-scaled legs (1,075) with Step 0's *primary* ratio-defined count (738). The legs were identical to Step 0 (2,351 legs; 738 primary ratio-defined; G1-down 50). Fixed the assert to the primary count (O12 / C15 fix-and-rerun), committed, relaunched once.

---

## Results (run 2026-10-26, `cr_cl_run.py` @ the run-assert commit, `CR_CL_PERIOD=y23`, launch 2 → `scripts/logs/cr_co/results.json`, `legs.parquet`, `run_stdout.log`)

200 sessions (2023-05-01 → 2024-04-02 after O5 / O6 exclusions); vol-scaled swing with IM_train 65.71 pts; 9 sessions without IM. 10,000 session-cluster draws, seed 20260920.

### Primary — G1-down on the vol-scaled swing → up, ±10 race

| | Value |
|---|---|
| Legs / sessions | 50 / 38; resolved and covered **50** (32 win, 18 loss, 0 tie, 0 unresolved); EDGE share 60 % |
| Win rate, group | **64.0 %** |
| Win rate, matched in-noise baseline | 50.7 % (262 down legs: 130 win / 110 loss / 22 unresolved) |
| Difference | **+13.3 pp, 95 % CI [−7.2, +32.0]**, two-sided bootstrap p 0.21 |
| **Verdict (K9)** | **INCONCLUSIVE** (CI spans 0; not refuted: upper bound > 0) |

Same direction and similar size as 2026 train (CR-CL: +17.4 pp [−1.2, +37.3], n 47), again with a CI that reaches below 0.

### Primary-cell readouts (unadjusted, no verdicts)

- ±20 race: +26.3 pp [−10.0, +51.0], n 28 resolved.
- Signed move: +30 min +1.05 pts [−1.98, +4.16]; +60 min +1.90 pts [−2.46, +6.31] (group +0.31 / +2.77 vs baseline −0.74 / +0.87).
- MFE before −10: median 13.75 pts (baseline 9.62); reaches +10 on 64 % (baseline 49.6 %); of those, +15 on 75 % (72 %) and +20 on 50 % (45 %).

### Other readouts (±10 race diff, 95 % CI, n resolved and covered)

| Readout | G1-up | **G1-down** | G2-up | G2-down |
|---|---|---|---|---|
| Vol-scaled (primary legs) | +5.6 [−16.1, +27.6] 34 | **+13.3 [−7.2, +32.0] 50** | +6.9 [−13.9, +29.5] 41 | −8.9 [−28.7, +13.3] 44 |
| Vol-scaled, ratio cut −0.5 / 1.5 | +7.9 [−14.6, +30.8] 33 | +12.2 [−8.3, +30.9] 49 | +14.8 [−8.4, +39.5] 32 | −5.3 [−26.8, +19.4] 38 |
| Vol-scaled, late legs | +4.8 [−71, +75] 5 | +5.4 [−70, +63] 8 | +4.5 [−41, +46] 10 | −1.1 [−49, +38] 12 |
| Fixed swing 12 | −2.5 [−43, +39] 14 | +13.9 [−22.2, +51.4] 18 | +23.9 [−7.3, +65.6] 18 | +4.1 [−34, +48] 14 |
| Fixed swing 8 | −3.3 [−25.0, +19.6] 33 | +11.6 [−10.5, +35.1] 40 | +9.5 [−12.4, +29.3] 40 | −18.5 [−38.9, +4.4] 46 |
| Fixed swing 16 | n 6 | +7.1 [−60, +67] 7 | n 4 | n 7 |

GEX readout: not computed (O10; every leg "no levels").

The G1-down point estimate is +12 to +14 pp in every version with n ≥ 40 (vol-scaled, ratio cut, swing 8) and in fixed swing 12; none has a CI excluding 0. G2 (skew relaxed as mechanics imply → continuation) shows nothing consistent.
