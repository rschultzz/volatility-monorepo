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
| A7 | honoured | Swing 12 primary; 8 / 16 and the vol-scaled swing (O8) are readouts. |
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
