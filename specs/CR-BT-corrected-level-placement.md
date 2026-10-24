# CR-BT — Corrected Level Placement and Chart Toggle

Spec freeze. Branch `feat/CR-BT-corrected-level-placement` off `origin/main` `e78774f` (the CR-BS merge, PR #70). Read-only against the database; no table writes, no cron.

Source: vault `Dash/CRs/CR-BT — Corrected Level Placement and Chart Toggle.md` (last_reviewed 2026-10-24). Body below is transcribed verbatim from that note (frontmatter omitted); the locked decisions L1–L10 were confirmed by the kickoff prompt (Ryan, 2026-10-24).

---


## Why

The colored GEX lines on the price chart are placed with `discounted_level` = strike × exp((r − q)(dte + 1)/252), using **the option's own DTE** (`apps/cron/job_orats_eod.py:121`). Chat review of the code on 2026-10-24 (no DB run) gives three problems:

1. **The direction is wrong for longer-dated options.** An option's gamma peaks when *its own expiry's forward* equals the strike. So in SPX cash the peak sits *below* the strike, by carry to that option's expiry. `discounted_level` moves it *up* by that carry.
2. **The ES contract's expiry is missing.** On an ES chart, every price sits above SPX cash by carry to the *ES* expiry. The correct ES-chart location is

   **L = K × e^(carry × (T_ES − T_option))**, equivalently **L = K × (ES price ÷ F_option)**, both prices taken at the same snapshot.

   - An option expiring with the ES contract plots exactly at its strike.
   - 0DTE plots near strike + ES basis.
   - Options expiring after ES plot below their strike.
3. **One strike becomes several lines.** `_fetch_gex_grouped_by_level` buckets by `discounted_level`. One strike's 0DTE, weekly and monthly OI are shifted by different, wrong amounts and land in different buckets.

**Chat's rough error estimate** (SPX ≈ 7700, ES Z26 ~55 days out, basis ~25, carry ~0.45 pts per calendar day; unmeasured):

| Expiry | Current shift | Correct shift | Current error |
|---|---|---|---|
| 0DTE | +1 | ~+25 | ~24 too low |
| 7 DTE | +10 | ~+22 | ~12 too low |
| 30 DTE | +38 | ~+11 | ~27 too high |
| 55 DTE (Dec quarterly) | +68 | ~0 | ~68 too high |
| 90 DTE | +112 | ~−16 | ~128 too high |

This was first flagged in CR-BI Step 0 (2026-09-20). On 2026-09-21 Ryan decided the level math must account for both the option's and the ES contract's expiry, built in parallel without touching the current system. This CR does that.

**Ryan's requirement (2026-10-24):** recalculate the levels, leave the old ones exactly as they are, and add a toggle to switch between the two.

## Locked decisions (Ryan to confirm before spec freeze)

| # | Decision | Value |
|---|---|---|
| L1 | Scope | (a) A pure shared function that places each `orats_oi_gamma` row on a target price axis. (b) A price-chart toggle, "Levels: Current \| Corrected", **default Current**. No table writes, no cron changes, no change to `discounted_level`, `orats_gex_landscape`, `bt_daily_features`, `bt_daily_outcomes`, or the landscape panel. |
| L2 | Placement formula | **L = K × (P_target ÷ F_e)**. F_e = the row's own `orats_oi_gamma.stock_price` (per-expiry parity-implied price at the prior session's 15:46 ET snapshot; field-conventions). P_target = the target-frame price at that same snapshot. No rates, no dividend yields, no day counts. |
| L3 | Target frames | One function, two callers. **ES chart:** P_target = ES front-month price at the prior session's 15:46 ET. **SPX cash** (used by CR-BU): P_target = S0 = `stock_price` of the nearest live expiry (CR-BK A1.1). |
| L4 | ES snapshot | Close of the `ironbeam_es_1m_bars` bar starting 15:45 ET on the API trade date. Bars carry the UTC bar *start* (field-conventions); DST is handled through `packages/shared/sessions.py` or `zoneinfo`, never a fixed UTC offset. Step 0 confirms the alignment. |
| L5 | Same selection pipeline | Corrected levels are aggregated and filtered with **exactly** the current machinery, so placement is the only difference: same bucket (`GEX_LEVEL_BUCKET`), same band padding, threshold, minimum spacing, maximum lines, color scale, line width and opacity. Only the level key changes: corrected L instead of `discounted_level`. |
| L6 | Per-expiry popup | The `gex_segments` per-expiration breakdown follows the toggle, built from corrected L, so its rows still sum to the line. |
| L7 | Old path untouched | With the toggle on Current (or the parameter absent), the `/api/ironbeam/bars` response is **byte-identical** to today's for the same inputs. A test enforces this. |
| L8 | Toggle mechanics | A query parameter on the existing route (`levels=corrected`; absent = current). The frontend control sits beside the existing GEX controls. Remembering the choice in `localStorage` is allowed (wrapped in try/catch). No other visible change: no layout, colors or spacing edits (CLAUDE.md visible-change discipline; this spec authorises only the toggle control and the alternate line set). |
| L9 | Intraday | Levels stay fixed through the session, as today (prior-day OI, prior-day snapshot). Intraday recasting is out of scope; see Not in scope. |
| L10 | Holdout | This CR reads no outcomes. Live display of any date is allowed. |

## Step 0 — diagnosis gate (answer all before implementation; contradiction-stop)

1. **`stock_price` is a per-expiry forward.** On 20 random train dates plus 2026-10-23:
   - plot or tabulate F_e / S0 − 1 against calendar days to expiry; it should rise smoothly with DTE;
   - report non-monotone dates and any expiry with a missing or zero `stock_price` (count rows, and share of total |GEX|).
2. **Price-space check against ES** (this is the measurement, per the CLAUDE.md rule):
   - For each sample date, take the SPX expiry nearest the front ES contract's expiry (the quarterly third Friday). Report |F_matched − ES(15:46 ET)|: median, p90 and max.
   - Gate: median ≤ 5 pts. If it fails, stop and report; do not substitute rates.
   - Flag the AM/PM third-Friday overwrite (field-conventions) where it touches the matched expiry.
3. **ES contract identity at the snapshot.** `ironbeam_es_1m_bars` has no contract column.
   - Confirm the live Ironbeam feed is the front month on 2026 dates, including the 2026-09 roll patch window.
   - Count train dates in 2023–2025 roll weeks, where the ES bars are the dying contract (calendar-roll `ES.c.0`).
   - Those dates are display-only caveats for this CR, but CR-BU needs the count.
4. **Worked day 2026-10-23.**
   - For the 10 largest |GEX| rows: strike, expiry, DTE, `discounted_level`, corrected L, and the difference.
   - Then the top selected lines under Current vs Corrected.
   - Compare against the chat estimate table above, and say where it was wrong.
5. **Render path.**
   - Identify exactly which payload fields the React price chart draws: `gex_levels`, `gex_segments`, or both (`react_price_preview/src/App.jsx`, `PriceChart.jsx`).
   - Identify the smallest change that adds the toggle.
   - Name every file to be touched. If anything outside `apps/web/modules/Ironbeam/`, `packages/shared/` and `react_price_preview/src/` is needed, stop and ask.
6. **Byte-identity test plan for L7.** Say how the test captures the current response (fixture date, mocked DB or recorded rows).

## Implementation order (one commit each)

1. `packages/shared/level_placement.py`: `place_levels(rows, p_target) -> rows with level_corrected`, pure, plus unit tests:
   - the matched-expiry row returns its strike to within float tolerance when F_e = P_target;
   - 0DTE shifts by about the basis;
   - an expiry after ES shifts down;
   - empty and NaN inputs are handled.
2. Backend:
   - a `levels` parameter on `/api/ironbeam/bars`;
   - corrected variants of `_fetch_gex_grouped_by_level` / `_fetch_gex_grouped_by_level_and_expir` built on rows + `place_levels`, with the existing functions untouched;
   - the L7 byte-identity test.
3. Frontend toggle (L8).
4. Smoke on 2026-10-23, one 2026 roll-adjacent date, and one third Friday:
   - screenshots of Current vs Corrected;
   - the per-expiry popup sums to its line in both modes.

## Outputs

- Spec `specs/CR-BT-corrected-level-placement.md` with the Step 0 answers and the smoke screenshots.
- Session note in `Dash/sessions/`; new field-conventions row for corrected L once Step 0 measures it.
- Deploy: the web service only. No cron changes.

## Not in scope

- Any change to stored levels, the landscape panel, regimes, outcomes or cards.
- Testing whether the levels work. That is CR-BU.
- Intraday recasting of gamma size (time decay, live IV), CR "C". It depends on CR-BU finding something, and it cannot add new OI.
- The CR-BK repriced-profile toggle on the landscape panel (separate, lower priority).
- The dealer-sign assumption.

## Kickoff prompt for Claude Code

```
Start CR-BT. First action: git fetch && git checkout -b feat/CR-BT-corrected-level-placement origin/main (confirm CR-BT is unused; rename if not).
Read the vault CR at "/Users/ryan/My Drive/Obsidian Vault/Anonymous/Dash/CRs/CR-BT — Corrected Level Placement and Chart Toggle.md" and components/field-conventions.md.
Commit 1: spec freeze to specs/CR-BT-corrected-level-placement.md, locked decisions transcribed verbatim.
Commit 2: Step 0 findings appended to the spec. Answer all six gate questions; stop and report to me before any implementation; stop immediately if gate 2 fails or anything contradicts the spec.
Read-only against the database. No table writes. Do not modify discounted_level, job_orats_eod.py, orats_gex_landscape, or the existing _fetch_gex_grouped_by_level* functions.
After I approve Step 0: implement steps 1-4, one commit each; with the toggle off the /api/ironbeam/bars response must be byte-identical to today's.
```

---

## Step 0 findings (2026-10-24)

Script `scripts/cr_bt_step0.py` (read-only session, SELECTs only; log `scripts/logs/cr_bt/step0.log`, git-ignored). Definitions fixed in the script docstring before the run: session date D = `orats_oi_gamma.trade_date`; API date A = `prev_trading_day(D)`; live rows = `expir_date >= D` (the chart's filter); S0 = `stock_price` of the nearest live expiry; ES snapshot = close of the `ironbeam_es_1m_bars` bar whose UTC start is 15:45 America/New_York on A (zoneinfo); front ES contract = first quarterly third Friday ≥ A (ingest rule, dying contract through expiry); sample = 20 train dates (2023-05-01 … 2026-06-05, numpy seed 17) + 2026-10-23.

**Verdict: STOP — gate 2 passes as written, but gate 1 contradicts L2 for every session before 2024-04-03, and the spec has no rule for sessions without an ES 15:45 ET bar. Amendment needed before implementation (see "Decisions needed").**

### Q1 — `stock_price` is a per-expiry forward — **CONTRADICTED before 2024-04-03**

- Universe: 958 sessions 2023-01-02 → 2026-10-23, 9,061,256 live rows. `stock_price` NULL/0 on **0** rows; `discounted_level` NULL on **0** live rows. `stock_price` is constant within (D, expiry) except on 3,525 of 44,754 groups (tiny spreads; the script uses MIN).
- Sample (F_e/S0 − 1 vs calendar days from A):

| era | sessions | Spearman(F_e, cd) | carry to ≤120 cd | F_e/S0−1 at 30 cd |
|---|---|---|---|---|
| < 2024-04-03 (7 sample dates) | 2023-05-16 … 2023-12-25 | −0.89 … −0.99 | **−0.09 … −0.23 pts/cd** | −1.6 … −33.6 bp |
| ≥ 2024-04-03 (14 sample dates) | 2024-06-12 … 2026-10-23 | +0.999 … +1.000 | +0.48 … +0.77 pts/cd | +19.7 … +42.2 bp |

- Every session, monthly (Spearman > 0.9 and positive slope = "forward-like"): **0–1 forward-like sessions per month 2023-01 → 2024-03; every session forward-like from 2024-04-03 on** (last non-forward-like session 2024-04-02; isolated flips 2023-03-07 and 2023-05-19). Train sessions not forward-like: **231 of 778**.
- Before 2024-04-03, `stock_price` falls with expiry at about S·q per day (q ≈ 1.2–1.5 %): it behaves like a dividend-discounted spot (≈ S·e^(−qT)), not the forward S·e^((r−q)T). From 2024-04-03 it rises at about S·(r−q) per day, as a forward should.
- Non-monotone after 2024-04-03: 0–6 consecutive-expiry dips per sample date, none larger than 1 pt (2024-09-19 has 3 dips > 1 pt, roll week).
- **This contradicts L2** ("F_e = per-expiry parity-implied price") and the field-conventions row for `orats_oi_gamma.stock_price` (CR-BK Step 0) for 2023-01 → 2024-04-02. Under L2, corrected L before 2024-04-03 would be shifted by about K·r·T too far upward (≈ 30 pts at the ES expiry in 2023).

### Q2 — price-space check against ES — **gate PASS as written (median 0.70 ≤ 5); FAILS for the pre-2024-04-03 era**

| set | n | median \|F_matched − ES\| | p90 | max | signed median |
|---|---|---|---|---|---|
| **gate: sample (21)** | 21 | **0.70** | 33.24 | 58.81 | |
| sample, ≥ 2024-04-03 | 14 | 0.18 | 1.39 | 3.18 | |
| sample, < 2024-04-03 | 7 | 25.32 | 55.05 | 58.81 | |
| all sessions with a bar | 930 | 0.48 | 39.89 | 71.79 | |
| ≥ 2024-04-03, excl. roll weeks | 536 | **0.23** | **0.85** | 8.52 | −0.07 |
| ≥ 2024-04-03, all | 625 | 0.23 | 0.97 | 69.05 (roll mismatch) | |
| < 2024-04-03, excl. roll weeks | 261 | **30.76** | 53.60 | 71.79 | −30.76 |
| 2025 excl. roll weeks | 208 | 0.25 | 1.12 | 3.18 | −0.10 |
| 2026 excl. roll weeks | 171 | 0.20 | 0.75 | 8.52 | −0.02 |

- The sample passes only because 14 of its 21 dates fall after 2024-04-03. In the forward-like era the matched expiry's `stock_price` reproduces the ES 15:45 ET bar close to a quarter of a point: the placement formula is price-validated there.
- **L4 alignment confirmed:** median |F_matched − ES close| by bar start, non-roll-week sessions: 15:43 2.39, 15:44 1.84, **15:45 0.45**, 15:46 1.82, 15:47 2.54 — a sharp minimum at the bar starting 15:45 ET (closing ≈ 15:46), matching ORATS' 15:46 ET snapshot.
- Matched expiry = the ES third Friday itself on 881 sessions (gap 1/3/4 days on 61/15/1 sessions). **AM/PM overwrite:** the matched expiry is a third Friday on every session, so it is always the PK-collided AM/PM row. AM and PM same-date forwards differ by hours of carry (< 0.1 pt), and the residual above is 0.23, so the overwrite does not show in F_e. It does still affect that expiry's OI/GEX (field-conventions), unchanged by this CR.
- **No ES 15:45 ET bar on A for 28 sessions** (spec has no rule): early closes (A = 2023-07-03, 2023-11-24, 2024-07-03, 2024-11-29, 2024-12-24, 2025-07-03, 2025-11-28, 2025-12-24, 2026-07-02), ES expiry Fridays in the Databento era (A = 2023-03-17, 06-16, 09-15, 12-15; 2024-03-15, 06-21, 09-20, 12-20; 2025-03-21, 06-20, 09-19, 12-19), the 2022-12-30 start, and 2026 feed gaps (A = 2026-02-17, 02-18, 06-18, 07-14, 07-15, 09-04).

### Q3 — ES contract identity at the snapshot

- 2026 Ironbeam feed vs the two quarterly forwards: the 15:45 ET close matches the **front** (ingest rule) contract to < 1 pt on all but these sessions, where it matches the **next** contract: D = 2026-03-23 (A = 03-20, H26 expired at that morning's open), D = 2026-06-16 → 06-22 (A = 06-15 → 06-18: the feed switched to U26 at Sun 2026-06-14 18:00 ET, five days **before** M26 expiry — the ingest did not carry the dying contract through expiry this time), D = 2026-09-15 → 09-21 (A = 09-14 → 09-18: the CR-BF patch window, ES Z26 by design). (The script's "next" list also shows 2026-02-18/19, 06-19, 07-03, 07-15/16, 09-07 — these are missing-bar NaNs, not contract matches.)
- Contract identity does not bias corrected placement as long as the snapshot and the displayed bars are the same contract: P_target is read from the same series that is drawn. **It does bias it when a contract switch falls between A 15:45 ET and the end of session D.** 2026 sessions of that kind: **2026-03-20** (switch at Fri 09:30 ET, +49 pts: ETH bars H26, RTH bars M26, levels H26-frame), **2026-06-15** (A = 06-12 M26 snapshot, bars U26 from Sun 18:00), **2026-09-14** (A = 09-11 U26 snapshot, bars Z26 from the patch). On those sessions corrected lines sit off by the calendar spread (~50–70 pts). In 2023–2025 (`ES.c.0`) the analogous session is the expiry Friday of each quarter; the following Monday has no snapshot (missing-bar list above).
- Train sessions in roll weeks (CR-BM/BO definition, Monday of the week before the third Friday → expiry Friday): 2023 **29**, 2024 **39**, 2025 **39** → **2023–2025: 107** (identical whether D or A is tested); 2026 train 10. Sessions with A = ES expiry Friday: 2023 3, 2024 4, 2025 4, 2026 1.

### Q4 — worked day 2026-10-23

A = 2026-10-22; ES 15:45 ET bar close **7686.00** (Ironbeam, Z26); S0 7649.40; ES − S0 **+36.60** (chat assumed ~25).

Top-10 |GEX| rows:

| strike | expiry | dte | F_e | `discounted_level` | corrected L | L − `discounted_level` |
|---|---|---|---|---|---|---|
| 7650 | 2026-10-30 | 7 | 7655.30 | 7659.94 | 7680.68 | +20.74 |
| 7750 | 2026-10-30 | 7 | 7655.30 | 7760.07 | 7781.08 | +21.01 |
| 7650 | 2026-10-23 | 0 | 7649.40 | 7651.24 | 7686.60 | +35.36 |
| 8000 | 2027-06-17 | 237 | 7840.02 | 8263.46 | 7842.83 | −420.62 |
| 7000 | 2027-06-17 | 237 | 7840.02 | 7230.52 | 6862.48 | −368.05 |
| 7700 | 2026-10-23 | 0 | 7649.40 | 7701.25 | 7736.84 | +35.59 |
| 7600 | 2026-10-23 | 0 | 7649.40 | 7601.23 | 7636.36 | +35.13 |
| 7700 | 2026-10-30 | 7 | 7655.30 | 7710.00 | 7730.88 | +20.87 |
| 7500 | 2026-10-30 | 7 | 7655.30 | 7509.74 | 7530.08 | +20.33 |
| 8040 | 2026-12-31 | 69 | 7697.90 | 8107.39 | 8027.57 | −79.82 |

Chat estimate vs measured (shift = level − strike, median over the expiry's live rows):

| expiry | current shift (chat) | corrected shift (chat) | current error (chat) |
|---|---|---|---|
| 0DTE 10-23 | +1.22 (+1) | **+35.85** (+25) | −34.6 (−24) |
| 7 DTE 10-30 | +9.47 (+10) | **+29.21** (+22) | −19.7 (−12) |
| 31 DTE 11-23 | +40.99 (+38) | +17.34 (+11) | +23.7 (+27) |
| 56 DTE 12-18 (Dec quarterly) | **+44.99** (+68) | −0.61 (0) | +45.6 (+68) |
| 84 DTE 2027-01-15 | +80.08 (+112) | −27.52 (−16) | +107.6 (+128) |

Where chat was wrong: (i) the basis was +36.6 that day, not ~25, so the short end is under-placed by ~35 pts at 0DTE and ~20 at 7 DTE, more than estimated; (ii) the carry slope in F_e is ~0.6 pts/cd, steeper than the 0.45 assumed, so corrected shifts below the ES expiry fall faster; (iii) the current shift on **third-Friday AM expiries is not monotone in DTE** — 12-18 +45 vs 12-11 +64, 2027-03-19 +122 vs 02-26 +139, 2027-06-17 +208 vs 05-21 +234, 2027-09-17 +301 vs 08-20 +352 — so `discounted_level` uses different (r, q) on AM monthlies; chat's +68 at the Dec quarterly is +45; (iv) **problem 3 of "Why" is only half right**: under the corrected formula one strike still lands in several buckets (706 of 887 strikes in > 1 bucket under both Current and Corrected), because each expiry's gamma peak really does sit at a different ES price. The correction fixes the amounts and direction, not the multiplicity.

Selected lines (band [7498.50, 7864.25] from 1,320 session bars; threshold 10 B; spacing 5; max 80): Current 60, Corrected 60. Top by magnitude:

| Current level | net B | mag B | | Corrected level | net B | mag B |
|---|---|---|---|---|---|---|
| 7660 | −353.9 | 401.5 | | 7681 | −352.7 | 400.3 |
| 7760 | −265.7 | 380.3 | | 7781 | −266.6 | 376.3 |
| 7651 | −280.5 | 324.2 | | 7687 | −271.7 | 319.0 |
| 7701 | −146.9 | 264.6 | | 7843 | +1.3 | 297.4 |
| 7601 | −211.6 | 239.5 | | 7737 | −130.4 | 265.7 |
| 7710 | −133.3 | 223.6 | | 7636 | −225.5 | 254.9 |
| 7751 | −54.4 | 190.6 | | 7731 | −135.2 | 228.3 |
| 7510 | −174.8 | 183.2 | | 7530 | −179.2 | 187.8 |

The two dominant lines move +21 pts (7660 → 7681, 7760 → 7781). The 2027-06-17 8000-strike LEAP mass (at 8263 under Current, outside the band) enters the band at 7843 under Corrected.

### Q5 — render path

- The React chart draws **`gex_segments` only**. `gex_levels` is in the payload but nothing in `react_price_preview/src` reads it (grep: only `gex_segments` in `App.jsx` and `PriceChart.jsx`). Each segment's `expirations` drives the popup and the GEX panel (Max-DTE slider recomputes net/call/put from `expirations`), so L6 is load-bearing.
- `/api/ironbeam/bars` is fetched in three places in `App.jsx` (center, multi-day extras, live refresh, lines ~1103/1192/1342); segments for every loaded session come from `_build_gex_segments_for_session(load_date, …)`, so corrected mode needs one snapshot per loaded session, not only the center date.
- GEX controls live in the PriceChart settings modal (draft state + `applySettings`, Min |GEX| slider at ~3190). Smallest change: a two-option "Levels: Current | Corrected" control in that modal beside Min |GEX|, applied via a new `onApplyLevelsMode` prop; `App.jsx` holds the state (localStorage key, try/catch), adds `levels=corrected` to the three URLs only when Corrected, and adds the state to the three effects' deps.
- Files to touch: `packages/shared/level_placement.py` (new), `packages/shared/tests/test_level_placement.py` (new), `apps/web/modules/Ironbeam/callbacks.py` (new functions + parameter threading only; existing `_fetch_gex_grouped_by_level*` untouched), `apps/web/modules/Ironbeam/tests/test_bars_levels_toggle.py` + a recorded fixture (new), `react_price_preview/src/App.jsx`, `react_price_preview/src/components/PriceChart.jsx`. **Nothing outside the three allowed directories.**
- Detail for step 2: the existing queries bucket with Postgres `ROUND()` on double (half away from zero); the corrected path buckets in Python and must use the same rounding (not numpy's half-to-even). The corrected row query keeps the existing WHERE clause (`discounted_level IS NOT NULL`, `expir_date >= :d`, ticker) so the row universe is identical (L5); it removes 0 live rows today.

### Q6 — byte-identity test plan (L7)

- Fixture: session 2026-10-23, `interval=1min`, phase `center` (and a `multi` call with `days_either_side=1`). Record, read-only, the exact DataFrames the payload builder consumes — `_fetch_bars_pt` output per session, `_fetch_gex_grouped_by_level` and `_fetch_gex_grouped_by_level_and_expir` per session, `fetch_skew_data` at 06:33 — to a JSON fixture under `apps/web/modules/Ironbeam/tests/fixtures/`.
- Golden: build the `/api/ironbeam/bars` response with the Flask test client from the **unmodified** `origin/main` module (`e78774f`), with those functions monkeypatched to return the fixture, and store the raw response bytes (golden file) before any edit to `callbacks.py`.
- Test: same harness on the branch; assert `response.data == golden` for (a) no `levels` param, (b) `levels=current`, (c) an unknown value (`levels=foo`), in both phases. A fourth case asserts `levels=corrected` differs and that each segment's `expirations` sum to its `net_gamma`.
- Live cross-check in the step-4 smoke: run the local server against the DB (read-only) for 2026-10-23 on `origin/main` and on the branch with no `levels` param, compare sha256 of the bodies.
- Design consequence: Current mode must not add any payload key; a `levels` echo / note appears only in Corrected responses.

### Decisions needed (amendment before implementation)

1. **Pre-2024-04-03 sessions** (`stock_price` not a forward; L2 invalid there; rates are ruled out). Recommendation: Corrected mode draws **no** lines for sessions D < 2024-04-03 and the payload carries a note; the toggle remains usable on later dates. Also amend the field-conventions row for `orats_oi_gamma.stock_price` (forward-like only from 2024-04-03) — CR-BK's S0 and CR-BU rely on it.
2. **No ES 15:45 ET bar on A** (28 sessions; early closes, Databento expiry Fridays, 2026 feed gaps). Recommendation: same as 1 — no corrected lines + note, rather than a substitute bar (the ORATS snapshot time on early-close days is unmeasured).
3. **Snapshot/display contract mismatch** (2026-03-20, 06-15, 09-14; one session per quarter in 2023–2025). Recommendation: accept as a display caveat and record the list; no code rule.
4. "Why" problem 3 correction (multiplicity remains under Corrected) — wording only, no change to L1–L10.

---

## Amendment A1 (Ryan, 2026-10-24, after Step 0)

Approved as recommended in Step 0 "Decisions needed". L1–L10 stand; these clauses add rules L2/L4 did not cover.

| # | Clause |
|---|---|
| A1.1 | **Forward-era floor.** Corrected placement is defined only for session dates D ≥ **2024-04-03** (first session from which `orats_oi_gamma.stock_price` is forward-like on every session, Step 0 Q1). For D < 2024-04-03, Corrected mode draws **no** GEX lines for that session (no segment, no `gex_levels` row). No rate/dividend substitute. |
| A1.2 | **Missing ES snapshot.** If the `ironbeam_es_1m_bars` bar starting 15:45 America/New_York on A = `prev_trading_day(D)` is absent (or its close is not a positive finite number), Corrected mode draws no lines for that session. No substitute bar. |
| A1.3 | **Payload note.** Corrected responses (only) carry `"levels": "corrected"` and `"levels_notes"`: a list of `{"session_date", "reason"}` for every loaded session without corrected lines, reason ∈ `pre_forward_era` / `no_es_snapshot` / `no_rows`. Current-mode responses gain no key (L7). |
| A1.4 | **Roll-mismatch sessions** (contract switch between the A 15:45 ET snapshot and the end of session D — 2026-03-20, 2026-06-15, 2026-09-14; the quarterly expiry Friday in 2023–2025) are accepted as a display caveat: no code rule; listed here and in the session note for CR-BU. |
| A1.5 | **"Why" problem 3, corrected wording.** The correction fixes the amount and direction of each expiry's shift, not the multiplicity: one strike's expiries legitimately land at different ES prices (706/887 strikes in > 1 bucket on 2026-10-23 under both modes). |
| A1.6 | **Bucketing.** The corrected path buckets in Python with half-away-from-zero rounding, matching Postgres `ROUND(double)` used by the existing queries, on the same row universe (same WHERE clause). |

---

## Implementation (2026-10-24)

| step | commit | what |
|---|---|---|
| amend | `9db3acd` | Amendment A1 |
| 1 | `0fc88d3` | `packages/shared/level_placement.py`: `place_levels`, `bucket_levels` (half away from zero = Postgres `ROUND`), `es_snapshot_bar_start_utc` (zoneinfo), `CORRECTED_LEVELS_FIRST_SESSION` 2024-04-03; 13 unit tests |
| 2 | `dfdadc8` | `callbacks.py`: `_fetch_gex_rows_for_placement`, `_fetch_es_snapshot_close`, `_corrected_gex_tables` (new); `levels` route parameter → `levels_mode`; `_build_gex_segments_for_session(gex_tables=…)`; Corrected-only `levels` / `levels_notes` keys. `_fetch_gex_grouped_by_level*` and `_select_levels` unchanged (diffed against `e78774f`). Tests: `tests/test_bars_levels_toggle.py` (9), record/replay harness `tests/bars_fixture.py`, `tests/record_bars_fixture.py`, fixtures |
| 3 | `e57d8fa` | `App.jsx` state + localStorage `ib-react-gex-levels-mode` (try/catch) + `levels=corrected` on the three bars fetches; `PriceChart.jsx` "Levels: Current / Corrected" radio in the settings modal under Min \|GEX\| |

**L7 golden.** `fixtures/cr_bt_bars_golden_center.json` (sha256 `86b5cb49…`, 245,329 B) and `…_multi.json` (`8645e12a…`, 240,164 B) were written by `record_bars_fixture.py --golden` while `callbacks.py` was byte-equal to `e78774f` (checked with `git diff --quiet e78774f`); live and replay bodies matched at capture.

## Step 4 — smoke (2026-10-24)

**Byte identity, live DB.** `origin/main` (`e78774f`, a detached worktree) vs the branch, Flask test client, 2026-10-23 / 2026-09-21 / 2026-10-16 / 2024-04-02 × center/multi(±1) × 1min/5min = 16 requests: **all 16 sha256-identical** with no `levels` parameter, and all 16 identical again with `levels=current`. Unit test: absent / `current` / `CURRENT` / `foo` / empty all equal the golden in both phases.

**Corrected mode, live DB** (gex_min_abs_b 10, 1min):

| D | center lines Current → Corrected | top-3 Current | top-3 Corrected | per-expiry sum mismatches (both modes) | notes |
|---|---|---|---|---|---|
| 2026-10-23 | 60 → 60 | 7660, 7760, 7651 | 7681, 7781, 7687 | 0 | [] |
| 2026-09-21 (first session after the Sept roll; A 09-18 is the Z26 patch) | 63 → 72 | 7612, 7701, 7712 | 7918, 7660, 7762 | 0 | [] |
| 2026-10-16 (third Friday) | 69 → 70 | 7751, 7801, 7809 | 7794, 7844, 7839 | 0 | [] |
| 2026-06-15 (roll-mismatch caveat A1.4) | 53 → 52 | 7494, 7401, 7501 | 7494, 7500, 7400 | 0 | [] — lines in the M26 frame over U26 bars, as accepted |
| 2024-04-02 | 52 → 0 | | | 0 | `pre_forward_era` |
| 2024-04-03 | 49 → 57 | 5200, 5139, 5207 | 5248, 5299, 5273 | 0 | [] |

**UI** (local Vite dev server + an auth-free smoke backend registering only the Ironbeam routes; screenshots 800 px, `specs/assets/cr-bt/`): the settings control (`settings-levels-control.jpg`); Current vs Corrected for 2026-10-23, 2026-09-21, 2026-10-16 (`<date>-current.jpg`, `<date>-corrected.jpg`); the Corrected popup for line 7687 on 2026-10-23 (`2026-10-23-corrected-popup-7687.jpg`): rows −272.16 + 0.00 − 0.58 − 0.02 + 1.10 = −271.66 B vs headline −271.67 B (display rounding). Applying Corrected stored `corrected` in localStorage and added `levels=corrected` to the center and multi requests. No console errors on the bars route. (The smoke backend did not allow origin `:5174` on the flow/skew/smile/ATM-IV routes, so those panels errored in this harness only.)

**Not verified here:** the deployed web service (no deploy done); the Dash-embedded copy of the React build (`/react-preview/`) — only the Vite dev build was exercised.
