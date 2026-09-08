# CR-AW — Setup v2 tab: max price, band evidence, KNN factors

> Authority: vault session note `Dash/sessions/2026-09-07 - CR-AW — Setup v2 Card.md`; design: `Dash/New Functionality/mockups/setup-v2-proposal-card.html` (two tabs: Trade, KNN factors)
> Branch: `feat/CR-AW-setup-v2-card` (off `origin/main` 04e13a2, the CR-AV merge)
> Scope: new Flask module `apps/web/modules/SetupV2/` (`/setup-v2`, `/api/setup-v2/card`), fair-value helper in `probability.py`, v2 frontend in `react_today_setup/`; `/today-setup` and its module untouched. **No deploy.**
> Mode: unattended through PR; halts at the first STOP gate that misses.

Drafted in chat 2026-09-07. **Authoritative mockup: `Dash/New Functionality/mockups/setup-v2-proposal-card.html`** (two tabs: Trade, KNN factors). It is the design to build; the earlier ledger/scorecard mockups are superseded and not in the vault. Ryan: **new tab beside the existing Today's Setup; old one stays until v2 is trusted.** Unattended through PR. Runs in parallel with CR-AU (worktree).

Terminology: analogues are **every corpus day inside the KNN similarity ceiling** (recency-weighted), not a fixed K. `_K_SAFETY=200` never binds. The card must never display "K="; it says "N analogues within the similarity ceiling".

## What the card must say, in order

1. **Max price to pay** for the proposed spread, from analogue closes. Quote vs max on a gauge; verdict "enter at the open" / "skip".
2. Fair value, expected P&L at the quote (with interval), market-implied (quote ÷ width).
3. **Distance band strip** with the reference cell numbers and today outlined.
4. **Analogue facts** with honest labels: reached the wall; finished above the short strike (full payout); finished above the long strike (any payout); spread worth at close.
5. **How to run it**: enter if under max (don't wait); hold to close; 15 DTE — each with its number. Untested watches (wall half-life, wall trend, vol state) in grey.
6. **Stamp**: reference `cr_id`, cell, n, re-run date, next run, fees included.
7. **Tab 2 — KNN factors**: every feature in the match, today's value, weight, percentile vs the magnet-above corpus, the analogues' middle-half band, a match-quality line.
8. Post-touch bars and the by-minute edge chart behind a detail toggle. No edge ratio anywhere on v2.

## Locked decisions

| # | Decision | Value |
|---|---|---|
| 1 | Route | New Flask module `apps/web/modules/SetupV2/` serving `/setup-v2` and `/api/setup-v2/card?date=`. New top-nav tab "Setup v2". `/today-setup` and its module untouched. |
| 2 | Fair value | For the walk-forward analogue set already returned by `compute_structural_probability(before_date=date)`: per analogue, `value = clamp(width − d, 0, width)` where `d = final_close_distance_from_target` (points below target; Step 0 confirms sign against the 11% close-above rate). `fair = mean(value)`. **Max price = 20th percentile of the bootstrap distribution of the mean** (1,000 resamples, seed = trade_date as int). Expected P&L = fair − quote; interval = bootstrap 2.5/97.5 of mean minus quote. Only analogues with `outcome_status='computed'`. Fees per shared config subtracted from expected P&L. |
| 3 | Any-payout rate | share of analogues with `d < width`; full-payout = `d ≤ 0`. |
| 4 | Band table | 3 rows + all from `bt_edge_backtest_results` at the latest `cr_id` matching `REF-%`, else `CR-AR`; `structure_type='debit'`, `outcome_type='close'`, `post_touch_pattern IS NULL`. Show `mean_pnl`, `win_rate`, Wilson, n. Today's band from the feature vector's signed distance. |
| 5 | Verdict | "enter at the open" if live quote ≤ max; "skip — quote above max" otherwise; "no listed structure" if the pair snapper raises. Never "wait". |
| 6 | KNN tab | Payload lists every feature key used by the KNN with: today's value, weight (read from the KNN config, not hardcoded), percentile of today vs magnet-above corpus ≤ date (`bt_daily_features_active`), 25th–75th percentile of the same feature across today's analogues, `in_band` bool. Match-quality = count in_band / total. Features with null today (IV rank, VRP) show "not populated". |
| 7 | Untested rows | Vol state, magnet horizon, wall trend: rendered from whatever the payload has (today: implied move percentile; the others null until CR-AT's wall table exists) with a fixed "untested" status. No colour other than grey. |
| 8 | Tests | Fair value on a synthetic analogue set (all above short → fair = width; all far below → 0; mixed); max ≤ fair; verdict for quote above / below / no structure; band selection; percentile computation; bundle grep: no "edge ×", no "supported", no "low-confidence" strings in the v2 bundle. |
| 9 | Not in scope | Changing `/today-setup`; condor card; wall table (CR-AT); weights fitting. |

## Gates

| Gate | Expected | On miss |
|---|---|---|
| G0 — `main` contains CR-AV; sign of `final_close_distance_from_target` confirmed (mean over analogues consistent with ~10% close-above) | yes | STOP |
| G1 — tests per decision 8; suites pass | yes | STOP |
| G2 — `git diff --stat` confined to SetupV2 module, Proposals (payload helpers only), probability.py (fair-value helper only), react_today_setup, tests, spec | yes | STOP |
| G3 — local `/api/setup-v2/card?date=2026-09-03`: fair, max, expected P&L, band, any-payout %, stamp, KNN tab all populated; verdict computed against the stored 06:34 quote | yes | STOP |
| G4 — `/today-setup` payload for 2026-09-03 byte-identical before and after | yes | STOP |

## Kickoff prompt

```
CR-AW — Setup v2 card. Unattended through PR. Authority: vault note
"Dash/sessions/2026-09-07 - CR-AW — Setup v2 Card.md". Read it and open the
mockup "Dash/New Functionality/mockups/setup-v2-proposal-card.html" in a
browser or as source — it is the design. Halt only at STOP gates.
DO NOT deploy. Work in worktree .claude/worktrees/cr-aw.

Branch: git fetch; git checkout -b feat/CR-AW-setup-v2-card origin/main.
Commit 1 — spec freeze. Step 0 → Commit 2: G0 sign check (query + one
  sentence); list the KNN feature keys and their weights from config.
Commit 3 — fair-value helper in probability.py + tests (decision 2, 3).
Commit 4 — SetupV2 module: route, payload (decisions 4–7) + tests.
Commit 5 — frontend: new tab, two-pane card from the mockups, post-touch
  and edge chart behind a toggle, bundle grep test. Build.
Step 1 — G3, G4. Commit 6 with the 2026-09-03 payload (numbers) pasted.
Wrap — Commit 7. Vault: run log; Sessions MOC. Push, open PR
  "CR-AW — Setup v2 tab: max price, band evidence, KNN factors".
  DO NOT MERGE. Print PR URL and the 2026-09-03 card numbers
  (fair, max, expected P&L, any-payout %, band, match quality).
```

## Decisions

- **Max = 20th percentile of the bootstrap mean.** A margin that scales with the analogue sample's uncertainty rather than a fixed discount. Locked here so the number means the same thing every day.
- **Verdict never says "wait".** Waiting selects adverse minutes (CR-AH/AR). Under max → enter; over → skip.
- **New tab, old untouched.** v2 earns trust beside v1; v1 is retired by a separate one-line CR when Ryan says so.

## Links

- [[2026-09-07 - CR-AV — Proposals Card, Evidence Only]], [[2026-09-06 - CR-AR — Pair-Snapping with Width Cap]], [[edge-zone-thresholds-need-recalibration-for-real-implied]], [[position-aware-edge-and-ev-display]], [[2026-09-07 - CR-AU — Live Out-of-Sample Stream]]

## Step 0 findings (2026-09-07, read-only)

Worktree `.claude/worktrees/cr-aw`, branch `feat/CR-AW-setup-v2-card` off `origin/main` 04e13a2 (PR #52, the CR-AV merge). Interpreter `/Users/ryan/code/volatility-monorepo/apps/web/.venv/bin/python` (arm64) for the DB reads and route runs; Rosetta repo venv for the Python suites; worktree-local `npm ci` for vitest.

| Gate | Expected | Actual | Result |
|---|---|---|---|
| G0 — `main` contains CR-AV | yes | `origin/main` = 04e13a2 (merge of PR #52, CR-AV); branch cut from it | PASS |
| G0 — sign of `final_close_distance_from_target` confirmed | consistent with ~10 % close-above | **Sign confirmed: `close − drift_target`** (`packages/shared/outcomes.py:247`, CR-022 spec), positive = closed above the wall. The "~10 %" the note expected is **`close_rate` = `reached_close` = \|d\| ≤ 0.25 IM (closed *at* the wall)**, not the share above it. On the 2026-09-03 walk-forward set (v1 route inputs, 70 within the ceiling, 68 computed): mean d **+0.9 pt**, median +3.6, share d ≥ 0 **51.5 %**, share d > −10 **57.4 %**, close_rate 10.3 %, touch_rate 73.5 %. Corpus-wide magnet-above (377 computed): 61.5 % above, mean +67. | PASS on the sign; the ~10 % prior was a mislabel — spec amended (A1) |

### G0 query (the one sentence)

`SELECT horizon_sessions, count(*), avg((final_close_distance_from_target > 0)::int), avg(final_close_distance_from_target), avg(reached_close::int) FROM bt_daily_outcomes_active WHERE feature_version = 'v0.6.0-openiv' AND outcome_status = 'computed' AND regime_kind_at_classification = 'magnet-above' GROUP BY 1` → h=5: n 98, 40.8 % above, mean −19.4, close_rate 17.3 %; h=20: n 173, 64.7 % above, mean +31.5, close_rate 8.1 %; h=60: n 103, 77.7 % above, mean +215, close_rate 3.9 %. **`final_close_distance_from_target` is signed close-minus-wall at the end of each analogue's bucket horizon; the 10 % figure is the at-wall rate, so "finished above the short strike" is `d ≥ 0`, not `close_rate`.**

Fair value under the locked definition with the correct sign on the 09-03 set: `value = clamp(10 + d, 0, 10)` → **fair 5.35**; bootstrap of the mean (1,000 resamples, seed 20260903): p2.5 4.23 · **p20 4.84 (max price)** · p50 5.35 · p97.5 6.52. (With the sign reversed the mean would be 5.10 — the two are close on this set only by coincidence; the tails differ.)

### KNN feature keys and weights (from `get_knn_config()` — canonical `v3`: ceiling 5.0, z_diff_cap 3.0, half-life 18 months)

| Feature | Weight | | Feature | Weight |
|---|---|---|---|---|
| is_pin_day | 2.5 | | n_clusters_total | 0.5 |
| is_magnet_day | 2.5 | | n_pin | 0.25 |
| is_bounded_day | 2.5 | | n_target | 0.25 |
| is_untethered_day | 2.5 | | n_feature | 0.25 |
| is_amplification_day | 2.5 | | n_clusters_above_spot | 0.5 |
| magnet_direction_signed | 2.0 | | n_clusters_below_spot | 0.5 |
| cluster_1_max_gex | 1.0 | | top_cluster_fraction_of_total_max_gex | 0.75 |
| cluster_1_quality_ordinal | 1.5 | | dominance_0DTE | 0.75 |
| cluster_1_signed_distance_sigma | 3.0 | | dominance_1_7 | 0.75 |
| cluster_2_max_gex | 0.5 | | dominance_8_30 | 0.75 |
| cluster_2_quality_ordinal | 0.25 | | dominance_30plus | 0.75 |
| cluster_2_signed_distance_sigma | 2.0 | | n_neg_zones | 0.3 |
| cluster_3_max_gex | 0.25 | | nearest_neg_signed_distance_sigma | 0.5 |
| cluster_3_quality_ordinal | 0.25 | | total_neg_max_gex | 0.3 |
| cluster_3_signed_distance_sigma | 1.5 | | implied_move_1d | 2.0 |
| atm_iv_percentile · skew_percentile · smile_convexity · term_structure_slope · vol_risk_premium | 1.0 each — **null today and across the corpus** (never populated; skipped by the NULL-aware distance) | | | |

34 keys (`FEATURE_NAMES`); the five vol-surface keys are null on 2026-09-03 and show "not populated" (decision 6).

### Other facts Step 0 surfaced

1. **Horizon of the analogue close.** `final_close_distance_from_target` is measured at `horizon_sessions` = `bucket_sessions(dominant_bucket)` — 1 / 5 / 20 / 60 sessions — not at 15 DTE. The 09-03 set is 34 × 5-session, 28 × 20-session, 6 × 60-session analogues (all magnet-above). `session_close_t15` exists for 458 / 475 computed rows but the wall price per analogue is not stored, so a 15-session value needs a landscape join; not in this CR.
2. **Band input.** Decision 4 says "the feature vector's signed distance"; that is `cluster_1_signed_distance_sigma`, which is **0 on 2026-09-03** (no confluence clusters; the wall comes from `walls`). The reference cells were banded on the harness's `sigma = (drift_target − table_spot) / IM`, IM from the 06:33 open straddle (`_OPEN_STRADDLE_SQL`, `compute_implied_move(…, dte=1)`), `distance_band()` thresholds 1.5 / 2.0. 09-03 under that measure: table_spot 7670.175, IV 0.1104, IM 53.34, wall 7806.175 → **σ 2.55 → far**. (The v1 route's own IM, `_resolve_implied_move` = 24.27, would give 4.2 σ, also far. The mockup's "near · +1.00 IM" is not reproducible from any stored measure.)
3. **Stored quote.** `orats_options_minute` holds, for trade_date 2026-09-03, the 18-Sep chain only at the 07:00 PT bar (`snapshot_pt` 06:59:54 — the v1 pl-data entry minute; 33 strikes 7625–7785 C/P) and a full-day crawl for three 25-Sep strikes (7800–7820). **There is no 06:34 bar for the 7775/7785 18-Sep calls.** The card prices at 06:34 PT through `price_proposal_legs` (cache write-through, CR-AO stale rule); the G3 run fetches that minute for the two legs if the environment allows, otherwise the stored 07:00 bar is used and the stamp says which.
4. **Analogue set.** The v1 route builds today's feature vector from the bars open (7704.25) and `_resolve_implied_move` (24.27); the stored feature row / harness inputs (table_spot 7670.175, open-straddle IM 53.34) give a different vector and only 20 analogues. Decision 2 binds the card to the v1 set, so the card reuses the v1 route's input helpers (`_load_landscape`, `fetch_rth_open`, `_resolve_implied_move`, `_fetch_carry_rates`, `get_effective_regime`, `build_proposals_response`) — imported, not copied, so the set is the same object v1 shows.
5. **Legs and expiry.** The card takes the v1 debit proposal: long 7796.175 / short 7806.175 ES → SPX 7775 / 7785 via `compute_spx_strike` at 15 DTE; expiry = trade_date + `expiry_dte_target` calendar days (the pl-data convention) = 2026-09-18. Surfaced, out of scope: the reference harness snaps the **undiscounted** ES-forward target (≈ 7805 SPX) at `nth_business_day(trade_date, 15)` (2026-09-25), i.e. ~20 pts above the SPX-equivalent wall and a week later — the reference cells and the CR-AU capture price a slightly different structure from the one the live card proposes (recorded under Open questions).
6. **Fees.** `packages/shared/config.py` does not exist on `main` (CR-AU creates it in parallel with `FEE_PER_CONTRACT_PER_LEG = 0.65`). The card imports it with a fallback constant of 0.65 so it picks up the shared value once CR-AU lands without a file overlap. A vertical round trip = 2 legs × 2 sides × $0.65 = $2.60 = **0.026 pts**.
7. **Mounting.** A Flask module cannot serve without `apps/web/app.py` importing and calling its `register_*` function; the "Setup v2" tab also lives in the Dash `dcc.Tabs` in `app.py`. G2's file list is read to include those lines in `app.py` (mount + tab + redirect), nothing else there.
8. **G4 "before".** `/api/setup/proposals?date=2026-09-03&ticker=SPX` captured via the Flask test client on the branch base: 200, 2,211 bytes, regime magnet-above, k 70 / 68, touch 0.7353, close 0.1029, mean days 4.84, one proposal (debit 15 DTE, 7775 / 7785 SPX). Bytes kept in the session scratchpad for the after-comparison.
9. **Reference cells.** `bt_edge_backtest_results` has no `REF-%` rows yet; latest is `cr_id = 'CR-AR'` (run `b07972a5`, 2026-09-07). Debit · close · pattern NULL · train, threshold 0.05: near +1.91 / 68.4 % [52.5, 80.9] n 38; mid +1.23 / 55.9 % [39.5, 71.1] n 34; far +1.99 / 48.4 % [32.0, 65.2] n 31; all +1.71 / 58.3 % n 103. Column names: `distance_band`, `mean_pnl`, `win_rate`, `wilson_lo`, `wilson_hi`, `n_settled` (n), `threshold`, `partition`.

## Spec amendments (Step 0, before any implementation code)

- **A1 — sign and "full payout".** `d = final_close_distance_from_target` is close − wall (positive = above). `value = clamp(width + d, 0, width)`; full payout = `d ≥ 0`; any payout = `d > −width`. The card's "finished above the short strike" line is the `d ≥ 0` share (51.5 % on 09-03), **not** `close_rate`; `close_rate` is shown as "closed at the wall (± 0.25 IM)" and the mockup's 10 % row is corrected accordingly.
- **A2 — horizon disclosure.** Fair value stays on `final_close_distance_from_target` (locked). The fair-value box states the horizon mix ("at each analogue's outcome horizon: 34 × 5, 28 × 20, 6 × 60 sessions") so the number is not read as a 15-DTE close. A 15-session variant is an open question, not built here.
- **A3 — band measure.** Today's band uses the harness's sigma (wall − `table_spot`) / open-straddle IM, the same measure the reference cells were banded on; the payload carries `sigma`, `im_open_straddle`, `table_spot` so the card can show it. The feature vector's `cluster_1_signed_distance_sigma` is not used (0 on cluster-less days).
- **A4 — quote minute.** Card quote = `price_proposal_legs` at **06:34 PT** (cache write-through, CR-AO stale rule); payload records `quote_minute`, `quote_valid`, `stale_quote`. "Clean" = valid at the minute and not stale.
- **A5 — mount.** `apps/web/app.py` gains the SetupV2 mount (import + `register_setup_v2_routes(server)`), the "Setup v2" tab and its redirect — the only lines touched there.
- **A6 — fees.** `FEE_PER_CONTRACT_PER_LEG` imported from `packages.shared.config` with a 0.65 fallback; `fee_pts = legs × 2 × fee / 100` subtracted from expected P&L; the stamp says "fees $0.65 / contract / leg included".
- **A7 — frontend entry.** The v2 page is a second Vite entry (`react_today_setup/setup-v2.html` → `dist/setup-v2.html`) so the v2 bundle can be grepped on its own; assets stay under `/today-setup/assets/` (already served). Both pages' top nav gains the "Setup v2" pill.

## Step 1 — gates and the 2026-09-03 card (2026-09-07)

| Gate | Expected | Actual | Result |
|---|---|---|---|
| G1 — tests per decision 8; suites pass | pass | fair value: all above short → width, all far below → 0, mixed, max ≤ fair, seed reproducible, empty → None (6 tests); verdict above / below / equal / no structure / no quote / never "wait" (6); band selection (REF- over CR-AR, filters, train + lowest threshold, thresholds 1.5 / 2.0) (6); percentile / quantile (2); KNN rows incl. "not populated" and match quality (4); fees / P&L (3); by-minute strip (3); route wiring (9); frontend: verdict enter / skip / no structure, "70 analogues within the similarity ceiling" and no "K=", band outline + horizon mix + at-wall rate, stamp, untested watches, detail toggle, KNN tab rows + tab switch (10), v2 bundle grep (1). Python: **951 passed, 1 skipped** across TodaySetup / Proposals / SetupV2 / packages.shared / options_cache / backtest; vitest **47 passed** (6 files) after `npm run build`. | PASS |
| G2 — diff confined | yes | `git diff --stat origin/main..HEAD`: `apps/web/app.py` (+8: mount, tab, redirect — A5), `apps/web/modules/SetupV2/**`, `packages/shared/probability.py` (+89: `analogue_fair_value` + `_quantile` + one docstring line), `packages/shared/tests/test_probability.py`, `react_today_setup/**` (v2 entry, components, tests, one nav pill in `App.jsx`, `vite.config.js` input), `specs/CR-AW-setup-v2-card.md`. Nothing in `apps/web/modules/TodaySetup/`, `Proposals/` or `packages/web-shared/`. | PASS |
| G3 — local `/api/setup-v2/card?date=2026-09-03` populated; verdict against the stored 06:34 quote | yes | 200 in 34 s, 17,094 bytes; every block populated (below). The 06:34 bar for the two legs was not in the store (Step 0 fact 3); `price_proposal_legs` fetched it through the cache (clean, not stale) — it is stored now, and the by-minute strip shows 06:34 and the earlier 07:00 bar. | PASS |
| G4 — `/api/setup/proposals?date=2026-09-03` byte-identical before / after | yes | 2,211 bytes both; `cmp` equal; md5 `b415e6d5…` both | PASS |

### The 2026-09-03 card (Flask test client on the branch, real DB)

| Field | Value |
|---|---|
| Structure | debit call spread, buy **7775** / sell **7785** SPX (ES 7796.175 / 7806.175), expiry **2026-09-18** (15 calendar DTE), width 10, listed, no warnings |
| Quote at 06:34 PT | **4.10** (7775 mid 42.05, 7785 mid 37.95; both legs valid, not stale); market-implied 41 % |
| Analogues | 70 within the similarity ceiling (5.0), **68** computed, 2023-05-02 … 2026-08-28; horizon mix 34 × 5, 28 × 20, 6 × 60 sessions |
| Fair value | **5.35** (mean of clamp(10 + d)); bootstrap of the mean p2.5 **4.16** · p20 **4.85** · p97.5 **6.47** (1,000 resamples, seed 20260903) |
| Max price | **4.85** |
| Expected P&L at 4.10 | **+1.23** [+0.03, +2.34] after fees (0.026 pt) |
| Verdict | **enter** — "quote is under the max — enter at the open" |
| Payout rates | full (d ≥ 0) **51.5 %**; any (d > −10) **57.4 %**; closed at the wall (± 0.25 IM) 10.3 %; reached the wall 73.5 % [62, 83], mean 4.84 sessions |
| Band | wall 7806.175, table_spot 7670.175, open-straddle IV 0.1104 → IM 53.34 → σ **2.55 → far** |
| Reference cells (CR-AR, run b07972a5, debit · close · train · threshold 0.05) | near +1.91 / 68.4 % [52.5, 80.9] n 38 · mid +1.23 / 55.9 % [39.5, 71.1] n 34 · **far +1.99 / 48.4 % [32.0, 65.2] n 31** (baseline +1.96, beat +0.03) |
| KNN tab | v3 (ceiling 5.0, cap 3.0, half-life 18 m); corpus 394 magnet-above days before 09-03; **match quality 29 of 30** populated factors inside the analogue band; outlier `implied_move_1d` (today 24.3 pt = 7.9th percentile; analogue middle half 33.4–52.9); the five vol-surface factors "not populated" |
| Stamp | CR-AR · far · debit · hold to close · n 31 · re-run 2026-09-07 · next 2026-10-01 · fees $0.65 / contract / leg included · quote 06:34 PT |

Browser check (scratch Flask server on 8061 serving the built entry): Trade and KNN tabs render, verdict / gauge / band strip / facts / stamp populated, no console errors.

Surfaced by the KNN tab: today's `implied_move_1d` in the live route is `_resolve_implied_move` (latest `orats_monies_minute` snapshot, nearest dte → 24.3 pt on 09-03) while the stored corpus vectors carry the 06:33 open-straddle move (53.3 pt on 09-03). The KNN therefore matches today's IM against a differently-measured corpus IM; the tab shows it as the one outlier. Pre-existing in v1; recorded under Open questions.

## What changed

- **`packages/shared/probability.py`** — `analogue_fair_value(close_distances, width, *, n_boot, seed, max_pct)`: per analogue `value = clamp(width + d, 0, width)` with `d = final_close_distance_from_target` (close − wall); `fair` = mean; `max_price` = 20th percentile of 1,000 bootstrap means (never above `fair`); `boot_lo / boot_hi` = 2.5 / 97.5; `full_payout_rate` (d ≥ 0), `any_payout_rate` (d > −width). Seed = trade date as YYYYMMDD. Pure; 6 tests.
- **`apps/web/modules/SetupV2/`** (new) — `service.py`: reference-cell selection (latest `REF-%` else `CR-AR`; debit · close · pooled; train + lowest threshold per band), harness band thresholds, `percentile_rank` (kind = mean) / `quantile`, `knn_factor_rows` (34 keys with label / group / low–high words, today, weight from `get_knn_config()`, percentile vs the magnet-above corpus before the date, analogue 25th–75th band in value and percentile space, `in_band`, `populated`), `match_quality`, `expected_pnl` (fair − quote − fees), `verdict` (enter / skip / no_structure / no_quote — never "wait"), `horizon_mix`, `next_reference_run`, `fee_points`, `net_debit_by_minute`; fee constant imported from `packages.shared.config` with a 0.65 fallback. `routes.py`: `GET /setup-v2` (serves `dist/setup-v2.html`), `GET /api/setup-v2/card?date=&ticker=`; `build_card` reuses the v1 input helpers (landscape, bars-open spot, `_resolve_implied_move`, carry rates, effective regime, `build_proposals_response`) so the analogue set is v1's; prices the debit legs at **06:34 PT** through `price_proposal_legs` (cache write-through, CR-AO stale rule); band σ from the harness measure (wall − `table_spot`) / open-straddle IM; cache-only by-minute quote strip over 06:30–07:00; 37 tests.
- **`apps/web/app.py`** — mount (`register_setup_v2_routes`), `TAB_SETUP_V2` "Setup v2" tab and its redirect to `/setup-v2`. Nothing else.
- **`react_today_setup/`** — second Vite entry `setup-v2.html` → `src/setupv2/` (`SetupV2App` with the shared top nav + date picker, `TradeTab` = top / mid / manage / stamp / detail toggle, `DetailPanel` = post-touch bars + quote-by-minute SVG, `KnnFactorsTab`, `format.js`, `setupv2.css` from the mockup). "Setup v2" pill added to the v1 nav. Tests: `SetupV2Card.test.jsx` (10) and `bundleV2Strings.test.js` (greps every chunk `dist/setup-v2.html` references: no `edge ×` / `edge_ratio` / badge phrases / `low-confidence` anywhere; no bare `supported` or string-literal `K=` in the app chunk).
- **Not embedded:** web-shared's `ProposalEdgeChart` (its legend and tooltip print `edge ×…`, which decision 8 and "no edge ratio anywhere on v2" forbid, and `packages/web-shared/` is outside G2). The detail toggle carries the post-touch bars and the spread's quote by minute against the max line instead (A8).

## Decisions

- **Decision:** proceed past G0's "~10 % close-above" expectation with an amendment rather than halt.
  **Rationale:** the gate exists to pin the sign; the sign is unambiguous in code (`close − drift_target`) and the 10 % figure in the note is `close_rate`, the at-wall rate, mislabelled. CLAUDE.md's rule for a wrong upstream assumption surfaced in Step 0 is to amend the spec and the note before implementation, which is what A1 does. Recorded here so the reviewer can veto at the PR.
- **Decision:** fair value on the locked column at each analogue's own outcome horizon, disclosed on the card.
  **Rationale:** decision 2 is locked; the horizon mix (5 / 20 / 60 sessions by dominant bucket) is stated in the fair-value box so the number is not read as a 15-DTE close. A single-horizon variant needs the analogue wall price (a landscape join) — open question.
- **Decision:** band from the harness's σ, not the feature vector's.
  **Rationale:** the reference cells were banded on (wall − `table_spot`) / open-straddle IM; `cluster_1_signed_distance_sigma` is 0 on cluster-less days such as 2026-09-03. Same measure, same band.
- **Decision:** the v2 page is its own Vite entry and shares nothing with the v1 page beyond React.
  **Rationale:** decision 8's bundle grep needs a v2 bundle to grep; v1 components carry `K=` and the edge-ratio strings.
- **Decision:** quote at 06:34 PT with cache write-through, not the stored-only read.
  **Rationale:** the store had no 06:34 bar for the 09-03 legs; the daily capture (CR-AU) will fill it for live days, and a miss should show a quote with a fetch rather than "no quote" on a day the pipeline hasn't captured.

## Open questions

- **15-session fair value.** `session_close_t15` exists for 458 / 475 computed rows; a per-analogue wall price (from `orats_gex_landscape.walls` via `pick_drift_target`) would give a value at a fixed 15-session horizon for comparison with the mixed-horizon number the card shows.
- **Implied-move measure mismatch (surfaced by the KNN tab).** The live route's `implied_move_1d` (24.3 pt on 09-03) is measured differently from the corpus's (53.3 pt, 06:33 open straddle); the KNN sees today as the 8th percentile and outside the analogue band on that feature alone. Which measure should the live vector use? (Also affects v1 — not changed here.)
- **Reference legs vs card legs.** The harness snaps the undiscounted ES-forward target (≈ 7805 SPX at `nth_business_day(15)` = 09-25) while the card prices 7775 / 7785 at date + 15 calendar days (09-18). The band cells the card cites were built on the former.
- **Mockup's "near · +1.00 IM"** is not reproducible from any stored measure (harness σ 2.55, v1 σ 4.2); the card shows far.
- **Card latency** ~30 s (v1's queries + two KNN ranks + corpus load + quote fetch). Acceptable for a tab; a shared rank between `compute_structural_probability` and the rows call would halve the KNN part.

## Spec amendments v2 (Ryan, 2026-09-07, before merge) — A2 / A3 / A4 restated

- **A2 — fair value at T+15 sessions.** For each computed analogue: `target = close_at_horizon − final_close_distance_from_target` (the wall re-derived from the stored outcome; `close_at_horizon` = ES RTH close on `horizon_end_date`, last 06:30–13:00 PT bar of `ironbeam_es_1m_bars` — checked against `pick_drift_target(walls)` on all 377 magnet-above rows: equal to the point); `d15 = target − session_close_t15` (points below the wall at T+15); `value = clamp(width − d15, 0, width)`. Analogues with NULL `session_close_t15` (or no horizon close) are excluded and counted (`n_no_t15_close`, `n_no_horizon_close`). Fair, max, expected P&L, full-payout (`d15 ≤ 0`) and any-payout (`d15 < width`) rates all move to this basis; `analogue_fair_value` takes points-below-target. The earlier A1 (sign) and A2 (horizon disclosure) are superseded by this.
- **A3 — expiry.** `expiry = nth_business_day(trade_date, 15)` (weekdays, NYSE holidays skipped — the harness's `_NYSE_HOLIDAYS`, copied into `SetupV2/service.py`) snapped to the nearest listed expiry in the prior-close chain (`orats_oi_gamma`; ties → later). 2026-09-03 → 2026-09-25 (22 calendar days). Card label "15 sessions (25 Sep)". The earlier fact 5 / date-plus-15-calendar-days convention is superseded.
- **A4 — the harness distance σ, verbatim from the code.** `scripts/cr_ah_step4_analysis.py::load_signal_entries`:
  ```
  spot = float(table_spot)                                       # orats_gex_landscape.table_spot
  floor_ts = datetime.combine(trade_date, time(6, 33, 0))
  iv_row = conn.execute(_OPEN_STRADDLE_SQL, (trade_date.isoformat(), TICKER, floor_ts)).fetchone()
  #   _OPEN_STRADDLE_SQL: first orats_monies_minute row with atmiv IS NOT NULL AND dte > 0
  #   AND snapshot_pt >= floor_ts, ORDER BY snapshot_pt ASC, dte ASC
  implied_move = compute_implied_move(spot, float(iv_row[0]), dte=1.0)   # spot × iv × sqrt(1 / 252)
  payload = _materialize_payload(landscape_rows, spot, implied_move)
  dt = (payload.get("regime") or {}).get("drift_target")
  sigma = (float(dt) - spot) / implied_move
  band = distance_band(sigma)     # near: σ < 1.5 · mid: 1.5 ≤ σ < 2.0 · far: σ ≥ 2.0  (backtest/models.py)
  ```
  The card's band strip is headed "σ (harness)" and shows the formula with today's inputs and the point distance beside it: `σ (harness) = (7806 − 7670) / 53.3 IM = 2.55 · 136 pt above the open`. The payload carries `sigma`, `distance_pts`, `table_spot`, `im_open_straddle`, `atmiv_open`, `sigma_formula`.

## Step 1b — G3 re-run after A2 / A3 / A4 (2026-09-07)

Commits: A2 20fd531 · A3 b222fc0 · A4 cc2dfce. G1: Python 960 passed / 1 skipped; vitest 47 passed after `npm run build`. G4 re-checked: `/api/setup/proposals?date=2026-09-03` still byte-identical to the pre-CR capture.

| Field | Before (calendar expiry, mixed-horizon close) | **After (A2 / A3)** |
|---|---|---|
| Structure | buy 7775 / sell 7785 SPX, 18 Sep (15 cal. DTE) | **buy 7770 / sell 7780 SPX, 25 Sep** — `nth_business_day(09-03, 15)` = 09-25 (Labor Day skipped), listed; 22 calendar days; ES 7796.175 / 7806.175 discount to 7770 / 7780 at that expiry |
| Quote 06:34 PT | 4.10 | **4.80** (7770 mid 62.90, 7780 mid 58.10; clean) · market-implied 48 % |
| Analogues valued | 68 (each at its bucket horizon) | **65 at T+15** (68 computed; 3 without a T+15 close excluded; 0 without a horizon close) |
| Fair value | 5.35 | **5.21** |
| Max price (bootstrap p20) | 4.85 (p2.5 4.16 · p97.5 6.52) | **4.67** (p2.5 4.00 · p97.5 6.36) |
| Expected P&L at the quote | +1.23 [+0.03, +2.34] | **+0.38 [−0.82, +1.53]** after fees (0.026) |
| Verdict | enter | **skip — quote above max** (4.80 > 4.67) |
| Full payout (d15 ≤ 0) | 51.5 % | **50.8 %** |
| Any payout (d15 < 10) | 57.4 % | **53.8 %** |
| Band | far, σ 2.55 | **far — σ (harness) = (7806 − 7670) / 53.3 IM = 2.55 · 136 pt above the open**; CR-AR far cell +1.99 / 48 % [32, 65] n 31 |
| KNN | 29 / 30 | 29 / 30 (unchanged; outlier `implied_move_1d`) |

The verdict flips to skip on this date: the 25-Sep quote carries a week more time value (4.80 vs 4.10) while the T+15 fair value is lower than the mixed-horizon one (5.21 vs 5.35), so the p20 max (4.67) sits under the quote.
