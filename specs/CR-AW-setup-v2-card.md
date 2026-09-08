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
