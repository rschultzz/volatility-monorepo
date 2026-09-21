# CR-BI — SPX-cash + NYSE-calendar outcomes become the canonical compute path

> Branch: `feat/CR-BI-spx-cash-canonical-outcomes` (off `origin/main` 9904afd)
> `data_safety_class: write_backfill`
> Source: ADR `2026-09-20 - Outcomes Evaluated on SPX Cash with NYSE-Calendar Sessions` + CR-BH spec (`specs/CR-BH-spx-cash-shadow-outcomes.md`) + kickoff prompt (2026-09-20).
> INSERT-only backfill. `v0.6.0-openiv` and `v0.6.0-openiv-spxcash` stay in the table as the record.
> **STOP after Step 0 and report before implementation.**

## Decision being implemented (ADR)

1. **Outcome prices are SPX cash**, built from 0DTE rows in `orats_monies_minute` with the CR-BH hybrid rules: `stock_price` from 2023-11-09; before that `spot_price` shifted −15 min on the lagged days with the close = 5-minute `stock_price` median; frozen-run + volatility-scaled spike filter; ES only as a per-day QA flag, never a price.
2. **Sessions come from `packages/shared/trading_calendar.py`** (NYSE), not from bar presence. One PT session-window definition (DST via zoneinfo); `compute_outcome`, containment, the t1/t5/t15 session columns and post-touch positions all read it. Retires the `cr_g` / `cr_i` fixed-UTC window.
3. **Promotion is through the normal compute path**, not by blessing the CR-BH snapshot: a new canonical `feature_version` computed by the same code the crons and the CR-AA sweep run.
4. **ES is demoted** to intraday execution context and flow. Remaining ES-vs-level comparisons must move to SPX cash or carry an explicit same-minute basis — inventoried here in Step 0; fixing them is *not* in this CR unless the spec is amended.

## Step-0 gate (read-only; all answered in this file before any code change)

1. **ES-vs-level inventory.** Every place an ES price is compared to an SPX-space level in the edge harness, the Setup v2 card path, `edge_zones.py`, `probability.py` and TodaySetup: (a) distance-band assignment, (b) touch-exit trigger, (c) entry/edge gate, (d) baselines, (e) card spot. File, function, and what it is compared against.
2. Does CR-AR close-path P&L (entry debit, settlement value) use anything other than SPX option quotes? Yes/no with the code path.
3. Cite CR-BH Step-0 item 5 (minute-sampling understatement of SPX high/low, median pts); compute it if it was not.
4. Quantify the `cr_g` / `cr_i` window error: for PST-month rows, distribution of (true 13:00 PT close − stored `session_close_t15`). Trace TodaySetup.
5. Version plan: new canonical `feature_version` name; does `bt_daily_features` need a matching bump for any join; which constant (`canonical_version.py`) promotes it; which readers change behaviour on promotion.

## Implementation (after go)

- One shared SPX-cash session builder (CR-BH rules) and one PT session-window definition with sessions from `packages/shared/trading_calendar.py`. `compute_outcome`, containment, t1/t5/t15 session columns and post-touch positions all read it. Retire the fixed-UTC windows in `cr_g` / `cr_i`.
- `cr_b_backfill_outcomes` and the CR-AA sweep compute the new version.
- Backfill all dates into the new version under the unattended backfill protocol, INSERT-only.
- Smoke: the new canonical must match the CR-BH shadow on touch/close for the 231 same-calendar rows; explain every difference on the other 248.
- Promote the canonical constant as its own commit, last.
- Out of scope: re-running CR-AR / locked findings (next CR), `contracts.py`, the Saturday 2026-02-21 bars, the `spot_price` drift.
- Crons need a redeploy after merge: list which (Ryan redeploys from chat).

## Implementation order

1. Step 0 — findings appended to this file (gate). **STOP, report.**
2. Step 1 — shared SPX-cash session builder + PT session window + trading-calendar sessions in `packages/shared`, with tests.
3. Step 2 — `outcomes_runner` / `cr_b_backfill_outcomes` / CR-AA sweep compute the new version (incl. t1/t5/t15 + post-touch); `cr_g` / `cr_i` fixed-UTC windows retired.
4. Step 3 — backfill all dates into the new version (INSERT-only, backfill protocol); smoke vs the CR-BH shadow.
5. Step 4 — promote the canonical constant (own commit, last).
6. Smoke + wrap; cron redeploy list.

## Step 0 — findings (2026-09-20, read-only; no code or data changed)

Method: two read-only code audits (inventory; CR-AR P&L path) with the load-bearing lines re-read by hand, `scripts/cr_bi_step0_window_error.py` for Q4, catalog queries + Render service list for Q5. **Gate result: all five answered. Q2 contradicts the ADR (Consequences 3) and Q1 is wider than the ADR anticipated — see "What Step 0 changes" at the end. STOP here.**

### Q1 — ES-vs-level inventory

Three price frames are in play, not two: **A** SPX cash (`table_spot`, `orats_monies_minute`, listed strikes); **B** the landscape axis — `orats_oi_gamma.discounted_level = strike × exp((r−q)(dte+1)/252)` → `walls[].price` → `drift_target` → proposal leg strikes (`strike_es` in code); **C** ES (`ironbeam_es_1m_bars` and everything derived: `session_*_t0/t1/t5/t15`, `fetch_rth_open`). The E6 docstring (`outcomes.py:104–121`) treats B ≡ C. `packages/shared/forward_math.compute_spx_strike` converts B → A; nothing in the repo converts C. B − A is the *option's* carry (≈ +12 pts at 15 DTE, less for the short-dated walls that dominate) — not quantified here; flagged below.

| # | cat. | File · function · lines | ES-derived quantity | Compared against | Effect of a +30-pt basis |
| --- | --- | --- | --- | --- | --- |
| b3 | (b) | `packages/shared/outcomes.py` `compute_outcome` 216–252 | ES daily RTH high / low / close (`cr_b._RTH_BARS_SQL`, sweep) | `drift_target` (B): touch, `reached_close` (± 0.25 × IM), `final_close_distance`, pin band | **The labels. Fixed by this CR.** CR-BH: touch 83.9 → 73.3 % |
| b6 | (b) | `outcomes_runner.py` `nearest_walls` / `compute_session_containment` 158–228 | ES session open / high / low / close | nearest walls (B) | `close_pos_in_band` inflated, `breach_side` skewed to `above`. **Fixed by this CR** |
| b5 | (b) | `probability.py` `classify_post_touch_positions` 59–124 ← `cr_i` 105–120, 223–231 | ES session close at `days_to_reach + N` | `drift_target ± 0.25 × IM` | basis > whole tolerance band → labels pushed to `+1`. **Fixed by this CR** (plus the UTC window, Q4) |
| c2 | (c) | `apps/web/modules/SetupV2/routes.py` `t15_distances_below_target` 179–211 → `analogue_fair_value` → `service.verdict` 331–348 | `session_close_t15` (ES); also `_fetch_rth_closes` 154–176 (ES close at horizon end) | `target = ES close − final_close_distance` = `drift_target` (B) | **Live enter/skip gate.** `d15` 30 pts too small → `fair` / `max_price` inflated → verdict too permissive. **Moves to SPX on promotion only if `_fetch_rth_closes` is also switched** — otherwise `target` becomes (ES close − SPX distance) and breaks. Must be handled in this CR |
| e1 | (e) | `Proposals/routes.py` `_fetch_anchor_data` 107–161 | card `current_spot` = `session_open_t0` (ES) | used at 422–593: leg IV calibration, P&L grid bounds vs `drift_target`, `build_real_strike_band(spot, …)` (463–468 — its docstring, `options_cache/pricing.py:541`, says *"spot is the SPX cash price"*: an ES price is passed), `compute_implied_pdf`, `compute_terminal_prob_in_range`, `compute_edge_zones`, greeks, `compute_todays_edge` | B-L strike band centred ~30 pts high; structural-vs-implied edge subtracts an SPX-space market probability from an ES-space projection. **Becomes SPX automatically on promotion** (`session_open_t0` = 06:33 SPX print) — a behaviour change to call out, in the ADR's direction |
| e2 | (f) | `structural_distribution.py` 60–73, 118–129, 185–196; `edge_today.py` 72–77, 144–169; `edge_zones.py` 139–140, 282–304 | analogue path `(session_close_tN − session_open_t0)/IM` — **ES-vs-ES, fine** — projected onto `today_spot` | bounds / `strike_es` = `drift_target` (B); implied PDF on SPX strikes | frame of the projection = frame of `today_spot`. ES today → one-sided inflation of `struct_close` for magnet-above. After promotion both analogue OHLC and `session_open_t0` are SPX → consistent, **provided the caller's spot is SPX too** |
| a3 / c3 | (a)(c) | `TodaySetup/routes.py` 249–272, 285; `SetupV2/routes.py` 325–347; `Bars/service.fetch_rth_open` 34–42 | live `spot` = first ES RTH bar open | `_materialize_payload` → `classify_regime` (every `dom_price ≷ spot`, `near_dist_pts` gate), `extract_features` σ-distances, `strike_anchors` call/put choice, `wall.above_spot` (SetupV2 586) | **Live query vector and regime are built in ES space; the stored corpus (`day_features.py:450, 525`) uses `table_spot` (SPX).** σ-distance features shifted ≈ −0.6σ vs their own corpus; walls within 30 pts of the open land on the wrong side; live and stored regime can disagree. **Not touched by promotion** — reads bars directly |
| b1 / b2 | (b) | `scripts/cr_ah_step4_analysis.py` `detect_touch` 593–653; `backtest/plugins/{debit_vertical,vertical}.is_touch` | ES 1m close (live query, not stored labels) | `close >= drift_target` (B) | harness touch-exit fires early / on days cash never touched. Exit *price* is SPX quotes. **Out of scope (next CR)** |
| d1 / d2 | (d) | harness `get_settlement_price` 703–719, `compute_pnl` 936–951; `debit_vertical.payoff` 41–57, `close_zone` 63–82 | **ES close 12:50–13:00 PT on expiry** as the settlement underlying | SPX listed strikes (A) | see Q2. Baseline shares it. Reaches the v2 card via `bt_edge_backtest_results` (`SetupV2/routes._fetch_reference_rows` 214–239). **Out of scope (next CR)** |
| c1 | (c) | harness `build_entry_scan` 558, 768–769 | none directly — `structural_prob` = KNN `touch_rate` from the ES labels (b3) | `edge = structural_prob − \|net\|/width ≥ T` | gate input inflated; changes which minute fills. Changes on promotion; re-run is the next CR |
| f3 | (f) | `scripts/cron_daily_leg_capture.py` 176–181, 256–259 | `session_open_t0` as `es_open` | `spx_open` (06:33 print): `basis_open`, skip if outside −40..100 | sanity guard only. **After promotion `es_open` is SPX → `basis_open` ≈ 0 always: the guard silently stops guarding and the logged field is meaningless.** Needs a one-line change (read ES from bars, or drop) — live cron |
| f1 | (f) | `scripts/backfill_es_minute_features_gex_walls.py` 71–125 → `packages/backtests/gex_fade.py` 70–77, 485–493 | ES minute close | `discounted_level` walls (B); 2-pt proximity gate | short/long GEX-wall backtests — same class, outside the named scope; not in this CR |

Clean (examined, SPX-vs-SPX or ES-vs-ES): distance band in the harness (`cr_ah_step4` 383–413) and on the v2 card (`SetupV2/routes` 474–483) — both `table_spot`; `direction_sanity` (`table_spot`); persisted corpus features; entry debit and touch-exit *prices* (`orats_options_minute`); `price_proposal_legs` / `fetch_horizon_delta` (convert B → A); condor strikes (06:33 SPX print by design); `Analogues._fetch_session_outcomes` (ES returns / ranges); `max_excursion`, `actual_realized_em_pct`, `range_over_im` (ES-vs-ES). `_resolve_implied_move` with an ES spot overstates IM by ≈ 0.2 pt — negligible.

Also surfaced, not quantified: the harness and the capture scripts snap the frame-B `drift_target` straight onto the frame-A listed chain (`snap_vertical_pair`, `cr_ah_step4` 460, `cron_daily_leg_capture` 148) while the live Proposals path converts with `compute_spx_strike` — so harness cells and the card do not describe exactly the same strikes. And the ADR's "targets and prices share one space" is A-vs-B after this CR: true to within the walls' own carry, which is small for short-dated walls but not zero.

### Q2 — CR-AR close-path P&L uses only SPX option quotes? **No.**

- Entry debit: SPX option mids only (`build_entry_scan` 504–570, `orats_options_minute`). Strikes: `drift_target` + `orats_oi_gamma` listed chain, `table_spot` tie-break — no ES. Date universe: `bt_daily_features` regime + quote availability; no `bt_daily_outcomes` filter at the CR-AR commit (`f22a69e`). Distance band: SPX.
- **Settlement: `get_settlement_price` (703–719) = last `ironbeam_es_1m_bars.close` in 12:50–13:00 PT on expiry day; `close_pnl = fill_net_credit + plugin.payoff(legs, settlement_price)` (936–942) = intrinsic value of SPX-strike legs at an *ES* price.** No option quote is read at settlement. The specs already say so (`CR-AN` :66, :347; `CR-AP` :129) — the ADR's "measured from SPX option quotes at entry and settlement" is wrong.
- The baseline settles on the same ES price (949–951); it cancels per trade but the two means use different denominators (103 vs 105).
- ES-derived touch labels enter through the edge gate (`structural_prob` = KNN `touch_rate`), which picks the fill minute and, at T = 0.05, which trades fill (104 → 103).
- Direction of the error: a debit *call* spread valued at an underlying ≈ 15–49 pts too high → pushed toward max profit. **The +1.71 mean / 58.3 % win is inflated by an amount not quantified here** (15-DTE expiries: the basis on the *expiry* date is what matters). The monthly `reference-rerun` cron wraps the same harness.
- Touch-exit path (not in the n = 103 cell): detection is ES-vs-`drift_target`; exit price is SPX quotes.

### Q3 — minute-sampling understatement (cited from CR-BH Step 0 item 5; computed there)

`specs/CR-BH-spx-cash-shadow-outcomes.md` Step 0 Q5 — ES as proxy, 50 random outcome dates (seed 20260920), 06:30–13:00 PT, range from 1m closes only vs true 1m high/low: **high understated median 0.75 pts** (mean 1.29, p90 1.85, max 11.5); **low understated median 1.38** (mean 1.82, p90 3.85, max 5.0); range median 2.25 (p90 5.8, max 13.75). Small against a 28-pt mean basis; biases touch slightly *down* (opposite direction to the basis bias). This is the number for ADR Consequences item 6.

### Q4 — `cr_g` / `cr_i` fixed-UTC window error (`scripts/cr_bi_step0_window_error.py`, read-only)

Mechanism proven: replicating `cr_g`'s pick (bars `13:30 ≤ t < 20:00 UTC`, grouped by UTC date, Nth bar-present "session") reproduces the stored `session_close_t1/t5/t15` on **767 / 764 / 756 of 767 / 764 / 756** non-NULL rows (0 differ), and `cr_i`'s labels on every row. On PDT sessions stored = the 12:59 PT bar close exactly (max |diff| 0.00).

**PST-session rows, true close − stored `session_close_t15`** (same session date, ES, PT-aware):

| true close definition | n | mean | median | median \|x\| | p75 | p90 | p99 | max | > 5 | > 10 | > 25 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 13:00 PT bar close (canonical t0 convention) | 221 | −1.05 | −0.75 | **7.25** | 14.00 | 25.00 | 54.30 | 103.25 | 141 | 82 | 22 |
| 12:59 PT bar close (`cr_g`'s own PDT convention) | 221 | −1.17 | −0.50 | 6.75 | 14.50 | 26.00 | 58.85 | 104.00 | 141 | 79 | 25 |

237 of 756 stored T+15 closes (31 %) fall on a PST session (16 of those dates have no 13:00/12:59 PT ES bar — early closes). T+1 and T+5 are the same picture (PST n = 217 / 214, median |x| 7.25, p90 ≈ 24–26, max 103). The error is unbiased (mean ≈ −1) — noise, not a level shift — but 37 % of PST rows are off by > 10 pts against a 0.25 × IM tolerance of ≈ 10–14 pts. For contrast, PDT rows vs the 13:00-bar close: median |x| 1.75, p90 4.75 (the excluded 13:00 bar only).

`position_tN_post_touch` labels that change with the true 13:00-bar close on the same session: **T+1 24 of 358 (15 of 106 PST) · T+5 23 of 360 (18 of 119 PST) · T+15 14 of 364 (13 of 122 PST)**. Separately, 24–25 rows per timeframe take their T+N from a "session" that is not an NYSE trading day (27 such dates in `cr_g`'s session list).

**TodaySetup trace.** `apps/web/modules/TodaySetup/routes.py:287` → `probability.compute_structural_probability` (default `feature_version = CANONICAL_FEATURE_VERSION`) → `_rank_analogues_with_outcomes` (`probability.py:575–670`, SELECT from `bt_daily_outcomes_active`: `position_t{1,5,15}_post_touch`, `session_open_t0`, all 12 `session_*_t{1,5,15}`) → `aggregate_post_touch_distribution` (`probability.py:197`, reads `position_tN_post_touch`, line 320) → `response["structural_probability"]["post_touch"]` and the direction-qualification badges (`routes.py:315–319`). So TodaySetup **does** consume the `cr_i` labels and carries the `cr_g` OHLC through the analogue payload. Its spot is ES: `fetch_rth_open(conn, trade_date)` (first RTH bar open from `ironbeam_es_1m_bars`, `routes.py:251–255`) — see Q1.

### Q5 — version plan

- **One constant versions both tables.** `packages/shared/canonical_version.CANONICAL_FEATURE_VERSION` is the version for `bt_daily_features` *and* `bt_daily_outcomes`: `apps/cron/job_orats_eod.py` writes features at it (`DAY_FEATURES_VERSION`), `scripts/cr_ab_open_implied_move.py` fills `implied_move_1d` at it, `cr_b_backfill_outcomes` selects feature rows at it and inserts outcomes at the *same* string (`_TARGET_DATES_SQL` NOT EXISTS on `o.feature_version = f.feature_version`), the sweep / `cr_as` / `cr_am` / `cr_ah` join `o ↔ f ON feature_version`, and the KNN (`probability.py`, Proposals, SetupV2, Analogues, `audit_overrides`, AuditFlags) reads features at it. Other tables with a `feature_version` column (`bt_strategies`, `bt_strategy_instances`: values `v1` / `v2`) are unrelated.
- **So yes — `bt_daily_features` needs a matching bump** (unlike the CR-BH shadow, which nothing read). Two ways:
  - **(B, recommended) bump both, copy features.** INSERT the 816 `v0.6.0-openiv` feature rows unchanged under the new version (writer has INSERT on `bt_daily_features`; PK `(ticker, trade_date, feature_version)`; precedent: CR-AB `cr_ab_backfill_openiv.py` wrote features at a new version the same way). Every join and reader keeps working untouched; promotion stays "edit one constant". Feature vectors are byte-identical, so KNN neighbours do not change — only the outcome labels attached to them.
  - (A) split the constant into `CANONICAL_FEATURE_VERSION` + a new `CANONICAL_OUTCOME_VERSION` and rewrite every `o ↔ f` join. Semantically cleaner (features did not change) but touches every live reader and both crons' SQL; more surface for the same result.
- **Name (proposed): `v0.7.0-spxcash`** — a minor bump, because the outcome *definition* changes (price series + session calendar + tN window), not just a data refresh. (`v0.6.0-openiv-spxcash` stays the CR-BH snapshot.)
- **Promotion:** edit `CANONICAL_FEATURE_VERSION` in `packages/shared/canonical_version.py` — own commit, last.
- **Readers that change behaviour on promotion** (all read the constant): `probability._rank_analogues_with_outcomes` / `compute_structural_probability` (touch / close / `days_to_reach` / tN / post-touch of analogues → TodaySetup, Proposals); `SetupV2/routes._fetch_analogue_outcomes` + fair value (`session_close_t15`, `final_close_distance_from_target`); `Proposals/routes` card spot (`session_open_t0` becomes the 06:33 SPX print instead of the ES 06:30 open — ≈ −28 pts on average); `scripts/cron_daily_leg_capture._ES_OPEN_SQL` (its `es_open` field becomes an SPX value — see Q1/Q2 for what it feeds); `job_orats_eod` and `cr_ab_open_implied_move` start writing features at the new version; `cr_b` / sweep write and promote outcomes at it; `run_reference_rerun.py` (monthly) reads whatever is canonical. `knn_config` versioning is independent (no change needed for identical features).
- **Deploy mechanics (Render, read 2026-09-20):** `autoDeploy = yes` on `main` for `sweep_pending_outcomes` (11:05 UTC), `backfill_outcomes` (13:40), `open-implied-move` (13:35), `orats-eod-gamma` (10:00, rootDir `apps/cron`) — they pick up the merge by themselves, so the merge must land with the backfill already complete. **`autoDeploy = no`: `Dash - MonoRepo` (web), `daily-leg-capture` (13:50), `reference-rerun` (monthly)** — these need the manual redeploy.

### What Step 0 changes — decisions needed before implementation

1. **ADR Consequences 3 is false** (Q2). The CR-AR reference is *expected to change*, and not only through labels: its settlement is an ES price. Fixing the harness stays out of scope (next CR) — but the ADR should be corrected before approval, and "locked" should come off +1.71 / 58.3 % until the re-run.
2. **Version plan** (Q5): bump both tables under one new version with a verbatim feature-row copy (recommended, CR-AB precedent) vs split the constant. Name proposed: `v0.7.0-spxcash`.
3. **Two live readers break or go stale on promotion unless touched in this CR** — a scope addition to approve: (i) `SetupV2/routes._fetch_rth_closes` (ES close used to reconstruct the target from an SPX-space distance) — switch to the shared SPX session frame, or read `drift_target` directly; (ii) `cron_daily_leg_capture` basis guard (`session_open_t0` as ES open).
4. **Live spot stays ES after promotion** (a3 / c3: TodaySetup + SetupV2 `fetch_rth_open` → regime, KNN query features, call/put choice). The ADR's item 4 says these must move or carry a basis. In or out of this CR? Recommendation: out — separate CR right after, because it changes live regime classification and needs its own before/after; but note the Proposals card spot *does* flip to SPX on promotion, so for a while the two cards will use different spots (they already do on the v2 card: `table_spot` next to ES).
5. **Auto-deploy:** the four data crons deploy on merge to `main`. The backfill must be complete and the constant commit must be in the same merge (or the merge must happen outside 10:00–13:50 UTC) so no cron runs on half a promotion.

## Step 0b — CR-AR re-settlement on SPX cash, AM-settled flags, quote settlement, frame B − A (2026-09-20, read-only)

Read-only (session `default_transaction_read_only = on`; harness `main()` never called → no `bt_backfill_runs` row, nothing in `bt_edge_backtest_results`; train only, universe ≤ 2026-06-05). Scripts: `scripts/cr_bi_step0b_resettle.py`, `scripts/cr_bi_step0b_frame_test.py`; outputs in `scripts/logs/cr_bi_step0b_*` (git-ignored). The replica reproduces the stored CR-AR debit cells to 4 decimals (all 103 / +1.7129 / 58.25 % / base +1.6157 / beat +0.0972; near, mid, far likewise), so the fills are identical.

**A. CR-AR debit, T = 0.05, re-settled on SPX cash** (CR-BH hybrid series, last print 12:50–13:00 PT on expiry; gross pts; like-for-like = the 103 trades settled under both):

| band | n | ES (old): mean · win [Wilson] · base · beat | SPX cash (new): mean · win [Wilson] · base · beat |
| --- | --- | --- | --- |
| all | 103 | +1.713 · 58.3 % [48.6, 67.3] · +1.564 · +0.149 | **+1.315 · 54.4 % [44.8, 63.7] · +1.166 · +0.149** |
| near | 38 | +1.912 · 68.4 % [52.5, 80.9] · +1.823 · +0.088 | +1.912 · 68.4 % [52.5, 80.9] · +1.823 · +0.088 |
| mid | 34 | +1.235 · 55.9 % [39.5, 71.1] · +0.914 · +0.321 | **+0.617 · 50.0 % [34.1, 65.9] · +0.296 · +0.321** |
| far | 31 | +1.994 · 48.4 % [32.0, 65.2] · +1.960 · +0.034 | **+1.349 · 41.9 % [26.4, 59.2] · +1.315 · +0.034** |

(Baseline here is on the same trades, so beat is unchanged by construction — the settlement term cancels per trade; the published +0.097 / +1.616 used a 105-trade baseline denominator.) With each series' own coverage the SPX cell is n = 105, +1.461, 55.2 % [45.7, 64.4], base +1.370 (n = 107), beat +0.091 — but the two extra trades are the AM-settled quarterly-OPEX expiries 2023-06-16 and 2024-06-21 (ES has no 12:50–13:00 bar; both score near max profit on a 13:00 print that is not their settlement), so the like-for-like row is the honest one. One trade (expiry 2025-07-03, early close) has no 12:50–13:00 print in either series.

Basis on expiry day (ES settle − SPX settle), 105 trades: mean +25.5, median +24.6, p10 +2.4, p90 +49.1, min −8.4, max +67.3. **Only 5 trades change** (a 10-wide vertical is binary except inside the strikes): Σ −41.02 pts, 4 wins → losses, 0 the other way.

| trade_date | expiry | band | long/short | fill | ES settle | SPX settle | basis | P&L ES → SPX |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2023-05-01 | 2023-05-22 | far | 4195/4205 | −4.45 | 4206.00 | 4192.77 | +13.2 | +5.55 → −4.45 |
| 2023-07-17 | 2023-08-07 | mid | 4525/4535 | −4.75 | 4537.00 | 4516.78 | +20.2 | +5.25 → −4.75 |
| 2024-10-02 | 2024-10-23 | mid | 5810/5820 | −3.40 | 5835.50 | 5796.48 | +39.0 | +6.60 → −3.40 |
| 2024-10-03 | 2024-10-24 | far | 5840/5850 | −2.60 | 5852.50 | 5811.07 | +41.4 | +7.40 → −2.60 |
| 2025-12-02 | 2025-12-23 | mid | 6900/6910 | −4.70 | 6958.25 | 6908.98 | +49.3 | +5.30 → +4.28 |

**B. AM-settled expiries: 5 of 108 trades, all filled** (third-Friday expiries; every leg is `OPRA_ROOT = 'SPX'`, and `orats_options_minute.expiry_tod` on the entry day is `am` for all rows — 2023-05-25 → 2023-06-16 is mixed am 600 / pm 179 under one symbol): 2023-05-25 → 06-16 (far), 2024-03-28 → 04-19 (near), 2024-05-30 → 06-21 (far), 2025-07-25 → 08-15 (near), 2026-01-29 → 02-20 (near). A 13:00 PT print is the wrong settlement for these (SET, opening prices). Counted, not fixed. Excluding them: ES n = 100, +1.794, 59.0 %; SPX n = 100, **+1.384, 55.0 % [45.2, 64.4]**.

**C. Quote-based settlement** (last valid spread mid 12:50–13:00 PT on expiry; captured for 104 of 106 filled trades; 102 have all three):

| band | n | ES | SPX cash | option quotes | mean \|SPX − quote\| | mean \|ES − quote\| |
| --- | --- | --- | --- | --- | --- | --- |
| all | 102 | +1.776 / 58.8 % | +1.374 / 54.9 % | **+1.305 / 54.9 %** | 0.073 | 0.472 |
| near | 37 | +2.092 / 70.3 % | +2.092 / 70.3 % | +1.967 / 70.3 % | 0.124 | 0.124 |
| mid | 34 | +1.235 / 55.9 % | +0.617 / 50.0 % | +0.580 / 50.0 % | 0.043 | 0.654 |
| far | 31 | +1.994 / 48.4 % | +1.349 / 41.9 % | +1.311 / 41.9 % | 0.046 | 0.686 |

SPX-cash settlement agrees with the option market (identical win/loss on every trade; the ≈ 0.07 gap is residual bid/ask on in-the-money spreads in the last minutes — six trades differ by 0.5–1.0). ES settlement does not. **Restated CR-AR debit reference: ≈ +1.3 pts, ≈ 55 % win (Wilson ≈ 45–64 %), not +1.71 / 58.3 %; the damage is in mid (+1.23 → +0.62) and far (+1.99 → +1.35); near is untouched.**

**D. Frame B − frame A — FAILS the 5-pt test by a wide margin. STOP.** Per date, kernel-weighted mean of (`discounted_level` − `strike`) over the positive-net-GEX rows that build the wall at `drift_target` (same Gaussian kernel as `gex_landscape`, σ = 8·√dte), 655 dates: **median +33.6 pts, mean +34.1, p75 +42.9, p90 +50.2, p99 +65.8, max +70.1**; weighted DTE of the wall median 42 days (p90 62). By year: 2023 +27.7 / 2024 +31.6 / 2025 +34.2 / 2026 +40.0 (median). On the CR-AR trade dates: median +33.0, p90 +46.7.

- Why so large: `job_orats_eod.compute_discounted_level` = `strike × exp((short_rate − div_yield) × (dte + 1) / 252)` with **calendar** `dte` over 252 and **`div_yield` = 0.0** in `orats_oi_gamma` — ≈ +1.1 pt per calendar day at SPX 7600 / r 3.7 % (7600 strike: +5.6 at 4 DTE, +18 at 15 DTE, ≈ +48 at 42 DTE), roughly twice the true carry (ORATS' own per-expiry forward rises ≈ +0.6 pt/day). And the dominant wall is built from ≈ 40-DTE options, not short-dated ones — the ADR's "mostly short-dated and therefore close to cash" is not what the data shows.
- Consequence: **B ≈ C on average.** Mean carry 31.9 vs mean ES basis 25.5 on the 479 computed rows; (basis − carry) mean −6.7, sd 15.8. The E6 convention was roughly right on level for the wrong reason; its error is the *mismatch* between the ES sawtooth and the wall's carry, not the whole basis. CR-BH's SPX-vs-B shadow removed the +25 basis but left the +32 carry, so it **under**states touch.

Three-frame comparison (read-only; SPX-vs-A = SPX cash vs target − carry(trade_date)):

| subset | n | touch: ES-vs-B (canonical) | SPX-vs-B (CR-BH shadow) | SPX-vs-A | close: ES-vs-B | SPX-vs-B | SPX-vs-A |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all | 479 | 83.9 % | 73.3 % | **84.6 %** | 8.1 % | 8.1 % | 5.4 % |
| all, same calendar | 231 | 74.5 % | 62.3 % | **76.6 %** | 9.1 % | 10.0 % | 6.9 % |
| magnet-above 5 | 99 | 61.6 % | 43.4 % | **66.7 %** | 17.2 % | 13.1 % | 13.1 % |
| magnet-above 20 | 176 | 90.9 % | 76.7 % | **90.3 %** | 8.0 % | 8.0 % | 5.1 % |
| magnet-above 60 | 103 | 88.3 % | 84.5 % | 90.3 % | 3.9 % | 4.9 % | 1.0 % |
| magnetic-pin | 98 | 90.8 % | 87.8 % | 87.8 % | 4.1 % | 7.1 % | 3.1 % |

5-session magnet-above touch by ES-basis bucket (< 20 · 20–35 · 35+): ES-vs-B 48 · 50 · 80 % (p 0.006); SPX-vs-B 33 · 43 · 50 % (p 0.18); **SPX-vs-A 63 · 57 · 78 % (p 0.20)** — the basis gradient is removed in a consistent frame too, but at the *original* level, not 18 pts lower. magnet-above flips ES-vs-B → SPX-vs-A: 5 T→F, 11 F→T.

**What this means.** (1) The CR-BH headline ("touch inflated ≈ 10 pts; honest magnet-above base rates ≈ 43 % / 77 %") and the ADR's evidence table are artifacts of comparing frame-A prices to frame-B targets — the base rates were about right (≈ 62–67 % / 90 %). What ES-vs-B genuinely gets wrong is the *time-varying* part (the basis gradient, roll seams, holiday sessions, the UTC window). (2) Implementing CR-BI as specified would make the artifact canonical. (3) The settlement finding (A–C) stands on its own: it is A-vs-A (SPX cash vs listed strikes) and is confirmed by option quotes. (4) "Where is the wall in cash?" is now the question: the listed strike K (frame A), the landscape's `discounted_level` (B, carry overstated ≈ 2×, q = 0), or the cash level at which those options are at-the-money-forward (≈ K·e^{−(r−q)T}, *below* K). The harness already treats `drift_target` as a cash strike when it snaps legs (Q1 d3) while the live card converts B → A — so the two already disagree by this same ≈ 30 pts on 40-DTE walls (less on the 15-DTE structure's own carry, ≈ 18).

**Options for the canonical switch (decision needed; nothing implemented):** (a) SPX cash vs targets converted to cash per date (SPX-vs-A above) — needs the carry per wall stored or recomputed at outcome time, and a decision on whether to also fix `compute_discounted_level` (calendar/252, q = 0), which would move every landscape, wall and regime label: a much bigger CR; (b) keep B targets and add the wall carry to the SPX series per row (what the frame test does) — smallest change, no landscape rebuild, but bakes a known-overstated carry into outcomes; (c) rebuild the landscape in cash space (strike axis), classify and target in A everywhere — cleanest, largest; (d) pause CR-BI, keep ES-vs-B canonical, ship only the frame-independent fixes (NYSE-calendar sessions, the `cr_g`/`cr_i` PT window, harness settlement on SPX cash).

## Amendment A1 (Ryan, 2026-09-20, after Step 0b) — re-scope to the frame-independent fixes (option d)

Supersedes "Implementation" / "Implementation order" above. The ADR `2026-09-20 - Outcomes Evaluated on SPX Cash with NYSE-Calendar Sessions` is suspended (marked from chat, with the three-frame table); not edited here. Step-0 decisions 2–5 from the first review are void except where restated.

**In scope**

1. **Sessions from `packages/shared/trading_calendar.py`; one PT session window.** New `packages/shared/sessions.py`: the RTH window (bar-open 06:30:00–13:00:00 PT inclusive — the existing t0 window, DST via `America/Los_Angeles`), `fetch_es_daily_bars` (ES RTH daily OHLC restricted to NYSE trading days), `session_ohlc_at` (T+N OHLC) and `post_touch_positions`. `cr_b_backfill_outcomes._fetch_daily_bars` (→ sweep, CR-BG) delegates to it; `cr_g_backfill_session_ohlc.py` and `cr_i_backfill_post_touch_positions.py` drop their fixed 13:30–20:00 UTC windows and use it. Consequence to state: tN closes move to the 13:00-bar convention of t0 (PDT rows shift by the 13:00 bar, median |x| ≈ 1.75 pts; PST rows by the full window error, median ≈ 7).
2. **New `feature_version` `v0.6.1-nyse-sessions`**: verbatim copy of the active `v0.6.0-openiv` feature rows + every outcome row recomputed (outcome metrics, t0 / containment, t1/t5/t15 OHLC, post-touch) through the shared path. **Outcomes stay ES-vs-B.** INSERT-only, backfill protocol. Smoke: t0 columns identical to `v0.6.0-openiv` on every row whose trade_date is a trading day; touch / close / `days_to_reach` identical on every computed row whose horizon contains no non-NYSE "session"; every other difference listed and attributed (holiday session removed / PT window).
3. **Promotion** of `CANONICAL_FEATURE_VERSION` → `v0.6.1-nyse-sessions` as its own commit, last (crons then write features and outcomes at it; the feature copy exists for exactly this). PR left open for Ryan to merge outside 10:00–13:50 UTC.
4. **Harness settlement on SPX cash.** `packages/shared/spx_cash.py` = the CR-BH hybrid builder moved from `scripts/cr_bh_spx_cash.py` (which becomes a re-export). `cr_ah_step4_analysis.get_settlement_price` → last clean SPX-cash print 12:50–13:00 PT on expiry. Option-quote settlement (last valid spread mid in the same window) is collected per trade and printed / stored in the run smoke as the cross-check; it does not enter the cells. **AM-settled expiries are excluded, not settled**: a trade whose entry-day leg quotes carry `expiry_tod = 'am'` gets `excluded_reason = 'am_settled_expiry'` (the SET value is not in the DB; a 13:00 print is wrong for them). `run_reference_rerun.py` wraps the harness and inherits all of it. The ES touch-exit path (`detect_touch`) is untouched (frame question, follow-up CR).
5. **Restated reference persisted under `cr_id = 'CR-BI'`** (train, universe_end = split = 2026-06-05, walk-forward). `CR-AR` rows are never touched. The v2 card reads the latest `REF-%` else `CR-AR` (`SetupV2/service.pick_reference_cr_id`), so `CR-BI` is not shown; the first `REF-2026-09` run (2026-10-01, after the redeploy) will carry the new settlement to the card by itself.
6. `v0.6.0-openiv-spxcash` stays in the table; its session note gets a do-not-promote line (frame-A prices vs frame-B targets).

**Out of scope (explicit):** SPX-cash outcomes; SetupV2 target reconstruction; the leg-capture basis guard; live spot; `compute_discounted_level` / landscape frame; `detect_touch`; card notes (no frame switch is happening, so nothing new to warn about); `contracts.py`; Saturday 2026-02-21 bars; `spot_price` drift.

**Implementation order:** Step 1 shared sessions + SPX-cash modules with tests → Step 2 `cr_b` / `cr_g` / `cr_i` on the shared window → Step 3 `v0.6.1-nyse-sessions` backfill + smoke → Step 4 harness settlement + AM exclusion + quote cross-check → Step 5 restated reference run (`CR-BI`) → Step 6 follow-up Step-0 material (read-only, below) → Step 7 promote the constant (own commit, last) → wrap + redeploy list.

## Pre-registration — follow-up CR Step 0, placebo tests (written before any run; one look each)

Frame: **SPX cash vs carry-corrected levels.** Prices = CR-BH hybrid SPX-cash daily OHLC (`packages/shared/spx_cash`, NYSE sessions by construction; open = first print ≥ 06:33 PT). Every wall price P on date d is converted to cash as P_A = P − carry(d, P), carry = kernel-weighted mean of (`discounted_level` − `strike`) over the `orats_oi_gamma` rows of d whose net GEX has the wall's sign, weight = |net| · exp(−(P − L)² / 2σ²), σ = 8·√max(dte, 0.5) — i.e. the level is put back on the strike axis. IM_d = `feature_vector.implied_move_1d` at `v0.6.0-openiv`. **Train only: signal date ≤ 2026-06-05 and the whole evaluation window ends ≤ 2026-06-05, for actual and placebo alike — no price after the split is read.** Not walk-forward. CI for every difference: cluster bootstrap over calendar months of the signal date, 10 000 resamples, seed 20260920, 95 % percentile interval. No threshold, subgroup or definition is changed after the run; anything further is a new registration.

**P1 — touch.** Signal set S_h, h ∈ {5, 20}: active `v0.6.0-openiv` rows with regime `magnet-above`, `horizon_sessions = h`, a positive-GEX `drift_target`, IM_d > 0, SPX sessions d…d+h−1 available. z_d = (target_A − open_d) / IM_d. Actual_d = 1 if max(high over the h sessions from d) ≥ target_A. Placebo_d = mean over every *other* train date e ≠ d (any regime, IM_e > 0, window available) of 1[max(high over h sessions from e) ≥ open_e + z_d · IM_e]. Report per h: n, mean Actual, mean Placebo, difference, CI. Rows with z_d ≤ 0 are kept (touch at the open counts for both). Single pre-registered secondary: the same with the placebo pool restricted to dates whose regime is not `magnet-above`.

**P2 — containment.** Signal set: every active train row with IM_d > 0, an SPX session on d, and a wall (any sign, from `orats_gex_landscape.walls`) strictly below and strictly above open_d after conversion to cash; nearest each side → offsets a_d = (above_A − open_d) / IM_d, b_d = (open_d − below_A) / IM_d. Actual: close-inside = below_A < close_d < above_A; range-inside = below_A < low_d and high_d < above_A (t0 session only, the CR-AQ definitions). Placebo_d = mean over every other train date e of the same two indicators for the band [open_e − b_d · IM_e, open_e + a_d · IM_e]. Report n, actual, placebo, difference, CI for close-inside and range-inside, pooled and by `regime_at_classification` of d (regimes with n < 20 are reported but flagged).

## Steps 1–2 — shared session definition; fixed-UTC windows retired (`7b1bb05`, `5531453`)

- `packages/shared/sessions.py`: `RTH_BARS_SQL` (bar-open 06:30:00–13:00:00 `America/Los_Angeles`, inclusive — the existing t0 window, DST-safe), `fetch_es_daily_bars` (NYSE trading days only, `trading_calendar.is_trading_day`), `session_ohlc_at`, `post_touch_positions`. Tests: `packages/shared/tests/test_sessions.py`.
- `packages/shared/spx_cash.py` = the CR-BH builder moved (git mv) + `settlement_prints`; `scripts/cr_bh_spx_cash.py` re-exports. Tests: `test_spx_cash.py`. (No pytest in any local venv; the 9 new tests were run with a function runner — all pass. The existing pytest suites could not be run locally.)
- `cr_b_backfill_outcomes._fetch_daily_bars` and `cr_aa_sweep_pending_outcomes._fetch_daily_bars` delegate to the shared fetch (their private SQL copies are gone); the sweep's `_fetch_session_dates` drops non-NYSE dates, so `_expected_horizon_end` counts real sessions. `cr_g` / `cr_i`: `RTH_START_UTC` / `RTH_END_UTC`, `_fetch_rth_minute_bars`, `_compute_session_ohlc`, `_fetch_rth_daily_bars` removed → `fetch_es_daily_bars` + `session_ohlc_at` / `post_touch_positions`. They remain manual NULL-fill tools (the crons do not fill tN / post-touch; unchanged).
- Side finding: `ironbeam_es_1m_bars` has **no RTH-window bars on 2023-12-15 and 2025-12-19** (December quarterly-expiry Fridays) — those trade dates have NULL t0 columns in every version.

## Step 3 — `v0.6.1-nyse-sessions` (run `a44f185c-06b3-4c27-be69-9c74d9eb7e79`, 2026-09-21 UTC)

`scripts/cr_bi_backfill_nyse_sessions.py`, INSERT-only, one transaction: **815 feature rows copied verbatim** (`feature_vector`, regime, config hash identical on 815/815) + **815 outcome rows** (computed 479 · na_regime 316 · na_data 15 · pending 5 — no status transitions). `v0.6.0-openiv` untouched (816 rows, max `computed_at` still CR-BG's). Outcomes are still ES-vs-B.

Smoke gate **PASS**:
- ES RTH-window dates 866 → NYSE sessions 840; the 26 dropped = the market holidays + Sat 2026-02-21 + 2026-09-07 listed in CR-BH Step 1.
- t0 / containment columns: **0 rows differ**; 3 rows were NULL in the source and are now filled (2026-09-09 / 10 / 11 — the sweep's null-fill had not landed; CR-BG side finding).
- Label columns (status, touch, close, `days_to_reach`, `horizon_end_date`, excursion, final distance, realized EM): 240 rows differ, **0 without a non-NYSE "session" inside the old horizon** — every difference is a holiday session removed (the horizon now runs one or more real sessions longer). Full list: `scripts/logs/cr_bi_nyse_sessions_label_diffs.csv`.
- **12 touch/close flips** (all explained by the longer real horizon): touch F→T 2023-05-26, 2024-08-22, 2024-11-25; close T→F 2024-01-09, 2024-11-12, 2024-11-19, 2024-11-27, 2025-11-06, 2026-06-12, 2026-06-18, 2026-07-01; close F→T 2023-05-26, 2024-01-11, 2024-11-25.
- T+N closes, old (UTC window, bar-present sessions) → new: non-NULL 767 / 764 / 756 → 785 / 783 / 776 (the script also fills rows `cr_g` had not reached); |Δ| median 2.25 / 2.75 / 5.25, p90 14.3 / 28.0 / 48.9 (PST window error + the 13:00 bar + T+N landing on a different date when a holiday was counted). Post-touch labels: non-NULL 387 → 405; changed where both exist: T+1 30, T+5 34, T+15 40.

## Step 4 — harness settlement (`931f572`)

`cr_ah_step4_analysis.get_settlement_price` = last clean SPX-cash print 12:50–13:00 PT on expiry (`spx_cash.settlement_prints`, series loaded once per process, ≈ 3.5 min). `get_settlement_quote_val` (cross-check only, printed + stored in the run smoke as `settlement_quote_crosscheck`). **AM-settled expiries are excluded, not settled** (`is_am_settled`: entry-day leg quotes with `expiry_tod = 'am'` → `excluded_reason = 'am_settled_expiry'`, reported beside the CR-AN decision-6 exclusions and in the smoke as `am_settled_excluded`). `run_reference_rerun.py` wraps the harness unchanged and inherits all three. `detect_touch` (ES vs `drift_target`) is untouched.

## Step 5 — restated reference, `cr_id = 'CR-BI'` (run `9b292f22-0af8-4865-89a3-416c3eea14d7`)

`--cr-id CR-BI --universe-end 2026-06-05 --split-date 2026-06-05 --structural-prob-mode walk-forward`, canonical still `v0.6.0-openiv` (fills identical to CR-AR). 8 cells persisted under `CR-BI`; the 8 `CR-AR` rows are untouched (`created_at` 2026-09-07). The v2 card reads `REF-%` else `CR-AR`, so `CR-BI` is not shown.

- Excluded as AM-settled: 5 debit + 5 credit trades (entry 2023-05-25, 2024-03-28, 2024-05-30, 2025-07-25, 2026-01-29). Settlement available: debit 102/103, credit 100/101 (expiry 2025-07-03 early close). Cross-check SPX-cash intrinsic vs option quotes: debit n = 101, mean |diff| 0.078, max 1.0, none > 1 pt; credit n = 99, mean 0.081, 2 trades > 1 pt (max 2.0).
- **The harness picks its threshold on train, and the pick moved: debit T = 0.10 (was 0.05).** Debit sweep: T 0.00 n 101 +1.38 / 55 % beat +0.09 · **T 0.05 n 100 +1.38 / 55 % beat +0.09** (the like-for-like cell; matches Step 0b's ex-third-Friday +1.384 / 55.0 %) · T 0.10 n 99 +1.46 / 56 % beat +0.17 ← chosen · T 0.15 n 98 +1.44 · T 0.20 n 95 +1.30.

| debit, persisted cells | CR-AR (ES settle, T 0.05) | CR-BI (SPX cash, AM-settled excluded, T 0.10) |
| --- | --- | --- |
| all | n 103 · +1.713 · 58.3 % [48.6, 67.3] · base +1.616 · beat +0.097 | n 99 · **+1.464 · 55.6 % [45.7, 65.0]** · base +1.290 · beat +0.174 |
| near | n 38 · +1.912 · 68.4 % · beat −0.034 | n 35 · +2.180 · 71.4 % [54.9, 83.7] · beat −0.003 |
| mid | n 34 · +1.235 · 55.9 % · beat +0.321 | n 34 · +0.642 · 50.0 % [34.1, 65.9] · beat +0.346 |
| far | n 31 · +1.994 · 48.4 % · beat +0.034 | n 30 · +1.560 · 43.3 % [27.4, 60.8] · beat +0.245 |

  Credit (T 0.05): n 20 · −1.602 · 60.0 % · beat −0.544 (CR-AR at its own T 0.00: n 27 · −2.045). Caveat, pre-existing: the threshold is selected in-sample by best beat, so "beat" at the chosen T is optimistic; the fixed-T 0.05 row is the comparable number.

## Follow-up CR Step 0 material (read-only; nothing written)

### A. `compute_discounted_level` — formula, inputs, direction

`apps/cron/job_orats_eod.py:121–125`: `discounted_level = strike × exp((short_rate − div_yield) × t)`, `t = (int(dte) + 1) / 252`.
- **Day count:** `dte` = calendar days (`expir_date − trade_date`, verified equal on every row of four sample dates) **+ 1, divided by 252** — a calendar count over a trading-day year, ≈ 1.45× too long (365/252), plus the +1.
- **Rate:** ORATS per-expiry `riskFreeRate` (fallback `riskFree30`): 4.5–5.5 % in 2023–24, 3.7–4.4 % in 2025–26.
- **Dividend:** ORATS per-expiry `yieldRate`; **exactly 0 on 40–55 % of rows** (the nearer expiries), 1.0–1.4 % on average. So for the expiries that build most walls the exponent is the full risk-free rate.
- **Direction: UP.** Every level is moved *above* its strike (7600 strike, 2026-06-01, r 3.72 %, q 0: +1.1 at 0 DTE, +5.6 at 4, +18.0 at 15, ≈ +48 at 42 DTE). Measured at the `drift_target` wall: median +33.6, p90 +50.2 (Step 0b D); across all walls on train dates: median +38.7, p90 +61.5.
- **What the physics says:** an option's gamma peaks where the forward to *its* expiry equals the strike, F_T = S·e^{(r−q)T} = K, i.e. at cash **S\* = K·e^{−(r−q)T} — below the strike** (further below by ≈ 1.5σ²T for the lognormal peak; small). The code applies the carry with the opposite sign — it answers "what forward does cash = K imply", not "what cash makes this option at-the-money-forward".
- **The level under each convention** (K = 7600, 42 DTE, r 3.72 %, true q ≈ 1.2 %): code **≈ 7648 (+48)**; (i) correct carry, act/365, r − q: **≈ 7578 (−22)**; (ii) strike axis, no adjustment: **7600 (0)**. At 15 DTE: +18 / −8 / 0. The gap between the code and (i) is ≈ 70 pts at 42 DTE, ≈ 26 at 15 DTE.
- In ES terms (what the canonical outcomes compare to): a wall at cash K·e^{−(r−q)T} shows up in the front ES contract at ≈ K·e^{(r−q)(T_ES − T)} — *at or slightly around* K + (small), not K + 34. So ES-vs-B works on average only because two errors (+34 carry on the target, +25 basis on the price) roughly cancel; neither is principled.

### B. Placebo tests (pre-registered above; single run `scripts/cr_bi_followup_placebo.py`, commit `b05954d` precedes the run; output `scripts/logs/cr_bi_followup_placebo.md`)

741 train dates with IM > 0 and an SPX session; SPX frame ends 2026-06-05 (no price after the split read); 1 096 walls converted (carry median +38.7, p90 +61.5).

**P1 — touch, magnet-above vs the same IM-scaled distance on all other days**

| horizon | n | actual | placebo | difference | 95 % CI |
| --- | --- | --- | --- | --- | --- |
| 5-session (pool 737) | 91 | 68.1 % | 66.6 % | **+1.5 pts** | [−10.8, +11.9] |
| 5-session, secondary (non-magnet-above pool 365) | 91 | 68.1 % | 66.0 % | +2.1 | [−10.2, +12.5] |
| 20-session (pool 723) | 160 | 89.4 % | 87.2 % | **+2.2 pts** | [−6.3, +8.8] |
| 20-session, secondary (pool 358) | 160 | 89.4 % | 88.0 % | +1.4 | [−7.1, +8.0] |

z = (target − open)/IM: median 0.86 (5-session) / 0.78 (20-session); z ≤ 0 on 17 / 29 rows (target at or below the open after conversion — counted as touched for actual and placebo alike).

**P2 — containment, two-sided wall band vs equal-shape bands on all other days** (256 of 741 rows have a wall on both sides of the open; band half-widths: above median 1.27 IM, below 2.56 IM)

| | regime | n | actual | placebo | difference | 95 % CI |
| --- | --- | --- | --- | --- | --- | --- |
| close-inside | pooled | 256 | 84.8 % | 84.7 % | **+0.1 pts** | [−4.6, +4.5] |
| | magnet-above | 156 | 87.8 % | 87.0 % | +0.8 | [−5.2, +5.8] |
| | amplification | 66 | 83.3 % | 82.6 % | +0.7 | [−7.3, +8.5] |
| | untethered | 23 | 78.3 % | 80.8 % | −2.6 | [−15.5, +12.0] |
| | bounded ⚑ | 7 | 57.1 % | 67.6 % | −10.5 | [−56.1, +8.9] |
| | magnetic-pin ⚑ | 4 | 75.0 % | 82.3 % | −7.3 | [−64.4, +31.3] |
| range-inside | pooled | 256 | 75.4 % | 72.8 % | **+2.6 pts** | [−2.0, +6.9] |
| | magnet-above | 156 | 80.8 % | 76.5 % | +4.2 | [−2.0, +9.6] |
| | amplification | 66 | 75.8 % | 68.8 % | +6.9 | [−0.2, +13.9] |
| | untethered | 23 | 56.5 % | 67.1 % | −10.5 | [−25.0, +5.6] |
| | bounded ⚑ | 7 | 28.6 % | 46.2 % | −17.6 | [−38.5, −0.4] |
| | magnetic-pin ⚑ | 4 | 50.0 % | 69.9 % | −19.9 | [−67.0, +27.2] |

⚑ n < 20. **Reading (one look, no re-cuts):** in this frame neither test separates the walls from a level at the same IM-scaled distance on any other day. Touch: +1.5 / +2.2 pts, CIs ± 8–11 pts, centred near zero — a high raw touch rate (68 % / 89 %) is what a level ≈ 0.8 IM above the open gets anyway over 5 / 20 sessions in a rising market. Containment: close-inside is exactly at placebo; range-inside is +2.6 pooled with the CI spanning zero (amplification +6.9 is the only cell whose interval nearly excludes zero — one of twelve cells). What this does **not** show: the tests are powered only for effects ≳ 8–10 pts (touch) / 5 pts (containment); the frame is the strike-axis conversion of a landscape whose kernel centres are themselves mis-placed (A), so a wall located correctly might behave differently; and the debit's P&L does not require a touch edge (its restated +1.4 comes mostly from near-band drift, where beat ≈ 0 — i.e. the same as buying the spread at the open without the signal).
