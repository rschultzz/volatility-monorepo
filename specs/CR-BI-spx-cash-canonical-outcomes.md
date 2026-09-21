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
