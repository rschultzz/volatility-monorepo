# CR-BH — Shadow-recompute `bt_daily_outcomes` on SPX cash; diff against `v0.6.0-openiv`

> Branch: `feat/CR-BH-spx-cash-shadow-outcomes` (off `origin/main` a11e958)
> `data_safety_class: write_backfill`
> Source: vault open-question `outcome-target-vs-es-roll-seam-and-basis-bias` (incl. its "Evidence — quantified from chat" section, taken as given — not re-derived) + kickoff prompt (2026-09-20).
> INSERT only. Do not promote the canonical version. Do not change `cr_b_backfill_outcomes`, the CR-AA sweep, the Proposals card, or any live cron. `vol_risk_premium` is out of scope.

## Why

`compute_outcome` compares an SPX forward-space wall target, raw, to front-contract ES bars. The measured ES−SPX basis is a sawtooth: 33–49 pts in the first month after each roll, 15–28 mid-quarter, mean ≈ 28. The 5-session `magnet-above` touch rate runs 44 % / 52 % / 79 % across low / mid / high basis buckets (p ≈ 0.003). All directional `computed` rows are `magnet-above`, so the bias inflates touch one-sidedly. SPX cash removes basis and roll seams entirely.

## Verified from chat (read-only SQL, 2026-09-20)

- `orats_monies_minute` has 0DTE SPX rows on all 815 active outcome dates: 850 days, 816 with ≥ 380 RTH minutes (median 391, 06:30–13:00 PT), 34 partial, 0 sparse.
- SPX cash per minute = `avg(spot_price) WHERE expir_date = trade_date AND ticker IN ('SPX','SPXW')`, grouped by `date_trunc('minute', snapshot_pt)`. This predicate matches partial index `idx_omm_front_minute_pt_ticker`. Use `spot_price`, not `stock_price`.
- Known bad prints exist (basis of 710.8 in 2023-09, 548.4 in 2023-11).
- `bt_daily_outcomes` PK = `(ticker, trade_date, feature_version)`. Versions present: `v0.6.0-openiv` (816 rows), `v0.5.0-rebuilt` (743).

## Step-0 diagnosis gate (all answered in this file before any write)

1. **Version leakage.** Grep every reader of `bt_daily_outcomes` (`apps/`, `scripts/`, `packages/`, SQL views). Does each one filter on `feature_version` or the canonical version constant? If ANY reader does not, a new version would pollute it: **STOP and ask** (the alternative is a separate shadow table, which needs owner DDL).
2. Does anything join `bt_daily_outcomes` to `bt_daily_features` on `feature_version` such that a new outcomes version needs matching feature rows? State how the shadow version avoids breaking that join.
3. Document the exact session window the current code uses for `session_*_t0` OHLC. Stored closes differ from a 06:30–13:00 PT RTH close by 0.25 to 7 pts and the 09-18 stored high exceeds the RTH high, so the window is not plain RTH. SPX cash ends 13:00 PT: state the window to be used and the mismatch it implies.
4. List every place ES price enters an outcome row: `compute_outcome` (touch / close / excursion / realized EM), `compute_session_containment` (`nearest_walls(walls, ES open)`), t1/t5/t15 session and post-touch columns. All of them switch to SPX cash in the shadow version.
5. Minute-sampled SPX understates true high/low. Quantify with ES as proxy: for 50 random days, compare range from ES 1m closes only vs true ES 1m high/low. Report the median understatement in pts (expect small vs a 28-pt basis).
6. Bad-print filter: propose a rule (`spot_price > 0` plus a robust neighbor-median check) and list every minute it drops. List the 34 partial days and classify each: early close vs data outage.

## Implementation

- Build SPX-cash session OHLC per trade_date in code (no new table unless Step 0 forces it).
- Recompute ALL rows (`computed`, `pending_*`, `na_*`) into a new `feature_version`, proposed `v0.6.0-openiv-spxcash`: same targets, same regime labels, same horizons, same tolerance rules. Only the price series changes.
- INSERT only, under the unattended backfill protocol (`get_backfill_db_conn`, `assert_role_or_die`, `backfill_run`). No UPDATE or DELETE of existing rows.
- Do NOT promote the canonical version. `v0.6.0-openiv` keeps running untouched.

## Deliverable — the diff (session note + append to the open-question)

- Per-day basis: `es_minutes.close` at 07:00 PT minus SPX cash at the same minute; buckets < 20, 20–35, 35+; drop basis outside −50..120.
- Old vs new `reached_touch` and `reached_close`: confusion matrix overall and by regime × horizon × basis bucket. List every trade_date whose label flips.
- Pooled touch/close rates old vs new, by regime × horizon.
- Confound test: in the NEW version, does 5-session `magnet-above` touch still vary by basis bucket? If the gradient disappears, it was the basis. If it persists, there is a real post-OPEX effect worth its own open-question.
- Status transitions (`computed` ↔ `na_*`, pending) caused by the switch.
- Leave the open-question `status: in-progress`. Promotion to canonical is a separate decision (ADR) made from chat after reading the diff.

## Implementation order

1. Step 0 — diagnosis findings appended to this file (gate).
2. Step 1 — SPX-cash session builder + shadow recompute script (`scripts/cr_bh_shadow_spxcash_outcomes.py`), dry-run output.
3. Step 2 — run the INSERT under the backfill protocol; smoke appended.
4. Step 3 — diff script + results.
5. Smoke + wrap.
