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
