---
type: cr
cr_id: CR-DD
title: Exact aggressor volume at price — live aggregation + Databento backfill (es_flow_price_1m)
status: in-progress
run_mode: interactive
data_safety_class: write_backfill
branch_name: feat/CR-DD-flow-price-1m
dependencies: [CR-CA]
depended_on_by: [CR-DC]
tags: [dash, cr, order-flow, aggressor, databento, ironbeam, data-pipeline]
---

# CR-DD — Exact aggressor volume at price

## Kickoff (Ryan, 2026-10-27, verbatim)

```
CR-DD — Exact aggressor volume at price: live aggregation + Databento backfill (write_backfill).

Branch first: git checkout -b feat/CR-DD-flow-price-1m off origin/main. Own worktree.
Read CLAUDE.md (backfill protocol), field-conventions.md, the CR-CA spec (Databento trades pull and its
budget cap), the CR-CX spec (per-session pricing ~$0.39–0.47), and CR-DC's Step 0 findings.
Build:
1. Table es_flow_price_1m: minute_utc, symbol, price (0.25 grid), buy_vol, sell_vol, trade_count, source
   ('ironbeam' | 'databento'), PK (minute_utc, symbol, price, source). Insert-only (no UPDATE/DELETE
   triggers needed beyond the role: dash_backfill_writer has no DELETE). DDL applied by the owner.
2. Live: a cron that aggregates ironbeam_es_trades into es_flow_price_1m for completed minutes, every
   5 minutes during the session and once after the close — and in all cases BEFORE prune-ironbeam-trades
   at 01:00 UTC. Idempotent (skip minutes already stored). Report the deploy spec for chat to create it.
3. Backfill: Databento trades (tick, with aggressor side) for 2026 train sessions 2026-01-06 → 2026-06-05,
   aggregated in memory to es_flow_price_1m — DO NOT store raw ticks. Front contract per session as CR-CA.
   Budget: price the full pull first; STOP if > $75. Under the backfill protocol (backfill_run 'CR-DD').
   Holdout (2026-06-06 → 2026-10-26) NOT fetched (B4). R-a deferred.
4. Checks: on 3 sessions, Σ per-minute volume and Σ(buy − sell) match es_flow_1m_databento within 1 %;
   on 2026-10-27, the live aggregate matches ironbeam_es_trades exactly.
PR, don't merge. Session note; copy logs before removing the worktree.
```

Consumer: CR-DC A1.3 reads `es_flow_price_1m` first (schema `(minute_utc, symbol, price, buy_vol,
sell_vol, trade_count, source)`; `databento` only for RTH of sessions ≤ 2026-06-05, `ironbeam` otherwise;
`price` taken as the 0.25-pt cell floor).

## Standing rulings conformance

Data-pipeline CR; no test, outcome or feature statistic is computed.

| Ruling | Status | Note |
|---|---|---|
| A1–A21 | n/a | no setup, level or exit logic |
| B1–B3 | n/a | no verdicts |
| B4 (data split) | honoured | backfill = train 2026-01-06 → 06-05 only; holdout neither fetched nor read (the CR-CA raw files on disk for holdout dates are not opened); the live cron writes the forward log (2026-10-27 →) once, insert-only; the 2026-10-27 check is mechanical (aggregate = trades), no direction / outcome statistic |
| B5 (Databento reference) | honoured | `source` keeps the two feeds apart; never pooled |
| B6–B8 | n/a | |
| C1, C2 | n/a | no UI |

Revoked / superseded R1–R7: none used.

## Step 0 findings (2026-10-28 03:15 UTC, read-only)

Script: `scripts/cr_dd_step0.py` (phases `cost`, `size`, `live`); output in `scripts/logs/cr_dd/step0_*.txt`
(git-ignored). No DB writes, no Databento download (`get_cost` / `get_billable_size` only, which are free).

### S1 — budget gate: PASS

- `get_cost` (GLBX.MDP3 `trades`, `stype_in=raw_symbol`, CR-CA window [06:15, 13:00) PT, CR-CA `contracts_for`)
  for 2026-01-06 → 06-05: 105 NYSE sessions, 115 session-contracts (10 roll-week sessions carry two contracts).
  ESH6 52 windows $25.78; ESM6 63 windows $27.73. **Priced total $53.51** (2.05 GB billable). Max per session $0.854
  (a roll-week session). **Within the $75 cap.**
- **All 115 windows are already on disk** as CR-CA L7 raw DBN files (`data/databento/raw/` in the main checkout).
  If they are reused, the actual new spend is **$0.00**. These are the files CR-CA downloaded; CR-DD stores no new raw ticks.

### S2 — size

- Databento ESH6 2026-01-06 (window): 266,264 trades, side B 138,295 / A 127,969 / N 0. 0 prices off the 0.25 grid.
  **3,171 (minute, price) rows** over 405 minutes.
- Backfill projection: about 365 k rows, under 0.05 GB.
- Live: the 2026-10-27 Globex day (10-26 22:00 → 10-27 21:00 UTC) has 10,507 (minute, symbol, price) rows,
  about 2.7 M rows a year (under 0.3 GB a year).
- DB is 26.74 GB now.

### S3 — `ironbeam_es_trades` (live source)

- Retained span: 2026-10-26 01:01 → now (prune keeps 2 days: `DELETE … WHERE ts_utc < now() − 2 days`, at
  01:00 UTC). One symbol (`XCME:ES.Z26`). 885 k rows. 0 prices off the 0.25 grid. `trade_id` is NULL on every
  row; `trade_key` (PK) is unique.
- `aggressor_side`: `BUY` 446 k, `SELL` 439 k, **`'0'` 2 trades / 110 contracts** (unknown).
- **Ingest lateness** (`received_at − ts_utc`): median 0.41 s, p99 2.2 s, p99.9 5.2 s. **One trade** of 885 k
  arrived ≥ 3 min after its minute started: **the 2026-10-27 closing print**, 20:59:59.989 UTC (4 contracts, BUY),
  received at 21:42:56. That is 43 min late, inside the daily maintenance break. On 10-26 the 20:59 minute's
  last receipt was 21:00:00.
- **10-27 retention clock:** the 01:00 UTC prune on **2026-10-29** deletes trades before 2026-10-27 01:00 UTC.
  That is the first 3 h of the 10-27 Globex day (from 10-26 22:00). Check 4b must run before then, or be
  limited to [10-27 01:00, 21:00) UTC.

### S4 — DDL

`es_flow_price_1m` does not exist. `dash_backfill_writer` has no CREATE on `public`, so the owner applies the DDL
(as the kickoff says).

### Decisions (recommendation first; Ryan to rule)

| # | Question | Options | Rec. |
|---|---|---|---|
| DD1 | Contracts in roll weeks ("front contract per session as CR-CA") | (a) CR-CA's stored set: front, plus the dying contract on its roll Monday → expiry Friday. This matches `es_flow_1m_databento`, so CR-DC's per-segment bar-match symbol always has cells. (b) Front only. | (a) |
| DD2 | Source of the ticks | (a) Reuse the 115 CR-CA raw files on disk ($0, same bytes Databento would send). (b) Re-download ($53.51) | (a) |
| DD3 | Unknown-side prints (Ironbeam `'0'`, Databento `N`) | (a) Counted in `trade_count` only; not in `buy_vol` / `sell_vol` (the schema has no unknown column, and CR-DC does not draw them). (b) Add `unknown_vol` (a schema change to CR-DC A1.3) | (a) |
| DD4 | When the live minute is "completed" | (a) A minute is stored once now ≥ its start + 3 min (p99.9 lateness 5 s), **except the last 15 min before the 17:00 ET daily close, which are held until 18:00 ET** (the reopen; the "once after the close" pass). This catches the 43-min-late closing print. (b) 3 min everywhere: that closing print would be lost. Either way, a trade that arrives after its minute is stored is lost (insert-only; skip-stored minutes). | (a) |
| DD5 | Schedule | One Render cron `*/5 * * * 0-5` (UTC). Every 5 min Sun → Fri, which covers the session, the after-close pass at 18:00 ET, and many runs before each 01:00 UTC prune (rows survive about 48 h). It is DST-safe because the hold rule is in ET. | as written |
| DD6 | Live write role and run tracking | (a′) Every run connects with `BACKFILL_DATABASE_URL` and calls `assert_role_or_die` (insert-only role). `backfill_run(conn, 'CR-DD-live')` + `update_run_smoke` are recorded only on runs that insert rows, which is about 1 row per 5 min while trading, or (a″) only on the post-close pass. (b) A `bt_backfill_runs` row on every 5-min run (288 a day, most of them empty). | (a′) |
| DD7 | Backfill window | CR-CA's [06:15, 13:00) PT (the raw files cover only this). This matches CR-DC's `databento` use (RTH ≤ 06-05). | as written |
| DD8 | 2026-10-27 check timing | After the owner applies the DDL, the cron script runs once from the laptop over the retained trades. The check compares it to `ironbeam_es_trades` for the whole 10-27 Globex day, before 2026-10-29 01:00 UTC. | as written |

## Rulings (Ryan, 2026-10-28, after Step 0)

DD1–DD8 approved as recommended. Then: (1) owner applies `scripts/cr_dd_ddl.sql` from the main checkout's
`.env` (`DATABASE_URL`, never printed) and verifies table, PK and grants (`dash_backfill_writer` SELECT / INSERT
only; PUBLIC nothing); (2) live aggregator + backfill as specified, backfill from the CR-CA files on disk under
`backfill_run('CR-DD')`, **no download**; (3) DD8 before 2026-10-29 01:00 UTC plus the 3-session Databento check;
(4) push, PR, don't merge; the session note carries the Render cron deploy spec (env var names only).

### DDL applied (2026-10-28 ~03:25 UTC, owner)

`es_flow_price_1m` created; PK (minute_utc, symbol, price, source); CHECK source ∈ {ironbeam, databento}.
ACL `{rschultz=arwdDxtm/rschultz, dash_backfill_writer=ar/rschultz}`: the backfill role has SELECT + INSERT
only; no PUBLIC entry.

## Implementation and results (2026-10-28)

| Step | Commit content |
|---|---|
| 1 | `packages/shared/flow_price_1m.py` (cell rule, DD4 hold, live SQL) + tests; `scripts/cron_flow_price_1m.py` (live cron) |
| 2 | `scripts/cr_dd_backfill.py` (Databento train backfill from the CR-CA raw files) |
| 3 | `scripts/cr_dd_check_live.py` (check 4b) |
| 4 | `render.yaml` documentation entry for the cron; this section |

- **DD4 SQL = Python:** with `now` = 2026-10-27 21:30 UTC the pending set holds minutes 20:40–20:44 and
  holds back 20:45–20:59. At 22:00 UTC (18:00 ET) it releases them. Unit tests cover EDT and EST.
- **Backfill** (run `9154637c-2b2a-4aa7-b4d4-741993be061e`, `backfill_run('CR-DD')`, completed): **663,288 rows**,
  105 sessions, 115 session-contracts (ESH6 / ESM6, both contracts in the 10 roll-week sessions), 0 off-grid prices,
  DB rows = built = inserted. **No download**: all ticks came from the CR-CA files on disk. Spend $0.
  (The 365 k projection was based on one quiet session.)
- **Check 4a** vs `es_flow_1m_databento` (tolerance 1 % of the minute's volume): **PASS, and exact** on all three sessions:

  | session | minutes | Σ volume (cells = flow 1m) | Σ(buy − sell) (cells = flow 1m) | minutes exact |
  |---|---|---|---|---|
  | 2026-01-06 | 405 | 950,406 | +10,274 | 405 / 405 |
  | 2026-03-16 (roll week, 2 contracts) | 810 | 1,400,926 | −10,072 | 810 / 810 |
  | 2026-05-15 | 405 | 1,381,961 | +23,541 | 405 / 405 |

  No minute is in one table only. (`N` side is 0 on these days, so buy + sell = `volume`.)
- **Live, first run** from the laptop as `dash_backfill_writer` (2026-10-28 03:27 UTC, run
  `014491cb-4f1d-43db-8a54-789d80d39a91`): 21,826 rows over 2,698 minutes (2026-10-26 04:27 → 10-28 03:24 UTC).
  The in-run re-sum vs trades was equal. A second run 9 s later found **0 pending** (idempotent).
- **Check 4b (DD8), Globex day 2026-10-27** [10-26 18:00, 10-27 17:00 EDT), trades retained from 10-26 01:01 UTC:
  **EXACT MATCH**. 10,507 cells over 1,380 minutes, all equal on buy / sell / count; none in one side only;
  buy 722,830, sell 705,636, prints 420,969 on both sides. Mechanical only (B4).
  Caveat: this run came after the fact, so the 43-min-late closing print was already in the table. Live, DD4 would
  have caught it too: it arrived 21:42 UTC, before the 22:00 UTC release.
