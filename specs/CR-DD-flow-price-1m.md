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
